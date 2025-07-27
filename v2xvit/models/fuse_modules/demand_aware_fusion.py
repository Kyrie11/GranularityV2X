import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# (The _create_grid_normalized helper class remains exactly the same as before)
def _create_grid_normalized(H: int, W: int, device: torch.device) -> torch.Tensor:
    """Creates a normalized 2D grid of reference points in the range [-1, 1]."""
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    return torch.stack((x_coords, y_coords), dim=-1)


class DualGuidanceAttentionFusion(nn.Module):
    # === FIX 1: Add `collaborator_dim` to handle collaborator feature dimensions ===
    def __init__(self, model_dim: int, collaborator_dim: int, num_heads: int, num_sampling_points: int,
                 demand_dim: int = 3):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_points = num_sampling_points
        self.head_dim = model_dim // num_heads

        # === FIX 2: Use `self.model_dim` instead of hardcoded 256 for robustness ===
        self.demand_encoder = nn.Sequential(
            nn.Conv2d(demand_dim, self.model_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.model_dim // 2, self.model_dim, kernel_size=1)
        )
        self.positional_embedding = None
        num_params_per_point = 2 + 1 + 3
        self.param_gen_proj = nn.Conv2d(model_dim, self.num_heads * self.num_points * num_params_per_point,
                                        kernel_size=1)

        # === FIX 1 (continued): `value_proj` now correctly maps from `collaborator_dim` to `model_dim` ===
        self.value_proj = nn.Conv2d(collaborator_dim, model_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(model_dim, model_dim, kernel_size=1)

    def forward(self,
                ego_features: torch.Tensor,
                ego_demand: torch.Tensor,
                collaborator_features_per_granularity: List[torch.Tensor],
                collaborator_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ego_features: Ego-agent's feature map (H_i). Shape: [1, C, H, W].
            ego_demand: Ego-agent's demand map (D_i). Shape: [1, 3, H, W].
            collaborator_features_per_granularity: List of 3 tensors [H''_j,g],
                each of shape [N-1, C_g, H, W]. Here C_g is `collaborator_dim`.
            collaborator_masks: Sparsity mask. Shape: [N-1, 3, H, W].

        Returns:
            torch.Tensor: The aggregated collaborative feature map (H_collab). Shape: [1, C, H, W].
        """
        if len(collaborator_features_per_granularity) == 0 or collaborator_features_per_granularity[0].shape[0] == 0:
            return torch.zeros_like(ego_features)

        # Note: num_collabs and H,W are derived from the collaborator features. C comes from the first tensor's channels.
        num_collabs, C_collab, H, W = collaborator_features_per_granularity[0].shape
        device = ego_features.device

        # --- 1. Formulate the comprehensive Query state Q_i ---
        if self.positional_embedding is None or self.positional_embedding.shape[-2:] != (H, W):
            self.positional_embedding = nn.Parameter(torch.zeros(1, self.model_dim, H, W, device=device))

        encoded_demand = self.demand_encoder(ego_demand)
        query_state = ego_features + encoded_demand + self.positional_embedding

        # --- 2. Generate Attention Parameters from Q_i ---
        params = self.param_gen_proj(query_state)
        # Reshape to separate heads and parameters
        params = params.view(1, self.num_heads, -1, H, W)

        offset_params, spatial_weight_params, granularity_weight_params = torch.split(
            params, [self.num_points * 2, self.num_points, self.num_points * 3], dim=2)

        # Reshape and apply activation functions
        offsets = offset_params.view(1, self.num_heads, self.num_points, 2, H, W).permute(0, 1, 4, 5, 2, 3)
        offsets = torch.tanh(offsets)

        spatial_weights = spatial_weight_params.view(1, self.num_heads, self.num_points, H, W).permute(0, 1, 3, 4, 2)
        spatial_weights = F.softmax(spatial_weights, dim=-1)

        granularity_weights = granularity_weight_params.view(1, self.num_heads, self.num_points, 3, H, W).permute(0, 1,
                                                                                                                  4, 5,
                                                                                                                  2, 3)
        granularity_weights = F.softmax(granularity_weights, dim=-1)

        # --- 3. Generate Value (V_j,g) from collaborator features (H''_j,g) ---
        # `value_proj` maps collaborator features from collaborator_dim -> model_dim
        values_per_granularity = [self.value_proj(feat) for feat in collaborator_features_per_granularity]

        # --- 4. Perform Dual-Guidance Sampling and Aggregation ---
        reference_grid = _create_grid_normalized(H, W, device).view(1, 1, H, W, 1, 2)

        # Reshape offsets to match grid_sample's expectations
        sampling_locations = (reference_grid + offsets).reshape(1, self.num_heads * H * W, self.num_points, 2)

        sampled_values = []
        for g in range(3):
            val_g = values_per_granularity[g]
            # `val_g` now has `model_dim` channels
            sampled_val = F.grid_sample(
                val_g,
                sampling_locations.expand(num_collabs, -1, -1, -1),
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            sampled_values.append(sampled_val)

        # Shape: [3, N-1, C, M*H*W, P]
        sampled_values = torch.stack(sampled_values, dim=0)

        # Reshape granularity weights for broadcasting
        # Shape becomes [3, 1, 1, M*H*W, P]
        gran_weights_bc = granularity_weights.permute(5, 0, 1, 2, 3, 4)  # [3, 1, M, H, W, P]
        gran_weights_bc = gran_weights_bc.reshape(3, 1, self.num_heads * H * W, self.num_points).unsqueeze(1)

        # Apply granularity weights
        fused_collab_values = torch.sum(sampled_values * gran_weights_bc, dim=0)

        # Apply sparsity mask
        unified_mask = collaborator_masks.sum(dim=1, keepdim=True) > 0
        sampled_mask = F.grid_sample(
            unified_mask.float(),
            sampling_locations.expand(num_collabs, -1, -1, -1),
            mode='nearest', padding_mode='zeros', align_corners=False
        )

        # Aggregate across collaborators
        summed_fused_values = (fused_collab_values * sampled_mask).sum(dim=0)
        num_agents_per_point = sampled_mask.sum(dim=0).clamp(min=1e-6)
        aggregated_value = summed_fused_values / num_agents_per_point

        # Apply spatial attention weights
        # Reshape spatial weights for broadcasting: [1, M*H*W, P]
        spatial_weights_bc = spatial_weights.reshape(1, self.num_heads * H * W, self.num_points)
        # Sum over sampling points dimension: [C, M*H*W]
        head_outputs = torch.sum(aggregated_value * spatial_weights_bc, dim=-1)

        # --- 5. Output Projection ---
        # Reshape from [C, M*H*W] to [C, M, H, W] to separate head outputs
        head_outputs_spatial = head_outputs.view(self.model_dim, self.num_heads, H, W)

        # Sum across the heads dimension, and add back the batch dimension
        aggregated_heads = head_outputs_spatial.sum(dim=1).unsqueeze(0)  # -> [1, C, H, W]

        H_collab = self.output_proj(aggregated_heads)

        return H_collab


class DemandDrivenFusionNetwork(nn.Module):
    """
    Orchestrates the entire demand-driven fusion process.
    """

    # === FIX 1 (continued): `g_out` now corresponds to `collaborator_dim` ===
    def __init__(self, model_dim: int, g_out: int, num_heads: int, num_sampling_points: int):
        super().__init__()
        self.fusion_attention = DualGuidanceAttentionFusion(
            model_dim=model_dim,
            collaborator_dim=g_out,  # Pass the collaborator dimension here
            num_heads=num_heads,
            num_sampling_points=num_sampling_points
        )
        self.final_layernorm = nn.LayerNorm(model_dim)
        self.refinement_block = nn.Sequential(
            nn.Conv2d(model_dim, model_dim * 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_dim * 2, model_dim, kernel_size=1, bias=False)
        )

    def forward(self,
                ego_features: torch.Tensor,
                ego_demand: torch.Tensor,
                collaborator_features: List[torch.Tensor],
                collaborator_masks: torch.Tensor,
                encoders: List[nn.Module]) -> torch.Tensor:
        # === Stage 1: Confidence-Aware Feature Encoding for Collaborators ===
        encoded_features_per_granularity = []
        for g_idx in range(3):
            # Each encoder outputs features of dimension `g_out` (C_ENCODED)
            encoded_g = encoders[g_idx](collaborator_features[g_idx])
            encoded_features_per_granularity.append(encoded_g)

        modulated_features_per_granularity = []
        for g_idx in range(3):
            encoded_g = encoded_features_per_granularity[g_idx]
            mask_g = collaborator_masks[:, g_idx:g_idx + 1]
            summed_features = (encoded_g * mask_g).sum(dim=0)
            num_agents = mask_g.sum(dim=0).clamp(min=1e-6)
            consensus_map_g = summed_features / num_agents
            similarity = F.cosine_similarity(encoded_g, consensus_map_g.unsqueeze(0), dim=1)
            confidence_g = 0.5 * (1 + similarity.unsqueeze(1))
            modulated_g = encoded_g * confidence_g * mask_g
            modulated_features_per_granularity.append(modulated_g)

        # === Stage 2: Dual-Guidance Attention Fusion ===
        # `modulated_features_per_granularity` (channel dim `g_out`) is passed in
        H_collab = self.fusion_attention(
            ego_features=ego_features,
            ego_demand=ego_demand,
            collaborator_features_per_granularity=modulated_features_per_granularity,
            collaborator_masks=collaborator_masks
        )

        # === Stage 3: Residual Fusion and Refinement ===
        normed_collab = self.final_layernorm(H_collab.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        H_fused = ego_features + normed_collab
        H_refined = H_fused + self.refinement_block(H_fused)
        return H_refined


class DetectionDecoder(nn.Module):
    """
    A simple decoder to generate detection predictions from the final feature map.
    """

    def __init__(self, in_channels: int, num_classes: int, num_regression_params: int):
        super().__init__()
        self.classification_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.regression_head = nn.Conv2d(in_channels, num_regression_params, kernel_size=1)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        class_preds = self.classification_head(x)
        reg_preds = self.regression_head(x)
        return class_preds, reg_preds


# --- Example Usage ---
if __name__ == '__main__':
    # --- Model Parameters ---
    C_H = 256  # Dimension of the final unified feature H_i (model_dim)
    C_ENCODED = 128  # The output dimension of each granularity encoder (g_out / collaborator_dim)
    C_V, C_F, C_R = 64, 128, 32  # Input channels for the 3 raw granularities
    H, W = 128, 128
    N = 5  # Total agents (1 ego + 4 collaborators)
    NUM_HEADS = 8
    NUM_SAMPLING_POINTS = 4
    NUM_CLASSES = 5
    NUM_REG_PARAMS = 8

    # --- Reusable Modules ---
    g1_encoder = nn.Conv2d(C_V, C_ENCODED, kernel_size=1)
    g2_encoder = nn.Conv2d(C_F, C_ENCODED, kernel_size=1)
    g3_encoder = nn.Conv2d(C_R, C_ENCODED, kernel_size=1)
    encoders_list = [g1_encoder, g2_encoder, g3_encoder]
    fusion_conv = nn.Conv2d(C_ENCODED * 3, C_H, kernel_size=1)

    # --- 1. Simulate EGO-AGENT feature generation ---
    ego_raw_F_V = torch.randn(1, C_V, H, W)
    ego_raw_F_F = torch.randn(1, C_F, H, W)
    ego_raw_F_R = torch.randn(1, C_R, H, W)
    ego_enc_V = g1_encoder(ego_raw_F_V)
    ego_enc_F = g2_encoder(ego_raw_F_F)
    ego_enc_R = g3_encoder(ego_raw_F_R)
    ego_concatenated = torch.cat([ego_enc_V, ego_enc_F, ego_enc_R], dim=1)
    H_i = fusion_conv(ego_concatenated)

    # --- 2. Prepare Inputs for the Fusion Network ---
    ego_demand_input = torch.rand(1, 3, H, W)
    collab_F_V = torch.randn(N - 1, C_V, H, W)
    collab_F_F = torch.randn(N - 1, C_F, H, W)
    collab_F_R = torch.randn(N - 1, C_R, H, W)
    collab_features_raw = [collab_F_V, collab_F_F, collab_F_R]

    # Generate a valid mask
    raw_masks = torch.rand(N - 1, 3, H, W)
    mask_indices = torch.argmax(F.softmax(raw_masks, dim=1), dim=1)
    collab_masks_input = F.one_hot(mask_indices, num_classes=3).permute(0, 3, 1, 2).float()

    # --- 3. Instantiate and run the Fusion Network ---
    # === FIX 1 (continued): Pass `g_out` during instantiation ===
    fusion_network = DemandDrivenFusionNetwork(
        model_dim=C_H,
        g_out=C_ENCODED,
        num_heads=NUM_HEADS,
        num_sampling_points=NUM_SAMPLING_POINTS
    )

    H_refined_output = fusion_network(
        ego_features=H_i,
        ego_demand=ego_demand_input,
        collaborator_features=collab_features_raw,
        collaborator_masks=collab_masks_input,
        encoders=encoders_list
    )

    # --- 4. Pass the refined feature map to the Decoder ---
    decoder = DetectionDecoder(
        in_channels=C_H,
        num_classes=NUM_CLASSES,
        num_regression_params=NUM_REG_PARAMS
    )
    class_predictions, reg_predictions = decoder(H_refined_output)

    print("--- Full Pipeline Simulation ---")
    print(f"Ego Feature (H_i) Shape:           {H_i.shape}")
    print(f"Refined Feature (H_refined) Shape: {H_refined_output.shape}")
    print(f"Classification Output Shape:       {class_predictions.shape}")
    print(f"Regression Output Shape:           {reg_predictions.shape}")

    assert H_refined_output.shape == H_i.shape
    assert class_predictions.shape == (1, NUM_CLASSES, H, W)
    assert reg_predictions.shape == (1, NUM_REG_PARAMS, H, W)
    print("\n✅ Full pipeline executed successfully. All shapes are correct.")
