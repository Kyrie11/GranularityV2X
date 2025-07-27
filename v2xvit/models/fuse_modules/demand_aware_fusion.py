import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DemandAwareCrossAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, demand_dim: int = 3):
        """
        Args:
            model_dim (int): The channel dimension of the unified features (C_UNIFIED).
            num_heads (int): The number of parallel attention heads (M).
            demand_dim (int): The channel dimension of the demand map (usually 3).
        """
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # 1. f_enc: Lightweight MLP to encode the demand map
        # We use a 1x1 Conv to act as a per-pixel MLP
        self.demand_encoder = nn.Sequential(
            nn.Conv2d(demand_dim, self.model_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.model_dim // 2, self.model_dim, kernel_size=1)
        )

        # 2. P_pos: Learnable positional embedding (initialized to zeros)
        # The network will learn spatial biases during training.
        # Shape: [1, C_UNIFIED, H, W] - will be dynamically inferred in forward pass
        self.positional_embedding = None

        # 3. Projection matrices W_m^Q, W_m^K, W_m^V
        # We use a single large Conv2d for efficiency and split the output.
        self.q_proj = nn.Conv2d(self.model_dim, self.model_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(self.model_dim, self.model_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(self.model_dim, self.model_dim, kernel_size=1)

        # 4. W^O: Output projection layer
        self.output_proj = nn.Conv2d(self.model_dim, self.model_dim, kernel_size=1)

    def forward(self, ego_features, ego_demand, collaborator_features):
        """
        Args:
            ego_features (torch.Tensor): The ego agent's unified feature map (H_hat_i).
                                         Shape: [1, C_UNIFIED, H, W].
            ego_demand (torch.Tensor): The ego agent's granular demand map (D_i).
                                       Shape: [1, 3, H, W].
            collaborator_features (torch.Tensor): The confidence-aware unified features
                                                  from all collaborators (H''_j).
                                                  Shape: [N-1, C_UNIFIED, H, W].

        Returns:
            torch.Tensor: The aggregated collaborative feature map (H_collab).
                          Shape: [1, C_UNIFIED, H, W].
        """

        num_collabs, C, H, W = collaborator_features.shape
        # Initialize positional embedding if it's the first run
        if self.positional_embedding is None:
            self.positional_embedding = nn.Parameter(torch.zeros(1, C, H, W, device=ego_features.device))

        # --- 1. Formulate the comprehensive Query state H'_i ---
        encoded_demand = self.demand_encoder(ego_demand)
        # Note: We use simple addition here instead of concatenation + projection
        # as it is more parameter-efficient and achieves the same goal of conditioning.
        # H'_i = Concat(H_hat_i, f_enc(D_i)) + P_pos
        query_state = ego_features + encoded_demand + self.positional_embedding

        # --- 2. Project to Q, K, V for all heads ---
        # Project and then split into M heads.
        # [B, C, H, W] -> [B, M, C_head, H, W]
        q = self.q_proj(query_state).view(1, self.num_heads, self.head_dim, H, W)
        k = self.k_proj(collaborator_features).view(num_collabs, self.num_heads, self.head_dim, H, W)
        v = self.v_proj(collaborator_features).view(num_collabs, self.num_heads, self.head_dim, H, W)

        # --- 3. Compute Attention Scores and Aggregate Values ---
        # Scaling factor
        d_k = self.head_dim
        scale = math.sqrt(d_k)

        # Expand q for broadcasting over collaborators: [1, 1, M, C_head, H, W]
        q_expanded = q.unsqueeze(1)

        # Compute scaled dot-product attention scores
        # Element-wise product and sum over head_dim is equivalent to batched dot-product
        # (q_expanded * k) -> [1, N-1, M, C_head, H, W]
        # torch.sum(...) -> [1, N-1, M, H, W]
        attn_scores = torch.sum(q_expanded * k, dim=3) / scale

        # Softmax over the collaborator dimension (dim=1)
        attention_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum of values
        # Unsqueeze weights and v for broadcasting
        # weights: [1, N-1, M, 1, H, W], v: [1, N-1, M, C_head, H, W]
        # output:  [1, N-1, M, C_head, H, W]
        weighted_v = attention_weights.unsqueeze(3) * v.unsqueeze(0)

        # Sum over the collaborator dimension (dim=1) to get head outputs
        # Shape: [1, M, C_head, H, W]
        head_outputs = torch.sum(weighted_v, dim=1)

        # --- 4. Concatenate heads and final projection ---
        # Reshape to [1, M * C_head, H, W] = [1, C_UNIFIED, H, W]
        concatenated_heads = head_outputs.reshape(1, self.model_dim, H, W)

        #  output projection W^O
        H_collab = self.output_proj(concatenated_heads)

        return H_collab

class STDNet(nn.Module):
    """
    Implements the full Spatio-Temporal and Demand-Aware Fusion Network.
    """

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.fusion_attention = DemandAwareCrossAttention(model_dim=model_dim, num_heads=num_heads)
        #  LayerNorm for the residual connection
        self.final_layernorm = nn.LayerNorm(model_dim)

    def forward(self, ego_features, ego_demand, collaborator_features_list, collaborator_masks, encoders, fusion_conv):
        """
        Args:
            ego_features (torch.Tensor): Ego agent's enhanced feature map (H_hat_i).
                                         You mentioned this is already implemented.
                                         Shape: [1, C_UNIFIED, H, W].
            ego_demand (torch.Tensor): Ego agent's demand map.
                                       Shape: [1, 3, H, W].
            collaborator_features_list (list[torch.Tensor]): List of sparse features.
                - [features_g1, features_g2, features_g3]
                - Each tensor is shape [N-1, C_g, H, W].
            collaborator_masks (torch.Tensor): Mask indicating data presence.
                                               Shape: [N-1, 3, H, W].

        Returns:
            torch.Tensor: The final, enhanced ego-agent feature map.
                          Shape: [1, C_UNIFIED, H, W].
        """
        # --- Stage 1: Confidence-Aware Feature Encoding ---
        modulated_features_per_granularity = []

        for g_idx in range(3):  # Iterate over 3 granularities
            features_g = collaborator_features_list[g_idx]  # [N-1, C_g, H, W]
            encoder_g = encoders[g_idx]
            # Mask for current granularity, unsqueezed for broadcasting
            # Shape: [N-1, 1, H, W]
            mask_g = collaborator_masks[:, g_idx:g_idx + 1, :, :]
            # Encode features: Epsilon_g(F''_j,g)
            encoded_features = encoder_g(features_g)  # [N-1, C_ENCODED, H, W]

            # Only consider non-zero data for consensus
            encoded_features_masked = encoded_features * mask_g

            # Compute Consensus Map C_g
            # Sum features and masks across the collaborator dimension (dim=0)
            summed_features = encoded_features_masked.sum(dim=0)  # [C_ENCODED, H, W]
            num_agents_per_pixel = mask_g.sum(dim=0).clamp(min=1e-8)  # [1, H, W]
            consensus_map_g = summed_features / num_agents_per_pixel  # [C_ENCODED, H, W]

            # Compute Confidence Score E_j,g using cosine similarity
            # F.cosine_similarity needs tensors of same shape or broadcastable
            # It computes similarity along dim=1 (channels) by default
            similarity = F.cosine_similarity(encoded_features, consensus_map_g.unsqueeze(0), dim=1)
            # Add a dimension for channel-wise multiplication: [N-1, H, W] -> [N-1, 1, H, W]
            confidence_score_g = 0.5 * (1 + similarity.unsqueeze(1))

            # Modulate features with confidence
            # Only apply confidence where data exists (important for sparsity)
            modulated_features = encoded_features * confidence_score_g * mask_g
            modulated_features_per_granularity.append(modulated_features)
        # Concatenate modulated granularities along the channel dimension
        concatenated_modulated = torch.cat(modulated_features_per_granularity, dim=1)

        # Unify to create H''_j
        H_j_prime_prime = fusion_conv(concatenated_modulated)  # [N-1, C_UNIFIED, H, W]

        # --- Stage 2: Demand-Driven Multi-Head Attention Fusion ---
        H_collab = self.fusion_attention(
            ego_features=ego_features,
            ego_demand=ego_demand,
            collaborator_features=H_j_prime_prime
        )

        # ---  Residual Connection ---
        # LayerNorm expects [B, C, H, W] -> [B, H, W, C] -> norm -> [B, C, H, W]
        normed_collab = self.final_layernorm(H_collab.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        H_final = ego_features + normed_collab

        return H_final