import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True, use_relu=True):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BEVPatchEmbed(nn.Module):
    def __init__(self, C_in, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(C_in, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, E, H', W']
        return x.flatten(2).transpose(1, 2)  # [B, H'*W', E]

class TemporalEncoderNtemp(nn.Module):
    def __init__(self, C_total_bev, D_temporal, num_layers=2, num_heads=4, dim_feedforward_factor=4):
        super().__init__()
        # Example: A simple ConvLSTM or a Transformer-based encoder for sequence of BEV maps
        # For simplicity, let's use a 3D Conv + Transformer approach
        self.input_proj = nn.Conv2d(C_total_bev, D_temporal, kernel_size=1)  # Project to D_temporal

        # Patching for Transformer
        # Assuming BEV_H, BEV_W, patch_size are known
        # self.patch_embed = BEVPatchEmbed(D_temporal, D_temporal, patch_size)
        # For simplicity, let's use global average pooling per frame + MLP for now
        # Or a full Transformer encoder if complexity allows

        # Simplified: Average historical frames and process with a Conv
        self.temporal_agg_conv = nn.Sequential(
            BasicConvBlock(D_temporal, D_temporal),
            BasicConvBlock(D_temporal, D_temporal)
        )
        # More advanced: A proper spatio-temporal Transformer
        # encoder_layer = nn.TransformerEncoderLayer(d_model=D_temporal, nhead=num_heads,
        #                                          dim_feedforward=D_temporal*dim_feedforward_factor,
        #                                          batch_first=True) # if tokens are [B*T, NumPatches, D]
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, historical_X_ego_list):
        # historical_X_ego_list: list of [1, C_total_bev, H, W] tensors from Ego's past
        if not historical_X_ego_list:
            # Return a zero tensor of expected shape if no history
            # This needs to be handled based on where historical_X_ego_list comes from
            # For now, assume it's not empty.
            # Example: B, C, H, W = 1, self.input_proj.out_channels, some_H, some_W
            # return torch.zeros(B, C, H, W, device=self.input_proj.conv.weight.device)
            raise ValueError("TemporalEncoderNtemp received empty history.")

        projected_history = [self.input_proj(x) for x in historical_X_ego_list]  # List of [1, D_temp, H, W]

        # Simple aggregation: average projected features
        stacked_history = torch.stack(projected_history, dim=2)  # [1, D_temp, K, H, W] (K = num_history_frames)
        avg_history_feat = torch.mean(stacked_history, dim=2)  # [1, D_temp, H, W]

        H_temporal = self.temporal_agg_conv(avg_history_feat)  # [1, D_temp, H, W]
        return H_temporal


class ConvHead(nn.Module):
    def __init__(self, in_channels, out_channels=1, use_sigmoid=True):
        super().__init__()
        self.conv = nn.Sequential(
            BasicConvBlock(in_channels, in_channels // 2),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class FuseEgoState(nn.Module):
    def __init__(self, C_total_bev, D_temporal, C_enhanced_out):
        super().__init__()
        # Fuse X_current_bev, M_occ, M_ano
        # M_occ and M_ano are single channel maps
        self.fusion_net = nn.Sequential(
            BasicConvBlock(C_total_bev + D_temporal + 1 + 1, C_enhanced_out * 2),  # X, H_temp (for M_occ), M_occ, M_ano
            SEBlock(C_enhanced_out * 2),
            BasicConvBlock(C_enhanced_out * 2, C_enhanced_out)
        )
        # Simpler if M_occ, M_ano are derived from H_temporal only
        # Then input to fusion_net is C_total_bev + 1 + 1
        self.fusion_net_v2 = nn.Sequential(
            BasicConvBlock(C_total_bev + 1 + 1, C_enhanced_out * 2),  # X_current, M_occ, M_ano
            SEBlock(C_enhanced_out * 2),
            BasicConvBlock(C_enhanced_out * 2, C_enhanced_out)
        )

    def forward(self, X_current_bev, M_occ, M_ano):
        # X_current_bev: [1, C_total, H, W]
        # M_occ: [1, 1, H, W]
        # M_ano: [1, 1, H, W]
        combined = torch.cat([X_current_bev, M_occ, M_ano], dim=1)
        return self.fusion_net_v2(combined)


class CrossAgentTransformerInteraction(nn.Module):
    def __init__(self, C_q_channels, C_kv_channels, C_hidden, num_layers=2, num_heads=4, dim_feedforward_factor=4,
                 patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.q_patch_embed = BEVPatchEmbed(C_q_channels, C_hidden, patch_size)
        self.kv_patch_embed = BEVPatchEmbed(C_kv_channels, C_hidden, patch_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=C_hidden, nhead=num_heads,
            dim_feedforward=C_hidden * dim_feedforward_factor, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Positional embeddings might be needed if not implicitly learned by convs in BEVPatchEmbed
        # self.q_pos_embed = nn.Parameter(...)
        # self.kv_pos_embed = nn.Parameter(...)

    def forward(self, query_bev, key_value_bev_list):
        # query_bev (X_ego_enhanced): [1, C_q, H, W]
        # key_value_bev_list (compensated X_j_bev): list of [1, C_kv, H, W]

        q_tokens = self.q_patch_embed(query_bev)  # [1, NumPatches_q, C_hidden]

        if not key_value_bev_list:  # No collaborators
            # If no collaborators, the "fused_context" is just based on ego
            # This needs careful thought: what should H_fused_context be?
            # Maybe just return q_tokens reshaped, or a projection of it.
            # For now, let'. assume the transformer handles empty memory.
            # Or, return q_tokens and let subsequent MLPs handle it.
            # Let's assume for now this won't be called if key_value_bev_list is empty,
            # or the calling HCMGF handles it.
            # If it must run, memory can be a zero tensor.
            # This will likely result in output being similar to query.
            num_q_patches = q_tokens.shape[1]
            memory_tokens = torch.zeros(1, 0, q_tokens.shape[2], device=q_tokens.device, dtype=q_tokens.dtype)

        else:
            kv_tokens_list = [self.kv_patch_embed(kv_bev) for kv_bev in key_value_bev_list]
            memory_tokens = torch.cat(kv_tokens_list, dim=1)  # [1, NumPatches_kv_total, C_hidden]

        # Add positional embeddings if used
        # q_tokens = q_tokens + self.q_pos_embed
        # memory_tokens = memory_tokens + self.kv_pos_embed (more complex if variable length)

        # TransformerDecoder expects target (query) and memory (key/value)
        # Output will have same seq_len as query_tokens
        output_tokens = self.transformer_decoder(tgt=q_tokens, memory=memory_tokens)  # [1, NumPatches_q, C_hidden]

        # Reassemble to BEV map
        # This requires knowing H', W' from patch_embed
        B, NumPatches, C = output_tokens.shape
        # Assuming NumPatches = H/patch_size * W/patch_size
        H_prime = query_bev.shape[2] // self.patch_size
        W_prime = query_bev.shape[3] // self.patch_size
        assert NumPatches == H_prime * W_prime

        H_fused_context = output_tokens.transpose(1, 2).reshape(B, C, H_prime, W_prime)
        # Upsample if needed, or use in token form for MLP_GN etc.
        # For simplicity, let's assume MLPs operate on per-patch token features, or upsample H_fused_context
        # The paper implies H_fused_context is a BEV map.
        # So we might need an upsampling/refinement layer here if patch_size > 1
        # For now, let's assume patch_size=1 or MLPs take token features.
        # If patch_size > 1:
        # H_fused_context = F.interpolate(H_fused_context,
        #                                 size=(query_bev.shape[2], query_bev.shape[3]),
        #                                 mode='bilinear', align_corners=False)
        return H_fused_context  # [1, C_hidden, H_prime, W_prime] or [1, C_hidden, H, W]


class MLPDecoder(nn.Module):
    def __init__(self, C_in_hidden, D_out, num_layers=2):
        super().__init__()
        layers = [nn.Linear(C_in_hidden, C_in_hidden // 2), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(C_in_hidden // 2, C_in_hidden // 2), nn.ReLU()])
        layers.append(nn.Linear(C_in_hidden // 2, D_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, H_fused_context_region_features):
        # H_fused_context_region_features: [N_regions, C_in_hidden]
        return self.mlp(H_fused_context_region_features)


class FuseWeightGenerator(nn.Module):
    def __init__(self, D_gn, D_occ, D_ano, D_uncertainty=1, D_consistency=1, D_weight_hidden=64):
        super().__init__()
        # V_GN, V_Occ, V_Ano are per-region. U_bev, M_consistency are per-region, per-collaborator.
        # This implies weights are generated per-region, per-collaborator.
        # Input to MLP should be features for a specific (region r, collaborator j)
        input_dim = D_gn + D_occ + D_ano + D_uncertainty + D_consistency
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, D_weight_hidden),
            nn.ReLU(),
            nn.Linear(D_weight_hidden, 1),  # Output a single logit for the weight
            # Sigmoid is applied in the paper, but for softmax normalization later, raw logits are fine.
        )

    def forward(self, V_gn_r, V_occ_r, V_ano_r, U_j_bev_r, M_j_consistency_r):
        # All inputs are for a specific region r and collaborator j (or derived from global context for r)
        # Shapes: V_gn_r [D_gn], V_occ_r [D_occ], V_ano_r [D_ano]
        # U_j_bev_r [D_uncertainty], M_j_consistency_r [D_consistency]
        combined = torch.cat([V_gn_r, V_occ_r, V_ano_r, U_j_bev_r, M_j_consistency_r], dim=-1)
        return self.mlp(combined)  # Returns logit for weight w_r,j->i


class HistoricalContextualMultiGranularityFuser(nn.Module):
    def __init__(self, args_hcmgf):
        super().__init__()
        self.C_total_bev = args_hcmgf['C_total_bev']  # Channel of X_bev (current and compensated)
        self.D_temporal = args_hcmgf['D_temporal']
        self.C_enhanced_ego = args_hcmgf['C_enhanced_ego']  # Output of FuseEgoState
        self.C_hidden_transformer = args_hcmgf['C_hidden_transformer']  # For CrossAgentTransformer

        self.D_gn = args_hcmgf['D_gn']
        self.D_occ_vec = args_hcmgf['D_occ_vec']  # Renamed from D_occ to avoid confusion with M_occ map
        self.D_ano_vec = args_hcmgf['D_ano_vec']

        self.temporal_encoder = TemporalEncoderNtemp(self.C_total_bev, self.D_temporal)
        self.occ_head = ConvHead(self.D_temporal, 1, use_sigmoid=True)  # For M_occ
        self.ano_head = ConvHead(self.D_temporal + self.C_total_bev, 1, use_sigmoid=True)  # For M_ano

        self.fuse_ego_state = FuseEgoState(self.C_total_bev, self.D_temporal, self.C_enhanced_ego)

        self.cross_agent_transformer = CrossAgentTransformerInteraction(
            self.C_enhanced_ego, self.C_total_bev, self.C_hidden_transformer,
            patch_size=args_hcmgf.get('fusion_patch_size', 1)  # if 1, no patching, direct conv for "embed"
        )

        # MLPs to decode V_GN, V_Occ, V_Ano from H_fused_context
        # These will operate on per-pixel/per-patch features of H_fused_context
        # If H_fused_context is [B, C_hidden, H', W'], then we need to adapt.
        # For now, assume these MLPs take [C_hidden] and output [D_X]
        self.mlp_gn = MLPDecoder(self.C_hidden_transformer, self.D_gn)
        self.mlp_occ_vec = MLPDecoder(self.C_hidden_transformer, self.D_occ_vec)  # For V_Occ,r
        self.mlp_ano_vec = MLPDecoder(self.C_hidden_transformer, self.D_ano_vec)  # For V_Ano,r

        # For self-supervised losses
        self.decoder_occ_recon = MLPDecoder(self.D_occ_vec, self.C_total_bev)  # Reconstruct X_j_bev from V_Occ
        self.proj_hist_occ_consistency = MLPDecoder(self.D_occ_vec, args_hcmgf['D_consistency_proj'])
        self.proj_curr_occ_consistency = MLPDecoder(self.C_total_bev, args_hcmgf['D_consistency_proj'])
        # For ano_contrastive, assume PositiveAnomalyFeatures etc. are handled externally or via hooks
        self.predictor_ano_diff = MLPDecoder(self.D_ano_vec, 1)  # Predict scalar distance

        self.fuse_weight_generator = FuseWeightGenerator(
            self.D_gn, self.D_occ_vec, self.D_ano_vec,
            D_uncertainty=1,  # Assuming U_j_bev is single channel
            D_consistency=1  # Assuming M_j_consistency is single channel
        )

        self.final_fusion_conv = BasicConvBlock(self.C_total_bev, args_hcmgf['C_final_fused_bev'])

        # Store channel info for splitting X_bev if needed by self-supervised losses
        self.C_V = args_hcmgf['C_V']
        self.C_F = args_hcmgf['C_F']
        self.C_D = args_hcmgf['C_D']

        # For simplicity, self-supervised losses are calculated outside or via specific methods.
        # This forward will focus on the main fusion path.

    def forward(self, current_compensated_features_list, ego_historical_X_list,
                collaborator_uncertainties_list=None,  # List of U_j_bev [1,1,H,W]
                collaborator_consistency_maps_list=None):  # List of M_j_consistency [1,1,H,W]
        # current_compensated_features_list:
        #   - elem 0: Ego's X_i^t [1, C_total, H, W]
        #   - elem 1..N: Collaborator j's compensated X_j_bev^t [1, C_total, H, W]
        # ego_historical_X_list: List of Ego's past X_i^{t-k} [1, C_total, H, W]

        X_ego_current_bev = current_compensated_features_list[0]
        compensated_collab_bevs = current_compensated_features_list[1:]

        # 1. Ego-Agent Temporal Enhancement
        H_ego_temporal = self.temporal_encoder(ego_historical_X_list)  # [1, D_temp, H, W]
        M_ego_occ = self.occ_head(H_ego_temporal)  # [1, 1, H, W]
        M_ego_ano = self.ano_head(torch.cat([H_ego_temporal, X_ego_current_bev], dim=1))  # [1, 1, H, W]

        X_ego_enhanced = self.fuse_ego_state(X_ego_current_bev, M_ego_occ, M_ego_ano)  # [1, C_enhanced_ego, H, W]

        # 2. Ego-Collaborator Cross-Agent Interaction
        H_fused_context_bev = self.cross_agent_transformer(X_ego_enhanced, compensated_collab_bevs)
        # H_fused_context_bev: [1, C_hidden_transformer, H_prime, W_prime] or [1, C_hidden_transformer, H, W]
        # Let's assume it's [1, C_hidden, H, W] for now (e.g. patch_size=1 or upsampled)
        # If it's [H_prime, W_prime], need to handle per-patch features for V_GN etc.
        # For simplicity, assume H_fused_context_bev is full BEV resolution.
        # If not, MLPs for V_GN etc. would operate on flattened tokens from H_fused_context_bev.

        # For per-region (pixel) V_GN, V_Occ, V_Ano:
        B, C_hidden, H, W = H_fused_context_bev.shape
        h_fused_flat_pixels = H_fused_context_bev.permute(0, 2, 3, 1).reshape(B * H * W,
                                                                              C_hidden)  # [N_pixels, C_hidden]

        V_gn_all_pixels = self.mlp_gn(h_fused_flat_pixels).reshape(B, H, W, self.D_gn)
        V_occ_vec_all_pixels = self.mlp_occ_vec(h_fused_flat_pixels).reshape(B, H, W, self.D_occ_vec)
        V_ano_vec_all_pixels = self.mlp_ano_vec(h_fused_flat_pixels).reshape(B, H, W, self.D_ano_vec)

        # 4. Dynamic Weight Generation and Final Fusion
        all_agent_features_for_sum = [X_ego_current_bev] + compensated_collab_bevs
        num_total_agents_for_sum = len(all_agent_features_for_sum)

        # Store raw weighted features before normalization for sum
        weighted_features_sum = torch.zeros_like(X_ego_current_bev)  # [1, C_total, H, W]
        sum_of_raw_weights_map = torch.zeros(B, 1, H, W, device=X_ego_current_bev.device) + 1e-8  # For normalization

        # Ego's weight (alpha_ego in paper)
        # Can be fixed, learned, or part of softmax. Let's make it learnable via the MLP.
        # For simplicity, let's assume ego has a fixed base weight or it's handled by softmax over all.

        # For each pixel (r_y, r_x):
        pixel_weights_logits_list = []

        # Ego "contribution" for weight generation (can be zeros or some self-assessment)
        # For simplicity, let's assume the weight for ego is also generated.
        # Or, as in paper, alpha_ego + sum(w_j * X_j).
        # We need a way to get V_GN_ego, U_ego, M_ego_consistency if ego is part of weighted sum.
        # Let's follow the formula: alpha_ego * X_ego + sum w_j * X_j
        # Alpha_ego could be 1.0 initially, and w_j are from MLP. Then normalize.

        # Weights for collaborators
        for j, X_j_bev_compensated in enumerate(compensated_collab_bevs):
            # Get U_j_bev[r] and M_j_consistency[r] for this collaborator
            # These need to be passed in or calculated.
            # Assuming they are [1,1,H,W] maps.
            U_j_bev_r_map = collaborator_uncertainties_list[j] if collaborator_uncertainties_list else torch.zeros_like(
                M_ego_occ)
            M_j_consistency_r_map = collaborator_consistency_maps_list[
                j] if collaborator_consistency_maps_list else torch.ones_like(M_ego_occ)

            # Permute for cat: [B,H,W,C]
            V_gn_r_map_p = V_gn_all_pixels.permute(0, 3, 1, 2)  # [1, D_gn, H, W]
            V_occ_r_map_p = V_occ_vec_all_pixels.permute(0, 3, 1, 2)
            V_ano_r_map_p = V_ano_vec_all_pixels.permute(0, 3, 1, 2)

            # Concat all inputs for MLP_fuse_weight for all pixels at once
            # Input to MLP_fuse_weight is [D_gn + D_occ + D_ano + 1 (unc) + 1 (cons)]
            # We need to cat [1, D_gn, H, W], [1, D_occ, H, W], ..., [1, 1, H, W]
            mlp_weight_input_j = torch.cat([
                V_gn_r_map_p, V_occ_r_map_p, V_ano_r_map_p,
                U_j_bev_r_map, M_j_consistency_r_map
            ], dim=1)  # [1, D_input_mlp_weight, H, W]

            # Reshape for MLP: [B*H*W, D_input_mlp_weight]
            mlp_weight_input_j_flat = mlp_weight_input_j.permute(0, 2, 3, 1).reshape(B * H * W, -1)

            w_r_j_logits_flat = self.fuse_weight_generator.mlp(mlp_weight_input_j_flat)  # [B*H*W, 1]
            w_r_j_logits_map = w_r_j_logits_flat.reshape(B, H, W, 1).permute(0, 3, 1, 2)  # [1,1,H,W]
            pixel_weights_logits_list.append(w_r_j_logits_map)

        # Softmax normalization of weights (Ego + Collaborators)
        # Let alpha_ego_logit be a learnable parameter or fixed
        alpha_ego_logit_map = torch.zeros(B, 1, H, W, device=X_ego_current_bev.device)  # Example: fixed ego logit

        all_logits_for_softmax = [alpha_ego_logit_map] + pixel_weights_logits_list
        stacked_logits = torch.cat(all_logits_for_softmax, dim=1)  # [B, NumTotalAgentsForSum, H, W]

        normalized_weights = F.softmax(stacked_logits, dim=1)  # Softmax over agents for each pixel

        alpha_ego_normalized = normalized_weights[:, 0:1, :, :]  # [B,1,H,W]
        collab_weights_normalized_list = [normalized_weights[:, j + 1:j + 2, :, :] for j in
                                          range(len(compensated_collab_bevs))]

        # Final weighted sum
        final_fused_bev_sum = alpha_ego_normalized * X_ego_current_bev
        for j, X_j_bev_compensated in enumerate(compensated_collab_bevs):
            final_fused_bev_sum += collab_weights_normalized_list[j] * X_j_bev_compensated

        # Final projection
        final_fused_bev_out = self.final_fusion_conv(final_fused_bev_sum)

        # --- Self-supervised losses would be calculated here or in a separate method ---
        # Using V_occ_vec_all_pixels, V_ano_vec_all_pixels, M_ego_occ, M_ego_ano,
        # compensated_collab_bevs, ego_historical_X_list etc.
        # This part is complex and depends on how GT/pseudo-GT for these losses are formed.
        # For now, we return the fused BEV. Losses can be added as separate outputs.

        # Placeholder for self-supervised losses
        ss_losses = {
            "occ_recon": torch.tensor(0.0, device=final_fused_bev_out.device),
            "occ_consistency": torch.tensor(0.0, device=final_fused_bev_out.device),
            "ano_contrastive": torch.tensor(0.0, device=final_fused_bev_out.device),
            "ano_diff_pred": torch.tensor(0.0, device=final_fused_bev_out.device),
        }

        return final_fused_bev_out, ss_losses
