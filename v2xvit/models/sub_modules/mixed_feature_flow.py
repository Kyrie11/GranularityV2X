import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

class BEVPatchEmbed(nn.Module):
    def __init__(self, C_in, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(C_in, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # [B, E, H', W']
        return x.flatten(2).transpose(1,2) # [B, H'*W', E]

class MixedHistoryTokenizer(nn.Module):
    def __init__(self, C_total_list, embed_dim, patch_size, max_seq_len,
                    num_recent_frames, num_distant_frames):
        super().__init__()
        self.patch_embed = BEVPatchEmbed(C_total_list, embed_dim, patch_size)
        self.max_seq_len = max_seq_len

        #learnable embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim)) #Positional for flattened sequence
        self.time_embed = nn.Embedding(num_recent_frames+num_distant_frames+1, embed_dim) #+1 for Safety
        self.seq_type_embed = nn.Embedding(2, embed_dim) #0 for recent, 1 for distant

        #Store these for mapping global time index to local embedding index
        self.num_recent_frames = num_recent_frames
        self.num_distant_frames = num_distant_frames

    def forward(self, historical_X_j_list_with_time_info):
        # historical_X_j_list_with_time_info: list of tuples
        # each tuple: (X_j_k [1, C_total, H, W], global_time_idx_k, seq_type_idx_k [0 or 1])
        # seq_type_idx_k: 0 for recent, 1 for distant
        # global_time_idx_k: unique index for each historical frame to feed time_embed

        all_tokens = []
        time_embeddings_to_add = []
        seq_type_embeddings_to_add = []

        current_token_count = 0
        for x_bev_k, time_idx_k, seq_type_k in historical_X_j_list_with_time_info:
            tokens_k = self.patch_embed(x_bev_k)  # [1, num_patches_k, E]
            num_patches_k = tokens_k.shape[1]

            if current_token_count + num_patches_k > self.max_seq_len:
                # Truncate if exceeding max_seq_len (simple strategy)
                tokens_k = tokens_k[:, :self.max_seq_len - current_token_count, :]
                num_patches_k = tokens_k.shape[1]
                if num_patches_k == 0: continue

            all_tokens.append(tokens_k.squeeze(0))  # Remove batch dim [num_patches_k, E]

            # Prepare embeddings to add (broadcast to num_patches_k)
            time_emb = self.time_embed(torch.tensor(time_idx_k, device=x_bev_k.device)).unsqueeze(0).repeat(
                num_patches_k, 1)
            seq_type_emb = self.seq_type_embed(torch.tensor(seq_type_k, device=x_bev_k.device)).unsqueeze(0).repeat(
                num_patches_k, 1)

            time_embeddings_to_add.append(time_emb)
            seq_type_embeddings_to_add.append(seq_type_emb)
            current_token_count += num_patches_k

        if not all_tokens:  # Should not happen if history is provided
            # Return a dummy sequence or handle error
            # For now, let's assume it won't be empty.
            # If it can be, need a robust way to handle this (e.g., return None and skip compensation)
            return torch.empty(0, self.pos_embed.shape[-1], device=self.pos_embed.device), \
                torch.empty(0, dtype=torch.long, device=self.pos_embed.device)

        # Concatenate all tokens and their respective embeddings
        final_tokens = torch.cat(all_tokens, dim=0)  # [Total_Patches, E]
        final_time_embed = torch.cat(time_embeddings_to_add, dim=0)
        final_seq_type_embed = torch.cat(seq_type_embeddings_to_add, dim=0)

        # Add all embeddings
        final_tokens = final_tokens + self.pos_embed[:, :final_tokens.shape[0], :] + \
                       final_time_embed + final_seq_type_embed

        # Store patch counts for easy reshaping later
        # (This part is tricky if patch counts per frame vary and we need to reconstruct per-frame maps from T_out)
        # For H_short, z_long, z_obj, we need to know which tokens in T_out correspond to which original frame/type.
        # A simpler way might be to pass indices or masks.
        # For now, let's assume the decoders can handle a flat T_out and use masks/indices if needed.
        # An alternative: return T_out and a list of (start_idx, end_idx, seq_type) for each frame's tokens in T_out.

        # Let's return the flattened tokens and a list indicating token origins
        token_origins = []  # list of (start_idx, num_patches, seq_type_k, original_time_idx_k)
        start_idx = 0
        for i, (x_bev_k, time_idx_k, seq_type_k) in enumerate(historical_X_j_list_with_time_info):
            # Recalculate num_patches after potential truncation
            # This is complex if truncation happened mid-list.
            # A fixed number of patches per frame simplifies this, or padding.
            # For now, assume patch_embed output is consistent or handled.
            # This part needs careful implementation of how tokens are mapped back.
            # Let's assume for now that the decoders get the full T_out and select based on seq_type.
            pass  # Placeholder for more complex token origin tracking

        return final_tokens.unsqueeze(0), None  # [1, Total_Patches, E], Placeholder for origin info

        # --- Transformer Encoder ---

    class TemporalTransformerEncoder(nn.Module):
        def __init__(self, embed_dim, num_heads, num_layers, dim_feedforward, dropout=0.1):
            super().__init__()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True  # Important: batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def forward(self, src_tokens):  # src_tokens: [B, SeqLen, E]
            return self.transformer_encoder(src_tokens)

        # --- Context Decoders ---

    class ShortTermDecoder(nn.Module):  # Decodes T_out to H_short
        def __init__(self, embed_dim, D_short, H_out, W_out, patch_size_ H_prime_W_prime_in_transformer

        ):
        super().__init__()
        # This needs to know the H', W' of the token grid from patch_embed
        # And the target H, W for H_short.
        # Example: Transposed convolutions to upsample and reshape
        # This is highly dependent on how tokens from recent frames are selected from T_out
        # and how many patches there were.
        # Simplified: assume we get a [B, NumRecentPatches, E] tensor
        # and NumRecentPatches = H_prime * W_prime (for a single recent frame, e.g., t0)
        self.H_prime, self.W_prime = H_prime_W_prime_in_transformer  # e.g. (H/patch_size, W/patch_size)
        self.embed_dim = embed_dim
        self.decoder = nn.Sequential(
            # Example: Reshape, then ConvTranspose
            nn.ConvTranspose2d(embed_dim, D_short * 2, kernel_size=patch_size, stride=patch_size),
            # Upsample to original H,W
            nn.ReLU(),
            nn.Conv2d(D_short * 2, D_short, kernel_size=3, padding=1)
        )

    def forward(self, t_out_recent_tokens):  # [B, NumRecentPatches, E]
        B, NumRecentPatches, E = t_out_recent_tokens.shape
        # Assert NumRecentPatches == self.H_prime * self.W_prime
        x = t_out_recent_tokens.transpose(1, 2).reshape(B, E, self.H_prime, self.W_prime)
        return self.decoder(x)  # [B, D_short, H, W]


class LongTermAggregator(nn.Module):  # Aggregates T_out to z_long
    def __init__(self, embed_dim, D_long):
        super().__init__()
        # Example: Attention pooling or simple mean pooling
        self.pooler = nn.AdaptiveAvgPool1d(1)  # Pool across sequence length
        self.fc = nn.Linear(embed_dim, D_long)

    def forward(self, t_out_distant_tokens):  # [B, NumDistantPatches, E]
        if t_out_distant_tokens.shape[1] == 0:  # No distant tokens
            return torch.zeros(t_out_distant_tokens.shape[0], self.fc.out_features, device=t_out_distant_tokens.device)
        pooled = self.pooler(t_out_distant_tokens.transpose(1, 2)).squeeze(-1)  # [B, E]
        return self.fc(pooled)  # [B, D_long]


class ObjectContextExtractor(nn.Module):  # Extracts z_obj from T_out and F_det_bev_t0
    def __init__(self, embed_dim, D_obj, bev_h_prime, bev_w_prime, patch_size, roi_output_size=7):
        super().__init__()
        self.bev_h_prime = bev_h_prime
        self.bev_w_prime = bev_w_prime
        self.patch_size = patch_size  # To map RoI coords to token grid
        self.roi_align = RoIAlign((roi_output_size, roi_output_size),
                                  spatial_scale=1.0 / patch_size,  # Scale RoI coords to token grid
                                  sampling_ratio=-1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * roi_output_size * roi_output_size, D_obj * 2),
            nn.ReLU(),
            nn.Linear(D_obj * 2, D_obj)
        )

    def forward(self, t_out_t0_tokens, object_rois_at_t0):
        # t_out_t0_tokens: [1, NumT0Patches, E] (tokens from X_j^t0)
        # object_rois_at_t0: list of [K, 5] tensors (batch_idx, x1, y1, x2, y2) for K objects in BEV pixel coords
        # Assumes t_out_t0_tokens can be reshaped to a feature map [1, E, H', W']
        if not object_rois_at_t0 or object_rois_at_t0[0].numel() == 0:
            return []  # No objects or RoIs

        B, NumT0Patches, E = t_out_t0_tokens.shape
        # Assert NumT0Patches == self.bev_h_prime * self.bev_w_prime
        feature_map_t0 = t_out_t0_tokens.transpose(1, 2).reshape(B, E, self.bev_h_prime, self.bev_w_prime)

        # RoIAlign expects a list of RoIs for the batch
        # object_rois_at_t0 is assumed to be for batch_idx 0 if B=1
        aligned_features = self.roi_align(feature_map_t0, object_rois_at_t0)  # [NumTotalRoIs, E, roi_h, roi_w]
        aligned_features = aligned_features.view(aligned_features.size(0), -1)  # Flatten

        z_obj_list = []
        # If object_rois_at_t0 was a list of tensors, need to split aligned_features
        # Assuming object_rois_at_t0 is a single tensor [K, 5] for simplicity here.
        z_objs = self.fc(aligned_features)  # [K, D_obj]
        for i in range(z_objs.shape[0]):
            z_obj_list.append(z_objs[i])
        return z_obj_list

    # --- Motion Predictors ---


class ObjectMotionPredictor(nn.Module):  # MLP_obj_motion
    def __init__(self, D_obj, D_long, D_out_affine=6):  # e.g., dx,dy,dz,droll,dpitch,dyaw or 2D: dx,dy,dtheta
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D_obj + D_long + 1, (D_obj + D_long) * 2),  # +1 for delay
            nn.ReLU(),
            nn.Linear((D_obj + D_long) * 2, D_out_affine)
        )

    def forward(self, z_obj_o, z_long, delay_scalar):  # delay_scalar is (t-t0)
        # z_obj_o: [D_obj], z_long: [D_long]
        delay_tensor = torch.tensor([delay_scalar], device=z_obj_o.device, dtype=z_obj_o.dtype)
        combined_input = torch.cat([z_obj_o, z_long, delay_tensor], dim=0)
        return self.mlp(combined_input)


class BEVMotionPredictor(nn.Module):  # Predictor_bev_motion (U-Net like)
    def __init__(self, D_short, D_long, H_bev, W_bev, num_flow_channels=2, num_uncertainty_channels=1):
        super().__init__()
        # Inspired by ReduceInfTC or a simpler U-Net
        # Input channels: D_short (from H_short) + D_long (tiled) + 1 (tiled delay)
        # This is a placeholder, a proper U-Net is needed.
        # Can reuse/adapt parts of the original author's ReduceInfTC if suitable
        input_c = D_short + D_long + 1
        self.unet_like_encoder = nn.Sequential(  # Simplified
            BasicConvBlock(input_c, 64),
            BasicConvBlock(64, 128, stride=2),  # Downsample
            BasicConvBlock(128, 256, stride=2),  # Downsample
        )
        self.unet_like_decoder = nn.Sequential(  # Simplified
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            BasicConvBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            BasicConvBlock(64, 64),
        )
        self.flow_head = nn.Conv2d(64, num_flow_channels, kernel_size=3, padding=1)
        self.uncertainty_head = nn.Conv2d(64, num_uncertainty_channels, kernel_size=3, padding=1)

        # Initialize flow_head to output small initial flows
        nn.init.zeros_(self.flow_head.weight)
        if self.flow_head.bias is not None:
            nn.init.zeros_(self.flow_head.bias)

    def forward(self, H_short, z_long, delay_scalar):  # H_short [1,D_short,H,W], z_long [1,D_long]
        B, _, H, W = H_short.shape
        delay_map = torch.full((B, 1, H, W), delay_scalar, device=H_short.device, dtype=H_short.dtype)
        z_long_tiled = z_long.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)

        combined_input = torch.cat([H_short, z_long_tiled, delay_map], dim=1)

        encoded = self.unet_like_encoder(combined_input)
        decoded = self.unet_like_decoder(encoded)

        flow = self.flow_head(decoded)  # [B, 2, H, W]
        uncertainty = torch.sigmoid(self.uncertainty_head(decoded))  # [B, 1, H, W]
        return flow, uncertainty

    # --- Feature Warper (from original author, slightly adapted) ---


class FeatureWarper(nn.Module):
    def get_grid(self, B, H, W, device):
        shifts_x = torch.arange(0, W, 1, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, H, 1, dtype=torch.float32, device=device)
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')  # Use 'ij' for H,W order
        grid_dst = torch.stack((shifts_x, shifts_y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]
        return grid_dst

    def flow_warp(self, feats, flow, scaled_delay):  # flow is per unit time
        # feats: [B, C, H, W], flow: [B, 2, H, W] (dx, dy)
        B, C, H, W = feats.shape
        grid_dst = self.get_grid(B, H, W, feats.device)  # Base grid for destination

        # Total displacement: flow * delay
        total_displacement = flow * scaled_delay  # scaled_delay is (t-t0)

        # Source coordinates in the original feature map: destination_coords - displacement
        # grid_src = grid_dst - total_displacement # This is if flow points from src to dst
        # If flow points from t0 to t (i.e. where a point at t0 moves to at t),
        # then to get value at grid_dst(t), we need to sample from grid_dst(t) - displacement = grid_src(t0)
        # So, the sampling grid should be grid_dst - total_displacement

        # The original code's get_grid for flow_warp_feats implies flow is added to grid_dst
        # flow_grid = ((flow + grid_dst) / workspace - 1)
        # This means 'flow' is the coordinate of the source pixel for each dest pixel.
        # If our 'flow' is displacement (how much a pixel moves), then:
        # source_coord_for_dest_pixel = dest_pixel_coord - displacement
        # Let's assume our predicted flow O_j_bev is the displacement vector.

        # Normalize coordinates for grid_sample: range [-1, 1]
        # x_new = (x_old / (W-1)) * 2 - 1
        # y_new = (y_old / (H-1)) * 2 - 1

        # sampling_grid_pixels = grid_dst - total_displacement # These are pixel coordinates in source
        sampling_grid_pixels = grid_dst + total_displacement  # If flow is where points *come from*

        # Normalize sampling_grid_pixels to [-1, 1]
        sampling_grid_normalized_x = (sampling_grid_pixels[:, 0, :, :] / ((W - 1) / 2)) - 1
        sampling_grid_normalized_y = (sampling_grid_pixels[:, 1, :, :] / ((H - 1) / 2)) - 1

        sampling_grid_normalized = torch.stack(
            [sampling_grid_normalized_x, sampling_grid_normalized_y], dim=-1
        )  # [B, H, W, 2] (x_coords, y_coords) for F.grid_sample

        warped_feats = F.grid_sample(
            feats, sampling_grid_normalized, mode="bilinear", padding_mode="border", align_corners=True
        )
        return warped_feats

    # --- Main MGDC Module ---


class MultiGranularityDelayCompensation(nn.Module):
    def __init__(self, args_mgdc):  # args_mgdc should contain all sub-module configs
        super().__init__()
        self.C_total_hist = args_mgdc['C_total_hist']  # Channel of historical X_j^k
        self.embed_dim = args_mgdc['transformer_embed_dim']
        patch_size = args_mgdc['patch_size']

        # Assuming BEV H,W are known for patch calculations
        self.bev_h, self.bev_w = args_mgdc['bev_shape']
        self.h_prime = self.bev_h // patch_size
        self.w_prime = self.bev_w // patch_size

        self.tokenizer = MixedHistoryTokenizer(
            self.C_total_hist, self.embed_dim, patch_size,
            args_mgdc['max_seq_len'],
            args_mgdc['num_recent_frames_ks'],
            args_mgdc['num_distant_frames_kl']
        )
        self.transformer_encoder = TemporalTransformerEncoder(
            self.embed_dim, args_mgdc['transformer_num_heads'], args_mgdc['transformer_num_layers'],
            args_mgdc['transformer_dim_feedforward']
        )
        self.short_term_decoder = ShortTermDecoder(
            self.embed_dim, args_mgdc['D_short'],
            self.bev_h, self.bev_w, patch_size, (self.h_prime, self.w_prime)
        )
        self.long_term_aggregator = LongTermAggregator(self.embed_dim, args_mgdc['D_long'])

        self.object_context_extractor = ObjectContextExtractor(
            self.embed_dim, args_mgdc['D_obj'],
            self.h_prime, self.w_prime, patch_size,
            roi_output_size=args_mgdc.get('roi_align_output_size', 7)
        )
        self.object_motion_predictor = ObjectMotionPredictor(
            args_mgdc['D_obj'], args_mgdc['D_long'], args_mgdc.get('D_out_affine', 6)
        )
        self.bev_motion_predictor = BEVMotionPredictor(
            args_mgdc['D_short'], args_mgdc['D_long'], self.bev_h, self.bev_w
        )
        self.feature_warper = FeatureWarper()

        # Store which historical frames are recent/distant for T_out splitting
        self.ks = args_mgdc['num_recent_frames_ks']
        self.kl = args_mgdc['num_distant_frames_kl']
        self.n_skip = args_mgdc['distant_frame_skip_n']

    def _prepare_history_with_time_info(self, historical_X_j_list_raw):
        # historical_X_j_list_raw: list of X_j^k, ordered from most recent to oldest
        # We need to select K_S recent and K_L distant and assign time/type indices

        # This logic needs to be robust, e.g., if len(historical_X_j_list_raw) is small
        prepared_history = []

        # Recent frames
        time_embed_idx_counter = 0
        for i in range(min(self.ks, len(historical_X_j_list_raw))):
            prepared_history.append((historical_X_j_list_raw[i], time_embed_idx_counter, 0))  # 0 for recent
            time_embed_idx_counter += 1

        # Distant frames
        # Start looking for distant frames after the K_S recent ones
        # And skip N_skip frames
        distant_collected = 0
        current_raw_idx = self.ks
        while distant_collected < self.kl and current_raw_idx < len(historical_X_j_list_raw):
            prepared_history.append(
                (historical_X_j_list_raw[current_raw_idx], time_embed_idx_counter, 1))  # 1 for distant
            time_embed_idx_counter += 1
            distant_collected += 1
            current_raw_idx += (self.n_skip + 1)  # Skip n_skip frames

        return prepared_history

    def _split_transformer_output(self, t_out_flat, historical_X_j_list_with_time_info, num_patches_per_frame):
        # t_out_flat: [1, TotalPatches, E]
        # num_patches_per_frame: H_prime * W_prime
        # This function needs to return t_out_recent_tokens and t_out_distant_tokens
        # based on the seq_type in historical_X_j_list_with_time_info

        t_out_recent_tokens_list = []
        t_out_distant_tokens_list = []
        t_out_t0_tokens = None  # Tokens corresponding to X_j^t0 (most recent in history)

        current_token_idx = 0
        for x_bev_k, time_idx_k, seq_type_k in historical_X_j_list_with_time_info:
            # Assuming each frame contributes num_patches_per_frame consistently for simplicity
            # (In reality, MixedHistoryTokenizer might truncate, making this complex)
            frame_tokens = t_out_flat[:, current_token_idx: current_token_idx + num_patches_per_frame, :]

            if seq_type_k == 0:  # Recent
                t_out_recent_tokens_list.append(frame_tokens)
                if time_idx_k == 0:  # Assuming the very first recent frame is t0
                    t_out_t0_tokens = frame_tokens
            else:  # Distant
                t_out_distant_tokens_list.append(frame_tokens)
            current_token_idx += num_patches_per_frame

        # Concatenate all recent/distant tokens respectively
        # Handle cases where lists might be empty
        t_out_recent = torch.cat(t_out_recent_tokens_list, dim=1) if t_out_recent_tokens_list else \
            torch.empty(1, 0, self.embed_dim, device=t_out_flat.device)
        t_out_distant = torch.cat(t_out_distant_tokens_list, dim=1) if t_out_distant_tokens_list else \
            torch.empty(1, 0, self.embed_dim, device=t_out_flat.device)

        if t_out_t0_tokens is None and t_out_recent_tokens_list:  # Fallback if t0 not explicitly first
            t_out_t0_tokens = t_out_recent_tokens_list[0]
        elif t_out_t0_tokens is None:  # No recent tokens at all
            t_out_t0_tokens = torch.empty(1, 0, self.embed_dim, device=t_out_flat.device)

        return t_out_recent, t_out_distant, t_out_t0_tokens

    def forward(self,
                historical_X_j_list_raw,  # List of [1, C_total_hist, H, W] for agent j, from t0-dt, t0-2dt ...
                current_F_vox_trans_t0,  # Sparse BEV map [1, C_V, H, W] or None
                current_F_feat_trans_t0,  # Sparse BEV map [1, C_F, H, W] or None
                current_f_det_list_trans_t0,  # List of object dicts at t0 or None
                delay_time_span_scalar):  # Scalar (t-t0)

        if not historical_X_j_list_raw:  # No history, cannot perform this type of compensation
            # Fallback: maybe return uncompensated data or apply a simpler compensation
            # For now, return as is if no history.
            print("Warning: MGDC called with no history. Returning uncompensated data.")
            return current_f_det_list_trans_t0, current_F_vox_trans_t0, current_F_feat_trans_t0

        # 1. Prepare history with time/type info and tokenize
        history_with_info = self._prepare_history_with_time_info(historical_X_j_list_raw)
        if not history_with_info:  # Still no usable history
            print("Warning: MGDC could not prepare history. Returning uncompensated data.")
            return current_f_det_list_trans_t0, current_F_vox_trans_t0, current_F_feat_trans_t0

        t_in_flat, _ = self.tokenizer(history_with_info)  # [1, TotalPatches, E]
        if t_in_flat.shape[1] == 0:  # Tokenizer produced no tokens
            print("Warning: MGDC tokenizer produced no tokens. Returning uncompensated data.")
            return current_f_det_list_trans_t0, current_F_vox_trans_t0, current_F_feat_trans_t0

        # 2. Transformer Encoding
        t_out_flat = self.transformer_encoder(t_in_flat)  # [1, TotalPatches, E]

        # 3. Context Feature Extraction
        # This requires knowing how many patches each frame in history_with_info contributed to t_out_flat
        num_patches_per_frame = self.h_prime * self.w_prime  # Assuming fixed patches per frame

        t_out_recent_tokens, t_out_distant_tokens, t_out_t0_tokens_for_obj_ctx = \
            self._split_transformer_output(t_out_flat, history_with_info, num_patches_per_frame)

        # H_short from t0 tokens (or all recent ones, paper says "mainly using t0")
        # Let's use t0 tokens for H_short for simplicity, assuming t0 is the first in recent.
        H_short = self.short_term_decoder(
            t_out_t0_tokens_for_obj_ctx if t_out_t0_tokens_for_obj_ctx.shape[1] > 0 else t_out_recent_tokens)
        z_long = self.long_term_aggregator(t_out_distant_tokens)  # [1, D_long]

        # 4. Motion Prediction & Compensation

        # --- Object-level ---
        compensated_f_det_list_t = []
        if current_f_det_list_trans_t0 is not None:
            # Convert detection list boxes to RoIs [K, 5] for RoIAlign
            # This assumes boxes are in pixel coordinates of the BEV map
            rois_for_obj_ctx = []
            original_obj_params = []  # To store original box params for transformation
            for obj_dict in current_f_det_list_trans_t0:
                # Example: obj_dict = {'box_bev_coords': [x1,y1,x2,y2], 'label':l, 'score':s}
                # Add batch index 0 for RoIAlign
                rois_for_obj_ctx.append([0] + obj_dict['box_bev_coords'])
                original_obj_params.append(obj_dict)  # Store full dict

            if rois_for_obj_ctx:
                rois_tensor = torch.tensor(rois_for_obj_ctx,
                                           device=t_out_flat.device,
                                           dtype=torch.float32)

                # z_obj_o_list: list of [D_obj] tensors
                z_obj_o_list = self.object_context_extractor(t_out_t0_tokens_for_obj_ctx, [rois_tensor])

                for i, z_obj_o in enumerate(z_obj_o_list):
                    T_obj_o_params = self.object_motion_predictor(
                        z_obj_o, z_long.squeeze(0), delay_time_span_scalar
                    )  # T_obj_o_params: [D_out_affine]

                    # Apply T_obj_o_params to original_obj_params[i]['box_bev_coords']
                    # This is a placeholder for actual transformation logic
                    # E.g., if D_out_affine is (dx, dy, d_theta) for 2D box center and orientation
                    compensated_box_params = self._apply_affine_to_box(original_obj_params[i], T_obj_o_params)
                    compensated_f_det_list_t.append(compensated_box_params)  # Store updated dict
            else:  # No objects in the list
                compensated_f_det_list_t = None

        # --- BEV-level ---
        O_bev, U_bev = self.bev_motion_predictor(H_short, z_long, delay_time_span_scalar)
        # O_bev: [1, 2, H, W], U_bev: [1, 1, H, W]

        hat_F_vox_t = None
        if current_F_vox_trans_t0 is not None:
            # Warper expects flow that directly maps src to dest coords, or displacement
            # Our O_bev is displacement. scaled_delay is already incorporated in predictor.
            # Paper: f_warp(F, (t-t0)*O_bev). So O_bev is flow_per_unit_delay.
            Z_vox_t = self.feature_warper.flow_warp(current_F_vox_trans_t0, O_bev, delay_time_span_scalar)
            hat_F_vox_t = Z_vox_t * (1 - U_bev)

        hat_F_feat_t = None
        if current_F_feat_trans_t0 is not None:
            Z_feat_t = self.feature_warper.flow_warp(current_F_feat_trans_t0, O_bev, delay_time_span_scalar)
            hat_F_feat_t = Z_feat_t * (1 - U_bev)

        return compensated_f_det_list_t, hat_F_vox_t, hat_F_feat_t

    def _apply_affine_to_box(self, original_obj_dict, affine_params_tensor):
        # Placeholder: Implement actual transformation of bounding box
        # based on the format of affine_params_tensor (e.g., dx, dy, dtheta)
        # For example:
        # box_center_x, box_center_y, w, l, heading = original_obj_dict['params']
        # dx, dy, d_heading = affine_params_tensor[0], affine_params_tensor[1], affine_params_tensor[2]
        # new_center_x = box_center_x + dx
        # new_center_y = box_center_y + dy
        # new_heading = heading + d_heading
        # return {'box_bev_coords': updated_coords, 'label': original_obj_dict['label'], ...}

        # Simple copy for now, assuming affine_params are deltas for the dict's values
        compensated_dict = original_obj_dict.copy()
        # Update compensated_dict['box_bev_coords'] or other relevant fields
        # This is highly dependent on your box representation and affine_params meaning
        # Example: if affine_params is [dx_center, dy_center, d_angle]
        # And box_bev_coords is [x1,y1,x2,y2], you'd need to convert to center, apply, convert back.
        # Or if box is [cx, cy, w, l, angle]

        # This is a very rough placeholder
        if 'box_bev_coords' in compensated_dict and len(
                affine_params_tensor) >= 2:  # Assume dx, dy for top-left for now
            compensated_dict['box_bev_coords'][0] += affine_params_tensor[0].item()
            compensated_dict['box_bev_coords'][1] += affine_params_tensor[1].item()
            compensated_dict['box_bev_coords'][2] += affine_params_tensor[
                0].item()  # Assuming same delta for bottom-right
            compensated_dict['box_bev_coords'][3] += affine_params_tensor[1].item()

        return compensated_dict
