import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign  # For ObjectContextExtractor


# Assuming BasicConvBlock and other utilities are defined elsewhere or standard
# from v2xvit.models.sub_modules.your_blocks import BasicConvBlock, ResNetBEVBackbone, ReduceInfTC
# For now, I'll use placeholders or simplified versions.
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


# --- Helper: Patch Embedding and Tokenizer ---
class BEVPatchEmbed(nn.Module):
    def __init__(self, C_in, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(C_in, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, E, H', W']
        return x.flatten(2).transpose(1, 2)  # [B, H'*W', E]

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

class TemporalContextEncoder(nn.Module):
    """Encodes a sequence of BEV features (single granularity)"""
    def __init__(self, C_in_granularity, D_out_context, num_history_frames):
        super().__init__()
        if num_history_frames>0:
            self.conv3d = nn.Conv3d(C_in_granularity, D_out_context,
                                    kernel_size=(min(num_history_frames, 3),3,3),# Kernel depth <= num_frames
                                    padding=(min(num_history_frames, 3)//2 if min(num_history_frames, 3)>1 else 0,1,1))
            self.norm = nn.BatchNorm2d(D_out_context)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool3d((1,None,None)) #Pool along time dim
        else: #No history
            self.dummy_param = nn.Parameter(torch.empty(0))

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, history_bev_list, record_len):
        # history_bev_list: list of [B, C_in, H, W]
        if not history_bev_list:
            return None

        B = len(record_len)
        batch_history_bev_list = self.regroup(history_bev_list, record_len)
        bev_final = []
        for b in range(B):
            N = record_len[b]
            history_bev = batch_history_bev_list[b]
            if history_bev.dim() == 3:
                stacked_history = torch.stack(history_bev, dim=1).unsqueeze(0) #[1,C,numframes,H,W]
            elif history_bev[0].dim() == 4:
                stacked_history = torch.cat(history_bev, dim=0).unsqueeze(0).permute(0,2,1,3,4) # [1, C, NumFrames, H,W]
            else:
                raise ValueError("Unsupported history tensor dim")

            x = self.relu(self.norm(self.conv3d(stacked_history)))
            x = self.pool(x)
            bev_final.append(x)
        bev_final = torch.cat(bev_final, dim=0)
        return bev_final

class BEVFlowPredictor(nn.Module):
    def __int__(self, D_short_ctx, D_long_ctx, D_current_feat,
                D_hidden, num_flow_channels=2, num_uncertainty_channels=1):
        super().__init__()
        input_channels = D_current_feat
        if D_short_ctx > 0: input_channels += D_short_ctx
        if D_long_ctx > 0: input_channels += D_long_ctx
        input_channels += 1 #For delay scalar map

        #U-Net like structure
        self.encoder1 = BasicConvBlock(input_channels, D_hidden)
        self.encoder2 = BasicConvBlock(D_hidden, D_hidden * 2, stride=2)
        self.encoder3 = BasicConvBlock(D_hidden * 2, D_hidden * 4, stride=2)

        self.decoder2 = BasicConvBlock(D_hidden * 4 + D_hidden * 2, D_hidden * 2)  # Skip conn
        self.upsample2 = nn.ConvTranspose2d(D_hidden * 2, D_hidden * 2, kernel_size=2, stride=2)

        self.decoder1 = BasicConvBlock(D_hidden * 2 + D_hidden, D_hidden)  # Skip conn
        self.upsample1 = nn.ConvTranspose2d(D_hidden, D_hidden, kernel_size=2, stride=2)

        self.flow_head = nn.Conv2d(D_hidden, num_flow_channels, kernel_size=3, padding=1)
        self.uncertainty_head = nn.Conv2d(D_hidden, num_uncertainty_channels, kernel_size=3, padding=1)

        nn.init.zeros_(self.flow_head.weight)
        if self.flow_head.bias is not None: nn.init.zeros_(self.flow_head.bias)

    def forward(self, current_F_cat_t0, short_term_ctx, long_term_ctx, delay_scalar_map):
        # current_F_cat_t0: [B, C_V+C_F+C_D, H, W]
        # short_term_ctx: [B, D_short_ctx, H, W] or None
        # long_term_ctx: [B, D_long_ctx, H, W] or None (can be global vector tiled too)
        # delay_scalar_map: [B, 1, H, W]
        input_list = [current_F_cat_t0]
        if short_term_ctx is not None:
            # Long-term context can modulate short-term context
            if long_term_ctx is not None:
                # Example: FiLM-like modulation or simple concat + conv
                # For simplicity, concat here
                # Ensure long_term_ctx is spatially aligned if it's a map
                # If long_term_ctx is global, tile it.
                # If both are maps:
                # combined_short_long = torch.cat([short_term_ctx, long_term_ctx], dim=1)
                # refined_short_ctx = self.short_long_refiner(combined_short_long) # some conv
                # input_list.append(refined_short_ctx)
                input_list.append(short_term_ctx)
                input_list.append(long_term_ctx)
            else:
                input_list.append(short_term_ctx)
        elif long_term_ctx is not None:# Only long term available
            input_list.append(long_term_ctx)
        input_list.append(delay_scalar_map)

        x = torch.cat(input_list, dim=1)

        enc1 = self.encoder1(x)  # D_hidden
        enc2 = self.encoder2(enc1)  # D_hidden*2
        enc3 = self.encoder3(enc2)  # D_hidden*4

        dec2 = self.upsample2(enc3)  # D_hidden*2
        dec2 = torch.cat([dec2, enc2], dim=1)  # Skip
        dec2 = self.decoder2(dec2)  # D_hidden*2

        dec1 = self.upsample1(dec2)  # D_hidden
        dec1 = torch.cat([dec1, enc1], dim=1)  # Skip
        dec1 = self.decoder1(dec1)  # D_hidden

        flow = self.flow_head(dec1)
        uncertainty = torch.sigmoid(self.uncertainty_head(dec1))
        return flow, uncertainty

class MultiGranularityBevDelayCompensation(nn.Module):
    def __init__(self, args_mgdc_bev):
        super().__init__()
        self.C_V = args_mgdc_bev['C_V']
        self.C_F = args_mgdc_bev['C_F']
        self.C_D = args_mgdc_bev['C_D']
        self.D_short_ctx_per_gran = args_mgdc_bev['D_short_ctx_per_granularity']
        self.D_long_ctx_per_gran = args_mgdc_bev['D_long_ctx_per_granularity']
        self.num_short_history = args_mgdc_bev['num_short_history_frames']
        self.num_long_history = args_mgdc_bev['num_long_history_frames']
        # Short-term encoders for each granularity
        if self.C_V > 0 and self.num_short_history > 0:
            self.short_encoder_vox = TemporalContextEncoder(self.C_V, self.D_short_ctx_per_gran, self.num_short_history)
        if self.C_F > 0 and self.num_short_history > 0:
            self.short_encoder_feat = TemporalContextEncoder(self.C_F, self.D_short_ctx_per_gran,
                                                             self.num_short_history)
        if self.C_D > 0 and self.num_short_history > 0:
            self.short_encoder_det = TemporalContextEncoder(self.C_D, self.D_short_ctx_per_gran, self.num_short_history)

        # Long-term encoders for each granularity
        if self.C_V > 0 and self.num_long_history > 0:
            self.long_encoder_vox = TemporalContextEncoder(self.C_V, self.D_long_ctx_per_gran,
                                                           self.num_long_history)
        if self.C_F > 0 and self.num_long_history > 0:
            self.long_encoder_feat = TemporalContextEncoder(self.C_F, self.D_long_ctx_per_gran,
                                                            self.num_long_history)
        if self.C_D > 0 and self.num_long_history > 0:
            self.long_encoder_det = TemporalContextEncoder(self.C_D, self.D_long_ctx_per_gran,
                                                           self.num_long_history)

        # Combine context from all granularities
        total_short_ctx_dim = 0
        if self.C_V > 0 and self.num_short_history > 0: total_short_ctx_dim += self.D_short_ctx_per_gran
        if self.C_F > 0 and self.num_short_history > 0: total_short_ctx_dim += self.D_short_ctx_per_gran
        if self.C_D > 0 and self.num_short_history > 0: total_short_ctx_dim += self.D_short_ctx_per_gran

        total_long_ctx_dim = 0
        if self.C_V > 0 and self.num_long_history > 0: total_long_ctx_dim += self.D_long_ctx_per_gran
        if self.C_F > 0 and self.num_long_history > 0: total_long_ctx_dim += self.D_long_ctx_per_gran
        if self.C_D > 0 and self.num_long_history > 0: total_long_ctx_dim += self.D_long_ctx_per_gran

        self.flow_predictor = BEVFlowPredictor(
            total_short_ctx_dim,
            total_long_ctx_dim,
            self.C_V + self.C_F + self.C_D,  # Channels of current concatenated F_t0
            args_mgdc_bev['flow_predictor_hidden_dim']
        )
        self.feature_warper = FeatureWarper()

    def forward(self,
                current_F_vox_t0, current_F_feat_t0, current_F_det_bev_t0,  # [B,C,H,W] or None
                short_history_vox, short_history_feat, short_history_det,  # list of [B,C,H,W]
                long_history_vox, long_history_feat, long_history_det,  # list of [B,C,H,W]
                delay_time_span_scalar
                ):

        b, _, h, w = current_F_vox_t0.shape if current_F_vox_t0 is not None else \
                    current_F_feat_t0.shape if current_F_feat_t0 is not None else \
                    current_F_det_bev_t0.shape

        if self.C_V > 0 and self.num_short_history > 0 and short_history_vox:
            s_ctx_vox = self.short_encoder_vox(short_history_vox)
        if self.C_F > 0 and self.num_short_history > 0 and short_history_feat:
            s_ctx_feat = self.short_encoder_feat(short_history_feat)
        if self.C_D > 0 and self.num_short_history > 0 and short_history_det:
            s_ctx_det = self.short_encoder_det(short_history_det)

        short_term_contexts = [c for c in [s_ctx_vox, s_ctx_feat, s_ctx_det] if c is not None]
        if short_term_contexts:
            full_short_ctx = torch.cat(short_term_contexts, dim=1)
        else:  # No short history processed
            full_short_ctx = None  # Or zeros: torch.zeros(b, 0, h, w, device=...)

        # Encode long-term history
        l_ctx_vox, l_ctx_feat, l_ctx_det = None, None, None
        if self.C_V > 0 and self.num_long_history > 0 and long_history_vox:
            l_ctx_vox = self.long_encoder_vox(long_history_vox)
        if self.C_F > 0 and self.num_long_history > 0 and long_history_feat:
            l_ctx_feat = self.long_encoder_feat(long_history_feat)
        if self.C_D > 0 and self.num_long_history > 0 and long_history_det:
            l_ctx_det = self.long_encoder_det(long_history_det)

        long_term_contexts = [c for c in [l_ctx_vox, l_ctx_feat, l_ctx_det] if c is not None]
        if long_term_contexts:
            full_long_ctx = torch.cat(long_term_contexts, dim=1)
        else:  # No long history processed
            full_long_ctx = None  # Or zeros

            # Prepare current features and delay map
            current_F_parts = []
            if current_F_vox_t0 is not None:
                current_F_parts.append(current_F_vox_t0)
            else:
                current_F_parts.append(torch.zeros(b, self.C_V, h, w,
                                                   device=l_ctx_vox.device if l_ctx_vox is not None else (
                                                       s_ctx_vox.device if s_ctx_vox is not None else 'cpu')))  # Add zeros if None

            if current_F_feat_t0 is not None:
                current_F_parts.append(current_F_feat_t0)
            else:
                current_F_parts.append(torch.zeros(b, self.C_F, h, w,
                                                   device=l_ctx_vox.device if l_ctx_vox is not None else (
                                                       s_ctx_vox.device if s_ctx_vox is not None else 'cpu')))

            if current_F_det_bev_t0 is not None:
                current_F_parts.append(current_F_det_bev_t0)
            else:
                current_F_parts.append(torch.zeros(b, self.C_D, h, w,
                                                   device=l_ctx_vox.device if l_ctx_vox is not None else (
                                                       s_ctx_vox.device if s_ctx_vox is not None else 'cpu')))

            current_F_cat_t0 = torch.cat(current_F_parts, dim=1)

            delay_map = torch.full((b, 1, h, w), delay_time_span_scalar,
                                   device=current_F_cat_t0.device, dtype=current_F_cat_t0.dtype)

            # Predict flow and uncertainty
            # The BEVFlowPredictor needs to handle None for full_short_ctx or full_long_ctx
            # (e.g., by having its input_channels calculation be dynamic or by passing zero tensors)
            # For now, flow_predictor init assumes fixed D_short_ctx, D_long_ctx.
            # If full_short_ctx is None, pass zeros of expected shape.
            if full_short_ctx is None:
                total_short_ctx_dim = self.flow_predictor.encoder1.conv.in_channels - \
                                      (self.C_V + self.C_F + self.C_D) - 1 - \
                                      (full_long_ctx.shape[1] if full_long_ctx is not None else 0)
                if total_short_ctx_dim > 0:
                    full_short_ctx = torch.zeros(b, total_short_ctx_dim, h, w, device=current_F_cat_t0.device)
                else:  # This case implies D_short_ctx was 0
                    full_short_ctx = None

            if full_long_ctx is None:
                total_long_ctx_dim = self.flow_predictor.encoder1.conv.in_channels - \
                                     (self.C_V + self.C_F + self.C_D) - 1 - \
                                     (full_short_ctx.shape[1] if full_short_ctx is not None else 0)
                if total_long_ctx_dim > 0:
                    full_long_ctx = torch.zeros(b, total_long_ctx_dim, h, w, device=current_F_cat_t0.device)
                else:
                    full_long_ctx = None

            flow_field, uncertainty_map = self.flow_predictor(
                current_F_cat_t0, full_short_ctx, full_long_ctx, delay_map
            )

            # Warp features
            compensated_F_vox_t, compensated_F_feat_t, compensated_F_det_bev_t = None, None, None
            if current_F_vox_t0 is not None:
                compensated_F_vox_t = self.feature_warper.flow_warp(current_F_vox_t0, flow_field,
                                                                    delay_time_span_scalar)
                compensated_F_vox_t = compensated_F_vox_t * (1 - uncertainty_map)
            if current_F_feat_t0 is not None:
                compensated_F_feat_t = self.feature_warper.flow_warp(current_F_feat_t0, flow_field,
                                                                     delay_time_span_scalar)
                compensated_F_feat_t = compensated_F_feat_t * (1 - uncertainty_map)
            if current_F_det_bev_t0 is not None:
                compensated_F_det_bev_t = self.feature_warper.flow_warp(current_F_det_bev_t0, flow_field,
                                                                        delay_time_span_scalar)
                compensated_F_det_bev_t = compensated_F_det_bev_t * (1 - uncertainty_map)

            # Note: Spatial transformation to ego frame is not explicitly done here.
            # It's assumed that the flow_field implicitly learns to predict motion
            # such that when warped, features are pseudo-aligned to where they *would be*
            # in the ego's view at current time, if they were still in collab's coord system.
            # A final explicit spatial transform using pairwise_t_matrix might still be needed
            # AFTER this compensation, before fusion with ego.
            # Or, the `delay_time_span_scalar` and history implicitly guide the flow predictor
            # to output flow in a "common" or "ego-aligned" sense if training data supports it.
            # This is a subtle but important point. Usually, flow is in pixel space of the input.

            return compensated_F_vox_t, compensated_F_feat_t, compensated_F_det_bev_t, uncertainty_map



