import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_relu: bool = True):
        super().__init__()
        print("convblock里的out_channels:", out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # Using GroupNorm is often more stable for BEV features than BatchNorm,
        # especially with varying or small batch sizes.
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class HistoryContextAdaptiveEnhancement(nn.Module):
    def __init__(self,
                 current_channels: int,
                 short_ctx_channels: int,
                 long_ctx_dim: int,
                 history_channels: int = 128,
                 output_channels: int = None):

        super().__init__()

        if output_channels is None:
            output_channels = current_channels

        # === Step 1: Context Unification Layers ===
        # Processor for the long-term context vector
        self.long_ctx_processor = nn.Sequential(
            nn.Linear(long_ctx_dim, history_channels),
            nn.ReLU(inplace=True)
        )
        # Processor for the short-term context map
        self.short_ctx_processor = nn.Conv2d(short_ctx_channels, history_channels, kernel_size=1)
        # Fusion layer for the combined history context
        self.history_fusion_conv = nn.Conv2d(history_channels * 2, history_channels, kernel_size=1)

        # === Step 2: Dual-Pathway Enhancement Layers ===
        # Pathway 1: Content Enhancement. Output channels must match current_channels for addition.
        self.content_pathway = ConvBlock(history_channels, current_channels)

        # Pathway 2: Attention Modulation. Output is a single-channel map for spatial gating.
        self.attention_pathway = ConvBlock(history_channels, 1)

        # === Step 3:  Fusion Layer ===
        # Takes concatenation of original, content-enhanced, and attention-modulated features.
        final_fusion_in_channels = current_channels * 3
        self.final_fusion_conv = nn.Conv2d(final_fusion_in_channels, output_channels, kernel_size=1)

    def forward(self,
                unified_bev: torch.Tensor,
                short_term_context: torch.Tensor,
                long_term_context: torch.Tensor) -> torch.Tensor:
        N, _, H, W = unified_bev.shape
        device = unified_bev.device

        # --- Step 1: Unify History Context ---
        # Process and expand long-term context to a spatial map
        l_ctx_processed = self.long_ctx_processor(long_term_context)
        l_ctx_spatial = l_ctx_processed.unsqueeze(-1).unsqueeze(-1).expand(N, -1, H, W)

        # Process short-term context
        s_ctx_processed = self.short_ctx_processor(short_term_context)

        # Concatenate and fuse into a single, unified history map
        concatenated_history = torch.cat([l_ctx_spatial, s_ctx_processed], dim=1)
        C_history = self.history_fusion_conv(concatenated_history)

        # --- Step 2: Parallel Dual-Pathway Enhancement ---
        # Pathway 1: Content Enhancement (Additive)
        delta_F_content = self.content_pathway(C_history)
        F_content_enhanced = unified_bev + delta_F_content

        # Pathway 2: Attention Modulation (Multiplicative)
        # Note: Sigmoid is applied here to create the gate
        attention_map = torch.sigmoid(self.attention_pathway(C_history))
        F_attention_modulated = unified_bev * attention_map

        # --- Step 3:  Fusion ---
        # Concatenate the original feature and the two enhanced versions
        concatenated_final = torch.cat([unified_bev, F_content_enhanced, F_attention_modulated], dim=1)
        F_enhanced = self.final_fusion_conv(concatenated_final)

        return F_enhanced




