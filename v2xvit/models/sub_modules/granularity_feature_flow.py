import torch
import torch.nn as nn
import torch.nn.functional as F

def warp_bev(bev_map, flow_field):
    B, _, H, W = bev_map.shape
    # Create a base grid of (x, y) coordinates
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    grid = torch.stack([xx, yy], dim=0).float().to(bev_map.device)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

    # The flow field gives offsets. Add them to the base grid.
    # The flow field dx corresponds to x-coordinates (W dimension), dy to y (H dimension)
    new_grid = grid + flow_field

    # Normalize the grid to the required [-1, 1] range for grid_sample
    # Normalize x coordinates (W dimension)
    new_grid[:, 0, :, :] = 2 * new_grid[:, 0, :, :] / (W - 1) - 1
    # Normalize y coordinates (H dimension)
    new_grid[:, 1, :, :] = 2 * new_grid[:, 1, :, :] / (H - 1) - 1

    # Perform the warping
    warped_map = F.grid_sample(bev_map, new_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return warped_map

class HierarchicalMotionPredictor(nn.Module):
    def __init__(self, c_vox, c_feat, c_det, hidden_dim=128):
        super(HierarchicalMotionPredictor, self).__init__()

        self.vox_encoder = self._make_stem(c_vox, hidden_dim // 4)
        self.feat_encoder = self._make_stem(c_feat, hidden_dim)
        self.det_encoder = self._make_stem(c_det, hidden_dim // 2)

        unified_dim = (hidden_dim // 4) + hidden_dim + (hidden_dim // 2)

        #Temporal Fusion
        self.temporal_processor = ConvGRUCell(unified_dim, hidden_dim, kernel_size=3)
        self.temporal_hidden_dim = hidden_dim

        # 3. Parallel Prediction Heads
        # Both heads take the final temporal hidden state as input
        self.object_flow_head = self._make_prediction_head(hidden_dim)
        self.residual_flow_head = self._make_prediction_head(hidden_dim)

    def _make_stem(selfself, in_channels, out_channels):
        """Creates a simple 2-layer Conv block to encode an input modality."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _make_prediction_head(self, in_channels):
        """Creates a head to predict a 2D flow field."""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 2, kernel_size=1)  # Output 2 channels for (dx, dy)
        )

    def forward(self, history_data):

