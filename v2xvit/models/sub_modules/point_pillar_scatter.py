import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']
        print("nx in scatter:", self.nx)
        assert self.nz == 1

    def forward(self, batch_dict):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict[
            'voxel_coords']

        raw_points = batch_dict['voxel_features'][:, :, :4] #原始点云，保存x,y,z,intensity
        #将原始点云按体素索引关联到BEV网格
        batch_dict['raw_points_mapped'] = {
            'coords': coords,  #体素坐标[num_voxels,4] (batch_idx, z, y, x)
            'raw_points': raw_points #原始点云 [num_voxels, max_pts_per_voxel, 4]
        }


        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]

            indices = this_coords[:, 1] + \
                      this_coords[:, 2] * self.nx + \
                      this_coords[:, 3]
            indices = indices.type(torch.long)

            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = \
            torch.stack(batch_spatial_features, 0)
        batch_spatial_features = \
            batch_spatial_features.view(batch_size, self.num_bev_features *
                                        self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features

        return batch_dict

