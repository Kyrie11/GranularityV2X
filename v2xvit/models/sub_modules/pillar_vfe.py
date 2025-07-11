"""
Pillar VFE, credits to OpenPCDet.
"""
from os import device_encoding

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomPointScatter(nn.Module):
    def __init__(self, grid_size, C_bev):
        super().__init__()
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1
        self.C_bev = C_bev

    def forward(self, point_features, voxel_coords):
        '''
        :param point_features: [N_pillars, N_points, C_point]
        :param voxel_coords:   [N_pillars, 4] => [batch_idx, z, y, x]
        :return:
        '''

        N_pillars, N_points, C_point = point_features.shape
        B = voxel_coords[:, 0].max().item() + 1
        H, W = int(self.ny), int(self.nx)

        #Step1:聚合每个pillar内部点的特征
        pillar_bev_feat = point_features.mean(dim=1) #[N_pillars, C_point]
        C_bev = pillar_bev_feat.shape[1]

        #Step2:创建空的BEV特征图
        spatial_features = torch.zeros((B, C_bev, H, W),
                                       dtype=point_features.dtype,
                                       device=point_features.device)

        # Step 3: 将pillar特征scatter到对应位置
        for i in range(N_pillars):
            b, z, y, x = voxel_coords[i]
            spatial_features[b, :, y, x] = pillar_bev_feat[i]

        return spatial_features

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0, device=inputs.device)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2,
                                                  1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size,
                 point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg['use_norm']
        self.with_distance = self.model_cfg['with_distance']

        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz']
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg['num_filters']
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                         last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.nx = - point_cloud_range[0] / self.voxel_x * 2
        self.ny = - point_cloud_range[1] / self.voxel_y * 2
        self.nz = 1

        self.num_bev_features = 8
        self.grid_size_x = round((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.grid_size_y = round((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict):

        voxel_features, voxel_num_points, coords = \
            batch_dict['voxel_features'], batch_dict['voxel_num_points'], \
            batch_dict['voxel_coords']

        if torch.isnan(coords).any() or torch.isinf(coords).any():
            print("!!! FATAL: Found NaN or Inf in coords tensor!")
            print(coords)  # 打印出有问题的coords以供分析
            # 你可以选择在这里抛出异常，以便立即定位问题
            raise ValueError("NaN/Inf in coords")

        record_len = batch_dict['record_len']
        batch_size = len(record_len)

        points_mean = \
            voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(
                    1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2,
                                     keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        # C = features.shape[2]
        # scatter = CustomPointScatter(grid_size=(self.nx, self.ny, self.nz), C_bev=C)
        # vox_bev = scatter(features, coords)
        # batch_dict['vox_bev'] = vox_bev

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count,
                                           axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features

        #点数(num of points)
        num_points_norm = voxel_num_points.view(-1, 1).float() / voxel_count
        #为了安全的除法， 防止体素中点数为0
        safe_voxel_num_points = voxel_num_points.view(-1, 1).float().clamp(min=1.0)

        #提取x,y,z,intensity
        points_xyz = voxel_features[:, :, :3]
        points_intensity = voxel_features[:, :, 3:4]

        #平均激光雷达强度
        sum_intensity = (points_intensity * mask).sum(dim=1)
        mean_intensity = sum_intensity / safe_voxel_num_points

        #平均高度
        sum_height = (points_xyz[:, :, 2:3] * mask).sum(dim=1)
        mean_height = sum_height / safe_voxel_num_points

        #最高点高度
        max_height = (points_xyz[:,:,2:3] * mask+(1-mask)*-1e6).max(dim=1)[0]

        #高度跨度(Height Span)
        min_height = (points_xyz[:, :, 2:3] * mask + (1 - mask) * 1e6).min(dim=1)[0]
        height_span = max_height - min_height

        #点坐标方差
        pillar_points_mean = (points_xyz * mask).sum(dim=1, keepdim=True) / safe_voxel_num_points.view(-1, 1, 1)
        points_sqr_dist = ((points_xyz - pillar_points_mean)**2 * mask).sum(dim=1) / safe_voxel_num_points.view(-1, 1)
        var_x = points_sqr_dist[:, 0:1]
        var_y = points_sqr_dist[:, 1:2]
        var_z = points_sqr_dist[:, 2:3]

        # 将所有手工设计的特征拼接在一起
        # 8个特征: [点数, 平均强度, 平均高度, 最高点高度, 高度跨度, x方差, y方差, z方差]
        pillar_bev_features = torch.cat([
            num_points_norm, mean_intensity, mean_height, max_height,
            height_span, var_x, var_y, var_z
        ], dim=1)
        # print("pillar_bev_features.shape=", pillar_bev_features.shape)

        vox_bev = torch.zeros(
            batch_size,
            self.num_bev_features,
            self.grid_size_y,
            self.grid_size_x,
            device = voxel_features.device)

        batch_indices = coords[:, 0].long()
        y_indices = coords[:, 2].long()
        x_indices = coords[:, 3].long()
        print(y_indices)
        batch_indices = torch.clamp(batch_indices, min=0, max=batch_size-1)
        # y_indices = torch.clamp(y_indices, min=0, max=self.grid_size_y - 1)
        # x_indices = torch.clamp(x_indices, min=0, max=self.grid_size_x - 1)

        vox_bev[batch_indices, :, y_indices, x_indices] = pillar_bev_features
        batch_dict['vox_bev'] = vox_bev
        # print("vox_bev.shape=", vox_bev.shape)

        return batch_dict
