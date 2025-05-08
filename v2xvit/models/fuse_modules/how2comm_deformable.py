from turtle import update
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import functional as F
from torch import batch_norm, einsum
from einops import rearrange, repeat
from icecream import ic

from v2xvit.models.sub_modules.torch_transformation_utils import warp_affine_simple
from v2xvit.models.comm_modules.communication import Communication
from v2xvit.models.sub_modules.how2comm_preprocess import How2commPreprocess
from v2xvit.models.fuse_modules.stcformer import STCFormer
from v2xvit.models.sub_modules.pillar_vfe import PillarVFE
from v2xvit.models.sub_modules.point_pillar_scatter import PointPillarScatter

class How2comm(nn.Module):
    def __init__(self, args, args_pre):
        super(How2comm, self).__init__()

        self.max_cav = 5 
        self.communication = False
        self.round = 1
        if 'communication' in args:
            self.communication = True
            self.naive_communication = Communication(args['communication'])
            if 'round' in args['communication']:
                self.round = args['communication']['round']
        print("communication:", self.communication)
        self.communication_flag = args['communication_flag']
        self.discrete_ratio = args['voxel_size'][0]  
        self.downsample_rate = args['downsample_rate']
        self.async_flag = True
        self.channel_fuse = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)

        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        self.how2comm = How2commPreprocess(args_pre, channel=64, delay=1)
        if self.multi_scale:
            layer_nums = args['layer_nums']  
            num_filters = args['num_filters'] 
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'STCFormer':
                    fuse_network = STCFormer(
                        channel=num_filters[idx], args=args['temporal_fusion'], idx=idx)
                self.fuse_modules.append(fuse_network)

        self.pillar_vfe = PillarVFE(args_pre['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args_pre['voxel_size'],
                                    point_cloud_range=args_pre['lidar_range'])
        self.scatter = PointPillarScatter(args_pre['point_pillar_scatter'])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, psm, record_len, pairwise_t_matrix, backbone=None, heads=None, history=None, raw_voxels=None, raw_coords=None):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [
            0, 1], :][:, :, :, :, [0, 1, 3]]  
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0,
                                                         2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1,
                                                         2] / (self.downsample_rate * self.discrete_ratio * H) * 2


        if history and self.async_flag: 
            feat_final, offset_loss = self.how2comm(x, history, record_len, backbone, heads)
            x = feat_final
        else:
            offset_loss = torch.zeros(1).to(x.device)
        neighbor_psm_list = []
        if history:
            his = history[0]
        else:
            his = x

        if self.multi_scale:
            ups = []
            ups_temporal = []
            ups_exclusive = []
            ups_common = []
            with_resnet = True if hasattr(backbone, 'resnet') else False  
            if with_resnet:
                feats = backbone.resnet(x)
                history_feats = backbone.resnet(his)

            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                his = history_feats[i] if with_resnet else backbone.blocks[i](his)

                if i == 0:
                    if self.communication:
                        batch_confidence_maps = self.regroup(psm, record_len)
                        _, _, confidence_maps = self.naive_communication(batch_confidence_maps)
                        
                        batch_temp_features = self.regroup(x, record_len)
                        batch_temp_features_his = self.regroup(his, record_len)
                        temp_list = []
                        temp_psm_list = []
                        history_list = []
                        for b in range(B):
                            N = record_len[b]
                            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                            temp_features = batch_temp_features[b]
                            C, H, W = temp_features.shape[1:]
                            neighbor_feature = warp_affine_simple(temp_features,
                                                                  t_matrix[0,
                                                                           :, :, :],
                                                                  (H, W)) 
                            temp_list.append(neighbor_feature)

                            temp_features_his = batch_temp_features_his[b]
                            C, H, W = temp_features_his.shape[1:]
                            neighbor_feature_his = warp_affine_simple(temp_features_his,
                                                                  t_matrix[0,
                                                                           :, :, :],
                                                                  (H, W))
                            history_list.append(neighbor_feature_his)
                            
                            temp_psm_list.append(warp_affine_simple(confidence_maps[b], t_matrix[0, :, :, :], (H, W)))  
                        x = torch.cat(temp_list, dim=0)
                        his = torch.cat(history_list, dim=0)
                        if self.communication_flag:
                            sparse_feats, commu_loss, communication_rates, sparse_history, sparse_voxels, sparse_coords = self.how2comm.communication(
                            x, record_len,history_list,temp_psm_list, raw_voxels, raw_coords)
                            x_comm = F.interpolate(sparse_feats, scale_factor=1, mode='bilinear', align_corners=False)
                            x_comm = self.channel_fuse(x_comm)
                            his_comm = F.interpolate(sparse_history, scale_factor=1, mode='bilinear', align_corners=False)
                            his_comm = self.channel_fuse(his_comm)

                            # ----- 并行融合实现 -----
                            #1. 计算Ego agent 原始 BEV特征
                            raw_voxel_list = self.regroup(raw_voxels, record_len)
                            raw_coord_list = self.regroup(raw_coords, record_len)
                            ego_bev_list = []
                            for b in range(len(raw_voxel_list)):
                                #取每一个batch中第一个agent(ego)，合并其体素信息
                                ego_voxels = raw_voxel_list[b][0].unsqueeze(0)  # [1, Np, 4]
                                ego_coords = raw_coord_list[b][0]  # [N_voxels, 4]
                                # 假设每个 voxel 点数相同或全为最大值
                                num_points = torch.full((ego_voxels.shape[0],), ego_voxels.shape[1],
                                                        dtype=torch.int32).to(ego_voxels.device)
                                batch_dict = {
                                    'voxel_features': ego_voxels,
                                    'voxel_coords': ego_coords,
                                    'voxel_num_points': num_points
                                }
                                pillar_feat = self.pillar_vfe(batch_dict)['pillar_features']  # [N_voxels, C']
                                batch_dict['pillar_features'] = pillar_feat
                                bev_ego = self.scatter(batch_dict)['spatial_features']        # [1, C', H, W]
                                ego_bev_list.append(bev_ego)
                            ego_bev = torch.cat(ego_bev_list, dim=0)                              # [B, C', H, W]

                            #2. 计算稀疏体素BEV特征
                            sparse_voxel_list = self.regroup(sparse_voxels, record_len)
                            sparse_coord_list = self.regroup(sparse_coords, record_len)
                            sparse_bev_list = []
                            for b in range(len(sparse_voxel_list)):
                                voxels_b = sparse_voxel_list[b]  # [N_sparse_voxels, Np, 4] (连接了各 agent)
                                coords_b = sparse_coord_list[b]  # [N_sparse_voxels, 4]
                                if voxels_b.numel() == 0:  # 防止无稀疏点云情况
                                    # 创建全零 BEV
                                    batch_size = ego_bev.shape[2]  # H
                                    bev_zeros = torch.zeros_like(ego_bev[b:b + 1])
                                    sparse_bev_list.append(bev_zeros)
                                    continue
                                num_pts2 = torch.full((voxels_b.shape[0],), voxels_b.shape[1], dtype=torch.int32).to(
                                    voxels_b.device)
                                batch_dict2 = {
                                    'voxel_features': voxels_b,
                                    'voxel_coords': coords_b,
                                    'voxel_num_points': num_pts2
                                }
                                pillar_feat2 = self.pillar_vfe(batch_dict2)['pillar_features']
                                batch_dict2['pillar_features'] = pillar_feat2
                                bev_sparse = self.scatter(batch_dict2)['spatial_features']  # [1, C', H, W]
                                sparse_bev_list.append(bev_sparse)
                            sparse_bev = torch.cat(sparse_bev_list, dim=0) # [B, C', H, W]

                            #3. 特征融合(采用加权求和)
                            #假设x_comm和ego_bev、sparse_bev具有相同通道数
                            fusion1 = ego_bev * 0.5 + x_comm * 0.5
                            fusion2 = ego_bev * 0.5 + sparse_bev * 0.5
                            #合并两个融合结果
                            x = fusion1 + fusion2
                            his = his_comm
                        else:
                            communication_rates = torch.tensor(0).to(x.device)
                            commu_loss = torch.zeros(1).to(x.device)
                    else:
                        communication_rates = torch.tensor(0).to(x.device)

                batch_node_features = self.regroup(x, record_len)
                batch_node_features_his = self.regroup(his, record_len)

                x_fuse = []
                x_temporal = []
                x_exclusive = []
                x_common = []
                for b in range(B):
                    N = record_len[b]
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    node_features_his = batch_node_features_his[b]
                    if i == 0:
                        neighbor_feature = node_features 
                        neighbor_feature_his = node_features_his
                        neighbor_psm = warp_affine_simple(
                            confidence_maps[b], t_matrix[0, :, :, :], (H, W))
                        
                    else:
                        C, H, W = node_features.shape[1:]  
                        neighbor_feature = warp_affine_simple(node_features,
                                                              t_matrix[0,
                                                                       :, :, :],
                                                              (H, W))
                        neighbor_feature_his = warp_affine_simple(node_features_his,
                                                              t_matrix[0,
                                                                       :, :, :],
                                                              (H, W)) 

                    feature_shape = neighbor_feature.shape
                    padding_len = self.max_cav - feature_shape[0]
                    padding_feature = torch.zeros(padding_len, feature_shape[1],
                                                  feature_shape[2], feature_shape[3])
                    padding_feature = padding_feature.to(
                        neighbor_feature.device)
                    neighbor_feature = torch.cat([neighbor_feature, padding_feature],
                                                 dim=0)

                    if i == 0:
                        padding_map = torch.zeros(
                            padding_len, 1, feature_shape[2], feature_shape[3])
                        padding_map = padding_map.to(neighbor_feature.device)
                        neighbor_psm = torch.cat(
                            [neighbor_psm, padding_map], dim=0)
                        neighbor_psm_list.append(neighbor_psm)
                        
                    if self.agg_mode == "STCFormer":
                        fusion, output_list = self.fuse_modules[i](neighbor_feature, neighbor_psm_list[b], neighbor_feature_his, i)
                        x_fuse.append(fusion)
                        x_temporal.append(output_list[0])
                        x_exclusive.append(output_list[1])
                        x_common.append(output_list[2])

                x_fuse = torch.stack(x_fuse)
                x_temporal = torch.stack(x_temporal)
                x_exclusive = torch.stack(x_exclusive)
                x_common = torch.stack(x_common)

                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                    ups_temporal.append(backbone.deblocks[i](x_temporal))
                    ups_exclusive.append(backbone.deblocks[i](x_exclusive))
                    ups_common.append(backbone.deblocks[i](x_common))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
                x_temporal = torch.cat(ups_temporal, dim=1)
                x_exclusive = torch.cat(ups_exclusive, dim=1)
                x_common = torch.cat(ups_common, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]

            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
                
        return x_fuse, communication_rates, {}, offset_loss, commu_loss, None, [x_temporal, x_exclusive, x_common]

    def _fuse_point_cloud(self, raw_points_list, sparse_points):
        """
        将稀疏点云与原始点云融合
        raw_points_list: List of dict, 原始体素数据
        sparse_points: [B, C, H, W] 稀疏点云BEV特征
        输出: 融合后的点云（格式需与后续处理兼容）
        """
        # 示例：简单拼接原始点云和稀疏点云
        fused_points = []
        for b in range(len(raw_points_list)):
            raw_coords = raw_points_list[b]['coords']
            raw_features = raw_points_list[b]['features']

            # 将 sparse_points 转换为体素坐标
            sparse_coords = self._bev_to_voxel_coords(sparse_points[b])
            sparse_features = sparse_points[b].permute(1, 2, 0).reshape(-1, sparse_points.shape[1])

            # 拼接
            fused_coords = torch.cat([raw_coords, sparse_coords], dim=0)
            fused_features = torch.cat([raw_features, sparse_features], dim=0)
            fused_points.append({'coords': fused_coords, 'features': fused_features})
        return fused_points