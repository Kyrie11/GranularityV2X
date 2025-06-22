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


class VoxelProjector(nn.Module):
    def __init__(self, in_channels=4, bev_channels=256, voxel_size=0.4):
        super().__init__()
        self.voxel_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 256)
        )
        self.bev_fusion = nn.Sequential(
            nn.Conv2d(bev_channels * 2, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU()
        )
        self.voxel_size = voxel_size

    def forward(self, bev_feat, record_len, sparse_voxels, sparse_coords, t_matrix):
        """
        :param bev_feat: [B, C, H, W] 原始BEV特征
        :param sparse_voxels: list[Tensor] 各agent的稀疏体素特征
        :param sparse_coords: list[Tensor] 各agent的体素坐标
        :param t_matrix: [B, L, L, 4,4] 坐标变换矩阵
        """
        batch_projected = []
        B= t_matrix.shape[0]
        for b in range(B):
            projected = []
            # 当前batch的变换矩阵
            cav_num = record_len[b]
            t_matrix_batch = t_matrix[b]  # [L, L, 4,4]
            # 初始化投影特征
            C, H, W = bev_feat.shape[1:]
            # 遍历每个协作agent（从索引1开始）
            for agent_id in range(0, cav_num-1):
                agent_projected = torch.zeros_like(bev_feat[b])
                print("len(sparse_voxels[b][i]=", sparse_voxels[b][agent_id].shape)
                # 坐标转换
                homog_coords = F.pad(sparse_coords[b][agent_id][:, 1:], (0, 1), value=1.0)
                # print("t_matrix_batch的形状是", t_matrix_batch[0, agent_id].shape)
                # print("homog_coords的形状是", homog_coords.shape)
                ego_coords = ((t_matrix_batch[0, agent_id]).double() @ (homog_coords.T).double()).T[:, :3]

                # 量化到BEV网格
                x_idx = (ego_coords[:, 0] / self.voxel_size).long().clamp(0, W - 1)
                print("x_idx:",x_idx)
                y_idx = (ego_coords[:, 1] / self.voxel_size).long().clamp(0, H - 1)
                voxel_features = sparse_voxels[b][agent_id].mean(dim=1)
                encoded = self.voxel_encoder(voxel_features)
                print("encoded.shape=", encoded.shape)
                valid_mask = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)
                print("projected的shape是：", agent_projected.shape)
                agent_projected.index_put_(
                    indices=(y_idx[valid_mask], x_idx[valid_mask]),
                    values=encoded[valid_mask],
                    accumulate=True
                )
                # print("voxel_features.shape=", voxel_features.shape)
                # 特征编码
                 # [64, N_selected]
                indices = y_idx*W + x_idx


                # 累积到投影特征
                # agent_projected.scatter_add_(1,
                #                        indices.unsqueeze(0).expand(C, -1),  # [C, N_selected]
                #                        encoded)
                # 与原始特征融合


                # fused = self.bev_fusion(torch.cat([
                #     bev_feat[b].unsqueeze(0),
                #     projected.unsqueeze(0)
                # ], dim=1))
                projected.append(agent_projected)
            batch_projected.append(projected)

        return batch_projected

class HierarchicalFusion(nn.Module):
    def __init__(self, in_channels=64, bev_channels=64):
        super().__init__()
        # 体素特征增强分支
        self.voxel_fusion = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU()
        )

        # 稀疏特征增强分支
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU()
        )

        # 跨模态注意力融合
        self.cross_attention = nn.Sequential(
            nn.Conv2d(bev_channels * 2, bev_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(bev_channels // 2, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, ego_feat, voxel_bev, sparse_feat):
        """
        ego_feat: [B,64,H,W] 原始特征
        voxel_bev: [B,64,H,W] 体素投影特征
        sparse_feat: [B,64,H,W] 稀疏通信特征
        """
        # 体素特征增强
        voxel_enhanced = self.voxel_fusion(voxel_bev)  # [B,64,H,W]

        # 稀疏特征增强
        feature_enhanced = self.feature_fusion(sparse_feat)  # [B,64,H,W]

        # 跨模态注意力权重
        attention = self.cross_attention(
            torch.cat([voxel_enhanced, feature_enhanced], dim=1))  # [B,2,H,W]

        # 加权融合
        fused_feat = (attention[:, 0:1] * voxel_enhanced +
                      attention[:, 1:2] * feature_enhanced)  # [B,64,H,W]

        # 残差连接
        return ego_feat + fused_feat


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
        self.async_flag = False
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

        self.voxel_projector = VoxelProjector(
            in_channels=32,
            bev_channels=64 if 'num_filters' not in args else args['num_filters'][0],
            voxel_size=args['voxel_size'][0]
        )

        self.hierarchical_fusion = HierarchicalFusion()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, fused_bev, psm, record_len, pairwise_t_matrix, backbone=None, heads=None, short_history=None, long_history=None):
        vox_bev, x, det_bev = fused_bev
        _, _, H, W = x.shape
        pairwise_t_matrix_4d = pairwise_t_matrix
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
            feat_final, offset_loss = self.how2comm(fused_bev, history, record_len, backbone, heads)
            comp_F_vox_t, comp_F_feat_t, comp_F_det_bev_t, _ = self.mgdc_bev_compensator(
            #
            )
            x = feat_final
        else:
            offset_loss = torch.zeros(1).to(x.device)
        neighbor_psm_list = []
        if history:
            his_vox, his, his_det = history[0][0],history[1][0],history[2][0]
        else:
            his_vox, his, his_det = vox_bev, x, det_bev
        if self.multi_scale:
            ups = []
            ups_temporal = []
            ups_exclusive = []
            ups_common = []
            with_resnet = True if hasattr(backbone, 'resnet') else False
            print("his.shape=", his.shape)
            if with_resnet:
                feats = backbone.resnet(x)
                history_feats = backbone.resnet(his)

            for i in range(self.num_levels):
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                his = history_feats[i] if with_resnet else backbone.blocks[i](his)
                print("在resnet后feat的大小为:", x.shape)

                if i == 0:
                    if self.communication:
                        batch_confidence_maps = self.regroup(psm, record_len)
                        _, _, confidence_maps = self.naive_communication(batch_confidence_maps)
                        
                        batch_temp_features = self.regroup(x, record_len)
                        batch_temp_features_his = self.regroup(his, record_len)

                        batch_vox_features = self.regroup(vox_bev, record_len)
                        batch_vox_features_his = self.regroup(his_vox, record_len)
                        batch_det_features = self.regroup(det_bev, record_len)
                        batch_det_features_his = self.regroup(his_det, record_len)

                        temp_list = []
                        temp_vox_list = []
                        temp_det_list = []
                        temp_psm_list = []
                        history_list = []
                        history_vox_list = []
                        history_det_list = []
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

                            #添加vox和det信息
                            temp_vox = batch_vox_features[b]
                            neighbor_vox = warp_affine_simple(temp_vox,
                                                              t_matrix[0,
                                                              :, :, :],
                                                              (H, W))
                            temp_vox_list.append(neighbor_vox)

                            temp_det = batch_det_features[b]
                            neighbor_det = warp_affine_simple(temp_det,
                                                              t_matrix[0,
                                                              :, :, :],
                                                              (H, W))
                            temp_det_list.append(neighbor_det)


                            temp_features_his = batch_temp_features_his[b]
                            C, H, W = temp_features_his.shape[1:]
                            neighbor_feature_his = warp_affine_simple(temp_features_his,
                                                                  t_matrix[0,
                                                                           :, :, :],
                                                                  (H, W))
                            history_list.append(neighbor_feature_his)
                            
                            temp_psm_list.append(warp_affine_simple(confidence_maps[b], t_matrix[0, :, :, :], (H, W)))  

                            #添加vox和det历史信息
                            temp_vox_his = batch_vox_features_his[b]
                            C,H,W = temp_vox_his.shape[1:]
                            neighbor_vox_his = warp_affine_simple(temp_vox_his,
                                                                  t_matrix[0,
                                                                  :, :, :],
                                                                  (H, W))
                            history_vox_list.append(neighbor_vox_his)

                            temp_det_his = batch_det_features_his[b]
                            C,H,W = temp_det_his.shape[1:]
                            neighbor_det_his = warp_affine_simple(temp_det_his,
                                                                  t_matrix[0,
                                                                  :, :, :],
                                                                  (H, W))
                            history_det_list.append(neighbor_det_his)


                        x = torch.cat(temp_list, dim=0)
                        vox_bev = torch.cat(temp_vox_list, dim=0)
                        det_bev = torch.cat(temp_det_list, dim=0)

                        his = torch.cat(history_list, dim=0)
                        his_vox = torch.cat(history_vox_list, dim=0)
                        his_det = torch.cat(history_det_list, dim=0)
                        if self.communication_flag:
                            all_agents_sparse_transmitted_data, total_loss, sparse_history = self.how2comm.communication(
                            vox_bev,x,det_bev,record_len,history_vox_list,history_list,history_det_list,temp_psm_list)


                        else:
                            communication_rates = torch.tensor(0).to(x.device)
                            commu_loss = torch.zeros(1).to(x.device)
                    else:
                        communication_rates = torch.tensor(0).to(x.device)

                batch_node_features = self.regroup(x, record_len)
                batch_node_features_his = self.regroup(his, record_len)


                for b in range(B):
                    N = record_len[b]
                    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                    node_features = batch_node_features[b]
                    node_features_his = batch_node_features_his[b]
                    final_fused_bev, hcmgf_ss_losses = self.hcmgfs_fuser(
                        node_features,
                        node_features_his,
                        collab_uncertainties,
                        collab_consistency
                    )


                
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