import torch
import torch.nn as nn
import torch.nn.functional as F
from v2xvit.models.sub_modules.pillar_vfe import PillarVFE
from v2xvit.models.sub_modules.point_pillar_scatter import PointPillarScatter
from v2xvit.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from v2xvit.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from v2xvit.models.sub_modules.downsample_conv import DownsampleConv
from v2xvit.models.sub_modules.naive_compress import NaiveCompressor
from v2xvit.models.fuse_modules.how2comm_deformable import How2comm



def transform_feature(feature_list, delay):
    return feature_list[delay]


class PointPillarHow2comm(nn.Module):
    def __init__(self, args):
        super(PointPillarHow2comm, self).__init__()
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.dcn = False

        self.fusion_net = How2comm(args['fusion_args'], args)
        self.frame = args['fusion_args']['frame']
        self.delay = args['fusion_args']['delay']
        self.discrete_ratio = args['fusion_args']['voxel_size'][0]
        self.downsample_rate = args['fusion_args']['downsample_rate']
        self.multi_scale = args['fusion_args']['multi_scale']

        self.anchor_number = args['anchor_number']
        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        if args['backbone_fix']:
            self.backbone_fix()

        self.history_max_len = args.get("history_max_len", 10)

        self.num_short_frames = 3
        self.num_long_frames = 3
        self.long_interval = 3

        self.score_threshold = 0.2


    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, short_term, long_term):
        short_his_g1, short_his_g2, short_his_g3 = [], [], []
        batch_dict_list = []
        feature_list = []
        feature_2d_list = []
        matrix_list = []
        regroup_feature_list = []
        regroup_feature_list_large = []

        for origin_data in short_term:
            data_dict = origin_data['ego']
            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
            record_len = data_dict['record_len']

            pairwise_t_matrix = data_dict['pairwise_t_matrix']
            batch_dict = {'voxel_features': voxel_features,
                          'voxel_coords': voxel_coords,
                          'voxel_num_points': voxel_num_points,
                          'record_len': record_len}
            # n, 4 -> n, c encoding voxel feature using point-pillar method
            batch_dict = self.pillar_vfe(batch_dict)
            # n, c -> N, C, H, W
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)
            # N, C, H', W'
            spatial_features_2d = batch_dict['spatial_features_2d']

            # downsample feature to reduce memory
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            # compressor
            if self.compression:
                spatial_features_2d = self.naive_compressor(
                    spatial_features_2d)
            # dcn
            if self.dcn:
                spatial_features_2d = self.dcn_net(spatial_features_2d)


            batch_dict_list.append(batch_dict)
            spatial_features = batch_dict['spatial_features']
            feature_list.append(spatial_features)
            feature_2d_list.append(spatial_features_2d)
            matrix_list.append(pairwise_t_matrix)
            regroup_feature_list.append(self.regroup(
                spatial_features_2d, record_len))
            regroup_feature_list_large.append(
                self.regroup(spatial_features, record_len))

            print("spatial_features.shape=", spatial_features.shape)
            print("spatial_features_2d.shape=", spatial_features_2d.shape)
            g1 = self.get_g1_bev(voxel_features, voxel_num_points, voxel_coords)
            print("vox_bev.shape=", g1.shape)
            psm = self.cls_head(spatial_features)
            rm = self.reg_head(spatial_features)
            g3 = self.get_g3_bev(psm, rm)
            print("det_bev.shape=", g3.shape)

        pairwise_t_matrix = matrix_list[0].clone().detach()

        history_feature = transform_feature(regroup_feature_list_large, self.delay)
        spatial_features = feature_list[0]
        spatial_features_2d = feature_2d_list[0]
        batch_dict = batch_dict_list[0]
        record_len = batch_dict['record_len']
        psm_single = self.cls_head(spatial_features_2d)

        # if delay == 0:
        #     fused_feature, commu_volume, offset_loss, commu_loss = self.fusion_net(
        #         g1_data=vox_bev, g2_data=feat_bev, g3_data=det_bev, record_len=record_len, pairwise_t_matrix=pairwise_t_matrix, backbone=self.backbone)
        # elif delay > 0:
        #     fused_feature, commu_volume, offset_loss, commu_loss = self.fusion_net(
        #         g1_data=vox_bev, g2_data=feat_bev, g3_data=det_bev, record_len=record_len, pairwise_t_matrix=pairwise_t_matrix, backbone=self.backbone, delay=delay, his_vox=his_vox, his_feat=his_feat, his_det=his_det)
        # print("fused_feat_list.shape=",fused_feature.shape)
        # # if self.shrink_flag:
        # #     fused_feature = self.shrink_conv(fused_feature)
        #
        # psm = self.cls_head(fused_feature)
        # rm = self.reg_head(fused_feature)
        # output_dict = {'psm':psm, 'rm':rm, 'commu_loss':commu_loss, 'offset_loss':offset_loss, 'commu_volume':commu_volume}
        # return output_dict
        return None

    def get_g1_bev(self, voxel_features, voxel_num_points, coords):
        max_points = voxel_features.shape[1]
        mask = self.pillar_vfe.get_paddings_indicator(voxel_num_points, max_points, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)

        #用掩码将无效填充点的数据清零
        masked_voxel_features = voxel_features * mask

        #计算8个物理量
        #为了避免除以0，添加一个极小值
        epsilon = 1e-6
        safe_voxel_num_points = voxel_num_points.view(-1, 1).float() + epsilon
        # 通道0：点数
        num_points_feat = voxel_num_points.view(-1, 1).float()
        # 通道 1: 平均强度 (Mean intensity) - 强度在第3个索引
        mean_intensity = torch.sum(masked_voxel_features[:,:,3], dim=1, keepdim=True) / safe_voxel_num_points
        # 通道 2: 平均高度 (Mean height) - z坐标在第2个索引
        mean_height = torch.sum(masked_voxel_features[:, :, 2], dim=1, keepdim=True) / safe_voxel_num_points
        # 通道 3: 最高点高度 (Max height)
        # 将掩码外的值设为极小值, 以免影响max的计算
        max_height = (masked_voxel_features[:, :, 2] + (1 - mask.squeeze(-1)) * -1e6).max(dim=1, keepdim=True)[0]
        # 通道 4: 高度跨度 (Height span)
        # 将掩码外的值设为极大值, 以免影响min的计算
        min_height = (masked_voxel_features[:, :, 2] + (1 - mask.squeeze(-1)) * 1e6).min(dim=1, keepdim=True)[0]
        height_span = max_height - min_height
        # 计算方差需要更精细的操作，以确保只在有效点上计算
        points_sum = torch.sum(masked_voxel_features[:, :, :3], dim=1)  # (M, 3)
        points_mean = points_sum / safe_voxel_num_points
        points_sq_sum = torch.sum(masked_voxel_features[:, :, :3] ** 2, dim=1)
        points_mean_sq = points_sq_sum / safe_voxel_num_points
        variance = torch.clamp(points_mean_sq - points_mean ** 2, min=0) #确保方差非负
        # 通道 5, 6, 7: x, y, z 方差 (x, y, z variance)
        xyz_variance = variance.split(1, dim=-1)
        x_variance, y_variance, z_variance = xyz_variance[0], xyz_variance[1], xyz_variance[2]
        # 4. 将8个特征合并
        # Shape: (M, 8)
        physical_pillar_features = torch.cat([
            num_points_feat, mean_intensity, mean_height,
            max_height, height_span, x_variance, y_variance, z_variance
        ], dim=1)

        # 5. 将物理特征散射到BEV图上 (借鉴PointPillarScatter的逻辑)
        # batch_size, H, W 可以从self.scatter获取
        N = coords[:, 0].max().int().item() + 1
        H_full, W_full = self.scatter.ny, self.scatter.nx
        downsample_factor = 2
        H_downsampled, W_downsampled = H_full // downsample_factor, W_full // downsample_factor

        # 创建下采样分辨率的BEV画布
        sum_bev_map = torch.zeros(N, 8, H_downsampled, W_downsampled, dtype=physical_pillar_features.dtype,
                                  device=physical_pillar_features.device)
        count_bev_map = torch.zeros(N, 8, H_downsampled, W_downsampled, dtype=physical_pillar_features.dtype,
                                    device=physical_pillar_features.device)
        epsilon = 1e-6

        # 准备用于scatter操作的索引
        # 索引需要是 (N, C, H, W) 的一维展开形式
        agent_indices = coords[:, 0].view(-1, 1).expand(-1, 8)  # (M, 8)
        downsampled_coords_y = (coords[:, 2] // downsample_factor).view(-1, 1).expand(-1, 8)
        downsampled_coords_x = (coords[:, 3] // downsample_factor).view(-1, 1).expand(-1, 8)

        # 将 pillar 特征散射并累加到 sum_bev_map
        # torch.scatter_add_(dim, index, src) -> self[index[i][j]] += src[i][j]
        # 我们需要将2D的(y,x)坐标和agent索引组合成一个可以在4D张量上操作的索引
        # 一个更高效的方法是先在2D的 H*W 上操作，然后 reshape
        sum_bev_map_flat = sum_bev_map.view(N, 8, H_downsampled, W_downsampled)
        count_bev_map_flat = count_bev_map.view(N, 8, H_downsampled, W_downsampled)

        # 计算一维的 BEV 索引
        bev_indices_flat = (downsampled_coords_y * W_downsampled + downsampled_coords_x).long()  # Shape: (M, 8)

        # 散射逻辑保持不变，但现在是在下采样后的网格上操作
        # 注意：多个原始Pillar可能会映射到同一个下采样后的Pillar。
        # 这里的简单实现是“后来者覆盖”，即后面的Pillar信息会覆盖前面的
        for agent_idx in range(N):
            agent_mask = (coords[:, 0] == agent_idx)
            if not agent_mask.any():
                continue
            # 获取当前 agent 的 pillar 特征和它们的目标索引
            src = physical_pillar_features[agent_mask]  # (num_pillars, 8)
            index = bev_indices_flat[agent_mask][:, 0].view(-1, 1).expand(-1, 8)  # (num_pillars, 8)

            # 在该 agent 对应的图层上进行累加
            sum_bev_map_flat[agent_idx].scatter_add_(dim=1, index=index.T, src=src.T)
            # 计数（每个pillar贡献1）
            ones_to_add = torch.ones_like(src)
            count_bev_map_flat[agent_idx].scatter_add_(dim=1, index=index.T, src=ones_to_add.T)

        # 计算平均值
        # 加上 epsilon 防止除以零
        physical_bev_map = sum_bev_map / (count_bev_map + epsilon)

        return physical_bev_map


    def get_g3_bev(self, psm, rm):
        N, anchor_num, H, W = psm.shape
        prob = torch.sigmoid(psm)
        max_probs, best_anchor_indices = torch.max(prob, dim=1)
        confidence_mask = max_probs > self.score_threshold
        object_map = torch.zeros(N, 8, H, W, device=psm.device, dtype=psm.dtype)
        if confidence_mask.any():
            object_map[:, 0, :, :][confidence_mask] = max_probs[confidence_mask]
            rm_reshaped = rm.view(N, anchor_num, 7, H, W)
            indices_for_gather = best_anchor_indices.unsqueeze(1).unsqueeze(1).expand(-1, -1, 7, -1, -1)
            selected_rm = torch.gather(rm_reshaped, dim=1, index=indices_for_gather)
            selected_rm = selected_rm.squeeze(1)
            expanded_confidence_mask = confidence_mask.unsqueeze(1).expand(-1, 7, -1, -1)
            object_map[:, 1:, :, :][expanded_confidence_mask] = selected_rm[expanded_confidence_mask]
        return object_map



