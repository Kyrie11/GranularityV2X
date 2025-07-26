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

    # g2的shape是[N,C,H,W]
    # g1的shape是[N,C',H/2,W/2]
    # 我们只加载一个batch数据，所以不再使用record_len
    def forward(self, current_data, short_term, long_term):
        #===========current时刻的数据================
        #返回的是三个元素个数为1的列表
        g1_data, g2_data, g3_data = self.get_histroy_granularity([current_data])
        # 从列表中分离
        g1_data = g1_data[0]
        g2_data = g2_data[0]
        g3_data = g3_data[0]
        current_data_dict = current_data['ego']
        pairwise_t_matrix = current_data_dict['pairwise_t_matrix'].clone().detach()
        print(f"pairwise_t_matrix.shape={pairwise_t_matrix.shape}")
        record_len = current_data_dict['record_len'][0] #只有一个batch
        print(f"g1_data.shape={g1_data.shape}")
        print(f"g2_data.shape={g2_data.shape}")
        print(f"g3_data.shape={g3_data.shape}")
        #所有agent的延迟时间
        delay = short_term[0]['ego']['time_delay'][record_len]
        print(f"delay={delay}")
        print(f"delay.shape={delay.shape}")

        short_his_g1, short_his_g2, short_his_g3 = self.get_histroy_granularity(short_term)
        long_his_g1, long_his_g2, long_his_g3 = self.get_histroy_granularity(long_term)
        print(f"long_his_g1[0].shape={long_his_g1[0].shape}")
        print(f"long_his_g2[0].shape={long_his_g2[0].shape}")
        print(f"long_his_g3[0].shape={long_his_g3[0].shape}")

        short_his = [short_his_g1, short_his_g2, short_his_g3]
        long_his = [long_his_g1, long_his_g2, long_his_g3]

        if len(long_term) <= 1:
            fused_feature, commu_volume, offset_loss, commu_loss = self.fusion_net(
                g1_data=g1_data, g2_data=g2_data, g3_data=g3_data, record_len=record_len, pairwise_t_matrix=pairwise_t_matrix, backbone=self.backbone)
        else:
            fused_feature, commu_volume, offset_loss, commu_loss = self.fusion_net(
                g1_data=g1_data, g2_data=g2_data, g3_data=g3_data, record_len=record_len, pairwise_t_matrix=pairwise_t_matrix, backbone=self.backbone, delay=delay, short_his=short_his, long_his=long_his)
        print("fused_feat_list.shape=",fused_feature.shape)
        # if self.shrink_flag:
        #     fused_feature = self.shrink_conv(fused_feature)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output_dict = {'psm':psm, 'rm':rm, 'commu_loss':commu_loss, 'offset_loss':offset_loss, 'commu_volume':commu_volume}
        return output_dict
        return None


    def get_histroy_granularity(self, history):
        his_g1, his_g2, his_g3 = [], [], []
        matrix_list = []
        for origin_frame in history:
            data_dict = origin_frame['ego']
            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
            record_len = data_dict['record_len']
            # print(f"voxel_features:{voxel_features}")
            pairwise_t_matrix = data_dict['pairwise_t_matrix']
            batch_dict = {'voxel_features': voxel_features,
                          'voxel_coords': voxel_coords,
                          'voxel_num_points': voxel_num_points,
                          'record_len': record_len}
            # n, 4 -> n, c encoding voxel feature using point-pillar method
            batch_dict = self.pillar_vfe(batch_dict)
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)
            spatial_features = batch_dict['spatial_features']
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

            spatial_features = batch_dict['spatial_features']
            his_g2.append(spatial_features)
            matrix_list.append(pairwise_t_matrix)
            g1 = self.get_g1_bev(voxel_features, voxel_num_points, voxel_coords)
            his_g1.append(g1)
            psm = self.cls_head(spatial_features_2d)
            rm = self.reg_head(spatial_features_2d)
            g3 = self.get_g3_bev(psm, rm)
            his_g3.append(g3)
        return his_g1, his_g2, his_g3

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

        # ####################################################################
        # ## 5. 修正后的散射逻辑 ##############################################
        # ####################################################################
        # 从scatter模块获取BEV尺寸信息
        batch_size = coords[:, 0].max().int().item() + 1
        H_full, W_full = self.scatter.ny, self.scatter.nx
        downsample_factor = 2  # 与backbone的下采样率匹配
        H_down, W_down = H_full // downsample_factor, W_full // downsample_factor

        # 创建空的BEV图用于累加
        sum_bev_map = torch.zeros(batch_size, 8, H_down, W_down,
                                  dtype=physical_pillar_features.dtype,
                                  device=physical_pillar_features.device)
        count_bev_map = torch.zeros_like(sum_bev_map)

        # 计算每个pillar在下采样BEV图中的一维索引 (所有pillar一起计算)
        # coords[:, 2] 是 y 坐标, coords[:, 3] 是 x 坐标
        y_indices = coords[:, 2] // downsample_factor
        x_indices = coords[:, 3] // downsample_factor
        # bev_indices 的 shape 是 (M,)
        bev_indices = (y_indices * W_down + x_indices).long()

        # 遍历批次中的每个样本
        for i in range(batch_size):
            # 找到属于当前样本的 pillar
            sample_mask = (coords[:, 0] == i)
            if not sample_mask.any():
                continue

            # 获取当前样本的 pillar 特征 (src) 和它们的目标索引 (index)
            src = physical_pillar_features[sample_mask]  # Shape: (num_pillars_in_sample, 8)
            index = bev_indices[sample_mask]  # Shape: (num_pillars_in_sample,)

            # 为了使用 scatter_add_，我们需要将 1D 的 index 扩展以匹配 src 的形状
            # 我们要在 H_down * W_down 这个维度上进行散射
            # 所以 index 需要被扩展成 (num_pillars_in_sample, 8)
            index_expanded = index.view(-1, 1).expand_as(src)

            # 在当前样本的BEV图层上进行操作
            # 先将 BEV 图 flatten 成 (8, H*W)
            # 然后将 src (num_pillars, 8) 转置成 (8, num_pillars)
            # 将 index (num_pillars, 8) 转置成 (8, num_pillars)
            # 这样就可以在 dim=1 (空间维度)上进行散射了
            sum_bev_map[i].view(8, -1).scatter_add_(dim=1, index=index_expanded.t(), src=src.t())

            # 同样的方法累加计数
            ones_to_add = torch.ones_like(src)
            count_bev_map[i].view(8, -1).scatter_add_(dim=1, index=index_expanded.t(), src=ones_to_add.t())

        # 计算平均值, 加上 epsilon 防止除以零
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



