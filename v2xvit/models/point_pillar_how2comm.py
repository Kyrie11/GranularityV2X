from numpy import record
import torch.nn as nn
from torch.nn import functional as F

from v2xvit.models.sub_modules.pillar_vfe import PillarVFE
from v2xvit.models.sub_modules.point_pillar_scatter import PointPillarScatter
from v2xvit.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from v2xvit.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from v2xvit.models.sub_modules.downsample_conv import DownsampleConv
from v2xvit.models.sub_modules.naive_compress import NaiveCompressor
from v2xvit.models.fuse_modules.how2comm_deformable import How2comm
import torch
from v2xvit.models.sub_modules.torch_transformation_utils import warp_affine_simple

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
        self.delay = 1
        self.discrete_ratio = args['fusion_args']['voxel_size'][0]
        self.downsample_rate = args['fusion_args']['downsample_rate']
        self.multi_scale = args['fusion_args']['multi_scale']

        detection_head_input_channels = args['base_bev_backbone']['num_upsample_filter'][-1]
        if self.shrink_flag:
            detection_head_input_channels = args['shrink_header']['dim'][-1]

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)


        # 新增：定义 F_det_bev 的目标通道数 (示例)
        # 通道0: 最高置信度 (objectness)
        # 通道1: 对应最高置信度的类别ID (整数)
        # (可选) 通道2-8: 对应最高置信度anchor的7个回归参数
        self.C_D = 2 + 7
        # self.num_classes = args['num_class']  # 获取类别数
        # self.num_anchors_per_loc = args['anchor_number']  # 获取每个位置的anchor数

        if args['backbone_fix']:
            self.backbone_fix()

    def generate_det_bev_from_heads(self, psm_single, rm_single, target_H_bev, target_W_bev):
        """
        从检测头的原始输出 psm_single 和 rm_single 生成 F_det_bev.
        基于方案A：每个像素的最佳Anchor信息编码。
        """
        batch_size = psm_single.shape[0]

        # 1. 上采样 (如果需要)
        current_H_head, current_W_head = psm_single.shape[2], psm_single.shape[3]
        if current_H_head != target_H_bev or current_W_head != target_W_bev:
            upsampled_psm = F.interpolate(psm_single, size=(target_H_bev, target_W_bev), mode='bilinear',
                                          align_corners=False)
            upsampled_rm = F.interpolate(rm_single, size=(target_H_bev, target_W_bev), mode='bilinear',
                                         align_corners=False)
        else:
            upsampled_psm = psm_single
            upsampled_rm = rm_single
        # upsampled_psm: [B, num_anchors * num_classes, H_bev, W_bev]
        # upsampled_rm: [B, num_anchors * 7, H_bev, W_bev]

        # 2. Reshape psm 以便处理每个anchor和每个类别
        # psm_single的通道是 num_anchors_per_loc * num_classes
        # 例如 B, 2*3, H, W -> B, 2, 3, H, W (假设3个class, 2个anchor)
        psm_reshaped = upsampled_psm.view(batch_size, self.num_anchors_per_loc, self.num_classes, target_H_bev,
                                          target_W_bev)

        # 3. 获取每个anchor每个位置的最佳类别概率和类别ID
        # (假设psm_reshaped中的值是logits，需要经过Sigmoid或Softmax得到概率)
        # 如果是多分类，通常在损失函数内部处理Softmax，这里可以直接用原始分数比较，或者先Softmax
        # 为了简化，我们假设已经是概率或者可以直接比较的分数
        # 如果是二分类（前景/背景），或者每个anchor只预测一个前景分数，逻辑会不同
        # 假设 psm_reshaped 已经是每个anchor对每个前景类的分数/概率

        # 找到每个anchor在所有类别中的最高分数及其类别索引
        anchor_best_score_per_class, anchor_best_class_id_per_class = torch.max(psm_reshaped, dim=2)
        # anchor_best_score_per_class: [B, num_anchors, H_bev, W_bev] (每个anchor对应的最高类别分数)
        # anchor_best_class_id_per_class: [B, num_anchors, H_bev, W_bev] (每个anchor对应的最高类别索引)

        # 4. 在所有anchor中找到每个位置的最佳分数和对应类别ID
        best_score_overall, best_anchor_idx = torch.max(anchor_best_score_per_class, dim=1)
        # best_score_overall: [B, H_bev, W_bev] (每个位置上所有anchor中的最高分数)
        # best_anchor_idx: [B, H_bev, W_bev] (每个位置上提供最高分数的anchor的索引)

        # 使用 best_anchor_idx 从 anchor_best_class_id_per_class 中收集对应的最佳类别ID
        # (需要小心处理索引，这里用gather)
        # best_class_id_overall = torch.gather(anchor_best_class_id_per_class, 1, best_anchor_idx.unsqueeze(1)).squeeze(1)
        # 上面这行可能有问题，因为anchor_best_class_id_per_class是[B,num_anchor,H,W], best_anchor_idx是[B,H,W]
        # 正确的gather方式：
        # 先将 anchor_best_class_id_per_class 调整为 [B, H, W, num_anchors]
        temp_class_ids = anchor_best_class_id_per_class.permute(0, 2, 3, 1)  # [B, H_bev, W_bev, num_anchors]
        # 将 best_anchor_idx 扩展维度以用于gather
        idx_for_gather = best_anchor_idx.unsqueeze(-1)  # [B, H_bev, W_bev, 1]
        best_class_id_overall = torch.gather(temp_class_ids, -1, idx_for_gather).squeeze(-1)  # [B, H_bev, W_bev]

        # 5. (可选) 提取对应最佳anchor的回归参数
        # upsampled_rm: [B, num_anchors * 7, H_bev, W_bev]
        # rm_reshaped = upsampled_rm.view(batch_size, self.num_anchors_per_loc, 7, target_H_bev, target_W_bev)
        # temp_reg_params = rm_reshaped.permute(0, 3, 4, 1, 2) # [B, H_bev, W_bev, num_anchors, 7]
        # best_reg_params = torch.gather(temp_reg_params, -2, idx_for_gather.unsqueeze(-1).expand(-1,-1,-1,-1,7)).squeeze(-2)
        # best_reg_params: [B, H_bev, W_bev, 7]

        # 6. 构建 F_det_bev
        F_det_bev = torch.zeros((batch_size, self.C_D, target_H_bev, target_W_bev), device=upsampled_psm.device,
                                dtype=upsampled_psm.dtype)

        # 通道0: 最高置信度 (objectness)
        F_det_bev[:, 0, :, :] = torch.sigmoid(best_score_overall)  # 假设best_score_overall是logits，用sigmoid转为概率

        # 通道1: 最佳类别ID (需要确保class_id是从0开始的整数)
        F_det_bev[:, 1, :, :] = best_class_id_overall.float()  # 转为float类型以匹配

        # 如果 C_D 包含回归参数 (示例)
        # if self.C_D == 2 + 7:
        #     F_det_bev[:, 2:, :, :] = best_reg_params.permute(0,3,1,2) # [B, 7, H_bev, W_bev]

        return F_det_bev

    def generate_vox_bev(self, batch_size, batch_dict, target_H, target_W):
        pillar_bev_coords_orig = batch_dict['voxel_coords'][:, [0,2,3]].long()# (b, y_vox_grid, x_vox_grid)

        valid_y_mask = (pillar_bev_coords_orig[:, 1] >= 0) & (pillar_bev_coords_orig[:, 1] < target_H)
        valid_x_mask = (pillar_bev_coords_orig[:, 2] >= 0) & (pillar_bev_coords_orig[:, 2] < target_W)
        valid_mask = valid_y_mask & valid_x_mask
        pillar_bev_coords = pillar_bev_coords_orig[valid_mask]

        num_points_per_pillar = batch_dict['voxel_num_points']


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

    def forward(self, data_dict_list):
        batch_dict_list = []  
        feature_list = []  
        feature_2d_list = []  
        matrix_list = []
        regroup_feature_list = []  
        regroup_feature_list_large = []

        raw_voxel_features_list = []
        raw_voxel_coords_list = []

        for origin_data in data_dict_list:
            data_dict = origin_data['ego']
            voxel_features = data_dict['processed_lidar']['voxel_features'].clone()
            print("voxel_features的shape", voxel_features.shape)
            raw_voxel_features_list.append(voxel_features)
            voxel_coords = data_dict['processed_lidar']['voxel_coords'].clone()
            raw_voxel_coords_list.append(voxel_coords)
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

        pairwise_t_matrix = matrix_list[0].clone().detach()
        history_feature = transform_feature(regroup_feature_list_large, self.delay)
        spatial_features = feature_list[0]
        spatial_features_2d = feature_2d_list[0]
        batch_dict = batch_dict_list[0]
        record_len = batch_dict['record_len']

        target_H, target_W = spatial_features.shape[2], spatial_features.shape[3]

        psm_single = self.cls_head(spatial_features_2d) #分类预测图 [batch_size, anchors, H, W]
        rm_single = self.reg_head(spatial_features_2d)  #回归预测图 [batch_size, anchors * cls_num, H, W]
        # det_bev = torch.cat([psm_single, rm_single], dim=0)
        # print("det_bev的形状：", det_bev.shape)

        if self.delay == 0:
            fused_feature, communication_rates, result_dict, offset_loss, commu_loss, _, _ = self.fusion_net(spatial_features, psm_single, record_len,pairwise_t_matrix,self.backbone,[self.shrink_conv, self.cls_head, self.reg_head], raw_voxels=raw_voxel_features_list[0], raw_coords=raw_voxel_coords_list[0])
        elif self.delay > 0:
            fused_feature, communication_rates, result_dict, offset_loss, commu_loss, _, _ = self.fusion_net(spatial_features, psm_single,record_len,pairwise_t_matrix,self.backbone,[self.shrink_conv, self.cls_head, self.reg_head], history=history_feature, raw_voxels=raw_voxel_features_list[0], raw_coords=raw_voxel_coords_list[0])
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        
        output_dict = {'psm': psm,
                        'rm': rm
                    }

        output_dict.update(result_dict)
        output_dict.update({'comm_rate': communication_rates,
                            "offset_loss": offset_loss,
                            'commu_loss': commu_loss
                            })
        return output_dict
