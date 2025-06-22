from numpy import record
import torch.nn as nn
import torch.nn.functional as F
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
        self.delay = 3
        self.discrete_ratio = args['fusion_args']['voxel_size'][0]
        self.downsample_rate = args['fusion_args']['downsample_rate']
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        if args['backbone_fix']:
            self.backbone_fix()

        self.history_max_len = args.get("history_max_len", 10)


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
        regroup_vox_list = []
        regroup_det_list = []


        for origin_data in data_dict_list:
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
            print("feature_2d_list.len=",len(feature_2d_list))
            matrix_list.append(pairwise_t_matrix)
            regroup_feature_list.append(self.regroup(
                spatial_features_2d, record_len))
            print("regroup_feature_list.len=",len(regroup_feature_list))
            regroup_feature_list_large.append(
                self.regroup(spatial_features, record_len))

            vox_bev = batch_dict['vox_bev']
            #下采样
            vox_bev = F.interpolate(vox_bev, scale_factor=0.5, mode="bilinear", align_corners=False)
            regroup_vox_list.append(self.regroup(vox_bev, record_len))
            psm = self.cls_head(spatial_features_2d)
            rm = self.reg_head(spatial_features_2d)
            # target_H, target_W = spatial_features.shape[2], spatial_features.shape[3]
            # psm = F.interpolate(psm, size=(target_H, target_W), mode='bilinear', align_corners=False)
            # rm = F.interpolate(rm, size=(target_H, target_W), mode="bilinear", align_corners=False)
            det_bev = torch.cat([psm, rm], dim=1)
            regroup_det_list.append(self.regroup(det_bev, record_len))



        pairwise_t_matrix = matrix_list[0].clone().detach()


        short_history_feature = regroup_feature_list_large[-1:-4:-1]
        long_history_feature = regroup_feature_list_large[len(regroup_feature_list)-1::-4]

        short_history_vox = regroup_vox_list[-1:-4:-1]
        long_history_vox = regroup_vox_list[len(regroup_feature_list)-1::-4]

        short_history_det = regroup_det_list[-1:-4:-1]
        long_history_det = regroup_det_list[len(regroup_feature_list)-1::-4]

        spatial_features = feature_list[0]
        spatial_features_2d = feature_2d_list[0]
        batch_dict = batch_dict_list[0]
        record_len = batch_dict['record_len']
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)

        # target_H, target_W = spatial_features.shape[2], spatial_features.shape[3]
        # upsampled_psm= F.interpolate(psm_single, size=(target_H, target_W), mode='bilinear', align_corners=False)
        # upsampled_rm = F.interpolate(rm_single, size=(target_H, target_W), mode="bilinear", align_corners=False)
        #得到三个粒度的bev
        vox_bev = torch.tensor(batch_dict['vox_bev'])
        # det_bev = torch.cat([upsampled_psm, upsampled_rm], dim=1)
        det_bev = torch.cat([psm_single, rm_single], dim=1)
        fused_bev = [vox_bev, spatial_features, det_bev]

        fused_long_his = [long_history_vox, long_history_feature, long_history_det]
        fused_short_his = [short_history_vox, short_history_feature, short_history_det]


        if self.delay == 0:
            fused_feature, communication_rates, result_dict, offset_loss, commu_loss, _, _ = self.fusion_net(
                fused_bev, psm_single, record_len, pairwise_t_matrix, self.backbone,
                [self.shrink_conv, self.cls_head, self.reg_head])
        elif self.delay > 0:
            fused_feature, communication_rates, result_dict, offset_loss, commu_loss, _, _ = self.fusion_net(
                fused_bev, psm_single, record_len, pairwise_t_matrix, self.backbone,
                [self.shrink_conv, self.cls_head, self.reg_head], short_history=fused_short_his, long_history=fused_long_his)
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
