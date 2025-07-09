from collections import OrderedDict

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

    def forward(self, data_dict_list, dataset):
        delay = 1
        batch_dict_list = []
        feature_2d_list = []
        matrix_list = []
        his_vox = []
        his_feat = []
        his_det = []
        for origin_data in data_dict_list:
            for cav_id, cav_content in origin_data.items():
                print("cav_id in origin_data:", cav_id)
                # print("cav_content:", cav_content)
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
            spatial_features = self.avg_pool(spatial_features)
            his_feat.append(spatial_features)

            # feature_2d_list.append(spatial_features_2d)
            matrix_list.append(pairwise_t_matrix)

            if delay>0:
                vox_bev = batch_dict['vox_bev']
                print("vox_bev.shape=", vox_bev.shape)
                # 下采样
                vox_bev = F.interpolate(vox_bev, scale_factor=0.5, mode="bilinear", align_corners=False)
                his_vox.append(vox_bev)

                psm = self.cls_head(spatial_features_2d)
                rm = self.reg_head(spatial_features_2d)
                temporal_output_dict = OrderedDict()
                temporal_output_dict['ego'] = {'psm': psm, 'rm': rm}
                pred_box_tensor, pred_score, _ = dataset.post_process(data_dict, temporal_output_dict)
                # target_H, target_W = spatial_features.shape[2], spatial_features.shape[3]
                # psm = F.interpolate(psm, size=(target_H, target_W), mode='bilinear', align_corners=False)
                # rm = F.interpolate(rm, size=(target_H, target_W), mode="bilinear", align_corners=False)
                det_bev = torch.cat([psm, rm], dim=1)
                his_det.append(det_bev)



        pairwise_t_matrix = matrix_list[0].clone().detach()

        # spatial_features_2d = feature_2d_list[0]
        batch_dict = batch_dict_list[0]
        record_len = batch_dict['record_len']
        # psm_single = self.cls_head(spatial_features_2d)
        # rm_single = self.reg_head(spatial_features_2d)
        # print("spatial_feature_2d.shape=", spatial_features_2d.shape)
        # print("spatial_feature.shape=", spatial_features.shape)
        # print("vox_bev.shape=",vox_bev.shape)
        # print("det_bev.shape=", det_bev.shape)



        # fused_his = [his_vox, his_feat, his_det]

        if delay == 0:
            fused_feature, commu_volume, offset_loss, commu_loss = self.fusion_net(
                record_len=record_len, pairwise_t_matrix=pairwise_t_matrix)
        elif delay > 0:
            fused_feature, commu_volume, offset_loss, commu_loss = self.fusion_net(
                record_len=record_len, pairwise_t_matrix=pairwise_t_matrix, delay=delay, his_vox=his_vox, his_feat=his_feat, his_det=his_det)
        print("fused_feat_list.shape=",fused_feature.shape)
        # if self.shrink_flag:
        #     fused_feature = self.shrink_conv(fused_feature)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output_dict = {'psm':psm, 'rm':rm, 'commu_loss':commu_loss, 'offset_loss':offset_loss, 'commu_volume':commu_volume}
        return output_dict
        # output_dict = {'psm': psm,
        #                'rm': rm
        #                }
        #
        # # output_dict.update(result_dict)
        # output_dict.update({'comm_rate': commu_volume,
        #                     "offset_loss": offset_loss,
        #                     'commu_loss': commu_loss
        #                     })
        # return output_dict
