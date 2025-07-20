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

    def forward(self, data_dict_list):
        delay = 0
        batch_dict_list = []
        feature_2d_list = []
        matrix_list = []
        his_vox = []
        his_feat = []
        his_det = []
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
            print("voxel_features.shape=", voxel_features.shape)
            print("voxel_coords.shape=", voxel_coords.shape)
            # n, 4 -> n, c encoding voxel feature using point-pillar method
            batch_dict = self.pillar_vfe(batch_dict)
            pillar_features = batch_dict['pillar_features']
            print("pillar_features.shape=", pillar_features.shape)
            # n, c -> N, C, H, W
            vox_bev = batch_dict['vox_bev']
            print("vox_bev.shape=", vox_bev.shape)
            # 下采样
            vox_bev = F.interpolate(vox_bev, scale_factor=0.5, mode="bilinear", align_corners=False)
            his_vox.append(vox_bev)
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



            psm = self.cls_head(spatial_features_2d)
            rm = self.reg_head(spatial_features_2d)

            B, anchor_num, H, W = psm.shape
            prob = torch.sigmoid(psm)
            max_probs, best_anchor_indices = torch.max(prob, dim=1)
            confidence_mask = max_probs > self.score_threshold
            object_map = torch.zeros(B, 8, H, W, device=psm.device, dtype=psm.dtype)
            if confidence_mask.any():
                object_map[:,0,:,:][confidence_mask] = max_probs[confidence_mask]
                rm_reshaped = rm.view(B, anchor_num, 7, H, W)
                indices_for_gather = best_anchor_indices.unsqueeze(1).unsqueeze(1).expand(-1,-1,7,-1,-1)
                selected_rm = torch.gather(rm_reshaped, dim=1, index=indices_for_gather)
                selected_rm = selected_rm.squeeze(1)
                expanded_confidence_mask = confidence_mask.unsqueeze(1).expand(-1,7,-1,-1)
                object_map[:,1:,:,:][expanded_confidence_mask] = selected_rm[expanded_confidence_mask]
            # temporal_output_dict = {'psm':psm, 'rm':rm}
            # detections = self.post_process(dataset, temporal_output_dict, record_len)
            # det_bev = torch.cat([psm, rm], dim=1)
            his_det.append(object_map)

        pairwise_t_matrix = matrix_list[0].clone().detach()

        # spatial_features_2d = feature_2d_list[0]
        batch_dict = batch_dict_list[0]
        record_len = batch_dict['record_len']

        vox_bev = his_vox[0]
        feat_bev = his_feat[0]
        det_bev = his_det[0]

        if delay == 0:
            fused_feature, commu_volume, offset_loss, commu_loss = self.fusion_net(
                g1_data=vox_bev, g2_data=feat_bev, g3_data=det_bev, record_len=record_len, pairwise_t_matrix=pairwise_t_matrix)
        elif delay > 0:
            fused_feature, commu_volume, offset_loss, commu_loss = self.fusion_net(
                g1_data=vox_bev, g2_data=feat_bev, g3_data=det_bev, record_len=record_len, pairwise_t_matrix=pairwise_t_matrix, delay=delay, his_vox=his_vox, his_feat=his_feat, his_det=his_det)
        print("fused_feat_list.shape=",fused_feature.shape)
        # if self.shrink_flag:
        #     fused_feature = self.shrink_conv(fused_feature)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output_dict = {'psm':psm, 'rm':rm, 'commu_loss':commu_loss, 'offset_loss':offset_loss, 'commu_volume':commu_volume}
        return output_dict


    def post_process(self, dataset, output_dict, record_len):
        psm = output_dict['psm']
        rm  = output_dict['rm']

        anchor_box = dataset.post_processor.generate_anchor_box()
        anchor_box = torch.tensor(anchor_box)
        anchor_box = anchor_box.to(psm.device)

        psm_by_batch = self.regroup(psm, record_len)
        rm_by_batch = self.regroup(rm, record_len)

        detections_by_batch = []
        batch_size = len(record_len)

        for i in range(batch_size):
            num_agents_in_sample = record_len[i]
            sample_detections = {}

            for j in range(num_agents_in_sample):
                agent_psm = psm_by_batch[i][j].unsqueeze(0)
                agent_rm = rm_by_batch[i][j].unsqueeze(0)

                local_boxes, local_scores = self.decode_single_agent_output(psm=agent_psm, rm=agent_rm, anchor_box=anchor_box)

                agent_key = f"agent_{j}"
                sample_detections[agent_key] = {
                    'box_tensor': local_boxes,
                    'score_tensor': local_scores
                }

            detections_by_batch.append(sample_detections)
        return detections_by_batch

    def decode_single_agent_output(self, psm, rm, anchor_box):
        """
            Decodes the raw output of a single agent into a list of bounding boxes
            and scores, without any coordinate transformation or NMS.
        """
        prob = torch.sigmoid(psm.permute(0, 2, 3, 1))
        prob = prob.reshape(1, -1)
        batch_box3d = self.delta_to_boxes3d(rm, anchor_box)
        mask = torch.gt(prob, self.score_threshold)
        mask = mask.view(1, -1)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)
        decoded_boxes = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
        decoded_scores = torch.masked_select(prob[0], mask[0])

        return decoded_boxes, decoded_scores

    def delta_to_boxes3d(self, deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d