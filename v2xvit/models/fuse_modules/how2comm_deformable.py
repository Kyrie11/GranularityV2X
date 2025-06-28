from turtle import update
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import functional as F
from torch import batch_norm, einsum
from einops import rearrange, repeat
from icecream import ic

from v2xvit.models.comm_modules.mutual_communication import AdvancedCommunication
from v2xvit.models.sub_modules.torch_transformation_utils import warp_affine_simple
from v2xvit.models.sub_modules.mixed_feature_flow import ContextFusionMotionPredictor
from v2xvit.loss.flow_loss import CompensationLoss
from v2xvit.models.fuse_modules.multi_granularity_fusion import AgentSelfEnhancement, MultiGranularityFusionNet


class How2comm(nn.Module):
    def __init__(self, args, args_pre):
        super(How2comm, self).__init__()
        self.communication = False
        self.communication_flag = args['communication_flag']
        self.downsample_rate = args['downsample_rate']
        self.async_flag = False
        self.discrete_ratio = args['voxel_size'][0]

        #时延补偿模块
        self.mgdc_bev_compensator = ContextFusionMotionPredictor(args['mgdc_bev_args'])
        #时延补偿损失
        self.compensation_criterion = CompensationLoss(args['mgdc_bev_args'])
        #ego增强模块
        self.ego_enhance_model = AgentSelfEnhancement(d_model=64, num_history_frames=3, nhead_transformer=8, num_transformer_layers=2)
        #最终融合模块
        self.granularity_fusion = MultiGranularityFusionNet(args['granularity_trans'])
        # 通信模块
        self.communication_net = AdvancedCommunication(c_vox=10, c_feat=64, c_det=10)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    #对ego-agent在current时刻的帧进行增强，对neighbor-agent在delay时刻的帧进行增强
    #我先实现对ego-agent的增强
    def get_enhanced_feature(self, curr_bev, his_bev, record_len):
        ego_history_list = []
        batch_size = len(record_len)
        curr_bev_batch = self.regroup(curr_bev, record_len)
        curr_ego_features = torch.stack([sample_bev[0] for sample_bev in curr_bev_batch], dim=0)
        ego_history_list.append(curr_ego_features)

        for his_bev_t in his_bev:
            his_bev_t_batch = self.regroup(his_bev_t, record_len)
            his_ego_features_t = torch.stack([sample_bev[0] for sample_bev in his_bev_t_batch], dim=0)
            ego_history_list.append(his_ego_features_t)

        enhanced_features, occlusion_map, abnormal_map = self.ego_enhance_model(ego_history_list)

        # updated_curr_bev = curr_bev.clone()
        ego_indices = [0] + torch.cumsum(torch.tensor(record_len[:-1]), dim=0).tolist()
        for i in range(batch_size):
            flat_idx = ego_indices[i]
            enhanced_feature_i = enhanced_features[i]
            curr_bev_batch[i][0] = enhanced_feature_i
            # updated_curr_bev[flat_idx] = enhanced_feature_i
        return torch.cat(curr_bev_batch, dim=0)

    def forward(self, bev_list, psm, record_len, pairwise_t_matrix, backbone=None, heads=None, history=None):
        vox_bev, feat_bev, det_bev = bev_list
        _, _, H, W = feat_bev.shape
        B, L = pairwise_t_matrix.shape[:2]
        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [
            0, 1], :][:, :, :, :, [0, 1, 3]]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0,
                                                         2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1,
                                                         2] / (self.downsample_rate * self.discrete_ratio * H) * 2



        c_vox = vox_bev.shape[1]
        c_feat = feat_bev.shape[1]
        if history:
            # feat_final, offset_loss = self.how2comm(fused_bev, short_history, long_history, record_len, backbone, heads)
            comp_F_fused, _, _ = self.mgdc_bev_compensator(history)
            comp_F_vox = comp_F_fused[:,0:c_vox,:,:]
            comp_F_feat = comp_F_fused[:,c_vox:c_vox+c_feat,:,:]
            comp_F_det = comp_F_fused[:,c_vox+c_feat:,:,:]
            offset_loss = self.compensation_criterion(predicted_bevs=[comp_F_vox, comp_F_feat, comp_F_det], ground_truth_bevs=bev_list)
            # 把ego-agent的当前帧补偿回去
            comp_F_vox_list = self.regroup(comp_F_vox, record_len)
            comp_F_feat_list = self.regroup(comp_F_feat, record_len)
            comp_F_det_list = self.regroup(comp_F_det, record_len)
            vox_bev_list = self.regroup(vox_bev, record_len)
            feat_bev_list = self.regroup(feat_bev, record_len)
            det_bev_list = self.regroup(det_bev, record_len)
            for bs in range(B):
                comp_F_vox_list[bs][0] = vox_bev_list[bs][0]
                comp_F_feat_list[bs][0] = feat_bev_list[bs][0]
                comp_F_det_list[bs][0] = det_bev_list[bs][0]
            vox_bev = torch.cat(comp_F_vox_list, dim=0)
            feat_bev = torch.cat(comp_F_feat_list, dim=0)
            det_bev = torch.cat(det_bev_list, dim=0)
        else:
            offset_loss = torch.zeros(1).to(feat_bev.device)

        #把增强后的ego特征放入
        his_feat = history[1]  # list of [B,C,H,W]
        # 对ego的帧进行增强
        feat_bev = self.get_enhanced_feature(feat_bev, his_feat[1:4], record_len)  # 取第0到第3帧作为历史

        fused_feat_list = []
        commu_volume = 0
        commu_loss = torch.tensor(0).to(feat_bev.device)
        #先不考虑multi_scale
        if self.communication:
            batch_confidence_maps = self.regroup(psm, record_len)
            _, _, confidence_maps = self.naive_communication(batch_confidence_maps)

            batch_temp_features = self.regroup(feat_bev, record_len)
            batch_vox_features = self.regroup(vox_bev, record_len)
            batch_det_features = self.regroup(det_bev, record_len)

            temp_list = []
            temp_vox_list = []
            temp_det_list = []
            temp_psm_list = []
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

                # 添加vox和det信息
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

                temp_psm_list.append(warp_affine_simple(confidence_maps[b], t_matrix[0, :, :, :], (H, W)))
            vox_bev = torch.cat(temp_vox_list, dim=0)
            feat_bev = torch.cat(temp_list, dim=0)
            det_bev = torch.cat(temp_det_list, dim=0)
            #稀疏多粒度数据传输
            if self.communication_flag:
                vox_bev, feat_bev, det_bev, commu_loss, commu_volume = self.communication_net(temp_vox_list, temp_list, temp_det_list)

                # sparse_vox = all_agents_sparse_transmitted_data[:, 0:self.c_d, :, :]
                # sparse_feat = all_agents_sparse_transmitted_data[:, self.c_d:self.c_d + self.c_f, :, :]
                # sparse_det = all_agents_sparse_transmitted_data[:, self.cd + self.c_f:, :, :]
                # vox_bev = F.interpolate(sparse_vox, scale_factor=1, mode="bilinear", align_corners=False)
                # vox_bev = self.channel_fuse(vox_bev)
                # feat_bev = F.interpolate(sparse_feat, scale_factor=1, mode="bilinear", align_corners=False)
                # feat_bev = self.channel_fuse(feat_bev)
                # det_bev = F.interpolate(sparse_det, scale_factor=1, mode="bilinear", align_corners=False)
                # det_bev = self.channel_fuse(det_bev)
            else:
                commu_volume = torch.tensor(0).to(feat_bev.device)
                commu_loss = torch.zeros(0).to(feat_bev.device)

            batch_node_feat = self.regroup(feat_bev, record_len)
            batch_node_vox = self.regroup(vox_bev, record_len)
            batch_node_det = self.regroup(det_bev, record_len)
            # batch_node_features_his = self.regroup(his, record_len)


            for b in range(B):
                N = record_len[b]
                # 这里已经是稀疏化的数据
                node_vox = batch_node_vox[b]
                node_feat = batch_node_feat[b]
                node_det = batch_node_det[b]
                fused_feat = self.granularity_fusion(node_vox, node_feat, node_det)
                fused_feat_list.append(fused_feat)
            fused_feat_list = torch.stack(fused_feat_list)

                
        return fused_feat_list, commu_volume, offset_loss, commu_loss