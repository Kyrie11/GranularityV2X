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
        self.communication = True
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
        self.communication_net = AdvancedCommunication(c_vox=10, c_feat=64, c_det=16)

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

        updated_curr_bev = curr_bev.clone()
        ego_indices = [0] + torch.cumsum(torch.tensor(record_len[:-1]), dim=0).tolist()
        for i in range(batch_size):
            curr_bev_batch[i][0] = enhanced_features[i]
        return torch.cat(curr_bev_batch, dim=0)



    def forward(self, record_len, pairwise_t_matrix, his_vox=None, his_feat=None, his_det=None):
        curr_vox_bev = his_vox[0]
        curr_feat_bev = his_feat[0]
        curr_det_bev = his_det[0]
        print("第一次检查feat_bev.shape=", curr_feat_bev.shape)
        _, _, H, W = curr_feat_bev.shape
        B, L = pairwise_t_matrix.shape[:2]
        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [
            0, 1], :][:, :, :, :, [0, 1, 3]]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0,
                                                         2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1,
                                                         2] / (self.downsample_rate * self.discrete_ratio * H) * 2


        c_vox = curr_vox_bev.shape[1]
        c_feat = curr_feat_bev.shape[1]
        if len(his_vox) > 1:
            # feat_final, offset_loss = self.how2comm(fused_bev, short_history, long_history, record_len, backbone, heads)
            comp_F_fused, _, _ = self.mgdc_bev_compensator(his_vox, his_feat, his_det, record_len)
            comp_F_vox = comp_F_fused[:,0:c_vox,:,:]
            comp_F_feat = comp_F_fused[:,c_vox:c_vox+c_feat,:,:]
            comp_F_det = comp_F_fused[:,c_vox+c_feat:,:,:]

            #把ego的帧返回回去
            #补偿后的结果
            comp_F_vox_batchs = self.regroup(comp_F_vox, record_len)
            comp_F_feat_batchs = self.regroup(comp_F_feat, record_len)
            comp_F_det_batchs = self.regroup(comp_F_det, record_len)
            #原始的帧
            curr_vox_batchs = self.regroup(curr_vox_bev, record_len)
            curr_feat_batchs = self.regroup(curr_feat_bev, record_len)
            curr_det_batchs = self.regroup(curr_det_bev, record_len)

            for bs in range(len(comp_F_vox_batchs)):
                comp_F_vox_batchs[bs][0] = curr_vox_batchs[bs][0]
                comp_F_feat_batchs[bs][0] = curr_feat_batchs[bs][0]
                comp_F_det_batchs[bs][0] = curr_det_batchs[bs][0]

            #补偿原始Ego的帧
            comp_F_vox = torch.cat(comp_F_vox_batchs, dim=0)
            comp_F_feat = torch.cat(comp_F_feat_batchs, dim=0)
            comp_F_det = torch.cat(comp_F_det_batchs, dim=0)

            offset_loss = self.compensation_criterion(comp_F_vox, comp_F_feat, comp_F_det, curr_vox_bev, curr_feat_bev, curr_det_bev)
            print("offset_loss=", offset_loss)
            # 把ego-agent的当前帧补偿回去

            vox_bev_copy = comp_F_vox
            feat_bev_copy = comp_F_feat
            det_bev_copy = comp_F_det
        else:
            vox_bev_copy = curr_vox_bev
            feat_bev_copy = curr_feat_bev
            det_bev_copy = curr_det_bev
            offset_loss = torch.zeros(1).to(curr_feat_bev.device)
        print("第二次检查feat_bev.shape=", feat_bev_copy.shape)
        #把增强后的ego特征放入
        # 对ego的帧进行增强
        feat_bev_copy = self.get_enhanced_feature(feat_bev_copy, his_feat[1:4], record_len)  # 取第0到第3帧作为历史
        print("第三次检查feat_bev.shape=", feat_bev_copy.shape)

        fused_feat_list = []
        fused_feat = torch.tensor(0).to(curr_feat_bev.device)
        commu_volume = 0
        commu_loss = torch.tensor(0).to(curr_feat_bev.device)
        #先不考虑multi_scale
        if self.communication:
            # batch_confidence_maps = self.regroup(psm, record_len)
            # _, _, confidence_maps = self.naive_communication(batch_confidence_maps)

            batch_temp_features = self.regroup(feat_bev_copy, record_len)
            batch_vox_features = self.regroup(vox_bev_copy, record_len)
            batch_det_features = self.regroup(det_bev_copy, record_len)

            temp_list = []
            temp_vox_list = []
            temp_det_list = []
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

                # temp_psm_list.append(warp_affine_simple(confidence_maps[b], t_matrix[0, :, :, :], (H, W)))
            vox_bev_copy = torch.cat(temp_vox_list, dim=0)
            feat_bev_copy = torch.cat(temp_list, dim=0)
            det_bev_copy = torch.cat(temp_det_list, dim=0)
            print("第四次检查feat_bev.shape=", feat_bev_copy.shape)
            #稀疏多粒度数据传输
            if self.communication_flag:
                vox_bev_copy, feat_bev_copy, det_bev_copy, commu_loss, commu_volume = self.communication_net(temp_vox_list, temp_list, temp_det_list)
                print("commu_loss=", commu_loss)
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
                commu_volume = torch.tensor(0).to(curr_feat_bev.device)
                commu_loss = torch.zeros(1).to(curr_feat_bev.device)
            print("第五次检查feat_bev.shape=", feat_bev_copy.shape)

            batch_node_feat = self.regroup(feat_bev_copy, record_len)
            batch_node_vox = self.regroup(vox_bev_copy, record_len)
            batch_node_det = self.regroup(det_bev_copy, record_len)
            # batch_node_features_his = self.regroup(his, record_len)


            for b in range(B):
                N = record_len[b]
                # 这里已经是稀疏化的数据
                node_vox = batch_node_vox[b]
                node_feat = batch_node_feat[b]
                node_det = batch_node_det[b]
                fused_feat = self.granularity_fusion(node_vox, node_feat, node_det)
                print("在batch循环里的fused_feat.shape=", fused_feat.shape)
                fused_feat_list.append(fused_feat)
            fused_feat = torch.cat(fused_feat_list, dim=0)

                
        return fused_feat, commu_volume, offset_loss, commu_loss