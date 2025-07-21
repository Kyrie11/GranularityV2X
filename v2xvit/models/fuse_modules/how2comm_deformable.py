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
from v2xvit.models.fuse_modules.gem_fusion import GEM_Fusion, ConvGRUCell
from v2xvit.models.sub_modules.delay_compensation import LatencyCompensator
from v2xvit.loss.hierarchical_delay_loss import HierarchicalDelayLoss


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

        self.c_temporal = 128
        self.gem_fusion = GEM_Fusion(c_g1=8, c_g2=64, c_g3=8, c_temporal=self.c_temporal, c_fusion=256)

        self.main_temporal_gru = ConvGRUCell(input_dim=256, hidden_dim=self.c_temporal, kernel_size=3)

        self.hidden_state = None

        self.latency_compensator = LatencyCompensator(c_g1=8, c_g2=64, c_g3=8, c_motion=32, c_fuse=128)

        self.hierarchical_delay_loss = HierarchicalDelayLoss()

    def reset_hidden_state(self):
        self.hidden_state = None

    def detach_hidden_state(self):
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()

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

    '''
        g1_data: vox-level data
        g2_data: feature-level data
        g3_data: detection-level data
    '''
    def forward(self, g1_data, g2_data, g3_data, record_len, pairwise_t_matrix, backbone=None, delay=0, short_his=None, long_his=None):
        curr_g1_data = g1_data
        curr_g2_data = g2_data
        curr_g3_data = g3_data
        device = g2_data.device
        print("第一次检查feat_bev.shape=", curr_g2_data.shape)
        _, _, H, W = curr_g2_data.shape

        B, L = pairwise_t_matrix.shape[:2]
        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [
                                                           0, 1], :][:, :, :, :, [0, 1, 3]]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0,
        2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1,
        2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]


        if short_his and long_his:
            # feat_final, offset_loss = self.how2comm(fused_bev, short_history, long_history, record_len, backbone, heads)
            comp_g1, comp_g2, comp_g3, compensation_params = self.latency_compensator(short_his, long_his, delay, record_len)
            delay_g1 = short_his[0][0]
            delay_g2 = short_his[1][0]
            delay_g3 = short_his[2][0]
            delay_loss = self.hierarchical_delay_loss(compensation_params, g1_data, g2_data, g3_data, delay_g1, delay_g2, delay_g3, record_len)

            g1_data = comp_g1.clone().detach()
            g2_data = comp_g2.clone().detach()
            g3_data = comp_g3.clone().detach()
        else:
            g1_data = curr_g1_data
            g2_data = curr_g2_data
            g3_data = curr_g3_data
            delay_loss = 0
        print("第二次检查feat_bev.shape=", g2_data.shape)
        #把增强后的ego特征放入
        # 对ego的帧进行增强
        # feat_bev_copy = self.get_enhanced_feature(feat_bev_copy, his_feat[1:-1], record_len)  # 取第0到第3帧作为历史
        # print("第三次检查feat_bev.shape=", feat_bev_copy.shape)

        fused_feat_list = []
        fused_feat = torch.tensor(0).to(device)
        commu_volume = 0
        commu_loss = torch.tensor(0).to(device)
        #先不考虑multi_scale
        if self.communication:
            batch_temp_g1_data = self.regroup(g1_data, record_len)
            batch_temp_g2_data = self.regroup(g2_data, record_len)
            batch_temp_g3_data = self.regroup(g3_data, record_len)

            temp_g1_list, temp_g2_list, temp_g3_list = [], [], []
            for b in range(B):
                N = record_len[b]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

                temp_g1_data = batch_temp_g1_data[b]
                print("temp_g1_data.shape=", temp_g1_data.shape)
                print("t_matrix.shape=", t_matrix.shape)
                C,H,W = temp_g1_data.shape[1:]
                neighbor_g1_data = warp_affine_simple(temp_g1_data, t_matrix[0,:,:,:], (H,W))
                temp_g1_list.append(neighbor_g1_data)

                temp_g2_data = batch_temp_g2_data[b]
                neighbor_g2_data = warp_affine_simple(temp_g2_data, t_matrix[0,:,:,:], (H,W))
                temp_g2_list.append(neighbor_g2_data)

                temp_g3_data = batch_temp_g3_data[b]
                neighbor_g3_data = warp_affine_simple(temp_g3_data, t_matrix[0,:,:,:], (H,W))
                temp_g3_list.append(neighbor_g3_data)
            # g1_data = torch.cat(temp_g1_list, dim=0)
            # g2_data = torch.cat(temp_g2_list, dim=0)
            # g3_data = torch.cat(temp_g3_list, dim=0)

            print("第四次检查feat_bev.shape=", g1_data.shape)
            #稀疏多粒度数据传输
            if self.communication_flag:
                g1_data, g2_data, g3_data, commu_loss, commu_volume = self.communication_net(temp_g1_list, temp_g2_list, temp_g3_list)
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
                commu_volume = torch.tensor(0).to(device)
                commu_loss = torch.zeros(0).to(device)
            print("第五次检查feat_bev.shape=", g1_data.shape)

            batch_node_g1 = self.regroup(g1_data, record_len)
            batch_node_g2 = self.regroup(g2_data, record_len)
            batch_node_g3 = self.regroup(g3_data, record_len)

            #初始化一个全零的隐藏状态 (记忆)
            if self.hidden_state is None or self.hidden_state.shape[0] != B:
                self.hidden_state = torch.zeros(B, self.c_temporal, H, W, device=device)

            for b in range(B):
                node_g1 = batch_node_g1[b]
                node_g2 = batch_node_g2[b]
                node_g3 = batch_node_g3[b]
                node_hidden_state = self.hidden_state[b:b+1]
                fused_feat = self.gem_fusion(node_g1, node_g2, node_g3, node_hidden_state)
                node_hidden_state = self.main_temporal_gru(fused_feat, node_hidden_state)
                fused_feat_list.append(fused_feat)
                # self.hidden_state[b:b+1] = node_hidden_state
            fused_feat = torch.cat(fused_feat_list, dim=0)

        return fused_feat, commu_volume, delay_loss, commu_loss