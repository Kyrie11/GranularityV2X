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
from v2xvit.models.sub_modules.hpc import ContextExtrapolator, AgentEncoder
from v2xvit.models.sub_modules.temporal_enhancement import HistoryContextAdaptiveEnhancement
from v2xvit.models.sub_modules.gain_gated_modulation import GainGatedModulation
from v2xvit.loss.distillation_loss import DistillationLoss
from v2xvit.models.fuse_modules.demand_aware_fusion import DemandDrivenFusionNetwork


class GranularityEncoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv_block(x)

class UnifiedBevEncoder(nn.Module):
    def __init__(self,g1_channels, g2_channels, g3_channels, g1_out, g2_out, g3_out, unified_channel):
        super().__init__()

        self.g1_encoder = GranularityEncoder(input_channels=g1_channels, output_channels=g1_out)
        self.g2_encoder = GranularityEncoder(input_channels=g2_channels, output_channels=g2_out)
        self.g3_encoder = GranularityEncoder(input_channels=g3_channels, output_channels=g3_out)

        total_output = g1_out + g2_out + g3_out
        self.fusion_conv = nn.Conv2d(total_output, unified_channel, kernel_size=1)  # 把三个粒度数据合并后再1x1卷积

    def forward(self, g1_data, g2_data, g3_data):
        ''''
        处理一帧的三粒度数据
        '''
        feature_g1 = self.g1_encoder(g1_data)
        feature_g2 = self.g2_encoder(g2_data)
        feature_g3 = self.g3_encoder(g3_data)
        concatenated_features = torch.cat([feature_g1, feature_g2, feature_g3], dim=1)
        unified_feature = self.fusion_conv(concatenated_features)
        return unified_feature


class How2comm(nn.Module):
    def __init__(self, args, args_pre):
        super(How2comm, self).__init__()
        self.communication = True
        self.communication_flag = args['communication_flag']
        self.downsample_rate = args['downsample_rate']
        self.async_flag = False
        self.discrete_ratio = args['voxel_size'][0]
        # self.long_intervals = args['train_params']['lsh']['p']
        self.long_intervals = 3


        s_ctx_channels = 32
        l_ctx_dim = 256
        feature_size = (100, 352)

        #============三个粒度数据的编码器================
        g_out = 128
        g1_in, g1_out = 8, g_out
        g2_in, g2_out = 256, g_out
        g3_in, g3_out = 8, g_out
        unified_channel = 256
        total_input = g1_out + g2_out + g3_out
        self.g1_encoder = GranularityEncoder(input_channels=g1_in, output_channels=g1_out)
        self.g2_encoder = GranularityEncoder(input_channels=g2_in, output_channels=g2_out)
        self.g3_encoder = GranularityEncoder(input_channels=g3_in, output_channels=g3_out)
        self.fusion_conv = nn.Conv2d(total_input, unified_channel, kernel_size=1)
        #============三个粒度数据的编码器================

        # 通信模块
        self.communication_net = AdvancedCommunication(c_g1=g1_in, c_g2=g2_in, c_g3=g3_in)

        #============时延预测模块============
        self.context_extrapolator = ContextExtrapolator(s_ctx_channels=s_ctx_channels, l_ctx_dim=l_ctx_dim, fusion_dim=128,
                                                        bev_feature_channels=g2_in, physical_info_channels=g1_in,
                                                        result_map_channels=g3_in, feature_size=feature_size)
        #============时延预测模块============
        self.temporal_context_encoder = AgentEncoder(total_input_channels=unified_channel,
                                                               s_ctx_channels=s_ctx_channels,
                                                               l_ctx_dim=l_ctx_dim,
                                                               feature_size=feature_size)

        self.distillation_loss = DistillationLoss(unified_bev_channels=unified_channel, g1_encoder=self.g1_encoder,
                                                  g2_encoder=self.g2_encoder, g3_encoder=self.g3_encoder, fusion_conv=self.fusion_conv)

        self.hdae_module = HistoryContextAdaptiveEnhancement(current_channels=unified_channel,
                                                             short_ctx_channels=s_ctx_channels,
                                                             long_ctx_dim=l_ctx_dim)

        self.gain_gated_module = GainGatedModulation(channels=unified_channel)

        self.fused_net = DemandDrivenFusionNetwork(model_dim=unified_channel, g_out=g_out, num_heads=4, num_sampling_points=8)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def get_unified_bev(self, g1_data, g2_data, g3_data):
        feature_g1 = self.g1_encoder(g1_data)
        feature_g2 = self.g2_encoder(g2_data)
        feature_g3 = self.g3_encoder(g3_data)
        concatenated_features = torch.cat([feature_g1, feature_g2, feature_g3], dim=1)
        unified_feature = self.fusion_conv(concatenated_features)
        return unified_feature

    def delay_compensation(self, g1_data, g2_data, g3_data, GT_data, short_his, long_his, delay):
        short_his_g1, short_his_g2, short_his_g3 = short_his
        long_his_g1, long_his_g2, long_his_g3 = long_his
        GT_g1_data, GT_g2_data, GT_g3_data = GT_data #用来作为监督

        # =======短期历史数据编码=======
        short_his_g1_stacked = torch.stack(short_his_g1, dim=1)  # [N,T,C,H,W]
        short_his_g2_stacked = torch.stack(short_his_g2, dim=1)
        short_his_g3_stacked = torch.stack(short_his_g3, dim=1)

        N, T_short, _, H, W = short_his_g1_stacked.shape

        short_his_g1_stacked = short_his_g1_stacked.view(N * T_short, -1, H, W)  # [N * T, C, H, W]
        short_his_g2_stacked = short_his_g2_stacked.view(N * T_short, -1, H, W)  # [N * T, C, H, W]
        short_his_g3_stacked = short_his_g3_stacked.view(N * T_short, -1, H, W)  # [N * T, C, H, W]

        short_his_unified_bev = self.get_unified_bev(short_his_g1_stacked, short_his_g2_stacked,
                                                         short_his_g3_stacked)  # Output: [N * T, C_unified, H, W]
        short_his_unified_bev = short_his_unified_bev.view(N, T_short, -1, H,
                                                           W)  ## [N * T, C_unified, H, W] -> [N, T, C_unified, H, W]

        # =======长期历史数据编码=======
        long_his_g1_stacked = torch.stack(long_his_g1, dim=1)  # [N,T,C,H,W]
        long_his_g2_stacked = torch.stack(long_his_g2, dim=1)
        long_his_g3_stacked = torch.stack(long_his_g3, dim=1)

        T_long = long_his_g1_stacked.shape[1]
        assert T_long == len(long_his_g1), "the index is wrong!!"

        long_his_g1_stacked = long_his_g1_stacked.view(N * T_long, -1, H, W)  # [N * T, C, H, W]
        long_his_g2_stacked = long_his_g2_stacked.view(N * T_long, -1, H, W)  # [N * T, C, H, W]
        long_his_g3_stacked = long_his_g3_stacked.view(N * T_long, -1, H, W)  # [N * T, C, H, W]

        long_his_unified_bev = self.get_unified_bev(long_his_g1_stacked, long_his_g2_stacked,
                                                        long_his_g3_stacked)  # Output: [N * T, C_unified, H, W]
        long_his_unified_bev = long_his_unified_bev.view(N, T_long, -1, H,
                                                         W)  ## [N * T, C_unified, H, W] -> [N, T, C_unified, H, W]

        long_time_gaps = [i * -100 * self.long_intervals for i in range(T_long)]  # 时间回溯gap,譬如[0,-300,-600,-900]
        print(f"长期历史延时时间为：{long_time_gaps}")
        # 编码历史上下文信息
        short_term_context, long_term_context = self.temporal_context_encoder(short_his_unified_bev, long_his_unified_bev, long_time_gaps)
        print("短期上下文的shape是：", short_term_context.shape)
        print("长期上下文的shape是：", long_term_context.shape)

        delayed_g1_frame = g1_data  # 要延迟补偿的帧
        delayed_g2_frame = g2_data  # 要延迟补偿的帧
        delayed_g3_frame = g3_data  # 要延迟补偿的帧
        # =================得到预测结果，包含预测的g1、g2、g3===============
        delay = delay * 100
        print("延迟时间是:", delay)
        predictions = self.context_extrapolator(
            s_ctx=short_term_context, #短期上下文
            l_ctx=long_term_context,  #长期上下文
            delayed_g1_frame=delayed_g1_frame,
            delayed_g2_frame=delayed_g2_frame,
            delayed_g3_frame=delayed_g3_frame,
            delays_ms=delay
        )

        predicted_g1 = predictions['predicted_g1']
        predicted_g2 = predictions['predicted_g2']
        predicted_g3 = predictions['predicted_g3']
        print(f"predicted_g1.shape={predicted_g1.shape}")
        # 沿着通道维度(dim=1)计算余弦相似度。
        # 输出的 cos_sim 的形状为 [N, H, W]
        # 将余弦相似度转换为损失。
        # 损失值范围为 [0, 2]，目标是最小化到 0。
        # 我们计算批次中所有像素位置损失的平均值。
        cos_sim1 = F.cosine_similarity(predicted_g1, GT_g1_data, dim=1)
        cos_sim1 = (1 - cos_sim1).mean()
        cos_sim2 = F.cosine_similarity(predicted_g2, GT_g2_data, dim=1)
        cos_sim2 = (1 - cos_sim2).mean()
        cos_sim3 = F.cosine_similarity(predicted_g3, GT_g3_data, dim=1)
        cos_sim3 = (1 - cos_sim3).mean()
        delay_loss = cos_sim1 + cos_sim2 + cos_sim3
        return predicted_g1, predicted_g2, predicted_g3, delay_loss, short_term_context, long_term_context

    def get_stacked_his(self, history):
        his_stacked = torch.stack(history, dim=1)
        N, T, _, H, W = his_stacked.shape
        assert len(history) == T, "the length is wrong!!!"
        his_stacked = his_stacked.view(N * T, -1, H, W)  # [N * T, C, H, W]
        return his_stacked


    '''
        g1_data: vox-level data
        g2_data: feature-level data
        g3_data: detection-level data
    '''
    def forward(self, current_g_data, record_len, pairwise_t_matrix, backbone=None, delay=0, short_his=None, long_his=None, GT_data=None):
        current_g1_data, current_g2_data, current_g3_data = current_g_data
        device = current_g2_data.device
        _, _, H, W = current_g2_data.shape
        B, L = pairwise_t_matrix.shape[:2]

        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [
                                                           0, 1], :][:, :, :, :, [0, 1, 3]]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0,
        2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1,
        2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        delay_loss = torch.tensor(0.0, device=device)
        if len(short_his) > 1 and len(long_his) > 1:
            predicted_g1, predicted_g2, predicted_g3,delay_loss, short_term_ctx, long_term_ctx = self.delay_compensation(current_g1_data,
                                                                                                                         current_g2_data,
                                                                                                                         current_g3_data,
                                                                                                                         GT_data,
                                                                                                                         short_his, long_his, delay)
            print(f"predicted_g1.shape={predicted_g1.shape}")
            # =====把预测的数据作为当前时刻的数据，但是要注意ego-agent的数据
            g1_data = predicted_g1.clone().detach()
            g2_data = predicted_g2.clone().detach()
            g3_data = predicted_g3.clone().detach()

            g1_data[0:1] = current_g1_data[0:1]
            g2_data[0:1] = current_g2_data[0:1]
            g3_data[0:1] = current_g3_data[0:1]

            ego_unified_bev = self.get_unified_bev(g1_data[0:1], g2_data[0:1], g3_data[0:1])

            ego_stx = short_term_ctx[0:1]
            ego_ltx = long_term_ctx[0:1]
            print("长期上下文ego_ltx的shape:", ego_ltx.shape)
            print("短期上下文ego_stx的shape:", ego_stx.shape)
            #=======对当前帧进行时序增强(只对ego增强了)=========
            ego_enhanced = self.hdae_module(ego_unified_bev, ego_stx, ego_ltx)

        else:
            g1_data = current_g1_data
            g2_data = current_g2_data
            g3_data = current_g3_data
        # #======对数据进行空间增强==============
        # current_unified_bev = self.gain_gated_module(current_unified_bev)
        # #==================================
        
        #坐标对齐1
        t_matrix = pairwise_t_matrix[0][:record_len, :record_len, :, :]
        H, W = g1_data.shape[2:]
        g1_data = warp_affine_simple(g1_data, t_matrix[0, :, :, :], (H,W))
        g2_data = warp_affine_simple(g2_data, t_matrix[0, :, :, :], (H,W))
        g3_data = warp_affine_simple(g3_data, t_matrix[0, :, :, :], (H,W))

        current_unified_bev = self.get_unified_bev(g1_data, g2_data, g3_data)
        if len(short_his) > 1 and len(long_his) > 1:
            current_unified_bev[0:1] = ego_enhanced

        if self.communication:
            #sparse_mask[N-1,3,H,W}
            ego_demand, sparse_g1, sparse_g2, sparse_g3, commu_volume, sparse_mask = self.communication_net(g1_data, g2_data,
                                                                                               g3_data, current_unified_bev)
            commu_loss = self.distillation_loss(ego_demand, current_unified_bev, sparse_g1, sparse_g2, sparse_g3)

            print("sparse_g1.shape=", sparse_g1.shape)
            print("ego_demand.shape=", ego_demand.shape)

            collab_sparse_data = [sparse_g1[1:].clone().detach(), sparse_g2[1:].clone().detach(), sparse_g3[1:].clone().detach()]

            ego_unified_bev = current_unified_bev[0:1].clone().detach()

            fused_feat = self.fused_net(ego_unified_bev, ego_demand.clone().detach(),
                                      collab_sparse_data, sparse_mask,
                                      [self.g1_encoder, self.g2_encoder, self.g3_encoder])

        return fused_feat, commu_volume, delay_loss, commu_loss