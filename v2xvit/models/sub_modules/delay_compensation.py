import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

def warp(feature_map, offset_field):
    """
    使用偏移场 (offset field) 对特征图进行扭曲 (warp)。

    Args:
        feature_map (torch.Tensor): 待扭曲的特征图, 形状为 [B, C, H, W]。
        offset_field (torch.Tensor): 像素级位移场, 形状为 [B, 2, H, W]。
                                      通道0是x方向位移, 通道1是y方向位移。

    Returns:
        torch.Tensor: 扭曲后的特征图, 形状与输入特征图相同。
    """
    B, _, H, W = feature_map.shape
    device = feature_map.device

    # 创建一个标准化的基础网格 (从-1到1)
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    # 归一化到 [-1, 1]
    grid_x = 2 * grid_x / (W - 1) - 1
    grid_y = 2 * grid_y / (H - 1) - 1
    # 组合成 [H, W, 2] 的网格
    base_grid = torch.stack((grid_x, grid_y), 2)
    # 扩展到批次维度 [B, H, W, 2]
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)

    # 将偏移场从 [B, 2, H, W] 转换为 [B, H, W, 2] 并归一化
    # grid_sample 要求偏移是相对整个特征图大小的比例
    offset_x, offset_y = offset_field[:, 0, ...], offset_field[:, 1, ...]
    norm_offset_x = 2 * offset_x / (W - 1)
    norm_offset_y = 2 * offset_y / (H - 1)
    # 组合成 [B, H, W, 2] 的归一化偏移
    normalized_offset = torch.stack((norm_offset_x, norm_offset_y), 3)

    # 新的采样网格 = 基础网格 + 归一化偏移
    sampling_grid = base_grid + normalized_offset

    # 使用 F.grid_sample 进行扭曲操作
    warped_feature = F.grid_sample(
        feature_map,
        sampling_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return warped_feature

class DualStreamMotionEncoder(nn.Module):
    """
    双流运动编码器，用于融合长短时历史信息。
    """
    def __init__(self, c_in, c_out):
        super(DualStreamMotionEncoder, self).__init__()
        # 短时流: 使用3D卷积捕捉精细运动
        self.short_term_stream = nn.Sequential(
            nn.Conv3d(c_in, c_out // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(c_out // 2),
            nn.ReLU(inplace=True)
        )

        # 长时流: 同样使用3D卷积捕捉宏观趋势
        self.long_term_stream = nn.Sequential(
            nn.Conv3d(c_in, c_out // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(c_out // 2),
            nn.ReLU(inplace=True)
        )

        # 融合模块，将两个流的输出特征整合成最终的运动特征
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, short_his, long_his):
        # 将列表堆叠成 [B, T, C, H, W] -> [B, C, T, H, W]
        short_tensor = torch.stack(short_his, dim=2)
        long_tensor = torch.stack(long_his, dim=2)

        # 通过各自的流进行处理
        short_features = self.short_term_stream(short_tensor)
        long_features = self.long_term_stream(long_tensor)

        # 3D卷积后，时间维度D会保持为3，我们需要将其压缩
        # 这里我们使用最大池化来聚合时间维度的信息
        short_features_2d = torch.max(short_features, dim=2)[0]
        long_features_2d = torch.max(long_features, dim=2)[0]

        #沿通道维度拼接
        combined_features = torch.cat([short_features_2d, long_features_2d], dim=1)

        #最终融合
        motion_feature = self.fusion_conv(combined_features)
        return motion_feature

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *  (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 是一个 [B, 1] 的张量，代表批次中每个样本的延迟(需为整数)
        return self.pe[x.long(), :].squeeze(1)


class HierarchicalPredictionHead(nn.Module):
    """
    分层预测头，生成基础位移、残差位移和共享置信度。
    """
    def __init__(self, c_in):
        super(HierarchicalPredictionHead, self).__init__()
        # 基础位移场 (G1) 的预测头
        self.base_offset_head = nn.Conv2d(c_in, 2, kernel_size=3, padding=1)

        # 残差位移场 (G2) 的预测头
        self.residual_offset_g2_head = nn.Conv2d(c_in, 2, kernel_size=3, padding=1)

        # 残差位移场 (G3) 的预测头
        self.residual_offset_g3_head = nn.Conv2d(c_in, 2, kernel_size=3, padding=1)

        # 共享置信度图 (Scale) 的预测头
        self.scale_head = nn.Conv2d(c_in, 1, kernel_size=3, padding=1)

    def forward(self, fused_motion_feature):
        offset_g1 = self.base_offset_head(fused_motion_feature)
        delta_offset_g2 = self.residual_offset_g2_head(fused_motion_feature)
        delta_offset_g3 = self.residual_offset_g3_head(fused_motion_feature)

        # 置信度通过 Sigmoid 激活到 [0, 1] 范围
        scale_shared = torch.sigmoid(self.scale_head(fused_motion_feature))

        return {
            'offset_g1': offset_g1,
            'delta_offset_g2': delta_offset_g2,
            'delta_offset_g3': delta_offset_g3,
            'scale_shared': scale_shared
        }

class LatencyCompensator(nn.Module):
    """
    完整的延迟补偿器。
    
    对于一个协作者，它接收其长短时历史数据和Ego的当前数据，
    然后预测出分层的补偿参数。
    """
    def __init__(self, c_g1, c_g2, c_g3, c_motion, c_fuse, c_delay_embed=32):
        super(LatencyCompensator, self).__init__()

        #延迟值编码器
        self.delay_encoder = PositionalEncoding(d_model=c_delay_embed)

        # 为每个粒度实例化一个双流运动编码器
        self.g1_encoder = DualStreamMotionEncoder(c_in=c_g1, c_out=c_motion)
        self.g2_encoder = DualStreamMotionEncoder(c_in=c_g2, c_out=c_motion)
        self.g3_encoder = DualStreamMotionEncoder(c_in=c_g3, c_out=c_motion)

        # 定义一个瓶颈层，用于融合所有提取出的运动特征和Ego的当前特征
        # 输入维度 = 3*运动特征 + 3*Ego特征
        fusion_input_dim = c_motion * 3 + (c_g1 + c_g2 + c_g3) * 2 + c_delay_embed
        self.fusion_bottleneck = nn.Sequential(
            nn.Conv2d(fusion_input_dim, c_fuse, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_fuse),
            nn.ReLU(inplace=True)
        )

        # 实例化分层预测头
        self.prediction_head = HierarchicalPredictionHead(c_in=c_fuse)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def get_his_collab(self, his, record_len):
        batch_size = len(record_len)

        his_g1, his_g2, his_g3 = his
        his_g1_collab, his_g2_collab, his_g3_collab = [], [], []
        frames = len(his_g1)
        for i in range(frames):
            his_g1_iterate = self.regroup(his_g1[i], record_len)
            his_g2_iterate = self.regroup(his_g2[i], record_len)
            his_g3_iterate = self.regroup(his_g3[i], record_len)

            his_g1_temporary, his_g2_temporary, his_g3_temporary = [], [], []

            for j in range(batch_size):
                his_g1_iterate_collab = his_g1_iterate[j][1:, :, :, :]
                his_g2_iterate_collab = his_g2_iterate[j][1:, :, :, :]
                his_g3_iterate_collab = his_g3_iterate[j][1:, :, :, :]

                his_g1_temporary.append(his_g1_iterate_collab)
                his_g2_temporary.append(his_g2_iterate_collab)
                his_g3_temporary.append(his_g3_iterate_collab)

            # his_g1_temporary = torch.cat(his_g1_temporary, dim=0)
            # his_g2_temporary = torch.cat(his_g2_temporary, dim=0)
            # his_g3_temporary = torch.cat(his_g3_temporary, dim=0)

            his_g1_collab.append(his_g1_temporary)
            his_g2_collab.append(his_g2_temporary)
            his_g3_collab.append(his_g3_temporary)

        return his_g1_collab, his_g2_collab, his_g3_collab

    def get_ego_data(self, his, record_len):
        his_g1, his_g2, his_g3 = his
        curr_g1 = his_g1[0]
        curr_g2 = his_g2[0]
        curr_g3 = his_g3[0]

        curr_g1 = self.regroup(curr_g1, record_len)
        curr_g2 = self.regroup(curr_g2, record_len)
        curr_g3 = self.regroup(curr_g3, record_len)

        batch_size = len(record_len)

        ego_g1, ego_g2, ego_g3 = [], [], []
        for b in range(batch_size):
            g1 = curr_g1[b][0:1, :, :, :]
            g2 = curr_g2[b][0:1, :, :, :]
            g3 = curr_g3[b][0:1, :, :, :]

            ego_g1.append(g1)
            ego_g2.append(g2)
            ego_g3.append(g3)
        # ego_g1 = torch.cat(ego_g1, dim=0)
        # ego_g2 = torch.cat(ego_g2, dim=0)
        # ego_g3 = torch.cat(ego_g3, dim=0)

        return ego_g1, ego_g2, ego_g3


    def forward(self, short_his, long_his, delay, record_len):
        short_his_g1, short_his_g2, short_his_g3 = short_his
        long_his_g1, long_his_g2, long_his_g3 = long_his
        device = short_his_g1[0].device
        # short_his_g1_collab, short_his_g2_collab, short_his_g3_collab = self.get_his_collab(short_his, record_len)
        # long_his_g1_collab, long_his_g2_collab, long_his_g3_collab = self.get_his_collab(long_his, record_len)
        ego_g1, ego_g2, ego_g3 = self.get_ego_data(short_his, record_len)

        B,_,H,W = short_his_g2[0].shape
        #对延迟值进行编码
        delay_embedding = self.delay_encoder(delay.unsqueeze(1)) #[B, c_delay_embed]

        # 为每个粒度提取运动特征
        motion_feat_g1 = self.g1_encoder(short_his_g1, long_his_g1)
        motion_feat_g2 = self.g2_encoder(short_his_g2, long_his_g2)
        motion_feat_g3 = self.g3_encoder(short_his_g3, long_his_g3)

        motion_feat_g1 = self.regroup(motion_feat_g1, record_len)
        motion_feat_g2 = self.regroup(motion_feat_g2, record_len)
        motion_feat_g3 = self.regroup(motion_feat_g3, record_len)

        compensated_g1, compensated_g2, compensated_g3 = [], [], []
        compensation_params = []
        for b in range(B):
            collab_num = record_len[b] - 1
            if collab_num <= 0:
                continue
            delay_channel = delay_embedding.unsqueeze(-1).unsqueeze(-1).expand(collab_num, -1, H, W)
            batch_motion_feat_g1 = motion_feat_g1[b][1:,:,:,:]
            batch_motion_feat_g2 = motion_feat_g2[b][1:,:,:,:]
            batch_motion_feat_g3 = motion_feat_g3[b][1:,:,:,:]

            batch_ego_g1 = ego_g1[b]
            batch_ego_g2 = ego_g2[b]
            batch_ego_g3 = ego_g3[b]

            batch_ego_g1 = batch_ego_g1.repeat_interleave(torch.tensor(collab_num, device=device), dim=0)
            batch_ego_g2 = batch_ego_g2.repeat_interleave(torch.tensor(collab_num, device=device), dim=0)
            batch_ego_g3 = batch_ego_g3.repeat_interleave(torch.tensor(collab_num, device=device), dim=0)
            combined_info = torch.cat([batch_ego_g1, batch_ego_g2, batch_ego_g3,
                                       batch_motion_feat_g1, batch_motion_feat_g2, batch_motion_feat_g3,
                                       short_his_g1[0][1:,:,:,:], short_his_g2[0][1:,:,:,:], short_his_g3[0][1:,:,:,:],
                                       delay_channel], dim=1)


            batch_fused_motion_feature = self.fusion_bottleneck(combined_info)
            batch_compensation_params = self.prediction_head(batch_fused_motion_feature)
            compensation_params.append(batch_compensation_params)

            feature_g1_to_compensate = short_his_g1[0][1:,:,:,:]
            feature_g2_to_compensate = short_his_g2[0][1:,:,:,:]
            feature_g3_to_compensate = short_his_g3[0][1:,:,:,:]

            offset_g2_reconstructed = batch_compensation_params['offset_g1'] + batch_compensation_params['delta_offset_g2']
            offset_g3_reconstructed = batch_compensation_params['offset_g1'] + batch_compensation_params['delta_offset_g3']

            batch_compensated_g1 = warp(feature_g1_to_compensate, batch_compensation_params['offset_g1'] * batch_compensation_params['scale_shared'])
            batch_compensated_g2 = warp(feature_g2_to_compensate, offset_g2_reconstructed) * batch_compensation_params['scale_shared']
            batch_compensated_g3 = warp(feature_g3_to_compensate, offset_g3_reconstructed) * batch_compensation_params['scale_shared']

            batch_compensated_g1 = torch.cat([batch_ego_g1, batch_compensated_g1], dim=0)
            batch_compensated_g2 = torch.cat([batch_ego_g2, batch_compensated_g2], dim=0)
            batch_compensated_g3 = torch.cat([batch_ego_g3, batch_compensated_g3], dim=0)

            compensated_g1.append(batch_compensated_g1)
            compensated_g2.append(batch_compensated_g2)
            compensated_g3.append(batch_compensated_g3)

        compensated_g1 = torch.cat(compensated_g1, dim=0)
        compensated_g2 = torch.cat(compensated_g2, dim=0)
        compensated_g3 = torch.cat(compensated_g3, dim=0)

        return compensated_g1, compensated_g2, compensated_g3, compensation_params





