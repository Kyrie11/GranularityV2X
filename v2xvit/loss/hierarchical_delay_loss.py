import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


# 辅助函数: warp, 用于根据位移场扭曲特征图
def warp(feature_map: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """
    根据位移场(offset)来扭曲(warp)输入的特征图(feature_map)。

    Args:
        feature_map (torch.Tensor): 输入的特征图，形状为 [B, C, H, W]。
        offset (torch.Tensor): 预测的位移场，形状为 [B, 2, H, W]，其中通道0为x方向位移，通道1为y方向位移。

    Returns:
        torch.Tensor: 扭曲后的特征图，形状与输入特征图相同。
    """
    B, _, H, W = feature_map.shape

    # 1. 创建一个标准网格，表示原始像素坐标
    # y_coords, x_coords 的形状都是 [H, W]
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    # 将网格移动到与输入张量相同的设备上
    grid = torch.stack((x_coords, y_coords), dim=0).float().to(feature_map.device)  # shape: [2, H, W]
    # 扩展网格以匹配批次大小
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # shape: [B, 2, H, W]

    # 2. 计算采样网格
    # 将预测的位移加到标准网格上
    sampling_grid = grid + offset

    # 3. 对采样坐标进行归一化
    # F.grid_sample 需要的坐标范围是 [-1, 1]
    # x方向归一化
    sampling_grid[:, 0, :, :] = 2 * sampling_grid[:, 0, :, :] / (W - 1) - 1
    # y方向归一化
    sampling_grid[:, 1, :, :] = 2 * sampling_grid[:, 1, :, :] / (H - 1) - 1

    # 4. 调整维度顺序以匹配 F.grid_sample 的要求
    # 需要从 [B, 2, H, W] 变为 [B, H, W, 2]
    sampling_grid = sampling_grid.permute(0, 2, 3, 1)

    # 5. 执行采样
    # mode='bilinear' 进行双线性插值
    # padding_mode='zeros' 对超出边界的区域填充0
    warped_feature = F.grid_sample(
        feature_map, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    return warped_feature

class HierarchicalDelayLoss(nn.Module):
    """
    分层蒸馏损失函数。

    该损失函数与分层预测头的结构完美对齐，包含三部分：
    1. 基础场监督 (L_base): 监督核心的 G1 位移场。
    2. 残差场精炼 (L_residual): 使用 .detach() 机制，专门监督 G2 和 G3 的残差位移。
    3. 置信度校准 (L_confidence): 使用 BCE 损失监督共享的置信度图。
    """
    def __init__(self, lambda_base = 1.0, lambda_res = 1.0, lambda_conf = 0.5):
        """
        初始化损失函数。

        Args:
            lambda_base (float): 基础场损失的权重。
            lambda_res (float): 残差场损失的权重。
            lambda_conf (float): 置信度损失的权重。
        """
        super(HierarchicalDelayLoss, self).__init__()
        self.lambda_base = lambda_base
        self.lambda_res = lambda_res
        self.lambda_conf = lambda_conf

        # 使用CosineSimilarity来计算特征相似度
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        # 使用BCEWithLogitsLoss来提高数值稳定性，它内部会先对输入应用sigmoid
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, predict_params, real_g1, real_g2, real_g3, delay_g1, delay_g2, delay_g3, record_len):
        """
        计算总损失。

        Args:
            predict_params (Dict): 模型预测的输出，包含 'offset_g1', 'delta_offset_g2',
                                'delta_offset_g3', 'scale_shared'。
            delay_g1, delay_g2, delay_g3: 协作者在 t-d 时刻的原始特征图。
            real_g1, real_g2, real_g3: t 时刻的真值特征图。

        Returns:
            Dict: 包含总损失和各分项损失的字典。
        """
        batch_size = len(record_len)

        real_g1 = self.regroup(real_g1, record_len)
        real_g2 = self.regroup(real_g2, record_len)
        real_g3 = self.regroup(real_g3, record_len)

        delay_g1 = self.regroup(delay_g1, record_len)
        delay_g2 = self.regroup(delay_g2, record_len)
        delay_g3 = self.regroup(delay_g3, record_len)

        total_loss = []
        for b in range(batch_size):
            collab_num = record_len[b] - 1
            if collab_num <=0:
                continue
            batch_predict_params = predict_params[b]
            # --- 1. 基础场监督 (L_base) ---

            batch_delay_g1 = delay_g1[b][1:, :, :, :]
            batch_delay_g2 = delay_g2[b][1:, :, :, :]
            batch_delay_g3 = delay_g3[b][1:, :, :, :]

            batch_real_g1 = real_g1[b][1:, :, :, :]
            batch_real_g2 = real_g2[b][1:, :, :, :]
            batch_real_g3 = real_g3[b][1:, :, :, :]

            offset_g1 = batch_predict_params['offset_g1']
            warped_g1 = warp(batch_delay_g1, offset_g1)

            # 计算 1 - cos_sim，并在批次和空间维度上取平均
            loss_base = (1-self.cosine_sim(warped_g1, batch_real_g1)).mean()

            # --- 2. 残差场精炼 (L_residual) ---
            # 关键步骤: 分离 offset_g1 的梯度
            offset_g1_detached = offset_g1.detach()

            #重建G2的最终位移场并计算损失
            offset_g2_final = offset_g1_detached + batch_predict_params['delta_offset_g2']
            warped_g2 = warp(batch_delay_g2, offset_g2_final)
            loss_res_g2 = (1-self.cosine_sim(warped_g2, batch_real_g2)).mean()

            #重建G3的最终位移场并计算损失
            offset_g3_final = offset_g1_detached + batch_predict_params['delta_offset_g3']
            warped_g3 = warp(batch_delay_g3, offset_g3_final)
            loss_res_g3 = (1-self.cosine_sim(warped_g3, batch_real_g3)).mean()

            loss_residual = loss_res_g2 + loss_res_g3

            # --- 3. 置信度校准 (L_confidence) ---
            scale_shared_logits = batch_predict_params['scale_shared']

            # 创建真值掩码 (Ground Truth Mask)
            # 只要任何一个粒度的真值特征不为0，该位置就应该被占据
            with torch.no_grad():
                gt_mask_g1 = torch.sum(torch.abs(batch_real_g1), dim=1, keepdim=True)
                gt_mask_g2 = torch.sum(torch.abs(batch_real_g2), dim=1, keepdim=True)
                gt_mask_g3 = torch.sum(torch.abs(batch_real_g3), dim=1, keepdim=True)

                gt_mask_combined = (gt_mask_g1 | gt_mask_g2 | gt_mask_g3).float()

            loss_confidence = self.bce_loss(scale_shared_logits, gt_mask_combined)

            # --- 4. 计算最终总损失 ---
            loss = (self.lambda_base * loss_base +
                          self.lambda_res * loss_residual +
                          self.lambda_conf * loss_confidence)

            total_loss.append(loss)
        final_loss = torch.mean(torch.stack(total_loss))
        return final_loss