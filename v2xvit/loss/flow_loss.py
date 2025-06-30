import torch
import torch.nn as nn
import torch.nn.functional as F
# 您需要安装 torchmetrics: pip install torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure


class FocalLoss(nn.Module):
    """
    一个健壮且标准的Focal Loss实现。
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs 和 targets 应该是 [B, C, H, W]
        # 我们需要在计算前将它们 flatten
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = p if y=1, 1-p if y=0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class CompensationLoss(nn.Module):
    """
    用于多粒度BEV延时补偿的混合损失函数。
    它实现了我们讨论的“优雅设计方案”。
    """

    def __init__(self, args_loss):
        super(CompensationLoss, self).__init__()

        # 从配置中获取权重参数
        self.lambda_feat = args_loss.get('lambda_feat', 1.0)
        self.lambda_det = args_loss.get('lambda_det', 1.0)
        self.lambda_vox = args_loss.get('lambda_vox', 1.0)

        # L1+SSIM 组合损失中，SSIM损失的权重
        self.alpha_ssim = args_loss.get('alpha_ssim', 0.1)

        # 1. feat_bev 的损失函数 (L1 + SSIM)
        self.l1_loss = nn.L1Loss(reduction='mean')
        # 假设特征值范围在0-1之间，如果不是，需要调整data_range
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        # 2. det_bev 的损失函数 (Focal Loss)
        focal_alpha = args_loss.get('focal_alpha', 0.25)
        focal_gamma = args_loss.get('focal_gamma', 2.0)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # 3. vox_bev 的损失函数 (L1)
        # 我们复用 self.l1_loss 即可

    def forward(self, predicted_bevs, ground_truth_bevs):
        """
        计算总损失。

        Args:
            predicted_bevs (dict): 包含 'vox_bev', 'feature_bev', 'det_bev' 的字典，值为模型预测的Tensor。
            ground_truth_bevs (dict): 包含 'vox_bev', 'feature_bev', 'det_bev' 的字典，值为t0时刻的真值Tensor。
        """
        # 从字典中获取预测值和真值
        pred_vox, pred_feat, pred_det = predicted_bevs

        gt_vox, gt_feat, gt_det = ground_truth_bevs

        # 确保SSIM的设备与输入一致
        self.ssim = self.ssim.to(pred_feat.device)

        # --- 计算 feat_bev 损失 ---
        loss_feat_l1 = self.l1_loss(pred_feat, gt_feat)
        # SSIM值越高越好，所以用 1.0 减去它作为损失。
        # 注意：SSIM需要输入至少4个通道，如果你的feat_bev通道少于4，需要调整
        # 这里我们假设它满足条件。如果特征值范围不是[0,1]，SSIM会不准确。
        loss_feat_ssim = 1.0 - self.ssim(pred_feat, gt_feat)
        loss_feat = loss_feat_l1 + self.alpha_ssim * loss_feat_ssim

        # --- 计算 det_bev 损失 ---
        # Focal Loss 通常用于logits，这里假设det_bev是类似heatmap的概率图
        # 因此，在使用FocalLoss之前最好对gt_det做clamp处理，避免log(0)
        gt_det_clamped = torch.clamp(gt_det, min=0, max=1)
        loss_det = self.focal_loss(pred_det, gt_det_clamped)

        # --- 计算 vox_bev 损失 ---
        loss_vox = self.l1_loss(pred_vox, gt_vox)

        # --- 计算加权总损失 ---
        total_loss = (self.lambda_feat * loss_feat +
                      self.lambda_det * loss_det +
                      self.lambda_vox * loss_vox)

        # 返回一个字典，方便日志记录和分析
        loss_dict = {
            'total_loss': total_loss,
            'loss_feat': loss_feat,
            'loss_det': loss_det,
            'loss_vox': loss_vox
        }
        return total_loss

