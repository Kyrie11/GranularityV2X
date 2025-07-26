import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# ===================================================================
#  模块 1: FiLM 参数生成器
# ===================================================================
class FiLMParameterGenerator(nn.Module):
    """
        从 ego-agent 的需求图 D 生成 FiLM 参数 γ 和 β。
    """
    def __init__(self, target_channels: int):
        """
        初始化。
        参数:
            target_channels (int): 目标特征图(unified_bev)的通道数，即 C'。
        """
        super().__init__()
        # 一个小型CNN来处理需求图 (3 channels) 并输出 2*C' 个通道
        self.param_generator = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2 * target_channels, kernel_size=1)
        )
        self.target_channels = target_channels

    def forward(self, ego_demand):
        """
        前向传播。
        参数:
            ego_demand (torch.Tensor): Ego-agent 的需求图 D。Shape: [1, 3, H, W]。
        返回:
            Tuple[torch.Tensor, torch.Tensor]: gamma 和 beta。
                                               Shape of both: [1, C', H, W]。
        """
        params = self.param_generator(ego_demand)
        # 沿通道维度分割，前一半是 gamma，后一半是 beta
        gamma, beta = torch.split(params, self.target_channels, dim=1)
        return gamma, beta

# ===================================================================
#  模块 2: 稀疏特征重建头
# ===================================================================

class ReconstructionHead(nn.Module):
    """
    使用与主模型共享的编码器，从稀疏数据重建统一特征图 H'。
    """
    def __init__(self, encoder_g1, encoder_g2, encoder_g3, fusion_conv):
        """
        初始化。
        参数:
            encoder_g1, encoder_g2, encoder_g3: 共享的粒度编码器。
            fusion_conv: 共享的1x1融合卷积层。
        """
        super().__init__()
        self.encoder_g1 = encoder_g1
        self.encoder_g2 = encoder_g2
        self.encoder_g3 = encoder_g3
        self.fusion_conv = fusion_conv

    def forward(self, sparse_g1_data: torch.Tensor, sparse_g2_data: torch.Tensor,
                sparse_g3_data: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        参数:
            sparse_gX_data (torch.Tensor): 协作 agents 的稀疏粒度数据。
                                            Shape: [N-1, C_gX, H, W]。
        返回:
            torch.Tensor: 重建的稀疏统一特征图 H'。Shape: [N-1, C', H, W]。
        """
        encoded_g1 = self.encoder_g1(sparse_g1_data)
        encoded_g2 = self.encoder_g2(sparse_g2_data)
        encoded_g3 = self.encoder_g3(sparse_g3_data)
        concatenated_features = torch.cat([encoded_g1, encoded_g2, encoded_g3], dim=1)

        h_prime = self.fusion_conv(concatenated_features)
        return h_prime

# ===================================================================
#  模块 3: 完整的蒸馏损失模块
# ==================================================================
class DistillationLoss(nn.Module):
    """
        计算稀疏重建特征 H' 与 FiLM 调制后的目标特征 I_aware 之间的蒸馏损失。
    """
    def __init__(self, unified_bev_channels: int, encoders: List[nn.Module], fusion_conv: nn.Module):
        """
        初始化。
        参数:
            unified_bev_channels (int): unified_bev 的通道数 C'。
            encoders (List[nn.Module]): 包含 [encoder_g1, encoder_g2, encoder_g3] 的列表。
            fusion_conv (nn.Module): 1x1 融合卷积层。
        """
        super().__init__()
        self.film_param_generator = FiLMParameterGenerator(unified_bev_channels)
        self.reconstruction_head = ReconstructionHead(encoders[0], encoders[1], encoders[2], fusion_conv)
        # FiLM Layer 本身只是一个操作，不需要是 nn.Module

    def forward(self,
                ego_demand: torch.Tensor,
                collaborator_unified_bevs: torch.Tensor,
                sparse_datas: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        计算最终的损失值。
        参数:
            ego_demand (torch.Tensor): Ego-agent 的需求图 D。Shape: [1, 3, H, W]。
            collaborator_unified_bevs (torch.Tensor): 所有协作 agents 的原始 unified_bev。
                                                      Shape: [N-1, C', H, W]。
            sparse_datas (Tuple[torch.Tensor, ...]): 包含三个稀疏数据张量的元组
                                                     (sparse_g1, sparse_g2, sparse_g3)。
                                                     每个张量 Shape: [N-1, C_gX, H, W]。
        返回:
            torch.Tensor: 计算出的标量损失值。
        """
        # 1. 生成 FiLM 参数
        gamma, beta = self.film_param_generator(ego_demand)

        # 2. 计算目标特征 I_{j,aware}
        #    广播 gamma 和 beta 以匹配协作 agents 的数量
        #    gamma, beta 的 shape 从 [1, C', H, W] 变为 [N-1, C', H, W]
        num_collaborators = collaborator_unified_bevs.shape[0]
        gamma = gamma.expand(num_collaborators, -1, -1, -1)
        beta = beta.expand(num_collaborators, -1, -1, -1)

        # 应用 FiLM 调制 (detach target to avoid gradients flowing back through it)
        # 我们希望 H' 学习 I_aware，而不是 I_aware 改变自己以适应 H'
        with torch.no_grad():
            i_aware = gamma * collaborator_unified_bevs + beta

        # 3. 从稀疏数据重建预测特征 H'_{j}
        sparse_g1, sparse_g2, sparse_g3 = sparse_datas
        h_prime = self.reconstruction_head(sparse_g1, sparse_g2, sparse_g3)

        # 4. 计算蒸馏损失 (L2-loss / MSE is a common choice)
        loss = F.mse_loss(h_prime, i_aware)

        return loss