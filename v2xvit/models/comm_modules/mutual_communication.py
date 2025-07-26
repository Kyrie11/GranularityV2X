import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from typing import List

# ===================================================================
#  模块 1: 粒度置信度模块 (Granularity Confidence Module, GCM)
# ===================================================================

class GranularityConfidenceModule(nn.Module):
    """
    粒度置信度模块 (GCM)。

    该模块接收一个 agent 的 unified_bev 特征图，并为图中的每个空间位置
    预测三个不同粒度（voxel, feature, result）的置信度。
    """
    def __init__(self, unified_bev_channels: int):
        super(GranularityConfidenceModule, self).__init__()
        # 定义一个小型卷积网络来处理特征并生成置信度图
        self.confidence_net = nn.Sequential(
            nn.Conv2d(unified_bev_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 1x1 卷积将特征维度映射到 3，分别对应三种粒度的置信度
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),

            # Sigmoid 激活函数将输出值归一化到 [0, 1] 范围，作为置信度
            nn.Sigmoid()
        )

    def forward(self, unified_bev: torch.Tensor) -> torch.Tensor:
        return self.confidence_net(unified_bev)

# ===================================================================
#  模块 2: 效用匹配模块 (Utility Matching Module, UMM)
# ===================================================================
class UtilityMatchingModule(nn.Module):
    """
    效用匹配模块 (UMM)。

    该模块将 ego-agent 的需求图 D 和一个协作 agent 的置信度图 C 进行匹配，
    计算出协作 agent 传输不同粒度数据的效用值。
    """

    def __init__(self):
        """
        初始化 UMM。
        """
        super(UtilityMatchingModule, self).__init__()

        # 输入通道为 6 (3 for demand D + 3 for confidence C)
        self.utility_net = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 1x1 卷积输出 3 个通道的效用值 (logits)
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, combined_feature) -> torch.Tensor:
        utility_map = self.utility_net(combined_feature)
        return utility_map

# ===================================================================
#  传输稀疏化模块 (Transmission Sparsification Module)
# ===================================================================
class TransmissionSparsificationModule(nn.Module):
    """
    传输稀疏化模块。

    根据净效用和决策阈值，生成最终的稀疏传输掩码。
    """
    def __init__(self, granularity_costs: List[int], alpha: List[float], selection_threshold: float):
        """
        初始化模块。

        参数:
            granularity_costs (List[int]): 每个粒度数据的单位成本 (即通道数)。
                                           例如: [C_g1, C_g2, C_g3]。
            alpha (float): 成本权重因子 α。
            selection_threshold (float): 净效用的选择阈值。
        """
        super(TransmissionSparsificationModule, self).__init__()
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32).view(1, 3, 1, 1)
        self.selection_threshold = selection_threshold
        # 将成本列表转换为 [1, 3, 1, 1] 的张量，以便于广播
        # register_buffer 会将张量注册到模型，并随模型移动到 CPU/GPU，但它不是模型参数
        costs_tensor = torch.tensor(granularity_costs, dtype=torch.float32).view(1, 3, 1, 1)
        weighted_costs = alpha_tensor * costs_tensor
        self.register_buffer('weighted_costs', weighted_costs)

    def forward(self, utility_maps: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            utility_maps (torch.Tensor): 来自 UMM 的效用图。
                                         Shape: [N-1, 3, H, W]。

        返回:
            torch.Tensor: 最终的稀疏传输掩码。
                          Shape: [N-1, 3, H, W]，值为 0 或 1。
        """
        # 1. 计算净效用: net_utility = utility - α * cost
        #    self.costs 的 shape 是 [1, 3, 1, 1]，会自动广播
        net_utility = utility_maps - self.weighted_costs

        # 2. 找到每个位置净效用最高的粒度及其对应的净效用值
        #    keepdim=True 保持维度为 [N-1, 1, H, W]，便于后续操作
        max_net_utility, best_granularity_indices = torch.max(net_utility, dim=1, keepdim=True)
        # 3. 判断净效用是否高于阈值
        is_above_threshold = max_net_utility > self.selection_threshold
        # 4. 生成最终的稀疏掩码
        #    a. 创建一个 one-hot 编码的掩码，标记出最优粒度的位置
        #       best_granularity_indices 的 shape 是 [N-1, 1, H, W]
        #       one_hot_selection 的 shape 是 [N-1, 3, H, W]
        one_hot_selection = torch.zeros_like(net_utility)
        one_hot_selection.scatter_(1, best_granularity_indices, 1)

        #    b. 结合阈值判断，生成最终掩码
        #       只有当一个粒度是“最优”且“效用高于阈值”时，才将其标记为 1
        #       is_above_threshold [N-1, 1, H, W] 会被广播到 [N-1, 3, H, W]
        sparse_maps = one_hot_selection * is_above_threshold.float()
        return sparse_maps

# ===================================================================
#  传输模块(Main)
# ===================================================================
class AdvancedCommunication(nn.Module):
    def __init__(self, c_vox, c_feat, c_det, c_semantic=32, lambda_rec=0.5):
        super(AdvancedCommunication, self).__init__()
        # self.channel_request = Channel_Request_Attention(in_planes)

        # 注册成本向量为buffer
        self.register_buffer('cost_vector', torch.tensor([c_vox, c_feat, c_det], dtype=torch.float32))

        # --- NEW: Hyperparameter to balance the losses ---
        self.lambda_rec = lambda_rec
        # Using L1 Loss is often better for image-to-image tasks as it's less blurry
        self.reconstruction_loss_fn = nn.L1Loss()

        self.thre = 0.1 #高于阈值时才传输
        self.alpha = [0.1, 0.2, 0.1] #调制粒度损失的参数

        self.gcm = GranularityConfidenceModule(unified_bev_channels=256)
        self.umm = UtilityMatchingModule()

        cost_list = [8,256,8]
        self.selection_net = TransmissionSparsificationModule(cost_list, self.alpha, self.thre)

    def get_sparse_data(self, g1_data, g2_data, g3_data, sparse_maps):
        ego_g1 = g1_data[0:1]
        ego_g2 = g2_data[0:1]
        ego_g3 = g3_data[0:1]

        collab_g1 = g1_data[1:]
        collab_g2 = g2_data[2:]
        collab_g3 = g3_data[3:]

        collab_sparse_g1 = collab_g1 * sparse_maps[:, 0:1, :, :]
        collab_sparse_g2 = collab_g2 * sparse_maps[:, 1:2, :, :]
        collab_sparse_g3 = collab_g3 * sparse_maps[:, 2:3, :, :]

        sparse_g1 = torch.stack([ego_g1, collab_sparse_g1], dim=0)
        sparse_g2 = torch.stack([ego_g2, collab_sparse_g2], dim=0)
        sparse_g3 = torch.stack([ego_g3, collab_sparse_g3], dim=0)

        return sparse_g1, sparse_g2, sparse_g3

    def forward(self, g1, g2, g3, unified_bev):
        '''
        :param g1_list: g1数据列表，长度为batch_size
        :param g2_list:
        :param g3_list:
        :param unified_list: 三个粒度数据通道拼接再经过1x1卷积
        :return:
        '''
        commu_volume = 0

        num_agents, c_g1, H, W = g1.shape
        ego_bev = unified_bev[0:1]  # Shape: [1, C', H, W]
        ego_confidence = self.gcm(ego_bev)  # Shape: [1, 3, H, W]
        ego_demand = 1.0 - ego_confidence  # Shape: [1, 3, H, W]

        if num_agents <= 1:
            # If no collaborators, append empty tensors to maintain output structure
            return ego_demand, g1, g2, g3, commu_volume

        collaborator_bevs = unified_bev[1:]  # Shape: [N-1, C', H, W]


        expanded_ego_demand = ego_demand.expand(num_agents-1, -1, -1, -1)

        collaborator_confidences = self.gcm(collaborator_bevs)  # Shape: [N-1, 3, H, W]

        combined_features = torch.cat((expanded_ego_demand, collaborator_confidences), dim=1)
        utility_maps = self.umm(combined_features) #[N-1,3,H,W]
        sparse_maps = self.selection_net(utility_maps) #[N-1,3,H,W]

        sparse_g1, sparse_g2, sparse_g3 = self.get_sparse_data(g1, g2, g3, sparse_maps) #返回的ego-agent的数据不变，其他的稀疏化

        commu_volume = sparse_maps.sum()

        return ego_demand, sparse_g1, sparse_g2, sparse_g3, commu_volume