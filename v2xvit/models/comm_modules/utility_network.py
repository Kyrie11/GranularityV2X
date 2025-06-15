import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation Layer."""

    def __init__(self, channels, condition_dim):
        super(FiLMLayer, self).__init__()
        self.channels = channels
        self.condition_dim = condition_dim
        self.projection = nn.Linear(condition_dim, 2 * channels)  # Predicts gamma and beta

    def forward(self, x, condition):
        # x: [B, C, H, W] (main feature map)
        # condition: [B, condition_dim] (conditional vector)

        gamma_beta = self.projection(condition)  # [B, 2*C]
        gamma = gamma_beta[:, :self.channels].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = gamma_beta[:, self.channels:].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        return gamma * x + beta


#计算效益网络的GroundTruth
class TargetUtilityCalculator(nn.Module):
    def __init__(self, single_granularity_channels_list,
                    full_bev_channels,
                    embedding_dim=64,
                    bandwidth_costs_list=None):
        super(TargetUtilityCalculator, self).__init__()

        self.num_granularites = len(single_granularity_channels_list)

        #为每种单一粒度数据创建一个嵌入层
        self.single_embedders = nn.ModuleList()
        for C_g in single_granularity_channels_list:
            self.single_embedders.append(nn.Linear(C_g, embedding_dim))

        #为全信息创建一个嵌入层
        self.full_embedder = nn.Linear(full_bev_channels, embedding_dim)

        if bandwidth_costs_list is None:
            #默认带宽成本
            bandwidth_costs_list = [10.0, 5.0, 2.0]

        self.register_buffer('bandwidth_costs', torch.tensor(bandwidth_costs_list, dtype=torch.float32))

        self.similarity_func = nn.CosineSimilarity(dim=-1) #计算最后一个维度的余弦相似度

    def forward(self, F_vox_bev, F_feat_bev, F_det_bev, X_bev):
        # F_vox_bev_i: [B, C_V, H, W]
        # F_feat_bev_i: [B, C_F, H, W] (假设已对齐到相同H,W)
        # F_det_bev_i: [B, C_D, H, W]
        # X_bev_i: [B, C_total, H, W] (全信息，由上面三个拼接而成)
        B,_,H,W = X_bev.shape
        target_utility_map = torch.zeros((B,H,W, self.num_granularites),
                                         device=X_bev.device, dtype=X_bev.dtype)

        # 将特征图转换为 [B, H, W, C] 的形式，方便逐像素处理
        F_vox_bev_permuted = F_vox_bev.permute(0, 2, 3, 1)
        F_feat_bev_permuted = F_feat_bev.permute(0, 2, 3, 1)
        F_det_bev_permuted = F_det_bev.permute(0, 2, 3, 1)
        X_bev_permuted = X_bev.permute(0, 2, 3, 1)
        #将特征图转换为[B,H,W,C]的形式，方便逐像素处理
        single_granularity_features = [
            F_vox_bev_permuted,
            F_feat_bev_permuted,
            F_det_bev_permuted
        ]

        embedded_full_info = self.full_embedder(X_bev_permuted)

        for g_idx in range(self.num_granularites):
            #获取当前粒度的原始BEV特征
            current_g_feat = single_granularity_features[g_idx] # [B, H, W, C_g]

            #嵌入当前粒度的特征
            embedded_single_g = self.single_embedders[g_idx](current_g_feat) # [B, H, W, C_g]

            #计算相似度
            similarity = self.similarity_func(embedded_single_g, embedded_full_info)  # [B, H, W]
            # Similarity 范围是 [-1, 1]，可以调整到 [0, 1] 作为信息保留度
            retain_degree = (similarity + 1.0) / 2.0

            # 计算单位带宽效用
            target_utility_map[:, :, :, g_idx] = retain_degree / self.bandwidth_costs[g_idx]
        return target_utility_map #[B,H,W,3]



            #后续要补全：对每个粒度的带宽衡量
class UtilityNetwork(nn.Module):
    def __init__(self, collab_bev_channels,
                 granularity_coeff_dim,  # X_C^{(i)} 的维度 (例如3)
                 semantic_coeff_dim,     # X_G^{(i)} 的维度 (例如C')
                 bandwidth_vector_dim,   # B 的维度 (3)
                 output_granularity_channels=3, # 输出3个粒度的效用
                 hidden_channels=64,
                 num_blocks=2):
        super(UtilityNetwork, self).__init__()
        self.output_granularity_channels = output_granularity_channels
        # 1. 初步卷积处理 Collaborator BEV 特征
        self.initial_conv = nn.Sequential(
            nn.Conv2d(collab_bev_channels+1, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),  # 可选
            nn.ReLU()
        )

        # 2. 将交互系数编码为条件向量
        # 空间交互系数 X_S 通常是 [B, C_S', H, W]，我们需要将其全局池化或通过卷积得到一个向量
        # 为了简化，假设 X_S 已经通过某种方式被处理成了一个可以影响全局的条件，
        # 或者它会与 collab_bev_features 在通道维度拼接（如果分辨率和通道数允许）
        # 这里我们先假设 X_S 会被用于调制
        # 将粒度系数、语义系数、带宽向量拼接成一个总的条件向量
        total_condition_dim = granularity_coeff_dim + semantic_coeff_dim + bandwidth_vector_dim
        # 3. 条件卷积块 (使用FiLM层进行调制)
        self.conditional_blocks = nn.ModuleList()
        current_channels = hidden_channels

        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Conv2d(current_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),  # 可选
                nn.ReLU()
            )
            film_layer = FiLMLayer(hidden_channels, total_condition_dim)
            self.conditional_blocks.append(nn.ModuleDict({'conv_block': block, 'film': film_layer}))
            current_channels = hidden_channels  # FiLM不改变通道数

        # 4. 输出头，生成每个像素对三种粒度的效用值
        self.output_conv = nn.Conv2d(hidden_channels, output_granularity_channels, kernel_size=1)

    def forward(self, collab_fused_bev,  # [B, C_collab_total, H, W]
                spatial_coefficient,  # [B, 1, H, W]
                granularity_coefficient,  # [B, 3]
                semantic_coefficient,  # [B, C']
                bandwidth_vector  # [B, 3]
                ):
        device = collab_fused_bev.device
        bandwidth_vector = bandwidth_vector.to(device)
        print("bandwidth_vector.shape=", bandwidth_vector.shape)
        print("granularity_coefficient.shape=", granularity_coefficient.shape)
        print("semantic_coefficient.shape=",semantic_coefficient.shape)
        x_input = torch.cat([collab_fused_bev, spatial_coefficient], dim=1)

        # a. 初步处理collab BEV特征
        x = self.initial_conv(x_input)  # [1, hidden_channels, H, W]

        # b. (可选) 融合空间交互系数
        # 如果 spatial_coefficient 是特征图，可以与 x 拼接后再通过一个1x1卷积
        # x = torch.cat([x, spatial_coefficient], dim=1)
        # x = self.initial_conv_merged(x)
        # 或者，更简单地，将 spatial_coefficient 的全局平均池化结果加入到下面的 condition_vector 中
        # 假设 spatial_coefficient 主要用于后续的选择机制，这里不直接融入网络（或已体现在collab_fused_bev中）

        # c. 构建总的条件向量 (用于FiLM调制)
        # (确保所有输入都是 [B, Dim] 的形状)
        condition_vector = torch.cat([
            granularity_coefficient,
            semantic_coefficient,
            bandwidth_vector
        ], dim=1)  # [B, total_condition_dim]

        # d. 通过条件卷积块
        for layer_module in self.conditional_blocks:
            conv_block = layer_module['conv_block']
            film_layer = layer_module['film']

            x_conv = conv_block(x)
            x = film_layer(x_conv, condition_vector)  # FiLM调制
            # 可以考虑加入残差连接 x = x + x_conv_modulated (如果维度匹配)

        # e. 输出效用图
        utility_map = self.output_conv(x)  # [B, output_granularity_channels=3, H, W]

        # 将通道维度放到最后，以匹配 [H,W,3] 的语义
        utility_map = utility_map.permute(0, 2, 3, 1)  # [B, H, W, 3]

        return utility_map