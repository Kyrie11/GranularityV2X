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
    def __init__(self, single_granularity_channels_list, #[C_V,C_F,C_D]
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


class TransmissionSelector(nn.Module):
    def __init__(self, C_V, C_F, C_D, bandwidth_costs, utility_network=None, no_transmission_utility_penalty=0.0):
        super(TransmissionSelector, self).__init__()
        self.C_V = C_V
        self.C_F = C_F
        self.C_D = C_D
        self.register_buffer('bandwidth_costs_tensor',
                             torch.tensor(bandwidth_costs, dtype=torch.float32))  # [B_vox, B_feat, B_det]
        self.utility_network = utility_network  # 传入预训练或正在训练的效用网络
        self.num_granularities = len(bandwidth_costs)
        self.total_channels_out = C_V + C_F + C_D

        # 引入一个小的惩罚项，使得在效用都接近0时，倾向于不传输
        # 或者，可以将其视为不传输的“效用”是负的（如果U是正向效用）
        self.no_transmission_utility_penalty = no_transmission_utility_penalty

    def selection_mechanism(self, utility_map_per_granularity, bandwidth_budget_per_sample):
        # utility_map_per_granularity: [B, H, W, 3] (每个像素对三种粒度的单位带宽效用)
        # bandwidth_budget: scalar (总带宽预算)

        B, H, W, G = utility_map_per_granularity.shape
        assert G == self.num_granularities, "Utility map granularity dim mismatch"
        device = utility_map_per_granularity.device

        # 初始化最终的选择索引图，-1代表不选择任何粒度
        selected_granularity_indices = torch.full((B, H, W), -1.0, device=device, dtype=torch.float32)

        # 1. 阶段一：区域内最优粒度选择 (并考虑不传输的选项)
        #    我们希望选择的粒度能最大化 U_g (单位带宽效用)
        #    如果所有粒度的 U_g 都很低 (例如，低于某个阈值或低于不传输的效用)，则不传输
        # 假设不传输的单位带宽效用是一个小的负值（或0，取决于U的范围）
        # 为了让模型有不选择的倾向，可以给“不传输”一个固定的效用值，例如0，
        # 如果所有粒度的预测效用都小于这个值，则不选。
        # 或者，在比较时，如果max(U_g) < threshold_for_transmission，则不选。
        # 这里我们直接取每个位置效用最高的粒度
        best_utility_at_pixel, best_granularity_idx_at_pixel = torch.max(utility_map_per_granularity, dim=3)
        # best_utility_at_pixel: [B, H, W]
        # best_granularity_idx_at_pixel: [B, H, W] (值为0, 1, 2)

        # 考虑一个最小效用阈值，低于此阈值的即使是局部最优也不选择
        # (这个阈值可以是一个超参数，或者根据no_transmission_utility_penalty调整)
        min_utility_threshold = self.no_transmission_utility_penalty # 示例：如果效用小于0就不值得传

        # 创建一个列表存储所有潜在的“区域最优传输选项”
        # 每个元素: (utility, cost, b_idx, r_y, r_x, g_idx)
        candidate_items = []

        for b_idx in range(B):
            for r_y in range(H):
                for r_x in range(W):
                    current_best_g_idx = best_granularity_idx_at_pixel[b_idx, r_y, r_x].item()
                    current_best_utility = best_utility_at_pixel[b_idx, r_y, r_x].item()

                    if current_best_utility > min_utility_threshold:  # 只有当最优粒度的效用足够高时才考虑
                        cost = self.bandwidth_costs_tensor[current_best_g_idx].item()
                        candidate_items.append({
                            'utility': current_best_utility,  # 这是单位带宽效用
                            'cost': cost,
                            'b_idx': b_idx,
                            'r_y': r_y,
                            'r_x': r_x,
                            'g_idx': current_best_g_idx
                        })

        # 2. 阶段二：跨区域带宽约束下的选择 (贪心)
        #    按照单位带宽效用（即 candidate_items中的 'utility'）降序排序
        candidate_items.sort(key=lambda x: x['utility'], reverse=True)
        current_sample_bandwidth_usage = torch.zeros(B, device=device)

        for item in candidate_items:
            b_idx, r_y, r_x, g_idx = item['b_idx'], item['r_y'], item['r_x'], item['g_idx']
            cost = item['cost']

            if current_sample_bandwidth_usage[b_idx] + cost <= bandwidth_budget_per_sample:
                selected_granularity_indices[b_idx, r_y, r_x] = float(g_idx)
                current_sample_bandwidth_usage[b_idx] += cost

        return selected_granularity_indices  # [B, H, W]

    def build_sparse_transmitted_bev(self, selected_granularity_indices,
                                     F_vox_bev, F_feat_bev, F_det_bev):
        # selected_granularity_indices: [B, H, W], 值为 -1, 0, 1, 2
        # F_vox_bev: [B, C_V, H, W]
        # F_feat_bev: [B, C_F, H, W]
        # F_det_bev: [B, C_D, H, W]

        B, H, W = selected_granularity_indices.shape
        device = selected_granularity_indices.device

        f_trans_bev = torch.zeros((B, self.total_channels_out, H, W), device=device, dtype=F_vox_bev.dtype)

        # 创建每个粒度的选择掩码
        mask_vox = (selected_granularity_indices == 0).unsqueeze(1)  # [B, 1, H, W]
        mask_feat = (selected_granularity_indices == 1).unsqueeze(1)
        mask_det = (selected_granularity_indices == 2).unsqueeze(1)

        # 填充体素通道
        # 需要确定体素通道在f_trans_bev中的范围，假设是前C_V个
        if self.C_V > 0:
            f_trans_bev[:, :self.C_V, :, :] = F_vox_bev * mask_vox

        # 填充特征通道
        # 假设特征通道在体素通道之后
        start_idx_feat = self.C_V
        end_idx_feat = self.C_V + self.C_F
        if self.C_F > 0:
            f_trans_bev[:, start_idx_feat:end_idx_feat, :, :] = F_feat_bev * mask_feat

        # 填充检测通道
        # 假设检测通道在特征通道之后
        start_idx_det = self.C_V + self.C_F
        end_idx_det = self.C_V + self.C_F + self.C_D
        if self.C_D > 0:
            f_trans_bev[:, start_idx_det:end_idx_det, :, :] = F_det_bev * mask_det

        return f_trans_bev

    def forward(self, collab_bev_data_list, bandwidth_budget, utility_map_list):
        # ego_requests: {'R_S_ego':..., 'R_C_ego':..., 'A_G_ego':...}
        # collab_states_list: list of {'A_S_collab':..., 'A_C_collab':..., 'A_G_collab':...}
        # collab_bev_data_list: list of {'vox_bev':..., 'feat_bev':..., 'det_bev':...}

        # ... (省略了调用utility_network得到utility_map_list的过程) ...
        # 假设 utility_map_list 是一个包含了每个协作方预测的效用图的列表
        cav_num = collab_bev_data_list.shape[0]

        all_sparse_trans_bevs = []
        all_selected_indices = []

        selected_granularity_indices = self.selection_mechanism(utility_map_list, bandwidth_budget/cav_num)
        sparse_trans_bev = self.build_sparse_transmitted_bev(selected_granularity_indices,
                                                             collab_bev_data_list[:, :self.C_V, :, :],
                                                             collab_bev_data_list[:, self.C_V:self.C_V + self.C_F, :, :],
                                                             collab_bev_data_list[:, self.C_V + self.C_F:self.C_V + self.C_F + self.C_D, :, :])

        # for i, utility_map_i in enumerate(utility_map_list):  # 遍历每个协作方的效用图
        #     # utility_map_i: [1, H, W, 3] (假设batch_size为1)
        #     print("utility_map_i.shape=",utility_map_i.shape)
        #     # 1. 进行选择 (这是核心的背包问题或其近似)
        #     selected_granularity_indices_i = self.selection_mechanism(utility_map_i, bandwidth_budget / cav_num)  # 简单均分预算
        #     all_selected_indices.append(selected_granularity_indices_i)
        #
        #     # 2. 构建稀疏传输图
        #     collab_data = collab_bev_data_list[i]
        #     sparse_trans_bev_i = self.build_sparse_transmitted_bev(
        #         selected_granularity_indices_i,
        #         collab_data[:self.C_V, :, :],
        #         collab_data[self.C_V:self.C_V+self.C_F, :, :],
        #         collab_data[self.C_V+self.C_F:self.C_V+self.C_F+self.C_D, :, :]
        #     )
        #     all_sparse_trans_bevs.append(sparse_trans_bev_i)

        # 返回所有协作方选择传输的稀疏BEV图列表，以及选择的索引（用于可能的损失计算或分析）
        return sparse_trans_bev, selected_granularity_indices



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