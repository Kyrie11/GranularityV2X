import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import random

from v2xvit.models.comm_modules.utility_network import UtilityNetwork, TargetUtilityCalculator


# class Channel_Request_Attention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(Channel_Request_Attention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.sharedMLP = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
#             nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avgout = self.sharedMLP(self.avg_pool(x))
#         maxout = self.sharedMLP(self.max_pool(x))
#         return self.sigmoid(avgout + maxout)


#[1,1,3]
class Granularity_Request_Attention(nn.Module):
    def __init__(self, in_channels, num_target_granularities=3, reduction_ratio=16):
        super(Granularity_Request_Attention, self).__init__()
        self.num_target_granularities = num_target_granularities


        # 使用全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP用于从全局特征生成粒度丰富度分数
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, num_target_granularities, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局池化
        avg_pool_out = self.avg_pool(x).view(x.size(0), -1)  # [B, C_total]
        max_pool_out = self.max_pool(x).view(x.size(0), -1)  # [B, C_total]
        # MLP处理
        # 可以选择只用一种池化结果，或者将它们相加/拼接
        # 这里以相加为例
        global_context = avg_pool_out + max_pool_out  # [B, C_total]

        A_C_scores = self.mlp(global_context)  # [B, num_target_granularities]
        A_C = self.sigmoid(A_C_scores)  # [B, 3], 每个值在0-1之间，表示对应粒度的丰富度

        return A_C

#[1,1,c]
class Semantic_Request_Attention(nn.Module):
    def __init__(self, in_channels, out_channels_C_prime, reduction_ratio=16):
        super(Semantic_Request_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # MLP用于生成语义上下文向量
        # 与GCAM不同，这里的MLP输出维度是 C_prime，而不是固定的3
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, out_channels_C_prime, bias=False)
        )

    def forward(self, x):  # x: [B, C_total, H, W]
        avg_pool_out = self.avg_pool(x).view(x.size(0), -1)
        max_pool_out = self.max_pool(x).view(x.size(0), -1)

        global_context = avg_pool_out + max_pool_out

        A_G = self.mlp(global_context)  # [B, C_prime]
        # 可选: A_G = torch.sigmoid(A_G)
        return A_G

class Spatial_Request_Attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(Spatial_Request_Attention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class StatisticsNetwork(nn.Module):
    def __init__(self, img_feature_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=img_feature_channels, out_channels=img_feature_channels * 2, kernel_size=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=img_feature_channels * 2, out_channels=img_feature_channels * 2, kernel_size=1, stride=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=img_feature_channels * 2, out_channels=1, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, concat_feature: torch.Tensor) -> torch.Tensor:
        x = self.conv1(concat_feature)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        local_statistics = self.conv3(x)
        return local_statistics


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, loss_coeff=1) -> None:
        super().__init__()
        self.loss_coeff = loss_coeff

    def __call__(self, T: torch.Tensor, T_prime: torch.Tensor) -> float:
        joint_expectation = (-F.softplus(-T)).mean()
        marginal_expectation = F.softplus(T_prime).mean()
        mutual_info = joint_expectation - marginal_expectation

        return -mutual_info * self.loss_coeff


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


class Communication(nn.Module):
    def __init__(self, args, in_planes):
        super(Communication, self).__init__()
        # self.channel_request = Channel_Request_Attention(in_planes)
        self.spatial_request = Spatial_Request_Attention()
        self.semantic_request = Semantic_Request_Attention(90, 16)
        self.granularity_request = Granularity_Request_Attention(90)
        self.channel_fusion = nn.Conv2d(in_planes * 2, in_planes, 1, bias=False)
        self.spatial_fusion = nn.Conv2d(2, 1, 1, bias=False)
        self.granularity_fusion = nn.Linear(3+3, 3) # R_C_ego (3) + A_C_collab (3) -> X_C (3)
        self.semantic_channel = 16
        self.semantic_fusion = nn.Linear(2*self.semantic_channel, self.semantic_channel)
        self.statisticsNetwork = StatisticsNetwork(in_planes * 2)
        self.mutual_loss = DeepInfoMaxLoss()
        self.request_flag = args['request_flag']
        self.bandwidth_budget = args['bandwidth']

        self.smooth = False
        self.thre = args['thre']
        if 'gaussian_smooth' in args:
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            self.kernel_size = kernel_size
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(
                1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False

        x = torch.arange(-(kernel_size - 1) // 2, (kernel_size + 1) // 2, dtype=torch.float32)
        d1_gaussian_filter = torch.exp(-x ** 2 / (2 * c_sigma ** 2))
        d1_gaussian_filter /= d1_gaussian_filter.sum()

        self.d1_gaussian_filter = d1_gaussian_filter.view(1, 1, kernel_size).cuda()

        #效益网络
        self.utility_net =UtilityNetwork(collab_bev_channels=90,
            granularity_coeff_dim=3,  # X_C 的维度
            semantic_coeff_dim=self.semantic_channel, # X_G 的维度
            bandwidth_vector_dim=3)
        self.target_utility_calculator = TargetUtilityCalculator([10,64,16],90)

        #每个粒度的带宽成本
        self.B_vox = torch.tensor(args.get("bandwidth_vox",10.0))
        self.B_feat = torch.tensor(args.get("bandwidth_feat",5.0))
        self.B_det = torch.tensor(args.get("bandwidth_det",2.0))
        self.bandwidth_vector = torch.tensor([self.B_vox, self.B_feat, self.B_det])

        self.transmission_selector = TransmissionSelector(C_V=10, C_F=64, C_D=16,
                                                          bandwidth_costs=self.bandwidth_vector,utility_network=self.utility_net)


    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center: k_size -
                                        center, 0 - center: k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) +
                                                   np.square(y)) / (2 * np.square(sigma)))
            return g

        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        gaussian_kernel = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.bias.data.zero_()

    def forward(self, vox_list,feat_list,det_list,confidence_map_list=None):
        all_agents_sparse_transmitted_data = [] #存储最终每个agent要传输的稀疏数据
        comm_rate_list = []
        # sparse_mask_list = []
        total_loss = torch.zeros(1).to(feat_list[0].device)
        for bs in range(len(feat_list)):
            agent_vox = vox_list[bs]
            print("agent_vox.shape=",agent_vox.shape)
            agent_feature = feat_list[bs]
            print("agent_feature.shape=",agent_feature.shape)
            agent_det = det_list[bs]
            print("agent_det.shape=",agent_det.shape)
            agent_fused_bev = torch.cat([agent_vox, agent_feature, agent_det], dim=1)
            cav_num, C, H, W = agent_feature.shape
            spatial_coefficients = []
            semantic_coefficients = []
            granularity_coefficients = []
            if cav_num == 1:
                all_agents_sparse_transmitted_data.append(agent_fused_bev)
                # send_feats.append(agent_feature)
                # ones_mask = torch.ones(cav_num, C, H, W).to(feat_list[0].device)
                # sparse_mask_list.append(ones_mask)
                continue

            # agent_channel_attention = self.channel_request(
            #     agent_feature)
            agent_spatial_attention = self.spatial_request(
                agent_fused_bev)
            agent_semantic_attention = self.semantic_request(
                agent_fused_bev)
            # print("agent_semantic_attention.shape=", agent_semantic_attention.shape)
            agent_granularity_attention = self.granularity_request(
                agent_fused_bev)

            # agent_activation = torch.mean(agent_feature, dim=1, keepdims=True).sigmoid()
            # agent_activation = self.gaussian_filter(agent_activation)

            # ego_channel_request = (
            #         1 - agent_channel_attention[0,]).unsqueeze(0)
            ego_spatial_request = (
                    1 - agent_spatial_attention[0,]).unsqueeze(0)
            # print("ego_spatial_request.shape=",ego_spatial_request.shape)
            ego_semantic_request = (
                    agent_semantic_attention[0,]).unsqueeze(0)
            # print("ego_semantic_request.shape=", ego_semantic_request.shape)
            ego_granularity_request = (
                    1 - agent_granularity_attention[0,]).unsqueeze(0)


            for i in range(cav_num - 1):
                if self.request_flag:
                    # channel_coefficient = self.channel_fusion(torch.cat(
                    #     [ego_channel_request, agent_channel_attention[i + 1,].unsqueeze(0)], dim=1))
                    spatial_coefficient = self.spatial_fusion(torch.cat(
                        [ego_spatial_request, agent_spatial_attention[i + 1,].unsqueeze(0)], dim=1))
                    semantic_coefficient = self.semantic_fusion(torch.cat(
                        [ego_semantic_request, agent_semantic_attention[i+1,].unsqueeze(0)], dim=1))
                    granularity_coefficient = self.granularity_fusion(torch.cat(
                        [ego_granularity_request, agent_granularity_attention[i+1,].unsqueeze(0)],dim=1))
                else:
                    # channel_coefficient = agent_channel_attention[i + 1,].unsqueeze(
                    #     0)
                    spatial_coefficient = agent_spatial_attention[i + 1,].unsqueeze(
                        0)
                    semantic_coefficient = agent_semantic_attention[i+1, ].unsqueeze(
                        0)
                    granularity_coefficient = agent_granularity_attention[i+1,].unsqueeze(
                        0)

                spatial_coefficient = spatial_coefficient.sigmoid()
                semantic_coefficient = semantic_coefficient.sigmoid()
                granularity_coefficient = granularity_coefficient.sigmoid()

                spatial_coefficient = self.gaussian_filter(spatial_coefficient)

                spatial_coefficients.append(spatial_coefficient)
                semantic_coefficients.append(semantic_coefficient)
                granularity_coefficients.append(granularity_coefficient)
                # comm_rate = sparse_mask.sum() / (C * H * W)
                # comm_rate_list.append(comm_rate)


            spatial_coefficients =  torch.cat(spatial_coefficients, dim=0)
            print("spatial_coefficients.shape=", spatial_coefficients.shape)
            semantic_coefficients = torch.cat(semantic_coefficients, dim=0)
            print("semantic_coefficients.shape", semantic_coefficients.shape)
            granularity_coefficients = torch.cat(granularity_coefficients, dim=0)

            collaborators_num = agent_fused_bev.shape[0] -1
            utility_map_list = self.utility_net(agent_fused_bev[1:, :, :, :],
                                                spatial_coefficients,
                                                semantic_coefficients,
                                                granularity_coefficients,
                                                self.bandwidth_vector.expand(collaborators_num, -1))
            print("utility_map_list.shape=",utility_map_list.shape)

            target_utility_map = self.target_utility_calculator(agent_vox[1:, :, :, :],
                                                           agent_feature[1:, :, :, :],
                                                           agent_det[1:, :, :, :],
                                                           agent_fused_bev[1:, :, :, :])

            Loss_utility_pred = F.mse_loss(utility_map_list, target_utility_map)



            all_sparse_trans_bevs, all_selected_indices = self.transmission_selector(agent_fused_bev[1:, :, :, :],
                                                                                     self.bandwidth_budget,
                                                                                     utility_map_list)

            Loss_recon = self.reconstruction(all_sparse_trans_bevs, agent_fused_bev[1:, :, :, :])
            Loss_bgs = Loss_utility_pred + 0.2 * Loss_recon


            sparse_feature = torch.cat(
                [agent_fused_bev[:1], all_sparse_trans_bevs], dim=0)

            all_agents_sparse_transmitted_data.append(sparse_feature)

            total_loss += Loss_bgs

        # if len(comm_rate_list) > 0:
        #     mean_rate = sum(comm_rate_list) / len(comm_rate_list)
        # else:
        #     mean_rate = torch.tensor(0).to(feat_list[0].device)

        return all_agents_sparse_transmitted_data, total_loss
