import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import random

from v2xvit.loss.contrastive_sparsity_loss import ContrastiveSparsityLoss
from v2xvit.loss.recon_loss import ReconstructionLoss


def _calculate_contribution(ref_bevs, contrib_bevs) -> torch.Tensor:
    """Helper to calculate the cosine similarity contribution of 'contrib' to 'ref'."""
    ref_vox, ref_feat, ref_det = ref_bevs
    contrib_vox, contrib_feat, contrib_det = contrib_bevs

    N, _, H, W = ref_vox.shape
    C_vox, C_feat, C_det = ref_vox.shape[1], ref_feat.shape[1], ref_det.shape[1]
    C_total = C_vox + C_feat + C_det

    # Create flattened reference vector
    ref_full = torch.cat(ref_bevs, dim=1).permute(0, 2, 3, 1).reshape(-1, C_total)

    # Create padded, flattened contribution vectors
    v_c = contrib_vox.permute(0, 2, 3, 1).reshape(-1, C_vox)
    v_c_pad = F.pad(v_c, (0, C_feat + C_det))

    f_c = contrib_feat.permute(0, 2, 3, 1).reshape(-1, C_feat)
    f_c_pad = F.pad(f_c, (C_vox, C_det))

    d_c = contrib_det.permute(0, 2, 3, 1).reshape(-1, C_det)
    d_c_pad = F.pad(d_c, (C_vox + C_feat, 0))

    # Calculate cosine similarity for each granularity
    sim_vox = F.cosine_similarity(v_c_pad, ref_full, dim=1)
    sim_feat = F.cosine_similarity(f_c_pad, ref_full, dim=1)
    sim_det = F.cosine_similarity(d_c_pad, ref_full, dim=1)

    # Stack and reshape back to [N, 3, H, W]
    contrib_flat = torch.stack([sim_vox, sim_feat, sim_det], dim=0)
    contribution = contrib_flat.reshape(3, N, H, W).permute(1, 0, 2, 3)

    # Normalize to [0,1] range
    return (contribution + 1) / 2

def _calculate_marginal_utility_gt(ego_bevs, collab_bevs, ego_requests, collab_semantic_attn) -> torch.Tensor:
    """Calculates the ideal 'marginal utility' GT using privileged information."""
    ego_vox, ego_feat, ego_det = ego_bevs
    collab_vox, collab_feat, collab_det = collab_bevs
    ego_req_spatial, ego_req_granularity, ego_req_semantic = ego_requests

    num_collaborators = collab_vox.shape[0]
    cost_vector = torch.tensor([ego_vox.shape[1], ego_feat.shape[1], ego_det.shape[1]], device=ego_vox.device)

    # Broadcast ego data to match collaborator batch size
    ego_vox_b = ego_vox.expand_as(collab_vox)
    ego_feat_b = ego_feat.expand_as(collab_feat)
    ego_det_b = ego_det.expand_as(collab_det)

    # 1. Create the 'ideal' fused BEV from the God's-eye view
    fused_vox = torch.maximum(ego_vox_b, collab_vox)
    fused_feat = torch.maximum(ego_feat_b, collab_feat)
    fused_det = torch.maximum(ego_det_b, collab_det)

    # 2. Calculate the collaborator's marginal potential w.r.t. the ideal BEV
    marginal_potential = _calculate_contribution(
        (fused_vox, fused_feat, fused_det),
        (collab_vox, collab_feat, collab_det)
    )

    # 3. Calculate semantic match score (as before)
    ego_req_sem_b = ego_req_semantic.expand_as(collab_semantic_attn)
    semantic_match = F.cosine_similarity(ego_req_sem_b, collab_semantic_attn, dim=1)
    semantic_match = ((semantic_match + 1) / 2).unsqueeze(1)  # Normalize to [0,1]

    # 4. Modulate by Ego's requests
    # Broadcast requests to match shape for element-wise multiplication
    ego_req_s_b = ego_req_spatial.expand_as(marginal_potential[:, 0:1, :, :])
    ego_req_g_b = ego_req_granularity.expand(num_collaborators, -1, -1, -1)

    absolute_utility = marginal_potential * ego_req_s_b * ego_req_g_b * semantic_match

    # 5.  GT is utility per unit of cost
    utility_gt = absolute_utility / cost_vector.view(1, 3, 1, 1)

    return utility_gt.detach()


class ReconstructionDecoder(nn.Module):
    """
    Fuses Ego's own BEV with the sparse BEV received from a collaborator,
    and attempts to reconstruct the collaborator's original, dense BEV.
    """

    def __init__(self, c_in_ego: int, c_in_sparse: int, c_out: int):
        super().__init__()
        # Input channels are the concatenation of ego's features and sparse features
        c_in = c_in_ego + c_in_sparse

        self.net = nn.Sequential(
            nn.Conv2d(c_in, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output channels should match the original feature's channels
            nn.Conv2d(64, c_out, kernel_size=1)
        )

    def forward(self, ego_bev: torch.Tensor, sparse_collab_bev: torch.Tensor) -> torch.Tensor:
        # Concatenate along the channel dimension
        fused_input = torch.cat([ego_bev, sparse_collab_bev], dim=1)
        reconstructed_bev = self.net(fused_input)
        return reconstructed_bev


class AttentionGenerator(nn.Module):
    """
        为单个Agent，从其多粒度BEV输入中生成三种Attention。
    """

    def __init__(self, c_vox: int, c_feat: int, c_det: int, c_semantic: int):
        super().__init__()
        c_total = c_vox + c_feat + c_det

        # 1. 空间Attention生成器
        self.spatial_attn_net = nn.Sequential(
            nn.Conv2d(c_total, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()  # 输出归一化到 [0, 1]
        )

        # 2. 粒度Attention生成器 (使用全局平均池化)
        self.granularity_attn_net = nn.Sequential(
            nn.Linear(c_total, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),  # 输出vox, feat, det三个粒度的重要性分数
            nn.Softmax(dim=-1)  # 使用Softmax让三种粒度竞争，总和为1
        )

        # 3. 语义Attention生成器 (主要依赖信息最丰富的feat_bev)
        self.semantic_attn_net = nn.Sequential(
            nn.Conv2d(c_feat, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, c_semantic, kernel_size=1)
            # 输出c'维的语义向量，不加激活，保留原始数值空间
        )



    def forward(self, vox_bev: torch.Tensor, feat_bev: torch.Tensor, det_bev: torch.Tensor):
        """
        Args:
            vox_bev (Tensor): 单个agent的体素BEV, [1, C_vox, H, W]
            feat_bev (Tensor): 单个agent的特征BEV, [1, C_feat, H, W]
            det_bev (Tensor): 单个agent的检测BEV, [1, C_det, H, W]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: 空间、粒度、语义Attention
        """
        # 准备空间attention的输入
        print("检查维度-----")
        print("vox_bev.shape=", vox_bev.shape)
        print("feat_bev.shape=", feat_bev.shape)
        print("det_bev.shape=", det_bev.shape)
        full_bev = torch.cat([vox_bev, feat_bev, det_bev], dim=1)

        # --- 计算空间Attention ---
        # [1, 1, H, W]
        spatial_attention = self.spatial_attn_net(full_bev)
        # --- 计算粒度Attention ---
        # 全局平均池化
        pooled_vox = F.adaptive_avg_pool2d(vox_bev, (1, 1)).flatten(1)
        pooled_feat = F.adaptive_avg_pool2d(feat_bev, (1, 1)).flatten(1)
        pooled_det = F.adaptive_avg_pool2d(det_bev, (1, 1)).flatten(1)
        pooled_full = torch.cat([pooled_vox, pooled_feat, pooled_det], dim=1)
        # [1, 3] -> [1, 3, 1, 1] 以便广播
        granularity_attention = self.granularity_attn_net(pooled_full).unsqueeze(-1).unsqueeze(-1)

        # --- 计算语义Attention ---
        # [1, c_semantic, H, W]
        semantic_attention = self.semantic_attn_net(feat_bev)
        return spatial_attention, granularity_attention, semantic_attention


class UtilityNetwork(nn.Module):
    """
        Learns to predict the final marginal utility.
        Input: A fusion of Ego's requests and Collaborator's self-attention.
        Output: A dense utility map predicting the value of transmitting each data chunk.
        """

    def __init__(self, c_semantic: int):
        super().__init__()
        # Input channels: spatial(1) + granularity(3) + semantic(C) for both ego and collab
        c_in = 2 * (1 + 3 + c_semantic)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=1),  # Output 3 channels for 3 granularities' utilities
            nn.ReLU()  # Utility must be non-negative
        )

    def forward(self, fused_attention_maps: torch.Tensor) -> torch.Tensor:
        return self.net(fused_attention_maps)

class QueryGenerator(nn.Module):
    def __init__(self, feature_channels, query_key_dim):
        super(QueryGenerator, self).__init__()

        self.query_projector = nn.Conv2d(
            in_channels=feature_channels,
            out_channels=query_key_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, feature_ego):
        q_ego = self.query_projector(feature_ego)
        return q_ego

class Channel_Request_Attention(nn.Module):
    def __init__(self, feature_channels, ratio=16):
        super(Channel_Request_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels//ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(feature_channels//ratio, feature_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class KeyGenerator(nn.Module):
    def __init__(self, g1_channels, g2_channels, g3_channels, query_key_dim):
        super(KeyGenerator, self).__init__()
        self.key_projector_g1 = nn.Conv2d(g1_channels, query_key_dim, 1, bias=False)
        self.key_projector_g2 = nn.Conv2d(g2_channels, query_key_dim, 1, bias=False)
        self.key_projector_g3 = nn.Conv2d(g3_channels, query_key_dim, 1, bias=False)

    def forward(self, g1_cav, g2_cav, g3_cav):
        k_g1 = self.key_projector_g1(g1_cav)
        k_g2 = self.key_projector_g2(g2_cav)
        k_g3 = self.key_projector_g3(g3_cav)

        return k_g1, k_g2, k_g3

class SemanticDemandAttention(nn.Module):
    def __init__(self, c_g1, c_g2, c_g3, mid_channels=64):
        super(SemanticDemandAttention, self).__init__()

        total_in_channels = c_g1 + c_g2 + c_g3
        self.fusion_network = nn.Sequential(
            nn.Conv2d(total_in_channels, mid_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.demand_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            # Depthwise
            nn.Conv2d(mid_channels, 3, kernel_size=1, bias=False),  # Pointwise, 3 for V, F, R
            nn.Sigmoid()  # 将每个需求分数归一化到[0, 1]
        )

    def forward(self, g1_data, g2_data, g3_data):
        combined_feature = torch.cat([g1_data, g2_data, g3_data], dim=1)
        fused_feature = self.fusion_network(combined_feature)
        demand_profile = self.demand_head(fused_feature)
        return demand_profile

class AdvancedCommunication(nn.Module):
    def __init__(self, c_vox, c_feat, c_det, c_semantic=32, lambda_rec=0.5):
        super(AdvancedCommunication, self).__init__()
        # self.channel_request = Channel_Request_Attention(in_planes)
        self.query_generator = QueryGenerator(c_feat, 16)
        self.channel_query = Channel_Request_Attention(c_feat, 16)
        self.key_generator = KeyGenerator(c_vox, c_feat, c_det, 16)
        #效益网络
        self.utility_network =UtilityNetwork(c_semantic)

        # 注册成本向量为buffer
        self.register_buffer('cost_vector', torch.tensor([c_vox, c_feat, c_det], dtype=torch.float32))

        self.recon_loss = ReconstructionLoss(90)

        self.attention_generator = AttentionGenerator(c_vox, c_feat, c_det, c_semantic)

        self.decoder_vox = ReconstructionDecoder(c_in_ego=c_vox, c_in_sparse=c_vox, c_out=c_vox)
        self.decoder_feat = ReconstructionDecoder(c_in_ego=c_feat, c_in_sparse=c_feat, c_out=c_feat)
        self.decoder_det = ReconstructionDecoder(c_in_ego=c_det, c_in_sparse=c_det, c_out=c_det)

        # --- NEW: Hyperparameter to balance the losses ---
        self.lambda_rec = lambda_rec
        # Using L1 Loss is often better for image-to-image tasks as it's less blurry
        self.reconstruction_loss_fn = nn.L1Loss()

        self.demand_analyzer = SemanticDemandAttention(8,64,8)
        self.fusion_conv = nn.Sequential(nn.Conv2d(6, 3, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True))
        self.thre = 0.01
        self.contrastive_sparsity_loss = ContrastiveSparsityLoss(8,64,8)

    def unravel_index_single(self, flat_index, shape):
        coords = []
        # 我们需要从内向外计算，所以反转形状
        for dim in reversed(shape):
            coords.append(flat_index % dim)
            flat_index = flat_index // dim
        # 因为是从内向外计算的，所以结果列表也需要反转
        return tuple(coords[::-1])

    def _calculate_marginal_utility_gt(self, ego_bevs, collab_bevs, ego_requests, collab_semantic_attn) -> torch.Tensor:
        """Calculates the ideal 'marginal utility' GT using privileged information."""
        """Calculates the ideal 'marginal utility' GT using privileged information."""
        ego_vox, ego_feat, ego_det = ego_bevs
        collab_vox, collab_feat, collab_det = collab_bevs
        ego_req_spatial, ego_req_granularity, ego_req_semantic = ego_requests

        num_collaborators = collab_vox.shape[0]
        cost_vector = torch.tensor([ego_vox.shape[1], ego_feat.shape[1], ego_det.shape[1]], device=ego_vox.device)

        # Broadcast ego data to match collaborator batch size
        ego_vox_b = ego_vox.expand_as(collab_vox)
        ego_feat_b = ego_feat.expand_as(collab_feat)
        ego_det_b = ego_det.expand_as(collab_det)

        # 1. Create the 'ideal' fused BEV from the God's-eye view
        fused_vox = torch.maximum(ego_vox_b, collab_vox)
        fused_feat = torch.maximum(ego_feat_b, collab_feat)
        fused_det = torch.maximum(ego_det_b, collab_det)

        # 2. Calculate the collaborator's marginal potential w.r.t. the ideal BEV
        marginal_potential = _calculate_contribution(
            (fused_vox, fused_feat, fused_det),
            (collab_vox, collab_feat, collab_det)
        )

        # 3. Calculate semantic match score (as before)
        ego_req_sem_b = ego_req_semantic.expand_as(collab_semantic_attn)
        semantic_match = F.cosine_similarity(ego_req_sem_b, collab_semantic_attn, dim=1)
        semantic_match = ((semantic_match + 1) / 2).unsqueeze(1)  # Normalize to [0,1]

        # 4. Modulate by Ego's requests
        # Broadcast requests to match shape for element-wise multiplication
        ego_req_s_b = ego_req_spatial.expand_as(marginal_potential[:, 0:1, :, :])
        ego_req_g_b = ego_req_granularity.expand(num_collaborators, -1, -1, -1)

        absolute_utility = marginal_potential * ego_req_s_b * ego_req_g_b * semantic_match

        # 5.  GT is utility per unit of cost
        utility_gt = absolute_utility / cost_vector.view(1, 3, 1, 1)

        return utility_gt.detach()

    def forward(self, g1_list, g2_list, g3_list):
        batch_size = len(g1_list)
        device = g1_list[0].device
        sparse_g1_out, sparse_g2_out, sparse_g3_out = [], [], []
        decision_mask_list = []
        total_commu_volume = []
        for b in range(batch_size):
            batch_g1, batch_g2, batch_g3 = g1_list[b], g2_list[b], g3_list[b]
            num_agents, c_g1, H, W = batch_g1.shape
            c_g3 = batch_g3.shape[1]

            if num_agents <= 1:
                # If no collaborators, append empty tensors to maintain output structure
                sparse_g1_out.append(batch_g1)
                sparse_g2_out.append(batch_g2)
                sparse_g3_out.append(batch_g3)
                continue

            demand_profiles = self.demand_analyzer(batch_g1, batch_g2, batch_g3)

            ego_need_profile = 1 - demand_profiles[0:1]

            combined_input = torch.cat([ego_need_profile.expand(num_agents-1, -1,-1,-1), demand_profiles[1:]], dim=1)
            utility_profiles = self.fusion_conv(combined_input)

            g1_cost = 8
            g2_cost = 64
            g3_cost = 8
            costs = torch.tensor([g1_cost, g2_cost, g3_cost], device=device).view(1,3,1,1)

            alpha_g1 = 0.01
            alpha_g2 = 0.01
            alpha_g3 = 0.01
            alphas = torch.tensor([alpha_g1, alpha_g2, alpha_g3], device=device).view(1,3,1,1)

            net_utilities = utility_profiles - alphas * costs
            max_net_utility, best_granularity_idx = torch.max(net_utilities, dim=1) #两个shape都是[N-1,H,W]
            decision_map = best_granularity_idx + 1
            # print("decision_map.shape=", decision_map.shape)
            decision_map[max_net_utility < self.thre] = 0 #[0:不通信，1:g1, 2:g2, 3:g3] [N-1,H,W]
            decision_map = decision_map.to(device)
            decision_mask_list.append(decision_map)
            # decision_map[max_net_utility < self.thre] = 0


            g1_decision_mask = decision_map == 1 #[N,H,W]
            g2_decision_mask = decision_map == 2
            g3_decision_mask = decision_map == 3

            g1_decision_mask = g1_decision_mask.unsqueeze(1)
            g2_decision_mask = g2_decision_mask.unsqueeze(1)
            g3_decision_mask = g3_decision_mask.unsqueeze(1)

            sparse_g1 = g1_decision_mask * batch_g1[1:]
            sparse_g2 = g2_decision_mask * batch_g2[1:]
            sparse_g3 = g3_decision_mask * batch_g3[1:]

            sparse_g1 = torch.cat([batch_g1[0:1], sparse_g1], dim=0)
            sparse_g2 = torch.cat([batch_g2[0:1], sparse_g2], dim=0)
            sparse_g3 = torch.cat([batch_g3[0:1], sparse_g3], dim=0)

            sparse_g1_out.append(sparse_g1)
            sparse_g2_out.append(sparse_g2)
            sparse_g3_out.append(sparse_g3)

            commu_volume = g1_decision_mask.sum() + g2_decision_mask.sum() + g3_decision_mask.sum()

            total_commu_volume.append(commu_volume)
        sparse_data = [sparse_g1_out, sparse_g2_out, sparse_g3_out]
        dense_data = [g1_list, g2_list, g3_list]
        loss = self.contrastive_sparsity_loss(sparse_data, dense_data, decision_mask_list)
        if len(total_commu_volume) < 1:
            mean_communication_volume = 0
        else:
            mean_communication_volume = torch.mean(torch.stack(total_commu_volume).float())
        return (torch.cat(sparse_g1_out, dim=0),
                torch.cat(sparse_g2_out, dim=0),
                torch.cat(sparse_g3_out, dim=0),
                loss,
                mean_communication_volume)