import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import random

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



class AdvancedCommunication(nn.Module):
    def __init__(self, c_vox, c_feat, c_det, c_semantic=32, lambda_rec=0.5):
        super(AdvancedCommunication, self).__init__()
        # self.channel_request = Channel_Request_Attention(in_planes)
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

    def forward(self, vox_list,feat_list,det_list):
        sparse_vox_out, sparse_feat_out, sparse_det_out = [], [], []
        total_loss = []
        total_communication_volume = []
        for i in range(len(feat_list)):
            utility_loss = None
            reconstruction_loss = None
            vox_i, feat_i, det_i = vox_list[i], feat_list[i], det_list[i]

            num_agents, c_vox, H, W = vox_i.shape
            c_feat = feat_i.shape[1]
            c_det = det_i.shape[1]

            if num_agents <= 1:
                # If no collaborators, append empty tensors to maintain output structure
                device = feat_i.device
                sparse_vox_out.append(torch.empty(0, c_vox, H, W, device=device))
                sparse_feat_out.append(torch.empty(0, c_feat, H, W, device=device))
                sparse_det_out.append(torch.empty(0, c_det, H, W, device=device))
                continue

            # 1. Separate Ego from Collaborators
            ego_vox, collab_vox = vox_i[0:1], vox_i[1:]
            ego_feat, collab_feat = feat_i[0:1], feat_i[1:]
            ego_det, collab_det = det_i[0:1], det_i[1:]
            num_collaborators = collab_vox.shape[0]

            #计算带宽上限
            max_possible_volume = H * W * num_collaborators * (c_vox+c_feat+c_det)
            min_budget = int(0.05 * max_possible_volume)
            max_budget = int(0.80 * max_possible_volume)
            min_budget = max(min_budget, torch.max(self.cost_vector).item())
            if min_budget >= max_budget:
                current_budget = max_budget
            else:
                current_budget = torch.randint(low=min_budget, high=max_budget + 1, size=(1,)).item()

            # 2. Generate Attention Maps (The "Common Language")
            ego_attn_spatial, ego_attn_granularity, ego_attn_semantic = self.attention_generator(ego_vox, ego_feat,
                                                                                                 ego_det)
            collab_attn_spatial, collab_attn_granularity, collab_attn_semantic = self.attention_generator(collab_vox,
                                                                                                          collab_feat,
                                                                                                          collab_det)
            collab_attn_granularity = collab_attn_granularity.expand(-1,-1,H,W)
            # Convert ego's self-attention into a "request" (1 - attention)
            ego_req_spatial = 1.0 - ego_attn_spatial
            ego_req_granularity = 1.0 - ego_attn_granularity

            # 3. Prepare Input for the UtilityNetwork
            # Broadcast ego's requests to match the number of collaborators for concatenation
            ego_req_s_b = ego_req_spatial.expand(num_collaborators, -1, -1, -1)
            ego_req_g_b = ego_req_granularity.expand(num_collaborators, -1, H, W)
            ego_req_sem_b = ego_attn_semantic.expand(num_collaborators, -1, -1, -1)
            print("————————检测维度——————")
            print("ego_req_s_b.shape=",ego_req_s_b.shape)
            print("ego_req_g_b.shape=",ego_req_g_b.shape)
            print("ego_req_sem_b.shape=",ego_req_sem_b.shape)
            print("collab_attn_spatial.shape=",collab_attn_spatial.shape)
            print("collab_attn_granularity.shape=",collab_attn_granularity.shape)
            print("collab_attn_semantic.shape=",collab_attn_semantic.shape)
            # Fuse all available information at inference time
            fused_input = torch.cat([
                ego_req_s_b, ego_req_g_b, ego_req_sem_b,
                collab_attn_spatial, collab_attn_granularity, collab_attn_semantic
            ], dim=1)

            # 4. Predict Utility
            predicted_utility = self.utility_network(fused_input)

            # 5. Compute Loss (only during training)
            # if self.training:
            #     # Use privileged information to calculate the "perfect" GT
            #     utility_gt = self._calculate_marginal_utility_gt(
            #         (ego_vox, ego_feat, ego_det),
            #         (collab_vox, collab_feat, collab_det),
            #         (ego_req_spatial, ego_req_granularity, ego_attn_semantic),
            #         collab_attn_semantic
            #     )
            #     utility_loss = F.mse_loss(predicted_utility, utility_gt)

            utility_gt = self._calculate_marginal_utility_gt(
                (ego_vox, ego_feat, ego_det),
                (collab_vox, collab_feat, collab_det),
                (ego_req_spatial, ego_req_granularity, ego_attn_semantic),
                collab_attn_semantic
            )
            utility_loss = F.mse_loss(predicted_utility, utility_gt)

            # 6. Top-K Selection based on Predicted Utility and Budget
            final_utility = predicted_utility
            best_utility_per_patch, best_granularity_idx = torch.max(final_utility, dim=1)

            # Flatten for efficient sorting
            flat_utility = best_utility_per_patch.flatten()
            flat_costs = self.cost_vector[best_granularity_idx.flatten()]

            # Sort patches by their predicted utility-per-cost
            # Adding a small epsilon to cost to avoid division by zero, though cost should be > 0
            sorted_indices = torch.argsort(flat_utility / (flat_costs + 1e-8), descending=True)

            # Greedily build the transmission mask
            transmission_mask = torch.zeros_like(final_utility, dtype=torch.bool)
            bandwidth_accumulator = 0

            for idx in sorted_indices:
                cost = flat_costs[idx]
                if bandwidth_accumulator + cost > current_budget:
                    continue

                bandwidth_accumulator += cost
                # Convert flat index back to multi-dimensional coordinates
                coords = self.unravel_index_single(idx.item(), best_utility_per_patch.shape)
                collab_idx, h_idx, w_idx = coords
                granularity_to_transmit = best_granularity_idx[coords]
                transmission_mask[collab_idx, granularity_to_transmit, h_idx, w_idx] = True

            sparse_vox_i = collab_vox * transmission_mask[:, 0:1, :, :]
            sparse_feat_i = collab_feat * transmission_mask[:, 1:2, :, :]
            sparse_det_i = collab_det * transmission_mask[:, 2:3, :, :]


            # if self.training:
            #     num_collaborators = collab_vox.shape[0]
            #     # Ego's BEVs need to be broadcasted to match the number of collaborators
            #     ego_vox_b = ego_vox.expand(num_collaborators, -1, -1, -1)
            #     ego_feat_b = ego_feat.expand(num_collaborators, -1, -1, -1)
            #     ego_det_b = ego_det.expand(num_collaborators, -1, -1, -1)
            #
            #     # Reconstruct each granularity
            #     recon_vox = self.decoder_vox(ego_vox_b, sparse_vox_i)
            #     recon_feat = self.decoder_feat(ego_feat_b, sparse_feat_i)
            #     recon_det = self.decoder_det(ego_det_b, sparse_det_i)
            #
            #     # Calculate loss against the ORIGINAL collaborator BEVs
            #     loss_vox = self.reconstruction_loss_fn(recon_vox, collab_vox)
            #     loss_feat = self.reconstruction_loss_fn(recon_feat, collab_feat)
            #     loss_det = self.reconstruction_loss_fn(recon_det, collab_det)
            #
            #     reconstruction_loss = loss_vox + loss_feat + loss_det

            num_collaborators = collab_vox.shape[0]
            # Ego's BEVs need to be broadcasted to match the number of collaborators
            ego_vox_b = ego_vox.expand(num_collaborators, -1, -1, -1)
            ego_feat_b = ego_feat.expand(num_collaborators, -1, -1, -1)
            ego_det_b = ego_det.expand(num_collaborators, -1, -1, -1)

            # Reconstruct each granularity
            recon_vox = self.decoder_vox(ego_vox_b, sparse_vox_i)
            recon_feat = self.decoder_feat(ego_feat_b, sparse_feat_i)
            recon_det = self.decoder_det(ego_det_b, sparse_det_i)

            # Calculate loss against the ORIGINAL collaborator BEVs
            loss_vox = self.reconstruction_loss_fn(recon_vox, collab_vox)
            loss_feat = self.reconstruction_loss_fn(recon_feat, collab_feat)
            loss_det = self.reconstruction_loss_fn(recon_det, collab_det)

            reconstruction_loss = loss_vox + loss_feat + loss_det

            # 把ego的数据拼接回去
            sparse_vox_i = torch.cat((ego_vox, sparse_vox_i), dim=0)
            sparse_feat_i = torch.cat((ego_feat, sparse_feat_i), dim=0)
            sparse_det_i = torch.cat((ego_det, sparse_det_i), dim=0)
            # 7. Apply Mask to Generate Sparse Features
            sparse_vox_out.append(sparse_vox_i)
            sparse_feat_out.append(sparse_feat_i)
            sparse_det_out.append(sparse_det_i)

            #计算通信量
            cost_reshaped = self.cost_vector.view(1, 3, 1, 1)
            volume_map = transmission_mask.float()*cost_reshaped
            total_communication_volume += torch.sum(volume_map)

            # if self.training:
            #     utility_loss = utility_loss if utility_loss is not None else 0
            #     reconstruction_loss = reconstruction_loss if reconstruction_loss is not None else 0
            #
            #     combined_loss = utility_loss + self.lambda_rec * reconstruction_loss
            #     total_loss.append(combined_loss)
            utility_loss = utility_loss if utility_loss is not None else 0
            reconstruction_loss = reconstruction_loss if reconstruction_loss is not None else 0

            combined_loss = utility_loss + self.lambda_rec * reconstruction_loss
            print("combined_loss=", combined_loss)
            total_loss.append(combined_loss)

        # 8. Aggregate batch results and return
        # final_loss = torch.mean(torch.stack(total_loss)) if self.training and total_loss else None
        final_loss = torch.mean(torch.stack(total_loss))

        mean_communication_volume = total_communication_volume/len(vox_list)

        return (torch.cat(sparse_vox_out, dim=0),
                torch.cat(sparse_feat_out, dim=0),
                torch.cat(sparse_det_out, dim=0),
                final_loss,
                mean_communication_volume)