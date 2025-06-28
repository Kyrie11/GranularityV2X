import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from v2xvit.models.comm_modules.basic_model import *
from typing import List, Dict

#----ego自我增强------
class AttentionMapHead(nn.Module):
    """
        A simple but effective head to decode a feature map into a single-channel attention map.
        It takes the high-dimensional context and projects it to a probability-like map.
    """

    def __init__(self, d_model: int, d_hidden: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(d_model, d_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid()  # Output a map with values in [0, 1]
        )

    def forward(self, context_feature: torch.Tensor) -> torch.Tensor:
        return self.head(context_feature)

class AgentSelfEnhancement(nn.Module):
    """
       Implements the per-agent feature enhancement logic using historical data.
       As per the design, this should be applied to all agents in the scene.
    """

    def __init__(self,
                 d_model: int = 64,
                 num_history_frames: int = 3,
                 nhead_transformer: int = 8,
                 num_transformer_layers: int = 2):
        super().__init__()

        self.d_model = d_model
        self.num_total_frames = num_history_frames + 1  # history + current
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_transformer,
            batch_first=True
        )
        self.spatio_temporal_transformer = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_layers
        )
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, self.num_total_frames, d_model))
        self.occlusion_head = AttentionMapHead(d_model)
        self.abnormal_head = AttentionMapHead(d_model)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(d_model + 2, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature_history: torch.Tensor):
        """
                Args:
                    feature_history (torch.Tensor): A batch of agent histories.
                                                    Shape: [N, T, C, H, W] where
                                                    N = number of agents
                                                    T = number of frames (current + history)
                                                    C = d_model
                                                    H, W = spatial dimensions

                Returns:
                    torch.Tensor: The enhanced feature_bev for the current frame for all agents.
                                  Shape: [N, C, H, W]
        """
        #目前我只传入了ego-agent的bev
        feature_history_tensor = torch.stack(feature_history, dim=1)
        B, T, C, H, W = feature_history_tensor.shape
        # 存储原始的当前帧特征 (列表中的第一个元素)，用于最后的残差连接
        current_feature = feature_history[0].clone()  # 形状: [B, C, H, W]
        # 为Transformer准备输入
        # [B, T, C, H, W] -> [B, H, W, T, C]
        temporal_context = feature_history_tensor.permute(0, 3, 4, 1, 2)
        # 添加时间位置编码 (会自动广播)
        print("temporal_context.shape=", temporal_context.shape)
        print("self.temporal_pos_embedding.unsqueeze(1).unsqueeze(1).shape=",self.temporal_pos_embedding.unsqueeze(1).unsqueeze(1).shape)
        temporal_context = temporal_context + self.temporal_pos_embedding.unsqueeze(1).unsqueeze(1)

        # 展平空间维度，送入Transformer
        # [B*H*W, T, C]
        transformer_input = temporal_context.reshape(B * H * W, T, C)
        transformer_output = self.spatio_temporal_transformer(transformer_input)

        # 解码遮挡图和异常图
        # [B*H*W, T, C] -> [B, H, W, T, C]
        context_vector = transformer_output.reshape(B, H, W, T, C)
        # 取出当前帧 (t=0) 对应的上下文向量，并转换回BEV格式 [B, C, H, W]
        current_context = context_vector[:, :, :, 0, :].permute(0, 3, 1, 2)

        occlusion_map = self.occlusion_head(current_context)  # [B, 1, H, W]
        abnormal_map = self.abnormal_head(current_context)  # [B, 1, H, W]

        # 融合与增强
        # 拼接原始特征和两个指导图
        fusion_input = torch.cat([current_feature, occlusion_map, abnormal_map], dim=1)

        # 卷积层学习如何利用指导图来增强特征
        enhancement = self.fusion_conv(fusion_input)

        # 残差连接：在原始特征基础上添加增强信息
        enhanced_feature = current_feature + enhancement
        return enhanced_feature, occlusion_map, abnormal_map


class GranularityEncoder(nn.Module):
    """
        1. 粒度编码器 (Granularity Encoder)
        设计出发点：解决输入数据异构性的核心模块。为三种物理意义完全不同的数据
        (vox, feat, det) 设计独立的投影器，将它们映射到统一的特征空间(d_model)。
        利用您提出的“空间互斥性”，通过逐元素相加进行高效、无参数的合并。
    """
    def __init__(self, c_vox, c_feat, c_det, d_model):
        super().__init__()
        self.c_vox, self.c_feat, self.c_det = c_vox, c_feat, c_det
        # 为每种粒度构建独立的投影器 (1x1卷积是最高效的特征空间变换方式)
        self.vox_projector = self._build_projector(c_vox, d_model)
        self.feat_projector = self._build_projector(c_feat, d_model)
        self.det_projector = self._build_projector(c_det, d_model)

    def _build_projector(self, c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, sparse_granu: torch.Tensor) -> torch.Tensor:
        """
                Args:
                    sparse_granu (Tensor): 稀疏的多粒度BEV图, [B, C_vox+C_feat+C_det, H, W]

                Returns:
                    Tensor: 统一后的密集特征图, [B, d_model, H, W]
        """

        # 将输入按通道切分成三种粒度
        vox_part = sparse_granu[:, :self.c_vox, ...]
        feat_part = sparse_granu[:, self.c_vox: self.c_vox + self.c_feat, ...]
        det_part = sparse_granu[:, self.c_vox + self.c_feat:, ...]

        # 独立投影
        proj_vox = self.vox_projector(vox_part)
        proj_feat = self.feat_projector(feat_part)
        proj_det = self.det_projector(det_part)

        # 利用空间互斥性，逐元素相加进行“缝合”
        unified_feature = proj_vox + proj_feat + proj_det
        return unified_feature


class TokenCrossAttention(nn.Module):
    """(无需修改) 负责计算ego-Q与collab-KV的匹配，并提炼出令牌 h_m。"""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        # ... (代码与上一版本完全相同)
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=n_heads,
                                               batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ego_feat_map: torch.Tensor, collab_feat_map: torch.Tensor) -> torch.Tensor:
        B, C, H, W = ego_feat_map.shape
        ego_seq = ego_feat_map.flatten(2).permute(0, 2, 1)
        collab_seq = collab_feat_map.flatten(2).permute(0, 2, 1)
        # 2. 【核心修改】创建全局 Ego 查询
        # q 的形状是 [B_ego, H*W, C]，我们只取 ego (B_ego=1)
        # .mean(dim=1, keepdim=True) -> [1, 1, C]
        q_global = ego_seq.mean(dim=1, keepdim=True)
        attn_output, _ = self.attention(query=q_global,
                                        key=collab_seq,
                                        value=collab_seq)

        fused_global_feat = attn_output.permute(0, 2, 1).view(1, C, 1, 1)

        # .expand(...) 将其广播到原始地图大小
        fused_map = fused_global_feat.expand(-1, -1, H, W)

        return fused_map


class TokenBevDecoder(nn.Module):
    """(无需修改) 负责将融合后的令牌 h_m 解码回BEV空间。"""

    def __init__(self, d_model: int, h: int, w: int):
        super().__init__()
        # ... (代码与上一版本完全相同)
        self.d_model = d_model
        self.h = h
        self.w = w
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True))

    def forward(self, h_m: torch.Tensor) -> torch.Tensor:
        print("h_m.shape=", h_m.shape)
        H_fused_content = self.decoder(h_m)
        return H_fused_content


# ======================================================================
# =========          这是为您重构的最终主网络模块          =========
# ======================================================================
class MultiGranularityFusionNet(nn.Module):
    """
    顶层编排网络。
    输入为包含场景中所有agent（ego+collaborators）的多粒度BEV张量。
    """

    def __init__(self, args: Dict):
        super().__init__()
        # 参数定义与之前相同
        self.c_vox = args['C_V']
        self.c_feat = args['C_F']
        self.c_det = args['C_D']
        self.d_model = args['d_model']
        self.bev_h = args['bev_h']
        self.bev_w = args['bev_w']

        # 实例化子模块 (与之前相同)
        self.granularity_encoder = GranularityEncoder(self.c_vox, self.c_feat, self.c_det, self.d_model)
        self.token_attention = TokenCrossAttention(self.d_model, n_heads=args.get('n_heads', 8))
        self.token_decoder = TokenBevDecoder(self.d_model, self.bev_h, self.bev_w)
        self.final_fusion_layer = nn.Conv2d(self.d_model, self.d_model, kernel_size=1)

    def forward(self, vox_bev: torch.Tensor, feat_bev: torch.Tensor, det_bev: torch.Tensor):
        """
        Args:
            vox_bev (Tensor): 场景中所有agent的体素BEV图。
                              形状: [B, C_vox, H, W]
            feat_bev (Tensor): 场景中所有agent的特征BEV图。
                               形状: [B, C_feat, H, W]。
                               其中feat_bev[0]应为增强后的ego-agent特征。
            det_bev (Tensor): 场景中所有agent的检测BEV图。
                              形状: [B, C_det, H, W]

        (B 是agent的数量, B=0是ego-agent, B=1...N-1是协作agents)

        Returns:
            Tensor: 最终融合后的特征图, [1, d_model, H, W] (只输出ego-agent的结果)
        """
        # --- 1. 数据分离：从输入张量中分离出Ego和Collaborators ---
        num_agents = vox_bev.shape[0]



        # Ego-agent的数据 (取第0个元素，并用[0:1]保持维度)
        # 核心假设：feat_bev[0]是已经过AgentSelfEnhancement模块增强的特征
        ego_vox = vox_bev[0:1]
        ego_feat = feat_bev[0:1]  # [1, C_feat, H, W]
        ego_det = det_bev[0:1]
        ego_granularity_cat = torch.cat([ego_vox, ego_feat, ego_det], dim=1)
        ego_encoded_feat = self.granularity_encoder(ego_granularity_cat)

        # 检查是否存在协作agents
        if num_agents <= 1:
            # 如果没有协作agent，则直接使用ego自身的特征
            final_feature = self.final_fusion_layer(ego_encoded_feat)
            print("当只有一个agent时，final_feature.shape=", final_feature.shape)
            return final_feature

        # Collaborator agents的数据
        collab_vox_bevs = vox_bev[1:]
        collab_feat_bevs = feat_bev[1:]
        collab_det_bevs = det_bev[1:]
        num_collaborators = collab_vox_bevs.shape[0]

        # --- 2. 循环处理每个协作agent，提取协作令牌 ---
        collaborator_tokens = []
        for i in range(num_collaborators):
            # 2a. 为当前协作agent准备稀疏多粒度输入
            # 从集合张量中切片出第i个协作agent的数据
            collab_vox_i = collab_vox_bevs[i:i + 1]  # [1, C_vox, H, W]
            collab_feat_i = collab_feat_bevs[i:i + 1]  # [1, C_feat, H, W]
            collab_det_i = collab_det_bevs[i:i + 1]  # [1, C_det, H, W]

            # 沿通道维度拼接成 GranularityEncoder 所需的格式
            sparse_granu_i = torch.cat([collab_vox_i, collab_feat_i, collab_det_i], dim=1)

            # 2b. 将其编码为统一特征图
            unified_collab_feat = self.granularity_encoder(sparse_granu_i)
            print("unified_collab_feat.shape=", unified_collab_feat.shape)
            print("ego_enhanced_feat.shape=", ego_encoded_feat.shape)

            # 2c. 通过交叉注意力，计算匹配令牌 h_m
            # ego_enhanced_feat作为Query, unified_collab_feat作为Key和Value
            h_m = self.token_attention(ego_encoded_feat, unified_collab_feat)
            collaborator_tokens.append(h_m)

        # --- 3. 聚合所有协作令牌并解码 ---
        # 将令牌列表堆叠成 [Num_collab, 1, C], 然后求和/平均
            fused_token = torch.stack(collaborator_tokens).sum(dim=0)  # [1, C]

        # 将聚合后的令牌解码为BEV特征图 H_fused_content
        H_fused_content = self.token_decoder(fused_token)

        # --- 4. 最终融合 ---
        # 与Ego-agent的原始增强特征进行最终融合 (残差连接)
        final_feature = ego_encoded_feat + H_fused_content
        final_feature = self.final_fusion_layer(final_feature)

        return final_feature
