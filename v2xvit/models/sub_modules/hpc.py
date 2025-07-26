import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


#========================
#短期历史编码器
#========================
class ShortTermEncoder(nn.Module):
    '''
    使用GRU处理连续的短期历史帧
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.gru = nn.GRU(input_size=input_dim,
        #                   hidden_dim=hidden_dim,
        #                   num_layers=1,
        #                   batch_first=True)
        self.conv3d = nn.Sequential(
            #Input: [N,C_in,T,H,W]
            nn.Conv3d(input_dim, 32, kernel_size=(3,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, output_dim, kernel_size=(1,3,3), padding=(0,1,1)),
            #output: [N,C_out,1,H,W]
        )

    def forward(self, short_term_history: torch.Tensor) -> torch.Tensor:
        # short_term_history shape: [N, T, C_in, H, W]
        # Conv3d expects: [N, C_in, T, H, W]
        short_term_history = short_term_history.permute(0,2,1,3,4)

        #可能T_short <3，进行动态填充
        T = short_term_history.shape[2]
        if T < 3:
            padding_needed = 3 - T
            padding_tuple = (0, 0, 0, 0, 0, padding_needed)
            short_term_history = F.pad(short_term_history, padding_tuple, mode="replicate")

        # 由于我们的T_short=3, kernel_size=3, padding=0, 时间维度会被压缩为1
        encoded_map = self.conv3d(short_term_history)

        # Squeeze the temporal dimension -> [N, C_out, H, W]
        return encoded_map.squeeze(2)


#=======================
#长期历史编码器
#=======================
class TemporalPositionalEncoder(nn.Module):
    """为Transformer添加时间位置编码"""
    def __init__(self, d_model, max_time_delta: float = 5000.0):
        super().__init__()
        self.d_model = d_model
        # 创建一个足够大的除数，用于缩放时间增量
        # log-scale for frequency division
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_time_delta) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, x: torch.Tensor, time_deltas: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入特征。Shape: (N, T, d_model)
            time_deltas (torch.Tensor): 每一帧相对于当前帧的时间差（例如 [0, -300, -600]）。
                                        Shape: (T,)

        Returns:
            torch.Tensor: 添加了时间编码的特征。Shape: (N, T, d_model)
        """
        # (1, T, 1) * (d_model/2) -> (1, T, d_model/2)
        time_deltas = time_deltas.unsqueeze(0).unsqueeze(-1)
        angles = time_deltas * self.div_term
        pe = torch.zeros(1, time_deltas.size(1), self.d_model, device=x.device)
        pe[0, :, 0::2] = torch.sin(angles)
        pe[0, :, 1::2] = torch.cos(angles)
        # pe shape: (1, T, d_model), can be broadcasted to (N, T, d_model)
        return x + pe

class LongTermEncoder(nn.Module):
    """
    使用Transformer处理稀疏采样的长期历史帧特征，并嵌入实际的时间间隔信息。
    """
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = TemporalPositionalEncoder(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim,
                                                   nhead=num_heads,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)

    def forward(self, long_term_history, time_intervals):
        """
        Args:
            long_term_history (torch.Tensor): 长期历史特征序列。
                                             Shape: (N, T_long, C, H, W)
            time_intervals (torch.Tensor): 长期历史中每帧的时间间隔(ms)。Shape: (T_long,)

        Returns:
            torch.Tensor: 编码后的长期上下文特征。Shape: (N, model_dim)
        """
        N, T_long, C, H, W = long_term_history.shape
        flattened_features = long_term_history.view(N, T_long, -1)
        projected_features = self.input_proj(flattened_features)

        #添加基于实际时间间隔的位置编码
        features_with_pos_enc = self.pos_encoder(projected_features, time_intervals)
        encoded_features = self.transformer_encoder(features_with_pos_enc)

        #使用最后一个时间步的输出作为长期历史的总结
        return encoded_features[:, 0, :]


#====================
#长短期信息Encoder
#====================
class TemporalContextEncoder(nn.Module):
    def __init__(self,
                 total_input_channels,
                 s_ctx_channels,
                 l_ctx_dim,
                 feature_size,
                 transformer_heads = 8,
                 transformer_layers = 2):
        super().__init__()
        feature_flat_dim = total_input_channels * feature_size[0] * feature_size[1] #特征展平

        self.short_term_encoder = ShortTermEncoder(
            input_dim=total_input_channels,
            output_dim=s_ctx_channels
        )

        self.long_term_encoder = LongTermEncoder(
            input_dim=feature_flat_dim,
            model_dim=l_ctx_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers
        )

    def forward(self, short_term_his, long_term_his, long_term_interval):
        '''
        :param short_term_his: [N,T_short,C,H,W]
        :param long_term_his: [N,T_long,C,H,W]
        :param long_term_interval: [N,1]
        :return:
        '''
        # short_term_tensor = torch.stack(short_term_his, dim=1) #[N,T_short,C,H,W]
        # long_term_tensor = torch.stack(long_term_his, dim=1) #[N,T_long,C,H,W]
        interval_tensor = torch.tensor(long_term_interval, device=short_term_his.device).float()

        #编码非对称上下文
        s_ctx = self.short_term_encoder(short_term_his) #shape : [N,s_ctx_channels,H,W]
        print("短期上下文的shape是：", s_ctx.shape)
        l_ctx = self.long_term_encoder(long_term_his, interval_tensor) #[N, l_ctx_dim]
        print("长期上下文的shape是：", l_ctx.shape)

        return {
            "short_term_context": s_ctx,
            "long_term_context": l_ctx,
        }

#==================
#动态门控开关
#==================
class DynamicGatingModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.delay_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid() #使用Sigmoid确保权重在[0,1]
        )

    def forward(self, delay: torch.Tensor):
        return self.delay_mlp(delay.float() / 1000.0) #将ms归一化到秒左右的范围

#=====================================
#分层运动预测头
#=====================================
class HierarchicalPredictionHead(nn.Module):
    """
    从融合时空上下文中分层预测运动场。
    - O_v: 主要的、密集的基础运动场。
    - O_f, O_r: 稀疏的残差运动修正。
    - S: 置信度调制图。
    """

    def __init__(self, input_channels, output_shape):
        """
        Args:
            feature_dim (int): 输入的融合特征维度。
            output_shape (Tuple[int, int]): 输出的BEV图谱的空间尺寸 (H, W)。
        """
        super().__init__()
        self.output_shape = output_shape

        self.decoder_trunk = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True)
        )

        # 独立的预测头
        # 头 for O_v: Base Motion Field (dx, dy)
        self.head_v = nn.Linear(64, 2, 1)

        # 头 for O_f: Feature Residual (dx, dy)
        # 我们在这里不施加硬性约束，而是依赖于外部的稀疏性损失
        self.head_f = nn.Linear(64, 2, 1)

        # 头 for O_r: Result Residual (dx, dy)
        self.head_r = nn.Linear(64, 2, 1)

        # 头 for S: Confidence Scaler
        # 使用 Sigmoid 激活函数将置信度缩放到 (0, 1) 区间，这符合其物理意义
        self.head_s = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, fused_context):
        x = self.decoder_trunk(fused_context)

        return {
            "base_motion_field": self.head_v(x),
            "feature_residual_field": self.head_f(x),
            "result_residual_field": self.head_r(x),
            "confidence_scaler": self.head_s(x)
        }

#==================================
#将时间上下文解码
#==================================
class ContextExtrapolator(nn.Module):
    '''
    运行在Ego-agent端来解码上下文并且外推
    '''
    def __init__(self,
                 s_ctx_channels: int,
                 l_ctx_dim: int,
                 fusion_dim: int,
                 bev_feature_channels: int,
                 physical_info_channels: int,
                 result_map_channels: int,
                 feature_size: Tuple[int, int]):
        super().__init__()
        self.feature_size = feature_size
        self.bev_channels = bev_feature_channels
        self.physical_channels = physical_info_channels
        self.result_channels = result_map_channels

        self.gating_module = DynamicGatingModule(fusion_dim)

        #1x1卷积yong'yu处理和对齐上下文通道
        self.s_ctx_processor = nn.Conv2d(s_ctx_channels, fusion_dim, 1)
        self.l_ctx_processor = nn.Conv2d(l_ctx_dim, fusion_dim, 1)

        #预测头
        self.prediction_head = HierarchicalPredictionHead(fusion_dim, feature_size)

        #Warping 辅助网格
        H, W = self.feature_size
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        identity_grid = torch.stack((grid_x, grid_y), 2).float()
        self.register_buffer('identity_grid', identity_grid.unsqueeze(0), persistent=False)

    def warp(self, feature_map, motion_field):
        N,_,H,W = feature_map.shape
        norm_identity_grid = self.identity_grid * 2.0 / torch.tensor([W-1, H-1], device=self.identity_grid.device) - 1.0
        motion_field_transposed = motion_field.permute(0, 2, 3, 1)
        norm_motion_field = motion_field_transposed * 2.0 / torch.tensor([W - 1, H - 1], device=motion_field.device)
        new_grid = norm_identity_grid + norm_motion_field
        return F.grid_sample(feature_map, new_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    def forward(self,
                s_ctx: torch.Tensor,
                l_ctx: torch.Tensor,
                delayed_g1_frame: torch.Tensor,
                delayed_g2_frame: torch.Tensor,
                delayed_g3_frame: torch.Tensor,
                delays_ms: List[float]
               ) -> Dict[str, torch.Tensor]:

        N,_,H,W = s_ctx.shape
        device = s_ctx.shape
        delay_tensor = torch.tensor(delays_ms, device=device).unsqueeze(1)

        # --- 1. 延迟感知的上下文融合 ---
        g = self.gating_module(delay_tensor).view(N, 1, 1, 1)  # Shape: [N, 1, 1, 1] for broadcasting

        # 处理短期上下文图
        s_ctx_processed = self.s_ctx_processor(s_ctx)

        # 广播并处理长期上下文向量
        l_ctx_broadcasted = l_ctx.unsqueeze(-1).unsqueeze(-1).expand(N, -1, H, W)
        l_ctx_processed = self.l_ctx_processor(l_ctx_broadcasted)

        # 动态门控融合
        fused_context_map = (1 - g) * s_ctx_processed + g * l_ctx_processed

        # --- 2. 基于融合上下文进行预测 ---
        params = self.prediction_head(fused_context_map)
        O_v = params["base_motion_field"]
        O_f = params["feature_residual_field"]
        O_r = params["result_residual_field"]
        S = params["confidence_scaler"]

        #---- 3.warp和scale
        F_v_warped = self._warp(delayed_g1_frame, O_v)
        F_f_warped = self._warp(delayed_g2_frame, O_v + O_f)
        F_r_warped = self._warp(delayed_g3_frame, O_v + O_r)

        # --- 4. 返回完整结果 ---
        return {
            "predicted_g1": S * F_v_warped,
            "predicted_g2": S * F_f_warped,
            "predicted_g3": S * F_r_warped,
            "feature_residual_field": O_f,  # for sparsity loss
            "result_residual_field": O_r,  # for sparsity loss
        }


