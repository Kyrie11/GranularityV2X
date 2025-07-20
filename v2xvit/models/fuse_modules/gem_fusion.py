import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class GIC_Update_Cell(nn.Module):
    """
    情景感知 GIC 更新单元 (Context-Aware GIC Update Cell).

    该单元接收一个先前的隐藏状态 (代表粗粒度理解), 一个当前的特征图 (代表细粒度细节),
    以及一个顶层的时序先验, 并输出一个更新后的隐藏状态。

    Args:
        c_h (int): 隐藏状态 h_prev 的通道数。
        c_f (int): 当前特征图 f_curr 的通道数。
        c_t (int): 时序先验 h_temporal_prior 的通道数。
        c_out (int): 输出隐藏状态 h_new 的通道数。
    """
    def __init__(self, c_h, c_f, c_t, c_out):
        super(GIC_Update_Cell, self).__init__()
        self.c_h = c_h
        self.c_f = c_f
        self.c_t = c_t
        self.c_out = c_out

        #输入通道总数
        c_in = c_h + c_f + c_t
        #计算更新门和重置门的卷积
        self.conv_gates = nn.Conv2d(c_in, 2*c_out, kernel_size=3, padding=1)
        #计算候选隐藏状态的卷积
        self.conv_candidate = nn.Conv2d(c_out+c_f+c_t, c_out, kernel_size=3, padding=1)

    def forward(self, h_prev, f_curr, h_temporal_prior):
        """
        Args:
            h_prev (torch.Tensor): 前一个隐藏状态 (e.g., H_1)。
            f_curr (torch.Tensor): 当前粒度的特征 (e.g., F_fused_G2)。
            h_temporal_prior (torch.Tensor): 来自顶层GRU的时序先验。

        Returns:
            torch.Tensor: 更新后的隐藏状态 H_new。
        """
        # 沿通道维度拼接所有输入
        combined_input = torch.cat([h_prev, f_curr, h_temporal_prior], dim=1)

        #计算门信号（更新门和重置门）
        gates = self.conv_gates(combined_input)
        update_gate, reset_gate = torch.split(gates, self.c_out, dim=1)
        update_gate = torch.sigmoid(update_gate)
        reset_gate = torch.sigmoid(reset_gate)

        # 计算候选隐藏状态
        # 重置门决定遗忘多少先前的隐藏状态
        candidate_input = torch.cat([reset_gate * h_prev, f_curr, h_temporal_prior], dim=1)
        h_tilde = torch.tanh(self.conv_candidate(candidate_input))

        # 计算新的隐藏状态
        # 更新门决定吸收多少新的候选状态
        h_new = (1 - update_gate) * h_prev + update_gate * h_tilde

        return h_new

class ASM_Gate(nn.Module):
    """
    双模态调制 ASM 门 (Dual-Modulation Agent-Specific Modulation Gate).

    该模块为每个协作者动态生成两个空间调制门 (专属补盲门和共识增强门),
    并以此为依据进行特征融合。

    Args:
        c_in (int): 该粒度特征的通道数。
        c_context (int): 上下文特征的中间通道维度。
    """
    def __init__(self, c_in, c_context=128):
        super(ASM_Gate, self).__init__()

        #生成共享上下文的网络
        self.context_net = nn.Sequential(
            nn.Conv2d(c_in * 2, c_context, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_context),
            nn.ReLU(inplace=True)
        )

        #生成调制门的网络(输入为上下文+特定协作者特征)
        gate_net_input_dim = c_context + c_in
        # self.gate_net = nn.Sequential(
        #     nn.Conv2d(gate_net_input_dim, c_in, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(c_in),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(c_in, c_in, kernel_size=1),
        #     nn.Sigmoid()  # Gate values should be in [0, 1]
        # )
        self.gate_net = nn.Sequential(
            nn.Conv2d(gate_net_input_dim, c_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),
        )

        #两个并行的卷积头，分别输出两种门
        self.exclusive_gate_head = nn.Conv2d(c_in, c_in, kernel_size=1)
        self.common_gate_head = nn.Conv2d(c_in, c_in, kernel_size=1)


    def forward(self, f_ego, f_cavs):
        """
        Args:
            f_ego (torch.Tensor): Ego-Agent 的特征图。
            f_cavs (List[torch.Tensor]): 协作者特征图的列表。

        Returns:
            torch.Tensor: 该粒度下融合后的特征图。
        """
        if not f_cavs:
            return f_ego

        # Aggregate collaborator features
        with torch.no_grad():
            f_cavs_tensor = torch.stack(f_cavs, dim=0)
            f_agg, _ = torch.max(f_cavs_tensor, dim=0)

        # Generate modulation contexts
        context_input = torch.cat([f_ego, f_agg], dim=1)
        context_map = self.context_net(context_input)

        sum_of_exclusive_features = 0
        sum_of_common_features = 0

        # Modulate and sum features from all collaboratorss
        for f_cav in f_cavs:
            gate_input = torch.cat([context_map, f_cav], dim=1)
            gate_features = self.gate_net(gate_input)

            #计算两种门并用Sigmoid激活
            exclusive_gate = torch.sigmoid(self.exclusive_gate_head(gate_features))
            common_gate = torch.sigmoid(self.common_gate_head(gate_features))

            #分别对特征进行 调制并累加
            sum_of_exclusive_features += exclusive_gate * f_cav
            sum_of_common_features += common_gate * f_cav

        f_fused = f_ego + sum_of_exclusive_features + sum_of_common_features

        return f_fused

class GIC(nn.Module):
    """
    情景感知粒度交互级联 (Context-Aware Granularity Interaction Cascade).

    该模块按粗到细的顺序处理来自 ASM Gate 的三张融合图, 并在每一步
    利用一个顶层的时序先验来指导融合。

    Args:
        c_g1, c_g2, c_g3 (int): 三个粒度级别的输入通道维度。
        c_temporal (int): 时序先验的通道维度。
        c_fusion (int): 隐藏状态和最终输出的统一通道维度。
    """
    def __init__(self, c_g1, c_g2, c_g3, c_temporal, c_fusion):
        super(GIC, self).__init__()

        # 将最粗粒度特征投影到融合空间的 1x1 卷积
        self.initial_proj = nn.Conv2d(c_g1, c_fusion, kernel_size=1)

        # Update cell for integrating G2 (medium) details
        self.gic_update_2 = GIC_Update_Cell(c_h=c_fusion, c_f=c_g2, c_t=c_temporal, c_out=c_fusion)

        # Update cell for integrating G3 (fine) details
        self.gic_update_3 = GIC_Update_Cell(c_h=c_fusion, c_f=c_g3, c_t=c_temporal, c_out=c_fusion)

    def forward(self, f_g1, f_g2, f_g3, h_temporal_prior):
        """
        Args:
            f_g1, f_g2, f_g3 (torch.Tensor): G1, G2, G3 的融合特征图。
            h_temporal_prior (torch.Tensor): 来自顶层 GRU 的时序先验。

        Returns:
            torch.Tensor: 最终完全融合的 BEV 特征图。
        """
        # 用最粗糙的上下文初始化隐藏状态
        h1 = self.initial_proj(f_g1)

        # 级联步骤 1: 用 h1 和时序先验来指导 G2 特征的融合
        h2 = self.gic_update_2(h1, f_g2, h_temporal_prior)

        # 级联步骤 2: 用 h2 和时序先验来指导 G3 特征的融合
        h3 = self.gic_update_3(h2, f_g3, h_temporal_prior)

        return h3

class GEM_Fusion(nn.Module):
    """
    完整的 GEM-Fusion

    该模块负责编排两阶段的融合过程:
    1. 双模态 ASM_Gate: 处理每个粒度级别上的智能体异构性。
    2. 情景感知 GIC: 在时序先验的指导下处理粒度间的互补性。

    Args:
        c_g1, c_g2, c_g3 (int): 三个粒度的输入通道维度。
        c_temporal (int): 顶层时序先验的通道维度。
        c_fusion (int): GIC 模块和最终输出的特征通道维度。
        c_asm_context (int): ASM_Gate 内部上下文的通道维度。
    """
    def __init__(self, c_g1, c_g2, c_g3, c_temporal, c_fusion, c_asm_context=64):
        super(GEM_Fusion, self).__init__()

        # --- 智能体特异性调制阶段 (Agent-Specific Modulation) ---
        self.asm_g1 = ASM_Gate(c_in=c_g1, c_context=c_asm_context)
        self.asm_g2 = ASM_Gate(c_in=c_g2, c_context=c_asm_context)
        self.asm_g3 = ASM_Gate(c_in=c_g3, c_context=c_asm_context)

        # --- 粒度交互级联阶段 (Granularity Interaction Cascade) ---
        self.gic = GIC(c_g1=c_g1, c_g2=c_g2, c_g3=c_g3, c_temporal=c_temporal, c_fusion=c_fusion)

    def forward(self, agent_data: List[Dict[str, torch.Tensor]], h_temporal_prior) -> torch.Tensor:
        """
        Args:
            agent_data (List[Dict[str, torch.Tensor]]): A list of dictionaries.
                The first element is the ego-agent, followed by collaborators.
                Each dictionary has keys 'G1', 'G2', 'G3' mapping to feature tensors.
                Example: [
                    {'G1': ego_g1, 'G2': ego_g2, 'G3': ego_g3}, # Ego
                    {'G1': cav1_g1, 'G2': cav1_g2, 'G3': cav1_g3}, # CAV 1
                    ...
                ]

        Returns:
            torch.Tensor: The final fused BEV feature map of shape [B, c_fusion, H, W].
        """
        ego_data = agent_data[0]
        cav_data_list = agent_data[1:]

        #ASM Fusion for each Granularity
        f_g1_fused = self.asm_g1(f_ego=ego_data['G1'], f_cavs=[cav['G1'] for cav in cav_data_list])

        f_g2_fused = self.asm_g2(f_ego=ego_data['G2'], f_cavs=[cav['G2'] for cav in cav_data_list])

        f_g3_fused = self.asm_g3(f_ego=ego_data['G3'], f_cavs=[cav['G3'] for cav in cav_data_list])

        final_bev_feature = self.gic(f_g1_fused, f_g2_fused, f_g3_fused, h_temporal_prior)

        return final_bev_feature

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size//2

        self.conv = nn.Conv2d(self.input_dim + self.hidden_dim, 2 * self.hidden_dim, self.kernel_size, padding=self.padding, bias=bias)
        self.conv_candidate = nn.Conv2d(self.input_dim+ + self.hidden_dim, self.hidden_dim, self.kernel_size, padding=self.padding, bias=bias)

    def forward(self, input_tensor,cur_state):
        h_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined_candidate = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_candidate(combined_candidate)
        h_next = torch.tanh(cc_cnm)

        h_new = (1-update_gate) * h_cur + update_gate * h_next
        return h_new

