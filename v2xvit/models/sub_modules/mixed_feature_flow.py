import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True, use_relu=True):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SingleInputFlowPredictor(nn.Module):
    """一个简单的网络，接收单一的融合特征图，预测光流 + 可信度"""

    def __init__(self, in_channels, middle_channels=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1)
        # 【修改】输出3个通道: dx, dy, scale
        self.conv_out = nn.Conv2d(middle_channels, 3, kernel_size=1)

    def forward(self, fused_context):
        x = F.relu(self.conv_in(fused_context))
        x = x + F.relu(self.conv_res(x))  # 加入残差连接
        outputs = self.conv_out(x)

        # 分离 flow 和 scale
        flow = outputs[:, 0:2, :, :]  # 取前两个通道作为光流
        scale = torch.sigmoid(outputs[:, 2:, :, :])  # 取最后一个通道，用sigmoid约束到0-1

        return flow, scale

class WarpingLayer(nn.Module):
    """
        使用光流场对特征图进行变形(warp)的模块。
        它接收一个特征图和一个光流场，输出变形后的特征图。
        这个模块的实现与输入特征图的通道数无关。
    """
    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow):
        B,C,H,W = x.shape
        # 1. 创建一个标准化的网格 (identity grid)
        # torch.meshgrid 创建了两个张量，分别代表每个像素的y和x坐标
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype),
                                        torch.arange(W, device=x.device, dtype=x.dtype),
                                        indexing='ij')
        # 将x和y坐标堆叠起来，形成一个形状为 [H, W, 2] 的坐标网格
        # 这个网格代表了每个输出像素应该从输入图像的哪个位置采样
        grid = torch.stack((grid_x, grid_y), 2)

        # 扩展到batch维度
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        # 2. 计算新的采样坐标
        # F.grid_sample 需要的是绝对采样坐标，所以我们将原始网格坐标与光流位移相加
        # flow 的形状是 [B, 2, H, W]，需要permute到 [B, H, W, 2] 来匹配grid
        new_grid = grid + flow.permute(0, 2, 3, 1)

        # 3. 将采样坐标归一化到 [-1, 1] 的范围
        # 这是 F.grid_sample 的要求。它期望坐标在 [-1, 1] 的范围内，
        # 其中 (-1, -1) 是左上角，(1, 1) 是右下角。
        new_grid[..., 0] = 2 * new_grid[..., 0] / (W - 1) - 1
        new_grid[..., 1] = 2 * new_grid[..., 1] / (H - 1) - 1

        # 4. 执行采样
        # F.grid_sample 会根据 new_grid 中的坐标，在输入特征图 x 上进行双线性插值采样
        # padding_mode='zeros' 表示超出边界的区域用0填充
        # align_corners=True 是一个重要的参数，确保了坐标变换的精确性
        warped_x = F.grid_sample(x, new_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped_x


class ContextFusionMotionPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.short_frames = args.get("short_frames", 3)
        self.long_interval = args.get("long_interval", 3)
        c_vox = args.get("C_V")
        c_feat = args.get("C_F")
        c_det = args.get("C_D")
        in_channels = c_vox + c_feat + c_det
        self.gru_hidden_channels = args.get("gru_dim", 32)
        delay_emb_dim = args.get("delay_dim", 16)
        self.max_delay = args.get("max_delay", 3)

        self.long_term_encoder = ConvGRU(in_channels, self.gru_hidden_channels, kernel_size=(3, 3))
        self.short_term_encoder = ConvGRU(in_channels, self.gru_hidden_channels, kernel_size=(3, 3))
        self.delay_embedding = nn.Embedding(self.max_delay + 1, delay_emb_dim)

        fusion_input_channels = self.gru_hidden_channels * 2 + delay_emb_dim
        self.context_fusion_net = nn.Sequential(
            nn.Conv2d(fusion_input_channels, self.gru_hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.gru_hidden_channels, self.gru_hidden_channels, kernel_size=1)
        )

        self.final_flow_predictor = SingleInputFlowPredictor(in_channels=self.gru_hidden_channels)
        self.warping_layer = WarpingLayer()
        self.refinement_net = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self,vox_list, feat_list, det_list, record_len):
        delay = random.randint(0, min(self.max_delay, len(vox_list)))

        fused_his = [torch.cat([vox_list[i], feat_list[i], det_list[i]], dim=1) for i in range(len(vox_list))]


        #我们要时延预测的帧
        fused_to_compensate = fused_his[delay]

        B,C,H,W = fused_to_compensate.shape

        device = fused_to_compensate.device

        #确保有足够的历史帧
        if delay+1 >= len(vox_list):
            # 如果没有历史（delay是最后一帧），则无法预测运动，直接返回原始帧
            # 同样返回零光流和中性scale，以保持输出格式一致
            return fused_to_compensate, torch.zeros(B, 2, H, W, device=device), torch.ones(B, 1, H, W, device=device)

        #提取delay后的帧作为历史
        fused_his_sequence = fused_his[delay+1:]

        if self.short_frames <= len(fused_his_sequence):
            #分割长短期历史
            short_term_his_fused = fused_his_sequence[:self.short_frames]
        else:
            short_term_his_fused = fused_his_sequence

        long_term_his_fused = fused_his_sequence[::self.long_interval]

        #远近时间倒排序
        if len(short_term_his_fused) > 0:
            his_for_short_gru = short_term_his_fused[::-1]
        else:
            his_for_short_gru = None
        if len(long_term_his_fused) > 0:
            his_for_long_gru = long_term_his_fused[::-1]
        else:
            his_for_long_gru = None

        # ---  编码长短期上下文 ---

        if his_for_short_gru is None and his_for_long_gru is None:
            # 没有历史信息，无法进行运动预测，直接返回
            return None, None, None
        # 长期上下文
        # long_term_context = self.init_hidden(B, (H, W), device, gru_hidden_channels)
        if his_for_long_gru is not None:
            long_term_context = self.long_term_encoder(his_for_long_gru)
        else:
            long_term_context = torch.zeros(B, self.gru_hidden_channels, H, W,
                                            device=fused_to_compensate.device,
                                            dtype=fused_to_compensate.dtype)

        # 短期上下文
        # short_term_context = self.init_hidden(B, (H, W), device, gru_hidden_channels)
        if his_for_short_gru is not None:
            short_term_context = self.short_term_encoder(his_for_short_gru)
        else:
            short_term_context = torch.zeros(B, self.gru_hidden_channels, H, W,
                                            device=fused_to_compensate.device,
                                            dtype=fused_to_compensate.dtype)

        # --- 3. 融合上下文 ---
        delay_tensor = torch.full((B,), delay, dtype=torch.long, device=vox_list[0].device)
        print("delay.shape=", delay_tensor.shape)
        delay_emb = self.delay_embedding(delay_tensor)
        print("delay_emb.shape=", delay_emb.shape)
        delay_map = delay_emb.view(B, -1, 1, 1).expand(B, -1, H, W)
        print("delay_map.shape=", delay_map.shape)
        fusion_input = torch.cat([long_term_context, short_term_context, delay_map], dim=1)
        print("fusion_input.shape=", fusion_input.shape)
        final_fused_context = self.context_fusion_net(fusion_input)

        # --- 4. 预测运动并外推 ---
        predicted_flow_at_delay, scale = self.final_flow_predictor(final_fused_context)

        delay_expanded = delay_tensor.float().view(B, 1, 1, 1)
        extrapolated_flow = predicted_flow_at_delay * delay_expanded

        # --- 5. 补偿与精炼 ---
        warped_feature = self.warping_layer(fused_to_compensate, extrapolated_flow)
        scaled_warped_feature = scale * warped_feature

        refined_prediction = self.refinement_net(scaled_warped_feature)

        return refined_prediction, extrapolated_flow, scale


    def init_hidden(self, batch_size, image_size, device, hidden_channels):
        H, W = image_size
        return torch.zeros(batch_size, hidden_channels, H, W, device=device)



class BEVFlowPredictor(nn.Module):
    def __init__(self, D_in, D_hidden):
        super().__init__()
        # D_current_feat: Channels of the concatenated F_vox_t0, F_feat_t0, F_det_t0

        # U-Net like structure
        self.encoder1 = BasicConvBlock(D_in, D_hidden)
        self.encoder2 = BasicConvBlock(D_hidden, D_hidden * 2, stride=2)
        self.encoder3 = BasicConvBlock(D_hidden * 2, D_hidden * 4, stride=2)

        self.decoder2 = BasicConvBlock(D_hidden * 4 + D_hidden * 2, D_hidden * 2)  # Skip conn
        self.upsample2 = nn.ConvTranspose2d(D_hidden * 2, D_hidden * 2, kernel_size=2, stride=2)

        self.decoder1 = BasicConvBlock(D_hidden * 2 + D_hidden, D_hidden)  # Skip conn
        self.upsample1 = nn.ConvTranspose2d(D_hidden, D_hidden, kernel_size=2, stride=2)

        # self.flow_head = nn.Conv2d(D_hidden, num_flow_channels, kernel_size=3, padding=1)
        self.object_flow_head = nn.Sequential(
            nn.Conv2d(D_hidden, D_hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(D_hidden // 2, 2, kernel_size=1)  # Output 2 channels for (dx, dy)
        )

        self.residual_flow_head = nn.Sequential(
            nn.Conv2d(D_hidden, D_hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(D_hidden // 2, 2, kernel_size=1)  # Output 2 channels for (dx, dy)
        )

        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(D_hidden, D_hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(D_hidden // 2, 1, kernel_size=1),  # Single channel for uncertainty scalar
            nn.Sigmoid()
        )

        for m in [self.object_flow_head, self.residual_flow_head]:
            for layer in m:
                if isinstance(layer, nn.Conv2d):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, input):
        # current_F_cat_t0: [B, C_V+C_F+C_D, H, W]
        # short_term_ctx: [B, D_short_ctx, H, W] or None
        # long_term_ctx: [B, D_long_ctx, H, W] or None (can be global vector tiled too)
        # delay_scalar_map: [B, 1, H, W]

        x = torch.cat(input, dim=1)
        enc1 = self.encoder1(x)  # D_hidden
        enc2 = self.encoder2(enc1)  # D_hidden*2
        enc3 = self.encoder3(enc2)  # D_hidden*4

        dec2 = self.upsample2(enc3)  # D_hidden*2
        dec2 = torch.cat([dec2, enc2], dim=1)  # Skip
        dec2 = self.decoder2(dec2)  # D_hidden*2

        dec1 = self.upsample1(dec2)  # D_hidden
        dec1 = torch.cat([dec1, enc1], dim=1)  # Skip
        dec1 = self.decoder1(dec1)  # D_hidden

        object_flow = self.object_flow_head(dec1)
        residual_flow = self.residual_flow_head(dec1)
        # uncertainty_map = self.uncertainty_head(dec1)



        return object_flow, residual_flow


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if isinstance(kernel_size, int):
            # 如果 kernel_size 是一个整数 (例如 3), padding 也应该是一个整数 (例如 1)
            padding = kernel_size // 2
        elif isinstance(kernel_size, tuple):
            # 如果 kernel_size 是一个元组 (例如 (3, 3) 或 (3, 5)),
            # padding 应该是一个对每个维度都计算的元组 (例如 (1, 1) 或 (1, 2))
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv_gates = nn.Conv2d(self.input_dim + self.hidden_dim, 2 * self.hidden_dim, kernel_size, padding=padding)
        self.conv_can = nn.Conv2d(self.input_dim + self.hidden_dim, self.hidden_dim, kernel_size, padding=padding)

    def forward(self, input_tensor, h_cur):
        # input_tensor: [B, C_in, H, W]
        # h_cur: [B, C_hidden, H, W]
        combined = torch.cat([input_tensor, h_cur], dim=1)

        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.sigmoid(gates).chunk(2, dim=1)

        combined_reset = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_can = torch.tanh(self.conv_can(combined_reset))

        h_next = (1 - update_gate) * h_cur + update_gate * cc_can
        return h_next

class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvGRUCell(input_dim, hidden_dim, kernel_size)

    def forward(self, input_sequence, hidden_state=None):
        b, c, h, w = input_sequence[0].shape
        if hidden_state is None:
            hidden_state = torch.zeros(b, self.hidden_dim, h, w, device=input_sequence[0].device)
        # 循环处理序列中的每一个时间步
        for input_tensor in input_sequence:
            hidden_state = self.cell(input_tensor, hidden_state)
        return hidden_state

class HierarchicalMotionPredictor(nn.Module):
    def __init__(self, args_mgdc_bev):
        super(HierarchicalMotionPredictor, self).__init__()
        self.c_vox = args_mgdc_bev.get('C_V', 10)
        self.c_feat = args_mgdc_bev.get('C_F', 64)
        self.c_det = args_mgdc_bev.get('C_D', 16)
        self.D_hidden = args_mgdc_bev.get('D_hidden', 128)

        self.delay_steps = args_mgdc_bev.get('delay', 3)
        self.delay_embedding_dim = args_mgdc_bev.get('delay_embedding_dim', 32)

        self.feature_encoder = nn.Sequential(
            nn.Conv2d(self.c_vox + self.c_feat + self.c_det, self.D_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # NEW: 定义ConvGRU单元用于融合时序信息
        # short_term和long_term将使用同一个GRU，但以不同的隐藏状态输入
        self.temporal_fusion_gru = ConvGRUCell(self.D_hidden, self.D_hidden, kernel_size=3)

        # NEW: 定义时间嵌入层
        self.delay_embedding = nn.Embedding(self.delay_steps+1, self.delay_embedding_dim)

        self.flow_predictor = BEVFlowPredictor(
            D_in=self.D_hidden + self.delay_embedding_dim,
            D_hidden=self.D_hidden
        )

    def forward(self, short_his_cat_tensors, long_his_cat_tensors, delay):
        #len(short_history_list)=帧数
        #len(short_history_list[0])=batch_size
        B,_,H,W = short_his_cat_tensors[0].shape

        # NEW: 初始化GRU的隐藏状态
        h_short = torch.zeros(B, self.temporal_fusion_gru.hidden_dim, H, W, device=short_his_cat_tensors[0].device)
        h_long = torch.zeros(B, self.temporal_fusion_gru.hidden_dim, H, W, device=short_his_cat_tensors[0].device)

        # NEW: 按时间顺序（从最远到最近）迭代处理历史帧
        # 处理长期历史
        for his_frame in long_his_cat_tensors:
            encoded_frame = self.feature_encoder(his_frame)
            h_long = self.temporal_fusion_gru(encoded_frame, h_long)

        # 注意：这里我们用h_long的最终状态作为h_short的初始状态，实现长短期信息的传递
        h_short = h_long
        for his_frame in short_his_cat_tensors:
            encoded_frame = self.feature_encoder(his_frame)
            h_short = self.temporal_fusion_gru(encoded_frame, h_short)

        delay = delay.long()
        delay_emb_vec = self.delay_embedding(delay)
        # 扩展成空间特征图
        delay_map = delay_emb_vec.view(B, self.delay_embedding_dim, 1, 1).expand(B, self.delay_embedding_dim, H, W)

        final_predictor_input = torch.cat([h_short, delay_map], dim=1)

        return self.flow_predictor(final_predictor_input)


class FeatureWarper(nn.Module):
    def get_grid(self, B, H, W, device):
        shifts_x = torch.arange(0, W, 1, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, H, 1, dtype=torch.float32, device=device)
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')  # Use 'ij' for H,W order
        grid_dst = torch.stack((shifts_x, shifts_y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]
        return grid_dst

    def flow_warp(self, feats, flow, delay_step):  # flow is per unit time
        # feats: [B, C, H, W], flow: [B, 2, H, W] (dx, dy)
        B, _, H, W = feats.shape
        grid_dst = self.get_grid(B, H, W, feats.device)  # Base grid for destination

        # Total displacement: flow * delay
        total_displacement = flow * delay_step  # scaled_delay is (t-t0)

        sampling_grid_pixels = grid_dst + total_displacement  # If flow is where points *come from*

        # Normalize sampling_grid_pixels to [-1, 1]
        sampling_grid_normalized_x = (sampling_grid_pixels[:, 0, :, :] / ((W - 1) / 2)) - 1
        sampling_grid_normalized_y = (sampling_grid_pixels[:, 1, :, :] / ((H - 1) / 2)) - 1

        sampling_grid_normalized = torch.stack(
            [sampling_grid_normalized_x, sampling_grid_normalized_y], dim=-1
        )  # [B, H, W, 2] (x_coords, y_coords) for F.grid_sample

        warped_feats = F.grid_sample(
            feats, sampling_grid_normalized, mode="bilinear", padding_mode="border", align_corners=True
        )
        return warped_feats

class MultiGranularityBevDelayCompensation(nn.Module):
    def __init__(self, args_mgdc_bev):
        super(MultiGranularityBevDelayCompensation, self).__init__()
        self.c_vox = args_mgdc_bev.get('C_V', 10)
        self.c_feat = args_mgdc_bev.get('C_F', 64)
        self.c_det = args_mgdc_bev.get('C_D', 16)
        self.short_frames = args_mgdc_bev.get("short_frames", 3)
        self.long_gaps = args_mgdc_bev.get("long_gaps", 3)

        self.motion_predictor = HierarchicalMotionPredictor(args_mgdc_bev)

        self.feature_warper = FeatureWarper()


    def _prepare_history_tensor_list(self, his_data_per_granularity):
        """
        Args:
        history_data_per_granularity (list): e.g., short_his_vox.
            结构: [[batch1_t0, batch2_t0, ...], [batch1_t1, batch2_t1, ...], ...]
        Return:
            list[torch.Tensor]:
            一个Tensor列表，每个Tensor代表一个时间步，形状为 [A, C, H, W]。
            e.g., [history_t0_tensor, history_t1_tensor, ...]
        """
        num_frames = len(his_data_per_granularity)
        output_tensor_list = []

        for i in range(num_frames):
            # history_data_per_granularity[i] 是一个包含该时间步所有batch数据的列表
            batched_tensor_at_t = torch.cat(his_data_per_granularity[i], dim=0)
            output_tensor_list.append(batched_tensor_at_t)
        return output_tensor_list

    def forward(self, fused_his, delay_steps, record_len):
        # len(record_len) = batch_size(B)
        # record_len[i]=cav_num_i
        #current_F_fused = [vox_bev, feat_bev, det_bev]
        #vox_bev:[_,C_vox,H,W]
        #feat_bev:[_,C_feat,H,W]
        #det_bev:[_,C_det,H,W]
        #short_his_list = [short_his_vox, short_his_feat, short_his_det]
        #long_his_list = [long_his_vox, long_his_feat, long_his_det]
        #len(short_his_vox)=num_short_frames
        #len(long_his_vox)=num_long_frames
        #short_his_vox[0] = [batch1_vox_bev, batch2_vox_bev...]  len(short_his_vox[0])=batch_size
        #batchi_vox_bev.shape = cav_num, C_vox, H, W (cav_num指每个batch中所有agent的数量)
        his_vox, his_feat, his_det = fused_his
        B, _, H, W = his_vox[0].shape
        short_his_vox = his_vox[delay_steps:delay_steps+self.short_frames]
        short_his_feat = his_feat[delay_steps:delay_steps+self.short_frames]
        short_his_det = his_det[delay_steps:delay_steps+self.short_frames]



        long_his_vox = his_vox[delay_steps::self.long_gaps]
        long_his_feat = his_feat[delay_steps::self.long_gaps]
        long_his_det = his_det[delay_steps::self.long_gaps]
        print("long_his_vox[0].shape=",long_his_vox[0].shape)
        print("long_his_feat[0].shape=",long_his_feat[0].shape)
        print("long_his_det[0].shape=",long_his_det[0].shape)
        num_short_frames = len(short_his_vox)
        num_long_frames = len(long_his_vox)

        # 将每个时间步的三个粒度特征拼接在一起
        short_his_cat_tensors = []
        for i in range(num_short_frames):
            short_his_cat_tensors.append(torch.cat([short_his_vox[i],short_his_feat[i],short_his_det[i]], dim=1))
        long_his_cat_tensors = []
        for i in range(num_long_frames):
            long_his_cat_tensors.append(torch.cat([long_his_vox[i],long_his_feat[i],long_his_det[i]], dim=1))

        # short_his_cat_tensors = [torch.cat([short_his_vox[i], short_his_feat[i], short_his_det[i]], dim=1) for i in range(num_short_frames)]
        # long_his_cat_tensors = [torch.cat([long_his_vox[i], long_his_feat[i], long_his_det[i]], dim=1) for i in range(num_long_frames)]
        print("long_his_cat_tensors[0].shape=", long_his_cat_tensors[0].shape)

        delay_tensor = torch.full((B,), delay_steps, dtype=torch.long, device=his_vox[0].device)
        #进行运动预测
        object_flow, residual_flow = self.motion_predictor(short_his_cat_tensors, long_his_cat_tensors, delay_tensor)

        final_flow = object_flow + residual_flow

        predicted_F_vox = self.feature_warper.flow_warp(short_his_vox[0], final_flow, delay_steps)
        predicted_F_fea = self.feature_warper.flow_warp(short_his_feat[0], final_flow, delay_steps)
        predicted_F_det = self.feature_warper.flow_warp(short_his_det[0], final_flow, delay_steps)

        return predicted_F_vox,predicted_F_fea,predicted_F_det







