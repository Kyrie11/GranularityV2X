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
        padding = kernel_size // 2

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
        self.delay_embedding = nn.Embedding(self.delay_steps, self.delay_embedding_dim)

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







