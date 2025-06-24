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

class TemporalContextEncoder(nn.Module):
    """Encodes a sequence of BEV features (single granularity)"""
    def __init__(self, C_in_granularity, D_out_context, num_history_frames):
        super().__init__()
        if num_history_frames>0:
            self.conv3d = nn.Conv3d(C_in_granularity, D_out_context,
                                    kernel_size=(min(num_history_frames, 3),3,3),# Kernel depth <= num_frames
                                    padding=(min(num_history_frames, 3)//2 if min(num_history_frames, 3)>1 else 0,1,1))
            self.norm = nn.BatchNorm3d(D_out_context)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool3d((1,None,None)) #Pool along time dim
        else: #No history
            self.dummy_param = nn.Parameter(torch.empty(0))

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, history_bev_list):
        #输入: history_sequence_for_batch, list of Tensors, each [N_agents, C, H, W]
        try:
            stacked_his = torch.stack(history_bev_list,dim=2) # [N_agents, C, num_frames, H,W]
        except Exception as e:
            raise e

        x = self.relu(self.norm(self.conv3d(stacked_his)))
        context_feat_map = self.pool(x).squeeze(2)

        return context_feat_map


class ShortTermMotionEncoder(nn.Module):
    def __init__(self, C_total_hist, D_short_out, num_short_frames=3, method='conv_cat'):
        super().__init__()
        self.method = method
        self.num_short_frames = num_short_frames
        if method=="conv_cat":
            self.conv_layers = nn.Sequential(
                BasicConvBlock((num_short_frames+1) * C_total_hist, D_short_out * 2),
                BasicConvBlock(D_short_out*2, D_short_out)
            )
        elif method=="conv3d":
            self.conv3d_encoder = TemporalContextEncoder(C_total_hist, D_short_out, num_short_frames+1) #+1 for current frame

    def forward(self, short_history, compensated_bev):
        if self.method == 'conv_cat':
            padded_history = self._pad_history(short_history, self.num_short_frames)
            inputs = [compensated_bev] + padded_history
            inputs = torch.cat(inputs, dim=1)
            short_ctx = self.conv_layers(inputs)
        elif self.method=='conv3d':
            inputs = short_history + [compensated_bev]
            short_ctx = self.conv3d_encoder(inputs)
        return short_ctx

class LongTermTrendEncoder(nn.Module):
    def __init__(self, C_total_hist, D_long_out, num_long_frames, output_type='map'):
        super().__init__()
        self.output_type = output_type
        self.conv3d_encoder = TemporalContextEncoder(C_total_hist, D_long_out, num_long_frames)
        if output_type=="vector":
            self.global_pool = nn.AdaptiveAvgPool2d(1) #To make it a vector from map

    def forward(self, long_history):
        long_context_map = self.conv3d_encoder(long_history)
        if self.output_type=="vector":
            long_context_map = self.global_pool(long_context_map).squeeze(-1).squeeze(-1)
        return long_context_map


class BEVFlowPredictor(nn.Module):
    def __init__(self, D_input_combined, D_hidden):
        super().__init__()
        # D_current_feat: Channels of the concatenated F_vox_t0, F_feat_t0, F_det_t0

        # U-Net like structure
        self.encoder1 = BasicConvBlock(D_input_combined, D_hidden)
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

    def forward(self, compensated_bev, short_term_ctx_batch, long_term_ctx_batch, delay_scalar_map):
        # current_F_cat_t0: [B, C_V+C_F+C_D, H, W]
        # short_term_ctx: [B, D_short_ctx, H, W] or None
        # long_term_ctx: [B, D_long_ctx, H, W] or None (can be global vector tiled too)
        # delay_scalar_map: [B, 1, H, W]
        input_list = []
        input_list.append(compensated_bev)
        input_list.append(short_term_ctx_batch)
        input_list.append(long_term_ctx_batch)
        input_list.append(delay_scalar_map)

        x = torch.cat(input_list, dim=1)
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
        uncertainty_map = self.uncertainty_head(dec1)



        return object_flow, residual_flow, uncertainty_map


class HierarchicalMotionPredictor(nn.Module):
    def __init__(self, C_total_bev, C_total_hist, D_short_ctx, D_long_ctx_map, D_long_ctx_vec,
                 num_short_frames, num_long_frames, flow_predictor_hidden_dim, long_ctx_type="map"):
        super().__init__()
        self.delay_embedding = nn.Embedding(delay_steps+1, delay_embedding_dim)
        self.short_term_encoder = ShortTermMotionEncoder(C_total_hist, D_short_ctx, num_short_frames, method="conv3d")
        self.long_ctx_type = long_ctx_type
        D_long_encoder_out = D_long_ctx_map if long_ctx_type == 'map' else D_long_ctx_vec
        self.long_term_encoder = LongTermTrendEncoder(C_total_hist, D_long_encoder_out, num_long_frames, output_type=long_ctx_type)

        if long_ctx_type == "vector":
            self.z_long_map_proj_channels = 32
            self.z_long_mlp = nn.Linear(D_long_ctx_vec, self.z_long_map_proj_channels)
        else:
            self.z_long_map_proj_channels = D_long_ctx_map

        current_feat_dim_for_flow_pred = C_total_bev

        D_input_combined = C_total_bev + D_short_ctx + self.z_long_map_proj_channels + delay_embedding_dim
        self.flow_predictor = BEVFlowPredictor(
            D_input_combined,
            flow_predictor_hidden_dim
        )

    def forward(self, compensated_bev, short_his_cat_tensors, long_his_cat_tensors, delay_step):
        #len(short_history_list)=帧数
        #len(short_history_list[0])=batch_size
        B,_,H,W = short_his_cat_tensors[0].shape
        delay_idx_tensor = torch.full((B, 1), float(delay_step), device=compensated_bev.device, dtype=torch.long)
        delay_embedded_vec = self.delay_embedding(delay_idx_tensor.squeeze(1))
        delay_map = delay_embedded_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        h_short_ctx = self.short_term_encoder(short_his_cat_tensors, compensated_bev)
        h_long_ctx = self.long_term_encoder(long_his_cat_tensors)
        if self.long_ctx_type == "vector":
            z_long_projected = self.z_long_mlp(h_long_ctx)
            h_long_ctx_flow = z_long_projected.unsqueeze(-1).unsqueeze(-1).expand(B,-1,H,W)
        else:
            h_long_ctx_flow = h_long_ctx

        object_flow_batch, residual_flow_batch, uncertainty_map_batch = self.flow_predictor(compensated_bev, h_short_ctx, h_long_ctx_flow, delay_map)
        return object_flow_batch, residual_flow_batch, uncertainty_map_batch

class FeatureWarper(nn.Module):
    def get_grid(self, B, H, W, device):
        shifts_x = torch.arange(0, W, 1, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, H, 1, dtype=torch.float32, device=device)
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')  # Use 'ij' for H,W order
        grid_dst = torch.stack((shifts_x, shifts_y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]
        return grid_dst

    def flow_warp(self, feats, flow, delay_step):  # flow is per unit time
        # feats: [B, C, H, W], flow: [B, 2, H, W] (dx, dy)
        B, C, H, W = feats.shape
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
        super().__init__()
        self.C_V = args_mgdc_bev['C_V']
        self.C_F = args_mgdc_bev['C_F']
        self.C_D = args_mgdc_bev['C_D']
        self.C_total_hist = self.C_V + self.C_F + self.C_D

        self.motion_predictor = HierarchicalMotionPredictor(
            C_total_bev = self.C_V + self.C_F + self.C_D,
            C_total_hist=self.C_total_hist,  # For history items
            D_short_ctx=args_mgdc_bev['D_short_ctx'],
            D_long_ctx_map=args_mgdc_bev['D_long_ctx_map'],  # if long_ctx_type='map'
            D_long_ctx_vec=args_mgdc_bev['D_long_ctx_vec'],  # if long_ctx_type='vector'
            num_short_frames=args_mgdc_bev['num_short_history_frames'],
            num_long_frames=args_mgdc_bev['num_long_history_frames'],
            flow_predictor_hidden_dim=args_mgdc_bev['flow_predictor_hidden_dim'],
            long_ctx_type=args_mgdc_bev.get('long_ctx_type', 'map')
        )

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

    def forward(self, compensated_bev, short_his_list, long_his_list, delay_steps):
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

        long_his_vox, long_his_feat, long_his_det = long_his_list


        short_his_vox_tensors = self._prepare_history_tensor_list(short_his_list[0])
        short_his_fea_tensors = self._prepare_history_tensor_list(short_his_list[1])
        short_his_det_tensors = self._prepare_history_tensor_list(short_his_list[2])

        long_his_vox_tensors = self._prepare_history_tensor_list(long_his_list[0])
        long_his_fea_tensors = self._prepare_history_tensor_list(long_his_list[1])
        long_his_det_tensors = self._prepare_history_tensor_list(long_his_list[2])

        num_short_frames = len(short_his_vox_tensors)
        num_long_frames = len(long_his_vox_tensors)

        #将每个时间步的三个粒度特征拼接在一起
        short_his_cat_tensors = [torch.cat([short_his_vox_tensors[i], short_his_fea_tensors[i], short_his_det_tensors[i]], dim=1) for i in range(num_short_frames)]
        long_his_cat_tensors = [torch.cat([long_his_vox_tensors[i], long_his_fea_tensors[i], long_his_det_tensors[i]], dim=1) for i in range(num_long_frames)]

        #进行运动预测
        object_flow, residual_flow, uncertainty= self.motion_predictor(compensated_bev, short_his_cat_tensors, long_his_cat_tensors, delay_steps)

        final_flow = object_flow + residual_flow

        predicted_F_vox = self.feature_warper.flow_warp(compensated_bev[:,0:self.C_V,:,:], final_flow, delay_steps)
        predicted_F_fea = self.feature_warper.flow_warp(compensated_bev[:,self.C_V:self.C_V+self.C_F,:,:], final_flow, delay_steps)
        predicted_F_det = self.feature_warper.flow_warp(compensated_bev[:,self.C_V+self.C_F:,:,:], final_flow, delay_steps)

        return predicted_F_vox,predicted_F_fea,predicted_F_det







