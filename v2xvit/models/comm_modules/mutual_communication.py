from asyncio import set_event_loop

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import random


class Channel_Request_Attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Channel_Request_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

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
            in_channels=img_feature_channels, out_channels=img_feature_channels*2, kernel_size=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=img_feature_channels*2, out_channels=img_feature_channels*2, kernel_size=1, stride=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=img_feature_channels*2, out_channels=1, kernel_size=1, stride=1)
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

        return -mutual_info*self.loss_coeff


class Communication(nn.Module):
    def __init__(self, args, in_planes):
        super(Communication, self).__init__()
        self.replace_mode = args.get("replace_mode", "random")
        self.replace_ratio = args.get("replace_ratio", 0.2)
        self.compression_rate = args.get("compression", 0.5)
        self.discrete_ratio = args['voxel_size'][0]
        self.channel_request = Channel_Request_Attention(in_planes) 
        self.spatial_request = Spatial_Request_Attention()
        self.channel_fusion = nn.Conv2d(in_planes*2, in_planes, 1, bias=False)
        self.spatial_fusion = nn.Conv2d(2, 1, 1, bias=False)
        self.statisticsNetwork = StatisticsNetwork(in_planes*2)
        self.mutual_loss = DeepInfoMaxLoss()
        self.request_flag = args['request_flag']

        self.smooth = False
        self.thre = args['thre']  
        if 'gaussian_smooth' in args:
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            self.kernel_size = kernel_size
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(
                1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False  

        x = torch.arange(-(kernel_size - 1) // 2, (kernel_size + 1) // 2, dtype=torch.float32)
        d1_gaussian_filter = torch.exp(-x**2 / (2 * c_sigma**2))
        d1_gaussian_filter /= d1_gaussian_filter.sum()

        self.d1_gaussian_filter = d1_gaussian_filter.view(1, 1, kernel_size).cuda()
        
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

    def forward(self, feat_list, confidence_map_list=None, raw_voxels=None, raw_coords=None):
        send_feats = []
        selected_voxels = []
        selected_coords = []
        comm_rate_list = []  
        sparse_mask_list = []  
        total_loss = torch.zeros(1).to(feat_list[0].device)
        batch_start = 0
        for bs in range(len(feat_list)): #按batch分割的feat
            selected_batch_voxels = []
            selected_batch_coords = []
            agent_feature = feat_list[bs]
            device = agent_feature.device
            cav_num, C, H, W = agent_feature.shape
            print("cav_num 2是", cav_num)
            batch_mask = (raw_coords[:, 0] >= batch_start) & \
                         (raw_coords[:, 0] < batch_start + cav_num)
            print("batch start 是", batch_start, ",batch_end是", batch_start+cav_num, ",batch_mask是", batch_mask)
            batch_voxel_coords = raw_coords[batch_mask]
            batch_voxel_features = raw_voxels[batch_mask]

            if cav_num == 1:
                send_feats.append(agent_feature)
                ones_mask = torch.ones(cav_num, C, H, W).to(feat_list[0].device)
                sparse_mask_list.append(ones_mask)
                continue
                
            collaborator_feature = torch.tensor([]).to(agent_feature.device)
            sparse_batch_mask = torch.tensor([]).to(agent_feature.device)

            agent_channel_attention = self.channel_request(
                agent_feature) 
            agent_spatial_attention = self.spatial_request(
                agent_feature)
            agent_activation = torch.mean(agent_feature, dim=1, keepdims=True).sigmoid()  
            agent_activation = self.gaussian_filter(agent_activation)

            ego_channel_request = (
                1 - agent_channel_attention[0, ]).unsqueeze(0)  
            ego_spatial_request = (
                1 - agent_spatial_attention[0, ]).unsqueeze(0)  



            for i in range(cav_num-1):
                global_agent_id = batch_start + i +1
                agent_mask = (batch_voxel_coords[:, 0] == global_agent_id)
                agent_coords = batch_voxel_coords[agent_mask].to(device)
                agent_features = batch_voxel_features[agent_mask].to(device)
                print("agent features的形状是:", agent_features.shape)
                if self.request_flag:
                    channel_coefficient = self.channel_fusion(torch.cat(
                        [ego_channel_request, agent_channel_attention[i+1, ].unsqueeze(0)], dim=1))  
                    spatial_coefficient = self.spatial_fusion(torch.cat(
                        [ego_spatial_request, agent_spatial_attention[i+1, ].unsqueeze(0)], dim=1))  
                else:  
                    channel_coefficient = agent_channel_attention[i+1, ].unsqueeze(
                        0)
                    spatial_coefficient = agent_spatial_attention[i+1, ].unsqueeze(
                        0)

                spatial_coefficient = spatial_coefficient.sigmoid()
                channel_coefficient = channel_coefficient.sigmoid()
                smoth_channel_coefficient = F.conv1d(channel_coefficient.reshape(1, 1, C), self.d1_gaussian_filter,
                                                     padding=(self.kernel_size - 1) // 2)
                channel_coefficient = smoth_channel_coefficient.reshape(1, C, 1, 1)
                spatial_coefficient = self.gaussian_filter(spatial_coefficient)
                sparse_matrix = channel_coefficient * spatial_coefficient 
                temp_activation = agent_activation[i+1, ].unsqueeze(0)
                sparse_matrix = sparse_matrix * temp_activation

                if self.thre > 0:
                    ones_mask = torch.ones_like(
                        sparse_matrix).to(sparse_matrix.device)
                    zeros_mask = torch.zeros_like(
                        sparse_matrix).to(sparse_matrix.device)
                    sparse_mask = torch.where(
                        sparse_matrix > self.thre, ones_mask, zeros_mask)
                else:
                    K = int(C * H * W * random.uniform(0, 0.3))
                    communication_maps = sparse_matrix.reshape(1, C * H * W)
                    _, indices = torch.topk(communication_maps, k=K, sorted=False)
                    communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                    ones_fill = torch.ones(1, K, dtype=communication_maps.dtype, device=communication_maps.device)
                    sparse_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(1, C, H, W)

                sparse_points_mask = torch.zeros((H, W)).bool()
                sparse_feature_mask = torch.zeros_like(sparse_mask).bool()
                if self.replace_mode == "random":
                    # 生成与sparse_mask同维度的随机掩码
                    replace_mask = torch.rand_like(sparse_mask, device=device)  # [C,H,W]
                    replace_mask = replace_mask < self.replace_ratio
                    sparse_points_mask = sparse_mask.bool() & replace_mask
                    sparse_feature_mask = sparse_mask.bool() & (~replace_mask)

                elif self.replace_mode == "topk":
                    # 对每个通道独立进行topk选择
                    C, H, W = sparse_mask.shape[-3:]
                    k_per_channel = int(H * W * self.replace_ratio)

                    sparse_points_mask = torch.zeros_like(sparse_mask, device=device).bool()
                    for c in range(C):
                        # 获取当前通道的置信度
                        channel_confidence = confidence_map_list[bs][i + 1][c]  # [H,W]

                        # 对当前通道进行topk选择
                        _, indices = torch.topk(channel_confidence.flatten(), k_per_channel)
                        channel_mask = torch.zeros_like(channel_confidence).bool().flatten()
                        channel_mask[indices] = True
                        sparse_points_mask[c] = channel_mask.reshape(H, W)

                    sparse_points_mask = sparse_mask.bool() & sparse_points_mask
                    sparse_feature_mask = sparse_mask.bool() & (~sparse_points_mask)

                elif self.replace_mode == "attention":
                    # 保持通道维度
                    attention_weights = agent_channel_attention[i + 1] * agent_spatial_attention[i + 1].to(device) # [C,H,W]
                    replace_mask = attention_weights > self.replace_ratio  # [C,H,W]
                    sparse_points_mask = sparse_mask.bool() & replace_mask
                    sparse_feature_mask = sparse_mask.bool() & (~replace_mask)

                sparse_points_mask = sparse_points_mask.squeeze(0)
                C, H, W = sparse_points_mask.shape

                x_idx = (agent_coords[:, 3] / self.discrete_ratio).long().clamp(0, W - 1)  # [K]
                y_idx = (agent_coords[:, 2] / self.discrete_ratio).long().clamp(0, H - 1)  # [K]

                # ==== 生成三维掩码索引 ====
                voxel_mask = sparse_points_mask[:, y_idx, x_idx].any(dim=0)
                print("voxel_mask:", len(voxel_mask))
                if False in voxel_mask:
                    print("确实存在False")
                selected_agent_coords = agent_coords[voxel_mask]
                selected_agent_voxels = agent_features[voxel_mask]
                selected_batch_voxels.append(selected_agent_voxels)
                selected_batch_coords.append(selected_agent_coords)
                print("len(selected_agent_coords)=", len(selected_agent_coords))
                print("len(agent_coords)=", len(agent_coords))
                feature_rate = sparse_feature_mask.sum() / (C * H * W)
                voxel_rate = sparse_points_mask.sum() / (H * W)
                comm_rate = feature_rate + 3.0 * voxel_rate
                comm_rate_list.append(comm_rate)

                collaborator_feature = torch.cat(
                    [collaborator_feature, agent_feature[i+1, ].unsqueeze(0)*sparse_feature_mask], dim=0)
                sparse_batch_mask = torch.cat(
                    [sparse_batch_mask, sparse_feature_mask], dim=0)

            selected_voxels.append(selected_batch_voxels)
            selected_coords.append(selected_batch_coords)

            org_feature = agent_feature.clone()
            sparse_feature = torch.cat(
                [agent_feature[:1], collaborator_feature], dim=0)
            send_feats.append(sparse_feature)
            ego_mask = torch.ones_like(agent_feature[:1]).to(
                agent_feature[:1].device)
            sparse_batch_mask = torch.cat(
                [ego_mask, sparse_batch_mask], dim=0)
            sparse_mask_list.append(sparse_batch_mask)

            org_feature_prime = torch.cat(
                [org_feature[1:], org_feature[0].unsqueeze(0)], dim=0)
            local_mutual = self.statisticsNetwork(
                torch.cat([org_feature, sparse_feature], dim=1))
            local_mutual_prime = self.statisticsNetwork(
                torch.cat([org_feature_prime, sparse_feature], dim=1))
            loss = self.mutual_loss(local_mutual, local_mutual_prime)
            # recon_loss = F.mse_loss(collaborator_points[:, :3], org_point_cloud[:, :3]) #增加重建损失
            # total_loss += loss + 0.5*recon_loss
            total_loss += loss

            batch_start += cav_num
        mean_rate = torch.stack(comm_rate_list).mean()
        sparse_mask = torch.cat(sparse_mask_list, dim=0)


        return send_feats, total_loss, mean_rate, sparse_mask, selected_voxels, selected_coords

    def transform_coords(self, coords, t_matrix):
        """
            :param coords: [K,3] (z,y,x) agent坐标系下的体素中心坐标
            :param t_matrix: [4,4] agent到ego的变换矩阵
            :return: [K,3] ego坐标系下的坐标
        """
        homog_coords = F.pad(coords, (0,1), value=1.0) #[K,4]
        ego_coords = (t_matrix @ homog_coords.T).T[:, :3]
        return ego_coords

    def generate_bev_mask(self, ego_coords, bev_shape=(200, 200), voxel_size=0.4):
        """
        :param ego_coords: [K,3] ego坐标系下的体素坐标
        :return: [H,W] bool类型BEV占据掩码
        """
        H, W = bev_shape
        # 计算BEV网格索引
        x = (ego_coords[:, 2] / voxel_size).long().clamp(0, W - 1)  # x对应W维度
        y = (ego_coords[:, 1] / voxel_size).long().clamp(0, H - 1)  # y对应H维度

        # 创建BEV掩码
        bev_mask = torch.zeros((H, W), dtype=bool)
        bev_mask[y, x] = True
        return bev_mask

    def _fuse_features(self, feature, points, voxel_coords, H, W):
        """
                feature: [N, C, H, W]
                points: [K, 4] (x,y,z,intensity均值)
                voxel_coords: [K, 4] (batch_idx, z, y, x)
        """
        x_indices = (voxel_coords[:, 3] * self.discrete_ratio).long()
        y_indices = (voxel_coords[:, 2] * self.discrete_ratio).long()

        feature[:, :3, y_indices, x_indices] = points[:, :3].T
        return feature

