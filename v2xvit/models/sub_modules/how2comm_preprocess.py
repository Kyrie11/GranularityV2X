import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from v2xvit.models.sub_modules.feature_flow import FlowGenerator, ResNetBEVBackbone
from v2xvit.models.comm_modules.mutual_communication import Communication


class How2commPreprocess(nn.Module):
    def __init__(self, args, channel, delay):
        super(How2commPreprocess, self).__init__()
        self.flow_flag = args['flow_flag'] 
        self.channel = channel
        self.frame = args['fusion_args']['frame']  
        self.delay = delay  
        self.flow = FlowGenerator(args)

        self.commu_module = Communication(
            args['fusion_args']['communication'], in_planes=self.channel)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def get_grid(self, flow):
        m, n = flow.shape[-2:]
        shifts_x = torch.arange(
            0, n, 1, dtype=torch.float32, device=flow.device)
        shifts_y = torch.arange(
            0, m, 1, dtype=torch.float32, device=flow.device)
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)

        grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)
        workspace = torch.tensor(
            [(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)

        flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)

        return flow_grid

    def resample(self, feats, flow):
        flow_grid = self.get_grid(flow)
        warped_feats = F.grid_sample(
            feats, flow_grid, mode="bilinear", paddi6ng_mode="border")

        return warped_feats

    def communication(self, vox_bev,feats,det_bev,record_len,history_vox_list,history_list,history_det_list,confidence_map_list):
        vox_list = self.regroup(vox_bev, record_len)
        feat_list = self.regroup(feats, record_len)
        det_list = self.regroup(det_bev, record_len)
        all_agents_sparse_transmitted_data, total_loss = self.commu_module(
            vox_list,feat_list,det_list,confidence_map_list)
        all_agents_sparse_transmitted_data = torch.cat(all_agents_sparse_transmitted_data, dim=0)
        sparse_history_list = []

        for i in range(len(all_agents_sparse_transmitted_data)):
            # print("history_vox_list[i][:1].shape=",history_vox_list[i][:1].shape)
            # print("history_list[i][:1].shape=",history_list[i][:1].shape)
            # print("history_det_list[i][:1].shape=",history_det_list[i][:1].shape)
            ego_history = torch.cat([history_vox_list[i][:1], history_list[i][:1], history_det_list[i][:1]], dim=1)
            print("ego_history.shape=", ego_history.shape)
            print("all_agents_sparse_transmitted_data[i].shape=",all_agents_sparse_transmitted_data[i].shape)
            sparse_history = torch.cat([ego_history, all_agents_sparse_transmitted_data[i][1:]], dim=0)
            sparse_history_list.append(sparse_history)

        sparse_history = torch.cat(sparse_history_list, dim=0)
        return all_agents_sparse_transmitted_data,  total_loss, sparse_history

    def forward(self, fused_curr, fused_history, record_len, backbone=None, heads=None):
        vox_curr, feat_curr, det_curr = fused_curr
        vox_curr = self.regroup(vox_curr, record_len)
        feat_curr = self.regroup(feat_curr, record_len)
        det_curr = self.regroup(det_curr, record_len)

        vox_history, feat_history, det_history = fused_history

        B = len(feat_curr)
        vox_list = [[] for _ in range(B)]
        feat_list = [[] for _ in range(B)]
        det_list = [[] for _ in range(B)]
        for bs in range(B):
            vox_list[bs] += [vox_curr[bs], vox_history[bs]]
            feat_list[bs] += [feat_curr[bs], feat_history[bs]]
            det_list[bs] += [det_curr[bs], det_history[bs]]

        fused_list = [vox_list, feat_list, det_list]
        if self.flow_flag:
            feat_final, offset_loss = self.flow(fused_list)
        else:
            offset_loss = torch.zeros(1).to(record_len.device)
            x_list = []
            for bs in range(B):
                delayed_colla_vox = vox_list[bs][self.delay][1:]
                delayed_colla_feat = feat_list[bs][self.delay][1:]
                delayed_colla_det = det_list[bs][self.delay][1:]
                ego_feat = feat_list[bs][0][:1]  
                x_list.append(
                    torch.cat([ego_feat, delayed_colla_feat], dim=0))
            feat_final = torch.cat(x_list, dim=0)

        return feat_final, offset_loss
