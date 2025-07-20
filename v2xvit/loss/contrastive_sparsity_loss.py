import torch
import torch.nn as nn
import torch.nn.functional as F

class GranularityEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化，得到 [B, D, 1, 1]
            nn.Flatten()  # 展平为 [B, D]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class ContrastiveSparsityLoss(nn.Module):
    def __init__(self, c_g1, c_g2, c_g3, feature_dim=128, temperature=0.1):
        '''
        :param c_g1:
        :param c_g2:
        :param c_g3:
        :param feature_dim: 编码后描述向量的维度
        :param temperature: InfoNCE损失的温度超参数
        '''
        super().__init__()

        self.g1_encoder = GranularityEncoder(c_g1, feature_dim)
        self.g2_encoder = GranularityEncoder(c_g2, feature_dim)
        self.g3_encoder = GranularityEncoder(c_g3, feature_dim)

        self.encoder = [self.g1_encoder, self.g2_encoder, self.g3_encoder]

        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, sparse_data, dense_data, decision_mask_list):
        sparse_g1, sparse_g2, sparse_g3 = sparse_data
        dense_g1, dense_g2, dense_g3 = dense_data

        device = sparse_g1[0].device
        total_loss = []

        batch_size = len(sparse_g1)
        print("batch_size=", batch_size)
        print("len=", len(decision_mask_list))
        for b in range(batch_size):
            decision_mask = decision_mask_list[b]
            batch_sparse_g1 = sparse_g1[b]
            batch_sparse_g2 = sparse_g2[b]
            batch_sparse_g3 = sparse_g3[b]
            batch_sparse_data = [batch_sparse_g1, batch_sparse_g2, batch_sparse_g3]

            batch_dense_g1 = dense_g1[b]
            batch_dense_g2 = dense_g2[b]
            batch_dense_g3 = dense_g3[b]
            # batch_dense_data = [batch_dense_g1, batch_dense_g2, batch_dense_g3]

            cav_num = batch_sparse_g1.shape[0]
            if cav_num <= 1:
                continue

            with torch.no_grad():
                dense_g1_keys = self.g1_encoder(batch_dense_g1)
                dense_g2_keys = self.g2_encoder(batch_dense_g2)
                dense_g3_keys = self.g3_encoder(batch_dense_g3)
                all_dense_keys = [dense_g1_keys, dense_g2_keys, dense_g3_keys]
            for i in range(1,  cav_num):
                unique_mask = decision_mask[i-1,:,:]
                unique_mask_cpu = unique_mask.cpu()
                print("decision_mask:", unique_mask.shape)
                for row in unique_mask_cpu:
                    for decision_val in row:
                        print("val:", decision_val)
                        if decision_val == 0:
                            continue
                        granularity_idx = decision_val - 1
                        encoder = self.encoder[granularity_idx]
                        anchor_sparse_data = batch_sparse_data[granularity_idx][i:i+1]
                        q = encoder[granularity_idx](anchor_sparse_data)

                        # Positive Key: 编码第i个CAV的对应稠密数据
                        k_pos = all_dense_keys[granularity_idx][i:i+1]

                        #构建负样本集
                        # a) 跨智能体负样本 (Inter-Agent Negatives)
                        # 对于当前粒度，其他所有CAV的稠密数据都是负样本
                        neg_keys_list = []
                        agent_indices = list(range(cav_num))
                        agent_indices.pop(i)
                        inter_agent_neg = all_dense_keys[granularity_idx][agent_indices]
                        neg_keys_list.append(inter_agent_neg)
                        # b) 跨粒度负样本 (Inter-Granularity Negatives)
                        # 对于当前CAV，其他所有粒度的稠密数据都是负样本
                        for other_gran_idx in range(3):
                            if other_gran_idx == granularity_idx:
                                continue
                            inter_agent_neg = all_dense_keys[other_gran_idx][i:i+1]
                            neg_keys_list.append(inter_agent_neg)

                        if not neg_keys_list:
                            continue

                        # ------ 计算InfoNCE损失 ------
                        k_neg = torch.cat(neg_keys_list, dim=0) #[N,D]
                        # L2-normalize all vectors
                        q = F.normalize(q, dim=1)
                        k_pos = F.normalize(k_pos, dim=1)
                        k_neg = F.normalize(k_neg, dim=1)

                        # 计算正样本相似度
                        l_pos = torch.einsum('bd,bd->b', q, k_pos)  # Shape: [1]
                        # 计算负样本相似度
                        l_neg = torch.einsum('bd,nd->bn', q, k_neg)  # Shape: [1, N_neg]
                        # 拼接成logits
                        logits = torch.cat([l_pos.unsqueeze(1), l_neg], dim=1)  # Shape: [1, 1 + N_neg]
                        # 应用温度系数
                        logits /= self.temperature

                        # 标签永远是第0个（正样本）
                        # 我们只有一个查询，所以batch_size是1
                        labels = torch.zeros(1, dtype=torch.long, device=q.device)

                        loss = self.criterion(logits, labels)
                        total_loss.append(loss)
            if not total_loss:
                return torch.tensor(0.0, device=device)

            final_loss = torch.mean(torch.stack(total_loss))

            return final_loss




