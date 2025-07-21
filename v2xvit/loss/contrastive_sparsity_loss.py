import torch
import torch.nn as nn
import torch.nn.functional as F

class GranularityEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class ContrastiveSparsityLoss(nn.Module):
    def __init__(self, c_g1, c_g2, c_g3, feature_dim=128, temperature=0.1, k=4096, m=0.999):
        '''
        :param c_g1:
        :param c_g2:
        :param c_g3:
        :param feature_dim: 编码后描述向量的维度
        :param temperature: InfoNCE损失的温度超参数
        '''
        super().__init__()

        self.temperature = temperature
        self.k = k
        self.m = m

        self.encoders = nn.ModuleList([GranularityEncoder(c_g1, feature_dim), GranularityEncoder(c_g2, feature_dim), GranularityEncoder(c_g3, feature_dim)])

        #创建负样本队列
        for gran_type in ['g1', 'g2', 'g3']:
            self.register_buffer(f"{gran_type}_queue", torch.randn(feature_dim, k))
            # 归一化队列
            queue = getattr(self, f"{gran_type}_queue")
            setattr(self, f"{gran_type}_queue", nn.functional.normalize(queue, dim=0))
            # 注册队列指针
            self.register_buffer(f"{gran_type}_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    @torch.no_grad()
    def _momentum_update_key_encoders(self):
        """动量更新key_encoders的权重"""
        for q_encoder, k_encoder in zip(self.query_encoders, self.key_encoders):
            for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, granularity_name):
        """
        将当前批次的keys入队，并将最旧的keys出队。
        Args:
            keys (Tensor): 当前批次的keys, shape [N, D]
        """
        batch_size = keys.shape[0]
        queue = getattr(self, f"{granularity_name}_queue")
        ptr = int(getattr(self, f"{granularity_name}_queue_ptr"))

        # 队列已满，替换旧的keys
        assert self.K % batch_size == 0
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # 移动指针

        getattr(self, f"{granularity_name}_queue_ptr")[0] = ptr

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
            batch_dense_data = [batch_dense_g1, batch_dense_g2, batch_dense_g3]

            cav_num = batch_sparse_g1.shape[0]
            if cav_num <= 1:
                continue

            with torch.no_grad():
                dense_g1_keys = self.encoders[0](batch_dense_g1)
                dense_g1_keys = F.normalize(dense_g1_keys, dim=1) #[N*H*W, D]
                dense_g1_keys.permute(0, 2, 3, 1).reshape(-1, dense_g1_keys.shape[1])

                dense_g2_keys = self.encoders[1](batch_dense_g2)
                dense_g2_keys = F.normalize(dense_g2_keys, dim=1)  # [N*H*W, D]
                dense_g2_keys.permute(0, 2, 3, 1).reshape(-1, dense_g2_keys.shape[1])

                dense_g3_keys = self.encoders[0](batch_dense_g3)
                dense_g3_keys = F.normalize(dense_g3_keys, dim=1)  # [N*H*W, D]
                dense_g3_keys.permute(0, 2, 3, 1).reshape(-1, dense_g3_keys.shape[1])
                all_dense_keys = [dense_g1_keys, dense_g2_keys, dense_g3_keys]

            negative_pool = torch.cat(all_dense_keys, dim=0)


            for gran_idx in range(1,4):
                mask = (decision_mask == gran_idx)
                if not mask.any():
                    continue

                agent_indices, y_indices, x_indices = torch.where(mask)

                num_anchors = agent_indices.shape[0]
                if num_anchors == 0:
                    continue

                sparse_data =  batch_sparse_data[gran_idx-1]
                q_map_encoded = self.encoders[gran_idx-1](sparse_data)
                q_map_encoded = F.normalize(q_map_encoded, dim=1)

                queries = q_map_encoded[agent_indices+1,:,y_indices,x_indices] #考虑到从ego-agent后开始索引
                dense_map = batch_dense_data[gran_idx-1]

                with torch.no_grad():
                    k_pos_map_encoded = self.encoders[gran_idx-1](dense_map)
                    k_pos_map_encoded = F.normalize(k_pos_map_encoded, dim=1)

                    # 提取采样点对应的positive key向量
                    positive_keys = k_pos_map_encoded[agent_indices+1,:,y_indices,x_indices]

                # 正样本相似度
                l_pos = torch.einsum('ad,ad->a', queries, positive_keys).unsqueeze(-1)  # [num_anchors, 1]
                # 负样本相似度 (所有锚点共享同一个巨大的负样本池)
                print("negative_pool.shape=", negative_pool.shape)
                print("queries.shape=",queries,shape)
                l_neg = torch.einsum('ad,nd->an', queries, negative_pool)  # [num_anchors, Total_Points]
                # 拼接 logits
                logits = torch.cat([l_pos, l_neg], dim=1)  # [num_anchors, 1 + Total_Points]
                logits /= self.temperature
                # 标签永远是第0个 (正样本)
                labels = torch.zeros(num_anchors, dtype=torch.long, device=logits.device)

                loss = self.criterion(logits, labels)
                total_loss.append(loss)

        if not total_loss:
            return torch.tensor(0.0, device=device)

        final_loss = torch.mean(torch.stack(total_loss))

        return final_loss




