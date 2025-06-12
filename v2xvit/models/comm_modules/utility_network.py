import torch
import torch.nn as nn


#后续要补全：对每个粒度的带宽衡量
class UtilityNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(UtilityNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, ego_request_spatial, collab_state_spatial,
                ego_request_granularity, collab_state_granularity,
                ego_semantic_context, collab_semantic_context,
                content_embedding_granularity_g, bandwidth_cost_g):
        #将所有输入展平拼接
        #空间相关的输入是针对空间r，粒度相关的输入是针对粒度g的
        #content_embedding_granularity_g是e_r,g
        inputs = torch.cat([
            ego_request_spatial.unsqueeze(-1), collab_state_spatial.unsqueeze(-1),
            ego_request_granularity.unsqueeze(-1), collab_state_granularity.unsqueeze(-1),
            ego_semantic_context.unsqueeze(0).expand(ego_request_spatial.shape[0],
                                                     -1) if ego_semantic_context.ndim == 1 else ego_semantic_context,
            # 假设ego_semantic_context是全局的，需要扩展
            collab_semantic_context.unsqueeze(0).expand(collab_state_spatial.shape[0],
                                                        -1) if collab_semantic_context.ndim == 1 else collab_semantic_context,
            content_embedding_granularity_g,
            bandwidth_cost_g.unsqueeze(-1)
        ], dim=-1)
        return self.net(inputs).squeeze(-1)