import torch
import torch.nn as nn

#传递稀疏数据时的重建损失
class ReconstructionLoss(nn.Module):
    def __init__(self, full_bev_channels, hidden_decoder_channels=128):
        super(ReconstructionLoss, self).__init__()
        #Decoder网络：从稀疏传输图重建全信息图
        #输入通道数是全信息图的通道数
        self.decoder = nn.Sequential(
            nn.Conv2d(full_bev_channels, hidden_decoder_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_decoder_channels, hidden_decoder_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_decoder_channels, full_bev_channels, kernel_size=1)
        )
        self.loss_func = nn.L1Loss(reduction="none")

    def forward(self, f_trans_bev, X_bev_full, valid_mask=None):
        #f_trans_bev: [B,C_total,H,W](期望的稀疏传输图，通过Gumbel-Softmax等得到)
        #X_bev_full: [B,C_total,H,W](原始的全信息)

        reconstructed_X_bev = self.decoder(f_trans_bev)
        loss_recon_pixelwise = self.loss_func(reconstructed_X_bev, X_bev_full.detach())

        if valid_mask is not None:
            #valid_mask [B,H,W] -> [B,1,H,W] for broadcasting
            loss_recon = (loss_recon_pixelwise * valid_mask.unsqueeze(1)).sum() / (valid_mask.sum()* X_bev_full.shape[1] + 1e-6)
        else:
            loss_recon = loss_recon_pixelwise.mean()

        return loss_recon