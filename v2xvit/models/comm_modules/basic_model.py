import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True, use_relu=True):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BEVPatchEmbed(nn.Module):
    def __init__(self, C_in, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(C_in, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, E, H', W']
        return x.flatten(2).transpose(1, 2)  # [B, H'*W', E]

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim_factor=2, num_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, in_dim * hidden_dim_factor), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(in_dim * hidden_dim_factor, in_dim * hidden_dim_factor), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(in_dim * hidden_dim_factor, out_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x): return self.mlp(x)
