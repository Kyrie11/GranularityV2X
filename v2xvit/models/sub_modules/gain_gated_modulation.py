import torch
import torch.nn as nn


class GainGatedModulation(nn.Module):
    """
    Performs Gain-Gated Modulation on a batch of features from multiple agents.
    It enhances features that are unique to each agent while suppressing common/consensus features.
    """

    def __init__(self, channels: int):
        """
        Args:
            channels (int): The number of channels C in the input feature map [N, C, H, W].
        """
        super().__init__()

        # A simple 1x1 Convolution to process the "uniqueness" map
        # and convert it into a single-channel gate.
        # This network is intentionally kept simple as requested.
        self.gate_generator = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, 1, kernel_size=1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): The input tensor of shape [N, C, H, W], where N
                                     is the total number of agents in the scene.

        Returns:
            torch.Tensor: The enhanced feature tensor of the same shape [N, C, H, W].
        """
        if features.shape[0] <= 1:
            # If there's only one agent (or no agents), no consensus can be formed.
            # Return the original features directly.
            return features

        # 1. Calculate the "consensus" feature by averaging across the agent dimension (N).
        # Shape: [1, C, H, W]
        consensus_feature = torch.mean(features, dim=0, keepdim=True)

        # 2. Calculate the "uniqueness" feature for each agent.
        # This is the absolute difference between each agent's feature and the consensus.
        # The broadcasting mechanism of PyTorch handles the shape difference automatically.
        # Shape: [N, C, H, W]
        uniqueness_feature = torch.abs(features - consensus_feature)

        # 3. Generate the gain gate for each agent.
        # The gate should have high values where the features are unique.
        # Shape: [N, 1, H, W]
        gate = torch.sigmoid(self.gate_generator(uniqueness_feature))

        # 4. Apply the Gain-Gated Modulation.
        # We use (1 + gate) to amplify unique features rather than just selecting them.
        # This preserves the original information and adds a "gain" based on uniqueness.
        enhanced_features = features * (1 + gate)

        return enhanced_features


