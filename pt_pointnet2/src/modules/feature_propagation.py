import torch
import torch.nn as nn

class FeaturePropagation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeaturePropagation, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, x_skip):
        # x: input features, x_skip: features from the previous layer
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Concatenate skip features
        x = torch.cat((x, x_skip), dim=1)
        return x