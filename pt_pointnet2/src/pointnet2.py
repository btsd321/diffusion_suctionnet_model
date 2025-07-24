import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.set_abstraction import SetAbstraction
from .modules.feature_propagation import FeaturePropagation

class PointNet2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2, self).__init__()
        self.sa1 = SetAbstraction(1024, 0.2, 3, [64, 64, 128])
        self.sa2 = SetAbstraction(256, 0.4, 128, [128, 128, 256])
        self.fp1 = FeaturePropagation(256, 128)
        self.fp2 = FeaturePropagation(128, 64)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, xyz):
        B, N, _ = xyz.size()
        l1_points, l1_indices = self.sa1(xyz)
        l2_points, l2_indices = self.sa2(l1_points)
        l3_points = self.fp1(l2_points, l1_points, l1_indices)
        l4_points = self.fp2(l3_points, xyz, l2_indices)
        out = self.fc(l4_points)
        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))