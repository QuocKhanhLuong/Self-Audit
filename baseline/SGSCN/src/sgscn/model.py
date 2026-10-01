# SPDX-License-Identifier: GPL-3.0
# Derived from Ahn et al., demo_final.py, pinned in upstream/receipt.json.
import torch.nn as nn
import torch.nn.functional as F


class CSNet(nn.Module):
    def __init__(self, input_dim, *, n_channels, n_conv):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, n_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for _ in range(n_conv - 1):
            self.conv2.append(nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(n_channels))
        self.conv3 = nn.Conv2d(n_channels, n_channels, 1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        for convolution, batch_norm in zip(self.conv2, self.bn2):
            x = batch_norm(F.relu(convolution(x)))
        return self.bn3(self.conv3(x))
