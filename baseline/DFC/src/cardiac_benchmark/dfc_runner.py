"""Minimal attributed mirror of kanezaki DFC direct/headless behavior (MIT)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .provenance import seed_everything


@dataclass(frozen=True)
class DFCConfig:
    profile_id: str = "DFC-Direct-2D-Default-MinL3"
    input_dim: int = 1
    nChannel: int = 100
    nConv: int = 2
    maxIter: int = 1000
    minLabels: int = 3
    lr: float = 0.1
    stepsize_sim: float = 1.0
    stepsize_con: float = 1.0
    momentum: float = 0.9
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    optimizer: str = "SGD"
    scheduler: str = "none"
    gradient_clipping: str = "none"
    amp: bool = False
    dtype: str = "float32"
    final_forward_policy: str = "fresh-post-update-train-mode-grad-enabled"
    bn_mode: str = "train"

    def __post_init__(self):
        frozen = {"profile_id": "DFC-Direct-2D-Default-MinL3", "input_dim": 1, "nChannel": 100, "nConv": 2, "maxIter": 1000, "minLabels": 3, "lr": 0.1, "stepsize_sim": 1.0, "stepsize_con": 1.0, "momentum": 0.9, "weight_decay": 0.0, "dampening": 0.0, "nesterov": False, "optimizer": "SGD", "scheduler": "none", "gradient_clipping": "none", "amp": False, "dtype": "float32", "final_forward_policy": "fresh-post-update-train-mode-grad-enabled", "bn_mode": "train"}
        if self.profile_id == frozen["profile_id"] and asdict(self) != frozen:
            raise ValueError("primary profile scientific fields are immutable")


class MyNet(nn.Module):
    """Structurally identical to demo.py::MyNet, parameterized instead of argparse."""
    def __init__(self, input_dim: int, n_channel: int = 100, n_conv: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, n_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.conv2 = nn.ModuleList([nn.Conv2d(n_channel, n_channel, kernel_size=3, stride=1, padding=1) for _ in range(n_conv - 1)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(n_channel) for _ in range(n_conv - 1)])
        self.conv3 = nn.Conv2d(n_channel, n_channel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(n_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(F.relu(self.conv1(x)))
        for conv, bn in zip(self.conv2, self.bn2): x = bn(F.relu(conv(x)))
        return self.bn3(self.conv3(x))


@dataclass
class DFCResult:
    raw_cluster_map: np.ndarray
    update_count: int
    forward_count: int
    stop_reason: str
    pre_stop_active_ids: list[int]
    pre_stop_active_count: int
    final_active_ids: list[int]
    final_active_count: int
    losses: list[dict[str, float]]


def loss_terms(response: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # response [1,C,H,W], exactly the original flatten/reshape arithmetic.
    _, channels, height, width = response.shape
    flat = response[0].permute(1, 2, 0).contiguous().view(-1, channels)
    grid = flat.reshape(height, width, channels)
    target = flat.max(1).indices
    sim = nn.CrossEntropyLoss(reduction="mean")(flat, target)
    row = torch.mean(torch.abs(grid[1:] - grid[:-1]))
    col = torch.mean(torch.abs(grid[:, 1:] - grid[:, :-1]))
    return flat, sim, row, col, target


def stop_after_update(pre_update_label_count: int, min_labels: int) -> bool:
    """Named only to make the frozen post-update check independently testable."""
    return pre_update_label_count <= min_labels


def run_dfc(data: torch.Tensor, config: DFCConfig, sample_seed: int, device: str = "cpu") -> DFCResult:
    if data.dtype != torch.float32 or data.shape[0] != 1 or data.shape[1] != config.input_dim or data.ndim != 4 or min(data.shape[-2:]) < 2 or not torch.isfinite(data).all():
        raise ValueError("DFC requires finite float32 [1,input_dim,H,W] input")
    seed_everything(sample_seed)  # immediately followed by CPU model construction.
    model = MyNet(config.input_dim, config.nChannel, config.nConv).to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay, dampening=config.dampening, nesterov=config.nesterov)
    image = data.to(device=device, dtype=torch.float32)
    forwards, losses, pre_ids = 0, [], []
    stop = "max_iter"
    for _ in range(config.maxIter):
        optimizer.zero_grad()
        response = model(image); forwards += 1
        flat, sim, row, col, target = loss_terms(response)
        pre_ids = sorted(int(x) for x in torch.unique(target).detach().cpu().tolist())
        total = config.stepsize_sim * sim + config.stepsize_con * (row + col)
        if not torch.isfinite(total): raise FloatingPointError("non-finite DFC loss")
        total.backward(); optimizer.step()
        losses.append({"similarity": float(sim.detach()), "row": float(row.detach()), "column": float(col.detach()), "total": float(total.detach())})
        if stop_after_update(len(pre_ids), config.minLabels):
            stop = "pre_update_labels_at_or_below_min_after_update"; break
    # Do not use eval/no_grad: official visualize=0 performs a graph-building train-mode final pass.
    final_response = model(image); forwards += 1
    final = final_response[0].permute(1, 2, 0).contiguous().view(-1, config.nChannel).max(1).indices.detach().cpu().numpy().reshape(data.shape[-2:])
    final = np.ascontiguousarray(final.astype("<i4", copy=False))
    final_ids = sorted(int(x) for x in np.unique(final).tolist())
    return DFCResult(final, len(losses), forwards, stop, pre_ids, len(pre_ids), final_ids, len(final_ids), losses)
