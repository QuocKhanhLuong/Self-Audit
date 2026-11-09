"""Shared teacher backbone blocks and local pretrained initialization.

DINOv3 is an explicit local pretrained initialization, never an online fallback.
The small encoder exists solely for CPU contracts and manufactured experiments;
Candidate A owns its teacher branches and decoder.
"""
from pathlib import Path
import hashlib
import subprocess
import torch
from torch import nn
from torch.nn import functional as F


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class CenterDifference(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])
        self.context = nn.Conv2d(2, 3, 3, padding=1, bias=False)
        nn.init.zeros_(self.context.weight)

    def forward(self, x):
        center = x[:, 1:2]
        diff = torch.cat((x[:, :1] - center, x[:, 2:3] - center), 1)
        return (center.expand(-1, 3, -1, -1) - self.mean) / self.std + self.context(diff)


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, channels), nn.GELU(),
                                  nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, channels))

    def forward(self, x):
        return F.gelu(x + self.body(x))


class TinyEncoder(nn.Module):
    channels = (8, 16, 32, 64)
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList(nn.Conv2d(c, d, 4 if i == 0 else 2, stride=4 if i == 0 else 2)
                                   for i, (c, d) in enumerate(zip((3, 8, 16, 32), self.channels)))
    def forward(self, x):
        outputs = []
        for layer in self.stages:
            x = F.gelu(layer(x)); outputs.append(x)
        return outputs


class OfficialStages(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.downsample_layers, self.stages = original.downsample_layers, original.stages
        self.channels = tuple(original.embed_dims)
        if self.channels != (96, 192, 384, 768):
            raise ValueError("expected DINOv3 ConvNeXt-Tiny channels")
    def forward(self, x):
        outputs = []
        for down, stage in zip(self.downsample_layers, self.stages):
            x = stage(down(x)); outputs.append(x)
        return outputs


def build_backbone(config, *, synthetic=False):
    """Load only the verified four-stage encoder used by Candidate A.

    The Candidate-A teacher must not instantiate and discard a native decoder.
    This backbone loader keeps provenance checks in one place while leaving
    branch and decoder construction to their owning modules.
    """

    name = config["backbone"]
    if name == "tiny_test":
        if not synthetic:
            raise ValueError("tiny_test is prohibited for patient training")
        return TinyEncoder(), {"pretrained": False, "synthetic": True}
    if name != "dinov3_convnext_tiny":
        raise ValueError("unsupported backbone; no random/legacy fallback")
    source, weights = Path(config["source_repo"]).resolve(), Path(config["weights"]).resolve()
    expected = config["weights_sha256"]
    if len(expected) != 64 or digest(weights) != expected:
        raise ValueError("pretrained weight hash mismatch")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != config["source_commit"]:
        raise ValueError("backbone source commit mismatch")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise ValueError("backbone tracked source is dirty")
    original = torch.hub.load(str(source), "dinov3_convnext_tiny", source="local", weights=str(weights), pretrained=True)
    return OfficialStages(original), {"pretrained": True, "weights_sha256": expected,
            "source_commit": head, "lineage": "user_supplied_hash_pinned;not_independent_upstream_certification"}
