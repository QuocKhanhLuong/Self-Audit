"""Native decoder: no frozen-Q bank, feature teacher, projection or dead head.

DINOv3 is an explicit local pretrained initialization, never an online fallback.
The small model exists solely for CPU contracts and manufactured experiments.
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


class NativeStudent(nn.Module):
    """All pixels are supervised at output resolution. Affine head for audit."""
    def __init__(self, encoder, widths=(96, 128, 192, 256)):
        super().__init__()
        self.encoder = encoder
        self.input_adapter = CenterDifference()
        self.lateral = nn.ModuleList(nn.Conv2d(c, d, 1) for c, d in zip(encoder.channels, widths))
        self.up = nn.ModuleList(nn.Conv2d(widths[i + 1], widths[i], 1) for i in range(3))
        self.refine = nn.ModuleList(nn.Sequential(Residual(d), Residual(d)) for d in widths)
        self.detail = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.GELU(), Residual(32))
        self.fuse = nn.Sequential(nn.Conv2d(widths[0] + 32, 32, 1), nn.GELU(), Residual(32))
        self.head = nn.Conv2d(32, 4, 1)

    def features(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("expected finite normalized Bx3xHxW input")
        h, w = x.shape[-2:]
        pad = (0, (-w) % 32, 0, (-h) % 32)
        norm, raw = F.pad(self.input_adapter(x), pad, mode="replicate"), F.pad(x, pad, mode="replicate")
        stages = self.encoder(norm)
        if len(stages) != 4:
            raise ValueError("four native backbone stages required")
        for i, (f, c) in enumerate(zip(stages, self.encoder.channels)):
            if f.shape[1] != c or f.shape[-2:] != (norm.shape[-2] // (4 * 2**i), norm.shape[-1] // (4 * 2**i)):
                raise ValueError("incorrect native backbone stride; no implicit feature resizing")
        d = self.refine[3](self.lateral[3](stages[3]))
        for i in (2, 1, 0):
            skip = self.lateral[i](stages[i])
            d = self.refine[i](skip + self.up[i](F.interpolate(d, skip.shape[-2:], mode="bilinear", align_corners=False)))
        f = self.fuse(torch.cat((F.interpolate(d, raw.shape[-2:], mode="bilinear", align_corners=False), self.detail(raw)), 1))
        return f[..., :h, :w]

    def forward(self, x):
        return self.head(self.features(x))


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


def build_model(config, *, synthetic=False):
    name = config["backbone"]
    if name == "tiny_test":
        if not synthetic:
            raise ValueError("tiny_test is prohibited for patient training")
        return NativeStudent(TinyEncoder(), (8, 16, 24, 32)), {"pretrained": False, "synthetic": True}
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
    return NativeStudent(OfficialStages(original)), {"pretrained": True, "weights_sha256": expected,
            "source_commit": head, "lineage": "user_supplied_hash_pinned;not_independent_upstream_certification"}
