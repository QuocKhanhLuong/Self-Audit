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
from .window import SemanticWindow, validate_window_config


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
    def __init__(self, encoder, widths=(96, 128, 192, 256), window=None, teacher_memory=None):
        super().__init__()
        self.encoder = encoder
        self.input_adapter = CenterDifference()
        self.lateral = nn.ModuleList(nn.Conv2d(c, d, 1) for c, d in zip(encoder.channels, widths))
        self.up = nn.ModuleList(nn.Conv2d(widths[i + 1], widths[i], 1) for i in range(3))
        self.refine = nn.ModuleList(nn.Sequential(Residual(d), Residual(d)) for d in widths)
        # Keep the native baseline's RNG stream and state_dict byte-for-byte
        # unchanged.  Window arms get their own construction-time RNG fork so
        # detail/fuse/head initialization remains matched across arms.
        self.window = None
        if window is not None:
            if isinstance(window, nn.Module):
                self.window = window
            else:
                window_cfg = validate_window_config(window)
                if window_cfg["mode"] != "none":
                    with torch.random.fork_rng(devices=[]):
                        self.window = SemanticWindow(widths[0], widths[1], window_cfg)
        self.detail = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.GELU(), Residual(32))
        self.fuse = nn.Sequential(nn.Conv2d(widths[0] + 32, 32, 1), nn.GELU(), Residual(32))
        self.head = nn.Conv2d(32, 4, 1)
        # None registers no module/state keys and leaves the baseline path intact.
        self.teacher_memory = teacher_memory

    def features(self, x, valid_mask=None, *, return_memory_feature=False):
        if x.ndim != 4 or x.shape[1] != 3 or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("expected finite normalized Bx3xHxW input")
        if valid_mask is not None:
            if valid_mask.ndim != 3 or valid_mask.shape != (x.shape[0], x.shape[2], x.shape[3]) or valid_mask.dtype is not torch.bool:
                raise ValueError("geometric valid mask must be a bool BxHxW tensor")
            valid_mask = valid_mask.to(device=x.device)
        h, w = x.shape[-2:]
        pad = (0, (-w) % 32, 0, (-h) % 32)
        norm, raw = F.pad(self.input_adapter(x), pad, mode="replicate"), F.pad(x, pad, mode="replicate")
        if self.window is not None:
            # A window arm must never read the replicate-padded tail when the
            # caller omits a mask.  Treat the original image as valid and the
            # internal padding as invalid; explicit all-true masks therefore
            # have exactly the same geometry.
            if valid_mask is None:
                valid_mask = torch.ones((x.shape[0], h, w), dtype=torch.bool, device=x.device)
            padded_valid = F.pad(valid_mask[:, None], pad, mode="constant", value=False)[:, 0]
        else:
            padded_valid = None
        stages = self.encoder(norm)
        if len(stages) != 4:
            raise ValueError("four native backbone stages required")
        for i, (f, c) in enumerate(zip(stages, self.encoder.channels)):
            if f.shape[1] != c or f.shape[-2:] != (norm.shape[-2] // (4 * 2**i), norm.shape[-1] // (4 * 2**i)):
                raise ValueError("incorrect native backbone stride; no implicit feature resizing")
        d = self.refine[3](self.lateral[3](stages[3]))
        memory_feature = None
        for i in (2, 1, 0):
            skip = self.lateral[i](stages[i])
            up = self.up[i](F.interpolate(d, skip.shape[-2:], mode="bilinear", align_corners=False))
            merged = skip + up
            if i == 0 and self.window is not None:
                if padded_valid is None:
                    window_valid = None
                else:
                    # The window lives on the stride-4 query grid.  A query
                    # cell is geometric support only when its full input
                    # footprint is real image area; padding never becomes a
                    # learned/label-dependent signal.
                    window_valid = F.interpolate(
                        padded_valid[:, None].float(), size=merged.shape[-2:], mode="area"
                    ).ge(1.0 - 1e-6)[:, 0]
                merged = merged + self.window(merged, d, window_valid)
            d = self.refine[i](merged)
            if i == 0 and return_memory_feature:
                # [B,widths[0],ceil(H/4),ceil(W/4)]: post-refine[0] memory query.
                memory_feature = d
        f = self.fuse(torch.cat((F.interpolate(d, raw.shape[-2:], mode="bilinear", align_corners=False), self.detail(raw)), 1))
        f = f[..., :h, :w]
        return (f, memory_feature) if return_memory_feature else f

    def forward(self, x, valid_mask=None):
        if self.teacher_memory is None:
            return self.head(self.features(x, valid_mask=valid_mask))
        features, memory_feature = self.features(x, valid_mask=valid_mask, return_memory_feature=True)
        base_logits = self.head(features)
        native_valid = valid_mask.to(device=x.device) if valid_mask is not None else None
        return base_logits + self.teacher_memory(memory_feature, base_logits, native_valid)


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


def build_model(config, *, synthetic=False, window=None, memory_bank=None,
                memory_mode="paired", memory_tau=0.2):
    if window is None and isinstance(config, dict) and "window" in config:
        window = config["window"]
    name = config["backbone"]
    def attach_memory(model):
        if memory_bank is not None:
            from .teacher_memory import TeacherMemory
            # Memory projection initialization must not perturb later RNG draws.
            with torch.random.fork_rng(devices=[]):
                model.teacher_memory = TeacherMemory(model.lateral[0].out_channels, memory_bank,
                                                     mode=memory_mode, tau=memory_tau)
        return model
    if name == "tiny_test":
        if not synthetic:
            raise ValueError("tiny_test is prohibited for patient training")
        return attach_memory(NativeStudent(TinyEncoder(), (8, 16, 24, 32), window=window)), {"pretrained": False, "synthetic": True}
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
    return attach_memory(NativeStudent(OfficialStages(original), window=window)), {"pretrained": True, "weights_sha256": expected,
            "source_commit": head, "lineage": "user_supplied_hash_pinned;not_independent_upstream_certification"}
