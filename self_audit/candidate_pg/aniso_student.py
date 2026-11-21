"""AnisoStudent: in-plane convolutional encoder/decoder with through-plane attention only at the deepest stage.

ACDC short-axis volumes are strongly anisotropic (about 10 mm slices, 1.5 mm in plane), so the stages E0-E3 never mix or down-sample along z
(kernels ``1x3x3``; one ``3x1x1`` mix at E3), and E4 adds a small Z-Transformer: self-attention along z only, for every (h, w) cell, with a learned
relative slice-position bias.  Same input/output contract as :class:`PGStudent` (``forward(image, geometry, ...) -> {"logits", "features", ...}``,
``phases``, ``attach_code_branch``), so the engine switches with ``student.arch = "aniso"``.  No z down-sampling anywhere.  Train-only outputs:
``sdf`` (3 channels, tanh) and ``aux_logits`` (deep supervision at the H/2 and H/4 decoder stages).  The stage-3 tap exposed for the code branch is
a 1x1x1 projection of E3 to the 128 channels the existing :class:`StudentAtomBranch` expects.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..candidate_fuse.student import FuseStudent, _shape
from ..candidate_q3d.alignment import GridPhase, grid_phase
from ..candidate_q3d.student import _ConvNormAct

ANISO_CHANNELS = (24, 48, 96, 160, 192)
MAX_SLICES = 32


class ZAttentionLayer(nn.Module):
    """Pre-norm self-attention along z (+ MLP) with a learned relative-position bias; padded slices are masked as keys."""

    def __init__(self, dim: int = 192, heads: int = 4, mlp_ratio: int = 2, max_slices: int = MAX_SLICES) -> None:
        super().__init__()
        self.dim, self.heads, self.max_slices = dim, heads, max_slices
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(torch.zeros(heads, 2 * max_slices - 1))
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x: Tensor, valid: Tensor | None = None) -> Tensor:
        """``x`` ``[N, D, C]`` (N = B*H*W cells), ``valid`` ``[N, D]`` bool (False = padded slice)."""
        n, d, c = x.shape
        if d > self.max_slices:
            raise ValueError(f"at most {self.max_slices} slices are supported")
        h = self.heads
        q, k, v = self.qkv(self.norm1(x)).reshape(n, d, 3, h, c // h).permute(2, 0, 3, 1, 4)
        idx = torch.arange(d, device=x.device)
        bias = self.rel_bias[:, (idx[:, None] - idx[None, :]) + self.max_slices - 1]  # [h, d, d]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(c // h) + bias[None]
        if valid is not None:
            scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        attn = scores.softmax(-1)
        if valid is not None:  # a fully padded row would be nan; padded queries are zeroed instead
            attn = torch.nan_to_num(attn, nan=0.0)
        x = x + self.proj((attn @ v).transpose(1, 2).reshape(n, d, c))
        x = x + self.mlp(self.norm2(x))
        return x if valid is None else x * valid[..., None].to(x.dtype)


class ZTransformer(nn.Module):
    def __init__(self, dim: int = 192, layers: int = 2, heads: int = 4, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(ZAttentionLayer(dim, heads, mlp_ratio) for _ in range(layers))

    def forward(self, x: Tensor, slice_valid: Tensor | None = None) -> Tensor:
        """``x`` ``[B,C,D,H,W]``; ``slice_valid`` ``[B,D]`` bool."""
        b, c, d, h, w = x.shape
        tokens = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, d, c)
        valid = None if slice_valid is None else slice_valid[:, None, None, :].expand(b, h, w, d).reshape(b * h * w, d)
        for layer in self.layers:
            tokens = layer(tokens, valid)
        return tokens.reshape(b, h, w, d, c).permute(0, 4, 3, 1, 2).contiguous()


def _conv(in_ch: int, out_ch: int, kernel=(1, 3, 3), stride=(1, 1, 1)) -> nn.Module:
    return _ConvNormAct(in_ch, out_ch, kernel_size=kernel, stride=stride)


class AnisoStudent(nn.Module):
    def __init__(self, config: Mapping[str, Any] | None = None, *, channels: tuple[int, ...] = ANISO_CHANNELS, deploy: bool = False) -> None:
        super().__init__()
        self.deploy = bool(deploy)
        model = (config or {}).get("model", config or {})
        if int((model.get("student", {}) if isinstance(model, Mapping) else {}).get("num_classes", 4)) != 4:
            raise ValueError("AnisoStudent requires four classes")
        c0, c1, c2, c3, c4 = channels
        self.channels = channels
        self.e0 = nn.Sequential(_conv(1, c0), _conv(c0, c0))
        self.e1 = nn.Sequential(_conv(c0, c1, stride=(1, 2, 2)), _conv(c1, c1))
        self.e2 = nn.Sequential(_conv(c1, c2, stride=(1, 2, 2)), _conv(c2, c2))
        self.e3 = nn.Sequential(_conv(c2, c3, stride=(1, 2, 2)), _conv(c3, c3, kernel=(3, 1, 1)), _conv(c3, c3))
        self.e4 = nn.Sequential(_conv(c3, c4, stride=(1, 2, 2)), _conv(c4, c4))
        self.z_transformer = ZTransformer(c4)
        self.up4, self.dec4 = nn.ConvTranspose3d(c4, c3, (1, 2, 2), (1, 2, 2)), _conv(2 * c3, c3)
        self.up3, self.dec3 = nn.ConvTranspose3d(c3, c2, (1, 2, 2), (1, 2, 2)), _conv(2 * c2, c2)
        self.up2, self.dec2 = nn.ConvTranspose3d(c2, c1, (1, 2, 2), (1, 2, 2)), _conv(2 * c1, c1)
        self.up1, self.dec1 = nn.ConvTranspose3d(c1, c0, (1, 2, 2), (1, 2, 2)), _conv(2 * c0, c0)
        self.logits_head = nn.Conv3d(c0, 4, 1)
        if not deploy:  # train-only modules are absent from the deployed (student-only) model
            self.sdf_head = nn.Conv3d(c0, 3, 1)
            self.aux_heads = nn.ModuleList([nn.Conv3d(c2, 4, 1), nn.Conv3d(c1, 4, 1)])  # D3 (H/4) and D2 (H/2)
            self.stage3_tap = nn.Conv3d(c3, 128, 1)  # feature for the code branch (stage-3 grid, 128 channels like PGStudent)
        self.teacher_phases = ((( 1, 1, 1), (0., 0., 0.)),) * 5
        convs = [self.e0[0].conv, self.e1[0].conv, self.e2[0].conv, self.e3[0].conv, self.e4[0].conv]
        self.phases = tuple(grid_phase(tuple(convs[: i + 1])) for i in range(5))
        self.code_branch: nn.Module | None = None

    def attach_code_branch(self, branch: nn.Module) -> "AnisoStudent":
        if not isinstance(branch, nn.Module):
            raise TypeError("code_branch must be a torch module")
        self.code_branch = branch
        return self

    _pad = staticmethod(FuseStudent._pad)
    _valid_centers = staticmethod(FuseStudent._valid_centers)

    def forward(self, image: Tensor, geometry: Any, *, teacher_features=None, keep_mask=None, skip_dropout: float = 0.0,
                return_stage3: bool = False, return_code: bool = False, dropout_generator: torch.Generator | None = None,
                slice_valid: Tensor | None = None) -> dict[str, Any]:
        if teacher_features is not None or keep_mask is not None:
            raise ValueError("AnisoStudent has no teacher-feature fusion path")
        if image.ndim != 5 or image.shape[1] != 1 or not image.is_floating_point() or not torch.isfinite(image).all():
            raise ValueError("input must be finite floating [B,1,D,H,W]")
        native = _shape(geometry, image)
        x = image[..., : native[1], : native[2]]
        if torch.any(x < 0) or torch.any(x > 1):
            raise ValueError("image must be normalized to [0,1]")
        x, (nh, nw) = self._pad(x)
        f0 = self.e0(x)
        f1 = self.e1(f0)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        f4 = self.z_transformer(self.e4(f3), slice_valid)

        def drop(t: Tensor) -> Tensor:
            return PGDrop(t, skip_dropout, self.training, dropout_generator)

        d3 = self.dec4(torch.cat([self.up4(drop(f4)), drop(f3)], 1))
        d2 = self.dec3(torch.cat([self.up3(d3), drop(f2)], 1))
        d1 = self.dec2(torch.cat([self.up2(d2), drop(f1)], 1))
        d0 = self.dec1(torch.cat([self.up1(d1), drop(f0)], 1))
        logits = self.logits_head(d0)[..., :nh, :nw]
        out: dict[str, Any] = {"logits": logits, "features": (f0, f1, f2, f3, f4), "gate_means": [0.0] * 5, "mimic_loss": logits.sum() * 0.0}
        if self.training and not self.deploy:
            out["sdf"] = torch.tanh(self.sdf_head(d0))[..., :nh, :nw]
            out["aux_logits"] = [F.interpolate(h(d), size=logits.shape[-3:], mode="trilinear", align_corners=False)
                                 for h, d in zip(self.aux_heads, (d2, d1))]
        if (return_stage3 or (return_code and self.code_branch is not None)) and not self.deploy:
            out["stage3"] = self.stage3_tap(f3)
        if self.training and return_code and self.code_branch is not None:
            valid = self._valid_centers(native, self.phases[3], out["stage3"].shape[-3:], out["stage3"].device)
            out["code_branch"] = self.code_branch(out["stage3"], valid_mask=valid.unsqueeze(0).expand(out["stage3"].shape[0], -1, -1, -1))
        return out

    def infer_logits(self, image: Tensor, geometry: Any) -> Tensor:
        return self.forward(image, geometry)["logits"]


def PGDrop(value: Tensor, probability: float, training: bool, generator: torch.Generator | None) -> Tensor:
    from .student_pg import PGStudent

    return PGStudent._skip_dropout(value, probability, training, generator)


TRAIN_ONLY_PREFIXES = ("sdf_head.", "aux_heads.", "stage3_tap.", "code_branch.")


def export_aniso_student(model: AnisoStudent, path: Any, *, epoch: int | None = None) -> dict[str, Any]:
    """Student-only artifact: train-only heads and any code branch are stripped; the deployed model is built with ``deploy=True``."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith(TRAIN_ONLY_PREFIXES)}
    torch.save({"schema": "self_audit.aniso_student.export.v1", "channels": list(model.channels), "state": state, "selected_epoch": epoch}, path)
    return {"path": str(path), "parameters_deployed": sum(v.numel() for v in state.values())}


def load_aniso_export(path: Any) -> AnisoStudent:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = AnisoStudent(channels=tuple(payload["channels"]), deploy=True)
    model.load_state_dict(payload["state"], strict=True)
    return model.eval()


def build_aniso_student(config: Mapping[str, Any] | None = None) -> AnisoStudent:
    return AnisoStudent(config)


def param_breakdown(model: nn.Module) -> dict[str, int]:
    rows = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
    rows["total"] = sum(p.numel() for p in model.parameters())
    return rows


__all__ = ["export_aniso_student", "load_aniso_export", "AnisoStudent", "ZTransformer", "ZAttentionLayer", "build_aniso_student", "param_breakdown", "ANISO_CHANNELS"]
