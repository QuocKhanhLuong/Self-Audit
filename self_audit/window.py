"""Small, differentiable semantic-support windows for the native decoder.

The sampler in this module deliberately avoids ``grid_sample``.  Its gather +
bilinear arithmetic has the same first-order values in the interior of a cell
and is usable by the second-order/finite-difference checks needed by the
meta-window pilot.  A sampled point is considered valid only when its complete
bilinear footprint is inside the supplied geometric-validity mask.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


WINDOW_DEFAULT = {
    "mode": "none",
    "dim": 32,
    "radius_min": 0.5,
    "radius_max": 4.0,
    "offset_limit": 2.0,
    "initial_radius": 1.0,
    "gamma_init": 1e-3,
}

_WINDOW_MODES = {"none", "fixed", "adaptive"}


def validate_window_config(config=None):
    """Return a validated window configuration without mutating its input.

    ``None`` means the exact native decoder path.  A ``mode='none'`` mapping is
    also accepted so a full experiment config can carry an explicit disabled
    arm.  The pilot intentionally keeps the parameter surface small: K=9 is a
    fixed architectural choice, while these values are the pre-registered
    geometry and residual-gate settings.
    """

    if config is None:
        return copy.deepcopy(WINDOW_DEFAULT)
    if not isinstance(config, dict):
        raise ValueError("window config must be a mapping or None")
    unknown = set(config) - set(WINDOW_DEFAULT)
    if unknown:
        raise ValueError(f"unknown window config fields: {sorted(unknown)}")
    out = copy.deepcopy(WINDOW_DEFAULT)
    out.update(config)
    if not isinstance(out["mode"], str) or out["mode"] not in _WINDOW_MODES:
        raise ValueError("window mode must be none, fixed, or adaptive")
    if type(out["dim"]) is not int or out["dim"] <= 0:
        raise ValueError("window dim must be a positive integer")
    for key in ("radius_min", "radius_max", "offset_limit", "initial_radius", "gamma_init"):
        value = out[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"invalid window {key}")
        out[key] = float(value)
    if out["radius_min"] <= 0 or out["radius_max"] <= out["radius_min"]:
        raise ValueError("window radii must satisfy 0 < radius_min < radius_max")
    if not 0 <= out["offset_limit"]:
        raise ValueError("window offset_limit must be nonnegative")
    if not out["radius_min"] < out["initial_radius"] < out["radius_max"]:
        raise ValueError("window initial_radius must lie strictly inside the radius bounds")
    if out["mode"] == "adaptive" and out["gamma_init"] <= 0:
        raise ValueError("adaptive window gamma_init must be positive")
    return out


def _as_valid_mask(mask: Optional[torch.Tensor], batch: int, height: int, width: int,
                   *, device: torch.device) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.ndim != 3 or mask.shape != (batch, height, width) or mask.dtype is not torch.bool:
        raise ValueError("geometric valid mask must be a bool BxHxW tensor")
    return mask.to(device=device)


def _gather_2d(feature: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gather BxC values at integer BxHq xWq xK coordinates."""

    b, c, h, w = feature.shape
    indices = (y * w + x).reshape(b, 1, -1).expand(-1, c, -1)
    values = torch.gather(feature.reshape(b, c, h * w), 2, indices)
    return values.reshape(b, c, *x.shape[1:])


def bilinear_sample(feature: torch.Tensor, coordinates: torch.Tensor,
                    valid_mask: Optional[torch.Tensor] = None):
    """Sample ``feature`` at pixel coordinates ``(x, y)``.

    Args:
        feature: ``B,C,H,W`` tensor.
        coordinates: ``B,Hq,Wq,K,2`` pixel coordinates in the feature grid.
        valid_mask: optional bool ``B,H,W`` geometric mask.  The returned
            validity is false unless all four points of the bilinear footprint
            are in bounds and valid.

    Returns:
        ``(values, valid)`` where values are ``B,C,Hq,Wq,K``.  Invalid values
        are zeroed, which makes the all-invalid attention fallback finite.
    """

    if feature.ndim != 4:
        raise ValueError("feature must be BxCxHxW")
    if coordinates.ndim != 5 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must be BxHq xWq xK x2")
    b, _, h, w = feature.shape
    if coordinates.shape[0] != b:
        raise ValueError("coordinate batch does not match feature batch")
    mask = _as_valid_mask(valid_mask, b, h, w, device=feature.device)
    # floor is piecewise constant; the bilinear weights retain the useful
    # coordinate derivative inside each cell.  Clamp only gather indices --
    # validity separately rejects an incomplete/out-of-bounds footprint.
    x, y = coordinates[..., 0], coordinates[..., 1]
    finite = torch.isfinite(x) & torch.isfinite(y)
    # Do not let NaN/Inf reach integer index conversion. Their gathered value
    # is discarded by ``finite`` below, so this replacement is only a
    # fail-closed indexing sentinel.
    safe_x = torch.where(finite, x, torch.zeros_like(x))
    safe_y = torch.where(finite, y, torch.zeros_like(y))
    x0_raw, y0_raw = torch.floor(safe_x), torch.floor(safe_y)
    x1_raw, y1_raw = x0_raw + 1, y0_raw + 1
    x0 = x0_raw.clamp(0, max(w - 1, 0)).long()
    x1 = x1_raw.clamp(0, max(w - 1, 0)).long()
    y0 = y0_raw.clamp(0, max(h - 1, 0)).long()
    y1 = y1_raw.clamp(0, max(h - 1, 0)).long()
    wx, wy = safe_x - x0_raw, safe_y - y0_raw

    v00 = _gather_2d(feature, x0, y0)
    v01 = _gather_2d(feature, x1, y0)
    v10 = _gather_2d(feature, x0, y1)
    v11 = _gather_2d(feature, x1, y1)
    values = (v00 * (1 - wx).unsqueeze(1) * (1 - wy).unsqueeze(1)
              + v01 * wx.unsqueeze(1) * (1 - wy).unsqueeze(1)
              + v10 * (1 - wx).unsqueeze(1) * wy.unsqueeze(1)
              + v11 * wx.unsqueeze(1) * wy.unsqueeze(1))

    in_bounds = finite & (x >= 0) & (x <= max(w - 1, 0)) & (y >= 0) & (y <= max(h - 1, 0))
    if mask is not None:
        m00 = _gather_2d(mask[:, None].to(dtype=feature.dtype), x0, y0)[:, 0].bool()
        m01 = _gather_2d(mask[:, None].to(dtype=feature.dtype), x1, y0)[:, 0].bool()
        m10 = _gather_2d(mask[:, None].to(dtype=feature.dtype), x0, y1)[:, 0].bool()
        m11 = _gather_2d(mask[:, None].to(dtype=feature.dtype), x1, y1)[:, 0].bool()
        in_bounds = in_bounds & m00 & m01 & m10 & m11
    values = values * in_bounds.unsqueeze(1).to(values.dtype)
    return values, in_bounds


def _inverse_sigmoid(value: float) -> float:
    if not 0 < value < 1:
        raise ValueError("initial radius cannot be represented inside the configured bounds")
    return math.log(value / (1.0 - value))


class _Geometry(nn.Module):
    """Zero-centered center/radius predictor with a bounded output."""

    def __init__(self, channels: int, *, radius_min: float,
                 radius_max: float, initial_radius: float):
        super().__init__()
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.offset = nn.Conv2d(channels, 4, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        ratio = (initial_radius - radius_min) / (radius_max - radius_min)
        # Do not silently clamp the requested radius: that would give fixed
        # and adaptive comparators different initial sampling locations.
        with torch.no_grad():
            self.offset.bias[2:].fill_(_inverse_sigmoid(ratio))

    def forward(self, query: torch.Tensor, offset_limit: float):
        raw = self.offset(query)
        center = raw[:, :2].tanh() * offset_limit
        radius = self.radius_min + (self.radius_max - self.radius_min) * raw[:, 2:].sigmoid()
        return center, radius


class SemanticWindow(nn.Module):
    """K=9 coarse-to-fine semantic support reader for NativeStudent.

    ``query`` is the stride-4 skip plus the upsampled stride-8 decoder feature;
    ``coarse`` is the stride-8 decoder feature.  The output is a stride-4
    residual with the same channel count as ``query``.
    """

    K = 9

    def __init__(self, query_dim: int, coarse_dim: int, config=None):
        super().__init__()
        cfg = validate_window_config(config)
        if cfg["mode"] == "none":
            raise ValueError("SemanticWindow requires fixed or adaptive mode")
        self.mode = cfg["mode"]
        self.embed_dim = cfg["dim"]
        self.offset_limit = cfg["offset_limit"]
        self.initial_radius = cfg["initial_radius"]
        self.radius_min = cfg["radius_min"]
        self.radius_max = cfg["radius_max"]
        self.to_q = nn.Conv2d(query_dim, self.embed_dim, 1)
        self.to_k = nn.Conv2d(coarse_dim, self.embed_dim, 1)
        self.to_v = nn.Conv2d(coarse_dim, self.embed_dim, 1)
        self.to_out = nn.Conv2d(self.embed_dim, query_dim, 1)
        self.geometry = _Geometry(query_dim, radius_min=self.radius_min,
                                  radius_max=self.radius_max, initial_radius=self.initial_radius)
        if self.mode == "fixed":
            self.set_geometry_requires_grad(False)
        self.gamma = nn.Parameter(torch.tensor(float(cfg["gamma_init"])))
        points = torch.tensor([[-1., -1.], [-1., 0.], [-1., 1.],
                               [0., -1.], [0., 0.], [0., 1.],
                               [1., -1.], [1., 0.], [1., 1.]])
        self.register_buffer("canonical_points", points, persistent=False)

    def set_geometry_requires_grad(self, enabled: bool):
        for parameter in self.geometry.parameters():
            parameter.requires_grad_(bool(enabled))

    @property
    def geometry_parameters(self):
        return tuple(self.geometry.parameters())

    def _coordinates(self, query: torch.Tensor, coarse: torch.Tensor):
        b, _, hq, wq = query.shape
        _, _, hc, wc = coarse.shape
        ys = torch.arange(hq, device=query.device, dtype=query.dtype)
        xs = torch.arange(wq, device=query.device, dtype=query.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack(((xx + .5) * (wc / wq) - .5,
                            (yy + .5) * (hc / hq) - .5), dim=-1)
        base = base.view(1, hq, wq, 1, 2)
        if self.mode == "fixed":
            center = query.new_zeros((b, hq, wq, 2))
            radius = query.new_full((b, hq, wq, 2), self.initial_radius)
        else:
            center, radius = self.geometry(query, self.offset_limit)
            center = center.permute(0, 2, 3, 1)
            radius = radius.permute(0, 2, 3, 1)
        return base + center.unsqueeze(-2) + radius.unsqueeze(-2) * self.canonical_points.to(query).view(1, 1, 1, self.K, 2)

    def forward(self, query: torch.Tensor, coarse: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None):
        if query.ndim != 4 or coarse.ndim != 4:
            raise ValueError("window query and coarse features must be BxCxHxW")
        if query.shape[0] != coarse.shape[0] or query.shape[2] < 1 or query.shape[3] < 1:
            raise ValueError("window feature shapes are incompatible")
        b, _, hq, wq = query.shape
        valid = _as_valid_mask(valid_mask, b, hq, wq, device=query.device)
        if valid is None:
            coarse_valid = None
            query_valid = None
        else:
            # Feature stages are exactly stride two here.  Area interpolation
            # keeps a coarse cell valid only when its entire fine footprint is
            # geometric image support; labels never enter this mask.
            hc, wc = coarse.shape[-2:]
            coarse_valid = F.interpolate(valid[:, None].float(), size=(hc, wc), mode="area").ge(1.0 - 1e-6)[:, 0]
            query_valid = valid
        coordinates = self._coordinates(query, coarse)
        sampled_k, sample_valid = bilinear_sample(self.to_k(coarse), coordinates, coarse_valid)
        sampled_v, _ = bilinear_sample(self.to_v(coarse), coordinates, coarse_valid)
        q = self.to_q(query)
        logits = (q.unsqueeze(-1) * sampled_k).sum(1) / math.sqrt(self.embed_dim)
        safe = logits.masked_fill(~sample_valid, float("-inf"))
        has_valid = sample_valid.any(-1)
        safe = torch.where(has_valid.unsqueeze(-1), safe, torch.zeros_like(safe))
        weights = F.softmax(safe, dim=-1) * sample_valid.to(logits.dtype)
        context = (sampled_v * weights.unsqueeze(1)).sum(-1)
        residual = self.to_out(context) * self.gamma
        # A projected bias must not reintroduce a residual when every sampled
        # footprint is invalid (this also covers out-of-bounds fixed windows
        # when no explicit image mask was supplied).
        residual = residual * has_valid[:, None].to(residual.dtype)
        if query_valid is not None:
            residual = residual * query_valid[:, None].to(residual.dtype)
        return residual


__all__ = ["WINDOW_DEFAULT", "validate_window_config", "bilinear_sample", "SemanticWindow"]
