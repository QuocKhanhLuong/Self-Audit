"""Operational, motion-aligned temporal consistency for native volumes (T00/T01).

This module implements support-aware, bounded 3D phase-correlation registration,
non-circular inverse warping with conservative interpolation support,
forward-backward occlusion detection with mapped-coordinate sampling, and
differentiable temporal consistency objectives across adjacent cardiac cine phases.

Key scientific contracts:
1. Spatial Z vs. Temporal T: All volume axes are spatial Z. Adjacent temporal
   phases are distinct cine volumes at t and t +/- 1.
2. Support-aware registration: Invalid background/padding textures are strictly
   masked out before cross-spectrum estimation, preventing invalid motion contamination.
3. Explicit physical bounds vs. clipping: Detected motions exceeding configured
   physical bounds are rejected as out-of-bounds/unobservable, never artificially
   clipped to boundaries or falsely certified by forward-backward checks.
4. Conservative interpolation support: Warped validity uses bilinear support
   thresholding (> 0.999), strictly rejecting voxels whose interpolation footprint
   touches invalid neighbors.
5. Forward-backward roundtrip consistency: Backward fields are sampled at forward-mapped
   coordinates to detect non-rigid deformation, out-of-bounds translation, and occlusion.
6. Detached teacher & registration: Neither registration fields nor teacher
   probabilities receive gradients; loss is differentiable strictly through
   current student logits.
7. Limited translational backend: Phase-correlation estimates bounded rigid
   translation with optional slicewise in-plane refinement. This provides motion
   consistency and alignment across adjacent cine frames, but is NOT a general
   non-rigid deformable cardiac registration backend. Unobservable constant or
   flat textures are rejected before apodization.
8. No U GT mask access: Registration and consistency are completely self-supervised
   from raw cine images and detached predictions. No GT masks are copied or read.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class TemporalConfig:
    """Configuration for temporal consistency."""

    mode: str = "none"  # "none" | "identity" | "motion" (aliases: "motion_aligned", "identity_diagnostic", "off")
    weight: float = 0.1
    max_disp_xyz: tuple[float, float, float] = (4.0, 16.0, 16.0)  # max displacement bound in voxels (Z, Y, X)
    consistency_threshold: float = 1.5  # forward-backward consistency check threshold in voxels for occlusion
    loss_type: str = "mse"  # "mse" | "kl"
    slice_wise: bool = True  # estimate per-slice in-plane motion to capture through-plane differential contraction
    window: str = "none"  # "none" | "hanning" | "tukey"

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> TemporalConfig:
        if cfg is None:
            return cls()
        if isinstance(cfg, TemporalConfig):
            return copy.deepcopy(cfg)
        d = dict(cfg)
        raw_mode = str(d.get("mode", "none")).lower()
        if raw_mode in {"motion_aligned", "t01", "phase_correlation", "registered", "motion"}:
            mode = "motion"
        elif raw_mode in {"identity_diagnostic", "diag", "identity"}:
            mode = "identity"
        elif raw_mode in {"off", "spatial_only", "t00", "none"}:
            mode = "none"
        else:
            raise ValueError(
                f"Unsupported temporal mode '{raw_mode}'. Supported modes are: "
                "'none', 'identity', 'motion'. No silent fallback is permitted."
            )
        d["mode"] = mode
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class RegistrationResult:
    """Detached registration displacement field and validation metadata."""

    flow_xyz: Tensor  # (B, 3, Z, H, W) float32 displacement in voxels: [u_x, u_y, u_z]
    occlusion_mask: Tensor  # (B, Z, H, W) bool, True where correspondence is occluded/inconsistent
    in_bounds_mask: Tensor  # (B, Z, H, W) bool, True where warped coordinates lie within grid and bounds
    metrics: dict[str, float] = field(default_factory=dict)


def _hanning_3d(z: int, y: int, x: int, device: torch.device) -> Tensor:
    """Create a 3D separable Hanning window."""
    wz = torch.hann_window(z, periodic=False, device=device) if z > 1 else torch.ones(1, device=device)
    wy = torch.hann_window(y, periodic=False, device=device) if y > 1 else torch.ones(1, device=device)
    wx = torch.hann_window(x, periodic=False, device=device) if x > 1 else torch.ones(1, device=device)
    return wz[:, None, None] * wy[None, :, None] * wx[None, None, :]


def _make_window_2d(h: int, w: int, device: torch.device, window_type: str = "none") -> Tensor:
    if window_type == "hanning":
        wy = torch.hann_window(h, periodic=False, device=device) if h > 1 else torch.ones(1, device=device)
        wx = torch.hann_window(w, periodic=False, device=device) if w > 1 else torch.ones(1, device=device)
        return wy[:, None] * wx[None, :]
    elif window_type == "tukey":
        from scipy.signal.windows import tukey

        wy = torch.as_tensor(tukey(h, alpha=0.15), dtype=torch.float32, device=device)
        wx = torch.as_tensor(tukey(w, alpha=0.15), dtype=torch.float32, device=device)
        return wy[:, None] * wx[None, :]
    return torch.ones((h, w), device=device)


def _make_window_3d(z: int, h: int, w: int, device: torch.device, window_type: str = "none") -> Tensor:
    if window_type == "hanning":
        return _hanning_3d(z, h, w, device)
    elif window_type == "tukey":
        from scipy.signal.windows import tukey

        wz = (
            torch.as_tensor(tukey(z, alpha=0.15), dtype=torch.float32, device=device)
            if z > 1
            else torch.ones(1, device=device)
        )
        wy = torch.as_tensor(tukey(h, alpha=0.15), dtype=torch.float32, device=device)
        wx = torch.as_tensor(tukey(w, alpha=0.15), dtype=torch.float32, device=device)
        return wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    return torch.ones((z, h, w), device=device)


def _apodize_support_3d(val: Tensor) -> Tensor:
    """Taper binary support mask boundaries to avoid rectangular edge artifacts in FFT."""
    val_float = val.float().unsqueeze(0).unsqueeze(0)
    kz = 3 if val.shape[0] >= 3 else 1
    pz = kz // 2
    blurred = F.avg_pool3d(val_float, kernel_size=(kz, 5, 5), stride=1, padding=(pz, 2, 2))
    return (blurred * val_float).squeeze(0).squeeze(0)


def _apodize_support_2d(val: Tensor) -> Tensor:
    """Taper binary support mask boundaries to avoid rectangular edge artifacts in FFT."""
    val_float = val.float().unsqueeze(0).unsqueeze(0)
    blurred = F.avg_pool2d(val_float, kernel_size=5, stride=1, padding=2)
    return (blurred * val_float).squeeze(0).squeeze(0)


def _phase_correlation_2d_support_aware(
    img1: Tensor,
    img2: Tensor,
    val1: Tensor,
    val2: Tensor,
    max_disp_y: float,
    max_disp_x: float,
    window: str = "none",
) -> tuple[float, float, bool]:
    """Compute subpixel 2D phase-correlation translation (dy, dx, in_bounds) with support awareness."""
    h, w = img1.shape
    device = img1.device

    v1_sum = int(val1.sum().item())
    v2_sum = int(val2.sum().item())
    if v1_sum == 0 or v2_sum == 0:
        return 0.0, 0.0, False

    # Reject unobservable valid image content BEFORE apodization (constant / zero variance)
    v1_pixels = img1[val1]
    v2_pixels = img2[val2]
    if v1_pixels.numel() < 2 or v2_pixels.numel() < 2:
        return 0.0, 0.0, False
    std1 = float(torch.std(v1_pixels.float()).item())
    std2 = float(torch.std(v2_pixels.float()).item())
    if std1 < 1e-4 or std2 < 1e-4:
        return 0.0, 0.0, False

    # Mask out invalid pixels with soft apodized support to prevent edge artifacts
    apod1 = _apodize_support_2d(val1)
    apod2 = _apodize_support_2d(val2)
    i1_clean = img1 * apod1
    i2_clean = img2 * apod2

    w2d = _make_window_2d(h, w, device, window)
    f1 = torch.fft.rfftn(i1_clean * w2d)
    f2 = torch.fft.rfftn(i2_clean * w2d)

    cross = f2 * torch.conj(f1)
    cross = cross / (torch.abs(cross) + 1e-7)

    corr = torch.fft.irfftn(cross, s=(h, w))
    corr = torch.fft.fftshift(corr)

    cy, cx = h // 2, w // 2
    max_idx = int(torch.argmax(corr).item())
    py = int(max_idx // w)
    px = int(max_idx % w)

    raw_dy = float(py - cy)
    raw_dx = float(px - cx)

    # Reject motions exceeding configured physical bounds instead of clipping
    if abs(raw_dy) > max_disp_y or abs(raw_dx) > max_disp_x:
        return raw_dy, raw_dx, False

    dy = raw_dy
    dx = raw_dx

    # Subpixel quadratic peak refinement
    if 0 < py < h - 1:
        c_m1 = float(corr[py - 1, px].item())
        c_0 = float(corr[py, px].item())
        c_p1 = float(corr[py + 1, px].item())
        denom = 2.0 * (2.0 * c_0 - c_m1 - c_p1)
        if abs(denom) > 1e-6:
            dy += (c_p1 - c_m1) / denom

    if 0 < px < w - 1:
        c_m1 = float(corr[py, px - 1].item())
        c_0 = float(corr[py, px].item())
        c_p1 = float(corr[py, px + 1].item())
        denom = 2.0 * (2.0 * c_0 - c_m1 - c_p1)
        if abs(denom) > 1e-6:
            dx += (c_p1 - c_m1) / denom

    in_bounds = (abs(dy) <= max_disp_y + 0.5) and (abs(dx) <= max_disp_x + 0.5)
    return dy, dx, in_bounds


def _phase_correlation_3d_support_aware(
    img1: Tensor,
    img2: Tensor,
    val1: Tensor,
    val2: Tensor,
    max_disp_z: float,
    max_disp_y: float,
    max_disp_x: float,
    window: str = "none",
) -> tuple[float, float, float, bool]:
    """Compute subpixel 3D phase-correlation translation (dz, dy, dx, in_bounds) with support awareness."""
    z, h, w = img1.shape
    device = img1.device

    v1_sum = int(val1.sum().item())
    v2_sum = int(val2.sum().item())
    if v1_sum == 0 or v2_sum == 0:
        return 0.0, 0.0, 0.0, False

    # Reject unobservable valid image content BEFORE apodization (constant / zero variance)
    v1_voxels = img1[val1]
    v2_voxels = img2[val2]
    if v1_voxels.numel() < 2 or v2_voxels.numel() < 2:
        return 0.0, 0.0, 0.0, False
    std1 = float(torch.std(v1_voxels.float()).item())
    std2 = float(torch.std(v2_voxels.float()).item())
    if std1 < 1e-4 or std2 < 1e-4:
        return 0.0, 0.0, 0.0, False

    # Mask out invalid pixels with soft apodized support to prevent edge artifacts
    apod1 = _apodize_support_3d(val1)
    apod2 = _apodize_support_3d(val2)
    i1_clean = img1 * apod1
    i2_clean = img2 * apod2

    w3d = _make_window_3d(z, h, w, device, window)
    f1 = torch.fft.rfftn(i1_clean * w3d)
    f2 = torch.fft.rfftn(i2_clean * w3d)

    cross = f2 * torch.conj(f1)
    cross = cross / (torch.abs(cross) + 1e-7)

    corr = torch.fft.irfftn(cross, s=(z, h, w))
    corr = torch.fft.fftshift(corr)

    cz, cy, cx = z // 2, h // 2, w // 2
    max_idx = int(torch.argmax(corr).item())
    pz = max_idx // (h * w)
    rem = max_idx % (h * w)
    py = rem // w
    px = rem % w

    raw_dz = float(pz - cz)
    raw_dy = float(py - cy)
    raw_dx = float(px - cx)

    # Reject out-of-bounds motion instead of clipping
    if abs(raw_dz) > max_disp_z or abs(raw_dy) > max_disp_y or abs(raw_dx) > max_disp_x:
        return raw_dz, raw_dy, raw_dx, False

    dz = raw_dz
    dy = raw_dy
    dx = raw_dx

    # Subpixel quadratic refinement along each axis
    if 0 < pz < z - 1:
        c_m1 = float(corr[pz - 1, py, px].item())
        c_0 = float(corr[pz, py, px].item())
        c_p1 = float(corr[pz + 1, py, px].item())
        denom = 2.0 * (2.0 * c_0 - c_m1 - c_p1)
        if abs(denom) > 1e-6:
            dz += (c_p1 - c_m1) / denom

    if 0 < py < h - 1:
        c_m1 = float(corr[pz, py - 1, px].item())
        c_0 = float(corr[pz, py, px].item())
        c_p1 = float(corr[pz, py + 1, px].item())
        denom = 2.0 * (2.0 * c_0 - c_m1 - c_p1)
        if abs(denom) > 1e-6:
            dy += (c_p1 - c_m1) / denom

    if 0 < px < w - 1:
        c_m1 = float(corr[pz, py, px - 1].item())
        c_0 = float(corr[pz, py, px].item())
        c_p1 = float(corr[pz, py, px + 1].item())
        denom = 2.0 * (2.0 * c_0 - c_m1 - c_p1)
        if abs(denom) > 1e-6:
            dx += (c_p1 - c_m1) / denom

    in_bounds = (
        (abs(dz) <= max_disp_z + 0.5)
        and (abs(dy) <= max_disp_y + 0.5)
        and (abs(dx) <= max_disp_x + 0.5)
    )
    return dz, dy, dx, in_bounds


def register_volumes(
    current_image: Tensor,
    adjacent_image: Tensor,
    current_valid: Tensor | None = None,
    adjacent_valid: Tensor | None = None,
    config: TemporalConfig | Mapping[str, Any] | None = None,
) -> RegistrationResult:
    """Estimate bounded 3D displacement field mapping current coordinates to adjacent.

    Support-aware: strictly ignores invalid background/padding regions.
    Explicitly bounds motion and rejects unobservable/out-of-bounds displacements.
    Forward-backward check samples backward flow at mapped coordinates.
    """
    cfg = TemporalConfig.from_mapping(config)
    b, c_img, z, h, w = current_image.shape
    device = current_image.device

    # Identity diagnostic mode
    if cfg.mode == "identity":
        flow_xyz = torch.zeros((b, 3, z, h, w), dtype=torch.float32, device=device)
        occ = torch.zeros((b, z, h, w), dtype=torch.bool, device=device)
        in_bounds = torch.ones((b, z, h, w), dtype=torch.bool, device=device)
        return RegistrationResult(
            flow_xyz=flow_xyz,
            occlusion_mask=occ,
            in_bounds_mask=in_bounds,
            metrics={"mean_displacement_voxels": 0.0, "is_identity_diagnostic": 1.0},
        )

    center_ch = 1 if c_img == 3 else 0
    i_curr = current_image[:, center_ch].detach()
    i_adj = adjacent_image[:, center_ch].detach()

    c_val = (
        current_valid.detach().to(device=device, dtype=torch.bool)
        if current_valid is not None
        else torch.ones((b, z, h, w), dtype=torch.bool, device=device)
    )
    a_val = (
        adjacent_valid.detach().to(device=device, dtype=torch.bool)
        if adjacent_valid is not None
        else torch.ones((b, z, h, w), dtype=torch.bool, device=device)
    )

    max_z, max_y, max_x = cfg.max_disp_xyz

    flows_list = []
    b_flows_list = []
    in_bounds_list = []
    disp_mags = []

    for bi in range(b):
        c_vol = i_curr[bi]
        a_vol = i_adj[bi]
        c_v = c_val[bi]
        a_v = a_val[bi]

        # 1. Support-aware 3D global volume displacement
        dz_g, dy_g, dx_g, fwd_ok = _phase_correlation_3d_support_aware(
            c_vol, a_vol, c_v, a_v, max_z, max_y, max_x, window=cfg.window
        )
        b_dz_g, b_dy_g, b_dx_g, bwd_ok = _phase_correlation_3d_support_aware(
            a_vol, c_vol, a_v, c_v, max_z, max_y, max_x, window=cfg.window
        )

        global_ok = fwd_ok and bwd_ok

        u_x = torch.full((z, h, w), dx_g if fwd_ok else 0.0, dtype=torch.float32, device=device)
        u_y = torch.full((z, h, w), dy_g if fwd_ok else 0.0, dtype=torch.float32, device=device)
        u_z = torch.full((z, h, w), dz_g if fwd_ok else 0.0, dtype=torch.float32, device=device)

        b_u_x = torch.full((z, h, w), b_dx_g if bwd_ok else 0.0, dtype=torch.float32, device=device)
        b_u_y = torch.full((z, h, w), b_dy_g if bwd_ok else 0.0, dtype=torch.float32, device=device)
        b_u_z = torch.full((z, h, w), b_dz_g if bwd_ok else 0.0, dtype=torch.float32, device=device)

        slice_ok = torch.full((z,), global_ok, dtype=torch.bool, device=device)

        # 2. Refine per-slice in-plane motion with through-plane offset target_zi
        if cfg.slice_wise and z > 1 and global_ok:
            for zi in range(z):
                target_zi = zi + int(round(dz_g))
                if 0 <= target_zi < z:
                    dy_s, dx_s, s_fwd_ok = _phase_correlation_2d_support_aware(
                        c_vol[zi], a_vol[target_zi], c_v[zi], a_v[target_zi], max_y, max_x, window=cfg.window
                    )
                    b_dy_s, b_dx_s, s_bwd_ok = _phase_correlation_2d_support_aware(
                        a_vol[target_zi], c_vol[zi], a_v[target_zi], c_v[zi], max_y, max_x, window=cfg.window
                    )
                    if s_fwd_ok and s_bwd_ok:
                        u_x[zi] = dx_s
                        u_y[zi] = dy_s
                        b_u_x[target_zi] = b_dx_s
                        b_u_y[target_zi] = b_dy_s
                    else:
                        slice_ok[zi] = False
                else:
                    # Slice shifted completely out of bounds in Z
                    slice_ok[zi] = False

        # Grid coordinates for current frame
        zz, yy, xx = torch.meshgrid(
            torch.arange(z, device=device, dtype=torch.float32),
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )

        mapped_x = xx + u_x
        mapped_y = yy + u_y
        mapped_z = zz + u_z

        # Explicit physical bound and spatial grid bound check
        in_b = (
            slice_ok[:, None, None]
            & (mapped_x >= 0.0)
            & (mapped_x <= float(w - 1))
            & (mapped_y >= 0.0)
            & (mapped_y <= float(h - 1))
            & (mapped_z >= 0.0)
            & (mapped_z <= float(z - 1))
            & (torch.abs(u_x) <= max_x)
            & (torch.abs(u_y) <= max_y)
            & (torch.abs(u_z) <= max_z)
        )

        flow_item = torch.stack([u_x, u_y, u_z], dim=0)  # (3, Z, H, W)
        b_flow_item = torch.stack([b_u_x, b_u_y, b_u_z], dim=0)  # (3, Z, H, W)

        flows_list.append(flow_item)
        b_flows_list.append(b_flow_item)
        in_bounds_list.append(in_b)

        disp_mag = torch.sqrt(u_x * u_x + u_y * u_y + u_z * u_z).mean().item()
        disp_mags.append(disp_mag)

    flow_xyz = torch.stack(flows_list, dim=0)  # (B, 3, Z, H, W)
    b_flow_xyz = torch.stack(b_flows_list, dim=0)  # (B, 3, Z, H, W)
    in_bounds_mask = torch.stack(in_bounds_list, dim=0)  # (B, Z, H, W)

    # 3. Forward-backward roundtrip consistency check sampled at mapped coordinates
    # Sample backward flow at coordinates mapped by forward flow
    warped_b_flow = warp_volume_3d(b_flow_xyz, flow_xyz, mode="bilinear", padding_mode="zeros")

    roundtrip_err = torch.sqrt(
        (flow_xyz[:, 0] + warped_b_flow[:, 0]) ** 2
        + (flow_xyz[:, 1] + warped_b_flow[:, 1]) ** 2
        + (flow_xyz[:, 2] + warped_b_flow[:, 2]) ** 2
    )

    occlusion_mask = (roundtrip_err > float(cfg.consistency_threshold)) | (~in_bounds_mask)

    return RegistrationResult(
        flow_xyz=flow_xyz,
        occlusion_mask=occlusion_mask,
        in_bounds_mask=in_bounds_mask,
        metrics={
            "mean_displacement_voxels": float(np.mean(disp_mags)),
            "is_identity_diagnostic": 0.0,
        },
    )


def warp_volume_3d(
    volume: Tensor,
    flow_xyz: Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tensor:
    """Warp a B,C,Z,H,W tensor from adjacent to current frame using flow_xyz [u_x, u_y, u_z].

    Zero-padding is enforced by default to preserve non-circular support.
    """
    b, c, z, h, w = volume.shape
    device = volume.device

    u_x = flow_xyz[:, 0]  # (B, Z, H, W)
    u_y = flow_xyz[:, 1]
    u_z = flow_xyz[:, 2]

    # Normalized grid coordinates in [-1, 1] for grid_sample
    zz, yy, xx = torch.meshgrid(
        torch.arange(z, device=device, dtype=torch.float32),
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # grid_sample expects (x_norm, y_norm, z_norm) in grid[..., 0], grid[..., 1], grid[..., 2]
    norm_x = ((xx[None] + u_x) / max(1.0, float(w - 1))) * 2.0 - 1.0 if w > 1 else torch.zeros_like(u_x)
    norm_y = ((yy[None] + u_y) / max(1.0, float(h - 1))) * 2.0 - 1.0 if h > 1 else torch.zeros_like(u_y)
    norm_z = ((zz[None] + u_z) / max(1.0, float(z - 1))) * 2.0 - 1.0 if z > 1 else torch.zeros_like(u_z)

    grid = torch.stack([norm_x, norm_y, norm_z], dim=-1)  # (B, Z, H, W, 3)

    return F.grid_sample(volume, grid, mode=mode, padding_mode=padding_mode, align_corners=True)


def temporal_consistency(
    student_logits: Tensor,
    teacher_probabilities: Tensor,
    current_image: Tensor,
    adjacent_image: Tensor,
    current_valid: Tensor,
    adjacent_valid: Tensor,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    config: TemporalConfig | Mapping[str, Any] | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Compute differentiable temporal consistency loss on student logits.

    Args:
        student_logits: Logits from student model at current time t (B, C, Z, H, W).
        teacher_probabilities: Detached teacher probabilities at adjacent time t +/- 1 (B, C, Z, H, W).
        current_image: Cine image volume at t (B, 3, Z, H, W) or (B, 1, Z, H, W).
        adjacent_image: Cine image volume at t +/- 1.
        current_valid: Geometric validity mask at t (B, Z, H, W).
        adjacent_valid: Geometric validity mask at t +/- 1 (B, Z, H, W).
        metadata: Optional sequence of metadata dicts.
        config: TemporalConfig or configuration mapping.

    Returns:
        loss: Scalar differentiable loss on student_logits.
        metrics: Dictionary of diagnostic metrics.
    """
    cfg = TemporalConfig.from_mapping(config)
    device = student_logits.device
    zero_loss = torch.tensor(0.0, device=device, requires_grad=True)

    metrics: dict[str, Any] = {
        "temporal_loss": 0.0,
        "temporal_valid_voxels": 0,
        "temporal_valid_fraction": 0.0,
        "mean_displacement_voxels": 0.0,
        "mean_displacement_mm": 0.0,
        "occlusion_fraction": 0.0,
        "mode": cfg.mode,
        "is_identity_diagnostic": cfg.mode == "identity",
    }

    if cfg.mode in {"none", "off"} or cfg.weight <= 0.0:
        return zero_loss, metrics

    # Detach teacher probabilities and registration inputs
    t_prob = teacher_probabilities.detach().to(device)
    c_img = current_image.detach().to(device)
    a_img = adjacent_image.detach().to(device)
    c_val = current_valid.to(device=device, dtype=torch.bool)
    a_val = adjacent_valid.to(device=device, dtype=torch.bool)

    # Run support-aware 3D registration or identity diagnostic
    reg_result = register_volumes(
        c_img,
        a_img,
        current_valid=c_val,
        adjacent_valid=a_val,
        config=cfg,
    )

    flow_xyz = reg_result.flow_xyz.detach()
    occlusion_mask = reg_result.occlusion_mask.detach()
    in_bounds_mask = reg_result.in_bounds_mask.detach()

    # Warp teacher probabilities to current frame coordinates
    warped_teacher_prob = warp_volume_3d(t_prob, flow_xyz, mode="bilinear", padding_mode="zeros").detach()

    # Conservative validity support:
    # Warp binary adjacent mask with bilinear interpolation; any pixel whose footprint touches an invalid
    # neighbor (< 0.999) is strictly rejected to prevent invalid neighbor contamination.
    a_val_float = a_val.unsqueeze(1).float()
    warped_adj_val = warp_volume_3d(a_val_float, flow_xyz, mode="bilinear", padding_mode="zeros").squeeze(1)
    is_conservative_valid = warped_adj_val > 0.999

    temporal_valid = c_val & is_conservative_valid & in_bounds_mask & (~occlusion_mask)
    n_valid = int(temporal_valid.sum().item())
    total_voxels = int(temporal_valid.numel())

    metrics["temporal_valid_voxels"] = n_valid
    metrics["temporal_valid_fraction"] = float(n_valid / max(1, total_voxels))
    metrics["mean_displacement_voxels"] = reg_result.metrics.get("mean_displacement_voxels", 0.0)
    metrics["occlusion_fraction"] = float(occlusion_mask.float().mean().item())

    # Physical displacement in mm if spacing is available
    if metadata is not None:
        if len(metadata) < student_logits.shape[0]:
            raise ValueError(
                f"Expected metadata length >= batch size {student_logits.shape[0]}, got {len(metadata)}"
            )
        disp_mm_list = []
        for bi in range(student_logits.shape[0]):
            m = metadata[bi]
            if not isinstance(m, Mapping):
                raise ValueError(f"Metadata entry {bi} must be a mapping, got {type(m).__name__}")
            sp = m.get("spacing_xyz_mm")
            if sp is None or len(sp) != 3 or any(not np.isfinite(x) or x <= 0 for x in sp):
                raise ValueError(
                    f"Metadata[{bi}] spacing_xyz_mm must contain 3 finite positive values, got {sp}"
                )
            sx, sy, sz = float(sp[0]), float(sp[1]), float(sp[2])
            u_x, u_y, u_z = flow_xyz[bi, 0], flow_xyz[bi, 1], flow_xyz[bi, 2]
            disp_mm_i = torch.sqrt((u_x * sx) ** 2 + (u_y * sy) ** 2 + (u_z * sz) ** 2).mean().item()
            disp_mm_list.append(disp_mm_i)
        metrics["mean_displacement_mm"] = float(np.mean(disp_mm_list))

    if n_valid == 0:
        return zero_loss, metrics

    # Differentiable student probabilities
    p_student = F.softmax(student_logits, dim=1)

    if cfg.loss_type == "kl":
        # Detached warped teacher as target distribution
        kl = warped_teacher_prob * (
            torch.log(warped_teacher_prob.clamp(min=1e-7)) - torch.log(p_student.clamp(min=1e-7))
        )
        voxel_err = kl.sum(dim=1)  # (B, Z, H, W)
    else:
        # Mean squared probability discrepancy
        diff = p_student - warped_teacher_prob
        voxel_err = (diff * diff).sum(dim=1)  # (B, Z, H, W)

    loss = (voxel_err * temporal_valid.float()).sum() / (float(n_valid) + 1e-7) * float(cfg.weight)
    metrics["temporal_loss"] = float(loss.detach())

    return loss, metrics


__all__ = [
    "RegistrationResult",
    "TemporalConfig",
    "register_volumes",
    "temporal_consistency",
    "warp_volume_3d",
]
