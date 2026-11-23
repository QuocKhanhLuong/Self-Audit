"""SliceSSMStudent: in-plane CNN with physically-stepped bidirectional z-SSM and misalignment-aware slice fusion.

Short-axis cine stacks are anisotropic (5-10 mm between slice centres against about 1.4 mm in plane, varying per patient)
and adjacent slices are misaligned because each slice is acquired in its own breath-hold.  This student keeps the
in-plane encoder/decoder of :class:`AnisoStudent` (kernels ``1x3x3``, no z down-sampling, deep supervision and SDF
train-only heads, the same stage-3 tap) and replaces through-plane mixing with two modules:

* :class:`PhysZSSM` (B1): a selective state-space scan along z, run in both directions for every in-plane cell.  The
  discretisation step is ``softplus(dt(x)) * (s_z / s_ref)``, i.e. the learned step is scaled by the patient's physical
  slice spacing ``s_z`` (mm, from ``geometry["spacing_zyx"][0]``).  Padded slices get a zero step, so they neither
  update nor read the state.  ``spacing_mode="none"`` drops the physical factor (ablation).
* :class:`SliceAlign` (B2): at stage 3 a small head predicts an in-plane offset field between each pair of adjacent
  slices; neighbour features are warped onto the current slice before they are fused.  The offsets can be supervised
  for free with :func:`shift_recovery_loss`: inject known per-slice shifts (:func:`apply_slice_shifts`) into a view and
  require the predicted offsets to change by exactly the injected relative shift.

Same engine contract as :class:`AnisoStudent` (``forward(image, geometry, ...) -> {"logits", "features", ...}``,
``phases``, ``attach_code_branch``); the engine selects it with ``student.arch = "slicessm"``.  No teacher-feature fusion
path.  Default sizing stays below 3 M parameters including train-only heads.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..candidate_fuse.student import _shape
from .aniso_student import ANISO_CHANNELS, AnisoStudent, PGDrop

SLICESSM_SCHEMA = "self_audit.slicessm_student.export.v1"
DEFAULT_SPACING_REF_MM = 10.0
PARAMETER_CAP = 3_000_000


def slice_spacing_mm(geometry: Any, batch: int, device: torch.device) -> Tensor:
    """Per-sample through-plane spacing ``[B]`` in mm from ``geometry["spacing_zyx"][0]``.

    ``geometry`` is the single mapping the PG engine passes (one patient geometry per call); a sequence of mappings
    (one per sample) is also accepted.  A missing spacing is an error, never a silent default.
    """

    def one(g: Any) -> float:
        if not isinstance(g, Mapping) or "spacing_zyx" not in g:
            raise ValueError("SliceSSMStudent requires geometry['spacing_zyx'] (physical slice spacing)")
        value = float(g["spacing_zyx"][0])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("slice spacing must be a finite positive number of mm")
        return value

    if isinstance(geometry, Mapping):
        values = [one(geometry)] * batch
    elif isinstance(geometry, Sequence) and not isinstance(geometry, (str, bytes)) and len(geometry) == batch:
        values = [one(g) for g in geometry]
    else:
        raise ValueError("geometry must be a mapping or one mapping per batch sample")
    return torch.tensor(values, dtype=torch.float32, device=device)


def _selective_scan(u: Tensor, dt: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
    """Diagonal selective scan.  ``u, dt``: ``[N, L, E]``; ``a``: ``[E, S]`` (negative); ``b, c``: ``[N, L, S]``.

    ``h_t = exp(dt_t * a) * h_{t-1} + dt_t * u_t * b_t``; ``y_t = <h_t, c_t> + d * u_t``.  ``L`` is the number of slices
    (at most a few dozen), so a plain loop is cheap and needs no custom kernel.
    """

    n, length, e = u.shape
    h = u.new_zeros(n, e, a.shape[1])
    ys = []
    for t in range(length):
        dt_t = dt[:, t, :, None]
        h = torch.exp(dt_t * a) * h + (dt_t * u[:, t, :, None]) * b[:, t, None, :]
        ys.append((h * c[:, t, None, :]).sum(-1))
    return torch.stack(ys, 1) + u * d


class _Direction(nn.Module):
    """Input-dependent ``dt, B, C`` and the diagonal ``A`` for one scan direction."""

    def __init__(self, inner: int, state: int, rank: int, dt_min: float, dt_max: float) -> None:
        super().__init__()
        self.x_proj = nn.Linear(inner, rank + 2 * state, bias=False)
        self.dt_proj = nn.Linear(rank, inner, bias=True)
        self.rank, self.state = rank, state
        self.a_log = nn.Parameter(torch.log(torch.arange(1, state + 1, dtype=torch.float32)).repeat(inner, 1))
        self.d = nn.Parameter(torch.ones(inner))
        # Mamba-style dt initialisation: softplus(bias) log-uniform in [dt_min, dt_max].
        dt = torch.exp(torch.rand(inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, u: Tensor, step_scale: Tensor) -> Tensor:
        """``u`` ``[N, L, E]``; ``step_scale`` ``[N, L, 1]`` (physical factor x slice validity)."""
        dbc = self.x_proj(u)
        dt_raw, b, c = dbc.split((self.rank, self.state, self.state), dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw)) * step_scale
        return _selective_scan(u, dt, -torch.exp(self.a_log), b, c, self.d)


class PhysZSSM(nn.Module):
    """Pre-norm bidirectional selective SSM along z with a gated output and a residual connection.

    ``step = softplus(dt(x)) * (s_z / s_ref)`` with ``spacing_mode="physical"``; ``"none"`` uses ``step = softplus(dt(x))``.
    Invalid (padded) slices get ``step = 0``: they leave the state unchanged and their outputs are zeroed.
    """

    def __init__(self, dim: int, *, state: int = 8, expand: float = 1.0, spacing_mode: str = "physical",
                 spacing_ref_mm: float = DEFAULT_SPACING_REF_MM, dt_min: float = 0.01, dt_max: float = 0.5) -> None:
        super().__init__()
        if spacing_mode not in ("physical", "none"):
            raise ValueError("spacing_mode must be 'physical' or 'none'")
        if spacing_ref_mm <= 0:
            raise ValueError("spacing_ref_mm must be positive")
        inner = max(4, int(round(dim * expand)))
        self.dim, self.inner, self.spacing_mode, self.spacing_ref_mm = dim, inner, spacing_mode, float(spacing_ref_mm)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * inner, bias=False)
        rank = max(1, math.ceil(dim / 16))
        self.forward_dir = _Direction(inner, state, rank, dt_min, dt_max)
        self.backward_dir = _Direction(inner, state, rank, dt_min, dt_max)
        self.out_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x: Tensor, spacing_mm: Tensor, slice_valid: Tensor | None = None) -> Tensor:
        """``x`` ``[B, C, D, H, W]``; ``spacing_mm`` ``[B]``; ``slice_valid`` ``[B, D]`` bool or ``None``."""
        bsz, ch, depth, height, width = x.shape
        tokens = x.permute(0, 3, 4, 2, 1).reshape(bsz * height * width, depth, ch)
        factor = spacing_mm / self.spacing_ref_mm if self.spacing_mode == "physical" else torch.ones_like(spacing_mm)
        scale = factor.to(x.dtype)[:, None].expand(bsz, depth)
        valid = torch.ones(bsz, depth, dtype=torch.bool, device=x.device) if slice_valid is None else slice_valid.to(torch.bool)
        scale = scale * valid.to(x.dtype)
        scale = scale[:, None, None, :].expand(bsz, height, width, depth).reshape(-1, depth, 1)
        vmask = valid[:, None, None, :].expand(bsz, height, width, depth).reshape(-1, depth, 1).to(x.dtype)
        u, gate = self.in_proj(self.norm(tokens)).chunk(2, dim=-1)
        u = F.silu(u) * vmask
        y_f = self.forward_dir(u, scale)
        y_b = self.backward_dir(u.flip(1), scale.flip(1)).flip(1)
        out = tokens + self.out_proj((y_f + y_b) * F.silu(gate)) * vmask
        return out.reshape(bsz, height, width, depth, ch).permute(0, 4, 3, 1, 2).contiguous()


def _warp_inplane(x: Tensor, flow: Tensor) -> Tensor:
    """Sample ``x`` ``[N, C, H, W]`` at ``p + flow``; ``flow`` ``[N, 2, H, W]`` in cells (dy, dx); zeros outside."""
    n, _, h, w = x.shape
    ys = torch.arange(h, device=x.device, dtype=x.dtype)
    xs = torch.arange(w, device=x.device, dtype=x.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    sy = gy[None] + flow[:, 0]
    sx = gx[None] + flow[:, 1]
    grid = torch.stack((2 * sx / max(w - 1, 1) - 1, 2 * sy / max(h - 1, 1) - 1), dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class SliceAlign(nn.Module):
    """Predict offsets between adjacent slices, warp both neighbours onto each slice and fuse them residually.

    ``flow[:, :, i]`` (cells, (dy, dx)) maps a location ``p`` of slice ``i`` to ``p + flow`` in slice ``i + 1``.  The last
    convolution of the offset head and the fusion projection are zero-initialised, so at initialisation the module is
    the identity and the offsets are zero.
    """

    def __init__(self, dim: int, *, hidden: int = 32, max_offset_cells: float = 4.0) -> None:
        super().__init__()
        self.max_offset_cells = float(max_offset_cells)
        self.reduce = nn.Conv3d(dim, hidden, 1)
        self.offset = nn.Sequential(
            nn.Conv3d(2 * hidden, hidden, (1, 3, 3), padding=(0, 1, 1)), nn.GELU(),
            nn.Conv3d(hidden, 2, (1, 3, 3), padding=(0, 1, 1)),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.fuse = nn.Conv3d(2 * dim, dim, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def flows(self, x: Tensor) -> Tensor:
        """Offsets ``[B, 2, D-1, H, W]`` between slices ``i`` and ``i+1``."""
        r = self.reduce(x)
        pair = torch.cat((r[:, :, :-1], r[:, :, 1:]), dim=1)
        return torch.tanh(self.offset(pair)) * self.max_offset_cells

    def forward(self, x: Tensor, slice_valid: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        bsz, ch, depth, height, width = x.shape
        if depth < 2:
            return x, None
        flow = self.flows(x)
        pairs = bsz * (depth - 1)
        nxt = x[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(pairs, ch, height, width)
        prv = x[:, :, :-1].permute(0, 2, 1, 3, 4).reshape(pairs, ch, height, width)
        f = flow.permute(0, 2, 1, 3, 4).reshape(pairs, 2, height, width)
        next_on_i = _warp_inplane(nxt, f).reshape(bsz, depth - 1, ch, height, width).permute(0, 2, 1, 3, 4)
        prev_on_next = _warp_inplane(prv, -f).reshape(bsz, depth - 1, ch, height, width).permute(0, 2, 1, 3, 4)
        if slice_valid is not None:
            v = slice_valid.to(x.dtype)
            pair_valid = (v[:, :-1] * v[:, 1:])[:, None, :, None, None]
            next_on_i = next_on_i * pair_valid
            prev_on_next = prev_on_next * pair_valid
        zero = x.new_zeros(bsz, ch, 1, height, width)
        aligned_next = torch.cat((next_on_i, zero), dim=2)   # slice i sees warped i+1 (none for the last slice)
        aligned_prev = torch.cat((zero, prev_on_next), dim=2)  # slice i sees warped i-1 (none for the first slice)
        out = x + self.fuse(torch.cat((aligned_next, aligned_prev), dim=1))
        if slice_valid is not None:
            out = out * slice_valid.to(x.dtype)[:, None, :, None, None]
        return out, flow


class SliceSSMStudent(AnisoStudent):
    """:class:`AnisoStudent` encoder/decoder with :class:`SliceAlign` at stage 3 and :class:`PhysZSSM` at stages 3-4."""

    def __init__(self, config: Mapping[str, Any] | None = None, *, channels: tuple[int, ...] = ANISO_CHANNELS,
                 deploy: bool = False, ssm_blocks: tuple[int, int] = (1, 2), spacing_mode: str | None = None,
                 align: bool | None = None) -> None:
        super().__init__(config, channels=channels, deploy=deploy)
        del self.z_transformer
        model = (config or {}).get("model", config or {})
        student_cfg = model.get("student", {}) if isinstance(model, Mapping) else {}
        opts = student_cfg.get("slicessm", {}) if isinstance(student_cfg, Mapping) else {}
        self.spacing_mode = str(spacing_mode or opts.get("spacing_mode", "physical"))
        self.align_enabled = bool(opts.get("align", True) if align is None else align)
        self.spacing_ref_mm = float(opts.get("spacing_ref_mm", DEFAULT_SPACING_REF_MM))
        state = int(opts.get("state", 8))
        n3, n4 = (int(v) for v in opts.get("ssm_blocks", ssm_blocks))
        c3, c4 = channels[3], channels[4]
        self.align3 = SliceAlign(c3) if self.align_enabled else None
        mk = lambda dim: PhysZSSM(dim, state=state, spacing_mode=self.spacing_mode, spacing_ref_mm=self.spacing_ref_mm)  # noqa: E731
        self.zssm3 = nn.ModuleList(mk(c3) for _ in range(n3))
        self.zssm4 = nn.ModuleList(mk(c4) for _ in range(n4))

    def _encode_to_e3(self, image: Tensor, geometry: Any, slice_valid: Tensor | None):
        if image.ndim != 5 or image.shape[1] != 1 or not image.is_floating_point() or not torch.isfinite(image).all():
            raise ValueError("input must be finite floating [B,1,D,H,W]")
        native = _shape(geometry, image)
        x = image[..., : native[1], : native[2]]
        if torch.any(x < 0) or torch.any(x > 1):
            raise ValueError("image must be normalized to [0,1]")
        spacing = slice_spacing_mm(geometry, image.shape[0], image.device)
        x, hw = self._pad(x)
        f0 = self.e0(x)
        f1 = self.e1(f0)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        flow = None
        if self.align3 is not None:
            f3, flow = self.align3(f3, slice_valid)
        for block in self.zssm3:
            f3 = block(f3, spacing, slice_valid)
        return native, hw, spacing, (f0, f1, f2, f3), flow

    def slice_flows(self, image: Tensor, geometry: Any, slice_valid: Tensor | None = None) -> Tensor | None:
        """Stage-3 adjacent-slice offsets only (encoder up to stage 3), for :func:`shift_recovery_loss`."""
        if self.align3 is None:
            return None
        native = _shape(geometry, image)
        x = image[..., : native[1], : native[2]]
        x, _ = self._pad(x)
        f3 = self.e3(self.e2(self.e1(self.e0(x))))
        return self.align3.flows(f3) if f3.shape[2] >= 2 else None

    def forward(self, image: Tensor, geometry: Any, *, teacher_features=None, keep_mask=None, skip_dropout: float = 0.0,
                return_stage3: bool = False, return_code: bool = False, dropout_generator: torch.Generator | None = None,
                slice_valid: Tensor | None = None) -> dict[str, Any]:
        if teacher_features is not None or keep_mask is not None:
            raise ValueError("SliceSSMStudent has no teacher-feature fusion path")
        native, (nh, nw), spacing, (f0, f1, f2, f3), flow = self._encode_to_e3(image, geometry, slice_valid)
        f4 = self.e4(f3)
        for block in self.zssm4:
            f4 = block(f4, spacing, slice_valid)

        def drop(t: Tensor) -> Tensor:
            return PGDrop(t, skip_dropout, self.training, dropout_generator)

        d3 = self.dec4(torch.cat([self.up4(drop(f4)), drop(f3)], 1))
        d2 = self.dec3(torch.cat([self.up3(d3), drop(f2)], 1))
        d1 = self.dec2(torch.cat([self.up2(d2), drop(f1)], 1))
        d0 = self.dec1(torch.cat([self.up1(d1), drop(f0)], 1))
        logits = self.logits_head(d0)[..., :nh, :nw]
        out: dict[str, Any] = {"logits": logits, "features": (f0, f1, f2, f3, f4), "gate_means": [0.0] * 5,
                               "mimic_loss": logits.sum() * 0.0}
        if flow is not None and self.training:
            out["slice_flow"] = flow
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


# ---------------------------------------------------------------------------------------------------------------
# Slice-shift injection and free supervision of the offset head (B2)
# ---------------------------------------------------------------------------------------------------------------

def sample_slice_shifts(batch: int, depth: int, max_px: int, *, device: torch.device | str = "cpu",
                        generator: torch.Generator | None = None) -> Tensor:
    """Integer per-slice in-plane shifts ``[B, D, 2]`` (dy, dx) uniform in ``[-max_px, max_px]``."""
    if max_px < 0:
        raise ValueError("max_px must be non-negative")
    shifts = torch.randint(-max_px, max_px + 1, (batch, depth, 2), generator=generator)
    return shifts.to(device=device, dtype=torch.float32)


def apply_slice_shifts(x: Tensor, shifts_px: Tensor, *, mode: str = "bilinear") -> tuple[Tensor, Tensor]:
    """Translate every slice of ``x`` ``[B, C, D, H, W]`` by ``shifts_px`` ``[B, D, 2]`` pixels (content moves by +shift).

    Returns the shifted tensor and an in-bounds mask ``[B, 1, D, H, W]`` (0 where content came from outside the slice).
    Use ``mode="nearest"`` for label maps.  Pseudo-labels, weights and validity of a shifted view must be shifted with
    exactly the same ``shifts_px``.
    """
    if x.ndim != 5 or shifts_px.shape != (x.shape[0], x.shape[2], 2):
        raise ValueError("x must be [B,C,D,H,W] and shifts [B,D,2]")
    bsz, ch, depth, height, width = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(bsz * depth, ch, height, width)
    is_int = not flat.is_floating_point()
    src = flat.to(torch.float32) if is_int else flat
    flow = -shifts_px.reshape(bsz * depth, 2, 1, 1).to(src.dtype).expand(-1, -1, height, width)
    moved = _warp_inplane(src, flow) if mode == "bilinear" else _warp_nearest(src, flow)
    inside = _warp_inplane(torch.ones_like(src[:, :1]), flow) > 0.999
    if is_int:
        moved = moved.round().to(flat.dtype)
    moved = moved.reshape(bsz, depth, ch, height, width).permute(0, 2, 1, 3, 4)
    inside = inside.reshape(bsz, depth, 1, height, width).permute(0, 2, 1, 3, 4)
    return moved, inside


def _warp_nearest(x: Tensor, flow: Tensor) -> Tensor:
    n, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype), torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    grid = torch.stack((2 * (gx[None] + flow[:, 1]) / max(w - 1, 1) - 1, 2 * (gy[None] + flow[:, 0]) / max(h - 1, 1) - 1), dim=-1)
    return F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=True)


def shift_recovery_loss(flow_shifted: Tensor, flow_clean: Tensor, shifts_px: Tensor, *, stride: int = 8,
                        slice_valid: Tensor | None = None) -> Tensor:
    """Smooth-L1 between the change of predicted offsets and the injected relative shift.

    The natural misalignment of the stack is unknown, so the target is a difference: on the shifted view the offset from
    slice ``i`` to ``i+1`` must grow by ``(s_{i+1} - s_i) / stride`` cells relative to the clean view (``flow_clean`` is
    detached).  ``stride`` is the in-plane stride of the stage that predicts the offsets (8 for stage 3).
    """
    if flow_shifted.shape != flow_clean.shape or flow_shifted.ndim != 5 or flow_shifted.shape[1] != 2:
        raise ValueError("flows must both be [B,2,D-1,H,W]")
    rel = (shifts_px[:, 1:] - shifts_px[:, :-1]) / float(stride)  # [B, D-1, 2]
    target = rel.permute(0, 2, 1)[..., None, None].to(flow_shifted.dtype)
    err = F.smooth_l1_loss(flow_shifted - flow_clean.detach(), target.expand_as(flow_shifted), reduction="none")
    if slice_valid is None:
        return err.mean()
    v = slice_valid.to(err.dtype)
    w = (v[:, :-1] * v[:, 1:])[:, None, :, None, None].expand_as(err)
    return (err * w).sum() / w.sum().clamp_min(1.0)


def shift_aux_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """``student.slicessm.shift_aux`` = ``{"weight", "max_px", "start_update"}``; ``None`` when absent or weight 0."""
    student_cfg = config.get("student", {}) if isinstance(config, Mapping) else {}
    opts = (student_cfg.get("slicessm", {}) or {}).get("shift_aux") if isinstance(student_cfg, Mapping) else None
    if not opts or float(opts.get("weight", 0.0)) <= 0.0:
        return None
    return {"weight": float(opts["weight"]), "max_px": int(opts.get("max_px", 8)), "start_update": int(opts.get("start_update", 0))}


def slice_shift_aux(student: nn.Module, images: Sequence[tuple[Tensor, Any]], *, seed: int, update: int, max_px: int) -> Tensor | None:
    """Shift-recovery loss averaged over ``(image, geometry)`` pairs; deterministic in ``(seed, update)`` for resume.

    Only the offset head and the encoder up to stage 3 receive gradient; the clean-view offsets are computed without
    gradient.  Returns ``None`` when the student has no offset head or every image has a single slice.
    """
    if not hasattr(student, "slice_flows"):
        return None
    gen = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(update))
    losses = []
    for image, geometry in images:
        if image.shape[2] < 2:
            continue
        shifts = sample_slice_shifts(image.shape[0], image.shape[2], max_px, device=image.device, generator=gen)
        shifted, _ = apply_slice_shifts(image, shifts)
        with torch.no_grad():
            clean = student.slice_flows(image, geometry)
        moved = student.slice_flows(shifted.clamp(0.0, 1.0), geometry)
        if clean is None or moved is None:
            continue
        losses.append(shift_recovery_loss(moved, clean, shifts, stride=8))
    return torch.stack(losses).mean() if losses else None


# ---------------------------------------------------------------------------------------------------------------
# Export, construction, accounting
# ---------------------------------------------------------------------------------------------------------------

TRAIN_ONLY_PREFIXES = ("sdf_head.", "aux_heads.", "stage3_tap.", "code_branch.")


def _options(model: SliceSSMStudent) -> dict[str, Any]:
    return {"spacing_mode": model.spacing_mode, "align": model.align_enabled, "spacing_ref_mm": model.spacing_ref_mm,
            "ssm_blocks": [len(model.zssm3), len(model.zssm4)], "state": model.zssm4[0].forward_dir.state if len(model.zssm4) else 8}


def export_slicessm_student(model: SliceSSMStudent, path: Any, *, epoch: int | None = None) -> dict[str, Any]:
    """Student-only artifact; train-only heads and any code branch are stripped."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith(TRAIN_ONLY_PREFIXES)}
    torch.save({"schema": SLICESSM_SCHEMA, "channels": list(model.channels), "options": _options(model), "state": state,
                "selected_epoch": epoch}, path)
    return {"path": str(path), "parameters_deployed": sum(v.numel() for v in state.values())}


def load_slicessm_export(path: Any) -> SliceSSMStudent:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SLICESSM_SCHEMA:
        raise ValueError("not a SliceSSMStudent export")
    model = SliceSSMStudent({"student": {"slicessm": payload["options"]}}, channels=tuple(payload["channels"]), deploy=True)
    model.load_state_dict(payload["state"], strict=True)
    return model.eval()


def build_slicessm_student(config: Mapping[str, Any] | None = None) -> SliceSSMStudent:
    """``student.slicessm.channels`` (five widths, default :data:`ANISO_CHANNELS`) and ``student.slicessm.parameter_cap`` (default 3 M, ``null`` = none)
    are the only knobs of the S/M/L width ablation; every other option and module is unchanged."""
    model_cfg = (config or {}).get("model", config or {})
    opts = (model_cfg.get("student", {}) if isinstance(model_cfg, Mapping) else {}).get("slicessm", {}) or {}
    channels = tuple(int(v) for v in opts.get("channels", ANISO_CHANNELS))
    if len(channels) != 5 or any(c < 8 or c % 8 for c in channels):
        raise ValueError("student.slicessm.channels must be five widths that are multiples of 8")
    cap = opts.get("parameter_cap", PARAMETER_CAP)
    model = SliceSSMStudent(config, channels=channels)
    total = sum(p.numel() for p in model.parameters())
    if cap is not None and total > int(cap):
        raise ValueError(f"SliceSSMStudent has {total} parameters, above the {int(cap)} cap")
    return model


__all__ = [
    "SliceSSMStudent", "PhysZSSM", "SliceAlign", "apply_slice_shifts", "shift_aux_config", "slice_shift_aux", "sample_slice_shifts", "shift_recovery_loss",
    "slice_spacing_mm", "export_slicessm_student", "load_slicessm_export", "build_slicessm_student", "PARAMETER_CAP",
]
