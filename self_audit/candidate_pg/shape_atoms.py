"""CBKD-SA: shape-atom codebook knowledge distillation (variant ``cbkd_sa_gated``).

A frozen codebook ``C`` (K x 64, cosine) is fitted on the weak teacher W's
level-3 features.  Each code owns a *shape atom*: a decoder ``D_A`` turns the
code embedding into a native-resolution patch of 4 class logits and 3
foreground SDF channels.  The student distils into this atom space:

* ``L_pm``  : teacher-side atom class probabilities ``q_T`` (margin- and
  arbiter-gated) pull the student's clean-weak probabilities on U.
* ``L_atom``: the student's stage-3 feature, vector-quantised with the frozen
  ``C`` and decoded by the frozen ``D_A``, must reproduce GT patches on L_K.

Guards carried over from the CBKD-v1 audit: ``log_softmax`` (never
clamp-then-log), l2-normalised ``z`` and ``C``, student code-usage perplexity
reported from the first CBKD update.  ``C`` and ``D_A`` never receive student
gradient; the student's ``P_S`` is the only trainable part of the branch.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import losses

N_CLASS = 4
N_SDF = 3
N_OUT = N_CLASS + N_SDF
TEACHER_CHANNELS = 256  # MedNeXt-L level-3 channels
STUDENT_CHANNELS = 128  # student stage3 channels
EMBED_DIM = 64
CODEBOOK_SIZE = 128
GRID_STRIDE_HW = 8


def l2n(x: Tensor, dim: int = 1) -> Tensor:
    return F.normalize(x, dim=dim, eps=1e-6)


class AtomDecoder(nn.Module):
    """``D_A``: code embedding ``[B,64,Dg,Hg,Wg]`` -> native patches ``[B,7,Dg*pd,Hg*8,Wg*8]``."""

    def __init__(self, embed_dim: int = EMBED_DIM, patch: tuple[int, int, int] = (1, 8, 8), hidden: int = 128) -> None:
        super().__init__()
        self.patch = tuple(int(v) for v in patch)
        voxels = self.patch[0] * self.patch[1] * self.patch[2]
        self.net = nn.Sequential(
            nn.Conv3d(embed_dim, hidden, 1), nn.ReLU(inplace=False),
            nn.Conv3d(hidden, hidden, 1), nn.ReLU(inplace=False),
            nn.Conv3d(hidden, N_OUT * voxels, 1),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        b, _, dg, hg, wg = embedding.shape
        pd, ph, pw = self.patch
        out = self.net(embedding).reshape(b, N_OUT, pd, ph, pw, dg, hg, wg)
        out = out.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, N_OUT, dg * pd, hg * ph, wg * pw)
        return out


def crop_to_native(full: Tensor, native_shape: Sequence[int], patch: Sequence[int]) -> Tensor:
    """Crop an assembled ``[B,C,Dg*pd,Hg*ph,Wg*pw]`` field to native ``(D,H,W)``.

    Student cells are centred: cell ``i`` is centred on voxel ``i*s``, so cell
    ``i`` covers voxels ``[i*s - s//2, i*s + s//2)``.  Native voxel ``v`` thus
    reads assembled index ``v + s//2``; the high side is replicate-padded.
    """

    out = full
    for axis, (n, s) in enumerate(zip(native_shape, patch)):
        shift = int(s) // 2
        dim = 2 + axis
        need = shift + int(n)
        if out.shape[dim] < need:
            pad_shape = list(out.shape)
            pad_shape[dim] = need - out.shape[dim]
            tail = out.narrow(dim, out.shape[dim] - 1, 1).expand(*pad_shape)
            out = torch.cat((out, tail), dim=dim)
        out = out.narrow(dim, shift, int(n))
    return out


def upsample_cells(cell_map: Tensor, native_shape: Sequence[int], patch: Sequence[int]) -> Tensor:
    """Nearest-upsample a cell map ``[B,C,Dg,Hg,Wg]`` to native resolution (same centring)."""

    full = cell_map
    for axis, s in enumerate(patch):
        full = full.repeat_interleave(int(s), dim=2 + axis)
    return crop_to_native(full, native_shape, patch)


def split_patches(patches: Tensor) -> tuple[Tensor, Tensor]:
    return patches[:, :N_CLASS], torch.tanh(patches[:, N_CLASS:])


def cosine_distances(z: Tensor, codes: Tensor) -> Tensor:
    """Squared euclidean distance on the unit sphere, ``[B,K,*spatial]``."""

    sim = torch.einsum("bc...,kc->bk...", z, codes)
    return (2.0 - 2.0 * sim).clamp_min(0.0)


def nearest_two(z: Tensor, codes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(idx, d1, d2)`` of the two nearest codes; ``d2 >= d1``."""

    dist = cosine_distances(z, codes)
    values, index = dist.topk(2, dim=1, largest=False)
    return index[:, 0], values[:, 0], values[:, 1]


def margin(d1: Tensor, d2: Tensor, tau_star: Tensor | float) -> Tensor:
    """``m = 1 - exp(-(d2 - d1)/tau*)`` in ``[0,1)``."""

    return 1.0 - torch.exp(-(d2 - d1).clamp_min(0.0) / torch.as_tensor(tau_star, dtype=d1.dtype, device=d1.device))


def code_usage(idx: Tensor, valid: Tensor | None, k: int) -> dict[str, float]:
    flat = idx[valid.bool()] if valid is not None else idx.reshape(-1)
    counts = torch.bincount(flat.reshape(-1).long(), minlength=k).double()
    total = counts.sum()
    if total <= 0:
        return {"perplexity": 0.0, "dead": int(k), "count": 0}
    freq = counts / total
    entropy = -(freq * freq.clamp_min(1e-12).log()).sum()
    return {"perplexity": float(entropy.exp()), "dead": int((counts == 0).sum()), "count": int(total)}


class ShapeAtomCodebook(nn.Module):
    """``P_T`` + ``C`` + ``D_A`` (+ ``tau*``): the teacher-side artifact fitted by arm CB."""

    def __init__(self, size: int = CODEBOOK_SIZE, dim: int = EMBED_DIM, patch: tuple[int, int, int] = (1, 8, 8),
                 teacher_channels: int = TEACHER_CHANNELS) -> None:
        super().__init__()
        self.size, self.dim, self.patch = int(size), int(dim), tuple(int(v) for v in patch)
        self.P_T = nn.Conv3d(teacher_channels, dim, 1)
        self.C = nn.Parameter(l2n(torch.randn(size, dim), dim=1))
        self.D_A = AtomDecoder(dim, self.patch)
        self.register_buffer("tau_star", torch.tensor(1.0))

    def codes(self) -> Tensor:
        return l2n(self.C, dim=1)

    def embed(self, feature: Tensor) -> Tensor:
        return l2n(self.P_T(feature))

    def decode(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        return split_patches(self.D_A(embedding))

    def forward(self, feature: Tensor) -> dict[str, Tensor]:
        z = self.embed(feature)
        codes = self.codes()
        idx, d1, d2 = nearest_two(z, codes)
        e = codes[idx].permute(0, 4, 1, 2, 3)
        logits, sdf = self.decode(e)
        return {"z": z, "e": e, "idx": idx, "d1": d1, "d2": d2, "margin": margin(d1, d2, self.tau_star),
                "class_logits": logits, "sdf": sdf}

    def freeze(self) -> "ShapeAtomCodebook":
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        return self

    def train(self, mode: bool = True) -> "ShapeAtomCodebook":
        # A frozen codebook never leaves eval mode.
        frozen = all(not p.requires_grad for p in self.parameters())
        return super().train(False if frozen else mode)


@torch.no_grad()
def teacher_atoms(codebook: ShapeAtomCodebook, feature_l3: Tensor, native_shape: Sequence[int]) -> dict[str, Tensor]:
    """Teacher-side ``q_T`` (class probabilities) and ``m_T`` at native resolution, no grad."""

    out = codebook(feature_l3.detach())
    logits = crop_to_native(out["class_logits"], native_shape, codebook.patch)
    m = upsample_cells(out["margin"].unsqueeze(1), native_shape, codebook.patch)[:, 0]
    sdf = crop_to_native(out["sdf"], native_shape, codebook.patch)
    return {"q": logits.softmax(dim=1), "margin": m, "sdf": sdf, "idx": out["idx"]}


class StudentAtomBranch(nn.Module):
    """Train-only student side: ``P_S`` (128->64) + frozen ``C``/``D_A`` (STE hard VQ)."""

    def __init__(self, codebook: ShapeAtomCodebook, student_channels: int = STUDENT_CHANNELS) -> None:
        super().__init__()
        self.P_S = nn.Conv3d(student_channels, codebook.dim, 1)
        self.patch = codebook.patch
        self.register_buffer("C", codebook.codes().detach().clone())
        self.D_A = copy.deepcopy(codebook.D_A)
        for parameter in self.D_A.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "StudentAtomBranch":
        super().train(mode)
        self.D_A.eval()
        return self

    def forward(self, stage3: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        z = l2n(self.P_S(stage3))
        idx, _, _ = nearest_two(z, self.C)
        e = self.C[idx].permute(0, 4, 1, 2, 3)
        zq = z + (e - z).detach()  # straight-through; C is a buffer
        logits, sdf = split_patches(self.D_A(zq))
        return {"z": z, "e": e, "idx": idx, "class_logits": logits, "sdf": sdf, "valid_mask": valid_mask}


def commitment(z: Tensor, e: Tensor, valid: Tensor | None) -> Tensor:
    err = (z - e.detach()).pow(2).sum(dim=1)
    if valid is None:
        return err.mean()
    mask = valid.bool()
    return (err * mask).sum() / mask.sum().clamp_min(1)


def atom_loss(branch_out: Mapping[str, Tensor], target: Tensor, sdf_target: Tensor | None, valid: Tensor,
              native_shape: Sequence[int], patch: Sequence[int]) -> tuple[Tensor, dict[str, Tensor]]:
    """CE + Dice on class logits and L1 on SDF against native GT (assembled atom patches)."""

    logits = crop_to_native(branch_out["class_logits"], native_shape, patch)
    weights = torch.ones_like(target, dtype=torch.float32)
    ce = losses.weighted_ce(logits, target, weights, valid)
    dice = losses.weighted_dice(logits.softmax(dim=1), target, weights, valid)
    terms = {"atom_ce": ce.detach(), "atom_dice": dice.detach()}
    total = ce + dice
    if sdf_target is not None:
        sdf = crop_to_native(branch_out["sdf"], native_shape, patch)
        l1 = ((sdf - sdf_target).abs().mean(dim=1) * valid.float()).sum() / valid.float().sum().clamp_min(1)
        terms["atom_sdf_l1"] = l1.detach()
        total = total + l1
    return total, terms


def pm_loss(p_student: Tensor, q_teacher: Tensor, m_teacher: Tensor, gate: Tensor, valid: Tensor) -> Tensor:
    """``sum g*m*valid*||p_S - q_T||^2 / sum g*m*valid`` (class-summed). Gradient only via ``p_student``."""

    weight = gate.detach().float() * m_teacher.detach().float() * valid.detach().float()
    err = (p_student - q_teacher.detach()).pow(2).sum(dim=1)
    denominator = weight.sum()
    if float(denominator) <= 0.0:
        return p_student.sum() * 0.0
    return (weight * err).sum() / denominator


def lambda_cb(update: int, *, lam: float = 0.5, onset: int = 1200, ramp: int = 400) -> float:
    if update < onset:
        return 0.0
    return float(lam) * min(1.0, (update - onset + 1) / max(1, ramp))


def fit_loss(codebook: ShapeAtomCodebook, feature: Tensor, class_target: Tensor, class_weight: Tensor,
             sdf_target: Tensor | None, sdf_weight: Tensor | None, native_shape: Sequence[int],
             valid: Tensor | None = None, *, beta: float = 0.25) -> tuple[Tensor, dict[str, Tensor]]:
    """Codebook fit loss on one frame.

    ``class_target`` is ``[B,4,D,H,W]`` probabilities (one-hot GT on L_K, W probabilities on U),
    ``class_weight`` ``[B,D,H,W]`` (1 on L_K, W confidence on U); ``sdf_target`` ``[B,3,D,H,W]`` with
    ``sdf_weight`` ``[B,D,H,W]`` (zero where W max-prob < 0.9 on U).  Codebook is learned by the
    VQ loss; the decoder is trained through the straight-through path.
    """

    z = codebook.embed(feature)
    codes = codebook.codes()
    idx, _, _ = nearest_two(z.detach(), codes)
    e = codes[idx].permute(0, 4, 1, 2, 3)
    zq = z + (e - z).detach()
    logits, sdf = codebook.decode(zq)
    logits = crop_to_native(logits, native_shape, codebook.patch)
    sdf = crop_to_native(sdf, native_shape, codebook.patch)
    support = torch.ones_like(class_weight, dtype=torch.bool) if valid is None else valid.bool()
    log_p = F.log_softmax(logits, dim=1)  # no clamp-then-log
    ce = -(class_target * log_p).sum(dim=1)
    w = class_weight.float() * support.float()
    l_ce = (ce * w).sum() / w.sum().clamp_min(1e-6)
    pred = log_p.exp()
    inter = (pred * class_target * w[:, None]).sum(dim=(0, 2, 3, 4))
    denom = ((pred + class_target) * w[:, None]).sum(dim=(0, 2, 3, 4))
    l_dice = 1.0 - ((2 * inter + 1e-6) / (denom + 1e-6))[1:].mean()
    total = l_ce + l_dice
    terms = {"fit_ce": l_ce.detach(), "fit_dice": l_dice.detach()}
    if sdf_target is not None and sdf_weight is not None:
        sw = sdf_weight.float() * support.float()
        l_sdf = ((sdf - sdf_target).abs().mean(dim=1) * sw).sum() / sw.sum().clamp_min(1e-6)
        total = total + l_sdf
        terms["fit_sdf_l1"] = l_sdf.detach()
    # VQ terms live on the cell grid: codebook term pulls C toward sg(z); commitment pulls z toward sg(e).
    l_code = (e - z.detach()).pow(2).sum(dim=1).mean()
    l_commit = (z - e.detach()).pow(2).sum(dim=1).mean()
    total = total + l_code + beta * l_commit
    terms.update({"fit_code": l_code.detach(), "fit_commit": l_commit.detach()})
    return total, terms


@torch.no_grad()
def calibrate_tau_star(codebook: ShapeAtomCodebook, features: Sequence[Tensor]) -> float:
    """``tau*`` = median of ``d2 - d1`` over TRAIN cells (set and returned)."""

    gaps = []
    for feature in features:
        _, d1, d2 = nearest_two(codebook.embed(feature), codebook.codes())
        gaps.append((d2 - d1).reshape(-1))
    value = float(torch.cat(gaps).median().clamp_min(1e-6))
    codebook.tau_star.fill_(value)
    return value


@torch.no_grad()
def restart_dead_codes(codebook: ShapeAtomCodebook, z_pool: Tensor, usage: Tensor) -> int:
    """Replace unused codes with random pooled ``z`` samples; returns the number restarted."""

    dead = (usage == 0).nonzero(as_tuple=False).reshape(-1)
    if dead.numel() == 0 or z_pool.shape[0] == 0:
        return 0
    pick = torch.randint(0, z_pool.shape[0], (dead.numel(),), device=z_pool.device)
    codebook.C.data[dead] = l2n(z_pool[pick], dim=1)
    return int(dead.numel())


def fg_dice(pred: Tensor, ref: Tensor) -> float:
    a, b = pred > 0, ref > 0
    denominator = float(a.sum() + b.sum())
    return 1.0 if denominator == 0 else 2.0 * float((a & b).sum()) / denominator


def class_dice(pred: Tensor, ref: Tensor, cls: int) -> float:
    a, b = pred == cls, ref == cls
    denominator = float(a.sum() + b.sum())
    return 1.0 if denominator == 0 else 2.0 * float((a & b).sum()) / denominator


# --------------------------------------------------------------------------- S3-SA stage
def patch_split(target: Tensor, patch: Sequence[int] = (1, 8, 8)) -> Tensor:
    """One-hot GT ``[D,H,W]`` -> non-overlapping patches ``[N, 4*pd*ph*pw]`` (high side padded with background)."""

    d, h, w = target.shape
    pd, ph, pw = (int(v) for v in patch)
    x = _onehot(target[None])[0]
    x = F.pad(x, (0, -w % pw, 0, -h % ph, 0, -d % pd))
    x[0, d:, :, :] = 1.0; x[0, :, h:, :] = 1.0; x[0, :, :, w:] = 1.0  # padding is background
    c, dd, hh, ww = x.shape
    x = x.reshape(c, dd // pd, pd, hh // ph, ph, ww // pw, pw).permute(1, 3, 5, 0, 2, 4, 6)
    return x.reshape(-1, c * pd * ph * pw)


def assemble_full_res(rows: Tensor, shape: Sequence[int], patch: Sequence[int] = (1, 8, 8)) -> Tensor:
    """Inverse of :func:`patch_split` (argmax labels ``[D,H,W]``; padded cells dropped)."""

    d, h, w = (int(v) for v in shape)
    pd, ph, pw = (int(v) for v in patch)
    gd, gh, gw = -(-d // pd), -(-h // ph), -(-w // pw)
    x = rows.reshape(gd, gh, gw, N_CLASS, pd, ph, pw).permute(3, 0, 4, 1, 5, 2, 6).reshape(N_CLASS, gd * pd, gh * ph, gw * pw)
    return x[:, :d, :h, :w].argmax(0)


def _onehot(labels: Tensor) -> Tensor:
    return F.one_hot(labels.clamp(0, N_CLASS - 1), N_CLASS).permute(0, 4, 1, 2, 3).float()


def _teacher_features(teacher: nn.Module, student: nn.Module, image: Tensor, geometry: Mapping[str, Any]) -> tuple[Tensor, Mapping[str, Any]]:
    """Aligned teacher level-3 feature on the student stage-3 grid plus the raw teacher packet (no grad)."""

    from . import codebook as legacy

    packet = legacy._teacher_packet(teacher, image, geometry)
    feature, _, _ = legacy._aligned_level3(packet["features"][3], geometry, teacher=teacher, student=student)
    return feature.detach(), packet


def _public_sdf(packet: Mapping[str, Any]) -> Tensor:
    sdf = packet.get("sdf")
    if isinstance(sdf, Tensor):
        return sdf.detach()
    return packet["sdf_all"][:, 1:].detach()


@torch.no_grad()
def reconstruction_report(codebook: ShapeAtomCodebook, teacher: nn.Module, student: nn.Module, data: Any,
                          device: torch.device, *, limit: int | None = None) -> dict[str, Any]:
    """Code-reconstructed teacher vs raw teacher (and vs GT) on labelled frames; FG/RV/MYO/LV Dice."""

    from . import codebook as legacy

    keys, _ = legacy._frame_lists(data, include_unlabeled=False)
    rows, usage_idx = [], []
    for key in keys[:limit]:
        frame = legacy._load_s3_frame(data, key, labeled=True)
        image, target, geometry = legacy._s3_frame_tensors(frame, device, labeled=True)
        feature, packet = _teacher_features(teacher, student, image, geometry)
        native = tuple(int(v) for v in image.shape[-3:])
        atoms = teacher_atoms(codebook, feature, native)
        raw = packet["region_logits"].argmax(1)
        rec = atoms["q"].argmax(1)
        row = {"fg_rec_vs_raw": fg_dice(rec, raw), "fg_rec_vs_gt": fg_dice(rec, target), "fg_raw_vs_gt": fg_dice(raw, target)}
        for cls, name in ((1, "rv"), (2, "myo"), (3, "lv")):
            row[f"{name}_rec_vs_gt"], row[f"{name}_raw_vs_gt"] = class_dice(rec, target, cls), class_dice(raw, target, cls)
        rows.append(row)
        usage_idx.append(atoms["idx"].reshape(-1))
    mean = {k: sum(r[k] for r in rows) / max(1, len(rows)) for k in rows[0]} if rows else {}
    usage = code_usage(torch.cat(usage_idx), None, codebook.size) if usage_idx else {"perplexity": 0.0, "dead": codebook.size, "count": 0}
    return {"frames": len(rows), "mean": mean, "perplexity": usage["perplexity"], "dead_codes": usage["dead"]}


def run_shape_atoms(config: Mapping[str, Any], out: Any, train_data: Any, val_data: Any, teacher: nn.Module,
                    student: nn.Module, *, device: str | torch.device = "cpu", max_updates: int | None = None,
                    resume: bool = False, checkpoint_every: int = 500, tau_frames: int = 128) -> dict[str, Any]:
    """S3-SA: fit ``P_T``, ``C``, ``D_A`` on TRAIN only with a frozen teacher, then calibrate ``tau*``.

    Step = one labelled frame + two U frames (L + mean(U1, U2)), as S3.  Targets: GT one-hot and GT SDF on
    L_K; teacher full-res probabilities (weight = max-prob) and teacher SDF (where max-prob >= 0.9) on U.
    Checkpoints are atomic every ``checkpoint_every`` updates and resumable.
    """

    import json
    import os
    import time
    from pathlib import Path

    from . import codebook as legacy
    from .data import PGSampler

    dev = torch.device(device)
    cb_cfg = config["codebook"]
    size, dim = int(cb_cfg["codebook"]["size"]), int(cb_cfg["codebook"]["dim"])
    updates = int(cb_cfg["updates"])
    if max_updates is not None:
        updates = min(updates, int(max_updates))
    opt_cfg = cb_cfg["optimizer"]
    restart_every = int(cb_cfg["codebook"].get("dead_code_restart_every", 500))
    entropy_weight = float(cb_cfg["codebook"].get("usage_entropy_weight", 0.01))
    beta = float(cb_cfg.get("commit_beta", 0.25))
    patch = (int(student.phases[3][0][0]), int(student.phases[3][0][1]), int(student.phases[3][0][2]))
    torch.manual_seed(int(config.get("seed", 42)))
    model = ShapeAtomCodebook(size, dim, patch).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["lr"]), weight_decay=float(opt_cfg["weight_decay"]))
    root = Path(out); root.mkdir(parents=True, exist_ok=True)
    ckpt_path = root / "checkpoint.pt"
    sampler = PGSampler(getattr(train_data, "train", train_data), seed=42)
    start, usage_counts = 0, torch.zeros(size, device=dev)
    if resume and ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"]); start = int(state["update"]); usage_counts = state["usage_counts"].to(dev)
        torch.set_rng_state(state["rng"].cpu())
    else:
        # Cold start: initialise C from real TRAIN teacher projections (not random).
        zs = []
        for key in list(train_data.train.labeled_keys())[:8] if hasattr(train_data, "train") else []:
            frame = train_data.train.load(key, labeled=True)
            image, _, geometry = legacy._s3_frame_tensors(frame, dev, labeled=True)
            feature, _ = _teacher_features(teacher, student, image, geometry)
            with torch.no_grad():
                zs.append(model.embed(feature).permute(0, 2, 3, 4, 1).reshape(-1, dim))
        if zs:
            pool = torch.cat(zs)
            model.C.data = l2n(pool[torch.randperm(pool.shape[0], generator=torch.Generator().manual_seed(42))[:size].to(dev)], dim=1) if pool.shape[0] >= size else model.C.data
    diag_path = root / "diag.jsonl"
    last_window_z: list[Tensor] = []
    view = getattr(train_data, "train", train_data)
    for update in range(start, updates):
        t0 = time.perf_counter()
        lk, uk, uk2 = sampler.next_step()
        total = torch.zeros((), device=dev)
        parts: dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)
        # labelled
        frame = view.load(lk, labeled=True)
        image, target, geometry = legacy._s3_frame_tensors(frame, dev, labeled=True)
        feature, packet = _teacher_features(teacher, student, image, geometry)
        from .teacher_training import _sdf_targets4
        native = tuple(int(v) for v in image.shape[-3:])
        sdf_gt = _sdf_targets4(target, geometry)[:, 1:].detach() if "spacing_zyx" in geometry else None
        ones = torch.ones(target.shape, device=dev)
        loss_l, t_l = fit_loss(model, feature, _onehot(target), ones, sdf_gt, ones if sdf_gt is not None else None, native, beta=beta)
        total = total + loss_l; parts["loss_l"] = float(loss_l.detach())
        u_losses = []
        for key in (uk, uk2):
            uframe = view.load(key, labeled=False)
            uimage, _, ugeom = legacy._s3_frame_tensors(uframe, dev, labeled=False)
            ufeat, upacket = _teacher_features(teacher, student, uimage, ugeom)
            probs = upacket["region_logits"].softmax(1).detach()
            conf = probs.max(1).values
            usdf = _public_sdf(upacket)
            lu, _ = fit_loss(model, ufeat, probs, conf, usdf, (conf >= 0.9).float(), tuple(int(v) for v in uimage.shape[-3:]), beta=beta)
            u_losses.append(lu)
            with torch.no_grad():
                z = model.embed(ufeat)
                last_window_z.append(z.permute(0, 2, 3, 4, 1).reshape(-1, dim)[::7].detach())
                idx = nearest_two(z, model.codes())[0].reshape(-1)
                usage_counts += torch.bincount(idx, minlength=size).float()
        loss_u = 0.5 * (u_losses[0] + u_losses[1])
        total = total + loss_u; parts["loss_u"] = float(loss_u.detach())
        # usage-entropy regulariser on soft assignments of the last U frame (maximise entropy)
        sim = torch.einsum("bc...,kc->bk...", model.embed(ufeat), model.codes())
        soft = (sim * 10.0).softmax(dim=1).mean(dim=(0, 2, 3, 4))
        entropy = -(soft * soft.clamp_min(1e-12).log()).sum()
        total = total - entropy_weight * entropy
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite S3-SA loss at update {update}")
        total.backward(); optimizer.step()
        update_no = update + 1
        if restart_every > 0 and update_no % restart_every == 0 and last_window_z:
            restarted = restart_dead_codes(model, torch.cat(last_window_z), usage_counts)
            usage_counts.zero_(); last_window_z.clear(); parts["restarted_codes"] = restarted
        if update_no % 50 == 0 or update_no == updates:
            freq = usage_counts / usage_counts.sum().clamp_min(1.0)
            ppl = float(torch.exp(-(freq * freq.clamp_min(1e-12).log()).sum()))
            with open(diag_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"update": update_no, "loss_total": float(total.detach()), "window_perplexity": ppl, "seconds": time.perf_counter() - t0, **parts}) + "\n")
            print(f"[S3-SA] update {update_no}/{updates} loss {float(total.detach()):.4f} ppl {ppl:.1f}", flush=True)
        if update_no % checkpoint_every == 0 or update_no == updates:
            tmp = ckpt_path.with_name(f".{ckpt_path.name}.{os.getpid()}.tmp")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
                        "update": update_no, "usage_counts": usage_counts.detach().cpu(), "rng": torch.get_rng_state()}, tmp)
            os.replace(tmp, ckpt_path)
    # tau* from TRAIN cells only: all labelled frames + an evenly spaced subset of U frames
    feats = []
    lkeys = list(view.labeled_keys()); ukeys = list(view.unlabeled_keys())
    stride = max(1, len(ukeys) // max(1, tau_frames))
    with torch.no_grad():
        for key, lab in [(k, True) for k in lkeys] + [(k, False) for k in ukeys[::stride][:tau_frames]]:
            fr = view.load(key, labeled=lab)
            im, _, ge = legacy._s3_frame_tensors(fr, dev, labeled=lab)
            feats.append(_teacher_features(teacher, student, im, ge)[0])
        tau = calibrate_tau_star(model, feats)
    model.freeze()
    artifact = root / "codebook.pt"
    torch.save({"codebook": model.state_dict(), "tau_star": tau, "patch": list(patch), "size": size, "dim": dim,
                "schema": "self_audit.candidate_pg.shape_atoms.v1"}, artifact)
    import hashlib
    receipt = {"status": "COMPLETED", "artifact": str(artifact), "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
               "tau_star": tau, "patch": list(patch), "updates": updates, "codebook_frozen": True,
               "val_reconstruction": reconstruction_report(model, teacher, student, val_data, dev)}
    (root / "S3_SA_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def load_shape_atoms(path: Any, device: str | torch.device = "cpu") -> ShapeAtomCodebook:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ShapeAtomCodebook(int(payload["size"]), int(payload["dim"]), tuple(payload["patch"])).to(device)
    model.load_state_dict(payload["codebook"], strict=True)
    return model.freeze()


__all__ = [
    "assemble_full_res", "load_shape_atoms", "patch_split", "reconstruction_report", "run_shape_atoms",
    "AtomDecoder", "ShapeAtomCodebook", "StudentAtomBranch", "atom_loss", "calibrate_tau_star", "class_dice",
    "code_usage", "commitment", "crop_to_native", "fg_dice", "fit_loss", "l2n", "lambda_cb", "margin",
    "nearest_two", "pm_loss", "restart_dead_codes", "split_patches", "teacher_atoms", "upsample_cells",
]
