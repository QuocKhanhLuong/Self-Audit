"""Candidate PG training engine.

This module deliberately owns orchestration only.  Backbone, view/reliability,
codebook and evaluation implementations are imported at run time so the PG
track remains additive and the historical tracks retain their public APIs.
The engine is runnable on a tiny synthetic CPU manifest and fails closed for a
real run when a pinned manifest or required stage artifact is absent.
"""
from __future__ import annotations

import copy
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import random
import subprocess
import time
from collections.abc import Sequence as ABCSequence
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

_SCHEMA = "self_audit.candidate_pg.v2"
_ARMS = ("A1", "A2", "A3", "A4")
_SWITCHES = {
    "A1": {"pg": False, "cbkd": False},
    "A2": {"pg": True, "cbkd": False},
    "A3": {"pg": False, "cbkd": True},
    "A4": {"pg": True, "cbkd": True},
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _source_git_sha() -> str:
    """Return the checked-out source commit for run provenance (never guessed)."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"
    value = completed.stdout.strip()
    return value if value else "UNKNOWN"


def config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    out: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np_state = state["numpy"]
    if isinstance(np_state, tuple):
        np.random.set_state(np_state)
    else:
        np.random.set_state((str(np_state["generator"]), np_state["keys"].cpu().numpy(), int(np_state["position"]), int(np_state["has_gauss"]), float(np_state["cached_gaussian"])))
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_save(payload: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return _sha(path)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _finite_or_none(value: Any) -> Any:
    """Diagnostics only: a non-finite float (e.g. a mean over an empty gate) is logged as null instead of aborting the run.

    The training loss itself is guarded separately (non-finite total loss raises); this only touches the diagnostic log."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(v) for v in value]
    return value


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _finite_or_none(dict(row)) if path.name == "diag.jsonl" else dict(row)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(clean, sort_keys=True, allow_nan=False, default=str) + "\n")


class _SyntheticFrame:
    def __init__(self, image: Tensor, target: Tensor | None, patient_id: str, frame_index: int):
        self.image, self.target = image, target
        self.patient_id, self.frame_index = patient_id, frame_index
        d, h, w = image.shape[-3:]
        self.geometry = {"shape": (d, h, w), "spacing_zyx": (1.0, 1.0, 1.0), "affine_zyx": np.eye(4).tolist(), "context": "full_volume"}


class _SyntheticStageView:
    """Minimal key/load view matching the real PG TRAIN/VAL/TEST contract."""

    def __init__(self, labeled: Sequence[_SyntheticFrame], unlabeled: Sequence[_SyntheticFrame]):
        self._labeled = {(str(frame.patient_id), int(frame.frame_index)): frame for frame in labeled}
        self._unlabeled = {(str(frame.patient_id), int(frame.frame_index)): frame for frame in unlabeled}
        self._frames = {**self._unlabeled, **self._labeled}

    def labeled_keys(self) -> list[tuple[str, int]]:
        return list(self._labeled)

    def unlabeled_keys(self) -> list[tuple[str, int]]:
        return list(self._unlabeled)

    def load(self, key: Any, *, labeled: bool) -> _SyntheticFrame:
        del labeled
        return self._frames[tuple(key)]


class _SyntheticStageBundle:
    def __init__(self, labeled: Sequence[_SyntheticFrame], unlabeled: Sequence[_SyntheticFrame]):
        self.train = _SyntheticStageView(labeled, unlabeled)


class _LazyTrainImageSequence(ABCSequence):
    """Indexable TRAIN image stream for S1 without eager volume materialisation.

    CineMA's S1 runner only needs ``len``/``getitem`` plus stable identity
    metadata.  Keeping keys and the FuseData view here means image payloads are
    read exactly once when a training or holdout index is consumed.
    """

    def __init__(self, source: Any, labeled_keys: Sequence[Any], unlabeled_keys: Sequence[Any]):
        self.source = source
        self._keys = [(key, True) for key in labeled_keys] + [(key, False) for key in unlabeled_keys]
        self.identities = [
            {"patient_id": str(key[0]), "frame_index": int(key[1]), "split": "train"}
            for key, _ in self._keys
        ]

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if not isinstance(index, int):
            raise TypeError("S1 TRAIN sequence index must be an integer")
        key, labeled = self._keys[index]
        frame = self.source.load(key, labeled=labeled)
        # The public CineMA canvas is a source-model view only. Resolve its
        # centre from native dimensions/spacing; never consult a TRAIN mask.
        # FuseData intentionally strips extra geometry keys, so attach this
        # explicit preprocessing declaration after loading the native frame.
        from .cinema_init import OFFICIAL_SAX_SIZE, OFFICIAL_SAX_SPACING
        geometry = copy.deepcopy(frame.geometry)
        native = tuple(int(v) for v in frame.image.shape[-3:])
        source_spacing = tuple(reversed(OFFICIAL_SAX_SPACING))
        resized = tuple(max(1, round(n * float(s) / t)) for n, s, t in
                        zip(native, geometry['spacing_zyx'], source_spacing, strict=True))
        target = (OFFICIAL_SAX_SIZE[2], OFFICIAL_SAX_SIZE[0], OFFICIAL_SAX_SIZE[1])
        geometry['cinema_crop'] = {
            'start_zyx': [max(0, (n - t) // 2) for n, t in zip(resized, target, strict=True)],
            'size_zyx': list(target), 'policy': 'image_center_no_gt_v1',
        }
        return {
            "image": frame.image,
            "geometry": geometry,
            "patient_id": str(frame.patient_id),
            "frame_index": int(frame.frame_index),
            "split": "train",
        }


class _TinyTeacher(nn.Module):
    """Shape-faithful synthetic teacher used only by CPU software checks.

    The PG contracts are channel-sensitive: S3 consumes frozen teacher level-3
    features at 256 channels, while the CBKD student branch consumes 128
    channels.  Keep the fixture's five feature levels identical in native
    geometry, but expose the locked MedNeXt channel tuple so the complete
    S3/A3/A4 path exercises the real codebook and branch APIs.
    """
    feature_channels = (32, 64, 128, 256, 512)
    feature_phases = (
        ((1, 1, 1), (0.0, 0.0, 0.0)),
        ((1, 2, 2), (0.0, 0.0, 0.0)),
        ((1, 4, 4), (0.0, 0.0, 0.0)),
        ((2, 8, 8), (0.0, 0.0, 0.0)),
        ((2, 16, 16), (0.0, 0.0, 0.0)),
    )
    teacher_phases = feature_phases

    def __init__(self):
        super().__init__()
        self.body = nn.Conv3d(1, self.feature_channels[0], 1)
        self.levels = nn.ModuleList([
            nn.Conv3d(self.feature_channels[index], self.feature_channels[index + 1], 1)
            for index in range(4)
        ])
        self.region = nn.Conv3d(self.feature_channels[0], 4, 1)
        self.boundary = nn.Conv3d(self.feature_channels[0], 4, 1)
        # Public PG SDF is foreground-three; q3d compatibility receives the
        # concatenated background + foreground four-channel representation.
        self.sdf = nn.Conv3d(self.feature_channels[0], 3, 1)

    def forward(self, image: Tensor, geometry: Mapping[str, Any] | None = None, *, return_features: bool = True, skip_dropout: float = 0.0, dropout_generator: torch.Generator | None = None) -> Mapping[str, Any]:
        del geometry, return_features, skip_dropout, dropout_generator
        feat = self.body(image)
        features = [feat]
        native = tuple(int(v) for v in image.shape[-3:])
        for index, level in enumerate(self.levels, start=1):
            level_feat = level(features[-1])
            phase = self.feature_phases[index][0]
            target = tuple(max(1, int(math.ceil(native[axis] / phase[axis]))) for axis in range(3))
            if tuple(level_feat.shape[-3:]) != target:
                level_feat = torch.nn.functional.interpolate(level_feat, size=target, mode="trilinear", align_corners=False)
            features.append(level_feat)
        sdf = torch.tanh(self.sdf(feat))
        return {
            "region_logits": self.region(feat),
            "boundary_logits": self.boundary(feat),
            "sdf": sdf,
            "sdf_all": torch.cat((torch.zeros_like(sdf[:, :1]), sdf), dim=1),
            "features": tuple(features),
        }


class _TinyCinemaEncoder(nn.Module):
    """Deterministic source fixture for the synthetic S1 runner seam.

    It exposes both locked CineMA tap names on the same H,W,D canvas shape
    expected by ``run_cinema_init``.  The fixture has no pretrained claim and
    is used only when ``runtime.synthetic`` is explicit.
    """

    def forward(self, image: Tensor) -> Mapping[str, Tensor]:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("synthetic CineMA fixture expects [B,1,H,W,D]")
        base = image.float()
        conv = torch.nn.functional.adaptive_avg_pool3d(base, (16, 24, 24)).expand(-1, 8, -1, -1, -1)
        token_grid = torch.nn.functional.adaptive_avg_pool3d(base, (16, 12, 12)).expand(-1, 8, -1, -1, -1)
        tokens = token_grid.permute(0, 2, 3, 4, 1).reshape(image.shape[0], -1, 8)
        return {"conv_1_8": conv, "tokens_1_16": tokens}


def _module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _build_teacher(config: Mapping[str, Any], *, synthetic: bool) -> tuple[nn.Module, dict[str, Any]]:
    if synthetic:
        return _TinyTeacher(), {"kind": "synthetic_fixture", "synthetic": True}
    if config.get("pg", {}).get("teacher_source", "mednext") == "m3hl":
        # M3HL: a hash-pinned external 2D U-Net, loaded from the config (no S2 receipt/state-dict path), frozen, probabilities only.
        from .teacher_m3hl import build_m3hl_teacher

        spec = config["pg"]["m3hl"]
        checkpoint = Path(spec["checkpoint"]["path"])
        if _sha(checkpoint) != spec["checkpoint"]["sha256"]:
            raise ValueError("M3HL teacher checkpoint SHA-256 differs from the config pin")
        root = Path(spec.get("source_root", "external/M3HL"))
        root = root if root.is_absolute() else Path(__file__).resolve().parents[2] / root
        return build_m3hl_teacher(checkpoint, root), {"kind": "m3hl", "checkpoint_sha256": spec["checkpoint"]["sha256"], "source_commit": spec["source_commit"]}
    mod = _module("self_audit.candidate_pg.teacher_mednext")
    if mod is None or not hasattr(mod, "build_teacher"):
        raise RuntimeError("PG real run requires candidate_pg.teacher_mednext.build_teacher")
    result = mod.build_teacher(config)
    return result if isinstance(result, tuple) else (result, {"kind": "mednext_l"})


def _build_student(config: Mapping[str, Any], *, synthetic: bool) -> nn.Module:
    mod = _module("self_audit.candidate_pg.student_pg")
    if mod is None:
        raise RuntimeError("PG real run requires candidate_pg.student_pg")
    factory = getattr(mod, "build_student", None)
    if not callable(factory):
        raise RuntimeError("candidate_pg.student_pg.build_student is required")
    return factory(config)


def _frame_tensors(frame: Any, device: torch.device, *, labeled: bool) -> tuple[Tensor, Tensor | None, Mapping[str, Any]]:
    image = torch.as_tensor(frame.image, dtype=torch.float32, device=device)
    if image.ndim == 4:
        image = image.unsqueeze(0)
    if image.ndim != 5 or image.shape[1] != 1 or not torch.isfinite(image).all():
        raise ValueError("PG frame image must be finite [B,1,D,H,W]")
    target = getattr(frame, "target", None)
    if labeled:
        if target is None:
            raise ValueError("labeled PG frame is missing target")
        target = torch.as_tensor(target, dtype=torch.long, device=device)
        if target.ndim == 3:
            target = target.unsqueeze(0)
        if target.shape != image.shape[:1] + image.shape[-3:] or bool(((target < 0) | (target > 3)).any()):
            raise ValueError("PG target shape/class range mismatch")
    return image, target, frame.geometry


def _synthetic_data(config: Mapping[str, Any]) -> dict[str, list[Any]]:
    # Keep the smoke canvas large enough for a non-empty 16-pixel aligned
    # CutMix box; this remains a tiny CPU fixture, not a clinical volume.
    shape = tuple(config.get("runtime", {}).get("synthetic_shape", (2, 64, 64)))
    n = int(config.get("runtime", {}).get("synthetic_frames", 8))
    labeled: list[Any] = []
    unlabeled: list[Any] = []
    for i in range(max(n, 2)):
        image = torch.full((1, 1, *shape), 0.5 + 0.01 * (i % 2))
        target = torch.zeros((1, *shape), dtype=torch.long)
        target[..., 1:-1, 1:-1] = (i % 3) + 1
        # Two synthetic patients give each patient multiple frames spanning
        # all three foreground classes while preserving different-patient U
        # pairing for the software-only sampler.
        labeled.append(_SyntheticFrame(image, target, f"synthetic{i%2}", i))
        unlabeled.append(_SyntheticFrame(image, None, f"synthetic{i%2}", i))
    return {"labeled": labeled, "unlabeled": unlabeled, "val": labeled[: min(4, len(labeled))], "test": labeled[: min(4, len(labeled))]}


def _load_data(config: Mapping[str, Any], *, synthetic: bool) -> dict[str, Any]:
    if synthetic:
        return _synthetic_data(config)
    from self_audit.candidate_pg.data import PGSampler, open_pg_data
    bundle = open_pg_data(config)
    train, val, test = bundle.train, bundle.val, bundle.test
    return {"labeled": [train.load(k, labeled=True) for k in train.labeled_keys()],
            # U is intentionally lazy: image-only TRAIN frames can be large,
            # and the locked sampler is the sole owner of the per-update
            # patient-balanced two-frame read contract.
            "unlabeled": [], "unlabeled_keys": list(train.unlabeled_keys()),
            "val": [val.load(k, labeled=True) for k in val.labeled_keys()],
            "test": [test.load(k, labeled=True) for k in test.labeled_keys()],
            "objects": bundle, "sampler": PGSampler(bundle, seed=42)}


def _grouped_backward(student: nn.Module, step_result: Mapping[str, Any], clip_norm: float, *, per_term: bool) -> dict[str, Any]:
    """Backward with the CBKD-SA terms clipped as their own group (rev 4 section 8.8).

    ``main`` = sup + lambda_u * PG, ``cbkd`` = lambda_cb * (L_pm + L_atom).  Each group is clipped to
    ``clip_norm`` separately and the clipped gradients are summed.  Logs group norms/factors and, on
    ``per_term`` rows, the sup / PG / CBKD norms.
    """

    diag = step_result["diagnostics"]
    lam_u, lam_cb = float(diag.get("lambda_u", 0.0)), float(diag.get("lambda_cb", 0.0))
    params = [p for p in student.parameters() if p.requires_grad]
    main = step_result["loss_sup"] + lam_u * step_result["loss_pg"]
    cb = lam_cb * step_result["loss_cbkd"]

    def grads(loss: Tensor, retain: bool) -> list[Tensor]:
        if not loss.requires_grad:
            return [torch.zeros_like(p) for p in params]
        out = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        return [torch.zeros_like(p) if g is None else g for p, g in zip(params, out)]

    def norm(gs: Sequence[Tensor]) -> float:
        return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs)))

    g_main = grads(main, retain=cb.requires_grad or per_term)
    g_cb = grads(cb, retain=per_term)
    n_main, n_cb = norm(g_main), norm(g_cb)
    f_main = min(1.0, clip_norm / (n_main + 1e-6)) if clip_norm > 0 else 1.0
    f_cb = min(1.0, clip_norm / (n_cb + 1e-6)) if clip_norm > 0 else 1.0
    for p, a, b in zip(params, g_main, g_cb):
        p.grad = a * f_main + b * f_cb
    info = {"norm_main": n_main, "norm_cbkd": n_cb, "clip_factor_main": f_main, "clip_factor_cbkd": f_cb}
    if per_term:
        info["norm_sup"] = norm(grads(step_result["loss_sup"], retain=True))
        info["norm_pg"] = norm(grads(lam_u * step_result["loss_pg"], retain=False)) if (lam_u * step_result["loss_pg"]).requires_grad else 0.0
    return info


def _lambda_u(update: int, total_updates: int, *, warmup_updates_labeled_only: int = 800,
              ramp_fraction: float = 0.1, maximum: float = 1.0) -> float:
    """Backward-compatible locked PG ramp helper used by focused tests.

    The production student path delegates ramping to ``training_step``; this
    pure helper remains part of the historical engine test surface.
    """
    if update < warmup_updates_labeled_only:
        return 0.0
    remaining = max(1, int(math.ceil(max(1, total_updates) * float(ramp_fraction))))
    return float(min(maximum, max(0, update - warmup_updates_labeled_only + 1) / remaining * maximum))


def _schedule_lr(optimizer: torch.optim.Optimizer, *, update: int, total_updates: int,
                 lr: float, warmup_updates: int = 0) -> float:
    """Reuse the canonical Q3D/FUSE cosine-warmup scheduler by value."""
    from self_audit.candidate_q3d.engine import _schedule_lr as canonical_schedule

    base_lrs = [float(lr)] * len(optimizer.param_groups)
    fraction = float(warmup_updates) / float(max(1, total_updates))
    canonical_schedule(
        optimizer, step=int(update), total=int(total_updates),
        base_lrs=base_lrs, warmup_fraction=fraction,
    )
    return float(optimizer.param_groups[0]["lr"])


def _extract_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if not isinstance(output, Mapping):
        raise TypeError("PG model forward must return a tensor or mapping")
    for key in ("logits", "region_logits", "fusion_logits"):
        value = output.get(key)
        if isinstance(value, Tensor):
            return value
    raise KeyError("PG model output has no logits/region_logits")


def _evaluate(model: nn.Module, frames: Sequence[Any], device: torch.device, *, path: str = "student_only", teacher: nn.Module | None = None) -> dict[str, Any]:
    if not frames:
        return {"fg_dice": float("nan")}
    was_training = model.training
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for frame in frames:
            image, target, geometry = _frame_tensors(frame, device, labeled=True)
            if path in {"fusion", "fusion_weak"}:
                if teacher is None:
                    raise ValueError("fusion_weak evaluation requires frozen teacher")
                with torch.no_grad():
                    target_output = teacher(image, geometry, return_features=True)
                features = target_output.get("features") if isinstance(target_output, Mapping) else None
                keep = torch.ones((image.shape[0], 1, 1, 1, 1), device=image.device)
                pred = _extract_logits(model(image, geometry, teacher_features=features, keep_mask=keep))
            else:
                pred = _extract_logits(model(image, geometry))
            cls = pred.argmax(1)
            records.append({
                "patient": str(getattr(frame, "patient_id", getattr(frame, "patient", "unknown"))),
                "frame": int(getattr(frame, "frame_index", getattr(frame, "frame", 0))),
                "target": target[0].detach().cpu(), "prediction": cls[0].detach().cpu(),
                "geometry": geometry,
            })
    model.train(was_training)
    evaluator = _module("self_audit.candidate_pg.evaluation")
    if evaluator is None or not hasattr(evaluator, "evaluate_records"):
        raise RuntimeError("PG evaluation.evaluate_records is required for epoch metrics")
    result = evaluator.evaluate_records(records, path="student_only")
    per = result.get("per_class", {}) if isinstance(result, Mapping) else {}
    return {
        "fg_dice": result.get("fg_dice"), "fg_hd95": result.get("fg_hd95"), "fg_assd": result.get("fg_assd"),
        "patient_count": result.get("patient_count", len(frames)),
        "class_1_dice": (per.get("RV") or {}).get("dice"), "class_2_dice": (per.get("MYO") or {}).get("dice"), "class_3_dice": (per.get("LV") or {}).get("dice"),
        "class_1_hd95": (per.get("RV") or {}).get("hd95_mm"), "class_2_hd95": (per.get("MYO") or {}).get("hd95_mm"), "class_3_hd95": (per.get("LV") or {}).get("hd95_mm"),
        "class_1_assd": (per.get("RV") or {}).get("assd_mm"), "class_2_assd": (per.get("MYO") or {}).get("assd_mm"), "class_3_assd": (per.get("LV") or {}).get("assd_mm"),
        "per_patient": result.get("per_patient", {}),
    }


def _epoch_full_splits() -> tuple[str, ...]:
    """Splits for the extra per-epoch Dice/Jaccard/HD95/ASD table (env ``EPOCH_FULL_METRICS=val,test``; off when unset).

    An environment switch, not a config field, so the resolved config hash and resume identity are unchanged.  Descriptive
    only: selection still reads val/student_only FG Dice from the locked epoch logger.
    """
    import os

    return tuple(s for s in (v.strip() for v in os.environ.get("EPOCH_FULL_METRICS", "").split(",")) if s in {"train", "val", "test"})


def _evaluate_full(model: nn.Module, frames: Sequence[Any], device: torch.device) -> dict[str, Any]:
    """Dice, Jaccard, HD95 and ASD per class and mean: native (mm, true spacing) and SSL4MIS-compatible (voxels, medpy)."""
    from .evaluation_ssl4mis_compat import evaluate_volumes

    if not frames:
        return {}
    was_training = model.training
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for frame in frames:
            image, target, geometry = _frame_tensors(frame, device, labeled=True)
            cls = _extract_logits(model(image, geometry)).argmax(1)
            spacing = geometry.get("spacing_zyx", geometry.get("spacing_zyx_mm"))
            records.append({"patient": str(getattr(frame, "patient_id", "unknown")), "frame": int(getattr(frame, "frame_index", 0)),
                            "target": target[0].cpu().numpy(), "prediction": cls[0].cpu().numpy(),
                            "geometry": {"spacing_zyx": [float(v) for v in spacing]}})
    model.train(was_training)
    def pack(result: Mapping[str, Any]) -> dict[str, Any]:
        return {"mean": result["mean"], "per_class": result["per_class"], "empty_cases": result["empty_cases"]}
    return {"volumes": len(records), "native_mm": pack(evaluate_volumes(records, voxelspacing_key="spacing_zyx")),
            "ssl4mis_voxels": pack(evaluate_volumes(records))}


def _append_epoch_full(arm_dir: Path, row: Mapping[str, Any]) -> None:
    """Append one epoch row to ``epoch_full_metrics.jsonl``/``.csv``; rows at or after this epoch (resume replays) are replaced."""
    import csv

    jpath = arm_dir / "epoch_full_metrics.jsonl"
    old = [json.loads(line) for line in jpath.read_text().splitlines() if line.strip()] if jpath.exists() else []
    old = [r for r in old if int(r["epoch"]) < int(row["epoch"])]
    rows = old + [dict(row)]
    jpath.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    flat = []
    for r in rows:
        for split in ("train", "val", "test"):
            entry = r.get(split)
            if not entry:
                continue
            for unit in ("native_mm", "ssl4mis_voxels"):
                for scope, values in [("mean", entry[unit]["mean"])] + [(c, v) for c, v in entry[unit]["per_class"].items()]:
                    flat.append({"epoch": r["epoch"], "update": r["update"], "split": split, "units": unit, "class": scope,
                                 "dice": values["dice"], "jaccard": values["jaccard"], "hd95": values["hd95"], "asd": values["asd"]})
    with (arm_dir / "epoch_full_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "update", "split", "units", "class", "dice", "jaccard", "hd95", "asd"])
        writer.writeheader()
        writer.writerows(flat)


def _save_checkpoint(out: Path, state: Mapping[str, Any], *, arm: str, epoch: int, update: int) -> dict[str, Any]:
    path = out / f"resume_{arm}.pt"
    sha = _atomic_save(dict(state), path)
    receipt = {"path": str(path.resolve()), "sha256": sha, "arm": arm, "epoch": epoch, "update": update}
    _json(out / f"resume_{arm}.sha256.json", receipt)
    return receipt


def _restore_checkpoint(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != _SCHEMA or payload.get("config_sha256") != expected.get("config_sha256"):
        raise ValueError("PG resume checkpoint schema/config identity mismatch")
    return payload


def _read_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"PG stage receipt missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"PG stage receipt must be a JSON object: {path}")
    return value


def _receipt_artifact(receipt: Mapping[str, Any], *, names: Sequence[str], label: str) -> Path:
    value = next((receipt.get(name) for name in names if receipt.get(name)), None)
    if value is None:
        raise ValueError(f"{label} receipt has no checkpoint/artifact path")
    artifact = Path(str(value))
    if not artifact.is_file():
        raise FileNotFoundError(f"{label} artifact missing: {artifact}")
    expected = receipt.get("checkpoint_sha256", receipt.get("artifact_sha256", receipt.get("sha256")))
    if expected and _sha(artifact) != str(expected):
        raise ValueError(f"{label} artifact SHA-256 mismatch")
    return artifact


def run_training(config: Mapping[str, Any], out: str | Path, *, device: str = "cpu", stage: str = "all", resume: bool = False, max_updates: int | None = None, arm: str | None = None) -> dict[str, Any]:
    """Execute S1 -> S2 -> S3 -> A1..A4 with deterministic resumable state."""
    cfg = copy.deepcopy(dict(config))
    if cfg.get("schema", _SCHEMA) != _SCHEMA:
        raise ValueError("PG config schema must be self_audit.candidate_pg.v2")
    if cfg.get("seed", 42) != 42:
        raise ValueError("PG seed is locked to 42")
    synthetic = bool(cfg.get("runtime", {}).get("synthetic", False))
    if stage not in {"all", "S1", "S2", "S3", "arms"}:
        raise ValueError("PG stage must be all, S1, S2, S3 or arms")
    if arm is not None and stage != "arms":
        raise ValueError("a single PG arm requires stage=arms after shared S1/S2/S3")
    selected_arms = (arm,) if arm else _ARMS
    if any(a not in _ARMS for a in selected_arms):
        raise ValueError("unknown PG arm")
    target_device = torch.device(device)
    out_path = Path(out).resolve()
    lock_path = out_path / "EXPERIMENT_LOCK.json"
    if lock_path.is_file():
        previous_lock = json.loads(lock_path.read_text())
        if previous_lock.get("requested_max_updates") != max_updates:
            raise ValueError("PG update limit differs from the existing run; use a separate profile directory")
    out_path.mkdir(parents=True, exist_ok=True)
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), Mapping) else {}
    if runtime_cfg.get("deterministic", True) is not True or runtime_cfg.get("tf32", False) is not False:
        raise ValueError("PG runtime requires deterministic=true and tf32=false")
    cublas_workspace = str(runtime_cfg.get("cublas_workspace_config", ":4096:8"))
    from .runtime import configure_runtime
    runtime_flags = configure_runtime(target_device, cublas_workspace)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Set deterministic/runtime process flags before any seed or CUDA context
    # initialisation (including S1's injected/official encoder path).
    _seed(42)
    warmup_override = runtime_cfg.get("synthetic_warmup_override")
    if warmup_override is not None:
        if not synthetic or not isinstance(warmup_override, Mapping):
            raise ValueError("synthetic_warmup_override is available only for explicit synthetic runs")
        _json(out_path / "SYNTHETIC_WARMUP_OVERRIDE.json", dict(warmup_override))
    if runtime_cfg.get("num_threads") is not None:
        torch.set_num_threads(int(runtime_cfg["num_threads"]))
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), Mapping) else {}
    data_identity = {
        "split_lock": data_cfg.get("split_lock"),
        "split_lock_sha256": data_cfg.get("split_lock_sha256"),
        "source_k8_sha256": data_cfg.get("source_k8_sha256"),
        "manifests": copy.deepcopy(data_cfg.get("manifests")),
    }
    identity = {
        "schema": _SCHEMA, "config_sha256": config_sha256(cfg), "seed": 42,
        "synthetic": synthetic, "deterministic": True, "tf32": False,
        "cublas_workspace_config": cublas_workspace,
        **runtime_flags,
        "source_git_sha": _source_git_sha(), "data_identity": data_identity,
        "requested_max_updates": max_updates,
        "profile_only": max_updates is not None and not synthetic,
    }
    _json(out_path / "EXPERIMENT_LOCK.json", identity)
    # A single predeclared adaptation may reduce two strong views to one for
    # every arm.  It is an explicit receipt, never a hidden retry or arm tune.
    runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), Mapping) else {}
    if bool(runtime.get("oom_reduce_strong_views", False)):
        cfg.setdefault("augmentation", {})["n_strong_views"] = 1
        identity["config_sha256"] = config_sha256(cfg)
        _json(out_path / "EXPERIMENT_LOCK.json", identity)
        _json(out_path / "OOM_ADAPTATION.json", {"status": "APPLIED", "reason": "explicit_runtime_oom_reduce_strong_views", "n_strong_views": 1, "arms": list(_ARMS)})
    data = _load_data(cfg, synthetic=synthetic)
    result: dict[str, Any] = {"schema": _SCHEMA, "status": "RUNNING", "identity": copy.deepcopy(identity), "profile_only": identity["profile_only"], "stages": {}, "arms": {}}
    # S1: defer to the CineMA worker when available. Synthetic mode records an
    # explicit fixture receipt and never pretends to be pretrained CineMA.
    if stage in {"all", "S1"}:
        s1_receipt_path = out_path / "S1_RECEIPT.json"
        reused_s1 = False
        if resume and s1_receipt_path.is_file():
            prior_s1 = _read_receipt(s1_receipt_path)
            if prior_s1.get("status") not in {"COMPLETED", "SOFTWARE_ONLY_SYNTHETIC"}:
                raise ValueError("PG S1 resume receipt is not a completed artifact")
            if prior_s1.get("params_sha256") != config_sha256(cfg.get("cinema_init", {})):
                raise ValueError("PG S1 resume recipe identity mismatch")
            if prior_s1.get("runtime_sha256") != config_sha256(runtime_cfg):
                raise ValueError("PG S1 resume runtime identity mismatch")
            _receipt_artifact(prior_s1, names=("checkpoint_path", "checkpoint", "student_init"), label="S1")
            result["stages"]["S1"] = {**prior_s1, "run_identity": copy.deepcopy(identity), "resume_reused": True}
            reused_s1 = True
        if not reused_s1:
            cine = _module("self_audit.candidate_pg.cinema_init")
            if cine is None or not hasattr(cine, "run_cinema_init"):
                raise RuntimeError("PG S1 requires candidate_pg.cinema_init.run_cinema_init")
        if not reused_s1 and synthetic:
            # Exercise the same S1 preprocessing, tap alignment, cosine loss,
            # checkpoint, and encoder-only artifact loop as a real run while
            # injecting only a tiny software source/student fixture.
            s1_cfg = copy.deepcopy(cfg)
            s1_cfg.setdefault("cinema_init", {})["synthetic"] = True
            s1_cfg["cinema_init"]["updates"] = int(runtime_cfg.get("synthetic_updates", 1))
            s1_student = _build_student(cfg, synthetic=True)
            with contextlib.redirect_stdout(io.StringIO()):
                result["stages"]["S1"] = cine.run_cinema_init(
                    s1_cfg, out_path / "S1", data["labeled"] + data["unlabeled"],
                    device=device, cinema_encoder=_TinyCinemaEncoder(), student_encoder=s1_student,
                    max_updates=max_updates,
                )
        elif not reused_s1:
            # CineMA initialization is image-only but must see the complete
            # TRAIN image pool: the eight locked labeled masks plus every
            # remaining unlabeled TRAIN cine frame.  Pass an indexable lazy
            # sequence so S1 can split/hash identities without eagerly reading
            # thousands of full volumes.
            train_bundle = data.get("objects")
            train_source = getattr(train_bundle, "train", None) if train_bundle is not None else None
            if train_source is None:
                raise RuntimeError("PG S1 requires the TRAIN data source for unlabeled CineMA images")
            train_records = _LazyTrainImageSequence(
                train_source,
                train_source.labeled_keys(),
                data.get("unlabeled_keys", train_source.unlabeled_keys()),
            )
            result["stages"]["S1"] = cine.run_cinema_init(cfg, out_path / "S1", train_records, device=device, max_updates=max_updates)
        if reused_s1:
            _json(out_path / "S1_RECEIPT.json", result["stages"]["S1"])
        # Bind the S1 receipt to the exact recipe/runtime/data identity used by
        # this dispatch.  The runner's source block carries the official
        # CineMA file SHAs for real executions; synthetic receipts remain
        # explicitly injected/software-only.
        if not reused_s1:
            s1_stage = dict(result["stages"]["S1"])
            s1_stage["run_identity"] = copy.deepcopy(identity)
            s1_stage["params_sha256"] = config_sha256(cfg.get("cinema_init", {}))
            s1_stage["runtime_sha256"] = config_sha256(runtime_cfg)
            data_list = out_path / "S1" / "data_list.json"
            if data_list.is_file():
                s1_stage["data_list_sha256"] = _sha(data_list)
            result["stages"]["S1"] = s1_stage
            _json(out_path / "S1_RECEIPT.json", result["stages"]["S1"])
    if stage == "S1":
        result["status"] = "PAUSED"; _json(out_path / "RESULT.json", result); return result
    # Stage boundaries are independently reproducible: teacher construction
    # must not depend on whether S1 was run in this process or resumed.
    _seed(42)
    teacher, teacher_meta = _build_teacher(cfg, synthetic=synthetic)
    teacher.to(target_device)
    view_mod = _module("self_audit.candidate_pg.views")
    # Stage-only invocations import the frozen S2 teacher; they never create a
    # fresh random teacher behind a successful receipt.
    if stage not in {"all", "S1", "S2"} and cfg.get("pg", {}).get("target_mode") != "self" and cfg.get("pg", {}).get("teacher_source", "mednext") != "m3hl":
        s2_receipt = _read_receipt(out_path / "S2_RECEIPT.json")
        s2_artifact = _receipt_artifact(s2_receipt, names=("checkpoint", "checkpoint_path"), label="S2")
        s2_payload = torch.load(s2_artifact, map_location=target_device, weights_only=False)
        state = s2_payload.get("model", s2_payload.get("teacher", s2_payload)) if isinstance(s2_payload, Mapping) else s2_payload
        if not isinstance(state, Mapping):
            raise ValueError("S2 checkpoint does not contain a teacher model state")
        teacher.load_state_dict(state, strict=True)
        teacher.eval(); teacher.requires_grad_(False)
    teacher_opt = None
    t_updates = int(cfg.get("runtime", {}).get("synthetic_updates", 1) if synthetic else cfg.get("teacher", {}).get("updates", 8000))
    if max_updates is not None: t_updates = min(t_updates, max_updates)
    if stage in {"all", "S2"}:
        # The dedicated S2 owner implements exact UniMatch streams, CutMix,
        # q3d teacher/SDF compatibility, threshold/ramp state, epoch metrics,
        # and resumable checkpoints.  Keep this orchestration layer additive:
        # it injects the already-constructed teacher and sampler, then consumes
        # the returned frozen artifact for S3/arms.
        teacher_training = _module("self_audit.candidate_pg.teacher_training")
        if teacher_training is None or not hasattr(teacher_training, "run_teacher_stage"):
            raise RuntimeError("PG S2 requires candidate_pg.teacher_training.run_teacher_stage")
        train_source = data.get("objects").train if data.get("objects") is not None else data
        val_source = data.get("objects").val if data.get("objects") is not None else data.get("val")
        test_source = data.get("objects").test if data.get("objects") is not None else data.get("test")
        s2_sampler = data.get("sampler") if data.get("sampler") is not None else None
        s2_view = view_mod.StatefulViewGenerator(seed=42 + 2003) if view_mod is not None else None
        s2_receipt = teacher_training.run_teacher_stage(
            cfg, out_path, train_source, val_source, test_source, device=device,
            teacher=teacher, sampler=s2_sampler, view_generator=s2_view,
            resume=resume, max_updates=t_updates, seed=42,
        )
        teacher.eval(); teacher.requires_grad_(False)
        result["stages"]["S2"] = {**dict(s2_receipt), "teacher": teacher_meta, "run_identity": copy.deepcopy(identity)}
        _json(out_path / "S2_RECEIPT.json", result["stages"]["S2"])
    if stage == "S2":
        result["status"] = "PAUSED"; _json(out_path / "RESULT.json", result); return result
    s3_codebook_model = None
    sa_variant = isinstance(cfg.get("cbkd"), Mapping) and cfg["cbkd"].get("variant") in ("cbkd_sa", "cbkd_sa_gated", "cbkd_sa_out")
    if sa_variant and synthetic:
        raise ValueError("cbkd_sa has no synthetic engine path; software evidence comes from the unit tests")
    if sa_variant and stage in {"all", "S3"}:
        from . import shape_atoms as _sa
        _seed(42)
        s3_alignment_student = _build_student(cfg, synthetic=False)
        sa_receipt = _sa.run_shape_atoms(
            cfg, out_path / "S3", data["objects"], data["objects"].val, teacher, s3_alignment_student,
            device=device, max_updates=max_updates, resume=resume,
        )
        sa_receipt = dict(sa_receipt); sa_receipt["run_identity"] = copy.deepcopy(identity)
        result["stages"]["S3"] = sa_receipt
        _json(out_path / "S3_RECEIPT.json", sa_receipt)
        s3_codebook_model = _sa.load_shape_atoms(sa_receipt["artifact"], target_device)
    elif stage in {"all", "S3"}:
        codebook = _module("self_audit.candidate_pg.codebook")
        if codebook is not None and hasattr(codebook, "run_codebook") and not synthetic:
            train_source = data.get("objects") if not synthetic else data
            val_source = data.get("objects").val if not synthetic else {"labeled": data.get("val", []), "unlabeled": data.get("unlabeled", [])}
            test_source = data.get("objects").test if not synthetic else {"labeled": data.get("test", []), "unlabeled": data.get("unlabeled", [])}
            # run_codebook aligns the teacher's level-3 tap to the student's
            # native phase metadata; inject a fresh architecture-only student
            # for that alignment, never for an S3 learning update.
            _seed(42)
            s3_alignment_student = _build_student(cfg, synthetic=synthetic)
            s3_result = codebook.run_codebook(
                cfg, out_path / "S3", train_source, val_source, test_source,
                teacher, student=s3_alignment_student, device=device,
                max_updates=max_updates, resume=resume,
            )
            s3_result = dict(s3_result)
            s3_result["run_identity"] = copy.deepcopy(identity)
            result["stages"]["S3"] = s3_result
            artifact = s3_result.get("artifact") if isinstance(s3_result, Mapping) else None
            if artifact and Path(artifact).is_file() and hasattr(codebook, "TeacherCodebook"):
                options = dict(cfg.get("codebook", {})); options.update({"teacher_channels": 256, "embedding_dim": int(options.get("dim", 64)), "codebook_size": int(options.get("size", 128))})
                s3_codebook_model = codebook.TeacherCodebook(options).to(target_device)
                payload = torch.load(artifact, map_location=target_device, weights_only=False)
                s3_codebook_model.load_state_dict(payload.get("codebook", payload), strict=True); s3_codebook_model.freeze()
        elif codebook is not None and hasattr(codebook, "run_codebook") and synthetic:
            # Synthetic smoke still executes the canonical S3 loop.  Only the
            # explicit fixture reduces K/D and update count; target pooling,
            # teacher detachment, sampler, optimizer, diagnostics, and qT
            # split evaluation remain owned by run_codebook itself.
            s3_cfg = copy.deepcopy(cfg)
            s3_cfg["codebook"]["updates"] = int(runtime_cfg.get("synthetic_updates", 1))
            s3_cfg["codebook"]["eval_every_updates"] = 1
            s3_cfg["codebook"]["codebook"] = dict(s3_cfg["codebook"].get("codebook", {}))
            s3_cfg["codebook"]["codebook"].update({"size": 8, "dim": 8})
            _seed(42)
            s3_alignment_student = _build_student(cfg, synthetic=True)
            s3_train = _SyntheticStageBundle(data["labeled"], data["unlabeled"])
            s3_val = _SyntheticStageBundle(data["val"], data["unlabeled"])
            s3_test = _SyntheticStageBundle(data["test"], data["unlabeled"])
            s3_result = codebook.run_codebook(
                s3_cfg, out_path / "S3", s3_train, s3_val, s3_test,
                teacher, student=s3_alignment_student, device=device,
                max_updates=max_updates or int(runtime_cfg.get("synthetic_updates", 1)), resume=resume,
            )
            artifact = s3_result.get("artifact")
            if not isinstance(artifact, str) or not Path(artifact).is_file():
                raise RuntimeError("synthetic S3 runner did not produce a codebook artifact")
            s3_result = dict(s3_result)
            s3_result["status"] = "SOFTWARE_ONLY_SYNTHETIC"
            s3_result["artifact_sha256"] = _sha(Path(artifact))
            s3_result["run_identity"] = copy.deepcopy(identity)
            result["stages"]["S3"] = s3_result
            payload = torch.load(artifact, map_location=target_device, weights_only=False)
            options = {"teacher_channels": 256, "embedding_dim": 8, "codebook_size": 8}
            s3_codebook_model = codebook.TeacherCodebook(options).to(target_device)
            s3_codebook_model.load_state_dict(payload.get("codebook", payload), strict=True)
            s3_codebook_model.freeze()
        else:
            result["stages"]["S3"] = {"status": "FAILED", "reason": "candidate_pg.codebook.run_codebook unavailable", "codebook_frozen": False}
        _json(out_path / "S3_RECEIPT.json", result["stages"]["S3"])
    if stage == "S3":
        result["status"] = "PAUSED"; _json(out_path / "RESULT.json", result); return result
    needs_codebook = any(_SWITCHES[a]["cbkd"] for a in selected_arms)
    if s3_codebook_model is None and sa_variant and needs_codebook:
        if stage not in {"arms"}:
            raise RuntimeError("PG arms require the completed S3-SA codebook stage")
        from . import shape_atoms as _sa
        s3_receipt = _read_receipt(out_path / "S3_RECEIPT.json")
        s3_codebook_model = _sa.load_shape_atoms(_receipt_artifact(s3_receipt, names=("artifact", "codebook"), label="S3"), target_device)
    if s3_codebook_model is None and needs_codebook:
        if stage not in {"arms"}:
            raise RuntimeError("PG arms require the completed S3 codebook stage")
        s3_receipt = _read_receipt(out_path / "S3_RECEIPT.json")
        s3_artifact = _receipt_artifact(s3_receipt, names=("artifact", "codebook"), label="S3")
        codebook = _module("self_audit.candidate_pg.codebook")
        if codebook is None or not hasattr(codebook, "TeacherCodebook"):
            raise RuntimeError("PG arms require candidate_pg.codebook.TeacherCodebook")
        if synthetic:
            options = {"teacher_channels": 256, "embedding_dim": 8, "codebook_size": 8}
        else:
            nested = cfg.get("codebook", {}).get("codebook", {}) if isinstance(cfg.get("codebook", {}).get("codebook", {}), Mapping) else {}
            options = {"teacher_channels": 256, "embedding_dim": int(nested.get("dim", 64)), "codebook_size": int(nested.get("size", 128))}
        s3_codebook_model = codebook.TeacherCodebook(options).to(target_device)
        payload = torch.load(s3_artifact, map_location=target_device, weights_only=False)
        state = payload.get("codebook", payload) if isinstance(payload, Mapping) else payload
        s3_codebook_model.load_state_dict(state, strict=True)
        s3_codebook_model.freeze()
    sampler_template = copy.deepcopy(data.get("sampler"))
    view_mod = _module("self_audit.candidate_pg.views")
    view_template = view_mod.StatefulViewGenerator(seed=42) if view_mod is not None else None
    selected_models: dict[str, nn.Module] = {}
    selected_epochs: dict[str, int] = {}
    epoch_logs: dict[str, list[dict[str, Any]]] = {}
    # Resolve one shared student initialization and copy it byte-for-byte to
    # every arm; switches are the sole arm difference.
    _seed(42)
    shared_student = _build_student(cfg, synthetic=synthetic)
    shared_student_state = copy.deepcopy(shared_student.state_dict())
    s1_checkpoint = result.get("stages", {}).get("S1", {}).get("checkpoint_path")
    if stage not in {"S1"}:
        s1_receipt_path = out_path / "S1_RECEIPT.json"
        if s1_checkpoint is None:
            if not s1_receipt_path.is_file():
                raise RuntimeError("PG arms require a completed S1_RECEIPT.json with encoder artifact")
            s1_receipt = _read_receipt(s1_receipt_path)
            s1_checkpoint = str(_receipt_artifact(
                s1_receipt,
                names=("checkpoint_path", "checkpoint", "student_init"),
                label="S1",
            ))
        else:
            s1_checkpoint = str(_receipt_artifact(
                result["stages"]["S1"],
                names=("checkpoint_path", "checkpoint", "student_init"),
                label="S1",
            ))
        cine_loader = _module("self_audit.candidate_pg.cinema_init")
        if cine_loader is None or not hasattr(cine_loader, "load_encoder_artifact"):
            raise RuntimeError("PG S1 requires cinema_init.load_encoder_artifact")
        # The same strict encoder-only artifact boundary is used for real and
        # synthetic runs; synthetic merely injects the source encoder inside
        # run_cinema_init and never imports a raw/random checkpoint directly.
        cine_loader.load_encoder_artifact(s1_checkpoint, shared_student, device="cpu")
    # Capture the final encoder-initialized state once; all arms clone it.
    shared_student_state = copy.deepcopy(shared_student.state_dict())
    for arm_name in selected_arms:
        print(f"[{arm_name}] start", flush=True)
        _seed(42)
        switches = _SWITCHES[arm_name]
        student = _build_student(cfg, synthetic=synthetic)
        student.load_state_dict(shared_student_state, strict=True)
        if switches["cbkd"] and sa_variant and hasattr(student, "attach_code_branch"):
            from . import shape_atoms as _sa
            student.attach_code_branch(_sa.StudentAtomBranch(s3_codebook_model))
        elif switches["cbkd"] and s3_codebook_model is not None and hasattr(student, "attach_code_branch"):
            branch_cfg = dict(cfg.get("cbkd", {})); branch_cfg.update({"student_channels": 128, "embedding_dim": int(getattr(s3_codebook_model, "embedding_dim", 64))})
            student.attach_code_branch(codebook.StudentCodeBranch(branch_cfg, s3_codebook_model))
        student = student.to(target_device)
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg.setdefault("pg", {})["enabled"] = bool(switches["pg"])
        arm_cfg.setdefault("cbkd", {})["enabled"] = bool(switches["cbkd"])
        arm_identity = {
            "schema": _SCHEMA, "config_sha256": config_sha256(arm_cfg),
            "seed": 42, "synthetic": synthetic, "arm": arm_name,
            "switches": dict(switches),
            "deterministic": identity["deterministic"], "tf32": identity["tf32"],
            **runtime_flags,
            "cublas_workspace_config": identity["cublas_workspace_config"],
            "source_git_sha": identity["source_git_sha"],
            "data_identity": copy.deepcopy(identity["data_identity"]),
        }
        train_opt_cfg = cfg.get("training", {}).get("optimizer", {}) if isinstance(cfg.get("training", {}).get("optimizer", {}), Mapping) else {}
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=float(train_opt_cfg.get("lr", cfg.get("training", {}).get("lr", 1e-3))),
            weight_decay=float(train_opt_cfg.get("weight_decay", 1e-4)),
        )
        rel_mod = _module("self_audit.candidate_pg.reliability")
        threshold = None
        if rel_mod is not None:
            threshold_cfg = cfg.get("pg", {}).get("threshold", {}) if isinstance(cfg.get("pg", {}).get("threshold", {}), Mapping) else {}
            threshold = rel_mod.ClassAdaptiveThreshold(
                4, ema=float(threshold_cfg.get("ema", 0.999)),
                clip=tuple(threshold_cfg.get("clip", (0.5, 0.95))),
            )
        updates = int(cfg.get("runtime", {}).get("synthetic_updates", 2) if synthetic else cfg.get("training", {}).get("updates", 8000))
        if max_updates is not None: updates = min(updates, max_updates)
        start = 0; best = -float("inf"); best_state: dict[str, Tensor] | None = None; best_epoch: int | None = None; rows: list[dict[str, Any]] = []
        epoch_loss_sum = [0.0, 0.0, 0.0, 0.0]; epoch_loss_steps = 0
        log_mod = _module("self_audit.candidate_pg.logging")
        if log_mod is None or not hasattr(log_mod, "EpochLogger"):
            raise RuntimeError("PG arms require candidate_pg.logging.EpochLogger")
        arm_dir = out_path / "arms" / arm_name
        _json(arm_dir / "EXPERIMENT_LOCK.json", arm_identity)
        # Every arm starts from identical sampler state, preserving the
        # locked 2x2 comparison rather than chaining arm-specific cursors.
        sampler = copy.deepcopy(sampler_template)
        view_gen = copy.deepcopy(view_template)
        keep_generator = torch.Generator(device=target_device).manual_seed(42 + 1009)
        fp_generator = torch.Generator(device=target_device).manual_seed(42 + 2003)
        train_data = data.get("objects").train if sampler is not None else None
        cp = out_path / f"resume_{arm_name}.pt"
        resume_arm = bool(resume or (stage == "arms" and cp.is_file()))
        resume_payload: dict[str, Any] | None = None
        if resume_arm and cp.is_file():
            # EpochLogger's constructor conservatively truncates rows beyond
            # the last commit marker.  A crash can leave a newer atomic
            # checkpoint just before logger.commit_epoch, so promote that
            # checkpoint's metrics boundary before constructing the logger.
            resume_payload = _restore_checkpoint(cp, arm_identity)
            receipt = resume_payload.get("metrics_receipt")
            if not isinstance(receipt, Mapping) or receipt.get("row_count") is None:
                raise ValueError("PG resume checkpoint lacks metrics_receipt row_count")
            commit_state = {
                "schema": "self_audit.candidate_pg.epoch_commit.v1",
                "epoch": int(receipt.get("epoch", 0)),
                "update": int(receipt.get("update", resume_payload.get("update", 0))),
                "row_count": int(receipt["row_count"]),
                "csv_offset": int(receipt.get("csv_offset", arm_dir.joinpath("epochs.csv").stat().st_size if arm_dir.joinpath("epochs.csv").exists() else 0)),
                "jsonl_offset": int(receipt.get("jsonl_offset", arm_dir.joinpath("epochs.jsonl").stat().st_size if arm_dir.joinpath("epochs.jsonl").exists() else 0)),
            }
            commit_state["metrics_offset"] = commit_state["jsonl_offset"]
            commit_path = arm_dir / "epochs.commit.json"
            tmp_commit = commit_path.with_name(f".{commit_path.name}.{os.getpid()}.tmp")
            tmp_commit.write_text(json.dumps(commit_state, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            os.replace(tmp_commit, commit_path)
        epoch_logger = log_mod.EpochLogger(arm_dir, stage="arms", arm=arm_name)
        if resume_payload is not None:
            payload = resume_payload
            student.load_state_dict(payload["student"]); optimizer.load_state_dict(payload["optimizer"]); start = int(payload["update"]); best = float(payload.get("best_val", best)); best_state = payload.get("best_student"); best_epoch = payload.get("best_epoch"); _restore_rng(payload["rng"])
            epoch_logger.recover_from_checkpoint(payload)
            rows = [dict(item) for item in epoch_logger.rows]
            saved_sum = payload.get("epoch_loss_sum")
            if isinstance(saved_sum, (list, tuple)) and len(saved_sum) == 4:
                epoch_loss_sum = [float(value) for value in saved_sum]
            epoch_loss_steps = int(payload.get("epoch_loss_steps", 0))
            if sampler is not None and payload.get("sampler") is not None:
                sampler.load_state_dict(payload["sampler"])
            if threshold is not None and payload.get("threshold") is not None:
                threshold.load_state_dict(payload["threshold"])
            if view_gen is not None and payload.get("view_generator") is not None:
                view_gen.load_state_dict(payload["view_generator"])
            if payload.get("keep_generator_state") is not None:
                keep_generator.set_state(payload["keep_generator_state"])
            if payload.get("fp_generator_state") is not None:
                fp_generator.set_state(payload["fp_generator_state"])
        epoch_started = time.perf_counter()
        w2s_watch = None
        if arm_cfg.get("pg", {}).get("target_mode") == "w2s_arbiter":
            from .w2s_checks import Watchdogs
            _fg = [float((torch.as_tensor(f.target) > 0).float().mean()) for f in data["labeled"]]
            w2s_watch = Watchdogs(sum(_fg) / max(1, len(_fg)), int(round(updates * float(arm_cfg["pg"]["w2s"]["alpha_start_fraction"]))),
                                  emit=lambda message: print(message, flush=True))
        for update in range(start, updates):
            step_started = time.perf_counter()
            if sampler is not None:
                lk, uk, uk2 = sampler.next_step()
                lf = train_data.load(lk, labeled=True)
                uf = train_data.load(uk, labeled=False)
                uf2 = train_data.load(uk2, labeled=False)
            else:
                lf = data["labeled"][update % len(data["labeled"])]
                # Locked update composition: exactly one labeled frame and two
                # patient-balanced image-only U frames. Synthetic fixtures may
                # contain fewer than two entries; cycling preserves the contract.
                uf = data["unlabeled"][update % len(data["unlabeled"])]
                uf2 = data["unlabeled"][(update + 1) % len(data["unlabeled"])]
            li, target, geom = _frame_tensors(lf, target_device, labeled=True)
            ui, _, uge = _frame_tensors(uf, target_device, labeled=False)
            ui2, _, uge2 = _frame_tensors(uf2, target_device, labeled=False)
            if view_gen is not None:
                valid_l = torch.ones((li.shape[0], *li.shape[-3:]), dtype=torch.bool, device=li.device)
                weak_l, strong_l, target_l, valid_l, geom_l = view_gen.joint_views(li, valid_l, geom, target)
                valid_u = torch.ones((ui.shape[0], *ui.shape[-3:]), dtype=torch.bool, device=ui.device)
                weak_u, strong_u, valid_u, geom_u = view_gen.joint_views(ui, valid_u, uge)
                valid_u2 = torch.ones((ui2.shape[0], *ui2.shape[-3:]), dtype=torch.bool, device=ui2.device)
                weak_u2, strong_u2, valid_u2, geom_u2 = view_gen.joint_views(ui2, valid_u2, uge2)
            else:
                weak_l, strong_l, target_l, valid_l, geom_l = li, li, target, torch.ones_like(target, dtype=torch.bool), geom
                weak_u, strong_u, valid_u, geom_u = ui, ui, torch.ones((ui.shape[0], *ui.shape[-3:]), dtype=torch.bool), uge
                weak_u2, strong_u2, valid_u2, geom_u2 = ui2, ui2, torch.ones((ui2.shape[0], *ui2.shape[-3:]), dtype=torch.bool), uge2
            # The dedicated student step owns all teacher/fusion/PG/CBKD
            # forward math.  The engine supplies deterministic views and RNG
            # streams, then owns only optimizer/schedule/checkpointing.
            strong_u2_view = view_gen.strong(weak_u) if view_gen is not None else strong_u
            strong_u2b_view = view_gen.strong(weak_u2) if view_gen is not None else strong_u2
            labeled_views = {"weak": weak_l, "strong": strong_l, "valid": valid_l, "geometry": geom_l, "target": target_l}
            u_views = {
                "a": {"weak": weak_u, "strong": strong_u, "valid": valid_u, "geometry": geom_u},
                "b": {"weak": weak_u2, "strong": strong_u2, "strong2": strong_u2b_view, "valid": valid_u2, "geometry": geom_u2},
            }
            if view_gen is not None:
                u_views["a"]["strong2"] = strong_u2_view
                u_views["cutmix_generator"] = view_gen.cutmix_generator
            if switches["pg"] and view_mod is not None and hasattr(view_mod, "cutmix_box"):
                cutmix_cfg = cfg.get("augmentation", {}).get("cutmix", {})
                partner_valid = view_mod.center_crop_pad(valid_u2, valid_u.shape[-3:], pad_value=0) if tuple(valid_u2.shape[-3:]) != tuple(valid_u.shape[-3:]) else valid_u2
                n_boxes = 2 if int(cfg.get("augmentation", {}).get("n_strong_views", 2)) == 2 else 1
                u_views["boxes"] = [view_mod.cutmix_box(
                    valid_u, view_gen.cutmix_generator if view_gen is not None else None,
                    partner_valid=partner_valid,
                    area_fraction=cutmix_cfg.get("area_fraction", (0.25, 0.5)),
                    align_px=int(cutmix_cfg.get("align_px", 16)), full_depth=True,
                ) for _ in range(n_boxes)]
                align = int(cutmix_cfg.get("align_px", 16))
                if any(any(int(coord) % align for coord in box) or int(box[1]) <= int(box[0]) or int(box[3]) <= int(box[2])
                       for box in u_views["boxes"]):
                    raise RuntimeError("PG CutMix produced a nonempty box with unaligned coordinates")
            optimizer.zero_grad(set_to_none=True)
            lr_now = _schedule_lr(optimizer, update=update, total_updates=updates,
                                  lr=float(cfg.get("training", {}).get("optimizer", {}).get("lr", cfg.get("training", {}).get("lr", 1e-3))),
                                  warmup_updates=int(math.ceil(float(cfg.get("training", {}).get("warmup_fraction", 0.05)) * updates)))
            step_mod = _module("self_audit.candidate_pg.training_step")
            if step_mod is None or not hasattr(step_mod, "student_training_step"):
                raise RuntimeError("PG arms require candidate_pg.training_step.student_training_step")
            amp_enabled = target_device.type == "cuda"
            amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled)
            with amp_context:
                step_result = step_mod.student_training_step(
                    arm_cfg, student, teacher, s3_codebook_model if switches["cbkd"] else None,
                    lf, uf, uf2, update, threshold,
                    labeled_views=labeled_views, u_views=u_views,
                    keep_generator=keep_generator, fp_generator=fp_generator,
                )
            loss = step_result["loss"]
            loss_sup = step_result["loss_sup"]
            loss_pg_term = step_result["loss_pg"]
            loss_cbkd_term = step_result["loss_cbkd"]
            diagnostics = step_result.get("diagnostics", {})
            diag_tau_c = [float(v) for v in getattr(threshold, "thresholds", torch.empty(0)).detach().cpu()] if threshold is not None else []
            diag_agreement = float(diagnostics.get("agreement_a")) if isinstance(diagnostics.get("agreement_a"), Tensor) else float("nan")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite PG loss at {arm_name} update {update}")
            clip_norm = float(runtime_cfg.get("gradient_clip_norm", 1.0))
            grad_diag: dict[str, Any] = {}
            if sa_variant and switches["cbkd"]:
                grad_diag = _grouped_backward(student, step_result, clip_norm, per_term=(update + 1) % 50 == 0)
            else:
                loss.backward()
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), clip_norm)
            optimizer.step()
            update_no = update + 1
            epoch_loss_sum[0] += float(loss.detach()); epoch_loss_sum[1] += float(loss_sup.detach())
            epoch_loss_sum[2] += float(loss_pg_term.detach()); epoch_loss_sum[3] += float(loss_cbkd_term.detach()); epoch_loss_steps += 1
            if update_no % int(cfg.get("training", {}).get("updates_per_epoch", 80)) == 0 or update_no == updates:
                epoch = math.ceil(update_no / max(1, int(cfg.get("training", {}).get("updates_per_epoch", 80))))
                # Locked epoch monitoring covers train/val/test and the two
                # student arm paths.  TEST rows are descriptive only;
                # selection below reads val/student_only exclusively.
                eval_rows = []
                patient_rows: list[dict[str, Any]] = []
                monitor_test = "TEST50" in cfg.get("training", {}).get("eval_splits", ["TEST50"])
                split_frames = (("train", data["labeled"]), ("val", data["val"])) + ((("test", data["test"]),) if monitor_test else ())
                eval_paths = cfg.get("training", {}).get("eval_paths", ["student_only", "fusion"])
                path_models = tuple((name, student) for name in ("student_only", "fusion") if name in eval_paths)
                mean_total = epoch_loss_sum[0] / max(1, epoch_loss_steps)
                mean_sup = epoch_loss_sum[1] / max(1, epoch_loss_steps)
                mean_pg = epoch_loss_sum[2] / max(1, epoch_loss_steps)
                mean_cbkd = epoch_loss_sum[3] / max(1, epoch_loss_steps)
                for split_name, frames in split_frames:
                    for path_name, eval_model in path_models:
                        metrics_eval = _evaluate(eval_model, frames, target_device, path=path_name, teacher=teacher if path_name == "fusion" else None)
                        eval_rows.append({
                            "stage": "arms", "arm": arm_name, "epoch": epoch,
                            "update": update_no, "split": split_name, "path": path_name,
                            "loss_total": mean_total if split_name == "train" else None,
                            "loss_sup": mean_sup if split_name == "train" else None,
                            "loss_pg": mean_pg if split_name == "train" else None,
                            "loss_cbkd": mean_cbkd if split_name == "train" else None,
                            "fg_dice": metrics_eval.get("fg_dice"),
                            "rv_dice": metrics_eval.get("class_1_dice"),
                            "myo_dice": metrics_eval.get("class_2_dice"),
                            "lv_dice": metrics_eval.get("class_3_dice"),
                            "fg_hd95": metrics_eval.get("fg_hd95"),
                            "rv_hd95": metrics_eval.get("class_1_hd95"),
                            "myo_hd95": metrics_eval.get("class_2_hd95"),
                            "lv_hd95": metrics_eval.get("class_3_hd95"),
                            "fg_assd": metrics_eval.get("fg_assd"),
                            "rv_assd": metrics_eval.get("class_1_assd"),
                            "myo_assd": metrics_eval.get("class_2_assd"),
                            "lv_assd": metrics_eval.get("class_3_assd"),
                            "lr": lr_now,
                        })
                        for patient_id, patient_metric in (metrics_eval.get("per_patient", {}) or {}).items():
                            classes = patient_metric.get("per_class", {}) if isinstance(patient_metric, Mapping) else {}
                            patient_rows.append({
                                "patient_id": str(patient_id), "split": split_name, "path": path_name,
                                "fg_dice": patient_metric.get("fg_dice"), "fg_hd95": patient_metric.get("fg_hd95"), "fg_assd": patient_metric.get("fg_assd"),
                                "rv_dice": (classes.get("RV") or {}).get("dice"), "rv_hd95": (classes.get("RV") or {}).get("hd95_mm"), "rv_assd": (classes.get("RV") or {}).get("assd_mm"),
                                "myo_dice": (classes.get("MYO") or {}).get("dice"), "myo_hd95": (classes.get("MYO") or {}).get("hd95_mm"), "myo_assd": (classes.get("MYO") or {}).get("assd_mm"),
                                "lv_dice": (classes.get("LV") or {}).get("dice"), "lv_hd95": (classes.get("LV") or {}).get("hd95_mm"), "lv_assd": (classes.get("LV") or {}).get("assd_mm"),
                            })
                for eval_row in eval_rows:
                    epoch_logger.log_epoch(eval_row)
                _full_splits = _epoch_full_splits()
                if _full_splits:
                    _full = {"epoch": epoch, "update": update_no, "arm": arm_name}
                    for _name in _full_splits:
                        _frames = {"train": data["labeled"], "val": data["val"], "test": data.get("test")}[_name]
                        if _frames:
                            _full[_name] = _evaluate_full(student, _frames, target_device)
                    _append_epoch_full(arm_dir, _full)
                    print("[full] ep {} | ".format(epoch) + " | ".join(
                        "{} dice {:.4f} jc {:.4f} hd95 {:.2f}mm asd {:.2f}mm".format(n, *(_full[n]["native_mm"]["mean"][k] for k in ("dice", "jaccard", "hd95", "asd")))
                        for n in _full_splits if n in _full), flush=True)
                if patient_rows:
                    epoch_logger.log_patient_rows(epoch, patient_rows)
                selected = epoch_logger.select_best(split="val", path="student_only", metric="fg_dice")
                best_val = float(selected["fg_dice"])
                if best_val > best:
                    best = best_val
                    best_epoch = int(selected["epoch"])
                    best_state = copy.deepcopy(student.state_dict())
                rows = [dict(item) for item in epoch_logger.rows]
                checkpoint_state = {
                    "schema": _SCHEMA, "config_sha256": arm_identity["config_sha256"],
                    "arm": arm_name, "update": update_no,
                    "student": student.state_dict(), "optimizer": optimizer.state_dict(),
                    "rng": _rng_state(), "sampler": sampler.state_dict() if sampler is not None else None,
                    "view_generator": view_gen.state_dict() if view_gen is not None else None,
                    "keep_generator_state": keep_generator.get_state(),
                    "fp_generator_state": fp_generator.get_state(),
                    "threshold": threshold.state_dict() if threshold is not None else None,
                    "best_val": best, "best_epoch": best_epoch, "best_student": best_state,
                    "switches": switches,
                    "epoch_loss_sum": epoch_loss_sum, "epoch_loss_steps": epoch_loss_steps,
                    "metrics_receipt": {
                        "row_count": len(epoch_logger.rows), "epoch": epoch, "update": update_no,
                        "csv_offset": arm_dir.joinpath("epochs.csv").stat().st_size if arm_dir.joinpath("epochs.csv").exists() else 0,
                        "jsonl_offset": arm_dir.joinpath("epochs.jsonl").stat().st_size if arm_dir.joinpath("epochs.jsonl").exists() else 0,
                    },
                }
                # Persist the immutable checkpoint first; the logger commit is
                # the subsequent boundary marker for a successfully completed
                # epoch and can be reconstructed from metrics_receipt.
                _save_checkpoint(out_path, checkpoint_state, arm=arm_name, epoch=epoch, update=update_no)
                epoch_logger.commit_epoch(epoch=epoch, update=update_no)
                console_metrics = {row["split"]: row["fg_dice"] for row in eval_rows
                                   if row["path"] == "student_only"}
                console_metrics.setdefault("test", None)
                print(log_mod.epoch_console_line(
                    arm_name, epoch, math.ceil(updates / int(cfg["training"]["updates_per_epoch"])),
                    mean_total, console_metrics["train"], console_metrics["val"],
                    best, best_epoch, console_metrics["test"], time.perf_counter() - epoch_started,
                ), flush=True)
                epoch_started = time.perf_counter()
                _append_jsonl(arm_dir / "run.log", {"level": "DEBUG", "stage": "arms", "arm": arm_name, "epoch": epoch, "update": update_no, "loss_total": mean_total, "loss_sup": mean_sup, "loss_pg": mean_pg, "loss_cbkd": mean_cbkd, "lr": lr_now})
                epoch_loss_sum = [0.0, 0.0, 0.0, 0.0]; epoch_loss_steps = 0
            if update_no % 50 == 0 or update_no == updates:
                pW_conf = diagnostics.get("pW_confidence_a") if isinstance(diagnostics, Mapping) else None
                pW_target = diagnostics.get("pW_target_a") if isinstance(diagnostics, Mapping) else None
                teacher_target = diagnostics.get("teacher_target_a") if isinstance(diagnostics, Mapping) else None
                coverage = None
                disagreement = None
                if isinstance(pW_conf, Tensor) and pW_conf.numel() and diag_tau_c:
                    threshold_tensor = torch.as_tensor(diag_tau_c, device=pW_conf.device, dtype=pW_conf.dtype)
                    if isinstance(pW_target, Tensor):
                        coverage = float((pW_conf >= threshold_tensor[pW_target.long()]).float().mean().detach().cpu())
                    if isinstance(teacher_target, Tensor) and isinstance(pW_target, Tensor):
                        disagreement = float((teacher_target != pW_target).float().mean().detach().cpu())
                memory_mb = None
                if target_device.type == "cuda":
                    memory_mb = float(torch.cuda.max_memory_allocated(target_device) / (1024.0 ** 2))
                code_perplexity = None; code_dead = None; code_coverage = None
                usage_ema = getattr(s3_codebook_model, "usage_ema", None) if switches["cbkd"] else None
                if isinstance(usage_ema, Tensor) and usage_ema.numel():
                    usage = usage_ema.detach().float().clamp_min(1e-12)
                    usage = usage / usage.sum().clamp_min(1e-12)
                    code_perplexity = float(torch.exp(-(usage * usage.log()).sum()).cpu())
                    code_dead = int((usage_ema <= float(getattr(s3_codebook_model, "dead_code_threshold", 0.0))).sum().cpu())
                    code_coverage = float((usage_ema > 0).float().mean().cpu())
                def _diag_json(value: Any) -> Any:
                    if isinstance(value, Tensor):
                        if value.numel() == 1:
                            return float(value.detach().cpu())
                        return value.detach().cpu().tolist()
                    if isinstance(value, Mapping):
                        return {str(key): _diag_json(item) for key, item in value.items()}
                    if isinstance(value, (tuple, list)):
                        return [_diag_json(item) for item in value]
                    if isinstance(value, np.generic):
                        return value.item()
                    return value

                exact_boxes = diagnostics.get("cutmix_boxes", u_views.get("boxes"))
                exact_agreement = diagnostics.get("agreement_a")
                if w2s_watch is not None:
                    w2s_watch.update(update_no, {k: _diag_json(diagnostics[k]) for k in ("alpha", "override_share_fg_disagree", "fg_coverage") if k in diagnostics})
                _append_jsonl(arm_dir / "diag.jsonl", {
                    "update": update_no, "arm": arm_name,
                    "loss_total": float(loss.detach()), "loss_sup": float(loss_sup.detach()),
                    "loss_pg": float(loss_pg_term.detach()), "loss_cbkd": float(loss_cbkd_term.detach()),
                    "tau_c": diag_tau_c, "agreement": diag_agreement if math.isfinite(diag_agreement) else None,
                    "confidence_coverage": coverage,
                    "gate_means": _diag_json(diagnostics.get("gate_means")),
                    "teacher_fusion_disagreement": disagreement,
                    # Persist the detached step diagnostics verbatim (after
                    # scalar/list conversion), not a reconstructed proxy.
                    "agreement_a": _diag_json(exact_agreement),
                    "coverage_per_class_a": _diag_json(diagnostics.get("coverage_per_class_a")),
                    "coverage_per_class_b": _diag_json(diagnostics.get("coverage_per_class_b")),
                    "boundary_band_coverage_a": _diag_json(diagnostics.get("boundary_band_coverage_a")),
                    "boundary_band_coverage_b": _diag_json(diagnostics.get("boundary_band_coverage_b")),
                    "teacher_fusion_disagreement_a": _diag_json(diagnostics.get("teacher_fusion_disagreement_a")),
                    "teacher_fusion_disagreement_b": _diag_json(diagnostics.get("teacher_fusion_disagreement_b")),
                    "code_perplexity": _diag_json(diagnostics.get("code_perplexity", code_perplexity)),
                    "code_dead": _diag_json(diagnostics.get("code_dead", code_dead)),
                    "code_valid_count": _diag_json(diagnostics.get("code_valid_count")),
                    "code_coverage": code_coverage,
                    "cutmix_boxes": _diag_json(exact_boxes),
                    "cutmix_nonempty": bool(exact_boxes) and any(int(box[1]) > int(box[0]) and int(box[3]) > int(box[2]) for box in exact_boxes),
                    "seconds": time.perf_counter() - step_started, "memory_mb": memory_mb,
                    **({"grad": grad_diag} if grad_diag else {}),
                    **{k: _diag_json(diagnostics[k]) for k in ("alpha", "tau", "tau_v", "target_mode", "rule_fraction", "rule_fraction_per_class",
                                                               "override_share_fg_disagree", "fg_coverage", "weight_mean", "loss_u",
                                                               "gate_fraction", "mean_m_T_gated", "lambda_u", "lambda_cb", "loss_sup",
                                                               "loss_pm", "loss_atom", "loss_commit", "code_perplexity", "code_dead") if k in diagnostics},
                    **{k: _diag_json(diagnostics[k]) for k in ("loss_pm", "loss_atom", "loss_commit", "m_T_mean", "student_code_perplexity",
                                                               "student_code_dead", "loss_pm_l", "loss_pm_u", "lambda_u", "lambda_cb") if k in diagnostics},
                })
            if update_no % 50 == 0 or update_no == updates or update_no % int(cfg.get("training", {}).get("updates_per_epoch", 80)) == 0:
                _save_checkpoint(out_path, {"schema": _SCHEMA, "config_sha256": arm_identity["config_sha256"], "arm": arm_name, "update": update_no, "student": student.state_dict(), "optimizer": optimizer.state_dict(), "rng": _rng_state(), "sampler": sampler.state_dict() if sampler is not None else None, "view_generator": view_gen.state_dict() if view_gen is not None else None, "keep_generator_state": keep_generator.get_state(), "fp_generator_state": fp_generator.get_state(), "threshold": threshold.state_dict() if threshold is not None else None, "best_val": best, "best_epoch": best_epoch, "best_student": best_state, "switches": switches, "epoch_loss_sum": epoch_loss_sum, "epoch_loss_steps": epoch_loss_steps, "metrics_receipt": {"row_count": len(epoch_logger.rows), "epoch": math.ceil(update_no / 80), "update": update_no}}, arm=arm_name, epoch=math.ceil(update_no / 80), update=update_no)
        if best_state is not None:
            student.load_state_dict(best_state, strict=True)
        rows = [dict(item) for item in epoch_logger.rows]
        if best_epoch is None:
            selected_row = epoch_logger.best_val
            if selected_row is not None:
                best_epoch = int(selected_row["epoch"])
        selected_models[arm_name] = copy.deepcopy(student).cpu()
        selected_epochs[arm_name] = int(best_epoch or (rows[-1].get("epoch", 0) if rows else 0))
        epoch_logs[arm_name] = [dict(item) for item in epoch_logger.rows]
        # Finalization owns the one authoritative export/cold-load transaction;
        # keep the arm result export slot populated from that receipt below.
        result["arms"][arm_name] = {"switches": switches, "updates": updates, "best_val": best, "metrics": rows, "export": None}
        print(f"[{arm_name}] end", flush=True)
    if set(selected_arms) == set(_ARMS):
        finalizer = _module("self_audit.candidate_pg.finalization")
        if finalizer is None or not hasattr(finalizer, "finalize_campaign"):
            raise RuntimeError("PG complete campaign requires candidate_pg.finalization.finalize_campaign")
        # Finalization is part of the campaign transaction, including the
        # synthetic software-only path.  Any export/cold-load/diagnostic or
        # final TEST failure must fail the run rather than being relabeled as a
        # successful campaign with an advisory receipt.
        result["finalization"] = finalizer.finalize_campaign(
            cfg, out_path, selected_models, selected_epochs, epoch_logs, data,
            teacher=teacher, device=device,
        )
        for arm_name, export in result["finalization"].get("exports", {}).items():
            if arm_name in result["arms"]:
                result["arms"][arm_name]["export"] = export
    result["status"] = "COMPLETED"; _json(out_path / "RESULT.json", result); return result


def train(config: Mapping[str, Any], output_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    return run_training(config, output_dir, **kwargs)


__all__ = ["run_training", "train", "config_sha256"]
