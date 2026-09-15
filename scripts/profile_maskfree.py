#!/usr/bin/env python
"""Bounded profiling harness for the canonical mask-free trainer.

This module deliberately contains no second training implementation.  It loads
``MaskfreeConfig`` and executes :class:`self_audit_maskfree.trainer.MaskfreeTrainer`
unchanged, adding process-local accounting around the existing
``TimingAccumulator``.  The harness is useful on a CPU-only checkout with
``--synthetic`` and on an inventoried image root with ``--config``; it never
opens a mask or a reference evaluator input.

The default protocol is five discarded warm-up batches followed by twenty
measured batches.  A bounded run remains a partial 150-epoch run: ``total_epochs``
is kept at 150, while ``max_steps`` stops after the requested batch count.  All
timing reports label this software evidence as neither CUDA/5070-Ti nor clinical
evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import cProfile
import dataclasses
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


PROFILE_SCHEMA_VERSION = "maskfree150.profile.v2"
SOURCE_SNAPSHOT_SCHEMA_VERSION = "maskfree150.source-snapshot.v2"
SYNTHETIC_FIXTURE_SCHEMA_VERSION = "maskfree150.synthetic-fixture.v2"
SYNTHETIC_DEPTH = 1024
SYNTHETIC_TRAIN_UNITS_MIN = 256
PARTIAL_EPOCH_MAX_BATCHES = SYNTHETIC_DEPTH // 8
TIMING_MODES = ("ordinary", "instrumented")
SCIENTIFIC_REQUIREMENTS = {
    "total_epochs": 150,
    "seed": 42,
    "batch_size": 8,
    "accumulation_steps": 1,
    "image_size": 128,
    "amp": False,
}

PROFILE_BUDGET_REQUIREMENTS = {
    "bank_size": 4,
    "rounds": 2,
    "max_fit_iterations": 5,
    "max_scored_regions": 32,
    "max_region_score_calls": 128,
}


def _json_default(value: Any) -> Any:
    """JSON conversion kept local so reports remain portable."""
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a deterministic linear-interpolated percentile or ``None``."""
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summary(values: list[float]) -> dict[str, float | int | None]:
    """Summarise a set of timing observations without fabricating empty values."""
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values) if values else None,
        "p50_seconds": percentile(values, 0.50),
        "p95_seconds": percentile(values, 0.95),
        "min_seconds": min(values) if values else None,
        "max_seconds": max(values) if values else None,
    }


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_details(details: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(details, default=_json_default))


def _rng_digest(runtime_module: ModuleType) -> dict[str, Any]:
    state = runtime_module.capture_rng_state()
    cuda = state.get("cuda")
    return {
        "digest": _sha256_payload(state),
        "cuda_state_count": len(cuda) if cuda is not None else 0,
        "components": sorted(state),
    }


def _harness_identity() -> dict[str, Any]:
    """Capture the profiler bytes and runtime knobs that affect paired runs.

    The canonical source snapshot deliberately excludes this script because it
    is orchestration code rather than production package code.  Recording its
    digest and the process/thread determinism settings in every report makes a
    later baseline/optimized comparison reject a silently changed harness.
    """
    import torch

    thread_keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCH_INTEROP_THREADS",
    )
    determinism_keys = (
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "TORCH_DETERMINISTIC",
        "CUDA_LAUNCH_BLOCKING",
    )
    try:
        deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    except Exception:  # pragma: no cover - defensive runtime boundary  # noqa: BLE001
        deterministic_algorithms = None
    return {
        "schema_version": "maskfree150.harness.v1",
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "thread_env": {key: os.environ.get(key) for key in thread_keys},
        "determinism_env": {key: os.environ.get(key) for key in determinism_keys},
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_deterministic_algorithms_preinit": deterministic_algorithms,
    }


def _tensor_digest(value: Any) -> str | None:
    """Hash tensor bytes for compact per-unit decision evidence."""
    try:
        import torch

        if not isinstance(value, torch.Tensor):
            return None
        tensor = value.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
        return digest.hexdigest()
    except Exception:  # pragma: no cover - diagnostics must not mask trainer errors  # noqa: BLE001
        return None


def _tensor_numeric_summary(value: Any) -> dict[str, Any] | None:
    """Summarise floating decision tensors for post-hash mismatch diagnosis."""
    try:
        import torch

        if not isinstance(value, torch.Tensor):
            return None
        tensor = value.detach().cpu().to(torch.float64).reshape(-1)
        if tensor.numel() == 0:
            return {"count": 0, "finite": True, "min": None, "max": None, "mean": None}
        finite = bool(torch.isfinite(tensor).all().item())
        finite_values = tensor[torch.isfinite(tensor)]
        return {
            "count": int(tensor.numel()),
            "finite": finite,
            "min": float(finite_values.min().item()) if finite_values.numel() else None,
            "max": float(finite_values.max().item()) if finite_values.numel() else None,
            "mean": float(finite_values.mean().item()) if finite_values.numel() else None,
        }
    except Exception:  # pragma: no cover - diagnostics must not mask trainer errors  # noqa: BLE001
        return None


def _score_record(score: Any) -> dict[str, Any] | None:
    """Serialize fit/select score identity plus numeric fields without tensors."""
    if score is None:
        return None
    available = bool(getattr(score, "available", True))
    count = int(getattr(score, "count", 0))
    raw_values = {
        "complexity": getattr(score, "complexity", None),
        "prior": getattr(score, "prior", None),
        "nll_sum": getattr(score, "nll_sum", None),
        "normalized_nll": getattr(score, "normalized_nll", None),
        "total": getattr(score, "total", None),
    }
    required = (
        ("complexity", "prior", "nll_sum", "normalized_nll", "total")
        if available
        else ("complexity", "prior")
    )
    finite_numeric = all(_numeric(raw_values[name]) is not None for name in required)
    finite_numeric = finite_numeric and (count > 0 if available else count == 0)
    return {
        "available": available,
        "count": count,
        "complexity": _numeric(raw_values["complexity"]),
        "prior": _numeric(raw_values["prior"]),
        "reason": getattr(score, "reason", None),
        "role": str(getattr(score, "role", "")),
        "nll_sum": _numeric(raw_values["nll_sum"]),
        "normalized_nll": _numeric(raw_values["normalized_nll"]),
        "total": _numeric(raw_values["total"]),
        "finite_numeric": finite_numeric,
    }


def _audit_decision_record(result: Any) -> dict[str, Any]:
    """Keep ordered, compact audit decisions for matched-workload checking."""
    trace = dict(getattr(result, "trace", {}) or {})
    bank = list(getattr(result, "bank", []) or [])
    candidates = []
    for candidate in bank:
        candidates.append(
            {
                "candidate_id": str(getattr(candidate, "candidate_id", "")),
                "source": str(getattr(candidate, "source", "")),
                "labels_sha256": _tensor_digest(getattr(candidate, "labels", None)),
                "probabilities_sha256": _tensor_digest(getattr(candidate, "probabilities", None)),
                "validity_sha256": _tensor_digest(getattr(candidate, "validity", None)),
                "semantic_unresolved": bool(getattr(candidate, "semantic_unresolved", False)),
            }
        )
    rejected_discrete = []
    rejected_gains = []
    for entry in list(trace.get("rejected_edits", []) or []):
        if not isinstance(entry, dict):
            rejected_discrete.append({"value": str(entry)})
            continue
        rejected_discrete.append(
            {
                key: value
                for key, value in entry.items()
                if key not in {"gain_nats_per_pixel", "score", "total"}
            }
        )
        if "gain_nats_per_pixel" in entry:
            rejected_gains.append(float(entry["gain_nats_per_pixel"]))
    evaluation_order = []
    for entry in list(trace.get("evaluation_order", []) or []):
        if isinstance(entry, dict):
            evaluation_order.append(
                {
                    key: entry.get(key)
                    for key in (
                        "candidate_index", "candidate_id", "round", "cached",
                        "fitting_steps", "capacity",
                    )
                }
            )
    search_notes = dict(trace.get("search_notes", {}) or {})
    budget = dict(trace.get("budget", {}) or {})
    return {
        "unit_id": str(trace.get("unit_id", "")),
        "initial_candidate_id": str(getattr(getattr(result, "initial", None), "candidate_id", "")),
        "selected_candidate_id": str(getattr(getattr(result, "selected", None), "candidate_id", "")),
        "selected_index": int(trace.get("selected_index", -1)),
        "accepted": bool(trace.get("accepted", False)),
        "search_inconclusive": bool(trace.get("search_inconclusive", False)),
        "semantic_unresolved": bool(trace.get("semantic_unresolved", False)),
        "acceptance_reason": trace.get("acceptance_reason"),
        "accepted_gain": _numeric(trace.get("improvement_nats_per_pixel")),
        "evaluation_order": _safe_details({"items": evaluation_order}),
        "per_unit_budget": {
            "fits_performed": int(trace.get("fits_performed", 0)),
            "primary_score_calls": int(trace.get("primary_score_calls", 0)),
            "regional_score_calls": int(trace.get("regional_score_calls", 0)),
            "regional_score_counts": [int(value) for value in trace.get("regional_score_counts", [])],
            "regional_refits": int(trace.get("regional_refits", 0)),
            "rounds_requested": int(trace.get("rounds_requested", 0)),
            "round_slots": _safe_details({"items": list(trace.get("round_slots", []) or [])}),
            "fitting_steps_total": sum(
                int(getattr(item, "fitting_steps", 0))
                for item in list(getattr(result, "fitted", []) or [])
            ),
            "max_fits": int(budget.get("max_fits", 0)),
            "fitting_iterations_per_candidate": int(
                budget.get("fitting_iterations_per_candidate", 0)
            ),
            "regions_total": int(search_notes.get("regions_total", 0)),
            "regions_scored": int(search_notes.get("regions_scored", 0)),
        },
        "rejected_decisions": _safe_details({"items": rejected_discrete}),
        "rejected_gains": rejected_gains,
        "fit_scores": [
            _score_record(getattr(item, "fit_score", None))
            for item in list(getattr(result, "fitted", []) or [])
        ],
        "select_scores": [
            _score_record(item)
            for item in list(getattr(result, "scores", []) or [])
        ],
        "audited_validity_sha256": _tensor_digest(getattr(result, "validity", None)),
        "regional_margin_sha256": _tensor_digest(getattr(result, "regional_margin", None)),
        "audited_validity_stats": _tensor_numeric_summary(getattr(result, "validity", None)),
        "regional_margin_stats": _tensor_numeric_summary(getattr(result, "regional_margin", None)),
        "bank": candidates,
    }


def _source_files(repo_root: Path) -> list[Path]:
    """Return source/config files in a stable order for an immutable snapshot."""
    package = repo_root / "src" / "self_audit_maskfree"
    files = sorted(package.rglob("*.py"))
    files.extend(sorted(repo_root.glob("configs/maskfree_*_150.y*ml")))
    return [path for path in files if "__pycache__" not in path.parts and path.is_file()]


def _source_manifest(repo_root: Path) -> dict[str, str]:
    return {
        source.relative_to(repo_root).as_posix(): _sha256_file(source)
        for source in _source_files(repo_root)
    }


def _snapshot_file_paths(destination: Path) -> set[str]:
    files_root = destination / "files"
    if not files_root.is_dir():
        return set()
    return {
        path.relative_to(files_root).as_posix()
        for path in files_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def verify_snapshot(destination: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Verify an immutable snapshot manifest and every copied target byte.

    Reusing a snapshot is intentionally strict: a missing, changed, or added
    file is a hard mismatch instead of an opportunity to refresh in place.
    """
    destination = destination.resolve()
    expected = {str(key): str(value) for key, value in snapshot.get("files", {}).items()}
    copied = _snapshot_file_paths(destination)
    missing: list[str] = []
    changed: list[str] = []
    for relative, digest in expected.items():
        target = destination / "files" / relative
        if not target.is_file():
            missing.append(relative)
            continue
        observed = _sha256_file(target)
        if observed != digest:
            changed.append(relative)
    added = sorted(copied - set(expected))
    observed_combined = _sha256_payload(
        {relative: _sha256_file(destination / "files" / relative)
         for relative in sorted(copied)}
    ) if copied else _sha256_payload({})
    manifest_combined = snapshot.get("combined_sha256")
    hash_mismatch = not isinstance(manifest_combined, str) or observed_combined != manifest_combined
    details = {
        "manifest_path": str(destination / "snapshot_manifest.json"),
        "expected_files": len(expected),
        "copied_files": len(copied),
        "missing_files": sorted(missing),
        "changed_files": sorted(changed),
        "added_files": added,
        "hash_mismatch": hash_mismatch,
        "ok": not (missing or changed or added or hash_mismatch),
        "manifest_combined_sha256": manifest_combined,
        "observed_combined_sha256": observed_combined,
    }
    if not details["ok"]:
        raise ValueError(
            "immutable source snapshot mismatch: " + json.dumps(details, sort_keys=True)
        )
    return details


def snapshot_source(repo_root: Path, destination: Path) -> dict[str, Any]:
    """Copy and hash the exact package/config bytes before a measured run.

    Existing snapshots are immutable: a second call verifies the old manifest
    instead of silently overwriting it.  Only package/config sources are copied
    so no data or reference-mask tree can enter the snapshot.
    """
    destination = destination.resolve()
    manifest_path = destination / "snapshot_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported source snapshot schema: {payload.get('schema_version')!r}"
            )
        verify_snapshot(destination, payload)
        return payload

    files_root = destination / "files"
    entries: dict[str, str] = {}
    for source in _source_files(repo_root):
        relative = source.relative_to(repo_root).as_posix()
        target = files_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries[relative] = _sha256_file(source)
    payload = {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(repo_root),
        "files": entries,
        "combined_sha256": _sha256_payload(entries),
        "note": "immutable bytes captured before profiling; no masks or reference inputs",
    }
    _write_json(manifest_path, payload)
    verify_snapshot(destination, payload)
    return payload


def source_snapshot_unchanged(repo_root: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    current = _source_manifest(repo_root)
    expected = {str(key): str(value) for key, value in snapshot.get("files", {}).items()}
    missing = sorted(set(expected) - set(current))
    added = sorted(set(current) - set(expected))
    changed = sorted(
        relative for relative in set(expected) & set(current)
        if current[relative] != expected[relative]
    )
    differences = sorted(set(missing) | set(added) | set(changed))
    return {
        "unchanged": not differences,
        "differences": differences,
        "missing_files": missing,
        "added_files": added,
        "changed_files": changed,
        "current_combined_sha256": _sha256_payload(current),
    }


def _resolve_source_root(source_root: Path) -> tuple[Path, Path]:
    """Resolve a source-root argument to ``sys.path`` and package directories."""
    root = source_root.expanduser().resolve()
    if (root / "src" / "self_audit_maskfree").is_dir():
        import_root = root / "src"
    elif (root / "self_audit_maskfree").is_dir():
        import_root = root
    elif root.name == "self_audit_maskfree" and root.is_dir():
        import_root = root.parent
    else:
        raise ValueError(
            f"--source-root must contain self_audit_maskfree package: {source_root}"
        )
    package = import_root / "self_audit_maskfree"
    if not (package / "__init__.py").is_file():
        raise ValueError(f"source-root package is missing __init__.py: {package}")
    return import_root, package


def _activate_source_root(source_root: Path) -> dict[str, Any]:
    """Select an import root before any ``self_audit_maskfree`` import occurs."""
    import_root, package = _resolve_source_root(source_root)
    for name in list(sys.modules):
        if name == "self_audit_maskfree" or name.startswith("self_audit_maskfree."):
            del sys.modules[name]
    importlib.invalidate_caches()
    # Remove any prior package roots, including the live checkout, so the
    # selected snapshot is the only candidate even under an inherited PYTHONPATH.
    sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != import_root]
    sys.path.insert(0, str(import_root))
    return {"import_root": str(import_root), "package_root": str(package)}


def _observed_import_identity(source_root: Path) -> dict[str, Any]:
    """Hash the modules actually imported, rejecting any live-source leakage."""
    import_root, package = _resolve_source_root(source_root)
    modules: dict[str, str] = {}
    file_hashes: dict[str, str] = {}
    outside: list[str] = []
    for name, module in sorted(sys.modules.items()):
        if not (name == "self_audit_maskfree" or name.startswith("self_audit_maskfree.")):
            continue
        path_value = getattr(module, "__file__", None)
        if not path_value:
            continue
        path = Path(path_value).resolve()
        modules[name] = str(path)
        try:
            relative = path.relative_to(package).as_posix()
        except ValueError:
            outside.append(str(path))
            continue
        if path.suffix == ".py" and path.is_file():
            file_hashes[relative] = _sha256_file(path)
    if outside:
        raise ValueError(f"imported self_audit_maskfree modules outside source-root: {outside}")
    package_files = {
        path.relative_to(package).as_posix(): _sha256_file(path)
        for path in package.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    unobserved = sorted(set(package_files) - set(file_hashes))
    # Modules imported lazily later are still tied to this package root; the
    # complete package hash records all observed source bytes.
    return {
        "import_root": str(import_root),
        "package_root": str(package),
        "module_paths": modules,
        "module_file_hashes": file_hashes,
        "package_files": package_files,
        "package_combined_sha256": _sha256_payload(package_files),
        "unobserved_package_files": unobserved,
        "outside_package_paths": outside,
    }


def _model_state_digest(trainer: Any) -> str:
    """Hash exact initial model tensors before the first profiled batch."""
    digest = hashlib.sha256()
    for model_name, model in sorted(getattr(trainer, "models", {}).items()):
        digest.update(model_name.encode("utf-8"))
        for name, tensor in sorted(model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _sample_order_digest(batches: list[dict[str, Any]]) -> str:
    return _sha256_payload([
        {
            "batch_sequence": int(batch.get("batch_sequence", -1)),
            "indices": list(batch.get("indices", [])),
            "unit_ids": list(batch.get("unit_ids", [])),
        }
        for batch in batches
    ])


def make_synthetic_fixture(
    root: Path,
    *,
    seed: int = 42,
    image_size: int = 128,
    depth: int = SYNTHETIC_DEPTH,
    file_format: str = "npy",
) -> dict[str, Any]:
    """Create one deterministic, file-backed image-only profiling cohort.

    The structured phantom has no segmentation labels, mask sidecar or
    reference intensity.  The default source has 1024 acquired slices, yielding
    at least 256 train units independent of warm-up/measured arguments while
    preserving the canonical ``ImageOnlyDataset`` and discovery path.
    """
    if image_size != 128:
        raise ValueError("the scientific profiling fixture is fixed at image_size=128")
    if depth < 256:
        raise ValueError("the profiling fixture must contain at least 256 acquired slices")
    if file_format not in ("npy", "nifti"):
        raise ValueError(f"unsupported synthetic format: {file_format!r}")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    suffix = ".nii.gz" if file_format == "nifti" else ".npy"
    source = root / f"synthetic_patient_{int(seed)}{suffix}"
    manifest_path = root / "fixture_manifest.json"

    if source.exists() or manifest_path.exists():
        if not source.is_file() or not manifest_path.is_file():
            raise ValueError(
                f"synthetic fixture reuse requires both source and fixture_manifest.json: {root}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = payload.get("fixture", payload)
        mismatches = {
            key: {"expected": value, "actual": expected.get(key)}
            for key, value in {
                "schema_version": SYNTHETIC_FIXTURE_SCHEMA_VERSION,
                "seed": int(seed),
                "image_size": int(image_size),
                "depth": int(depth),
                "format": file_format,
            }.items()
            if expected.get(key) != value
        }
        if mismatches:
            raise ValueError(f"synthetic fixture recipe mismatch: {mismatches}")
        observed_hash = _sha256_file(source)
        if observed_hash != expected.get("source_sha256"):
            raise ValueError("synthetic fixture source bytes changed since creation")
        return dict(expected)

    rng = np.random.default_rng(int(seed))
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, image_size, dtype=np.float32),
        np.linspace(-1.0, 1.0, image_size, dtype=np.float32),
        indexing="ij",
    )
    radius = np.sqrt(xx * xx + yy * yy)
    frames = []
    for index in range(depth):
        phase = np.float32(index / max(depth - 1, 1))
        anatomy = (
            0.45 * np.exp(-((xx - 0.23 * np.cos(phase * math.pi)) ** 2 +
                             (yy - 0.15 * np.sin(phase * math.pi)) ** 2) / 0.06)
            + 0.30 * np.exp(-((xx + 0.30) ** 2 + (yy + 0.18) ** 2) / 0.035)
            + 0.10 * np.cos(3.0 * xx + phase) * np.sin(2.0 * yy - phase)
            - 0.08 * radius
        )
        noise = rng.normal(loc=0.0, scale=0.002, size=(image_size, image_size)).astype(np.float32)
        frames.append((anatomy + noise + np.float32(phase * 0.01)).astype(np.float32))
    array = np.stack(frames, axis=2)
    if file_format == "nifti":
        try:
            import nibabel as nib
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("--synthetic-format nifti requires nibabel") from exc
        nib.save(nib.Nifti1Image(array, np.eye(4, dtype=np.float64)), str(source))
    else:
        np.save(source, array)
    fixture = {
        "schema_version": SYNTHETIC_FIXTURE_SCHEMA_VERSION,
        "root": str(root),
        "source": str(source),
        "format": file_format,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "source_sha256": _sha256_file(source),
        "mask_inputs_used": False,
        "reference_inputs_used": False,
        "seed": int(seed),
        "image_size": int(image_size),
        "depth": int(depth),
        "train_units_minimum": SYNTHETIC_TRAIN_UNITS_MIN,
    }
    _write_json(manifest_path, {"fixture": fixture})
    return fixture


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found; no install attempted"}
    query = (
        "index,uuid,name,memory.used,memory.total,utilization.gpu,"
        "utilization.memory,power.draw"
    )
    try:
        completed = subprocess.run(
            [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"nvidia-smi failed: {type(exc).__name__}: {exc}"}
    return {
        "available": completed.returncode == 0,
        "command": [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        "returncode": completed.returncode,
        "rows": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
        "stderr": completed.stderr.strip() or None,
    }


class TimingProbe:
    """Attach per-call inclusive/exclusive timing to one existing accumulator."""

    def __init__(self, trainer: Any, *, mode: str = "instrumented",
                 runtime_module: ModuleType | None = None,
                 warmup_batches: int = 0,
                 measured_batches: int | None = None) -> None:
        self.trainer = trainer
        if mode not in TIMING_MODES:
            raise ValueError(f"unknown timing mode: {mode!r}")
        self.mode = mode
        self.runtime_module = runtime_module
        self.warmup_batches = int(warmup_batches)
        self.measured_batches = None if measured_batches is None else int(measured_batches)
        self.original_stage = trainer.timing.stage
        self.stack: list[dict[str, Any]] = []
        self.all_events: list[dict[str, Any]] = []
        self.batches: list[dict[str, Any]] = []
        self.active_batch: dict[str, Any] | None = None
        self.checkpoint = {"calls": 0, "wall_seconds": 0.0, "process_cpu_seconds": 0.0}
        self.progress = {
            "update_calls": 0,
            "update_seconds": 0.0,
            "dashboard_calls": 0,
            "dashboard_seconds": 0.0,
            "event_calls": 0,
            "event_seconds": 0.0,
        }
        self.cuda_sync = {"calls": 0, "wall_seconds": 0.0, "process_cpu_seconds": 0.0}
        self.connected_components = {
            "calls": 0,
            "wall_seconds": 0.0,
            "process_cpu_seconds": 0.0,
        }
        self.logical_work = {
            "counts": defaultdict(int),
            "wall_seconds": defaultdict(float),
            "process_cpu_seconds": defaultdict(float),
            "items": defaultdict(int),
        }
        self.cache_before_run: dict[str, Any] | None = None
        self.cache_after_warmup: dict[str, Any] | None = None
        self.cache_after_measured: dict[str, Any] | None = None
        self.initial_state: dict[str, Any] | None = None
        self._batch_sequence = 0

    def record_initial_state(self, *, indices: list[int], unit_ids: list[str]) -> None:
        if self.initial_state is not None:
            return
        rng = _rng_digest(self.runtime_module) if self.runtime_module is not None else None
        try:
            import torch

            deterministic_effective = bool(torch.are_deterministic_algorithms_enabled())
        except Exception:  # pragma: no cover - defensive runtime boundary  # noqa: BLE001
            deterministic_effective = None
        config = getattr(self.trainer, "config", None)
        resume = getattr(config, "resume", None)
        resume_path = Path(resume).expanduser().resolve() if resume else None
        self.initial_state = {
            "model_state_sha256": _model_state_digest(self.trainer),
            "rng": rng,
            "torch_deterministic_algorithms_effective": deterministic_effective,
            "first_batch_indices": list(indices),
            "first_batch_unit_ids": list(unit_ids),
            "checkpoint_resume": {
                "requested": str(resume_path) if resume_path else None,
                "exists": bool(resume_path and resume_path.is_file()),
                "start_epoch": int(getattr(self.trainer, "start_epoch", 0)),
                "start_batch": int(getattr(self.trainer, "start_batch", 0)),
                "last_completed_epoch": int(getattr(self.trainer, "last_completed_epoch", -1)),
                "resume_position": list(getattr(self.trainer, "_resume_position", (0, 0, []))[:2]),
            },
        }

    def record_logical_call(self, name: str, *, wall_seconds: float,
                            process_cpu_seconds: float, item_count: int | None = None) -> None:
        self.logical_work["counts"][name] += 1
        self.logical_work["wall_seconds"][name] += float(wall_seconds)
        self.logical_work["process_cpu_seconds"][name] += float(process_cpu_seconds)
        if item_count is not None:
            self.logical_work["items"][name] += int(item_count)
        batch = self.active_batch
        if batch is not None:
            rows = batch.setdefault("logical_work", {})
            row = rows.setdefault(
                name,
                {"calls": 0, "items": 0, "wall_seconds": 0.0, "process_cpu_seconds": 0.0},
            )
            row["calls"] += 1
            row["wall_seconds"] += float(wall_seconds)
            row["process_cpu_seconds"] += float(process_cpu_seconds)
            if item_count is not None:
                row["items"] += int(item_count)

    def _cache_stats(self) -> dict[str, Any]:
        dataset = getattr(self.trainer, "dataset", None)
        getter = getattr(dataset, "cache_stats", None)
        if not callable(getter):
            return {"available": False, "reason": "dataset.cache_stats unavailable"}
        try:
            stats = getter()
        except Exception as exc:  # pragma: no cover - defensive diagnostics  # noqa: BLE001
            return {"available": False, "reason": f"cache_stats failed: {type(exc).__name__}: {exc}"}
        if stats is None:
            return {"available": False, "reason": "dataset.cache_stats returned null"}
        normalized = {str(key): int(value) for key, value in dict(stats).items()}
        normalized["resident_bytes"] = int(normalized.get("bytes", 0))
        return {"available": True, "stats": normalized}

    @contextlib.contextmanager
    def stage(self, name: str, **details: Any) -> Iterator[None]:
        frame: dict[str, Any] = {
            "name": str(name),
            "details": _safe_details(details),
            "batch": self.active_batch,
            "children": [],
            "start": time.perf_counter(),
            "process_start": time.process_time(),
        }
        if self.stack:
            self.stack[-1]["children"].append(frame)
        self.stack.append(frame)
        try:
            with self.original_stage(name, **details):
                yield
        finally:
            frame["inclusive_seconds"] = max(0.0, time.perf_counter() - frame["start"])
            frame["process_cpu_seconds"] = max(0.0, time.process_time() - frame["process_start"])
            child_seconds = sum(float(child.get("inclusive_seconds", 0.0)) for child in frame["children"])
            child_process_seconds = sum(
                float(child.get("process_cpu_seconds", 0.0)) for child in frame["children"]
            )
            frame["exclusive_seconds"] = max(0.0, frame["inclusive_seconds"] - child_seconds)
            frame["exclusive_process_cpu_seconds"] = max(
                0.0, frame["process_cpu_seconds"] - child_process_seconds
            )
            self.stack.pop()
            self.all_events.append(frame)
            batch = frame.get("batch")
            if batch is not None:
                if not self.stack:
                    batch["_top_level_stage_inclusive_seconds"] = (
                        float(batch.get("_top_level_stage_inclusive_seconds", 0.0))
                        + float(frame["inclusive_seconds"])
                    )
                by_stage = batch.setdefault("_stage_events", defaultdict(lambda: {"inclusive": 0.0, "exclusive": 0.0, "calls": 0}))
                row = by_stage[frame["name"]]
                row["inclusive"] += frame["inclusive_seconds"]
                row["exclusive"] += frame["exclusive_seconds"]
                row["process_inclusive"] = row.get("process_inclusive", 0.0) + frame["process_cpu_seconds"]
                row["process_exclusive"] = row.get("process_exclusive", 0.0) + frame["exclusive_process_cpu_seconds"]
                row["calls"] += 1

    def install_timing(self) -> None:
        if self.mode == "instrumented":
            self.trainer.timing.stage = self.stage

    def begin_batch(self, *, epoch: int, batch_index: int, indices: list[int]) -> dict[str, Any]:
        if self.active_batch is not None:
            raise RuntimeError("profiling batch overlap")
        unit_ids = []
        dataset = getattr(self.trainer, "dataset", None)
        if dataset is not None:
            unit_ids = [str(dataset.unit_ids[index]) for index in indices]
        batch = {
            "batch_sequence": self._batch_sequence,
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "indices": list(indices),
            "unit_ids": unit_ids,
            "phase": "warmup",
            "_stage_events": defaultdict(lambda: {"inclusive": 0.0, "exclusive": 0.0, "calls": 0}),
            "_started_wall": time.perf_counter(),
            "_started_process": time.process_time(),
            "audit": defaultdict(float),
            "loss_subcomponents": {},
            "finite": {
                "loss": True if self.mode == "instrumented" else None,
                "gradient": True if self.mode == "instrumented" else None,
            },
            "gradient_counts": {},
            "component_steps_before": dict(getattr(self.trainer, "component_steps", {})),
            "connected_components_calls": 0,
            "connected_components_seconds": 0.0,
            "connected_components_process_cpu_seconds": 0.0,
            "optimizer_calls": 0,
            "student_loss_calls": 0,
            "producer_loss_calls": 0,
            "skipped_student_loss_calls": 0,
            "skipped_producer_loss_calls": 0,
            "_top_level_stage_inclusive_seconds": 0.0,
            "logical_work": {},
            "audit_decisions": [],
        }
        self._batch_sequence += 1
        if batch["batch_sequence"] == 0:
            self.cache_before_run = self._cache_stats()
        if batch["batch_sequence"] == self.warmup_batches:
            self.cache_after_warmup = self._cache_stats()
        if self.initial_state is None:
            self.record_initial_state(indices=list(indices), unit_ids=unit_ids)
        self.active_batch = batch
        return batch

    def end_batch(self, batch: dict[str, Any], *, success: bool, error: str | None = None) -> None:
        batch["wall_seconds"] = max(0.0, time.perf_counter() - batch.pop("_started_wall"))
        batch["process_cpu_seconds"] = max(0.0, time.process_time() - batch.pop("_started_process"))
        batch["success"] = bool(success)
        if error:
            batch["error"] = error
        steps_before = batch.pop("component_steps_before")
        steps_after = dict(getattr(self.trainer, "component_steps", {}))
        batch["component_steps"] = {
            name: int(steps_after.get(name, 0) - steps_before.get(name, 0))
            for name in sorted(set(steps_before) | set(steps_after))
        }
        batch["stage_seconds"] = {
            name: {
                "inclusive_seconds": float(values["inclusive"]),
                "exclusive_seconds": float(values["exclusive"]),
                "process_cpu_inclusive_seconds": float(values.get("process_inclusive", 0.0)),
                "process_cpu_exclusive_seconds": float(values.get("process_exclusive", 0.0)),
                "calls": int(values["calls"]),
            }
            for name, values in sorted(batch.pop("_stage_events").items())
        }
        # Direct root stages are accumulated as they close.  Do not search all
        # prior events (which made end_batch O(total_events squared)).
        batch["top_level_stage_inclusive_seconds"] = float(
            batch.pop("_top_level_stage_inclusive_seconds", 0.0)
        )
        batch["unattributed_batch_seconds"] = max(
            0.0, float(batch["wall_seconds"]) - float(batch["top_level_stage_inclusive_seconds"])
        )
        batch["audit"] = {key: (int(value) if float(value).is_integer() else float(value))
                           for key, value in sorted(batch["audit"].items())}
        batch["finite"] = {
            key: (None if value is None else bool(value))
            for key, value in batch["finite"].items()
        }
        batch["gpu_stats"] = _gpu_stats(getattr(self.trainer, "device", None))
        batch["nvidia_smi_sample"] = None
        self.batches.append(batch)
        if (
            self.measured_batches is not None
            and batch["batch_sequence"] == self.warmup_batches + self.measured_batches - 1
        ):
            self.cache_after_measured = self._cache_stats()
        self.active_batch = None

    def add_audit(self, result: Any) -> None:
        batch = self.active_batch
        if batch is None:
            return
        trace = dict(getattr(result, "trace", {}) or {})
        bank = list(getattr(result, "bank", []) or [])
        fitted = list(getattr(result, "fitted", []) or [])
        notes = dict(trace.get("search_notes", {}) or {})
        batch.setdefault("audit_decisions", []).append(_audit_decision_record(result))
        batch["audit"]["units"] += 1
        batch["audit"]["candidates"] += int(trace.get("candidate_count", len(bank)))
        batch["audit"]["fits"] += int(trace.get("fits_performed", len(fitted)))
        batch["audit"]["fit_steps"] += sum(int(getattr(item, "fitting_steps", 0)) for item in fitted)
        batch["audit"]["score_calls"] += int(trace.get("score_calls_total", 0))
        batch["audit"]["primary_score_calls"] += int(trace.get("primary_score_calls", 0))
        batch["audit"]["regional_score_calls"] += int(trace.get("regional_score_calls", 0))
        batch["audit"]["regional_score_observation_count"] += int(trace.get("regional_score_observation_count", 0))
        batch["audit"]["regional_refits"] += int(trace.get("regional_refits", 0))
        batch["audit"]["regions_total"] += int(notes.get("regions_total", 0))
        batch["audit"]["regions_scored"] += int(notes.get("regions_scored", 0))
        batch["audit"]["accepted_edits"] += int(bool(trace.get("accepted", False)))
        batch["audit"]["search_inconclusive"] += int(bool(trace.get("search_inconclusive", False)))
        batch["audit"]["semantic_unresolved"] += int(bool(trace.get("semantic_unresolved", False)))
        batch["audit"]["connected_components_converged"] += int(bool(notes.get("components_converged", True)))

    def add_loss(self, kind: str, loss: Any, metrics: dict[str, Any] | None) -> None:
        batch = self.active_batch
        if batch is None:
            return
        if self.mode == "ordinary":
            if kind == "producer":
                batch["producer_loss_calls"] += 1
                if (metrics or {}).get("skipped"):
                    batch["skipped_producer_loss_calls"] += 1
            else:
                index = int(batch["student_loss_calls"])
                batch["student_loss_calls"] += 1
                if (metrics or {}).get("skipped"):
                    batch["skipped_student_loss_calls"] += 1
            return
        import torch

        finite = bool(torch.isfinite(loss).all().item()) if isinstance(loss, torch.Tensor) else math.isfinite(float(loss))
        if batch["finite"]["loss"] is not None:
            batch["finite"]["loss"] = bool(batch["finite"]["loss"] and finite)
        if kind == "producer":
            batch["producer_loss_calls"] += 1
            target = "producer"
            if (metrics or {}).get("skipped"):
                batch["skipped_producer_loss_calls"] += 1
        else:
            index = int(batch["student_loss_calls"])
            batch["student_loss_calls"] += 1
            target = "student_no_audit" if index == 0 else "student_audited"
            if (metrics or {}).get("skipped"):
                batch["skipped_student_loss_calls"] += 1
        values: dict[str, float | None] = {}
        for name, value in (metrics or {}).items():
            number = _numeric(value)
            values[str(name)] = number
        loss_value = _numeric(loss.detach().cpu().item() if isinstance(loss, torch.Tensor) else loss)
        values.setdefault("loss", loss_value)
        batch["loss_subcomponents"][target] = values

    def add_optimizer(self) -> None:
        batch = self.active_batch
        if batch is not None:
            batch["optimizer_calls"] += 1

    def add_gradients(self, finite: bool, counts: dict[str, int]) -> None:
        batch = self.active_batch
        if batch is not None:
            if batch["finite"]["gradient"] is not None:
                batch["finite"]["gradient"] = bool(batch["finite"]["gradient"] and finite)
            batch["gradient_counts"] = dict(counts)

    def record_sync(self, wall: float, process: float) -> None:
        self.cuda_sync["calls"] += 1
        self.cuda_sync["wall_seconds"] += wall
        self.cuda_sync["process_cpu_seconds"] += process

    def record_connected_components(self, wall: float, process: float) -> None:
        self.connected_components["calls"] += 1
        self.connected_components["wall_seconds"] += wall
        self.connected_components["process_cpu_seconds"] += process
        batch = self.active_batch
        if batch is not None:
            batch["connected_components_calls"] += 1
            batch["connected_components_seconds"] += wall
            batch["connected_components_process_cpu_seconds"] += process


def _gpu_stats(device: Any) -> dict[str, Any] | None:
    if device is None:
        return None
    try:
        import torch
        if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
            return None
        index = device.index if device.index is not None else torch.cuda.current_device()
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        }
    except Exception:  # pragma: no cover - defensive hardware boundary  # noqa: BLE001
        return None


class Instrumentation:
    """Install and restore all process-local wrappers used by ``TimingProbe``."""

    def __init__(self, trainer: Any, probe: TimingProbe) -> None:
        self.trainer = trainer
        self.probe = probe
        self._restorers: list[tuple[Any, str, Any]] = []
        self.audit_hook_name: str | None = None
        self.bulk_audit_enabled_before: bool | None = None
        self.bulk_audit_enabled_during: bool | None = None
        self.bulk_audit_enabled_after: bool | None = None

    def _set(self, owner: Any, name: str, value: Any) -> None:
        self._restorers.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def _bulk_enabled(self) -> bool | None:
        checker = getattr(self.trainer, "_bulk_audit_enabled", None)
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception:  # pragma: no cover - focused compatibility diagnostics  # noqa: BLE001
            return False

    def _install_audit_hook(self) -> None:
        """Hook exactly one canonical audit result sink across source eras."""
        if callable(getattr(self.trainer, "_record_audit", None)):
            name = "_record_audit"
        elif callable(getattr(self.trainer, "_audit_unit", None)):
            name = "_audit_unit"
        else:
            raise TypeError("trainer exposes neither _record_audit nor _audit_unit")
        original = getattr(self.trainer, name)

        def audit_result(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self.probe.add_audit(result)
            return result

        self.audit_hook_name = name
        self._set(self.trainer, name, audit_result)

    def _install_observation_wrappers(self) -> None:
        """Wrap class methods while preserving canonical bound-method identity.

        The trainer's bulk gate compares ``bound.__func__`` with the class
        attribute.  Patching only the instance makes that identity differ and
        silently selects the scalar path.  A class-level wrapper keeps the two
        references identical and records timing only for this trainer's target
        instance; all wrappers are restored after the run.
        """
        observation = self.trainer.observation
        owner = type(observation)
        for name in ("fit", "fit_many", "score", "score_many", "prepare_score", "score_region"):
            original = getattr(owner, name, None)
            if not callable(original):
                continue

            def wrapped(
                instance: Any,
                *args: Any,
                __name: str = name,
                __original: Any = original,
                **kwargs: Any,
            ) -> Any:
                started = time.perf_counter()
                process_started = time.process_time()
                try:
                    return __original(instance, *args, **kwargs)
                finally:
                    if instance is self.trainer.observation:
                        wall = time.perf_counter() - started
                        process = time.process_time() - process_started
                        item_count = None
                        if __name.endswith("_many") and args:
                            try:
                                item_count = len(args[0])
                            except TypeError:
                                item_count = None
                        self.probe.record_logical_call(
                            __name,
                            wall_seconds=wall,
                            process_cpu_seconds=process,
                            item_count=item_count,
                        )

            self._set(owner, name, wrapped)
        self.bulk_audit_enabled_during = self._bulk_enabled()

    def install(self) -> None:
        import torch

        from self_audit_maskfree import auditor, hypotheses, ontology

        # TimingAccumulator remains authoritative for aggregate totals; this
        # wrapper only adds per-call records and leaves its updates intact.
        self.probe.install_timing()
        self.bulk_audit_enabled_before = self._bulk_enabled()

        original_batch = self.trainer._train_batch

        def train_batch(*args: Any, **kwargs: Any) -> Any:
            indices = list(kwargs.get("indices", []))
            batch = self.probe.begin_batch(
                epoch=int(kwargs.get("epoch", -1)),
                batch_index=int(kwargs.get("batch_index", -1)),
                indices=indices,
            )
            try:
                output = original_batch(*args, **kwargs)
            except Exception as exc:
                self.probe.end_batch(batch, success=False, error=f"{type(exc).__name__}: {exc}")
                raise
            else:
                self.probe.end_batch(batch, success=True)
                return output

        self._set(self.trainer, "_train_batch", train_batch)

        # Ordinary mode intentionally stops at the minimal whole-batch timer;
        # finite-gradient hooks, per-loss timing, and progress wrappers belong
        # only to the separate instrumented run.  Lightweight trace/call counts
        # remain so ordinary and instrumented reports can prove matched work.
        if self.probe.mode == "ordinary":
            self._install_audit_hook()
            original_optimizer = self.trainer._optimizer_step

            def optimizer_step_count(*args: Any, **kwargs: Any) -> Any:
                self.probe.add_optimizer()
                return original_optimizer(*args, **kwargs)

            self._set(self.trainer, "_optimizer_step", optimizer_step_count)
            components = self.trainer.components
            original_producer_loss = components.producer_loss
            original_student_loss = components.student_loss

            def producer_loss_count(*args: Any, **kwargs: Any) -> Any:
                result = original_producer_loss(*args, **kwargs)
                self.probe.add_loss("producer", result[0], result[1])
                return result

            def student_loss_count(*args: Any, **kwargs: Any) -> Any:
                result = original_student_loss(*args, **kwargs)
                self.probe.add_loss("student", result[0], result[1])
                return result

            self._set(
                self.trainer,
                "components",
                dataclasses.replace(
                    components,
                    producer_loss=producer_loss_count,
                    student_loss=student_loss_count,
                ),
            )
            self.bulk_audit_enabled_during = self._bulk_enabled()
            return

        self._install_audit_hook()

        original_optimizer = self.trainer._optimizer_step

        def optimizer_step(*args: Any, **kwargs: Any) -> Any:
            finite = True
            counts: dict[str, int] = {}
            for name, model in getattr(self.trainer, "models", {}).items():
                seen = 0
                for parameter in model.parameters():
                    if parameter.grad is None:
                        continue
                    seen += 1
                    finite = finite and bool(torch.isfinite(parameter.grad.detach()).all().item())
                counts[name] = seen
            self.probe.add_gradients(finite, counts)
            self.probe.add_optimizer()
            return original_optimizer(*args, **kwargs)

        self._set(self.trainer, "_optimizer_step", optimizer_step)

        # The frozen Components dataclass can be replaced without modifying
        # production code; both wrappers preserve all return values exactly.
        components = self.trainer.components
        original_producer_loss = components.producer_loss
        original_student_loss = components.student_loss

        def producer_loss(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            process_started = time.process_time()
            try:
                result = original_producer_loss(*args, **kwargs)
                self.probe.add_loss("producer", result[0], result[1])
                return result
            finally:
                self.probe.record_logical_call(
                    "producer_loss",
                    wall_seconds=time.perf_counter() - started,
                    process_cpu_seconds=time.process_time() - process_started,
                )

        def student_loss(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            process_started = time.process_time()
            try:
                result = original_student_loss(*args, **kwargs)
                self.probe.add_loss("student", result[0], result[1])
                return result
            finally:
                self.probe.record_logical_call(
                    "student_loss",
                    wall_seconds=time.perf_counter() - started,
                    process_cpu_seconds=time.process_time() - process_started,
                )

        self._set(
            self.trainer,
            "components",
            dataclasses.replace(
                components,
                producer_loss=producer_loss,
                student_loss=student_loss,
            ),
        )

        original_save = self.trainer._save_checkpoint

        def save_checkpoint(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            process_started = time.process_time()
            try:
                return original_save(*args, **kwargs)
            finally:
                self.probe.checkpoint["calls"] += 1
                self.probe.checkpoint["wall_seconds"] += time.perf_counter() - started
                self.probe.checkpoint["process_cpu_seconds"] += time.process_time() - process_started

        self._set(self.trainer, "_save_checkpoint", save_checkpoint)

        # Progress calls are kept separate from scientific stages.  Quiet
        # progress is the default, but this remains useful with compact output.
        progress = self._current_progress()
        for name in ("update", "dashboard", "event"):
            original = getattr(progress, name)

            def wrapped(*args: Any, __name: str = name, __original: Any = original, **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return __original(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - started
                    self.probe.progress[f"{__name}_calls"] += 1
                    self.probe.progress[f"{__name}_seconds"] += elapsed

            self._set(progress, name, wrapped)

        # Count every actual canonical connected_components invocation exactly
        # once.  The three modules share one wrapper around the original
        # function; wrapping aliases around aliases would double count nested
        # calls from ontology._holes.
        original_components = ontology.connected_components

        def connected_components(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            process_started = time.process_time()
            try:
                return original_components(*args, **kwargs)
            finally:
                self.probe.record_connected_components(
                    time.perf_counter() - started,
                    time.process_time() - process_started,
                )

        for module, name in ((ontology, "connected_components"),
                             (auditor, "connected_components"),
                             (hypotheses, "connected_components")):
            self._set(module, name, connected_components)

        # Count logical observation APIs without changing the trainer's bulk
        # eligibility identity (see _install_observation_wrappers).
        self._install_observation_wrappers()

        if torch.cuda.is_available():
            original_sync = torch.cuda.synchronize

            def synchronize(*args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                process_started = time.process_time()
                try:
                    return original_sync(*args, **kwargs)
                finally:
                    self.probe.record_sync(
                        time.perf_counter() - started,
                        time.process_time() - process_started,
                    )

            self._set(torch.cuda, "synchronize", synchronize)

    def _current_progress(self) -> Any:
        from self_audit_maskfree.progress import current_progress
        return current_progress()

    def restore(self) -> None:
        # Reverse order is important for aliases and nested wrappers.
        for owner, name, original in reversed(self._restorers):
            setattr(owner, name, original)
        self._restorers.clear()
        self.bulk_audit_enabled_after = self._bulk_enabled()


def _aggregate_stage_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "inclusive": [],
            "exclusive": [],
            "process_inclusive": [],
            "process_exclusive": [],
        }
    )
    calls: dict[str, int] = defaultdict(int)
    for batch in batches:
        for name, row in batch.get("stage_seconds", {}).items():
            values[name]["inclusive"].append(float(row["inclusive_seconds"]))
            values[name]["exclusive"].append(float(row["exclusive_seconds"]))
            values[name]["process_inclusive"].append(
                float(row.get("process_cpu_inclusive_seconds", 0.0))
            )
            values[name]["process_exclusive"].append(
                float(row.get("process_cpu_exclusive_seconds", 0.0))
            )
            calls[name] += int(row.get("calls", 0))
    return {
        name: {
            "inclusive": summary(rows["inclusive"]),
            "exclusive": summary(rows["exclusive"]),
            "process_cpu_inclusive": summary(rows["process_inclusive"]),
            "process_cpu_exclusive": summary(rows["process_exclusive"]),
            "calls": count,
        }
        for name, rows in sorted(values.items())
        for count in [calls[name]]
    }


def _component_steps(batches: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for batch in batches:
        for name, value in batch.get("component_steps", {}).items():
            totals[name] += int(value)
    return dict(sorted(totals.items()))


def _aggregate_audit(batches: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = defaultdict(float)
    for batch in batches:
        for name, value in batch.get("audit", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[name] += float(value)
    return {
        name: int(value) if value.is_integer() else value
        for name, value in sorted(totals.items())
    }


def _write_hotspot_report(output: Path, report: dict[str, Any]) -> None:
    stages = report.get("timing", {}).get("stages", {})
    rows = []
    for name, values in stages.items():
        p95 = values.get("inclusive", {}).get("p95_seconds")
        exclusive = values.get("exclusive", {}).get("mean_seconds")
        rows.append((float(p95) if p95 is not None else -1.0, name, p95, exclusive))
    rows.sort(reverse=True)
    lines = [
        "# Maskfree150 local profiling hotspot summary",
        "",
        f"- status: **{report.get('status')}**",
        f"- benchmark kind: `{report.get('benchmark_kind')}`",
        f"- measured batches: `{report.get('profiler', {}).get('measured_batches_captured')}`",
        "- evidence class: bounded local software timing; not RTX 5070 Ti, ACDC, clinical, or production evidence",
        "",
        "## Stage timing (inclusive p95 / exclusive mean)",
        "",
        "| stage | inclusive p95 (s) | exclusive mean (s) |",
        "| --- | ---: | ---: |",
    ]
    for _, name, p95, exclusive in rows:
        lines.append(f"| `{name}` | {p95 if p95 is not None else 'null'} | {exclusive if exclusive is not None else 'null'} |")
    batch = report.get("timing", {}).get("full_scientific_batch_seconds", {})
    lines += [
        "",
        "## Batch accounting",
        "",
        f"- full `_train_batch` p50: `{batch.get('p50_seconds')}` s",
        f"- full `_train_batch` p95: `{batch.get('p95_seconds')}` s",
        f"- process CPU p50: `{report.get('timing', {}).get('process_cpu_seconds', {}).get('p50_seconds')}` s",
        f"- unattributed batch p95 (Python/interval work): `{report.get('timing', {}).get('unattributed_batch_seconds', {}).get('p95_seconds')}` s",
        f"- checkpoint calls/time: `{report.get('timing', {}).get('checkpoint_overhead')}`",
        f"- connected-component calls/time: `{report.get('accounting', {}).get('connected_components')}`",
        "",
        "No optimization is implied by this ranking; preserve bank, rounds, max-fit, regional, student and RNG contracts when comparing a later run.",
        "",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "hotspot_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _assert_fresh_output(output: Path) -> None:
    """Reject output reuse, stale JSONL, and mixed ordinary/instrumented runs."""
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError(f"profile output is not a directory: {output}")
    entries = [path for path in output.iterdir() if path.name not in {".DS_Store"}]
    if entries:
        raise ValueError(
            "profile output must be fresh; refusing reuse or mixed jsonl: "
            + json.dumps(sorted(path.name for path in entries))
        )


def _budget_contract(source_modules: dict[str, ModuleType]) -> dict[str, Any]:
    auditor = source_modules["auditor"]
    observation = source_modules["observation"]
    trainer_module = source_modules["trainer"]
    model = observation.ObservationModel(
        max_iterations=5,
        variance_floor=0.05,
        beta=0.01,
    )
    actual = {
        "bank_size": int(getattr(source_modules["hypotheses"], "BANK_SIZE", -1)),
        # The trainer's canonical audit call is the authority for the number
        # of rounds used by a measured batch; do not hard-code this contract
        # in the profiler and accidentally bless a changed trainer budget.
        "rounds": int(getattr(trainer_module, "AUDIT_ROUNDS", -1)),
        "max_fit_iterations": int(getattr(model, "max_iterations", -1)),
        "max_scored_regions": int(getattr(auditor, "MAX_SCORED_REGIONS", -1)),
        "max_region_score_calls": int(getattr(auditor, "MAX_REGION_SCORE_CALLS", -1)),
    }
    return {
        "required": dict(PROFILE_BUDGET_REQUIREMENTS),
        "actual": actual,
        "matches": actual == PROFILE_BUDGET_REQUIREMENTS,
    }


def _logical_work_report(probe: TimingProbe) -> dict[str, Any]:
    return {
        "counts": {name: int(value) for name, value in sorted(probe.logical_work["counts"].items())},
        "items": {name: int(value) for name, value in sorted(probe.logical_work["items"].items())},
        "wall_seconds": {
            name: float(value) for name, value in sorted(probe.logical_work["wall_seconds"].items())
        },
        "process_cpu_seconds": {
            name: float(value)
            for name, value in sorted(probe.logical_work["process_cpu_seconds"].items())
        },
    }


def _cache_report(probe: TimingProbe) -> dict[str, Any]:
    """Report warmup/measured cache state and counter deltas when available."""
    before = probe.cache_before_run
    after_warmup = probe.cache_after_warmup
    after_measured = probe.cache_after_measured
    if not all(
        state is not None and bool(state.get("available"))
        for state in (before, after_warmup, after_measured)
    ):
        return {
            "available": False,
            "before_run": before,
            "after_warmup": after_warmup,
            "after_measured": after_measured,
            "delta_measured": None,
        }
    after_warmup_stats = dict(after_warmup["stats"])
    after_measured_stats = dict(after_measured["stats"])
    delta = {
        key: int(after_measured_stats.get(key, 0) - after_warmup_stats.get(key, 0))
        for key in sorted(set(after_measured_stats) | set(after_warmup_stats))
        if isinstance(after_measured_stats.get(key, 0), (int, float))
        and isinstance(after_warmup_stats.get(key, 0), (int, float))
    }
    return {
        "available": True,
        "before_run": before,
        "after_warmup": after_warmup,
        "after_measured": after_measured,
        "delta_measured": delta,
    }


def _image_inventory(manifest: dict[str, Any]) -> dict[str, Any]:
    """Hash actual image source/frame identities for real-data pair matching."""
    rows = []
    for record in list(manifest.get("records", []) or []):
        rows.append(
            {
                key: record.get(key)
                for key in (
                    "unit_id", "split", "path", "source_hash", "frame_fingerprint",
                    "frame_axis", "depth_axis", "frame_index",
                )
            }
        )
    rows.sort(key=lambda row: str(row.get("unit_id", "")))
    return {
        "record_count": len(rows),
        "source_count": len({str(row.get("path")) for row in rows if row.get("path")}),
        "records_sha256": _sha256_payload(rows),
        "source_paths_sha256": _sha256_payload(
            sorted({str(row.get("path")) for row in rows if row.get("path")})
        ),
    }


def _compare_reports(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable equivalence gate for ordinary/instrumented or A/B runs."""
    checks: dict[str, Any] = {}
    numeric_differences: dict[str, Any] = {}

    def _nonempty(value: Any) -> bool:
        return value is not None and value != "" and value != [] and value != {}

    checks["status"] = (
        reference.get("status") == "completed_bounded"
        and candidate.get("status") == "completed_bounded"
    )
    checks["timing_mode"] = (
        reference.get("profiler", {}).get("timing_mode")
        == candidate.get("profiler", {}).get("timing_mode")
    )
    ref_scientific_hash = reference.get("config", {}).get("scientific_hash")
    cand_scientific_hash = candidate.get("config", {}).get("scientific_hash")
    checks["scientific_hash"] = _nonempty(ref_scientific_hash) and ref_scientific_hash == cand_scientific_hash
    ref_fixture = reference.get("units", {}).get("fixture") or {}
    cand_fixture = candidate.get("units", {}).get("fixture") or {}
    fixture_keys = ("source_sha256", "shape", "dtype", "seed", "format")
    both_synthetic = bool(ref_fixture) and bool(cand_fixture)
    both_real = not ref_fixture and not cand_fixture
    checks["fixture_content"] = both_synthetic and all(
        _nonempty(ref_fixture.get(key))
        and _nonempty(cand_fixture.get(key))
        and ref_fixture.get(key) == cand_fixture.get(key)
        for key in fixture_keys
    ) if both_synthetic else both_real
    checks["fixture_path"] = (
        both_synthetic
        and _nonempty(ref_fixture.get("source"))
        and _nonempty(cand_fixture.get("source"))
        and ref_fixture.get("source") == cand_fixture.get("source")
    ) if both_synthetic else both_real
    checks["image_inventory"] = (
        both_real
        and _nonempty(reference.get("units", {}).get("image_inventory"))
        and reference.get("units", {}).get("image_inventory")
        == candidate.get("units", {}).get("image_inventory")
    ) or (
        both_synthetic
        and _nonempty(reference.get("units", {}).get("image_inventory"))
        and reference.get("units", {}).get("image_inventory")
        == candidate.get("units", {}).get("image_inventory")
    )
    ref_manifest = reference.get("units", {}).get("manifest_hash")
    cand_manifest = candidate.get("units", {}).get("manifest_hash")
    checks["manifest_hash"] = _nonempty(ref_manifest) and ref_manifest == cand_manifest
    ref_unit_ids = reference.get("units", {}).get("unit_ids")
    cand_unit_ids = candidate.get("units", {}).get("unit_ids")
    checks["unit_ids"] = _nonempty(ref_unit_ids) and ref_unit_ids == cand_unit_ids
    ref_sample_order = reference.get("sample_order_digest")
    cand_sample_order = candidate.get("sample_order_digest")
    checks["sample_order_digest"] = _nonempty(ref_sample_order) and ref_sample_order == cand_sample_order
    ref_initial_model = reference.get("initial_state", {}).get("model_state_sha256")
    cand_initial_model = candidate.get("initial_state", {}).get("model_state_sha256")
    checks["initial_model_state"] = _nonempty(ref_initial_model) and ref_initial_model == cand_initial_model
    ref_initial_rng = reference.get("initial_state", {}).get("rng", {}).get("digest")
    cand_initial_rng = candidate.get("initial_state", {}).get("rng", {}).get("digest")
    checks["initial_rng"] = _nonempty(ref_initial_rng) and ref_initial_rng == cand_initial_rng
    ref_final_model = reference.get("final_state", {}).get("model_state_sha256")
    cand_final_model = candidate.get("final_state", {}).get("model_state_sha256")
    checks["final_model_state"] = _nonempty(ref_final_model) and ref_final_model == cand_final_model
    ref_final_rng = reference.get("final_state", {}).get("rng", {}).get("digest")
    cand_final_rng = candidate.get("final_state", {}).get("rng", {}).get("digest")
    checks["final_rng"] = _nonempty(ref_final_rng) and ref_final_rng == cand_final_rng
    def _source_valid(report: dict[str, Any]) -> bool:
        source = report.get("source", {})
        observed = source.get("observed_import", {})
        activation = source.get("activation", {})
        return bool(
            source.get("snapshot_check", {}).get("ok") is True
            and _nonempty(source.get("snapshot_check", {}).get("manifest_combined_sha256"))
            and source.get("snapshot_check", {}).get("manifest_combined_sha256")
            == source.get("snapshot_check", {}).get("observed_combined_sha256")
            and _nonempty(source.get("snapshot_path"))
            and _nonempty(activation.get("import_root"))
            and activation.get("import_root") == observed.get("import_root")
            and _nonempty(observed.get("package_combined_sha256"))
            and _nonempty(observed.get("package_root"))
            and not observed.get("outside_package_paths")
        )

    checks["source_identity"] = _source_valid(reference) and _source_valid(candidate)
    def _bulk_identity_valid(report: dict[str, Any]) -> bool:
        bulk = report.get("profiler", {}).get("canonical_bulk_audit", {})
        before = bulk.get("enabled_before_hooks")
        during = bulk.get("enabled_during_hooks")
        after = bulk.get("enabled_after_restore")
        return bool(
            (before is True and during is True and after is True)
            or (before is False and during is False and after is False)
        )

    checks["canonical_bulk_identity"] = (
        _bulk_identity_valid(reference) and _bulk_identity_valid(candidate)
    )
    ref_harness = reference.get("harness", {})
    cand_harness = candidate.get("harness", {})
    checks["harness_sha256"] = bool(ref_harness.get("script_sha256")) and (
        ref_harness.get("script_sha256") == cand_harness.get("script_sha256")
    )
    checks["harness_runtime"] = all(
        _nonempty(ref_harness.get(key))
        and _nonempty(cand_harness.get(key))
        and ref_harness.get(key) == cand_harness.get(key)
        for key in (
            "python_executable",
            "python_version",
            "thread_env",
            "determinism_env",
            "torch_num_threads",
            "torch_num_interop_threads",
            "torch_deterministic_algorithms_preinit",
            "torch_deterministic_algorithms_effective",
        )
    )
    checks["effective_determinism"] = (
        ref_harness.get("torch_deterministic_algorithms_effective") is True
        and cand_harness.get("torch_deterministic_algorithms_effective") is True
    )
    checks["amp_disabled"] = (
        reference.get("trainer", {}).get("amp_enabled") is False
        and candidate.get("trainer", {}).get("amp_enabled") is False
    )
    ref_accounting = reference.get("accounting", {})
    cand_accounting = candidate.get("accounting", {})
    for key in ("component_steps", "physical_batch_api_calls", "logical_units"):
        checks[f"accounting_{key}"] = (
            _nonempty(ref_accounting.get(key))
            and _nonempty(cand_accounting.get(key))
            and ref_accounting.get(key) == cand_accounting.get(key)
        )
    ref_budget = reference.get("budget_contract", {})
    cand_budget = candidate.get("budget_contract", {})
    checks["budget_contract"] = bool(
        ref_budget.get("matches") is True
        and cand_budget.get("matches") is True
        and _nonempty(ref_budget.get("required"))
        and ref_budget == cand_budget
    )
    ref_audit = ref_accounting.get("audit", {})
    cand_audit = cand_accounting.get("audit", {})
    budget_fields = (
        "units", "candidates", "fits", "fit_steps", "primary_score_calls",
        "regional_score_calls", "regional_score_observation_count", "regions_total",
        "regions_scored", "score_calls", "accepted_edits", "semantic_unresolved",
    )
    for key in budget_fields:
        checks[f"audit_budget_{key}"] = (
            _nonempty(ref_audit.get(key))
            and _nonempty(cand_audit.get(key))
            and ref_audit.get(key) == cand_audit.get(key)
        )
    checks["all_budget_counts"] = checks["budget_contract"] and all(
        checks[f"audit_budget_{key}"] for key in budget_fields
    )
    checks["all_sample_ids"] = checks["sample_order_digest"] and checks["unit_ids"]
    def _batch_contract_valid(report: dict[str, Any]) -> bool:
        rows = [batch for batch in report.get("batches", []) if batch.get("phase") == "measured"]
        if not rows:
            return False
        return all(
            bool(batch.get("success"))
            and int(batch.get("producer_loss_calls", 0)) == 1
            and int(batch.get("student_loss_calls", 0)) == 2
            and int(batch.get("optimizer_calls", 0)) == 1
            and all(int(batch.get("component_steps", {}).get(name, 0)) == 1
                    for name in ("producer", "student_audited", "student_no_audit"))
            and int(batch.get("skipped_student_loss_calls", 0)) == 0
            and int(batch.get("skipped_producer_loss_calls", 0)) == 0
            for batch in rows
        )

    checks["per_batch_accounting"] = (
        _batch_contract_valid(reference) and _batch_contract_valid(candidate)
    )
    ref_logical = ref_accounting.get("logical_work", {})
    cand_logical = cand_accounting.get("logical_work", {})
    numeric_differences["logical_work"] = {
        "counts_reference": ref_logical.get("counts"),
        "counts_candidate": cand_logical.get("counts"),
        "items_reference": ref_logical.get("items"),
        "items_candidate": cand_logical.get("items"),
    }
    ref_decisions = [
        decision
        for batch in reference.get("batches", [])
        if batch.get("phase") == "measured"
        for decision in batch.get("audit_decisions", [])
    ]
    cand_decisions = [
        decision
        for batch in candidate.get("batches", [])
        if batch.get("phase") == "measured"
        for decision in batch.get("audit_decisions", [])
    ]
    def _discrete_decision(value: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item
            for key, item in value.items()
            if key not in {
                "rejected_gains",
                "accepted_gain",
                "fit_scores",
                "select_scores",
                "audited_validity_sha256",
                "regional_margin_sha256",
                "audited_validity_stats",
                "regional_margin_stats",
            }
        }

    discrete_match = bool(ref_decisions) and [
        _discrete_decision(value) for value in ref_decisions
    ] == [
        _discrete_decision(value) for value in cand_decisions
    ]
    gain_deltas: list[float] = []
    accepted_gain_deltas: list[float] = []
    score_metadata_match = True
    score_numeric_deltas: list[dict[str, Any]] = []
    tensor_differences: list[dict[str, Any]] = []
    if len(ref_decisions) == len(cand_decisions):
        for reference_decision, candidate_decision in zip(ref_decisions, cand_decisions):
            tensor_differences.append(
                {
                    "unit_id": reference_decision.get("unit_id"),
                    "audited_validity_hash_equal": (
                        _nonempty(reference_decision.get("audited_validity_sha256"))
                        and reference_decision.get("audited_validity_sha256")
                        == candidate_decision.get("audited_validity_sha256")
                    ),
                    "regional_margin_hash_equal": (
                        _nonempty(reference_decision.get("regional_margin_sha256"))
                        and reference_decision.get("regional_margin_sha256")
                        == candidate_decision.get("regional_margin_sha256")
                    ),
                    "audited_validity_stats_reference": reference_decision.get("audited_validity_stats"),
                    "audited_validity_stats_candidate": candidate_decision.get("audited_validity_stats"),
                    "regional_margin_stats_reference": reference_decision.get("regional_margin_stats"),
                    "regional_margin_stats_candidate": candidate_decision.get("regional_margin_stats"),
                }
            )
            reference_gains = list(reference_decision.get("rejected_gains", []))
            candidate_gains = list(candidate_decision.get("rejected_gains", []))
            if len(reference_gains) != len(candidate_gains):
                gain_deltas.append(1e300)
            else:
                gain_deltas.extend(
                    abs(float(left) - float(right))
                    for left, right in zip(reference_gains, candidate_gains)
                )
            reference_gain = reference_decision.get("accepted_gain")
            candidate_gain = candidate_decision.get("accepted_gain")
            if reference_gain is None or candidate_gain is None:
                accepted_gain_deltas.append(1e300)
            else:
                accepted_gain_deltas.append(abs(float(reference_gain) - float(candidate_gain)))

            for score_kind in ("fit_scores", "select_scores"):
                reference_scores = list(reference_decision.get(score_kind, []) or [])
                candidate_scores = list(candidate_decision.get(score_kind, []) or [])
                if len(reference_scores) != len(candidate_scores):
                    score_metadata_match = False
                    score_numeric_deltas.append(
                        {"kind": score_kind, "length_mismatch": True, "max_abs_delta": 1e300}
                    )
                    continue
                for index, (reference_score, candidate_score) in enumerate(
                    zip(reference_scores, candidate_scores)
                ):
                    if reference_score is None or candidate_score is None:
                        score_metadata_match = False
                        score_numeric_deltas.append(
                            {
                                "kind": score_kind,
                                "index": index,
                                "metadata_equal": False,
                                "max_abs_delta": 1e300,
                            }
                        )
                        continue
                    metadata_fields = (
                        "available", "count", "reason", "role", "finite_numeric",
                    )
                    metadata_equal = all(
                        reference_score.get(field) == candidate_score.get(field)
                        for field in metadata_fields
                    )
                    exact_fields_equal = all(
                        reference_score.get(field) == candidate_score.get(field)
                        for field in ("complexity", "prior")
                    )
                    score_deltas: dict[str, float] = {}
                    for field in ("complexity", "prior", "nll_sum", "normalized_nll", "total"):
                        left = reference_score.get(field)
                        right = candidate_score.get(field)
                        if left is None or right is None:
                            score_deltas[field] = 1e300
                        else:
                            score_deltas[field] = abs(float(left) - float(right))
                    score_metadata_match = score_metadata_match and metadata_equal and exact_fields_equal
                    score_numeric_deltas.append(
                        {
                            "kind": score_kind,
                            "index": index,
                            "metadata_equal": metadata_equal,
                            "exact_fields_equal": exact_fields_equal,
                            "field_abs_deltas": score_deltas,
                            "max_abs_delta": max(score_deltas.values(), default=0.0),
                        }
                    )
    else:
        gain_deltas.append(1e300)
        accepted_gain_deltas.append(1e300)
        score_metadata_match = False
    max_gain_delta = max(gain_deltas, default=0.0)
    max_accepted_gain_delta = max(accepted_gain_deltas, default=0.0)
    max_score_delta = max(
        float(row.get("max_abs_delta", 0.0)) for row in score_numeric_deltas
    ) if score_numeric_deltas else 0.0
    score_tolerance_ok = True
    for row in score_numeric_deltas:
        field_deltas = row.get("field_abs_deltas", {})
        if row.get("length_mismatch") or row.get("metadata_equal") is False:
            score_tolerance_ok = False
            continue
        if field_deltas.get("complexity", 1e300) != 0.0 or field_deltas.get("prior", 1e300) != 0.0:
            score_tolerance_ok = False
        if field_deltas.get("nll_sum", 1e300) > 3e-10:
            score_tolerance_ok = False
        if max(
            field_deltas.get("normalized_nll", 1e300),
            field_deltas.get("total", 1e300),
        ) > 1e-12:
            score_tolerance_ok = False
    checks["audit_decisions"] = discrete_match
    checks["audit_tensor_hashes"] = bool(tensor_differences) and all(
        row["audited_validity_hash_equal"] and row["regional_margin_hash_equal"]
        for row in tensor_differences
    )
    checks["audit_gain_tolerance"] = max_gain_delta <= 1e-12
    checks["audit_accepted_gain_tolerance"] = max_accepted_gain_delta <= 1e-12
    checks["audit_score_metadata"] = score_metadata_match
    checks["audit_score_tolerance"] = score_tolerance_ok
    if any(not math.isfinite(value) for value in gain_deltas + accepted_gain_deltas):
        checks["audit_gain_tolerance"] = False
    numeric_differences["audit_gain_max_abs_delta"] = max_gain_delta
    numeric_differences["audit_accepted_gain_max_abs_delta"] = max_accepted_gain_delta
    numeric_differences["audit_score_max_abs_delta"] = max_score_delta
    numeric_differences["audit_score_fields"] = score_numeric_deltas
    numeric_differences["audit_tensor_fields"] = tensor_differences
    return {
        "schema_version": "maskfree150.matched-report.v1",
        "reference_kind": reference.get("benchmark_kind"),
        "candidate_kind": candidate.get("benchmark_kind"),
        "checks": checks,
        "matched": all(bool(value) for value in checks.values()),
        "source_hashes": {
            "reference": reference.get("source", {}).get("observed_import", {}).get("package_combined_sha256"),
            "candidate": candidate.get("source", {}).get("observed_import", {}).get("package_combined_sha256"),
        },
        "numeric_differences": numeric_differences,
        "note": "source hashes may differ for baseline versus optimized code; all data/model/sample contracts must match",
    }


def _prepare_config(args: argparse.Namespace, output: Path) -> tuple[Any, dict[str, Any]]:
    from self_audit_maskfree.config import MaskfreeConfig, load_config

    fixture: dict[str, Any] | None = None
    if args.synthetic:
        if args.seed != 42:
            raise ValueError("the matched maskfree150 profiler fixture is fixed at seed=42")
        fixture_root = (
            Path(args.synthetic_root).expanduser().resolve()
            if args.synthetic_root is not None
            else output / "synthetic_data"
        )
        fixture = make_synthetic_fixture(
            fixture_root,
            seed=args.seed,
            image_size=128,
            depth=SYNTHETIC_DEPTH,
            file_format=args.synthetic_format,
        )
        config = MaskfreeConfig(
            dataset="acdc",
            data_root=fixture["root"],
            output_dir=str(output / "trainer_workspace"),
            total_epochs=150,
            seed=args.seed,
            batch_size=8,
            accumulation_steps=1,
            image_size=128,
            lr=0.001,
            weight_decay=0.0001,
            warmup_epochs=5,
            width=16,
            feature_dim=16,
            protocol="auto",
            depth_axis=2,
            device=args.device or "cpu",
            num_workers=0,
            amp=False,
            wandb_mode="disabled",
            wandb_project="self-audit-maskfree",
            run_id=args.run_id,
            max_steps=args.warmup_batches + args.measured_batches,
            max_epochs=None,
            resume=None,
            allow_cpu=(args.device or "cpu") == "cpu",
            epoch_validation=True,
            epoch_reference_config=None,
        )
    else:
        if args.config is None:
            raise ValueError("--config or --synthetic is required")
        config = load_config(args.config)
        requested_device = args.device or config.device
        config = config.replace(
            output_dir=str(output / "trainer_workspace"),
            run_id=args.run_id,
            device=requested_device,
            max_steps=args.warmup_batches + args.measured_batches,
            max_epochs=None,
            resume=None,
            wandb_mode="disabled",
            allow_cpu=bool(config.allow_cpu or requested_device == "cpu"),
        )

    values = {name: getattr(config, name) for name in SCIENTIFIC_REQUIREMENTS}
    mismatches = {
        name: {"expected": expected, "actual": values[name]}
        for name, expected in SCIENTIFIC_REQUIREMENTS.items()
        if values[name] != expected
    }
    if mismatches:
        raise ValueError(
            "profiling requires the scientific150 FP32 contract; mismatches: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return config, {"synthetic_fixture": fixture}


def _run_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one bounded trainer run and write a JSON report."""
    if args.warmup_batches < 0 or args.measured_batches <= 0:
        raise ValueError("warmup-batches must be non-negative and measured-batches must be positive")
    if args.warmup_batches + args.measured_batches >= PARTIAL_EPOCH_MAX_BATCHES:
        raise ValueError(
            "profiling bound crosses the fixed 1024-unit synthetic epoch; "
            "keep warmup+measured below 128 batches"
        )
    output = Path(args.output).expanduser().resolve()
    _assert_fresh_output(output)
    output.mkdir(parents=True, exist_ok=True)
    harness = _harness_identity()

    # A baseline with no explicit root snapshots the live checkout once, then
    # imports exclusively from those copied bytes.  This is deliberately done
    # before the first self_audit_maskfree import.
    external_source = Path(args.source_root).expanduser().resolve() if args.source_root else None
    if external_source is None:
        snapshot_dir = output / "source_snapshot"
        snapshot = snapshot_source(REPO_ROOT, snapshot_dir)
        selected_source_root = snapshot_dir / "files"
        source_snapshot_path = snapshot_dir
    else:
        if (external_source / "snapshot_manifest.json").is_file():
            source_snapshot_path = external_source
            selected_source_root = external_source / "files"
        elif (external_source.parent / "snapshot_manifest.json").is_file() and external_source.name == "files":
            source_snapshot_path = external_source.parent
            selected_source_root = external_source
        else:
            raise ValueError(
                "--source-root must point to an immutable source snapshot directory "
                "or its files/ child"
            )
        snapshot = json.loads(
            (source_snapshot_path / "snapshot_manifest.json").read_text(encoding="utf-8")
        )
        verify_snapshot(source_snapshot_path, snapshot)

    activation = _activate_source_root(selected_source_root)
    from self_audit_maskfree import auditor, hypotheses, observation, runtime
    from self_audit_maskfree.trainer import MaskfreeTrainer

    observed_import_before = _observed_import_identity(selected_source_root)
    config, preparation = _prepare_config(args, output)
    _write_json(output / "resolved_config.json", config.to_dict())
    if args.reference_snapshot:
        reference = Path(args.reference_snapshot).expanduser().resolve()
        if not (reference / "snapshot_manifest.json").is_file():
            raise ValueError(f"reference snapshot manifest not found: {reference}")
        reference_manifest = json.loads(
            (reference / "snapshot_manifest.json").read_text(encoding="utf-8")
        )
        verify_snapshot(reference, reference_manifest)
        _write_json(output / "reference_snapshot_pointer.json", {"path": str(reference)})

    source_before = runtime.source_identity(REPO_ROOT)
    rng_before = _rng_digest(runtime)
    nvidia = _nvidia_smi()
    trainer = MaskfreeTrainer(config, repo_root=REPO_ROOT)
    probe = TimingProbe(
        trainer,
        mode=args.timing_mode,
        runtime_module=runtime,
        warmup_batches=args.warmup_batches,
        measured_batches=args.measured_batches,
    )
    instrumentation = Instrumentation(trainer, probe)
    instrumentation.install()
    profile = cProfile.Profile() if args.cprofile else None
    result: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None = None
    started_wall = time.perf_counter()
    started_process = time.process_time()
    try:
        if profile is not None:
            profile.enable()
        result = trainer.run()
    except Exception as exc:  # noqa: BLE001 - preserve trainer failure evidence
        error_payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if profile is not None:
            profile.disable()
            profile_path = Path(args.cprofile_output).expanduser() if args.cprofile_output else output / "profile.prof"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile.dump_stats(str(profile_path))
        instrumentation.restore()
    elapsed_wall = max(0.0, time.perf_counter() - started_wall)
    elapsed_process = max(0.0, time.process_time() - started_process)

    # Recompute the package identity after run completion rather than trusting
    # the cached hash used by the trainer's run identity.
    source_after = dict(runtime.source_identity(REPO_ROOT))
    source_after["package"] = runtime.package_source_hash(refresh=True)
    snapshot_check = verify_snapshot(source_snapshot_path, snapshot)
    live_source_check = source_snapshot_unchanged(REPO_ROOT, snapshot)
    observed_import_after = _observed_import_identity(selected_source_root)
    rng_after = _rng_digest(runtime)
    final_model_state_sha256 = _model_state_digest(trainer)
    import torch

    harness["torch_deterministic_algorithms_effective"] = bool(
        torch.are_deterministic_algorithms_enabled()
    )
    all_batches = list(probe.batches)
    for batch in all_batches:
        batch["phase"] = "warmup" if batch["batch_sequence"] < args.warmup_batches else "measured"
    measured = [batch for batch in all_batches if batch["phase"] == "measured"]
    warmup = [batch for batch in all_batches if batch["phase"] == "warmup"]
    for batch in all_batches:
        batch["nvidia_smi_sample"] = nvidia if batch["batch_sequence"] == 0 else None
        _append_jsonl(output / "batch_metrics.jsonl", batch)

    batch_wall_values = [float(batch["wall_seconds"]) for batch in measured if batch.get("success")]
    process_values = [float(batch["process_cpu_seconds"]) for batch in measured if batch.get("success")]
    unattributed_values = [float(batch["unattributed_batch_seconds"]) for batch in measured if batch.get("success")]
    gpu_rows = [batch.get("gpu_stats") for batch in measured if batch.get("gpu_stats") is not None]
    peak_gpu = None
    if gpu_rows:
        peak_gpu = {
            key: max(int(row.get(key, 0)) for row in gpu_rows)
            for key in ("allocated_bytes", "reserved_bytes", "max_allocated_bytes", "max_reserved_bytes")
        }
    timing = {
        "full_scientific_batch_seconds": summary(batch_wall_values),
        "process_cpu_seconds": summary(process_values),
        "unattributed_batch_seconds": summary(unattributed_values),
        "stages": _aggregate_stage_batches([batch for batch in measured if batch.get("success")]),
        "interval_overhead": dict(probe.progress),
        "checkpoint_overhead": dict(probe.checkpoint),
        "cuda_sync": dict(probe.cuda_sync),
        "gpu_memory_peak": peak_gpu,
        "run_wall_seconds": elapsed_wall,
        "run_process_cpu_seconds": elapsed_process,
        "stage_aggregation_note": "inclusive nested stages overlap; do not sum them into batch latency",
        "batch_scope": "trainer._train_batch; progress/checkpoint overhead is reported separately",
        "timing_mode": args.timing_mode,
        "logical_work": _logical_work_report(probe),
        "raw_neural_compute_note": "producer/feature/student stages exclude audit, optimizer, checkpoint, and logging; nested values are not summed",
    }
    finite_loss_values = [batch.get("finite", {}).get("loss") for batch in measured]
    finite_gradient_values = [batch.get("finite", {}).get("gradient") for batch in measured]
    finite_loss_known = [value for value in finite_loss_values if value is not None]
    finite_gradient_known = [value for value in finite_gradient_values if value is not None]
    logical_work = _logical_work_report(probe)
    cache = _cache_report(probe)
    accounting = {
        "warmup_batches": len(warmup),
        "measured_batches": len(measured),
        "requested_measured_batches": args.measured_batches,
        "component_steps": _component_steps(measured),
        "audit": _aggregate_audit(measured),
        "connected_components": dict(probe.connected_components),
        "loss_subcomponents": {
            str(batch["batch_sequence"]): batch.get("loss_subcomponents", {}) for batch in measured
        },
        "finite_loss_batches": sum(bool(value) for value in finite_loss_known) if finite_loss_known else None,
        "finite_gradient_batches": sum(bool(value) for value in finite_gradient_known) if finite_gradient_known else None,
        "finite_checks_available": bool(finite_loss_known and finite_gradient_known),
        "student_loss_calls": sum(int(batch.get("student_loss_calls", 0)) for batch in measured),
        "producer_loss_calls": sum(int(batch.get("producer_loss_calls", 0)) for batch in measured),
        "optimizer_calls": sum(int(batch.get("optimizer_calls", 0)) for batch in measured),
        "student_loss_skips": sum(int(batch.get("skipped_student_loss_calls", 0)) for batch in measured),
        "producer_loss_skips": sum(int(batch.get("skipped_producer_loss_calls", 0)) for batch in measured),
        "physical_batch_api_calls": len(all_batches),
        "logical_units": int(sum(float(batch.get("audit", {}).get("units", 0)) for batch in measured)),
        "logical_work": logical_work,
        "cache": cache,
        "student_steps_note": "zero student steps/skips are retained as observed; no work is manufactured",
    }
    manifest = getattr(trainer, "manifest", {}) or {}
    unit_ids = list(getattr(trainer, "unit_ids", []) or [])
    budget_contract = _budget_contract({
        "auditor": auditor,
        "hypotheses": hypotheses,
        "observation": observation,
        "trainer": importlib.import_module(MaskfreeTrainer.__module__),
    })
    validation_errors: list[str] = []
    audit_totals = accounting.get("audit", {})
    expected_units = int(config.batch_size) * int(args.measured_batches)
    nonvacuous_expectations = {
        "units": expected_units,
        "candidates": 4 * expected_units,
        "fits": 4 * expected_units,
        "fit_steps": 5 * 4 * expected_units,
        "primary_score_calls": 4 * expected_units,
    }
    for name, expected in nonvacuous_expectations.items():
        observed = int(audit_totals.get(name, 0))
        if observed != expected:
            validation_errors.append(
                f"audit accounting {name} mismatch: expected {expected}, observed {observed}"
            )
    if int(audit_totals.get("regional_score_calls", 0)) <= 0:
        validation_errors.append("regional score calls are zero")
    if int(audit_totals.get("score_calls", 0)) <= 0:
        validation_errors.append("score calls are zero")
    for batch in measured:
        expected_steps = {"producer": 1, "student_audited": 1, "student_no_audit": 1}
        if int(batch.get("producer_loss_calls", 0)) != 1:
            validation_errors.append(
                f"batch {batch.get('batch_sequence')} producer loss calls are not exactly one"
            )
        if int(batch.get("student_loss_calls", 0)) != 2:
            validation_errors.append(
                f"batch {batch.get('batch_sequence')} student loss calls are not exactly two"
            )
        if int(batch.get("optimizer_calls", 0)) != 1:
            validation_errors.append(
                f"batch {batch.get('batch_sequence')} optimizer calls are not exactly one"
            )
        observed_steps = batch.get("component_steps", {})
        for name, expected in expected_steps.items():
            if int(observed_steps.get(name, 0)) != expected:
                validation_errors.append(
                    f"batch {batch.get('batch_sequence')} component step {name} is not one"
                )
        if int(batch.get("skipped_student_loss_calls", 0)) or int(
            batch.get("skipped_producer_loss_calls", 0)
        ):
            validation_errors.append(
                f"batch {batch.get('batch_sequence')} contains skipped loss calls"
            )
    measured_decisions = [
        decision
        for batch in measured
        for decision in batch.get("audit_decisions", [])
    ]
    if len(measured_decisions) != expected_units:
        validation_errors.append("ordered per-unit audit decision evidence is incomplete")
    if any(
        not decision.get("audited_validity_sha256")
        or not decision.get("regional_margin_sha256")
        for decision in measured_decisions
    ):
        validation_errors.append("audited validity/regional margin hash evidence is incomplete")
    if any(
        (decision.get("audited_validity_stats") or {}).get("finite") is not True
        or (decision.get("regional_margin_stats") or {}).get("finite") is not True
        or any(
            score is None or score.get("finite_numeric") is not True
            for score in list(decision.get("fit_scores", []))
            + list(decision.get("select_scores", []))
        )
        for decision in measured_decisions
    ):
        validation_errors.append("nonfinite or incomplete per-unit score/decision evidence")
    bulk_state = {
        "before": instrumentation.bulk_audit_enabled_before,
        "during": instrumentation.bulk_audit_enabled_during,
        "after": instrumentation.bulk_audit_enabled_after,
    }
    if bulk_state["before"] and not (bulk_state["during"] and bulk_state["after"]):
        validation_errors.append(
            "canonical bulk eligibility changed while instrumentation hooks were installed"
        )
    if args.timing_mode == "instrumented" and bulk_state["during"]:
        fit_many_items = [
            int(batch.get("logical_work", {}).get("fit_many", {}).get("items", 0))
            for batch in measured
        ]
        if any(item != 32 for item in fit_many_items):
            validation_errors.append(
                "canonical bulk fit_many item count mismatch: expected 32 per measured batch"
            )
    if result is None:
        validation_errors.append("trainer returned no result")
    elif result.get("status") not in {"partial", "completed"}:
        validation_errors.append(f"trainer status is not runnable: {result.get('status')!r}")
    if len(warmup) != args.warmup_batches:
        validation_errors.append(
            f"warmup batch capture incomplete: expected {args.warmup_batches}, observed {len(warmup)}"
        )
    if len(measured) != args.measured_batches:
        validation_errors.append(
            f"measured batch capture incomplete: expected {args.measured_batches}, observed {len(measured)}"
        )
    if any(not bool(batch.get("success")) for batch in all_batches):
        validation_errors.append("one or more requested batches failed")
    if args.timing_mode == "instrumented" and any(
        value is False for value in finite_loss_values + finite_gradient_values
    ):
        validation_errors.append("nonfinite loss or gradient observed")
    if not snapshot_check.get("ok", False):
        validation_errors.append("immutable imported source snapshot drifted")
    if observed_import_after.get("import_root") != activation.get("import_root"):
        validation_errors.append("observed import root differs from activated source snapshot")
    if observed_import_after.get("outside_package_paths"):
        validation_errors.append("observed package imported modules outside selected snapshot")
    if not budget_contract["matches"]:
        validation_errors.append("profiling budget constants differ from the required contract")
    if args.synthetic and len(unit_ids) < SYNTHETIC_TRAIN_UNITS_MIN:
        validation_errors.append(
            f"synthetic train cohort is too small: {len(unit_ids)} < {SYNTHETIC_TRAIN_UNITS_MIN}"
        )
    if args.warmup_batches + args.measured_batches >= PARTIAL_EPOCH_MAX_BATCHES:
        validation_errors.append(
            "requested profiling bound crosses the fixed synthetic epoch; use a partial-epoch bound"
        )
    if source_before.get("package", {}).get("combined") != source_after.get("package", {}).get("combined"):
        validation_errors.append("observed imported package source hash changed during run")
    initial_state = probe.initial_state or {}
    if "torch_deterministic_algorithms_effective" in initial_state:
        harness["torch_deterministic_algorithms_effective"] = initial_state[
            "torch_deterministic_algorithms_effective"
        ]
    if initial_state.get("torch_deterministic_algorithms_effective") is not True:
        validation_errors.append("effective deterministic-algorithms state was not true at first batch")
    report = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "status": "completed_bounded" if not validation_errors else "failed",
        "benchmark_kind": args.kind,
        "software_evidence_only": True,
        "hardware_or_clinical_claim": None,
        "harness": harness,
        "config": {
            "path": str(Path(args.config).expanduser().resolve()) if args.config else None,
            "resolved": config.to_dict(),
            "scientific_identity": config.scientific_identity(),
            "scientific_hash": runtime.sha256_json(config.scientific_identity()),
        },
        "source": {
            "snapshot": snapshot,
            "snapshot_path": str(source_snapshot_path),
            "snapshot_check": snapshot_check,
            "live_source_check": live_source_check,
            "activation": activation,
            "observed_import": observed_import_after,
            "observed_import_before": observed_import_before,
            "identity_before": source_before,
            "identity_after": source_after,
            "package_unchanged": source_before.get("package", {}).get("combined") == source_after.get("package", {}).get("combined"),
        },
        "rng": {
            "before": rng_before,
            "after": rng_after,
            "seed": config.seed,
            "note": "fixture uses a local NumPy Generator; trainer RNG streams are not replaced",
        },
        "device": {
            "requested": config.device,
            "resolved": runtime.device_identity(getattr(trainer, "device", None)),
            "gpu_stats_after": runtime.gpu_stats(getattr(trainer, "device", None)),
            "nvidia_smi_before_training": nvidia,
            "nvidia_smi_before_training_note": (
                "pre-training observation only; no measured GPU utilization claim"
            ),
            "gpu_utilization_claim": None,
        },
        "profiler": {
            "warmup_batches": args.warmup_batches,
            "measured_batches_requested": args.measured_batches,
            "measured_batches_captured": len(measured),
            "cprofile_enabled": bool(args.cprofile),
            "cprofile_path": str(Path(args.cprofile_output).expanduser().resolve()) if args.cprofile_output else (str((output / "profile.prof").resolve()) if args.cprofile else None),
            "overhead_note": "Timing wrappers and cProfile (when enabled) add overhead; cProfile runs are not ordinary throughput measurements",
            "instrumentation_scope": "ordinary minimal batch timer" if args.timing_mode == "ordinary" else "existing TimingAccumulator.stage plus process-local wrappers",
            "timing_mode": args.timing_mode,
            "fine_gradient_checks": args.timing_mode == "instrumented",
            "producer_loss_attribution_note": (
                "producer_loss wall/process time is measured as one call; individual "
                "contrastive, reconstruction, and equivariance sub-loss timings are "
                "unavailable without invasive scientific tracing. Per-term loss values "
                "are recorded separately."
            ),
            "decision_trace_note": (
                "per-unit decision hashes and compact numeric summaries are recorded; "
                "full regional float arrays are intentionally not serialized, so trace "
                "overhead is identical across ordinary and instrumented modes."
            ),
            "canonical_bulk_audit": {
                "enabled_before_hooks": instrumentation.bulk_audit_enabled_before,
                "enabled_during_hooks": instrumentation.bulk_audit_enabled_during,
                "enabled_after_restore": instrumentation.bulk_audit_enabled_after,
                "audit_hook_name": instrumentation.audit_hook_name,
            },
        },
        "timing": timing,
        "accounting": accounting,
        "units": {
            "manifest_id": manifest.get("manifest_id"),
            "manifest_hash": runtime.sha256_json(manifest) if manifest else None,
            "image_inventory": _image_inventory(manifest) if manifest else None,
            "split": "train",
            "unit_ids": unit_ids,
            "unit_count": len(unit_ids),
            "mask_inputs_used": False,
            "reference_inputs_used": False,
            "fixture": preparation.get("synthetic_fixture"),
        },
        "budget_contract": budget_contract,
        "initial_state": initial_state,
        "final_state": {
            "model_state_sha256": final_model_state_sha256,
            "rng": rng_after,
        },
        "sample_order_digest": _sample_order_digest(all_batches),
        "trainer": result,
        "batches": all_batches,
        "error": error_payload,
        "validation_errors": validation_errors,
        "command": list(sys.argv),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
    }
    _write_json(output / "profile_report.json", report)
    _write_hotspot_report(output, report)
    if args.reference_report:
        reference_report_path = Path(args.reference_report).expanduser().resolve()
        if not reference_report_path.is_file():
            raise ValueError(f"reference report not found: {reference_report_path}")
        reference_report = json.loads(reference_report_path.read_text(encoding="utf-8"))
        matched = _compare_reports(reference_report, report)
        _write_json(output / "matched_report.json", matched)
        report["matched_report"] = matched
        _write_json(output / "profile_report.json", report)
    return report


def _restore_source_import_environment(
    previous_path: list[str],
    previous_modules: dict[str, ModuleType],
) -> None:
    """Restore caller imports after a snapshot-backed in-process smoke run."""
    for name in list(sys.modules):
        if name == "self_audit_maskfree" or name.startswith("self_audit_maskfree."):
            del sys.modules[name]
    sys.modules.update(previous_modules)
    sys.path[:] = previous_path


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Run one profile while leaving the caller's module/import state intact."""
    previous_path = list(sys.path)
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "self_audit_maskfree" or name.startswith("self_audit_maskfree.")
    }
    try:
        return _run_profile(args)
    finally:
        _restore_source_import_environment(previous_path, previous_modules)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded profiling harness for canonical MaskfreeTrainer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="canonical maskfree150 YAML for a real image-root run")
    source.add_argument("--synthetic", action="store_true", help="generate and profile a deterministic image-only fixture")
    parser.add_argument("--synthetic-format", choices=("npy", "nifti"), default="npy")
    parser.add_argument(
        "--synthetic-root",
        "--fixture-root",
        dest="synthetic_root",
        type=Path,
        default=None,
        help="shared immutable file-backed fixture root for matched baseline/optimized runs",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=REPO_ROOT / "reports/maskfree150/local_baseline",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--warmup-batches", "--warmup", dest="warmup_batches", type=int, default=5)
    parser.add_argument("--measured-batches", "--measured", dest="measured_batches", type=int, default=20)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--kind", choices=("early", "baseline", "optimized"), default="baseline")
    parser.add_argument("--timing-mode", choices=TIMING_MODES, default="instrumented")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="immutable snapshot directory (or files/ child) imported before trainer construction",
    )
    parser.add_argument("--cprofile", "--profile", action="store_true")
    parser.add_argument("--cprofile-output", "--profile-output", dest="cprofile_output", type=Path, default=None)
    parser.add_argument("--reference-snapshot", type=Path, default=None)
    parser.add_argument("--reference-report", type=Path, default=None,
                        help="machine-readable profile_report.json used for matched_report.json")
    parser.add_argument("--seed", type=int, default=42, help="synthetic seed; ignored for real config scientific seed")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.warmup_batches < 0 or args.measured_batches <= 0:
        parser.error("warmup-batches must be non-negative and measured-batches must be positive")
    if args.warmup_batches + args.measured_batches >= PARTIAL_EPOCH_MAX_BATCHES:
        parser.error("warmup-batches + measured-batches must stay below the fixed epoch (128 batches)")
    if args.run_id is None:
        args.run_id = f"profile-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{os.getpid()}"
    try:
        report = run_profile(args)
    except Exception as exc:  # configuration/snapshot errors happen before trainer.run  # noqa: BLE001
        print(f"profile setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    captured = report.get("profiler", {}).get("measured_batches_captured", 0)
    matched_report = report.get("matched_report")
    summary = {
        "status": report.get("status"),
        "measured_batches": captured,
        "output": str(Path(args.output).expanduser().resolve()),
    }
    if args.reference_report is not None:
        summary["matched"] = bool(matched_report and matched_report.get("matched"))
    print(json.dumps(summary, sort_keys=True))
    return 0 if (
        report.get("status") == "completed_bounded"
        and (args.reference_report is None or bool(matched_report and matched_report.get("matched")))
    ) else 3


if __name__ == "__main__":
    raise SystemExit(main())
