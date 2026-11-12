#!/usr/bin/env python3
"""Build a protocol-bound, SHA-sealed Q-3D physiology cache for TRAIN U.

For real runs, load a verified Q-3D research checkpoint after stage T (and Q
for VQ arms), before J. Online O1 refreshes are managed by the trainer. The synthetic-only
default builds the repository's tiny smoke teacher and seals its exact state
hash.  This command never trains, reads a U mask, or accesses VAL/TEST data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.candidate_q3d import CHECKPOINT_SCHEMA  # noqa: E402

from self_audit.candidate_a3d.data import CandidateA3DData  # noqa: E402
from self_audit.candidate_q3d.physiology import (  # noqa: E402
    compute_physiology_cache_rows,
    file_sha256,
    save_physiology_cache,
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{name} must be a nested mapping")
    return value


def _resolve_path(raw: str, *, base: Path = ROOT) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _manifest_identity(config: Mapping[str, Any], config_path: Path) -> tuple[Path, str, bool]:
    dataset = _mapping(config.get("dataset"), "dataset")
    manifests = _mapping(dataset.get("manifests"), "dataset.manifests")
    record = _mapping(manifests.get("train"), "dataset.manifests.train")
    raw_path = record.get("path")
    manifest_sha = record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(manifest_sha, str):
        raise ValueError("dataset.manifests.train must declare path and SHA-256")
    manifest_path = _resolve_path(raw_path, base=config_path.parent)
    actual = file_sha256(manifest_path)
    if actual != manifest_sha:
        raise ValueError("configured TRAIN manifest SHA-256 mismatch")
    runtime = _mapping(config.get("runtime", {}), "runtime")
    synthetic = runtime.get("synthetic", False)
    if type(synthetic) is not bool:
        raise ValueError("runtime.synthetic must be a boolean")
    return manifest_path, manifest_sha, synthetic


def _teacher_checkpoint(
    config_path: Path,
    *,
    synthetic: bool,
    override: str | None,
    declared_sha256: str | None,
):
    if synthetic and override is None:
        if declared_sha256 is not None:
            raise ValueError("a synthetic teacher state hash is computed after construction")
        return None, None
    if not isinstance(override, str) or not override:
        raise ValueError("real physiology caching requires the frozen post-T teacher checkpoint path")
    if not isinstance(declared_sha256, str) or len(declared_sha256) != 64:
        raise ValueError("real physiology caching requires --teacher-sha256 for the post-T snapshot")
    path = _resolve_path(override, base=config_path.parent)
    actual_sha = file_sha256(path)
    if actual_sha != declared_sha256:
        raise ValueError("teacher post-T checkpoint differs from the requested SHA-256")
    return path, actual_sha


def _codebook_identity(config: Mapping[str, Any], override: str | None) -> str | None:
    arm = str(config.get("arm", "V1")).upper()
    tokenizer = _mapping(config.get("tokenizer", {}), "tokenizer")
    value = override or tokenizer.get("codebook_sha256")
    if arm in {"C0", "C1"}:
        if value is not None:
            raise ValueError("continuous Q-3D arms do not bind a codebook SHA")
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("VQ physiology cache requires tokenizer.codebook_sha256")
    return value


def _load_factory(specification: str):
    if ":" not in specification:
        raise ValueError("teacher factory must use module:function syntax")
    module_name, function_name = specification.split(":", 1)
    if not module_name or not function_name:
        raise ValueError("teacher factory must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise ValueError(f"teacher factory {specification!r} is not callable")
    return factory


def _model_state_sha256(model: torch.nn.Module) -> str:
    # The cache contract uses the same model-state digest as the trainer;
    # checkpoint-file bytes have a separate digest in the CLI report.
    from self_audit.candidate_q3d.engine import _state_sha256

    return _state_sha256(model)


def _source_identity(data: CandidateA3DData, protocol: str, manifest_sha: str) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "dataset": data.manifest.get("dataset"),
        "student_manifest_sha256": manifest_sha,
        "unlabeled_patients": [
            {"id": record["id"], "image_sha256": record["image_sha256"]}
            for record in data.unlabeled
        ],
    }


def _load_post_t_teacher(config, checkpoint_path, expected_sha256, device):
    """Restore a complete teacher from a safe, protocol-bound research snapshot."""
    from self_audit.candidate_q3d.config import validate_config
    from self_audit.candidate_q3d.engine import (
        _build_teacher, _build_tokenizer, _data_identity_without_labels, _load_data, _object_sha256,
    )

    synthetic = bool(config.get("runtime", {}).get("synthetic", False))
    normalized = validate_config(config, real_run=not synthetic)
    if file_sha256(checkpoint_path) != expected_sha256:
        raise ValueError("teacher post-T checkpoint SHA-256 mismatch")
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    sidecar_fields = sidecar.read_text().split() if sidecar.is_file() else []
    if not sidecar_fields or sidecar_fields[0] != expected_sha256:
        raise ValueError("teacher post-T checkpoint requires its matching SHA-256 sidecar")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("teacher post-T checkpoint must use the Q-3D research schema")
    if payload.get("config_sha256") != _object_sha256(normalized):
        raise ValueError("teacher post-T checkpoint config identity mismatch")
    data = _load_data(normalized, None)
    if payload.get("data_identity") != _data_identity_without_labels(data):
        raise ValueError("teacher post-T checkpoint TRAIN protocol/cohort mismatch")
    completed = payload.get("completed_stages")
    if not isinstance(completed, list) or "T" not in completed:
        raise ValueError("teacher checkpoint has not completed stage T")
    if "J" in completed or payload.get("stage") == "J":
        raise ValueError("physiology cache requires a post-T teacher snapshot before joint stage J")
    if not isinstance(payload.get("teacher_state"), Mapping):
        raise ValueError("teacher checkpoint contains no teacher state")
    teacher, _ = _build_teacher(normalized, synthetic=synthetic)
    teacher.load_state_dict(payload["teacher_state"], strict=True)
    teacher.to(device).eval().requires_grad_(False)
    expected_state_sha = payload.get("teacher_best_sha256")
    if not isinstance(expected_state_sha, str) or expected_state_sha != _model_state_sha256(teacher):
        raise ValueError("teacher checkpoint is not the frozen VAL-selected T snapshot")
    teacher._q3d_cache_codebook_sha256 = None
    if normalized["tokenizer"]["enabled"]:
        if "Q" not in completed or not isinstance(payload.get("tokenizer_state"), Mapping):
            raise ValueError("VQ cache requires completed Q and its frozen tokenizer")
        tokenizer = _build_tokenizer(normalized, teacher, device)
        tokenizer.load_state_dict(payload["tokenizer_state"], strict=True)
        metadata = tokenizer.artifact_metadata()
        if not metadata["initialized"] or not metadata["frozen"]:
            raise ValueError("VQ cache requires an initialized, frozen tokenizer")
        teacher._q3d_cache_codebook_sha256 = metadata["sha256"]
    return teacher


def build_cache(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    teacher_factory: str | None,
    output_path: Path,
    device: str = "cpu",
    teacher_checkpoint: str | None = None,
    teacher_sha256: str | None = None,
    codebook_sha256: str | None = None,
    refreshed_epoch: int = 0,
) -> dict[str, Any]:
    """Load configured inputs, run frozen inference on U, and seal the cache."""

    protocol_section = _mapping(config.get("protocol"), "protocol")
    protocol = protocol_section.get("variant")
    if protocol not in {"A", "B"}:
        raise ValueError("protocol.variant must be A or B")
    manifest_path, manifest_sha, synthetic = _manifest_identity(config, config_path)
    checkpoint_path, checkpoint_sha = _teacher_checkpoint(
        config_path, synthetic=synthetic, override=teacher_checkpoint,
        declared_sha256=teacher_sha256,
    )
    if checkpoint_path is not None and refreshed_epoch != 0:
        raise ValueError("the standalone pre-J cache uses epoch 0; O1 refreshes are managed by the trainer")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable; no silent CPU fallback")

    if teacher_factory is not None:
        if not synthetic:
            raise ValueError("custom teacher factories are restricted to synthetic software fixtures")
        if checkpoint_path is not None:
            raise ValueError("a checkpoint must use the verified post-T loader, not a custom fixture factory")
        teacher = _load_factory(teacher_factory)(
            config=config,
            checkpoint_path=checkpoint_path,
            expected_sha256=checkpoint_sha,
            device=torch_device,
        )
    elif checkpoint_path is not None:
        teacher = _load_post_t_teacher(config, checkpoint_path, checkpoint_sha, torch_device)
    elif synthetic:
        from self_audit.candidate_q3d.engine import _build_teacher

        teacher, _ = _build_teacher(config, synthetic=True)
    else:
        raise ValueError("real cache construction requires a verified post-T teacher checkpoint")
    if not isinstance(teacher, torch.nn.Module):
        raise TypeError("teacher factory must return torch.nn.Module")
    snapshot_codebook_sha = getattr(teacher, "_q3d_cache_codebook_sha256", None)
    if snapshot_codebook_sha is not None and codebook_sha256 is not None and snapshot_codebook_sha != codebook_sha256:
        raise ValueError("requested codebook SHA differs from the frozen checkpoint tokenizer")
    codebook_sha = _codebook_identity(config, snapshot_codebook_sha or codebook_sha256)
    teacher.to(torch_device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model_sha = _model_state_sha256(teacher)

    data = CandidateA3DData(manifest_path, manifest_sha, synthetic=synthetic)
    if not data.uids:
        raise ValueError("TRAIN manifest has no image-only U cohort for physiology caching")
    physiology_config = config.get("physiology", config.get("physio", {}))
    if not isinstance(physiology_config, Mapping) or not physiology_config.get("enabled", True):
        raise ValueError("this arm/config does not declare physiology weighting enabled")
    rows = compute_physiology_cache_rows(
        teacher, data, teacher_sha256=model_sha,
        codebook_sha256=codebook_sha, protocol=protocol,
        config=config, device=torch_device,
    )
    source_identity = _source_identity(data, protocol, manifest_sha)
    cache_sha = save_physiology_cache(
        output_path, rows, teacher_sha256=model_sha,
        codebook_sha256=codebook_sha, protocol=protocol,
        refreshed_epoch=refreshed_epoch,
        config=config,
        source_identity=source_identity,
    )
    if synthetic and checkpoint_path is None:
        snapshot_kind = "synthetic_initialized_state"
    elif synthetic:
        snapshot_kind = "synthetic_verified_checkpoint"
    else:
        snapshot_kind = "verified_post_T_checkpoint"
    return {
        "status": "SOFTWARE_CACHE_WRITTEN",
        "protocol": protocol,
        "teacher_sha256": model_sha,
        "teacher_state_sha256": model_sha,
        "teacher_checkpoint_file_sha256": checkpoint_sha,
        "teacher_snapshot_kind": snapshot_kind,
        "codebook_sha256": codebook_sha,
        "source_identity": source_identity,
        "source_identity_sha256": hashlib.sha256(
            json.dumps(source_identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "train_manifest_sha256": manifest_sha,
        "cache_path": str(output_path.resolve()),
        "cache_sha256": cache_sha,
        "patient_count": len(rows),
        "patient_ids": [row["id"] for row in rows],
        "synthetic_fixture": synthetic,
        "device": str(torch_device),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--teacher-factory", help="module:function synthetic-only teacher fixture")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", help="Q-3D research snapshot after T (and Q for VQ arms), before J")
    parser.add_argument("--teacher-sha256", help="required SHA-256 for the frozen post-T checkpoint")
    parser.add_argument("--codebook-sha256", help="optional assertion against snapshot C0; synthetic fixture VQ requires it")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--refreshed-epoch", type=int, default=0)
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    from self_audit.candidate_q3d.config import load_config

    config = load_config(config_path)
    summary = build_cache(
        config,
        config_path=config_path,
        teacher_factory=args.teacher_factory,
        output_path=args.output.expanduser().resolve(),
        device=args.device,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_sha256=args.teacher_sha256,
        codebook_sha256=args.codebook_sha256,
        refreshed_epoch=args.refreshed_epoch,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
