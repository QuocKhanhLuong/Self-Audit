"""Immutable SHA-pinned manifests for reusable Q-3D T/Q/P stage outputs.

Stage checkpoints contain a complete engine snapshot for their own phase.
Manifests bind that snapshot to the exact data/protocol identity, the
arm-independent stage source configuration, and the manifests of its parents.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


STAGE_MANIFEST_SCHEMA = "self_audit.candidate_q3d.stage_manifest.v1"
_STAGES = ("T", "Q", "P")
_IDENTITY_REQUIRED = ("dataset", "protocol", "k", "seed")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = {
    "schema",
    "stage",
    "checkpoint",
    "identity",
    "source_config",
    "source_config_sha256",
    "dependencies",
    "content_hashes",
    "payload_sha256",
}
_OPTIONAL_MANIFEST_KEYS = {"teacher_quality"}
_TEACHER_QUALITY_KEYS = {
    "teacher_dev_fg_dice",
    "teacher_dev_per_class",
    "teacher_selected_dev_update",
    "teacher_terminal_dev_fg_dice",
    "teacher_terminal_dev_per_class",
    "dev_curve_still_improving",
}


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stage manifest values must be canonical JSON data: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file's exact bytes; used for checkpoints and manifest references."""

    file_path = Path(path).expanduser().resolve(strict=True)
    if not file_path.is_file():
        raise ValueError(f"stage manifest file reference is not a file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _normalize_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError("stage manifest identity must be a mapping")
    missing = [key for key in _IDENTITY_REQUIRED if key not in identity]
    if missing:
        raise ValueError(f"stage manifest identity is missing required keys: {missing}")
    dataset = identity.get("dataset")
    protocol = identity.get("protocol")
    k = identity.get("k")
    seed = identity.get("seed")
    if not isinstance(dataset, str) or dataset not in {"acdc", "mnms"}:
        raise ValueError("stage manifest identity dataset must be acdc or mnms")
    if not isinstance(protocol, str) or protocol not in {"A", "B"}:
        raise ValueError("stage manifest identity protocol must be A or B")
    if type(k) is not int or k < 1:
        raise ValueError("stage manifest identity K must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("stage manifest identity seed must be a nonnegative integer")
    normalized = dict(identity)
    _canonical_json(normalized)
    return normalized


def _normalize_source_config(source_config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source_config, Mapping):
        raise ValueError("stage manifest source_config must be a mapping")
    normalized = dict(source_config)
    _canonical_json(normalized)
    return normalized


def _teacher_quality_from_checkpoint(path: Path, *, stage: str) -> dict[str, Any] | None:
    """Extract reporting-only T quality from a completed engine snapshot."""

    if stage != "T":
        return None
    try:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch compatibility
            payload = torch.load(path, map_location="cpu")
    except Exception:  # noqa: BLE001 - legacy low-level fixtures may be raw bytes
        return None
    if not isinstance(payload, Mapping):
        return None
    # A published T stage checkpoint is a Q3D engine snapshot.  Such a
    # checkpoint must carry the complete teacher-quality receipt; silently
    # omitting it would make the immutable manifest look complete while the
    # required DEV/curve evidence was never sealed.  Raw-byte fixtures and
    # legacy non-stage checkpoints remain supported by returning None.
    is_published_t = stage == "T" and payload.get("stage_artifact_stage") == "T"
    missing = _TEACHER_QUALITY_KEYS - set(payload)
    if is_published_t and missing:
        raise ValueError(
            "published T checkpoint is missing teacher-quality fields: "
            f"{sorted(missing)}"
        )
    quality = {key: payload[key] for key in _TEACHER_QUALITY_KEYS if key in payload}
    if not quality:
        return None
    try:
        _canonical_json(quality)
    except ValueError as exc:
        raise ValueError("T teacher quality metadata must be canonical JSON data") from exc
    return quality


def _normalize_dependencies(
    stage: str,
    dependencies: Mapping[str, Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    if not isinstance(dependencies, Mapping) or not set(dependencies).issubset(_STAGES):
        raise ValueError("stage manifest dependencies must map only T/Q/P to manifest refs")
    if stage == "T" and dependencies:
        raise ValueError("T stage manifest cannot depend on another stage")
    if stage == "Q" and set(dependencies) != {"T"}:
        raise ValueError("Q stage manifest must depend on exactly the matching T manifest")
    if stage == "P" and ("T" not in dependencies or "J" in dependencies):
        raise ValueError("P stage manifest must depend on T and may also depend on Q")
    if "Q" in dependencies and "T" not in dependencies:
        raise ValueError("a Q dependency requires its T dependency")
    normalized: dict[str, dict[str, str]] = {}
    for dependency_stage, ref in dependencies.items():
        if dependency_stage not in _STAGES or not isinstance(ref, Mapping):
            raise ValueError("stage dependency ref must contain path and manifest-file SHA-256")
        ref_path = ref.get("path")
        if not isinstance(ref_path, str) or not Path(ref_path).is_absolute():
            raise ValueError(f"{dependency_stage} dependency path must be absolute")
        dependency_path = Path(ref_path).resolve(strict=True)
        if not dependency_path.is_file():
            raise ValueError(f"{dependency_stage} dependency manifest is not a file")
        expected_file_sha = _sha256(ref.get("sha256"), f"dependencies.{dependency_stage}.sha256")
        actual_file_sha = file_sha256(dependency_path)
        if actual_file_sha != expected_file_sha:
            raise ValueError(f"{dependency_stage} dependency manifest file SHA-256 mismatch")
        # Recursive load verifies the parent's stage payload, checkpoint bytes,
        # source config seal, identity, and its own dependency chain.
        parent = load_stage_manifest(dependency_path, identity=identity)
        if parent["stage"] != dependency_stage:
            raise ValueError(f"dependency key {dependency_stage} points at a {parent['stage']} manifest")
        normalized[dependency_stage] = {
            "path": str(dependency_path),
            "sha256": expected_file_sha,
        }
    return normalized


def _normalize_content_hashes(stage: str, content_hashes: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(content_hashes, Mapping):
        raise ValueError("stage manifest content_hashes must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in content_hashes.items():
        if not isinstance(key, str) or not key:
            raise ValueError("stage manifest content hash names must be nonempty strings")
        normalized[key] = _sha256(value, f"content_hashes.{key}")
    required = {
        "T": {"teacher_state_sha256"},
        "Q": {"teacher_state_sha256", "codebook_sha256", "sampling_phase_sha256"},
        "P": {"teacher_state_sha256", "cache_sha256"},
    }[stage]
    missing = required - set(normalized)
    if missing:
        raise ValueError(f"{stage} stage manifest is missing content hashes: {sorted(missing)}")
    return normalized


def _manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    return _canonical_sha256({key: value for key, value in manifest.items() if key != "payload_sha256"})


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable Q-3D stage manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_stage_manifest(
    path: str | Path,
    *,
    stage: str,
    checkpoint_path: str | Path,
    identity: Mapping[str, Any],
    source_config: Mapping[str, Any],
    dependencies: Mapping[str, Mapping[str, Any]],
    content_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Write a sealed, immutable manifest for one completed T/Q/P producer."""

    if stage not in _STAGES:
        raise ValueError("stage manifest stage must be T, Q, or P")
    normalized_identity = _normalize_identity(identity)
    normalized_source = _normalize_source_config(source_config)
    normalized_dependencies = _normalize_dependencies(stage, dependencies, normalized_identity)
    normalized_hashes = _normalize_content_hashes(stage, content_hashes)
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError(f"stage checkpoint must be a file: {checkpoint}")
    teacher_quality = _teacher_quality_from_checkpoint(checkpoint, stage=stage)
    normalized_hashes = dict(normalized_hashes)
    if teacher_quality is not None:
        normalized_hashes["teacher_quality_sha256"] = _canonical_sha256(teacher_quality)
    manifest: dict[str, Any] = {
        "schema": STAGE_MANIFEST_SCHEMA,
        "stage": stage,
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        "identity": normalized_identity,
        "source_config": normalized_source,
        "source_config_sha256": _canonical_sha256(normalized_source),
        "dependencies": normalized_dependencies,
        "content_hashes": normalized_hashes,
    }
    if teacher_quality is not None:
        manifest["teacher_quality"] = teacher_quality
    manifest["payload_sha256"] = _manifest_payload_sha256(manifest)
    manifest_path = Path(path).expanduser().resolve()
    _write_immutable_json(manifest_path, manifest)
    return manifest


def _load_stage_manifest(
    path: Path,
    *,
    identity: Mapping[str, Any],
    source_config: Mapping[str, Any] | None,
    visiting: set[Path],
) -> dict[str, Any]:
    manifest_path = path.resolve(strict=True)
    if manifest_path in visiting:
        raise ValueError(f"cyclic Q-3D stage manifest dependency: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError(f"Q-3D stage manifest must be a file: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Q-3D stage manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("Q-3D stage manifest has missing or unknown fields")
    manifest_keys = set(manifest)
    if manifest_keys - (_MANIFEST_KEYS | _OPTIONAL_MANIFEST_KEYS) \
            or (_MANIFEST_KEYS - manifest_keys):
        raise ValueError("Q-3D stage manifest has missing or unknown fields")
    if manifest.get("schema") != STAGE_MANIFEST_SCHEMA:
        raise ValueError("unsupported Q-3D stage manifest schema")
    stage = manifest.get("stage")
    if stage not in _STAGES:
        raise ValueError("Q-3D stage manifest stage must be T, Q, or P")
    actual_identity = _normalize_identity(manifest.get("identity", {}))
    if actual_identity != _normalize_identity(identity):
        raise ValueError("Q-3D stage manifest dataset/protocol/K/seed/data identity mismatch")
    actual_source = _normalize_source_config(manifest.get("source_config", {}))
    source_sha = _sha256(manifest.get("source_config_sha256"), "source_config_sha256")
    if _canonical_sha256(actual_source) != source_sha:
        raise ValueError("Q-3D stage manifest source_config SHA-256 mismatch")
    if source_config is not None and actual_source != _normalize_source_config(source_config):
        raise ValueError("Q-3D stage manifest source_config differs from the requested source")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"path", "sha256"}:
        raise ValueError("Q-3D stage manifest checkpoint ref is malformed")
    checkpoint_path = checkpoint.get("path")
    if not isinstance(checkpoint_path, str) or not Path(checkpoint_path).is_absolute():
        raise ValueError("Q-3D stage checkpoint path must be absolute")
    checkpoint_file = Path(checkpoint_path).resolve(strict=True)
    expected_checkpoint_sha = _sha256(checkpoint.get("sha256"), "checkpoint.sha256")
    if not checkpoint_file.is_file() or file_sha256(checkpoint_file) != expected_checkpoint_sha:
        raise ValueError("Q-3D stage checkpoint file SHA-256 mismatch")
    checkpoint_quality = _teacher_quality_from_checkpoint(checkpoint_file, stage=stage)
    dependencies = _normalize_dependencies_for_load(
        stage,
        manifest.get("dependencies"),
        actual_identity,
        visiting | {manifest_path},
    )
    hashes = _normalize_content_hashes(stage, manifest.get("content_hashes", {}))
    normalized = {
        "schema": STAGE_MANIFEST_SCHEMA,
        "stage": stage,
        "checkpoint": {"path": str(checkpoint_file), "sha256": expected_checkpoint_sha},
        "identity": actual_identity,
        "source_config": actual_source,
        "source_config_sha256": source_sha,
        "dependencies": dependencies,
        "content_hashes": hashes,
    }
    if "teacher_quality" in manifest:
        if stage != "T":
            raise ValueError("teacher_quality metadata is only valid on the T stage manifest")
        teacher_quality = manifest.get("teacher_quality")
        if not isinstance(teacher_quality, Mapping):
            raise ValueError("T teacher_quality must be a mapping")
        teacher_quality = dict(teacher_quality)
        unknown_quality = set(teacher_quality) - _TEACHER_QUALITY_KEYS
        if unknown_quality:
            raise ValueError(f"T teacher_quality has unknown fields: {sorted(unknown_quality)}")
        _canonical_json(teacher_quality)
        expected_quality_sha = hashes.get("teacher_quality_sha256")
        if expected_quality_sha != _canonical_sha256(teacher_quality):
            raise ValueError("T teacher_quality SHA-256 mismatch")
        if checkpoint_quality is not None and teacher_quality != checkpoint_quality:
            raise ValueError("T teacher_quality differs from the checkpoint payload")
        normalized["teacher_quality"] = teacher_quality
    elif checkpoint_quality is not None:
        # The checkpoint has a quality receipt, so its manifest must expose
        # and hash the same receipt for downstream consumers.
        raise ValueError("T stage manifest omits teacher_quality from a published checkpoint")
    payload_sha = _sha256(manifest.get("payload_sha256"), "payload_sha256")
    if _manifest_payload_sha256(normalized) != payload_sha:
        raise ValueError("Q-3D stage manifest canonical payload SHA-256 mismatch")
    normalized["payload_sha256"] = payload_sha
    return normalized


def _normalize_dependencies_for_load(
    stage: str,
    dependencies: Any,
    identity: Mapping[str, Any],
    visiting: set[Path],
) -> dict[str, dict[str, str]]:
    if not isinstance(dependencies, Mapping) or not set(dependencies).issubset(_STAGES):
        raise ValueError("Q-3D stage manifest dependencies are malformed")
    if stage == "T" and dependencies:
        raise ValueError("T stage manifest cannot depend on another stage")
    if stage == "Q" and set(dependencies) != {"T"}:
        raise ValueError("Q stage manifest must depend on exactly T")
    if stage == "P" and "T" not in dependencies:
        raise ValueError("P stage manifest must depend on T")
    if "Q" in dependencies and "T" not in dependencies:
        raise ValueError("a Q dependency requires its T dependency")
    normalized: dict[str, dict[str, str]] = {}
    for dependency_stage, ref in dependencies.items():
        if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
            raise ValueError(f"{dependency_stage} dependency manifest ref is malformed")
        ref_path = ref.get("path")
        if not isinstance(ref_path, str) or not Path(ref_path).is_absolute():
            raise ValueError(f"{dependency_stage} dependency manifest path must be absolute")
        dependency_path = Path(ref_path).resolve(strict=True)
        expected_sha = _sha256(ref.get("sha256"), f"dependencies.{dependency_stage}.sha256")
        if file_sha256(dependency_path) != expected_sha:
            raise ValueError(f"{dependency_stage} dependency manifest file SHA-256 mismatch")
        parent = _load_stage_manifest(
            dependency_path,
            identity=identity,
            source_config=None,
            visiting=visiting,
        )
        if parent["stage"] != dependency_stage:
            raise ValueError(f"dependency key {dependency_stage} points at a {parent['stage']} manifest")
        normalized[dependency_stage] = {"path": str(dependency_path), "sha256": expected_sha}
    return normalized


def load_stage_manifest(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    source_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a stage manifest, re-hashing its checkpoint and all parents."""

    manifest_path = Path(path).expanduser().resolve(strict=True)
    return _load_stage_manifest(
        manifest_path,
        identity=identity,
        source_config=source_config,
        visiting=set(),
    )


def stage_manifest_file_ref(path: str | Path) -> dict[str, str]:
    """Return the path and raw file SHA to store as a later manifest dependency."""

    manifest_path = Path(path).expanduser().resolve(strict=True)
    return {"path": str(manifest_path), "sha256": file_sha256(manifest_path)}
