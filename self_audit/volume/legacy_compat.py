"""Explicit compatibility checks for historical ``self_audit.semi_v2`` runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpoint import _file_sha256, source_sha256 as current_execution_source_sha256


def _normalise_module_path(name: str) -> str:
    value = str(name).replace("\\", "/")
    if value.endswith(".py"):
        return value
    return value.replace(".", "/") + ".py"


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def _trusted_canonical_file_map() -> dict[str, str]:
    """Read the pinned external canonical map, never the checkpoint manifest."""
    root = Path(__file__).resolve().parents[2]
    candidates = (
        root / "reports" / "teacher_memory_full_20261024" / "completed_20261025" / "ASSET_LINEAGE.json",
        root / "reports" / "native_volume_pilot" / "START.json",
    )
    for path in candidates:
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        mapping = raw.get("canonical_runtime_file_sha256") or raw.get("canonical_hashes_verified")
        if isinstance(mapping, Mapping) and mapping:
            return {_normalise_module_path(k): str(v).lower() for k, v in mapping.items()}
    raise ValueError("trusted canonical file map is unavailable; historical loading is disabled")


def _verify_canonical_files(expected: Mapping[str, str]) -> None:
    trusted = _trusted_canonical_file_map()
    normalised = {_normalise_module_path(k): str(v).lower() for k, v in expected.items()}
    if not normalised:
        raise ValueError("legacy compatibility requires a non-empty canonical module map")
    if set(normalised) != set(trusted):
        missing = sorted(set(trusted) - set(normalised))
        extra = sorted(set(normalised) - set(trusted))
        raise ValueError(f"legacy canonical module map is incomplete (missing={missing}, extra={extra})")
    if any(normalised[key] != trusted[key] for key in trusted):
        raise ValueError("legacy canonical module map differs from the pinned external map")
    root = Path(__file__).resolve().parents[2]
    for relative, expected_digest in trusted.items():
        # The current router is deliberately different.  Its historical hash
        # remains pinned in the original source_sha256; only canonical model,
        # data, loss and window files must still be the executed originals.
        if relative == "train.py":
            continue
        path = root / relative
        if not path.is_file() or _file_sha256(path) != expected_digest:
            raise ValueError(f"executed canonical module differs from pinned file: {relative}")


def verify_legacy_manifest(folder: str | Path, *, expected_manifest_sha256: str | None = None,
                           expected_source_sha256: str | None = None,
                           expected_data_sha256: str | None = None,
                           expected_payload_sha256: str | None = None,
                           expected_model: Mapping[str, Any] | None = None,
                           expected_canonical_modules: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Verify a historical checkpoint without mutating its source digest.

    This function is intentionally explicit: a caller must provide the source
    and data hashes it accepts.  No source-digest monkeypatch or silent schema
    migration is performed.
    """
    folder = Path(folder)
    manifest_path = folder / "MANIFEST.json"
    actual_manifest_sha256 = _file_sha256(manifest_path)
    if expected_manifest_sha256 is not None and actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("legacy manifest hash mismatch")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid legacy manifest") from exc
    if manifest.get("schema") != "self_audit.checkpoint.v2":
        raise ValueError("not a historical self_audit.checkpoint.v2 manifest")
    if not _is_sha256(expected_source_sha256):
        raise ValueError("legacy compatibility requires an explicit source_sha256")
    if manifest.get("source_sha256") != expected_source_sha256:
        raise ValueError("legacy checkpoint source hash is not the explicitly accepted hash")
    if not _is_sha256(expected_data_sha256):
        raise ValueError("legacy compatibility requires an explicit data_sha256")
    if manifest.get("data_sha256") != expected_data_sha256:
        raise ValueError("legacy checkpoint data hash mismatch")
    if not isinstance(expected_model, Mapping) or not expected_model:
        raise ValueError("legacy compatibility requires a complete non-empty model provenance map")
    actual_model = manifest.get("backbone")
    if not isinstance(actual_model, Mapping):
        actual_model = (manifest.get("config") or {}).get("model")
    if not isinstance(actual_model, Mapping) or set(expected_model) != set(actual_model):
        raise ValueError("legacy model provenance map is incomplete")
    for key, value in expected_model.items():
        if actual_model.get(key) != value:
            raise ValueError(f"legacy model provenance mismatch for {key}")
    if not isinstance(expected_canonical_modules, Mapping) or not expected_canonical_modules:
        raise ValueError("legacy compatibility requires a complete non-empty canonical module map")
    _verify_canonical_files(expected_canonical_modules)
    payload = folder / "weights.pt"
    actual_payload_sha256 = _file_sha256(payload)
    if actual_payload_sha256 != manifest.get("payload_sha256"):
        raise ValueError("legacy checkpoint payload hash mismatch")
    if not _is_sha256(expected_payload_sha256):
        raise ValueError("legacy compatibility requires an explicit payload_sha256")
    if actual_payload_sha256 != expected_payload_sha256:
        raise ValueError("legacy checkpoint payload hash is not the explicitly accepted hash")
    return manifest


def load_legacy_checkpoint(folder: str | Path, *, manifest_sha256: str, source_sha256: str,
                           data_sha256: str | None = None, payload_sha256: str | None = None,
                           expected_model: Mapping[str, Any] | None = None,
                           expected_canonical_modules: Mapping[str, str] | None = None,
                           device: str = "cpu", return_optimizers: bool = False):
    """Load verified historical student and EMA states without old digest gates.

    The caller supplies the accepted historical provenance explicitly.  The
    current native-volume source digest remains truthful in its own manifests;
    this compatibility path never compares an old payload against it and never
    mutates a historical manifest.
    """
    if expected_model is None or expected_canonical_modules is None:
        raise ValueError("historical loading requires explicit model and canonical-module hashes")
    manifest = verify_legacy_manifest(
        folder, expected_manifest_sha256=manifest_sha256,
        expected_source_sha256=source_sha256,
        expected_data_sha256=data_sha256,
        expected_payload_sha256=payload_sha256,
        expected_model=expected_model,
        expected_canonical_modules=expected_canonical_modules,
    )
    payload = torch.load(Path(folder) / "weights.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or "student" not in payload or "teacher" not in payload:
        raise ValueError("historical checkpoint must contain student and teacher states")
    if not isinstance(payload.get("optimizers"), list):
        raise ValueError("historical checkpoint must contain declared optimizer states")

    def build(state: Mapping[str, Any]):
        config = manifest.get("config")
        if not isinstance(config, Mapping) or not isinstance(config.get("model"), Mapping):
            raise ValueError("historical checkpoint lacks canonical model configuration")
        model_config = dict(config["model"])
        memory_bank = None
        teacher_memory = config.get("teacher_memory") or {}
        if isinstance(teacher_memory, Mapping) and teacher_memory.get("arm") in {"A2", "A3", "A4", "A5", "A6"}:
            from self_audit.teacher_memory import MemoryBank
            prefix = "teacher_memory.bank."
            keys = {prefix + "endpoints", prefix + "confidences", prefix + "valid"}
            if not keys.issubset(state):
                raise ValueError("historical memory arm is missing embedded bank buffers")
            memory_bank = MemoryBank(
                state[prefix + "endpoints"].numpy(),
                state[prefix + "confidences"].numpy(),
                state[prefix + "valid"].numpy(),
                {"provenance": {"source": "historical_checkpoint_embedded"}},
            )
        from self_audit.model import build_model
        memory_mode = "unpaired" if teacher_memory.get("arm") == "A2" else "paired"
        model, _ = build_model(
            model_config,
            synthetic=bool(manifest.get("synthetic", False)),
            window=config.get("window"),
            memory_bank=memory_bank,
            memory_mode=memory_mode,
            memory_tau=float(teacher_memory.get("tau", 0.2)),
        )
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()

    student = build(payload["student"])
    teacher = build(payload["teacher"])
    cfg = manifest.get("config")
    if not isinstance(cfg, Mapping) or not all(k in cfg for k in ("body_lr", "head_lr", "weight_decay")):
        raise ValueError("historical checkpoint lacks declared optimizer configuration")
    body = [p for n, p in student.named_parameters() if p.requires_grad and not n.startswith("head.")]
    head = [p for n, p in student.named_parameters() if p.requires_grad and n.startswith("head.")]
    optimizers: list[torch.optim.Optimizer] = []
    if body:
        optimizers.append(torch.optim.AdamW(body, lr=float(cfg["body_lr"]), weight_decay=float(cfg["weight_decay"])))
    if head:
        optimizers.append(torch.optim.SGD(head, lr=float(cfg.get("head_lr", 0.0)), momentum=0.0, weight_decay=0.0))
    if len(optimizers) != len(payload["optimizers"]):
        raise ValueError("historical optimizer state count differs from declared model groups")
    for optimizer, state in zip(optimizers, payload["optimizers"], strict=True):
        optimizer.load_state_dict(state)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if return_optimizers:
        return student, teacher, optimizers, manifest
    return student, teacher, manifest


__all__ = ["current_execution_source_sha256", "load_legacy_checkpoint", "verify_legacy_manifest"]
