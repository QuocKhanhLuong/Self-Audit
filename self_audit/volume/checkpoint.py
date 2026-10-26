"""Hash-pinned, resumable checkpoints for the native-volume trainer."""

from __future__ import annotations

import hashlib
import json
import os
import random
import copy
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import SCHEMA, config_sha256, validate_volume_config
from .engine import VolumeTrainer


def _source_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    canonical = [
        root / "train.py",
        root / "self_audit" / "__init__.py",
        root / "self_audit" / "model.py",
        root / "self_audit" / "window.py",
        root / "self_audit" / "data.py",
        root / "self_audit" / "losses.py",
        # Frozen native evaluation imports the historical metric helpers;
        # deployment/checkpoint identity must include that executed module.
        root / "self_audit" / "evaluation.py",
    ]
    canonical.extend((root / "self_audit" / "volume").rglob("*.py"))
    h = hashlib.sha256()
    seen: set[Path] = set()
    for path in sorted(canonical, key=lambda p: p.relative_to(root).as_posix()):
        if path in seen or not path.is_file() or any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        seen.add(path)
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


source_sha256 = _source_sha256


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_sha256_pin(value: Any, name: str) -> str:
    """Return a non-null, syntactically valid SHA-256 identity pin.

    Checkpoint provenance is an identity contract, not an optional annotation:
    accepting ``None`` or arbitrary strings would make resume silently skip the
    data comparison in ``train_volume``.
    """

    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"checkpoint data provenance {name} must be a non-null 64-hex SHA-256 pin")
    return value.lower()


def _validate_rng_state(state: Mapping[str, Any]) -> None:
    """Validate every RNG payload using isolated generators.

    ``random.setstate``/``np.random.set_state`` mutate process-global streams;
    use isolated generators first so malformed state fails before any supplied
    model, region cache, or global RNG can be changed.  CUDA state is rejected
    when CUDA is unavailable rather than being silently dropped on restore.
    """

    if not isinstance(state, Mapping):
        raise ValueError("volume checkpoint RNG state must be an object")
    required = {"python", "numpy", "torch", "cuda"}
    if not required.issubset(state):
        raise ValueError(f"volume checkpoint RNG state missing required fields: {sorted(required - set(state))}")

    try:
        random.Random().setstate(copy.deepcopy(state["python"]))
    except Exception as exc:  # random raises both TypeError and IndexError by version
        raise ValueError("invalid Python RNG state") from exc

    try:
        isolated_numpy = np.random.RandomState()
        isolated_numpy.set_state(copy.deepcopy(state["numpy"]))
    except Exception as exc:
        raise ValueError("invalid NumPy RNG state") from exc

    torch_rng = state["torch"]
    if not isinstance(torch_rng, torch.Tensor) or torch_rng.device.type != "cpu" or torch_rng.dtype != torch.uint8:
        raise ValueError("volume checkpoint torch RNG must be a CPU uint8 tensor")
    try:
        torch.Generator(device="cpu").set_state(torch_rng.detach().clone())
    except Exception as exc:
        raise ValueError("invalid Torch RNG state") from exc

    cuda_rng = state["cuda"]
    if cuda_rng is None:
        if torch.cuda.is_available():
            raise ValueError("CUDA checkpoint RNG state is missing while CUDA is available")
        return
    if not torch.cuda.is_available():
        raise ValueError("CUDA RNG state cannot be restored because CUDA is unavailable")
    if not isinstance(cuda_rng, list) or len(cuda_rng) != torch.cuda.device_count():
        raise ValueError("volume checkpoint CUDA RNG must contain one CPU tensor per CUDA device")
    for index, value in enumerate(cuda_rng):
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8:
            raise ValueError("volume checkpoint CUDA RNG must contain CPU uint8 tensors")
        try:
            torch.Generator(device=f"cuda:{index}").set_state(value.detach().clone())
        except Exception as exc:
            raise ValueError(f"invalid CUDA RNG state for device {index}") from exc


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rng_state() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def _restore_rng(state: Mapping[str, Any]) -> None:
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def save_volume_checkpoint(folder: str | Path, trainer: VolumeTrainer, config: Mapping[str, Any], *,
                           data_provenance: Mapping[str, Any] | None = None, step: int | None = None) -> str:
    """Persist a complete trainer state and return the manifest SHA-256.

    The destination must not already exist, making accidental overwrite of a
    prior checkpoint impossible.  All state needed for CPU resume is embedded
    in ``weights.pt``; manifests contain only JSON metadata and hashes.
    """
    c = validate_volume_config(config, allow_full_run=True)
    provenance = dict(data_provenance or {})
    required_provenance = {"synthetic", "manifest_sha256", "geometry_sha256"}
    if not required_provenance.issubset(provenance):
        raise ValueError(f"checkpoint data provenance missing required fields: {sorted(required_provenance - set(provenance))}")
    if type(provenance.get("synthetic")) is not bool:
        raise ValueError("checkpoint data provenance synthetic must be an exact bool")
    provenance["manifest_sha256"] = _require_sha256_pin(provenance.get("manifest_sha256"), "manifest_sha256")
    provenance["geometry_sha256"] = _require_sha256_pin(provenance.get("geometry_sha256"), "geometry_sha256")
    folder = Path(folder)
    if folder.exists():
        raise ValueError(f"checkpoint destination already exists: {folder}")
    folder.mkdir(parents=True)
    payload = {
        "student": trainer.model.state_dict(),
        "teacher": trainer.teacher.state_dict(),
        "optimizers": [o.state_dict() for o in trainer.optimizers],
        "schedulers": [s.state_dict() for s in trainer.schedulers],
        "scaler": trainer.scaler.state_dict() if trainer.scaler is not None else None,
        "trainer": trainer.state_dict(),
        "rng": _rng_state(),
    }
    payload_path = folder / "weights.pt"
    torch.save(payload, payload_path)
    manifest: dict[str, Any] = {
        "schema": "self_audit.volume.checkpoint.v1",
        "volume_schema": SCHEMA,
        "config": c,
        "config_sha256": config_sha256(c),
        "step": int(trainer.outer_step if step is None else step),
        "source_sha256": _source_sha256(),
        "payload_sha256": _file_sha256(payload_path),
        "data_provenance": provenance,
        "prediction_model": c["prediction_model"],
        "teacher_external": False,
        "ema_updates": trainer.ema_updates,
        "pseudo_updates": trainer.pseudo_updates,
        "supervised_updates": trainer.supervised_updates,
        "supervised_extra_updates": trainer.supervised_extra_updates,
        "optimizer_updates": trainer.optimizer_updates,
        "scheduler_total_steps": [int(getattr(scheduler, "total_steps", 0)) for scheduler in trainer.schedulers],
    }
    manifest_path = folder / "MANIFEST.json"
    _json_write(manifest_path, manifest)
    return _file_sha256(manifest_path)


def _validate_manifest(folder: Path, expected_sha256: str | None) -> dict[str, Any]:
    manifest_path = folder / "MANIFEST.json"
    if expected_sha256 is not None and _file_sha256(manifest_path) != expected_sha256:
        raise ValueError("volume checkpoint manifest hash mismatch")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid volume checkpoint manifest") from exc
    if manifest.get("schema") != "self_audit.volume.checkpoint.v1" or manifest.get("volume_schema") != SCHEMA:
        raise ValueError("unsupported volume checkpoint schema")
    if config_sha256(manifest.get("config", {})) != manifest.get("config_sha256"):
        raise ValueError("volume checkpoint config SHA-256 mismatch")
    payload_path = folder / "weights.pt"
    if _file_sha256(payload_path) != manifest.get("payload_sha256"):
        raise ValueError("volume checkpoint payload hash mismatch")
    if manifest.get("source_sha256") != _source_sha256():
        raise ValueError("executed volume source differs from checkpoint")
    provenance = manifest.get("data_provenance")
    required_provenance = {"synthetic", "manifest_sha256", "geometry_sha256"}
    if not isinstance(provenance, Mapping) or not required_provenance.issubset(provenance):
        raise ValueError(f"volume checkpoint data provenance missing required fields: {sorted(required_provenance - set(provenance or {}))}")
    if type(provenance.get("synthetic")) is not bool:
        raise ValueError("checkpoint data provenance synthetic must be an exact bool")
    _require_sha256_pin(provenance.get("manifest_sha256"), "manifest_sha256")
    _require_sha256_pin(provenance.get("geometry_sha256"), "geometry_sha256")
    return manifest


def load_volume_checkpoint(folder: str | Path, expected_sha256: str | None = None, *, model: Any | None = None,
                           device: str | torch.device = "cpu", restore_rng: bool = True,
                           region_manager: Any | None = None) -> tuple[VolumeTrainer, dict[str, Any]]:
    """Load a checkpoint and return ``(trainer, manifest)``.

    Supplying ``model`` avoids a dependency on model-owner code and is useful
    for exact resume tests.  Otherwise ``build_volume_model`` is called with
    the manifest configuration.
    """
    folder = Path(folder)
    manifest = _validate_manifest(folder, expected_sha256)
    config = validate_volume_config(manifest["config"], allow_full_run=True)
    payload = torch.load(folder / "weights.pt", map_location="cpu", weights_only=False)
    required_payload = {"student", "teacher", "optimizers", "schedulers", "scaler", "trainer", "rng"}
    if not isinstance(payload, Mapping):
        raise ValueError("volume checkpoint payload must be an object")
    if not required_payload.issubset(payload):
        raise ValueError(f"volume checkpoint payload missing required fields: {sorted(required_payload - set(payload))}")
    if not isinstance(payload["student"], Mapping) or not isinstance(payload["teacher"], Mapping):
        raise ValueError("volume checkpoint student/teacher states must be mappings")
    if not isinstance(payload["optimizers"], list) or not all(isinstance(s, Mapping) for s in payload["optimizers"]):
        raise ValueError("volume checkpoint optimizers must be a list of state mappings")
    if not isinstance(payload["schedulers"], list) or not all(isinstance(s, Mapping) for s in payload["schedulers"]):
        raise ValueError("volume checkpoint schedulers must be a list of state mappings")
    _validate_rng_state(payload["rng"])
    required_trainer = {"outer_step", "supervised_updates", "supervised_extra_updates", "pseudo_updates",
                        "optimizer_updates", "ema_updates", "skipped_supervised_updates", "skipped_pseudo_updates",
                        "history", "optimizer_metadata", "sampler_state", "region_state"}
    trainer_state = payload["trainer"]
    if not isinstance(trainer_state, Mapping) or not required_trainer.issubset(trainer_state):
        raise ValueError(f"volume checkpoint trainer state missing required fields: {sorted(required_trainer - set(trainer_state if isinstance(trainer_state, Mapping) else {}))}")
    counter_fields = ("outer_step", "supervised_updates", "supervised_extra_updates", "pseudo_updates",
                      "optimizer_updates", "ema_updates", "skipped_supervised_updates", "skipped_pseudo_updates")
    if any(type(trainer_state[key]) is not int or trainer_state[key] < 0 for key in counter_fields):
        raise ValueError("volume checkpoint trainer counters must be non-negative integers")
    if not isinstance(trainer_state["history"], list) or not isinstance(trainer_state["optimizer_metadata"], (Mapping, list)):
        raise ValueError("volume checkpoint trainer history must be a list and optimizer_metadata a mapping/list")
    if trainer_state.get("sampler_state") is None or not isinstance(trainer_state.get("sampler_state"), Mapping):
        raise ValueError("volume checkpoint trainer sampler_state is required")
    if config["regions"].get("mode") != "none" and (trainer_state.get("region_state") is None or not isinstance(trainer_state.get("region_state"), Mapping)):
        raise ValueError("region-enabled checkpoint is missing region cache state")
    if not isinstance(manifest.get("scheduler_total_steps"), list) or not all(type(x) is int and x >= 0 for x in manifest["scheduler_total_steps"]):
        raise ValueError("volume checkpoint scheduler horizon is required")
    for key in ("step", "ema_updates", "pseudo_updates", "supervised_updates", "supervised_extra_updates", "optimizer_updates"):
        if type(manifest.get(key)) is not int or manifest[key] < 0:
            raise ValueError(f"volume checkpoint manifest counter {key} is invalid")

    def restore_into(target: VolumeTrainer) -> None:
        target.model.load_state_dict(payload["student"], strict=True)
        target.teacher.load_state_dict(payload["teacher"], strict=True)
        if len(payload["optimizers"]) != len(target.optimizers):
            raise ValueError("checkpoint optimizer count differs from current recipe")
        for optimizer, state in zip(target.optimizers, payload["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        if len(payload["schedulers"]) != len(target.schedulers):
            raise ValueError("checkpoint scheduler count differs from current recipe")
        for scheduler, state in zip(target.schedulers, payload["schedulers"], strict=True):
            scheduler.load_state_dict(state)
        actual_horizon = [int(getattr(scheduler, "total_steps", 0)) for scheduler in target.schedulers]
        if list(manifest["scheduler_total_steps"]) != actual_horizon:
            raise ValueError("volume checkpoint scheduler horizon mismatch")
        if target.scaler is not None and payload["scaler"] is None:
            raise ValueError("AMP checkpoint is missing scaler state")
        if target.scaler is None and payload["scaler"] is not None:
            raise ValueError("FP32 checkpoint must not contain scaler state")
        if target.scaler is not None:
            if not isinstance(payload["scaler"], Mapping):
                raise ValueError("AMP checkpoint scaler state must be a mapping")
            target.scaler.load_state_dict(payload["scaler"])
        target.load_state_dict(trainer_state)
        for key in ("ema_updates", "pseudo_updates", "supervised_updates", "supervised_extra_updates", "optimizer_updates"):
            if manifest[key] != int(getattr(target, key)):
                raise ValueError(f"volume checkpoint manifest counter mismatch: {key}")
        if manifest["step"] != int(target.outer_step):
            raise ValueError("volume checkpoint manifest step mismatch")

    if model is None:
        from .models import build_volume_model  # type: ignore
        # _validate_manifest already enforces an exact bool; preserve that
        # identity rather than coercing malformed values during model build.
        synthetic = manifest["data_provenance"]["synthetic"]
        model, _ = build_volume_model(config, synthetic=synthetic)
    # Validate on a deep copy first: malformed optimizer/cache/RNG payloads
    # must not mutate a caller-supplied model or region manager.
    candidate_model = copy.deepcopy(model).to(device)
    candidate_regions = copy.deepcopy(region_manager) if region_manager is not None else None
    candidate = VolumeTrainer(candidate_model, config, region_manager=candidate_regions)
    restore_into(candidate)
    if model is not candidate_model:
        model.to(device)
        trainer = VolumeTrainer(model, config, region_manager=region_manager)
        restore_into(trainer)
    else:
        trainer = candidate
    if restore_rng:
        _restore_rng(payload["rng"])
    return trainer, manifest


# Short aliases make the volume package convenient to embed while retaining a
# distinct implementation from ``self_audit.engine.load_checkpoint``.
save_checkpoint = save_volume_checkpoint
load_checkpoint = load_volume_checkpoint


__all__ = ["load_checkpoint", "load_volume_checkpoint", "save_checkpoint", "save_volume_checkpoint", "source_sha256"]
