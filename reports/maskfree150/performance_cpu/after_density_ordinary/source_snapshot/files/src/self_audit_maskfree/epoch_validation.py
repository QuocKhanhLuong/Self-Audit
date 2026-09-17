"""Image-only epoch snapshots and a report-only, isolated reference process.

This module never imports a reference reader or opens annotation files. Both
students predict every development image before a complete native-grid freeze
permits the child evaluator to read references. Its scalar report is used only
by terminal/reporting code, never by an optimizer, scheduler or checkpoint rule.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

import torch

from . import runtime
from .data.cache import manifest_digest as source_manifest_digest
from .progress import current_progress

ARMS = ("student_no_audit", "student_audited")
METHODS = tuple(f"{arm}_full_input" for arm in ARMS)
SCHEMA = "maskfree150.epoch_validation.v1"
REFERENCE_TIMEOUT_SECONDS = 3600


def _source_cache_snapshot() -> dict[str, int] | None:
    """Return process-local source-cache counters for validation receipts."""
    try:
        from .data.cache import default_source_cache

        return default_source_cache().stats()
    except Exception:  # pragma: no cover - defensive for partial component imports
        return None


def _source_cache_delta(before: dict[str, int] | None) -> dict[str, Any] | None:
    """Expose cumulative cache state and this validation call's counter delta."""
    after = _source_cache_snapshot()
    if after is None:
        return None
    if before is None:
        return {"after": after, "delta": None}
    counters = (
        "hits",
        "misses",
        "evictions",
        "invalidations",
        "loads",
        "loaded_bytes",
        "uncacheable",
    )
    return {
        "after": after,
        "delta": {name: int(after[name]) - int(before.get(name, 0)) for name in counters},
    }


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _identity(trainer: Any, epoch: int) -> dict[str, Any]:
    return {
        "epoch": epoch, "run_id": trainer.run_id,
        "manifest_hash": runtime.sha256_json(trainer.manifest),
        "scientific_hash": runtime.sha256_json(trainer.config.scientific_identity()),
        "source_hash": trainer.source["package"]["combined"],
        "model_hashes": {arm: _state_hash(trainer.models[arm].state_dict()) for arm in ARMS},
    }


def _unavailable(epoch: int, reason: str, *, status: str = "UNAVAILABLE") -> dict[str, Any]:
    return {"schema_version": SCHEMA, "epoch": epoch, "status": status,
            "available": False, "reason": reason, "freeze_id": None, "students": {}}


def _save_receipt(path: Path, result: dict[str, Any]) -> None:
    runtime.atomic_write_json(path, result)
    failure_path = path.parent / "last_failure.json"
    if result.get("status") != "FAILED" and failure_path.exists():
        failure_path.rename(path.parent / f"recovered_failure_{time.time_ns()}.json")


def _reference_report_valid(receipt: dict[str, Any]) -> bool:
    try:
        path = Path(receipt["reference_report"])
        return path.is_file() and runtime.sha256_file(path) == receipt["reference_report_sha256"]
    except (KeyError, TypeError, OSError):
        return False


def _run_reference(freeze_path: Path, image_manifest: Path, output: Path,
                   reference_config: str | None) -> dict[str, Any]:
    """Only the subprocess may open a reference config, CSV metadata or masks."""
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_maskfree_epoch.py"
    command = [sys.executable, str(script), "--frozen-manifest", str(freeze_path),
               "--image-manifest", str(image_manifest), "--output", str(output)]
    if reference_config is not None:
        command.extend(["--reference-config", str(Path(reference_config).expanduser().absolute())])
    environment = dict(os.environ)
    # Evaluation is CPU-only; it must not initialise another CUDA context.
    environment["CUDA_VISIBLE_DEVICES"] = ""
    log_path = output / "reference_process.log"
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=environment, timeout=REFERENCE_TIMEOUT_SECONDS, check=False)
    report_path = output / "epoch_validation.json"
    if not report_path.is_file():
        raise RuntimeError(f"reference evaluator exited {result.returncode} without a report; {log_path}")
    report = json.loads(report_path.read_text())
    if result.returncode != 0:
        raise RuntimeError(f"reference evaluator exited {result.returncode}: "
                           f"{report.get('reason') or log_path}")
    if report.get("status") not in {"COMPLETED", "UNAVAILABLE"}:
        raise RuntimeError(f"invalid reference evaluator status: {report.get('status')!r}")
    return report


def observe_epoch(trainer: Any, epoch: int) -> dict[str, Any]:
    """Observe a saved epoch boundary, preserving RNG, weights and module modes.

    The trainer saves its resume checkpoint BEFORE calling this function. An
    interruption can therefore retry the same epoch snapshot before advancing
    training. Frozen exports are reused and validated; rolling ``last.pt`` is
    never a freeze dependency. Reference results stay outside checkpoint history.
    """
    started = time.monotonic()
    progress = current_progress()
    root = trainer.paths.root / "validation" / f"epoch_{epoch + 1:04d}"
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "receipt.json"
    source_cache_before = _source_cache_snapshot()
    reference_request = {"config_path": str(Path(trainer.config.epoch_reference_config).expanduser().absolute())
                         if trainer.config.epoch_reference_config else None}
    records = [r for r in trainer.manifest.get("records", []) if r.get("split") == "dev"]
    unit_ids = sorted(str(r["unit_id"]) for r in records)
    rng = runtime.capture_rng_state()
    modes = [(module, module.training) for arm in ARMS for module in trainer.models[arm].modules()]
    identity: dict[str, Any] | None = None
    snapshot_owned = False
    freeze_path = root / "exports" / "freeze_manifest.json"
    saved = None
    try:
        identity = _identity(trainer, epoch)
        owner_path = root / "snapshot_identity.json"
        if owner_path.exists():
            if json.loads(owner_path.read_text()) != identity:
                raise RuntimeError("epoch snapshot identity differs from this saved model boundary")
        else:
            runtime.atomic_write_json(owner_path, identity)
        snapshot_owned = True

        if receipt_path.is_file():
            saved = json.loads(receipt_path.read_text())
            if saved.get("snapshot_identity") != identity:
                raise RuntimeError("validation receipt identity mismatch")
            if (saved.get("status") == "COMPLETED" and saved.get("reference_request") == reference_request and
                    _reference_report_valid(saved)):
                trainer.components.validate_freeze(json.loads(freeze_path.read_text()), require_complete=True)
                saved = dict(saved)
                saved["source_cache"] = _source_cache_delta(source_cache_before)
                _save_receipt(receipt_path, saved)
                return {**saved, "cached": True}

        if not records:
            result = _unavailable(epoch, "image-only patient split contains no development images")
            result.update(snapshot_identity=identity, elapsed_seconds=time.monotonic() - started,
                          source_cache=_source_cache_delta(source_cache_before))
            _save_receipt(receipt_path, result)
            return result
        if len(unit_ids) != len(set(unit_ids)):
            raise RuntimeError("duplicate validation unit IDs")
        train_patients = {r["patient_id"] for r in trainer.manifest["records"] if r.get("split") == "train"}
        if train_patients & {r["patient_id"] for r in records}:
            raise RuntimeError("development patients overlap training patients")

        batch_size = trainer.config.batch_size
        batches = math.ceil(len(unit_ids) / batch_size)
        progress.event("validation.start", epoch=epoch + 1, global_epoch=epoch, total_epochs=trainer.config.total_epochs,
                       units=len(unit_ids), batches=batches, physical_batch=batch_size)
        inference_started = time.monotonic()
        image_manifest_path = root / "image_manifest.json"
        if not image_manifest_path.exists():
            runtime.atomic_write_json(image_manifest_path, trainer.manifest)
        elif json.loads(image_manifest_path.read_text()) != trainer.manifest:
            raise RuntimeError("epoch image manifest changed")

        if not freeze_path.exists():
            # The development cohort and manifest are frozen for this epoch
            # snapshot.  Compute the complete source-manifest digest once and
            # pass it only to canonical loaders that explicitly accept the
            # private override.  Custom test/deployment components retain their
            # historical call signature and therefore stay byte-for-byte
            # compatible with the validation contract.
            load_full_input = trainer.components.load_full_input
            try:
                load_parameters = inspect.signature(load_full_input).parameters.values()
                accepts_manifest_digest = any(
                    parameter.name == "_manifest_digest_value"
                    and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
                    for parameter in load_parameters
                )
            except (TypeError, ValueError):
                accepts_manifest_digest = False
            load_kwargs = {"image_size": trainer.config.image_size}
            if accepts_manifest_digest:
                load_kwargs["_manifest_digest_value"] = source_manifest_digest(trainer.manifest)
            checkpoints = {"validation_image_manifest": str(image_manifest_path)}
            for arm in ARMS:
                trainer.models[arm].eval()
                path = root / "checkpoints" / f"{arm}.pt"
                if path.exists():
                    saved = torch.load(path, map_location="cpu", weights_only=False)
                    if saved.get("snapshot_identity") != identity or _state_hash(saved["model"]) != identity["model_hashes"][arm]:
                        raise RuntimeError(f"{arm} epoch checkpoint identity mismatch")
                else:
                    runtime.atomic_save_torch(path, {
                        "model": {k: v.detach().cpu().clone() for k, v in trainer.models[arm].state_dict().items()},
                        "snapshot_identity": identity, "arm": arm, "epoch": epoch,
                    })
                checkpoints[arm] = str(path)
            entries = []
            with torch.inference_mode():
                for offset in range(0, len(unit_ids), batch_size):
                    ids = unit_ids[offset:offset + batch_size]
                    with progress.stage("validation.images", operation="load validation images", batch=offset // batch_size + 1):
                        loaded = [load_full_input(trainer.manifest, unit_id, **load_kwargs) for unit_id in ids]
                        inputs = torch.stack([item[0] for item in loaded]).to(trainer.device).float()
                    with progress.stage("validation.predict", operation="predict both students"):
                        probabilities = {
                            arm: torch.softmax(trainer.models[arm](inputs).float(), dim=1).cpu() for arm in ARMS
                        }
                    with progress.stage("validation.export", operation="write native prediction shards"):
                        for index, (_, record) in enumerate(loaded):
                            for arm in ARMS:
                                probs = probabilities[arm][index]
                                entries.append(trainer.components.export_prediction(
                                    root / "exports", record=record, prediction_name=f"{arm}_full_input",
                                    labels=probs.argmax(0).long(), probabilities=probs,
                                    validity=torch.ones_like(probs[0]), alternatives=[],
                                    checkpoint_id=checkpoints[arm], version=trainer.pseudo_label_version(epoch)))
                    progress.event("validation.batch", epoch=epoch + 1, global_epoch=epoch, batch=offset // batch_size + 1,
                                   batches=batches, units_completed=offset + len(ids))
            with progress.stage("validation.freeze", operation="assemble and freeze both students"):
                frozen = trainer.components.freeze_predictions(
                    root / "exports", entries, checkpoints, dataset=trainer.config.dataset,
                    protocol=trainer.manifest["resolved_protocol"], epoch=epoch,
                    required_methods=METHODS, required_unit_ids=unit_ids)
        else:
            frozen = json.loads(freeze_path.read_text())
        trainer.components.validate_freeze(frozen, require_complete=True)
        if set(frozen.get("required_methods", [])) != set(METHODS):
            # Exporter stores the authoritative method set in completeness.
            declared = frozen.get("completeness", {}).get("required", [])
            if set(declared) != set(METHODS):
                raise RuntimeError("epoch freeze does not declare exactly the two student arms")
        inference_seconds = time.monotonic() - inference_started
        reference_started = time.monotonic()
        reference_output = root / "reference"
        if saved is not None:
            attempt = str(time.time_ns())
            runtime.atomic_write_json(root / "receipt_history" / f"{attempt}.json", saved)
            reference_output = reference_output / f"attempt_{attempt}"
        with progress.stage("validation.reference", operation="load reference masks and score frozen volumes"):
            result = _run_reference(freeze_path, image_manifest_path, reference_output,
                                    trainer.config.epoch_reference_config)
        if result.get("freeze_id") != frozen["freeze_id"] or result.get("epoch") != epoch:
            raise RuntimeError("reference report does not match the frozen epoch")
        result.update(
            snapshot_identity=identity, inference_seconds=inference_seconds,
            reference_seconds=time.monotonic() - reference_started,
            elapsed_seconds=time.monotonic() - started,
            physical_batch=batch_size, accumulation_steps=1,
            validation_view="full_input", split="dev", reference_report=str(reference_output / "reference_metrics.json"),
            reference_request=reference_request,
            freeze_manifest=str(freeze_path), checkpoint_selection="none", cached=False,
            source_cache=_source_cache_delta(source_cache_before),
        )
        if result.get("status") == "COMPLETED":
            result["reference_report_sha256"] = runtime.sha256_file(result["reference_report"])
        _save_receipt(receipt_path, result)
        return result
    except Exception as error:
        result = _unavailable(epoch, f"{type(error).__name__}: {error}", status="FAILED")
        result.update(snapshot_identity=identity, elapsed_seconds=time.monotonic() - started,
                      traceback=traceback.format_exc(), source_cache=_source_cache_delta(source_cache_before))
        # An integrity failure must never overwrite the prior valid receipt.
        runtime.atomic_write_json(root / "last_failure.json", result)
        if snapshot_owned and not receipt_path.exists():
            runtime.atomic_write_json(receipt_path, result)
        return result
    finally:
        for module, training in modes:
            module.training = training
        runtime.restore_rng_state(rng)
        progress.event("validation.end", epoch=epoch + 1, global_epoch=epoch)


def validation_status(run_root: Path, *, enabled: bool, epochs: int) -> dict[str, Any]:
    """Small report status, separate from training/checkpoint completion."""
    counts = {name: 0 for name in ("COMPLETED", "UNAVAILABLE", "FAILED", "NOT_STARTED")}
    for epoch in range(epochs):
        path = run_root / "validation" / f"epoch_{epoch + 1:04d}" / "receipt.json"
        if (path.parent / "last_failure.json").is_file():
            status = "FAILED"
        else:
            try:
                receipt = json.loads(path.read_text()) if path.is_file() else {}
                status = receipt.get("status", "FAILED") if path.is_file() else "NOT_STARTED"
                if status == "COMPLETED" and not _reference_report_valid(receipt):
                    status = "FAILED"
            except (OSError, ValueError, TypeError, AttributeError):
                status = "FAILED"
        counts[status if status in counts else "FAILED"] += 1
    status = ("NOT_STARTED" if not enabled or not epochs else
              "FAILED" if counts["FAILED"] else
              "COMPLETED" if counts["COMPLETED"] == epochs else "UNAVAILABLE")
    return {"enabled": enabled, "status": status, "epochs_expected": epochs,
            "epoch_counts": counts, "split": "dev", "checkpoint_selection": "none",
            "reports": str(run_root / "validation")}
