"""Wave 4 integration tests for the trainer's best/last commit protocol.

Scope: the commit *transaction* at trainer level -- crash points between the
selected-best snapshot, the ``last.pt`` commit and the public alias publish;
tampered selection metadata on resume; and the post-commit telemetry contract
(the durable report a W&B adapter actually reads, and telemetry not perturbing
the training streams).

Exact-resume continuity across schedule positions is a separate, larger
fixture owned by another worker (``tests/test_runtime_resume.py``); this file
deliberately uses one-epoch runs so it exercises the commit paths without
re-running that comparison.

Reference environment: Python 3.10.21 / torch 2.4.1 (CPU), single-threaded.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from self_audit.models.self_audit_net import SelfAuditNet
from self_audit.provenance import file_sha256
from self_audit.training._utils import COMMITTED_BEST_REFERENCE_KEY, save_checkpoint
from self_audit.training.checkpoint_commit import (
    PUBLIC_BEST_NAME,
    SELECTED_BEST_DIRNAME,
)
from self_audit.training.unified_config import parse_unified_config
from self_audit.training.unified_trainer import UnifiedTrainer


class _SyntheticDataset(Dataset):
    def __init__(self, count: int = 4, size: int = 32) -> None:
        torch.manual_seed(42)
        self.images = torch.randn(count, 3, size, size)
        self.masks = torch.randint(0, 4, (count, size, size))
        self.case_ids = [f"case_{index // 2:03d}" for index in range(count)]

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.images[index],
            "mask": self.masks[index],
            "case_id": self.case_ids[index],
        }


def _tiny_net(seed: int = 99) -> SelfAuditNet:
    torch.manual_seed(seed)
    return SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=8,
        window_k=2,
        max_turns=1,
    ).eval()


def _trainer(tmp_path: Path, *, total_epochs: int = 1) -> UnifiedTrainer:
    with open("configs/self_audit_full.yaml", "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)
    cfg_dict["model"].update(
        {"pretrained_encoder": False, "fallback": True, "shared_channels": 8, "window_k": 2, "max_turns": 1}
    )
    cfg_dict["checkpoint"]["output_dir"] = str(tmp_path / "weights")
    cfg_dict["checkpoint"]["best_selection_min_epoch"] = 0
    cfg_dict["dataset"]["dataloader"]["num_workers"] = 0
    cfg_dict["logging"]["report_dir"] = str(tmp_path / "reports")
    cfg_dict["training"]["amp"]["enabled"] = False
    cfg_dict["training"]["schedule"]["total_epochs"] = total_epochs
    # A valid *gated* interval: selection is defined only inside the
    # threshold-gated joint interval and only on final_foreground_macro_dice,
    # so the fixture has to be the real gated shape rather than a bootstrap
    # interval relying on a fallback metric.
    cfg_dict["training"]["schedule"]["intervals"] = [
        {
            "start_epoch": 0,
            "end_epoch": total_epochs,
            "name": "joint_gated",
            "trainable": "all",
            "encoder_lr": 0.0001,
            "annotation_lr": 0.001,
            "auditor_lr": 0.001,
            "annotation_weight": 1.0,
            "audit_weight": 1.0,
            "objective": "retained_final_annotation",
            "transition_population": "active_attempted",
            "rollout": "threshold_gate",
            "batch_size": 2,
            "accumulation_steps": 1,
            "augment": False,
            "reset_optimizer": True,
        }
    ]
    cfg = parse_unified_config(cfg_dict)
    trainer = UnifiedTrainer(cfg, model=_tiny_net(), device=torch.device("cpu"), disable_tqdm=True)

    dataset = _SyntheticDataset(count=4)
    loaders = (
        DataLoader(dataset, batch_size=2, shuffle=False),
        DataLoader(dataset, batch_size=2, shuffle=False),
    )
    for interval in cfg.training.schedule.intervals:
        trainer._loader_cache[(interval.batch_size, interval.augment)] = loaders
    return trainer


def _weights_dir(tmp_path: Path) -> Path:
    return tmp_path / "weights"


def _last_payload(tmp_path: Path) -> dict[str, Any]:
    payload = torch.load(_weights_dir(tmp_path) / "last.pt", map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    return payload


def _rewrite_last(tmp_path: Path, model: nn.Module, payload: dict[str, Any], **overrides: Any) -> None:
    """Re-emit last.pt with tampered selection metadata, keeping it loadable."""

    reserved = {"format_version", "model", "epoch", "global_step", "optimizer_step", "rng_state", "provenance",
                "optimizer", "scheduler", "scaler", "config"}
    extra = {key: value for key, value in payload.items() if key not in reserved}
    extra.update(overrides)
    save_checkpoint(
        _weights_dir(tmp_path) / "last.pt",
        model,
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        optimizer_step=int(payload["optimizer_step"]),
        config=payload.get("config"),
        extra=extra,
    )


def _snapshots(tmp_path: Path) -> list[Path]:
    store = _weights_dir(tmp_path) / SELECTED_BEST_DIRNAME
    return sorted(store.glob("*.pt")) if store.is_dir() else []


# ---------------------------------------------------------------------------
# 1. Commit order and selection metadata cross-consistency
# ---------------------------------------------------------------------------

def test_commit_writes_snapshot_then_last_then_alias(tmp_path: Path) -> None:
    """last.pt commits a hash-verified reference and best.pt is its published view."""

    trainer = _trainer(tmp_path)
    report = trainer.train()
    assert report["completed"] is True

    snapshots = _snapshots(tmp_path)
    assert len(snapshots) == 1, "one improving epoch must leave exactly one immutable snapshot"

    payload = _last_payload(tmp_path)
    reference = payload[COMMITTED_BEST_REFERENCE_KEY]
    assert isinstance(reference, dict)
    assert reference["path"] == f"{SELECTED_BEST_DIRNAME}/{snapshots[0].name}"
    assert reference["sha256"] == file_sha256(snapshots[0])

    # Selection metadata is derived from the reference, so it cannot drift.
    assert payload["best_metric"] == pytest.approx(float(reference["metric"]))
    assert int(payload["best_epoch"]) == int(reference["epoch"]) == 1
    assert payload["best_checkpoint_hash"] == reference["sha256"]
    assert payload["best_checkpoint_path"] == PUBLIC_BEST_NAME
    assert payload["best_selection_status"] == "selected"
    assert payload["role"] == "last"

    # The public alias is a byte-identical view of the committed snapshot.
    alias = _weights_dir(tmp_path) / PUBLIC_BEST_NAME
    assert file_sha256(alias) == reference["sha256"]
    assert not alias.is_symlink()

    # The declared selection metric is recorded, never left implicit.
    assert payload["selection_metric"] == "final_foreground_macro_dice"
    assert report["epochs"][-1]["selection_metric"] == "final_foreground_macro_dice"


def _patched_validation(trainer: UnifiedTrainer, **overrides: Any) -> Any:
    """Run the real validator, then override selection-metric keys."""

    real = UnifiedTrainer.validate_epoch

    def _wrapped(self: UnifiedTrainer, *args: Any, **kwargs: Any) -> dict[str, Any]:
        stats = dict(real(self, *args, **kwargs))
        for key, value in overrides.items():
            if value is _DROP:
                stats.pop(key, None)
            else:
                stats[key] = value
        return stats

    return patch.object(UnifiedTrainer, "validate_epoch", _wrapped)


_DROP = object()


def test_gated_selection_refuses_a_missing_dice(tmp_path: Path) -> None:
    """The gated interval selects on the Dice only; an absent Dice is a refusal."""

    trainer = _trainer(tmp_path)
    with _patched_validation(trainer, final_foreground_macro_dice=_DROP):
        with pytest.raises(ValueError) as excinfo:
            trainer.train()

    message = str(excinfo.value)
    assert "final_foreground_macro_dice" in message
    assert "Refusing to fall back to 'primary_metric'" in message


@pytest.mark.parametrize("undefined", [None, float("nan")])
def test_undefined_dice_cannot_improve_the_best(tmp_path: Path, undefined: Any) -> None:
    """An undefined Dice stays undefined: no selection, no coercion, no zero."""

    trainer = _trainer(tmp_path)
    with _patched_validation(trainer, final_foreground_macro_dice=undefined):
        trainer.train()

    payload = _last_payload(tmp_path)
    assert payload[COMMITTED_BEST_REFERENCE_KEY] is None
    assert payload["best_selection_status"] == "skipped_nonfinite_metric"
    assert payload["best_metric"] == -float("inf")
    assert payload["best_epoch"] is None
    assert payload["best_checkpoint_hash"] is None
    assert _snapshots(tmp_path) == []
    assert not (_weights_dir(tmp_path) / PUBLIC_BEST_NAME).exists()
    # The undefined metric is reported as an explicit null, never as 0.0.
    assert trainer.report["epochs"][-1]["selection_metric_value"] is None


# ---------------------------------------------------------------------------
# 2. Crash points: alias publication, and last.pt commit
# ---------------------------------------------------------------------------

def test_alias_failure_leaves_last_authoritative_and_resume_repairs_it(tmp_path: Path) -> None:
    """An interrupted alias publication is recovered from the committed reference."""

    trainer = _trainer(tmp_path)
    with patch(
        "self_audit.training.unified_trainer.publish_best_alias",
        side_effect=OSError(28, "No space left on device"),
    ):
        with pytest.raises(OSError):
            trainer.train()

    weights = _weights_dir(tmp_path)
    alias = weights / PUBLIC_BEST_NAME
    assert not alias.exists(), "the alias publish never landed"

    payload = _last_payload(tmp_path)
    reference = payload[COMMITTED_BEST_REFERENCE_KEY]
    snapshot = weights / reference["path"]
    assert snapshot.is_file(), "last.pt must reference durable bytes even though the alias failed"
    assert file_sha256(snapshot) == reference["sha256"]

    # Resume resolves the committed reference and republishes the alias.
    resumed = _trainer(tmp_path)
    resumed.resume_from_checkpoint(weights / "last.pt")

    assert alias.is_file()
    assert file_sha256(alias) == reference["sha256"]
    assert resumed.best_reference == reference
    assert resumed.best_checkpoint_hash == reference["sha256"]
    assert resumed.best_epoch == int(reference["epoch"])
    assert resumed.best_metric == pytest.approx(float(reference["metric"]))
    assert resumed.written_best_path == alias


def test_failed_last_commit_preserves_the_previously_referenced_snapshot(tmp_path: Path) -> None:
    """Audit R9 regression at trainer level: a failed commit cannot orphan the old last."""

    trainer = _trainer(tmp_path, total_epochs=2)
    real_save = save_checkpoint

    def _fail_second_last(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path).name == "last.pt" and int(kwargs.get("epoch", 0)) == 2:
            raise OSError(28, "No space left on device")
        return real_save(path, *args, **kwargs)

    with patch("self_audit.training.unified_trainer.save_checkpoint", side_effect=_fail_second_last):
        with pytest.raises(OSError):
            trainer.train()

    weights = _weights_dir(tmp_path)
    payload = _last_payload(tmp_path)
    assert int(payload["epoch"]) == 1, "the previous last.pt must survive intact"

    reference = payload[COMMITTED_BEST_REFERENCE_KEY]
    snapshot = weights / reference["path"]
    assert snapshot.is_file()
    assert file_sha256(snapshot) == reference["sha256"]
    # The alias the surviving last.pt names still matches it, so the pair is
    # consistent -- which is exactly what the pre-protocol ordering destroyed.
    assert file_sha256(weights / PUBLIC_BEST_NAME) == reference["sha256"]
    # The epoch-2 candidate snapshot may exist as inert bytes; nothing was deleted.
    assert snapshot in _snapshots(tmp_path)


# ---------------------------------------------------------------------------
# 3. Tampered selection metadata and corrupted snapshots
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("best_metric", 0.99, "best_metric"),
        ("best_epoch", 7, "best_epoch"),
        ("best_checkpoint_hash", "0" * 64, "best_checkpoint_hash"),
    ],
)
def test_tampered_selection_metadata_refuses_resume(
    tmp_path: Path, field: str, value: Any, expected: str
) -> None:
    trainer = _trainer(tmp_path)
    trainer.train()
    payload = _last_payload(tmp_path)

    _rewrite_last(tmp_path, _tiny_net(), payload, **{field: value})

    resumed = _trainer(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        resumed.resume_from_checkpoint(_weights_dir(tmp_path) / "last.pt")
    message = str(excinfo.value)
    assert "Tampered selection metadata" in message
    assert expected in message


def test_corrupted_snapshot_refuses_resume(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    trainer.train()
    snapshots = _snapshots(tmp_path)
    snapshots[0].write_bytes(b"not-the-committed-bytes")

    resumed = _trainer(tmp_path)
    with pytest.raises(ValueError, match="does not verify"):
        resumed.resume_from_checkpoint(_weights_dir(tmp_path) / "last.pt")


def test_missing_snapshot_refuses_resume(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    trainer.train()
    _snapshots(tmp_path)[0].unlink()

    resumed = _trainer(tmp_path)
    with pytest.raises(ValueError, match="does not verify"):
        resumed.resume_from_checkpoint(_weights_dir(tmp_path) / "last.pt")


def test_explicit_absent_selection_with_finite_metric_refuses_resume(tmp_path: Path) -> None:
    """A finite best_metric with no verifiable selection would suppress selection forever."""

    trainer = _trainer(tmp_path)
    trainer.train()
    payload = _last_payload(tmp_path)

    _rewrite_last(
        tmp_path,
        _tiny_net(),
        payload,
        **{COMMITTED_BEST_REFERENCE_KEY: None, "best_metric": 0.5, "best_epoch": None,
           "best_checkpoint_hash": None, "best_checkpoint_path": None},
    )

    resumed = _trainer(tmp_path)
    with pytest.raises(ValueError, match="explicit -inf sentinel"):
        resumed.resume_from_checkpoint(_weights_dir(tmp_path) / "last.pt")


def test_explicit_absent_selection_with_sentinel_resumes_and_can_still_select(
    tmp_path: Path,
) -> None:
    """The explicit no-selection sentinel is accepted and leaves selection open."""

    trainer = _trainer(tmp_path)
    trainer.train()
    payload = _last_payload(tmp_path)

    _rewrite_last(
        tmp_path,
        _tiny_net(),
        payload,
        **{COMMITTED_BEST_REFERENCE_KEY: None, "best_metric": -float("inf"), "best_epoch": None,
           "best_checkpoint_hash": None, "best_checkpoint_path": None},
    )

    resumed = _trainer(tmp_path)
    resumed.resume_from_checkpoint(_weights_dir(tmp_path) / "last.pt")

    assert resumed.best_reference is None
    assert resumed.best_metric == -float("inf")
    assert resumed.best_epoch is None
    assert resumed.written_best_path is None


# ---------------------------------------------------------------------------
# 3b. Verified progress history across resume
# ---------------------------------------------------------------------------

def test_resumed_report_preserves_the_committed_epoch_history(tmp_path: Path) -> None:
    """A resumed attempt keeps reporting the epochs it already committed.

    The history travels in the checkpoint, so it is covered by the same safe
    load and provenance as the weights, and is validated rather than trusted.
    """

    trainer = _trainer(tmp_path, total_epochs=3)
    real_train_epoch = UnifiedTrainer.train_epoch

    def _crash_on_third(self: UnifiedTrainer, *args: Any, **kwargs: Any) -> Any:
        if self.global_epoch == 2:
            raise RuntimeError("simulated crash entering epoch 3")
        return real_train_epoch(self, *args, **kwargs)

    with patch.object(UnifiedTrainer, "train_epoch", _crash_on_third):
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer.train()

    payload = _last_payload(tmp_path)
    assert int(payload["epoch"]) == 2
    assert [entry["epoch"] for entry in payload["epoch_history"]] == [1, 2]

    resumed = _trainer(tmp_path, total_epochs=3)
    start = resumed.resume_from_checkpoint(_weights_dir(tmp_path) / "last.pt")
    assert start == 2
    assert [entry["epoch"] for entry in resumed.report["epochs"]] == [1, 2]

    report = resumed.train(start_epoch=start)
    assert [entry["epoch"] for entry in report["epochs"]] == [1, 2, 3]
    assert report["completed_epochs"] == 3
    durable = json.loads((tmp_path / "reports" / "pipeline_report.json").read_text())
    assert [entry["epoch"] for entry in durable["epochs"]] == [1, 2, 3]


def test_saved_history_tail_is_the_committed_row_and_survives_resume(tmp_path: Path) -> None:
    """The row inside last.pt already describes the commit it is part of.

    Saving the pre-commit row recorded ``checkpoint_committed=False`` and a
    stale ``completed_epochs`` inside the checkpoint, and a resume then carried
    that tail into the final report even though the durable report disagreed.
    The committed row is now proposed before the save and adopted afterwards,
    so the bytes and the report agree -- with the one deliberate exception that
    a checkpoint cannot claim the later JSON report commit.
    """

    trainer = _trainer(tmp_path, total_epochs=3)
    real_train_epoch = UnifiedTrainer.train_epoch

    def _crash_on_third(self: UnifiedTrainer, *args: Any, **kwargs: Any) -> Any:
        if self.global_epoch == 2:
            raise RuntimeError("simulated crash entering epoch 3")
        return real_train_epoch(self, *args, **kwargs)

    with patch.object(UnifiedTrainer, "train_epoch", _crash_on_third):
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer.train()

    # Baseline: what the first attempt durably reported for epochs 1 and 2.
    baseline = json.loads((tmp_path / "reports" / "pipeline_report.json").read_text())
    baseline_rows = {int(entry["epoch"]): entry for entry in baseline["epochs"]}
    assert baseline["completed_epochs"] == 2

    payload = _last_payload(tmp_path)
    history = payload["epoch_history"]
    assert [entry["epoch"] for entry in history] == [1, 2]

    tail = history[-1]
    # The saved tail is a committed row, not a pre-commit one.
    assert tail["checkpoint_committed"] is True
    assert tail["status"] == "checkpoint_committed"
    assert tail["completed_epochs"] == 2
    assert tail["last_checkpoint_committed_epoch"] == 2
    assert tail["global_step"] == int(payload["global_step"])
    assert tail["optimizer_step"] == int(payload["optimizer_step"])
    assert tail["interval"] == payload["interval_name"]
    assert tail["selection_metric"] == "final_foreground_macro_dice"
    assert tail["best_selection_status"] == payload["best_selection_status"]
    assert tail["best_metric"] == pytest.approx(payload["best_metric"])
    assert tail["best_epoch"] == payload["best_epoch"]
    # ...and it truthfully does not claim the later JSON report commit.
    assert tail["report_committed"] is False
    assert history[0]["report_committed"] is True

    resumed = _trainer(tmp_path, total_epochs=3)
    start = resumed.resume_from_checkpoint(_weights_dir(tmp_path) / "last.pt")
    report = resumed.train(start_epoch=start)

    resumed_rows = {int(entry["epoch"]): entry for entry in report["epochs"]}
    assert sorted(resumed_rows) == [1, 2, 3]
    # Field-by-field equality with the baseline committed metadata, not just a
    # matching epoch list.  ``report_committed`` is the one field a checkpoint
    # cannot certify, so it is asserted separately rather than patched.
    for epoch in (1, 2):
        expected = dict(baseline_rows[epoch])
        observed = dict(resumed_rows[epoch])
        expected.pop("report_committed", None)
        carried = observed.pop("report_committed", None)
        assert observed == expected, f"resumed row {epoch} lost committed metadata"
        assert carried is (epoch == 1)

    durable = json.loads((tmp_path / "reports" / "pipeline_report.json").read_text())
    durable_rows = {int(entry["epoch"]): entry for entry in durable["epochs"]}
    for epoch in (1, 2):
        assert durable_rows[epoch]["checkpoint_committed"] is True
        assert durable_rows[epoch]["completed_epochs"] == epoch


def test_resume_restores_the_saved_completed_validation(tmp_path: Path) -> None:
    """``last_completed_validation`` is read from where save_checkpoint puts it.

    ``save_checkpoint`` merges ``extra`` into the payload, so the value sits at
    the top level; resume read ``payload["extra"]``, found nothing, and left
    the completed validation as ``None`` -- losing exactly the evidence a
    failure report written right after resume has to show.
    """

    trainer = _trainer(tmp_path, total_epochs=2)
    real_train_epoch = UnifiedTrainer.train_epoch

    def _crash_on_second(self: UnifiedTrainer, *args: Any, **kwargs: Any) -> Any:
        if self.global_epoch == 1:
            raise RuntimeError("simulated crash entering epoch 2")
        return real_train_epoch(self, *args, **kwargs)

    with patch.object(UnifiedTrainer, "train_epoch", _crash_on_second):
        with pytest.raises(RuntimeError, match="simulated crash"):
            trainer.train()

    payload = _last_payload(tmp_path)
    saved_validation = payload["last_completed_validation"]
    assert isinstance(saved_validation, dict) and saved_validation
    assert "final_foreground_macro_dice" in saved_validation

    resumed = _trainer(tmp_path, total_epochs=2)
    start = resumed.resume_from_checkpoint(_weights_dir(tmp_path) / "last.pt")
    assert start == 1
    assert isinstance(resumed.last_completed_validation, dict)
    assert resumed.last_completed_validation["final_foreground_macro_dice"] == pytest.approx(
        float(saved_validation["final_foreground_macro_dice"])
    )

    # A failure immediately after resume must still report that validation.
    with patch.object(UnifiedTrainer, "train_epoch", side_effect=RuntimeError("post-resume failure")):
        with pytest.raises(RuntimeError, match="post-resume failure"):
            resumed.train(start_epoch=start)

    failure = json.loads((tmp_path / "reports" / "failure.json").read_text())
    recorded = failure["last_completed_validation"]
    assert isinstance(recorded, dict) and recorded
    assert recorded["final_foreground_macro_dice"] == pytest.approx(
        float(saved_validation["final_foreground_macro_dice"])
    )


def test_resume_refuses_a_truncated_or_missing_history(tmp_path: Path) -> None:
    """A history that does not describe epochs 1..N is refused, not truncated."""

    trainer = _trainer(tmp_path, total_epochs=2)
    trainer.train()
    payload = _last_payload(tmp_path)
    assert [entry["epoch"] for entry in payload["epoch_history"]] == [1, 2]

    _rewrite_last(tmp_path, _tiny_net(), payload, epoch_history=[dict(payload["epoch_history"][0])])
    with pytest.raises(ValueError, match="refusing a truncated history"):
        _trainer(tmp_path, total_epochs=2).resume_from_checkpoint(_weights_dir(tmp_path) / "last.pt")

    _rewrite_last(tmp_path, _tiny_net(), payload, epoch_history=None)
    with pytest.raises(ValueError, match="carries no 'epoch_history'"):
        _trainer(tmp_path, total_epochs=2).resume_from_checkpoint(_weights_dir(tmp_path) / "last.pt")


@pytest.mark.parametrize("counter", ["epoch", "global_step", "optimizer_step"])
def test_resume_rejects_invalid_counters(tmp_path: Path, counter: str) -> None:
    """Counters are validated, never coerced or defaulted to 0."""

    trainer = _trainer(tmp_path)
    trainer.train()
    payload = _last_payload(tmp_path)

    corrupt = dict(payload)
    corrupt[counter] = -1
    reserved = {"format_version", "model", "rng_state", "provenance", "optimizer", "scheduler",
                "scaler", "config", "epoch", "global_step", "optimizer_step"}
    extra = {key: value for key, value in corrupt.items() if key not in reserved}
    weights = _weights_dir(tmp_path)
    # save_checkpoint validates counters itself, so write the tampered payload
    # directly to prove the resume path is what refuses it.
    raw = torch.load(weights / "last.pt", map_location="cpu", weights_only=True)
    raw[counter] = -1
    torch.save(raw, weights / "last.pt")

    with pytest.raises(ValueError, match=counter):
        _trainer(tmp_path).resume_from_checkpoint(weights / "last.pt")


# ---------------------------------------------------------------------------
# 4. Post-commit telemetry contract
# ---------------------------------------------------------------------------

class _DiskReadingLogger:
    """A mocked adapter that reads the artifacts a real W&B callback would see."""

    def __init__(self, weights: Path, reports: Path) -> None:
        self.weights = weights
        self.reports = reports
        self.observations: list[dict[str, Any]] = []
        self.enabled = True
        self.failed_log_count = 0
        self.telemetry_errors: list[str] = []
        self.last_error: Exception | None = None
        self.finish_status = "not_finished"
        self.finish_count = 0
        self.identity_summary: dict[str, Any] = {"init_status": "ok", "mode": "offline"}

    def log(self, payload: Any, step: Any = None) -> None:
        durable_report = json.loads((self.reports / "pipeline_report.json").read_text())
        durable_last = torch.load(self.weights / "last.pt", map_location="cpu", weights_only=True)
        self.observations.append(
            {
                "payload": dict(payload),
                "report_completed_epochs": durable_report["completed_epochs"],
                "report_last_row": durable_report["epochs"][-1],
                "checkpoint_epoch": int(durable_last["epoch"]),
                "checkpoint_interval": durable_last["interval_name"],
                "checkpoint_reference": durable_last[COMMITTED_BEST_REFERENCE_KEY],
                "checkpoint_best_metric": durable_last["best_metric"],
                "checkpoint_schedule": dict(durable_last["resolved_schedule"]),
                "checkpoint_history": [dict(entry) for entry in durable_last["epoch_history"]],
            }
        )

    def set_summary(self, payload: Any) -> None:
        return None

    def log_images(self, payload: Any, step: Any = None) -> None:
        return None

    def finish(self, exit_code: Any = None) -> None:
        self.finish_count += 1
        self.finish_status = "ok"


def _install_logger(trainer: UnifiedTrainer, logger: Any) -> None:
    """Pin a mocked adapter, bypassing delayed initialization."""

    trainer._wandb_started = True
    trainer.logger = logger
    trainer.wandb_identity = dict(getattr(logger, "identity_summary", {}))


def test_post_commit_log_sees_the_advanced_count_on_disk(tmp_path: Path) -> None:
    """The durable report a post-commit adapter reads carries the current epoch count.

    Previously the atomic report was written before the counters advanced, so a
    post-commit reader saw the *previous* completed_epochs.
    """

    trainer = _trainer(tmp_path, total_epochs=2)
    logger = _DiskReadingLogger(_weights_dir(tmp_path), tmp_path / "reports")
    _install_logger(trainer, logger)

    report = trainer.train()

    assert len(logger.observations) == 2
    for index, observed in enumerate(logger.observations, start=1):
        assert observed["report_completed_epochs"] == index, "durable count must not lag the commit"
        assert observed["checkpoint_epoch"] == index
        assert observed["checkpoint_interval"] == "joint_gated"
        row = observed["report_last_row"]
        assert row["epoch"] == index
        assert row["checkpoint_committed"] is True
        assert row["completed_epochs"] == index
        assert row["interval"] == "joint_gated"
        payload = observed["payload"]
        schedule = observed["checkpoint_schedule"]
        # Schedule weights, rollout and LRs: the same resolved values in the
        # committed row, in the checkpoint and in what the adapter receives.
        for field, logged_key in (
            ("annotation_weight", "schedule/annotation_weight"),
            ("audit_weight", "schedule/audit_weight"),
        ):
            assert schedule[field] == pytest.approx(row["resolved_schedule"][field])
            assert payload[logged_key] == pytest.approx(schedule[field])
        assert schedule["annotation_weight"] == pytest.approx(1.0)
        assert schedule["audit_weight"] == pytest.approx(1.0)
        assert payload["schedule/rollout_mode"] == schedule["rollout_mode"] == "threshold_gate"
        assert schedule["interval_name"] == row["interval"] == "joint_gated"
        assert schedule["interval_index"] == row["interval_index"] == 0
        # Every learning rate the adapter logs comes from the committed row.
        for logged_key, row_keys in (
            ("train/lr_encoder", ("lr_encoder",)),
            ("train/lr_annotation", ("lr_annotation_heads", "lr_annotation")),
            ("train/lr_auditor", ("lr_auditor",)),
        ):
            assert logged_key in payload, f"the adapter must receive {logged_key}"
            row_key = next((key for key in row_keys if key in row), None)
            assert row_key is not None, f"committed row carries none of {row_keys}"
            assert row[row_key] == pytest.approx(payload[logged_key]), (
                f"{logged_key} disagrees with the committed row {row_key}"
            )
        # Configured interval LRs are recorded in the checkpoint too.
        for field in ("encoder_lr", "annotation_lr", "auditor_lr"):
            assert schedule[field] == pytest.approx(row["resolved_schedule"][field])

        # Canonical Dice family: whatever the adapter is given must be exactly
        # what the durable row measured, and an undefined metric stays absent
        # rather than being fabricated as 0.
        for logged_key, row_keys in (
            ("val/initial_dice", ("initial_foreground_macro_dice", "modes/initial_dice", "initial_dice")),
            ("val/final_dice", ("final_foreground_macro_dice", "modes/self_audit_dice", "final_dice")),
        ):
            row_key = next((key for key in row_keys if row.get(key) is not None), None)
            if logged_key in payload:
                assert row_key is not None, f"{logged_key} logged with no measured row value"
                assert payload[logged_key] == pytest.approx(row[row_key])
            else:
                assert row_key is None, f"{logged_key} measured in the row but not logged"
        if "val/candidate_gain" in payload:
            assert payload["val/candidate_gain"] != 0.0 or row.get("modes/candidate_path_gain") == 0.0
        # The committed selection metric value is the Dice, or an explicit null.
        selection_value = row["selection_metric_value"]
        assert row["selection_metric"] == "final_foreground_macro_dice"
        if selection_value is None:
            assert row["best_selection_status"] == "skipped_nonfinite_metric"
        else:
            assert selection_value == pytest.approx(row["final_foreground_macro_dice"])
        # The phase-C branch reports the Dice family rather than a generic
        # primary_metric key, so assert on whichever the adapter received.
        if "val/primary_metric" in payload:
            assert row["primary_metric"] == pytest.approx(payload["val/primary_metric"])
        else:
            assert row["primary_metric"] == pytest.approx(payload["val/final_dice"])
        assert observed["checkpoint_best_metric"] == pytest.approx(
            float(observed["checkpoint_reference"]["metric"])
        )
        # The checkpoint carries the full committed history, not only this row.
        history = observed["checkpoint_history"]
        assert [entry["epoch"] for entry in history] == list(range(1, index + 1))

    assert report["completed_epochs"] == 2
    assert report["telemetry"]["finalization_status"] == "finalized"


def test_finalization_status_reports_a_failed_finish(tmp_path: Path) -> None:
    """A logger whose finish fails must not be reported as finalized."""

    class _FailingFinishLogger(_DiskReadingLogger):
        def finish(self, exit_code: Any = None) -> None:
            self.finish_count += 1
            self.finish_status = "failed"
            raise RuntimeError("finish exploded")

    trainer = _trainer(tmp_path)
    logger = _FailingFinishLogger(_weights_dir(tmp_path), tmp_path / "reports")
    _install_logger(trainer, logger)

    report = trainer.train()

    assert report["telemetry"]["finalization_status"] == "finish_failed"
    # The adapter raised instead of recording its own error; it is recorded here.
    assert any("finish: finish exploded" in entry for entry in logger.telemetry_errors)
    assert logger.failed_log_count >= 1
    durable = json.loads((tmp_path / "reports" / "pipeline_report.json").read_text())
    assert durable["telemetry"]["finalization_status"] == "finish_failed"


def test_telemetry_cannot_consume_the_training_random_streams(tmp_path: Path) -> None:
    """A drawing adapter must not change training or the checkpointed RNG."""

    class _RngDrawingLogger(_DiskReadingLogger):
        def log(self, payload: Any, step: Any = None) -> None:
            super().log(payload, step)
            random.random()
            np.random.rand(3)
            torch.randn(3)

        def set_summary(self, payload: Any) -> None:
            torch.randn(5)

        def finish(self, exit_code: Any = None) -> None:
            random.random()
            super().finish(exit_code)

    quiet = _trainer(tmp_path / "quiet", total_epochs=2)
    _install_logger(quiet, _DiskReadingLogger(_weights_dir(tmp_path / "quiet"), tmp_path / "quiet" / "reports"))
    quiet.train()
    quiet_last = torch.load(
        _weights_dir(tmp_path / "quiet") / "last.pt", map_location="cpu", weights_only=True
    )

    noisy = _trainer(tmp_path / "noisy", total_epochs=2)
    _install_logger(noisy, _RngDrawingLogger(_weights_dir(tmp_path / "noisy"), tmp_path / "noisy" / "reports"))
    noisy.train()
    noisy_last = torch.load(
        _weights_dir(tmp_path / "noisy") / "last.pt", map_location="cpu", weights_only=True
    )

    for name, expected in quiet_last["model"].items():
        assert torch.equal(expected, noisy_last["model"][name]), f"telemetry perturbed {name}"
    assert torch.equal(quiet_last["rng_state"]["torch"], noisy_last["rng_state"]["torch"])
    assert torch.equal(
        quiet_last["rng_state"]["numpy"][1], noisy_last["rng_state"]["numpy"][1]
    )
    assert quiet_last["rng_state"]["python"] == noisy_last["rng_state"]["python"]


# ---------------------------------------------------------------------------
# 5. Delayed W&B initialization and recorded identity
# ---------------------------------------------------------------------------

def test_logger_initialization_is_delayed_until_after_resume(tmp_path: Path) -> None:
    """Construction must not open a run, so a resume cannot orphan one."""

    trainer = _trainer(tmp_path)
    # The placeholder is a disabled no-op: nothing external was started.
    assert trainer._wandb_started is False
    assert trainer.logger.enabled is False
    assert trainer.logger.identity_summary["init_status"] == "not_attempted"

    trainer.train()
    assert trainer._wandb_started is True
    # The canonical fixture keeps W&B disabled, so identity stays truthful
    # rather than claiming a run.
    assert trainer.wandb_identity["init_status"] == "not_attempted"
    assert trainer.wandb_identity["backend_resume_performed"] is False

    payload = _last_payload(tmp_path)
    assert payload["wandb_identity"]["init_status"] == "not_attempted"
    report = json.loads((tmp_path / "reports" / "pipeline_report.json").read_text())
    assert report["wandb_identity"]["init_status"] == "not_attempted"


def test_resumed_attempt_requests_the_recovered_run_identity(tmp_path: Path) -> None:
    """A resumed attempt asks to continue the recovered id, not a fresh run."""

    trainer = _trainer(tmp_path)
    trainer.train()
    original_run_id = trainer.run_id

    resumed = _trainer(tmp_path)
    resumed.resume_from_checkpoint(_weights_dir(tmp_path) / "last.pt")
    assert resumed.run_id == original_run_id
    assert resumed.is_resumed is True

    resumed._wandb_settings["enabled"] = True
    resumed._wandb_settings["mode"] = "online"
    captured: dict[str, Any] = {}

    class _Recording:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.enabled = False
            self.failed_log_count = 0
            self.telemetry_errors: list[str] = []
            self.finish_status = "not_finished"
            self.identity_summary = {"requested_run_id": kwargs.get("run_id"),
                                     "requested_resume": kwargs.get("resume"),
                                     "init_status": "ok"}

    with patch("self_audit.training.unified_trainer.WandbLogger", _Recording):
        identity = resumed.start_logging()

    assert captured["run_id"] == original_run_id
    assert captured["resume"] == "allow"
    assert identity["requested_run_id"] == original_run_id


# ---------------------------------------------------------------------------
# 6. Worker lifecycle default
# ---------------------------------------------------------------------------

def test_training_loaders_never_use_persistent_workers(tmp_path: Path) -> None:
    """Persistent worker RNG cannot be checkpointed, so the runtime default is off."""

    trainer = _trainer(tmp_path)
    trainer._loader_cache.clear()
    interval = trainer.schedule.intervals[0]

    captured: list[dict[str, Any]] = []

    def _spy(dataset: Any, config: Any, **kwargs: Any) -> Any:
        captured.append(dict(config))
        return DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)

    with patch("self_audit.training.unified_trainer.build_patient_dataset", return_value=_SyntheticDataset(4)), \
         patch("self_audit.training.unified_trainer.build_data_loader", side_effect=_spy):
        trainer.get_loaders(interval)

    assert captured, "loaders must be built through build_data_loader"
    for config in captured:
        assert config["persistent_workers"] is False
