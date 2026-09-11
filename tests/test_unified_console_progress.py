"""Tests for console progress bar behavior, running loss telemetry, and persistent epoch summaries.

Validates:
1. Native Candidate C runner (scripts/run_acdc_mnms_candidate_c.sh) has removed default --no_tqdm flags
   while preserving python -u, tee redirection, device assignment, and independent datasets.
2. CLI --no_tqdm opt-out flag remains fully supported across argument parsing and UnifiedTrainer.
3. train_epoch configures tqdm with leave=True so completed epoch bars remain visible.
4. train_epoch closes progress bars via try/finally on normal completion, early break, and exceptions,
   safely handling fake-tqdm instances without close().
5. train_epoch postfix includes running annotation/audit losses with exact existing reduction divisors
   (sample-weighted for Phase A, batch-mean for Phase B/C), without fabricating unmeasured Dice.
6. Bounded max_steps halts do not lose the final processed batch postfix.
7. Persistent console epoch summaries format explicitly-named metrics per phase:
   - Phase A: train_loss, val_loss, val_macro_foreground_dice (plus initial_dice only if measured).
   - Phase B: train_loss, val_loss, on_policy_<source> primary metric, and local/combined fix & regress F1.
   - Phase C: train_loss, initial/final foreground macro dice, net gain, harmful acceptance & beneficial rejection rates.
8. Missing, null, or NaN metrics are omitted or marked N/A, never fabricated as zeros or mislabeled as Dice.
9. Disabling tqdm (--no_tqdm) disables progress bars but preserves the persistent epoch summary with flush=True.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import math
from pathlib import Path
import sys
from typing import Any
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from self_audit.training.schedule import ScheduleInterval
from self_audit.training.unified_config import apply_overrides, load_unified_config
from self_audit.training.unified_trainer import UnifiedTrainer
from self_audit.training._utils import add_wandb_and_tqdm_args
import scripts.train_self_audit as runner


class _ToyProgressDataset(Dataset):
    """Synthetic dataset generating minimal batches for trainer loop execution."""

    def __init__(self, count: int = 4, size: int = 16) -> None:
        torch.manual_seed(42)
        self.images = torch.randn(count, 3, size, size)
        self.masks = torch.randint(0, 4, (count, size, size))
        self.case_ids = [f"toy_{idx:03d}" for idx in range(count)]

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.images[index],
            "mask": self.masks[index],
            "case_id": self.case_ids[index],
        }


class _CapturingFakeTqdm:
    """Mock tqdm that records initialization options, postfix calls, and lifecycle events."""

    instances: list[_CapturingFakeTqdm] = []

    def __init__(
        self,
        iterable: Any = None,
        desc: str = "",
        disable: bool = False,
        leave: bool = False,
        total: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.iterable = iterable
        self.desc = desc
        self.disable = disable
        self.leave = leave
        self.total = total
        self.n = 0
        self.kwargs = kwargs
        self.postfix_history: list[dict[str, Any]] = []
        self.closed = False
        _CapturingFakeTqdm.instances.append(self)

    def __iter__(self) -> Iterator[Any]:
        if self.iterable is not None:
            for item in self.iterable:
                yield item

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix(self, postfix: dict[str, Any]) -> None:
        self.postfix_history.append(dict(postfix))

    def close(self) -> None:
        self.closed = True


class _ToyLinearModel(nn.Module):
    """Tiny linear model satisfying UnifiedTrainer structure."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.annotation_heads = nn.Linear(4, 4)
        self.auditor = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.annotation_heads(self.encoder(x))


def _build_toy_trainer(*, disable_tqdm: bool = False) -> UnifiedTrainer:
    config = load_unified_config("configs/self_audit_full.yaml")
    model = _ToyLinearModel()
    trainer = UnifiedTrainer(
        config,
        model=model,
        device=torch.device("cpu"),
        disable_tqdm=disable_tqdm,
    )
    return trainer


# ============================================================================
# 1. Candidate C Runner Default Flags & CLI Support
# ============================================================================


def test_candidate_c_runner_defaults_no_tqdm_flag_removed() -> None:
    """scripts/run_acdc_mnms_candidate_c.sh must not pass --no_tqdm by default."""
    script_path = Path("scripts/run_acdc_mnms_candidate_c.sh")
    assert script_path.exists(), f"Missing runner script: {script_path}"
    content = script_path.read_text(encoding="utf-8")

    # Confirm no_tqdm is removed from both ACDC and M&Ms invocations
    assert "--no_tqdm" not in content, "Found default --no_tqdm in Candidate C runner script"

    # Confirm python -u and tee are preserved
    assert "python -u scripts/train_self_audit.py" in content
    assert 'tee "logs/${ACDC_RUN}.log"' in content
    assert 'tee "logs/${MNMS_RUN}.log"' in content

    # Confirm batch size and GPU device variables preserved
    assert 'BATCH_SIZE="${BATCH_SIZE:-8}"' in content
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"' in content


def test_cli_user_opt_out_no_tqdm_remains_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI continues to support --no_tqdm when explicitly passed by the user."""
    # Shared helper behavior
    parser = argparse.ArgumentParser()
    add_wandb_and_tqdm_args(parser)
    assert parser.parse_args([]).no_tqdm is False
    assert parser.parse_args(["--no_tqdm"]).no_tqdm is True

    # scripts/train_self_audit.py parser behavior
    monkeypatch.setattr(sys, "argv", ["train_self_audit.py", "--config", "configs/self_audit_full.yaml"])
    args_default = runner._parse_args()
    assert args_default.no_tqdm is False

    monkeypatch.setattr(sys, "argv", ["train_self_audit.py", "--config", "configs/self_audit_full.yaml", "--no_tqdm"])
    args_explicit = runner._parse_args()
    assert args_explicit.no_tqdm is True


# ============================================================================
# 2. Progress Bar Visibility (leave=True) and Safe Closing
# ============================================================================


def test_train_epoch_tqdm_configured_with_leave_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """train_epoch must instantiate tqdm with leave=True to keep completed bars visible."""
    _CapturingFakeTqdm.instances.clear()
    monkeypatch.setattr("self_audit.training.unified_trainer.tqdm", _CapturingFakeTqdm)

    trainer = _build_toy_trainer(disable_tqdm=False)
    interval = trainer.schedule.intervals[0]
    trainer.setup_interval_optimizer_and_scheduler(interval, num_batches=2)

    # Monkeypatch compute_batch_loss to return a tiny scalar loss
    dummy_loss = torch.tensor(0.5, requires_grad=True)
    monkeypatch.setattr(
        trainer,
        "compute_batch_loss",
        lambda batch, interval: (dummy_loss, {"parts": {"loss": 0.5}}),
    )

    dataset = _ToyProgressDataset(count=2)
    loader = DataLoader(dataset, batch_size=1)

    trainer.train_epoch(interval, loader)

    assert len(_CapturingFakeTqdm.instances) == 1
    pbar = _CapturingFakeTqdm.instances[0]
    assert pbar.leave is True, f"Expected leave=True on train progress bar, got {pbar.leave}"
    assert pbar.closed is True, "Progress bar should be closed at epoch completion"


def test_train_epoch_progress_bar_closes_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Progress bar must close even if an exception occurs during training."""
    _CapturingFakeTqdm.instances.clear()
    monkeypatch.setattr("self_audit.training.unified_trainer.tqdm", _CapturingFakeTqdm)

    trainer = _build_toy_trainer(disable_tqdm=False)
    interval = trainer.schedule.intervals[0]
    trainer.setup_interval_optimizer_and_scheduler(interval, num_batches=2)

    def _failing_compute(batch: Any, interval: Any) -> Any:
        raise RuntimeError("Simulated mid-epoch training error")

    monkeypatch.setattr(trainer, "compute_batch_loss", _failing_compute)

    dataset = _ToyProgressDataset(count=2)
    loader = DataLoader(dataset, batch_size=1)

    with pytest.raises(RuntimeError, match="Simulated mid-epoch training error"):
        trainer.train_epoch(interval, loader)

    assert len(_CapturingFakeTqdm.instances) == 1
    pbar = _CapturingFakeTqdm.instances[0]
    assert pbar.closed is True, "Progress bar close() must be called in finally block on exception"


# ============================================================================
# 3. Postfix Telemetry & Correct Existing Divisors (No Fabricated Dice)
# ============================================================================


@pytest.mark.parametrize(
    ("interval_idx", "expected_ann_key", "expected_aud_key"),
    [
        (0, "ann_loss", None),          # weighted_a0_a3: sample-weighted annotation loss
        (1, None, "audit_loss"),        # counterfactual_audit: batch-mean audit loss
        (2, "ann_loss", "audit_loss"),  # retained_final_annotation: batch-mean joint components
    ],
)
def test_train_epoch_postfix_running_losses(
    monkeypatch: pytest.MonkeyPatch,
    interval_idx: int,
    expected_ann_key: str | None,
    expected_aud_key: str | None,
) -> None:
    """Postfix must display available running losses with correct divisors and never fabricate Dice."""
    _CapturingFakeTqdm.instances.clear()
    monkeypatch.setattr("self_audit.training.unified_trainer.tqdm", _CapturingFakeTqdm)

    trainer = _build_toy_trainer(disable_tqdm=False)
    interval = trainer.schedule.intervals[interval_idx]
    trainer.setup_interval_optimizer_and_scheduler(interval, num_batches=2)

    dummy_loss = torch.tensor(0.4000, requires_grad=True)
    details = {
        "parts": {"loss": 0.3500},
        "annotation_loss": 0.3500,
        "audit_loss": 0.2500,
    }
    monkeypatch.setattr(trainer, "compute_batch_loss", lambda batch, interval: (dummy_loss, details))

    dataset = _ToyProgressDataset(count=2)
    loader = DataLoader(dataset, batch_size=1)

    trainer.train_epoch(interval, loader)

    pbar = _CapturingFakeTqdm.instances[0]
    assert len(pbar.postfix_history) >= 1
    last_postfix = pbar.postfix_history[-1]

    # Baseline keys
    assert "loss" in last_postfix
    assert "avg" in last_postfix
    assert "opt_step" in last_postfix

    # Never fabricate Dice in training postfix
    for k in last_postfix:
        assert "dice" not in k.lower(), f"Unmeasured Dice was placed in train postfix: {k}"

    if expected_ann_key is not None:
        assert expected_ann_key in last_postfix
        assert float(last_postfix[expected_ann_key]) > 0.0
    else:
        assert "ann_loss" not in last_postfix

    if expected_aud_key is not None:
        assert expected_aud_key in last_postfix
        assert float(last_postfix[expected_aud_key]) > 0.0
    else:
        assert "audit_loss" not in last_postfix


# ============================================================================
# 4. Final Bounded Step Postfix Preservation
# ============================================================================


def test_train_epoch_max_steps_preserves_final_postfix(monkeypatch: pytest.MonkeyPatch) -> None:
    """When max_steps breaks training mid-epoch, the final postfix must reflect that processed step and count correctly."""
    _CapturingFakeTqdm.instances.clear()
    monkeypatch.setattr("self_audit.training.unified_trainer.tqdm", _CapturingFakeTqdm)

    trainer = _build_toy_trainer(disable_tqdm=False)
    interval = trainer.schedule.intervals[0]
    trainer.setup_interval_optimizer_and_scheduler(interval, num_batches=4)

    dummy_loss = torch.tensor(0.3210, requires_grad=True)
    monkeypatch.setattr(
        trainer,
        "compute_batch_loss",
        lambda batch, interval: (dummy_loss, {"annotation_loss": 0.3210}),
    )

    dataset = _ToyProgressDataset(count=4)
    loader = DataLoader(dataset, batch_size=1)

    # Halt after exactly 1 step
    stats = trainer.train_epoch(interval, loader, max_steps=1)

    pbar = _CapturingFakeTqdm.instances[0]
    assert len(pbar.postfix_history) >= 1
    final_postfix = pbar.postfix_history[-1]

    # Postfix was updated for the 1st step before the break occurred
    assert final_postfix["opt_step"] == 1
    assert final_postfix["loss"] == "0.3210"
    assert pbar.closed is True
    assert pbar.total == 4
    assert pbar.n == 1


def test_train_epoch_real_tqdm_progress_count_on_bounded_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """When max_steps=1 halts after 1 batch, real tqdm must register n=1, total=4, and '1/4' in output."""
    import io
    from tqdm import tqdm as real_tqdm

    created_pbars: list[real_tqdm] = []
    buf = io.StringIO()

    def _tqdm_factory(*args: Any, **kwargs: Any) -> real_tqdm:
        kwargs["file"] = buf
        pbar = real_tqdm(*args, **kwargs)
        created_pbars.append(pbar)
        return pbar

    monkeypatch.setattr("self_audit.training.unified_trainer.tqdm", _tqdm_factory)

    trainer = _build_toy_trainer(disable_tqdm=False)
    interval = trainer.schedule.intervals[0]
    trainer.setup_interval_optimizer_and_scheduler(interval, num_batches=4)

    dummy_loss = torch.tensor(0.3210, requires_grad=True)
    monkeypatch.setattr(
        trainer,
        "compute_batch_loss",
        lambda batch, interval: (dummy_loss, {"annotation_loss": 0.3210}),
    )

    dataset = _ToyProgressDataset(count=4)
    loader = DataLoader(dataset, batch_size=1)

    trainer.train_epoch(interval, loader, max_steps=1)

    assert len(created_pbars) == 1
    pbar = created_pbars[0]
    assert pbar.total == 4
    assert pbar.n == 1
    output = buf.getvalue()
    assert "1/4" in output
    assert pbar.postfix is not None
    assert "opt_step=1" in pbar.postfix
    assert "loss=0.3210" in pbar.postfix


# ============================================================================
# 5. Persistent Console Epoch Summary
# ============================================================================


def test_format_epoch_summary_phase_a_annotation_headroom() -> None:
    """Phase A summary must show train_loss, val_loss, val_macro_foreground_dice, and initial_dice when measured."""
    trainer = _build_toy_trainer()
    interval = trainer.schedule.intervals[0]  # annotation_bootstrap / weighted_a0_a3
    trainer.optimizer_step = 12

    train_stats = {"train_loss": 0.4567, "loss": 0.4567}
    val_stats = {
        "val_loss": 0.3891,
        "val_macro_foreground_dice": 0.7654,
        "phase_a/a0_dice": 0.6800,  # Measured initial dice from headroom
    }

    summary = trainer.format_epoch_summary(epoch=0, interval=interval, train_stats=train_stats, val_stats=val_stats)

    assert f"[{interval.name}] Epoch 001/130" in summary
    assert "train_loss=0.4567" in summary
    assert "val_loss=0.3891" in summary
    assert "val_macro_foreground_dice=0.7654" in summary
    assert "initial_dice=0.6800" in summary
    assert "opt_step=12" in summary


def test_format_epoch_summary_phase_a_omits_unmeasured_initial_dice() -> None:
    """When headroom is disabled in Phase A, initial_dice must be omitted (never fabricated as 0.0000)."""
    trainer = _build_toy_trainer()
    interval = trainer.schedule.intervals[0]
    trainer.optimizer_step = 5

    train_stats = {"train_loss": 0.5000}
    val_stats = {
        "val_loss": 0.4500,
        "val_macro_foreground_dice": 0.7200,
        # No initial_dice or phase_a/a0_dice
    }

    summary = trainer.format_epoch_summary(epoch=2, interval=interval, train_stats=train_stats, val_stats=val_stats)

    assert "val_macro_foreground_dice=0.7200" in summary
    assert "initial_dice" not in summary, "Unmeasured initial_dice must be omitted"


def test_format_epoch_summary_phase_b_auditor() -> None:
    """Phase B summary must explicitly identify on_policy metric source and F1 scores, never calling accuracy Dice."""
    trainer = _build_toy_trainer()
    interval = trainer.schedule.intervals[1]  # auditor_training / counterfactual_audit (epochs 101-120)
    trainer.optimizer_step = 45

    train_stats = {"train_loss": 0.2345}
    val_stats = {
        "audit_loss": 0.2876,
        "primary_metric": 0.8123,
        "primary_metric_source": "on_policy_auroc",
        "audit/on_policy/local_fix_f1": 0.7410,
        "audit/on_policy/local_regress_f1": 0.6920,
        "local_fix_f1": 0.7250,
        "local_regress_f1": 0.6800,
    }

    # Epoch 100 corresponds to canonical Phase B Epoch 101/130
    summary = trainer.format_epoch_summary(epoch=100, interval=interval, train_stats=train_stats, val_stats=val_stats)

    assert f"[{interval.name}] Epoch 101/130" in summary
    assert "train_loss=0.2345" in summary
    assert "val_loss=0.2876" in summary
    assert "on_policy_auroc=0.8123" in summary
    assert "on_policy_fix_f1=0.7410" in summary
    assert "on_policy_regress_f1=0.6920" in summary
    assert "combined_fix_f1=0.7250" in summary
    assert "combined_regress_f1=0.6800" in summary
    assert "opt_step=45" in summary

    # Auditor accuracy / metrics must NEVER be called Dice
    assert "dice" not in summary.lower()


def test_format_epoch_summary_phase_b_fallback_accuracy() -> None:
    """When on_policy_auroc is not selected, resolve_primary_metric uses on_policy_improve_regress_accuracy verbatim."""
    trainer = _build_toy_trainer()
    interval = trainer.schedule.intervals[1]
    train_stats = {"train_loss": 0.2100}
    val_stats = {
        "audit_loss": 0.2500,
        "primary_metric": 0.7500,
        "primary_metric_source": "on_policy_improve_regress_accuracy",
    }
    summary = trainer.format_epoch_summary(epoch=100, interval=interval, train_stats=train_stats, val_stats=val_stats)
    assert "on_policy_improve_regress_accuracy=0.7500" in summary
    assert "dice" not in summary.lower()


def test_format_epoch_summary_phase_b_undefined_source() -> None:
    """When primary_metric_source is 'undefined' or missing, emit primary_metric=N/A."""
    trainer = _build_toy_trainer()
    interval = trainer.schedule.intervals[1]
    train_stats = {"train_loss": 0.2100}
    val_stats = {
        "audit_loss": 0.2500,
        "primary_metric": float("nan"),
        "primary_metric_source": "undefined",
    }
    summary = trainer.format_epoch_summary(epoch=100, interval=interval, train_stats=train_stats, val_stats=val_stats)
    assert "primary_metric=N/A" in summary


def test_format_epoch_summary_phase_c_joint() -> None:
    """Phase C summary must report initial/final dice, net gain, and acceptance/rejection rates."""
    trainer = _build_toy_trainer()
    interval = trainer.schedule.intervals[2]  # joint_self_audit / retained_final_annotation (epochs 121-130)
    trainer.optimizer_step = 100

    train_stats = {"train_loss": 0.1500}
    val_stats = {
        "initial_foreground_macro_dice": 0.6500,
        "final_foreground_macro_dice": 0.7400,
        "net_dice_gain": 0.0900,
        "harmful_acceptance_rate": 0.0350,
        "beneficial_rejection_rate": 0.0200,
    }

    # Epoch 120 corresponds to canonical Phase C Epoch 121/130
    summary = trainer.format_epoch_summary(epoch=120, interval=interval, train_stats=train_stats, val_stats=val_stats)

    assert f"[{interval.name}] Epoch 121/130" in summary
    assert "train_loss=0.1500" in summary
    assert "initial_foreground_macro_dice=0.6500" in summary
    assert "final_foreground_macro_dice=0.7400" in summary
    assert "net_gain=0.0900" in summary
    assert "harmful_acceptance_rate=0.0350" in summary
    assert "beneficial_rejection_rate=0.0200" in summary
    assert "opt_step=100" in summary


# ============================================================================
# 6. Missing-Metric Nonfabrication & Robust Formatting
# ============================================================================


def test_format_epoch_summary_nan_metrics_marked_na_never_zero() -> None:
    """NaN or empty slice metrics in validation must display N/A and never fake 0.0000."""
    trainer = _build_toy_trainer()
    interval = trainer.schedule.intervals[2]

    train_stats = {"train_loss": 0.1200}
    val_stats = {
        "initial_foreground_macro_dice": float("nan"),
        "final_foreground_macro_dice": float("nan"),
        "net_gain": float("nan"),
    }

    summary = trainer.format_epoch_summary(epoch=120, interval=interval, train_stats=train_stats, val_stats=val_stats)

    assert "initial_foreground_macro_dice=N/A" in summary
    assert "final_foreground_macro_dice=N/A" in summary
    assert "0.0000" not in summary


def test_format_epoch_summary_nonfinite_and_malformed_marked_na() -> None:
    """Non-finite floats (+/-inf) and malformed objects are rendered as N/A."""
    trainer = _build_toy_trainer()
    interval = trainer.schedule.intervals[1]
    train_stats = {"train_loss": 0.2000}
    val_stats = {
        "audit_loss": float("inf"),
        "primary_metric": "invalid_metric_string",
        "primary_metric_source": "on_policy_auroc",
    }
    summary = trainer.format_epoch_summary(epoch=100, interval=interval, train_stats=train_stats, val_stats=val_stats)
    assert "val_loss=N/A" in summary
    assert "on_policy_auroc=N/A" in summary


# ============================================================================
# 7. --no_tqdm Mode Preserves Persistent Summary
# ============================================================================


def test_no_tqdm_mode_disables_bars_but_preserves_printed_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """When disable_tqdm=True, progress bars are disabled but format_epoch_summary output is printed by _train_impl."""
    _CapturingFakeTqdm.instances.clear()
    monkeypatch.setattr("self_audit.training.unified_trainer.tqdm", _CapturingFakeTqdm)

    config = load_unified_config("configs/self_audit_full.yaml")
    config = apply_overrides(
        config,
        {
            "output_dir": str(tmp_path / "weights"),
            "report_dir": str(tmp_path / "reports"),
        },
    )
    model = _ToyLinearModel()
    trainer = UnifiedTrainer(
        config,
        model=model,
        device=torch.device("cpu"),
        disable_tqdm=True,
    )

    dataset = _ToyProgressDataset(count=4)
    loader = DataLoader(dataset, batch_size=2)
    trainer.get_loaders = lambda interval: (loader, loader)

    dummy_loss = torch.tensor(0.5, requires_grad=True)
    monkeypatch.setattr(
        trainer,
        "compute_batch_loss",
        lambda batch, interval: (dummy_loss, {"parts": {"loss": 0.5}}),
    )
    monkeypatch.setattr(
        trainer,
        "validate_epoch",
        lambda interval, val_loader, max_val_batches=None: {
            "val_loss": 0.4500,
            "val_macro_foreground_dice": 0.7000,
        },
    )

    # Execute production _train_impl directly
    trainer._train_impl(start_epoch=0, max_steps=1, max_val_batches=1)

    # Verify tqdm was disabled
    assert len(_CapturingFakeTqdm.instances) >= 1
    assert all(p.disable is True for p in _CapturingFakeTqdm.instances)

    # Verify production _train_impl printed the summary string directly to stdout
    captured = capsys.readouterr()
    assert "[annotation_bootstrap] Epoch 001/130" in captured.out
    assert "train_loss=0.5000" in captured.out
    assert "val_macro_foreground_dice=0.7000" in captured.out
    assert "opt_step=1" in captured.out
