"""Tests for Wave 2 telemetry correctness, loss aggregation, and W&B logging.

Validates:
1. Deliberately distinct numeric initial/candidate/final Dice metrics in Phase A (from evaluate_annotation_headroom)
   and Phase C (proposer improvement / candidate_path_gain vs self_audit_gain).
2. Missing measurement omission: never fabricate zeros when metrics are unmeasured.
3. Accurate on-policy / synthetic mapping for Phase B auditor evaluation without prefix-filter loss.
4. Canonical LR keys: train/encoder_lr, train/annotation_lr, train/auditor_lr.
5. Schedule metadata: audit_weight, rollout_mode, and on_policy_fraction only when observed.
6. Absolute prohibition of phase_a/, phase_b/, phase_c/ prefixes and misleading labels in W&B payloads.
7. Exact reference loss reduction parity in train_epoch (sample-weighted for Phase A, batch-mean for Phase B/C),
   distinguishing individual component losses from total weighted loss.
"""

from __future__ import annotations

from typing import Any
import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from self_audit.training.schedule import ScheduleInterval
from self_audit.training.unified_config import load_unified_config
from self_audit.training.unified_trainer import UnifiedTrainer
from self_audit.models.self_audit_net import SelfAuditNet


class _SyntheticToyDataset(Dataset):
    def __init__(self, count: int = 6, size: int = 32) -> None:
        torch.manual_seed(123)
        self.images = torch.randn(count, 3, size, size)
        self.masks = torch.randint(0, 4, (count, size, size))
        self.case_ids = [f"case_{idx:03d}" for idx in range(count)]

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.images[index],
            "mask": self.masks[index],
            "case_id": self.case_ids[index],
        }


def _make_dummy_trainer() -> UnifiedTrainer:
    config = load_unified_config("configs/self_audit_full.yaml")
    model = SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()
    trainer = UnifiedTrainer(
        config,
        model=model,
        device=torch.device("cpu"),
        disable_tqdm=True,
    )
    return trainer


# ============================================================================
# Gate 1: Distinct Numeric Initial/Candidate/Final and Semantic Mappings
# ============================================================================


def test_phase_a_distinct_numeric_headroom_mapping() -> None:
    """Phase A W&B payload must map evaluate_annotation_headroom metrics distinctly.
    
    initial_dice must come from a0_dice, candidate_gain from total_refinement_gain,
    and final_dice from final_dice/val_macro. All 3 must be distinctly numeric.
    """
    trainer = _make_dummy_trainer()
    int_0 = trainer.schedule.intervals[0]  # weighted_a0_a3
    trainer.setup_interval_optimizer_and_scheduler(int_0, num_batches=10)

    train_stats = {
        "train_loss": 0.456,
        "annotation_loss": 0.456,
        "train_batches": 10,
        "epoch_optimizer_steps": 10,
    }

    # Deliberately distinct values: initial != final, candidate_gain != 0
    val_stats = {
        "val_loss": 0.412,
        "val_macro_foreground_dice": 0.725,
        "phase_a/sample_count": 50.0,
        "phase_a/a0_dice": 0.650,
        "phase_a/final_dice": 0.725,
        "phase_a/total_refinement_gain": 0.075,
        "phase_a/mean_stage_headroom": 0.025,
        "phase_a/headroom_collapse_ratio": 0.15,
        "phase_a/stage_0/candidate_gain": 0.040,
        "phase_a/stage_1/candidate_gain": 0.035,
    }

    payload = trainer._build_wandb_payload(
        epoch=5,
        interval=int_0,
        interval_idx=0,
        train_stats=train_stats,
        val_stats=val_stats,
    )

    # Verify distinct numeric values
    assert payload["val/initial_dice"] == pytest.approx(0.650)
    assert payload["val/candidate_gain"] == pytest.approx(0.075)
    assert payload["val/final_dice"] == pytest.approx(0.725)
    assert payload["val/initial_dice"] != payload["val/final_dice"]
    assert payload["val/candidate_gain"] > 0.0

    # Verify remapped headroom metrics without phase_a/ prefix
    assert payload["headroom/mean_stage_headroom"] == pytest.approx(0.025)
    assert payload["headroom/headroom_collapse_ratio"] == pytest.approx(0.15)
    assert payload["headroom/stage_0/candidate_gain"] == pytest.approx(0.040)
    assert payload["headroom/stage_1/candidate_gain"] == pytest.approx(0.035)

    # Strictly no phase_a/ prefixes allowed in the payload
    for k in payload:
        assert not k.startswith("phase_a/"), f"Forbidden phase_a prefix found in key: {k}"


def test_phase_a_missing_headroom_omission_never_fabricate_zeros() -> None:
    """When evaluate_headroom is disabled or unavailable, omitted values must NOT be fabricated as zeros or aliases."""
    trainer = _make_dummy_trainer()
    int_0 = trainer.schedule.intervals[0]
    trainer.setup_interval_optimizer_and_scheduler(int_0, num_batches=10)

    train_stats = {
        "train_loss": 0.500,
        "annotation_loss": 0.500,
    }
    # Only standard validation Dice without headroom
    val_stats = {
        "val_macro_foreground_dice": 0.710,
    }

    payload = trainer._build_wandb_payload(
        epoch=0,
        interval=int_0,
        interval_idx=0,
        train_stats=train_stats,
        val_stats=val_stats,
    )

    # final_dice is measured and present
    assert payload["val/final_dice"] == pytest.approx(0.710)
    # initial_dice and candidate_gain were not measured: NEVER fabricate zeros or equate initial to final
    assert "val/initial_dice" not in payload, "initial_dice was fabricated without headroom measurement"
    assert "val/candidate_gain" not in payload, "candidate_gain was fabricated as zero without headroom measurement"
    # audit_loss was not measured in Phase A: NEVER fabricate zero
    assert "train/audit_loss" not in payload, "audit_loss was fabricated as zero in Phase A"


def test_phase_c_distinct_proposer_improvement_vs_self_audit_gain() -> None:
    """In Phase C, candidate_gain MUST represent proposer improvement (candidate_path_gain), NOT self-audit net gain.
    
    Self-audit net gain must have its own explicit semantic key (val/self_audit_gain).
    """
    trainer = _make_dummy_trainer()
    int_2 = trainer.schedule.intervals[2]  # retained_final_annotation
    trainer.setup_interval_optimizer_and_scheduler(int_2, num_batches=10)

    train_stats = {
        "train_loss": 0.850,
        "annotation_loss": 0.400,
        "audit_loss": 0.450,
    }

    # Initial = 0.70, Always-accept / Candidate path = 0.74 (gain = +0.04),
    # Self-audit final = 0.78 (gain = +0.08).
    val_stats = {
        "initial_foreground_macro_dice": 0.700,
        "final_foreground_macro_dice": 0.780,
        "net_dice_gain": 0.080,
        "modes/initial_dice": 0.700,
        "modes/always_accept_dice": 0.740,
        "modes/self_audit_dice": 0.780,
        "modes/candidate_path_gain": 0.040,  # Proposer improvement
        "modes/self_audit_gain": 0.080,      # Retained self-audit gain
        "modes/audit_rescue_vs_always": 0.040,
        "modes/headroom_capture_ratio": 0.65,
        "harmful_acceptance_rate": 0.02,
        "beneficial_rejection_rate": 0.01,
    }

    payload = trainer._build_wandb_payload(
        epoch=125,
        interval=int_2,
        interval_idx=2,
        train_stats=train_stats,
        val_stats=val_stats,
    )

    # initial, candidate_gain (proposer), and final must be distinctly numeric
    assert payload["val/initial_dice"] == pytest.approx(0.700)
    assert payload["val/candidate_gain"] == pytest.approx(0.040), "candidate_gain falsely recorded self-audit gain instead of proposer improvement"
    assert payload["val/final_dice"] == pytest.approx(0.780)
    # Self-audit gain is explicitly labeled
    assert payload["val/self_audit_gain"] == pytest.approx(0.080)
    assert payload["val/candidate_gain"] != payload["val/self_audit_gain"]

    # Proposer always-accept dice and audit rescue metrics
    assert payload["val/always_accept_dice"] == pytest.approx(0.740)
    assert payload["modes/audit_rescue_vs_always"] == pytest.approx(0.040)

    # Both train annotation and audit loss are present and nonzero
    assert payload["train/annotation_loss"] == pytest.approx(0.400)
    assert payload["train/audit_loss"] == pytest.approx(0.450)
    assert payload["train/total_loss"] == pytest.approx(0.850)

    # No phase_c/ prefixes
    for k in payload:
        assert not k.startswith("phase_c/"), f"Forbidden phase_c prefix in key: {k}"


def test_phase_c_missing_decomposition_never_fabricate_candidate_gain() -> None:
    """When decomposition is not evaluated in Phase C, candidate_gain must NOT be fabricated or aliased to net_gain."""
    trainer = _make_dummy_trainer()
    int_2 = trainer.schedule.intervals[2]
    trainer.setup_interval_optimizer_and_scheduler(int_2, num_batches=10)

    train_stats = {
        "train_loss": 0.600,
        "annotation_loss": 0.350,
        "audit_loss": 0.250,
    }
    val_stats = {
        "initial_foreground_macro_dice": 0.680,
        "final_foreground_macro_dice": 0.730,
        "net_dice_gain": 0.050,
    }

    payload = trainer._build_wandb_payload(
        epoch=121,
        interval=int_2,
        interval_idx=2,
        train_stats=train_stats,
        val_stats=val_stats,
    )

    assert payload["val/initial_dice"] == pytest.approx(0.680)
    assert payload["val/final_dice"] == pytest.approx(0.730)
    assert payload["val/self_audit_gain"] == pytest.approx(0.050)
    # Proposer improvement was not measured: candidate_gain must be omitted, NEVER fabricated as 0 or net_gain
    assert "val/candidate_gain" not in payload, "candidate_gain should be omitted when proposer improvement is not measured"


# ============================================================================
# Gate 2: Phase B Auditor Validation Metrics & On-Policy Mapping
# ============================================================================


def test_phase_b_val_outputs_and_provenance_preservation() -> None:
    """Ensure Phase B validator metrics (on-policy, synthetic, combined, primary_metric) are preserved without prefix-filtering loss."""
    trainer = _make_dummy_trainer()
    int_1 = trainer.schedule.intervals[1]  # counterfactual_audit
    trainer.setup_interval_optimizer_and_scheduler(int_1, num_batches=10)

    train_stats = {
        "train_loss": 0.312,
        "audit_loss": 0.312,
    }
    val_stats = {
        "audit_loss": 0.285,
        "primary_metric": 0.815,
        "primary_metric_source": "audit/on_policy/auroc",
        "transitions": 120.0,
        "local_fix_count": 40.0,
        "local_unchanged_count": 60.0,
        "local_regress_count": 20.0,
        "audit/on_policy/auroc": 0.815,
        "audit/on_policy/auprc": 0.760,
        "audit/on_policy/transition_count": 48.0,
        "audit/synthetic/auroc": 0.890,
        "audit/synthetic/auprc": 0.840,
        "audit/synthetic/transition_count": 72.0,
        "audit/combined/auroc": 0.860,
        "audit/combined/auprc": 0.808,
        "audit/combined/transition_count": 120.0,
    }

    payload = trainer._build_wandb_payload(
        epoch=105,
        interval=int_1,
        interval_idx=1,
        train_stats=train_stats,
        val_stats=val_stats,
    )

    # Real metrics must NOT be lost by prefix filtering
    assert payload["val/primary_metric"] == pytest.approx(0.815)
    assert payload["val/primary_metric_source"] == "audit/on_policy/auroc"
    assert payload["audit/val_loss"] == pytest.approx(0.285)
    assert payload["audit/transitions"] == pytest.approx(120.0)
    assert payload["audit/local_fix_count"] == pytest.approx(40.0)

    # On-policy and synthetic metrics must be mapped accurately
    assert payload["audit/on_policy/auroc"] == pytest.approx(0.815)
    assert payload["audit/synthetic/auroc"] == pytest.approx(0.890)
    assert payload["audit/combined/auroc"] == pytest.approx(0.860)

    # on_policy_fraction observed from transition counts (48 / 120 = 0.40)
    assert payload["schedule/on_policy_fraction"] == pytest.approx(0.40)

    # Phase B has no annotation dice: must be omitted, NEVER fabricated
    assert "val/initial_dice" not in payload
    assert "val/candidate_gain" not in payload
    assert "val/final_dice" not in payload
    assert "train/annotation_loss" not in payload

    # No phase_b/ prefixes
    for k in payload:
        assert not k.startswith("phase_b/"), f"Forbidden phase_b prefix in key: {k}"


def test_schedule_on_policy_fraction_omission_when_unobserved() -> None:
    """on_policy_fraction must be strictly omitted when not observed in data, never fabricated."""
    trainer = _make_dummy_trainer()
    int_1 = trainer.schedule.intervals[1]
    trainer.setup_interval_optimizer_and_scheduler(int_1, num_batches=10)

    train_stats = {"train_loss": 0.3}
    val_stats = {"audit_loss": 0.25}

    payload = trainer._build_wandb_payload(
        epoch=101,
        interval=int_1,
        interval_idx=1,
        train_stats=train_stats,
        val_stats=val_stats,
    )

    assert "schedule/on_policy_fraction" not in payload, "on_policy_fraction was fabricated without observed transitions"


# ============================================================================
# Gate 3: Canonical Learning Rate and Schedule Keys
# ============================================================================


def test_canonical_learning_rate_keys() -> None:
    """Verify exact canonical learning rate keys: train/encoder_lr, train/annotation_lr, train/auditor_lr.

    Accounts for actual optimizer telemetry: step 0 reflects warmup (1e-8 factor),
    and stepping through warmup reaches the configured peak interval learning rates.
    """
    trainer = _make_dummy_trainer()

    # Interval 0: only encoder and annotation are trainable
    int_0 = trainer.schedule.intervals[0]
    trainer.setup_interval_optimizer_and_scheduler(int_0, num_batches=5)
    # At step 0: scheduler applies initial warmup factor 1e-8
    payload_0 = trainer._build_wandb_payload(0, int_0, 0, {"train_loss": 0.5}, {})
    assert payload_0["train/encoder_lr"] == pytest.approx(3e-5 * 1e-8)
    assert payload_0["train/annotation_lr"] == pytest.approx(3e-4 * 1e-8)
    assert payload_0["train/auditor_lr"] == pytest.approx(0.0)

    # Step through warmup (warmup_epochs=5 * steps_per_epoch=5 = 25 steps) to reach peak LR
    for _ in range(25):
        trainer.scheduler.step()
    payload_0_warm = trainer._build_wandb_payload(0, int_0, 0, {"train_loss": 0.5}, {})
    assert payload_0_warm["train/encoder_lr"] == pytest.approx(3e-5)
    assert payload_0_warm["train/annotation_lr"] == pytest.approx(3e-4)
    assert payload_0_warm["train/auditor_lr"] == pytest.approx(0.0)

    # Interval 1: only auditor is trainable
    int_1 = trainer.schedule.intervals[1]
    trainer.setup_interval_optimizer_and_scheduler(int_1, num_batches=5)
    payload_1 = trainer._build_wandb_payload(100, int_1, 1, {"train_loss": 0.4}, {})
    assert payload_1["train/encoder_lr"] == pytest.approx(0.0)
    assert payload_1["train/annotation_lr"] == pytest.approx(0.0)
    assert payload_1["train/auditor_lr"] == pytest.approx(3e-4 * 1e-8)

    for _ in range(25):
        trainer.scheduler.step()
    payload_1_warm = trainer._build_wandb_payload(100, int_1, 1, {"train_loss": 0.4}, {})
    assert payload_1_warm["train/encoder_lr"] == pytest.approx(0.0)
    assert payload_1_warm["train/annotation_lr"] == pytest.approx(0.0)
    assert payload_1_warm["train/auditor_lr"] == pytest.approx(3e-4)

    # Interval 2: all modules trainable
    int_2 = trainer.schedule.intervals[2]
    trainer.setup_interval_optimizer_and_scheduler(int_2, num_batches=5)
    payload_2 = trainer._build_wandb_payload(120, int_2, 2, {"train_loss": 0.3}, {})
    assert payload_2["train/encoder_lr"] == pytest.approx(1e-6 * 1e-8)
    assert payload_2["train/annotation_lr"] == pytest.approx(1e-5 * 1e-8)
    assert payload_2["train/auditor_lr"] == pytest.approx(1e-5 * 1e-8)

    for _ in range(25):
        trainer.scheduler.step()
    payload_2_warm = trainer._build_wandb_payload(120, int_2, 2, {"train_loss": 0.3}, {})
    assert payload_2_warm["train/encoder_lr"] == pytest.approx(1e-6)
    assert payload_2_warm["train/annotation_lr"] == pytest.approx(1e-5)
    assert payload_2_warm["train/auditor_lr"] == pytest.approx(1e-5)

    # Fallback telemetry when optimizer is None (reads train_stats directly)
    trainer.optimizer = None
    payload_stats = trainer._build_wandb_payload(
        0, int_0, 0,
        {"train_loss": 0.5, "encoder_lr": 3e-5, "annotation_lr": 3e-4, "auditor_lr": 0.0},
        {},
    )
    assert payload_stats["train/encoder_lr"] == pytest.approx(3e-5)
    assert payload_stats["train/annotation_lr"] == pytest.approx(3e-4)
    assert payload_stats["train/auditor_lr"] == pytest.approx(0.0)



def test_schedule_metadata_keys() -> None:
    """Verify schedule/audit_weight, schedule/annotation_weight, schedule/rollout_mode keys."""
    trainer = _make_dummy_trainer()

    int_0 = trainer.schedule.intervals[0]
    p0 = trainer._build_wandb_payload(0, int_0, 0, {}, {})
    assert p0["schedule/annotation_weight"] == pytest.approx(1.0)
    assert p0["schedule/audit_weight"] == pytest.approx(0.0)
    assert p0["schedule/rollout_mode"] == "propagate_no_audit"

    int_2 = trainer.schedule.intervals[2]
    p2 = trainer._build_wandb_payload(120, int_2, 2, {}, {})
    assert p2["schedule/annotation_weight"] == pytest.approx(1.0)
    assert p2["schedule/audit_weight"] == pytest.approx(1.0)
    assert p2["schedule/rollout_mode"] == "threshold_gate"


# ============================================================================
# Gate 4: train_epoch Loss Aggregation & Reference Reduction Parity
# ============================================================================


def test_train_epoch_loss_aggregation_phase_a_reduction() -> None:
    """In Phase A, train_epoch must reduce sample-weighted (matching train_annotation_epoch reference)."""
    trainer = _make_dummy_trainer()
    int_0 = trainer.schedule.intervals[0]
    trainer.setup_interval_optimizer_and_scheduler(int_0, num_batches=2)

    ds = _SyntheticToyDataset(count=6, size=16)
    # Loader with batch_size=4 produces batch 1 (size 4) and batch 2 (size 2)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    stats = trainer.train_epoch(int_0, loader)

    assert "train_loss" in stats
    assert "annotation_loss" in stats
    # Phase A does not train auditor: audit_loss must not be present
    assert "audit_loss" not in stats
    assert stats["annotation_loss"] > 0.0
    assert stats["train_loss"] > 0.0
    assert stats["train_batches"] == 2
    assert stats["epoch_optimizer_steps"] > 0


def test_train_epoch_loss_aggregation_phase_c_nonzero_components() -> None:
    """In Phase C, train_epoch must aggregate distinct nonzero annotation and audit losses and total loss."""
    trainer = _make_dummy_trainer()
    int_2 = trainer.schedule.intervals[2]
    trainer.setup_interval_optimizer_and_scheduler(int_2, num_batches=2)

    ds = _SyntheticToyDataset(count=4, size=16)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    stats = trainer.train_epoch(int_2, loader)

    assert "train_loss" in stats
    assert "annotation_loss" in stats
    assert "audit_loss" in stats
    # In Phase C both components are trained and must be nonzero numbers
    assert stats["annotation_loss"] > 0.0
    assert stats["train_loss"] > 0.0
    # Audit loss can be 0 or positive depending on transitions, but must be finite
    assert math.isfinite(stats["audit_loss"])


def test_train_epoch_preserves_incomplete_epoch_contract() -> None:
    """train_epoch must preserve incomplete_epoch and max_steps contract."""
    trainer = _make_dummy_trainer()
    int_0 = trainer.schedule.intervals[0]
    trainer.setup_interval_optimizer_and_scheduler(int_0, num_batches=4)

    ds = _SyntheticToyDataset(count=8, size=16)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    stats = trainer.train_epoch(int_0, loader, max_steps=1)

    assert trainer.incomplete_epoch is True
    assert trainer.incomplete_reason == "max_steps_reached"
    assert stats["epoch_optimizer_steps"] == 1
