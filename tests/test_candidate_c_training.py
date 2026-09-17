"""Training-side gates for the opt-in predicted-history exposure curriculum.

Scope of these tests: they prove the *software* contract only -- that the
option is off by default and leaves the baseline schedule bit-identical, that
turning it on actually routes an accepted-predicted-history rollout into the
loss, that the auditor-only stage cannot reach annotation parameters, that the
six sample-level counters mean what they are documented to mean, and that
replay records are invalidated around optimizer updates.

They prove nothing about whether the exposed evidence is informative.  The
bootstrap Auditor is untrained, so the evidence it supplies is cold and
uncalibrated by construction; no test here claims otherwise, and the synthetic
Candidate-C diagnostics rows below are stubs, never evidence that a solver ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from self_audit.models.self_audit_net import SelfAuditNet
from self_audit.training.finetune_joint import (
    ORDINARY_REAL_EVIDENCE_KEY,
    _counter_masks,
    _ordinary_attempt_masks,
    PREDICTED_HISTORY_COUNTER_KEYS,
    accumulate_rollout_counters,
    compute_joint_losses,
    compute_predicted_history_audit_loss,
    rollout_counters,
    zero_rollout_counters,
)
from self_audit.training.train_annotation import phase_a_loss
from self_audit.training.unified_config import RolloutConfig, load_unified_config
from self_audit.training.unified_trainer import UnifiedTrainer


CPU = torch.device("cpu")


@dataclass(frozen=True)
class _RolloutSettings(RolloutConfig):
    """Stand-in for the config worker's extended ``training.rollout`` block.

    The trainer reads the two curriculum fields defensively, so a test can
    supply them before the schema worker lands them without editing any config
    file this worker does not own.
    """

    predicted_history_exposure: Any = False
    predicted_history_weight: Any = 0.1


def _tiny_net(seed: int = 7) -> SelfAuditNet:
    torch.manual_seed(seed)
    return SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()


def _config(**rollout_kwargs: Any):
    config = load_unified_config("configs/self_audit_full.yaml")
    if rollout_kwargs:
        current = config.training.rollout
        object.__setattr__(
            config.training,
            "rollout",
            _RolloutSettings(tau=current.tau, max_turns=current.max_turns, **rollout_kwargs),
        )
    return config


def _batch(batch_size: int = 2, size: int = 32) -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    return {
        "image": torch.randn(batch_size, 3, size, size),
        "mask": torch.randint(0, 4, (batch_size, size, size)),
    }


class _SyntheticDataset(Dataset):
    def __init__(self, count: int = 4, size: int = 32) -> None:
        torch.manual_seed(5)
        self.images = torch.randn(count, 3, size, size)
        self.masks = torch.randint(0, 4, (count, size, size))

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.images[index],
            "mask": self.masks[index],
            "case_id": f"case_{index:03d}",
        }


# ---------------------------------------------------------------------------
# Counter semantics
# ---------------------------------------------------------------------------


def _mask(*values: int) -> torch.Tensor:
    return torch.tensor([bool(value) for value in values], dtype=torch.bool)


class _WindowModeStub:
    def __init__(self, window_mode: str) -> None:
        self.window_mode = window_mode


def test_counters_never_credit_rejected_history_as_accepted() -> None:
    """Rejected attempts HALT, so they never become accepted history."""

    # Three rows, two turns.  Turn 0: rows 0 and 1 attempted, only row 0
    # accepted.  Turn 1: only row 0 may still be attempted.
    output = {
        "logits": torch.zeros(3, 4, 2, 2),
        "transition_active_masks": [_mask(1, 1, 0), _mask(1, 0, 0)],
        "transition_state_masks": [_mask(1, 0, 0), _mask(0, 0, 0)],
    }
    counters = rollout_counters(output, model=_WindowModeStub("current"), batch_size=3)

    assert counters["attempted_turn_samples"] == 3
    # Only the single turn-0 acceptance; the rejected row 1 contributes nothing.
    assert counters["accepted_history_count"] == 1
    # Exactly one attempt consumed real accepted predicted history (row 0, turn 1).
    assert counters["real_evidence_attempts"] == 1
    assert counters["rollout_samples"] == 3
    assert counters["rollout_batches"] == 1


def test_candidate_c_counters_are_gated_on_the_actual_window_mode() -> None:
    """C counters are measured only when the C solver is the active mode."""

    rows = [
        # A record-consuming attempt that solved cleanly.
        {
            "turn": 1, "sample_index": 0, "record_turn": 0, "record_kind": "ordinary",
            "eligible": True, "feasible": True, "c1_passed": True, "fallback_reason": None,
            "accepted_path": "counterfactual",
        },
        # A record-consuming attempt whose C1 replay FAILED.  The core marks it
        # ineligible for the solve and routes it back to normal annotation; it
        # is still an attempt, and it is the only way replay_failure can ever
        # increment.
        {
            "turn": 1, "sample_index": 1, "record_turn": 0, "record_kind": "ordinary",
            "eligible": False, "feasible": False, "c1_passed": False,
            "fallback_reason": "replay_failure", "accepted_path": "ordinary",
        },
        # A record-consuming attempt with no REGRESS mass: eligible, infeasible.
        {
            "turn": 2, "sample_index": 0, "record_turn": 1, "record_kind": "ordinary",
            "eligible": True, "feasible": False, "c1_passed": True,
            "fallback_reason": "no_regress", "accepted_path": "ordinary",
        },
        # Not an attempt: a fresh ordinary turn holding no record at all.
        {
            "turn": 0, "sample_index": 0, "record_turn": None, "record_kind": None,
            "eligible": False, "feasible": None, "c1_passed": None,
            "fallback_reason": "no_accepted_history", "accepted_path": "ordinary",
        },
        # Not an attempt: the held record is itself a restitution, so it is
        # ineligible history and no replay is attempted.
        {
            "turn": 2, "sample_index": 1, "record_turn": None, "record_kind": "candidate_c",
            "eligible": False, "feasible": None, "c1_passed": None,
            "fallback_reason": "previous_action_not_replayable", "accepted_path": "ordinary",
        },
    ]
    output = {
        "logits": torch.zeros(2, 4, 2, 2),
        "transition_active_masks": [_mask(1, 1), _mask(1, 1), _mask(1, 1)],
        "transition_state_masks": [_mask(1, 1), _mask(1, 1), _mask(1, 1)],
        "candidate_c_diagnostics": rows,
    }

    measured = rollout_counters(output, model=_WindowModeStub("candidate_c"), batch_size=2)
    assert measured["candidate_c_diagnostics_available"] == 1.0
    # Three record-consuming attempts; eligibility does not gate the denominator.
    assert measured["candidate_c_attempts"] == 3
    assert measured["candidate_c_feasible"] == 1
    # A real core C1 failure can now increment this counter.
    assert measured["candidate_c_replay_failure"] == 1
    assert measured["candidate_c_fallback"] == 2

    for inactive_mode in ("current", "feature_only", "free_offsets", "direct_rollback"):
        gated = rollout_counters(output, model=_WindowModeStub(inactive_mode), batch_size=2)
        assert gated["candidate_c_diagnostics_available"] == 0.0
        assert gated["candidate_c_attempts"] == 0
        assert gated["candidate_c_feasible"] == 0
        assert gated["candidate_c_fallback"] == 0
        assert gated["candidate_c_replay_failure"] == 0


def test_an_eligible_row_without_a_record_is_not_a_candidate_c_attempt() -> None:
    """Only a held replayable ordinary record makes a turn a C attempt."""

    output = {
        "logits": torch.zeros(1, 4, 2, 2),
        "transition_active_masks": [_mask(1)],
        "transition_state_masks": [_mask(1)],
        "candidate_c_diagnostics": [
            {
                "turn": 0, "sample_index": 0, "record_turn": None, "record_kind": None,
                "eligible": True, "feasible": True, "c1_passed": True,
                "fallback_reason": None, "accepted_path": "ordinary",
            }
        ],
    }
    counters = rollout_counters(output, model=_WindowModeStub("candidate_c"), batch_size=1)
    assert counters["candidate_c_attempts"] == 0
    assert counters["candidate_c_feasible"] == 0


def test_an_explicit_attempted_flag_overrides_the_record_inference() -> None:
    output = {
        "logits": torch.zeros(1, 4, 2, 2),
        "transition_active_masks": [_mask(1)],
        "transition_state_masks": [_mask(1)],
        "candidate_c_diagnostics": [
            {
                "turn": 0, "sample_index": 0, "record_turn": None, "record_kind": None,
                "attempted": True, "eligible": False, "feasible": None,
                "c1_passed": False, "fallback_reason": "stale_record",
                "accepted_path": "ordinary",
            }
        ],
    }
    counters = rollout_counters(output, model=_WindowModeStub("candidate_c"), batch_size=1)
    assert counters["candidate_c_attempts"] == 1
    assert counters["candidate_c_replay_failure"] == 1
    assert counters["candidate_c_fallback"] == 1


def test_missing_diagnostics_are_reported_unavailable_not_measured() -> None:
    output = {
        "logits": torch.zeros(1, 4, 2, 2),
        "transition_active_masks": [_mask(1)],
        "transition_state_masks": [_mask(1)],
    }
    counters = rollout_counters(output, model=_WindowModeStub("candidate_c"), batch_size=1)
    assert counters["candidate_c_diagnostics_available"] == 0.0
    assert all(counters[f"candidate_c_{name}"] == 0 for name in ("attempts", "feasible", "fallback", "replay_failure"))


def test_accumulation_conjoins_availability_over_rollout_batches() -> None:
    total = zero_rollout_counters()
    available = dict(zero_rollout_counters())
    available.update({"rollout_batches": 1, "attempted_turn_samples": 2, "candidate_c_diagnostics_available": 1.0})
    unavailable = dict(zero_rollout_counters())
    unavailable.update({"rollout_batches": 1, "attempted_turn_samples": 3, "candidate_c_diagnostics_available": 0.0})

    accumulate_rollout_counters(total, available)
    assert total["candidate_c_diagnostics_available"] == 1.0
    accumulate_rollout_counters(total, unavailable)
    # One batch without diagnostics makes the whole stage aggregate unavailable.
    assert total["candidate_c_diagnostics_available"] == 0.0
    assert total["attempted_turn_samples"] == 5
    assert total["rollout_batches"] == 2


# ---------------------------------------------------------------------------
# Option default OFF: baseline must be untouched
# ---------------------------------------------------------------------------


def test_bootstrap_baseline_is_unchanged_when_the_option_is_off() -> None:
    config = _config()
    assert getattr(config.training.rollout, "predicted_history_exposure", False) is False

    model = _tiny_net(21)
    trainer = UnifiedTrainer(config, model=model, device=CPU, disable_tqdm=True)
    trainer.amp_enabled = False
    interval = config.training.schedule.intervals[0]
    batch = _batch()

    torch.manual_seed(3)
    loss, details = trainer.compute_batch_loss(batch, interval)

    torch.manual_seed(3)
    reference, parts = phase_a_loss(
        model.forward_annotation(batch["image"]),
        batch["mask"],
        stage_weights=config.training.annotation_loss.stage_weights,
    )
    reference = reference * float(interval.annotation_weight)

    assert torch.allclose(loss, reference, atol=1e-6)
    assert details["auxiliary_batches"] == 0
    assert details["predicted_history_exposure"] is False
    counters = details["rollout_counters"]
    # No rollout ran at all: every counter is zero and the stage says so.
    assert counters["rollout_batches"] == 0
    assert all(counters[key] == 0 for key in PREDICTED_HISTORY_COUNTER_KEYS)


def test_settings_reject_out_of_contract_values() -> None:
    model = _tiny_net(22)

    trainer = UnifiedTrainer(_config(predicted_history_weight=-1.0), model=model, device=CPU, disable_tqdm=True)
    with pytest.raises(ValueError, match="predicted_history_weight"):
        trainer.predicted_history_settings()

    trainer = UnifiedTrainer(
        _config(predicted_history_weight=float("nan")), model=model, device=CPU, disable_tqdm=True
    )
    with pytest.raises(ValueError, match="predicted_history_weight"):
        trainer.predicted_history_settings()

    trainer = UnifiedTrainer(
        _config(predicted_history_exposure="yes"), model=model, device=CPU, disable_tqdm=True
    )
    with pytest.raises(TypeError, match="predicted_history_exposure"):
        trainer.predicted_history_settings()

    # Default when the schema worker has not added the fields yet.
    trainer = UnifiedTrainer(_config(), model=model, device=CPU, disable_tqdm=True)
    assert trainer.predicted_history_settings() == (False, 0.1)


# ---------------------------------------------------------------------------
# Option ON: meaningful exposure
# ---------------------------------------------------------------------------


def test_bootstrap_exposure_adds_a_measured_accepted_history_rollout() -> None:
    weight = 0.5
    model = _tiny_net(23)
    interval = _config().training.schedule.intervals[0]
    batch = _batch()

    baseline_trainer = UnifiedTrainer(_config(), model=model, device=CPU, disable_tqdm=True)
    baseline_trainer.amp_enabled = False
    torch.manual_seed(3)
    baseline_loss, _ = baseline_trainer.compute_batch_loss(batch, interval)

    exposed_trainer = UnifiedTrainer(
        _config(predicted_history_exposure=True, predicted_history_weight=weight),
        model=model,
        device=CPU,
        disable_tqdm=True,
    )
    exposed_trainer.amp_enabled = False
    torch.manual_seed(3)
    exposed_loss, details = exposed_trainer.compute_batch_loss(batch, interval)

    assert details["auxiliary_batches"] == 1
    assert details["evidence_calibration"] == "cold_untrained_predicted"
    # The auxiliary term is really in the total, at the configured weight.
    expected = float(baseline_loss) + float(interval.annotation_weight) * weight * details[
        "auxiliary_annotation_loss"
    ]
    assert exposed_loss.item() == pytest.approx(expected, abs=1e-5)
    assert exposed_loss.item() != pytest.approx(float(baseline_loss), abs=1e-6)

    counters = details["rollout_counters"]
    assert counters["rollout_batches"] == 1
    assert counters["rollout_samples"] == int(batch["image"].shape[0])
    # The rollout genuinely attempted turns for this batch.
    assert counters["attempted_turn_samples"] > 0
    assert counters["accepted_history_count"] >= counters["real_evidence_attempts"]

    # The auxiliary term must reach annotation parameters (that is the point of
    # the exposure), and must not be a detached constant.
    assert exposed_loss.requires_grad
    exposed_loss.backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in model.annotation_expert.parameters()
    )


def test_auditor_stage_exposure_cannot_update_the_frozen_annotator() -> None:
    config = _config(predicted_history_exposure=True, predicted_history_weight=0.5)
    model = _tiny_net(24)
    trainer = UnifiedTrainer(config, model=model, device=CPU, disable_tqdm=True)
    trainer.amp_enabled = False
    interval = config.training.schedule.intervals[1]
    assert interval.trainable == "auditor"
    trainer.set_module_trainability(interval.trainable)

    # The stage freezes the annotator before anything runs.
    assert all(not parameter.requires_grad for parameter in model.annotation_expert.parameters())
    assert all(parameter.requires_grad for parameter in model.auditor.parameters())
    assert model.annotation_expert.training is False

    model.zero_grad(set_to_none=True)
    auxiliary, counters = compute_predicted_history_audit_loss(
        model,
        _batch(),
        tau_accept=config.training.rollout.tau,
        t_max=config.training.rollout.max_turns,
        neutral_margin=config.training.audit_loss.neutral_margin,
        local_weighting=True,
        audit_margin=config.training.audit_loss.audit_margin,
    )
    assert auxiliary is not None, "the rollout produced no transition to audit"
    assert counters["rollout_batches"] == 1
    assert counters["attempted_turn_samples"] > 0
    auxiliary.backward()

    auditor_ids = {id(parameter) for parameter in model.auditor.parameters()}
    for name, parameter in model.named_parameters():
        if id(parameter) in auditor_ids:
            continue
        assert parameter.grad is None or bool(parameter.grad.abs().sum() == 0), (
            f"auditor-only loss leaked a gradient into {name}"
        )
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in model.auditor.parameters()
    )


def test_joint_stage_does_not_run_a_duplicate_auxiliary_rollout() -> None:
    config = _config(predicted_history_exposure=True, predicted_history_weight=0.5)
    model = _tiny_net(25)
    trainer = UnifiedTrainer(config, model=model, device=CPU, disable_tqdm=True)
    trainer.amp_enabled = False
    interval = config.training.schedule.intervals[2]
    batch = _batch()

    torch.manual_seed(99)
    loss, details = trainer.compute_batch_loss(batch, interval)

    torch.manual_seed(99)
    reference, _ = compute_joint_losses(
        model,
        batch,
        tau_accept=config.training.rollout.tau,
        t_max=config.training.rollout.max_turns,
        lambda_audit=interval.audit_weight,
        neutral_margin=config.training.audit_loss.neutral_margin,
        local_weighting=True,
    )

    # Joint already rolls out with accepted predicted history; the exposure
    # option must add nothing on top of it.
    assert torch.allclose(loss, reference, atol=1e-6)
    assert details["auxiliary_batches"] == 0
    # Counters are still measured, from the joint rollout itself.
    assert details["rollout_counters"]["rollout_batches"] == 1
    assert details["rollout_counters"]["attempted_turn_samples"] > 0


# ---------------------------------------------------------------------------
# Stage telemetry and record invalidation
# ---------------------------------------------------------------------------


class _InvalidationCountingNet(SelfAuditNet):
    """Tiny net exposing the optional record-invalidation hook."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.invalidation_calls = 0
        self.record_generation = 0

    def invalidate_candidate_c_records(self) -> None:
        self.invalidation_calls += 1
        self.record_generation += 1


def _counting_net(seed: int = 31) -> _InvalidationCountingNet:
    torch.manual_seed(seed)
    return _InvalidationCountingNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()


def _run_one_epoch(trainer: UnifiedTrainer, interval: Any) -> dict[str, Any]:
    loader = DataLoader(_SyntheticDataset(count=4), batch_size=2, shuffle=False)
    trainer.setup_interval_optimizer_and_scheduler(interval, num_batches=len(loader))
    return trainer.train_epoch(interval, loader)


def test_records_are_invalidated_around_every_batch_and_optimizer_update() -> None:
    config = _config()
    model = _counting_net()
    trainer = UnifiedTrainer(config, model=model, device=CPU, disable_tqdm=True)
    trainer.amp_enabled = False
    interval = config.training.schedule.intervals[0]

    stats = _run_one_epoch(trainer, interval)

    batches = int(stats["train_batches"])
    steps = int(stats["epoch_optimizer_steps"])
    assert batches == 2 and steps == 2
    # One epoch-start invalidation, one before each batch, one after each update.
    assert model.invalidation_calls == 1 + batches + steps
    assert stats["rollout/candidate_c_invalidations"] == model.invalidation_calls
    assert model.record_generation == model.invalidation_calls


def test_every_stage_reports_the_six_counters_with_coherent_denominators() -> None:
    config = _config()
    interval = config.training.schedule.intervals[0]
    trainer = UnifiedTrainer(config, model=_tiny_net(41), device=CPU, disable_tqdm=True)
    trainer.amp_enabled = False

    off_stats = _run_one_epoch(trainer, interval)
    for key in PREDICTED_HISTORY_COUNTER_KEYS:
        assert off_stats[f"rollout/{key}"] == 0
    assert off_stats["rollout/rollout_batches"] == 0
    assert off_stats["rollout/predicted_history_exposure"] == 0.0
    assert off_stats["rollout/candidate_c_diagnostics_available"] == 0.0
    assert off_stats["rollout/auxiliary_batches"] == 0
    assert off_stats["rollout/evidence_calibration"] == "cold_untrained_predicted"

    on_config = _config(predicted_history_exposure=True, predicted_history_weight=0.25)
    on_trainer = UnifiedTrainer(on_config, model=_tiny_net(41), device=CPU, disable_tqdm=True)
    on_trainer.amp_enabled = False
    on_stats = _run_one_epoch(on_trainer, on_config.training.schedule.intervals[0])

    for key in PREDICTED_HISTORY_COUNTER_KEYS:
        assert f"rollout/{key}" in on_stats
    assert on_stats["rollout/rollout_batches"] == 2
    assert on_stats["rollout/rollout_samples"] == 4
    assert on_stats["rollout/attempted_turn_samples"] > 0
    assert on_stats["rollout/predicted_history_exposure"] == 1.0
    assert on_stats["rollout/predicted_history_weight"] == pytest.approx(0.25)
    assert on_stats["rollout/auxiliary_batches"] == 2
    assert on_stats["rollout/auxiliary_seconds"] > 0.0
    # Accepted history can never exceed the attempts that could produce it.
    assert on_stats["rollout/accepted_history_count"] <= on_stats["rollout/attempted_turn_samples"]
    assert on_stats["rollout/real_evidence_attempts"] <= on_stats["rollout/attempted_turn_samples"]
    # The ordinary-annotation companion is a subset of the broader counter.
    assert (
        on_stats[f"rollout/{ORDINARY_REAL_EVIDENCE_KEY}"]
        <= on_stats["rollout/real_evidence_attempts"]
    )
    # No restitution path exists in this configuration, so every acceptance is
    # ordinary by construction and the flag says the derivation was structural.
    assert on_stats["rollout/ordinary_accepted_path_from_diagnostics"] == 0.0
    assert (
        on_stats[f"rollout/{ORDINARY_REAL_EVIDENCE_KEY}"]
        == on_stats["rollout/real_evidence_attempts"]
    )
    # No C solver is configured, so the C counters are zero AND flagged unavailable.
    assert on_stats["rollout/candidate_c_diagnostics_available"] == 0.0


def test_wandb_payload_carries_the_same_counter_fields_as_the_report_row() -> None:
    config = _config(predicted_history_exposure=True, predicted_history_weight=0.25)
    trainer = UnifiedTrainer(config, model=_tiny_net(42), device=CPU, disable_tqdm=True)
    trainer.amp_enabled = False
    interval = config.training.schedule.intervals[0]
    train_stats = _run_one_epoch(trainer, interval)

    payload = trainer._build_wandb_payload(0, interval, 0, train_stats, {"primary_metric": 0.0})
    for key, value in train_stats.items():
        if not key.startswith("rollout/"):
            continue
        assert key in payload, f"{key} missing from the W&B payload"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert payload[key] == pytest.approx(float(value))
        else:
            assert payload[key] == value


def test_counters_read_the_real_solver_diagnostics_when_candidate_c_is_active() -> None:
    """The counter derivation matches the core module's actual diagnostics rows.

    This asserts only that the trainer reads the rows the core solver emits and
    classifies them by the documented rules.  It makes no claim about whether
    the solver improved anything.
    """

    torch.manual_seed(7)
    model = SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
        window_mode="candidate_c",
    ).eval()
    output = model.infer(torch.randn(2, 3, 32, 32), mode="self_audit", tau_accept=0.0, t_max=2)

    rows = output.get("candidate_c_diagnostics")
    assert isinstance(rows, list) and rows, "the C solver emitted no per-attempt rows"
    counters = rollout_counters(output, model=model, batch_size=2)

    assert counters["candidate_c_diagnostics_available"] == 1.0
    attempts = [
        row
        for row in rows
        if isinstance(row.get("record_turn"), int) and row.get("record_kind") == "ordinary"
    ]
    assert counters["candidate_c_attempts"] == len(attempts)
    # A first-turn row holds no record at all, so it is not a C attempt.
    assert any(row.get("record_turn") is None for row in rows)
    assert counters["candidate_c_attempts"] < len(rows)
    assert counters["candidate_c_feasible"] <= counters["candidate_c_attempts"]
    assert counters["candidate_c_fallback"] <= counters["candidate_c_attempts"]
    assert counters["candidate_c_replay_failure"] <= counters["candidate_c_attempts"]
    assert counters["accepted_history_count"] <= counters["attempted_turn_samples"]
    # The solver publishes accepted_path, so the ordinary split is derived
    # per sample rather than assumed, and can only narrow the broad counter.
    assert counters["ordinary_accepted_path_from_diagnostics"] == 1.0
    assert counters[ORDINARY_REAL_EVIDENCE_KEY] <= counters["real_evidence_attempts"]

    # Independently recompute the per-turn ordinary exposure straight from the
    # rows, keying on the CURRENT turn's path, and require the counter to agree.
    paths = {(row["turn"], row["sample_index"]): row.get("accepted_path") for row in rows}
    active = _counter_masks(output, "transition_active_masks", "active_masks")
    state = _counter_masks(output, "transition_state_masks", "state_masks")
    expected = [0]
    for turn in range(1, len(active)):
        expected.append(
            sum(
                1
                for sample in range(int(active[turn].numel()))
                if bool(state[turn - 1][sample])
                and bool(active[turn][sample])
                and paths.get((turn, sample)) == "ordinary"
            )
        )
    assert _per_turn_ordinary_evidence(output) == expected
    assert counters[ORDINARY_REAL_EVIDENCE_KEY] == sum(expected)


# ---------------------------------------------------------------------------
# Ordinary vs restitution acceptance
# ---------------------------------------------------------------------------


def _diagnostics_row(turn: int, sample: int, accepted_path: str) -> dict[str, Any]:
    return {
        "turn": turn,
        "sample_index": sample,
        "eligible": turn > 0,
        "record_turn": turn - 1 if turn > 0 else None,
        "record_kind": "ordinary" if turn > 0 else None,
        "accepted": True,
        "accepted_path": accepted_path,
        "c1_passed": None,
        "feasible": None,
        "fallback_reason": None,
    }


def _per_turn_ordinary_evidence(output: dict[str, Any]) -> list[int]:
    """Per-turn contributions to the ordinary-annotation evidence counter.

    Totals can coincide under a reversed temporal convention, so the per-turn
    breakdown is what actually pins the semantics down.
    """

    active = _counter_masks(output, "transition_active_masks", "active_masks")
    state = _counter_masks(output, "transition_state_masks", "state_masks")
    ordinary, _ = _ordinary_attempt_masks(output, active)
    contributions = [0]
    for index in range(1, len(active)):
        contributions.append(
            int((state[index - 1] & active[index] & ordinary[index]).sum().item())
        )
    return contributions


def test_ordinary_restitution_ordinary_splits_the_evidence_counters() -> None:
    """One sample walking ordinary -> restitution -> ordinary.

    ``real_evidence_attempts`` counts both post-acceptance attempts, because it
    is about the latest accepted evidence whatever produced it.  Only the turn
    whose OWN forward is ordinary annotation counts as generator exposure: at
    turn 1 the consumer is the solver (whose innovation is detached and whose
    audit input may have been ``None``), at turn 2 it is the trainable
    generator.  The total here is the same under either temporal convention,
    which is exactly why the asymmetric per-turn fixtures below exist.
    """

    output = {
        "logits": torch.zeros(1, 4, 2, 2),
        "transition_active_masks": [_mask(1), _mask(1), _mask(1)],
        "transition_state_masks": [_mask(1), _mask(1), _mask(1)],
        "candidate_c_diagnostics": [
            _diagnostics_row(0, 0, "ordinary"),
            _diagnostics_row(1, 0, "counterfactual"),
            _diagnostics_row(2, 0, "ordinary"),
        ],
    }
    counters = rollout_counters(output, model=_WindowModeStub("candidate_c"), batch_size=1)

    assert counters["attempted_turn_samples"] == 3
    # ALL accepted transitions, restitution included -- this is not an
    # ordinary-only count and the docstring no longer says it is.
    assert counters["accepted_history_count"] == 3
    # Turn 1 follows the ordinary turn-0 acceptance; turn 2 follows the
    # restitution acceptance at turn 1.
    assert counters["real_evidence_attempts"] == 2
    assert counters[ORDINARY_REAL_EVIDENCE_KEY] == 1
    assert counters["ordinary_accepted_path_from_diagnostics"] == 1.0
    # The exposure sits on turn 2, the ordinary one -- not on turn 1.
    assert _per_turn_ordinary_evidence(output) == [0, 0, 1]


def test_ordinary_counter_equals_the_broad_counter_without_a_restitution_path() -> None:
    """Baseline ordinary rollout: every post-accept attempt qualifies."""

    output = {
        "logits": torch.zeros(1, 4, 2, 2),
        "transition_active_masks": [_mask(1), _mask(1), _mask(1)],
        "transition_state_masks": [_mask(1), _mask(1), _mask(1)],
    }
    counters = rollout_counters(output, model=_WindowModeStub("current"), batch_size=1)

    assert counters["real_evidence_attempts"] == 2
    assert counters[ORDINARY_REAL_EVIDENCE_KEY] == 2
    # Derived structurally, not from a published accepted_path.
    assert counters["ordinary_accepted_path_from_diagnostics"] == 0.0


def test_a_rejected_restitution_row_cannot_create_ordinary_evidence() -> None:
    """A HALTed row contributes to neither evidence counter."""

    output = {
        "logits": torch.zeros(2, 4, 2, 2),
        "transition_active_masks": [_mask(1, 1), _mask(1, 0)],
        "transition_state_masks": [_mask(1, 0), _mask(0, 0)],
        "candidate_c_diagnostics": [
            _diagnostics_row(0, 0, "ordinary"),
            {**_diagnostics_row(0, 1, "ordinary"), "accepted": False},
        ],
    }
    counters = rollout_counters(output, model=_WindowModeStub("candidate_c"), batch_size=2)

    assert counters["accepted_history_count"] == 1
    assert counters["real_evidence_attempts"] == 1
    assert counters[ORDINARY_REAL_EVIDENCE_KEY] == 1


# ---------------------------------------------------------------------------
# Auxiliary auditor path cost
# ---------------------------------------------------------------------------


def test_auditor_exposure_encodes_once_and_reuses_the_collected_features() -> None:
    """One encoder forward, and the Auditor sees exactly those shared features."""

    config = _config(predicted_history_exposure=True, predicted_history_weight=0.5)
    model = _tiny_net(26)
    model.eval()

    encode_calls: list[torch.Tensor] = []
    original_encode = model.encode

    def counting_encode(images: torch.Tensor) -> Any:
        encoded = original_encode(images)
        encode_calls.append(encoded["shared"])
        return encoded

    model.encode = counting_encode

    # The official gate runs inside infer under no_grad; the loss passes run
    # with gradients enabled.  Recording the grad mode separates the two.
    auditor_calls: list[tuple[torch.Tensor, bool]] = []
    original_forward = model.auditor.forward

    def capturing_forward(features: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        auditor_calls.append((features, torch.is_grad_enabled()))
        return original_forward(features, *args, **kwargs)

    model.auditor.forward = capturing_forward

    try:
        # tau below every delta_q keeps all rows active for every turn, so the
        # feature tensor is used whole and must not be copied per turn.
        auxiliary, counters = compute_predicted_history_audit_loss(
            model,
            _batch(),
            tau_accept=-1.0,
            t_max=config.training.rollout.max_turns,
            neutral_margin=config.training.audit_loss.neutral_margin,
            local_weighting=True,
            audit_margin=config.training.audit_loss.audit_margin,
        )
    finally:
        del model.encode
        del model.auditor.forward

    assert auxiliary is not None
    # Exactly one encoder forward for the whole auxiliary path: the rollout's.
    assert len(encode_calls) == 1

    shared = encode_calls[0]
    assert counters["rollout_batches"] == 1
    gate_passes = [features for features, grad in auditor_calls if not grad]
    loss_passes = [features for features, grad in auditor_calls if grad]

    # Documented added cost: the official Auditor collection passes plus an
    # equal number of gradient-bearing loss passes over the same transitions.
    assert gate_passes, "the official acceptance gate did not run"
    assert loss_passes, "no gradient-bearing auditor pass ran"
    assert len(gate_passes) == len(loss_passes)

    for supplied in loss_passes:
        assert supplied.shape == shared.shape
        assert torch.equal(supplied, shared)
        # Same storage as the rollout's own features: no second encode, and the
        # full feature map is not cloned once per turn.
        assert supplied.data_ptr() == shared.data_ptr()
    # The gate saw the identical values (infer row-selects its own copy).
    for supplied in gate_passes:
        assert torch.equal(supplied, shared)


def test_auditor_exposure_refuses_to_invent_shared_features() -> None:
    """A rollout output without shared features must fail, not use raw images."""

    class _NoSharedFeatures(SelfAuditNet):
        def infer(self, images: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
            output = dict(super().infer(images, **kwargs))
            output.pop("shared_features", None)
            return output

    torch.manual_seed(27)
    model = _NoSharedFeatures(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=2,
    ).eval()

    with pytest.raises(KeyError, match="shared_features"):
        compute_predicted_history_audit_loss(model, _batch(), tau_accept=0.0, t_max=2)


def test_two_turn_ordinary_then_candidate_c_consumes_evidence_in_the_solver() -> None:
    """Previous ordinary -> current C: the SOLVER consumes the evidence.

    The asymmetric case the totals-only fixture cannot catch.  Both samples
    consume real accepted evidence at turn 1, so ``real_evidence_attempts`` is
    2, but the receiving forward at turn 1 is the Candidate-C solver, not the
    trainable ordinary generator, so the ordinary counter is 0.
    """

    output = {
        "logits": torch.zeros(2, 4, 2, 2),
        "transition_active_masks": [_mask(1, 1), _mask(1, 1)],
        "transition_state_masks": [_mask(1, 1), _mask(1, 1)],
        "candidate_c_diagnostics": [
            _diagnostics_row(0, 0, "ordinary"),
            _diagnostics_row(0, 1, "ordinary"),
            _diagnostics_row(1, 0, "counterfactual"),
            _diagnostics_row(1, 1, "counterfactual"),
        ],
    }
    counters = rollout_counters(output, model=_WindowModeStub("candidate_c"), batch_size=2)

    assert counters["real_evidence_attempts"] == 2
    assert counters[ORDINARY_REAL_EVIDENCE_KEY] == 0
    assert counters["ordinary_accepted_path_from_diagnostics"] == 1.0
    assert _per_turn_ordinary_evidence(output) == [0, 0]


def test_three_turn_ordinary_c_ordinary_places_the_exposure_on_the_last_turn() -> None:
    """Previous C -> current ordinary: the trainable GENERATOR consumes it.

    Per-turn contributions must be ``[0, 0, 2]``.  A reversed convention that
    keyed on the previous turn's path would produce ``[0, 2, 0]`` -- the same
    total, the wrong turn, and the wrong consuming module.
    """

    output = {
        "logits": torch.zeros(2, 4, 2, 2),
        "transition_active_masks": [_mask(1, 1), _mask(1, 1), _mask(1, 1)],
        "transition_state_masks": [_mask(1, 1), _mask(1, 1), _mask(1, 1)],
        "candidate_c_diagnostics": [
            _diagnostics_row(0, 0, "ordinary"),
            _diagnostics_row(0, 1, "ordinary"),
            _diagnostics_row(1, 0, "counterfactual"),
            _diagnostics_row(1, 1, "counterfactual"),
            _diagnostics_row(2, 0, "ordinary"),
            _diagnostics_row(2, 1, "ordinary"),
        ],
    }
    counters = rollout_counters(output, model=_WindowModeStub("candidate_c"), batch_size=2)

    assert _per_turn_ordinary_evidence(output) == [0, 0, 2]
    assert counters["real_evidence_attempts"] == 4
    assert counters[ORDINARY_REAL_EVIDENCE_KEY] == 2


def test_c1_failure_routed_to_ordinary_counts_as_both_attempt_and_exposure() -> None:
    """A failed replay still fed the real previous predicted evidence forward.

    The turn attempted a record-consuming C solve (so it belongs in the C
    denominator and increments ``candidate_c_replay_failure``) and was routed
    back to normal annotation, which really did consume the previous accepted
    predicted evidence in the trainable generator.
    """

    output = {
        "logits": torch.zeros(1, 4, 2, 2),
        "transition_active_masks": [_mask(1), _mask(1)],
        "transition_state_masks": [_mask(1), _mask(1)],
        "candidate_c_diagnostics": [
            _diagnostics_row(0, 0, "ordinary"),
            {
                **_diagnostics_row(1, 0, "ordinary"),
                "eligible": False,
                "c1_passed": False,
                "feasible": False,
                "fallback_reason": "replay_failure",
            },
        ],
    }
    counters = rollout_counters(output, model=_WindowModeStub("candidate_c"), batch_size=1)

    assert counters["candidate_c_attempts"] == 1
    assert counters["candidate_c_replay_failure"] == 1
    assert counters["candidate_c_fallback"] == 1
    assert counters["candidate_c_feasible"] == 0
    # Both counted: the solve was attempted AND the ordinary generator consumed
    # the previous predicted evidence.
    assert counters["real_evidence_attempts"] == 1
    assert counters[ORDINARY_REAL_EVIDENCE_KEY] == 1
    assert _per_turn_ordinary_evidence(output) == [0, 1]
