"""Stage-wise attribution metrics for the Self-Audit branch.

This module is evaluation-only. Ground truth is used strictly after deployable
inference to measure candidate headroom and the value added by the audit gate.

Aggregation contract
--------------------
Every rate and ratio in this module is computed **exactly once**, at the end,
over the full population of per-sample / per-transition values.  Nothing here
averages per-batch scalars, so all reported numbers are invariant to how a
loader happens to be partitioned into batches.  The one decision margin comes
from :mod:`self_audit.audit.semantics`; this module contains no local
improve/regress threshold of its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from self_audit.audit.semantics import (
    BENEFICIAL,
    HARMFUL,
    NEUTRAL,
    classify_delta,
    resolve_neutral_margin,
)
from self_audit.audit.targets import multiclass_dice
from self_audit.training._utils import move_batch


def _tensor_from(output: Mapping[str, Any], *names: str) -> Tensor | None:
    for name in names:
        value = output.get(name)
        if torch.is_tensor(value):
            return value
    return None


def _transition_value(value: Any, index: int) -> Any | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.ndim == 1:
            return value if index == 0 else None
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return None


def _bool_mask(
    value: Any | None,
    *,
    batch_size: int,
    device: torch.device,
    default: bool,
) -> Tensor:
    if value is None:
        return torch.full((batch_size,), bool(default), dtype=torch.bool, device=device)
    result = value if torch.is_tensor(value) else torch.as_tensor(value, device=device)
    result = result.to(device=device, dtype=torch.bool).reshape(-1)
    if result.numel() != batch_size:
        raise ValueError(
            f"Expected transition mask with {batch_size} rows, got {result.numel()}"
        )
    return result


def _select_rows(value: Tensor, mask: Tensor) -> Tensor:
    if value.shape[0] == mask.numel():
        return value.index_select(0, mask.nonzero(as_tuple=False).flatten().to(value.device))
    if value.shape[0] == int(mask.sum().item()):
        return value
    raise ValueError(
        f"Transition rows {value.shape[0]} do not match batch {mask.numel()} "
        f"or active count {int(mask.sum().item())}"
    )


def _safe_rate(numerator: Tensor, denominator: Tensor) -> float:
    count = int(denominator.sum().item())
    if count == 0:
        return float("nan")
    return float((numerator & denominator).sum().item() / count)


def _mean_or_nan(value: Tensor) -> float:
    return float(value.double().mean().item()) if value.numel() else float("nan")


def _rate(count: int, total: int) -> float:
    """Fraction, or ``nan`` when the population is empty (never a silent 0.0)."""

    return float(count) / float(total) if total > 0 else float("nan")


def _classify_masks(
    delta: Tensor, neutral_margin: float | None
) -> tuple[Tensor, Tensor, Tensor]:
    """Beneficial / neutral / harmful masks from the single shared rule.

    Delegates to :func:`self_audit.audit.semantics.classify_delta` so this
    module holds no second implementation of the decision margin.
    """

    codes = classify_delta(delta.detach().cpu().numpy(), neutral_margin)
    codes_tensor = torch.as_tensor(codes, device=delta.device)
    return (
        codes_tensor == BENEFICIAL,
        codes_tensor == NEUTRAL,
        codes_tensor == HARMFUL,
    )


# --------------------------------------------------------------------------
# Raw per-transition / per-sample containers
# --------------------------------------------------------------------------


@dataclass
class StageTransition:
    """Raw per-transition rows for one refinement stage.

    ``delta``, ``accepted``, ``previous_dice`` and ``candidate_dice`` all carry
    ``attempt_count`` rows -- the *active* rows only.  ``row_count`` is the
    number of batch rows the stage was offered to, so ``attempt_rate`` can be
    recomputed correctly after concatenating several batches.
    """

    delta: Tensor
    accepted: Tensor
    previous_dice: Tensor
    candidate_dice: Tensor
    attempt_count: int
    row_count: int

    @classmethod
    def empty(cls, device: torch.device, *, row_count: int = 0) -> "StageTransition":
        zeros = torch.zeros((0,), dtype=torch.float32, device=device)
        return cls(
            delta=zeros,
            accepted=torch.zeros((0,), dtype=torch.bool, device=device),
            previous_dice=zeros.clone(),
            candidate_dice=zeros.clone(),
            attempt_count=0,
            row_count=int(row_count),
        )

    def merged_with(self, other: "StageTransition") -> "StageTransition":
        return StageTransition(
            delta=torch.cat([self.delta, other.delta.to(self.delta.device)]),
            accepted=torch.cat([self.accepted, other.accepted.to(self.accepted.device)]),
            previous_dice=torch.cat(
                [self.previous_dice, other.previous_dice.to(self.previous_dice.device)]
            ),
            candidate_dice=torch.cat(
                [self.candidate_dice, other.candidate_dice.to(self.candidate_dice.device)]
            ),
            attempt_count=self.attempt_count + other.attempt_count,
            row_count=self.row_count + other.row_count,
        )


@dataclass
class AuditModeSamples:
    """Per-sample mode scores plus the raw per-stage transition rows."""

    initial_dice: Tensor
    always_dice: Tensor
    self_dice: Tensor
    oracle_dice: Tensor
    stages: list[StageTransition] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return int(self.initial_dice.numel())

    @property
    def stage_deltas(self) -> list[Tensor]:
        return [stage.delta for stage in self.stages]

    @property
    def stage_accepted(self) -> list[Tensor]:
        return [stage.accepted for stage in self.stages]

    @property
    def attempt_counts(self) -> list[int]:
        return [stage.attempt_count for stage in self.stages]

    @property
    def row_counts(self) -> list[int]:
        return [stage.row_count for stage in self.stages]

    def merged_with(self, other: "AuditModeSamples") -> "AuditModeSamples":
        stages: list[StageTransition] = []
        device = self.initial_dice.device
        for index in range(max(len(self.stages), len(other.stages))):
            left = self.stages[index] if index < len(self.stages) else StageTransition.empty(device)
            right = other.stages[index] if index < len(other.stages) else StageTransition.empty(device)
            stages.append(left.merged_with(right))
        return AuditModeSamples(
            initial_dice=torch.cat([self.initial_dice, other.initial_dice.to(device)]),
            always_dice=torch.cat([self.always_dice, other.always_dice.to(device)]),
            self_dice=torch.cat([self.self_dice, other.self_dice.to(device)]),
            oracle_dice=torch.cat([self.oracle_dice, other.oracle_dice.to(device)]),
            stages=stages,
        )


# --------------------------------------------------------------------------
# Stage-wise attribution
# --------------------------------------------------------------------------

#: Keys emitted as ``nan`` for a stage that was never attempted.
_STAGE_NAN_KEYS = (
    "prev_dice",
    "candidate_dice",
    "candidate_gain",
    "headroom",
    "positive_headroom",
    "realized_gain",
    "audit_gate_value",
    "accept_rate",
    "reject_rate",
    "audit_involvement_rate",
    "beneficial_candidate_rate",
    "harmful_candidate_rate",
    "neutral_candidate_rate",
    "beneficial_capture_rate",
    "harmful_block_rate",
    "attribution_residual",
)


def extract_stage_transitions(
    output: Mapping[str, Any],
    ground_truth: Tensor,
) -> list[StageTransition]:
    """Pull the raw per-transition rows out of one ``self_audit`` inference output.

    No thresholds and no aggregation happen here -- this is the tensor half of
    :func:`decompose_self_audit_output`, split out so a loader-level caller can
    concatenate rows across batches and reduce exactly once.
    """

    if not isinstance(output, Mapping):
        raise TypeError("Self-Audit decomposition requires a mapping inference output")
    if ground_truth.ndim != 3:
        raise ValueError(f"Expected ground_truth [B,H,W], got {tuple(ground_truth.shape)}")

    previous_states = output.get("transition_previous", [])
    candidate_states = output.get("transition_candidates", output.get("candidates", []))
    audits = output.get("audits", [])
    active_values = output.get("transition_active_masks", output.get("active_masks"))
    accepted_values = output.get("transition_state_masks", output.get("state_masks"))

    batch_size = int(ground_truth.shape[0])
    device = ground_truth.device

    stages: list[StageTransition] = []
    transition_count = min(len(previous_states), len(candidate_states))
    for stage in range(transition_count):
        previous = previous_states[stage]
        candidate = candidate_states[stage]
        audit = audits[stage] if stage < len(audits) else None

        active_raw = _transition_value(active_values, stage)
        if active_raw is None and isinstance(audit, Mapping):
            active_raw = audit.get("active_mask", audit.get("audit_mask"))
        active = _bool_mask(active_raw, batch_size=batch_size, device=device, default=True)
        active_count = int(active.sum().item())

        if active_count == 0:
            stages.append(StageTransition.empty(device, row_count=batch_size))
            continue

        previous_active = _select_rows(previous.detach(), active)
        candidate_active = _select_rows(candidate.detach(), active)
        target_active = _select_rows(ground_truth.detach(), active)

        previous_dice = multiclass_dice(previous_active, target_active)
        candidate_dice = multiclass_dice(candidate_active, target_active)

        accepted_raw = _transition_value(accepted_values, stage)
        if accepted_raw is None and isinstance(audit, Mapping):
            accepted_raw = audit.get("state_mask", audit.get("accepted"))
        accepted_full = _bool_mask(
            accepted_raw, batch_size=batch_size, device=device, default=False
        )
        accepted = _select_rows(accepted_full, active)

        stages.append(
            StageTransition(
                delta=candidate_dice - previous_dice,
                accepted=accepted,
                previous_dice=previous_dice,
                candidate_dice=candidate_dice,
                attempt_count=active_count,
                row_count=batch_size,
            )
        )
    return stages


def _stage_metrics(
    prefix: str,
    stage: StageTransition,
    neutral_margin: float,
) -> dict[str, float]:
    result: dict[str, float] = {
        f"{prefix}/attempt_rate": _rate(stage.attempt_count, stage.row_count),
        f"{prefix}/attempt_count": float(stage.attempt_count),
    }
    if stage.attempt_count == 0:
        for key in _STAGE_NAN_KEYS:
            result[f"{prefix}/{key}"] = float("nan")
        return result

    delta = stage.delta
    accepted = stage.accepted
    beneficial, neutral, harmful = _classify_masks(delta, neutral_margin)

    realized = delta * accepted.to(delta.dtype)
    gate_value = realized - delta
    headroom = torch.clamp(delta, min=0.0)
    accepted_count = int(accepted.sum().item())
    total = int(delta.numel())

    result[f"{prefix}/prev_dice"] = _mean_or_nan(stage.previous_dice)
    result[f"{prefix}/candidate_dice"] = _mean_or_nan(stage.candidate_dice)
    result[f"{prefix}/candidate_gain"] = _mean_or_nan(delta)
    result[f"{prefix}/headroom"] = _mean_or_nan(headroom)
    # Renamed from ``oracle_gain``: this is clamp(delta, min=0) measured on the
    # SELF-AUDIT trajectory, not on the oracle trajectory.  See module notes.
    result[f"{prefix}/positive_headroom"] = result[f"{prefix}/headroom"]
    result[f"{prefix}/realized_gain"] = _mean_or_nan(realized)
    result[f"{prefix}/audit_gate_value"] = _mean_or_nan(gate_value)
    result[f"{prefix}/accept_rate"] = _rate(accepted_count, total)
    result[f"{prefix}/reject_rate"] = _rate(total - accepted_count, total)
    result[f"{prefix}/audit_involvement_rate"] = result[f"{prefix}/reject_rate"]
    result[f"{prefix}/beneficial_candidate_rate"] = _rate(int(beneficial.sum().item()), total)
    result[f"{prefix}/harmful_candidate_rate"] = _rate(int(harmful.sum().item()), total)
    result[f"{prefix}/neutral_candidate_rate"] = _rate(int(neutral.sum().item()), total)
    result[f"{prefix}/beneficial_capture_rate"] = _safe_rate(accepted, beneficial)
    result[f"{prefix}/harmful_block_rate"] = _safe_rate(~accepted, harmful)
    result[f"{prefix}/attribution_residual"] = abs(
        result[f"{prefix}/candidate_gain"]
        + result[f"{prefix}/audit_gate_value"]
        - result[f"{prefix}/realized_gain"]
    )
    return result


def stage_transition_metrics(
    stages: Sequence[StageTransition],
    *,
    neutral_margin: float | None = None,
) -> dict[str, float]:
    """Reduce already-accumulated transition rows to the ``stage_*``/``audit/`` metrics."""

    margin = resolve_neutral_margin(neutral_margin)
    result: dict[str, float] = {}
    for index, stage in enumerate(stages):
        result.update(_stage_metrics(f"stage_{index}", stage, margin))

    populated = [stage for stage in stages if stage.attempt_count > 0]
    if not populated:
        result["audit/transition_count"] = 0.0
        return result

    device = populated[0].delta.device
    delta = torch.cat([stage.delta.to(device) for stage in populated])
    accepted = torch.cat([stage.accepted.to(device) for stage in populated])
    beneficial, neutral, harmful = _classify_masks(delta, margin)
    realized = delta * accepted.to(delta.dtype)
    gate_value = realized - delta
    total = int(delta.numel())
    accepted_count = int(accepted.sum().item())

    candidate_gain = _mean_or_nan(delta)
    audit_gate_value = _mean_or_nan(gate_value)
    realized_gain = _mean_or_nan(realized)
    result.update(
        {
            "audit/transition_count": float(total),
            "audit/candidate_gain": candidate_gain,
            "audit/headroom": _mean_or_nan(torch.clamp(delta, min=0.0)),
            "audit/positive_headroom": _mean_or_nan(torch.clamp(delta, min=0.0)),
            "audit/realized_gain": realized_gain,
            "audit/gate_value": audit_gate_value,
            "audit/accept_rate": _rate(accepted_count, total),
            "audit/involvement_rate": _rate(total - accepted_count, total),
            "audit/beneficial_candidate_rate": _rate(int(beneficial.sum().item()), total),
            "audit/harmful_candidate_rate": _rate(int(harmful.sum().item()), total),
            "audit/neutral_candidate_rate": _rate(int(neutral.sum().item()), total),
            "audit/beneficial_capture_rate": _safe_rate(accepted, beneficial),
            "audit/harmful_block_rate": _safe_rate(~accepted, harmful),
            "audit/attribution_residual": abs(
                candidate_gain + audit_gate_value - realized_gain
            ),
        }
    )
    return result


def decompose_self_audit_output(
    output: Mapping[str, Any],
    ground_truth: Tensor,
    *,
    neutral_margin: float | None = None,
) -> dict[str, float]:
    """Attribute annotation-candidate gain and audit-gate value stage by stage.

    For each attempted transition, with oracle-only validation delta
    ``d = Dice(candidate, GT) - Dice(previous, GT)`` and hard gate ``g``:

    candidate_gain = d
    audit_gate_value = (g * d) - d
    realized_gain = g * d

    Therefore ``candidate_gain + audit_gate_value == realized_gain``.
    Positive audit-gate value means the Auditor rescued a harmful proposal;
    negative value means it rejected a beneficial proposal.

    ``neutral_margin=None`` resolves to the canonical
    :data:`self_audit.audit.semantics.DEFAULT_NEUTRAL_MARGIN`.
    """

    stages = extract_stage_transitions(output, ground_truth)
    return stage_transition_metrics(stages, neutral_margin=neutral_margin)


def _foreground_dice(logits: Tensor, target: Tensor) -> Tensor:
    return multiclass_dice(logits, target, num_classes=int(logits.shape[1]))


# --------------------------------------------------------------------------
# Four-mode headroom attribution
# --------------------------------------------------------------------------


def audit_mode_metrics(
    samples: AuditModeSamples,
    *,
    neutral_margin: float | None = None,
) -> dict[str, float]:
    """Reduce accumulated per-sample mode scores to the ``modes/`` metrics.

    ``headroom_capture_ratio`` is a **ratio of sums** restricted to rows with
    *material* headroom (``oracle_headroom > neutral_margin``, strictly
    greater).  The previous mean-of-ratios estimator divided by a ``1e-8``
    floor, so a single row of numerically-zero headroom could drive the
    headline statistic to ~1e6.  With no eligible row the ratio is ``nan``:
    reporting ``0.0`` would claim the audit captured nothing when in fact
    nothing was available to capture.
    """

    margin = resolve_neutral_margin(neutral_margin)
    initial = samples.initial_dice.double()
    always = samples.always_dice.double()
    self_dice = samples.self_dice.double()
    oracle = samples.oracle_dice.double()

    oracle_headroom = oracle - initial
    self_gain = self_dice - initial
    candidate_path_gain = always - initial
    audit_rescue = self_dice - always

    eligible = oracle_headroom > margin
    eligible_count = int(eligible.sum().item())
    total = int(oracle_headroom.numel())
    oracle_headroom_sum = float(oracle_headroom[eligible].sum().item()) if eligible_count else 0.0
    self_gain_sum = float(self_gain[eligible].sum().item()) if eligible_count else 0.0
    capture_ratio = (
        self_gain_sum / oracle_headroom_sum if eligible_count and oracle_headroom_sum != 0.0
        else float("nan")
    )

    return {
        "modes/sample_count": float(total),
        "modes/initial_dice": _mean_or_nan(initial),
        "modes/always_accept_dice": _mean_or_nan(always),
        "modes/self_audit_dice": _mean_or_nan(self_dice),
        "modes/oracle_dice": _mean_or_nan(oracle),
        "modes/candidate_path_gain": _mean_or_nan(candidate_path_gain),
        "modes/audit_rescue_vs_always": _mean_or_nan(audit_rescue),
        "modes/self_audit_gain": _mean_or_nan(self_gain),
        "modes/oracle_headroom": _mean_or_nan(oracle_headroom),
        "modes/headroom_capture_ratio": capture_ratio,
        "modes/headroom_available_rate": _rate(eligible_count, total),
        "modes/headroom_eligible_count": float(eligible_count),
        "modes/oracle_headroom_sum": oracle_headroom_sum,
        "modes/self_gain_on_headroom_sum": self_gain_sum,
    }


@torch.no_grad()
def evaluate_audit_modes_batch(
    model: torch.nn.Module,
    images: Tensor,
    ground_truth: Tensor,
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    neutral_margin: float | None = None,
    return_per_sample: bool = False,
) -> dict[str, float] | tuple[dict[str, float], AuditModeSamples]:
    """Run initial/always/self/oracle on one batch and quantify audit headroom.

    With ``return_per_sample=False`` (the default) the return value is the
    scalar metric dict, unchanged in shape from previous releases.  With
    ``return_per_sample=True`` the return value is
    ``(metrics, AuditModeSamples)``; the samples object carries the raw
    ``[B]`` per-sample mode Dice tensors and the per-stage
    ``delta``/``accepted``/attempt counts, so a loader-level caller can
    concatenate and reduce exactly once.
    """

    margin = resolve_neutral_margin(neutral_margin)

    initial_out = model.infer(images, mode="initial_only", t_max=0)
    always_out = model.infer(
        images,
        mode="always_accept_refinement",
        tau_accept=-float("inf"),
        t_max=int(t_max),
    )
    self_out = model.infer(
        images,
        mode="self_audit",
        tau_accept=float(tau_accept),
        t_max=int(t_max),
    )
    oracle_out = model.infer(
        images,
        mode="oracle_accept",
        oracle_target=ground_truth,
        t_max=int(t_max),
    )

    initial_logits = _tensor_from(initial_out, "initial_logits", "a0_logits", "logits")
    always_logits = _tensor_from(always_out, "logits")
    self_logits = _tensor_from(self_out, "logits")
    oracle_logits = _tensor_from(oracle_out, "logits")
    if any(value is None for value in (initial_logits, always_logits, self_logits, oracle_logits)):
        raise ValueError("One or more inference modes did not return annotation logits")

    samples = AuditModeSamples(
        initial_dice=_foreground_dice(initial_logits, ground_truth),
        always_dice=_foreground_dice(always_logits, ground_truth),
        self_dice=_foreground_dice(self_logits, ground_truth),
        oracle_dice=_foreground_dice(oracle_logits, ground_truth),
        stages=extract_stage_transitions(self_out, ground_truth),
    )

    result = audit_mode_metrics(samples, neutral_margin=margin)
    result.update(stage_transition_metrics(samples.stages, neutral_margin=margin))
    if return_per_sample:
        return result, samples
    return result


def _aggregate_gt_firewall(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Combine per-batch firewall probes without averaging rates."""

    if not rows:
        return {}
    max_abs = max(float(row.get("gt_firewall/max_abs_logit_diff", 0.0)) for row in rows)
    mismatches = sum(float(row.get("gt_firewall/decision_mismatch_count", 0.0)) for row in rows)
    total = sum(float(row.get("gt_firewall/decision_total", 0.0)) for row in rows)
    mismatch_rate = mismatches / total if total > 0 else 0.0
    return {
        "gt_firewall/num_batches_checked": float(len(rows)),
        "gt_firewall/max_abs_logit_diff": max_abs,
        "gt_firewall/decision_mismatch_count": mismatches,
        "gt_firewall/decision_total": total,
        "gt_firewall/decision_mismatch_rate": float(mismatch_rate),
        "gt_firewall/passed": float(max_abs <= 1e-7 and mismatch_rate == 0.0),
    }


@torch.no_grad()
def evaluate_audit_decomposition(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    neutral_margin: float | None = None,
    max_batches: int | None = None,
    disable_tqdm: bool = False,
    gt_probe_batches: int = 1,
) -> dict[str, float]:
    """Aggregate stage-wise audit attribution over a validation loader.

    Per-sample and per-transition rows are accumulated across the whole loader
    and every rate and ratio is computed once at the end, so the result is
    invariant to the loader's batch partition.  Nothing is averaged per batch
    and no batch is silently dropped for being ``nan``.
    """

    margin = resolve_neutral_margin(neutral_margin)
    was_training = model.training
    model.eval()
    accumulated: AuditModeSamples | None = None
    firewall_rows: list[dict[str, float]] = []
    try:
        pbar = tqdm(loader, desc="Audit decomposition", disable=disable_tqdm, leave=False)
        for batch_index, raw_batch in enumerate(pbar):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch(raw_batch, device)
            _, samples = evaluate_audit_modes_batch(
                model,
                batch["image"],
                batch["mask"],
                tau_accept=float(tau_accept),
                t_max=int(t_max),
                neutral_margin=margin,
                return_per_sample=True,
            )
            accumulated = samples if accumulated is None else accumulated.merged_with(samples)
            if batch_index < int(gt_probe_batches):
                firewall_rows.append(
                    probe_gt_leakage(
                        model,
                        batch["image"],
                        batch["mask"],
                        tau_accept=float(tau_accept),
                        t_max=int(t_max),
                    )
                )
    finally:
        model.train(was_training)

    if accumulated is None:
        return {}

    result = audit_mode_metrics(accumulated, neutral_margin=margin)
    result.update(stage_transition_metrics(accumulated.stages, neutral_margin=margin))
    result.update(_aggregate_gt_firewall(firewall_rows))
    return result


@torch.no_grad()
def probe_gt_leakage(
    model: torch.nn.Module,
    images: Tensor,
    ground_truth: Tensor,
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
) -> dict[str, float]:
    """Dynamic firewall probe: permuting GT must not change self-audit inference.

    ``oracle_target`` exists for the analysis-only ``oracle_accept`` mode.
    This probe deliberately supplies two different oracle tensors while
    running ``mode='self_audit'``. A non-zero difference means deployable
    inference has accidentally become GT-dependent.
    """

    if ground_truth.shape[0] != images.shape[0]:
        raise ValueError("ground_truth and images must have the same batch size")
    if ground_truth.shape[0] > 1:
        permutation = torch.arange(ground_truth.shape[0] - 1, -1, -1, device=ground_truth.device)
        shuffled = ground_truth.index_select(0, permutation)
    else:
        shuffled = torch.roll(ground_truth, shifts=1, dims=-1)

    first = model.infer(
        images,
        mode="self_audit",
        tau_accept=float(tau_accept),
        t_max=int(t_max),
        oracle_target=ground_truth,
    )
    second = model.infer(
        images,
        mode="self_audit",
        tau_accept=float(tau_accept),
        t_max=int(t_max),
        oracle_target=shuffled,
    )
    logits_a = _tensor_from(first, "logits")
    logits_b = _tensor_from(second, "logits")
    if logits_a is None or logits_b is None:
        raise ValueError("Leakage probe requires final logits")
    max_abs = float((logits_a - logits_b).abs().max().item())

    accepted_a = _tensor_from(first, "accepted_count")
    accepted_b = _tensor_from(second, "accepted_count")
    halt_a = _tensor_from(first, "halt_turn")
    halt_b = _tensor_from(second, "halt_turn")
    decision_mismatches = 0
    decision_total = 0
    for left, right in ((accepted_a, accepted_b), (halt_a, halt_b)):
        if left is not None and right is not None:
            decision_mismatches += int((left != right).sum().item())
            decision_total += int(left.numel())
    mismatch_rate = decision_mismatches / max(decision_total, 1)

    return {
        "gt_firewall/max_abs_logit_diff": max_abs,
        "gt_firewall/decision_mismatch_count": float(decision_mismatches),
        "gt_firewall/decision_total": float(decision_total),
        "gt_firewall/decision_mismatch_rate": float(mismatch_rate),
        "gt_firewall/passed": float(max_abs <= 1e-7 and mismatch_rate == 0.0),
    }


# --------------------------------------------------------------------------
# Phase-A annotation headroom
# --------------------------------------------------------------------------


def extract_annotation_scores(
    output: Mapping[str, Any],
    ground_truth: Tensor,
) -> list[Tensor]:
    """Return the per-sample Dice of every state on the annotation trajectory."""

    trace = output.get("state_trace", output.get("all_states"))
    if not isinstance(trace, (list, tuple)) or not trace:
        initial = _tensor_from(output, "initial_logits", "a0_logits", "A0_logits")
        states = output.get("states", [])
        trace = ([initial] if initial is not None else []) + list(states)
    trace = [value for value in trace if torch.is_tensor(value)]
    if not trace:
        raise ValueError("Annotation output contains no trajectory")
    return [multiclass_dice(value.detach(), ground_truth) for value in trace]


def annotation_headroom_metrics(
    scores: Sequence[Tensor],
    *,
    neutral_margin: float | None = None,
) -> dict[str, float]:
    """Reduce accumulated per-state Dice to the ``phase_a/`` metrics."""

    if not scores:
        raise ValueError("Annotation headroom requires at least one trajectory state")
    margin = resolve_neutral_margin(neutral_margin)
    result: dict[str, float] = {
        "phase_a/sample_count": float(scores[0].numel()),
        "phase_a/a0_dice": _mean_or_nan(scores[0]),
        "phase_a/final_dice": _mean_or_nan(scores[-1]),
        "phase_a/total_refinement_gain": _mean_or_nan(scores[-1] - scores[0]),
    }
    positive_mass: list[Tensor] = []
    for stage, (previous_score, candidate_score) in enumerate(zip(scores[:-1], scores[1:])):
        delta = candidate_score - previous_score
        beneficial, neutral, harmful = _classify_masks(delta, margin)
        headroom = torch.clamp(delta, min=0.0)
        positive_mass.append(headroom)
        total = int(delta.numel())
        prefix = f"phase_a/stage_{stage}"
        result[f"{prefix}/prev_dice"] = _mean_or_nan(previous_score)
        result[f"{prefix}/candidate_dice"] = _mean_or_nan(candidate_score)
        result[f"{prefix}/candidate_gain"] = _mean_or_nan(delta)
        result[f"{prefix}/headroom"] = _mean_or_nan(headroom)
        result[f"{prefix}/beneficial_rate"] = _rate(int(beneficial.sum().item()), total)
        result[f"{prefix}/harmful_rate"] = _rate(int(harmful.sum().item()), total)
        result[f"{prefix}/neutral_rate"] = _rate(int(neutral.sum().item()), total)
    if positive_mass:
        mean_headroom = _mean_or_nan(torch.cat(positive_mass))
        result["phase_a/mean_stage_headroom"] = mean_headroom
        result["phase_a/neutral_margin"] = float(margin)
        # Fraction of the decision margin that the mean stage headroom fails to
        # reach.  Uses the *passed* margin -- it used to divide by a literal
        # 0.005, so any other margin silently desynchronised this one metric.
        if margin > 0.0:
            result["phase_a/headroom_collapse_ratio"] = float(
                1.0 - min(max(mean_headroom / margin, 0.0), 1.0)
            )
        else:
            result["phase_a/headroom_collapse_ratio"] = float("nan")
    return result


def decompose_annotation_output(
    output: Mapping[str, Any],
    ground_truth: Tensor,
    *,
    neutral_margin: float | None = None,
) -> dict[str, float]:
    """Measure A0 saturation and available recurrent correction headroom."""

    scores = extract_annotation_scores(output, ground_truth)
    return annotation_headroom_metrics(scores, neutral_margin=neutral_margin)


@torch.no_grad()
def evaluate_annotation_headroom(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    neutral_margin: float | None = None,
    max_batches: int | None = None,
    disable_tqdm: bool = False,
) -> dict[str, float]:
    """Aggregate Phase-A A0/refinement headroom over validation data.

    Like :func:`evaluate_audit_decomposition`, per-sample scores are
    concatenated across the loader and reduced once, so the result does not
    depend on the batch partition.
    """

    margin = resolve_neutral_margin(neutral_margin)
    was_training = model.training
    model.eval()
    accumulated: list[list[Tensor]] = []
    try:
        pbar = tqdm(loader, desc="Annotation headroom", disable=disable_tqdm, leave=False)
        for batch_index, raw_batch in enumerate(pbar):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch(raw_batch, device)
            output = model.forward_annotation(batch["image"])
            scores = extract_annotation_scores(output, batch["mask"])
            if not accumulated:
                accumulated = [[value] for value in scores]
            elif len(scores) != len(accumulated):
                raise ValueError(
                    "Annotation trajectory length changed across batches "
                    f"({len(accumulated)} then {len(scores)}); cannot aggregate exactly"
                )
            else:
                for bucket, value in zip(accumulated, scores):
                    bucket.append(value)
    finally:
        model.train(was_training)
    if not accumulated:
        return {}
    return annotation_headroom_metrics(
        [torch.cat(bucket) for bucket in accumulated],
        neutral_margin=margin,
    )


__all__ = [
    "AuditModeSamples",
    "StageTransition",
    "annotation_headroom_metrics",
    "audit_mode_metrics",
    "decompose_annotation_output",
    "decompose_self_audit_output",
    "evaluate_annotation_headroom",
    "evaluate_audit_decomposition",
    "evaluate_audit_modes_batch",
    "extract_annotation_scores",
    "extract_stage_transitions",
    "probe_gt_leakage",
    "stage_transition_metrics",
]
