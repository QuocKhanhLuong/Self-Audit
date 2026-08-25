"""Stage-wise attribution metrics for the Self-Audit branch.

This module is evaluation-only. Ground truth is used strictly after deployable
inference to measure candidate headroom and the value added by the audit gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    return float(value.float().mean().item()) if value.numel() else float("nan")


def decompose_self_audit_output(
    output: Mapping[str, Any],
    ground_truth: Tensor,
    *,
    neutral_margin: float = 0.005,
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
    result: dict[str, float] = {}

    all_delta: list[Tensor] = []
    all_realized: list[Tensor] = []
    all_gate_value: list[Tensor] = []
    all_accepted: list[Tensor] = []
    all_beneficial: list[Tensor] = []
    all_harmful: list[Tensor] = []
    all_neutral: list[Tensor] = []

    transition_count = min(len(previous_states), len(candidate_states))
    for stage in range(transition_count):
        previous = previous_states[stage]
        candidate = candidate_states[stage]
        audit = audits[stage] if stage < len(audits) else None

        active_raw = _transition_value(active_values, stage)
        if active_raw is None and isinstance(audit, Mapping):
            active_raw = audit.get("active_mask", audit.get("audit_mask"))
        active = _bool_mask(
            active_raw,
            batch_size=batch_size,
            device=device,
            default=True,
        )
        active_count = int(active.sum().item())
        prefix = f"stage_{stage}"
        result[f"{prefix}/attempt_rate"] = active_count / max(batch_size, 1)
        result[f"{prefix}/attempt_count"] = float(active_count)

        if active_count == 0:
            for key in (
                "prev_dice",
                "candidate_dice",
                "candidate_gain",
                "headroom",
                "oracle_gain",
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
            ):
                result[f"{prefix}/{key}"] = float("nan")
            continue

        previous_active = _select_rows(previous.detach(), active)
        candidate_active = _select_rows(candidate.detach(), active)
        target_active = _select_rows(ground_truth.detach(), active)

        previous_dice = multiclass_dice(previous_active, target_active)
        candidate_dice = multiclass_dice(candidate_active, target_active)
        delta = candidate_dice - previous_dice

        accepted_raw = _transition_value(accepted_values, stage)
        if accepted_raw is None and isinstance(audit, Mapping):
            accepted_raw = audit.get("state_mask", audit.get("accepted"))
        accepted_full = _bool_mask(
            accepted_raw,
            batch_size=batch_size,
            device=device,
            default=False,
        )
        accepted = _select_rows(accepted_full, active)
        accepted_float = accepted.to(delta.dtype)

        beneficial = delta > float(neutral_margin)
        harmful = delta < -float(neutral_margin)
        neutral = ~(beneficial | harmful)

        realized = delta * accepted_float
        gate_value = realized - delta
        headroom = torch.clamp(delta, min=0.0)

        result[f"{prefix}/prev_dice"] = _mean_or_nan(previous_dice)
        result[f"{prefix}/candidate_dice"] = _mean_or_nan(candidate_dice)
        result[f"{prefix}/candidate_gain"] = _mean_or_nan(delta)
        result[f"{prefix}/headroom"] = _mean_or_nan(headroom)
        result[f"{prefix}/oracle_gain"] = _mean_or_nan(headroom)
        result[f"{prefix}/realized_gain"] = _mean_or_nan(realized)
        result[f"{prefix}/audit_gate_value"] = _mean_or_nan(gate_value)
        result[f"{prefix}/accept_rate"] = float(accepted.float().mean().item())
        result[f"{prefix}/reject_rate"] = float((~accepted).float().mean().item())
        result[f"{prefix}/audit_involvement_rate"] = result[f"{prefix}/reject_rate"]
        result[f"{prefix}/beneficial_candidate_rate"] = float(beneficial.float().mean().item())
        result[f"{prefix}/harmful_candidate_rate"] = float(harmful.float().mean().item())
        result[f"{prefix}/neutral_candidate_rate"] = float(neutral.float().mean().item())
        result[f"{prefix}/beneficial_capture_rate"] = _safe_rate(accepted, beneficial)
        result[f"{prefix}/harmful_block_rate"] = _safe_rate(~accepted, harmful)

        result[f"{prefix}/attribution_residual"] = abs(
            result[f"{prefix}/candidate_gain"]
            + result[f"{prefix}/audit_gate_value"]
            - result[f"{prefix}/realized_gain"]
        )

        all_delta.append(delta)
        all_realized.append(realized)
        all_gate_value.append(gate_value)
        all_accepted.append(accepted)
        all_beneficial.append(beneficial)
        all_harmful.append(harmful)
        all_neutral.append(neutral)

    if all_delta:
        delta = torch.cat(all_delta)
        realized = torch.cat(all_realized)
        gate_value = torch.cat(all_gate_value)
        accepted = torch.cat(all_accepted)
        beneficial = torch.cat(all_beneficial)
        harmful = torch.cat(all_harmful)
        neutral = torch.cat(all_neutral)
        result.update(
            {
                "audit/transition_count": float(delta.numel()),
                "audit/candidate_gain": _mean_or_nan(delta),
                "audit/headroom": _mean_or_nan(torch.clamp(delta, min=0.0)),
                "audit/realized_gain": _mean_or_nan(realized),
                "audit/gate_value": _mean_or_nan(gate_value),
                "audit/accept_rate": float(accepted.float().mean().item()),
                "audit/involvement_rate": float((~accepted).float().mean().item()),
                "audit/beneficial_candidate_rate": float(beneficial.float().mean().item()),
                "audit/harmful_candidate_rate": float(harmful.float().mean().item()),
                "audit/neutral_candidate_rate": float(neutral.float().mean().item()),
                "audit/beneficial_capture_rate": _safe_rate(accepted, beneficial),
                "audit/harmful_block_rate": _safe_rate(~accepted, harmful),
            }
        )
    else:
        result["audit/transition_count"] = 0.0

    return result


def _foreground_dice(logits: Tensor, target: Tensor) -> Tensor:
    return multiclass_dice(logits, target, num_classes=int(logits.shape[1]))


@torch.no_grad()
def evaluate_audit_modes_batch(
    model: torch.nn.Module,
    images: Tensor,
    ground_truth: Tensor,
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    neutral_margin: float = 0.005,
) -> dict[str, float]:
    """Run initial/always/self/oracle on one batch and quantify audit headroom."""

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

    initial_dice = _foreground_dice(initial_logits, ground_truth)
    always_dice = _foreground_dice(always_logits, ground_truth)
    self_dice = _foreground_dice(self_logits, ground_truth)
    oracle_dice = _foreground_dice(oracle_logits, ground_truth)

    oracle_headroom = oracle_dice - initial_dice
    self_gain = self_dice - initial_dice
    candidate_path_gain = always_dice - initial_dice
    audit_rescue = self_dice - always_dice

    positive_headroom = oracle_headroom > 1e-8
    capture = torch.full_like(oracle_headroom, float("nan"))
    capture[positive_headroom] = self_gain[positive_headroom] / oracle_headroom[positive_headroom]

    result = {
        "modes/initial_dice": _mean_or_nan(initial_dice),
        "modes/always_accept_dice": _mean_or_nan(always_dice),
        "modes/self_audit_dice": _mean_or_nan(self_dice),
        "modes/oracle_dice": _mean_or_nan(oracle_dice),
        "modes/candidate_path_gain": _mean_or_nan(candidate_path_gain),
        "modes/audit_rescue_vs_always": _mean_or_nan(audit_rescue),
        "modes/self_audit_gain": _mean_or_nan(self_gain),
        "modes/oracle_headroom": _mean_or_nan(oracle_headroom),
        "modes/headroom_capture_ratio": _mean_or_nan(capture[torch.isfinite(capture)]),
        "modes/headroom_available_rate": float(positive_headroom.float().mean().item()),
    }
    result.update(
        decompose_self_audit_output(
            self_out,
            ground_truth,
            neutral_margin=float(neutral_margin),
        )
    )
    return result


@torch.no_grad()
def evaluate_audit_decomposition(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    neutral_margin: float = 0.005,
    max_batches: int | None = None,
    disable_tqdm: bool = False,
) -> dict[str, float]:
    """Aggregate stage-wise audit attribution over a validation loader."""

    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []
    try:
        pbar = tqdm(loader, desc="Audit decomposition", disable=disable_tqdm, leave=False)
        for batch_index, raw_batch in enumerate(pbar):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch(raw_batch, device)
            row = evaluate_audit_modes_batch(
                model,
                batch["image"],
                batch["mask"],
                tau_accept=float(tau_accept),
                t_max=int(t_max),
                neutral_margin=float(neutral_margin),
            )
            if batch_index == 0:
                row.update(
                    probe_gt_leakage(
                        model,
                        batch["image"],
                        batch["mask"],
                        tau_accept=float(tau_accept),
                        t_max=int(t_max),
                    )
                )
            rows.append(row)
    finally:
        model.train(was_training)

    if not rows:
        return {}

    keys = sorted(set().union(*(row.keys() for row in rows)))
    aggregate: dict[str, float] = {}
    for key in keys:
        values = torch.tensor(
            [row[key] for row in rows if key in row and torch.isfinite(torch.tensor(row[key]))],
            dtype=torch.float64,
        )
        aggregate[key] = float(values.mean().item()) if values.numel() else float("nan")
    return aggregate


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
        "gt_firewall/decision_mismatch_rate": float(mismatch_rate),
        "gt_firewall/passed": float(max_abs <= 1e-7 and mismatch_rate == 0.0),
    }


def decompose_annotation_output(
    output: Mapping[str, Any],
    ground_truth: Tensor,
    *,
    neutral_margin: float = 0.005,
) -> dict[str, float]:
    """Measure A0 saturation and available recurrent correction headroom."""

    trace = output.get("state_trace", output.get("all_states"))
    if not isinstance(trace, (list, tuple)) or not trace:
        initial = _tensor_from(output, "initial_logits", "a0_logits", "A0_logits")
        states = output.get("states", [])
        trace = ([initial] if initial is not None else []) + list(states)
    trace = [value for value in trace if torch.is_tensor(value)]
    if not trace:
        raise ValueError("Annotation output contains no trajectory")

    scores = [multiclass_dice(value.detach(), ground_truth) for value in trace]
    result: dict[str, float] = {
        "phase_a/a0_dice": _mean_or_nan(scores[0]),
        "phase_a/final_dice": _mean_or_nan(scores[-1]),
        "phase_a/total_refinement_gain": _mean_or_nan(scores[-1] - scores[0]),
    }
    positive_mass: list[Tensor] = []
    for stage, (previous_score, candidate_score) in enumerate(zip(scores[:-1], scores[1:])):
        delta = candidate_score - previous_score
        beneficial = delta > float(neutral_margin)
        harmful = delta < -float(neutral_margin)
        neutral = ~(beneficial | harmful)
        headroom = torch.clamp(delta, min=0.0)
        positive_mass.append(headroom)
        prefix = f"phase_a/stage_{stage}"
        result[f"{prefix}/prev_dice"] = _mean_or_nan(previous_score)
        result[f"{prefix}/candidate_dice"] = _mean_or_nan(candidate_score)
        result[f"{prefix}/candidate_gain"] = _mean_or_nan(delta)
        result[f"{prefix}/headroom"] = _mean_or_nan(headroom)
        result[f"{prefix}/beneficial_rate"] = float(beneficial.float().mean().item())
        result[f"{prefix}/harmful_rate"] = float(harmful.float().mean().item())
        result[f"{prefix}/neutral_rate"] = float(neutral.float().mean().item())
    if positive_mass:
        result["phase_a/mean_stage_headroom"] = _mean_or_nan(torch.cat(positive_mass))
        result["phase_a/headroom_collapse_ratio"] = float(
            1.0 - min(max(result["phase_a/mean_stage_headroom"] / 0.005, 0.0), 1.0)
        )
    return result


@torch.no_grad()
def evaluate_annotation_headroom(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    neutral_margin: float = 0.005,
    max_batches: int | None = None,
    disable_tqdm: bool = False,
) -> dict[str, float]:
    """Aggregate Phase-A A0/refinement headroom over validation data."""

    was_training = model.training
    model.eval()
    rows: list[dict[str, float]] = []
    try:
        pbar = tqdm(loader, desc="Annotation headroom", disable=disable_tqdm, leave=False)
        for batch_index, raw_batch in enumerate(pbar):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch(raw_batch, device)
            output = model.forward_annotation(batch["image"])
            rows.append(
                decompose_annotation_output(
                    output,
                    batch["mask"],
                    neutral_margin=float(neutral_margin),
                )
            )
    finally:
        model.train(was_training)
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    result: dict[str, float] = {}
    for key in keys:
        finite = [row[key] for row in rows if key in row and torch.isfinite(torch.tensor(row[key]))]
        result[key] = float(torch.tensor(finite, dtype=torch.float64).mean().item()) if finite else float("nan")
    return result
