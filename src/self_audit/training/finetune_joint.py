"""Phase C: threshold-controlled joint fine-tuning with separated gradients.

**Metric space.**  Every Dice this module *reports* is
:data:`~self_audit.audit.semantics.METRIC_SPACE_SLICE_PROXY` -- 2-D per-slice
foreground macro Dice on the resized network grid, averaged over slices.  It
is a training/monitoring proxy, **not** a paper metric; quotable per-volume
numbers live in :mod:`self_audit.evaluation.volume_inference`.  Phase A now
computes the identical quantity through the identical helper
(:func:`~self_audit.evaluation.metrics.slice_proxy_dice`), so the two phases'
headline numbers are comparable.

The transition *targets* that drive the audit loss keep using
:func:`~self_audit.audit.targets.multiclass_dice` unchanged, so training is
bit-identical to before this module's reporting was realigned.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.audit.semantics import (
    AUDIT_TARGET_LEGACY_ONE_V1,
    FOREGROUND_DICE_EXCLUDE_V1,
    METRIC_SPACE_SLICE_PROXY,
    resolve_empty_policy,
)
from self_audit.audit.targets import build_transition_targets, multiclass_dice
from self_audit.evaluation.contracts import (
    ContractMismatchError,
    FOREGROUND_DICE_EXCLUDE_V1_CONTRACT,
    MetricContract,
    compute_dice_from_stats,
    compute_sufficient_statistics,
    resolve_metric_contract,
    score_state,
)
from self_audit.evaluation.threshold import CACHE_SCHEMA_VERSION
from self_audit.evaluation.metrics import (
    acceptance_metrics,
    slice_proxy_dice,
    transition_audit_metrics,
)
from self_audit.losses.annotation import annotation_loss
from self_audit.losses.audit import audit_loss
from self_audit.training._utils import (
    add_wandb_and_tqdm_args,
    autocast_context,
    build_data_loader,
    build_grad_scaler,
    build_model_from_config,
    build_patient_dataset,
    build_training_scheduler,
    checkpoint_progress,
    encoder_head_optimizer,
    extract_initial_logits,
    finalize_optimizer_step,
    is_finite,
    load_checkpoint,
    load_config,
    move_batch,
    print_model_parameter_summary,
    resolve_amp,
    resolve_device,
    save_checkpoint,
    seed_everything,
    setup_wandb_logger,
    validate_accumulation_steps,
    validate_dataset_splits,
)


def _zero_audit_term(model: torch.nn.Module, reference: torch.Tensor) -> torch.Tensor:
    """Keep the total differentiable when a hard cap produces no transition."""

    auditor = getattr(model, "auditor", None)
    if auditor is None:
        return reference.new_zeros(())
    return sum((parameter.sum() * 0.0 for parameter in auditor.parameters()), reference.new_zeros(()))


def _local_class_weights(
    target: torch.Tensor,
    *,
    num_classes: int = 3,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Return bounded inverse-frequency weights for local audit targets."""

    values = target.reshape(-1).long()
    counts = torch.bincount(values, minlength=int(num_classes)).float()
    present = counts > 0
    weights = torch.ones(int(num_classes), device=target.device, dtype=torch.float32)
    if bool(present.any()):
        total = counts[present].sum()
        n_present = present.sum().float()
        weights[present] = (total / (n_present * counts[present])).clamp(0.25, float(max_weight))
    return weights


def _mask_tensor(value: Any, *, device: torch.device, batch_size: int) -> torch.Tensor:
    mask = value if torch.is_tensor(value) else torch.as_tensor(value, device=device)
    mask = mask.to(device=device, dtype=torch.bool).reshape(-1)
    if mask.numel() != int(batch_size):
        raise ValueError(f"Transition mask must have {batch_size} rows, got {mask.numel()}")
    return mask


def select_audit_output(
    audit_output: Any,
    active_mask: torch.Tensor | Iterable[bool] | None = None,
) -> Any | None:
    """Select active rows from a batch-aligned Auditor output.

    The model stores full ``[B,...]`` audit tensors for every transition so
    that transition alignment is preserved.  Phase C calls this helper before
    computing GT-derived targets.  If no explicit mask is supplied, the
    helper reads ``active_mask``/``audit_mask`` from a mapping output.
    """

    if audit_output is None:
        return None
    if active_mask is None and isinstance(audit_output, Mapping):
        active_mask = audit_output.get("active_mask", audit_output.get("audit_mask"))
    if active_mask is None:
        return audit_output

    if torch.is_tensor(active_mask):
        mask = active_mask.to(dtype=torch.bool).reshape(-1)
    else:
        mask = torch.as_tensor(list(active_mask), dtype=torch.bool).reshape(-1)
    if not bool(mask.any()):
        return None
    indices = mask.nonzero(as_tuple=False).flatten()

    def select(value: Any) -> Any:
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == mask.numel():
            return value.index_select(0, indices.to(device=value.device))
        return value

    if isinstance(audit_output, Mapping):
        return {key: select(value) for key, value in audit_output.items()}
    if torch.is_tensor(audit_output):
        return select(audit_output)
    # AuditOutput is a small dataclass in the model namespace.  Returning a
    # mapping keeps this helper independent of that class and is accepted by
    # audit_loss's dict/object compatibility layer.
    local_logits = getattr(audit_output, "local_logits", None)
    delta_q = getattr(audit_output, "delta_q", None)
    if torch.is_tensor(local_logits) or torch.is_tensor(delta_q):
        return {
            "local_logits": select(local_logits),
            "delta_q": select(delta_q),
        }
    return audit_output


def _transition_value_at(value: Any, index: int) -> Any | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.ndim == 1:
            return value if index == 0 else None
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return None


def _transition_active_mask(
    output: Mapping[str, Any],
    audit_output: Any,
    index: int,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    masks = output.get("transition_active_masks", output.get("active_masks"))
    value = _transition_value_at(masks, index)
    if value is None and isinstance(audit_output, Mapping):
        value = audit_output.get("active_mask", audit_output.get("audit_mask"))
    if value is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    return _mask_tensor(value, device=device, batch_size=batch_size)


def _select_batch_rows(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        raise ValueError("Transition tensors must have a batch dimension")
    if value.shape[0] == mask.numel():
        return value.index_select(0, mask.nonzero(as_tuple=False).flatten().to(value.device))
    if value.shape[0] == int(mask.sum().item()):
        return value
    raise ValueError(
        f"Transition batch rows {value.shape[0]} do not match mask {mask.numel()} "
        f"or active rows {int(mask.sum().item())}"
    )


def compute_joint_losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    lambda_audit: float = 1.0,
    neutral_margin: float = 0.005,
    local_weighting: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute annotation and Auditor losses with an explicit gradient split.

    The inference gate is discrete and receives detached ``DeltaQ``.  The
    annotation term sees only the retained final state.  The audit term uses
    detached transition inputs and therefore can update only Auditor weights.
    """

    if hasattr(model, "infer"):
        output = model.infer(
            batch["image"],
            mode="self_audit",
            tau_accept=float(tau_accept),
            t_max=int(t_max),
        )
    elif hasattr(model, "forward_joint"):  # pragma: no cover - compatibility fallback
        output = model.forward_joint(batch["image"])
    else:  # pragma: no cover
        output = model(batch["image"])

    if isinstance(output, dict):
        final_logits = output.get("logits")
        if not torch.is_tensor(final_logits):
            final_logits = extract_initial_logits(output)
    else:
        final_logits = extract_initial_logits(output)
    annotation_term = annotation_loss(final_logits, batch["mask"])[0]

    audit_terms: list[torch.Tensor] = []
    audit_parts: list[dict[str, torch.Tensor]] = []
    audit_sample_counts: list[int] = []
    audit_transition_masks: list[torch.Tensor] = []
    if isinstance(output, dict):
        previous_states = output.get("transition_previous", [])
        candidate_states = output.get("transition_candidates", output.get("candidates", []))
        audit_outputs = output.get("audits", [])
        batch_size = int(batch["image"].shape[0])
        for index, (previous, candidate, audit_output) in enumerate(
            zip(previous_states, candidate_states, audit_outputs)
        ):
            active_mask = _transition_active_mask(
                output,
                audit_output,
                index,
                batch_size=batch_size,
                device=batch["image"].device,
            )
            if not bool(active_mask.any()):
                continue
            selected_audit = select_audit_output(audit_output, active_mask)
            if selected_audit is None:
                continue
            # Targets and Auditor inputs are explicitly detached.  The Auditor
            # also defensively detaches its inputs internally.
            previous_audit = _select_batch_rows(previous.detach(), active_mask)
            candidate_audit = _select_batch_rows(candidate.detach(), active_mask)
            ground_truth = _select_batch_rows(batch["mask"], active_mask)
            targets = build_transition_targets(previous_audit, candidate_audit, ground_truth)
            weights = _local_class_weights(targets.local) if local_weighting else None
            audit_term, parts = audit_loss(
                selected_audit,
                targets,
                neutral_margin=float(neutral_margin),
                local_class_weights=weights,
            )
            audit_terms.append(audit_term)
            audit_parts.append(parts)
            audit_sample_counts.append(int(active_mask.sum().item()))
            audit_transition_masks.append(active_mask.detach())
    audit_term = torch.stack(audit_terms).mean() if audit_terms else _zero_audit_term(model, annotation_term)
    total = annotation_term + float(lambda_audit) * audit_term
    details: dict[str, Any] = {
        "annotation_loss": annotation_term.detach(),
        "audit_loss": audit_term.detach(),
        "total_loss": total.detach(),
        # Differentiable references are intentionally exposed for phase-level
        # gradient tests and diagnostics; callers should backpropagate only
        # the total or the explicitly selected term.
        "annotation_loss_tensor": annotation_term,
        "audit_loss_tensor": audit_term,
        "total_loss_tensor": total,
        "audit_parts": audit_parts,
        "transition_count": len(audit_terms),
        "audit_sample_counts": audit_sample_counts,
        "audit_transition_masks": audit_transition_masks,
        "output": output,
    }
    return total, details


#: Sample-level rollout counters reported for every schedule interval.
#:
#: A zero here is a **measured** zero: the stage ran an accepted-history
#: rollout and observed none of that event.  A stage that could not observe a
#: counter family at all reports the matching ``*_available`` flag as ``0.0``
#: instead of presenting the zeros as valid telemetry.
PREDICTED_HISTORY_COUNTER_KEYS = (
    "real_evidence_attempts",
    "accepted_history_count",
    "candidate_c_attempts",
    "candidate_c_feasible",
    "candidate_c_fallback",
    "candidate_c_replay_failure",
)

#: Window modes under which the Candidate-C counters describe a solver that is
#: actually running.  Under ``current``/``feature_only``/``free_offsets``/
#: ``direct_rollback`` there is no C solver, so the C counters are reported as
#: zero with ``candidate_c_diagnostics_available = 0.0``.
CANDIDATE_C_WINDOW_MODES = ("candidate_c", "candidate_c_no_fix")


#: Post-acceptance attempts whose own forward is the trainable ordinary
#: annotation generator, whatever produced the evidence they consume.  Reported
#: alongside -- never instead of -- the six agreed counters, because
#: ``real_evidence_attempts`` deliberately counts every attempt that consumed
#: the latest accepted evidence, including turns where a solver consumes it.
ORDINARY_REAL_EVIDENCE_KEY = "ordinary_annotation_real_evidence_attempts"


def zero_rollout_counters() -> dict[str, float]:
    """Return the counter block for a stage/batch that ran no rollout at all."""

    counters: dict[str, float] = {key: 0 for key in PREDICTED_HISTORY_COUNTER_KEYS}
    counters[ORDINARY_REAL_EVIDENCE_KEY] = 0
    counters["attempted_turn_samples"] = 0
    counters["rollout_samples"] = 0
    counters["rollout_batches"] = 0
    counters["candidate_c_diagnostics_available"] = 0.0
    counters["ordinary_accepted_path_from_diagnostics"] = 0.0
    return counters


def _counter_masks(output: Mapping[str, Any], *names: str) -> list[torch.Tensor]:
    """Return per-turn ``[B]`` boolean masks from the first present key."""

    for name in names:
        value = output.get(name)
        if isinstance(value, (list, tuple)) and value and all(torch.is_tensor(item) for item in value):
            return [item.detach().reshape(-1).to(dtype=torch.bool) for item in value]
        if torch.is_tensor(value) and value.ndim == 2:
            return [row.detach().reshape(-1).to(dtype=torch.bool) for row in value]
    return []


def candidate_c_counters_meaningful(model: Any) -> bool:
    """Return whether ``model`` actually runs the Candidate-C solver."""

    return str(getattr(model, "window_mode", "")) in CANDIDATE_C_WINDOW_MODES


#: ``accepted_path`` value denoting an ordinary annotation transition.  Any
#: other published path (a replayed factual support, a counterfactual
#: restitution, a direct rollback) is not an ordinary trainable annotation.
ORDINARY_ACCEPTED_PATH = "ordinary"


def _row_sample_indices(row: Mapping[str, Any]) -> list[int]:
    """Return the sample rows one diagnostics row refers to."""

    index = row.get("sample_index")
    if isinstance(index, int) and not isinstance(index, bool):
        return [index]
    active = row.get("active_indices")
    if isinstance(active, (list, tuple)):
        return [int(value) for value in active if isinstance(value, int) and not isinstance(value, bool)]
    if torch.is_tensor(active):
        return [int(value) for value in active.detach().reshape(-1).tolist()]
    return []


def _ordinary_attempt_masks(
    output: Mapping[str, Any],
    active_masks: list[torch.Tensor],
) -> tuple[list[torch.Tensor], bool]:
    """Mark, per turn, the attempts whose **own** path is ordinary annotation.

    Returns ``(masks, from_diagnostics)``.  ``masks[t]`` is the subset of the
    rows attempted at turn ``t`` whose diagnostics row for that turn reports
    ``accepted_path == "ordinary"`` -- that is, rows where the *receiving*
    forward at turn ``t`` is the trainable ordinary annotation generator.

    The path is read from the **current** turn, never the previous one.  What
    produced the consumed evidence does not determine which module consumes it:
    a previous ordinary acceptance followed by a Candidate-C turn has its
    evidence consumed inside the solver, while a previous restitution followed
    by an ordinary turn has its evidence consumed by the trainable generator.
    A row whose C solve failed C1 and was routed back to normal annotation
    reports ``accepted_path == "ordinary"`` and therefore counts here: it really
    did feed the previous predicted evidence into the ordinary generator.

    A row is cleared only when it publishes a path that is not ordinary; a
    missing ``accepted_path`` leaves the attempt as it stands, and a rollout
    that publishes no rows at all has no restitution path by construction, so
    every attempt is ordinary and the flag records that this was structural.
    """

    rows = output.get("candidate_c_diagnostics")
    if not isinstance(rows, (list, tuple)) or not rows:
        return [mask.clone() for mask in active_masks], False

    ordinary = [mask.clone() for mask in active_masks]
    usable = False
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        turn = row.get("turn")
        path = row.get("accepted_path")
        if not isinstance(turn, int) or isinstance(turn, bool) or path is None:
            continue
        if turn < 0 or turn >= len(ordinary):
            continue
        usable = True
        if str(path) == ORDINARY_ACCEPTED_PATH:
            continue
        for sample in _row_sample_indices(row):
            if 0 <= sample < int(ordinary[turn].numel()):
                ordinary[turn][sample] = False
    if not usable:
        return [mask.clone() for mask in active_masks], False
    return ordinary, True


def _candidate_c_attempted(row: Mapping[str, Any]) -> bool:
    """Return whether one diagnostics row is an actual record-consuming attempt.

    An attempt is a turn that reached the solver holding a replayable ordinary
    record -- a valid ``record_turn`` with ``record_kind == "ordinary"`` -- and
    is counted **regardless of whether the solve was eligible or succeeded**.
    Gating on ``eligible`` would make a C1 replay failure or a zero-REGRESS
    turn invisible, which would in turn make ``candidate_c_replay_failure``
    unable to increment for a real core failure.  A fresh ordinary turn holding
    no record is not an attempt.  An explicit ``attempted`` boolean from the
    core module overrides the inference when present.
    """

    explicit = row.get("attempted")
    if isinstance(explicit, bool):
        return explicit
    record_turn = row.get("record_turn")
    if not isinstance(record_turn, int) or isinstance(record_turn, bool) or record_turn < 0:
        return False
    return str(row.get("record_kind")) == "ordinary"


def rollout_counters(
    output: Any,
    *,
    model: Any = None,
    batch_size: int | None = None,
) -> dict[str, float]:
    """Derive the six sample-level rollout counters from one ``infer`` output.

    Definitions (all counted in *sample-turn attempts*, never in batches):

    ``real_evidence_attempts``
        Attempts in which the annotator consumed the latest real predicted
        accepted evidence, i.e. rows whose *previous* turn was accepted and
        which were attempted again this turn.  A rejected attempt HALTs and can
        never contribute, so attempted-but-rejected history is never counted
        here.  **Whatever produced that acceptance counts**: under a
        restitution window mode the previous acceptance may be a Candidate-C or
        direct-rollback action, whose evidence is consumed by the solver rather
        than by a trainable ordinary annotation forward.  Use
        ``ordinary_annotation_real_evidence_attempts`` for the narrower
        quantity.
    ``accepted_history_count``
        **All** accepted transitions **created** by this rollout, ordinary and
        restitution alike, read from the state masks.  It includes a final-turn
        acceptance that no later turn could consume, so it is a creation count,
        not a reuse count.
    ``candidate_c_attempts`` / ``candidate_c_feasible`` /
    ``candidate_c_fallback`` / ``candidate_c_replay_failure``
        Derived from the per-attempt sample rows in ``candidate_c_diagnostics``
        and only when ``model.window_mode`` actually selects the C solver.
        An **attempt** is a turn that reached the solver holding a replayable
        ordinary record (valid ``record_turn`` with ``record_kind ==
        "ordinary"``, or an explicit ``attempted`` flag from the core), counted
        whether or not the solve was eligible or succeeded -- so a C1 replay
        failure and a zero-REGRESS turn both land in the denominator.  A fresh
        ordinary turn holding no record is not an attempt.  The other three are
        counted strictly within those attempted rows: ``feasible is True``,
        a non-null ``fallback_reason``, and ``c1_passed is False`` (``None`` is
        unmeasured and does not count).

    Reported alongside the six, and clearly named so it cannot be mistaken for
    one of them:

    ``ordinary_annotation_real_evidence_attempts``
        Post-acceptance attempts whose **own** forward is the trainable ordinary
        annotation generator: accepted at ``t-1``, attempted at ``t``, and the
        row for turn ``t`` reports ``accepted_path == "ordinary"``.  The
        receiving path is what matters, not the path that produced the
        evidence -- previous-ordinary followed by a C turn has its evidence
        consumed inside the solver and does **not** count, while
        previous-C followed by an ordinary turn does.  A C1 replay failure
        routed back to normal annotation counts here too (and still counts as a
        C attempt above).  Under a baseline ordinary rollout no other path
        exists, so every post-accept attempt qualifies and the count equals
        ``real_evidence_attempts``; ``ordinary_accepted_path_from_diagnostics``
        says which derivation was used (``1.0`` = per-sample ``accepted_path``).

        This counts **forward exposure**, not a guaranteed parameter gradient:
        an accepted conditional update or a gradient probe would be a separate
        measurement, and none is claimed here.

    Denominators reported alongside: ``attempted_turn_samples`` (total
    sample-turn attempts), ``rollout_samples`` (rows entering the rollout) and
    ``rollout_batches``.
    """

    counters = zero_rollout_counters()
    if not isinstance(output, Mapping):
        return counters

    active_masks = _counter_masks(output, "transition_active_masks", "active_masks")
    state_masks = _counter_masks(output, "transition_state_masks", "state_masks")

    counters["attempted_turn_samples"] = sum(int(mask.sum().item()) for mask in active_masks)
    counters["accepted_history_count"] = sum(int(mask.sum().item()) for mask in state_masks)

    real_evidence = 0
    for index in range(1, len(active_masks)):
        if index - 1 >= len(state_masks):
            break
        previous_accepted = state_masks[index - 1]
        attempt = active_masks[index]
        if previous_accepted.numel() != attempt.numel():
            continue
        real_evidence += int((previous_accepted & attempt.to(previous_accepted.device)).sum().item())
    counters["real_evidence_attempts"] = real_evidence

    ordinary_masks, from_diagnostics = _ordinary_attempt_masks(output, active_masks)
    counters["ordinary_accepted_path_from_diagnostics"] = 1.0 if from_diagnostics else 0.0
    ordinary_evidence = 0
    for index in range(1, len(active_masks)):
        if index - 1 >= len(state_masks) or index >= len(ordinary_masks):
            break
        previous_accepted = state_masks[index - 1]
        attempt = active_masks[index]
        ordinary_now = ordinary_masks[index]
        if previous_accepted.numel() != attempt.numel() or ordinary_now.numel() != attempt.numel():
            continue
        device = previous_accepted.device
        ordinary_evidence += int(
            (
                previous_accepted
                & attempt.to(device)
                & ordinary_now.to(device)
            ).sum().item()
        )
    counters[ORDINARY_REAL_EVIDENCE_KEY] = ordinary_evidence

    if batch_size is None:
        logits = output.get("logits")
        batch_size = int(logits.shape[0]) if torch.is_tensor(logits) else 0
    counters["rollout_samples"] = int(batch_size)
    counters["rollout_batches"] = 1

    rows = output.get("candidate_c_diagnostics")
    if model is not None and candidate_c_counters_meaningful(model) and isinstance(rows, (list, tuple)):
        counters["candidate_c_diagnostics_available"] = 1.0
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if not _candidate_c_attempted(row):
                continue
            counters["candidate_c_attempts"] += 1
            if row.get("feasible") is True:
                counters["candidate_c_feasible"] += 1
            if row.get("c1_passed") is False:
                counters["candidate_c_replay_failure"] += 1
            if row.get("fallback_reason") is not None:
                counters["candidate_c_fallback"] += 1
    return counters


def accumulate_rollout_counters(
    total: dict[str, float],
    update: Mapping[str, float],
) -> dict[str, float]:
    """Accumulate one batch's counters into a stage total.

    Counts add.  The two ``*_available`` / ``*_from_diagnostics`` flags are
    conjunctions over the batches that actually ran a rollout: a stage claims a
    provenance only when every measured rollout batch supplied it.
    """

    flags = ("candidate_c_diagnostics_available", "ordinary_accepted_path_from_diagnostics")
    had_rollout = float(total.get("rollout_batches", 0)) > 0
    for key, value in update.items():
        if key in flags:
            continue
        total[key] = total.get(key, 0) + value
    if float(update.get("rollout_batches", 0)) > 0:
        for flag in flags:
            incoming = float(update.get(flag, 0.0))
            current = float(total.get(flag, 0.0))
            total[flag] = min(current, incoming) if had_rollout else incoming
    return total


def _rollout_shared_features(output: Mapping[str, Any]) -> torch.Tensor:
    """Return the detached shared features the rollout itself was built on.

    ``infer`` already publishes the exact encoder output the official Auditor
    was conditioned on for this trajectory, so the auxiliary auditor loss reuses
    it instead of running a second encoder forward.  There is deliberately no
    fallback: raw images are not shared features, and silently substituting
    them would score the Auditor on an input it never saw.
    """

    shared = output.get("shared_features")
    if not torch.is_tensor(shared):
        raise KeyError(
            "Predicted-history auditor exposure requires 'shared_features' in the infer "
            "output; refusing to re-encode or to substitute raw images"
        )
    return shared.detach()


def _select_active_rows(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select active rows, avoiding a full copy when every row is active."""

    if bool(mask.all()):
        return value
    return _select_batch_rows(value, mask)


def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    values = probs.clamp_min(1e-8)
    return -(values * values.log()).sum(dim=1, keepdim=True)


def compute_predicted_history_annotation_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Auxiliary annotation loss over the official accepted-predicted-history rollout.

    This runs the deployable ``model.infer(mode="self_audit")`` path -- the same
    official gate, the same strict ``delta_q > tau`` acceptance, rejection still
    HALTs -- and applies the ordinary annotation loss to the **retained final**
    state.  Ground truth enters only as the supervised target; no ground truth
    is passed into the runtime proposal, no acceptance is forced, and no
    oracle or synthetic audit map is injected.

    The audit evidence consumed by the annotator here is the model's own
    *predicted* local evidence, detached inside ``infer``.  During the
    annotation bootstrap that evidence comes from an **untrained** Auditor: it
    is explicitly uncalibrated exposure to the shape of a history-conditioned
    input, not evidence that the history is informative.
    """

    if not hasattr(model, "infer"):
        raise AttributeError("Predicted-history exposure requires model.infer")
    output = model.infer(
        batch["image"],
        mode="self_audit",
        tau_accept=float(tau_accept),
        t_max=int(t_max),
    )
    if not isinstance(output, Mapping):
        raise TypeError("Predicted-history exposure requires a mapping inference output")
    final_logits = output.get("logits")
    if not torch.is_tensor(final_logits):
        final_logits = extract_initial_logits(output)
    loss = annotation_loss(final_logits, batch["mask"])[0]
    counters = rollout_counters(output, model=model, batch_size=int(batch["image"].shape[0]))
    return loss, counters


@torch.no_grad()
def _collect_predicted_history_rollout(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, float], torch.Tensor]:
    """Collect on-policy transition pairs from an accepted-history rollout.

    Runs under ``no_grad`` so the annotation network cannot be reached from
    anything built here; the returned probabilities are detached leaves.  The
    pairs are genuinely on-policy (the network produced them at inference), so
    they carry ``provenance_kind = "on_policy"`` with a fine-grained
    ``provenance`` naming the predicted-history rollout.
    """

    if not hasattr(model, "infer"):
        raise AttributeError("Predicted-history exposure requires model.infer")
    output = model.infer(
        batch["image"],
        mode="self_audit",
        tau_accept=float(tau_accept),
        t_max=int(t_max),
    )
    if not isinstance(output, Mapping):
        raise TypeError("Predicted-history exposure requires a mapping inference output")

    device = batch["image"].device
    batch_size = int(batch["image"].shape[0])
    transitions: list[dict[str, Any]] = []
    previous_states = output.get("transition_previous", [])
    candidate_states = output.get("transition_candidates", output.get("candidates", []))
    audit_outputs = output.get("audits", [])
    for index, (previous, candidate, audit_output) in enumerate(
        zip(previous_states, candidate_states, audit_outputs)
    ):
        active_mask = _transition_active_mask(
            output,
            audit_output,
            index,
            batch_size=batch_size,
            device=device,
        )
        if not bool(active_mask.any()):
            continue
        transitions.append(
            {
                # ``infer`` states are logits; softmax matches the probability
                # convention ``build_auditor_transitions`` feeds the Auditor.
                "previous": previous.detach().softmax(dim=1),
                "candidate": candidate.detach().softmax(dim=1),
                "turn_index": index,
                "provenance": "on_policy:predicted_history",
                "provenance_kind": "on_policy",
                "valid_mask": active_mask.detach(),
            }
        )
    counters = rollout_counters(output, model=model, batch_size=batch_size)
    return transitions, counters, _rollout_shared_features(output)


def collect_predicted_history_transitions(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Public form of :func:`_collect_predicted_history_rollout`.

    Callers that only want the transition pairs do not have to handle the
    shared-feature tensor the auditor loss needs.
    """

    transitions, counters, _ = _collect_predicted_history_rollout(
        model,
        batch,
        tau_accept=tau_accept,
        t_max=t_max,
    )
    return transitions, counters


def compute_predicted_history_audit_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    neutral_margin: float = 0.005,
    local_weighting: bool = True,
    audit_margin: float = 0.05,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Auditor-only loss over predicted-accepted-history transitions.

    The rollout itself runs under ``no_grad`` and every Auditor input --
    features, previous probabilities, candidate probabilities -- is a detached
    leaf, so this term can update Auditor parameters only.  The annotation
    network stays frozen, exactly as the auditor-only interval requires.
    Ground truth is used only to build the supervised transition targets.

    **Added cost, stated explicitly.**  Exactly one encoder forward runs, inside
    ``infer``; the shared features the official Auditor was conditioned on are
    reused here rather than re-derived.  The Auditor itself runs twice per
    active transition: once inside ``infer`` under ``no_grad`` as the official
    acceptance gate, and once here with gradients enabled to produce the loss.

    Under the **baseline** ``window_mode="current"`` rollout that is ``1``
    encoder forward, ``T`` annotator forwards and ``2 x T`` Auditor forwards per
    batch for ``T`` active turns, plus one backward through the Auditor only.

    Under a **restitution** window mode the annotator cost is strictly larger
    and ``T`` is no longer the whole story: each record-consuming attempt adds
    one differentiable historical replay of the stored support and up to two
    candidate checks per solver group, plus a coordinate backward inside the
    solver.  Those replays and checks are extra annotation-expert evaluations
    the solver performs on its own, and a factual batch can exceed the count of
    active rows because the solver groups supports.  This function does not
    perform or double-count them -- they belong to ``infer`` -- but the
    auxiliary path inherits them, so the baseline ``T`` figure must not be
    quoted for a Candidate-C or direct-rollback run.
    """

    auditor = getattr(model, "auditor", None)
    if auditor is None:
        raise AttributeError("Predicted-history auditor exposure requires model.auditor")

    transitions, counters, features = _collect_predicted_history_rollout(
        model,
        batch,
        tau_accept=tau_accept,
        t_max=t_max,
    )
    device = batch["image"].device

    losses: list[torch.Tensor] = []
    for transition in transitions:
        valid_mask = transition["valid_mask"].to(device=device, dtype=torch.bool)
        if not bool(valid_mask.any()):
            continue
        previous = _select_active_rows(transition["previous"], valid_mask)
        candidate = _select_active_rows(transition["candidate"], valid_mask)
        target = _select_active_rows(batch["mask"], valid_mask)
        targets = build_transition_targets(previous, candidate, target)
        weights = _local_class_weights(targets.local) if local_weighting else None
        with torch.no_grad():
            entropy_previous = _entropy_from_probs(previous)
            entropy_candidate = _entropy_from_probs(candidate)
        with autocast_context(enabled=amp_enabled, device=device, dtype=amp_dtype):
            audit_output = auditor(
                _select_active_rows(features, valid_mask),
                previous,
                candidate,
                candidate - previous,
                entropy_previous=entropy_previous,
                entropy_candidate=entropy_candidate,
            )
        loss, _ = audit_loss(
            audit_output,
            targets,
            margin=float(audit_margin),
            neutral_margin=float(neutral_margin),
            local_class_weights=weights,
        )
        losses.append(loss)

    counters["predicted_history_transitions"] = len(losses)
    if not losses:
        return None, counters
    return torch.stack(losses).mean(), counters


def _audit_output_tensor(output: Any, *names: str) -> torch.Tensor | None:
    for name in names:
        value = output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    return None


def _foreground_dice_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int | None = None,
    empty_policy: str | None = None,
) -> torch.Tensor:
    """Return per-sample foreground macro Dice in metric space ``"slice_proxy"``.

    Routed through :func:`~self_audit.evaluation.metrics.slice_proxy_dice`, the
    single shared proxy Phase A also calls, so the two phases can no longer
    disagree about the same predictions.

    **Behaviour change (loud):** this used to call
    :func:`~self_audit.audit.targets.multiclass_dice`, which scores a class
    that is empty in *both* prediction and target as ``1.0``.  Under the
    canonical ``"exclude"`` policy such a class is excluded instead, so a
    slice with no foreground anywhere now yields ``nan`` rather than ``1.0``.
    Reported Phase-C Dice will therefore *drop* on datasets with empty slices;
    that drop is the removal of an inflation, not a regression.  Callers must
    aggregate nan-aware (``torch.nanmean``).

    The audit **loss targets** are untouched: ``build_transition_targets``
    still uses ``multiclass_dice``, so training remains bit-identical.
    """

    classes = int(logits.shape[1]) if num_classes is None else int(num_classes)
    scores = slice_proxy_dice(
        logits,
        target,
        num_classes=classes,
        empty_policy=resolve_empty_policy(empty_policy),
    )
    return torch.as_tensor(scores, dtype=torch.float32, device=logits.device)


def _nanmean(values: torch.Tensor) -> float:
    """nan-aware mean; ``nan`` when nothing finite was contributed."""

    if values.numel() == 0:
        return float("nan")
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return float("nan")
    return float(values[finite].float().mean())


@torch.no_grad()
def validate_phase_c(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    max_batches: int | None = None,
    epoch: int = 0,
    total_epochs: int = 1,
    disable_tqdm: bool = False,
) -> dict[str, Any]:
    """Validate Phase C behavior with GT used only after deployable inference.

    The returned annotation metrics compare initial ``A0`` and retained final
    states.  Transition targets are constructed only in this validation
    helper, after ``model.infer(..., mode="self_audit")`` has completed.
    """

    was_training = model.training
    model.eval()
    initial_scores: list[torch.Tensor] = []
    final_scores: list[torch.Tensor] = []
    attempted_counts: list[torch.Tensor] = []
    accepted_counts: list[torch.Tensor] = []
    accepted_values: list[torch.Tensor] = []
    actual_deltas: list[torch.Tensor] = []
    audit_local_predictions: list[torch.Tensor] = []
    audit_local_targets: list[torch.Tensor] = []
    audit_delta_predictions: list[torch.Tensor] = []
    audit_delta_targets: list[torch.Tensor] = []

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch + 1:03d}/{total_epochs:03d} [Val C]",
        disable=disable_tqdm,
        leave=False,
    )
    try:
        for batch_index, raw_batch in enumerate(pbar):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch(raw_batch, device)
            output = model.infer(
                batch["image"],
                mode="self_audit",
                tau_accept=float(tau_accept),
                t_max=int(t_max),
            )
            if not isinstance(output, Mapping):
                raise TypeError("Phase-C validation requires a mapping inference output")
            initial_logits = output.get("initial_logits")
            if not torch.is_tensor(initial_logits):
                initial_logits = extract_initial_logits(output)
            final_logits = output.get("logits")
            if not torch.is_tensor(final_logits):
                final_logits = initial_logits
            ground_truth = batch["mask"]
            initial_scores.append(_foreground_dice_per_sample(initial_logits, ground_truth).detach())
            final_scores.append(_foreground_dice_per_sample(final_logits, ground_truth).detach())

            batch_size = int(batch["image"].shape[0])
            attempted_fallback = torch.zeros(batch_size, dtype=torch.long, device=device)
            accepted_fallback = torch.zeros(batch_size, dtype=torch.long, device=device)
            previous_states = output.get("transition_previous", [])
            candidate_states = output.get("transition_candidates", output.get("candidates", []))
            audit_outputs = output.get("audits", [])
            for index, (previous, candidate, audit_output) in enumerate(
                zip(previous_states, candidate_states, audit_outputs)
            ):
                active_mask = _transition_active_mask(
                    output,
                    audit_output,
                    index,
                    batch_size=batch_size,
                    device=device,
                )
                attempted_fallback += active_mask.to(torch.long)
                if not bool(active_mask.any()):
                    continue
                state_mask_value = _transition_value_at(
                    output.get("transition_state_masks", output.get("state_masks")),
                    index,
                )
                if state_mask_value is None and isinstance(audit_output, Mapping):
                    state_mask_value = audit_output.get("state_mask", audit_output.get("accepted"))
                accepted_mask = (
                    torch.zeros(batch_size, dtype=torch.bool, device=device)
                    if state_mask_value is None
                    else _mask_tensor(state_mask_value, device=device, batch_size=batch_size)
                )
                accepted_fallback += accepted_mask.to(torch.long)

                previous_selected = _select_batch_rows(previous.detach(), active_mask)
                candidate_selected = _select_batch_rows(candidate.detach(), active_mask)
                target_selected = _select_batch_rows(ground_truth, active_mask)
                targets = build_transition_targets(
                    previous_selected,
                    candidate_selected,
                    target_selected,
                )
                accepted_values.append(_select_batch_rows(accepted_mask.detach(), active_mask))
                actual_deltas.append(targets.delta_dice.reshape(-1).detach())

                selected_audit = select_audit_output(audit_output, active_mask)
                local_prediction = _audit_output_tensor(selected_audit, "local_logits", "local")
                delta_prediction = _audit_output_tensor(selected_audit, "delta_q", "global_delta_q", "delta_quality")
                if local_prediction is not None and delta_prediction is not None:
                    audit_local_predictions.append(local_prediction.detach())
                    audit_local_targets.append(targets.local.detach())
                    audit_delta_predictions.append(delta_prediction.detach().reshape(-1))
                    audit_delta_targets.append(targets.delta_dice.detach().reshape(-1))

            attempted_value = output.get("num_attempted_turns")
            accepted_value = output.get("accepted_count")
            attempted_counts.append(
                attempted_value.detach().to(torch.long)
                if torch.is_tensor(attempted_value)
                else attempted_fallback
            )
            accepted_counts.append(
                accepted_value.detach().to(torch.long)
                if torch.is_tensor(accepted_value)
                else accepted_fallback
            )
            if initial_scores and final_scores:
                cur_init = _nanmean(torch.cat(initial_scores))
                cur_final = _nanmean(torch.cat(final_scores))
                pbar.set_postfix({"init_dice": f"{cur_init:.4f}", "final_dice": f"{cur_final:.4f}"})
    finally:
        model.train(was_training)

    scored_slices = 0
    excluded_slices = 0
    if initial_scores:
        all_initial = torch.cat(initial_scores)
        all_final = torch.cat(final_scores)
        # nan-aware: under the "exclude" empty-class policy a slice with no
        # foreground in either prediction or target scores nan, and a plain
        # mean would poison the whole epoch.
        initial_mean = _nanmean(all_initial)
        final_mean = _nanmean(all_final)
        mean_attempted = float(torch.cat(attempted_counts).float().mean())
        mean_accepted = float(torch.cat(accepted_counts).float().mean())
        scored_slices = int(torch.isfinite(all_initial).sum())
        excluded_slices = int(all_initial.numel()) - scored_slices
    else:
        initial_mean = float("nan")
        final_mean = float("nan")
        mean_attempted = float("nan")
        mean_accepted = float("nan")

    if accepted_values and actual_deltas:
        acceptance = acceptance_metrics(
            torch.cat(accepted_values),
            torch.cat(actual_deltas),
        )
    else:
        acceptance = {
            "harmful_acceptance_rate": 0.0,
            "beneficial_rejection_rate": 0.0,
            "net_dice_gain_after_auditing": 0.0,
        }

    if audit_local_predictions:
        audit_metrics = transition_audit_metrics(
            torch.cat(audit_local_predictions),
            torch.cat(audit_local_targets),
            torch.cat(audit_delta_predictions),
            torch.cat(audit_delta_targets),
        )
    else:
        audit_metrics = {
            "improve_regress_accuracy": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "correlation_delta_q_delta_dice": float("nan"),
            "local_fix_f1": float("nan"),
            "local_regress_f1": float("nan"),
        }

    result: dict[str, Any] = {
        "metric_space": METRIC_SPACE_SLICE_PROXY,
        "scored_slice_count": float(scored_slices),
        "excluded_empty_slice_count": float(excluded_slices),
        "initial_foreground_macro_dice": initial_mean,
        "final_foreground_macro_dice": final_mean,
        "net_dice_gain": final_mean - initial_mean,
        "net_gain": final_mean - initial_mean,
        "harmful_acceptance_rate": acceptance["harmful_acceptance_rate"],
        "beneficial_rejection_rate": acceptance["beneficial_rejection_rate"],
        "net_dice_gain_after_auditing": acceptance["net_dice_gain_after_auditing"],
        "mean_attempted_turns": mean_attempted,
        "mean_accepted_turns": mean_accepted,
        "per_sample_mean_attempted_turns": mean_attempted,
        "per_sample_mean_accepted_turns": mean_accepted,
        "audit_transition_count": int(sum(int(values.numel()) for values in actual_deltas)),
        "audit_metrics": audit_metrics,
    }
    result.update({f"audit_{key}": value for key, value in audit_metrics.items()})
    return result


# Explicit alias for callers that use the validation stage name.
validate_joint = validate_phase_c


@torch.no_grad()
def collect_validation_transition_cache(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    t_max: int = 3,
    disable_tqdm: bool = False,
    empty_policy: str | None = None,
    metric_contract: MetricContract | str | None = None,
) -> dict[str, Any]:
    """Cache full validation transitions for threshold calibration.

    ``always_accept_refinement`` is used only to expose the full candidate
    trajectory.  The saved cache contains GT-derived deltas for calibration;
    it is never consumed by deployable inference.

    Metric contract:
    Transitions are evaluated under the specified ``metric_contract``
    (defaults to ``foreground_dice_exclude_v1``). ``q_previous`` and ``q_candidate``
    are recorded alongside ``actual_delta_dice`` under the same metric contract.
    Historical training targets (``multiclass_dice`` with ``legacy_one``) are
    recorded in ``legacy_actual_delta_dice``.

    Blank trajectory preservation:
    Blank A0 slices are NEVER dropped: all trajectories are preserved so that
    threshold replay can monitor blank slices and detect blank-to-hallucination
    regressions. Sufficient statistics (``tp_initial``, ``fp_initial``, ``fn_initial``,
    ``tp_candidate``, ``fp_candidate``, ``fn_candidate``) are recorded per sample
    and per transition for exact volumetric recomputation.
    """

    model.eval()
    if metric_contract is not None:
        contract = resolve_metric_contract(metric_contract)
    elif empty_policy is not None:
        policy = resolve_empty_policy(empty_policy)
        contract = MetricContract(
            name=f"foreground_dice_{policy}_v1",
            empty_policy=policy,
        )
    else:
        contract = FOREGROUND_DICE_EXCLUDE_V1_CONTRACT

    if contract.metric_space != METRIC_SPACE_SLICE_PROXY:
        raise ContractMismatchError(
            f"collect_validation_transition_cache operates on 2-D slice proxies and rejects "
            f"volume contract {contract.name!r} with metric_space={contract.metric_space!r}. "
            f"Per-slice Q values must never be labeled volume scores."
        )

    initial_values: list[torch.Tensor] = []
    tp_initial_values: list[torch.Tensor] = []
    fp_initial_values: list[torch.Tensor] = []
    fn_initial_values: list[torch.Tensor] = []

    quality_values: list[torch.Tensor] = []
    actual_values: list[torch.Tensor] = []
    legacy_actual_values: list[torch.Tensor] = []
    active_values: list[torch.Tensor] = []
    q_prev_values: list[torch.Tensor] = []
    q_cand_values: list[torch.Tensor] = []
    tp_cand_values: list[torch.Tensor] = []
    fp_cand_values: list[torch.Tensor] = []
    fn_cand_values: list[torch.Tensor] = []
    all_case_ids: list[str] = []

    num_classes = max(contract.classes) + 1

    pbar = tqdm(loader, desc="Caching transitions", disable=disable_tqdm, leave=False)
    for raw_batch in pbar:
        batch = move_batch(raw_batch, device)
        output = model.infer(
            batch["image"],
            mode="always_accept_refinement",
            tau_accept=-float("inf"),
            t_max=int(t_max),
        )
        initial = output.get("initial_logits")
        if not torch.is_tensor(initial):
            initial = extract_initial_logits(output)

        b_size = int(batch["image"].shape[0])
        init_preds = initial.argmax(dim=1).detach().cpu().numpy()
        gt_masks = batch["mask"].detach().cpu().numpy()

        batch_cids: list[str] | None = None
        for key in ("case_id", "subject_id", "volume_id"):
            if key in batch:
                raw_c = batch[key]
                batch_cids = [str(c.item()) if torch.is_tensor(c) else str(c) for c in raw_c]
                break
        if batch_cids is not None:
            all_case_ids.extend(batch_cids)

        batch_init_scores: list[float] = []
        batch_tp_init: list[list[int]] = []
        batch_fp_init: list[list[int]] = []
        batch_fn_init: list[list[int]] = []
        for i in range(b_size):
            stats = compute_sufficient_statistics(
                init_preds[i], gt_masks[i], classes=contract.classes
            )
            _, macro = compute_dice_from_stats(stats, contract)
            batch_init_scores.append(macro)
            tp_row = [0] * num_classes
            fp_row = [0] * num_classes
            fn_row = [0] * num_classes
            for c in contract.classes:
                tp_row[c] = stats.tp.get(c, 0)
                fp_row[c] = stats.fp.get(c, 0)
                fn_row[c] = stats.fn.get(c, 0)
            batch_tp_init.append(tp_row)
            batch_fp_init.append(fp_row)
            batch_fn_init.append(fn_row)

        initial_values.append(torch.as_tensor(batch_init_scores, dtype=torch.float32))
        tp_initial_values.append(torch.as_tensor(batch_tp_init, dtype=torch.int64))
        fp_initial_values.append(torch.as_tensor(batch_fp_init, dtype=torch.int64))
        fn_initial_values.append(torch.as_tensor(batch_fn_init, dtype=torch.int64))

        batch_quality: list[torch.Tensor] = []
        batch_actual: list[torch.Tensor] = []
        batch_legacy_actual: list[torch.Tensor] = []
        batch_active: list[torch.Tensor] = []
        batch_q_prev: list[torch.Tensor] = []
        batch_q_cand: list[torch.Tensor] = []
        batch_tp_cand: list[torch.Tensor] = []
        batch_fp_cand: list[torch.Tensor] = []
        batch_fn_cand: list[torch.Tensor] = []

        for index, (previous, candidate, audit_output) in enumerate(
            zip(output.get("transition_previous", []), output.get("transition_candidates", []), output.get("audits", []))
        ):
            active_mask = _transition_active_mask(
                output,
                audit_output,
                index,
                batch_size=b_size,
                device=device,
            )
            delta_q = _audit_output_tensor(audit_output, "delta_q", "global_delta_q", "delta_quality")
            if delta_q is None:
                raise ValueError("Auditor output lacks delta_q for threshold cache")

            prev_labels = previous.detach().argmax(dim=1).cpu().numpy()
            cand_labels = candidate.detach().argmax(dim=1).cpu().numpy()

            legacy_targets = build_transition_targets(previous.detach(), candidate.detach(), batch["mask"])
            batch_legacy_actual.append(legacy_targets.delta_dice.detach().cpu().float().reshape(-1))

            turn_deltas: list[float] = []
            turn_q_prev: list[float] = []
            turn_q_cand: list[float] = []
            turn_tp_cand: list[list[int]] = []
            turn_fp_cand: list[list[int]] = []
            turn_fn_cand: list[list[int]] = []

            for i in range(b_size):
                prev_stats = compute_sufficient_statistics(
                    prev_labels[i], gt_masks[i], classes=contract.classes
                )
                cand_stats = compute_sufficient_statistics(
                    cand_labels[i], gt_masks[i], classes=contract.classes
                )
                _, q_p = compute_dice_from_stats(prev_stats, contract)
                _, q_c = compute_dice_from_stats(cand_stats, contract)
                d = float(q_c - q_p) if np.isfinite(q_c) and np.isfinite(q_p) else float("nan")

                turn_q_prev.append(q_p)
                turn_q_cand.append(q_c)
                turn_deltas.append(d)

                tp_row = [0] * num_classes
                fp_row = [0] * num_classes
                fn_row = [0] * num_classes
                for c in contract.classes:
                    tp_row[c] = cand_stats.tp.get(c, 0)
                    fp_row[c] = cand_stats.fp.get(c, 0)
                    fn_row[c] = cand_stats.fn.get(c, 0)
                turn_tp_cand.append(tp_row)
                turn_fp_cand.append(fp_row)
                turn_fn_cand.append(fn_row)

            batch_quality.append(delta_q.detach().reshape(-1).cpu().float())
            batch_actual.append(torch.as_tensor(turn_deltas, dtype=torch.float32))
            batch_active.append(active_mask.detach().cpu())
            batch_q_prev.append(torch.as_tensor(turn_q_prev, dtype=torch.float32))
            batch_q_cand.append(torch.as_tensor(turn_q_cand, dtype=torch.float32))
            batch_tp_cand.append(torch.as_tensor(turn_tp_cand, dtype=torch.int64))
            batch_fp_cand.append(torch.as_tensor(turn_fp_cand, dtype=torch.int64))
            batch_fn_cand.append(torch.as_tensor(turn_fn_cand, dtype=torch.int64))

        if batch_quality:
            quality_values.append(torch.stack(batch_quality, dim=1))
            actual_values.append(torch.stack(batch_actual, dim=1))
            legacy_actual_values.append(torch.stack(batch_legacy_actual, dim=1))
            active_values.append(torch.stack(batch_active, dim=1))
            q_prev_values.append(torch.stack(batch_q_prev, dim=1))
            q_cand_values.append(torch.stack(batch_q_cand, dim=1))
            tp_cand_values.append(torch.stack(batch_tp_cand, dim=1))
            fp_cand_values.append(torch.stack(batch_fp_cand, dim=1))
            fn_cand_values.append(torch.stack(batch_fn_cand, dim=1))

    if not initial_values:
        raise ValueError("Cannot cache threshold transitions from an empty validation loader")

    provenance = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "metric_space": contract.metric_space,
        "empty_class_policy": contract.empty_policy,
        "metric_contract": contract.name,
        "metric_contract_version": contract.version,
        "neutral_margin": contract.neutral_margin,
        "excluded_empty_slice_count": 0,
    }

    size = int(torch.cat(initial_values).shape[0])
    if not quality_values:
        empty = torch.empty((size, 0), dtype=torch.float32)
        empty_stats = torch.empty((size, 0, num_classes), dtype=torch.int64)
        result = {
            "initial_dice": torch.cat(initial_values),
            "delta_q": empty,
            "actual_delta_dice": empty,
            "legacy_actual_delta_dice": empty,
            "active_mask": empty.bool(),
            "q_previous": empty,
            "q_candidate": empty,
            "tp_initial": torch.cat(tp_initial_values),
            "fp_initial": torch.cat(fp_initial_values),
            "fn_initial": torch.cat(fn_initial_values),
            "tp_candidate": empty_stats,
            "fp_candidate": empty_stats,
            "fn_candidate": empty_stats,
            **provenance,
        }
    else:
        result = {
            "initial_dice": torch.cat(initial_values),
            "delta_q": torch.cat(quality_values),
            "actual_delta_dice": torch.cat(actual_values),
            "legacy_actual_delta_dice": torch.cat(legacy_actual_values),
            "active_mask": torch.cat(active_values),
            "q_previous": torch.cat(q_prev_values),
            "q_candidate": torch.cat(q_cand_values),
            "tp_initial": torch.cat(tp_initial_values),
            "fp_initial": torch.cat(fp_initial_values),
            "fn_initial": torch.cat(fn_initial_values),
            "tp_candidate": torch.cat(tp_cand_values),
            "fp_candidate": torch.cat(fp_cand_values),
            "fn_candidate": torch.cat(fn_cand_values),
            **provenance,
        }

    if all_case_ids and len(all_case_ids) == size:
        result["case_ids"] = all_case_ids
    return result


def joint_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    lambda_audit: float = 1.0,
    neutral_margin: float = 0.005,
    local_weighting: bool = True,
) -> torch.Tensor:
    """Backward-compatible scalar Phase-C objective."""

    return compute_joint_losses(
        model,
        batch,
        tau_accept=tau_accept,
        t_max=t_max,
        lambda_audit=lambda_audit,
        neutral_margin=neutral_margin,
        local_weighting=local_weighting,
    )[0]


def finetune_joint_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    tau_accept: float = 0.0,
    t_max: int = 3,
    lambda_audit: float = 1.0,
    neutral_margin: float = 0.005,
    local_weighting: bool = True,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    gradient_accumulation_steps: int = 1,
    grad_clip: float | None = 3.0,
    epoch: int = 0,
    total_epochs: int = 1,
    max_steps: int | None = None,
    disable_tqdm: bool = False,
) -> dict[str, float]:
    model.train()
    accumulation_steps = validate_accumulation_steps(gradient_accumulation_steps)
    if max_steps is not None and int(max_steps) < 1:
        raise ValueError("max_steps must be positive when provided")
    if scaler is None:
        scaler = build_grad_scaler(enabled=amp_enabled, device=device, dtype=amp_dtype)
    running = 0.0
    running_annotation = 0.0
    running_audit = 0.0
    transitions = 0
    steps = 0
    optimizer_steps = 0
    pending = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch + 1:03d}/{total_epochs:03d} [Train C]",
        disable=disable_tqdm,
        leave=False,
    )
    for batch_index, raw_batch in enumerate(pbar):
        if max_steps is not None and optimizer_steps >= int(max_steps):
            break
        batch = move_batch(raw_batch, device)
        with autocast_context(enabled=amp_enabled, device=device, dtype=amp_dtype):
            loss, details = compute_joint_losses(
                model,
                batch,
                tau_accept=tau_accept,
                t_max=t_max,
                lambda_audit=lambda_audit,
                neutral_margin=neutral_margin,
                local_weighting=local_weighting,
            )
        if not is_finite(loss):
            raise FloatingPointError(
                f"Non-finite Phase-C loss at epoch={epoch} step={batch_index}: "
                f"loss={float(loss.detach())!r} annotation={float(details['annotation_loss'])!r} "
                f"audit={float(details['audit_loss'])!r}"
            )
        scaled_loss = loss / float(accumulation_steps)
        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        pending += 1
        running += float(loss.detach())
        running_annotation += float(details["annotation_loss"])
        running_audit += float(details["audit_loss"])
        transitions += int(details["transition_count"])
        steps += 1
        if pending == accumulation_steps:
            step_ok, _ = finalize_optimizer_step(
                model,
                optimizer,
                scaler,
                scheduler=scheduler,
                pending_batches=pending,
                accumulation_steps=accumulation_steps,
                grad_clip=grad_clip,
            )
            if not step_ok:
                raise FloatingPointError(f"Non-finite Phase-C gradients at epoch={epoch} step={batch_index}")
            pending = 0
            optimizer_steps += 1
        pbar.set_postfix({
            "loss": f"{float(loss.detach()):.4f}",
            "annot": f"{float(details['annotation_loss']):.4f}",
            "audit": f"{float(details['audit_loss']):.4f}",
            "trans": f"{int(details['transition_count'])}",
            "lr": f"{float(optimizer.param_groups[0]['lr']):.2e}",
        })
    if pending:
        step_ok, _ = finalize_optimizer_step(
            model,
            optimizer,
            scaler,
            scheduler=scheduler,
            pending_batches=pending,
            accumulation_steps=accumulation_steps,
            grad_clip=grad_clip,
        )
        if not step_ok:
            raise FloatingPointError(f"Non-finite Phase-C gradients at epoch={epoch} final_partial_step")
        optimizer_steps += 1
    divisor = max(steps, 1)
    return {
        "loss": running / divisor,
        "annotation_loss": running_annotation / divisor,
        "audit_loss": running_audit / divisor,
        "transitions": float(transitions),
        "optimizer_steps": float(optimizer_steps),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Audit Phase C joint fine-tuning")
    parser.add_argument("--config", default="configs/self_audit_joint.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_manifest", default=None, help="Override split manifest JSON path")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lambda_audit", type=float, default=None)
    parser.add_argument("--tau_accept", type=float, default=None)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    add_wandb_and_tqdm_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.split_manifest is not None:
        config["split_manifest"] = args.split_manifest
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.image_size is not None:
        config["image_size"] = args.image_size
    device = resolve_device(args.device or config.get("device"))
    seed_everything(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", False)))
    split_stats = validate_dataset_splits(config)
    print(f"split_stats={split_stats}")
    model = build_model_from_config(config, device)
    initial_checkpoint = args.resume or args.checkpoint
    if initial_checkpoint is None:
        raise ValueError("Phase C requires --checkpoint or --resume")
    train_dataset = build_patient_dataset(config, split=str(config.get("train_split", "train")), train=True)
    val_dataset = build_patient_dataset(config, split=str(config.get("val_split", "val")), train=False)
    loader = build_data_loader(train_dataset, config, device=device, train=True)
    val_loader = build_data_loader(val_dataset, config, device=device, train=False)
    optimizer = encoder_head_optimizer(
        model,
        lr=float(config.get("joint_lr", 1e-5)),
        encoder_lr=float(config.get("joint_encoder_lr", 1e-6)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(args.epochs or config.get("epochs", 10))
    accumulation_steps = validate_accumulation_steps(config.get("gradient_accumulation_steps", 1))
    scheduler_config = dict(config)
    scheduler_config["accumulation_steps"] = accumulation_steps
    scheduler = build_training_scheduler(optimizer, scheduler_config, num_batches=len(loader), epochs=epochs)
    amp_enabled, amp_dtype = resolve_amp(config, device)
    scaler = build_grad_scaler(enabled=amp_enabled, device=device, dtype=amp_dtype)
    wandb_logger = setup_wandb_logger(args, config, phase="joint")
    start_epoch = 0
    best_metric = float("-inf")
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch, _, _ = checkpoint_progress(payload)
        best_metric = float(payload.get("best_metric", float("-inf")))
    else:
        load_checkpoint(args.checkpoint, model=model, map_location=device)
    print_model_parameter_summary(model, title="Phase C: Joint Model Parameters (All Unfrozen)")
    audit_config = config.get("audit", {})
    if not isinstance(audit_config, dict):
        raise ValueError("audit config must be a mapping")
    lambda_audit = float(args.lambda_audit if args.lambda_audit is not None else config.get("lambda_audit", 1.0))
    tau_accept = float(args.tau_accept if args.tau_accept is not None else audit_config.get("tau_accept", 0.0))
    t_max = int(args.t_max if args.t_max is not None else audit_config.get("t_max", config.get("model", {}).get("max_turns", 3)))
    output_target = Path(config.get("output", "weights/self_audit/phase_c_joint.pt"))
    output_dir = output_target.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for epoch in range(start_epoch, epochs):
            stats = finetune_joint_epoch(
                model,
                loader,
                optimizer,
                device,
                tau_accept=tau_accept,
                t_max=t_max,
                lambda_audit=lambda_audit,
                neutral_margin=float(audit_config.get("neutral_margin", 0.005)),
                local_weighting=audit_config.get("local_class_weighting", True) != "none",
                scheduler=scheduler,
                scaler=scaler,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                gradient_accumulation_steps=accumulation_steps,
                grad_clip=config.get("grad_clip", 3.0),
                epoch=epoch,
                total_epochs=epochs,
                max_steps=args.max_steps,
                disable_tqdm=args.no_tqdm,
            )
            validation = validate_phase_c(
                model,
                val_loader,
                device,
                tau_accept=tau_accept,
                t_max=t_max,
                max_batches=args.max_val_batches,
                epoch=epoch,
                total_epochs=epochs,
                disable_tqdm=args.no_tqdm,
            )
            metric = float(validation["final_foreground_macro_dice"])
            is_best = is_finite(metric) and metric > best_metric
            if is_best:
                best_metric = metric
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                config=config,
                extra={"best_metric": best_metric, "phase": "joint"},
            )
            if is_best:
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    config=config,
                    extra={"best_metric": best_metric, "phase": "joint"},
                )
                save_checkpoint(
                    output_target,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    config=config,
                    extra={"best_metric": best_metric, "phase": "joint"},
                )
            log_payload = {
                "epoch": epoch + 1,
                "train/loss": stats["loss"],
                "train/annotation_loss": stats["annotation_loss"],
                "train/audit_loss": stats["audit_loss"],
                "train/lr": stats["lr"],
                "val/initial_dice": validation["initial_foreground_macro_dice"],
                "val/final_dice": validation["final_foreground_macro_dice"],
                "val/net_gain": validation["net_gain"],
                "val/harmful_acceptance_rate": validation["harmful_acceptance_rate"],
                "val/beneficial_rejection_rate": validation["beneficial_rejection_rate"],
                "val/mean_attempted_turns": validation["mean_attempted_turns"],
                "val/mean_accepted_turns": validation["mean_accepted_turns"],
                "val/audit_auroc": validation["audit_auroc"],
                "val/audit_fix_f1": validation["audit_local_fix_f1"],
                "val/audit_regress_f1": validation["audit_local_regress_f1"],
                "best_final_macro_dice": best_metric,
            }
            wandb_logger.log(log_payload, step=epoch + 1)
            print(
                f"epoch={epoch + 1:03d} lr={stats['lr']:.3e} loss={stats['loss']:.5f} "
                f"annotation={stats['annotation_loss']:.5f} audit={stats['audit_loss']:.5f} "
                f"initial_dice={validation['initial_foreground_macro_dice']:.4f} "
                f"final_dice={validation['final_foreground_macro_dice']:.4f} "
                f"net_gain={validation['net_gain']:.4f} "
                f"harmful_acceptance={validation['harmful_acceptance_rate']:.4f} "
                f"beneficial_rejection={validation['beneficial_rejection_rate']:.4f} "
                f"mean_attempted={validation['mean_attempted_turns']:.3f} "
                f"mean_accepted={validation['mean_accepted_turns']:.3f} "
                f"audit_AUROC={validation['audit_auroc']:.4f} "
                f"audit_FIX_F1={validation['audit_local_fix_f1']:.4f} "
                f"audit_REGRESS_F1={validation['audit_local_regress_f1']:.4f} "
                f"audit_corr={validation['audit_correlation_delta_q_delta_dice']:.4f} "
                f"tau={tau_accept:.4f}"
            )
            if args.max_steps is not None:
                break

        if args.visualize:
            try:
                from self_audit.evaluation.visualizer import plot_phase_c_audit_trace, save_figure
                model.eval()
                sample_imgs, sample_msks, sample_inits, sample_finals, sample_cids = [], [], [], [], []
                sample_candidates, sample_halts = [], []
                with torch.no_grad():
                    for v_batch in val_loader:
                        v_batch = move_batch(v_batch, device)
                        out = model.infer(v_batch["image"], mode="self_audit", tau_accept=float(tau_accept), t_max=int(t_max))
                        init_log = out.get("initial_logits", out.get("a0_logits", out.get("logits")))
                        fin_log = out.get("logits")
                        cands = out.get("transition_candidates", [])
                        halts = out.get("halt_turn")
                        for idx in range(v_batch["image"].shape[0]):
                            sample_imgs.append(v_batch["image"][idx].cpu())
                            sample_msks.append(v_batch["mask"][idx].cpu())
                            sample_inits.append(init_log[idx].argmax(dim=0).cpu())
                            sample_finals.append(fin_log[idx].argmax(dim=0).cpu())
                            if cands and len(cands) > 0 and idx < cands[0].shape[0]:
                                sample_candidates.append(cands[0][idx].argmax(dim=0).cpu())
                            sample_cids.append(v_batch.get("case_id", [f"Case_{len(sample_imgs)}"])[idx] if isinstance(v_batch.get("case_id"), list) else f"Case_{len(sample_imgs)}")
                            if halts is not None and torch.is_tensor(halts) and idx < len(halts):
                                sample_halts.append(int(halts[idx].cpu()))
                            if len(sample_imgs) >= max(int(args.vis_samples), 1):
                                break
                        if len(sample_imgs) >= max(int(args.vis_samples), 1):
                            break
                if sample_imgs:
                    fig = plot_phase_c_audit_trace(
                        sample_imgs[:args.vis_samples],
                        sample_msks[:args.vis_samples],
                        sample_inits[:args.vis_samples],
                        sample_finals[:args.vis_samples],
                        transition_candidates=sample_candidates[:args.vis_samples] if sample_candidates else None,
                        halted_turns=sample_halts[:args.vis_samples] if sample_halts else None,
                        tau_accept=float(tau_accept),
                        case_ids=sample_cids[:args.vis_samples],
                        title=f"Phase C Self-Audit Decisions (Epoch {epoch+1})",
                    )
                    vis_path = Path(args.vis_dir) / "phase_c_val_trace.png"
                    saved_img = save_figure(fig, vis_path)
                    wandb_logger.log_images({"val/phase_c_audit_trace": saved_img}, step=epoch + 1)
                    print(f"visualizations_saved={saved_img}")
            except Exception as vis_err:
                print(f"[Visualizer Warning] Failed to export Phase C visualization: {vis_err}")
    finally:
        wandb_logger.finish()
    print(f"saved_last={output_dir / 'last.pt'} saved_best={output_dir / 'best.pt'} saved_target={output_target}")


if __name__ == "__main__":  # pragma: no cover
    main()
