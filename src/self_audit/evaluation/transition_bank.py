"""Frozen transition-bank export (W4): proposal generation, evaluator, strict JSON.

A transition bank is a frozen, replayable record of *what the model proposed*
and *how good those proposals actually turned out to be*.  The two halves are
deliberately separate modules of work:

``generate_on_policy_proposals`` / ``generate_synthetic_proposals``
    Produce real transitions by running a real model.  The deployable
    on-policy entry point takes an image tensor plus identity metadata and
    nothing else -- there is no parameter through which query ground truth
    could reach the rollout, and a batch *mapping* is refused outright
    because the dataset batch mapping carries ``mask``.

    SCOPE NOTE: The GT-free claim is strictly scoped to this proposal
    generation API (and ``model.infer``), which receives no ground truth.
    Upstream dataset loader selection (such as ``foreground_only=True`` in
    volume slice datasets) can consume reference GT during sample/slice
    selection prior to calling this API; an end-to-end GT-free claim
    requires verifying that the data loader does not perform GT-dependent
    slice filtering.
``TransitionEvaluator``
    Attaches the actual quality ``Q`` *after* generation, under an explicit
    versioned metric contract.  Nothing it computes can change a proposal, a
    state identifier, a predicted ``delta_q`` or a gate decision.

Synthetic proposals may use training ground truth -- that is what a
counterfactual edit is -- and every such row is stamped
``gt_used_in_generation=True``.  Those rows are never a GT-free deployment
claim.

Three further properties are structural rather than advisory:

* A state identifier is a digest of the *actual state tensor content* and its
  declared representation, never of a stage number.  Two different stages that
  hold the same tensor share an id; the same stage recomputed from mutated
  weights does not.
* An on-policy sample that is rejected HALTs.  No later transition is emitted
  for it, and :func:`validate_bank` rejects a bank that contains one.
* An undefined quality score (every foreground class empty in both prediction
  and reference under the ``exclude`` policy) keeps its row, its coverage
  counters and its explicit ``*_defined=False`` flag.  It is serialized as
  JSON ``null``; ``NaN`` never reaches the file.

This module builds a *tested export path*.  It does not create, and must not
be described as creating, a real bank in this environment: that needs a real
trained checkpoint and a real cohort.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ..audit.counterfactual import CounterfactualGenerator
from ..audit.semantics import (
    DEFAULT_NEUTRAL_MARGIN,
    METRIC_SPACE_SLICE_PROXY,
    classify_delta,
    resolve_neutral_margin,
)
from ..audit.targets import LOCAL_AUDIT_NAMES
from ..provenance import (
    MEMBERSHIP_COVERAGE,
    UNKNOWN,
    cohort_identity,
    git_source_provenance,
    preprocessing_descriptor,
    resolve_model_identity,
    state_digest,
)
from .contracts import (
    ContractMismatchError,
    MetricContract,
    SufficientStatistics,
    compute_dice_from_stats,
    resolve_metric_contract,
    score_transition,
    validate_contract,
)
from ..artifact_io import atomic_write_json
from .threshold import json_safe

BANK_SCHEMA_VERSION = 1

#: Where a transition physically came from.
SOURCE_ON_POLICY = "on_policy"
SOURCE_SYNTHETIC = "synthetic"
SOURCE_ALWAYS_ACCEPT_PREFIX = "always_accept_prefix"
TRANSITION_SOURCES = (SOURCE_ON_POLICY, SOURCE_SYNTHETIC, SOURCE_ALWAYS_ACCEPT_PREFIX)

#: Which rollout produced the trajectory.  This is recorded separately from
#: the gate decision: ``always_accept_refinement`` walks past a transition the
#: threshold gate would have rejected, so it is an analysis trajectory and is
#: never the deployable one.
ROLLOUT_SELF_AUDIT = "self_audit"
ROLLOUT_ALWAYS_ACCEPT = "always_accept_refinement"
ROLLOUT_SYNTHETIC = "synthetic_perturbation"
ROLLOUT_POLICIES = (ROLLOUT_SELF_AUDIT, ROLLOUT_ALWAYS_ACCEPT, ROLLOUT_SYNTHETIC)

_SOURCE_FOR_ROLLOUT = {
    ROLLOUT_SELF_AUDIT: SOURCE_ON_POLICY,
    ROLLOUT_ALWAYS_ACCEPT: SOURCE_ALWAYS_ACCEPT_PREFIX,
    ROLLOUT_SYNTHETIC: SOURCE_SYNTHETIC,
}

#: Declared representations a state identifier may be taken over.  The label
#: is part of the hashed payload, so the same bytes read as logits and as
#: probabilities do not collide.
STATE_REPRESENTATIONS = ("logits_v1", "probs_v1", "class_labels_v1")

#: Channel semantics of the auditor's local evidence map.
LOCAL_EVIDENCE_CHANNELS = tuple(LOCAL_AUDIT_NAMES)
LOCAL_EVIDENCE_SEMANTICS = "auditor_local_softmax_mean_over_pixels_v1"

#: The bank's own integrity signature *is* a content hash of its rows and
#: header.  The split membership signature carried inside the provenance block
#: is not: it stays labelled with the narrower guarantee it actually has.
BANK_CONTENT_COVERAGE = "bank_header_and_rows_content_hash"

_ROW_REQUIRED_FIELDS = (
    "schema_version",
    "patient_id",
    "case_id",
    "slice_index",
    "stage",
    "trajectory_id",
    "state_representation",
    "previous_state_id",
    "candidate_state_id",
    "delta_q",
    "accepted",
    "hypothetical_gate",
    "source",
    "rollout_policy",
    "gt_used_in_generation",
    "generation_operation",
    "generation_valid",
    "halted_after",
    "local_evidence",
)

_EVALUATED_REQUIRED_FIELDS = (
    "q_previous",
    "q_candidate",
    "delta_dice",
    "q_previous_defined",
    "q_candidate_defined",
    "delta_defined",
    "delta_class",
    "neutral_margin",
    "per_class_dice_previous",
    "per_class_dice_candidate",
    "metric_contract",
    "metric_contract_version",
    "metric_space",
    "empty_policy",
    "sufficient_statistics",
)

#: A state identifier is the declared representation plus a full SHA-256 hex
#: digest.  A truncated or upper-cased digest is refused rather than accepted
#: as "close enough".
_STATE_ID_DIGEST = re.compile(r"^[0-9a-f]{64}$")

#: Required keys of a complete bound-export checkpoint identity.
_BOUND_CHECKPOINT_FIELDS = (
    "checkpoint_path",
    "checkpoint_sha256",
    "state_digest",
    "file_state_digest",
    "model_identity",
    "producer",
)

EXPORT_BOUND = "bound_checkpoint_export"
EXPORT_UNVERIFIED_DEMO = "unverified_demo_export"
EXPORT_IDENTITY_CLASSES = (EXPORT_BOUND, EXPORT_UNVERIFIED_DEMO)


class BankValidationError(ValueError):
    """Raised when a bank payload fails a required identity or contract check."""


# ---------------------------------------------------------------------------
# Strict scalar readers
# ---------------------------------------------------------------------------


def _require_int(value: Any, where: str) -> int:
    """Read an integer without accepting ``True`` or truncating a float."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise BankValidationError(f"{where} must be an integer, got {value!r}")
    return int(value)


def _require_bool(value: Any, where: str) -> bool:
    """Read a boolean without accepting ``1``/``0`` or a non-empty string."""

    if not isinstance(value, bool):
        raise BankValidationError(f"{where} must be a boolean, got {value!r}")
    return bool(value)


def _require_finite_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BankValidationError(f"{where} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise BankValidationError(
            f"{where} must be finite, got {value!r}; a non-finite score is never a valid "
            "measurement and must not be silently serialized as null."
        )
    return number


def _require_nonempty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BankValidationError(f"{where} must be a non-empty string, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# State identity
# ---------------------------------------------------------------------------


def state_tensor_id(state: Tensor, *, representation: str = "logits_v1") -> str:
    """Return a deterministic identifier for one sample's state tensor.

    The identifier is a digest of the tensor's *content*, shape, dtype and its
    declared ``representation``.  It is explicitly not derived from the stage
    index, the sample index, or anything else about where the state sat in a
    trajectory: a rejected candidate that is discarded and a later state that
    happens to hold the same values are the same state and get the same id.
    """

    if representation not in STATE_REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {list(STATE_REPRESENTATIONS)}, got {representation!r}"
        )
    if not torch.is_tensor(state):
        raise TypeError(f"state must be a tensor, got {type(state).__name__}")
    if state.ndim not in (2, 3):
        raise ValueError(
            "state_tensor_id identifies ONE sample: expected [C,H,W] or [H,W], "
            f"got {tuple(state.shape)}. Index the batch first so ids are not batch-dependent."
        )
    digest = state_digest({"representation": representation, "state": state.detach()})
    return f"{representation}:sha256:{digest}"


def state_identity(state: Tensor, *, representation: str = "logits_v1") -> dict[str, Any]:
    """Full state descriptor: id plus the representation facts it was taken over."""

    return {
        "state_id": state_tensor_id(state, representation=representation),
        "representation": str(representation),
        "shape": [int(dim) for dim in state.shape],
        "dtype": str(state.dtype),
    }


def _labels_from_state(state: Tensor, *, representation: str) -> np.ndarray:
    if representation == "class_labels_v1":
        return state.detach().to("cpu").to(torch.int64).numpy()
    return state.detach().to("cpu").argmax(dim=0).to(torch.int64).numpy()


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionProposal:
    """One generated transition, before any ground truth has been consulted.

    ``previous_labels`` / ``candidate_labels`` are the hard label maps the
    evaluator scores.  They are derived from the same tensors the state ids
    were taken over and are deliberately not part of the serialized row.
    """

    patient_id: str
    case_id: str
    slice_index: int
    stage: int
    trajectory_id: str
    state_representation: str
    previous_state_id: str
    candidate_state_id: str
    delta_q: float
    accepted: bool
    hypothetical_accept: bool
    tau_accept: float
    source: str
    rollout_policy: str
    gt_used_in_generation: bool
    generation_operation: str
    generation_valid: bool
    halted_after: bool
    local_evidence: dict[str, Any]
    previous_labels: np.ndarray
    candidate_labels: np.ndarray
    schema_version: int = BANK_SCHEMA_VERSION

    def key(self) -> tuple[str, str, int]:
        return (str(self.patient_id), str(self.case_id), int(self.slice_index))

    def to_row(self) -> dict[str, Any]:
        """The serializable proposal half of a bank row.  No GT-derived field."""

        return {
            "schema_version": int(self.schema_version),
            "patient_id": str(self.patient_id),
            "case_id": str(self.case_id),
            "slice_index": int(self.slice_index),
            "stage": int(self.stage),
            "trajectory_id": str(self.trajectory_id),
            "state_representation": str(self.state_representation),
            "previous_state_id": str(self.previous_state_id),
            "candidate_state_id": str(self.candidate_state_id),
            "delta_q": float(self.delta_q),
            "accepted": bool(self.accepted),
            "hypothetical_gate": {
                "tau_accept": float(self.tau_accept),
                "accept": bool(self.hypothetical_accept),
                "note": (
                    "threshold decision recomputed from delta_q; under "
                    f"{ROLLOUT_ALWAYS_ACCEPT!r} the rollout ignores it"
                ),
            },
            "source": str(self.source),
            "rollout_policy": str(self.rollout_policy),
            "gt_used_in_generation": bool(self.gt_used_in_generation),
            "generation_operation": str(self.generation_operation),
            "generation_valid": bool(self.generation_valid),
            "halted_after": bool(self.halted_after),
            "local_evidence": dict(self.local_evidence),
        }


def _identity_lists(
    batch_size: int,
    patient_ids: Sequence[Any],
    case_ids: Sequence[Any],
    slice_indices: Sequence[Any],
) -> tuple[list[str], list[str], list[int]]:
    lengths = {len(patient_ids), len(case_ids), len(slice_indices)}
    if lengths != {batch_size}:
        raise ValueError(
            "patient_ids/case_ids/slice_indices must each have one entry per sample: "
            f"batch={batch_size}, got {len(patient_ids)}/{len(case_ids)}/{len(slice_indices)}"
        )
    patients = [str(value.item()) if torch.is_tensor(value) else str(value) for value in patient_ids]
    cases = [str(value.item()) if torch.is_tensor(value) else str(value) for value in case_ids]
    slices = [int(value.item()) if torch.is_tensor(value) else int(value) for value in slice_indices]
    return patients, cases, slices


def _local_evidence_summary(local_logits: Tensor) -> dict[str, Any]:
    probabilities = local_logits.detach().float().softmax(dim=0)
    mean = probabilities.mean(dim=(1, 2))
    argmax = probabilities.argmax(dim=0)
    pixels = int(argmax.numel())
    fractions = [float((argmax == index).sum().item()) / float(pixels) for index in range(probabilities.shape[0])]
    return {
        "channels": list(LOCAL_EVIDENCE_CHANNELS),
        "semantics": LOCAL_EVIDENCE_SEMANTICS,
        "description": (
            "mean auditor local softmax probability per channel over all pixels, plus the "
            "fraction of pixels whose argmax channel is that channel"
        ),
        "mean_probability": {
            name: float(mean[index].item()) for index, name in enumerate(LOCAL_EVIDENCE_CHANNELS)
        },
        "argmax_pixel_fraction": {
            name: fractions[index] for index, name in enumerate(LOCAL_EVIDENCE_CHANNELS)
        },
        "pixels": pixels,
    }


def generate_on_policy_proposals(
    model: nn.Module,
    images: Tensor,
    *,
    patient_ids: Sequence[Any],
    case_ids: Sequence[Any],
    slice_indices: Sequence[Any],
    tau_accept: float = 0.0,
    t_max: int | None = None,
    rollout_policy: str = ROLLOUT_SELF_AUDIT,
) -> list[TransitionProposal]:
    """Generate real rollout transitions.  This API cannot see query GT.

    ``images`` must be the image tensor itself.  A dataset batch *mapping* is
    refused: that mapping carries ``mask``, and accepting it would put query
    ground truth one dictionary lookup away from the deployable rollout.

    Under :data:`ROLLOUT_SELF_AUDIT` a rejected sample halts and no later
    transition is produced for it.  Under :data:`ROLLOUT_ALWAYS_ACCEPT` the
    rollout walks to the cap regardless of the gate; those rows are tagged
    :data:`SOURCE_ALWAYS_ACCEPT_PREFIX` and their ``hypothetical_gate`` records
    what the threshold *would* have decided.

    SCOPE NOTE: The GT-free claim is strictly scoped to this proposal
    generation API, which does not accept or inspect query ground truth.
    Upstream dataset loaders configured with ``foreground_only=True`` can
    consume reference GT to filter slices before this API is called; full
    pipeline GT-freedom requires ensuring the loader does not perform
    GT-dependent slice selection.
    """

    if isinstance(images, Mapping):
        raise TypeError(
            "Deployable proposal generation does not accept a batch mapping: a dataset "
            "batch carries 'mask' (query ground truth). Pass batch['image'] and the "
            "identity metadata explicitly."
        )
    if not torch.is_tensor(images):
        raise TypeError(f"images must be a tensor, got {type(images).__name__}")
    if rollout_policy not in (ROLLOUT_SELF_AUDIT, ROLLOUT_ALWAYS_ACCEPT):
        raise ValueError(
            f"rollout_policy must be {ROLLOUT_SELF_AUDIT!r} or {ROLLOUT_ALWAYS_ACCEPT!r} "
            f"for on-policy generation, got {rollout_policy!r}"
        )

    batch_size = int(images.shape[0])
    patients, cases, slices = _identity_lists(batch_size, patient_ids, case_ids, slice_indices)
    source = _SOURCE_FOR_ROLLOUT[rollout_policy]
    tau = float(tau_accept)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model.infer(
                images,
                mode=rollout_policy,
                tau_accept=tau,
                t_max=t_max,
            )
    finally:
        model.train(was_training)

    previous_states = output["transition_previous"]
    candidate_states = output["transition_candidates"]
    audits = output["audits"]

    proposals: list[TransitionProposal] = []
    halted: list[bool] = [False] * batch_size
    for stage, (previous, candidate, audit) in enumerate(
        zip(previous_states, candidate_states, audits)
    ):
        attempted = audit["active_mask"].detach().cpu()
        accepted = audit["accepted"].detach().cpu()
        delta_q = audit["delta_q"].detach().cpu().reshape(batch_size, -1)[:, 0]
        local_logits = audit["local_logits"].detach().cpu()
        for index in range(batch_size):
            if not bool(attempted[index]):
                continue
            if halted[index]:
                raise BankValidationError(
                    f"Rollout emitted stage {stage} for sample {index} after it halted; "
                    "a rejected on-policy sample must produce no later transition."
                )
            previous_row = previous[index].detach().cpu()
            candidate_row = candidate[index].detach().cpu()
            row_delta_q = float(delta_q[index].item())
            row_accepted = bool(accepted[index])
            trajectory = f"{source}:{patients[index]}:{cases[index]}:{slices[index]}:{rollout_policy}"
            proposals.append(
                TransitionProposal(
                    patient_id=patients[index],
                    case_id=cases[index],
                    slice_index=slices[index],
                    stage=int(stage),
                    trajectory_id=trajectory,
                    state_representation="logits_v1",
                    previous_state_id=state_tensor_id(previous_row, representation="logits_v1"),
                    candidate_state_id=state_tensor_id(candidate_row, representation="logits_v1"),
                    delta_q=row_delta_q,
                    accepted=row_accepted,
                    hypothetical_accept=bool(row_delta_q > tau),
                    tau_accept=tau,
                    source=source,
                    rollout_policy=rollout_policy,
                    gt_used_in_generation=False,
                    generation_operation="annotation_expert_transition",
                    generation_valid=True,
                    halted_after=bool(rollout_policy == ROLLOUT_SELF_AUDIT and not row_accepted),
                    local_evidence=_local_evidence_summary(local_logits[index]),
                    previous_labels=_labels_from_state(previous_row, representation="logits_v1"),
                    candidate_labels=_labels_from_state(candidate_row, representation="logits_v1"),
                )
            )
            if rollout_policy == ROLLOUT_SELF_AUDIT and not row_accepted:
                halted[index] = True
    return proposals


def generate_synthetic_proposals(
    model: nn.Module,
    images: Tensor,
    ground_truth: Tensor,
    *,
    patient_ids: Sequence[Any],
    case_ids: Sequence[Any],
    slice_indices: Sequence[Any],
    generator: CounterfactualGenerator | None = None,
    kind: str = "negative",
    operation: str | None = None,
    tau_accept: float = 0.0,
    draw_index: int = 0,
) -> list[TransitionProposal]:
    """Generate counterfactual transitions around the model's own initial state.

    This path *does* consume ground truth: a counterfactual repair or
    regression is defined relative to GT.  It is therefore a training-side
    diagnostic and every row it emits carries ``gt_used_in_generation=True``.
    It is never evidence about GT-free deployable behaviour.
    """

    if isinstance(images, Mapping):
        raise TypeError("images must be a tensor, not a batch mapping")
    batch_size = int(images.shape[0])
    patients, cases, slices = _identity_lists(batch_size, patient_ids, case_ids, slice_indices)
    sampler = generator if generator is not None else CounterfactualGenerator()
    tau = float(tau_accept)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            encoded = model.encode(images)
            shared = encoded["shared"]
            initial = model.initial_head(shared, output_size=images.shape[-2:])
            previous_probs = initial.softmax(dim=1)
            sample = sampler.generate(previous_probs, ground_truth, kind=kind, operation=operation)
            candidate_probs = sample.candidate_probs
            entropy_previous = -(
                previous_probs.clamp_min(1e-8) * previous_probs.clamp_min(1e-8).log()
            ).sum(dim=1, keepdim=True)
            entropy_candidate = -(
                candidate_probs.clamp_min(1e-8) * candidate_probs.clamp_min(1e-8).log()
            ).sum(dim=1, keepdim=True)
            audit = model.auditor(
                shared,
                previous_probs,
                candidate_probs,
                candidate_probs - previous_probs,
                entropy_previous=entropy_previous,
                entropy_candidate=entropy_candidate,
            )
    finally:
        model.train(was_training)

    delta_q = audit.delta_q.detach().cpu().reshape(batch_size, -1)[:, 0]
    local_logits = audit.local_logits.detach().cpu()
    valid_mask = sample.valid_mask
    proposals: list[TransitionProposal] = []
    for index in range(batch_size):
        previous_row = previous_probs[index].detach().cpu()
        candidate_row = candidate_probs[index].detach().cpu()
        row_delta_q = float(delta_q[index].item())
        row_accepted = bool(row_delta_q > tau)
        trajectory = (
            f"{SOURCE_SYNTHETIC}:{patients[index]}:{cases[index]}:{slices[index]}:"
            f"{sample.operation}:{int(draw_index)}"
        )
        proposals.append(
            TransitionProposal(
                patient_id=patients[index],
                case_id=cases[index],
                slice_index=slices[index],
                stage=0,
                trajectory_id=trajectory,
                state_representation="probs_v1",
                previous_state_id=state_tensor_id(previous_row, representation="probs_v1"),
                candidate_state_id=state_tensor_id(candidate_row, representation="probs_v1"),
                delta_q=row_delta_q,
                accepted=row_accepted,
                hypothetical_accept=row_accepted,
                tau_accept=tau,
                source=SOURCE_SYNTHETIC,
                rollout_policy=ROLLOUT_SYNTHETIC,
                gt_used_in_generation=True,
                generation_operation=str(sample.operation),
                generation_valid=bool(True if valid_mask is None else bool(valid_mask[index])),
                halted_after=False,
                local_evidence=_local_evidence_summary(local_logits[index]),
                previous_labels=_labels_from_state(previous_row, representation="probs_v1"),
                candidate_labels=_labels_from_state(candidate_row, representation="probs_v1"),
            )
        )
    return proposals


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _stats_row(stats: Any, classes: Sequence[int]) -> dict[str, dict[str, int]]:
    return {
        "tp": {str(int(c)): int(stats.tp.get(int(c), 0)) for c in classes},
        "fp": {str(int(c)): int(stats.fp.get(int(c), 0)) for c in classes},
        "fn": {str(int(c)): int(stats.fn.get(int(c), 0)) for c in classes},
    }


def _defined_or_none(value: float, defined: bool) -> float | None:
    return float(value) if defined else None


@dataclass
class TransitionEvaluator:
    """Attach actual quality to already-generated proposals.

    The evaluator is the only component that sees reference labels, and it
    runs strictly after generation.  It cannot alter a proposal: it returns new
    row dictionaries that carry the proposal's own fields verbatim plus the
    GT-derived block.
    """

    contract: MetricContract = field(default_factory=lambda: resolve_metric_contract(None))
    neutral_margin: float | None = None

    def __post_init__(self) -> None:
        self.contract = validate_contract(self.contract)
        if self.contract.metric_space != METRIC_SPACE_SLICE_PROXY:
            raise ContractMismatchError(
                "Transition-bank rows are per-slice proxies and reject volume contract "
                f"{self.contract.name!r} with metric_space={self.contract.metric_space!r}."
            )
        self.neutral_margin = resolve_neutral_margin(
            self.contract.neutral_margin if self.neutral_margin is None else self.neutral_margin
        )

    def evaluate(
        self,
        proposals: Sequence[TransitionProposal],
        *,
        targets: Mapping[tuple[str, str, int], Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        classes = self.contract.classes
        for proposal in proposals:
            key = proposal.key()
            if key not in targets:
                raise BankValidationError(
                    f"No reference label supplied for {key}; the evaluator refuses to score "
                    "a transition it has no ground truth for rather than dropping the row."
                )
            reference = targets[key]
            score = score_transition(
                proposal.previous_labels,
                proposal.candidate_labels,
                reference,
                self.contract,
            )
            per_class_previous, _ = compute_dice_from_stats(score.stats_previous, self.contract)
            per_class_candidate, _ = compute_dice_from_stats(score.stats_candidate, self.contract)
            delta_class = (
                int(classify_delta(score.delta_dice, self.neutral_margin))
                if score.is_defined_delta
                else None
            )
            row = proposal.to_row()
            row.update(
                {
                    "q_previous": _defined_or_none(score.q_previous, score.is_defined_previous),
                    "q_candidate": _defined_or_none(score.q_candidate, score.is_defined_candidate),
                    "delta_dice": _defined_or_none(score.delta_dice, score.is_defined_delta),
                    "q_previous_defined": bool(score.is_defined_previous),
                    "q_candidate_defined": bool(score.is_defined_candidate),
                    "delta_defined": bool(score.is_defined_delta),
                    "delta_class": delta_class,
                    "neutral_margin": float(self.neutral_margin),
                    "metric_contract": str(self.contract.name),
                    "metric_contract_version": int(self.contract.version),
                    "metric_space": str(self.contract.metric_space),
                    "empty_policy": str(self.contract.empty_policy),
                    "per_class_dice_previous": {
                        str(int(c)): (
                            float(per_class_previous[c])
                            if np.isfinite(per_class_previous[c])
                            else None
                        )
                        for c in classes
                    },
                    "per_class_dice_candidate": {
                        str(int(c)): (
                            float(per_class_candidate[c])
                            if np.isfinite(per_class_candidate[c])
                            else None
                        )
                        for c in classes
                    },
                    "sufficient_statistics": {
                        "previous": _stats_row(score.stats_previous, classes),
                        "candidate": _stats_row(score.stats_candidate, classes),
                        "note": (
                            "retained for every row, including rows whose macro score is "
                            "undefined, so empty-foreground coverage is never lost"
                        ),
                    },
                }
            )
            rows.append(row)
        return rows


# ---------------------------------------------------------------------------
# Provenance blocks
# ---------------------------------------------------------------------------


def generation_provenance_record(
    *,
    model: nn.Module | None = None,
    binding: Mapping[str, Any] | None = None,
    loader: Any = None,
    split_name: str = "val",
    rollout_policy: str = ROLLOUT_SELF_AUDIT,
    gt_used_in_generation: bool = False,
    max_batches: int | None = None,
    batch_size: int | None = None,
    observed_samples: int | None = None,
    unverified_demo_reason: str | None = None,
) -> dict[str, Any]:
    """Describe the state and cohort that *produced* the proposals.

    There are exactly two admissible shapes, and the record says which one it
    is in ``export_identity_class``:

    :data:`EXPORT_BOUND`
        A real export.  It requires a complete checkpoint binding (file hash,
        live state digest, on-disk state digest, resolved model identity,
        producer record) *and* a cohort identity.  Nothing here may be absent.
    :data:`EXPORT_UNVERIFIED_DEMO`
        A bank produced without a bound checkpoint -- a fixture or a smoke
        run.  It is admissible only with an explicit
        ``unverified_demo_reason`` and is never a measurement of a trained
        model.

    An absent checkpoint therefore cannot masquerade as a bound export, and a
    bound export cannot be declared while its identity fields are missing.
    """

    if rollout_policy not in ROLLOUT_POLICIES:
        raise ValueError(f"rollout_policy must be one of {list(ROLLOUT_POLICIES)}, got {rollout_policy!r}")
    if binding is None and not (isinstance(unverified_demo_reason, str) and unverified_demo_reason.strip()):
        raise ValueError(
            "A bank exported without a bound checkpoint must declare unverified_demo_reason; "
            "an unbound export is never reported as a measurement of a trained model."
        )
    if binding is not None and unverified_demo_reason is not None:
        raise ValueError("unverified_demo_reason is only for exports with no checkpoint binding")
    record: dict[str, Any] = {
        "generation_code": git_source_provenance(),
        "rollout_policy": str(rollout_policy),
        "gt_used_in_generation": bool(gt_used_in_generation),
        "checkpoint_bound": binding is not None,
        "export_identity_class": EXPORT_BOUND if binding is not None else EXPORT_UNVERIFIED_DEMO,
        "unverified_demo_reason": None if binding is not None else str(unverified_demo_reason),
        "checkpoint": None if binding is None else dict(binding),
        "model_identity": None if model is None else resolve_model_identity(model),
        "producer": (
            {"producer_git_sha": UNKNOWN, "producer_recorded": False}
            if binding is None
            else dict(dict(binding).get("producer", {}) or {"producer_git_sha": UNKNOWN})
        ),
    }
    if loader is not None:
        record["preprocessing"] = preprocessing_descriptor(loader)
        record["cohort"] = cohort_identity(
            loader,
            split_name=split_name,
            max_batches=max_batches,
            batch_size=batch_size,
            observed_samples=observed_samples,
        )
        record["split_coverage"] = MEMBERSHIP_COVERAGE
    else:
        record["cohort"] = None
        record["split_coverage"] = MEMBERSHIP_COVERAGE
    record["split_name"] = str(split_name)
    return record


def evaluation_provenance_record(
    *,
    contract: MetricContract | str | None = None,
    neutral_margin: float | None = None,
    reference_source: str = UNKNOWN,
) -> dict[str, Any]:
    """Describe the contract and reference labels the evaluator used.

    Kept separate from the generation record on purpose: re-scoring a frozen
    bank under a new contract changes this block and must not be readable as a
    change to the model that produced the rows.
    """

    resolved = validate_contract(resolve_metric_contract(contract))
    return {
        "evaluation_code": git_source_provenance(),
        "metric_contract": resolved.name,
        "metric_contract_version": int(resolved.version),
        "metric_contract_definition": resolved.to_dict(),
        "metric_space": resolved.metric_space,
        "empty_policy": resolved.empty_policy,
        "neutral_margin": float(
            resolved.neutral_margin if neutral_margin is None else resolve_neutral_margin(neutral_margin)
        ),
        "reference_source": str(reference_source),
        "evaluator_reads_gt": True,
    }


# ---------------------------------------------------------------------------
# Bank assembly, serialization and validation
# ---------------------------------------------------------------------------


def _canonical(payload: Any) -> str:
    return json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def bank_content_signature(bank: Mapping[str, Any]) -> str:
    """SHA-256 over the header and every row, excluding the integrity block."""

    payload = {key: value for key, value in bank.items() if key != "integrity"}
    return hashlib.sha256(f"transition_bank.v1\x00{_canonical(payload)}".encode("utf-8")).hexdigest()


def _sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("source")),
        str(row.get("patient_id")),
        str(row.get("case_id")),
        int(row.get("slice_index", -1)),
        str(row.get("trajectory_id")),
        int(row.get("stage", -1)),
    )


def _reject_nonfinite_scores(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject non-finite scores on the RAW rows, before JSON coercion.

    ``json_safe`` turns any non-finite float into ``null``.  Run unchecked,
    that would quietly convert a corrupt *defined* score into something the
    loader reads as a legitimately undefined one.  So the raw values are
    checked first: ``delta_q`` and ``tau_accept`` must always be finite, and a
    quality score may be non-finite only where its own ``*_defined`` flag says
    the score is undefined.
    """

    for position, row in enumerate(rows):
        if "delta_q" not in row:
            raise BankValidationError(f"Row {position} is missing delta_q")
        _require_finite_float(row["delta_q"], f"Row {position} delta_q")
        gate = row.get("hypothetical_gate")
        if isinstance(gate, Mapping):
            _require_finite_float(gate.get("tau_accept"), f"Row {position} hypothetical_gate.tau_accept")
        for name, flag in (
            ("q_previous", "q_previous_defined"),
            ("q_candidate", "q_candidate_defined"),
            ("delta_dice", "delta_defined"),
        ):
            if name not in row:
                continue
            value = row[name]
            if value is None:
                continue
            if _require_bool(row[flag], f"Row {position} {flag}"):
                _require_finite_float(value, f"Row {position} {name}")
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                raise BankValidationError(
                    f"Row {position} marks {flag} false but carries the finite value {value!r}"
                )


def build_bank(
    rows: Sequence[Mapping[str, Any]],
    *,
    generation: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble a validated bank payload from evaluated rows."""

    _reject_nonfinite_scores(rows)
    bank: dict[str, Any] = {
        "bank_schema_version": int(BANK_SCHEMA_VERSION),
        "generation": json_safe(dict(generation)),
        "evaluation": json_safe(dict(evaluation)),
        "protocol": json_safe(dict(protocol)),
        "rows": [json_safe(dict(row)) for row in sorted(rows, key=_sort_key)],
    }
    bank["integrity"] = {
        "coverage": BANK_CONTENT_COVERAGE,
        "num_rows": len(bank["rows"]),
        "content_signature": bank_content_signature(bank),
    }
    validate_bank(bank)
    return bank


def _stats_from_row(block: Any, classes: Sequence[int], where: str) -> SufficientStatistics:
    if not isinstance(block, Mapping):
        raise BankValidationError(f"{where} must be a mapping of tp/fp/fn counts")
    declared_classes = [int(c) for c in classes]
    declared_keys = {str(c) for c in declared_classes}
    counts: dict[str, dict[int, int]] = {}
    for name in ("tp", "fp", "fn"):
        entry = block.get(name)
        if not isinstance(entry, Mapping):
            raise BankValidationError(f"{where}.{name} must be a mapping keyed by class id")
        raw_keys = list(entry.keys())
        seen_norm: set[int] = set()
        for key in raw_keys:
            if not isinstance(key, str):
                raise BankValidationError(
                    f"{where}.{name} class key {key!r} must be a string, got {type(key).__name__}"
                )
            if key not in declared_keys:
                raise BankValidationError(
                    f"{where}.{name} contains undeclared or alias class key {key!r}; "
                    f"expected exact declared keys {sorted(declared_keys)}"
                )
            try:
                norm = int(key)
            except ValueError:
                raise BankValidationError(
                    f"{where}.{name} class key {key!r} is not a valid integer class id"
                )
            if norm in seen_norm:
                raise BankValidationError(
                    f"{where}.{name} contains duplicate/colliding class alias {key!r}"
                )
            seen_norm.add(norm)
        missing = sorted(declared_keys - set(raw_keys))
        if missing:
            raise BankValidationError(
                f"{where}.{name} is missing declared class(es): {missing}"
            )
        parsed: dict[int, int] = {}
        for c in declared_classes:
            k = str(c)
            value = entry[k]
            count = _require_int(value, f"{where}.{name}[{k}]")
            if count < 0:
                raise BankValidationError(
                    f"{where}.{name}[{k}] count must be non-negative, got {count}"
                )
            parsed[c] = count
        counts[name] = parsed
    return SufficientStatistics(tp=counts["tp"], fp=counts["fp"], fn=counts["fn"])


def _check_recomputed_quality(
    position: int,
    row: Mapping[str, Any],
    contract: MetricContract,
) -> None:
    """Recompute Q from the row's own sufficient statistics and enforce it.

    A content hash proves the file was not altered in transport.  It proves
    nothing about whether the numbers agree with each other, so every quality
    field is recomputed here from the counts the row itself carries, under the
    bank's declared contract, using the same helpers the evaluator used.
    """

    stats_block = row["sufficient_statistics"]
    if not isinstance(stats_block, Mapping):
        raise BankValidationError(f"Row {position} sufficient_statistics must be a mapping")
    classes = contract.classes
    for side, score_key, defined_key, per_class_key in (
        ("previous", "q_previous", "q_previous_defined", "per_class_dice_previous"),
        ("candidate", "q_candidate", "q_candidate_defined", "per_class_dice_candidate"),
    ):
        stats = _stats_from_row(
            stats_block.get(side), classes, f"Row {position} sufficient_statistics.{side}"
        )
        per_class, macro = compute_dice_from_stats(stats, contract)
        defined = _require_bool(row[defined_key], f"Row {position} {defined_key}")
        if bool(np.isfinite(macro)) != defined:
            raise BankValidationError(
                f"Row {position} {defined_key}={defined} contradicts its own sufficient "
                f"statistics, which recompute to {'a defined' if np.isfinite(macro) else 'an undefined'} score"
            )
        stored = row[score_key]
        if defined:
            if abs(float(stored) - float(macro)) > 1e-9:
                raise BankValidationError(
                    f"Row {position} {score_key}={stored!r} does not match the score recomputed "
                    f"from its sufficient statistics ({macro!r}) under contract {contract.name!r}"
                )
        elif stored is not None:
            raise BankValidationError(
                f"Row {position} {score_key} must be null when the score is undefined"
            )
        recorded_per_class = row[per_class_key]
        if not isinstance(recorded_per_class, Mapping):
            raise BankValidationError(f"Row {position} {per_class_key} must be a mapping")
        for cls in classes:
            expected = per_class[int(cls)]
            found = recorded_per_class.get(str(int(cls)))
            if np.isfinite(expected):
                if found is None or abs(float(found) - float(expected)) > 1e-9:
                    raise BankValidationError(
                        f"Row {position} {per_class_key}[{cls}]={found!r} disagrees with the "
                        f"recomputed value {float(expected)!r}"
                    )
            elif found is not None:
                raise BankValidationError(
                    f"Row {position} {per_class_key}[{cls}] must be null for an excluded class"
                )


def _check_local_evidence(position: int, evidence: Mapping[str, Any]) -> None:
    if list(evidence.get("channels", [])) != list(LOCAL_EVIDENCE_CHANNELS):
        raise BankValidationError(
            f"Row {position} local_evidence must declare channels {list(LOCAL_EVIDENCE_CHANNELS)}"
        )
    _require_nonempty_str(evidence.get("semantics"), f"Row {position} local_evidence.semantics")
    if _require_int(evidence.get("pixels"), f"Row {position} local_evidence.pixels") <= 0:
        raise BankValidationError(f"Row {position} local_evidence.pixels must be positive")
    for block in ("mean_probability", "argmax_pixel_fraction"):
        entry = evidence.get(block)
        if not isinstance(entry, Mapping) or set(entry) != set(LOCAL_EVIDENCE_CHANNELS):
            raise BankValidationError(
                f"Row {position} local_evidence.{block} must carry exactly the declared channels"
            )
        total = 0.0
        for channel in LOCAL_EVIDENCE_CHANNELS:
            value = _require_finite_float(entry[channel], f"Row {position} local_evidence.{block}[{channel}]")
            if value < -1e-9 or value > 1.0 + 1e-9:
                raise BankValidationError(
                    f"Row {position} local_evidence.{block}[{channel}]={value!r} is outside [0,1]"
                )
            total += value
        if abs(total - 1.0) > 1e-5:
            raise BankValidationError(
                f"Row {position} local_evidence.{block} sums to {total!r}, not 1; the local "
                "evidence summary must stay on the simplex."
            )


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: MetricContract,
    protocol: Mapping[str, Any],
) -> None:
    trajectories: dict[str, list[Mapping[str, Any]]] = {}
    declared_sources = {str(value) for value in protocol.get("sources", [])}
    protocol_tau = _require_finite_float(protocol.get("tau_accept"), "Bank protocol tau_accept")
    protocol_t_max = _require_int(protocol.get("t_max"), "Bank protocol t_max")
    for position, row in enumerate(rows):
        missing = [key for key in _ROW_REQUIRED_FIELDS if key not in row]
        missing += [key for key in _EVALUATED_REQUIRED_FIELDS if key not in row]
        if missing:
            raise BankValidationError(
                f"Row {position} is missing required field(s): {', '.join(sorted(set(missing)))}"
            )
        if _require_int(row["schema_version"], f"Row {position} schema_version") != BANK_SCHEMA_VERSION:
            raise BankValidationError(
                f"Row {position} declares schema_version={row['schema_version']!r}, "
                f"expected {BANK_SCHEMA_VERSION}"
            )
        _require_nonempty_str(row["patient_id"], f"Row {position} patient_id")
        _require_nonempty_str(row["case_id"], f"Row {position} case_id")
        if _require_int(row["slice_index"], f"Row {position} slice_index") < 0:
            raise BankValidationError(f"Row {position} slice_index must be non-negative")
        stage = _require_int(row["stage"], f"Row {position} stage")
        if stage < 0:
            raise BankValidationError(f"Row {position} stage must be non-negative")
        _require_nonempty_str(row["trajectory_id"], f"Row {position} trajectory_id")
        _require_bool(row["accepted"], f"Row {position} accepted")
        _require_bool(row["gt_used_in_generation"], f"Row {position} gt_used_in_generation")
        _require_bool(row["generation_valid"], f"Row {position} generation_valid")
        _require_nonempty_str(row["generation_operation"], f"Row {position} generation_operation")
        halted_after = _require_bool(row["halted_after"], f"Row {position} halted_after")
        expected_halted = bool(row["source"] == SOURCE_ON_POLICY and not row["accepted"])
        if halted_after != expected_halted:
            raise BankValidationError(
                f"Row {position} halted_after={halted_after!r} does not match on-policy rejection "
                f"(expected {expected_halted!r})"
            )
        if row["source"] not in TRANSITION_SOURCES:
            raise BankValidationError(
                f"Row {position} has unknown source {row['source']!r}; "
                f"expected one of {list(TRANSITION_SOURCES)}"
            )
        if row["rollout_policy"] not in ROLLOUT_POLICIES:
            raise BankValidationError(
                f"Row {position} has unknown rollout_policy {row['rollout_policy']!r}"
            )
        if _SOURCE_FOR_ROLLOUT[str(row["rollout_policy"])] != row["source"]:
            raise BankValidationError(
                f"Row {position} pairs rollout_policy {row['rollout_policy']!r} with source "
                f"{row['source']!r}; a rollout policy fixes its source tag."
            )
        if row["source"] == SOURCE_SYNTHETIC and not bool(row["gt_used_in_generation"]):
            raise BankValidationError(
                f"Row {position} is synthetic but claims gt_used_in_generation=False; "
                "a counterfactual edit is defined against ground truth."
            )
        if row["source"] != SOURCE_SYNTHETIC and bool(row["gt_used_in_generation"]):
            raise BankValidationError(
                f"Row {position} is {row['source']!r} but claims GT was used during generation; "
                "an on-policy rollout has no ground-truth input."
            )
        if row["state_representation"] not in STATE_REPRESENTATIONS:
            raise BankValidationError(
                f"Row {position} has unknown state_representation {row['state_representation']!r}"
            )
        prefix = f"{row['state_representation']}:sha256:"
        for field_name in ("previous_state_id", "candidate_state_id"):
            value = _require_nonempty_str(row[field_name], f"Row {position} {field_name}")
            if not value.startswith(prefix) or not _STATE_ID_DIGEST.match(value[len(prefix):]):
                raise BankValidationError(
                    f"Row {position} field {field_name}={value!r} is not a full SHA-256 state "
                    f"identifier for representation {row['state_representation']!r}"
                )
        if row["source"] not in declared_sources:
            raise BankValidationError(
                f"Row {position} has source {row['source']!r} which the bank protocol does not "
                f"declare (declared: {sorted(declared_sources)})"
            )
        if row["source"] != SOURCE_SYNTHETIC and stage >= protocol_t_max:
            raise BankValidationError(
                f"Row {position} is at stage {stage} but the protocol caps the rollout at "
                f"t_max={protocol_t_max}"
            )
        delta_q = _require_finite_float(row["delta_q"], f"Row {position} delta_q")
        gate = row["hypothetical_gate"]
        if not isinstance(gate, Mapping) or "tau_accept" not in gate or "accept" not in gate:
            raise BankValidationError(
                f"Row {position} hypothetical_gate must record tau_accept and accept"
            )
        row_tau = _require_finite_float(gate["tau_accept"], f"Row {position} hypothetical_gate.tau_accept")
        if abs(row_tau - protocol_tau) > 1e-12:
            raise BankValidationError(
                f"Row {position} was gated at tau={row_tau!r} but the bank protocol declares "
                f"tau_accept={protocol_tau!r}"
            )
        if _require_bool(gate["accept"], f"Row {position} hypothetical_gate.accept") != bool(
            delta_q > row_tau
        ):
            raise BankValidationError(
                f"Row {position} hypothetical_gate disagrees with delta_q > tau_accept"
            )
        if row["rollout_policy"] == ROLLOUT_SELF_AUDIT and bool(row["accepted"]) != bool(gate["accept"]):
            raise BankValidationError(
                f"Row {position} is a self-audit rollout whose recorded acceptance "
                "disagrees with its own threshold decision"
            )
        if row["source"] == SOURCE_ALWAYS_ACCEPT_PREFIX and not _require_bool(
            row["accepted"], f"Row {position} accepted"
        ):
            raise BankValidationError(
                f"Row {position} has source {SOURCE_ALWAYS_ACCEPT_PREFIX!r} but accepted is false; "
                "an always-accept rollout always accepts each proposal."
            )
        if str(row["metric_space"]) != contract.metric_space:
            raise BankValidationError(
                f"Row {position} metric_space {row['metric_space']!r} disagrees with the bank "
                f"contract metric space {contract.metric_space!r}"
            )
        _require_nonempty_str(row["empty_policy"], f"Row {position} empty_policy")
        if str(row["empty_policy"]) != contract.empty_policy:
            raise BankValidationError(
                f"Row {position} empty_policy {row['empty_policy']!r} disagrees with the bank "
                f"contract empty policy {contract.empty_policy!r}"
            )
        defined = _require_bool(row["delta_defined"], f"Row {position} delta_defined")
        if defined != (
            _require_bool(row["q_previous_defined"], f"Row {position} q_previous_defined")
            and _require_bool(row["q_candidate_defined"], f"Row {position} q_candidate_defined")
        ):
            raise BankValidationError(
                f"Row {position} delta_defined disagrees with the two state validity flags"
            )
        for name, flag in (
            ("q_previous", "q_previous_defined"),
            ("q_candidate", "q_candidate_defined"),
            ("delta_dice", "delta_defined"),
        ):
            value = row[name]
            if bool(row[flag]):
                if value is None:
                    raise BankValidationError(
                        f"Row {position} marks {flag} true but stores {name}=null"
                    )
            elif value is not None:
                raise BankValidationError(
                    f"Row {position} marks {flag} false but stores {name}={value!r}; an "
                    "undefined score must serialize as JSON null, never a sentinel."
                )
        if defined:
            expected = float(row["q_candidate"]) - float(row["q_previous"])
            if abs(expected - float(row["delta_dice"])) > 1e-6:
                raise BankValidationError(
                    f"Row {position} delta_dice does not equal q_candidate - q_previous"
                )
        margin = _require_finite_float(row["neutral_margin"], f"Row {position} neutral_margin")
        if abs(margin - float(contract.neutral_margin)) > 1e-12:
            raise BankValidationError(
                f"Row {position} neutral_margin={margin!r} disagrees with the bank contract "
                f"margin {float(contract.neutral_margin)!r}"
            )
        if defined:
            delta_class = _require_int(row["delta_class"], f"Row {position} delta_class")
            expected_class = int(classify_delta(float(row["delta_dice"]), margin))
            if delta_class != expected_class:
                raise BankValidationError(
                    f"Row {position} delta_class={delta_class!r} disagrees with the "
                    f"canonical classification {expected_class!r} at margin {margin!r}"
                )
        else:
            if row["delta_class"] is not None:
                raise BankValidationError(
                    f"Row {position} delta_class must be null when delta is undefined, got {row['delta_class']!r}"
                )
        _check_recomputed_quality(position, row, contract)
        evidence = row["local_evidence"]
        if not isinstance(evidence, Mapping):
            raise BankValidationError(f"Row {position} local_evidence must be a mapping")
        _check_local_evidence(position, evidence)
        trajectories.setdefault(str(row["trajectory_id"]), []).append(row)

    for trajectory_id, group in trajectories.items():
        ordered = sorted(group, key=lambda row: int(row["stage"]))
        identity = {
            (
                str(row["patient_id"]),
                str(row["case_id"]),
                int(row["slice_index"]),
                str(row["source"]),
                str(row["rollout_policy"]),
                str(row["state_representation"]),
            )
            for row in ordered
        }
        if len(identity) != 1:
            raise BankValidationError(
                f"Trajectory {trajectory_id!r} mixes {len(identity)} distinct sample identities; "
                "one trajectory belongs to exactly one patient/case/slice under one policy."
            )
        stages = [int(row["stage"]) for row in ordered]
        if stages != list(range(len(stages))):
            raise BankValidationError(
                f"Trajectory {trajectory_id!r} has stages {stages}; stages must start at 0 "
                "and be contiguous with no duplicates."
            )
        for previous_row, next_row in zip(ordered, ordered[1:]):
            if bool(previous_row["accepted"]):
                if next_row["previous_state_id"] != previous_row["candidate_state_id"]:
                    raise BankValidationError(
                        f"Trajectory {trajectory_id!r} breaks state continuity at stage "
                        f"{next_row['stage']}: an accepted candidate must be the next previous state."
                    )
            elif next_row["previous_state_id"] != previous_row["previous_state_id"]:
                raise BankValidationError(
                    f"Trajectory {trajectory_id!r} breaks state continuity at stage "
                    f"{next_row['stage']}: a rejected candidate must leave the state unchanged."
                )
            if str(previous_row["source"]) == SOURCE_ON_POLICY and not bool(previous_row["accepted"]):
                raise BankValidationError(
                    f"Trajectory {trajectory_id!r} continues past a rejected on-policy "
                    f"transition at stage {previous_row['stage']}: a rejected sample HALTs and "
                    "no later transition may be recorded for it."
                )


def _check_source_identity(
    block: Any,
    where: str,
    *,
    is_bound: bool = False,
    schema_version: int = BANK_SCHEMA_VERSION,
) -> None:
    """An unknown source signature must never be recorded as verified.

    The producer (W3.1) records ``source_content_signature`` alongside an
    explicit ``source_content_signature_known`` flag.  The bank does not
    compare revisions -- it is not a lineage consumer -- but it refuses to
    carry a record that claims a known signature while storing ``"unknown"``,
    or that stores a malformed digest under a known flag.

    For fresh schema banks (schema_version >= 1), source identity blocks must
    be non-empty mappings with required source content signatures; no
    generation_code={} is accepted as a bound export.  Historical checkpoints
    with an unknown producer (producer_git_sha == "unknown") are preserved.
    """

    if not isinstance(block, Mapping) or not block:
        if is_bound or schema_version >= 1:
            raise BankValidationError(f"{where} must be a non-empty mapping")
        return
    if "source_content_signature" not in block:
        if is_bound or schema_version >= 1:
            raise BankValidationError(f"{where} is missing required source_content_signature")
        return
    known = _require_bool(
        block.get("source_content_signature_known"),
        f"{where}.source_content_signature_known",
    )
    signature = str(block.get("source_content_signature"))
    if known:
        if not _STATE_ID_DIGEST.match(signature):
            raise BankValidationError(
                f"{where}.source_content_signature is marked known but is not a full "
                f"SHA-256 digest ({signature!r})"
            )
    elif signature != UNKNOWN:
        raise BankValidationError(
            f"{where}.source_content_signature_known is false but a signature "
            f"({signature!r}) is stored; an unverified signature must stay {UNKNOWN!r}."
        )


def validate_bank(bank: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema, provenance, contract, integrity and state continuity."""

    if not isinstance(bank, Mapping):
        raise BankValidationError(f"bank must be a mapping, got {type(bank).__name__}")
    for key in ("bank_schema_version", "generation", "evaluation", "protocol", "rows", "integrity"):
        if key not in bank:
            raise BankValidationError(f"Bank is missing required top-level key {key!r}")
    version = bank["bank_schema_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != BANK_SCHEMA_VERSION:
        raise BankValidationError(
            f"Unsupported bank_schema_version {version!r}; only {BANK_SCHEMA_VERSION} is supported"
        )

    generation = bank["generation"]
    if not isinstance(generation, Mapping):
        raise BankValidationError("Bank generation provenance must be a mapping")
    for key in (
        "generation_code",
        "rollout_policy",
        "checkpoint_bound",
        "split_coverage",
        "export_identity_class",
        "cohort",
    ):
        if key not in generation:
            raise BankValidationError(f"Bank generation provenance is missing {key!r}")
    if generation["split_coverage"] != MEMBERSHIP_COVERAGE:
        raise BankValidationError(
            f"Bank generation split_coverage must stay {MEMBERSHIP_COVERAGE!r}: the split "
            "signature hashes membership only and must not be reported as content coverage."
        )
    identity_class = str(generation["export_identity_class"])
    if identity_class not in EXPORT_IDENTITY_CLASSES:
        raise BankValidationError(
            f"Unknown export_identity_class {identity_class!r}; expected one of "
            f"{list(EXPORT_IDENTITY_CLASSES)}"
        )
    bound = _require_bool(generation["checkpoint_bound"], "Bank generation checkpoint_bound")
    _check_source_identity(
        generation.get("generation_code"),
        "Bank generation.generation_code",
        is_bound=bound,
        schema_version=version,
    )
    if bound != (identity_class == EXPORT_BOUND):
        raise BankValidationError(
            f"Bank generation checkpoint_bound={bound} contradicts export_identity_class "
            f"{identity_class!r}"
        )
    if bound:
        checkpoint = generation.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise BankValidationError(
                "A bound export must carry its complete checkpoint identity, not a null checkpoint"
            )
        missing = [key for key in _BOUND_CHECKPOINT_FIELDS if not checkpoint.get(key)]
        if missing:
            raise BankValidationError(
                f"Bound export checkpoint identity is incomplete; missing or empty: "
                f"{', '.join(sorted(missing))}"
            )
        for key in ("state_digest", "file_state_digest", "checkpoint_sha256"):
            value = str(checkpoint[key])
            if not _STATE_ID_DIGEST.match(value):
                raise BankValidationError(
                    f"Bound export checkpoint {key}={value!r} is not a full SHA-256 digest"
                )
        if str(checkpoint["state_digest"]) != str(checkpoint["file_state_digest"]):
            raise BankValidationError(
                f"Bound checkpoint state_digest ({checkpoint['state_digest']!r}) does not "
                f"match file_state_digest ({checkpoint['file_state_digest']!r})"
            )
        live_model_identity = generation.get("model_identity")
        if not isinstance(live_model_identity, Mapping):
            raise BankValidationError("A bound export must record the live model identity")
        binding_model_identity = checkpoint.get("model_identity")
        if not isinstance(binding_model_identity, Mapping):
            raise BankValidationError("A bound export checkpoint must record model_identity")
        if dict(live_model_identity) != dict(binding_model_identity):
            raise BankValidationError(
                "Live model_identity does not match checkpoint binding model_identity"
            )
        cohort = generation.get("cohort")
        if not isinstance(cohort, Mapping) or not cohort:
            raise BankValidationError(
                "A bound export must record the exact cohort it ran over; an empty or null cohort "
                "is not an identified measurement."
            )
        cohort_split = _require_nonempty_str(
            cohort.get("split_name"), "Bank generation.cohort.split_name"
        )
        if cohort_split != str(generation.get("split_name")):
            raise BankValidationError(
                f"Bank generation.cohort.split_name ({cohort_split!r}) disagrees with "
                f"generation.split_name ({generation.get('split_name')!r})"
            )
        membership_sig = _require_nonempty_str(
            cohort.get("membership_signature"), "Bank generation.cohort.membership_signature"
        )
        if not _STATE_ID_DIGEST.match(membership_sig):
            raise BankValidationError(
                f"Bank generation.cohort.membership_signature is not a full SHA-256 digest ({membership_sig!r})"
            )
        if cohort.get("coverage") != MEMBERSHIP_COVERAGE:
            raise BankValidationError(
                f"Bank generation.cohort.coverage must be {MEMBERSHIP_COVERAGE!r}, got {cohort.get('coverage')!r}"
            )
        num_records = _require_int(cohort.get("num_records"), "Bank generation.cohort.num_records")
        if num_records <= 0:
            raise BankValidationError("Bank generation.cohort.num_records must be positive")
    else:
        _require_nonempty_str(
            generation.get("unverified_demo_reason"),
            "Bank generation unverified_demo_reason",
        )

    evaluation = bank["evaluation"]
    if not isinstance(evaluation, Mapping):
        raise BankValidationError("Bank evaluation provenance must be a mapping")
    for key in (
        "evaluation_code",
        "metric_contract",
        "metric_contract_definition",
        "metric_contract_version",
        "metric_space",
        "empty_policy",
        "neutral_margin",
    ):
        if key not in evaluation:
            raise BankValidationError(f"Bank evaluation provenance is missing {key!r}")
    contract = validate_contract(evaluation["metric_contract_definition"])
    _check_source_identity(
        evaluation.get("evaluation_code"),
        "Bank evaluation.evaluation_code",
        is_bound=bound,
        schema_version=version,
    )
    if contract.name != evaluation["metric_contract"]:
        raise BankValidationError(
            f"Bank declares metric_contract {evaluation['metric_contract']!r} but its "
            f"definition is {contract.name!r}"
        )
    eval_version = _require_int(
        evaluation["metric_contract_version"], "Bank evaluation metric_contract_version"
    )
    if eval_version != contract.version:
        raise BankValidationError(
            f"Bank evaluation metric_contract_version ({eval_version}) disagrees with "
            f"contract version ({contract.version})"
        )
    if str(evaluation["metric_space"]) != contract.metric_space:
        raise BankValidationError(
            f"Bank evaluation metric_space ({evaluation['metric_space']!r}) disagrees with "
            f"contract metric_space ({contract.metric_space!r})"
        )
    if str(evaluation["empty_policy"]) != contract.empty_policy:
        raise BankValidationError(
            f"Bank evaluation empty_policy ({evaluation['empty_policy']!r}) disagrees with "
            f"contract empty_policy ({contract.empty_policy!r})"
        )
    eval_margin = _require_finite_float(
        evaluation["neutral_margin"], "Bank evaluation neutral_margin"
    )
    if abs(eval_margin - float(contract.neutral_margin)) > 1e-12:
        raise BankValidationError(
            f"Bank evaluation neutral_margin ({eval_margin!r}) disagrees with "
            f"contract margin ({float(contract.neutral_margin)!r})"
        )

    protocol = bank["protocol"]
    if not isinstance(protocol, Mapping):
        raise BankValidationError("Bank protocol must be a mapping")
    for key in ("tau_accept", "t_max", "sources"):
        if key not in protocol:
            raise BankValidationError(f"Bank protocol is missing {key!r}")
    sources = protocol["sources"]
    if not isinstance(sources, list) or not sources:
        raise BankValidationError("Bank protocol sources must be a non-empty list")
    unknown = [value for value in sources if value not in TRANSITION_SOURCES]
    if unknown:
        raise BankValidationError(
            f"Bank protocol declares unknown source tag(s) {unknown}; expected values from "
            f"{list(TRANSITION_SOURCES)}"
        )
    if _require_int(protocol["t_max"], "Bank protocol t_max") < 0:
        raise BankValidationError("Bank protocol t_max must be non-negative")

    rows = bank["rows"]
    if not isinstance(rows, list):
        raise BankValidationError("Bank rows must be a list")
    for position, row in enumerate(rows):
        if str(row.get("metric_contract")) != contract.name:
            raise BankValidationError(
                f"Row {position} metric_contract {row.get('metric_contract')!r} disagrees with the bank "
                f"contract {contract.name!r}"
            )
        row_version = _require_int(
            row.get("metric_contract_version"), f"Row {position} metric_contract_version"
        )
        if row_version != contract.version:
            raise BankValidationError(
                f"Row {position} metric_contract_version ({row_version}) disagrees with the bank "
                f"contract ({contract.version})"
            )
        if "empty_policy" not in row:
            raise BankValidationError(f"Row {position} is missing required field 'empty_policy'")
        if str(row.get("empty_policy")) != contract.empty_policy:
            raise BankValidationError(
                f"Row {position} empty_policy {row.get('empty_policy')!r} disagrees with the bank "
                f"contract empty policy {contract.empty_policy!r}"
            )
    _validate_rows(rows, contract=contract, protocol=protocol)

    integrity = bank["integrity"]
    if not isinstance(integrity, Mapping) or "content_signature" not in integrity:
        raise BankValidationError("Bank integrity block must carry content_signature")
    num_rows = _require_int(integrity.get("num_rows"), "Bank integrity num_rows")
    if num_rows != len(rows):
        raise BankValidationError(
            f"Bank integrity num_rows={num_rows!r} disagrees with {len(rows)} rows"
        )
    expected = bank_content_signature(bank)
    if str(integrity["content_signature"]) != expected:
        raise BankValidationError(
            "Bank content signature mismatch: the header or rows were modified after export "
            f"(expected {expected}, found {integrity['content_signature']})"
        )
    return dict(bank)


def dump_bank(bank: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and write the bank as strict JSON (``allow_nan=False``)."""

    validate_bank(bank)
    target = Path(path)
    return atomic_write_json(target, dict(bank), indent=2, sort_keys=True)


def load_bank(path: str | Path) -> dict[str, Any]:
    """Read a bank back and re-validate every boundary check."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_json_constant)
    return validate_bank(payload)


def _reject_json_constant(name: str) -> Any:
    raise BankValidationError(
        f"Bank JSON contains the non-standard constant {name!r}; undefined scores must be null."
    )


def bank_summary(bank: Mapping[str, Any]) -> dict[str, Any]:
    """Small counting summary for report text.  Purely descriptive."""

    rows = list(bank.get("rows", []))
    by_source: dict[str, int] = {}
    for row in rows:
        by_source[str(row.get("source"))] = by_source.get(str(row.get("source")), 0) + 1
    return {
        "num_rows": len(rows),
        "rows_by_source": by_source,
        "accepted": sum(1 for row in rows if bool(row.get("accepted"))),
        "halted_after": sum(1 for row in rows if bool(row.get("halted_after"))),
        "undefined_delta": sum(1 for row in rows if not bool(row.get("delta_defined"))),
        "trajectories": len({str(row.get("trajectory_id")) for row in rows}),
    }


__all__ = [
    "BANK_CONTENT_COVERAGE",
    "BANK_SCHEMA_VERSION",
    "BankValidationError",
    "DEFAULT_NEUTRAL_MARGIN",
    "EXPORT_BOUND",
    "EXPORT_IDENTITY_CLASSES",
    "EXPORT_UNVERIFIED_DEMO",
    "LOCAL_EVIDENCE_CHANNELS",
    "LOCAL_EVIDENCE_SEMANTICS",
    "ROLLOUT_ALWAYS_ACCEPT",
    "ROLLOUT_POLICIES",
    "ROLLOUT_SELF_AUDIT",
    "ROLLOUT_SYNTHETIC",
    "SOURCE_ALWAYS_ACCEPT_PREFIX",
    "SOURCE_ON_POLICY",
    "SOURCE_SYNTHETIC",
    "STATE_REPRESENTATIONS",
    "TRANSITION_SOURCES",
    "TransitionEvaluator",
    "TransitionProposal",
    "bank_content_signature",
    "bank_summary",
    "build_bank",
    "dump_bank",
    "evaluation_provenance_record",
    "generate_on_policy_proposals",
    "generate_synthetic_proposals",
    "generation_provenance_record",
    "load_bank",
    "state_identity",
    "state_tensor_id",
    "validate_bank",
]
