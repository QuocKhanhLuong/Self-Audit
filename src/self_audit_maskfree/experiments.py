"""Paired-bank selector experiments and negative controls (W6).

What this module is for
-----------------------
The scientific question is not "does our pipeline produce a mask" but "does
*predictive evidence on withheld observations* choose better drafts than
confidence, consistency, or fitting-side score do?". That question is only
answerable if every selector is offered **the same candidate bank**, so that a
difference between selectors cannot be explained by one of them having searched
harder or seen more candidates.

:func:`run_bank_experiments` therefore runs E0-E5 and six negative controls
against one already-generated bank for one unit, and separately compares the
active challenger's candidates against an *equal-number, equal-fitting-cost* set
of random edits. Without that second comparison, any benefit of active challenge
could simply be a larger search budget.

Honesty rules enforced in code, not only in prose
-------------------------------------------------
* E4 and E5 may select the exact same candidate on a common bank. That is the
  expected outcome when their only difference is ambiguity reporting, and it is
  reported as ``tie_with_e4`` - never converted into an invented gain.
* ``cuts_inspired_control`` is an in-repo simplified feature-grouping control.
  Every row carries ``external_cuts_reproduction='pending'``; nothing here is a
  reproduction of the published CUTS result.
* Negative controls are diagnostics. They are returned in their own section and
  are never folded into the selector comparison or fed back as training labels.
* Shuffled-evidence control: when no incompatible study has been cached yet, the
  control is ``available=False`` with a reason and the current unit's scores are
  pushed into the cache for a later unit. No foreign evidence is fabricated.
* No manual mask, no reference, no ground truth is read anywhere in this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .contracts import (
    VERSION,
    AuditResult,
    EvidenceScore,
    FittedHypothesis,
    FittingView,
    Hypothesis,
    ScoringView,
    hypothesis_from_labels,
)
from .evaluation.metrics import COVERAGE_LEVELS, metric_row
from .evaluation.partitions import (
    all_background_hypothesis,
    class_permutation_hypothesis,
    excessive_partition_hypothesis,
    intensity_grouping_hypothesis,
    random_mask_hypothesis,
)
from .evaluation.verification import verify_frozen_bank  # re-exported for W5

NUM_CLASSES = 4
#: Selector experiment identifiers in the order the suite reports them.
SELECTOR_IDS = (
    "E0_intensity_grouping",
    "E1_cuts_inspired_control",
    "E2_confidence_consistency",
    "E3_fitting_score",
    "E4_selection_evidence",
    "E5_evidence_plus_challenge",
)
#: Negative-control identifiers. Diagnostics only.
CONTROL_IDS = (
    "control_random_same_budget",
    "control_shuffled_evidence",
    "control_all_background",
    "control_random_spatial_masks",
    "control_class_permutation",
    "control_excessive_partition",
)


# ---------------------------------------------------------------------------
# foreign evidence cache for the shuffled-evidence control
# ---------------------------------------------------------------------------
@dataclass
class ForeignEvidenceCache:
    """Scores from *other* studies, kept so the shuffled-evidence control is real.

    The control asks: if a selector is handed evidence computed on a different,
    incompatible study, does it still "work"? A selector that still produces a
    confident choice is responding to something other than the evidence.

    The cache is filled as units stream past. Before any other study has been
    seen there is nothing to shuffle in, and the control reports itself
    unavailable rather than inventing numbers.
    """

    capacity: int = 32
    entries: list[dict[str, Any]] = field(default_factory=list)
    # A cache is evidence from one dataset/run only.  The optional scope keeps
    # the small public fixture API backwards compatible while preventing a
    # process-level cache from silently crossing independent experiments.
    dataset: str | None = None
    run_id: str | None = None

    def push(
        self,
        study_id: str,
        unit_id: str,
        scores: Mapping[str, float],
        *,
        dataset: str | None = None,
        run_id: str | None = None,
        candidate_slots: Mapping[str, int] | None = None,
    ) -> None:
        """Queue one unit's finite evidence for a later *different* study.

        Candidate ids are usually unit-specific, so the shuffled control cannot
        join studies by id.  ``candidate_slots`` is the preregistered bank
        position (incumbent, boundary edit, split/merge, semantic alternative)
        and is the only correspondence used by :func:`run_bank_experiments`.
        When a caller omits slots, deterministic sorted-key slots are recorded
        as a conservative compatibility path for old fixtures.
        """
        usable = {
            candidate_id: float(value)
            for candidate_id, value in scores.items()
            if value is not None and math.isfinite(float(value))
        }
        if not usable:
            return
        resolved_dataset = dataset if dataset is not None else self.dataset
        resolved_run = run_id if run_id is not None else self.run_id
        slot_scores: dict[str, float] = {}
        slot_by_candidate: dict[str, int] = {}
        if candidate_slots:
            for candidate_id, value in usable.items():
                raw_slot = candidate_slots.get(candidate_id)
                if raw_slot is None:
                    continue
                slot = int(raw_slot)
                slot_scores[str(slot)] = value
                slot_by_candidate[str(candidate_id)] = slot
        if not slot_scores:
            # The bank order is the preregistered correspondence.  Sorted ids
            # are only a fallback for legacy callers that did not expose it.
            for slot, candidate_id in enumerate(sorted(usable)):
                slot_scores[str(slot)] = usable[candidate_id]
                slot_by_candidate[str(candidate_id)] = slot
        self.entries.append(
            {
                "study_id": study_id,
                "unit_id": unit_id,
                "dataset": resolved_dataset,
                "run_id": resolved_run,
                "scores": usable,
                "slot_scores": slot_scores,
                "candidate_slots": slot_by_candidate,
            }
        )
        if len(self.entries) > self.capacity:
            del self.entries[0 : len(self.entries) - self.capacity]

    def foreign(
        self,
        study_id: str,
        *,
        dataset: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Most recent cached entry from a *different* study, or None."""
        resolved_dataset = dataset if dataset is not None else self.dataset
        resolved_run = run_id if run_id is not None else self.run_id
        for entry in reversed(self.entries):
            if entry["study_id"] == study_id:
                continue
            if resolved_dataset is not None and entry.get("dataset") != resolved_dataset:
                continue
            if resolved_run is not None and entry.get("run_id") != resolved_run:
                continue
            return entry
        return None


#: Process-level default cache. A caller that wants isolation (tests, or a second
#: dataset in the same process) passes its own instance.
DEFAULT_FOREIGN_CACHE = ForeignEvidenceCache()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _score_entry(score: EvidenceScore | None) -> dict[str, Any] | None:
    if score is None:
        return None
    return {
        "nll_sum": score.nll_sum,
        "count": score.count,
        "normalized_nll": score.normalized_nll,
        "complexity": score.complexity,
        "prior": score.prior,
        "total": score.total,
        "role": score.role,
        "available": score.available,
        "reason": score.reason,
    }


def _total(score: EvidenceScore | None) -> float | None:
    if score is None or not score.available or score.total is None:
        return None
    return float(score.total)


def confidence_consistency(hypothesis: Hypothesis) -> dict[str, Any]:
    """Fit-only confidence and consistency metadata for the E2 selector.

    ``confidence`` is the mean maximum class probability of the draft;
    ``consistency`` is the fraction of 4-neighbour pixel pairs that agree.
    Neither looks at any observation the draft was not built from, which is
    exactly the point of E2: it is the "trust the network's own certainty"
    policy that predictive evidence has to beat.
    """
    probabilities = hypothesis.probabilities.detach().to(torch.float32)
    confidence_map = probabilities.max(dim=0).values
    confidence = float(confidence_map.mean().item())
    # ``hypothesis_from_labels`` necessarily produces one-hot probabilities.
    # In that common path confidence is identically one, so E2 is a smoothness
    # (consistency) control.  Keep the value for traceability, but disclose the
    # constant criterion instead of presenting it as an independent confidence
    # signal.
    confidence_constant = bool(torch.allclose(confidence_map, torch.ones_like(confidence_map)))
    labels = hypothesis.labels.detach()
    horizontal = labels[:, 1:] == labels[:, :-1]
    vertical = labels[1:, :] == labels[:-1, :]
    total = horizontal.numel() + vertical.numel()
    consistency = (
        float(horizontal.sum().item() + vertical.sum().item()) / float(total) if total else 1.0
    )
    return {
        "confidence": confidence,
        "consistency": consistency,
        "combined": 0.5 * (confidence + consistency),
        "confidence_constant": confidence_constant,
        "selection_basis": "smoothness_only" if confidence_constant else "confidence_and_smoothness",
        "confidence_note": (
            "one-hot draft confidence is constant at 1.0; this E2 choice is driven by "
            "spatial consistency only"
            if confidence_constant
            else "confidence is derived from the fit-side draft probabilities"
        ),
    }


def _cache_scope(view: FittingView) -> tuple[str | None, str | None]:
    """Return the dataset/run identity used to isolate foreign evidence."""
    metadata = view.metadata if isinstance(view.metadata, Mapping) else {}
    dataset = metadata.get("dataset")
    run_id = metadata.get("run_id") or metadata.get("experiment_id") or metadata.get("manifest_id")
    return (
        None if dataset is None else str(dataset),
        None if run_id is None else str(run_id),
    )


def _candidate_slots(hypotheses: Sequence[Hypothesis]) -> dict[str, int]:
    """Resolve the frozen, positional bank slots used for cross-study shuffles."""
    slots: dict[str, int] = {}
    used: set[int] = set()
    for position, hypothesis in enumerate(hypotheses):
        raw = hypothesis.metadata.get("candidate_slot", hypothesis.metadata.get("slot", position))
        try:
            slot = int(raw)
        except (TypeError, ValueError):
            slot = position
        if slot in used:
            # Duplicate metadata is not allowed to collapse two candidates into
            # one foreign-evidence slot; fall back to the stable bank position.
            slot = position
        while slot in used:
            slot += 1
        used.add(slot)
        slots[hypothesis.candidate_id] = slot
    return slots


def random_edit(
    source: Hypothesis,
    *,
    seed: int,
    index: int,
    max_fraction: float = 0.15,
    candidate_id: str | None = None,
) -> Hypothesis:
    """One bounded random edit of a draft: relabel a random rectangle.

    Deliberately cheap and content-blind. These edits exist only to give the
    active challenger an equal-number, equal-cost opponent: if a random edit set
    of the same size and the same fitting budget matches the challenger's gain,
    then the gain came from search budget, not from the challenge policy.
    """
    generator = torch.Generator().manual_seed(seed * 1000 + index)
    labels = source.labels.detach().clone()
    height, width = labels.shape
    side_h = max(1, int(round(height * math.sqrt(max_fraction))))
    side_w = max(1, int(round(width * math.sqrt(max_fraction))))
    box_h = int(torch.randint(1, side_h + 1, (1,), generator=generator).item())
    box_w = int(torch.randint(1, side_w + 1, (1,), generator=generator).item())
    top = int(torch.randint(0, max(1, height - box_h + 1), (1,), generator=generator).item())
    left = int(torch.randint(0, max(1, width - box_w + 1), (1,), generator=generator).item())
    new_class = int(torch.randint(0, NUM_CLASSES, (1,), generator=generator).item())
    labels[top : top + box_h, left : left + box_w] = new_class
    name = candidate_id or f"random_edit_{index}"
    hypothesis = hypothesis_from_labels(labels, name, "random_edit_control")
    hypothesis.validity = source.validity.detach().clone()
    hypothesis.semantic_unresolved = bool(source.semantic_unresolved)
    hypothesis.metadata = dict(source.metadata)
    hypothesis.metadata.update(
        {
            "edit": "random_rectangle",
            "edited_from": source.candidate_id,
            "box": [top, left, box_h, box_w],
            "new_class": new_class,
            "note": "edit type does not determine whether an edit is beneficial",
        }
    )
    return hypothesis


def _resolve_roles(draft: Hypothesis, fitting_view: FittingView) -> Hypothesis:
    """Name E0's anonymous intensity groups with W2's image-only ontology.

    W6 never invents anatomical role rules and never matches groups to a
    reference: the group ids handed to ``ontology.resolve_roles`` are anonymous,
    and the resolver alone decides which of them is RV, MYO or LV. E0 must use
    the *same* ontology as the audited arm, otherwise the baseline would be
    handicapped by a naming difference rather than by its partition.

    If the resolver refuses this partition (for example because orientation
    metadata is missing), the draft is returned with ``semantic_unresolved``
    set and the reason recorded. That is the honest state: intensity bins are
    not anatomy.
    """
    def mark_failure(reason: str) -> Hypothesis:
        draft.semantic_unresolved = True
        # Keep the draft labels available for diagnostics, but never let an
        # ontology failure masquerade as an all-valid training target.
        draft.validity = torch.zeros_like(draft.validity, dtype=torch.float32)
        draft.metadata["ontology_status"] = "fail"
        draft.metadata["ontology_reason"] = reason
        return draft

    try:
        from .ontology import resolve_roles  # type: ignore
    except Exception as error:  # ontology is W2-owned and may not exist yet
        return mark_failure(
            f"ontology.resolve_roles unavailable ({type(error).__name__}); "
            "intensity groups are not anatomical roles"
        )
    try:
        resolved = resolve_roles(
            draft.labels, fitting_view, candidate_id=draft.candidate_id, source=draft.source
        )
    except Exception as error:
        return mark_failure(
            f"ontology.resolve_roles refused this partition: {type(error).__name__}: {error}"
        )
    for key, value in draft.metadata.items():
        resolved.metadata.setdefault(key, value)
    resolved.metadata["ontology_reason"] = "resolved by ontology.resolve_roles"
    resolved.metadata.setdefault("ontology_status", "resolved")
    if resolved.semantic_unresolved:
        # A resolver is allowed to return alternatives, but unresolved anatomy
        # cannot retain all-valid support.  The conservative v1 policy gates
        # the complete draft until the roles are resolved explicitly.
        resolved.validity = torch.zeros_like(resolved.validity, dtype=torch.float32)
        resolved.metadata["ontology_status"] = "unresolved"
    return resolved


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------
def _stable_order(validity: torch.Tensor) -> torch.Tensor:
    """Descending validity, ties broken by flat index. Deterministic and GT-free."""
    flat = validity.detach().reshape(-1).to(torch.float64)
    index = torch.arange(flat.numel(), dtype=torch.float64)
    # Sorting by (-validity, index): scale index into the tie-break position.
    keys = torch.stack([-flat, index], dim=1)
    order = sorted(range(flat.numel()), key=lambda i: (float(keys[i, 0]), float(keys[i, 1])))
    return torch.tensor(order, dtype=torch.long)


def matched_coverage(
    validity_a: torch.Tensor, validity_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two boolean masks with an identical number of supported pixels.

    A selective method can always look better by keeping fewer pixels. Matched
    coverage removes that degree of freedom: both arms are truncated to the
    smaller of their supported counts, each keeping its own highest-validity
    pixels under a stable, deterministic ordering. No reference is consulted, so
    this can run inside the mask-free pipeline.
    """
    if validity_a.shape != validity_b.shape:
        raise ValueError("matched_coverage requires two validity maps on the same grid")
    supported_a = int((validity_a.detach() > 0).sum().item())
    supported_b = int((validity_b.detach() > 0).sum().item())
    budget = min(supported_a, supported_b)
    masks: list[torch.Tensor] = []
    for validity in (validity_a, validity_b):
        mask = torch.zeros(validity.shape, dtype=torch.bool)
        if budget > 0:
            order = _stable_order(validity)[:budget]
            flat = mask.reshape(-1)
            flat[order] = True
            mask = flat.reshape(validity.shape)
        masks.append(mask)
    return masks[0], masks[1]


def coverage_masks(
    validity: torch.Tensor, levels: Sequence[float] = COVERAGE_LEVELS
) -> dict[float, torch.Tensor]:
    """Boolean masks retaining the top ``level`` fraction of pixels by validity.

    The preregistered levels are 25/50/75/100 percent. Reporting an accuracy at
    one of these levels without its coverage is forbidden by the metric
    contract; keeping the level in the key makes that hard to do by accident.
    """
    total = validity.numel()
    order = _stable_order(validity)
    masks: dict[float, torch.Tensor] = {}
    for level in levels:
        if not (0.0 < float(level) <= 1.0):
            raise ValueError("coverage levels must lie in (0,1]")
        keep = max(1, min(total, int(round(float(level) * total)))) if total else 0
        mask = torch.zeros(validity.shape, dtype=torch.bool)
        flat = mask.reshape(-1)
        flat[order[:keep]] = True
        masks[float(level)] = flat.reshape(validity.shape)
    return masks


# ---------------------------------------------------------------------------
# bank bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class _BankIndex:
    """Bank candidates aligned with their fits and their selection-role scores.

    Two distinct notions of "challenge" live here and must not be conflated:

    ``challenge_ids``
        the candidates the *challenger* proposed, i.e. everything the bank holds
        beyond the incumbent. These are what the equal-number, equal-cost
        random-edit set has to be matched against.
    ``e4_pool_ids``
        the pool E4 is allowed to choose from. On a fixed shared bank this is
        the whole bank, which is why E4 and E5 may tie exactly; it narrows only
        when an audit trace explicitly marks which candidates arrived in a later
        challenge round.
    """

    hypotheses: list[Hypothesis]
    fitted: dict[str, FittedHypothesis]
    selection: dict[str, EvidenceScore]
    challenge_ids: list[str]
    e4_pool_ids: list[str]
    challenge_source: str
    pool_source: str


def _index_bank(audit_result: AuditResult) -> _BankIndex:
    bank = list(audit_result.bank)
    fitted_map: dict[str, FittedHypothesis] = {}
    for fitted in audit_result.fitted:
        fitted_map[fitted.hypothesis.candidate_id] = fitted

    selection_map: dict[str, EvidenceScore] = {}
    scores = list(audit_result.scores)
    if len(scores) == len(audit_result.fitted):
        for fitted, score in zip(audit_result.fitted, scores):
            selection_map[fitted.hypothesis.candidate_id] = score
    elif len(scores) == len(bank):
        for hypothesis, score in zip(bank, scores):
            selection_map[hypothesis.candidate_id] = score

    trace = audit_result.trace if isinstance(audit_result.trace, dict) else {}
    declared = trace.get("challenge_candidate_ids") or trace.get("challenger_candidate_ids")
    incumbent_id = audit_result.initial.candidate_id
    if isinstance(declared, (list, tuple)):
        challenge_ids = [str(value) for value in declared]
        challenge_source = "declared by the audit trace"
        e4_pool_ids = [h.candidate_id for h in bank if h.candidate_id not in set(challenge_ids)]
        pool_source = "pre-challenge candidates named by the audit trace"
    else:
        # The W2 bank is generated once: every non-incumbent member is a
        # challenger proposal (bounded boundary edit, split/merge, semantic
        # alternative). That set - not a later "round" - is what the random-edit
        # control has to match in number and in fitting cost.
        challenge_ids = [h.candidate_id for h in bank if h.candidate_id != incumbent_id]
        challenge_source = "non-incumbent bank members (the challenger's proposals)"
        e4_pool_ids = [h.candidate_id for h in bank]
        pool_source = (
            "the whole shared bank; the audit trace marks no separate challenge round, so E4 "
            "and E5 choose from an identical pool and may tie exactly"
        )
    return _BankIndex(
        bank, fitted_map, selection_map, challenge_ids, e4_pool_ids, challenge_source, pool_source
    )


def _fit_and_score(
    model: Any,
    hypothesis: Hypothesis,
    fitting_view: FittingView,
    selection_view: ScoringView,
    budget: dict[str, int],
) -> tuple[FittedHypothesis, EvidenceScore]:
    """One matched-budget fit plus one selection-role score, with budget accounting."""
    fitted = model.fit(hypothesis, fitting_view)
    budget["fits"] += 1
    budget["fitting_steps"] += int(fitted.fitting_steps)
    score = model.score(fitted, selection_view)
    budget["scores"] += 1
    # Preserve the score alongside the pre-freeze fit.  Verification can use
    # this provenance without ever recomputing or refitting on a sealed view.
    fitted.metadata.setdefault("selection_score", score.total)
    return fitted, score


def _prediction_row(
    name: str,
    family: str,
    candidate: Hypothesis | None,
    *,
    selection: EvidenceScore | None,
    fit: EvidenceScore | None,
    provenance: Mapping[str, Any],
    available: bool = True,
    reason: str | None = None,
    value_override: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """One serializable experiment row with full provenance and honest availability."""
    total = _total(selection) if value_override is None else float(value_override)
    row = metric_row(
        f"{name}.selection_total",
        total if (available and total is not None) else None,
        unit="nats/pixel + beta*complexity + prior",
        dataset=str(provenance.get("dataset", "unknown")),
        split=str(provenance.get("split", "train")),
        protocol=str(provenance.get("protocol", "unknown")),
        epoch=provenance.get("epoch"),
        checkpoint=str(provenance.get("checkpoint", "unfrozen")),
        population="observation unit",
        count=int(selection.count) if selection is not None else 0,
        available=bool(available and total is not None),
        reason=(
            reason
            if reason is not None
            else (None if (available and total is not None) else "no available selection score")
        ),
        experiment=name,
        family=family,
        candidate_id=None if candidate is None else candidate.candidate_id,
        candidate_source=None if candidate is None else candidate.source,
        semantic_unresolved=None if candidate is None else bool(candidate.semantic_unresolved),
        fit_total=_total(fit),
        selection_score=_score_entry(selection),
        fit_score=_score_entry(fit),
        external_cuts_reproduction="pending",
        note="predictive NLL on withheld selection observations; not accuracy",
        **extra,
    )
    return row


# ---------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------
def run_bank_experiments(
    audit_result: AuditResult,
    fitting_view: FittingView,
    selection_view: ScoringView,
    model: Any,
    *,
    seed: int = 42,
    foreign_cache: ForeignEvidenceCache | None = None,
) -> dict[str, Any]:
    """Run E0-E5, six negative controls and the paired search comparison on one bank.

    Parameters
    ----------
    audit_result:
        The already-audited bank for this unit. Its candidates, fits and
        selection scores are reused so that every selector sees an identical
        bank; only E0 and the controls introduce partitions of their own, and
        those are fitted at the same budget.
    fitting_view / selection_view:
        Fitting observations and the adaptive selection observations. The
        verification role is not accepted here and is not reachable from this
        function.
    model:
        The observation model. ``fit`` is called only with ``fitting_view``.
    foreign_cache:
        Cache supplying evidence from an incompatible study for the shuffled
        evidence control. Defaults to the process-level cache.

    Returns a dict with ``rows`` (serializable metric rows), ``predictions``
    (name to :class:`Hypothesis`, for the exporter), ``selectors``,
    ``controls``, ``search_comparison``, ``coverage`` and ``budget``.
    """
    if selection_view.role != "select":
        raise ValueError(
            f"bank experiments run on the adaptive selection role; got {selection_view.role!r}"
        )
    cache = DEFAULT_FOREIGN_CACHE if foreign_cache is None else foreign_cache
    index = _index_bank(audit_result)
    dataset_scope, run_scope = _cache_scope(fitting_view)
    slot_by_candidate = _candidate_slots(index.hypotheses)
    candidate_by_slot = {slot: candidate_id for candidate_id, slot in slot_by_candidate.items()}
    budget = {"fits": 0, "scores": 0, "fitting_steps": 0}
    provenance = {
        "dataset": fitting_view.metadata.get("dataset", "unknown"),
        "split": fitting_view.metadata.get("split", "train"),
        "protocol": fitting_view.protocol,
        "epoch": fitting_view.metadata.get("epoch"),
        "checkpoint": fitting_view.metadata.get("checkpoint", "unfrozen"),
    }

    rows: list[dict[str, Any]] = []
    predictions: dict[str, Hypothesis] = {}
    # Every method exported for post-freeze evaluation must retain the exact
    # pre-freeze fit that generated its selection score.  Verification receives
    # this map and only calls ``score`` on these objects after the freeze.
    fitted_predictions: dict[str, FittedHypothesis] = {}
    control_fitted: dict[str, FittedHypothesis] = {}
    fitted_prediction_reasons: dict[str, str] = {}
    selectors: dict[str, dict[str, Any]] = {}
    notes: list[str] = []

    # ---------------- E0: image-only intensity grouping, same ontology -------
    e0_partition = intensity_grouping_hypothesis(fitting_view.image, fitting_view.support)
    e0_candidate = _resolve_roles(e0_partition, fitting_view)
    e0_fitted, e0_score = _fit_and_score(model, e0_candidate, fitting_view, selection_view, budget)
    selectors["E0_intensity_grouping"] = {
        "experiment": "E0",
        "description": "simple image-only intensity grouping with the same ontology, no bank, no audit",
        "candidate_id": e0_candidate.candidate_id,
        "selection": _score_entry(e0_score),
        "fit": _score_entry(e0_fitted.fit_score),
        "from_common_bank": False,
    }
    predictions["E0_intensity_grouping"] = e0_candidate
    fitted_predictions["E0_intensity_grouping"] = e0_fitted
    rows.append(
        _prediction_row("E0_intensity_grouping", "selector", e0_candidate, selection=e0_score,
                        fit=e0_fitted.fit_score, provenance=provenance, from_common_bank=False)
    )

    # ---------------- E1: cuts_inspired_control (feature grouping, no audit) --
    incumbent = audit_result.initial
    e1_fitted = index.fitted.get(incumbent.candidate_id)
    e1_score = index.selection.get(incumbent.candidate_id)
    if e1_fitted is None:
        e1_fitted, e1_score = _fit_and_score(model, incumbent, fitting_view, selection_view, budget)
    elif e1_score is None:
        e1_score = model.score(e1_fitted, selection_view)
        budget["scores"] += 1
    selectors["E1_cuts_inspired_control"] = {
        "experiment": "E1",
        "name": "cuts_inspired_control",
        "description": (
            "in-repo simplified self-supervised feature grouping plus ontology, no audit; "
            "NOT a reproduction of the published CUTS result"
        ),
        "candidate_id": incumbent.candidate_id,
        "selection": _score_entry(e1_score),
        "fit": _score_entry(e1_fitted.fit_score),
        "from_common_bank": True,
        "external_cuts_reproduction": "pending",
    }
    predictions["E1_cuts_inspired_control"] = incumbent
    if e1_fitted is not None:
        fitted_predictions["E1_cuts_inspired_control"] = e1_fitted
    else:
        fitted_prediction_reasons["E1_cuts_inspired_control"] = "incumbent fit unavailable"
    rows.append(
        _prediction_row("E1_cuts_inspired_control", "selector", incumbent, selection=e1_score,
                        fit=None if e1_fitted is None else e1_fitted.fit_score,
                        provenance=provenance, from_common_bank=True)
    )
    notes.append(
        "E1 is named cuts_inspired_control; exact external CUTS reproduction is pending, not completed."
    )

    # ---------------- E2: confidence / consistency only ----------------------
    confidence_table = {
        hypothesis.candidate_id: confidence_consistency(hypothesis)
        for hypothesis in index.hypotheses
    }
    if confidence_table:
        e2_id = max(confidence_table, key=lambda key: confidence_table[key]["combined"])
        e2_candidate = next(h for h in index.hypotheses if h.candidate_id == e2_id)
        e2_score = index.selection.get(e2_id)
        e2_fitted = index.fitted.get(e2_id)
        selectors["E2_confidence_consistency"] = {
            "experiment": "E2",
            "description": "choose from the shared bank by fit-only confidence/consistency, no observation evidence",
            "candidate_id": e2_id,
            "criterion": confidence_table[e2_id],
            "all_candidates": confidence_table,
            "selection": _score_entry(e2_score),
            "from_common_bank": True,
            "confidence_constant": bool(confidence_table[e2_id]["confidence_constant"]),
            "criterion_note": confidence_table[e2_id]["confidence_note"],
        }
        predictions["E2_confidence_consistency"] = e2_candidate
        if e2_fitted is not None:
            fitted_predictions["E2_confidence_consistency"] = e2_fitted
        else:
            fitted_prediction_reasons["E2_confidence_consistency"] = (
                "selected bank candidate fit unavailable"
            )
        rows.append(
            _prediction_row("E2_confidence_consistency", "selector", e2_candidate, selection=e2_score,
                            fit=None if e2_fitted is None else e2_fitted.fit_score,
                            provenance=provenance, from_common_bank=True,
                            confidence=confidence_table[e2_id]["confidence"],
                            consistency=confidence_table[e2_id]["consistency"],
                            confidence_constant=confidence_table[e2_id]["confidence_constant"],
                            criterion_note=confidence_table[e2_id]["confidence_note"])
        )
    else:  # pragma: no cover - a bank is never empty under the W2 contract
        selectors["E2_confidence_consistency"] = {
            "experiment": "E2", "available": False, "reason": "empty bank"}

    # ---------------- E3: fitting-side score only ----------------------------
    fit_totals = {
        candidate_id: _total(fitted.fit_score) for candidate_id, fitted in index.fitted.items()
    }
    available_fit = {k: v for k, v in fit_totals.items() if v is not None}
    if available_fit:
        e3_id = min(available_fit, key=lambda key: available_fit[key])
        e3_candidate = index.fitted[e3_id].hypothesis
        e3_score = index.selection.get(e3_id)
        selectors["E3_fitting_score"] = {
            "experiment": "E3",
            "description": "choose by fitting-side score only; no withheld observation is consulted",
            "candidate_id": e3_id,
            "fit_totals": fit_totals,
            "selection": _score_entry(e3_score),
            "from_common_bank": True,
        }
        predictions["E3_fitting_score"] = e3_candidate
        fitted_predictions["E3_fitting_score"] = index.fitted[e3_id]
        rows.append(
            _prediction_row("E3_fitting_score", "selector", e3_candidate, selection=e3_score,
                            fit=index.fitted[e3_id].fit_score, provenance=provenance,
                            from_common_bank=True)
        )
    else:
        selectors["E3_fitting_score"] = {
            "experiment": "E3", "available": False,
            "reason": "no candidate reported an available fitting-side total",
        }

    # ---------------- E4: withheld selection evidence, no active challenge ---
    initial_pool = index.e4_pool_ids or [h.candidate_id for h in index.hypotheses]
    selection_totals = {
        candidate_id: _total(score) for candidate_id, score in index.selection.items()
    }
    e4_pool = {
        candidate_id: value
        for candidate_id, value in selection_totals.items()
        if value is not None and candidate_id in set(initial_pool)
    }
    e4_id: str | None = None
    if e4_pool:
        e4_id = min(e4_pool, key=lambda key: e4_pool[key])
        e4_candidate = next(
            (h for h in index.hypotheses if h.candidate_id == e4_id),
            index.fitted[e4_id].hypothesis if e4_id in index.fitted else None,
        )
        selectors["E4_selection_evidence"] = {
            "experiment": "E4",
            "description": (
                "choose by withheld selection predictive evidence over the pre-challenge "
                "candidates only; no active challenger"
            ),
            "candidate_id": e4_id,
            "pool": sorted(e4_pool),
            "selection": _score_entry(index.selection.get(e4_id)),
            "from_common_bank": True,
            "pool_source": index.pool_source,
        }
        if e4_candidate is not None:
            predictions["E4_selection_evidence"] = e4_candidate
            if e4_id in index.fitted:
                fitted_predictions["E4_selection_evidence"] = index.fitted[e4_id]
            else:
                fitted_prediction_reasons["E4_selection_evidence"] = (
                    "selected pre-challenge candidate fit unavailable"
                )
        rows.append(
            _prediction_row("E4_selection_evidence", "selector", e4_candidate, selection=index.selection.get(e4_id),
                            fit=index.fitted[e4_id].fit_score if e4_id in index.fitted else None,
                            provenance=provenance, from_common_bank=True)
        )
    else:
        selectors["E4_selection_evidence"] = {
            "experiment": "E4", "available": False,
            "reason": "no pre-challenge candidate produced an available selection score",
        }

    # ---------------- E5: full evidence selection plus active challenge ------
    selected = audit_result.selected
    e5_score = index.selection.get(selected.candidate_id)
    e5_fitted = index.fitted.get(selected.candidate_id)
    tie_with_e4 = e4_id is not None and e4_id == selected.candidate_id
    selectors["E5_evidence_plus_challenge"] = {
        "experiment": "E5",
        "description": "full predictive-evidence selection with the active challenger",
        "candidate_id": selected.candidate_id,
        "selection": _score_entry(e5_score),
        "from_common_bank": True,
        "tie_with_e4": bool(tie_with_e4),
        "tie_note": (
            "E4 and E5 selected the same candidate on this bank; this is the expected outcome "
            "when their only difference is ambiguity reporting, and it is not a mask-quality gain"
            if tie_with_e4
            else None
        ),
        "semantic_unresolved": bool(selected.semantic_unresolved),
        "alternatives": len(selected.alternatives),
    }
    predictions["E5_evidence_plus_challenge"] = selected
    if e5_fitted is not None:
        fitted_predictions["E5_evidence_plus_challenge"] = e5_fitted
    else:
        fitted_prediction_reasons["E5_evidence_plus_challenge"] = (
            "auditor-selected candidate fit unavailable"
        )
    rows.append(
        _prediction_row("E5_evidence_plus_challenge", "selector", selected, selection=e5_score,
                        fit=None if e5_fitted is None else e5_fitted.fit_score,
                        provenance=provenance, from_common_bank=True,
                        tie_with_e4=bool(tie_with_e4))
    )
    if tie_with_e4:
        notes.append(
            "E4 and E5 selected the identical candidate; no difference in mask quality is claimed."
        )

    # ---------------- negative controls --------------------------------------
    controls: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(seed)

    # 1. random same-budget selection from the shared bank
    if index.hypotheses:
        pick = int(torch.randint(0, len(index.hypotheses), (1,), generator=generator).item())
        random_candidate = index.hypotheses[pick]
        random_score = index.selection.get(random_candidate.candidate_id)
        random_fitted = index.fitted.get(random_candidate.candidate_id)
        if random_score is None:
            random_fitted, random_score = _fit_and_score(
                model, random_candidate, fitting_view, selection_view, budget
            )
        controls.append(
            {
                "control": "control_random_same_budget",
                "description": "uniform random pick from the same bank at the same selection budget",
                "candidate_id": random_candidate.candidate_id,
                "selection": _score_entry(random_score),
                "available": True,
                "reason": None,
            }
        )
        predictions["control_random_same_budget"] = random_candidate
        if random_fitted is not None:
            control_fitted["control_random_same_budget"] = random_fitted
            fitted_predictions["control_random_same_budget"] = random_fitted
        else:
            fitted_prediction_reasons["control_random_same_budget"] = "bank fit unavailable"
        rows.append(
            _prediction_row("control_random_same_budget", "negative_control", random_candidate,
                            selection=random_score,
                            fit=None if random_fitted is None else random_fitted.fit_score,
                            provenance=provenance)
        )

    # 2. shuffled evidence from an incompatible study
    foreign = cache.foreign(
        fitting_view.study_id, dataset=dataset_scope, run_id=run_scope
    )
    if foreign is None:
        controls.append(
            {
                "control": "control_shuffled_evidence",
                "description": "select using evidence computed on a different, incompatible study",
                "available": False,
                "reason": (
                    "no incompatible study has been cached yet; this unit's scores were queued "
                    "for a later unit rather than fabricating foreign evidence"
                ),
                "queued": True,
            }
        )
        rows.append(
            _prediction_row("control_shuffled_evidence", "negative_control", None, selection=None,
                            fit=None, provenance=provenance, available=False,
                            reason="no cached incompatible study available yet")
        )
    else:
        # Candidate ids are unit-specific.  The foreign selector therefore
        # uses only the preregistered bank-slot correspondence recorded by the
        # cache, never an id join across studies.
        raw_slot_scores = foreign.get("slot_scores")
        if not isinstance(raw_slot_scores, Mapping):
            # Legacy cache entries are converted to deterministic slots.  This
            # path remains explicit in the trace so it cannot be mistaken for
            # an exact-id match.
            raw_scores = foreign.get("scores", {})
            raw_slot_scores = {
                str(slot): float(value)
                for slot, (_, value) in enumerate(sorted(raw_scores.items()))
                if value is not None and math.isfinite(float(value))
            }
        shared_slots = [slot for slot in raw_slot_scores if int(slot) in candidate_by_slot]
        if shared_slots:
            shuffled_slot = min(shared_slots, key=lambda key: float(raw_slot_scores[key]))
            shuffled_id = candidate_by_slot[int(shuffled_slot)]
            shuffled_candidate = next(h for h in index.hypotheses if h.candidate_id == shuffled_id)
            foreign_total = float(raw_slot_scores[shuffled_slot])
            foreign_slot_mapping = {
                str(slot): candidate_by_slot[int(slot)] for slot in shared_slots
            }
            controls.append(
                {
                    "control": "control_shuffled_evidence",
                    "description": "select using evidence computed on a different, incompatible study",
                    "candidate_id": shuffled_id,
                    "foreign_study_id": foreign["study_id"],
                    "foreign_unit_id": foreign["unit_id"],
                    "foreign_candidate_slot": int(shuffled_slot),
                    "foreign_selection_total": foreign_total,
                    "slot_mapping": foreign_slot_mapping,
                    "selection": _score_entry(index.selection.get(shuffled_id)),
                    "agrees_with_e5": shuffled_id == selected.candidate_id,
                    "available": True,
                    "reason": None,
                }
            )
            predictions["control_shuffled_evidence"] = shuffled_candidate
            shuffled_fitted = index.fitted.get(shuffled_id)
            if shuffled_fitted is not None:
                control_fitted["control_shuffled_evidence"] = shuffled_fitted
                fitted_predictions["control_shuffled_evidence"] = shuffled_fitted
            else:
                fitted_prediction_reasons["control_shuffled_evidence"] = "bank fit unavailable"
            rows.append(
                _prediction_row("control_shuffled_evidence", "negative_control", shuffled_candidate,
                                selection=index.selection.get(shuffled_id), fit=None,
                                provenance=provenance, foreign_study_id=foreign["study_id"],
                                foreign_unit_id=foreign["unit_id"],
                                foreign_candidate_slot=int(shuffled_slot),
                                foreign_selection_total=foreign_total,
                                value_override=foreign_total)
            )
        else:
            controls.append(
                {
                    "control": "control_shuffled_evidence",
                    "available": False,
                    "reason": (
                        "cached foreign study has no shared preregistered bank slot with this bank; "
                        "shuffled evidence cannot be applied without inventing a mapping"
                    ),
                }
            )
            rows.append(
                _prediction_row("control_shuffled_evidence", "negative_control", None,
                                selection=None, fit=None, provenance=provenance, available=False,
                                reason="cached foreign study has no shared preregistered bank slot")
            )
    cache.push(
        fitting_view.study_id,
        fitting_view.unit_id,
        {cid: value for cid, value in selection_totals.items() if value is not None},
        dataset=dataset_scope,
        run_id=run_scope,
        candidate_slots=slot_by_candidate,
    )

    # 3-6. degenerate partitions, each fitted at the same budget as a real candidate
    shape = tuple(fitting_view.support.shape)
    # Kept so that finalization can freeze these fits alongside every other
    # compared method and post-freeze verification can score them without
    # refitting, which the observation model forbids.
    fitted_controls: dict[str, FittedHypothesis] = {}
    degenerates = [
        ("control_all_background", all_background_hypothesis(shape)),
        ("control_random_spatial_masks", random_mask_hypothesis(shape, seed=seed)),
        ("control_class_permutation", class_permutation_hypothesis(selected)),
        ("control_excessive_partition", excessive_partition_hypothesis(shape)),
    ]
    selected_total = _total(e5_score)
    for name, candidate in degenerates:
        fitted, score = _fit_and_score(model, candidate, fitting_view, selection_view, budget)
        total = _total(score)
        beats = total is not None and selected_total is not None and total < selected_total
        controls.append(
            {
                "control": name,
                "description": f"degenerate partition diagnostic: {candidate.metadata.get('degenerate')}",
                "candidate_id": candidate.candidate_id,
                "selection": _score_entry(score),
                "fit": _score_entry(fitted.fit_score),
                "beats_selected": bool(beats),
                "available": total is not None,
                "reason": None if total is not None else "no available selection score",
                "note": "diagnostic only; never used as a training label",
            }
        )
        predictions[name] = candidate
        fitted_controls[name] = fitted
        control_fitted[name] = fitted
        fitted_predictions[name] = fitted
        rows.append(
            _prediction_row(name, "negative_control", candidate, selection=score,
                            fit=fitted.fit_score, provenance=provenance,
                            beats_selected=bool(beats))
        )

    control_failures = [entry["control"] for entry in controls if entry.get("beats_selected")]
    if control_failures:
        notes.append(
            "Falsification signal: degenerate control(s) "
            + ", ".join(control_failures)
            + " scored better than the selected candidate on the selection observations."
        )

    # ---------------- paired search comparison -------------------------------
    search = _random_edit_comparison(
        index, incumbent, fitting_view, selection_view, model, budget, seed=seed
    )
    rows.extend(search.pop("rows"))
    best_random = search.pop("best_random_edit_hypothesis", None)
    search.pop("fitted_random_edits", None)
    best_random_fitted = search.pop("best_random_edit_fitted", None)
    if best_random is not None:
        # The random-edit arm is exported as its own compared method, so the
        # freeze can show that the challenger was measured against a real,
        # equal-cost opponent rather than against an absent one.
        predictions["control_random_edit_search"] = best_random
        if best_random_fitted is not None:
            control_fitted["control_random_edit_search"] = best_random_fitted
            fitted_predictions["control_random_edit_search"] = best_random_fitted
        else:
            fitted_prediction_reasons["control_random_edit_search"] = (
                "best random edit fit unavailable"
            )
    elif index.challenge_ids:
        # This is only reached when the random search had no finite score.  The
        # reason is kept separate from a missing method so finalization can
        # report an unavailable comparison honestly.
        fitted_prediction_reasons["control_random_edit_search"] = (
            "random edit search produced no available candidate"
        )

    # ---------------- coverage -----------------------------------------------
    coverage = {
        "levels": list(COVERAGE_LEVELS),
        "selected_natural_coverage": float((selected.validity > 0).to(torch.float64).mean().item()),
        "incumbent_natural_coverage": float(
            (incumbent.validity > 0).to(torch.float64).mean().item()
        ),
        "note": (
            "natural coverage of each arm; a matched-coverage comparison is required before "
            "attributing any quality difference to better labels rather than to discarding more pixels"
        ),
    }
    matched_selected, matched_incumbent = matched_coverage(selected.validity, incumbent.validity)
    coverage["matched_supported_pixels"] = int(matched_selected.sum().item())
    coverage["matched_equal"] = int(matched_selected.sum().item()) == int(
        matched_incumbent.sum().item()
    )

    return {
        "contract_version": VERSION,
        "study_id": fitting_view.study_id,
        "unit_id": fitting_view.unit_id,
        "partition_id": fitting_view.partition_id,
        "protocol": fitting_view.protocol,
        "seed": seed,
        "rows": rows,
        "predictions": predictions,
        "selectors": selectors,
        "controls": controls,
        "fitted_controls": fitted_controls,
        # ``control_fitted`` includes every control with a real pre-freeze fit;
        # ``fitted_controls`` is retained as the historical degenerate-only
        # alias used by focused tests and callers.
        "control_fitted": control_fitted,
        "fitted_predictions": fitted_predictions,
        "fitted_prediction_reasons": fitted_prediction_reasons,
        "search_comparison": search,
        "coverage": coverage,
        "budget": budget,
        "bank": {
            "candidate_ids": [h.candidate_id for h in index.hypotheses],
            "candidate_slots": slot_by_candidate,
            "challenge_candidate_ids": index.challenge_ids,
            "challenge_source": index.challenge_source,
            "e4_pool_candidate_ids": index.e4_pool_ids,
            "pool_source": index.pool_source,
            "common_bank": True,
        },
        "notes": notes,
        "external_cuts_reproduction": "pending",
    }


def _random_edit_comparison(
    index: _BankIndex,
    incumbent: Hypothesis,
    fitting_view: FittingView,
    selection_view: ScoringView,
    model: Any,
    budget: dict[str, int],
    *,
    seed: int,
) -> dict[str, Any]:
    """Equal-number, equal-fitting-cost random edits versus the active challenger.

    Without this comparison, any advantage of the challenger could be explained
    by it simply having produced more candidates. The random-edit set is given
    the same number of candidates and the same per-candidate fitting budget, so
    the only remaining difference is *which* edits were proposed.
    """
    challenge_ids = index.challenge_ids
    rows: list[dict[str, Any]] = []
    provenance = {
        "dataset": fitting_view.metadata.get("dataset", "unknown"),
        "split": fitting_view.metadata.get("split", "train"),
        "protocol": fitting_view.protocol,
        "epoch": fitting_view.metadata.get("epoch"),
        "checkpoint": fitting_view.metadata.get("checkpoint", "unfrozen"),
    }
    if not challenge_ids:
        return {
            "best_random_edit_hypothesis": None,
            "best_random_edit_fitted": None,
            "fitted_random_edits": {},
            "available": False,
            "reason": (
                "the audit trace did not identify challenge-generated candidates, so an "
                "equal-number random-edit comparison has no target set to match"
            ),
            "random_edits": [],
            "rows": rows,
        }

    challenge_totals = {
        candidate_id: _total(index.selection.get(candidate_id)) for candidate_id in challenge_ids
    }
    usable_challenge = {k: v for k, v in challenge_totals.items() if v is not None}
    baseline_total = _total(index.selection.get(incumbent.candidate_id))

    random_entries: list[dict[str, Any]] = []
    random_hypotheses: dict[str, Hypothesis] = {}
    fitted_random_edits: dict[str, FittedHypothesis] = {}
    steps_before = budget["fitting_steps"]
    for position in range(len(challenge_ids)):
        candidate = random_edit(incumbent, seed=seed, index=position)
        random_hypotheses[candidate.candidate_id] = candidate
        fitted, score = _fit_and_score(model, candidate, fitting_view, selection_view, budget)
        fitted_random_edits[candidate.candidate_id] = fitted
        random_entries.append(
            {
                "candidate_id": candidate.candidate_id,
                "selection": _score_entry(score),
                "fit": _score_entry(fitted.fit_score),
                "total": _total(score),
                "fitting_steps": int(fitted.fitting_steps),
            }
        )
        rows.append(
            _prediction_row(candidate.candidate_id, "random_edit_control", candidate,
                            selection=score, fit=fitted.fit_score, provenance=provenance)
        )
    random_steps = budget["fitting_steps"] - steps_before
    challenge_steps = sum(
        int(index.fitted[candidate_id].fitting_steps)
        for candidate_id in challenge_ids
        if candidate_id in index.fitted
    )

    usable_random = {
        entry["candidate_id"]: entry["total"]
        for entry in random_entries
        if entry["total"] is not None
    }
    best_random_id = min(usable_random, key=lambda key: usable_random[key]) if usable_random else None
    best_random_hypothesis = random_hypotheses.get(best_random_id) if best_random_id else None
    best_challenge = min(usable_challenge.values()) if usable_challenge else None
    best_random = min(usable_random.values()) if usable_random else None
    gain = (
        None
        if best_challenge is None or best_random is None
        else float(best_random - best_challenge)
    )
    return {
        "available": best_challenge is not None and best_random is not None,
        "reason": (
            None
            if best_challenge is not None and best_random is not None
            else "one side produced no available selection score"
        ),
        "challenge_candidate_ids": challenge_ids,
        "challenge_totals": challenge_totals,
        "best_challenge_total": best_challenge,
        "random_edits": random_entries,
        "best_random_edit_total": best_random,
        "best_random_edit_candidate_id": best_random_id,
        "best_random_edit_hypothesis": best_random_hypothesis,
        "best_random_edit_fitted": (
            None if best_random_id is None else fitted_random_edits.get(best_random_id)
        ),
        "fitted_random_edits": fitted_random_edits,
        "equal_number": len(random_entries) == len(challenge_ids),
        "challenge_fitting_steps": challenge_steps,
        "random_edit_fitting_steps": random_steps,
        "equal_fitting_cost": challenge_steps == random_steps,
        "incumbent_total": baseline_total,
        "challenge_gain_over_random_edits": gain,
        "interpretation": (
            "positive gain means the challenger's proposals scored better than an equal-number, "
            "equal-cost random-edit set; a non-positive gain attributes any apparent benefit to "
            "search budget rather than to the challenge policy"
        ),
        "rows": rows,
    }


__all__ = [
    "CONTROL_IDS",
    "DEFAULT_FOREIGN_CACHE",
    "SELECTOR_IDS",
    "ForeignEvidenceCache",
    "confidence_consistency",
    "coverage_masks",
    "matched_coverage",
    "random_edit",
    "run_bank_experiments",
    "verify_frozen_bank",
]
