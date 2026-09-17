"""Evidence auditor and challenger over a frozen candidate bank (W2).

Scientific role
---------------
:func:`audit_bank` is the *directly computed* evidence auditor demanded by the
contract: it fits each banked candidate on fitting observations only, scores each
frozen fit against withheld **selection** observations through the W1
:class:`~self_audit_maskfree.observation.ObservationModel`, and keeps the
incumbent unless a challenger beats it by a preregistered strict margin.

There is no learned correctness head here, no ground truth, and no access to the
verification role. The auditor consumes selection observations, which is adaptive
information use and is logged as such.

What the numbers are
--------------------
``EvidenceScore.total`` is a predictive negative log-likelihood in nats per valid
observed pixel plus declared complexity and prior terms. The regional margin this
module returns is a *difference of such scores*, not a probability that a pixel
is correctly labelled, and ``AuditResult.validity`` is an evidence weight, not a
correctness certificate.

Budget
------
Provisional fixed budget, logged rather than tuned: :data:`DEFAULT_ROUNDS` rounds
over the bank's candidate slots, at most one fit per candidate (the incumbent fit
is computed once and cached), and the fitting iteration cap belongs to the
observation model. Two rounds over four slots means **four** fits in total, not
eight. Candidates are fitted and scored one at a time.

Anti-starvation
---------------
A rejected challenger rejects *that edit only*. The remaining rounds still run,
the remaining candidates are still fitted and scored, and the unit stays a legal
self-supervised learning sample. Nothing here halts training.

Empty or degenerate challenge
-----------------------------
If no candidate other than the incumbent produces an available score, or every
candidate is pixel-identical to the incumbent, the trace records
``search_inconclusive=True`` and the regional margin is zero everywhere. An
unchallenged incumbent is not a confident incumbent.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .contracts import (
    VERSION,
    AuditResult,
    EvidenceScore,
    FittedHypothesis,
    FittingView,
    Hypothesis,
    ScoringView,
)
from .observation import ObservationModel
from .ontology import connected_components

#: Preregistered rounds over the bank's candidate slots.
DEFAULT_ROUNDS = 2
#: Preregistered strict improvement in nats per valid observed pixel. A
#: challenger below this margin loses and the incumbent is retained.
DEFAULT_IMPROVEMENT_THRESHOLD = 0.001
#: Scale, in nats per pixel, at which a regional margin saturates its weight.
#: Provisional constant, recorded, not fitted.
MARGIN_SCALE = 0.01
#: Hard budget ceilings. The audit refuses inputs that exceed them rather than
#: quietly spending more than the preregistered budget.
MAX_BANK_CANDIDATES = 4
MAX_FIT_ITERATIONS = 5
MAX_ROUNDS = 2
#: Largest connected regions of the selection that receive a scored margin.
#: Regions beyond this get zero evidence and are counted, never silently
#: credited with the score of a region they are not part of.
MAX_SCORED_REGIONS = 32
#: Ceiling on the extra ``score`` calls one audit may spend on regional margins.
#: Scoring is cheap relative to fitting, but it is still budgeted and logged.
MAX_REGION_SCORE_CALLS = 128
#: Ceiling on the validity an unchallenged pixel can reach from same-bank
#: agreement alone. Agreement inside one bank is not reproducibility: it is only
#: a diagnostic of dissent within this one bounded search.
AGREEMENT_WEIGHT_CAP = 0.5
#: Fixed region order used when enumerating the selection's regions.
ROLE_LABELS = ((0, "BG"), (1, "RV"), (2, "MYO"), (3, "LV"))


class AuditError(ValueError):
    """Raised when an audit input violates the frozen maskfree150 contract."""


def _score_dict(score: EvidenceScore) -> dict[str, Any]:
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
        "unit": "nats per valid observed pixel (plus beta*complexity and prior)",
    }


def _round_slots(count: int, rounds: int) -> list[list[int]]:
    """Split candidate slots 1..count-1 over ``rounds`` rounds, incumbent aside.

    Slot 0 is the incumbent and is fitted exactly once, before round 1. The
    remaining slots are dealt out so that ``sum(len(r) for r in rounds) ==
    count - 1``: the rounds partition one bank, they do not multiply it.
    """
    challengers = list(range(1, count))
    if rounds < 1:
        raise AuditError("rounds must be >= 1")
    size = -(-len(challengers) // rounds) if challengers else 0
    slots = [challengers[index: index + size] for index in range(0, len(challengers), size or 1)]
    slots = slots[:rounds] if slots else []
    assigned = [slot for group in slots for slot in group]
    leftover = [slot for slot in challengers if slot not in assigned]
    if leftover and slots:
        slots[-1].extend(leftover)
    elif leftover:  # pragma: no cover - rounds >= 1 always yields a group
        slots = [leftover]
    while len(slots) < rounds:
        slots.append([])
    return slots


def _distinctness(bank: Sequence[Hypothesis]) -> list[dict[str, Any]]:
    rows = []
    for index, candidate in enumerate(bank):
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "source": candidate.source,
                "hamming_vs_incumbent": float(
                    (candidate.labels != bank[0].labels).float().mean().item()
                ),
                "distinct_from_incumbent": bool((candidate.labels != bank[0].labels).any())
                if index > 0
                else False,
                "semantic_unresolved": bool(candidate.semantic_unresolved),
                "prior_penalty": float(candidate.metadata.get("prior_penalty", 0.0)),
                "prior_violations": dict(
                    candidate.metadata.get("ontology_trace", {}).get("prior_violations", {})
                ),
                "generation": dict(candidate.metadata.get("generation", {})),
            }
        )
    return rows


def _restricted_selection_view(scoring_view: ScoringView, region: torch.Tensor) -> ScoringView:
    """A selection view narrowed to one region. Same role, same ids, no new data.

    This reads nothing the caller did not already hold: it is the *same*
    selection observations with a smaller support. It exists so that the same
    frozen fits can be scored on exactly the same region support, which is what
    makes a per-region comparison fair.
    """
    support = scoring_view.support & region
    image = torch.where(support, scoring_view.image[0], torch.zeros_like(scoring_view.image[0]))
    return ScoringView(
        image=image.unsqueeze(0),
        support=support,
        study_id=scoring_view.study_id,
        unit_id=scoring_view.unit_id,
        role=scoring_view.role,
        partition_id=scoring_view.partition_id,
    )


def _selected_regions(labels: torch.Tensor) -> tuple[list[dict[str, Any]], bool]:
    """Connected regions of the selected partition, largest first.

    A region, not a class: two disconnected pieces of the same anatomical role
    are separate pieces of evidence and are scored separately.
    """
    regions: list[dict[str, Any]] = []
    converged_all = True
    for role, name in ROLE_LABELS:
        role_mask = labels == role
        if not bool(role_mask.any()):
            continue
        components, converged = connected_components(role_mask)
        converged_all &= converged
        if not converged:
            regions.append({"role": name, "component": 0, "mask": role_mask})
            continue
        for component in torch.unique(components[role_mask]).tolist():
            regions.append(
                {"role": name, "component": int(component), "mask": components == int(component)}
            )
    regions.sort(key=lambda entry: int(entry["mask"].sum().item()), reverse=True)
    return regions, converged_all


def _regional_margin(
    bank: Sequence[Hypothesis],
    selected_index: int,
    fitted: Sequence[FittedHypothesis | None],
    model: ObservationModel,
    selection_view: ScoringView,
    *,
    prepared: Sequence[Any | None] | None = None,
    physical_work: dict[str, int] | None = None,
) -> tuple[torch.Tensor, bool, dict[str, Any]]:
    """Per-region predictive evidence margin on withheld selection observations.

    For each connected region of the *selected* partition, the frozen fit of the
    selection and the frozen fit of every candidate that actually disagrees
    inside that region are scored on the **same** region support. The margin is
    the smallest normalized-NLL gap among those disagreeing candidates, in nats
    per valid observed pixel of that region:

    ``margin(r) = min_c [ NLL_c(O_select in r) - NLL_selected(O_select in r) ]``

    This is a genuinely regional quantity. It is not the unit-level score
    difference copied onto pixels, which would have told every region the same
    story regardless of where the candidates actually differ.

    A region with no disagreeing candidate, and a region with no observed
    selection pixel, both get margin **zero**: an unchallenged or unobserved
    region has earned no predictive evidence and must not be handed confidence
    it never demonstrated. Regions beyond the scoring budget also get zero and
    are counted.

    If W1's optional ``prepare_score``/``score_region`` API is available, each
    frozen fit is prepared once and regional reductions use that immutable
    snapshot.  Otherwise the original scalar ``score`` fallback is used.  No
    candidate is refitted, and the number of extra score calls and each
    region's valid denominator are recorded.
    """
    selected = bank[selected_index]
    if physical_work is None:
        physical_work = {}
    if prepared is not None and len(prepared) != len(fitted):
        raise AuditError("prepared regional scores must align with fitted candidates")
    margin = torch.zeros_like(selected.labels, dtype=torch.float32)
    report: dict[str, Any] = {
        "regions": [],
        "score_calls": 0,
        "score_counts": [],
        "refits": 0,
        "regions_total": 0,
        "regions_scored": 0,
        "regions_unchallenged": 0,
        "regions_unobserved": 0,
        "regions_over_budget": 0,
        "components_converged": True,
        "definition": (
            "min over disagreeing candidates of the normalized selection NLL gap on the "
            "same region support, in nats per valid observed pixel"
        ),
    }
    def close_report() -> None:
        """Expose explicit accounting for the extra regional score work."""
        counts = [int(count) for count in report["score_counts"]]
        report["extra_score_calls"] = int(report["score_calls"])
        report["regional_score_calls"] = int(report["score_calls"])
        report["regional_score_observation_count"] = int(sum(counts))
        report["regional_score_counts"] = counts
        report["regional_refits"] = int(report["refits"])

    if fitted[selected_index] is None:
        report["unavailable_reason"] = "the selection has no frozen fit to score"
        close_report()
        return margin, True, report

    challengers = [
        index
        for index in range(len(bank))
        if index != selected_index
        and fitted[index] is not None
        and bool((bank[index].labels != selected.labels).any())
    ]
    if not challengers:
        report["unavailable_reason"] = "no distinct challenger produced a frozen fit"
        close_report()
        return margin, True, report

    regions, converged = _selected_regions(selected.labels)
    report["components_converged"] = converged
    report["regions_total"] = len(regions)
    scored_anywhere = False

    for position, region in enumerate(regions):
        mask = region["mask"]
        pixels = int(mask.sum().item())
        observed = int((selection_view.support & mask).sum().item())
        # A candidate is a regional alternative only when it disagrees on an
        # observed selection pixel in this region. Differences confined to fit
        # pixels cannot earn an O_select margin.
        observed_mask = mask & selection_view.support
        disagreeing_anywhere = [
            index
            for index in challengers
            if bool((bank[index].labels != selected.labels)[mask].any())
        ]
        disagreeing = [
            index
            for index in challengers
            if bool((bank[index].labels != selected.labels)[observed_mask].any())
        ]
        row: dict[str, Any] = {
            "role": region["role"],
            "component": region["component"],
            "pixels": pixels,
            "observed_selection_pixels": observed,
            "disagreeing_candidates": [bank[index].candidate_id for index in disagreeing],
            "unobserved_disagreeing_candidates": [
                bank[index].candidate_id
                for index in disagreeing_anywhere
                if index not in disagreeing
            ],
            "margin_nats_per_pixel": 0.0,
            "available": False,
        }
        if position >= MAX_SCORED_REGIONS or report["score_calls"] >= MAX_REGION_SCORE_CALLS:
            row["reason"] = "region scoring budget exhausted; zero evidence recorded"
            report["regions_over_budget"] += 1
            report["regions"].append(row)
            continue
        if observed == 0 and disagreeing_anywhere:
            row["reason"] = "no observed selection pixel in this region"
            report["regions_unobserved"] += 1
            report["regions"].append(row)
            continue
        if not disagreeing:
            row["reason"] = "no banked candidate disagrees inside this region"
            report["regions_unchallenged"] += 1
            report["regions"].append(row)
            continue
        # The region becomes evidence-bearing only after at least one challenger
        # produces an available score on this same O_select support.

        sub_view = _restricted_selection_view(selection_view, mask)

        def score_one(
            index: int,
            *,
            _mask: torch.Tensor = mask,
            _sub_view: ScoringView = sub_view,
        ) -> EvidenceScore:
            snapshot = prepared[index] if prepared is not None else None
            score_region = getattr(model, "score_region", None)
            if snapshot is not None and callable(score_region):
                physical_work["score_region_calls"] = physical_work.get("score_region_calls", 0) + 1
                return score_region(snapshot, _mask)
            physical_work["scalar_region_score_calls"] = (
                physical_work.get("scalar_region_score_calls", 0) + 1
            )
            return model.score(fitted[index], _sub_view)  # type: ignore[arg-type]

        selected_score = score_one(selected_index)
        report["score_calls"] += 1
        report["score_counts"].append(int(selected_score.count))
        if not selected_score.available or selected_score.normalized_nll is None:
            row["reason"] = selected_score.reason or "selection score unavailable in this region"
            report["regions"].append(row)
            continue
        best_gap: float | None = None
        best_candidate: str | None = None
        for index in disagreeing:
            if report["score_calls"] >= MAX_REGION_SCORE_CALLS:
                row["reason"] = "region scoring budget exhausted mid-region"
                break
            challenger_score = score_one(index)
            report["score_calls"] += 1
            report["score_counts"].append(int(challenger_score.count))
            if not challenger_score.available or challenger_score.normalized_nll is None:
                continue
            gap = float(challenger_score.normalized_nll) - float(selected_score.normalized_nll)
            if best_gap is None or gap < best_gap:
                best_gap, best_candidate = gap, bank[index].candidate_id
        if best_gap is None:
            row.setdefault("reason", "no disagreeing candidate produced an available region score")
            report["regions"].append(row)
            continue
        margin[mask] = float(best_gap)
        scored_anywhere = True
        row.update(
            {
                "margin_nats_per_pixel": float(best_gap),
                "selected_normalized_nll": float(selected_score.normalized_nll),
                "closest_challenger": best_candidate,
                "available": True,
            }
        )
        report["regions_scored"] += 1
        report["regions"].append(row)

    # A distinct candidate outside O_select, or a distinct candidate in an
    # unobserved region, is not a predictive challenge. Both cases therefore
    # leave the margin at zero and make the search inconclusive.
    inconclusive = not scored_anywhere
    if inconclusive:
        report["unavailable_reason"] = (
            "no distinct banked candidate disagrees on an observed selection region"
        )
    close_report()
    return margin, inconclusive, report


def _same_bank_agreement(bank: Sequence[Hypothesis], selected_index: int,
                         available: Sequence[bool]) -> torch.Tensor:
    """Fraction of scorable bank members agreeing with the selection per pixel.

    This is explicitly *same-bank agreement*, not reproducibility or repeated-run
    stability. Nothing here was executed twice, re-seeded or re-sampled, so it
    only says how much dissent this particular bounded search produced. Agreement
    alone is capped below full validity; a measured regional margin must supply
    the remaining evidence.
    """
    selected = bank[selected_index]
    usable = [index for index, ok in enumerate(available) if ok]
    if not usable:
        return torch.zeros_like(selected.labels, dtype=torch.float32)
    agreement = torch.zeros_like(selected.labels, dtype=torch.float32)
    for index in usable:
        agreement += (bank[index].labels == selected.labels).float()
    return agreement / float(len(usable))


def _validate_audit_inputs(
    bank: Sequence[Hypothesis],
    fitting_view: FittingView,
    selection_view: ScoringView,
    model: ObservationModel,
    *,
    rounds: int,
    improvement_threshold: float,
) -> None:
    """Validate one bank using the unchanged scalar audit guards."""
    if not isinstance(bank, Sequence) or not bank:
        raise AuditError("bank must be a non-empty sequence of Hypothesis")
    if any(not isinstance(candidate, Hypothesis) for candidate in bank):
        raise AuditError("bank entries must be contracts.Hypothesis")
    if not isinstance(fitting_view, FittingView):
        raise AuditError("audit_bank requires a contracts.FittingView")
    if not isinstance(selection_view, ScoringView):
        raise AuditError("audit_bank requires a contracts.ScoringView")
    if selection_view.role != "select":
        raise AuditError(
            "audit_bank accepts role='select' only; verification observations are sealed "
            "until every prediction and checkpoint is frozen"
        )
    if not isinstance(model, ObservationModel):
        raise AuditError("audit_bank requires the W1 ObservationModel")
    if not (improvement_threshold > 0.0):
        raise AuditError("improvement_threshold must be strictly positive")
    if len(bank) > MAX_BANK_CANDIDATES:
        raise AuditError(
            f"bank has {len(bank)} candidates; maximum is {MAX_BANK_CANDIDATES}"
        )
    if rounds < 1 or rounds > MAX_ROUNDS:
        raise AuditError(f"rounds must be in [1, {MAX_ROUNDS}]")
    if model.max_iterations > MAX_FIT_ITERATIONS:
        raise AuditError(
            f"model.max_iterations={model.max_iterations} exceeds the audit cap of "
            f"{MAX_FIT_ITERATIONS}"
        )
    fitting_view.validate()
    selection_view.validate()
    if fitting_view.unit_id != selection_view.unit_id:
        raise AuditError("fitting and selection views describe different units")
    if fitting_view.partition_id != selection_view.partition_id:
        raise AuditError("fitting and selection views come from different observation partitions")


def _prepare_regional_scores(
    fitted: Sequence[FittedHypothesis | None],
    fitting_view: FittingView,
    selection_view: ScoringView,
    model: ObservationModel,
    *,
    physical_work: dict[str, int],
) -> list[Any | None] | None:
    """Prepare each fit once when W1 exposes the immutable regional API."""
    prepare = getattr(model, "prepare_score", None)
    score_region = getattr(model, "score_region", None)
    if not callable(prepare) or not callable(score_region):
        return None
    snapshots: list[Any | None] = []
    for entry in fitted:
        if entry is None:
            snapshots.append(None)
            continue
        snapshots.append(prepare(entry, selection_view))
        physical_work["prepare_score_calls"] = physical_work.get("prepare_score_calls", 0) + 1
    return snapshots


def _finish_audit(
    bank: Sequence[Hypothesis],
    fitting_view: FittingView,
    selection_view: ScoringView,
    model: ObservationModel,
    *,
    rounds: int,
    improvement_threshold: float,
    fitted: Sequence[FittedHypothesis | None],
    scores: Sequence[EvidenceScore | None],
    order: Sequence[dict[str, Any]],
    physical_work: dict[str, int] | None = None,
    prepared: Sequence[Any | None] | None = None,
) -> AuditResult:
    """Apply the scalar selection/validity/trace decisions to evaluated rows.

    Both :func:`audit_bank` and :func:`audit_banks` call this function.  Keeping
    this decision path single-sourced is what makes a bulk likelihood pass
    unable to change candidate ordering, threshold boundaries, regional guards,
    labels, or validity semantics.
    """
    if len(fitted) != len(bank) or len(scores) != len(bank):
        raise AuditError("evaluated fits and scores must align with the bank")
    physical_work = dict(physical_work or {})
    totals: list[float | None] = [
        score.total if score is not None and score.available else None for score in scores
    ]
    available = [total is not None for total in totals]
    incumbent_total = totals[0]

    selected_index = 0
    improvement = 0.0
    rejected: list[dict[str, Any]] = []
    if incumbent_total is None:
        acceptance_reason = "incumbent has no available selection score; no edit is comparable"
    else:
        best_index, best_gain = 0, 0.0
        for index in range(1, len(bank)):
            total = totals[index]
            if total is None:
                rejected.append(
                    {
                        "candidate_id": bank[index].candidate_id,
                        "reason": (scores[index].reason if scores[index] is not None
                                   else "not evaluated"),
                    }
                )
                continue
            gain = float(incumbent_total) - float(total)
            # Preserve the original closed threshold and first-candidate tie
            # behavior exactly: equality at the threshold is admissible, but a
            # gain equal to the current best does not replace it.
            if gain >= improvement_threshold and gain > best_gain:
                best_index, best_gain = index, gain
            elif index != 0:
                rejected.append(
                    {
                        "candidate_id": bank[index].candidate_id,
                        "reason": "improvement below the preregistered threshold",
                        "gain_nats_per_pixel": gain,
                    }
                )
        selected_index, improvement = best_index, best_gain
        acceptance_reason = (
            "strict improvement over the incumbent on selection observations"
            if selected_index != 0
            else "no challenger reached the preregistered strict improvement; incumbent retained"
        )

    regional_work = dict(physical_work)
    selected = bank[selected_index]
    margin, search_inconclusive, margin_notes = _regional_margin(
        bank, selected_index, fitted, model, selection_view,
        prepared=prepared,
        physical_work=regional_work,
    )
    same_bank_agreement = _same_bank_agreement(bank, selected_index, available)

    if search_inconclusive:
        # An unchallenged incumbent earns no margin evidence. Same-bank agreement
        # remains a bounded diagnostic and cannot buy full validity on its own.
        margin_weight = torch.zeros_like(margin)
    else:
        margin_weight = 1.0 - torch.exp(-margin.clamp_min(0.0) / MARGIN_SCALE)
    same_bank_weight = (AGREEMENT_WEIGHT_CAP * same_bank_agreement).clamp(0.0, 1.0)
    evidence_weight = 1.0 - (1.0 - same_bank_weight) * (1.0 - margin_weight)
    validity = (selected.validity * evidence_weight).clamp(0.0, 1.0)

    class_counts = {
        name: int((selected.labels == role).sum().item())
        for role, name in ((0, "BG"), (1, "RV"), (2, "MYO"), (3, "LV"))
    }
    valid_class_counts = {
        name: float(((selected.labels == role).float() * validity).sum().item())
        for role, name in ((0, "BG"), (1, "RV"), (2, "MYO"), (3, "LV"))
    }
    fits_performed = sum(1 for entry in order if not entry["cached"])
    regional_score_calls = int(margin_notes.get("regional_score_calls", 0))
    # Regional physical counters are per-unit and deliberately separate from
    # the trainer's shared batch API counters.
    regional_work.setdefault("regional_prepare_calls", regional_work.get("prepare_score_calls", 0))
    regional_work.setdefault("regional_score_region_calls", regional_work.get("score_region_calls", 0))
    regional_work.setdefault("regional_scalar_score_calls", regional_work.get("scalar_region_score_calls", 0))
    trace: dict[str, Any] = {
        "contract_version": VERSION,
        "study_id": fitting_view.study_id,
        "unit_id": fitting_view.unit_id,
        "partition_id": fitting_view.partition_id,
        "protocol": fitting_view.protocol,
        "bank_id": selected.metadata.get("bank_id"),
        "bank_size": len(bank),
        "rounds_requested": rounds,
        "round_slots": _round_slots(len(bank), rounds),
        "fits_performed": fits_performed,
        "primary_score_calls": fits_performed,
        "score_calls_total": fits_performed + regional_score_calls,
        "regional_score_calls": regional_score_calls,
        "regional_score_observation_count": int(
            margin_notes.get("regional_score_observation_count", 0)
        ),
        "regional_score_counts": list(margin_notes.get("regional_score_counts", [])),
        "regional_refits": int(margin_notes.get("regional_refits", 0)),
        "evaluation_order": list(order),
        "budget": {
            "max_fits": len(bank),
            "fitting_iterations_per_candidate": model.max_iterations,
            "capacity": model.capacity,
            "improvement_threshold_nats_per_pixel": improvement_threshold,
            "margin_scale_nats_per_pixel": MARGIN_SCALE,
        },
        "accepted": selected_index != 0,
        "selected_candidate_id": selected.candidate_id,
        "selected_index": selected_index,
        "improvement_nats_per_pixel": improvement,
        "acceptance_reason": acceptance_reason,
        "rejected_edits": rejected,
        "rejection_is_edit_scoped": True,
        "learning_sample_retained": True,
        "search_inconclusive": search_inconclusive,
        "search_notes": margin_notes,
        "distinctness": _distinctness(bank),
        "selection_observations_consumed": int(selection_view.support.sum().item()),
        "adaptive_information_use": "selection observations were read to rank candidates",
        "semantic_unresolved": bool(selected.semantic_unresolved),
        "semantic_alternatives": len(selected.alternatives),
        "class_pixel_counts": class_counts,
        "valid_class_weight": valid_class_counts,
        "class_absent": [name for name, count in class_counts.items() if count == 0],
        "coverage_mean_validity": float(validity.mean().item()),
        "coverage_fraction_above_zero": float((validity > 0).float().mean().item()),
        "same_bank_agreement": {
            "mean": float(same_bank_agreement.mean().item()),
            "min": float(same_bank_agreement.min().item()),
            "max": float(same_bank_agreement.max().item()),
            "validity_weight_cap": AGREEMENT_WEIGHT_CAP,
            "interpretation": (
                "agreement within this single bank; it is not repeated-run stability"
            ),
        },
        "stability_diagnostic": "same_bank_agreement",
        "interpretation": (
            "scores are predictive NLL in nats per valid observed pixel; margins are score "
            "differences, not calibrated correctness probabilities"
        ),
        "physical_work": regional_work,
    }
    return AuditResult(
        initial=bank[0],
        selected=selected,
        bank=list(bank),
        fitted=[entry for entry in fitted if entry is not None],
        scores=[score for score in scores if score is not None],
        regional_margin=margin,
        validity=validity,
        trace=trace,
    )


def audit_bank(
    bank: Sequence[Hypothesis],
    fitting_view: FittingView,
    selection_view: ScoringView,
    model: ObservationModel,
    *,
    rounds: int = DEFAULT_ROUNDS,
    improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
) -> AuditResult:
    """Fit, score and select over a frozen bank using selection observations only.

    Parameters
    ----------
    bank:
        The frozen candidate bank; ``bank[0]`` is the incumbent. Selectors
        compared downstream must receive this identical object.
    fitting_view:
        The only data any nuisance fit may see.
    selection_view:
        Must carry ``role='select'``. A verification view is refused: sealed
        observations may not touch any trained or selected state.
    model:
        The W1 observation model. All candidates are fitted at the same capacity
        and iteration budget, which the model enforces on its own.

    Returns
    -------
    AuditResult
        Selection, full bank, fits, scores, regional margin, pseudo-label
        validity and a trace of budget, costs, distinctness and acceptance.
    """
    _validate_audit_inputs(
        bank, fitting_view, selection_view, model,
        rounds=rounds, improvement_threshold=improvement_threshold,
    )

    slots = _round_slots(len(bank), rounds)
    fitted: list[FittedHypothesis | None] = [None] * len(bank)
    scores: list[EvidenceScore | None] = [None] * len(bank)
    order: list[dict[str, Any]] = []

    def evaluate(index: int, round_index: int) -> None:
        if fitted[index] is not None:
            order.append({"candidate_index": index, "round": round_index, "cached": True})
            return
        # Streamed: one candidate is fitted, scored and released before the next
        # is touched, so no bank-wide graph or shared nuisance state exists.
        candidate_fit = model.fit(bank[index], fitting_view)
        candidate_score = model.score(candidate_fit, selection_view)
        fitted[index] = candidate_fit
        scores[index] = candidate_score
        order.append(
            {
                "candidate_index": index,
                "candidate_id": bank[index].candidate_id,
                "round": round_index,
                "cached": False,
                "fitting_steps": candidate_fit.fitting_steps,
                "capacity": candidate_fit.capacity,
                "fit_score": _score_dict(candidate_fit.fit_score),
                "select_score": _score_dict(candidate_score),
            }
        )

    evaluate(0, 0)
    for round_index, group in enumerate(slots, start=1):
        for index in group:
            # A rejection in an earlier round never stops a later one: every
            # slot in the preregistered budget is spent.
            evaluate(index, round_index)

    physical_work = {
        "mode": "scalar",
        "fit_calls": len([entry for entry in order if not entry["cached"]]),
        "fit_items": len([entry for entry in order if not entry["cached"]]),
        "primary_score_calls": len([entry for entry in order if not entry["cached"]]),
        "primary_score_items": len([entry for entry in order if not entry["cached"]]),
    }
    # The scalar public method remains the independent reference path.  It
    # deliberately uses ``ObservationModel.score`` for regional reductions;
    # only the canonical bulk entry point opts into prepared snapshots.
    prepared = None
    return _finish_audit(
        bank, fitting_view, selection_view, model,
        rounds=rounds, improvement_threshold=improvement_threshold,
        fitted=fitted, scores=scores, order=order,
        physical_work=physical_work, prepared=prepared,
    )


def audit_banks(
    banks: Sequence[Sequence[Hypothesis]],
    fitting_views: Sequence[FittingView],
    selection_views: Sequence[ScoringView],
    model: ObservationModel,
    *,
    rounds: int = DEFAULT_ROUNDS,
    improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
) -> list[AuditResult]:
    """Audit a batch of units with one ordered ``fit_many``/``score_many`` pass.

    Candidate banks are still generated independently per unit and retain their
    original slot/round order.  Only the expensive primary likelihood calls are
    flattened across units; :func:`_finish_audit` then applies the exact scalar
    threshold, tie, regional, validity and trace decisions to each unit.  If a
    custom observation double does not expose both bulk methods, this function
    deliberately falls back to :func:`audit_bank` for compatibility.

    The returned traces keep logical per-unit score counts unchanged.  Their
    ``physical_work`` contains only regional prepare/reduction calls with
    ``scope='unit'``; shared fit_many/score_many API calls are owned by the
    trainer's batch record and are never duplicated across unit traces.
    """
    if not isinstance(banks, Sequence):
        raise AuditError("banks must be a sequence of candidate banks")
    if not isinstance(fitting_views, Sequence) or not isinstance(selection_views, Sequence):
        raise AuditError("fitting_views and selection_views must be sequences")
    if len(banks) != len(fitting_views) or len(banks) != len(selection_views):
        raise AuditError("banks, fitting_views and selection_views must have equal lengths")
    if not banks:
        return []

    for bank, fitting_view, selection_view in zip(banks, fitting_views, selection_views):
        _validate_audit_inputs(
            bank, fitting_view, selection_view, model,
            rounds=rounds, improvement_threshold=improvement_threshold,
        )

    # Subclasses or instance monkey-patches may override scalar fit/score or
    # consume RNG.  They must remain on the scalar reference path even if they
    # inherit optional bulk methods from ObservationModel.
    def canonical_method(name: str) -> bool:
        bound = getattr(model, name, None)
        base = getattr(ObservationModel, name, None)
        return callable(bound) and getattr(bound, "__func__", bound) is base

    canonical_model = type(model) is ObservationModel and all(
        canonical_method(name)
        for name in (
            "fit", "score", "fit_many", "score_many", "prepare_score", "score_region"
        )
    )
    fit_many = getattr(model, "fit_many", None)
    score_many = getattr(model, "score_many", None)
    if not canonical_model or not callable(fit_many) or not callable(score_many):
        # A custom observation model may intentionally only implement the
        # scalar contract; preserving the scalar callback path is mandatory.
        return [
            audit_bank(
                bank, fitting_view, selection_view, model,
                rounds=rounds, improvement_threshold=improvement_threshold,
            )
            for bank, fitting_view, selection_view in zip(banks, fitting_views, selection_views)
        ]

    flat_hypotheses: list[Hypothesis] = []
    flat_fitting: list[FittingView] = []
    flat_selection: list[ScoringView] = []
    spans: list[tuple[int, int]] = []
    for bank, fitting_view, selection_view in zip(banks, fitting_views, selection_views):
        start = len(flat_hypotheses)
        flat_hypotheses.extend(bank)
        flat_fitting.extend([fitting_view] * len(bank))
        flat_selection.extend([selection_view] * len(bank))
        spans.append((start, len(flat_hypotheses)))

    # One ordered bulk call for the full BxK candidate set.  The observation
    # implementation may group compatible rows internally, but it remains the
    # authority for actual kernel/evaluation counts; this layer records only
    # the API invocation and item cardinality.
    fitted_flat = list(fit_many(flat_hypotheses, flat_fitting))
    if len(fitted_flat) != len(flat_hypotheses):
        raise AuditError(
            "ObservationModel.fit_many returned a result count different from the flattened bank"
        )
    scored_flat = list(score_many(fitted_flat, flat_selection))
    if len(scored_flat) != len(fitted_flat):
        raise AuditError(
            "ObservationModel.score_many returned a result count different from fit_many"
        )
    if any(not isinstance(entry, FittedHypothesis) for entry in fitted_flat):
        raise AuditError("ObservationModel.fit_many returned a non-FittedHypothesis entry")
    if any(not isinstance(entry, EvidenceScore) for entry in scored_flat):
        raise AuditError("ObservationModel.score_many returned a non-EvidenceScore entry")

    results: list[AuditResult] = []
    for unit_index, (bank, fitting_view, selection_view) in enumerate(
        zip(banks, fitting_views, selection_views)
    ):
        start, end = spans[unit_index]
        fitted_unit = fitted_flat[start:end]
        scores_unit = scored_flat[start:end]
        slots = _round_slots(len(bank), rounds)
        order: list[dict[str, Any]] = []
        # Reconstruct the exact scalar evaluation order even though all rows
        # were evaluated in a flattened call above.
        ordered_indices = [0]
        for group in slots:
            ordered_indices.extend(group)
        for index in ordered_indices:
            candidate_fit = fitted_unit[index]
            candidate_score = scores_unit[index]
            round_index = 0 if index == 0 else next(
                (round_no for round_no, group in enumerate(slots, start=1) if index in group),
                1,
            )
            order.append(
                {
                    "candidate_index": index,
                    "candidate_id": bank[index].candidate_id,
                    "round": round_index,
                    "cached": False,
                    "fitting_steps": candidate_fit.fitting_steps,
                    "capacity": candidate_fit.capacity,
                    "fit_score": _score_dict(candidate_fit.fit_score),
                    "select_score": _score_dict(candidate_score),
                }
            )
        # Shared fit_many/score_many invocation counts belong to the trainer's
        # batch record.  The per-unit trace carries only regional physical work
        # so it can be safely aggregated over units.
        regional_work: dict[str, int | str] = {"mode": "bulk", "scope": "unit"}
        prepared = _prepare_regional_scores(
            fitted_unit, fitting_view, selection_view, model,
            physical_work=regional_work,  # type: ignore[arg-type]
        )
        result = _finish_audit(
            bank, fitting_view, selection_view, model,
            rounds=rounds, improvement_threshold=improvement_threshold,
            fitted=fitted_unit, scores=scores_unit, order=order,
            physical_work=regional_work, prepared=prepared,
        )
        results.append(result)
    return results


__all__ = [
    "AGREEMENT_WEIGHT_CAP",
    "DEFAULT_IMPROVEMENT_THRESHOLD",
    "DEFAULT_ROUNDS",
    "MAX_BANK_CANDIDATES",
    "MAX_FIT_ITERATIONS",
    "MAX_REGION_SCORE_CALLS",
    "MAX_ROUNDS",
    "MAX_SCORED_REGIONS",
    "AuditError",
    "audit_bank",
    "audit_banks",
]
