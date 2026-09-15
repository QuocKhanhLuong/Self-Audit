"""Post-freeze image-only verification of a frozen candidate bank (W6).

This is the last step that is still allowed to look at observations, and it is
the only step that may touch ``O_verify``. It runs *after* every compared
prediction and checkpoint is frozen and hashed, and it is structurally incapable
of changing anything upstream:

* it never calls :meth:`ObservationModel.fit`;
* it takes already-fitted candidates and only calls :meth:`ObservationModel.score`;
* it returns a receipt, not a decision, and exposes no tuning entry point;
* a second verification of the same freeze is refused unless the caller
  explicitly opts in, and then the reuse is recorded, because unlimited
  verification reuse is exactly how a "held-out" set stops being held out.

Reported quantities are image-only predictive numbers. None of them is accuracy
and none of them is a probability of correctness.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..contracts import VERSION, EvidenceScore, FittedHypothesis, ScoringView
from .freeze import FreezeValidationError, validate_freeze_manifest


#: ``freeze + study + unit + partition`` keys already verified in this process.
#: A freeze contains many units, so guarding only the freeze id would reject a
#: valid second unit and would fail to prevent a repeated verification of one
#: unit.  Counts are retained to make deliberate repeats explicit.
_VERIFIED_FREEZES: dict[str, int] = {}

VERIFICATION_LEDGER_SCHEMA = "maskfree150.verification-ledger.v1"


class VerificationReuseError(RuntimeError):
    """Raised when a frozen set is verified more than once without an explicit opt-in."""


def _verification_key(freeze_id: str, verify_view: ScoringView) -> str:
    """Stable key for one sealed unit and its observation partition."""
    return json.dumps(
        {
            "freeze_id": str(freeze_id),
            "study_id": str(verify_view.study_id),
            "unit_id": str(verify_view.unit_id),
            "partition_id": str(verify_view.partition_id),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _read_ledger(path: Path) -> dict[str, dict[str, Any]]:
    """Read a durable verification ledger, accepting the v1 mapping shape."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise VerificationReuseError(f"verification ledger is unreadable: {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise VerificationReuseError(f"verification ledger must be an object: {path}")
    raw = payload.get("verifications", payload)
    if not isinstance(raw, Mapping):
        raise VerificationReuseError(f"verification ledger verifications must be an object: {path}")
    result: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping):
            count = value.get("count", 0)
            entry = dict(value)
        else:
            count = value
            entry = {}
        if isinstance(count, bool) or not isinstance(count, (int, float)) or int(count) < 0:
            raise VerificationReuseError(f"verification ledger has invalid count for {key!r}")
        entry["count"] = int(count)
        result[str(key)] = entry
    return result


def _write_ledger(path: Path, entries: Mapping[str, Mapping[str, Any]]) -> None:
    """Atomically persist counts and completed result payloads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": VERIFICATION_LEDGER_SCHEMA,
        "verifications": {str(key): dict(value) for key, value in entries.items()},
    }
    from ..runtime import atomic_write_json
    atomic_write_json(path, payload)


def _ledger_result(path: Path, key: str) -> dict[str, Any] | None:
    """Return a previously completed result for resume, if one is present."""
    entry = _read_ledger(path).get(key)
    if not entry:
        return None
    result = entry.get("result")
    return dict(result) if isinstance(result, Mapping) else None


def _score_entry(score: EvidenceScore) -> dict[str, Any]:
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


def _perturbed_views(view: ScoringView, *, seed: int) -> list[tuple[str, ScoringView, str]]:
    """Deterministic perturbations of the verification observations.

    Two families, both applied to the *observations* and never to the model:

    ``noise_sigma_*``
        additive Gaussian noise inside the verification support, testing how
        brittle the predictive score is to intensity nuisance.
    ``dropout_*``
        a fraction of verification pixels removed from the support (and zeroed,
        preserving the ScoringView invariant), testing how much the score
        depends on which withheld pixels happened to be drawn.
    """
    generator = torch.Generator().manual_seed(seed)
    variants: list[tuple[str, ScoringView, str]] = []
    for sigma in (0.05, 0.10):
        noise = torch.randn(view.image.shape, generator=generator, dtype=view.image.dtype)
        image = view.image + noise * sigma
        image = image * view.support.unsqueeze(0)
        variants.append(
            (
                f"noise_sigma_{sigma:g}",
                replace(view, image=image),
                "additive Gaussian intensity noise on verification observations",
            )
        )
    for fraction in (0.25, 0.50):
        uniform = torch.rand(view.support.shape, generator=generator)
        keep = view.support & (uniform >= fraction)
        if not bool(keep.any()):
            continue
        image = view.image * keep.unsqueeze(0)
        variants.append(
            (
                f"dropout_{fraction:g}",
                replace(view, image=image, support=keep),
                "random removal of verification pixels from the scored support",
            )
        )
    return variants


def _annotation_stability(candidates: Sequence[FittedHypothesis], ranking: Sequence[int]) -> dict[str, Any]:
    """Pixel agreement between the verification-best draft and its runner-up.

    High agreement means the verification ranking is not choosing between
    materially different annotations; low agreement means the choice matters.
    Neither direction is a correctness statement.
    """
    if len(ranking) < 2:
        return {
            "available": False,
            "reason": "fewer than two scored candidates; stability is undefined",
            "agreement": None,
            "compared": [],
        }
    best = candidates[ranking[0]].hypothesis.labels
    runner_up = candidates[ranking[1]].hypothesis.labels
    agreement = float((best == runner_up).to(torch.float64).mean().item())
    return {
        "available": True,
        "reason": None,
        "agreement": agreement,
        "compared": [
            candidates[ranking[0]].hypothesis.candidate_id,
            candidates[ranking[1]].hypothesis.candidate_id,
        ],
        "note": "pixel agreement between the two best verification candidates, not accuracy",
    }


def _stable_validity_mask(validity: torch.Tensor, count: int) -> torch.Tensor:
    """Keep ``count`` highest frozen-validity pixels with a flat-index tie break."""
    flat = validity.detach().reshape(-1).to(torch.float64)
    order = sorted(range(flat.numel()), key=lambda index: (-float(flat[index]), index))
    keep = torch.zeros(flat.numel(), dtype=torch.bool, device=validity.device)
    if count > 0:
        keep[torch.as_tensor(order[:count], dtype=torch.long, device=validity.device)] = True
    return keep.reshape(validity.shape)


def _student_view(view: ScoringView, validity: torch.Tensor, support: torch.Tensor) -> ScoringView:
    """Restrict a sealed view to a frozen student validity map."""
    keep = support & (validity.detach() > 0)
    return replace(view, image=view.image * keep.unsqueeze(0), support=keep)


def verify_student_coverage(
    fitted_by_arm: Mapping[str, FittedHypothesis],
    verify_view: ScoringView,
    model: Any,
) -> dict[str, Any]:
    """Score frozen student fits at natural and equal supported coverage.

    ``fitted_by_arm`` must contain pre-freeze fits.  The helper only derives
    deterministic support subsets from their frozen validity maps and calls the
    model's score method; it never fits, tunes, calibrates, or reads a reference.
    This makes a matched-coverage student comparison a real predictive NLL
    comparison rather than a coverage-count assertion.
    """
    if not isinstance(verify_view, ScoringView) or verify_view.role != "verify":
        raise ValueError("student coverage verification requires role='verify'")
    arms = {str(name): fit for name, fit in fitted_by_arm.items()}
    if not arms:
        return {
            "available": False,
            "reason": "no pre-freeze student fits were supplied",
            "natural": {},
            "matched": {},
            "matched_supported_pixels": 0,
        }
    for name, fit in arms.items():
        if not isinstance(fit, FittedHypothesis):
            raise TypeError(f"student arm {name!r} is not a FittedHypothesis")

    natural: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for name, fit in arms.items():
        validity = fit.hypothesis.validity.detach()
        if validity.shape != verify_view.support.shape:
            natural[name] = {
                "score": None,
                "supported_pixels": 0,
                "available": False,
                "reason": "student validity shape does not match verify support",
            }
            counts[name] = 0
            continue
        support = verify_view.support & (validity > 0)
        counts[name] = int(support.sum().item())
        score = model.score(fit, _student_view(verify_view, validity, verify_view.support))
        natural[name] = {
            "score": _score_entry(score),
            "supported_pixels": counts[name],
            "available": bool(score.available and score.total is not None),
            "reason": None if score.available and score.total is not None else score.reason,
        }

    matched_count = min(counts.values()) if counts else 0
    matched: dict[str, dict[str, Any]] = {}
    for name, fit in arms.items():
        validity = fit.hypothesis.validity.detach()
        if validity.shape != verify_view.support.shape:
            matched[name] = {
                "score": None,
                "supported_pixels": 0,
                "available": False,
                "reason": "student validity shape does not match verify support",
            }
            continue
        ranked_validity = torch.where(
            verify_view.support,
            validity,
            torch.full_like(validity, float("-inf")),
        )
        keep = _stable_validity_mask(ranked_validity, int(matched_count))
        # The support mask is applied before scoring and is independent of the
        # verification intensities.  It is therefore a frozen coverage choice.
        score = model.score(fit, replace(verify_view, image=verify_view.image * keep.unsqueeze(0),
                                         support=keep))
        matched[name] = {
            "score": _score_entry(score),
            "supported_pixels": int(keep.sum().item()),
            "available": bool(score.available and score.total is not None),
            "reason": None if score.available and score.total is not None else score.reason,
        }

    return {
        "available": bool(matched and matched_count > 0 and
                           all(item["available"] for item in matched.values())),
        "reason": None if matched_count > 0 else "no positive frozen validity support is shared",
        "natural": natural,
        "matched": matched,
        "supported_pixels_by_arm": counts,
        "matched_supported_pixels": int(matched_count),
        "note": (
            "matched predictive NLL uses equal-count support selected from frozen validity; "
            "it is not a reference quality metric"
        ),
    }


def verify_frozen_bank(
    fitted: FittedHypothesis | Sequence[FittedHypothesis] | Mapping[str, FittedHypothesis],
    verify_view: ScoringView,
    model: Any,
    freeze_manifest: Mapping[str, Any],
    *,
    selection_scores: Mapping[str, float] | None = None,
    degenerate_fitted: Sequence[FittedHypothesis] | None = None,
    control_fitted: Mapping[str, FittedHypothesis] | None = None,
    method_to_candidate: Mapping[str, str] | None = None,
    selected_candidate_id: str | None = None,
    student_fitted_by_arm: Mapping[str, FittedHypothesis] | None = None,
    seed: int = 42,
    allow_repeat: bool = False,
    resume_cached: bool = False,
    ledger_path: str | Path | None = None,
    freeze_root: str | None = None,
    required_methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score a frozen bank on ``O_verify`` and emit a one-time verification receipt.

    Parameters
    ----------
    fitted:
        One or more already-fitted candidates. They are consumed read-only; no
        fit is performed or repeated here.
    verify_view:
        A ``ScoringView`` with ``role='verify'``. Any other role is refused: the
        selection observations have their own, adaptive, path.
    model:
        The observation model used for the original fits. Only ``score`` is
        called on it.
    freeze_manifest:
        The manifest produced at finalization. Every listed file and checkpoint
        is re-hashed before a single verification number is computed.
    selection_scores:
        Optional map of candidate id to the selection-role total that the
        auditor already computed, used for the selection-to-verification gap.
        When absent, the gap falls back to
        ``FittedHypothesis.metadata['selection_score']`` and is otherwise
        reported unavailable rather than recomputed here.
    degenerate_fitted:
        Degenerate challengers (all-background, random masks, class permutation,
        excessive partitioning) that were **fitted before the freeze** and frozen
        with every other compared method. Verification only scores them. They
        cannot be manufactured here: fitting after a freeze is not permitted, and
        the observation model refuses to score a partition swapped into another
        candidate's parameters. Without them the degenerate challenge is reported
        as unavailable, never as "no degenerate won".
    control_fitted:
        Mapping of control method name to its exact pre-freeze fit. These fits
        are merged with ``fitted`` and scored once per unique candidate, so E0,
        random-edit, degenerate, and student methods cannot become orphaned
        post-freeze.
    method_to_candidate:
        Optional method-to-candidate mapping used to report a verification score
        for every frozen comparison, including selectors sharing one bank fit.
    selected_candidate_id:
        The candidate selected before the freeze. Any verification ranking is
        diagnostic; this id drives perturbation and degenerate comparisons.
    student_fitted_by_arm:
        Optional mapping of student arm name to its pre-freeze fit. When present,
        natural and equal-coverage predictive scores are returned under
        ``student_coverage``.
    resume_cached:
        Return a completed result stored in ``ledger_path`` without rescoring the
        sealed observations. This is the only permitted resume path.
    ledger_path:
        Optional durable JSON ledger for one verification key. Results are
        written atomically after scoring so a resumed run can reuse them.

    Returns a dictionary with per-candidate verification scores, the
    selection-to-verification gap, perturbation sensitivity, annotation
    stability, degenerate-challenge outcomes, and the freeze receipt.
    """
    method_names_from_fitted: dict[str, str] = {}
    if isinstance(fitted, Mapping):
        candidates = list(fitted.values())
        method_names_from_fitted = {
            str(name): fit.hypothesis.candidate_id
            for name, fit in fitted.items()
            if isinstance(fit, FittedHypothesis)
        }
    elif isinstance(fitted, FittedHypothesis):
        candidates = [fitted]
    else:
        candidates = list(fitted)
    if degenerate_fitted:
        # Degenerate challengers are ordinary frozen fits for verification.  Add
        # them to the same candidate pool so each candidate is scored exactly
        # once, then derive the falsification comparison from that score.
        candidates.extend(degenerate_fitted)
    if control_fitted:
        candidates.extend(control_fitted.values())
    if not candidates:
        raise ValueError("verify_frozen_bank needs at least one fitted candidate")
    if not isinstance(verify_view, ScoringView):
        raise TypeError("verify_frozen_bank requires a contracts.ScoringView")
    if verify_view.role != "verify":
        raise ValueError(
            f"verification requires role='verify'; refusing a {verify_view.role!r} view"
        )

    receipt = validate_freeze_manifest(
        freeze_manifest, root=freeze_root, required_methods=required_methods
    )
    freeze_id = receipt["freeze_id"] or "unnamed-freeze"
    verification_key = _verification_key(freeze_id, verify_view)
    durable_ledger = _read_ledger(Path(ledger_path)) if ledger_path is not None else {}
    durable_entry = durable_ledger.get(verification_key, {})
    durable_count = int(durable_entry.get("count", 0)) if durable_entry else 0
    reuse_count = max(_VERIFIED_FREEZES.get(verification_key, 0), durable_count)
    if resume_cached and isinstance(durable_entry.get("result"), Mapping):
        cached = dict(durable_entry["result"])
        cached["resumed_cached"] = True
        cached["freeze_verification_key"] = verification_key
        cached["verification_ledger_path"] = str(ledger_path) if ledger_path is not None else None
        return cached
    if reuse_count and not allow_repeat:
        raise VerificationReuseError(
            f"freeze/unit/partition {verification_key!r} has already been verified "
            f"{reuse_count} time(s); repeated verification turns a sealed set into an "
            "adaptively reused one. Pass allow_repeat=True only to record a deliberate, "
            "disclosed reuse."
        )
    next_count = reuse_count + 1

    # De-duplicate shared bank fits.  Multiple frozen methods intentionally
    # point at one candidate (E1/E2/E3/E4/E5 often do), and scoring the same fit
    # repeatedly would inflate verification work and obscure method provenance.
    unique_candidates: list[FittedHypothesis] = []
    candidate_index: dict[str, int] = {}
    for candidate in candidates:
        if not isinstance(candidate, FittedHypothesis):
            raise TypeError("fitted entries must be contracts.FittedHypothesis")
        candidate_id = candidate.hypothesis.candidate_id
        if candidate_id not in candidate_index:
            candidate_index[candidate_id] = len(unique_candidates)
            unique_candidates.append(candidate)
    candidates = unique_candidates

    method_map: dict[str, str] = dict(method_names_from_fitted)
    if method_to_candidate is not None:
        method_map.update({str(name): str(candidate_id)
                           for name, candidate_id in method_to_candidate.items()})
    if control_fitted:
        method_map.update({str(name): fit.hypothesis.candidate_id
                           for name, fit in control_fitted.items()})
    for candidate in candidates:
        method_map.setdefault(candidate.hypothesis.candidate_id, candidate.hypothesis.candidate_id)

    scored: list[dict[str, Any]] = []
    totals: list[tuple[int, float]] = []
    for index, candidate in enumerate(candidates):
        score = model.score(candidate, verify_view)
        entry = {
            "candidate_id": candidate.hypothesis.candidate_id,
            "source": candidate.hypothesis.source,
            "semantic_unresolved": bool(candidate.hypothesis.semantic_unresolved),
            "fit_score": _score_entry(candidate.fit_score),
            "verify_score": _score_entry(score),
        }
        selection_total: float | None = None
        if selection_scores is not None:
            raw = selection_scores.get(candidate.hypothesis.candidate_id)
            selection_total = None if raw is None else float(raw)
        if selection_total is None:
            raw = candidate.metadata.get("selection_score")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                selection_total = float(raw)
        if selection_total is None or score.total is None:
            entry["selection_to_verification_gap"] = None
            entry["gap_reason"] = (
                "no recorded selection-role total for this candidate"
                if selection_total is None
                else "verification score unavailable"
            )
        else:
            entry["selection_to_verification_gap"] = float(score.total) - selection_total
            entry["selection_total"] = selection_total
            entry["gap_reason"] = None
        scored.append(entry)
        if score.available and score.total is not None:
            totals.append((index, float(score.total)))

    ranking = [index for index, _ in sorted(totals, key=lambda item: item[1])]
    best_index = ranking[0] if ranking else None
    preselected_index = (
        candidate_index.get(str(selected_candidate_id))
        if selected_candidate_id is not None
        else None
    )
    if selected_candidate_id is not None and preselected_index is None:
        raise ValueError(
            f"selected_candidate_id {selected_candidate_id!r} has no supplied frozen fit"
        )
    comparison_index = preselected_index if preselected_index is not None else best_index
    comparison_reason = (
        "preselected E5 candidate"
        if preselected_index is not None
        else "verification-best diagnostic fallback; caller did not provide selected_candidate_id"
    )

    perturbations: list[dict[str, Any]] = []
    if comparison_index is None:
        perturbation_reason: str | None = "no candidate produced an available verification score"
    else:
        perturbation_reason = None
        baseline = scored[comparison_index]["verify_score"]["total"]
        for name, variant, description in _perturbed_views(verify_view, seed=seed):
            variant_score = model.score(candidates[comparison_index], variant)
            perturbations.append(
                {
                    "perturbation": name,
                    "description": description,
                    "score": _score_entry(variant_score),
                    "delta_vs_unperturbed": (
                        None
                        if variant_score.total is None or baseline is None
                        else float(variant_score.total) - float(baseline)
                    ),
                }
            )

    # ``control_fitted`` is the preferred complete source.  Keep the legacy
    # ``degenerate_fitted`` sequence for callers that still pass only the four
    # degenerate controls.
    degenerate_candidates: list[FittedHypothesis] = []
    seen_degenerate: set[str] = set()
    if degenerate_fitted:
        for candidate in degenerate_fitted:
            if candidate.hypothesis.candidate_id not in seen_degenerate:
                seen_degenerate.add(candidate.hypothesis.candidate_id)
                degenerate_candidates.append(candidate)
    if control_fitted:
        for name, candidate in control_fitted.items():
            if name in {
                "control_all_background",
                "control_random_spatial_masks",
                "control_class_permutation",
                "control_excessive_partition",
            } or candidate.hypothesis.metadata.get("degenerate"):
                if candidate.hypothesis.candidate_id not in seen_degenerate:
                    seen_degenerate.add(candidate.hypothesis.candidate_id)
                    degenerate_candidates.append(candidate)

    degenerate: list[dict[str, Any]] = []
    degenerate_reason: str | None = None
    if not degenerate_candidates:
        degenerate_reason = (
            "no pre-freeze degenerate fits were supplied; they cannot be produced here because "
            "fitting after a freeze is not permitted and the observation model refuses a "
            "partition swapped into another candidate's frozen parameters"
        )
    elif comparison_index is None:
        degenerate_reason = "preselected candidate has no available verification score to compare against"
    else:
        baseline = scored[comparison_index]["verify_score"]["total"]
        for challenger in degenerate_candidates:
            if not isinstance(challenger, FittedHypothesis):
                raise TypeError("degenerate_fitted entries must be contracts.FittedHypothesis")
            challenger_index = candidate_index.get(challenger.hypothesis.candidate_id)
            if challenger_index is None:  # defensive; all challengers are merged above
                raise ValueError(
                    f"degenerate candidate {challenger.hypothesis.candidate_id!r} was not "
                    "included in the frozen candidate pool"
                )
            challenger_score = scored[challenger_index]["verify_score"]
            beats = (
                challenger_score["total"] is not None
                and baseline is not None
                and float(challenger_score["total"]) < float(baseline)
            )
            degenerate.append(
                {
                    "candidate_id": challenger.hypothesis.candidate_id,
                    "degenerate": challenger.hypothesis.metadata.get("degenerate"),
                    "score": challenger_score,
                    "beats_selected": bool(beats),
                    "method": "scored a pre-freeze fit on sealed observations; no refit",
                }
            )

    degenerate_wins = [entry["candidate_id"] for entry in degenerate if entry["beats_selected"]]

    method_scores: dict[str, dict[str, Any]] = {}
    for method, candidate_id in sorted(method_map.items()):
        index = candidate_index.get(candidate_id)
        if index is None:
            method_scores[method] = {
                "candidate_id": candidate_id,
                "verify_score": None,
                "available": False,
                "reason": "method has no supplied pre-freeze fitted hypothesis",
            }
        else:
            verify_score = scored[index]["verify_score"]
            method_scores[method] = {
                "candidate_id": candidate_id,
                "verify_score": verify_score,
                "available": bool(verify_score.get("available") and verify_score.get("total") is not None),
                "reason": None if verify_score.get("available") else verify_score.get("reason"),
                "selection_to_verification_gap": scored[index].get("selection_to_verification_gap"),
            }

    student_coverage = None
    if student_fitted_by_arm:
        student_coverage = verify_student_coverage(student_fitted_by_arm, verify_view, model)

    result = {
        "contract_version": VERSION,
        "role": verify_view.role,
        "study_id": verify_view.study_id,
        "unit_id": verify_view.unit_id,
        "partition_id": verify_view.partition_id,
        "verified_at": time.time(),
        "freeze_receipt": receipt,
        "freeze_reuse_count": reuse_count,
        "freeze_verification_key": verification_key,
        "verification_ledger_path": str(ledger_path) if ledger_path is not None else None,
        "repeat_verification": bool(reuse_count),
        "candidates": scored,
        "methods": method_scores,
        "method_to_candidate": method_map,
        "verification_ranking": [candidates[index].hypothesis.candidate_id for index in ranking],
        "verification_ranking_is_diagnostic": True,
        "selected_by_verification": (
            None if best_index is None else candidates[best_index].hypothesis.candidate_id
        ),
        "preselected_candidate_id": selected_candidate_id,
        "comparison_baseline": comparison_reason,
        "perturbation_sensitivity": perturbations,
        "perturbation_reason": perturbation_reason,
        "bank_agreement": _annotation_stability(candidates, ranking),
        "annotation_stability": {
            "available": False,
            "reason": "no independent repeated annotation experiment; bank agreement is reported separately",
            "agreement": None,
        },
        "degenerate_challenges": degenerate,
        "degenerate_challenge_reason": degenerate_reason,
        "degenerate_wins": degenerate_wins,
        "falsification_flag": bool(degenerate_wins),
        "refit_performed": False,
        "note": (
            "image-only predictive verification. Lower normalized NLL means the partition "
            "predicted withheld intensities better under the frozen constrained model; it is "
            "not accuracy and not a probability of correctness."
        ),
    }

    if student_coverage is not None:
        result["student_coverage"] = student_coverage

    # Update in-memory and durable reuse ledgers only after every score and
    # diagnostic has completed.  The complete result is written in the same
    # atomic ledger update, allowing an interruption to resume without opening
    # O_verify a second time.
    if ledger_path is not None:
        durable_ledger[verification_key] = {
            "count": next_count,
            "freeze_id": str(freeze_id),
            "study_id": str(verify_view.study_id),
            "unit_id": str(verify_view.unit_id),
            "partition_id": str(verify_view.partition_id),
            "completed_at": time.time(),
            "result": result,
        }
        _write_ledger(Path(ledger_path), durable_ledger)
    # Record consumption only after the durable result (when requested) has
    # been atomically persisted.  A failed write must remain resumable.
    _VERIFIED_FREEZES[verification_key] = next_count
    return result


def verification_registry_state() -> dict[str, int]:
    """Read-only snapshot of which freezes have been verified in this process."""
    return dict(_VERIFIED_FREEZES)


def reset_verification_registry() -> None:
    """Clear the in-process reuse guard.

    Test-support only. Calling this in a pipeline would erase the evidence that a
    sealed set had already been consumed, which is precisely the abuse the guard
    exists to make visible.
    """
    _VERIFIED_FREEZES.clear()


__all__ = [
    "FreezeValidationError",
    "VerificationReuseError",
    "reset_verification_registry",
    "verify_student_coverage",
    "verification_registry_state",
    "verify_frozen_bank",
]
