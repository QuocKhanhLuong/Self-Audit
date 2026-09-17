"""End-to-end locked Self-Audit annotation and transition-audit model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .annotation_expert import (
    _freeze_tensor,
    AnnotationExpert,
    ExpertReplayRecord,
    RECORD_KIND_CANDIDATE_C,
    RECORD_KIND_ORDINARY,
    RECORD_KIND_ROLLBACK,
    StaleReplayRecordError,
)
from .annotation_head import InitialAnnotationHead
from .auditor import CounterfactualAuditor
from .dynamic_window import normalized_pixel_step
from .encoder import ConvNeXtTinyEncoder, build_encoder
from .fpn import LightweightFPN


#: Local audit channel semantics, mirroring ``self_audit.audit.targets``.
LOCAL_FIX: int = 0
LOCAL_UNCHANGED: int = 1
LOCAL_REGRESS: int = 2

#: Window/ablation modes.  ``current`` is the shipped baseline and is exactly
#: the pre-Candidate-C computation.
WINDOW_MODES: tuple[str, ...] = (
    "current",
    "feature_only",
    "free_offsets",
    "candidate_c",
    "candidate_c_no_fix",
    "direct_rollback",
)

#: Modes that consume a bounded accepted-transition record.
RECORD_CONSUMING_MODES: frozenset[str] = frozenset(
    {"candidate_c", "candidate_c_no_fix", "direct_rollback"}
)
SOLVER_MODES: frozenset[str] = frozenset({"candidate_c", "candidate_c_no_fix"})

#: A factual winning margin at or below this is reported as a tie.  It is a
#: DIAGNOSTIC threshold only: a tied predicted FIX pixel is protected exactly
#: like any other, because ``argmax`` gives it a deterministic predicted class
#: and that class must survive.  What a tie does change is which constraint
#: binds -- its required margin is zero, so the explicit class-identity check
#: is the one doing the work there.
MARGIN_TIE_EPSILON: float = 1e-6

#: Hard ceiling on candidate checks per solver invocation, enforced by
#: ``CandidateCConfig`` itself so no entry point can build an unbounded loop.
MAX_CANDIDATE_CHECKS: int = 2


@dataclass(frozen=True)
class CandidateCConfig:
    """Runtime settings for the Candidate C restitution solver.

    There is deliberately no ``enabled`` / ``enforce_fix`` / ``detach_innovation``
    field: ``window_mode`` alone selects activation (``candidate_c``), the FIX
    constraint (``candidate_c`` vs ``candidate_c_no_fix``) and the no-solver
    control (``direct_rollback``).  Differentiating through the solver is out of
    scope, so the restitution innovation and its gate are always detached.
    """

    rho_feature_pixels: float = 1.0
    lam: float = 1.0
    lr: float | None = None
    fix_threshold: float = 0.5
    regress_threshold: float = 0.5
    margin_fraction: float = 0.5
    min_regress_mass: float = 1e-3
    replay_atol: float = 1e-5
    replay_rtol: float = 1e-5
    max_backtracks: int = 2

    def __post_init__(self) -> None:
        for name in (
            "rho_feature_pixels",
            "lam",
            "fix_threshold",
            "regress_threshold",
            "margin_fraction",
            "min_regress_mass",
            "replay_atol",
            "replay_rtol",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"candidate_c.{name} must be a real number, got {value!r}")
            value = float(value)
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"candidate_c.{name} must be finite, got {value!r}")
            if value < 0.0:
                raise ValueError(f"candidate_c.{name} must be non-negative, got {value!r}")
            object.__setattr__(self, name, value)
        # A non-positive trust region is not "no movement": it makes the
        # normalized proposal meaningless, so it is rejected outright.
        if self.rho_feature_pixels <= 0.0:
            raise ValueError(
                f"candidate_c.rho_feature_pixels must be strictly positive, got {self.rho_feature_pixels}"
            )
        for name in ("fix_threshold", "regress_threshold", "margin_fraction", "min_regress_mass"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"candidate_c.{name} is compared against a probability and must lie in [0, 1], got {value}"
                )
        if self.lr is not None:
            if not isinstance(self.lr, (int, float)) or isinstance(self.lr, bool):
                raise TypeError(f"candidate_c.lr must be a real number or None, got {self.lr!r}")
            lr = float(self.lr)
            if lr <= 0.0 or lr != lr or lr == float("inf"):
                raise ValueError(f"candidate_c.lr must be a positive finite float, got {self.lr!r}")
            object.__setattr__(self, "lr", lr)
        if isinstance(self.max_backtracks, bool) or not isinstance(self.max_backtracks, int):
            raise TypeError(f"candidate_c.max_backtracks must be an int, got {self.max_backtracks!r}")
        # Hard upper bound: the contract allows at most two candidate checks in
        # TOTAL (full projected proposal, then half step).  This is enforced at
        # every entry point, not only by the YAML schema, so no caller can
        # construct an unbounded solver loop.
        if not 0 <= self.max_backtracks <= MAX_CANDIDATE_CHECKS:
            raise ValueError(
                "candidate_c.max_backtracks bounds the TOTAL number of candidate checks and "
                f"must be an integer in 0..{MAX_CANDIDATE_CHECKS}, got {self.max_backtracks}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rho_feature_pixels": self.rho_feature_pixels,
            "lam": self.lam,
            "lr": self.lr,
            "fix_threshold": self.fix_threshold,
            "regress_threshold": self.regress_threshold,
            "margin_fraction": self.margin_fraction,
            "min_regress_mass": self.min_regress_mass,
            "replay_atol": self.replay_atol,
            "replay_rtol": self.replay_rtol,
            "max_backtracks": self.max_backtracks,
        }


_CANDIDATE_C_FIELDS: frozenset[str] = frozenset(CandidateCConfig().as_dict())


def resolve_candidate_c_config(value: Any) -> CandidateCConfig:
    """Accept ``None``, a mapping or a :class:`CandidateCConfig` strictly."""

    if value is None:
        return CandidateCConfig()
    if isinstance(value, CandidateCConfig):
        return value
    if isinstance(value, Mapping):
        unknown = sorted(set(map(str, value.keys())) - _CANDIDATE_C_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown candidate_c keys {unknown}; allowed keys are {sorted(_CANDIDATE_C_FIELDS)}"
            )
        return CandidateCConfig(**{str(k): v for k, v in value.items()})
    raise TypeError(f"candidate_c must be None, a mapping or CandidateCConfig, got {type(value)!r}")


def resolve_window_mode(value: str) -> str:
    mode = str(value).lower().strip()
    if mode not in WINDOW_MODES:
        raise ValueError(f"window_mode must be one of {list(WINDOW_MODES)}, got {value!r}")
    return mode


@dataclass
class _AcceptedOrdinaryRecord:
    """The single most-recent accepted ORDINARY transition, per active row.

    Rows whose most recent accepted action was produced by the Candidate C
    solver or by direct rollback are *removed* from this record rather than
    relabelled: their factual output is not an ordinary AnnotationExpert update
    of their immediate pre-state, so C1 does not apply to them.
    """

    indices: Tensor
    expert_record: ExpertReplayRecord
    audit_evidence: Tensor
    turn: int
    #: Which recorded rows had their ordinary transition ACCEPTED.  The record
    #: keeps every row of the factual forward so the replay can reproduce its
    #: exact batch composition; only accepted rows are replay-eligible.
    accepted: Tensor


def _runner_up(logits: Tensor, winner: Tensor) -> Tensor:
    """``max_{c != winner} logits[:, c]`` for a per-pixel winner index."""

    masked = logits.scatter(1, winner.unsqueeze(1), torch.finfo(logits.dtype).min)
    return masked.amax(dim=1)


def _winning_margin(logits: Tensor, winner: Tensor) -> Tensor:
    chosen = logits.gather(1, winner.unsqueeze(1)).squeeze(1)
    return chosen - _runner_up(logits, winner)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _materialize(value: Tensor) -> Tensor:
    """Copy into an ordinary detached tensor usable outside inference mode."""

    with torch.inference_mode(False), torch.no_grad():
        result = torch.empty_like(value)
        result.copy_(value)
    return result


def _project_supports(
    factual: Sequence[Tensor],
    step: Sequence[Tensor],
    trust_region: Tensor,
) -> tuple[Tensor, ...]:
    """Project ``factual + step`` onto the trust region and the sampling domain.

    The trust region is an axis-wise L-infinity box of ``rho`` feature pixels
    around the realized factual support; the domain is the normalized
    ``[-1, 1]`` sampling square.  Because the factual point already lies in the
    domain, clamping after the box projection can only move the result *toward*
    the factual point, so both constraints hold simultaneously.
    """

    bound = trust_region.view(1, 1, 1, 1, 2)
    return tuple(
        (base + torch.clamp(offset, min=-bound, max=bound)).clamp(-1.0, 1.0)
        for base, offset in zip(factual, step)
    )


def _halve_towards_factual(
    factual: Sequence[Tensor], proposal: Sequence[Tensor]
) -> tuple[Tensor, ...]:
    """The midpoint of the factual point and an already-projected proposal.

    Halving the *realized* displacement is the only honest backtrack: halving
    the pre-projection direction would leave every saturated component exactly
    where it was, so the second check would re-evaluate the first point.  The
    midpoint of two points that satisfy the box and the domain satisfies both.
    """

    return tuple(base + 0.5 * (point - base) for base, point in zip(factual, proposal))


def _displacement_pixels(
    candidate: Sequence[Tensor], factual: Sequence[Tensor], pixel_step: Tensor
) -> Tensor:
    """Per-row mean squared coordinate displacement in FEATURE-PIXEL units."""

    safe = torch.where(pixel_step > 0, pixel_step, torch.ones_like(pixel_step)).view(1, 1, 1, 1, 2)
    total: Tensor | None = None
    count = 0
    for new, base in zip(candidate, factual):
        scaled = ((new - base) / safe).pow(2).flatten(1).sum(1)
        total = scaled if total is None else total + scaled
        count += int(new[0].numel())
    if total is None:
        raise ValueError("no supports to measure")
    return total / float(max(count, 1))


def _restitution_objective(
    logits: Tensor,
    restoration_target: Tensor,
    weights: Tensor,
    candidate: Sequence[Tensor],
    factual: Sequence[Tensor],
    pixel_step: Tensor,
    lam: float,
) -> Tensor:
    """C2 objective: REGRESS-weighted restoration CE + lambda * displacement."""

    cross_entropy = torch.nn.functional.cross_entropy(logits, restoration_target, reduction="none")
    numerator = (weights * cross_entropy).flatten(1).sum(1)
    denominator = weights.flatten(1).sum(1).clamp_min(1e-8)
    displacement = _displacement_pixels(candidate, factual, pixel_step)
    return numerator / denominator + float(lam) * displacement


def _run_restitution_solver(
    expert: AnnotationExpert,
    record: _AcceptedOrdinaryRecord,
    current_state_rows: Tensor,
    *,
    config: CandidateCConfig,
    enforce_fix: bool,
    capture_geometry: bool = False,
) -> dict[str, Any]:
    """Solve C1-C2 and return the detached C3 innovation for each row.

    All work happens inside ``torch.inference_mode(False)`` with grad enabled so
    the solver still functions when the caller wrapped inference in
    ``no_grad``/``inference_mode``.  The replay is functional with detached
    parameters and buffers, so no ``requires_grad`` flag is mutated, no
    ``.grad`` buffer is touched, and the encoder, Auditor, annotation
    parameters and patient image receive no solver gradient.

    The record is replayed at its **exact original batch composition**: rows
    that have since halted are replayed too and their results discarded.
    Replaying a subset would change kernel tiling and reduction order and could
    break the strict C1 tolerance under AMP, which must never be worked around
    by loosening that tolerance.
    """

    expert_record = record.expert_record
    factual_logits = expert_record.factual_candidate_logits
    rows = int(factual_logits.shape[0])
    device = factual_logits.device
    dtype = factual_logits.dtype
    evidence = record.audit_evidence
    atol = float(config.replay_atol)
    rtol = float(config.replay_rtol)

    reason: list[str | None] = [None] * rows
    evals = {"factual_replay": 0, "coordinate_backward": 0, "candidate_checks": 0, "total_forward": 0}

    with torch.inference_mode(False), torch.enable_grad():
        retained = _materialize(current_state_rows).to(dtype=dtype)
        pixel_step = normalized_pixel_step(
            expert_record.feature_hw[0], expert_record.feature_hw[1], device=device, dtype=dtype
        )
        trust_region = float(config.rho_feature_pixels) * pixel_step
        factual_supports = tuple(item.detach() for item in expert_record.coordinates)
        variables = tuple(item.detach().clone().requires_grad_(True) for item in factual_supports)

        # --- C1: exact frozen factual-support replay ---------------------
        replayed = expert.replay(expert_record, variables, capture_geometry=capture_geometry)
        replay_logits = replayed.candidate_logits
        evals["factual_replay"] += 1
        evals["total_forward"] += 1

        with torch.no_grad():
            absolute = (replay_logits - factual_logits).abs()
            tolerance = atol + rtol * factual_logits.abs()
            c1_abs_error = absolute.flatten(1).amax(1)
            c1_rel_error = (
                absolute / factual_logits.abs().clamp_min(torch.finfo(dtype).tiny)
            ).flatten(1).amax(1)
            c1_passed = (absolute <= tolerance).flatten(1).all(1)
            # The stored factual output must still be the retained state, or
            # the innovation would be transferred onto the wrong annotation.
            stale = ~(
                (retained - factual_logits).abs() <= atol + rtol * factual_logits.abs()
            ).flatten(1).all(1)

            regress_raw = evidence[:, LOCAL_REGRESS]
            fix_raw = evidence[:, LOCAL_FIX]
            selected = regress_raw >= float(config.regress_threshold)
            weights = regress_raw * selected.to(dtype=dtype)
            regress_mass = weights.flatten(1).mean(1)
            has_regress = regress_mass >= float(config.min_regress_mass)

            # ``argmax`` over the factual logits is the predicted class: it is
            # a deterministic function of the stored factual output, defined
            # for tied pixels exactly as it is for any other (lowest winning
            # index).  EVERY thresholded predicted FIX pixel is therefore
            # protected -- a pixel is never dropped from protection because
            # its winning margin is zero or merely small, which would leave
            # precisely the most fragile predictions unguarded.
            winner = factual_logits.argmax(dim=1)
            factual_margin = _winning_margin(factual_logits, winner)
            protected = fix_raw >= float(config.fix_threshold)
            # Diagnostic only: how many protected pixels are tied or nearly
            # tied.  These pixels are protected like all the others; the count
            # exists so a run can report how much of the protected set sits at
            # the margin floor, never to exclude anything from protection.
            fix_tie = protected & (factual_margin <= MARGIN_TIE_EPSILON)
            # A tied pixel has ``factual_margin == 0`` and therefore a required
            # margin of zero; the class-identity constraint below is what
            # actually pins its prediction.
            required_margin = float(config.margin_fraction) * factual_margin
            restoration_target = expert_record.annotation_logits.argmax(dim=1)

            # Rows whose ordinary transition was REJECTED stay in the replay
            # only to preserve the factual batch composition; they are never
            # eligible.  This is stated explicitly rather than inferred from
            # "retained != factual", which would be a coincidence, not a rule.
            accepted_rows = record.accepted.to(device=device, dtype=torch.bool)
            eligible = accepted_rows & c1_passed & ~stale & has_regress
            for row in range(rows):
                if not bool(accepted_rows[row]):
                    reason[row] = "record_row_not_accepted"
                elif not bool(c1_passed[row]):
                    reason[row] = "replay_failure"
                elif bool(stale[row]):
                    reason[row] = "stale_record"
                elif not bool(has_regress[row]):
                    reason[row] = "no_regress_evidence"

        objective_before = _restitution_objective(
            replay_logits,
            restoration_target,
            weights,
            variables,
            factual_supports,
            pixel_step,
            config.lam,
        )
        with torch.no_grad():
            finite_objective = torch.isfinite(objective_before)
            for row in range(rows):
                if bool(eligible[row]) and not bool(finite_objective[row]):
                    reason[row] = "nonfinite_objective"
            eligible = eligible & finite_objective

        settled = torch.zeros(rows, dtype=torch.bool, device=device)
        best_logits = factual_logits.clone()
        factual_geometry = tuple(replayed.geometry)
        best_coordinates = [item.detach().clone() for item in factual_supports]
        best_preclamp = [item.detach().clone() for item in expert_record.coordinates_preclamp]
        best_attention = (
            [item["attention"].detach().clone() for item in factual_geometry]
            if capture_geometry and factual_geometry
            else None
        )
        # ``*_after`` describe the proposal that was actually CHECKED for this
        # row (the accepted one, else the last one evaluated).  They are null
        # only when no proposal was evaluated at all -- never a stand-in zero,
        # and never conflated with the separate question of what was selected.
        objective_after: list[float | None] = [None] * rows
        violation_after: list[float | None] = [None] * rows
        displacement_after: list[float | None] = [None] * rows
        checked_feasible: list[bool | None] = [None] * rows
        checked_improved: list[bool | None] = [None] * rows
        proposal_checks: list[int] = [0] * rows
        selected_displacement: list[float | None] = [None] * rows

        if bool(eligible.any()):
            # --- one gradient at the factual point -----------------------
            total = torch.where(
                eligible, objective_before, torch.zeros_like(objective_before)
            ).sum()
            gradients: list[Tensor | None]
            if torch.isfinite(total):
                # ``autograd.grad`` rather than ``backward``: no ``.grad``
                # buffer anywhere in the process is written or cleared.
                gradients = list(
                    torch.autograd.grad(total, variables, allow_unused=True)
                )
                evals["coordinate_backward"] += 1
            else:
                gradients = [None]

            with torch.no_grad():
                usable = eligible.clone()
                if any(gradient is None for gradient in gradients):
                    usable = torch.zeros_like(usable)
                    for row in range(rows):
                        if bool(eligible[row]):
                            reason[row] = "nonfinite_objective"
                    direction: tuple[Tensor, ...] = ()
                    magnitude = torch.zeros(rows, device=device, dtype=dtype)
                else:
                    # The proposal is built in FEATURE-PIXEL coordinates
                    # z = (Q - P) / pixel_step, whose gradient is
                    # grad_z = grad_Q * pixel_step by the chain rule.  A step
                    # normalized in z has a well-defined size in pixels on a
                    # non-square grid; normalizing in raw normalized-coordinate
                    # units instead makes every component saturate against the
                    # box, which would silently turn the half step into a
                    # repeat of the full step.
                    axis = pixel_step.view(1, 1, 1, 1, 2)
                    finite_gradient = torch.ones(rows, dtype=torch.bool, device=device)
                    magnitude = torch.zeros(rows, device=device, dtype=dtype)
                    pixel_gradients: list[Tensor] = []
                    for gradient in gradients:
                        finite_gradient &= torch.isfinite(gradient).flatten(1).all(1)
                        in_pixels = gradient * axis
                        pixel_gradients.append(in_pixels)
                        magnitude = torch.maximum(magnitude, in_pixels.abs().flatten(1).amax(1))
                    if config.lr is not None:
                        # Explicit step in feature pixels per unit pixel-gradient.
                        scale = torch.full_like(magnitude, float(config.lr))
                    else:
                        # Normalized one-pixel proposal: across every internal
                        # depth of one sample, the largest pixel-gradient
                        # component moves exactly rho feature pixels.
                        scale = float(config.rho_feature_pixels) / magnitude.clamp_min(
                            torch.finfo(dtype).tiny
                        )
                    direction = tuple(
                        -in_pixels * scale.view(-1, 1, 1, 1, 1) * axis
                        for in_pixels in pixel_gradients
                    )
                    for row in range(rows):
                        if bool(eligible[row]) and (
                            not bool(finite_gradient[row]) or float(magnitude[row]) <= 0.0
                        ):
                            reason[row] = (
                                "nonfinite_gradient"
                                if not bool(finite_gradient[row])
                                else "zero_gradient"
                            )
                    usable = eligible & finite_gradient & (magnitude > 0.0)

                if bool(usable.any()) and direction:
                    checks = min(int(config.max_backtracks), MAX_CANDIDATE_CHECKS)
                    projected = _project_supports(factual_supports, direction, trust_region)
                    proposals = [projected][:checks]
                    if checks > 1:
                        proposals.append(_halve_towards_factual(factual_supports, projected))
                    for proposal in proposals:
                        if bool((usable & ~settled).sum() == 0):
                            break
                        candidate_output = expert.replay(
                            expert_record, proposal, capture_geometry=capture_geometry
                        )
                        candidate_logits = candidate_output.candidate_logits
                        evals["candidate_checks"] += 1
                        evals["total_forward"] += 1
                        finite_candidate = torch.isfinite(candidate_logits).flatten(1).all(1)
                        objective_candidate = _restitution_objective(
                            candidate_logits,
                            restoration_target,
                            weights,
                            proposal,
                            factual_supports,
                            pixel_step,
                            config.lam,
                        )
                        improved = finite_candidate & torch.isfinite(objective_candidate)
                        improved &= objective_candidate < objective_before
                        candidate_margin = _winning_margin(candidate_logits, winner)
                        gap = (required_margin - candidate_margin).clamp_min(0.0) * protected.to(dtype=dtype)
                        worst = gap.flatten(1).amax(1)
                        # Two distinct requirements on the counterfactual, both
                        # over ALL protected pixels: the fractional margin
                        # floor, and the predicted class itself.  The margin
                        # test alone is vacuous where the required margin is
                        # zero (a tie), so the class-identity test is stated
                        # separately rather than inferred from the margin.
                        class_kept = (candidate_logits.argmax(dim=1) == winner) | ~protected
                        class_kept_row = class_kept.flatten(1).all(1)
                        feasible = (
                            torch.ones(rows, dtype=torch.bool, device=device)
                            if not enforce_fix
                            else (worst <= 0.0) & class_kept_row
                        )
                        accept = usable & ~settled & improved & feasible
                        moved = _displacement_pixels(proposal, factual_supports, pixel_step)
                        for row in range(rows):
                            if not bool(usable[row]) or bool(settled[row]):
                                continue
                            # Measured for every checked proposal, including the
                            # ones that fail: a failed feasibility check must
                            # expose its violation, not hide behind a null.
                            proposal_checks[row] += 1
                            checked_feasible[row] = bool(feasible[row])
                            checked_improved[row] = bool(improved[row])
                            violation_after[row] = float(worst[row])
                            objective_after[row] = (
                                float(objective_candidate[row])
                                if bool(torch.isfinite(objective_candidate[row]))
                                else None
                            )
                            displacement_after[row] = float(moved[row])
                        if bool(accept.any()):
                            picked = accept.view(-1, 1, 1, 1)
                            best_logits = torch.where(picked, candidate_logits, best_logits)
                            support_mask = accept.view(-1, 1, 1, 1, 1)
                            for depth in range(len(proposal)):
                                best_coordinates[depth] = torch.where(
                                    support_mask, proposal[depth], best_coordinates[depth]
                                )
                                best_preclamp[depth] = torch.where(
                                    support_mask, proposal[depth], best_preclamp[depth]
                                )
                                if best_attention is not None and candidate_output.geometry:
                                    best_attention[depth] = torch.where(
                                        support_mask,
                                        candidate_output.geometry[depth]["attention"],
                                        best_attention[depth],
                                    )
                            for row in range(rows):
                                if bool(accept[row]):
                                    selected_displacement[row] = float(moved[row])
                                    reason[row] = None
                            settled |= accept
                        for row in range(rows):
                            if bool(usable[row]) and not bool(settled[row]):
                                # Distinguish the two distinct failure modes
                                # instead of calling every rejection infeasible.
                                if enforce_fix and not bool(feasible[row]):
                                    reason[row] = "infeasible"
                                else:
                                    reason[row] = "no_improvement"

        with torch.no_grad():
            zero = torch.zeros_like(factual_logits)
            innovation = torch.where(settled.view(-1, 1, 1, 1), best_logits - factual_logits, zero)
            for row in range(rows):
                if not bool(settled[row]):
                    # Identity fallback: the factual support was selected, so
                    # the selected displacement is exactly zero, not unknown.
                    selected_displacement[row] = 0.0
            gate = torch.sigmoid(expert_record.factual_update_gate).detach()
            # Final C3 preservation check on the actual transferred candidate,
            # over the SAME full protected set as the counterfactual check:
            # neither floating arithmetic nor an approximate C1 may be allowed
            # to invalidate the convex-mixture preservation argument.  The
            # mixture is convex only between the factual output and the
            # candidate; the retained state is allowed to differ from the
            # factual output within the C1 tolerance, so a tied protected pixel
            # can still flip here even after a feasible counterfactual, and
            # this check is what catches it.
            projected_candidate = retained + gate * innovation
            preserved = (projected_candidate.argmax(dim=1) == winner) | ~protected
            preserved_row = preserved.flatten(1).all(1)
            broken = settled & ~preserved_row
            if enforce_fix and bool(broken.any()):
                innovation = torch.where(broken.view(-1, 1, 1, 1), zero, innovation)
                revert = broken.view(-1, 1, 1, 1, 1)
                for depth in range(len(best_coordinates)):
                    best_coordinates[depth] = torch.where(
                        revert, factual_supports[depth], best_coordinates[depth]
                    )
                    best_preclamp[depth] = torch.where(
                        revert, expert_record.coordinates_preclamp[depth], best_preclamp[depth]
                    )
                    if best_attention is not None and factual_geometry:
                        best_attention[depth] = torch.where(
                            revert, factual_geometry[depth]["attention"], best_attention[depth]
                        )
                for row in range(rows):
                    if bool(broken[row]):
                        selected_displacement[row] = 0.0
                        reason[row] = "c3_preservation_failed"
                settled = settled & ~broken

    return {
        "innovation": innovation.detach(),
        "gate": gate.detach(),
        "settled": settled,
        "eligible": eligible,
        "c1_passed": c1_passed,
        "c1_abs_error": c1_abs_error,
        "c1_rel_error": c1_rel_error,
        "regress_mass": regress_mass,
        "regress_mass_raw": regress_raw.flatten(1).mean(1),
        "fix_mass": fix_raw.flatten(1).mean(1),
        "num_protected": protected.flatten(1).sum(1),
        # Retained at a constant zero for backward diagnostic compatibility:
        # no thresholded predicted FIX pixel is excluded from protection any
        # more, so the field it used to carry is by construction empty.
        "num_ties_excluded": torch.zeros_like(protected.flatten(1).sum(1)),
        "num_fix_ties": fix_tie.flatten(1).sum(1),
        "objective_before": objective_before.detach(),
        "objective_after": objective_after,
        "violation_after": violation_after,
        "displacement_after": displacement_after,
        "selected_displacement": selected_displacement,
        "checked_feasible": checked_feasible,
        "checked_improved": checked_improved,
        "proposal_checks": proposal_checks,
        "stale": stale,
        "reason": reason,
        "evals": evals,
        "geometry": (
            _pack_geometry(
                expert_record,
                factual_geometry,
                best_coordinates,
                best_preclamp,
                best_attention,
            )
            if capture_geometry
            else None
        ),
    }


def _pack_geometry(
    expert_record: ExpertReplayRecord,
    factual_geometry: Sequence[dict[str, Any]],
    chosen_coordinates: Sequence[Tensor],
    chosen_preclamp: Sequence[Tensor],
    chosen_attention: Sequence[Tensor] | None,
) -> dict[str, Any]:
    """Batched factual/chosen realized supports, one entry per internal depth."""

    factual: list[dict[str, Any]] = []
    chosen: list[dict[str, Any]] = []
    for depth in range(expert_record.depth):
        captured = factual_geometry[depth] if depth < len(factual_geometry) else {}
        # Report the identity the conditioning embeddings ACTUALLY used.  The
        # raw request can exceed the embedding table and be clamped -- at
        # turn_index=2 the third internal depth requests 4 but the table only
        # reaches 3 -- so the raw value would misdescribe the computation.
        # The effective values are read back from the capture on the record.
        requested = int(expert_record.iteration_index) + depth
        effective_turn = (
            expert_record.effective_turn_indices[depth]
            if depth < len(expert_record.effective_turn_indices)
            else int(expert_record.turn_index)
        )
        effective_iteration = (
            expert_record.effective_iteration_indices[depth]
            if depth < len(expert_record.effective_iteration_indices)
            else requested
        )
        common = {
            "depth_index": depth,
            "turn_index": int(effective_turn),
            "iteration_index": int(effective_iteration),
            "requested_iteration_index": requested,
            "state_identity": expert_record.state_identity,
        }
        factual.append(
            {
                **common,
                "coordinates": expert_record.coordinates[depth],
                "coordinates_preclamp": expert_record.coordinates_preclamp[depth],
                "attention": captured.get("attention"),
            }
        )
        chosen.append(
            {
                **common,
                "coordinates": chosen_coordinates[depth],
                "coordinates_preclamp": chosen_preclamp[depth],
                "attention": None if chosen_attention is None else chosen_attention[depth],
            }
        )
    return {
        "feature_hw": expert_record.feature_hw,
        "k": int(expert_record.coordinates[0].shape[-2]),
        "factual": tuple(factual),
        "chosen": tuple(chosen),
    }


def _row_geometry(packed: dict[str, Any] | None, offset: int) -> dict[str, Any] | None:
    """Slice one batched geometry payload down to a single sample row."""

    if packed is None:
        return None

    def _slice(entries: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for entry in entries:
            item = dict(entry)
            for key in ("coordinates", "coordinates_preclamp", "attention"):
                value = item.get(key)
                item[key] = None if value is None else value[offset : offset + 1]
            rows.append(item)
        return tuple(rows)

    return {
        "feature_hw": packed["feature_hw"],
        "k": packed["k"],
        "factual": _slice(packed["factual"]),
        "chosen": _slice(packed["chosen"]),
    }


def _run_direct_rollback(
    record: _AcceptedOrdinaryRecord,
    current_state_rows: Tensor,
    *,
    config: CandidateCConfig,
) -> dict[str, Any]:
    """Matched control: no solver, no replay, same gate and same audit gate.

    At predicted previous-REGRESS positions the retained logits are mixed
    toward the pre-transition logits.  The *threshold mask* form is used (not
    the raw probability weight) so the control acts on exactly the same
    thresholded REGRESS set as C2, making the comparison about coordinate
    restitution versus direct rollback rather than about the weighting.
    """

    expert_record = record.expert_record
    factual_logits = expert_record.factual_candidate_logits
    dtype = factual_logits.dtype
    rows = int(factual_logits.shape[0])
    device = factual_logits.device
    atol = float(config.replay_atol)
    rtol = float(config.replay_rtol)
    evidence = record.audit_evidence

    with torch.no_grad():
        retained = _materialize(current_state_rows).to(dtype=dtype)
        stale = ~(
            (retained - factual_logits).abs() <= atol + rtol * factual_logits.abs()
        ).flatten(1).all(1)
        regress_raw = evidence[:, LOCAL_REGRESS]
        selected = (regress_raw >= float(config.regress_threshold)).to(dtype=dtype)
        regress_mass = (regress_raw * selected).flatten(1).mean(1)
        has_regress = regress_mass >= float(config.min_regress_mass)
        accepted_rows = record.accepted.to(device=device, dtype=torch.bool)
        settled = accepted_rows & ~stale & has_regress
        gate = torch.sigmoid(expert_record.factual_update_gate).detach()
        target = expert_record.annotation_logits - factual_logits
        innovation = torch.where(
            settled.view(-1, 1, 1, 1),
            selected.unsqueeze(1) * target,
            torch.zeros_like(factual_logits),
        )
        reason: list[str | None] = [None] * rows
        for row in range(rows):
            if not bool(accepted_rows[row]):
                reason[row] = "record_row_not_accepted"
            elif bool(stale[row]):
                reason[row] = "stale_record"
            elif not bool(has_regress[row]):
                reason[row] = "no_regress_evidence"

    return {
        "innovation": innovation,
        "gate": gate,
        "settled": settled,
        "eligible": accepted_rows & ~stale,
        "stale": stale,
        "c1_passed": [None] * rows,
        "c1_abs_error": [None] * rows,
        "c1_rel_error": [None] * rows,
        "regress_mass": regress_mass,
        "regress_mass_raw": regress_raw.flatten(1).mean(1),
        "fix_mass": evidence[:, LOCAL_FIX].flatten(1).mean(1),
        "num_protected": [None] * rows,
        "num_ties_excluded": [None] * rows,
        "num_fix_ties": [None] * rows,
        "objective_before": [None] * rows,
        "objective_after": [None] * rows,
        "violation_after": [None] * rows,
        "displacement_after": [None] * rows,
        # Direct rollback runs no solver, so nothing is proposed or checked and
        # the selected support is always the factual one.
        "selected_displacement": [0.0] * rows,
        "checked_feasible": [None] * rows,
        "checked_improved": [None] * rows,
        "proposal_checks": [0] * rows,
        "reason": reason,
        "evals": {"factual_replay": 0, "coordinate_backward": 0, "candidate_checks": 0, "total_forward": 0},
        "geometry": None,
    }


class SelfAuditNet(nn.Module):
    """2.5-D ConvNeXt/FPN annotation model with thresholded self-audit."""

    def __init__(
        self,
        *,
        encoder_name: str = "convnext_tiny",
        pretrained_encoder: bool = False,
        encoder_allow_fallback: bool = True,
        shared_channels: int = 96,
        num_classes: int = 4,
        window_k: int = 8,
        max_turns: int = 3,
        window_mode: str = "current",
        candidate_c: Any = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.max_turns = int(max_turns)
        self.window_mode = resolve_window_mode(window_mode)
        self.candidate_c = resolve_candidate_c_config(candidate_c)
        self.encoder: ConvNeXtTinyEncoder = build_encoder(
            name=encoder_name,
            pretrained=pretrained_encoder,
            in_channels=3,
            allow_fallback=encoder_allow_fallback,
        )
        self.fpn = LightweightFPN(self.encoder.out_channels, out_channels=int(shared_channels))
        self.initial_head = InitialAnnotationHead(int(shared_channels), self.num_classes)
        self.annotation_expert = AnnotationExpert(
            feature_channels=int(shared_channels),
            num_classes=self.num_classes,
            window_k=int(window_k),
            max_turns=self.max_turns,
            audit_conditioning="feature_only" if self.window_mode == "feature_only" else "full",
            offset_mode="free" if self.window_mode == "free_offsets" else "structured",
        )
        self.auditor = CounterfactualAuditor(feature_channels=int(shared_channels), num_classes=self.num_classes)

    @property
    def uses_transition_records(self) -> bool:
        return self.window_mode in RECORD_CONSUMING_MODES

    def invalidate_candidate_c_records(self) -> int:
        """Fail every outstanding replay record closed after a weight update.

        Records are already scoped to a single :meth:`infer` call and never
        survive it; this remains available so a trainer can make the guarantee
        explicit around an ``optimizer.step()``.
        """

        return self.annotation_expert.invalidate_replay_records()

    def encode(self, images: Tensor) -> dict[str, Any]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Self-Audit input must be [B,3,H,W], got {tuple(images.shape)}")
        features = self.encoder(images)
        shared = self.fpn(features)
        return {"features": features, "shared": shared}

    def forward_annotation(self, images: Tensor, *, turns: int | None = None) -> dict[str, Any]:
        """Phase-A path: produce soft intermediate states without audit decisions."""

        encoded = self.encode(images)
        shared = encoded["shared"]
        initial = self.initial_head(shared, output_size=images.shape[-2:])
        state = initial
        audit_evidence = None
        states: list[Tensor] = []
        expert_outputs = []
        for turn in range(min(int(turns if turns is not None else self.max_turns), self.max_turns)):
            expert_output = self.annotation_expert(
                shared,
                state,
                previous_audit_evidence=audit_evidence,
                turn_index=turn,
                iteration_index=turn,
                return_metadata=False,
            )
            state = expert_output.candidate_logits
            states.append(state)
            expert_outputs.append(expert_output)
        return {
            "initial_logits": initial,
            "a0_logits": initial,
            "A0_logits": initial,
            "logits": state,
            "A_t": state,
            # ``states`` remains refinement-only for compatibility.  The
            # explicit trace is the source of truth for adjacent transitions.
            "state_trace": [initial] + states,
            "all_states": [initial] + states,
            "states": states,
            "refinement_logits": states,
            "expert_outputs": expert_outputs,
            "shared_features": shared,
            "features": encoded["features"],
        }

    def infer(
        self,
        images: Tensor,
        *,
        mode: str = "self_audit",
        tau_accept: float = 0.0,
        threshold: float | None = None,
        t_max: int | None = None,
        oracle_target: Tensor | None = None,
        capture_geometry: bool = False,
    ) -> dict[str, Any]:
        """Run initial-only, always-accept, or threshold-controlled inference.

        ``capture_geometry`` additionally emits ``candidate_c_geometry``: raw
        factual/chosen realized supports and attention, one element per
        ``candidate_c_diagnostics`` row and in the same order.  It is a debug
        payload; the ordinary rollout output stays JSON-serializable.
        """

        modes = {"initial_only", "always_accept_refinement", "self_audit", "oracle_accept"}
        if mode not in modes:
            raise ValueError(f"mode must be one of {sorted(modes)}, got {mode!r}")
        if threshold is not None:
            tau_accept = float(threshold)
        cap = self.max_turns if t_max is None else int(t_max)
        if cap < 0:
            raise ValueError("t_max must be non-negative")
        if mode == "oracle_accept" and oracle_target is None:
            raise ValueError("oracle_accept is analysis-only and requires oracle_target explicitly")
        encoded = self.encode(images)
        shared = encoded["shared"]
        initial = self.initial_head(shared, output_size=images.shape[-2:])
        state = initial
        previous_audit = None
        audits: list[dict[str, Tensor]] = []
        candidates: list[Tensor] = []
        accepted_turns: list[int] = []
        halted_turns: list[int] = []
        transition_previous: list[Tensor] = []
        transition_candidates: list[Tensor] = []
        batch_size = int(images.shape[0])
        active = torch.ones(batch_size, dtype=torch.bool, device=images.device)
        accepted_count = torch.zeros(batch_size, dtype=torch.long, device=images.device)
        halt_turn = torch.full((batch_size,), -1, dtype=torch.long, device=images.device)
        num_attempted_turns = torch.zeros(batch_size, dtype=torch.long, device=images.device)
        transition_active_masks: list[Tensor] = []
        transition_state_masks: list[Tensor] = []
        # Candidate C bookkeeping.  The record lives for exactly this call and
        # is never a buffer, never serialized and never shared across batches.
        use_records = self.uses_transition_records
        record: _AcceptedOrdinaryRecord | None = None
        previous_kind: dict[int, str] = {}
        candidate_c_diagnostics: list[dict[str, Any]] = []
        candidate_c_geometry: list[dict[str, Any]] = []
        if mode != "initial_only":
            for turn in range(cap):
                if not bool(active.any()):
                    break

                # Run the recurrent expert and Auditor only for rows that are
                # still active.  This keeps a halted row from receiving a new
                # candidate or audit evidence in a mixed batch.
                attempted = active.clone()
                active_indices = attempted.nonzero(as_tuple=False).flatten()
                transition_active_masks.append(attempted)
                num_attempted_turns = num_attempted_turns + attempted.to(torch.long)

                shared_active = shared.index_select(0, active_indices)
                state_active = state.index_select(0, active_indices)
                previous_audit_active = (
                    None
                    if previous_audit is None
                    else previous_audit.index_select(0, active_indices)
                )
                (
                    candidate_active,
                    ordinary_record,
                    ordinary_positions,
                    restitution_positions,
                    restitution_result,
                    turn_rows,
                    turn_geometry,
                ) = self._turn_candidates(
                    turn=turn,
                    active_indices=active_indices,
                    shared_active=shared_active,
                    state_active=state_active,
                    state_full=state,
                    previous_audit_active=previous_audit_active,
                    record=record,
                    previous_kind=previous_kind,
                    use_records=use_records,
                    capture_geometry=capture_geometry,
                )
                candidate_active_full = self._scatter_rows(state, active_indices, candidate_active)
                candidate = torch.where(
                    attempted.view(-1, 1, 1, 1),
                    candidate_active_full,
                    state,
                )
                transition_previous.append(state)
                transition_candidates.append(candidate)
                previous_probs = state_active.detach().softmax(dim=1)
                candidate_probs = candidate_active.detach().softmax(dim=1)
                entropy_previous = -(
                    previous_probs.clamp_min(1e-8)
                    * previous_probs.clamp_min(1e-8).log()
                ).sum(dim=1, keepdim=True)
                entropy_candidate = -(
                    candidate_probs.clamp_min(1e-8)
                    * candidate_probs.clamp_min(1e-8).log()
                ).sum(dim=1, keepdim=True)
                audit_output = self.auditor(
                    shared_active.detach(),
                    previous_probs.detach(),
                    candidate_probs.detach(),
                    (candidate_probs - previous_probs).detach(),
                    entropy_previous=entropy_previous.detach(),
                    entropy_candidate=entropy_candidate.detach(),
                )
                local_logits_active = self._audit_tensor(audit_output, "local_logits")
                delta_q_active = self._audit_tensor(audit_output, "delta_q")
                delta_q = delta_q_active.detach().reshape(-1)
                if mode == "always_accept_refinement":
                    accepted_active = torch.ones_like(delta_q, dtype=torch.bool)
                elif mode == "oracle_accept":
                    oracle_target_active = oracle_target.index_select(0, active_indices)
                    accepted_active = self._candidate_improves(
                        state_active,
                        candidate_active,
                        oracle_target_active,
                    )
                else:
                    accepted_active = delta_q > float(tau_accept)
                accepted = self._scatter_mask(active, active_indices, accepted_active)

                # The state update and the recurrent audit evidence are both
                # row-masked.  A rejected row keeps its previous state and no
                # local evidence is fed back into a later active row.
                state = torch.where(accepted[:, None, None, None], candidate, state)
                local_logits = self._scatter_rows_from_values(
                    batch_size,
                    active_indices,
                    local_logits_active,
                )
                local_evidence = local_logits.detach().softmax(dim=1)
                previous_audit = torch.where(
                    accepted[:, None, None, None],
                    local_evidence,
                    torch.zeros_like(local_evidence),
                )
                delta_q_full = self._scatter_rows_from_values(
                    batch_size,
                    active_indices,
                    audit_output.delta_q,
                )
                candidates.append(candidate)
                audits.append({
                    "local_logits": local_logits,
                    "delta_q": delta_q_full,
                    "accepted": accepted,
                    "active_mask": attempted,
                    "audit_mask": attempted,
                    "state_mask": accepted,
                })
                transition_state_masks.append(accepted)
                accepted_count = accepted_count + accepted.to(torch.long)
                rejected = attempted & ~accepted
                halt_now = rejected & (halt_turn < 0)
                halt_turn = torch.where(
                    halt_now,
                    torch.full_like(halt_turn, int(turn)),
                    halt_turn,
                )
                if bool(accepted.any()):
                    accepted_turns.append(turn)
                if bool(rejected.any()):
                    halted_turns.append(turn)
                if use_records:
                    for position, row in enumerate(turn_rows):
                        row["delta_q"] = float(delta_q[position])
                        row["accepted"] = bool(accepted_active[position])
                    candidate_c_diagnostics.extend(turn_rows)
                    candidate_c_geometry.extend(turn_geometry)
                    record = self._update_records(
                        turn=turn,
                        active_indices=active_indices,
                        accepted_active=accepted_active,
                        ordinary_record=ordinary_record,
                        ordinary_positions=ordinary_positions,
                        restitution_positions=restitution_positions,
                        restitution_result=restitution_result,
                        local_logits_active=local_logits_active,
                        previous_kind=previous_kind,
                    )
                if mode == "self_audit" or mode == "oracle_accept":
                    active = accepted
                # always_accept intentionally runs to the hard cap; the cap is
                # a safety limit, never a requirement for the self-audit path.
        transition_active_tensor = self._stack_masks(
            transition_active_masks,
            batch_size=batch_size,
            device=images.device,
        )
        transition_state_tensor = self._stack_masks(
            transition_state_masks,
            batch_size=batch_size,
            device=images.device,
        )
        return {
            "logits": state,
            "initial_logits": initial,
            "a0_logits": initial,
            "A0_logits": initial,
            "A_t": state,
            "states": candidates,
            "candidates": candidates,
            "transition_previous": transition_previous,
            "transition_candidates": transition_candidates,
            "audits": audits,
            "accepted_turns": accepted_turns,
            "halted_turns": halted_turns,
            "accepted_count": accepted_count,
            "halt_turn": halt_turn,
            "num_attempted_turns": num_attempted_turns,
            "active_mask": active,
            "final_active": active,
            # Per-turn lists retain the explicit [B] alignment contract.
            "active_masks": transition_active_masks,
            "transition_active_masks": transition_active_masks,
            "audit_masks": transition_active_masks,
            "transition_audit_masks": transition_active_masks,
            "state_masks": transition_state_masks,
            "transition_state_masks": transition_state_masks,
            # Stacked forms are convenience views for callers that want a
            # [T,B] tensor without changing the per-turn public contract.
            "active_mask_tensor": transition_active_tensor,
            "transition_active_mask_tensor": transition_active_tensor,
            "state_mask_tensor": transition_state_tensor,
            "transition_state_mask_tensor": transition_state_tensor,
            "shared_features": shared,
            "features": encoded["features"],
            "window_mode": self.window_mode,
            "candidate_c_diagnostics": candidate_c_diagnostics,
            **({"candidate_c_geometry": candidate_c_geometry} if capture_geometry else {}),
        }

    # ------------------------------------------------------------------
    # Candidate C turn plumbing
    # ------------------------------------------------------------------

    def _turn_candidates(
        self,
        *,
        turn: int,
        active_indices: Tensor,
        shared_active: Tensor,
        state_active: Tensor,
        state_full: Tensor,
        previous_audit_active: Tensor | None,
        record: _AcceptedOrdinaryRecord | None,
        previous_kind: dict[int, str],
        use_records: bool,
        capture_geometry: bool = False,
    ) -> tuple[
        Tensor, Any, Tensor, Tensor, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]
    ]:
        """Produce this turn's candidate for every active row.

        Active rows are partitioned into rows that carry an eligible accepted
        ORDINARY record and rows that do not.  The restitution partition is
        resolved FIRST, because a row whose C1 replay identity fails, whose
        retained state is stale, or whose record no longer matches the live
        model state must fall back to the **ordinary annotator** for this turn
        -- not to a zero-innovation identity candidate.  Those rows therefore
        join the ordinary partition before its single forward runs, so the
        replay record built from that forward preserves its exact factual batch
        composition.  Both partitions go on to the same official Auditor gate.
        """

        device = state_active.device
        count = int(active_indices.numel())
        if not use_records:
            expert_output = self.annotation_expert(
                shared_active,
                state_active,
                previous_audit_evidence=previous_audit_active,
                turn_index=turn,
                iteration_index=turn,
                return_metadata=False,
            )
            empty = torch.empty(0, dtype=torch.long, device=device)
            return expert_output.candidate_logits, None, empty, empty, None, [], []

        position_of = {int(value): index for index, value in enumerate(active_indices.tolist())}
        eligible_positions: list[int] = []
        eligible_offsets: list[int] = []
        if record is not None:
            for record_position, global_index in enumerate(record.indices.tolist()):
                if not bool(record.accepted[record_position]):
                    continue
                position = position_of.get(int(global_index))
                if position is not None:
                    eligible_positions.append(position)
                    eligible_offsets.append(record_position)

        rows: list[dict[str, Any]] = [{} for _ in range(count)]
        geometry_rows: list[dict[str, Any]] = [{} for _ in range(count)]
        restitution_result: dict[str, Any] | None = None
        replay_state_failure: str | None = None
        # One solver invocation per turn at most.  The id is stable within this
        # ``infer`` call so a consumer can deduplicate the per-group eval counts
        # that every row of the group reports.
        invocation = f"turn{int(turn)}" if eligible_positions else None
        path = "rollback" if self.window_mode == "direct_rollback" else "counterfactual"

        # --- phase 1: resolve the restitution partition ----------------------
        if eligible_positions and record is not None:
            # The record is replayed whole, at its original batch composition:
            # rows that have since halted or were rejected are replayed and
            # discarded rather than dropped, because a different batch shape
            # can change kernel tiling and break the strict C1 tolerance
            # under AMP.
            retained_group = state_full.index_select(0, record.indices.to(state_full.device))
            try:
                if self.window_mode == "direct_rollback":
                    restitution_result = _run_direct_rollback(
                        record, retained_group, config=self.candidate_c
                    )
                else:
                    restitution_result = _run_restitution_solver(
                        self.annotation_expert,
                        record,
                        retained_group,
                        config=self.candidate_c,
                        enforce_fix=self.window_mode == "candidate_c",
                        capture_geometry=capture_geometry,
                    )
            except StaleReplayRecordError as error:
                # Narrow: the record no longer describes the live model state.
                # Any other exception is a programmer error and must surface.
                restitution_result = None
                replay_state_failure = f"stale_record_state: {error}"

        fallback_positions: list[int] = []
        fallback_offsets: list[int] = []
        restitution_positions_list: list[int] = []
        restitution_offsets: list[int] = []
        for position, offset in zip(eligible_positions, eligible_offsets):
            if restitution_result is None:
                fallback_positions.append(position)
                fallback_offsets.append(offset)
                continue
            # Tri-state, and never ``is False``: for the solver this entry is a
            # 0-dim ``torch.bool`` tensor, and ``tensor(False) is False`` is
            # ALWAYS false, which silently left every numeric replay failure on
            # the identity path.  ``None`` means the mode performs no replay
            # (direct rollback), which is not a failure.
            replay_ok = restitution_result["c1_passed"][offset]
            replay_failed = replay_ok is not None and not bool(replay_ok)
            stale = bool(restitution_result["stale"][offset])
            if stale or replay_failed:
                fallback_positions.append(position)
                fallback_offsets.append(offset)
            else:
                restitution_positions_list.append(position)
                restitution_offsets.append(offset)

        restitution_positions = torch.tensor(
            restitution_positions_list, dtype=torch.long, device=device
        )
        used = set(restitution_positions_list)
        ordinary_list = [index for index in range(count) if index not in used]
        ordinary_positions = torch.tensor(ordinary_list, dtype=torch.long, device=device)

        candidate_active = torch.zeros_like(state_active)
        ordinary_record = None

        # --- phase 2: one ordinary forward over the combined partition -------
        if int(ordinary_positions.numel()) > 0:
            expert_output = self.annotation_expert(
                shared_active.index_select(0, ordinary_positions),
                state_active.index_select(0, ordinary_positions),
                previous_audit_evidence=(
                    None
                    if previous_audit_active is None
                    else previous_audit_active.index_select(0, ordinary_positions)
                ),
                turn_index=turn,
                iteration_index=turn,
                return_metadata=False,
                capture_geometry=True,
            )
            candidate_active = candidate_active.index_copy(
                0, ordinary_positions, expert_output.candidate_logits
            )
            ordinary_record = self.annotation_expert.build_replay_record(
                shared_features=shared_active.index_select(0, ordinary_positions),
                annotation_logits=state_active.index_select(0, ordinary_positions),
                previous_audit_evidence=(
                    None
                    if previous_audit_active is None
                    else previous_audit_active.index_select(0, ordinary_positions)
                ),
                turn_index=turn,
                iteration_index=turn,
                output=expert_output,
                record_kind=RECORD_KIND_ORDINARY,
            )
            fallback_offset_of = dict(zip(fallback_positions, fallback_offsets))
            for position in ordinary_list:
                global_index = int(active_indices[position])
                offset = fallback_offset_of.get(position)
                if offset is None:
                    rows[position] = self._diagnostic_row(
                        turn=turn,
                        sample_index=global_index,
                        eligible=False,
                        record_kind=previous_kind.get(global_index),
                        accepted_path="ordinary",
                        fallback_reason=(
                            "previous_action_not_replayable"
                            if global_index in previous_kind
                            else "no_accepted_history"
                        ),
                    )
                    unavailable = (
                        "ordinary annotation path: no accepted ordinary record was replayed"
                    )
                else:
                    # Candidate C was disabled for this transition; the row took
                    # the ordinary annotation path and the measured replay
                    # failure is reported rather than hidden behind an identity.
                    reason = (
                        replay_state_failure
                        if restitution_result is None
                        else restitution_result["reason"][offset]
                    )
                    rows[position] = self._diagnostic_row(
                        turn=turn,
                        sample_index=global_index,
                        eligible=False,
                        record_kind=RECORD_KIND_ORDINARY,
                        accepted_path="ordinary",
                        fallback_reason=reason,
                        result=restitution_result,
                        offset=offset,
                        record_turn=record.turn if record is not None else None,
                        solver_invocation_id=invocation,
                    )
                    # No support and no innovation were used by this row.
                    rows[position]["coordinate_displacement"] = None
                    rows[position]["innovation_magnitude"] = None
                    if restitution_result is None:
                        # C1 requires a VALID RECORDED STATE as well as numeric
                        # identity.  A preflight state failure is a C1 failure
                        # and must be counted as one, but no numeric comparison
                        # happened, so the errors stay null -- never a zero.
                        rows[position]["c1_passed"] = False
                        rows[position]["c1_failure_kind"] = "record_state"
                        rows[position]["c1_max_abs_err"] = None
                        rows[position]["c1_max_rel_err"] = None
                    unavailable = f"candidate C disabled for this transition ({reason})"
                geometry_rows[position] = {
                    "turn": int(turn),
                    "record_turn": None if offset is None or record is None else int(record.turn),
                    "sample_index": global_index,
                    "unavailable_reason": unavailable,
                }

        # --- phase 3: transfer the restitution innovation --------------------
        if restitution_positions_list and restitution_result is not None and record is not None:
            # C3: current retained state plus the recorded class-shared gate
            # times the exact factual/counterfactual logit difference.  The
            # innovation and the gate are detached; gradients reach the
            # annotator only through the retained state.
            group = torch.tensor(restitution_offsets, dtype=torch.long, device=device)
            retained_rows = state_active.index_select(0, restitution_positions)
            innovation = restitution_result["innovation"].index_select(0, group).to(
                dtype=retained_rows.dtype
            )
            gate = restitution_result["gate"].index_select(0, group).to(dtype=retained_rows.dtype)
            candidate_active = candidate_active.index_copy(
                0, restitution_positions, retained_rows + gate * innovation
            )
            for index, position in enumerate(restitution_positions_list):
                offset = restitution_offsets[index]
                global_index = int(active_indices[position])
                settled = bool(restitution_result["settled"][offset])
                rows[position] = self._diagnostic_row(
                    turn=turn,
                    sample_index=global_index,
                    eligible=bool(restitution_result["eligible"][offset]),
                    record_kind=RECORD_KIND_ORDINARY,
                    accepted_path=path if settled else "factual_support",
                    fallback_reason=restitution_result["reason"][offset],
                    result=restitution_result,
                    offset=offset,
                    innovation=innovation[index],
                    record_turn=record.turn,
                    solver_invocation_id=invocation,
                )
                if capture_geometry:
                    packed = _row_geometry(restitution_result.get("geometry"), offset)
                    geometry_rows[position] = (
                        {
                            "turn": int(turn),
                            "record_turn": int(record.turn),
                            "sample_index": global_index,
                            **packed,
                        }
                        if packed is not None
                        else {
                            "turn": int(turn),
                            "record_turn": int(record.turn),
                            "sample_index": global_index,
                            "unavailable_reason": (
                                "direct rollback performs no replay, so no realized support is solved"
                                if self.window_mode == "direct_rollback"
                                else "solver captured no geometry"
                            ),
                        }
                    )
        return (
            candidate_active,
            ordinary_record,
            ordinary_positions,
            restitution_positions,
            restitution_result,
            rows,
            geometry_rows if capture_geometry else [],
        )

    @staticmethod
    def _item(value: Any, offset: int) -> Any:
        if value is None:
            return None
        if torch.is_tensor(value):
            element = value[offset]
            if element.dtype == torch.bool:
                return bool(element)
            if not element.is_floating_point():
                return int(element)
            return float(element)
        return value[offset]

    def _diagnostic_row(
        self,
        *,
        turn: int,
        sample_index: int,
        eligible: bool,
        record_kind: str | None,
        accepted_path: str,
        fallback_reason: str | None,
        result: dict[str, Any] | None = None,
        offset: int | None = None,
        innovation: Tensor | None = None,
        record_turn: int | None = None,
        solver_invocation_id: str | None = None,
    ) -> dict[str, Any]:
        """One JSON-compatible per-sample attempt row.

        ``None`` always means *unmeasured*; it is never a stand-in zero.
        """

        row: dict[str, Any] = {
            "turn": int(turn),
            "sample_index": int(sample_index),
            "eligible": bool(eligible),
            "record_kind": record_kind,
            # Turn index of the accepted ORDINARY transition being replayed, so
            # an offline triplet can be aligned without any extra evaluation:
            # transition_previous[record_turn], transition_candidates[record_turn],
            # transition_candidates[turn].  ``None`` when there is no record.
            "record_turn": None if record_turn is None else int(record_turn),
            "c1_passed": None,
            # Which half of C1 failed: "record_state" when the record no longer
            # matches the live model state (no numeric comparison was possible,
            # so the error stays null rather than a fabricated zero), "numeric"
            # when the replay ran but did not reproduce the factual output.
            "c1_failure_kind": None,
            "c1_max_abs_err": None,
            "c1_max_rel_err": None,
            "regress_mass": None,
            "regress_mass_raw": None,
            "fix_mass": None,
            "num_protected": None,
            "num_ties_excluded": None,
            # Additive diagnostic: protected pixels whose factual winning
            # margin is at or below the tie epsilon.  They are protected like
            # every other thresholded predicted FIX pixel; the count only
            # reports how much of the protected set sits at the margin floor.
            "num_fix_ties": None,
            "feasible": None,
            "improved": None,
            "objective_before": None,
            "objective_after": None,
            "constraint_violation": None,
            "proposal_checks_run": None,
            "proposal_displacement": None,
            "coordinate_displacement": None,
            "innovation_magnitude": None,
            "fallback_reason": fallback_reason,
            "accepted_path": accepted_path,
            "evals": {
                "factual_replay": 0,
                "coordinate_backward": 0,
                "candidate_checks": 0,
                "total_forward": 0,
            },
            "solver_group_size": None,
            # Rows sharing this id shared ONE solver invocation and therefore
            # one set of eval counts; deduplicate on it before summing.
            "solver_invocation_id": solver_invocation_id,
            "delta_q": None,
            "accepted": None,
        }
        if result is None or offset is None:
            return row
        row.update(
            {
                "c1_passed": self._item(result["c1_passed"], offset),
                "c1_max_abs_err": self._item(result["c1_abs_error"], offset),
                "c1_max_rel_err": self._item(result["c1_rel_error"], offset),
                "regress_mass": self._item(result["regress_mass"], offset),
                "regress_mass_raw": self._item(result["regress_mass_raw"], offset),
                "fix_mass": self._item(result["fix_mass"], offset),
                "num_protected": self._item(result["num_protected"], offset),
                "num_ties_excluded": self._item(result["num_ties_excluded"], offset),
                "num_fix_ties": self._item(result["num_fix_ties"], offset),
                # ``feasible``/``improved``/``objective_after``/
                # ``constraint_violation`` describe the proposal that was
                # actually CHECKED (the accepted one, else the last evaluated);
                # they are null only when no proposal was evaluated.  They are
                # NOT synonyms for acceptance: a proposal can be feasible but
                # not improving, and a failed feasibility check reports its
                # measured violation rather than a null.
                "feasible": self._item(result["checked_feasible"], offset),
                "improved": self._item(result["checked_improved"], offset),
                "objective_before": self._item(result["objective_before"], offset),
                "objective_after": self._item(result["objective_after"], offset),
                "constraint_violation": self._item(result["violation_after"], offset),
                "proposal_checks_run": self._item(result["proposal_checks"], offset),
                "proposal_displacement": self._item(result["displacement_after"], offset),
                # Displacement of the SELECTED support: exactly 0.0 on the
                # identity fallback, because the factual support was selected.
                "coordinate_displacement": self._item(result["selected_displacement"], offset),
                # Evaluation counts are per solver invocation and are shared by
                # every row of the batched group; ``solver_group_size`` says how
                # many rows shared them.
                "evals": dict(result["evals"]),
                "solver_group_size": int(result["settled"].numel()),
            }
        )
        if row["c1_passed"] is False:
            row["c1_failure_kind"] = "numeric"
        if innovation is not None:
            row["innovation_magnitude"] = float(innovation.detach().abs().max())
        return row

    def _update_records(
        self,
        *,
        turn: int,
        active_indices: Tensor,
        accepted_active: Tensor,
        ordinary_record: Any,
        ordinary_positions: Tensor,
        restitution_positions: Tensor,
        restitution_result: dict[str, Any] | None,
        local_logits_active: Tensor,
        previous_kind: dict[int, str],
    ) -> _AcceptedOrdinaryRecord | None:
        """Keep at most one most-recent accepted ORDINARY transition per row.

        An accepted restitution or rollback action is *not* an ordinary
        AnnotationExpert update of its immediate pre-state, so it can never
        become a replay record.  It clears the row's ordinary history and only
        leaves an ineligible ``previous_kind`` marker, which forces the next
        active turn back onto the ordinary path.  This is what prevents an
        unbounded recursion of restitution-on-restitution records and what
        stops a stale older ordinary action from being relabelled as the most
        recent accepted action.
        """

        if int(restitution_positions.numel()) > 0 and restitution_result is not None:
            kind = (
                RECORD_KIND_ROLLBACK
                if self.window_mode == "direct_rollback"
                else RECORD_KIND_CANDIDATE_C
            )
            for offset in range(int(restitution_positions.numel())):
                position = int(restitution_positions[offset])
                global_index = int(active_indices[position])
                if bool(accepted_active[position]):
                    # Even the factual-support fallback is an identity action
                    # taken on the restitution path, not an ordinary
                    # AnnotationExpert update of its immediate pre-state, so it
                    # is never relabelled as replay-eligible history.
                    previous_kind[global_index] = kind
        if ordinary_record is None or int(ordinary_positions.numel()) == 0:
            return None
        accepted_ordinary = accepted_active.index_select(0, ordinary_positions)
        if not bool(accepted_ordinary.any()):
            return None
        # The record keeps EVERY row of the factual ordinary forward, not just
        # the accepted ones, so the replay reproduces that forward's exact batch
        # composition; ``accepted`` marks which rows are replay-eligible.
        evidence = local_logits_active.index_select(0, ordinary_positions).detach().softmax(dim=1)
        for offset in accepted_ordinary.nonzero(as_tuple=False).flatten().tolist():
            previous_kind.pop(int(active_indices[int(ordinary_positions[int(offset)])]), None)
        return _AcceptedOrdinaryRecord(
            indices=active_indices.index_select(0, ordinary_positions),
            expert_record=ordinary_record,
            audit_evidence=_freeze_tensor(evidence),
            turn=int(turn),
            accepted=accepted_ordinary.detach().clone(),
        )

    @staticmethod
    def _scatter_rows(reference: Tensor, indices: Tensor, values: Tensor) -> Tensor:
        """Scatter active-row values into a full batch while retaining autograd."""

        return SelfAuditNet._scatter_rows_from_values(reference.shape[0], indices, values)

    @staticmethod
    def _scatter_rows_from_values(batch_size: int, indices: Tensor, values: Tensor) -> Tensor:
        if values.ndim == 0 or values.shape[0] != indices.numel():
            raise ValueError(
                "Cannot scatter active rows: "
                f"values shape={tuple(values.shape)}, indices={indices.numel()}"
            )
        full = values.new_zeros((int(batch_size), *values.shape[1:]))
        return full.index_copy(0, indices, values)

    @staticmethod
    def _scatter_mask(reference: Tensor, indices: Tensor, values: Tensor) -> Tensor:
        if values.ndim != 1 or values.shape[0] != indices.numel():
            raise ValueError(
                "Cannot scatter active mask: "
                f"values shape={tuple(values.shape)}, indices={indices.numel()}"
            )
        full = torch.zeros_like(reference, dtype=torch.bool)
        return full.index_copy(0, indices, values.to(dtype=torch.bool))

    @staticmethod
    def _stack_masks(masks: list[Tensor], *, batch_size: int, device: torch.device) -> Tensor:
        if not masks:
            return torch.empty((0, int(batch_size)), dtype=torch.bool, device=device)
        return torch.stack(masks, dim=0)

    @staticmethod
    def _audit_tensor(output: Any, name: str) -> Tensor:
        if isinstance(output, dict):
            value = output.get(name)
        else:
            value = getattr(output, name, None)
        if not torch.is_tensor(value):
            raise TypeError(f"Auditor output must contain tensor {name!r}")
        return value

    def _candidate_improves(self, previous: Tensor, candidate: Tensor, target: Tensor) -> Tensor:
        previous_labels = previous.detach().argmax(dim=1)
        candidate_labels = candidate.detach().argmax(dim=1)
        target = target.detach().long()
        previous_score = self._dice(previous_labels, target)
        candidate_score = self._dice(candidate_labels, target)
        return candidate_score > previous_score

    @staticmethod
    def _dice(prediction: Tensor, target: Tensor) -> Tensor:
        values = []
        for cls in (1, 2, 3):
            pred = prediction == cls
            true = target == cls
            denominator = pred.flatten(1).sum(1) + true.flatten(1).sum(1)
            score = torch.where(denominator == 0, torch.ones_like(denominator, dtype=torch.float32), 2.0 * (pred & true).flatten(1).sum(1).float() / denominator.clamp_min(1).float())
            values.append(score)
        return torch.stack(values, dim=1).mean(dim=1)

    def forward(self, images: Tensor, **kwargs: Any) -> dict[str, Any]:
        return self.infer(images, **kwargs)


def build_self_audit_net(**kwargs: Any) -> SelfAuditNet:
    return SelfAuditNet(**kwargs)


SelfAudit = SelfAuditNet
