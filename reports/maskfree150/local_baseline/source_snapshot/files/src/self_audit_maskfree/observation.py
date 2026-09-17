"""Constrained region-conditioned appearance observation model (W1).

Scientific role
---------------
This module answers one narrow question: given a candidate anatomical partition
``Hypothesis`` and the *fitting* observations of a study unit, how well does that
partition explain observations it was never fitted on?

It is deliberately a small, inspectable, fully specified generative model:

* four region-conditioned appearance components (the fixed named semantic order
  ``0=BG, 1=RV, 2=MYO, 3=LV``), optionally two nuisance background modes;
* a restricted affine-in-XY smooth bias plane shared by all regions;
* a fixed variance floor expressed in fit-normalized intensity units squared.

There is no neural renderer, no per-pixel free parameter, no skip connection
from the scored image, and no refitting at scoring time. The only way a
candidate can lower its scored negative log-likelihood is by placing region
boundaries so that the *fit-side* appearance statistics also explain the
withheld observations. Every candidate in a bank receives exactly the same
capacity, the same initialization recipe and the same fitting budget, so a score
difference between two candidates cannot be produced by an unequal fit.

What this score is NOT
----------------------
``EvidenceScore.total`` is a predictive negative log-likelihood in nats per valid
observed pixel plus declared complexity and prior terms. It is not segmentation
accuracy, not a calibrated probability of correctness, and not evidence of
novelty. A lower score means the partition predicted withheld intensities better
under this specific constrained model, nothing more.

Frozen fitted identity
----------------------
``fit`` deep-copies the candidate before using it, so a ``FittedHypothesis``
never shares tensors with the caller's ``Hypothesis``, and it records a SHA-256
digest of the frozen partition. ``score`` re-checks that digest and refuses a
snapshot that was mutated in place. A stored fit therefore always describes the
partition whose appearance parameters it holds. ``score`` also requires the
scoring model's full recipe — capacity, background modes, iterations, variance
floor, bias ridge, beta and density family — to equal the one recorded at fit
time, because capacity alone does not make two numbers comparable.

Observation firewall
--------------------
``fit`` accepts only :class:`~self_audit_maskfree.contracts.FittingView`, whose
withheld pixels and guard bands are physically zero in every channel. ``score``
accepts only :class:`~self_audit_maskfree.contracts.ScoringView` and mutates
nothing: fitted parameters are frozen before scoring and are re-validated
afterwards by the caller if desired. Consequently no scored intensity can reach
any fitted parameter.

Temporal support
----------------
Only ``spatial_predictive`` is implemented. ``cine_predictive`` requires an
explicitly parameterized fit-time nonrigid temporal deformation model that this
module does not yet provide; requesting it raises rather than silently degrading
to a spatial fit that would be mislabeled as a temporal experiment.
"""
from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import numpy as np

from .contracts import (
    VERSION,
    EvidenceScore,
    FittedHypothesis,
    FittingView,
    Hypothesis,
    ScoringView,
)

NUM_REGIONS = 4
SUPPORTED_PROTOCOLS = ("spatial_predictive",)
#: Absolute tolerance on the per-pixel class-probability sum. float32 sums of
#: four values err by ~1e-7, so this only forgives genuine rounding.
PROBABILITY_SUM_TOLERANCE = 1e-4
#: Fixed additive penalty per nuisance component beyond the four region
#: components. Capacity is matched across a bank, so this is a constant offset
#: within one comparison; it only matters if a caller ever compares fits made at
#: different capacity, which the contract forbids.
COMPONENT_PENALTY = 0.01
#: Minimum number of fit pixels a region needs before its own appearance
#: statistics are trusted. Smaller regions fall back to the fit-global
#: distribution and are counted in ``parameters['fallback_regions']``.
MIN_REGION_PIXELS = 8
#: Student-t degrees of freedom when ``distribution='student_t'``. Fixed, not
#: fitted, so that capacity stays identical across candidates.
STUDENT_T_DOF = 4.0

Distribution = Literal["gaussian", "student_t"]


class ObservationContractError(ValueError):
    """Raised when an input violates the frozen maskfree150 observation contract."""


@dataclass(frozen=True)
class _Component:
    """One appearance component: a region index plus its fitted density."""

    region: int
    mean: float
    variance: float
    log_weight: float
    fallback: bool


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float64).item())


def _check_hypothesis(hypothesis: Hypothesis, height: int, width: int) -> None:
    if not isinstance(hypothesis, Hypothesis):
        raise ObservationContractError("fit/score require a contracts.Hypothesis")
    labels = hypothesis.labels
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long:
        raise ObservationContractError("hypothesis.labels must be a long tensor")
    if tuple(labels.shape) != (height, width):
        raise ObservationContractError(
            f"hypothesis.labels must be [H,W]={(height, width)}, got {tuple(labels.shape)}"
        )
    if int(labels.min()) < 0 or int(labels.max()) >= NUM_REGIONS:
        raise ObservationContractError("hypothesis.labels must lie in the fixed semantic order 0..3")
    probabilities = hypothesis.probabilities
    if not isinstance(probabilities, torch.Tensor) or probabilities.dtype != torch.float32:
        raise ObservationContractError("hypothesis.probabilities must be a float32 tensor")
    if tuple(probabilities.shape) != (NUM_REGIONS, height, width):
        raise ObservationContractError("hypothesis.probabilities must be [4,H,W]")
    if not torch.isfinite(probabilities).all():
        raise ObservationContractError("hypothesis.probabilities contains nonfinite values")
    if float(probabilities.min()) < 0.0:
        raise ObservationContractError("hypothesis.probabilities must be non-negative")
    sums = probabilities.sum(dim=0)
    if float((sums - 1.0).abs().max()) > PROBABILITY_SUM_TOLERANCE:
        raise ObservationContractError(
            "hypothesis.probabilities must sum to one over the four semantic classes at "
            f"every pixel (max deviation {float((sums - 1.0).abs().max()):.3e})"
        )
    validity = hypothesis.validity
    if not isinstance(validity, torch.Tensor) or validity.dtype != torch.float32:
        raise ObservationContractError("hypothesis.validity must be a float32 tensor")
    if tuple(validity.shape) != (height, width):
        raise ObservationContractError("hypothesis.validity must be [H,W]")
    if not torch.isfinite(validity).all():
        raise ObservationContractError("hypothesis.validity contains nonfinite values")
    if float(validity.min()) < 0.0 or float(validity.max()) > 1.0:
        raise ObservationContractError("hypothesis.validity must lie in [0,1]")


def _freeze_hypothesis(hypothesis: Hypothesis) -> Hypothesis:
    """Return a private deep copy of a candidate, detached from the caller.

    ``Hypothesis`` is a mutable dataclass holding mutable tensors. If a
    ``FittedHypothesis`` merely referenced the caller's object, a later in-place
    edit — a challenger round rewriting labels, a bank being reused — would
    silently change which partition a stored fit is understood to describe,
    while the fitted appearance parameters still belonged to the old partition.
    Freezing at fit time makes the fitted identity immutable from outside.
    """
    return Hypothesis(
        candidate_id=hypothesis.candidate_id,
        labels=hypothesis.labels.detach().clone(),
        probabilities=hypothesis.probabilities.detach().clone(),
        validity=hypothesis.validity.detach().clone(),
        source=hypothesis.source,
        semantic_unresolved=hypothesis.semantic_unresolved,
        alternatives=copy.deepcopy(hypothesis.alternatives),
        metadata=copy.deepcopy(hypothesis.metadata),
    )


def _hypothesis_digest(hypothesis: Hypothesis) -> str:
    """SHA-256 over everything of a candidate that can change its likelihood.

    Recorded at fit time and re-checked at scoring time, so that mutating the
    frozen snapshot in place is rejected rather than silently rescoring a
    different partition against parameters fitted for another one.
    """
    digest = hashlib.sha256()
    digest.update(hypothesis.candidate_id.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(hypothesis.source.encode("utf-8"))
    digest.update(b"\x01" if hypothesis.semantic_unresolved else b"\x00")
    for tensor in (hypothesis.labels, hypothesis.probabilities, hypothesis.validity):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(repr(tuple(contiguous.shape)).encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(contiguous.numpy().tobytes())
    digest.update(repr(_prior_penalty(hypothesis)).encode("utf-8"))
    return digest.hexdigest()


def _prior_penalty(hypothesis: Hypothesis) -> float:
    """Read the declared anatomical-prior penalty produced by the ontology layer.

    W1 never invents anatomical priors. The ontology/hypothesis owner records a
    non-negative penalty in nats under ``metadata['prior_penalty']``; anything
    absent scores as zero prior. Semantic ambiguity deliberately adds nothing:
    a semantic permutation that leaves appearance unchanged must leave the
    likelihood unchanged instead of acquiring an invented distinction.
    """
    raw = hypothesis.metadata.get("prior_penalty", 0.0)
    if isinstance(raw, torch.Tensor):
        raw = _as_float(raw)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ObservationContractError("metadata['prior_penalty'] must be a real number")
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ObservationContractError("metadata['prior_penalty'] must be finite and non-negative")
    return value


def _neighbor_disagreement(labels: torch.Tensor) -> float:
    """Fraction of 4-neighbor pixel pairs whose labels differ.

    This is the whole complexity term for the partition itself: an over-split
    partition pays for its boundary length, which is what stops the model from
    buying likelihood with arbitrarily many small regions.
    """
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    total = horizontal.numel() + vertical.numel()
    if total == 0:
        return 0.0
    return float(horizontal.sum().item() + vertical.sum().item()) / float(total)


def _plane_basis(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Design matrix [H*W,3] for the restricted affine bias plane ``a0+a1*x+a2*y``."""
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=torch.float64)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(grid_x)
    return torch.stack([ones.reshape(-1), grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)


def _log_density(
    values: torch.Tensor,
    mean: float,
    variance: float,
    distribution: Distribution,
) -> torch.Tensor:
    """Log density of ``values`` under one component, in nats.

    ``variance`` is the squared scale parameter in both families. For the
    Student-t this makes ``variance`` the squared scale, not the distribution
    variance (which is ``variance * dof/(dof-2)``); the fit is a moment-matched
    heavy-tail robustification of the Gaussian fit, not an EM t-fit, and is
    documented as such rather than presented as a maximum-likelihood t estimate.
    """
    deviation = values - mean
    if distribution == "gaussian":
        return -0.5 * math.log(2.0 * math.pi * variance) - deviation.pow(2) / (2.0 * variance)
    dof = STUDENT_T_DOF
    normalizer = (
        math.lgamma((dof + 1.0) / 2.0)
        - math.lgamma(dof / 2.0)
        - 0.5 * math.log(dof * math.pi * variance)
    )
    return normalizer - ((dof + 1.0) / 2.0) * torch.log1p(deviation.pow(2) / (dof * variance))


class ObservationModel:
    """Deterministic constrained appearance model over anatomical partitions.

    Parameters
    ----------
    max_iterations:
        Outer alternating passes (region statistics, then bias plane). Every
        candidate in a bank uses exactly this many passes; the executed count is
        reported in ``FittedHypothesis.fitting_steps``.
    variance_floor:
        Lower bound on every component variance, in fit-normalized intensity
        units squared. Prevents a degenerate partition from winning by driving a
        tiny region's variance to zero.
    beta:
        Weight of the complexity term in the total score.
    background_modes:
        1 or 2. Two modes give background the nuisance capacity to absorb
        non-anatomical structure. The value is a property of the *model*, not of
        a candidate, so capacity is matched across every member of a bank.
    bias_ridge:
        Ridge coefficient of the affine bias solve; contract minimum 0.1.
    distribution:
        ``'gaussian'`` or ``'student_t'``.

    The model holds no per-candidate state. ``fit`` returns all fitted
    parameters inside its :class:`FittedHypothesis`, so two candidates can never
    share mutable nuisance parameters.
    """

    supports_temporal = False

    def __init__(
        self,
        max_iterations: int = 5,
        variance_floor: float = 0.05,
        beta: float = 0.01,
        *,
        background_modes: int = 1,
        bias_ridge: float = 0.1,
        distribution: Distribution = "gaussian",
    ) -> None:
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ObservationContractError("max_iterations must be a positive int")
        if not (variance_floor > 0.0 and math.isfinite(variance_floor)):
            raise ObservationContractError("variance_floor must be finite and positive")
        if not (beta >= 0.0 and math.isfinite(beta)):
            raise ObservationContractError("beta must be finite and non-negative")
        if background_modes not in (1, 2):
            raise ObservationContractError("background_modes must be exactly 1 or 2")
        if not (bias_ridge >= 0.1 and math.isfinite(bias_ridge)):
            raise ObservationContractError("bias_ridge must be >= 0.1 per the contract")
        if distribution not in ("gaussian", "student_t"):
            raise ObservationContractError("distribution must be 'gaussian' or 'student_t'")
        self.max_iterations = max_iterations
        self.variance_floor = float(variance_floor)
        self.beta = float(beta)
        self.background_modes = int(background_modes)
        self.bias_ridge = float(bias_ridge)
        self.distribution: Distribution = distribution

    @property
    def capacity(self) -> int:
        """Total number of appearance components: 3 foreground + background modes."""
        return (NUM_REGIONS - 1) + self.background_modes

    def recipe(self) -> dict[str, Any]:
        """The full comparison recipe, not just capacity.

        Two scores are only comparable if every knob that moves the likelihood
        was identical: capacity alone is not enough, because a different
        variance floor, bias ridge, density family, iteration count or beta
        produces a different number on the same data. ``score`` refuses a fit
        produced under any other recipe.
        """
        return {
            "contract_version": VERSION,
            "capacity": self.capacity,
            "background_modes": self.background_modes,
            "max_iterations": self.max_iterations,
            "variance_floor": self.variance_floor,
            "bias_ridge": self.bias_ridge,
            "beta": self.beta,
            "distribution": self.distribution,
            "student_t_dof": STUDENT_T_DOF if self.distribution == "student_t" else None,
            "min_region_pixels": MIN_REGION_PIXELS,
            "component_penalty": COMPONENT_PENALTY * float(self.capacity - NUM_REGIONS),
        }

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    def fit(self, hypothesis: Hypothesis, fitting_view: FittingView) -> FittedHypothesis:
        """Fit nuisance appearance parameters for one candidate on fitting observations only.

        The scored observations are not an argument and cannot be reached from
        here. All tensors are detached: the nuisance fit is a closed-form
        alternating estimator, never a gradient path back into the producer.
        """
        if not isinstance(fitting_view, FittingView):
            raise ObservationContractError("fit requires a contracts.FittingView")
        fitting_view.validate()
        if fitting_view.protocol not in SUPPORTED_PROTOCOLS:
            raise ObservationContractError(
                f"protocol {fitting_view.protocol!r} is not implemented by ObservationModel; "
                "cine_predictive needs an explicit fit-time temporal deformation model and "
                "must not be approximated by a spatial fit"
            )
        height, width = fitting_view.support.shape
        _check_hypothesis(hypothesis, height, width)
        # Freeze immediately: everything below, and everything the returned
        # FittedHypothesis exposes, uses this private copy. A caller mutating
        # its own Hypothesis afterwards cannot change what was fitted or what
        # will later be scored.
        hypothesis = _freeze_hypothesis(hypothesis)

        image = fitting_view.image.detach().to(torch.float64)[0]
        support = fitting_view.support.detach()
        labels = hypothesis.labels.detach().to(support.device)
        flat_support = support.reshape(-1)
        flat_image = image.reshape(-1)[flat_support]
        flat_labels = labels.reshape(-1)[flat_support]
        basis = _plane_basis(height, width, image.device)[flat_support]

        observed = int(flat_support.sum().item())
        if observed == 0:  # pragma: no cover - FittingView.validate already rejects this
            raise ObservationContractError("fitting view has empty support")

        global_mean = _as_float(flat_image.mean())
        global_variance = max(
            _as_float(flat_image.var(unbiased=False)) if observed > 1 else 0.0,
            self.variance_floor,
        )

        bias_coefficients = torch.zeros(3, dtype=torch.float64, device=image.device)
        components: list[_Component] = []
        region_counts = {region: 0 for region in range(NUM_REGIONS)}
        fallback_regions: list[int] = []

        for _ in range(self.max_iterations):
            residual = flat_image - basis @ bias_coefficients
            components, region_counts, fallback_regions = self._fit_components(
                residual, flat_labels, global_mean, global_variance
            )
            bias_coefficients = self._fit_bias(residual + basis @ bias_coefficients,
                                               flat_labels, basis, components)

        residual = flat_image - basis @ bias_coefficients
        nll = -self._log_mixture(residual, flat_labels, components)
        nll_sum = _as_float(nll.sum())

        complexity = _neighbor_disagreement(labels)
        prior = _prior_penalty(hypothesis)
        component_penalty = COMPONENT_PENALTY * float(self.capacity - NUM_REGIONS)
        complexity_total = complexity + component_penalty
        normalized = nll_sum / float(observed)

        fit_score = EvidenceScore(
            nll_sum=nll_sum,
            count=observed,
            normalized_nll=normalized,
            complexity=complexity_total,
            prior=prior,
            total=normalized + self.beta * complexity_total + prior,
            role="fit",
            available=True,
            reason=None,
        )

        parameters: dict[str, Any] = {
            "fit_support_shape": list(support.shape),
            "fit_support_bits": np.packbits(support.detach().cpu().numpy()).tobytes().hex(),
            "bias_coefficients": [float(value) for value in bias_coefficients.tolist()],
            "bias_ridge": self.bias_ridge,
            "components": [
                {
                    "region": component.region,
                    "mean": component.mean,
                    "variance": component.variance,
                    "log_weight": component.log_weight,
                    "fallback": component.fallback,
                }
                for component in components
            ],
            "distribution": self.distribution,
            "student_t_dof": STUDENT_T_DOF if self.distribution == "student_t" else None,
            "variance_floor": self.variance_floor,
            "global_mean": global_mean,
            "global_variance": global_variance,
            "region_counts": dict(region_counts),
            "fallback_regions": list(fallback_regions),
            "fallback_region_count": len(fallback_regions),
            "observed_fit_pixels": observed,
        }

        return FittedHypothesis(
            hypothesis=hypothesis,
            parameters=parameters,
            fit_score=fit_score,
            fitting_steps=self.max_iterations,
            capacity=self.capacity,
            study_id=fitting_view.study_id,
            unit_id=fitting_view.unit_id,
            partition_id=fitting_view.partition_id,
            metadata={
                "contract_version": VERSION,
                "protocol": fitting_view.protocol,
                "beta": self.beta,
                "background_modes": self.background_modes,
                "component_penalty": component_penalty,
                "max_iterations": self.max_iterations,
                "temporal_support": self.supports_temporal,
                "recipe": self.recipe(),
                "hypothesis_digest": _hypothesis_digest(hypothesis),
            },
        )

    def _fit_components(
        self,
        residual: torch.Tensor,
        flat_labels: torch.Tensor,
        global_mean: float,
        global_variance: float,
    ) -> tuple[list[_Component], dict[int, int], list[int]]:
        """Closed-form region statistics on the bias-corrected residual."""
        components: list[_Component] = []
        region_counts: dict[int, int] = {}
        fallback_regions: list[int] = []
        for region in range(NUM_REGIONS):
            selected = residual[flat_labels == region]
            count = int(selected.numel())
            region_counts[region] = count
            modes = self.background_modes if region == 0 else 1
            if count < MIN_REGION_PIXELS:
                # Empty or near-empty region: fall back to the fit-global
                # distribution rather than inventing a sharp component that a
                # degenerate partition could exploit. Counted explicitly.
                fallback_regions.append(region)
                for _ in range(modes):
                    components.append(
                        _Component(
                            region=region,
                            mean=global_mean,
                            variance=global_variance,
                            log_weight=-math.log(modes),
                            fallback=True,
                        )
                    )
                continue
            if modes == 1:
                mean = _as_float(selected.mean())
                variance = max(_as_float(selected.var(unbiased=False)), self.variance_floor)
                components.append(_Component(region, mean, variance, 0.0, False))
                continue
            components.extend(self._fit_background_modes(selected, global_variance))
        return components, region_counts, fallback_regions

    def _fit_background_modes(
        self, values: torch.Tensor, global_variance: float
    ) -> list[_Component]:
        """Deterministic two-mode nuisance fit for the background region.

        Hard assignment seeded at the 25th/75th residual percentiles and refined
        for a fixed three passes. No RNG, so two identical inputs always give
        identical modes, and the capacity is identical for every candidate.
        """
        quantiles = torch.tensor([0.25, 0.75], dtype=values.dtype, device=values.device)
        centers = torch.quantile(values, quantiles)
        if _as_float(centers[1] - centers[0]) <= 0.0:
            spread = math.sqrt(global_variance)
            centers = torch.tensor(
                [_as_float(values.mean()) - spread, _as_float(values.mean()) + spread],
                dtype=values.dtype,
                device=values.device,
            )
        assignment = torch.zeros_like(values, dtype=torch.long)
        for _ in range(3):
            distances = torch.stack([(values - centers[0]).abs(), (values - centers[1]).abs()])
            assignment = distances.argmin(dim=0)
            for mode in (0, 1):
                member = values[assignment == mode]
                if member.numel() >= MIN_REGION_PIXELS:
                    centers[mode] = member.mean()
        components: list[_Component] = []
        total = float(values.numel())
        for mode in (0, 1):
            member = values[assignment == mode]
            count = int(member.numel())
            if count < MIN_REGION_PIXELS:
                components.append(
                    _Component(0, _as_float(centers[mode]), global_variance,
                               math.log(max(count, 1) / total), True)
                )
                continue
            mean = _as_float(member.mean())
            variance = max(_as_float(member.var(unbiased=False)), self.variance_floor)
            components.append(_Component(0, mean, variance, math.log(count / total), False))
        return components

    def _fit_bias(
        self,
        values: torch.Tensor,
        flat_labels: torch.Tensor,
        basis: torch.Tensor,
        components: list[_Component],
    ) -> torch.Tensor:
        """Weighted ridge solve for the restricted affine bias plane.

        The bias has three degrees of freedom for the whole image, so it can
        shade the field under a ridge penalty, and it cannot act
        as a per-pixel free reconstruction parameter.
        """
        target_mean = torch.zeros_like(values)
        weights = torch.zeros_like(values)
        for region in range(NUM_REGIONS):
            region_components = [c for c in components if c.region == region]
            if not region_components:  # pragma: no cover - every region emits >=1 component
                continue
            mask = flat_labels == region
            if not bool(mask.any()):
                continue
            weight_sum = sum(math.exp(c.log_weight) for c in region_components)
            mean = sum(math.exp(c.log_weight) * c.mean for c in region_components) / weight_sum
            variance = sum(math.exp(c.log_weight) * c.variance for c in region_components) / weight_sum
            target_mean[mask] = mean
            weights[mask] = 1.0 / variance
        weighted_basis = basis * weights.unsqueeze(1)
        normal_matrix = basis.transpose(0, 1) @ weighted_basis
        normal_matrix = normal_matrix + self.bias_ridge * torch.eye(
            3, dtype=basis.dtype, device=basis.device
        )
        right_hand = weighted_basis.transpose(0, 1) @ (values - target_mean)
        return torch.linalg.solve(normal_matrix, right_hand)

    def _log_mixture(
        self,
        residual: torch.Tensor,
        flat_labels: torch.Tensor,
        components: list[_Component],
    ) -> torch.Tensor:
        """Per-pixel log density under the component mixture of that pixel's region."""
        output = torch.full_like(residual, float("nan"))
        for region in range(NUM_REGIONS):
            mask = flat_labels == region
            if not bool(mask.any()):
                continue
            region_components = [c for c in components if c.region == region]
            if not region_components:  # pragma: no cover
                raise ObservationContractError(f"no fitted component for region {region}")
            values = residual[mask]
            stacked = torch.stack(
                [
                    _log_density(values, c.mean, c.variance, self.distribution) + c.log_weight
                    for c in region_components
                ]
            )
            output[mask] = torch.logsumexp(stacked, dim=0)
        if torch.isnan(output).any():  # pragma: no cover - every label is covered above
            raise ObservationContractError("unscored pixels remain in the mixture density")
        return output

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def score(self, fitted: FittedHypothesis, scoring_view: ScoringView) -> EvidenceScore:
        """Score a *frozen* fit against withheld observations. Never refits.

        The full draft partition enters the likelihood. Validity is deliberately
        not used as a weight here: a candidate must not be able to hide a badly
        explained region by declaring it invalid.
        """
        if not isinstance(fitted, FittedHypothesis):
            raise ObservationContractError("score requires a contracts.FittedHypothesis")
        if not isinstance(scoring_view, ScoringView):
            raise ObservationContractError("score requires a contracts.ScoringView")
        scoring_view.validate()
        if fitted.study_id != scoring_view.study_id:
            raise ObservationContractError(
                f"study mismatch: fitted {fitted.study_id!r} vs scoring {scoring_view.study_id!r}"
            )
        if fitted.unit_id != scoring_view.unit_id:
            raise ObservationContractError(
                f"unit mismatch: fitted {fitted.unit_id!r} vs scoring {scoring_view.unit_id!r}"
            )
        if fitted.partition_id != scoring_view.partition_id:
            raise ObservationContractError(
                "partition mismatch: a fit may only be scored against observations of the "
                "same frozen observation partition"
            )
        expected = self.recipe()
        recorded = fitted.metadata.get("recipe")
        if recorded is None:
            raise ObservationContractError(
                "fitted hypothesis carries no recipe record; it was not produced by this "
                "ObservationModel version and cannot be scored"
            )
        differing = sorted(
            key for key in expected
            if recorded.get(key, "<missing>") != expected[key]
        )
        if differing:
            details = ", ".join(
                f"{key}: fit={recorded.get(key, '<missing>')!r} vs model={expected[key]!r}"
                for key in differing
            )
            raise ObservationContractError(
                "recipe mismatch, candidates must be compared under one identical recipe "
                f"({details})"
            )
        if fitted.capacity != self.capacity:  # pragma: no cover - implied by the recipe check
            raise ObservationContractError(
                f"capacity mismatch: fit used {fitted.capacity}, this model has {self.capacity}"
            )

        height, width = scoring_view.support.shape
        support_bits = fitted.parameters.get("fit_support_bits")
        if not isinstance(support_bits, str) or fitted.parameters.get("fit_support_shape") != [height, width]:
            raise ObservationContractError("missing or incompatible fitting support")
        packed = np.frombuffer(bytes.fromhex(support_bits), dtype=np.uint8)
        if len(packed) != (height * width + 7) // 8:
            raise ObservationContractError("invalid fitting support encoding")
        fitting_support = torch.from_numpy(np.unpackbits(packed, count=height * width).astype(bool).reshape(height, width))
        if torch.any(fitting_support.to(scoring_view.support.device) & scoring_view.support):
            raise ObservationContractError("scoring observations overlap fitting observations")
        hypothesis = fitted.hypothesis
        _check_hypothesis(hypothesis, height, width)
        recorded_digest = fitted.metadata.get("hypothesis_digest")
        if recorded_digest is None:
            raise ObservationContractError(
                "fitted hypothesis carries no identity digest; refusing to score an "
                "unverifiable partition"
            )
        if _hypothesis_digest(hypothesis) != recorded_digest:
            raise ObservationContractError(
                "fitted hypothesis was mutated after fitting; its appearance parameters "
                "belong to the partition that was fitted, not to this one. Refit instead."
            )

        complexity = _neighbor_disagreement(hypothesis.labels) + float(
            fitted.metadata.get("component_penalty", 0.0)
        )
        prior = _prior_penalty(hypothesis)

        support = scoring_view.support.detach()
        count = int(support.sum().item())
        if count == 0:
            return EvidenceScore(
                nll_sum=0.0,
                count=0,
                normalized_nll=None,
                complexity=complexity,
                prior=prior,
                total=None,
                role=scoring_view.role,
                available=False,
                reason="no observed pixels in this scoring role for this unit",
            )

        components = [
            _Component(
                region=int(entry["region"]),
                mean=float(entry["mean"]),
                variance=float(entry["variance"]),
                log_weight=float(entry["log_weight"]),
                fallback=bool(entry["fallback"]),
            )
            for entry in fitted.parameters["components"]
        ]
        bias_coefficients = torch.tensor(
            fitted.parameters["bias_coefficients"], dtype=torch.float64,
            device=scoring_view.image.device,
        )

        flat_support = support.reshape(-1)
        flat_image = scoring_view.image.detach().to(torch.float64)[0].reshape(-1)[flat_support]
        flat_labels = hypothesis.labels.detach().to(support.device).reshape(-1)[flat_support]
        basis = _plane_basis(height, width, scoring_view.image.device)[flat_support]

        residual = flat_image - basis @ bias_coefficients
        nll = -self._log_mixture(residual, flat_labels, components)
        nll_sum = _as_float(nll.sum())
        normalized = nll_sum / float(count)

        return EvidenceScore(
            nll_sum=nll_sum,
            count=count,
            normalized_nll=normalized,
            complexity=complexity,
            prior=prior,
            total=normalized + self.beta * complexity + prior,
            role=scoring_view.role,
            available=True,
            reason=None,
        )
