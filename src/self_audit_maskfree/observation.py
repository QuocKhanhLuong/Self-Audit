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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import torch

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


@dataclass(frozen=True, init=False)
class PreparedScore:
    """Immutable-by-contract full-support score snapshot.

    PyTorch tensors remain mutable even when held by a frozen dataclass.  The
    cached tensors are private detached clones and ``score_region`` verifies a
    digest (plus the cheap tensor version counters) before every reduction.  A
    caller can inspect cloned properties, but cannot mutate the snapshot
    without being rejected on the next operation.
    """

    _nll_per_pixel: torch.Tensor = field(repr=False, compare=False)
    _support: torch.Tensor = field(repr=False, compare=False)
    nll_sum: float
    count: int
    normalized_nll: float | None
    complexity: float
    prior: float
    total: float | None
    role: str
    available: bool
    reason: str | None
    study_id: str
    unit_id: str
    partition_id: str
    recipe_snapshot: tuple[tuple[str, Any], ...]
    hypothesis_digest: str
    _nll_version: int = field(repr=False, compare=False)
    _support_version: int = field(repr=False, compare=False)
    _integrity: str = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        nll_per_pixel: torch.Tensor,
        support: torch.Tensor,
        nll_sum: float,
        count: int,
        normalized_nll: float | None,
        complexity: float,
        prior: float,
        total: float | None,
        role: str,
        available: bool,
        reason: str | None,
        study_id: str,
        unit_id: str,
        partition_id: str,
        recipe_snapshot: tuple[tuple[str, Any], ...],
        hypothesis_digest: str,
    ) -> None:
        if not isinstance(nll_per_pixel, torch.Tensor) or not isinstance(support, torch.Tensor):
            raise ObservationContractError("prepared score storage must be tensors")
        nll_clone = nll_per_pixel.detach().clone()
        support_clone = support.detach().clone()
        object.__setattr__(self, "_nll_per_pixel", nll_clone)
        object.__setattr__(self, "_support", support_clone)
        object.__setattr__(self, "nll_sum", float(nll_sum))
        object.__setattr__(self, "count", int(count))
        object.__setattr__(self, "normalized_nll", normalized_nll)
        object.__setattr__(self, "complexity", float(complexity))
        object.__setattr__(self, "prior", float(prior))
        object.__setattr__(self, "total", total if total is None else float(total))
        object.__setattr__(self, "role", str(role))
        object.__setattr__(self, "available", bool(available))
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "study_id", str(study_id))
        object.__setattr__(self, "unit_id", str(unit_id))
        object.__setattr__(self, "partition_id", str(partition_id))
        object.__setattr__(self, "recipe_snapshot", tuple(recipe_snapshot))
        object.__setattr__(self, "hypothesis_digest", str(hypothesis_digest))
        object.__setattr__(self, "_nll_version", int(nll_clone._version))
        object.__setattr__(self, "_support_version", int(support_clone._version))
        object.__setattr__(
            self,
            "_integrity",
            _prepared_digest(
                nll_clone,
                support_clone,
                nll_sum=float(nll_sum),
                count=int(count),
                normalized_nll=normalized_nll,
                complexity=float(complexity),
                prior=float(prior),
                total=total if total is None else float(total),
                role=str(role),
                available=bool(available),
                reason=reason,
                study_id=str(study_id),
                unit_id=str(unit_id),
                partition_id=str(partition_id),
                recipe_snapshot=tuple(recipe_snapshot),
                hypothesis_digest=str(hypothesis_digest),
            ),
        )

    @property
    def nll_per_pixel(self) -> torch.Tensor:
        """A detached copy of cached per-pixel NLL (never the private storage)."""
        return self._nll_per_pixel.detach().clone()

    @property
    def support(self) -> torch.Tensor:
        """A detached copy of the full scoring support."""
        return self._support.detach().clone()

    def _assert_integrity(self) -> None:
        """Reject mutation of private tensor storage or scalar snapshot fields."""
        if self._nll_per_pixel._version != self._nll_version:
            raise ObservationContractError("prepared score NLL storage was mutated")
        if self._support._version != self._support_version:
            raise ObservationContractError("prepared score support storage was mutated")
        current = _prepared_digest(
            self._nll_per_pixel,
            self._support,
            nll_sum=self.nll_sum,
            count=self.count,
            normalized_nll=self.normalized_nll,
            complexity=self.complexity,
            prior=self.prior,
            total=self.total,
            role=self.role,
            available=self.available,
            reason=self.reason,
            study_id=self.study_id,
            unit_id=self.unit_id,
            partition_id=self.partition_id,
            recipe_snapshot=self.recipe_snapshot,
            hypothesis_digest=self.hypothesis_digest,
        )
        if current != self._integrity:
            raise ObservationContractError("prepared score snapshot was mutated")


@dataclass(frozen=True)
class _ScoreContext:
    """Validated scalar score inputs used by the non-reference fast paths."""

    fitted: FittedHypothesis
    scoring_view: ScoringView
    hypothesis: Hypothesis
    complexity: float
    prior: float
    support: torch.Tensor
    count: int
    components: tuple[_Component, ...]
    bias_coefficients: torch.Tensor
    height: int
    width: int


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


def _freeze_hypothesis(
    hypothesis: Hypothesis,
    *,
    device: torch.device | None = None,
) -> Hypothesis:
    """Return a private deep copy of a candidate, detached from the caller.

    ``Hypothesis`` is a mutable dataclass holding mutable tensors. If a
    ``FittedHypothesis`` merely referenced the caller's object, a later in-place
    edit — a challenger round rewriting labels, a bank being reused — would
    silently change which partition a stored fit is understood to describe,
    while the fitted appearance parameters still belonged to the old partition.
    Freezing at fit time makes the fitted identity immutable from outside.
    """
    def clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
        clone = tensor.detach().clone()
        if device is not None:
            clone = clone.to(device=device)
        return clone

    def clone_nested(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return clone_tensor(value)
        if isinstance(value, dict):
            return {copy.deepcopy(key): clone_nested(nested) for key, nested in value.items()}
        if isinstance(value, list):
            return [clone_nested(nested) for nested in value]
        if isinstance(value, tuple):
            return tuple(clone_nested(nested) for nested in value)
        return copy.deepcopy(value)

    return Hypothesis(
        candidate_id=hypothesis.candidate_id,
        labels=clone_tensor(hypothesis.labels),
        probabilities=clone_tensor(hypothesis.probabilities),
        validity=clone_tensor(hypothesis.validity),
        source=hypothesis.source,
        semantic_unresolved=hypothesis.semantic_unresolved,
        alternatives=clone_nested(hypothesis.alternatives),
        metadata=clone_nested(hypothesis.metadata),
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


def _prepared_digest(
    nll_per_pixel: torch.Tensor,
    support: torch.Tensor,
    *,
    nll_sum: float,
    count: int,
    normalized_nll: float | None,
    complexity: float,
    prior: float,
    total: float | None,
    role: str,
    available: bool,
    reason: str | None,
    study_id: str,
    unit_id: str,
    partition_id: str,
    recipe_snapshot: tuple[tuple[str, Any], ...],
    hypothesis_digest: str,
) -> str:
    """Digest prepared scalar metadata and private tensors for tamper detection."""
    digest = hashlib.sha256()
    for tensor in (nll_per_pixel, support):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(repr(tuple(contiguous.shape)).encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(contiguous.numpy().tobytes())
    values = (
        nll_sum,
        count,
        normalized_nll,
        complexity,
        prior,
        total,
        role,
        available,
        reason,
        study_id,
        unit_id,
        partition_id,
        recipe_snapshot,
        hypothesis_digest,
    )
    digest.update(repr(values).encode("utf-8"))
    return digest.hexdigest()


def _freeze_recipe_value(value: Any) -> Any:
    """Recursively detach recipe values so prepared snapshots never alias dicts/lists."""
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_recipe_value(nested))
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_recipe_value(nested) for nested in value)
    if isinstance(value, set):
        return frozenset(_freeze_recipe_value(nested) for nested in value)
    return value


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


def _log_density_batch(
    values: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    distribution: Distribution,
    normalizers: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized counterpart of :func:`_log_density` in float64.

    ``values`` is ``[B,N]`` and ``means``/``variances`` are ``[B,M,1]``;
    the result is ``[B,M,N]``.  The expression deliberately mirrors the scalar
    implementation so only reduction order, not the density definition, can
    differ between paths.
    """
    deviation = values[:, None, :] - means
    # The scalar reference uses Python ``math.log`` on each component scale.
    # Callers pass normalizers built from the frozen Python component values, so
    # this hot path never synchronously copies a GPU tensor to the host.  The
    # fallback keeps this helper self-contained for direct unit tests.
    if normalizers is None:
        normalizer_values: list[float] = []
        for value in variances.detach().to(torch.float64).reshape(-1).tolist():
            if distribution == "gaussian":
                normalizer_values.append(-0.5 * math.log(2.0 * math.pi * value))
            else:
                dof = STUDENT_T_DOF
                normalizer_values.append(
                    math.lgamma((dof + 1.0) / 2.0)
                    - math.lgamma(dof / 2.0)
                    - 0.5 * math.log(dof * math.pi * value)
                )
        normalizers = torch.tensor(
            normalizer_values,
            dtype=torch.float64,
            device=variances.device,
        ).reshape_as(variances)
    if distribution == "gaussian":
        return normalizers - deviation.pow(2) / (2.0 * variances)
    dof = STUDENT_T_DOF
    return normalizers - ((dof + 1.0) / 2.0) * torch.log1p(
        deviation.pow(2) / (dof * variances)
    )


def _coerce_views(
    views: Any,
    length: int,
    expected_type: type,
    name: str,
) -> list[Any]:
    """Normalize optional shared view convenience to a corresponding list."""
    if isinstance(views, expected_type):
        return [views] * length
    try:
        result = list(views)
    except TypeError as exc:
        raise ObservationContractError(
            f"{name} must be a {expected_type.__name__} or sequence of them"
        ) from exc
    if len(result) != length:
        raise ObservationContractError(
            f"{name} sequence length {len(result)} does not match input length {length}"
        )
    if any(not isinstance(view, expected_type) for view in result):
        raise ObservationContractError(
            f"{name} sequence entries must be {expected_type.__name__}"
        )
    return result


def _components_from_parameters(fitted: FittedHypothesis) -> tuple[_Component, ...]:
    """Decode and validate the frozen component snapshot for fast score paths."""
    raw_components = fitted.parameters.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ObservationContractError("fitted hypothesis carries no component parameters")
    components: list[_Component] = []
    for entry in raw_components:
        if not isinstance(entry, dict):
            raise ObservationContractError("fitted component parameters are malformed")
        try:
            component = _Component(
                region=int(entry["region"]),
                mean=float(entry["mean"]),
                variance=float(entry["variance"]),
                log_weight=float(entry["log_weight"]),
                fallback=bool(entry["fallback"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ObservationContractError("fitted component parameters are malformed") from exc
        if component.region not in range(NUM_REGIONS):
            raise ObservationContractError("fitted component region is outside 0..3")
        if not all(
            math.isfinite(value)
            for value in (component.mean, component.variance, component.log_weight)
        ) or component.variance <= 0.0:
            raise ObservationContractError("fitted component parameters must be finite")
        components.append(component)
    return tuple(components)


def _decode_fitting_support(
    fitted: FittedHypothesis,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Decode the packed fit support and enforce shape/encoding invariants."""
    support_bits = fitted.parameters.get("fit_support_bits")
    if not isinstance(support_bits, str) or fitted.parameters.get("fit_support_shape") != [height, width]:
        raise ObservationContractError("missing or incompatible fitting support")
    try:
        packed = np.frombuffer(bytes.fromhex(support_bits), dtype=np.uint8)
    except (TypeError, ValueError) as exc:
        raise ObservationContractError("invalid fitting support encoding") from exc
    if len(packed) != (height * width + 7) // 8:
        raise ObservationContractError("invalid fitting support encoding")
    unpacked = np.unpackbits(packed, count=height * width).astype(bool).reshape(height, width)
    return torch.from_numpy(unpacked).to(device=device)


def _validate_score_context(
    model: ObservationModel,
    fitted: FittedHypothesis,
    scoring_view: ScoringView,
) -> _ScoreContext:
    """Apply scalar score guards once for a prepared/bulk score operation."""
    if not isinstance(fitted, FittedHypothesis):
        raise ObservationContractError("score requires a contracts.FittedHypothesis")
    if not isinstance(scoring_view, ScoringView):
        raise ObservationContractError("score requires a contracts.ScoringView")
    model._validate_execution_view_device(scoring_view)
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

    expected = model.recipe()
    recorded = fitted.metadata.get("recipe")
    if not isinstance(recorded, dict):
        raise ObservationContractError(
            "fitted hypothesis carries no recipe record; it was not produced by this "
            "ObservationModel version and cannot be scored"
        )
    differing = sorted(key for key in expected if recorded.get(key, "<missing>") != expected[key])
    if differing:
        details = ", ".join(
            f"{key}: fit={recorded.get(key, '<missing>')!r} vs model={expected[key]!r}"
            for key in differing
        )
        raise ObservationContractError(
            "recipe mismatch, candidates must be compared under one identical recipe "
            f"({details})"
        )
    if fitted.capacity != model.capacity:
        raise ObservationContractError(
            f"capacity mismatch: fit used {fitted.capacity}, this model has {model.capacity}"
        )

    height, width = scoring_view.support.shape
    fitting_support = _decode_fitting_support(
        fitted, height, width, scoring_view.support.device
    )
    support = scoring_view.support.detach()
    if torch.any(fitting_support & support):
        raise ObservationContractError("scoring observations overlap fitting observations")

    hypothesis = fitted.hypothesis
    compute_hypothesis = (
        hypothesis
        if not model._uses_cuda_execution
        else _freeze_hypothesis(hypothesis, device=torch.device("cpu"))
    )
    _check_hypothesis(compute_hypothesis, height, width)
    recorded_digest = fitted.metadata.get("hypothesis_digest")
    if not isinstance(recorded_digest, str):
        raise ObservationContractError(
            "fitted hypothesis carries no identity digest; refusing to score an "
            "unverifiable partition"
        )
    if _hypothesis_digest(hypothesis) != recorded_digest:
        raise ObservationContractError(
            "fitted hypothesis was mutated after fitting; its appearance parameters "
            "belong to the partition that was fitted, not to this one. Refit instead."
        )

    complexity = _neighbor_disagreement(compute_hypothesis.labels) + float(
        fitted.metadata.get("component_penalty", 0.0)
    )
    prior = _prior_penalty(compute_hypothesis)
    components = _components_from_parameters(fitted)
    raw_bias = fitted.parameters.get("bias_coefficients")
    if not isinstance(raw_bias, (list, tuple)) or len(raw_bias) != 3:
        raise ObservationContractError("fitted hypothesis carries malformed bias coefficients")
    try:
        target_device = (
            model.execution_device
            if model._uses_cuda_execution
            else scoring_view.image.device
        )
        bias_coefficients = torch.tensor(
            [float(value) for value in raw_bias], dtype=torch.float64,
            device=target_device,
        )
    except (TypeError, ValueError) as exc:
        raise ObservationContractError("fitted hypothesis carries malformed bias coefficients") from exc
    if not torch.isfinite(bias_coefficients).all():
        raise ObservationContractError("fitted bias coefficients must be finite")
    source_support = support.detach()
    compute_support = (
        source_support
        if not model._uses_cuda_execution
        else source_support.to(device=target_device)
    )
    return _ScoreContext(
        fitted=fitted,
        scoring_view=scoring_view,
        hypothesis=compute_hypothesis,
        complexity=complexity,
        prior=prior,
        support=compute_support,
        count=int(source_support.sum().item()),
        components=components,
        bias_coefficients=bias_coefficients,
        height=height,
        width=width,
    )


def _component_batch_tensors(
    components: Sequence[Sequence[_Component]],
    device: torch.device,
    max_modes: int,
    distribution: Distribution,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack per-candidate components as ``[B,4,M]`` float64 tensors."""
    batch = len(components)
    means = torch.zeros((batch, NUM_REGIONS, max_modes), dtype=torch.float64, device=device)
    variances = torch.ones_like(means)
    log_weights = torch.full_like(means, float("-inf"))
    normalizers = torch.zeros_like(means)
    seen = [[0] * NUM_REGIONS for _ in range(batch)]
    for index, candidate_components in enumerate(components):
        for component in candidate_components:
            mode = seen[index][component.region]
            if mode >= max_modes:
                raise ObservationContractError("fitted component count exceeds model capacity")
            means[index, component.region, mode] = component.mean
            variances[index, component.region, mode] = component.variance
            log_weights[index, component.region, mode] = component.log_weight
            if distribution == "gaussian":
                normalizers[index, component.region, mode] = -0.5 * math.log(
                    2.0 * math.pi * component.variance
                )
            else:
                dof = STUDENT_T_DOF
                normalizers[index, component.region, mode] = (
                    math.lgamma((dof + 1.0) / 2.0)
                    - math.lgamma(dof / 2.0)
                    - 0.5 * math.log(dof * math.pi * component.variance)
                )
            seen[index][component.region] += 1
        if any(count == 0 for count in seen[index]):
            raise ObservationContractError("fitted component parameters omit a semantic region")
    return means, variances, log_weights, normalizers


def _batch_log_mixture(
    residual: torch.Tensor,
    labels: torch.Tensor,
    components: Sequence[Sequence[_Component]],
    distribution: Distribution,
    max_modes: int,
) -> torch.Tensor:
    """Return per-pixel log mixture for ``residual``/``labels`` shaped ``[B,N]``."""
    means, variances, log_weights, normalizers = _component_batch_tensors(
        components, residual.device, max_modes, distribution
    )
    values = residual

    # Every pixel has one hard semantic label, so evaluating all four region
    # mixtures is redundant.  Gather each pixel's region parameters first and
    # evaluate one ``[B,M,N]`` density tensor.  The gather keeps candidate and
    # mode axes independent, while preserving the existing component ordering
    # (and therefore the same logsumexp reduction order).
    valid_labels = (labels >= 0) & (labels < NUM_REGIONS)
    if labels.is_floating_point():
        integer_labels = labels.to(dtype=torch.long)
        valid_labels = valid_labels & torch.isfinite(labels) & (labels == integer_labels)
    else:
        integer_labels = labels.to(dtype=torch.long)
    safe_labels = torch.where(valid_labels, integer_labels, torch.zeros_like(integer_labels))
    region_index = safe_labels.unsqueeze(-1).expand(-1, -1, max_modes)
    selected_means = means.gather(1, region_index).transpose(1, 2)
    selected_variances = variances.gather(1, region_index).transpose(1, 2)
    selected_log_weights = log_weights.gather(1, region_index).transpose(1, 2)
    selected_normalizers = normalizers.gather(1, region_index).transpose(1, 2)
    density = _log_density_batch(
        values,
        selected_means,
        selected_variances,
        distribution,
        selected_normalizers,
    )
    density = density + selected_log_weights
    output = torch.logsumexp(density, dim=1)
    if not bool(valid_labels.all()):
        output = torch.where(valid_labels, output, torch.full_like(output, float("nan")))
    if torch.isnan(output).any():
        raise ObservationContractError("unscored pixels remain in the mixture density")
    return output


def _batch_log_mixture_tensors(
    residual: torch.Tensor,
    labels: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    log_weights: torch.Tensor,
    normalizers: torch.Tensor,
    distribution: Distribution,
    max_modes: int,
) -> torch.Tensor:
    """Evaluate a mixture from already-device-resident component tensors.

    ``_batch_log_mixture`` intentionally accepts the portable Python component
    dictionaries stored in a :class:`FittedHypothesis`.  The explicit
    ``execution_device`` fitting path keeps the corresponding sufficient
    statistics as tensors throughout all fitting iterations and calls this
    helper only once, at the operation boundary.  Keeping this expression in a
    single helper avoids a second density implementation while avoiding any
    candidate-wise scalar device transfers in the hot loop.
    """
    if residual.ndim != 2 or labels.ndim != 2:
        raise ObservationContractError("batched mixture inputs must be [B,N]")
    if tuple(residual.shape) != tuple(labels.shape):
        raise ObservationContractError("batched mixture residual/labels shape mismatch")
    batch_size, _pixel_count = residual.shape
    expected_shape = (batch_size, NUM_REGIONS, max_modes)
    for name, tensor in (
        ("means", means),
        ("variances", variances),
        ("log_weights", log_weights),
        ("normalizers", normalizers),
    ):
        if tuple(tensor.shape) != expected_shape:
            raise ObservationContractError(
                f"batched mixture {name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
            )
    valid_labels = (labels >= 0) & (labels < NUM_REGIONS)
    if labels.is_floating_point():
        integer_labels = labels.to(dtype=torch.long)
        valid_labels = valid_labels & torch.isfinite(labels) & (labels == integer_labels)
    else:
        integer_labels = labels.to(dtype=torch.long)
    safe_labels = torch.where(valid_labels, integer_labels, torch.zeros_like(integer_labels))
    region_index = safe_labels.unsqueeze(-1).expand(-1, -1, max_modes)
    selected_means = means.gather(1, region_index).transpose(1, 2)
    selected_variances = variances.gather(1, region_index).transpose(1, 2)
    selected_log_weights = log_weights.gather(1, region_index).transpose(1, 2)
    selected_normalizers = normalizers.gather(1, region_index).transpose(1, 2)
    density = _log_density_batch(
        residual,
        selected_means,
        selected_variances,
        distribution,
        selected_normalizers,
    )
    density = density + selected_log_weights
    output = torch.logsumexp(density, dim=1)
    if not bool(valid_labels.all()):
        output = torch.where(valid_labels, output, torch.full_like(output, float("nan")))
    if torch.isnan(output).any():
        raise ObservationContractError("unscored pixels remain in the mixture density")
    return output


def _score_from_nll(
    *,
    nll_sum: float,
    count: int,
    complexity: float,
    prior: float,
    beta: float,
    role: str,
    reason: str | None = None,
) -> EvidenceScore:
    """Build the canonical score fields from a reduction."""
    if count <= 0:
        return EvidenceScore(
            nll_sum=0.0,
            count=0,
            normalized_nll=None,
            complexity=complexity,
            prior=prior,
            total=None,
            role=role,
            available=False,
            reason=reason or "no observed pixels in this scoring role for this unit",
        )
    normalized = nll_sum / float(count)
    return EvidenceScore(
        nll_sum=nll_sum,
        count=count,
        normalized_nll=normalized,
        complexity=complexity,
        prior=prior,
        total=normalized + beta * complexity + prior,
        role=role,
        available=True,
        reason=None,
    )


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
    execution_device:
        Optional explicit device for detached tensor-heavy fitting and scoring
        math. ``None`` and explicit ``'cpu'`` preserve the legacy CPU reference
        path (CPU capability views are required for the latter); explicit CUDA
        selects the tensorized path. CUDA fits serialize portable CPU
        hypotheses and Python parameter values, and execution placement is not
        part of the comparison recipe.

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
        execution_device: str | torch.device | None = None,
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
        if execution_device is not None and not isinstance(execution_device, (str, torch.device)):
            raise ObservationContractError(
                "execution_device must be None, a device string, or torch.device"
            )
        try:
            normalized_device = (
                None if execution_device is None else torch.device(execution_device)
            )
        except (TypeError, RuntimeError, ValueError) as exc:
            raise ObservationContractError("execution_device is not a valid torch device") from exc
        if normalized_device is not None and normalized_device.type not in ("cpu", "cuda"):
            raise ObservationContractError(
                "execution_device supports only CPU and CUDA devices"
            )
        if normalized_device is not None and normalized_device.type == "cuda":
            if normalized_device.index is not None and normalized_device.index < 0:
                raise ObservationContractError("execution_device CUDA index must be non-negative")
            if not torch.cuda.is_available():
                raise ObservationContractError(
                    "execution_device CUDA is unavailable; refusing CPU fallback"
                )
            if (
                normalized_device.index is not None
                and normalized_device.index >= torch.cuda.device_count()
            ):
                raise ObservationContractError(
                    "execution_device CUDA index is unavailable; refusing fallback"
                )
        self.max_iterations = max_iterations
        self.variance_floor = float(variance_floor)
        self.beta = float(beta)
        self.background_modes = int(background_modes)
        self.bias_ridge = float(bias_ridge)
        self.distribution: Distribution = distribution
        # ``None`` and explicit CPU retain the legacy reference arithmetic;
        # explicit CUDA opts into the detached tensorized path.
        self.execution_device = normalized_device

    def _target_device(self, input_device: torch.device) -> torch.device:
        """Resolve the heavy-math device without changing the legacy default."""
        return self.execution_device if self._uses_cuda_execution else input_device

    @property
    def _uses_cuda_execution(self) -> bool:
        """Whether public dispatch should use the explicit tensorized CUDA path."""
        return self.execution_device is not None and self.execution_device.type == "cuda"

    def _validate_execution_view_device(
        self,
        view: FittingView | ScoringView,
    ) -> None:
        """Reject capability tensors that would silently bypass explicit CPU placement."""
        if self.execution_device is None or self.execution_device.type != "cpu":
            return
        tensors: list[torch.Tensor] = [view.image, view.support]
        if isinstance(view, FittingView):
            tensors.append(view.context)
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise ObservationContractError(
                "execution_device='cpu' requires CPU capability view tensors"
            )

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
        self._validate_execution_view_device(fitting_view)
        fitting_view.validate()
        if fitting_view.protocol not in SUPPORTED_PROTOCOLS:
            raise ObservationContractError(
                f"protocol {fitting_view.protocol!r} is not implemented by ObservationModel; "
                "cine_predictive needs an explicit fit-time temporal deformation model and "
                "must not be approximated by a spatial fit"
            )
        height, width = fitting_view.support.shape
        if self._uses_cuda_execution:
            # Keep validation/topology and the frozen public hypothesis portable
            # even when the fitting view arrives on CUDA.  The tensorized helper
            # below moves only detached working arrays to ``execution_device``.
            if not isinstance(hypothesis, Hypothesis):
                raise ObservationContractError("fit/score require a contracts.Hypothesis")
            frozen = _freeze_hypothesis(hypothesis, device=torch.device("cpu"))
            _check_hypothesis(frozen, height, width)
            return self._fit_many_group([frozen], [fitting_view])[0]
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

    # ------------------------------------------------------------------
    # batched fitting
    # ------------------------------------------------------------------
    def fit_many(
        self,
        hypotheses: Sequence[Hypothesis],
        fitting_views: FittingView | Sequence[FittingView],
    ) -> list[FittedHypothesis]:
        """Fit corresponding candidates/views with deterministic float64 batching.

        The returned list preserves input order.  Compatible geometry/device
        groups share only temporary image/basis tensors; every candidate still
        gets a private frozen hypothesis, component dictionary, digest and fit
        score.  Heterogeneous groups use the scalar reference method, which is
        intentionally an explicit fallback rather than a silent contract
        change.
        """
        try:
            candidates = list(hypotheses)
        except TypeError as exc:
            raise ObservationContractError("hypotheses must be a sequence") from exc
        if not candidates:
            # A shared view is not meaningful for an empty bank, but accepting
            # it keeps the ordered bulk API total and unsurprising.
            return []
        views = _coerce_views(fitting_views, len(candidates), FittingView, "fitting_views")
        frozen: list[Hypothesis] = []
        for candidate, view in zip(candidates, views):
            if not isinstance(view, FittingView):  # pragma: no cover - _coerce_views checks
                raise ObservationContractError("fit_many requires FittingView inputs")
            self._validate_execution_view_device(view)
            view.validate()
            if view.protocol not in SUPPORTED_PROTOCOLS:
                raise ObservationContractError(
                    f"protocol {view.protocol!r} is not implemented by ObservationModel; "
                    "cine_predictive needs an explicit fit-time temporal deformation model "
                    "and must not be approximated by a spatial fit"
                )
            height, width = view.support.shape
            if not self._uses_cuda_execution:
                _check_hypothesis(candidate, height, width)
                frozen.append(_freeze_hypothesis(candidate))
            else:
                if not isinstance(candidate, Hypothesis):
                    raise ObservationContractError("fit/score require a contracts.Hypothesis")
                frozen_candidate = _freeze_hypothesis(candidate, device=torch.device("cpu"))
                _check_hypothesis(frozen_candidate, height, width)
                frozen.append(frozen_candidate)

        groups: dict[tuple[Any, ...], list[int]] = {}
        for index, view in enumerate(views):
            device = view.image.device
            # Scalar fit requires image/support indexing on one device.  Keep a
            # mismatched entry on the explicit scalar fallback path.
            if view.support.device != device:
                key = ("scalar", index)
            else:
                key = (
                    tuple(view.support.shape),
                    device.type,
                    device.index,
                    view.protocol,
                    str(view.image.dtype),
                )
            groups.setdefault(key, []).append(index)

        output: list[FittedHypothesis | None] = [None] * len(candidates)
        for key, indices in groups.items():
            if key[0] == "scalar" or len(indices) == 1:
                for index in indices:
                    output[index] = self.fit(candidates[index], views[index])
                continue
            batch = self._fit_many_group(
                [frozen[index] for index in indices],
                [views[index] for index in indices],
            )
            if len(batch) != len(indices):
                raise ObservationContractError(
                    "fit_many fast group returned an unexpected result count"
                )
            for index, fitted in zip(indices, batch):
                output[index] = fitted
        # Every slot is filled above; the assertion keeps type checkers honest
        # without exposing a mutable partial result.
        if any(entry is None for entry in output):
            raise ObservationContractError("fit_many did not fill every ordered result slot")
        return cast(list[FittedHypothesis], output)

    def _fit_many_group(
        self,
        hypotheses: Sequence[Hypothesis],
        fitting_views: Sequence[FittingView],
    ) -> list[FittedHypothesis]:
        """Batch the heavy region/bias equations for one compatible view group."""
        if not hypotheses:
            return []
        if self._uses_cuda_execution:
            return self._fit_many_group_on_device(hypotheses, fitting_views)
        height, width = fitting_views[0].support.shape
        device = fitting_views[0].image.device
        batch_size = len(hypotheses)
        images = torch.stack(
            [view.image.detach().to(torch.float64)[0] for view in fitting_views], dim=0
        ).reshape(batch_size, -1)
        supports = torch.stack(
            [view.support.detach() for view in fitting_views], dim=0
        ).reshape(batch_size, -1)
        labels = torch.stack(
            [hypothesis.labels.detach().to(device=device) for hypothesis in hypotheses], dim=0
        ).reshape(batch_size, -1)
        basis_full = _plane_basis(height, width, device)
        observed = supports.sum(dim=1).to(torch.float64)
        if bool((observed <= 0).any()):  # pragma: no cover - view.validate rejects this
            raise ObservationContractError("fitting view has empty support")

        # Keep the scalar ``Tensor.mean``/``Tensor.var`` reduction algorithm for
        # each candidate's global fallback statistics.  These four tiny loops
        # avoid silently changing the variance estimator while the bias/density
        # work below remains batched.
        global_mean_values: list[float] = []
        global_variance_values: list[float] = []
        for index in range(batch_size):
            selected = images[index][supports[index]]
            mean = _as_float(selected.mean())
            variance = max(
                _as_float(selected.var(unbiased=False)) if selected.numel() > 1 else 0.0,
                self.variance_floor,
            )
            global_mean_values.append(mean)
            global_variance_values.append(variance)
        global_mean_tensor = torch.tensor(
            global_mean_values, dtype=torch.float64, device=device
        )
        global_variance_tensor = torch.tensor(
            global_variance_values, dtype=torch.float64, device=device
        )
        global_mean = [float(value) for value in global_mean_tensor.tolist()]
        global_variance = [float(value) for value in global_variance_tensor.tolist()]

        bias_coefficients = torch.zeros((batch_size, 3), dtype=torch.float64, device=device)
        components_batch: list[list[_Component]] = [[] for _ in range(batch_size)]
        region_counts_batch: list[dict[int, int]] = [
            {region: 0 for region in range(NUM_REGIONS)} for _ in range(batch_size)
        ]
        fallback_batch: list[list[int]] = [[] for _ in range(batch_size)]

        for _ in range(self.max_iterations):
            residual = images - (basis_full @ bias_coefficients.transpose(0, 1)).transpose(0, 1)
            components_batch = [[] for _ in range(batch_size)]
            region_means = torch.empty(
                (batch_size, NUM_REGIONS), dtype=torch.float64, device=device
            )
            region_variances = torch.empty_like(region_means)
            region_counts_batch = [
                {region: 0 for region in range(NUM_REGIONS)} for _ in range(batch_size)
            ]
            fallback_batch = [[] for _ in range(batch_size)]

            for region in range(NUM_REGIONS):
                region_mask = supports & (labels == region)
                counts = region_mask.sum(dim=1)
                means_values: list[float] = []
                variance_values: list[float] = []
                for index in range(batch_size):
                    selected = residual[index][region_mask[index]]
                    count = int(selected.numel())
                    if count < MIN_REGION_PIXELS:
                        means_values.append(global_mean[index])
                        variance_values.append(global_variance[index])
                    else:
                        means_values.append(_as_float(selected.mean()))
                        variance_values.append(
                            max(_as_float(selected.var(unbiased=False)), self.variance_floor)
                        )
                means = torch.tensor(means_values, dtype=torch.float64, device=device)
                variances = torch.tensor(variance_values, dtype=torch.float64, device=device)
                region_means[:, region] = means
                region_variances[:, region] = variances

                for index in range(batch_size):
                    count = int(counts[index].item())
                    region_counts_batch[index][region] = count
                    modes = self.background_modes if region == 0 else 1
                    if count < MIN_REGION_PIXELS:
                        fallback_batch[index].append(region)
                        for _mode in range(modes):
                            components_batch[index].append(
                                _Component(
                                    region=region,
                                    mean=global_mean[index],
                                    variance=global_variance[index],
                                    log_weight=-math.log(modes),
                                    fallback=True,
                                )
                            )
                    elif modes == 1:
                        components_batch[index].append(
                            _Component(
                                region=region,
                                mean=float(means[index].item()),
                                variance=float(variances[index].item()),
                                log_weight=0.0,
                                fallback=False,
                            )
                        )
                    else:
                        selected = residual[index][region_mask[index]]
                        background_components = self._fit_background_modes(
                            selected, global_variance[index]
                        )
                        components_batch[index].extend(background_components)
                        weight_sum = sum(math.exp(component.log_weight) for component in background_components)
                        region_means[index, region] = sum(
                            math.exp(component.log_weight) * component.mean
                            for component in background_components
                        ) / weight_sum
                        region_variances[index, region] = sum(
                            math.exp(component.log_weight) * component.variance
                            for component in background_components
                        ) / weight_sum

            # Match scalar fit's reconstructed-value expression exactly rather
            # than reusing the original image tensor after the residual solve.
            reconstructed = residual + (
                basis_full @ bias_coefficients.transpose(0, 1)
            ).transpose(0, 1)
            bias_coefficients = self._fit_bias_batch(
                reconstructed,
                labels,
                supports,
                basis_full,
                region_means,
                region_variances,
            )

        residual = images - (basis_full @ bias_coefficients.transpose(0, 1)).transpose(0, 1)
        log_mixture = _batch_log_mixture(
            residual,
            labels,
            components_batch,
            self.distribution,
            self.background_modes,
        )
        nll = -log_mixture
        nll_sum_tensor = torch.where(supports, nll, torch.zeros_like(nll)).sum(dim=1)

        result: list[FittedHypothesis] = []
        component_penalty = COMPONENT_PENALTY * float(self.capacity - NUM_REGIONS)
        for index, (hypothesis, view) in enumerate(zip(hypotheses, fitting_views)):
            observed_count = int(observed[index].item())
            nll_sum = float(nll_sum_tensor[index].item())
            complexity = _neighbor_disagreement(hypothesis.labels) + component_penalty
            prior = _prior_penalty(hypothesis)
            normalized = nll_sum / float(observed_count)
            fit_score = EvidenceScore(
                nll_sum=nll_sum,
                count=observed_count,
                normalized_nll=normalized,
                complexity=complexity,
                prior=prior,
                total=normalized + self.beta * complexity + prior,
                role="fit",
                available=True,
                reason=None,
            )
            parameters: dict[str, Any] = {
                "fit_support_shape": list(view.support.shape),
                "fit_support_bits": np.packbits(view.support.detach().cpu().numpy()).tobytes().hex(),
                "bias_coefficients": [float(value) for value in bias_coefficients[index].tolist()],
                "bias_ridge": self.bias_ridge,
                "components": [
                    {
                        "region": component.region,
                        "mean": component.mean,
                        "variance": component.variance,
                        "log_weight": component.log_weight,
                        "fallback": component.fallback,
                    }
                    for component in components_batch[index]
                ],
                "distribution": self.distribution,
                "student_t_dof": STUDENT_T_DOF if self.distribution == "student_t" else None,
                "variance_floor": self.variance_floor,
                "global_mean": global_mean[index],
                "global_variance": global_variance[index],
                "region_counts": dict(region_counts_batch[index]),
                "fallback_regions": list(fallback_batch[index]),
                "fallback_region_count": len(fallback_batch[index]),
                "observed_fit_pixels": observed_count,
            }
            result.append(
                FittedHypothesis(
                    hypothesis=hypothesis,
                    parameters=parameters,
                    fit_score=fit_score,
                    fitting_steps=self.max_iterations,
                    capacity=self.capacity,
                    study_id=view.study_id,
                    unit_id=view.unit_id,
                    partition_id=view.partition_id,
                    metadata={
                        "contract_version": VERSION,
                        "protocol": view.protocol,
                        "beta": self.beta,
                        "background_modes": self.background_modes,
                        "component_penalty": component_penalty,
                        "max_iterations": self.max_iterations,
                        "temporal_support": self.supports_temporal,
                        "recipe": self.recipe(),
                        "hypothesis_digest": _hypothesis_digest(hypothesis),
                    },
                )
            )
        return result

    def _fit_background_modes_batch_on_device(
        self,
        values: torch.Tensor,
        support: torch.Tensor,
        global_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fit the deterministic two-mode background mixture without host syncs.

        ``values`` is a full ``[B,N]`` residual grid and ``support`` selects the
        background pixels for each candidate.  The scalar reference uses
        ``torch.quantile`` over a ragged selection; sorting masked rows and
        applying the same linear interpolation gives the equivalent quartiles
        while keeping all five fitting iterations on the execution device.
        Rows with fewer than ``MIN_REGION_PIXELS`` are replaced by the caller's
        global fallback, so their temporary masked values are never serialized.
        """
        batch_size, pixel_count = values.shape
        device = values.device
        dtype = values.dtype
        counts = support.sum(dim=1).to(dtype=torch.long)
        safe_counts = counts.clamp_min(1)
        masked = torch.where(
            support,
            values,
            torch.full_like(values, float("inf")),
        )
        sorted_values = torch.sort(masked, dim=1).values
        quantile_values: list[torch.Tensor] = []
        for quantile in (0.25, 0.75):
            position = (safe_counts.to(dtype) - 1.0) * quantile
            lower = position.floor().to(dtype=torch.long)
            upper = position.ceil().to(dtype=torch.long)
            fraction = position - lower.to(dtype)
            lower_value = sorted_values.gather(1, lower.unsqueeze(1)).squeeze(1)
            upper_value = sorted_values.gather(1, upper.unsqueeze(1)).squeeze(1)
            quantile_values.append(lower_value + fraction * (upper_value - lower_value))
        centers = torch.stack(quantile_values, dim=1)
        spread = global_variance.clamp_min(self.variance_floor).sqrt()
        degenerate = (~torch.isfinite(centers).all(dim=1)) | (
            centers[:, 1] - centers[:, 0] <= 0.0
        )
        global_mean = torch.where(
            support,
            values,
            values.new_zeros(()),
        ).sum(dim=1) / safe_counts.to(dtype)
        fallback_centers = torch.stack(
            (global_mean - spread, global_mean + spread), dim=1
        )
        centers = torch.where(degenerate.unsqueeze(1), fallback_centers, centers)

        assignment = torch.zeros((batch_size, pixel_count), dtype=torch.long, device=device)
        for _ in range(3):
            distances = torch.stack(
                (
                    (values - centers[:, 0].unsqueeze(1)).abs(),
                    (values - centers[:, 1].unsqueeze(1)).abs(),
                ),
                dim=1,
            )
            assignment = distances.argmin(dim=1)
            mode_masks = torch.stack(
                (support & (assignment == 0), support & (assignment == 1)), dim=1
            )
            mode_counts = mode_masks.sum(dim=2)
            mode_safe = mode_counts.clamp_min(1).to(dtype)
            mode_values = values.unsqueeze(1) * mode_masks.to(dtype)
            mode_means = mode_values.sum(dim=2) / mode_safe
            enough = mode_counts >= MIN_REGION_PIXELS
            centers = torch.where(enough, mode_means, centers)

        mode_masks = torch.stack(
            (support & (assignment == 0), support & (assignment == 1)), dim=1
        )
        mode_counts = mode_masks.sum(dim=2)
        mode_safe = mode_counts.clamp_min(1).to(dtype)
        mode_values = values.unsqueeze(1) * mode_masks.to(dtype)
        raw_means = mode_values.sum(dim=2) / mode_safe
        enough = mode_counts >= MIN_REGION_PIXELS
        means = torch.where(enough, raw_means, centers)
        deviations = values.unsqueeze(1) - means.unsqueeze(2)
        raw_variances = (
            deviations.pow(2) * mode_masks.to(dtype)
        ).sum(dim=2) / mode_safe
        variances = torch.where(
            enough,
            raw_variances.clamp_min(self.variance_floor),
            global_variance.unsqueeze(1).expand(-1, 2),
        )
        log_weights = torch.log(
            mode_counts.clamp_min(1).to(dtype)
            / counts.clamp_min(1).to(dtype).unsqueeze(1)
        )
        fallback = ~enough
        return means, variances, log_weights, fallback

    def _fit_many_group_on_device(
        self,
        hypotheses: Sequence[Hypothesis],
        fitting_views: Sequence[FittingView],
        device: torch.device | None = None,
    ) -> list[FittedHypothesis]:
        """Tensorized explicit-device fitting with one host serialization boundary."""
        if not hypotheses:
            return []
        height, width = fitting_views[0].support.shape
        if device is None:
            device = self.execution_device or fitting_views[0].image.device
        batch_size = len(hypotheses)
        # Stack on the source device first, then perform one bulk transfer per
        # logical array.  This avoids one ``.to(device)`` call for every
        # candidate when a CPU bank is dispatched to CUDA.
        images = torch.stack(
            [view.image.detach()[0] for view in fitting_views], dim=0
        ).reshape(batch_size, -1).to(device=device, dtype=torch.float64)
        supports = torch.stack(
            [view.support.detach() for view in fitting_views], dim=0
        ).reshape(batch_size, -1).to(device=device)
        labels = torch.stack(
            [hypothesis.labels.detach() for hypothesis in hypotheses], dim=0
        ).reshape(batch_size, -1).to(device=device, dtype=torch.long)
        basis = _plane_basis(height, width, device)
        dtype = torch.float64
        support_weights = supports.to(dtype=dtype)
        observed = support_weights.sum(dim=1)
        if bool((observed <= 0).any()):  # validation should have caught this on the source device
            raise ObservationContractError("fitting view has empty support")
        safe_observed = observed.clamp_min(1.0)
        global_mean_tensor = (images * support_weights).sum(dim=1) / safe_observed
        global_variance_tensor = (
            (images - global_mean_tensor.unsqueeze(1)).pow(2) * support_weights
        ).sum(dim=1) / safe_observed
        global_variance_tensor = global_variance_tensor.clamp_min(self.variance_floor)

        bias_coefficients = torch.zeros((batch_size, 3), dtype=dtype, device=device)
        region_ids = torch.arange(NUM_REGIONS, dtype=torch.long, device=device).view(1, -1, 1)
        mode_count = self.background_modes
        component_means = torch.zeros(
            (batch_size, NUM_REGIONS, mode_count), dtype=dtype, device=device
        )
        component_variances = torch.ones_like(component_means)
        component_log_weights = torch.full_like(component_means, float("-inf"))
        component_fallback = torch.zeros(
            (batch_size, NUM_REGIONS, mode_count), dtype=torch.bool, device=device
        )
        # Labels/supports are fixed for every fitting pass.  Build their
        # region masks and sufficient-count tensors once, outside the hot loop;
        # only residual-dependent moments and the bias solve are repeated.
        region_masks = supports.unsqueeze(1) & (labels.unsqueeze(1) == region_ids)
        region_masks_float = region_masks.to(dtype=dtype)
        region_counts = region_masks.sum(dim=2)
        region_safe = region_counts.clamp_min(1).to(dtype)
        enough_regions = region_counts >= MIN_REGION_PIXELS

        for _ in range(self.max_iterations):
            residual = images - (basis @ bias_coefficients.transpose(0, 1)).transpose(0, 1)
            region_values = residual.unsqueeze(1) * region_masks_float
            raw_means = region_values.sum(dim=2) / region_safe
            deviations = residual.unsqueeze(1) - raw_means.unsqueeze(2)
            raw_variances = (
                deviations.pow(2) * region_masks_float
            ).sum(dim=2) / region_safe
            region_means = torch.where(
                enough_regions,
                raw_means,
                global_mean_tensor.unsqueeze(1).expand(-1, NUM_REGIONS),
            )
            region_variances = torch.where(
                enough_regions,
                raw_variances.clamp_min(self.variance_floor),
                global_variance_tensor.unsqueeze(1).expand(-1, NUM_REGIONS),
            )

            component_means.zero_()
            component_variances.fill_(1.0)
            component_log_weights.fill_(float("-inf"))
            component_fallback.zero_()
            component_means[:, :, 0] = region_means
            component_variances[:, :, 0] = region_variances
            component_log_weights[:, :, 0] = 0.0
            component_fallback[:, :, 0] = ~enough_regions

            if mode_count == 2:
                background_mask = region_masks[:, 0]
                bg_means, bg_variances, bg_log_weights, bg_fallback = (
                    self._fit_background_modes_batch_on_device(
                        residual,
                        background_mask,
                        global_variance_tensor,
                    )
                )
                bg_enough = enough_regions[:, 0].unsqueeze(1)
                bg_mean_fallback = global_mean_tensor.unsqueeze(1).expand(-1, 2)
                bg_variance_fallback = global_variance_tensor.unsqueeze(1).expand(-1, 2)
                bg_log_fallback = torch.full_like(bg_log_weights, -math.log(2.0))
                bg_means = torch.where(bg_enough, bg_means, bg_mean_fallback)
                bg_variances = torch.where(bg_enough, bg_variances, bg_variance_fallback)
                bg_log_weights = torch.where(bg_enough, bg_log_weights, bg_log_fallback)
                bg_fallback = torch.where(
                    bg_enough,
                    bg_fallback,
                    torch.ones_like(bg_fallback),
                )
                component_means[:, 0, :] = bg_means
                component_variances[:, 0, :] = bg_variances
                component_log_weights[:, 0, :] = bg_log_weights
                component_fallback[:, 0, :] = bg_fallback
                background_weights = bg_log_weights.exp()
                background_weight_sum = background_weights.sum(dim=1).clamp_min(torch.finfo(dtype).tiny)
                region_means[:, 0] = (
                    (background_weights * bg_means).sum(dim=1) / background_weight_sum
                )
                region_variances[:, 0] = (
                    (background_weights * bg_variances).sum(dim=1) / background_weight_sum
                )

            reconstructed = residual + (
                basis @ bias_coefficients.transpose(0, 1)
            ).transpose(0, 1)
            bias_coefficients = self._fit_bias_batch(
                reconstructed,
                labels,
                supports,
                basis,
                region_means,
                region_variances,
            )

        residual = images - (basis @ bias_coefficients.transpose(0, 1)).transpose(0, 1)
        normalizers = torch.zeros_like(component_means)
        if self.distribution == "gaussian":
            normalizers = -0.5 * torch.log(2.0 * math.pi * component_variances)
        else:
            dof = STUDENT_T_DOF
            normalizers = (
                math.lgamma((dof + 1.0) / 2.0)
                - math.lgamma(dof / 2.0)
                - 0.5 * torch.log(dof * math.pi * component_variances)
            )
        log_mixture = _batch_log_mixture_tensors(
            residual,
            labels,
            component_means,
            component_variances,
            component_log_weights,
            normalizers,
            self.distribution,
            mode_count,
        )
        nll = -log_mixture
        nll_sum_tensor = torch.where(supports, nll, torch.zeros_like(nll)).sum(dim=1)

        # One detached transfer per sufficient-statistic family at the logical
        # operation boundary.  No candidate-wise scalar ``.item()`` occurs in
        # the fitting loop above.
        means_cpu = component_means.detach().cpu()
        variances_cpu = component_variances.detach().cpu()
        log_weights_cpu = component_log_weights.detach().cpu()
        fallback_cpu = component_fallback.detach().cpu()
        bias_cpu = bias_coefficients.detach().cpu().tolist()
        global_mean_cpu = global_mean_tensor.detach().cpu().tolist()
        global_variance_cpu = global_variance_tensor.detach().cpu().tolist()
        counts_cpu = region_counts.detach().cpu()
        observed_cpu = observed.detach().cpu()
        nll_sum_cpu = nll_sum_tensor.detach().cpu().tolist()
        result: list[FittedHypothesis] = []
        component_penalty = COMPONENT_PENALTY * float(self.capacity - NUM_REGIONS)
        for index, (hypothesis, view) in enumerate(zip(hypotheses, fitting_views)):
            observed_count = int(observed_cpu[index].item())
            nll_sum = float(nll_sum_cpu[index])
            counts = [int(value) for value in counts_cpu[index].tolist()]
            fallback_regions = [region for region, value in enumerate(counts) if value < MIN_REGION_PIXELS]
            complexity = _neighbor_disagreement(hypothesis.labels.detach().cpu()) + component_penalty
            prior = _prior_penalty(hypothesis)
            normalized = nll_sum / float(observed_count)
            fit_score = EvidenceScore(
                nll_sum=nll_sum,
                count=observed_count,
                normalized_nll=normalized,
                complexity=complexity,
                prior=prior,
                total=normalized + self.beta * complexity + prior,
                role="fit",
                available=True,
                reason=None,
            )
            components: list[dict[str, Any]] = []
            for region in range(NUM_REGIONS):
                modes = mode_count if region == 0 else 1
                for mode in range(modes):
                    components.append(
                        {
                            "region": region,
                            "mean": float(means_cpu[index, region, mode].item()),
                            "variance": float(variances_cpu[index, region, mode].item()),
                            "log_weight": float(log_weights_cpu[index, region, mode].item()),
                            "fallback": bool(fallback_cpu[index, region, mode].item()),
                        }
                    )
            parameters: dict[str, Any] = {
                "fit_support_shape": list(view.support.shape),
                "fit_support_bits": np.packbits(view.support.detach().cpu().numpy()).tobytes().hex(),
                "bias_coefficients": [float(value) for value in bias_cpu[index]],
                "bias_ridge": self.bias_ridge,
                "components": components,
                "distribution": self.distribution,
                "student_t_dof": STUDENT_T_DOF if self.distribution == "student_t" else None,
                "variance_floor": self.variance_floor,
                "global_mean": float(global_mean_cpu[index]),
                "global_variance": float(global_variance_cpu[index]),
                "region_counts": {region: counts[region] for region in range(NUM_REGIONS)},
                "fallback_regions": fallback_regions,
                "fallback_region_count": len(fallback_regions),
                "observed_fit_pixels": observed_count,
            }
            result.append(
                FittedHypothesis(
                    hypothesis=hypothesis,
                    parameters=parameters,
                    fit_score=fit_score,
                    fitting_steps=self.max_iterations,
                    capacity=self.capacity,
                    study_id=view.study_id,
                    unit_id=view.unit_id,
                    partition_id=view.partition_id,
                    metadata={
                        "contract_version": VERSION,
                        "protocol": view.protocol,
                        "beta": self.beta,
                        "background_modes": self.background_modes,
                        "component_penalty": component_penalty,
                        "max_iterations": self.max_iterations,
                        "temporal_support": self.supports_temporal,
                        "recipe": self.recipe(),
                        "hypothesis_digest": _hypothesis_digest(hypothesis),
                    },
                )
            )
        return result

    def _fit_bias_batch(
        self,
        values: torch.Tensor,
        labels: torch.Tensor,
        support: torch.Tensor,
        basis: torch.Tensor,
        region_means: torch.Tensor,
        region_variances: torch.Tensor,
    ) -> torch.Tensor:
        """Batched weighted ridge solve retained as a small testable helper."""
        target = region_means.gather(1, labels)
        weights = region_variances.gather(1, labels).reciprocal()
        weights = torch.where(support, weights, torch.zeros_like(weights))
        weighted_basis = basis.unsqueeze(0) * weights.unsqueeze(2)
        # Keep scalar ``basis.T @ (basis * weights)`` multiplication order:
        # putting the unweighted basis first matters for float64 parity.
        normal = torch.einsum("nd,bne->bde", basis, weighted_basis)
        normal = normal + self.bias_ridge * torch.eye(
            3, dtype=basis.dtype, device=basis.device
        )
        rhs = torch.einsum("bnd,bn->bd", weighted_basis, values - target)
        return torch.linalg.solve(normal, rhs)

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
    def _score_on_device(
        self,
        fitted: FittedHypothesis,
        scoring_view: ScoringView,
    ) -> EvidenceScore:
        """Score one fit after routing detached residual/density work to target."""
        context = _validate_score_context(self, fitted, scoring_view)
        if context.count == 0:
            return _score_from_nll(
                nll_sum=0.0,
                count=0,
                complexity=context.complexity,
                prior=context.prior,
                beta=self.beta,
                role=scoring_view.role,
            )
        support = context.support
        device = support.device
        height, width = context.height, context.width
        flat_support = support.reshape(-1)
        image = scoring_view.image.detach().to(device=device, dtype=torch.float64)[0].reshape(-1)
        labels = context.hypothesis.labels.detach().to(device=device, dtype=torch.long).reshape(-1)
        basis = _plane_basis(height, width, device)[flat_support]
        residual = image[flat_support] - basis @ context.bias_coefficients
        log_mixture = _batch_log_mixture(
            residual.unsqueeze(0),
            labels[flat_support].unsqueeze(0),
            [context.components],
            self.distribution,
            self.background_modes,
        )[0]
        nll_sum = _as_float((-log_mixture).sum())
        return _score_from_nll(
            nll_sum=nll_sum,
            count=context.count,
            complexity=context.complexity,
            prior=context.prior,
            beta=self.beta,
            role=scoring_view.role,
        )

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
        self._validate_execution_view_device(scoring_view)
        if self._uses_cuda_execution:
            return self._score_on_device(fitted, scoring_view)
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

    def prepare_score(
        self,
        fitted: FittedHypothesis,
        scoring_view: ScoringView,
    ) -> PreparedScore:
        """Cache an immutable full-support per-pixel NLL for regional reductions.

        This method has its own score arithmetic rather than delegating to the
        public scalar ``score`` reference.  The resulting snapshot contains no
        references to the input fitted hypothesis or scoring view; callers may
        mutate either after preparation without changing the cached result.
        """
        context = _validate_score_context(self, fitted, scoring_view)
        support = context.support
        height, width = context.height, context.width
        device = support.device
        nll_full = torch.zeros((height, width), dtype=torch.float64, device=device)
        if context.count:
            flat_support = support.reshape(-1)
            flat_image = (
                scoring_view.image.detach().to(device=device, dtype=torch.float64)[0]
                .reshape(-1)[flat_support]
            )
            flat_labels = (
                context.hypothesis.labels.detach().to(device=device, dtype=torch.long)
                .reshape(-1)[flat_support]
            )
            basis = _plane_basis(height, width, device)[flat_support]
            residual = flat_image - basis @ context.bias_coefficients
            if not self._uses_cuda_execution:
                nll_values = -self._log_mixture(residual, flat_labels, list(context.components))
            else:
                nll_values = -_batch_log_mixture(
                    residual.unsqueeze(0),
                    flat_labels.unsqueeze(0),
                    [context.components],
                    self.distribution,
                    self.background_modes,
                )[0]
            nll_full.reshape(-1)[flat_support] = nll_values
            nll_sum = _as_float(nll_values.sum())
        else:
            nll_sum = 0.0
        result = _score_from_nll(
            nll_sum=nll_sum,
            count=context.count,
            complexity=context.complexity,
            prior=context.prior,
            beta=self.beta,
            role=scoring_view.role,
        )
        recipe_snapshot = tuple(
            (key, _freeze_recipe_value(value))
            for key, value in sorted(self.recipe().items(), key=lambda item: item[0])
        )
        # Regional auditing supplies CPU connected-component masks and performs
        # integrity hashing on CPU.  CUDA likelihood work therefore crosses one
        # full-grid D2H boundary here; the immutable PreparedScore and all
        # subsequent score_region reductions stay portable CPU tensors.
        prepared_nll = nll_full.detach().cpu() if self._uses_cuda_execution else nll_full
        prepared_support = support.detach().cpu() if self._uses_cuda_execution else support
        return PreparedScore(
            nll_per_pixel=prepared_nll,
            support=prepared_support,
            nll_sum=result.nll_sum,
            count=result.count,
            normalized_nll=result.normalized_nll,
            complexity=result.complexity,
            prior=result.prior,
            total=result.total,
            role=result.role,
            available=result.available,
            reason=result.reason,
            study_id=scoring_view.study_id,
            unit_id=scoring_view.unit_id,
            partition_id=scoring_view.partition_id,
            recipe_snapshot=recipe_snapshot,
            hypothesis_digest=str(fitted.metadata["hypothesis_digest"]),
        )

    def score_region(self, prepared: PreparedScore, support: torch.Tensor) -> EvidenceScore:
        """Reduce a prepared full-support NLL on one exact boolean subset."""
        if not isinstance(prepared, PreparedScore):
            raise ObservationContractError("score_region requires a PreparedScore")
        prepared._assert_integrity()
        if not isinstance(support, torch.Tensor) or support.dtype != torch.bool:
            raise ObservationContractError("regional support must be a bool tensor")
        if tuple(support.shape) != tuple(prepared._support.shape):
            raise ObservationContractError(
                f"regional support must have shape {tuple(prepared._support.shape)}"
            )
        if support.device != prepared._support.device:
            raise ObservationContractError("regional support must stay on the prepared score device")
        subset = prepared._support & support.detach()
        count = int(subset.sum().item())
        if count == 0:
            return _score_from_nll(
                nll_sum=0.0,
                count=0,
                complexity=prepared.complexity,
                prior=prepared.prior,
                beta=float(dict(prepared.recipe_snapshot).get("beta", 0.0)),
                role=prepared.role,
                reason=prepared.reason or "no observed pixels in this scoring role for this unit",
            )
        nll_sum = _as_float(prepared._nll_per_pixel[subset].sum())
        beta = float(dict(prepared.recipe_snapshot).get("beta", 0.0))
        return _score_from_nll(
            nll_sum=nll_sum,
            count=count,
            complexity=prepared.complexity,
            prior=prepared.prior,
            beta=beta,
            role=prepared.role,
        )

    def score_many(
        self,
        fitted: Sequence[FittedHypothesis],
        scoring_views: ScoringView | Sequence[ScoringView],
    ) -> list[EvidenceScore]:
        """Score corresponding frozen fits/views with deterministic float64 batching."""
        try:
            fits = list(fitted)
        except TypeError as exc:
            raise ObservationContractError("fitted must be a sequence") from exc
        if not fits:
            return []
        views = _coerce_views(scoring_views, len(fits), ScoringView, "scoring_views")
        contexts = [_validate_score_context(self, fit, view) for fit, view in zip(fits, views)]

        groups: dict[tuple[Any, ...], list[int]] = {}
        for index, view in enumerate(views):
            device = view.image.device
            if view.support.device != device:
                key = ("scalar", index)
            else:
                key = (
                    tuple(view.support.shape),
                    device.type,
                    device.index,
                    view.role,
                    str(view.image.dtype),
                )
            groups.setdefault(key, []).append(index)

        output: list[EvidenceScore | None] = [None] * len(fits)
        for key, indices in groups.items():
            if key[0] == "scalar" or len(indices) == 1:
                for index in indices:
                    output[index] = self.score(fits[index], views[index])
                continue
            scores = self._score_many_group([contexts[index] for index in indices])
            if len(scores) != len(indices):
                raise ObservationContractError(
                    "score_many fast group returned an unexpected result count"
                )
            for index, score in zip(indices, scores):
                output[index] = score
        if any(entry is None for entry in output):
            raise ObservationContractError("score_many did not fill every ordered result slot")
        return cast(list[EvidenceScore], output)

    def _score_many_group(self, contexts: Sequence[_ScoreContext]) -> list[EvidenceScore]:
        """Batch full-grid residual/density evaluation for compatible contexts."""
        if not contexts:
            return []
        height, width = contexts[0].height, contexts[0].width
        device = self._target_device(contexts[0].scoring_view.image.device)
        batch_size = len(contexts)
        images = torch.stack(
            [
                context.scoring_view.image.detach().to(device=device, dtype=torch.float64)[0]
                for context in contexts
            ],
            dim=0,
        ).reshape(batch_size, -1)
        supports = torch.stack(
            [context.support.detach().to(device=device) for context in contexts], dim=0
        ).reshape(batch_size, -1)
        labels = torch.stack(
            [
                context.hypothesis.labels.detach().to(device=device, dtype=torch.long)
                for context in contexts
            ],
            dim=0,
        ).reshape(batch_size, -1)
        basis = _plane_basis(height, width, device)
        bias = torch.stack(
            [context.bias_coefficients.to(device=device) for context in contexts], dim=0
        )
        residual = images - (basis @ bias.transpose(0, 1)).transpose(0, 1)
        components = [context.components for context in contexts]
        log_mixture = _batch_log_mixture(
            residual,
            labels,
            components,
            self.distribution,
            self.background_modes,
        )
        nll = -log_mixture
        nll_sum = torch.where(supports, nll, torch.zeros_like(nll)).sum(dim=1)
        result: list[EvidenceScore] = []
        for index, context in enumerate(contexts):
            result.append(
                _score_from_nll(
                    nll_sum=float(nll_sum[index].item()),
                    count=context.count,
                    complexity=context.complexity,
                    prior=context.prior,
                    beta=self.beta,
                    role=context.scoring_view.role,
                )
            )
        return result
