#!/usr/bin/env python3
"""Bounded, evidence-safe CPU/CUDA gate for the mask-free auditor.

This is an *independent gate*, not another implementation of the auditor.  It
freezes a candidate bank once on CPU, keeps the public bank/views on CPU as the
trainer does, sets ``ObservationModel(execution_device=...)`` for private
tensor-heavy math, and calls the canonical :func:`audit_banks` entry point on
both sides.  Only image-only tensors are used by the built-in fixture.  A
real-data mode can instead load a bounded list of :class:`TrainingUnit` objects
through the public manifest/dataset API; it never asks the training dataset for
verification data.

The gate is intentionally conservative:

* CUDA is ``NOT RUN`` (a non-success status) when the requested device is not
  available, rather than silently falling back to CPU.
* candidate content and discrete decisions must be exact;
* FP64 fit/score arithmetic and FP32 regional outputs are compared under
  separate strict tolerances, with maximum absolute differences recorded;
* an acceleration ratio is emitted only after decisions, full work counters and
  CUDA residency all pass;
* output files are fresh-only and default to the portable system temp directory.

The synthetic fixture is deliberately non-degenerate, but no latent labels are
passed to the pipeline.  Its image is generated from deterministic analytic
intensity structure and seeded noise, then split into disjoint fit/selection
supports before candidate generation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_audit_maskfree import observation as observation_module
from self_audit_maskfree import runtime as runtime_module
from self_audit_maskfree.auditor import (
    DEFAULT_IMPROVEMENT_THRESHOLD,
    DEFAULT_ROUNDS,
    MAX_FIT_ITERATIONS,
    audit_banks,
)
from self_audit_maskfree.contracts import (
    AuditResult,
    EvidenceScore,
    FittedHypothesis,
    FittingView,
    Hypothesis,
    ScoringView,
    TrainingUnit,
)
from self_audit_maskfree.hypotheses import BANK_SIZE, generate_bank
from self_audit_maskfree.observation import ObservationModel

SCHEMA_VERSION = "maskfree150.audit-device-gate.v1"
DEFAULT_RESOLUTION = 224
DEFAULT_SEED = 42
DEFAULT_MAX_UNITS = 1
DEFAULT_WARMUP = 1
DEFAULT_REPEATS = 3
FP64_RTOL = 5e-10
FP64_ATOL = 5e-10
FP32_RTOL = 5e-6
FP32_ATOL = 5e-6
CUBLAS_WORKSPACE_CONFIG_DEFAULT = ":4096:8"
CUBLAS_WORKSPACE_CONFIG_ALLOWED = (":4096:8", ":16:8")
HEAVY_CUDA_OP_TOKENS = (
    "cdist",
    "linalg_solve",
    "logsumexp",
    "quantile",
    "einsum",
    "matmul",
    "mm",
)


class GateError(RuntimeError):
    """Raised for a requested gate setup or contract violation."""


@dataclass(frozen=True)
class GateCase:
    """CPU-frozen observations and candidate banks for one bounded audit."""

    units: tuple[TrainingUnit, ...]
    banks: tuple[tuple[Hypothesis, ...], ...]
    source: dict[str, Any]


def _tensor_hash(value: torch.Tensor) -> str:
    """Hash shape, dtype and exact contiguous bytes independent of device."""
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Require matching shape/dtype and exact bytes, not only equal values."""
    return _tensor_hash(left) == _tensor_hash(right)


def _unit_tensor_identity(unit: TrainingUnit) -> dict[str, Any]:
    """Return immutable image/support digests for one bounded input unit.

    The gate deliberately records only the image and support tensors consumed
    by the fitting and selection views.  Hashing happens while the case is
    frozen, before any timed pass, so the receipt can establish that both arms
    saw identical content without retaining another tensor copy in the report.
    """
    return {
        "unit_id": str(unit.fitting.unit_id),
        "fitting": {
            "image_sha256": _tensor_hash(unit.fitting.image),
            "support_sha256": _tensor_hash(unit.fitting.support),
            "image_shape": list(unit.fitting.image.shape),
            "support_shape": list(unit.fitting.support.shape),
            "image_dtype": str(unit.fitting.image.dtype),
            "support_dtype": str(unit.fitting.support.dtype),
        },
        "selection": {
            "image_sha256": _tensor_hash(unit.selection.image),
            "support_sha256": _tensor_hash(unit.selection.support),
            "image_shape": list(unit.selection.image.shape),
            "support_shape": list(unit.selection.support.shape),
            "image_dtype": str(unit.selection.image.dtype),
            "support_dtype": str(unit.selection.support.dtype),
        },
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _exact_snapshot(value: Any) -> Any:
    """JSON-safe representation retaining the exact hexadecimal float value."""
    if isinstance(value, float):
        return {"__float_hex__": value.hex()}
    if isinstance(value, torch.Tensor):
        return {"__tensor_sha256__": _tensor_hash(value)}
    if isinstance(value, Mapping):
        return {str(key): _exact_snapshot(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_exact_snapshot(item) for item in value]
    return value


def _exact_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(_exact_snapshot(value)).encode("utf-8")).hexdigest()


def _clone_hypothesis(candidate: Hypothesis, *, device: torch.device) -> Hypothesis:
    """Clone a frozen candidate while changing only tensor residency."""
    return Hypothesis(
        candidate_id=str(candidate.candidate_id),
        labels=candidate.labels.detach().to(device=device).clone(),
        probabilities=candidate.probabilities.detach().to(device=device).clone(),
        validity=candidate.validity.detach().to(device=device).clone(),
        source=str(candidate.source),
        semantic_unresolved=bool(candidate.semantic_unresolved),
        alternatives=copy.deepcopy(candidate.alternatives),
        metadata=copy.deepcopy(candidate.metadata),
    )


def _clone_fitting_view(view: FittingView, *, device: torch.device) -> FittingView:
    return FittingView(
        image=view.image.detach().to(device=device).clone(),
        support=view.support.detach().to(device=device).clone(),
        context=view.context.detach().to(device=device).clone(),
        study_id=view.study_id,
        unit_id=view.unit_id,
        protocol=view.protocol,
        metadata=copy.deepcopy(view.metadata),
        partition_id=view.partition_id,
    )


def _clone_scoring_view(view: ScoringView, *, device: torch.device) -> ScoringView:
    return ScoringView(
        image=view.image.detach().to(device=device).clone(),
        support=view.support.detach().to(device=device).clone(),
        study_id=view.study_id,
        unit_id=view.unit_id,
        role=view.role,
        partition_id=view.partition_id,
    )


def _clone_unit(unit: TrainingUnit, *, device: torch.device) -> TrainingUnit:
    return TrainingUnit(
        fitting=_clone_fitting_view(unit.fitting, device=device),
        selection=_clone_scoring_view(unit.selection, device=device),
        record=copy.deepcopy(unit.record),
    )


def _analytic_image(size: int, seed: int) -> torch.Tensor:
    """Create a non-degenerate image-only phantom without exposing labels."""
    if size != DEFAULT_RESOLUTION:
        raise GateError(f"the audit gate is frozen at resolution {DEFAULT_RESOLUTION}, got {size}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radius = (xx.square() + yy.square()).sqrt()
    # Three smooth intensity structures, a low-frequency field and seeded noise
    # produce distinct appearance regions while retaining no segmentation truth.
    central = torch.exp(-((xx - 0.18).square() + (yy + 0.08).square()) / 0.045)
    annulus = torch.exp(-((radius - 0.42).square()) / 0.018)
    neighbour = torch.exp(-((xx + 0.42).square() + (yy - 0.20).square()) / 0.025)
    field = 0.06 * torch.sin(3.0 * xx + 0.7) * torch.cos(2.0 * yy - 0.4) - 0.08 * radius
    noise = torch.randn((size, size), generator=generator, dtype=torch.float32) * 0.004
    return (0.70 * central + 0.34 * annulus + 0.22 * neighbour + field + noise).contiguous()


def build_synthetic_units(*, size: int = DEFAULT_RESOLUTION, seed: int = DEFAULT_SEED,
                          count: int = 1) -> list[TrainingUnit]:
    """Build deterministic image-only ``TrainingUnit`` fixtures."""
    if size != DEFAULT_RESOLUTION:
        raise GateError(f"synthetic audit fixtures require image_size={DEFAULT_RESOLUTION}")
    if count < 1:
        raise GateError("at least one synthetic unit is required")
    blocks = torch.arange(size, dtype=torch.long) // 8
    by, bx = torch.meshgrid(blocks, blocks, indexing="ij")
    # Disjoint, spatially interleaved roles ensure every smooth structure is
    # represented in both fit and selection observations.
    selection_support = ((by + 3 * bx + int(seed)) % 5) == 0
    fitting_support = ~selection_support
    if not bool(selection_support.any()) or not bool(fitting_support.any()):
        raise GateError("synthetic fixture generated an empty observation role")
    units: list[TrainingUnit] = []
    for index in range(int(count)):
        # Keep every unit deterministic but distinct.  Reusing one image for
        # multiple unit IDs would let a flattened bulk call pass while hiding
        # cross-unit ordering or stale-cache joins.
        unit_seed = int(seed) + index
        image = _analytic_image(size, unit_seed)
        fit_image = torch.where(fitting_support, image, torch.zeros_like(image)).unsqueeze(0)
        select_image = torch.where(selection_support, image, torch.zeros_like(image)).unsqueeze(0)
        unit_id = f"audit-device-synthetic-{int(seed)}-{index:02d}"
        partition_id = f"audit-device-partition-{int(seed)}"
        fitting = FittingView(
            image=fit_image.clone(),
            support=fitting_support.clone(),
            context=fit_image.repeat(3, 1, 1),
            study_id="maskfree150-audit-device-synthetic",
            unit_id=unit_id,
            protocol="spatial_predictive",
            metadata={"image_size": int(size), "dataset": "synthetic_image_only"},
            partition_id=partition_id,
        )
        selection = ScoringView(
            image=select_image.clone(),
            support=selection_support.clone(),
            study_id=fitting.study_id,
            unit_id=unit_id,
            role="select",
            partition_id=partition_id,
        )
        fitting.validate()
        selection.validate()
        units.append(
            TrainingUnit(
                fitting=fitting,
                selection=selection,
                record={
                    "unit_id": unit_id,
                    "study_id": fitting.study_id,
                    "split": "synthetic",
                    "image_only": True,
                    "mask_inputs_used": False,
                    "reference_inputs_used": False,
                    "image_size": int(size),
                    "seed": unit_seed,
                },
            )
        )
    return units


def load_bounded_training_units(
    manifest_path: str | Path,
    *,
    split: str = "train",
    image_size: int = DEFAULT_RESOLUTION,
    seed: int = DEFAULT_SEED,
    max_units: int = DEFAULT_MAX_UNITS,
    unit_ids: Sequence[str] | None = None,
) -> list[TrainingUnit]:
    """Load bounded real image-only units through the existing public APIs."""
    if image_size != DEFAULT_RESOLUTION:
        raise GateError(f"real-data audit gate requires image_size={DEFAULT_RESOLUTION}")
    if max_units < 1:
        raise GateError("max_units must be positive")
    from self_audit_maskfree.data import ImageOnlyDataset, load_manifest

    manifest = load_manifest(manifest_path)
    dataset = ImageOnlyDataset(
        manifest,
        split=split,
        image_size=image_size,
        seed=seed,
        # Avoid process-global source-cache joins in an independent gate.  The
        # unit tensors remain frozen in this case and no cross-batch cache is
        # used by the audit itself.
        cache=None,
    )
    requested = list(unit_ids or ())
    if requested:
        indices = [dataset.unit_index(unit_id) for unit_id in requested]
    else:
        indices = list(range(min(len(dataset), int(max_units))))
    if len(indices) > max_units:
        raise GateError(f"requested {len(indices)} units but max_units={max_units}")
    if not indices:
        raise GateError(f"manifest split {split!r} contains no units")
    units = [dataset[index] for index in indices]
    for unit in units:
        unit.fitting.validate()
        unit.selection.validate()
        if unit.selection.role != "select":
            raise GateError("real TrainingUnit selection role is not select")
    return units


def freeze_case(
    units: Sequence[TrainingUnit], *, seed: int = DEFAULT_SEED, source: Mapping[str, Any] | None = None
) -> GateCase:
    """Generate the candidate banks exactly once on CPU and freeze their content."""
    if not units:
        raise GateError("cannot freeze an empty audit case")
    banks: list[tuple[Hypothesis, ...]] = []
    for unit in units:
        if unit.fitting.image.device.type != "cpu":
            raise GateError("candidate bank generation must start from CPU fitting views")
        bank = generate_bank(unit.fitting, features=None, seed=seed)
        if len(bank) != BANK_SIZE:
            raise GateError(f"generate_bank returned {len(bank)} candidates, expected {BANK_SIZE}")
        # Cloning prevents any later device transfer or audit call from mutating
        # the CPU source object that defines the comparison identity.
        banks.append(tuple(_clone_hypothesis(candidate, device=torch.device("cpu")) for candidate in bank))
    return GateCase(tuple(units), tuple(banks), dict(source or {}))


def _device(value: str | torch.device) -> torch.device:
    requested = torch.device(value)
    if requested.type not in ("cpu", "cuda"):
        raise GateError(f"audit device must be CPU or CUDA, got {requested}")
    return requested


def _model_for_device(device: torch.device) -> ObservationModel:
    """Construct the worker's model without weakening the fixed audit budget."""
    kwargs: dict[str, Any] = {"max_iterations": MAX_FIT_ITERATIONS}
    # This gate targets the explicit worker API.  A source snapshot without the
    # keyword is not silently treated as an equivalent implementation: fail
    # closed so the report names the missing backend contract.
    try:
        parameters = inspect.signature(ObservationModel).parameters
    except (TypeError, ValueError):  # pragma: no cover - unusual extension type
        parameters = {}
    if "execution_device" not in parameters:
        raise GateError("ObservationModel does not expose the required execution_device API")
    kwargs["execution_device"] = str(device)
    model = ObservationModel(**kwargs)
    return model


@contextmanager
def _deterministic_algorithms() -> Iterator[None]:
    """Enable deterministic algorithms and a supported cuBLAS workspace.

    ``CUBLAS_WORKSPACE_CONFIG`` is read when cuBLAS handles are created, so it
    is set before any explicit CUDA execution.  An existing unsupported value
    is rejected rather than silently producing a non-deterministic gate.
    """
    _configure_cublas_workspace()
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _configure_cublas_workspace() -> str:
    """Set/validate the deterministic cuBLAS workspace before CUDA context use."""
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is None:
        workspace = CUBLAS_WORKSPACE_CONFIG_DEFAULT
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    elif workspace not in CUBLAS_WORKSPACE_CONFIG_ALLOWED:
        raise GateError(
            "CUBLAS_WORKSPACE_CONFIG must be one of "
            + ", ".join(CUBLAS_WORKSPACE_CONFIG_ALLOWED)
            + f", got {workspace!r}"
        )
    return workspace


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_activities() -> list[Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if hasattr(torch.profiler, "ProfilerActivity"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Yield nested tensors without exposing their values in the report."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _iter_tensors(item)


@contextmanager
def _method_and_residency_probe() -> Iterator[dict[str, Any]]:
    """Measure canonical bulk calls and private heavy-math tensor residency.

    ``audit_banks`` checks ``type(model) is ObservationModel`` and that public
    methods are canonical.  Therefore the wrappers are temporary class-level
    attributes: while installed, the auditor's ``base`` lookup sees the same
    wrapper and remains on the bulk path.  All attributes are restored in a
    ``finally`` block, including on an exception.
    """
    counters: dict[str, Any] = {
        "fit_many_calls": 0,
        "fit_many_items": 0,
        "score_many_calls": 0,
        "score_many_items": 0,
        "heavy_helpers": [],
    }
    original_fit_many = ObservationModel.fit_many
    original_score_many = ObservationModel.score_many
    original_fit_bias = ObservationModel._fit_bias_batch
    original_mix = observation_module._batch_log_mixture
    original_mix_tensors = observation_module._batch_log_mixture_tensors

    def fit_many_probe(self: ObservationModel, hypotheses: Sequence[Hypothesis], fitting_views: Any):  # type: ignore[no-untyped-def]
        counters["fit_many_calls"] += 1
        counters["fit_many_items"] += len(hypotheses)
        return original_fit_many(self, hypotheses, fitting_views)

    def score_many_probe(self: ObservationModel, fitted: Sequence[FittedHypothesis], scoring_views: Any):  # type: ignore[no-untyped-def]
        counters["score_many_calls"] += 1
        counters["score_many_items"] += len(fitted)
        return original_score_many(self, fitted, scoring_views)

    def fit_bias_probe(self: ObservationModel, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        tensors = list(_iter_tensors(args)) + list(_iter_tensors(kwargs))
        counters["heavy_helpers"].append(
            {
                "helper": "ObservationModel._fit_bias_batch",
                "tensor_devices": sorted({str(tensor.device) for tensor in tensors}),
                "all_tensor_devices": len(tensors) > 0 and len({tensor.device for tensor in tensors}) == 1,
            }
        )
        return original_fit_bias(self, *args, **kwargs)

    def mixture_probe(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        tensors = list(_iter_tensors(args)) + list(_iter_tensors(kwargs))
        counters["heavy_helpers"].append(
            {
                "helper": "observation._batch_log_mixture",
                "tensor_devices": sorted({str(tensor.device) for tensor in tensors}),
                "all_tensor_devices": len(tensors) > 0 and len({tensor.device for tensor in tensors}) == 1,
            }
        )
        return original_mix(*args, **kwargs)

    def mixture_tensors_probe(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        tensors = list(_iter_tensors(args)) + list(_iter_tensors(kwargs))
        counters["heavy_helpers"].append(
            {
                "helper": "observation._batch_log_mixture_tensors",
                "tensor_devices": sorted({str(tensor.device) for tensor in tensors}),
                "all_tensor_devices": len(tensors) > 0 and len({tensor.device for tensor in tensors}) == 1,
            }
        )
        return original_mix_tensors(*args, **kwargs)

    ObservationModel.fit_many = fit_many_probe  # type: ignore[method-assign]
    ObservationModel.score_many = score_many_probe  # type: ignore[method-assign]
    ObservationModel._fit_bias_batch = fit_bias_probe  # type: ignore[method-assign]
    observation_module._batch_log_mixture = mixture_probe  # type: ignore[assignment]
    observation_module._batch_log_mixture_tensors = mixture_tensors_probe  # type: ignore[assignment]
    try:
        yield counters
    finally:
        ObservationModel.fit_many = original_fit_many  # type: ignore[method-assign]
        ObservationModel.score_many = original_score_many  # type: ignore[method-assign]
        ObservationModel._fit_bias_batch = original_fit_bias  # type: ignore[method-assign]
        observation_module._batch_log_mixture = original_mix  # type: ignore[assignment]
        observation_module._batch_log_mixture_tensors = original_mix_tensors  # type: ignore[assignment]


def _event_cuda_time(event: Any) -> float:
    for name in ("self_device_time_total", "device_time_total", "self_cuda_time_total", "cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number > 0.0:
                return number
    return 0.0


def _event_is_cuda(event: Any) -> bool:
    text = str(getattr(event, "device_type", "")).lower()
    if "cuda" in text:
        return True
    return _event_cuda_time(event) > 0.0


def _heavy_cuda_event_rows(profiler: Any | None) -> list[dict[str, Any]]:
    """Return token-matching CUDA events with positive recorded device time.

    A profiler event can carry ``device_type=CUDA`` metadata even when no
    kernel was recorded (for example a zero-duration fake event or an empty
    profile).  Such metadata is not residency evidence.  The gate therefore
    requires a finite, strictly positive device-time value on a heavy-op token
    before marking CUDA compute as observed.
    """
    if profiler is None or not hasattr(profiler, "key_averages"):
        return []
    rows: list[dict[str, Any]] = []
    try:
        events = profiler.key_averages()
    except (RuntimeError, TypeError, ValueError):
        return rows
    for event in events:
        if not _event_is_cuda(event):
            continue
        key = str(getattr(event, "key", ""))
        device_time = _event_cuda_time(event)
        if device_time <= 0.0 or not math.isfinite(device_time):
            continue
        if any(token in key.lower() for token in HEAVY_CUDA_OP_TOKENS):
            rows.append({"name": key, "device_time": device_time})
    return rows


def _residency_report(
    model: ObservationModel,
    units: Sequence[TrainingUnit],
    banks: Sequence[Sequence[Hypothesis]],
    results: Sequence[AuditResult],
    device: torch.device,
    profiler: Any | None,
    probe: Mapping[str, Any],
    require_profiler: bool = False,
) -> dict[str, Any]:
    """Check tensors and profiler events, not only a model device attribute.

    The explicit-device observation implementation serializes fitted parameter
    dictionaries and hypotheses to CPU at the logical fit boundary.  That host
    serialization is expected; the gate checks the device-resident inputs,
    selected/regional outputs, and heavy CUDA profiler events separately.
    """
    checks: dict[str, Any] = {
        "requested_device": str(device),
        "model_execution_device": None,
        "model_execution_device_matches": False,
        # The canonical trainer keeps public views and frozen banks on CPU;
        # ObservationModel(execution_device=CUDA) transfers detached working
        # arrays internally.  Verify this route explicitly instead of requiring
        # callers to move topology/masks to CUDA.
        "public_inputs_on_cpu": True,
        "public_bank_on_cpu": True,
        "audit_outputs_on_cpu": True,
        "fit_hypotheses_serialized_to_cpu": True,
        "heavy_helper_calls": list(probe.get("heavy_helpers", [])),
        "heavy_helpers_on_requested_device": False,
        "heavy_cuda_events": [],
        "heavy_cuda_event_observed": False,
        "profiler_required": bool(require_profiler),
    }
    declared = getattr(model, "execution_device", None)
    if declared is None:
        declared = getattr(model, "device", None)
    if declared is not None:
        checks["model_execution_device"] = str(declared)
        try:
            checks["model_execution_device_matches"] = torch.device(declared) == device
        except (TypeError, RuntimeError, ValueError):
            checks["model_execution_device_matches"] = False
    for unit in units:
        for tensor in (unit.fitting.image, unit.fitting.support, unit.fitting.context,
                       unit.selection.image, unit.selection.support):
            checks["public_inputs_on_cpu"] &= tensor.device.type == "cpu"
    for bank in banks:
        for candidate in bank:
            for tensor in (candidate.labels, candidate.probabilities, candidate.validity):
                checks["public_bank_on_cpu"] &= tensor.device.type == "cpu"
    for result in results:
        for tensor in (result.regional_margin, result.validity,
                       result.selected.labels, result.selected.probabilities, result.selected.validity):
            checks["audit_outputs_on_cpu"] &= tensor.device.type == "cpu"
        # Explicit-device fitting deliberately serializes fitted parameters and
        # portable hypotheses to CPU at the complete operation boundary.  This
        # is expected host output, not evidence that the heavy fit ran on CPU;
        # profiler events below provide the independent residency check.
        for fitted in result.fitted:
            checks["fit_hypotheses_serialized_to_cpu"] &= (
                fitted.hypothesis.labels.device.type == "cpu"
                and fitted.hypothesis.probabilities.device.type == "cpu"
                and fitted.hypothesis.validity.device.type == "cpu"
            )
    checks["heavy_cuda_events"] = _heavy_cuda_event_rows(profiler)
    checks["heavy_cuda_event_observed"] = bool(checks["heavy_cuda_events"])
    def helper_device_matches(text: str) -> bool:
        if device.type == "cuda":
            # A bare ``cuda`` request resolves to the current visible device;
            # CUDA helper tensors therefore commonly report ``cuda:0``.
            return text == "cuda" or text.startswith("cuda:")
        return text == "cpu"

    helper_rows = checks["heavy_helper_calls"]
    if device.type == "cuda":
        checks["heavy_helpers_on_requested_device"] = bool(helper_rows) and all(
            any(helper_device_matches(text) for text in row.get("tensor_devices", []))
            and row.get("all_tensor_devices", False)
            for row in helper_rows
        )
    else:
        checks["heavy_helpers_on_requested_device"] = bool(helper_rows) and all(
            row.get("all_tensor_devices", False) for row in helper_rows
        )
    checks["passed"] = bool(
        checks["public_inputs_on_cpu"]
        and checks["public_bank_on_cpu"]
        and checks["audit_outputs_on_cpu"]
        and checks["fit_hypotheses_serialized_to_cpu"]
        and checks["heavy_helpers_on_requested_device"]
        and (device.type != "cuda" or checks["model_execution_device_matches"])
        and (not require_profiler or checks["heavy_cuda_event_observed"])
    )
    return checks


def _aggregate_work(
    case: GateCase, results: Sequence[AuditResult], *, api_calls: Mapping[str, int]
) -> dict[str, Any]:
    """Derive complete logical work counters for one canonical audit pass."""
    if len(results) != len(case.banks):
        raise GateError("audit result count does not match frozen case")
    candidate_count = sum(len(bank) for bank in case.banks)
    regional_prepare = 0
    regional_score_region = 0
    regional_scalar = 0
    logical_primary = 0
    logical_total = 0
    for result in results:
        trace = result.trace
        logical_primary += int(trace.get("primary_score_calls", 0))
        logical_total += int(trace.get("score_calls_total", 0))
        work = trace.get("physical_work", {})
        regional_prepare += int(work.get("regional_prepare_calls", 0))
        regional_score_region += int(work.get("regional_score_region_calls", 0))
        regional_scalar += int(work.get("regional_scalar_score_calls", 0))
    unit_budget_match = all(
        int(result.trace.get("bank_size", -1)) == BANK_SIZE
        and int(result.trace.get("rounds_requested", -1)) == DEFAULT_ROUNDS
        and int(result.trace.get("fits_performed", -1)) == BANK_SIZE
        and int(result.trace.get("primary_score_calls", -1)) == BANK_SIZE
        and int(result.trace.get("budget", {}).get("fitting_iterations_per_candidate", -1))
        == MAX_FIT_ITERATIONS
        and all(int(fitted.fitting_steps) == MAX_FIT_ITERATIONS for fitted in result.fitted)
        for result in results
    )
    bulk_api_match = bool(
        int(api_calls.get("fit_many_calls", 0)) == 1
        and int(api_calls.get("score_many_calls", 0)) == 1
        and int(api_calls.get("fit_many_items", 0)) == candidate_count
        and int(api_calls.get("score_many_items", 0)) == candidate_count
    )
    return {
        "units": len(case.banks),
        "candidate_items": candidate_count,
        "fit_calls": candidate_count,
        "fit_items": candidate_count,
        "primary_score_calls": logical_primary,
        "primary_score_items": candidate_count,
        "score_calls_total": logical_total,
        "regional_prepare_calls": regional_prepare,
        "regional_score_region_calls": regional_score_region,
        "regional_scalar_score_calls": regional_scalar,
        "fit_many_calls": int(api_calls.get("fit_many_calls", 0)),
        "fit_many_items": int(api_calls.get("fit_many_items", 0)),
        "score_many_calls": int(api_calls.get("score_many_calls", 0)),
        "score_many_items": int(api_calls.get("score_many_items", 0)),
        "full_budget": {
            "bank_size": BANK_SIZE,
            "rounds": DEFAULT_ROUNDS,
            "max_fit_iterations": MAX_FIT_ITERATIONS,
            "all_units_match": bool(unit_budget_match and bulk_api_match),
            "unit_budget_match": bool(unit_budget_match),
            "bulk_api_calls_match": bulk_api_match,
        },
    }


def _run_audit_pass(
    case: GateCase,
    *,
    device: torch.device,
    profile: bool = False,
) -> dict[str, Any]:
    """Run one fresh canonical audit pass on ``device``."""
    # Match the trainer boundary: banks and observation capability views remain
    # CPU objects; only ObservationModel's detached tensor-heavy working arrays
    # move to its explicit execution_device.
    public_device = torch.device("cpu")
    units = [_clone_unit(unit, device=public_device) for unit in case.units]
    banks = [
        [_clone_hypothesis(candidate, device=public_device) for candidate in bank]
        for bank in case.banks
    ]
    model = _model_for_device(device)
    profiler: Any | None = None
    profile_context: Any = nullcontext()
    if profile and device.type == "cuda":
        try:
            profile_context = torch.profiler.profile(
                activities=_profile_activities(),
                record_shapes=False,
                profile_memory=False,
            )
        except (RuntimeError, TypeError):  # pragma: no cover - version dependent
            profile_context = nullcontext()
    # Configure cuBLAS before synchronize/model work can initialize a CUDA
    # context.  The context manager validates it again as a cheap invariant.
    _configure_cublas_workspace()
    _synchronize(device)
    started = time.perf_counter()
    deterministic_effective = False
    with _method_and_residency_probe() as probe, profile_context as active_profiler, _deterministic_algorithms():
        deterministic_effective = bool(torch.are_deterministic_algorithms_enabled())
        results = audit_banks(
            banks,
            [unit.fitting for unit in units],
            [unit.selection for unit in units],
            model,
            rounds=DEFAULT_ROUNDS,
            improvement_threshold=DEFAULT_IMPROVEMENT_THRESHOLD,
        )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    profiler = active_profiler if active_profiler is not None and hasattr(active_profiler, "key_averages") else None
    api_calls = {
        "fit_many_calls": int(probe.get("fit_many_calls", 0)),
        "fit_many_items": int(probe.get("fit_many_items", 0)),
        "score_many_calls": int(probe.get("score_many_calls", 0)),
        "score_many_items": int(probe.get("score_many_items", 0)),
    }
    work = _aggregate_work(case, results, api_calls=api_calls)
    residency = _residency_report(
        model,
        units,
        banks,
        results,
        device,
        profiler,
        probe,
        require_profiler=bool(profile and device.type == "cuda"),
    )
    return {
        "device": str(device),
        "elapsed_seconds": float(elapsed),
        "deterministic_algorithms_effective": deterministic_effective,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "results": results,
        "work": work,
        "residency": residency,
        "profiler_available": profiler is not None,
    }


def _iter_numeric(value: Any, path: str = "") -> Iterator[tuple[str, float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        number = float(value)
        yield path, number
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _iter_numeric(value[key], f"{path}.{key}" if path else str(key))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            yield from _iter_numeric(item, f"{path}[{index}]")


def _compare_structured(
    left: Any,
    right: Any,
    *,
    rtol: float,
    atol: float,
    label: str,
    path: str = "",
) -> dict[str, Any]:
    """Compare complete payload structure, exact discrete leaves and floats.

    Every key/list length and every non-floating leaf is exact.  Floating leaves
    must be finite on both sides and satisfy the supplied strict tolerance;
    NaN/Inf is never ignored as an absent numeric path.
    """
    mismatches: list[dict[str, Any]] = []
    nonfinite: list[dict[str, Any]] = []
    maximum = 0.0

    def walk(a: Any, b: Any, location: str) -> None:
        nonlocal maximum
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                mismatches.append({"path": location, "left": type(a).__name__, "right": type(b).__name__})
                return
            if not torch.equal(a.detach().cpu(), b.detach().cpu()):
                mismatches.append({"path": location, "left": "tensor", "right": "tensor"})
            return
        if isinstance(a, bool) or isinstance(b, bool):
            if not isinstance(a, bool) or not isinstance(b, bool) or a != b:
                mismatches.append({"path": location, "left": a, "right": b})
            return
        if isinstance(a, int) and not isinstance(a, bool) or isinstance(b, int) and not isinstance(b, bool):
            if not (isinstance(a, int) and not isinstance(a, bool)
                    and isinstance(b, int) and not isinstance(b, bool) and a == b):
                mismatches.append({"path": location, "left": a, "right": b})
            return
        if isinstance(a, float) or isinstance(b, float):
            if not isinstance(a, (int, float)) or isinstance(a, bool) or not isinstance(b, (int, float)) or isinstance(b, bool):
                mismatches.append({"path": location, "left": a, "right": b})
                return
            af, bf = float(a), float(b)
            if not math.isfinite(af) or not math.isfinite(bf):
                nonfinite.append({"path": location, "left": af, "right": bf})
                return
            difference = abs(af - bf)
            maximum = max(maximum, difference)
            if not math.isclose(af, bf, rel_tol=rtol, abs_tol=atol):
                mismatches.append({"path": location, "left": af, "right": bf, "abs_difference": difference})
            return
        if isinstance(a, Mapping) or isinstance(b, Mapping):
            if not isinstance(a, Mapping) or not isinstance(b, Mapping):
                mismatches.append({"path": location, "left": type(a).__name__, "right": type(b).__name__})
                return
            akeys, bkeys = set(a), set(b)
            if akeys != bkeys:
                mismatches.append({"path": location, "missing_left": sorted(bkeys - akeys, key=str), "missing_right": sorted(akeys - bkeys, key=str)})
            for key in sorted(akeys & bkeys, key=str):
                walk(a[key], b[key], f"{location}.{key}" if location else str(key))
            return
        if isinstance(a, Sequence) and not isinstance(a, (str, bytes, bytearray)) or isinstance(b, Sequence) and not isinstance(b, (str, bytes, bytearray)):
            if not (isinstance(a, Sequence) and not isinstance(a, (str, bytes, bytearray))
                    and isinstance(b, Sequence) and not isinstance(b, (str, bytes, bytearray))):
                mismatches.append({"path": location, "left": type(a).__name__, "right": type(b).__name__})
                return
            if len(a) != len(b):
                mismatches.append({"path": location, "left_length": len(a), "right_length": len(b)})
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                walk(item_a, item_b, f"{location}[{index}]")
            return
        if a != b:
            mismatches.append({"path": location, "left": a, "right": b})

    walk(left, right, path)
    return {
        "label": label,
        "passed": not mismatches and not nonfinite,
        "max_abs_difference": maximum,
        "rtol": rtol,
        "atol": atol,
        "mismatches": mismatches[:64],
        "mismatch_count": len(mismatches),
        "nonfinite": nonfinite[:32],
        "nonfinite_count": len(nonfinite),
    }


def _compare_numeric_maps(
    left: Any,
    right: Any,
    *,
    rtol: float,
    atol: float,
    label: str,
) -> dict[str, Any]:
    lmap = dict(_iter_numeric(left))
    rmap = dict(_iter_numeric(right))
    paths = sorted(set(lmap) | set(rmap))
    missing = [path for path in paths if path not in lmap or path not in rmap]
    rows: list[dict[str, Any]] = []
    maximum = 0.0
    passed = not missing
    for path in paths:
        if path in missing:
            continue
        a, b = lmap[path], rmap[path]
        difference = abs(a - b)
        maximum = max(maximum, difference)
        close = math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
        passed &= close
        if not close:
            rows.append({"path": path, "left": a, "right": b, "abs_difference": difference})
    return {
        "label": label,
        "passed": bool(passed),
        "max_abs_difference": maximum,
        "rtol": rtol,
        "atol": atol,
        "missing_paths": missing,
        "mismatches": rows[:32],
        "mismatch_count": len(rows),
    }


def _score_payload(score: EvidenceScore) -> dict[str, Any]:
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


def _fit_payload(fitted: FittedHypothesis) -> dict[str, Any]:
    return {
        "candidate_id": fitted.hypothesis.candidate_id,
        "parameters": fitted.parameters,
        "fit_score": _score_payload(fitted.fit_score),
        "fitting_steps": fitted.fitting_steps,
        "capacity": fitted.capacity,
        "study_id": fitted.study_id,
        "unit_id": fitted.unit_id,
        "partition_id": fitted.partition_id,
        "metadata": fitted.metadata,
    }


def _discrete_projection(value: Any) -> Any:
    """Retain exact decision leaves while eliding tolerated float payloads."""
    if isinstance(value, float):
        return "<floating-field>"
    if isinstance(value, Mapping):
        return {str(key): _discrete_projection(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_discrete_projection(item) for item in value]
    return value


def _candidate_signature(candidate: Hypothesis) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "semantic_unresolved": bool(candidate.semantic_unresolved),
        "alternatives": copy.deepcopy(candidate.alternatives),
        "labels_sha256": _tensor_hash(candidate.labels),
        "probabilities_sha256": _tensor_hash(candidate.probabilities),
        "validity_sha256": _tensor_hash(candidate.validity),
        "labels_shape": list(candidate.labels.shape),
        "probabilities_shape": list(candidate.probabilities.shape),
        "validity_shape": list(candidate.validity.shape),
        "labels_dtype": str(candidate.labels.dtype),
        "probabilities_dtype": str(candidate.probabilities.dtype),
        "validity_dtype": str(candidate.validity.dtype),
    }


def _augment_case_source(
    source: Mapping[str, Any],
    case: GateCase,
    *,
    manifest_path: str | Path | None,
) -> dict[str, Any]:
    """Add immutable input and frozen-bank identities to the gate source.

    These digests are computed exactly once after loading/freeze and before the
    first audit pass.  They make a real-data receipt reproducible without
    opening verification masks or introducing a cross-batch cache.
    """
    enriched = copy.deepcopy(dict(source))
    enriched["unit_tensor_identity"] = [_unit_tensor_identity(unit) for unit in case.units]
    enriched["frozen_bank_signatures"] = [
        [_candidate_signature(candidate) for candidate in bank] for bank in case.banks
    ]
    if manifest_path is not None:
        path = Path(manifest_path).expanduser().resolve()
        try:
            manifest_sha256 = runtime_module.sha256_file(path)
        except (OSError, RuntimeError) as exc:
            raise GateError(f"unable to hash manifest for immutable receipt: {path}: {exc}") from exc
        enriched["manifest_sha256"] = manifest_sha256
    return enriched


def _collect_provenance(*, requested_device: str, resolved_device: str) -> dict[str, Any]:
    """Collect code/runtime/hardware identity once, outside timed passes.

    Hardware fields are intentionally null when CUDA was requested but is not
    available.  This prevents a CPU fallback or an unresolvable runtime from
    being misread as an RTX result.
    """
    try:
        source_identity = runtime_module.source_identity(REPO_ROOT)
        source_error = None
    except Exception as exc:  # noqa: BLE001 - provenance must remain explicit
        source_identity = None
        source_error = f"{type(exc).__name__}: {exc}"
    try:
        gate_script_sha256 = runtime_module.sha256_file(Path(__file__).resolve())
        gate_script_error = None
    except (OSError, RuntimeError) as exc:
        gate_script_sha256 = None
        gate_script_error = f"{type(exc).__name__}: {exc}"

    visible = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }
    cuda_available = bool(torch.cuda.is_available())
    hardware: dict[str, Any] = {
        "requested_device": str(requested_device),
        "resolved_device": str(resolved_device),
        "status": "UNKNOWN",
        "device_identity": None,
        "visible_cuda_uuid": None,
        "visible_device_environment": visible,
        "reason": None,
    }
    if resolved_device == "cuda":
        if not cuda_available:
            hardware.update(
                status="NOT AVAILABLE",
                reason="torch.cuda.is_available() is false; no CUDA hardware identity was resolved",
            )
        else:
            try:
                identity = runtime_module.device_identity(torch.device("cuda"))
            except Exception as exc:  # noqa: BLE001 - report unresolved hardware
                hardware.update(
                    status="UNKNOWN",
                    reason=f"unable to resolve CUDA device identity: {type(exc).__name__}: {exc}",
                )
            else:
                hardware.update(
                    status="AVAILABLE",
                    device_identity=identity,
                    visible_cuda_uuid=identity.get("uuid")
                    or (visible.get("CUDA_VISIBLE_DEVICES") if str(visible.get("CUDA_VISIBLE_DEVICES") or "").startswith("GPU-") else None),
                )
    else:
        try:
            identity = runtime_module.device_identity(torch.device("cpu"))
        except Exception as exc:  # noqa: BLE001 - still expose the failure
            hardware.update(status="UNKNOWN", reason=f"unable to resolve CPU identity: {type(exc).__name__}: {exc}")
        else:
            hardware.update(status="CPU", device_identity=identity)
    return {
        "source_identity": source_identity,
        "source_identity_error": source_error,
        "gate_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": gate_script_sha256,
            "sha256_error": gate_script_error,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": runtime_module.environment_identity().get("platform"),
            "torch": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": cuda_available,
        },
        "hardware": hardware,
    }


def _decision_signature(result: AuditResult) -> dict[str, Any]:
    trace = result.trace
    return {
        "unit_id": trace.get("unit_id"),
        "partition_id": trace.get("partition_id"),
        "bank_id": trace.get("bank_id"),
        "candidate_ids": [candidate.candidate_id for candidate in result.bank],
        "evaluation_order": [
            {
                "candidate_index": row.get("candidate_index"),
                "candidate_id": row.get("candidate_id"),
                "round": row.get("round"),
                "cached": row.get("cached"),
            }
            for row in trace.get("evaluation_order", [])
        ],
        "round_slots": trace.get("round_slots"),
        "selected_candidate_id": trace.get("selected_candidate_id"),
        "selected_index": trace.get("selected_index"),
        "accepted": trace.get("accepted"),
        "acceptance_reason": trace.get("acceptance_reason"),
        "rejected_edits": [
            {"candidate_id": row.get("candidate_id"), "reason": row.get("reason")}
            for row in trace.get("rejected_edits", [])
        ],
        "search_inconclusive": trace.get("search_inconclusive"),
        "semantic_unresolved": bool(result.selected.semantic_unresolved),
        "semantic_alternatives": len(result.selected.alternatives),
        "learning_sample_retained": trace.get("learning_sample_retained"),
        "class_absent": trace.get("class_absent"),
        "selection_observations_consumed": trace.get("selection_observations_consumed"),
        # Regional challenger identity, availability/reason and budget fields
        # are discrete decisions too; keep the complete search-note structure
        # for an exact structural comparison below.
        "search_notes": _discrete_projection(trace.get("search_notes")),
    }


def compare_audit_passes(cpu_pass: Mapping[str, Any], gpu_pass: Mapping[str, Any]) -> dict[str, Any]:
    """Compare canonical results, exact bank content and all decision fields."""
    cpu_results = list(cpu_pass.get("results", []))
    gpu_results = list(gpu_pass.get("results", []))
    result_count_match = len(cpu_results) == len(gpu_results)
    candidate_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    floating_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    regional_trace_rows: list[dict[str, Any]] = []
    bitwise_rows: list[dict[str, Any]] = []
    for index, (cpu_result, gpu_result) in enumerate(zip(cpu_results, gpu_results)):
        cpu_bank, gpu_bank = cpu_result.bank, gpu_result.bank
        bank_match = len(cpu_bank) == len(gpu_bank)
        bank_detail: list[dict[str, Any]] = []
        for candidate_index, (cpu_candidate, gpu_candidate) in enumerate(zip(cpu_bank, gpu_bank)):
            labels_equal = _tensor_bitwise_equal(cpu_candidate.labels, gpu_candidate.labels)
            probabilities_equal = _tensor_bitwise_equal(
                cpu_candidate.probabilities, gpu_candidate.probabilities
            )
            validity_equal = _tensor_bitwise_equal(cpu_candidate.validity, gpu_candidate.validity)
            alternatives_compare = _compare_structured(
                cpu_candidate.alternatives,
                gpu_candidate.alternatives,
                rtol=FP32_RTOL,
                atol=FP32_ATOL,
                label=f"unit[{index}].candidate[{candidate_index}].alternatives",
            )
            metadata_compare = _compare_structured(
                cpu_candidate.metadata,
                gpu_candidate.metadata,
                rtol=FP32_RTOL,
                atol=FP32_ATOL,
                label=f"unit[{index}].candidate[{candidate_index}].metadata",
            )
            detail = {
                "candidate_index": candidate_index,
                "candidate_id_equal": cpu_candidate.candidate_id == gpu_candidate.candidate_id,
                "source_equal": cpu_candidate.source == gpu_candidate.source,
                "semantic_unresolved_equal": bool(cpu_candidate.semantic_unresolved)
                == bool(gpu_candidate.semantic_unresolved),
                "alternatives_equal": alternatives_compare["passed"],
                "metadata_equal": metadata_compare["passed"],
                "alternatives_comparison": alternatives_compare,
                "metadata_comparison": metadata_compare,
                "labels_bitwise_equal": labels_equal,
                "probabilities_bitwise_equal": probabilities_equal,
                "validity_bitwise_equal": validity_equal,
                "cpu": _candidate_signature(cpu_candidate),
                "gpu": _candidate_signature(gpu_candidate),
            }
            bank_detail.append(detail)
            bank_match &= bool(
                detail["candidate_id_equal"]
                and detail["source_equal"]
                and detail["semantic_unresolved_equal"]
                and detail["alternatives_equal"]
                and detail["metadata_equal"]
                and labels_equal
                and probabilities_equal
                and validity_equal
            )
        candidate_rows.append({"unit_index": index, "passed": bool(bank_match), "candidates": bank_detail})

        cpu_decision, gpu_decision = _decision_signature(cpu_result), _decision_signature(gpu_result)
        decision_equal = _canonical_json(cpu_decision) == _canonical_json(gpu_decision)
        decision_rows.append({
            "unit_index": index,
            "passed": decision_equal,
            "cpu": cpu_decision,
            "gpu": gpu_decision,
        })
        structure_rows.append(_compare_structured(
            cpu_result.trace,
            gpu_result.trace,
            # Trace mixes FP64 scores with FP32 coverage/margin summaries;
            # strict FP32 bounds cover the latter while dedicated fit/score
            # payload checks above retain FP64 bounds.
            rtol=FP32_RTOL,
            atol=FP32_ATOL,
            label=f"unit[{index}].trace",
        ))
        cpu_notes = cpu_result.trace.get("search_notes", {})
        gpu_notes = gpu_result.trace.get("search_notes", {})
        regional_trace_rows.append(_compare_structured(
            {
                "regions": cpu_notes.get("regions", []),
                "score_counts": cpu_notes.get("score_counts", []),
                "regional_score_counts": cpu_notes.get("regional_score_counts", []),
                "regional_score_observation_count": cpu_notes.get("regional_score_observation_count"),
            },
            {
                "regions": gpu_notes.get("regions", []),
                "score_counts": gpu_notes.get("score_counts", []),
                "regional_score_counts": gpu_notes.get("regional_score_counts", []),
                "regional_score_observation_count": gpu_notes.get("regional_score_observation_count"),
            },
            rtol=FP64_RTOL,
            atol=FP64_ATOL,
            label=f"unit[{index}].regional_trace_fp64",
        ))

        # Fit parameters and score dictionaries are Python/FP64 observations;
        # regional margin and validity retain their FP32 tensor contract.
        cpu_fits = [_fit_payload(entry) for entry in cpu_result.fitted]
        gpu_fits = [_fit_payload(entry) for entry in gpu_result.fitted]
        floating_rows.append(_compare_structured(
            cpu_fits, gpu_fits, rtol=FP64_RTOL, atol=FP64_ATOL, label=f"unit[{index}].fit_parameters"
        ))
        floating_rows.append(_compare_structured(
            [_score_payload(entry) for entry in cpu_result.scores],
            [_score_payload(entry) for entry in gpu_result.scores],
            rtol=FP64_RTOL,
            atol=FP64_ATOL,
            label=f"unit[{index}].scores",
        ))
        cpu_fit_hash, gpu_fit_hash = _exact_hash(cpu_fits), _exact_hash(gpu_fits)
        cpu_score_hash = _exact_hash([_score_payload(entry) for entry in cpu_result.scores])
        gpu_score_hash = _exact_hash([_score_payload(entry) for entry in gpu_result.scores])
        bitwise_rows.extend(
            [
                {
                    "label": f"unit[{index}].fit_parameters",
                    "bitwise_equal": cpu_fit_hash == gpu_fit_hash,
                    "cpu_sha256": cpu_fit_hash,
                    "gpu_sha256": gpu_fit_hash,
                },
                {
                    "label": f"unit[{index}].scores",
                    "bitwise_equal": cpu_score_hash == gpu_score_hash,
                    "cpu_sha256": cpu_score_hash,
                    "gpu_sha256": gpu_score_hash,
                },
            ]
        )
        for field in ("regional_margin", "validity"):
            left = getattr(cpu_result, field).detach().cpu()
            right = getattr(gpu_result, field).detach().cpu()
            shape_equal = tuple(left.shape) == tuple(right.shape)
            dtype_equal = left.dtype == right.dtype
            if shape_equal:
                difference = (left.to(torch.float64) - right.to(torch.float64)).abs()
                finite = bool(torch.isfinite(difference).all())
                close = bool(
                    dtype_equal
                    and torch.allclose(left, right, rtol=FP32_RTOL, atol=FP32_ATOL, equal_nan=False)
                )
                max_abs_difference = float(difference.max().item()) if difference.numel() else 0.0
            else:
                finite = False
                close = False
                max_abs_difference = float("inf")
            close = bool(shape_equal and dtype_equal and finite and close)
            bitwise = _tensor_bitwise_equal(left, right)
            floating_rows.append({
                "label": f"unit[{index}].{field}",
                "passed": bool(finite and close),
                "shape_equal": shape_equal,
                "dtype_equal": dtype_equal,
                "max_abs_difference": max_abs_difference,
                "rtol": FP32_RTOL,
                "atol": FP32_ATOL,
                "shape": list(left.shape),
            })
            bitwise_rows.append({
                "label": f"unit[{index}].{field}",
                "bitwise_equal": bitwise,
                "cpu_sha256": _tensor_hash(left),
                "gpu_sha256": _tensor_hash(right),
            })
    work_equal = _canonical_json(cpu_pass.get("work")) == _canonical_json(gpu_pass.get("work"))
    residency_pass = bool(gpu_pass.get("residency", {}).get("passed", False))
    deterministic_pass = bool(
        cpu_pass.get("deterministic_algorithms_effective", False)
        and gpu_pass.get("deterministic_algorithms_effective", False)
    )
    cublas_workspace_pass = (
        cpu_pass.get("cublas_workspace_config") == gpu_pass.get("cublas_workspace_config")
        and cpu_pass.get("cublas_workspace_config") in CUBLAS_WORKSPACE_CONFIG_ALLOWED
    )
    candidate_pass = result_count_match and all(row["passed"] for row in candidate_rows)
    decision_pass = result_count_match and all(row["passed"] for row in decision_rows)
    floating_pass = result_count_match and all(row["passed"] for row in floating_rows)
    structure_pass = result_count_match and all(row["passed"] for row in structure_rows)
    regional_trace_pass = result_count_match and all(row["passed"] for row in regional_trace_rows)
    full_budget_pass = bool(
        cpu_pass.get("work", {}).get("full_budget", {}).get("all_units_match", False)
        and gpu_pass.get("work", {}).get("full_budget", {}).get("all_units_match", False)
    )
    return {
        "result_count_match": result_count_match,
        "candidate_content": {"passed": candidate_pass, "units": candidate_rows},
        "decisions": {"passed": decision_pass, "units": decision_rows},
        "structure": {"passed": structure_pass, "traces": structure_rows},
        "regional_trace": {"passed": regional_trace_pass, "comparisons": regional_trace_rows},
        "floating": {"passed": floating_pass, "comparisons": floating_rows},
        "bitwise": {"regional_outputs": bitwise_rows},
        "work_counters": {"passed": work_equal, "cpu": cpu_pass.get("work"), "gpu": gpu_pass.get("work")},
        "full_budget": {"passed": full_budget_pass},
        "residency": {"passed": residency_pass, "gpu": gpu_pass.get("residency")},
        "deterministic_algorithms": {
            "passed": deterministic_pass,
            "cpu": cpu_pass.get("deterministic_algorithms_effective"),
            "gpu": gpu_pass.get("deterministic_algorithms_effective"),
        },
        "cublas_workspace": {
            "passed": cublas_workspace_pass,
            "cpu": cpu_pass.get("cublas_workspace_config"),
            "gpu": gpu_pass.get("cublas_workspace_config"),
        },
        "passed": bool(
            candidate_pass
            and decision_pass
            and floating_pass
            and structure_pass
            and regional_trace_pass
            and work_equal
            and full_budget_pass
            and residency_pass
            and deterministic_pass
            and cublas_workspace_pass
        ),
    }


def _unique_output(path: Path | None) -> Path:
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = Path(tempfile.gettempdir()) / f"maskfree150-audit-device-{stamp}-{os.getpid()}.json"
    path = path.expanduser().resolve()
    if path.exists():
        raise GateError(f"refusing to overwrite existing gate output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_fresh_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise GateError(f"refusing to overwrite existing gate output: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _timed_device_passes(
    case: GateCase,
    *,
    device: torch.device,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if warmup < 0 or repeats < 1:
        raise GateError("warmup must be >=0 and repeats must be >=1")
    if repeats > 5:
        raise GateError("bounded GPU timing allows at most five repeats")
    baseline = reference
    warmup_checks: list[bool] = []
    for _ in range(warmup):
        warmup_pass = _run_audit_pass(case, device=device, profile=False)
        if baseline is not None:
            if device.type == "cpu":
                warmup_checks.append(bool(compare_audit_passes(baseline, warmup_pass)["passed"]))
            else:
                # Compare every CUDA warmup against the same CPU equivalence
                # reference; this catches state/cache drift before timing.
                warmup_checks.append(bool(compare_audit_passes(baseline, warmup_pass)["passed"]))
    timings: list[float] = []
    passes: list[dict[str, Any]] = []
    repeat_checks: list[dict[str, Any]] = []
    for _ in range(repeats):
        result = _run_audit_pass(case, device=device, profile=False)
        timings.append(float(result["elapsed_seconds"]))
        passes.append(result)
        if baseline is not None:
            repeat_checks.append(compare_audit_passes(baseline, result))
    return {
        "warmup": int(warmup),
        "repeats": int(repeats),
        "seconds": timings,
        "median_seconds": statistics.median(timings),
        "last_pass": passes[-1],
        "warmup_checks": warmup_checks,
        "repeat_checks": repeat_checks,
        "repeat_checks_passed": bool(all(warmup_checks) and all(check["passed"] for check in repeat_checks)),
    }


def run_gate(
    *,
    device: str = "cuda",
    cpu_selfcheck: bool = False,
    size: int = DEFAULT_RESOLUTION,
    seed: int = DEFAULT_SEED,
    synthetic_units: int = 1,
    manifest: str | Path | None = None,
    split: str = "train",
    max_units: int = DEFAULT_MAX_UNITS,
    unit_ids: Sequence[str] | None = None,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run the bounded gate and return a JSON-serialisable report mapping."""
    # Set/validate the deterministic cuBLAS workspace before even probing CUDA
    # availability; some driver probes initialize the CUDA context.
    _configure_cublas_workspace()
    if size != DEFAULT_RESOLUTION:
        raise GateError(f"resolution {size} is not allowed; this gate is frozen at {DEFAULT_RESOLUTION}")
    requested = str(device)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in ("cpu", "cuda"):
        raise GateError("device must be auto, cpu or cuda")
    if requested == "cpu" and not cpu_selfcheck:
        # A CPU-only invocation is intentionally an explicit selfcheck.  This
        # status cannot be confused with a CUDA gate or a speed claim.
        cpu_selfcheck = True
    provenance = _collect_provenance(requested_device=str(device), resolved_device=requested)
    with _deterministic_algorithms():
        if manifest is not None:
            units = load_bounded_training_units(
                manifest,
                split=split,
                image_size=size,
                seed=seed,
                max_units=max_units,
                unit_ids=unit_ids,
            )
            source = {
                "kind": "real_training_units",
                "manifest": str(Path(manifest).expanduser().resolve()),
                "split": split,
                "max_units": int(max_units),
                "unit_ids": [unit.fitting.unit_id for unit in units],
                "mask_inputs_used": False,
                "reference_inputs_used": False,
            }
        else:
            units = build_synthetic_units(size=size, seed=seed, count=synthetic_units)
            source = {
                "kind": "synthetic_image_only",
                "image_size": int(size),
                "seed": int(seed),
                "units": len(units),
                "mask_inputs_used": False,
                "reference_inputs_used": False,
            }
        case = freeze_case(units, seed=seed, source=source)
        source = _augment_case_source(source, case, manifest_path=manifest)
        case = GateCase(case.units, case.banks, source)
        cpu_pass = _run_audit_pass(case, device=torch.device("cpu"), profile=False)

        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_device": str(device),
            "resolved_device": requested,
            "status": None,
            "success": False,
            "source": source,
            "provenance": provenance,
            "contract": {
                "resolution": DEFAULT_RESOLUTION,
                "seed": int(seed),
                "bank_size": BANK_SIZE,
                "max_fit_iterations": MAX_FIT_ITERATIONS,
                "rounds": DEFAULT_ROUNDS,
                "improvement_threshold": DEFAULT_IMPROVEMENT_THRESHOLD,
                "deterministic_algorithms": True,
                "cublas_workspace_config": cpu_pass["cublas_workspace_config"],
                "observation_arithmetic": "FP64",
                "neural_models": "none in image-only gate; no dtype/AMP changes",
            },
            "cpu": {
                "elapsed_seconds": cpu_pass["elapsed_seconds"],
                "deterministic_algorithms_effective": cpu_pass["deterministic_algorithms_effective"],
                "work": cpu_pass["work"],
                "residency": cpu_pass["residency"],
                "status": "PASS" if cpu_pass["work"]["full_budget"]["all_units_match"] else "FAIL",
            },
            "gpu": {"status": "NOT RUN", "reason": None},
            "comparison": None,
            "timing": {
                "scope": "audit-only ObservationModel fit/score/regional work; excludes data I/O, producer, students, and optimizer",
                "speedup": None,
                "speedup_claim": "NOT CLAIMED",
            },
            "remaining_cpu_work": (
                "This gate executes one canonical audit_banks pass over every frozen unit and "
                "candidate (4 candidate fits per bank, fit max_iterations=5, rounds=2), plus "
                "regional score reductions inside the fixed 128-call/32-region ceilings."
            ),
            "real_data_gaps": [
                "No real-data claim is made unless --manifest is supplied and the bounded units load successfully.",
                "No verification/ground-truth masks are opened; clinical and scientific efficacy remain untested.",
                "CUDA timing and residency require the requested RTX device and matching worker API.",
            ],
        }

        if requested == "cuda" and not torch.cuda.is_available():
            report["status"] = "CUDA NOT RUN"
            report["gpu"] = {
                "status": "NOT RUN",
                "reason": "requested CUDA gate but torch.cuda.is_available() is false; no CPU fallback is treated as CUDA evidence",
            }
        elif cpu_selfcheck:
            # CPU selfcheck compares two fresh canonical CPU passes.  It is a
            # software gate for tests, not a hardware equivalence claim.
            cpu_second = _run_audit_pass(case, device=torch.device("cpu"), profile=False)
            comparison = compare_audit_passes(cpu_pass, cpu_second)
            report["comparison"] = comparison
            report["gpu"] = {
                "status": "NOT RUN",
                "reason": "CPU selfcheck requested; CUDA equivalence intentionally not attempted",
            }
            report["status"] = "CPU SELFCHECK PASS" if comparison["passed"] else "CPU SELFCHECK FAIL"
            report["success"] = bool(comparison["passed"])
        else:
            try:
                gpu_equivalence = _run_audit_pass(case, device=torch.device("cuda"), profile=True)
                comparison = compare_audit_passes(cpu_pass, gpu_equivalence)
                report["comparison"] = comparison
                report["gpu"] = {
                    "status": "PASS" if comparison["passed"] else "FAIL",
                    "elapsed_seconds": gpu_equivalence["elapsed_seconds"],
                    "work": gpu_equivalence["work"],
                    "residency": gpu_equivalence["residency"],
                    "profiler_available": gpu_equivalence["profiler_available"],
                }
                if comparison["passed"]:
                    # Symmetric warm-up/repeat protocol: a cold one-shot CPU
                    # denominator cannot be compared to a warmed GPU median.
                    cpu_timed = _timed_device_passes(
                        case,
                        device=torch.device("cpu"),
                        warmup=warmup,
                        repeats=repeats,
                        reference=cpu_pass,
                    )
                    gpu_timed = _timed_device_passes(
                        case,
                        device=torch.device("cuda"),
                        warmup=warmup,
                        repeats=repeats,
                        reference=cpu_pass,
                    )
                    repeat_pass = bool(
                        cpu_timed["repeat_checks_passed"] and gpu_timed["repeat_checks_passed"]
                    )
                    report["timing"] = {
                        "scope": "audit-only ObservationModel fit/score/regional work; excludes data I/O, producer, students, and optimizer",
                        "warmup": warmup,
                        "repeats": repeats,
                        "cpu_seconds": cpu_timed["seconds"],
                        "cpu_median_seconds": cpu_timed["median_seconds"],
                        "gpu_seconds": gpu_timed["seconds"],
                        "gpu_median_seconds": gpu_timed["median_seconds"],
                        "speedup": (
                            cpu_timed["median_seconds"] / gpu_timed["median_seconds"]
                            if repeat_pass
                            else None
                        ),
                        "repeat_checks_passed": repeat_pass,
                        "cpu_repeat_checks": [bool(check["passed"]) for check in cpu_timed["repeat_checks"]],
                        "gpu_repeat_checks": [bool(check["passed"]) for check in gpu_timed["repeat_checks"]],
                        "speedup_claim": (
                            "ALLOWED: work counters, decisions, floating tolerances, CUDA residency, "
                            "and every warmup/repeat equivalence check passed"
                            if repeat_pass
                            else "NOT CLAIMED: a warmup/repeat equivalence check drifted"
                        ),
                    }
                    report["success"] = repeat_pass
                    report["status"] = "PASS" if repeat_pass else "FAIL"
                else:
                    report["status"] = "FAIL"
                    report["timing"] = {
                        "scope": "audit-only ObservationModel fit/score/regional work; excludes data I/O, producer, students, and optimizer",
                        "speedup": None,
                        "speedup_claim": "NOT CLAIMED: equivalence/residency/work gate failed",
                    }
            except Exception as exc:  # noqa: BLE001 - hardware boundary must be reported
                report["status"] = "CUDA FAIL"
                report["gpu"] = {
                    "status": "FAIL",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                report["timing"] = {
                    "scope": "audit-only ObservationModel fit/score/regional work; excludes data I/O, producer, students, and optimizer",
                    "speedup": None,
                    "speedup_claim": "NOT CLAIMED: CUDA pass failed before a complete gate",
                }
        destination = _unique_output(Path(output) if output is not None else None)
        report["output_path"] = str(destination)
        _write_fresh_json(destination, report)
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-selfcheck", action="store_true", help="run two fresh CPU passes; never a CUDA claim")
    parser.add_argument("--manifest", type=Path, help="optional frozen image-only manifest")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--unit-id", action="append", dest="unit_ids", default=[])
    parser.add_argument("--max-units", type=int, default=DEFAULT_MAX_UNITS)
    parser.add_argument("--synthetic-units", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path, help="fresh JSON output; defaults to the system temp directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_gate(
            device=args.device,
            cpu_selfcheck=args.cpu_selfcheck,
            size=args.image_size,
            seed=args.seed,
            synthetic_units=args.synthetic_units,
            manifest=args.manifest,
            split=args.split,
            max_units=args.max_units,
            unit_ids=args.unit_ids,
            warmup=args.warmup,
            repeats=args.repeats,
            output=args.output,
        )
    except Exception as exc:  # noqa: BLE001 - CLI emits non-success evidence
        print(f"audit device gate error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if report["status"] in ("PASS", "CPU SELFCHECK PASS"):
        return 0
    if report["status"] == "CUDA NOT RUN":
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
