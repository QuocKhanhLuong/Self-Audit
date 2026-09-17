"""Independent mask-free audit device-gate checks.

These tests are bounded CPU software checks.  They do not turn a CPU run into
CUDA, RTX, scientific, clinical, or throughput evidence; a requested CUDA run
is explicitly ``NOT RUN`` when no CUDA device is available.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import scripts.check_maskfree_audit_device as gate


def test_synthetic_fixture_is_resolution_224_image_only_and_non_degenerate() -> None:
    units = gate.build_synthetic_units(size=224, seed=42, count=1)
    unit = units[0]
    assert tuple(unit.fitting.image.shape) == (1, 224, 224)
    assert tuple(unit.selection.image.shape) == (1, 224, 224)
    assert unit.fitting.support.dtype is torch.bool
    assert unit.selection.support.dtype is torch.bool
    assert int(unit.fitting.support.sum()) > 0
    assert int(unit.selection.support.sum()) > 0
    assert not bool((unit.fitting.support & unit.selection.support).any())
    assert unit.record["mask_inputs_used"] is False
    assert unit.record["reference_inputs_used"] is False
    assert float(unit.fitting.image[0][unit.fitting.support].std()) > 0.0


def test_freeze_case_uses_four_cpu_generated_candidates() -> None:
    case = gate.freeze_case(gate.build_synthetic_units(size=224, seed=42, count=2), seed=42)
    assert len(case.banks) == 2
    assert all(len(bank) == 4 for bank in case.banks)
    assert all(candidate.labels.device.type == "cpu" for bank in case.banks for candidate in bank)
    assert all(bank[0].metadata["bank_id"] for bank in case.banks)
    assert len({candidate.candidate_id for candidate in case.banks[0]}) == 4
    assert case.banks[0][0].metadata["bank_id"] != case.banks[1][0].metadata["bank_id"]


def test_cpu_selfcheck_passes_full_budget_without_speedup_claim(tmp_path: Path) -> None:
    output = tmp_path / "cpu-selfcheck.json"
    report = gate.run_gate(
        device="cpu",
        cpu_selfcheck=True,
        size=224,
        seed=42,
        synthetic_units=1,
        warmup=0,
        repeats=1,
        output=output,
    )
    assert report["status"] == "CPU SELFCHECK PASS"
    assert report["success"] is True
    assert report["comparison"]["passed"] is True
    assert report["comparison"]["full_budget"]["passed"] is True
    assert report["comparison"]["deterministic_algorithms"]["passed"] is True
    assert report["cpu"]["work"]["full_budget"]["all_units_match"] is True
    assert report["timing"]["speedup"] is None
    assert report["timing"]["speedup_claim"] == "NOT CLAIMED"
    assert report["provenance"]["source_identity"]["git"]["commit"]
    assert report["provenance"]["source_identity"]["package"]["combined"]
    assert report["provenance"]["gate_script"]["sha256"]
    assert report["provenance"]["hardware"]["status"] == "CPU"
    assert report["source"]["unit_tensor_identity"][0]["fitting"]["image_sha256"]
    assert len(report["source"]["frozen_bank_signatures"][0]) == 4
    assert output.is_file()
    with pytest.raises(gate.GateError, match="refusing to overwrite"):
        gate.run_gate(
            device="cpu",
            cpu_selfcheck=True,
            size=224,
            seed=42,
            synthetic_units=1,
            warmup=0,
            repeats=1,
            output=output,
        )


def test_requested_cuda_without_device_is_not_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = gate.run_gate(
        device="cuda",
        size=224,
        seed=42,
        synthetic_units=1,
        warmup=0,
        repeats=1,
        output=tmp_path / "cuda-not-run.json",
    )
    assert report["status"] == "CUDA NOT RUN"
    assert report["success"] is False
    assert report["gpu"]["status"] == "NOT RUN"
    assert "no CPU fallback" in report["gpu"]["reason"]
    assert report["timing"]["speedup"] is None
    assert report["provenance"]["hardware"]["device_identity"] is None
    assert report["provenance"]["hardware"]["status"] == "NOT AVAILABLE"


def test_resolution_is_frozen_at_224() -> None:
    with pytest.raises(gate.GateError, match="resolution 128 is not allowed"):
        gate.run_gate(device="cpu", cpu_selfcheck=True, size=128, output=Path("/private/tmp/unused.json"))


def test_default_output_is_fresh_under_system_tmp() -> None:
    output = gate._unique_output(None)
    try:
        assert output.parent == Path(tempfile.gettempdir()).resolve()
        assert not output.exists()
    finally:
        if output.exists():
            output.unlink()


def test_invalid_cublas_workspace_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "unsupported")
    with pytest.raises(gate.GateError, match="CUBLAS_WORKSPACE_CONFIG"):
        gate.run_gate(device="cpu", cpu_selfcheck=True, size=224, output=Path("/private/tmp/unused.json"))


def _fresh_cpu_passes() -> tuple[dict, dict]:
    case = gate.freeze_case(gate.build_synthetic_units(size=224, seed=42, count=1), seed=42)
    return (
        gate._run_audit_pass(case, device=torch.device("cpu")),
        gate._run_audit_pass(case, device=torch.device("cpu")),
    )


def test_comparator_rejects_nonfinite_fit_parameter() -> None:
    left, right = _fresh_cpu_passes()
    right["results"][0].fitted[0].parameters["components"][0]["mean"] = float("nan")
    comparison = gate.compare_audit_passes(left, right)
    assert comparison["passed"] is False
    assert any(row["nonfinite_count"] for row in comparison["floating"]["comparisons"])


def test_comparator_rejects_discrete_fallback_flag_change() -> None:
    left, right = _fresh_cpu_passes()
    component = right["results"][0].fitted[0].parameters["components"][0]
    component["fallback"] = not component["fallback"]
    comparison = gate.compare_audit_passes(left, right)
    assert comparison["passed"] is False
    assert comparison["floating"]["passed"] is False


def test_comparator_rejects_candidate_dtype_change_even_when_values_match() -> None:
    left, right = _fresh_cpu_passes()
    candidate = right["results"][0].bank[0]
    candidate.probabilities = candidate.probabilities.to(torch.float64)
    comparison = gate.compare_audit_passes(left, right)
    assert comparison["passed"] is False
    assert comparison["candidate_content"]["passed"] is False
    assert comparison["candidate_content"]["units"][0]["candidates"][0]["probabilities_bitwise_equal"] is False


def test_comparator_rejects_score_availability_and_reason_change() -> None:
    left, right = _fresh_cpu_passes()
    score = right["results"][0].scores[0]
    score.available = not score.available
    score.reason = "injected comparator negative control"
    comparison = gate.compare_audit_passes(left, right)
    assert comparison["passed"] is False
    assert comparison["floating"]["passed"] is False


def test_comparator_rejects_regional_challenger_trace_change() -> None:
    left, right = _fresh_cpu_passes()
    regions = right["results"][0].trace["search_notes"]["regions"]
    regions[0]["disagreeing_candidates"] = ["injected-regional-challenger"]
    comparison = gate.compare_audit_passes(left, right)
    assert comparison["passed"] is False
    assert comparison["structure"]["passed"] is False


def test_comparator_uses_fp64_bound_for_regional_score_trace() -> None:
    left, right = _fresh_cpu_passes()
    regions = right["results"][0].trace["search_notes"]["regions"]
    scored = next((row for row in regions if row.get("available")), None)
    if scored is None:
        pytest.skip("fixture produced no scored regional row")
    scored["margin_nats_per_pixel"] += 1e-7
    comparison = gate.compare_audit_passes(left, right)
    assert comparison["passed"] is False
    assert comparison["regional_trace"]["passed"] is False


def test_comparator_rejects_regional_output_dtype_change() -> None:
    left, right = _fresh_cpu_passes()
    result = right["results"][0]
    result.regional_margin = result.regional_margin.to(torch.float64)
    comparison = gate.compare_audit_passes(left, right)
    assert comparison["passed"] is False
    regional = next(
        row for row in comparison["floating"]["comparisons"] if row["label"].endswith("regional_margin")
    )
    assert regional["dtype_equal"] is False


def test_timing_repeats_are_checked_against_equivalence_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    case = gate.freeze_case(gate.build_synthetic_units(size=224, seed=42, count=1), seed=42)
    reference = gate._run_audit_pass(case, device=torch.device("cpu"))
    original = gate._run_audit_pass

    def drifted(case_arg, *, device, profile=False):  # type: ignore[no-untyped-def]
        current = original(case_arg, device=device, profile=profile)
        current["results"][0].scores[0].available = not current["results"][0].scores[0].available
        return current

    monkeypatch.setattr(gate, "_run_audit_pass", drifted)
    timed = gate._timed_device_passes(
        case,
        device=torch.device("cpu"),
        warmup=0,
        repeats=1,
        reference=reference,
    )
    assert timed["repeat_checks_passed"] is False
    assert timed["repeat_checks"][0]["passed"] is False


class _FakeProfiler:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    def key_averages(self) -> list[object]:
        return self._events


class _FakeEvent:
    def __init__(self, *, device_time: float, key: str = "aten::matmul") -> None:
        self.device_type = "DeviceType.CUDA"
        self.device_time_total = device_time
        self.key = key


def test_empty_profiler_has_no_heavy_cuda_residency_evidence() -> None:
    assert gate._heavy_cuda_event_rows(_FakeProfiler([])) == []


def test_zero_duration_cuda_event_has_no_heavy_cuda_residency_evidence() -> None:
    profiler = _FakeProfiler([_FakeEvent(device_time=0.0)])
    assert gate._heavy_cuda_event_rows(profiler) == []
