"""Scoped tests for the bounded canonical mask-free profiling harness."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import pytest

import scripts.profile_maskfree as profiler
from scripts.profile_maskfree import (
    REPO_ROOT,
    Instrumentation,
    TimingProbe,
    _compare_reports,
    make_synthetic_fixture,
    percentile,
    snapshot_source,
    verify_snapshot,
)


def test_percentile_is_empty_safe_and_deterministic() -> None:
    assert percentile([], 0.5) is None
    assert percentile([1.0], 0.95) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_synthetic_fixture_is_image_only_and_reproducible(tmp_path: Path) -> None:
    first = make_synthetic_fixture(tmp_path / "one", seed=17)
    second = make_synthetic_fixture(tmp_path / "two", seed=17)
    assert first["shape"] == [128, 128, 1024]
    assert first["depth"] >= 256
    assert first["dtype"] == "float32"
    assert first["source_sha256"] == second["source_sha256"]
    assert first["mask_inputs_used"] is False
    assert first["reference_inputs_used"] is False
    assert not list(tmp_path.rglob("*mask*"))


def test_source_snapshot_rejects_added_changed_and_missing_bytes(tmp_path: Path) -> None:
    """Snapshot reuse must fail closed instead of mixing live source bytes."""
    for mutation in ("added", "changed", "missing"):
        destination = tmp_path / mutation
        manifest = snapshot_source(REPO_ROOT, destination)
        target = destination / "files" / "src" / "self_audit_maskfree" / "auditor.py"
        if mutation == "added":
            target.with_name("unexpected.py").write_text("# drift\n", encoding="utf-8")
        elif mutation == "changed":
            target.write_bytes(target.read_bytes() + b"\n# drift\n")
        else:
            target.unlink()
        with pytest.raises(ValueError, match="immutable source snapshot mismatch"):
            verify_snapshot(destination, manifest)
    manifest_destination = tmp_path / "manifest-hash"
    manifest = snapshot_source(REPO_ROOT, manifest_destination)
    manifest_path = manifest_destination / "snapshot_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["combined_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable source snapshot mismatch"):
        verify_snapshot(manifest_destination, manifest_payload)


class _FakeTiming:
    @contextmanager
    def stage(self, _name: str, **_details):
        yield


class _FakeTrainer:
    timing = _FakeTiming()
    dataset: ClassVar[Any] = None
    component_steps: ClassVar[dict[str, int]] = {"producer": 0}
    device: ClassVar[Any] = None


class _LegacyAuditTrainer:
    def _audit_unit(self):
        return "legacy-result"


class _AuditProbe:
    def __init__(self) -> None:
        self.results: list[Any] = []

    def add_audit(self, result: Any) -> None:
        self.results.append(result)


def test_audit_hook_falls_back_to_legacy_unit_sink_without_double_hooking() -> None:
    trainer = _LegacyAuditTrainer()
    probe = _AuditProbe()
    instrumentation = Instrumentation(trainer, probe)  # type: ignore[arg-type]
    instrumentation._install_audit_hook()
    assert instrumentation.audit_hook_name == "_audit_unit"
    assert trainer._audit_unit() == "legacy-result"
    assert probe.results == ["legacy-result"]
    instrumentation.restore()
    assert trainer._audit_unit() == "legacy-result"


def test_timing_probe_keeps_nested_inclusive_and_exclusive_accounting() -> None:
    probe = TimingProbe(_FakeTrainer())
    probe.begin_batch(epoch=0, batch_index=0, indices=[])
    with probe.stage("outer"), probe.stage("inner"):
        pass
    batch = probe.active_batch
    assert batch is not None
    probe.end_batch(batch, success=True)
    outer = probe.batches[0]["stage_seconds"]["outer"]
    inner = probe.batches[0]["stage_seconds"]["inner"]
    assert outer["inclusive_seconds"] >= inner["inclusive_seconds"]
    assert outer["exclusive_seconds"] >= 0.0
    assert inner["exclusive_seconds"] >= 0.0
    assert probe.batches[0]["top_level_stage_inclusive_seconds"] >= outer["inclusive_seconds"]


def test_one_measured_batch_accounts_for_both_students_without_fabricating_steps(tmp_path: Path) -> None:
    """The scoped integration gate executes the real canonical trainer once.

    It catches a profiler that accidentally replaces the dataset/model path or
    hides a skipped student update.  One eight-slice fixture is exactly one
    physical batch, so the assertion is independent of measured-batch count.
    """
    source_snapshot = tmp_path / "source_snapshot"
    snapshot_source(REPO_ROOT, source_snapshot)
    output = tmp_path / "profile"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "profile_maskfree.py"),
            "--synthetic",
            "--source-root",
            str(source_snapshot),
            "--output",
            str(output),
            "--warmup",
            "0",
            "--measured",
            "1",
            "--run-id",
            "pytest-profile",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "profile_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_bounded"
    assert report["profiler"]["measured_batches_captured"] == 1
    batch = next(row for row in report["batches"] if row["phase"] == "measured")
    assert batch["success"] is True
    assert batch["student_loss_calls"] == 2
    assert batch["producer_loss_calls"] == 1
    assert batch["optimizer_calls"] == 1
    assert report["accounting"]["component_steps"] == {
        "producer": 1,
        "student_audited": 1,
        "student_no_audit": 1,
    }
    assert batch["finite"] == {"gradient": True, "loss": True}
    assert batch["audit"]["fits"] > 0
    assert batch["audit"]["score_calls"] >= batch["audit"]["primary_score_calls"]
    assert batch["connected_components_calls"] > 0
    assert report["units"]["mask_inputs_used"] is False
    assert report["units"]["reference_inputs_used"] is False
    assert report["timing"]["full_scientific_batch_seconds"]["p50_seconds"] is not None
    assert report["units"]["unit_count"] >= 256
    assert report["source"]["observed_import"]["package_root"].endswith("self_audit_maskfree")
    assert report["source"]["observed_import"]["import_root"] == report["source"]["activation"]["import_root"]
    assert all(
        path.startswith(report["source"]["activation"]["package_root"])
        for path in report["source"]["observed_import"]["module_paths"].values()
    )
    assert report["source"]["snapshot_check"]["ok"] is True
    assert report["initial_state"]["model_state_sha256"]
    assert report["sample_order_digest"]
    assert report["accounting"]["audit"]["units"] == 8
    assert report["accounting"]["audit"]["fits"] == 32
    assert report["accounting"]["audit"]["fit_steps"] == 160
    assert report["accounting"]["audit"]["score_calls"] > 0
    bulk = report["profiler"]["canonical_bulk_audit"]
    if bulk["enabled_before_hooks"]:
        assert bulk["enabled_during_hooks"] is True
        assert bulk["enabled_after_restore"] is True
        assert batch["logical_work"]["fit_many"]["items"] == 32


def test_original_snapshot_uses_legacy_audit_hook_in_subprocess(tmp_path: Path) -> None:
    """The immutable pre-optimization source remains executable via fallback."""
    source_snapshot = REPO_ROOT / "reports" / "maskfree150" / "local_baseline" / "source_snapshot"
    if not (source_snapshot / "snapshot_manifest.json").is_file():
        pytest.skip("original local baseline snapshot is not present")
    output = tmp_path / "original-profile"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "profile_maskfree.py"),
            "--synthetic",
            "--source-root",
            str(source_snapshot),
            "--output",
            str(output),
            "--warmup",
            "0",
            "--measured",
            "1",
            "--run-id",
            "pytest-original-profile",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "profile_report.json").read_text(encoding="utf-8"))
    assert report["profiler"]["canonical_bulk_audit"]["audit_hook_name"] == "_audit_unit"
    assert report["accounting"]["audit"]["units"] == 8
    assert report["accounting"]["audit"]["fits"] == 32
    assert report["accounting"]["audit"]["fit_steps"] == 160
    assert report["source"]["snapshot_check"]["ok"] is True
    self_match = _compare_reports(report, report)
    assert self_match["matched"] is True
    tampered_counter = json.loads(json.dumps(report))
    tampered_counter["accounting"]["audit"]["units"] -= 1
    assert _compare_reports(report, tampered_counter)["matched"] is False
    tampered_gain = json.loads(json.dumps(report))
    tampered_gain["batches"][0]["audit_decisions"][0]["accepted_gain"] = 1.0
    assert _compare_reports(report, tampered_gain)["matched"] is False
    tampered_score = json.loads(json.dumps(report))
    score_record = tampered_score["batches"][0]["audit_decisions"][0]["fit_scores"][0]
    score_record["total"] = (score_record["total"] or 0.0) + 1.0
    assert _compare_reports(report, tampered_score)["matched"] is False
    missing_harness = json.loads(json.dumps(report))
    missing_harness["harness"]["script_sha256"] = None
    assert _compare_reports(report, missing_harness)["matched"] is False
    tampered_bulk = json.loads(json.dumps(report))
    tampered_bulk["profiler"]["canonical_bulk_audit"] = {
        "enabled_before_hooks": True,
        "enabled_during_hooks": False,
        "enabled_after_restore": True,
    }
    assert _compare_reports(report, tampered_bulk)["matched"] is False


def test_main_fails_closed_when_reference_match_is_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        profiler,
        "run_profile",
        lambda _args: {
            "status": "completed_bounded",
            "profiler": {"measured_batches_captured": 1},
            "matched_report": {"matched": False},
        },
    )
    code = profiler.main(
        [
            "--synthetic",
            "--output",
            str(tmp_path / "stub-profile"),
            "--measured",
            "1",
            "--reference-report",
            str(tmp_path / "reference.json"),
        ]
    )
    assert code == 3
    assert json.loads(capsys.readouterr().out)["matched"] is False
