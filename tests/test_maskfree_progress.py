"""Terminal visibility is software evidence; no GPU/data quality claim."""
import io
import json
import random
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch

from self_audit_maskfree.progress import TerminalProgress, current_progress


def test_blocked_load_has_live_heartbeat_and_exact_item(tmp_path):
    stream = io.StringIO()
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    journal = tmp_path / "progress.jsonl"
    with TerminalProgress(stream=stream, heartbeat_seconds=.02) as progress:
        progress.attach(journal)
        progress.update(epoch=1, batch=1, batches=3)
        with progress.stage("data_load"):
            with progress.stage("data.decode_slice_stack", path="patient001_4d.nii.gz", frame=7):
                progress.update(loading_frame=7)
                # Wait for the actual background reporter, not a manually called heartbeat.
                deadline = time.monotonic() + 2
                while "heartbeat" not in stream.getvalue() and time.monotonic() < deadline:
                    threading.Event().wait(.01)
                assert "heartbeat" in stream.getvalue()
                assert "stage.done" not in stream.getvalue()
        progress.event("after_load")
        with pytest.raises(RuntimeError, match="broken header"):
            with progress.stage("inventory.header", path="bad.nii.gz"):
                raise RuntimeError("broken header")
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    heartbeat = next(row for row in events if row["event"] == "heartbeat")
    assert heartbeat["path"] == "patient001_4d.nii.gz"
    assert heartbeat["frame"] == 7
    assert heartbeat["phase_path"] == "data_load > data.decode_slice_stack"
    assert heartbeat["phase_seconds"] > 0
    assert any(row["event"] == "stage.error" and row["path"] == "bad.nii.gz" for row in events)
    assert "loading_frame" not in next(row for row in events if row["event"] == "after_load")
    assert not current_progress().enabled
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_gui_terminal_dashboard_is_readable_through_pipe_and_journal_appends(tmp_path):
    stream = io.StringIO()  # no isatty capability; same display under tee
    journal = tmp_path / "progress.jsonl"
    for epoch in (1, 2):
        with TerminalProgress(stream=stream) as progress:
            progress.attach(journal)
            progress.update(dataset="acdc", epoch=epoch, epochs=150, batch=1, batches=10)
            progress.dashboard(force=True, lr=.001, component_steps={"producer": 1},
                metrics={"producer/loss": .5, "student_no_audit/loss": .8,
                         "student_audited/loss": .7, "audit/select_nll": None},
                metric_counts={"producer/loss": 1}, gpu_stats=None,
                verification_status="locked_until_all_predictions_frozen")
    output = stream.getvalue()
    assert "epoch=2/150 batch=1/10" in output
    assert "student_no_audit: loss=0.8" in output
    assert "student_audited: loss=0.7" in output
    assert "select_nll=n/a" in output
    assert "\x1b" not in output  # captured GUI output has no escape/cursor garbage
    metrics = [r for r in map(json.loads, journal.read_text().splitlines()) if r["event"] == "metrics"]
    assert len(metrics) == 2
    assert metrics[0]["gpu_stats"] is None


def test_training_with_telemetry_matches_quiet_training(tmp_path):
    from test_maskfree_trainer import build_components, build_config, _flat_state
    from self_audit_maskfree.trainer import MaskfreeTrainer
    torch.set_num_threads(1)
    quiet = MaskfreeTrainer(build_config(tmp_path / "quiet", max_steps=1), components=build_components())
    quiet.run()
    stream = io.StringIO()
    with TerminalProgress(stream=stream, heartbeat_seconds=.02) as progress:
        loud = MaskfreeTrainer(build_config(tmp_path / "loud", max_steps=1), components=build_components())
        progress.attach(loud.paths.reports / "progress.jsonl")
        loud.run()
    for key, value in _flat_state(quiet).items():
        assert torch.equal(value, _flat_state(loud)[key]), key
    assert quiet.history[0]["audit"] == loud.history[0]["audit"]
    assert quiet.history[0]["coverage"] == loud.history[0]["coverage"]
    events = list(map(json.loads, (loud.paths.reports / "progress.jsonl").read_text().splitlines()))
    row = next(r for r in events if r["event"] == "metrics")
    assert row["metric_counts"]["producer/loss"] == 1
    assert row["global_optimizer_steps"] == 1
    assert row["gpu_stats"] is None
    assert any(r["event"] == "epoch.summary" and not r["epoch_complete"] for r in events)


def test_inventory_cli_reports_geometry_failure_before_training(tmp_path):
    import nibabel as nib
    data = tmp_path / "images"
    data.mkdir()
    for index in range(2):
        affine = np.eye(4)
        affine[0, 3] = index * 4
        image = nib.Nifti1Image(np.full((24, 24, 2), index, dtype=np.float32), affine)
        image.header.set_xyzt_units("mm", "sec")
        nib.save(image, data / f"patient001_frame{index:02d}.nii.gz")
    output = tmp_path / "inventory"
    result = subprocess.run([sys.executable, "scripts/prepare_maskfree_data.py", "--dataset", "acdc",
        "--root", str(data), "--output", str(output), "--no-probe"],
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 3, result.stdout + result.stderr
    inventory = json.loads((output / "inventory_acdc.json").read_text())
    assert inventory["inventory_status"] == "FAILED"
    assert inventory["training_status"] == "NOT_STARTED"
    assert inventory["geometry_conflict"]["study_id"] == "acdc:patient001"
    assert len(inventory["geometry_conflict"]["geometries"]) == 2
    assert not (output / "manifest_acdc.json").exists()
    assert "imports.load" in result.stderr
    assert "patient001_frame00.nii.gz" in result.stderr
    assert "patient001_frame01.nii.gz" in result.stderr
