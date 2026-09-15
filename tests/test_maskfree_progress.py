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
    with TerminalProgress(stream=stream, heartbeat_seconds=.02, mode="verbose") as progress:
        progress.attach(journal)
        progress.update(epoch=1, batch=1, batches=3)
        with progress.stage("data_load"), progress.stage("data.decode_slice_stack",
                                                           path="patient001_4d.nii.gz", frame=7):
            progress.update(loading_frame=7)
            # Wait for the actual background reporter, not a manually called heartbeat.
            deadline = time.monotonic() + 2
            while "heartbeat" not in stream.getvalue() and time.monotonic() < deadline:
                threading.Event().wait(.01)
            assert "heartbeat" in stream.getvalue()
            assert "stage.done" not in stream.getvalue()
        progress.event("after_load")
        with pytest.raises(RuntimeError, match="broken header"), progress.stage(
            "inventory.header", path="bad.nii.gz"
        ):
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
        with TerminalProgress(stream=stream, mode="verbose") as progress:
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
    from test_maskfree_trainer import _flat_state, build_components, build_config

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
        capture_output=True, text=True, timeout=45, check=False)
    assert result.returncode == 3, result.stdout + result.stderr
    inventory = json.loads((output / "inventory_acdc.json").read_text())
    assert inventory["inventory_status"] == "FAILED"
    assert inventory["training_status"] == "NOT_STARTED"
    assert inventory["geometry_conflict"]["study_id"] == "acdc:patient001"
    assert len(inventory["geometry_conflict"]["geometries"]) == 2
    assert not (output / "manifest_acdc.json").exists()
    assert str(output / "inventory_acdc.json") in result.stderr


def test_compact_tqdm_hides_detail_but_preserves_journal_and_epoch_summary(tmp_path):
    stream = io.StringIO()
    with TerminalProgress(stream=stream, mode="compact") as progress:
        progress.attach(tmp_path / "progress.jsonl")
        progress.update(dataset="acdc", epoch=1, epochs=150, batches=2)
        progress.event("epoch.start", resumed_from_batch=0)
        for batch in (1, 2):
            progress.update(batch=batch)
            with progress.stage("data.decode_slice_stack", path="do_not_print_this_file.nii.gz"):
                progress.event("discovery.header_read", native_affine=[[1, 0], [0, 1]])
                progress.heartbeat()
            progress.dashboard(force=True, metrics={"producer/loss": .5,
                "student_no_audit/loss": .8, "student_audited/loss": .7})
        progress.event("epoch.summary", global_epoch=0, epoch_complete=True, epoch_seconds=12.3,
            batch_cursor=2, producer={"producer/loss": .5},
            students={"student_no_audit/loss": .8, "student_audited/loss": .7},
            audit={"select_nll_mean": 1.5})
    output = stream.getvalue()
    assert "Epoch 1/150 Train" in output
    assert "2/2" in output
    assert "time=12.3s" in output
    assert output.count("loss P/U/A=") == 1
    assert "val Dice=--" in output
    for hidden in ("do_not_print_this_file", "native_affine", "heartbeat", "stage.start", "metric_counts"):
        assert hidden not in output
    events = list(map(json.loads, (tmp_path / "progress.jsonl").read_text().splitlines()))
    assert any(row["event"] == "heartbeat" for row in events)
    assert any("native_affine" in row for row in events)
    assert any(row.get("path") == "do_not_print_this_file.nii.gz" for row in events)


def test_delayed_phase_bar_closes_line_before_next_console_message():
    stream = io.StringIO()
    with TerminalProgress(stream=stream, mode="compact") as progress:
        with progress.stage("model.load"):
            threading.Event().wait(.55)
            progress.heartbeat()
        progress.event("run.result", status="partial", report="report.json")
    assert "model.load" in stream.getvalue()
    assert "\n[maskfree] run.result:" in stream.getvalue()


def test_epoch_train_and_validation_use_separate_compact_bars(tmp_path):
    stream = io.StringIO()
    journal = tmp_path / "progress.jsonl"
    with TerminalProgress(stream=stream, mode="compact") as progress:
        progress.attach(journal)
        progress.update(dataset="acdc", epoch=3, epochs=150, batches=2)
        progress.event("epoch.start", resumed_from_batch=0)
        progress.dashboard(force=True, metrics={
            "producer/loss": 0.5,
            "student_no_audit/loss": 0.8,
            "student_audited/loss": 0.7,
        })
        progress.event("epoch.train_done", epoch=3)
        progress.event("validation.start", epoch=99, global_epoch=2, total_epochs=150, units=4,
                       batches=2, operation="images")
        progress.event("validation.batch", global_epoch=2, batch=1, batches=2, operation="images")
        with progress.stage("validation.freeze", operation="freeze"):
            progress.event("validation.batch", global_epoch=2, batch=2, batches=2, operation="freeze")
        progress.event("validation.end", epoch=99, global_epoch=2)
        progress.event(
            "epoch.summary",
            global_epoch=2,
            epochs=150,
            epoch_complete=True,
            epoch_seconds=12.3,
            total_elapsed_seconds=15.7,
            producer={"producer/loss": 0.5},
            students={
                "student_no_audit/loss": 0.8,
                "student_audited/loss": 0.7,
            },
            audit={"select_nll_mean": 1.5},
            validation={
                "status": "available",
                "available": True,
                "reason": None,
                "elapsed_seconds": 3.4,
                "inference_seconds": 2.8,
                "reference_seconds": 0.6,
                "students": {
                    "student_no_audit": {
                        "dice": 0.81,
                        "iou": 0.70,
                        "hd95": 4.2,
                        "assd": 1.1,
                        "dice_per_class": {"RV": 0.71, "MYO": 0.81, "LV": 0.91},
                        "patients": 2,
                        "volumes": 4,
                    },
                    "student_audited": {
                        "dice": 0.82,
                        "iou": 0.72,
                        "hd95": 4.0,
                        "assd": 1.0,
                        "dice_per_class": {"RV": 0.72, "MYO": 0.82, "LV": 0.92},
                        "patients": 2,
                        "volumes": 4,
                    },
                },
            },
        )

    output = stream.getvalue()
    assert output.index("Train") < output.index("Val")
    assert output.count("loss P/U/A=") == 1
    assert "Epoch 3/150" in output
    assert "train=12.3s val=3.4s total=15.7s" in output
    assert "student_no_audit: Dice=0.8100 IoU=0.7000 Dice(RV/MYO/LV)=0.7100/0.8100/0.9100 val=3.4s" in output
    assert "student_audited: Dice=0.8200 IoU=0.7200 Dice(RV/MYO/LV)=0.7200/0.8200/0.9200 val=3.4s" in output
    assert "freeze" in output
    assert "validation.freeze" not in output
    assert "\x1b" not in output

    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert any(row["event"] == "epoch.train_done" for row in events)
    assert any(row["event"] == "validation.start" and row["operation"] == "images"
               for row in events)
    assert sum(row["event"] == "validation.batch" for row in events) == 2
    assert any(row["event"] == "stage.start" and row["phase"] == "validation.freeze"
               and row["operation"] == "freeze" for row in events)
    summary = next(row for row in events if row["event"] == "epoch.summary")
    assert summary["validation"]["students"]["student_audited"]["dice"] == 0.82


def test_epoch_summary_keeps_unavailable_validation_as_dashes(tmp_path):
    stream = io.StringIO()
    with TerminalProgress(stream=stream, mode="compact") as progress:
        progress.attach(tmp_path / "progress.jsonl")
        progress.event(
            "epoch.summary",
            global_epoch=0,
            epochs=150,
            epoch_complete=True,
            epoch_seconds=1.0,
            producer={"producer/loss": 0.1},
            students={},
            audit={},
            validation={
                "status": "unavailable",
                "available": False,
                "reason": "reference masks absent",
                "elapsed_seconds": 2.0,
                "students": {
                    "student_no_audit": {"dice": 0.0},
                    "student_audited": {"dice": 0.0},
                },
            },
        )
    output = stream.getvalue()
    assert "train=1.0s val=2.0s total=--" in output
    assert "val Dice=-- (reference masks absent)" in output
    assert "student_no_audit:" not in output
    assert "student_audited:" not in output
    assert "Dice=0.0000" not in output


def test_zero_width_pty_keeps_tqdm_visible():
    import os
    import pty
    import fcntl
    import struct
    import termios

    master, slave = pty.openpty()
    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
        with os.fdopen(os.dup(slave), "w") as stream:
            with TerminalProgress(stream=stream, mode="compact") as progress:
                progress.event("epoch.start", dataset="acdc", epoch=1, total_epochs=150, batches=2)
                progress.event("epoch.train_done")
        output = os.read(master, 8192).decode()
        assert "Epoch 1/150 Train" in output
    finally:
        os.close(master)
        os.close(slave)
