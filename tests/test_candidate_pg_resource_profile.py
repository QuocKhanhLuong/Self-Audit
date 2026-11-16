from __future__ import annotations

import pytest

from scripts.pg_resource_profile import (
    MEASURED_STEPS,
    THREAD_COUNTS,
    WARMUPS,
    choose_largest_frame,
    run_bounded_steps,
)


def test_choose_largest_native_train_frame_is_stable_and_does_not_accept_eval():
    records = [
        {"patient_id": "p2", "frame_index": 4, "shape": (4, 20, 30), "split": "train"},
        {"patient_id": "p1", "frame_index": 7, "shape": (5, 20, 24), "split": "train"},
        {"patient_id": "p1", "frame_index": 2, "shape": (5, 20, 24), "split": "train"},
    ]
    assert choose_largest_frame(records) == records[2]
    with pytest.raises(ValueError, match="TRAIN records only"):
        choose_largest_frame([dict(records[0], split="VAL18")])


def test_bounded_schedule_runs_three_warmups_and_five_measured_updates():
    calls = []
    ticks = iter(float(i) for i in range(100))
    syncs = []
    timings = run_bounded_steps(
        lambda: calls.append("step"), synchronize=lambda: syncs.append("sync"),
        warmups=WARMUPS, measured_steps=MEASURED_STEPS, clock=lambda: next(ticks),
    )
    assert len(calls) == WARMUPS + MEASURED_STEPS
    assert len(timings) == MEASURED_STEPS
    assert timings == [1.0] * MEASURED_STEPS
    assert len(syncs) == 2 + MEASURED_STEPS
    assert THREAD_COUNTS == (2, 4, 8)


def test_bounded_schedule_rejects_empty_measurements():
    with pytest.raises(ValueError, match="measured_steps positive"):
        run_bounded_steps(lambda: None, synchronize=lambda: None, measured_steps=0)
