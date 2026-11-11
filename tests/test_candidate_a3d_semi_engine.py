"""Integration contracts for image-only U; manufactured data only."""
import json
from pathlib import Path

import numpy as np
import pytest

from self_audit.candidate_a3d.config import DEFAULT_SEMI_SUPERVISED_CONFIG
from self_audit.candidate_a3d.data import synthetic_fixture
from self_audit.candidate_a3d.engine import _semi_epoch_plan, train
from self_audit.candidate_a3d.io import read_checkpoint


def test_semi_schedule_covers_patients_and_cycles_frames():
    labeled = [(0, 0), (0, 3)]
    unlabeled = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    rng = np.random.default_rng(42)
    visits = {}
    first_l, first_u = _semi_epoch_plan(labeled, unlabeled, 2, rng, visits)
    second_l, second_u = _semi_epoch_plan(labeled, unlabeled, 2, rng, visits)
    assert set(first_l) == set(second_l) == set(labeled)
    assert set(first_u) == {(0, 0), (1, 0)}
    assert set(second_u) == {(0, 1), (1, 1)}
    with pytest.raises(ValueError, match="cover"):
        _semi_epoch_plan(labeled, unlabeled, 1, rng, visits)


def test_teacher_free_semi_uses_u_without_main_teacher_or_gt(tmp_path):
    fixture = synthetic_fixture(tmp_path / "data", n_labeled=2, n_unlabeled=2,
                                n_evaluation=1, shape_tzyx=(2, 8, 16, 16))
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/candidate_a3d/smoke.json").read_text())
    config.update(use_teacher=False, freeze_teacher=True, validation_every=0,
                  checkpoint_every_epochs=0)
    config["semi_supervised"] = dict(DEFAULT_SEMI_SUPERVISED_CONFIG,
                                     enabled=True, confidence_threshold=0.0)
    output = tmp_path / "run"
    report = train(config, fixture["train"], fixture["train_sha256"], output,
                   device="cpu", synthetic=True)
    assert report["label_reads"] == ["train_labeled_000", "train_labeled_001"]
    assert report["unlabeled_image_reads"] == ["train_unlabeled_000", "train_unlabeled_001"]
    assert report["unlabeled_samples"] == 4
    assert report["student_unlabeled_updates"] == 4
    assert report["teacher_updates"] == report["teacher_unlabeled_updates"] == 0
    payload, metadata = read_checkpoint(output / "final", report["final_manifest_sha256"])
    assert payload["teacher"] is None
    assert payload["training_state"]["teacher_ema"] is None
    assert payload["training_state"]["student_ema"]
    assert "image-only U consistency" in metadata["supervision"]
    updates = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    assert all(row["unlabeled"]["policy_targets"] == 0 for row in updates)
    assert all(row["unlabeled"]["ground_truth_reads"] == 0 for row in updates)
