"""Native full-volume Candidate-A 3-D integration contracts."""

import csv
import json

import pytest
import torch

from scripts.smoke_candidate_a3d import run_smoke
from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
from self_audit.candidate_a3d.geometry import canonical_geometry
from self_audit.candidate_a3d.io import read_checkpoint
from self_audit.model import digest


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("candidate_a3d") / "run"
    return out, run_smoke(out)


def test_full_volume_stages_export_and_metrics(smoke):
    out, receipt = smoke
    assert receipt["status"] == "PASS_SYNTHETIC_ONLY"
    assert receipt["export_bitwise_parity"]
    assert receipt["export_payload_keys"] == ["student"]
    assert receipt["stages"]["teacher_pretrain"]["student_updates"] == 0
    assert receipt["stages"]["teacher_warmup"]["student_updates"] == 0
    assert receipt["stages"]["joint"]["student_updates"] > 0
    assert (out / "student" / "weights.pt").is_file()
    assert (out / "evaluation" / "EPOCH_METRICS.json").is_file()


def test_each_epoch_covers_frames_not_slices_and_logs_branches(smoke):
    out, _ = smoke
    # Two labeled patients and T=2 means four native full-volume updates.
    epoch = json.loads((out / "joint" / "epochs.jsonl").read_text().splitlines()[0])
    assert epoch["updates"] == 4
    rows = list(csv.DictReader((out / "joint" / "metrics.csv").open()))
    assert {row["class"] for row in rows} == {"BG", "RV", "MYO", "LV"}
    assert {row["branch"] for row in rows if row["split"] == "train_online"} >= {
        "student", "teacher_region", "teacher_boundary"
    }
    for stage in ("teacher_pretrain", "teacher_warmup", "joint"):
        stage_rows = list(csv.DictReader((out / stage / "metrics.csv").open()))
        expected = {"student"} if stage == "joint" else {"teacher_region", "teacher_boundary"}
        assert {row["branch"] for row in stage_rows if row["split"] == "validation"} == expected
        for split, branch in {(row["split"], row["branch"]) for row in stage_rows}:
            group = [row for row in stage_rows if (row["split"], row["branch"]) == (split, branch)]
            assert {row["class"] for row in group} == {"BG", "RV", "MYO", "LV"}
        for row in stage_rows:
            assert row["aggregation"] == "pooled_voxels"
            if row["reference_present"] == "True":
                tp, fp, fn = (int(row[key]) for key in ("TP", "FP", "FN"))
                assert float(row["dice"]) == pytest.approx(2 * tp / (2 * tp + fp + fn))
                if row["split"] == "validation":
                    assert row["macro_patient_dice"] != ""


def test_data_contract_is_native_full_volume_and_geometry_is_not_resampled(tmp_path):
    fixture = synthetic_fixture(
        tmp_path / "fixture", n_labeled=2, n_unlabeled=1, n_evaluation=1,
        shape_tzyx=(2, 3, 5, 6),
    )
    data = CandidateA3DData(fixture["train"], fixture["train_sha256"], synthetic=True)
    keys = data.labeled_frame_keys()
    assert len(keys) == 4
    image, labels, geometry_data, _ = data.load_labeled_frame(keys[0])
    geometry = canonical_geometry(geometry_data)
    assert tuple(image.shape) == (1, 1, 3, 5, 6)
    assert tuple(labels.shape) == (1, 3, 5, 6)
    assert tuple(geometry["shape"]) == (3, 5, 6)
    assert geometry["context"] == "full_volume"
    assert tuple(geometry["spacing_zyx"]) == (2.5, 0.7, 1.3)


def test_final_checkpoint_contains_teacher_but_export_does_not(smoke):
    out, _ = smoke
    payload, metadata = read_checkpoint(
        out / "joint" / "final", digest(out / "joint" / "final" / "MANIFEST.json"))
    assert payload["teacher"] is not None
    exported = torch.load(out / "student" / "weights.pt", weights_only=True)
    assert set(exported) == {"student"}
    assert metadata["prediction_model"] == "student"
