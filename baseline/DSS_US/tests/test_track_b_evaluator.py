"""DSS-US native Track B: Step I evaluator behaviour and sealed-run execution (synthetic only)."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src")]
from shared_benchmark.native_artifacts import (  # noqa: E402
    ImageInventory, NativeContractError, RawRunWriter, array_hash, write_seal_receipt,
)
from shared_benchmark.native_protocol import ProtocolBlocked  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(f"dss_track_b_test_{name}", BASE / "evaluation/track_b" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matching = _load("matching")
step1 = _load("step1")
run = _load("run")


def test_equal_counts_use_hungarian_and_are_raw_id_invariant():
    gt = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
    partition = np.array([[7, 7, 3, 3], [7, 7, 9, 9]])
    mapping, branch = matching.match_segments(partition, gt)
    assert branch == "hungarian" and mapping == {7: 0, 3: 1, 9: 2}
    renamed = np.vectorize({7: 40, 3: 41, 9: 42}.get)(partition)
    assert matching.match_segments(renamed, gt)[0] == {40: 0, 41: 1, 42: 2}


def test_exclusive_majority_later_class_wins_only_on_strictly_greater_iou():
    # IoU(segment 5, class 1) = 1/3 (tied with segment 7, first index kept); IoU(segment 5, class 2) = 0.6.
    partition = np.array([[5, 5, 5, 5, 5, 7, 0, 0, 8]])
    gt = np.array([[1, 1, 2, 2, 2, 1, 0, 0, 0]])
    mapping, branch = matching.match_segments(partition, gt)
    assert branch == "exclusive_majority"
    assert mapping == {0: 0, 5: 2}   # class 2 displaces class 1; class 1 is not re-matched to segment 7


def test_exclusive_majority_loser_stays_unmatched_and_tie_keeps_the_earlier_class():
    # IoU(segment 5, class 1) = 0.5 > IoU(segment 5, class 2) = 0.4: class 2 loses and is not re-matched to 6.
    partition = np.array([[5, 5, 5, 5, 6, 0, 0, 9]])
    gt = np.array([[1, 1, 2, 2, 2, 0, 0, 0]])
    assert matching.match_segments(partition, gt) == ({0: 0, 5: 1}, "exclusive_majority")
    # Exact tie (0.5 vs 0.5) on segment 4: the earlier class 1 keeps it.
    partition = np.array([[4, 4, 9, 9, 11, 12]])
    gt = np.array([[1, 2, 0, 0, 0, 0]])
    assert matching.match_segments(partition, gt) == ({9: 0, 4: 1}, "exclusive_majority")


def test_unmatched_segments_become_background_and_absent_label_scores_zero():
    gt = np.array([[1, 1, 0, 0]])
    partition = np.array([[3, 3, 4, 5]])
    result = step1.per_image_segment_dice([("s", partition, gt)], n_classes=3)
    row = result["images"][0]
    assert row["dice_per_foreground_label"] == [1.0, 0.0]  # label 2 absent from GT and prediction
    assert row["dice"] == 0.5


def test_aggregation_is_mean_and_population_std():
    perfect = ("a", np.array([[3, 3, 4, 4]]), np.array([[1, 1, 0, 0]]))
    half = ("b", np.array([[3, 4, 4, 4]]), np.array([[1, 1, 0, 0]]))
    result = step1.per_image_segment_dice([perfect, half], n_classes=2)
    values = np.array([row["dice"] for row in result["images"]])
    assert result["dice_mean"] == pytest.approx(values.mean())
    assert result["dice_std_population"] == pytest.approx(values.std(ddof=0))


def test_grid_mismatch_and_non_integer_gt_are_refused():
    with pytest.raises(ValueError, match="grid mismatch"):
        step1.per_image_segment_dice([("s", np.zeros((2, 2), int), np.zeros((2, 3), int))], n_classes=2)
    with pytest.raises(ValueError, match="integer"):
        step1.per_image_segment_dice([("s", np.zeros((2, 2), int), np.zeros((2, 2), float))], n_classes=2)


def _sealed_step1_run(tmp_path, profile="step1_ours_comb"):
    import cv2
    config = json.loads((BASE / "config/native" / f"{profile}.yaml").read_text())
    staging = tmp_path / "images"
    staging.mkdir()
    image = np.random.default_rng(0).integers(0, 255, (4, 4, 3), dtype=np.uint8)
    cv2.imwrite(str(staging / "p1.png"), image)
    manifest = {"schema": "medical-native.image-only.v1", "dataset": "CAMUS",
                "records": [{"sample_id": "p1", "role": "image", "image_path": "images/p1.png",
                             "image_sha256": hashlib.sha256((staging / "p1.png").read_bytes()).hexdigest()}]}
    (tmp_path / "inventory.json").write_text(json.dumps(manifest))
    inventory = ImageInventory(tmp_path / "inventory.json", image_root=staging)
    writer = RawRunWriter(tmp_path / "raw", inventory=inventory, config=config, provenance={"seed": 1})
    partition = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], np.int32)
    writer.write(inventory.records[0], partition, {"input_native_hw": [4, 4], "decoded_image_sha256": array_hash(image)})
    writer.seal([])
    receipt = tmp_path / "receipts" / "raw.json"
    write_seal_receipt(tmp_path / "raw", receipt)
    gt = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], np.uint8)
    gt_path = tmp_path / "gt" / "p1.npy"
    gt_path.parent.mkdir()
    np.save(gt_path, gt)
    gt_manifest = tmp_path / "gt_manifest.json"
    gt_manifest.write_text(json.dumps({"schema": run.GT_SCHEMA, "dataset": "CAMUS", "records": [
        {"sample_id": "p1", "gt_path": "gt/p1.npy", "gt_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest()}]}))
    return tmp_path / "raw", receipt, gt_manifest, partition, gt


def test_step1_evaluates_a_sealed_run_after_receipt_and_seal(tmp_path):
    raw, receipt, gt_manifest, partition, gt = _sealed_step1_run(tmp_path)
    result = run.evaluate_native(raw, seal_receipt=receipt, gt_manifest=gt_manifest,
                                 protocol_path=BASE / "config/native/step1_ours_comb.yaml")
    expected = step1.per_image_segment_dice([("p1", partition, gt)], n_classes=4)
    assert result["dice_mean"] == expected["dice_mean"] == 1.0
    assert result["images"][0]["branch"] == "hungarian"
    assert result["raw_seal_sha256"] and result["gt_manifest_sha256"]


def test_tampered_raw_is_rejected_before_any_label_access(tmp_path):
    raw, receipt, gt_manifest, _, _ = _sealed_step1_run(tmp_path)
    np.savez_compressed(raw / "p1" / "raw.npz", partition=np.zeros((4, 4), "<i4"))
    gt_manifest.unlink()  # GT must never be needed when the raw run fails verification
    with pytest.raises(NativeContractError):
        run.evaluate_native(raw, seal_receipt=receipt, gt_manifest=gt_manifest,
                            protocol_path=BASE / "config/native/step1_ours_comb.yaml")


def test_reference_manifest_must_cover_exactly_the_sealed_samples_and_match_hashes(tmp_path):
    raw, receipt, gt_manifest, _, _ = _sealed_step1_run(tmp_path)
    document = json.loads(gt_manifest.read_text())
    document["records"][0]["gt_sha256"] = "0" * 64
    gt_manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="GT hash mismatch"):
        run.evaluate_native(raw, seal_receipt=receipt, gt_manifest=gt_manifest,
                            protocol_path=BASE / "config/native/step1_ours_comb.yaml")
    document["records"] = []
    gt_manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="exactly the sealed samples"):
        run.evaluate_native(raw, seal_receipt=receipt, gt_manifest=gt_manifest,
                            protocol_path=BASE / "config/native/step1_ours_comb.yaml")


def test_step1_evaluator_refuses_a_raw_run_from_a_different_profile(tmp_path):
    raw, receipt, gt_manifest, _, _ = _sealed_step1_run(tmp_path, profile="step1_dss_baseline")
    with pytest.raises(ValueError, match="stage/protocol mismatch"):
        run.evaluate_native(raw, seal_receipt=receipt, gt_manifest=gt_manifest,
                            protocol_path=BASE / "config/native/step1_ours_comb.yaml")


def test_step2_evaluation_stays_blocked_before_gt(tmp_path):
    raw, receipt, gt_manifest, _, _ = _sealed_step1_run(tmp_path)
    gt_manifest.unlink()
    with pytest.raises(ProtocolBlocked):
        run.evaluate_native(raw, seal_receipt=receipt, gt_manifest=gt_manifest,
                            protocol_path=BASE / "config/native/step2_ours_comb_ours.yaml")
