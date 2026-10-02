"""Shared ACDC adaptation for the native baselines: producer plumbing, seals, Track A and Track B.

Synthetic fixtures only (``acdc_synthetic``); no real ACDC data, checkpoint or GT is used.
Test names avoid annotation tokens because pytest derives temporary paths from them.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(HERE)]
import acdc_synthetic  # noqa: E402
from shared_benchmark import acdc_native as an  # noqa: E402
from shared_benchmark import acdc_tracks as at  # noqa: E402
from shared_benchmark.native_artifacts import NativeContractError  # noqa: E402
from shared_benchmark.native_protocol import file_hash, value_hash  # noqa: E402
from shared_benchmark.semantic_contract import BG, LV, MYO, RV, VOID, array_hash  # noqa: E402

ADAPTATIONS = sorted(ROOT.glob("baseline/*/config/acdc/*.json"))
PROVENANCE = {"repository": {"repository_commit_sha": "test"}, "code_identity": {"sha256": "test"}}


def _intensity_partition(source):
    """Deterministic image-only stand-in method: 3 intensity bands with deliberately odd raw IDs."""
    outputs = {}
    for record in source.records:
        gray = source.read_bgr(record)[:, :, 0]
        partition = np.where(gray < 85, 7, np.where(gray < 170, 1000, 3)).astype(np.int32)
        outputs[record["sample_id"]] = (partition, {"stand_in": "intensity_bands"})
    return outputs


def _produce(tmp_path, produce=_intensity_partition, *, patients=("patient001",), name="run"):
    contract, records, image_root, reference_root = acdc_synthetic.build(tmp_path / f"data_{name}", patients=patients)
    adaptation = {"schema": an.ADAPTATION_SCHEMA, "adaptation_id": "synthetic"}
    result = an.run_acdc_producer(
        contract=contract, records=records, image_root=image_root, output_root=tmp_path / name,
        receipt_path=tmp_path / "receipts" / f"{name}.json", method="SYNTHETIC", baseline_mode="synthetic-bands",
        adaptation=adaptation, adaptation_sha256=value_hash(adaptation), provenance=PROVENANCE, produce=produce,
        seed_for_record=lambda record: 5, trusted_roots=[sys.prefix, sys.base_prefix, ROOT / "src", HERE])
    return result, records, image_root, reference_root


# ---------------------------------------------------------------- input and producer contract

def test_input_conversion_is_explicit_deterministic_and_image_only():
    plane = np.linspace(-2.0, 3.0, 224 * 224, dtype=np.float32).reshape(224, 224)
    converted = an.central_plane_to_uint8(plane)
    assert converted.dtype == np.uint8 and converted.min() == 0 and converted.max() == 255
    assert np.array_equal(converted, an.central_plane_to_uint8(plane.copy()))
    assert not an.central_plane_to_uint8(np.full((4, 4), 1.5)).any()  # constant plane -> 0
    three = an.gray_to_three_channels(converted)
    assert three.shape == (224, 224, 3) and all(np.array_equal(three[:, :, c], converted) for c in range(3))
    assert an.INPUT_CONVERSION_SPEC["gt_derived"] is False


def test_producer_seals_unmapped_ids_on_the_shared_grid(tmp_path):
    result, records, _, _ = _produce(tmp_path)
    artifacts = result["verified"]["artifacts"]
    assert set(artifacts) == {record["sample_id"] for record in records}
    for record in records:
        artifact = artifacts[record["sample_id"]]
        assert artifact.partition.shape == (224, 224) and artifact.partition.dtype == np.int32
        assert set(np.unique(artifact.partition)) <= {3, 7, 1000}  # raw IDs exactly as produced
        assert artifact.metadata["source_image_sha256"] == record["source"]["sha256"]
        assert artifact.metadata["seed"] == 5
        provenance = artifact.metadata["baseline_provenance"]
        assert provenance["input_conversion"] == an.INPUT_CONVERSION
        source = an.AcdcImageSource([record], tmp_path / "data_run" / "acdc")
        assert provenance["central_image_sha256"] == array_hash(source.central_plane(record))


def test_producer_decodes_only_cohort_images_and_methods_touch_nothing_under_the_image_root(tmp_path):
    result, records, image_root, _ = _produce(tmp_path)
    run = result["verified"]["run"]
    assert run["provenance"]["decoded_sources"] == [{"locator": r["source"]["locator"], "sha256": r["source"]["sha256"]}
                                                    for r in records[:1]]
    log = json.loads((tmp_path / "run" / "access_log.json").read_text())
    assert all(entry["allowed"] for entry in log)
    root = str(image_root.resolve())
    assert not [entry for entry in log if entry["path"].startswith(root + "/")]


def test_decoding_refuses_a_source_whose_hash_changed(tmp_path):
    contract, records, image_root, _ = acdc_synthetic.build(tmp_path / "data")
    with (image_root / records[0]["source"]["locator"]).open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(Exception, match="hash mismatch"):
        an.AcdcImageSource(records, image_root)


def test_firewall_stops_a_producer_that_opens_the_reference(tmp_path):
    def peeking(source):
        locator = source.records[0]["source"]["locator"].replace(".nii", "_gt.nii")
        (source.image_root / locator).read_bytes()

    with pytest.raises(NativeContractError, match="GT firewall"):
        _produce(tmp_path, peeking)
    assert not (tmp_path / "run" / "acdc_raw_seal.json").exists()
    assert not (tmp_path / "receipts" / "run.json").exists()


def test_raw_maps_must_be_on_the_shared_grid(tmp_path):
    def wrong(source):
        return {r["sample_id"]: (np.zeros((223, 224), np.int32), {}) for r in source.records}

    with pytest.raises(NativeContractError, match="shared grid"):
        _produce(tmp_path, wrong)


def test_seal_receipt_and_tamper_rejection(tmp_path):
    result, _, _, _ = _produce(tmp_path)
    root, receipt = tmp_path / "run", tmp_path / "receipts" / "run.json"
    an.verify_acdc_seal_receipt(root, receipt)
    with pytest.raises(FileExistsError):
        an.write_acdc_seal_receipt(root, receipt)
    with pytest.raises(NativeContractError, match="outside"):
        an.write_acdc_seal_receipt(root, root / "inside.json")
    artifact = next(iter(result["verified"]["artifacts"].values()))
    tampered = artifact.partition.copy()
    tampered[0, 0] = 999
    np.save(artifact.directory / "raw_partition.npy", tampered)
    with pytest.raises((NativeContractError, ValueError)):
        an.verify_acdc_seal_receipt(root, receipt)


def test_run_level_tamper_and_stray_files_are_rejected(tmp_path):
    _produce(tmp_path)
    root = tmp_path / "run"
    (root / "notes.txt").write_text("x")
    with pytest.raises(NativeContractError, match="unsealed"):
        an.verify_acdc_raw_run(root)
    (root / "notes.txt").unlink()
    run = json.loads((root / "run.json").read_text())
    run["expected_samples"] = run["expected_samples"][:1]
    (root / "run.json").write_text(json.dumps(run))
    with pytest.raises(NativeContractError, match="run.json"):
        an.verify_acdc_raw_run(root)


def test_repeat_runs_are_deterministic(tmp_path):
    first, _, _, _ = _produce(tmp_path, name="first")
    second, _, _, _ = _produce(tmp_path, name="second")
    a, b = first["verified"]["artifacts"], second["verified"]["artifacts"]
    assert {k: v.metadata["raw_partition_sha256"] for k, v in a.items()} == \
           {k: v.metadata["raw_partition_sha256"] for k, v in b.items()}


# ---------------------------------------------------------------- adaptation configs

@pytest.mark.parametrize("path", ADAPTATIONS, ids=lambda p: p.stem)
def test_adaptations_bind_unchanged_runnable_profiles(path):
    adaptation, source, _ = an.load_adaptation(path)
    from shared_benchmark.native_protocol import load_lock
    profile = load_lock(source)
    assert adaptation["source_profile"]["sha256"] == file_hash(source)
    assert adaptation["source_profile"]["paper_equivalence"] == profile["paper_equivalence"]
    for forbidden in ("scientific", "paper_unspecified", "implementation_conventions", "conditional_values"):
        assert forbidden not in adaptation  # plumbing only: no scientific value is redeclared
    assert adaptation["acdc_contract"]["input_conversion"] == an.INPUT_CONVERSION_SPEC


def test_exactly_the_runnable_profiles_are_adapted():
    names = {path.stem for path in ADAPTATIONS}
    assert names == {"acdc_ph2_paper_faithful_declared_conventions", "acdc_sysu_us_paper_faithful_declared_conventions",
                     "acdc_ph2_official_reference", "acdc_sysu_us_official_reference",
                     "acdc_step2_dss_baseline_dss_paper_faithful_declared_conventions"}


def test_producer_code_never_imports_track_b():
    producers = [ROOT / "src/shared_benchmark/acdc_native.py", ROOT / "baseline/SGSCN/scripts/run_acdc.py",
                 ROOT / "baseline/DSS_US/scripts/run_acdc.py"]
    for path in producers:
        text = path.read_text()
        assert "acdc_tracks" not in text and "majority" not in text and "gt_root" not in text


# ---------------------------------------------------------------- Track A

def test_track_a_has_no_reference_input_and_reads_only_images(tmp_path):
    assert not any("gt" in name or "reference" in name for name in inspect.signature(at.run_track_a).parameters)
    _, records, image_root, _ = _produce(tmp_path)
    index = at.run_track_a(tmp_path / "run", tmp_path / "receipts" / "run.json", image_root, tmp_path / "track_a",
                           trusted_roots=[sys.prefix, sys.base_prefix, ROOT / "src", HERE])
    log = json.loads((tmp_path / "track_a" / "access_log.json").read_text())
    assert all(entry["allowed"] for entry in log) and not any("_gt" in entry["path"] for entry in log)
    assert {row["sample_id"] for row in index["samples"]} == {r["sample_id"] for r in records}
    _, _, maps = at.verify_track_a(tmp_path / "run", tmp_path / "receipts" / "run.json", image_root, tmp_path / "track_a")
    assert all(m.dtype == np.uint8 and set(np.unique(m)) <= {BG, RV, MYO, LV, VOID} for m in maps.values())


def test_track_a_refuses_a_tampered_raw_run(tmp_path):
    result, _, image_root, _ = _produce(tmp_path)
    (tmp_path / "run" / "access_log.json").write_text("[]\n")
    with pytest.raises(NativeContractError):
        at.run_track_a(tmp_path / "run", tmp_path / "receipts" / "run.json", image_root, tmp_path / "track_a")
    assert not (tmp_path / "track_a").exists()


# ---------------------------------------------------------------- Track B rule

def test_majority_vote_simple_case():
    raw = np.array([[5, 5, 9], [5, 9, 9]])
    reference = np.array([[0, 0, 3], [1, 3, 3]])
    semantic, rows = at.raw_id_majority_vote(raw, reference)
    assert semantic.tolist() == [[BG, BG, LV], [BG, LV, LV]]
    assert {row["raw_id"]: row["assigned"] for row in rows} == {5: BG, 9: LV}


def test_majority_vote_maps_many_ids_to_one_class():
    raw = np.array([[1, 2, 3, 4]])
    reference = np.array([[2, 2, 2, 0]])
    semantic, _ = at.raw_id_majority_vote(raw, reference)
    assert semantic.tolist() == [[MYO, MYO, MYO, BG]]


@pytest.mark.parametrize("values,expected", [([0, 1], BG), ([1, 2], RV), ([2, 3], MYO), ([3, 1], RV), ([3, 2, 1, 0], BG)])
def test_majority_vote_tie_order(values, expected):
    semantic, _ = at.raw_id_majority_vote(np.zeros((1, len(values)), int), np.array([values]))
    assert set(np.unique(semantic)) == {expected}


def test_majority_vote_never_splits_ids_or_components():
    raw = np.array([[4, 0, 4], [0, 0, 0], [4, 0, 4]])  # raw ID 4 = four separate components
    reference = np.array([[3, 0, 3], [0, 0, 0], [3, 0, 1]])
    semantic, _ = at.raw_id_majority_vote(raw, reference)
    assert set(semantic[raw == 4]) == {LV}  # the RV corner stays with its raw ID


def test_majority_vote_void_without_valid_reference_pixels():
    raw = np.array([[1, 1, 2]])
    reference = np.array([[255, 255, 0]])
    semantic, rows = at.raw_id_majority_vote(raw, reference)
    assert semantic.tolist() == [[VOID, VOID, BG]]
    assert next(r for r in rows if r["raw_id"] == 1)["valid_counts"] == [0, 0, 0, 0]


def test_raw_restoration_uses_the_frozen_nearest_exact_inverse():
    import torch
    import torch.nn.functional as F
    raw = np.random.default_rng(0).integers(0, 50, (224, 224))
    restored = at.restore_raw_partition(raw, (40, 48))
    expected = F.interpolate(torch.from_numpy(raw.astype(np.float32))[None, None], size=(40, 48), mode="nearest-exact")
    assert np.array_equal(restored, expected[0, 0].numpy().astype(np.int64))


# ---------------------------------------------------------------- both tracks, one raw run

def test_both_tracks_consume_the_same_sealed_raw_maps(tmp_path):
    result, records, image_root, reference_root = _produce(tmp_path, patients=("patient001", "patient002"))
    raw, receipt = tmp_path / "run", tmp_path / "receipts" / "run.json"
    at.run_track_a(raw, receipt, image_root, tmp_path / "track_a",
                   trusted_roots=[sys.prefix, sys.base_prefix, ROOT / "src", HERE])
    summary_a = at.evaluate_track_a(raw, receipt, image_root, tmp_path / "track_a", reference_root, tmp_path / "eval_a")
    summary_b = at.evaluate_track_b(raw, receipt, reference_root, tmp_path / "eval_b")
    seal = result["verified"]["seal"]["seal_sha256"]
    assert summary_a["raw_seal_sha256"] == summary_b["raw_seal_sha256"] == seal
    assert summary_a["tracks_contract_sha256"] == summary_b["tracks_contract_sha256"] == file_hash(an.TRACKS_CONTRACT)
    assert summary_b["kind"] == "GT_ASSISTED_DIAGNOSTIC_ONLY" and summary_b["rule"] == "raw_id_majority_vote_v1"
    assert len(summary_b["per_volume"]) == len(summary_a["per_volume"]) == 2
    for record in records:
        semantic = json.loads((tmp_path / "track_a" / next(
            row["directory"] for row in json.loads((tmp_path / "track_a" / "track_a_index.json").read_text())["samples"]
            if row["sample_id"] == record["sample_id"]) / "metadata.json").read_text())
        raw_metadata = result["verified"]["artifacts"][record["sample_id"]].metadata
        assert semantic["raw_partition_sha256"] == raw_metadata["raw_partition_sha256"]
        assert semantic["raw_artifact_scientific_hash"] == raw_metadata["scientific_payload_hash"]
    # Track B never writes into the raw or Track A roots
    assert sorted(p.name for p in raw.iterdir()) == sorted(an._RUN_ENTRIES)
    assert sorted(p.name for p in (tmp_path / "track_a").iterdir()) == sorted(at._TRACK_A_ENTRIES)


def test_track_b_verifies_the_seal_before_any_reference_access(tmp_path):
    _produce(tmp_path)
    (tmp_path / "run" / "access_log.json").write_text("[]\n")
    with pytest.raises(NativeContractError):
        at.evaluate_track_b(tmp_path / "run", tmp_path / "receipts" / "run.json", tmp_path / "absent", tmp_path / "eval_b")
    assert not (tmp_path / "eval_b").exists()


def test_track_b_output_cannot_live_in_the_raw_root(tmp_path):
    _, _, _, reference_root = _produce(tmp_path)
    with pytest.raises(at.TrackError):
        at.evaluate_track_b(tmp_path / "run", tmp_path / "receipts" / "run.json", reference_root, tmp_path / "run" / "b")


def test_common_metrics_need_complete_volumes():
    record = {"volume_id": "v", "depth": 3, "slice_index": 0}
    with pytest.raises(at.TrackError, match="complete volumes"):
        at._volumes([record])


# ---------------------------------------------------------------- CLIs

def test_tracks_cli_runs_track_b_on_a_sealed_run(tmp_path):
    import subprocess
    _, _, _, reference_root = _produce(tmp_path)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_acdc_native_tracks.py"), "track-b",
                             "--raw-root", str(tmp_path / "run"), "--receipt", str(tmp_path / "receipts" / "run.json"),
                             "--gt-root", str(reference_root), "--output-dir", str(tmp_path / "eval_b")],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout)["kind"] == "GT_ASSISTED_DIAGNOSTIC_ONLY"


def test_readiness_reports_separate_statuses_and_blocked_data(tmp_path):
    import subprocess
    result = subprocess.run([sys.executable, str(ROOT / "scripts/acdc_native_readiness.py"),
                             "--image-root", str(tmp_path / "absent")], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout)
    assert report["ACDC_DATA_STATUS"] == "BLOCKED_DATA" and report["ACDC_FULL_RUN_STATUS"].startswith("NOT_STARTED")
    for method in ("DSS-US", "SGSCN"):
        status = report[method]
        assert status["ACDC_PRODUCER_STATUS"] == "PROTOCOL_READY" and status["ACDC_PRODUCER_RUNNABLE_NOW"] is False
        assert status["ACDC_TRACK_A_STATUS"].startswith("ADAPTER_READY")
        assert status["ACDC_TRACK_B_STATUS"].startswith("EVALUATOR_READY")
    assert any("checkpoint" in reason for reason in report["DSS-US"]["data_reasons"])
    assert not any("checkpoint" in reason for reason in report["SGSCN"]["data_reasons"])
