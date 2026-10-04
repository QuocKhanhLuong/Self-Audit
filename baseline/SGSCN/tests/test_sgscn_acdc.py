"""SGSCN on the frozen ACDC contract: synthetic end-to-end producer runs and CLI gates.

Synthetic NIfTI fixtures only; no real ACDC data. Each run trains one network per slice
exactly as the source profile declares (locked architecture, optimizer, LR, losses, stopping).
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src"), str(BASE / "scripts"), str(ROOT / "tests/native_baselines")]
import acdc_synthetic  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("sgscn_run_acdc", BASE / "scripts/run_acdc.py")
run_acdc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_acdc)
from sgscn.paper_faithful import PAPER_LEARNING_RATE, sample_seed  # noqa: E402
from shared_benchmark.acdc_native import verify_acdc_seal_receipt  # noqa: E402

ADAPTATIONS = sorted((BASE / "config/acdc").glob("*.json"))
PAPER_FAITHFUL = BASE / "config/acdc/acdc_ph2_paper_faithful_declared_conventions.json"
OFFICIAL = BASE / "config/acdc/acdc_sysu_us_official_reference.json"


def _run(tmp_path, adaptation, name):
    contract, records, image_root, _ = acdc_synthetic.build(tmp_path / f"data_{name}", shape=(40, 48, 1))
    result = run_acdc.run(contract=contract, records=records, image_root=image_root, output=tmp_path / name,
                          receipt=tmp_path / "receipts" / f"{name}.json", adaptation_path=adaptation, threads=2,
                          extra_trusted_roots=[HERE])
    return result, records


@pytest.mark.parametrize("path", ADAPTATIONS, ids=lambda p: p.stem)
def test_every_adaptation_is_protocol_ready_and_keeps_its_profile(path):
    adaptation, _, config, settings, *_ = run_acdc.load_profile(path)
    assert adaptation["source_profile"]["profile"] == config["profile"]
    assert settings["learning_rate"] == (PAPER_LEARNING_RATE[config["dataset"]] if "paper_faithful" in path.stem
                                         else config["scientific"]["learning_rate"])
    result = subprocess.run([sys.executable, str(BASE / "scripts/run_acdc.py"), "--adaptation", str(path), "--check-protocol"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "ACDC_PRODUCER_READY"


def test_missing_acdc_images_are_blocked_data_without_output(tmp_path):
    output = tmp_path / "raw"
    result = subprocess.run([sys.executable, str(BASE / "scripts/run_acdc.py"), "--adaptation", str(PAPER_FAITHFUL),
                             "--image-root", str(tmp_path / "absent"), "--output", str(output),
                             "--receipt", str(tmp_path / "receipt.json")], capture_output=True, text=True, timeout=120)
    assert result.returncode == 3, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "BLOCKED_DATA" and any("ACDC image root" in reason for reason in report["reasons"])
    assert not output.exists()


def test_a_changed_source_profile_blocks_the_adaptation(tmp_path):
    changed = json.loads(PAPER_FAITHFUL.read_text())
    changed["source_profile"]["sha256"] = "0" * 64
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))
    result = subprocess.run([sys.executable, str(BASE / "scripts/run_acdc.py"), "--adaptation", str(path), "--check-protocol"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 2 and "source native profile changed" in result.stdout


def test_paper_faithful_profile_runs_end_to_end_deterministically(tmp_path):
    first, records = _run(tmp_path, PAPER_FAITHFUL, "first")
    second, _ = _run(tmp_path, PAPER_FAITHFUL, "second")
    verified = verify_acdc_seal_receipt(tmp_path / "first", tmp_path / "receipts" / "first.json")
    for record in records:
        artifact = verified["artifacts"][record["sample_id"]]
        provenance = artifact.metadata["baseline_provenance"]
        assert artifact.partition.shape == (224, 224) and artifact.partition.dtype == np.int32
        assert artifact.metadata["seed"] == sample_seed(1, record["sample_id"]) == provenance["seed"]
        assert provenance["stop_rule"] == "official_max_iterations_min_labels" and provenance["iterations"] <= 50
        assert provenance["input_conversion"] == "acdc_historical224_central_plane_minmax_uint8_gray3_v1"
        repeat = second["verified"]["artifacts"][record["sample_id"]]
        assert np.array_equal(artifact.partition, repeat.partition)
    run = verified["run"]
    assert run["provenance"]["profile_class"] == "PAPER_FAITHFUL_WITH_DECLARED_CONVENTIONS"
    assert run["provenance"]["paper_equivalence"].startswith("NOT_EXACT_PAPER_REPRODUCTION")
    assert run["method"] == "SGSCN" and run["baseline_mode"] == "acdc_v12_ph2_paper_faithful_declared_conventions"
    assert all(entry["allowed"] for entry in json.loads((tmp_path / "first" / "access_log.json").read_text()))


def test_official_reference_profile_runs_and_stays_labelled(tmp_path):
    result, records = _run(tmp_path, OFFICIAL, "official")
    run = result["verified"]["run"]
    assert run["provenance"]["implementation_kind"] == "official_code_reference"
    assert run["provenance"]["profile_class"] == "OFFICIAL_REFERENCE"
    artifact = result["verified"]["artifacts"][records[0]["sample_id"]]
    assert artifact.partition.shape == (224, 224)


def test_both_tracks_consume_the_sgscn_raw_run(tmp_path):
    from shared_benchmark import acdc_tracks
    contract, records, image_root, reference_root = acdc_synthetic.build(tmp_path / "data_tracks", shape=(40, 48, 1))
    result = run_acdc.run(contract=contract, records=records, image_root=image_root, output=tmp_path / "raw",
                          receipt=tmp_path / "receipts" / "raw.json", adaptation_path=OFFICIAL, threads=2,
                          extra_trusted_roots=[HERE])
    raw, receipt = tmp_path / "raw", tmp_path / "receipts" / "raw.json"
    index = acdc_tracks.run_track_a(raw, receipt, image_root, tmp_path / "track_a",
                                    trusted_roots=[sys.prefix, sys.base_prefix, ROOT / "src", HERE])
    summary_a = acdc_tracks.evaluate_track_a(raw, receipt, image_root, tmp_path / "track_a", reference_root, tmp_path / "eval_a")
    summary_b = acdc_tracks.evaluate_track_b(raw, receipt, reference_root, tmp_path / "eval_b")
    assert summary_a["raw_seal_sha256"] == summary_b["raw_seal_sha256"] == result["verified"]["seal"]["seal_sha256"]
    assert index["raw_seal_sha256"] == result["verified"]["seal"]["seal_sha256"]
    mapping = summary_b["raw_id_mappings"][records[0]["sample_id"]]
    raw_ids = set(np.unique(result["verified"]["artifacts"][records[0]["sample_id"]].partition).tolist())
    assert {row["raw_id"] for row in mapping} <= raw_ids  # whole raw IDs, never split
