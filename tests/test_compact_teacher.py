"""Unit tests for compact teacher identity, lineage preflight, cache, safe historical bridge, and CLI."""

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from self_audit.compact_data import CompactTrainingData, synthetic_compact_fixture
from self_audit.compact_teacher import (
    HISTORICAL_BASELINE_PINS,
    HISTORICAL_MANIFEST_SHA256,
    HISTORICAL_PAYLOAD_SHA256,
    HISTORICAL_SOURCE_CLOSURE,
    HISTORICAL_SOURCE_COMMIT,
    FixedTeacherCache,
    build_teacher_cache,
    find_checkpoint_payload,
    import_historical_package,
    is_historical_baseline_checkpoint,
    load_teacher_for_cache,
    preflight_teacher_lineage,
    validate_teacher_identity,
    verify_historical_source,
)
from self_audit.model import build_model, digest


def test_historical_pins_match_repo_evidence():
    """Verify HISTORICAL_BASELINE_PINS match historical report manifests."""
    evidence_path = Path("reports/baseline_recovery/HISTORICAL_EVIDENCE.json")
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text())
        assert evidence["refs"]["history"] == HISTORICAL_SOURCE_COMMIT
        assert evidence["historical_source_closure"] == HISTORICAL_SOURCE_CLOSURE

    manifest_path = Path("reports/window_pilot_test50_20261024/baseline/CHECKPOINT_MANIFEST.json")
    if manifest_path.is_file():
        assert digest(manifest_path) == HISTORICAL_MANIFEST_SHA256
        manifest = json.loads(manifest_path.read_text())
        assert manifest["source_sha256"] == HISTORICAL_SOURCE_CLOSURE
        assert manifest["payload_sha256"] == HISTORICAL_PAYLOAD_SHA256
        assert manifest["prediction_model"] == "student"
        assert manifest["labeled_ids"] == HISTORICAL_BASELINE_PINS["labeled_ids"]


def test_validate_teacher_identity_canonical_requires_all_hashes_no_pins():
    """Canonical teacher accepts NO pins: mandatory checkpoint/source/payload hashes, full EMA, quality receipt."""
    # Missing hashes
    incomplete = {
        "teacher_kind": "canonical",
        "inference_state": "student",
        "labeled_ids": ["p001"],
        "label_budget": 1,
    }
    with pytest.raises(ValueError, match="mandatory 64-hex 'manifest_sha256'"):
        validate_teacher_identity(incomplete)

    # Hashes present but not full EMA method
    with_hashes = {
        "teacher_kind": "canonical",
        "inference_state": "student",
        "manifest_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "payload_sha256": "c" * 64,
        "training_method": "supervised",  # Not full EMA
        "labeled_ids": ["p001"],
        "label_budget": 1,
    }
    with pytest.raises(ValueError, match="requires a full EMA training method"):
        validate_teacher_identity(with_hashes)

    # Missing quality receipt
    with_hashes["training_method"] = "ema_full"
    with pytest.raises(ValueError, match="requires an explicit, verified quality receipt"):
        validate_teacher_identity(with_hashes)

    # Full canonical identity with quality receipt
    with_hashes["quality_receipt"] = {"status": "COMPLETED", "macro4": 0.88, "foreground_dice": 0.84}
    receipt = validate_teacher_identity(with_hashes)
    # Cannot stamp verified=True from claimed fields alone without actual manifest checked
    assert receipt["verified"] is False
    assert receipt["verification_status"] == "CLAIMED_METADATA_UNVERIFIED"


def test_validate_teacher_identity_forged_manifest_rejected(tmp_path):
    """Negative test: Forged manifest file or SHA mismatch must be rejected."""
    manifest_file = tmp_path / "forged_manifest.json"
    manifest_file.write_text(json.dumps({"tampered": True}))

    identity = {
        "teacher_kind": "canonical",
        "inference_state": "student",
        "manifest_sha256": "f" * 64,  # Forged expected SHA
        "source_sha256": "1" * 64,
        "payload_sha256": "2" * 64,
        "training_method": "ema_full",
        "quality_receipt": {"macro4": 0.88},
        "labeled_ids": ["p001"],
        "label_budget": 1,
    }
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        validate_teacher_identity(identity, manifest_path=manifest_file)


def test_validate_teacher_identity_forged_payload_rejected(tmp_path):
    """Negative test: Forged payload file or SHA mismatch must be rejected."""
    manifest_file = tmp_path / "valid_manifest.json"
    manifest_file.write_text(json.dumps({"schema": "self_audit.checkpoint.v2"}))
    m_sha = digest(manifest_file)

    payload_file = tmp_path / "forged_payload.pt"
    payload_file.write_bytes(b"tampered_payload_bytes")

    identity = {
        "teacher_kind": "canonical",
        "inference_state": "student",
        "manifest_sha256": m_sha,
        "source_sha256": "1" * 64,
        "payload_sha256": "9" * 64,  # Forged expected SHA
        "training_method": "ema_full",
        "quality_receipt": {"macro4": 0.88},
        "labeled_ids": ["p001"],
        "label_budget": 1,
    }
    with pytest.raises(ValueError, match="payload hash mismatch"):
        validate_teacher_identity(
            identity,
            manifest_path=manifest_file,
            payload_path=payload_file,
        )


def test_validate_teacher_identity_unknown_lineage_rejected():
    """Negative test: Missing lineage or empty IDs must fail closed."""
    with pytest.raises(ValueError, match="teacher labeled IDs are mandatory"):
        validate_teacher_identity({
            "teacher_kind": "canonical",
            "inference_state": "student",
            "labeled_ids": [],
            "label_budget": 0,
        })

    with pytest.raises(ValueError, match="invalid character in patient ID"):
        validate_teacher_identity({
            "teacher_kind": "canonical",
            "inference_state": "student",
            "labeled_ids": ["../evil_patient"],
            "label_budget": 1,
        })


def test_k8_teacher_combined_with_k1_student_rejected(tmp_path):
    """Negative test: K=8 teacher combined with K=1 student budget must be rejected."""
    manifest_path, manifest_sha = synthetic_compact_fixture(tmp_path / "k1_data", n_labeled=1, n_unlabeled=2)
    data = CompactTrainingData(manifest_path, manifest_sha, synthetic=True, allow_k1=True)

    # Teacher with 8 labeled IDs
    teacher_lids = [f"patient_{i:03d}" for i in range(8)]
    with pytest.raises(ValueError, match="exceeds declared student budget K=1"):
        validate_teacher_identity(
            {
                "teacher_kind": "synthetic",
                "inference_state": "student",
                "labeled_ids": teacher_lids,
                "label_budget": 8,
            },
            data=data,
        )

    # Direct preflight must also reject K=8 union for K=1 budget
    with pytest.raises(ValueError, match="exceeds label budget K=1"):
        preflight_teacher_lineage(
            teacher_labeled_ids=teacher_lids,
            data=data,
            label_budget=1,
        )


def test_synthetic_teacher_with_real_data_rejected(tmp_path):
    """Negative test: Synthetic teacher kind is strictly prohibited with real patient data."""
    # Create non-synthetic manifest
    data_dir = tmp_path / "real_sim_data"
    manifest_path, manifest_sha = synthetic_compact_fixture(data_dir, n_labeled=2, n_unlabeled=2)
    raw_m = json.loads(manifest_path.read_text())
    raw_m["synthetic"] = False
    manifest_path.write_text(json.dumps(raw_m))
    real_sha = digest(manifest_path)

    data = CompactTrainingData(manifest_path, real_sha, synthetic=False)

    with pytest.raises(ValueError, match="synthetic teacher kind is strictly prohibited with real patient data"):
        validate_teacher_identity(
            {
                "teacher_kind": "synthetic",
                "inference_state": "student",
                "labeled_ids": list(data.lids),
                "label_budget": len(data.lids),
            },
            data=data,
        )


def test_crop_flip_geometry_and_bad_frame_rejected(tmp_path):
    """Negative test: Bad frame indices rejected, crop and flip geometry synchronized."""
    data_dir = tmp_path / "geom_data"
    manifest_path, manifest_sha = synthetic_compact_fixture(data_dir, n_labeled=1, n_unlabeled=2, size=32)
    data = CompactTrainingData(manifest_path, manifest_sha, synthetic=True)

    teacher_model, _ = build_model({"backbone": "tiny_test"}, synthetic=True)
    teacher_id = {
        "schema": "self_audit.compact_teacher_identity.v1",
        "teacher_kind": "synthetic",
        "inference_state": "student",
        "label_budget": 1,
        "labeled_ids": list(data.lids),
    }

    cache_dir = tmp_path / "geom_cache"
    receipt = build_teacher_cache(teacher_model, data, cache_dir, teacher_id)
    cache = FixedTeacherCache(cache_dir, receipt["cache_manifest_sha256"], data, teacher_id)

    # Bad frame index out of bounds
    bad_meta = {"patient": data.uids[0], "frame": 999, "slice": 0}
    with pytest.raises(ValueError, match="rejected bad frame/slice index"):
        cache.sample_batch(bad_meta)

    # Bad slice index out of bounds
    bad_slice = {"patient": data.uids[0], "frame": 0, "slice": 999}
    with pytest.raises(ValueError, match="rejected bad frame/slice index"):
        cache.sample_batch(bad_slice)

    # Valid metadata with crop and horizontal flip
    valid_meta = {
        "patient": data.uids[0],
        "frame": 0,
        "slice": 0,
        "crop": {
            "pad_lrtb": [2, 2, 2, 2],
            "top": 2,
            "left": 2,
            "size": 16,
        },
        "hflip": True,
    }
    p, v = cache.sample_batch(valid_meta)
    assert p.shape == (1, 4, 16, 16)
    assert v.shape == (1, 16, 16)
    assert v.dtype == torch.bool
    assert torch.allclose(p.sum(dim=1), torch.ones(1, 16, 16), atol=1e-3)

    cache.close()


def test_streaming_memmap_disk_and_telemetry_report(tmp_path):
    """Verify build_teacher_cache streams open_memmap and reports disk bytes, latency, and VRAM."""
    data_dir = tmp_path / "stream_data"
    manifest_path, manifest_sha = synthetic_compact_fixture(data_dir, n_labeled=1, n_unlabeled=2, size=32)
    data = CompactTrainingData(manifest_path, manifest_sha, synthetic=True)

    teacher_model, _ = build_model({"backbone": "tiny_test"}, synthetic=True)
    teacher_id = {
        "schema": "self_audit.compact_teacher_identity.v1",
        "teacher_kind": "synthetic",
        "inference_state": "student",
        "label_budget": 1,
        "labeled_ids": list(data.lids),
    }

    cache_dir = tmp_path / "stream_cache"
    receipt = build_teacher_cache(
        teacher_model,
        data,
        cache_dir,
        teacher_id,
        precision="float32",
    )
    assert receipt["status"] == "PASS"
    assert receipt["total_disk_bytes"] > 0
    assert receipt["total_latency_seconds"] > 0

    # Inspect manifest telemetry
    manifest_data = json.loads(Path(receipt["cache_manifest_path"]).read_text())
    assert "telemetry" in manifest_data
    telemetry = manifest_data["telemetry"]
    assert "total_latency_seconds" in telemetry
    assert "mean_seconds_per_volume" in telemetry
    assert "total_disk_bytes" in telemetry


def test_safe_historical_bridge_importlib_isolated(tmp_path):
    """Verify safe historical bridge verifies commit 0f54455b closure 3c7fa85f and imports cleanly."""
    # Export historical commit 0f54455b into a temporary directory
    hist_dir = tmp_path / "historical_checkout"
    hist_dir.mkdir()

    tar_proc = subprocess.Popen(
        ["git", "archive", HISTORICAL_SOURCE_COMMIT],
        stdout=subprocess.PIPE,
    )
    subprocess.check_call(["tar", "-x", "-C", str(hist_dir)], stdin=tar_proc.stdout)
    tar_proc.wait()

    # 1. verify_historical_source on export
    receipt = verify_historical_source(hist_dir)
    assert receipt["status"] == "PASS"
    assert receipt["source_closure"] == HISTORICAL_SOURCE_CLOSURE

    # 2. Tampered historical source must fail closed
    tampered_file = hist_dir / "self_audit" / "losses.py"
    tampered_file.write_text(tampered_file.read_text() + "\n# tampered")
    with pytest.raises(ValueError, match="historical source closure mismatch"):
        verify_historical_source(hist_dir)

    # 3. Clean checkout import under unique importlib namespace
    clean_dir = tmp_path / "clean_hist"
    clean_dir.mkdir()
    tar_proc2 = subprocess.Popen(
        ["git", "archive", HISTORICAL_SOURCE_COMMIT],
        stdout=subprocess.PIPE,
    )
    subprocess.check_call(["tar", "-x", "-C", str(clean_dir)], stdin=tar_proc2.stdout)
    tar_proc2.wait()

    pkg = import_historical_package(clean_dir, pkg_alias="_test_hist_isolated")
    assert pkg.__name__ == "_test_hist_isolated"
    assert "_test_hist_isolated" in sys.modules
    # Ensure current self_audit was not clobbered
    assert "self_audit" in sys.modules and sys.modules["self_audit"].__name__ == "self_audit"


def test_find_checkpoint_payload_inspects_student_and_weights(tmp_path):
    """find_checkpoint_payload finds student.pt or weights.pt matching SHA; fails closed otherwise."""
    cd = tmp_path / "ckpt"
    cd.mkdir()

    # student.pt present
    student_file = cd / "student.pt"
    student_file.write_bytes(b"student_payload_bytes_123")
    sha = digest(student_file)

    found = find_checkpoint_payload(cd, sha)
    assert found == student_file

    # SHA mismatch fails closed
    with pytest.raises(FileNotFoundError, match="Real missing assets BLOCKED"):
        find_checkpoint_payload(cd, "0" * 64)


def test_load_teacher_for_cache_strict_state_no_fallback(tmp_path):
    """load_teacher_for_cache requires exact state in payload; no fallback allowed."""
    cd = tmp_path / "ckpt_no_student"
    cd.mkdir()

    manifest_file = cd / "CHECKPOINT_MANIFEST.json"
    payload_file = cd / "student.pt"

    # Save payload without 'student' key (only has 'model')
    torch.save({"model": {}}, payload_file)
    p_sha = digest(payload_file)

    m_data = {
        "schema": "self_audit.checkpoint.v2",
        "config": {"method": "ema_full", "model": {"backbone": "tiny_test"}},
        "labeled_ids": ["p001"],
        "label_budget": 1,
        "payload_sha256": p_sha,
        "synthetic": True,
    }
    manifest_file.write_text(json.dumps(m_data))
    m_sha = digest(manifest_file)

    teacher_id = {
        "teacher_kind": "synthetic",
        "inference_state": "student",
        "manifest_sha256": m_sha,
        "training_method": "ema_full",
        "payload_sha256": p_sha,
        "labeled_ids": ["p001"],
        "label_budget": 1,
    }

    # Must raise KeyError on missing 'student' state — NO fallback to payload directly!
    with pytest.raises(KeyError, match="'student' state missing from payload keys"):
        load_teacher_for_cache(
            cd,
            teacher_identity=teacher_id,
            synthetic=True,
        )


def test_cli_preflight_script_with_historical_source(tmp_path):
    """CLI preflight script with student manifest and historical source."""
    manifest_path, _ = synthetic_compact_fixture(tmp_path / "cli_data", n_labeled=1, n_unlabeled=1)

    # Export clean historical checkout
    hist_dir = tmp_path / "hist_cli"
    hist_dir.mkdir()
    tar_proc = subprocess.Popen(["git", "archive", HISTORICAL_SOURCE_COMMIT], stdout=subprocess.PIPE)
    subprocess.check_call(["tar", "-x", "-C", str(hist_dir)], stdin=tar_proc.stdout)
    tar_proc.wait()

    res = subprocess.run(
        [
            sys.executable,
            "scripts/preflight_compact_teacher.py",
            "--teacher-labeled-ids", "synthetic_p000",
            "--student-manifest", str(manifest_path),
            "--student-manifest-sha256", digest(manifest_path),
            "--historical-source", str(hist_dir),
            "--label-budget", "1",
            "--declared-labeled-ids", "synthetic_p000",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = json.loads(res.stdout)
    assert receipt["status"] == "PASS"
    assert "historical_source_bridge" in receipt
    assert receipt["historical_source_bridge"]["verified_clean"] is True


def test_cli_build_cache_script_end_to_end(tmp_path):
    """CLI build cache script end-to-end with open_memmap and telemetry output."""
    manifest_path, manifest_sha = synthetic_compact_fixture(tmp_path / "cache_data", n_labeled=1, n_unlabeled=1)
    cache_dir = tmp_path / "cli_cache"

    res = subprocess.run(
        [
            sys.executable,
            "scripts/build_compact_teacher_cache.py",
            "--data-manifest", str(manifest_path),
            "--data-sha256", manifest_sha,
            "--output-dir", str(cache_dir),
            "--synthetic-teacher",
            "--precision", "float32",
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = json.loads(res.stdout)
    assert receipt["status"] == "PASS"
    assert (cache_dir / "CACHE_MANIFEST.json").is_file()
    assert receipt["total_disk_bytes"] > 0
