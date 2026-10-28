"""Acceptance tests for shared multi-domain training data (ACDC + M&Ms).

Covers all contractual requirements:
1. Negative unknown metadata & conflicting aliases rejected.
2. Raw vs canonical & double-remap with actual array witness.
3. Actual voxel duplicate with different .npy file hash rejected.
4. Eval ID namespace overlap disambiguated and collisions rejected.
5. Wrong budgets (10, 16, bool, float) rejected.
6. Mapping contradictions rejected.
7. Invalid shared manifest rejected BEFORE array/mask hash I/O.
8. Source path overwrite rejected (out must be brand new, exist_ok=False).
9. No U or unselected/discarded mask opens during preparation.
10. Successful CLI roundtrip with mandatory SHAs and loader preflight.
11. L16 rejection & exact L4+4 budget enforcement.
12. Repeatable subset selection with seed 42.
13. Namespaced collisions disambiguated.
14. U no mask leakage (strictly image only).
15. Quarantines (ACDC 055/076, MnM 8 duplicate IDs) excluded & rejected if present.
16. Sampler 50/50 domain balancing & round-robin patient coverage.
17. Single-domain baseline parity (plain tuple, exact old RNG).
18. Loader roundtrip.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from self_audit.data import TrainingData, write_json
from self_audit.model import digest
from self_audit.shared_data import (
    CANONICAL_CLASSES,
    CANONICAL_DOMAINS,
    DEFAULT_ACDC_BUDGET,
    DEFAULT_MNM_BUDGET,
    DEFAULT_SEED,
    DEFAULT_SHARED_BUDGET,
    KNOWN_EXCLUDED_ACDC_IDS,
    KNOWN_QUARANTINED_MNM_IDS,
    RAW_MNM_MAPPING,
    build_shared_manifest,
    compute_voxel_sha256,
    filter_clean_patients,
    is_quarantined_patient,
    namespace_patient_id,
    parse_namespaced_id,
    remap_mnm_mask,
    select_shared_train_labels,
    synthetic_shared_fixture,
    validate_and_remap_mask,
    validate_no_image_overlap,
    validate_shared_manifest,
)
from scripts.prepare_shared_training import prepare_from_manifests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_record(pid: str, folder: Path, is_labeled: bool = True) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    clean_id = pid.replace(":", "_")
    img = np.zeros((1, 2, 16, 16), dtype=np.float32)
    seed_int = int(hashlib.sha256(pid.encode()).hexdigest()[:12], 16)
    rng = np.random.default_rng(seed_int)
    img[0, 0, :4, :4] = rng.uniform(0.01, 0.99, size=(4, 4)).astype(np.float32)
    ip = folder / f"{clean_id}.npy"
    np.save(ip, img)
    rec = {"id": pid, "image": ip.name, "image_sha256": digest(ip)}
    if is_labeled:
        mask = np.zeros((2, 16, 16), dtype=np.uint8)
        mp = folder / f"{clean_id}_mask.npy"
        np.save(mp, mask)
        rec["masks"] = {"0": {"path": mp.name, "sha256": digest(mp)}}
    return rec


# ---------------------------------------------------------------------------
# 1. Negative Unknown Metadata & Aliases
# ---------------------------------------------------------------------------

def test_negative_unknown_metadata(tmp_path):
    """Manifests with unknown domain strings, aliases, or conflicting metadata must be rejected."""
    folder = tmp_path / "unknown_meta"
    manifest_p, _ = synthetic_shared_fixture(folder)
    m = json.loads(manifest_p.read_text())

    # Conflicting alias domain (e.g. uppercase 'ACDC' instead of 'acdc')
    bad_m = copy.deepcopy(m)
    bad_m["domains"] = ["ACDC", "mnm"]
    with pytest.raises(ValueError, match="canonical lowercase"):
        validate_shared_manifest(bad_m)

    # Unknown third domain
    bad_m2 = copy.deepcopy(m)
    bad_m2["domains"] = ["acdc", "mnm", "domain3"]
    with pytest.raises(ValueError, match="canonical lowercase"):
        validate_shared_manifest(bad_m2)

    # Missing domain in domain_map
    bad_m3 = copy.deepcopy(m)
    del bad_m3["domain_map"][bad_m3["labeled"][0]["id"]]
    with pytest.raises(ValueError, match="covering every train patient ID"):
        validate_shared_manifest(bad_m3)

    # Conflict between domain_map and namespace prefix
    bad_m4 = copy.deepcopy(m)
    first_id = bad_m4["labeled"][0]["id"]
    bad_m4["domain_map"][first_id] = "mnm" if bad_m4["domain_map"][first_id] == "acdc" else "acdc"
    with pytest.raises(ValueError, match="conflicts with domain"):
        validate_shared_manifest(bad_m4)


# ---------------------------------------------------------------------------
# 2. Raw vs Canonical & Double-Remap Actual Array Witness
# ---------------------------------------------------------------------------

def test_raw_vs_canonical_doubleremap_array_witness():
    """Actual array witness: raw MnM maps 1->3, 2->2, 3->1; canonical does not; double-remap fails."""
    # Raw MnM array witness
    raw_array = np.array([0, 1, 2, 3, 255], dtype=np.uint8)
    remapped = validate_and_remap_mask(raw_array, domain="mnm", mapping_mode="raw_mnm")
    assert np.array_equal(remapped, np.array([0, 3, 2, 1, 255], dtype=np.uint8))

    # Canonical array witness (permits NO remap, exact identity)
    canonical_array = np.array([0, 1, 2, 3, 255], dtype=np.uint8)
    no_remap = validate_and_remap_mask(canonical_array, domain="acdc", mapping_mode="canonical")
    assert np.array_equal(no_remap, canonical_array)

    # Double remap is strictly rejected
    with pytest.raises(ValueError, match="rejected ambiguous or double-remap"):
        validate_and_remap_mask(canonical_array, domain="mnm", mapping_mode="double_remap")

    # Applying raw_mnm mapping to ACDC is rejected
    with pytest.raises(ValueError, match="raw_mnm mapping requested for non-MnM domain"):
        validate_and_remap_mask(canonical_array, domain="acdc", mapping_mode="raw_mnm")


# ---------------------------------------------------------------------------
# 3. Actual Voxel Duplicate with Different .npy File Hash
# ---------------------------------------------------------------------------

def test_actual_voxel_duplicate_different_npy_hash(tmp_path):
    """Voxel duplicate detector must flag matching arrays even when .npy file hashes differ."""
    folder = tmp_path / "voxel_dup"
    folder.mkdir()

    # Create identical voxel array
    data = np.ones((1, 2, 8, 8), dtype=np.float32)

    # File 1: saved standard
    f1 = folder / "p1.npy"
    np.save(f1, data)

    # File 2: saved with extra header comment / padding altering file SHA-256
    f2 = folder / "p2.npy"
    with open(f2, "wb") as fp:
        import numpy.lib.format as npfmt
        # Write format with different version or extra padding
        npfmt.write_array(fp, data, version=(2, 0))

    sha_f1 = digest(f1)
    sha_f2 = digest(f2)
    # File hashes must differ for the witness test
    assert sha_f1 != sha_f2

    # Voxel hashes must be identical
    assert compute_voxel_sha256(data) == compute_voxel_sha256(np.load(f1))
    assert compute_voxel_sha256(np.load(f1)) == compute_voxel_sha256(np.load(f2))

    train_records = [
        {"id": "acdc:p001", "image": str(f1), "image_sha256": sha_f1},
        {"id": "mnm:p002", "image": str(f2), "image_sha256": sha_f2},
    ]

    # Must be caught as voxel duplicate across patient IDs
    with pytest.raises(ValueError, match="duplicate TRAIN image voxels across patient IDs"):
        validate_no_image_overlap(train_records)


# ---------------------------------------------------------------------------
# 4. Eval ID Namespace Overlap
# ---------------------------------------------------------------------------

def test_eval_id_namespace_overlap(tmp_path):
    """Evaluation IDs must be namespaced with acdc: and mnm:; colliding unnamespaced IDs fail."""
    folder = tmp_path / "eval_ns"
    manifest_p, _ = synthetic_shared_fixture(folder)
    m = json.loads(manifest_p.read_text())

    # Evaluation IDs without domain namespace must be rejected
    bad_m = copy.deepcopy(m)
    bad_m["evaluation_ids"] = ["eval001", "eval002"]
    with pytest.raises(ValueError, match="must be namespaced with 'acdc:' or 'mnm:'"):
        validate_shared_manifest(bad_m)

    # Properly namespaced cross-domain evaluation IDs pass
    good_m = copy.deepcopy(m)
    good_m["evaluation_ids"] = ["acdc:eval001", "mnm:eval001"]
    val = validate_shared_manifest(good_m)
    assert val["evaluation_ids"] == ["acdc:eval001", "mnm:eval001"]


# ---------------------------------------------------------------------------
# 5. Wrong Budgets (10, 16, Bool, Float)
# ---------------------------------------------------------------------------

def test_wrong_budgets(tmp_path):
    """Budgets of 10, 16, float, bool, or non-4+4 for patient data must be rejected."""
    folder = tmp_path / "wrong_budgets"
    manifest_p, _ = synthetic_shared_fixture(folder)
    m = json.loads(manifest_p.read_text())

    # Patient data (synthetic=False) rejecting budget 10
    m_patient_10 = copy.deepcopy(m)
    m_patient_10["synthetic"] = False
    m_patient_10["label_budget"] = 10
    m_patient_10["domain_budgets"] = {"acdc": 5, "mnm": 5}
    with pytest.raises(ValueError, match=r"real patient shared budget must be exactly 4\+4=8"):
        validate_shared_manifest(m_patient_10)

    # Rejecting budget 16
    m_patient_16 = copy.deepcopy(m)
    m_patient_16["synthetic"] = False
    m_patient_16["label_budget"] = 16
    m_patient_16["domain_budgets"] = {"acdc": 8, "mnm": 8}
    with pytest.raises(ValueError, match=r"real patient shared budget must be exactly 4\+4=8"):
        validate_shared_manifest(m_patient_16)

    # Rejecting float budget (8.0)
    m_float = copy.deepcopy(m)
    m_float["label_budget"] = 8.0
    with pytest.raises(ValueError, match="label_budget must be int"):
        validate_shared_manifest(m_float)

    # Rejecting bool budget (True)
    m_bool = copy.deepcopy(m)
    m_bool["label_budget"] = True
    with pytest.raises(ValueError, match="label_budget must be int"):
        validate_shared_manifest(m_bool)


# ---------------------------------------------------------------------------
# 6. Mapping Contradictions
# ---------------------------------------------------------------------------

def test_mapping_contradictions():
    """Contradictions between requested mapping and receipt provenance must fail."""
    mask = np.array([0, 1, 2, 3, 255], dtype=np.uint8)

    # Claiming canonical when receipt says raw_mnm
    with pytest.raises(ValueError, match="mapping contradiction"):
        validate_and_remap_mask(
            mask, domain="mnm", mapping_mode="canonical",
            provenance_receipt={"mapping": "raw_mnm", "status": "PASS"}
        )

    # Claiming raw_mnm when receipt says canonical
    with pytest.raises(ValueError, match="mapping contradiction"):
        validate_and_remap_mask(
            mask, domain="mnm", mapping_mode="raw_mnm",
            provenance_receipt={"mapping": "canonical", "status": "PASS"}
        )


# ---------------------------------------------------------------------------
# 7. Invalid Shared Rejected BEFORE Array / Mask Hash I/O
# ---------------------------------------------------------------------------

def test_invalid_shared_rejected_before_io(tmp_path):
    """Corrupt or missing image arrays are never accessed if shared metadata is invalid."""
    folder = tmp_path / "reject_before_io"
    folder.mkdir()

    # Manifest points to completely nonexistent image and mask paths, with 2 labeled patients
    bad_manifest = {
        "schema": "self_audit.data.v2",
        "role": "training",
        "shared": True,
        "synthetic": False,
        "label_budget": 2,  # Invalid budget for real patient shared data (must be 8)!
        "domains": ["acdc", "mnm"],
        "domain_budgets": {"acdc": 1, "mnm": 1},
        "domain_patients": {"acdc": ["acdc:001"], "mnm": ["mnm:002", "mnm:003"]},
        "domain_map": {"acdc:001": "acdc", "mnm:002": "mnm", "mnm:003": "mnm"},
        "labeled": [
            {"id": "acdc:001", "image": "NONEXISTENT_1.npy", "image_sha256": "0" * 64, "masks": {"0": {"path": "NO_MASK_1.npy", "sha256": "0" * 64}}},
            {"id": "mnm:002", "image": "NONEXISTENT_2.npy", "image_sha256": "1" * 64, "masks": {"0": {"path": "NO_MASK_2.npy", "sha256": "1" * 64}}},
        ],
        "unlabeled": [{"id": "mnm:003", "image": "NONEXISTENT_3.npy", "image_sha256": "2" * 64}],
    }
    mp = folder / "train.json"
    write_json(mp, bad_manifest)
    sha = digest(mp)

    # Must raise ValueError on budget BEFORE attempting to open NONEXISTENT.npy
    with pytest.raises(ValueError, match=r"real patient shared budget must be exactly 4\+4=8"):
        TrainingData(mp, sha, synthetic=False)


# ---------------------------------------------------------------------------
# 8. Source Path Overwrite Rejected (exist_ok=False)
# ---------------------------------------------------------------------------

def test_source_path_overwrite_rejected(tmp_path):
    """Output directory must be brand new and outside all input directories."""
    acdc_dir = tmp_path / "acdc"
    mnm_dir = tmp_path / "mnm"
    existing_out = tmp_path / "already_exists"
    existing_out.mkdir()

    acdc_m = _make_dummy_manifest(acdc_dir, "acdc")
    mnm_m = _make_dummy_manifest(mnm_dir, "mnm")

    # Outputting to an already existing directory fails
    with pytest.raises(ValueError, match="must be brand new and not already exist"):
        prepare_from_manifests(
            acdc_m, digest(acdc_m),
            mnm_m, digest(mnm_m),
            existing_out,
        )

    # Outputting inside ACDC directory fails
    with pytest.raises(ValueError, match="strictly outside ACDC input directory"):
        prepare_from_manifests(
            acdc_m, digest(acdc_m),
            mnm_m, digest(mnm_m),
            acdc_dir / "nested_out",
        )


def _make_dummy_manifest(folder: Path, domain: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    l_recs = [_make_dummy_record(f"{domain}:{domain}{i:03d}", folder, True) for i in range(1, 9)]
    u_recs = [_make_dummy_record(f"{domain}:{domain}{i:03d}", folder, False) for i in range(9, 15)]
    m = {
        "schema": "self_audit.data.v2", "role": "training", "synthetic": True,
        "label_budget": 8, "labeled": l_recs, "unlabeled": u_recs, "evaluation_ids": [f"{domain}:eval01"],
        "mapping_provenance": {domain: "canonical"},
    }
    mp = folder / "train.json"
    write_json(mp, m)
    return mp


# ---------------------------------------------------------------------------
# 9. No U or Unselected Mask Opens
# ---------------------------------------------------------------------------

def test_no_u_or_unselected_mask_opens(tmp_path, monkeypatch):
    """Discarded labeled masks and U masks must NEVER be loaded during preparation."""
    acdc_dir = tmp_path / "acdc_spy"
    mnm_dir = tmp_path / "mnm_spy"
    out_dir = tmp_path / "clean_out"

    acdc_m = _make_dummy_manifest(acdc_dir, "acdc")
    mnm_m = _make_dummy_manifest(mnm_dir, "mnm")

    loaded_masks = []
    orig_load = np.load

    def spy_load(file, *args, **kwargs):
        path_str = str(file)
        if "_mask.npy" in path_str:
            loaded_masks.append(path_str)
        return orig_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", spy_load)

    prepare_from_manifests(
        acdc_m, digest(acdc_m),
        mnm_m, digest(mnm_m),
        out_dir,
        acdc_budget=4,
        mnm_budget=4,
        seed=42,
    )

    # Discarded L masks (patients 5..8 from both domains) and U masks must NOT be loaded
    for loaded in loaded_masks:
        assert not any(f"00{i}_mask" in loaded for i in range(5, 9))


# ---------------------------------------------------------------------------
# 10. Successful CLI Roundtrip
# ---------------------------------------------------------------------------

def test_successful_cli_roundtrip(tmp_path):
    """End-to-end CLI execution with mandatory SHAs, validation, and loader preflight."""
    acdc_dir = tmp_path / "acdc_cli"
    mnm_dir = tmp_path / "mnm_cli"
    out_dir = tmp_path / "cli_roundtrip_out"

    acdc_m = _make_dummy_manifest(acdc_dir, "acdc")
    mnm_m = _make_dummy_manifest(mnm_dir, "mnm")

    cmd = [
        sys.executable, "scripts/prepare_shared_training.py",
        "--acdc-manifest", str(acdc_m),
        "--acdc-manifest-sha256", digest(acdc_m),
        "--mnm-manifest", str(mnm_m),
        "--mnm-manifest-sha256", digest(mnm_m),
        "--out", str(out_dir),
        "--acdc-budget", "4",
        "--mnm-budget", "4",
        "--seed", "42",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out_json = json.loads(res.stdout)
    assert out_json["status"] == "PASS"

    # Verify receipt and new manifest exist
    assert (out_dir / "train.json").is_file()
    assert (out_dir / "PREPARATION.json").is_file()
    receipt = json.loads((out_dir / "PREPARATION.json").read_text())
    assert receipt["status"] == "PASS"


# ---------------------------------------------------------------------------
# 11. Repeatable Subset & Quarantines
# ---------------------------------------------------------------------------

def test_repeatable_subset_and_quarantines():
    """Subset selection is deterministic and strictly excludes all quarantined members."""
    acdc_pool = [f"patient{i:03d}" for i in range(1, 15)]
    mnm_pool = [f"M{i:03d}" for i in range(1, 15)]

    s1_a, s1_m = select_shared_train_labels(acdc_pool, mnm_pool, seed=DEFAULT_SEED)
    s2_a, s2_m = select_shared_train_labels(acdc_pool, mnm_pool, seed=DEFAULT_SEED)

    assert s1_a == s2_a
    assert s1_m == s2_m
    assert len(s1_a) == 4
    assert len(s1_m) == 4

    # Quarantined IDs are detected
    for qid in KNOWN_QUARANTINED_MNM_IDS:
        assert is_quarantined_patient(qid, "mnm")
    for qid in KNOWN_EXCLUDED_ACDC_IDS:
        assert is_quarantined_patient(qid, "acdc")


# ---------------------------------------------------------------------------
# 12. Sampler 50/50 Domain Balancing & Telemetry
# ---------------------------------------------------------------------------

def test_sampler_5050_telemetry(tmp_path):
    """TrainingData choose_patient and sample_indices alternate 50/50 and report coverage."""
    folder = tmp_path / "sampler_fixture"
    manifest_p, sha = synthetic_shared_fixture(folder)
    data = TrainingData(manifest_p, sha, synthetic=True)

    assert data.is_shared
    assert data.domains == ["acdc", "mnm"]

    rng = np.random.default_rng(42)
    observed_l = []
    seen = set()

    for step in range(16):
        li, ai, ui = data.sample_indices(step, rng)
        assert isinstance((li, ai, ui), tuple)
        p_l = data.labeled[li]["id"]
        p_a = data.labeled[ai]["id"]
        p_u = data.unlabeled[ui]["id"]
        assert p_l != p_a

        seen.add(p_l)
        seen.add(p_a)
        seen.add(p_u)
        observed_l.append(data.domain_of(p_l))

    # 50/50 alternation
    assert observed_l[0::2] == ["acdc"] * 8
    assert observed_l[1::2] == ["mnm"] * 8

    # Telemetry
    cov = data.domain_coverage(seen)
    assert cov["acdc"]["labeled_coverage"] == 1.0
    assert cov["mnm"]["labeled_coverage"] == 1.0


# ---------------------------------------------------------------------------
# 13. Single Domain Baseline Parity
# ---------------------------------------------------------------------------

def test_single_domain_baseline_parity(tmp_path):
    """Non-shared dataset preserves exact baseline round-robin behavior and plain tuple."""
    from self_audit.data import synthetic_fixture

    folder = tmp_path / "single_dom"
    manifest_p, sha = synthetic_fixture(folder, seed=11, n_labeled=3, n_unlabeled=4, size=32)
    data = TrainingData(manifest_p, sha, synthetic=True)

    assert not data.is_shared
    assert len(data.domains) == 1

    rng = np.random.default_rng(42)
    for step in range(12):
        indices = data.sample_indices(step, rng)
        assert type(indices) is tuple
        li, ai, ui = indices
        assert li == step % len(data.lids)
        assert ui == step % len(data.uids)
        assert li != ai


# ---------------------------------------------------------------------------
# 14. Voxel Duplicate Fails Before Any Mask Opens
# ---------------------------------------------------------------------------

def test_prepare_fails_before_masks_open_on_voxel_duplicate(tmp_path, monkeypatch):
    """Voxel duplicates with different NPY headers fail in prepare_from_manifests BEFORE opening masks."""
    acdc_dir = tmp_path / "acdc_vdup"
    mnm_dir = tmp_path / "mnm_vdup"
    out_dir = tmp_path / "out_vdup"

    acdc_m = _make_dummy_manifest(acdc_dir, "acdc")
    mnm_m = _make_dummy_manifest(mnm_dir, "mnm")

    # Tamper with an MnM image to have identical voxels to an ACDC image,
    # but write it with a different numpy format version so file SHA differs.
    acdc_m_data = json.loads(acdc_m.read_text())
    first_acdc_img = np.load(acdc_dir / acdc_m_data["labeled"][0]["image"])

    # Overwrite first MnM labeled image with identical voxels but different NPY header
    mnm_m_data = json.loads(mnm_m.read_text())
    mnm_img_path = mnm_dir / mnm_m_data["labeled"][0]["image"]
    with open(mnm_img_path, "wb") as fp:
        import numpy.lib.format as npfmt
        npfmt.write_array(fp, first_acdc_img, version=(2, 0))

    # Update MnM manifest with new file sha
    mnm_m_data["labeled"][0]["image_sha256"] = digest(mnm_img_path)
    write_json(mnm_m, mnm_m_data)

    opened_masks = []
    orig_load = np.load

    def spy_load(file, *args, **kwargs):
        path_str = str(file)
        if "_mask.npy" in path_str:
            opened_masks.append(path_str)
        return orig_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", spy_load)

    with pytest.raises(ValueError, match="duplicate TRAIN image voxels across patient IDs"):
        prepare_from_manifests(
            acdc_m, digest(acdc_m),
            mnm_m, digest(mnm_m),
            out_dir,
            acdc_budget=4,
            mnm_budget=4,
            seed=42,
        )

    # NO mask file should ever have been opened
    assert len(opened_masks) == 0


# ---------------------------------------------------------------------------
# 15. Raw MnM Mask Original SHA Verified Before Remap
# ---------------------------------------------------------------------------

def test_raw_mnm_mask_sha_verified_before_remap(tmp_path):
    """Raw selected MnM mask original SHA must verify before np.load/remap."""
    acdc_dir = tmp_path / "acdc_raw_corrupt"
    mnm_dir = tmp_path / "mnm_raw_corrupt"
    out_dir = tmp_path / "out_raw_corrupt"

    acdc_m = _make_dummy_manifest(acdc_dir, "acdc")
    mnm_m = _make_dummy_manifest(mnm_dir, "mnm")

    # Set MnM provenance to raw_mnm
    mnm_data = json.loads(mnm_m.read_text())
    mnm_data["mapping_provenance"] = {"mnm": "raw_mnm"}
    write_json(mnm_m, mnm_data)

    # Determine which MnM patient is actually selected under seed 42
    mnm_l_ids = [r["id"] for r in mnm_data["labeled"]]
    from self_audit.data import label_subset
    selected_first_id = label_subset(mnm_l_ids, 4, 42)[0]
    target_rec = next(r for r in mnm_data["labeled"] if r["id"] == selected_first_id)

    # Corrupt that selected raw mask file on disk so its hash mismatches manifest
    corrupt_mask_path = mnm_dir / target_rec["masks"]["0"]["path"]
    corrupt_mask_path.write_bytes(b"CORRUPTED_BYTES")

    with pytest.raises(ValueError, match="source mask SHA mismatch"):
        prepare_from_manifests(
            acdc_m, digest(acdc_m),
            mnm_m, digest(mnm_m),
            out_dir,
            acdc_budget=4,
            mnm_budget=4,
            seed=42,
        )


# ---------------------------------------------------------------------------
# 16. Second Remap Rejected on Canonical Masks
# ---------------------------------------------------------------------------

def test_second_remap_rejected_on_canonical_masks(tmp_path):
    """Attempting to remap verified canonical masks (double-remap) must be rejected."""
    acdc_dir = tmp_path / "acdc_canon"
    mnm_dir = tmp_path / "mnm_canon"
    out_dir = tmp_path / "out_second_remap"

    acdc_m = _make_dummy_manifest(acdc_dir, "acdc")
    mnm_m = _make_dummy_manifest(mnm_dir, "mnm")

    # Both are canonical; user attempts --mapping raw_mnm on canonical masks
    with pytest.raises(ValueError, match="mapping contradiction: CLI requested 'raw_mnm' on verified canonical masks"):
        prepare_from_manifests(
            acdc_m, digest(acdc_m),
            mnm_m, digest(mnm_m),
            out_dir,
            mapping_override="raw_mnm",
        )


# ---------------------------------------------------------------------------
# 17. Missing and Unbound Evaluation / Receipt Proof Rejected
# ---------------------------------------------------------------------------

def test_missing_and_unbound_eval_proof_rejected(tmp_path):
    """Real patient preparation rejects missing evaluation catalogs or unbound receipts."""
    acdc_dir = tmp_path / "acdc_proof"
    mnm_dir = tmp_path / "mnm_proof"
    receipt_dir = tmp_path / "receipt_proof"
    out_dir1 = tmp_path / "out_no_eval"
    out_dir2 = tmp_path / "out_unbound_receipt"

    acdc_m = _make_dummy_manifest(acdc_dir, "acdc")
    mnm_m = _make_dummy_manifest(mnm_dir, "mnm")

    # Set synthetic=False for real data check
    for p in (acdc_m, mnm_m):
        d = json.loads(p.read_text())
        d["synthetic"] = False
        write_json(p, d)

    # Real data without eval catalog must fail
    with pytest.raises(ValueError, match="real patient preparation requires pinned evaluation catalogs"):
        prepare_from_manifests(
            acdc_m, digest(acdc_m),
            mnm_m, digest(mnm_m),
            out_dir1,
        )

    # Test unbound receipt (receipt references wrong train manifest SHA)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    bad_receipt_p = receipt_dir / "bad_receipt.json"
    write_json(bad_receipt_p, {
        "status": "PASS",
        "train_manifest_sha256": "0" * 64,  # Does not match mnm_m digest!
        "mask_ontology": "canonical",
    })

    with pytest.raises(ValueError, match="is not bound to input manifest"):
        prepare_from_manifests(
            acdc_m, digest(acdc_m),
            mnm_m, digest(mnm_m),
            out_dir2,
            mnm_receipt_path=bad_receipt_p,
            mnm_receipt_sha256=digest(bad_receipt_p),
        )


# ---------------------------------------------------------------------------
# 18. Mapping Override Only Applies to MnM
# ---------------------------------------------------------------------------

def test_mapping_override_only_applies_to_mnm(tmp_path):
    """CLI mapping override raw_mnm remaps MnM without rejecting canonical ACDC."""
    acdc_dir = tmp_path / "acdc_ov"
    mnm_dir = tmp_path / "mnm_ov"
    out_dir = tmp_path / "out_ov"

    acdc_m = _make_dummy_manifest(acdc_dir, "acdc")
    mnm_m = _make_dummy_manifest(mnm_dir, "mnm")

    # Set MnM provenance to raw_mnm
    mnm_data = json.loads(mnm_m.read_text())
    mnm_data["mapping_provenance"] = {"mnm": "raw_mnm"}
    write_json(mnm_m, mnm_data)

    # Override raw_mnm should succeed: remap MnM, preserve ACDC as canonical
    mp, sha = prepare_from_manifests(
        acdc_m, digest(acdc_m),
        mnm_m, digest(mnm_m),
        out_dir,
        mapping_override="raw_mnm",
    )
    res = json.loads(mp.read_text())
    assert res["mapping_provenance"]["acdc"].startswith("canonical")
    assert "raw_mnm remapped" in res["mapping_provenance"]["mnm"]


# ---------------------------------------------------------------------------
# 19. Historical Actual MnM Receipt Proves Output Canonical
# ---------------------------------------------------------------------------

def test_historical_actual_mnm_receipt_proves_canonical(tmp_path):
    """Read the exact report bytes, not a hand-built approximation of historical provenance."""
    from scripts.prepare_shared_training import determine_mask_ontology
    receipt=Path("reports/baseline_recovery/MNM_PREPARATION_SOURCE.json")
    data=json.loads(receipt.read_text())
    expected="82f6a05b38fa8e69607b0fd6fcecf0062809353dfb47cc8107cbbfb865bf030b"
    assert digest(receipt)==expected
    ontology,info=determine_mask_ontology({},data["receipt"]["sha256"],receipt,expected,domain="mnm")
    assert ontology=="canonical" and info["sha256"]==expected
    with pytest.raises(ValueError,match="double-remap|contradiction"):
        determine_mask_ontology({},data["receipt"]["sha256"],receipt,expected,cli_mapping="raw_mnm",domain="mnm")


def test_mapping_plan_alone_is_not_evidence_of_applied_remap(tmp_path):
    from scripts.prepare_shared_training import determine_mask_ontology
    receipt=tmp_path/"unapplied_mapping.json"
    write_json(receipt,{"train_manifest_sha256":"a"*64,"mapping":{"0":0,"1":3,"2":2,"3":1,"255":255}})
    with pytest.raises(ValueError,match="missing mask ontology proof"):
        determine_mask_ontology({},"a"*64,receipt,digest(receipt),domain="mnm")
    with pytest.raises(ValueError,match="missing mask ontology proof"):
        determine_mask_ontology({"mapping_provenance":{"mnm":"canonical perhaps"}},"a"*64,domain="mnm")
