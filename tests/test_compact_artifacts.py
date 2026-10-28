"""Targeted unit tests for CompactStudent artifact persistence, export, CLI, and standalone execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import pytest
import torch

from self_audit.compact_model import CompactStudent, build_compact_model
from self_audit.compact_artifacts import (
    compact_training_source_sha256,
    digest,
    export_compact_student,
    load_compact_checkpoint,
    load_compact_export,
    runtime_model_file_sha256,
    save_compact_checkpoint,
)


class DummyValidData:
    def __init__(self, tmp_path, synthetic=True, duplicate_lids=False, overlap=False):
        Path(tmp_path).mkdir(parents=True, exist_ok=True)
        if duplicate_lids:
            self.lids = ["patient001", "patient001"]
            self.uids = ["patient003", "patient004"]
        elif overlap:
            self.lids = ["patient001", "patient002"]
            self.uids = ["patient002", "patient003"]
        else:
            self.lids = ["patient001", "patient002"]
            self.uids = ["patient003", "patient004", "patient005"]
        self.manifest = {"dataset": "test_acdc"}
        if synthetic is not None:
            self.manifest["synthetic"] = synthetic
        data_file = tmp_path / "data_manifest.json"
        data_file.write_text(json.dumps(self.manifest))
        self.path = str(data_file)
        self.data_sha256 = digest(data_file)


def test_checkpoint_roundtrip_with_mandatory_sha(tmp_path):
    model = build_compact_model({"widths": [16, 32, 64, 128]})
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = DummyValidData(tmp_path, synthetic=True)
    config = {
        "schema": "self_audit.compact.v1",
        "method": "joint_guard",
        "training": {"steps": 10},
    }
    cp_dir = tmp_path / "checkpoint_001"

    manifest_sha = save_compact_checkpoint(
        model=model,
        optimizer=opt,
        config=config,
        data=data,
        out=cp_dir,
        step=5,
        teacher_identity={"teacher_sha256": "a" * 64, "lineage": "frozen_full_ema"},
        extra={"supervised_loss": 0.42},
    )

    assert isinstance(manifest_sha, str)
    assert len(manifest_sha) == 64

    # Load with valid expected SHA
    loaded_model, meta = load_compact_checkpoint(cp_dir, expected_sha=manifest_sha)
    assert meta["step"] == 5
    assert meta["data_provenance"]["lids"] == ["patient001", "patient002"]
    assert meta["data_provenance"]["label_budget"] == 2
    assert meta["data_provenance"]["synthetic"] is True
    assert meta["weights_only_bytes"] > 0
    assert meta["serialization_bytes"] > meta["weights_only_bytes"]
    assert meta["parameters_total"] == sum(p.numel() for p in model.parameters())

    # Bitwise logit equivalence
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        out_orig = model(x)
        out_loaded = loaded_model(x)
    torch.testing.assert_close(out_orig, out_loaded, rtol=0, atol=0)


def test_checkpoint_verifies_model_file_sha_before_reconstruct(tmp_path):
    """Verify that load_compact_checkpoint verifies model file SHA before instantiating model."""
    model = build_compact_model({"widths": [16, 32, 64, 128]})
    data = DummyValidData(tmp_path)
    cp_dir = tmp_path / "cp_before_recon"
    save_compact_checkpoint(model, None, {}, data, cp_dir, step=0)

    # Tamper with model_file_sha256 in checkpoint MANIFEST
    manifest_path = cp_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_file_sha256"] = "e" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2))
    tampered_manifest_sha = digest(manifest_path)

    with pytest.raises(ValueError, match="checkpoint model file SHA mismatch"):
        load_compact_checkpoint(cp_dir, expected_sha=tampered_manifest_sha)


def test_checkpoint_mandatory_sha_validation(tmp_path):
    model = build_compact_model({"widths": [16, 32, 64, 128]})
    data = DummyValidData(tmp_path)
    cp_dir = tmp_path / "cp_sha_val"
    save_compact_checkpoint(model, None, {}, data, cp_dir, step=0)

    # Missing / None expected_sha
    with pytest.raises(TypeError):
        load_compact_checkpoint(cp_dir)  # argument missing

    with pytest.raises(ValueError, match="mandatory 64-hex SHA-256 pin"):
        load_compact_checkpoint(cp_dir, expected_sha=None)

    # Short / non-hex string
    with pytest.raises(ValueError, match="mandatory 64-hex SHA-256 pin"):
        load_compact_checkpoint(cp_dir, expected_sha="not_a_sha")

    # Mismatched SHA
    with pytest.raises(ValueError, match="checkpoint manifest hash mismatch"):
        load_compact_checkpoint(cp_dir, expected_sha="0" * 64)


def test_strict_id_uniqueness_and_disjointness(tmp_path):
    model = build_compact_model({"widths": [16, 32, 64, 128]})

    # Duplicate lids
    bad_data1 = DummyValidData(tmp_path / "d1", duplicate_lids=True)
    with pytest.raises(ValueError, match="duplicate labeled IDs"):
        save_compact_checkpoint(model, None, {}, bad_data1, tmp_path / "cp_dup_lids", step=0)

    # Duplicate uids
    bad_data2 = {
        "lids": ["p1"],
        "uids": ["p2", "p2"],
        "manifest": {"synthetic": True},
        "data_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="duplicate unlabeled IDs"):
        save_compact_checkpoint(model, None, {}, bad_data2, tmp_path / "cp_dup_uids", step=0)

    # Overlap between lids and uids
    bad_data3 = DummyValidData(tmp_path / "d3", overlap=True)
    with pytest.raises(ValueError, match="labeled and unlabeled IDs must be strictly disjoint"):
        save_compact_checkpoint(model, None, {}, bad_data3, tmp_path / "cp_overlap", step=0)


def test_legacy_train_manifest_synthetic_default_false(tmp_path):
    """Verify real legacy TRAIN manifest lacking 'synthetic' cleanly resolves to False."""
    model = build_compact_model({"widths": [16, 32, 64, 128]})
    # Data without 'synthetic' in manifest
    legacy_data = DummyValidData(tmp_path / "legacy", synthetic=None)
    cp_dir = tmp_path / "cp_legacy"
    cp_sha = save_compact_checkpoint(model, None, {}, legacy_data, cp_dir, step=0)

    _, meta = load_compact_checkpoint(cp_dir, expected_sha=cp_sha)
    assert meta["data_provenance"]["synthetic"] is False


def test_checkpoint_rejects_nonfinite_weights(tmp_path):
    model = build_compact_model({"widths": [16, 32, 64, 128]})
    with torch.no_grad():
        for p in model.parameters():
            p[0] = float("nan")
            break

    data = DummyValidData(tmp_path)
    cp_dir = tmp_path / "cp_nan_weights"
    with pytest.raises(ValueError, match="non-finite values"):
        save_compact_checkpoint(model, None, {}, data, cp_dir, step=0)


def test_export_strict_provenance_and_clean_config(tmp_path):
    model = build_compact_model({"widths": [16, 32, 64, 128]})
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = DummyValidData(tmp_path)
    config = {
        "schema": "self_audit.compact.v1",
        "cache_path": "/fake/training/cache.pt",
        "teacher_path": "/fake/training/teacher.pt",
        "weights": "/fake/pretrained/dino.pth",
        "teacher_memory": {"arm": "A5"},
    }
    cp_dir = tmp_path / "checkpoint_source"
    cp_sha = save_compact_checkpoint(model, opt, config, data, cp_dir, step=12)

    export_dir = tmp_path / "student_export"
    export_sha = export_compact_student(cp_dir, expected_sha=cp_sha, out=export_dir)

    assert isinstance(export_sha, str)
    assert len(export_sha) == 64

    # Load export strictly
    exported_model, export_meta = load_compact_export(export_dir, expected_sha=export_sha)
    assert export_meta["schema"] == "self_audit.compact.export.v1"
    assert export_meta["teacher_included"] is False
    assert export_meta["external_teacher_required"] is False
    assert export_meta["teacher_memory_included"] is False
    assert export_meta["dino_required"] is False
    assert export_meta["pretrained_weights_required"] is False
    assert export_meta["synthetic"] is True

    # Check exact deploy metadata
    prep = export_meta["preprocessing"]
    assert prep["input_channels"] == 3
    assert prep["context_slices"] == [-1, 0, 1]
    assert "normalization_recipes" in prep
    assert prep["normalization_recipes"]["prepared"]["value_range"] == [0.0, 1.0]
    assert prep["normalization_recipes"]["raw"]["percentiles"] == [0.5, 99.5]

    cmap = export_meta["class_map"]
    assert cmap["0"] == "background"
    assert cmap["1"] == "right_ventricle"
    assert cmap["2"] == "myocardium"
    assert cmap["3"] == "left_ventricle"

    lineage = export_meta["lineage"]
    assert lineage["lids"] == ["patient001", "patient002"]
    assert lineage["label_budget"] == 2
    assert lineage["data_sha256"] == data.data_sha256

    # Cleaned config (paths nullified)
    assert export_meta["config"]["cache_path"] is None
    assert export_meta["config"]["teacher_path"] is None
    assert export_meta["config"]["weights"] is None
    assert export_meta["config"]["teacher_memory"] is None

    # Strict payload check
    payload = torch.load(export_dir / "weights.pt", map_location="cpu", weights_only=True)
    assert set(payload.keys()) == {"student"}

    # Bitwise match
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(x), exported_model(x), rtol=0, atol=0)


def test_export_mandatory_sha_validation(tmp_path):
    model = build_compact_model({"widths": [16, 32, 64, 128]})
    data = DummyValidData(tmp_path)
    cp_dir = tmp_path / "cp_for_exp_sha"
    cp_sha = save_compact_checkpoint(model, None, {}, data, cp_dir, step=0)

    # Missing / None expected_sha in export
    with pytest.raises(TypeError):
        export_compact_student(cp_dir, out=tmp_path / "exp1")

    with pytest.raises(ValueError, match="mandatory 64-hex SHA-256 pin"):
        export_compact_student(cp_dir, expected_sha="invalid", out=tmp_path / "exp2")

    # Mismatched expected_sha
    with pytest.raises(ValueError, match="checkpoint manifest hash mismatch"):
        export_compact_student(cp_dir, expected_sha="1" * 64, out=tmp_path / "exp3")


def test_load_compact_export_mandatory_model_file_sha_and_tamper(tmp_path):
    model = build_compact_model({"widths": [16, 32, 64, 128]})
    data = DummyValidData(tmp_path)
    cp_dir = tmp_path / "cp_for_tamper"
    cp_sha = save_compact_checkpoint(model, None, {}, data, cp_dir, step=0)

    exp_dir = tmp_path / "export_tamper"
    exp_sha = export_compact_student(cp_dir, expected_sha=cp_sha, out=exp_dir)

    # Missing model_file_sha256
    manifest_path = exp_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["model_file_sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    sha_no_model = digest(manifest_path)
    with pytest.raises(ValueError, match="mandatory 64-hex SHA-256 pin"):
        load_compact_export(exp_dir, expected_sha=sha_no_model)

    # Tampered model_file_sha256 in manifest
    manifest["model_file_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2))
    tampered_sha = digest(manifest_path)
    with pytest.raises(ValueError, match="runtime model file SHA mismatch"):
        load_compact_export(exp_dir, expected_sha=tampered_sha)


def test_load_compact_export_validates_metadata_consistency(tmp_path):
    model = build_compact_model({"widths": [16, 32, 64, 128]})
    data = DummyValidData(tmp_path)
    cp_dir = tmp_path / "cp_meta_cons"
    cp_sha = save_compact_checkpoint(model, None, {}, data, cp_dir, step=0)

    exp_dir = tmp_path / "export_meta_cons"
    export_compact_student(cp_dir, expected_sha=cp_sha, out=exp_dir)

    manifest_path = exp_dir / "MANIFEST.json"
    base_manifest = json.loads(manifest_path.read_text())

    # Corrupt preprocessing input_channels
    bad_prep = copy_manifest = json.loads(json.dumps(base_manifest))
    bad_prep["preprocessing"]["input_channels"] = 1
    manifest_path.write_text(json.dumps(bad_prep, indent=2))
    with pytest.raises(ValueError, match="input_channels must be 3"):
        load_compact_export(exp_dir, expected_sha=digest(manifest_path))

    # Corrupt class_map
    bad_cmap = json.loads(json.dumps(base_manifest))
    bad_cmap["class_map"]["1"] = "wrong_organ"
    manifest_path.write_text(json.dumps(bad_cmap, indent=2))
    with pytest.raises(ValueError, match="export class_map mismatch"):
        load_compact_export(exp_dir, expected_sha=digest(manifest_path))

    # Corrupt lineage (overlapping IDs)
    bad_lineage = json.loads(json.dumps(base_manifest))
    bad_lineage["lineage"]["uids"] = [bad_lineage["lineage"]["lids"][0]]
    manifest_path.write_text(json.dumps(bad_lineage, indent=2))
    with pytest.raises(ValueError, match="lineage lids and uids must be strictly disjoint"):
        load_compact_export(exp_dir, expected_sha=digest(manifest_path))


def test_export_cli_negative_and_positive(tmp_path):
    cli_path = Path(__file__).resolve().parents[1] / "scripts" / "export_compact_student.py"
    model = build_compact_model({"widths": [16, 32, 64, 128]})
    data = DummyValidData(tmp_path)
    cp_dir = tmp_path / "cp_cli"
    cp_sha = save_compact_checkpoint(model, None, {}, data, cp_dir, step=2)

    # Missing mandatory --checkpoint-sha256 flag
    res_missing = subprocess.run([sys.executable, str(cli_path), "--checkpoint", str(cp_dir), "--out", str(tmp_path / "out1")], capture_output=True, text=True)
    assert res_missing.returncode != 0

    # Wrong expected SHA
    res_wrong = subprocess.run([sys.executable, str(cli_path), "--checkpoint", str(cp_dir), "--checkpoint-sha256", "0" * 64, "--out", str(tmp_path / "out2")], capture_output=True, text=True)
    assert res_wrong.returncode != 0

    # Valid invocation
    out_valid = tmp_path / "out_valid"
    res_valid = subprocess.run([sys.executable, str(cli_path), "--checkpoint", str(cp_dir), "--checkpoint-sha256", cp_sha, "--out", str(out_valid)], capture_output=True, text=True, check=True)
    receipt = json.loads(res_valid.stdout)
    assert receipt["status"] == "EXPORTED"
    assert receipt["teacher_included"] is False
    assert len(receipt["manifest_sha256"]) == 64

    # Load from the CLI-exported artifact
    loaded_model, _ = load_compact_export(out_valid, expected_sha=receipt["manifest_sha256"])
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(x), loaded_model(x), rtol=0, atol=0)


def test_standalone_isolated_subprocess_after_deleting_originals(tmp_path):
    """Verify export runs in isolated sandbox after deleting original checkpoint & training source."""
    repo_root = Path(__file__).resolve().parents[1]
    model = build_compact_model({"widths": [16, 32, 64, 128]})
    data = DummyValidData(tmp_path)
    cp_dir = tmp_path / "original_cp"
    cp_sha = save_compact_checkpoint(model, None, {}, data, cp_dir, step=1)

    export_dir = tmp_path / "exported_student"
    export_sha = export_compact_student(cp_dir, expected_sha=cp_sha, out=export_dir)

    # Create an isolated deployment sandbox
    sandbox = tmp_path / "isolated_deployment_sandbox"
    sandbox.mkdir()

    # Copy package modules (self_audit/compact_model.py and self_audit/compact_artifacts.py only)
    target_pkg = sandbox / "self_audit"
    target_pkg.mkdir()
    (target_pkg / "__init__.py").write_text('"""Standalone self_audit package."""\n')
    shutil.copy(repo_root / "self_audit" / "compact_model.py", target_pkg / "compact_model.py")
    shutil.copy(repo_root / "self_audit" / "compact_artifacts.py", target_pkg / "compact_artifacts.py")

    # Move export directory to sandbox
    sandbox_export = sandbox / "export"
    shutil.copytree(export_dir, sandbox_export)

    # DELETE the original checkpoint, data, and source export directory
    shutil.rmtree(cp_dir)
    shutil.rmtree(export_dir)
    Path(data.path).unlink()

    # Run isolated inference script in sandbox with PYTHONPATH pointing only to sandbox
    test_script = (
        "import sys, torch, json\n"
        "from pathlib import Path\n"
        "from self_audit.compact_artifacts import load_compact_export\n"
        f"model, meta = load_compact_export('{sandbox_export}', expected_sha='{export_sha}')\n"
        "x = torch.ones(1, 3, 32, 32) * 0.5\n"
        "out = model(x)\n"
        "assert out.shape == (1, 4, 32, 32)\n"
        "assert torch.isfinite(out).all()\n"
        "print('STANDALONE_SUCCESS')\n"
    )
    test_file = sandbox / "run_eval.py"
    test_file.write_text(test_script)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(sandbox)

    res = subprocess.run(
        [sys.executable, str(test_file)],
        cwd=str(sandbox),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "STANDALONE_SUCCESS" in res.stdout
