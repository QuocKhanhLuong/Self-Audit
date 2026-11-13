import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from self_audit.candidate_fuse.config import (
    config_sha256,
    default_config,
    load_config,
    normalize_config,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fuse_template_and_copied_teacher_contract():
    cfg = load_config(ROOT / "configs/candidate_fuse/v1.json")
    assert cfg["seed"] == 42
    assert cfg["model"]["teacher"] == json.loads(
        (ROOT / "configs/candidate_q3d/v1.json").read_text()
    )["model"]["teacher"]
    assert cfg["data"]["train_manifest"] is None
    assert len(cfg["model"]["student"]["channels"]) == 5
    with pytest.raises(ValueError, match="existing pinned"):
        validate_config(cfg, real_run=True)


def test_synthetic_config_and_stable_canonical_hash():
    cfg = default_config(synthetic=True)
    assert cfg["model"]["student"]["channels"] == [4, 8, 12, 16, 32]
    assert cfg["training"]["teacher_epochs"] == cfg["training"]["joint_epochs"] == 1
    assert config_sha256(cfg) == config_sha256(dict(reversed(list(cfg.items()))))
    with pytest.raises(ValueError, match="locked to 42"):
        cfg["seed"] = 7
        normalize_config(cfg)


def test_cli_help_does_not_import_fuse_models():
    result = subprocess.run([sys.executable, "train.py", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--fuse-stage" in result.stdout
    assert "--fuse-teacher-artifact" in result.stdout
    assert "self_audit.candidate_fuse" not in result.stdout


def test_evaluate_fuse_schema_dispatch_is_lazy():
    code = (
        "import sys, evaluate; fn=evaluate.evaluator_for_schema('self_audit.candidate_fuse.export.v1'); "
        "assert callable(fn); assert 'self_audit.candidate_fuse.evaluation' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_evaluate_help_does_not_import_fuse_models():
    result = subprocess.run([sys.executable, "evaluate.py", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--fuse-mode" in result.stdout



def test_fuse_cli_requires_explicit_development_split(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"placeholder")
    import hashlib
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "model.pt.manifest.json"
    manifest.write_text(json.dumps({"schema": "self_audit.candidate_fuse.export.v1", "artifact": artifact.name,
                                    "artifact_sha256": checksum, "artifact_bytes": artifact.stat().st_size}))
    result = subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(artifact),
        "--checkpoint-sha256", checksum, "--data", "unused", "--data-sha256", "0" * 64,
        "--out", str(tmp_path / "out"),
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FUSE evaluation requires explicit --split dev" in result.stderr


@pytest.mark.parametrize("field,value", [
    ("teacher_lr_encoder", 2e-4),
    ("teacher_lr_head", 2e-3),
    ("student_lr", 6e-4),
    ("weight_decay", 2e-4),
    ("warmup_fraction", 0.1),
    ("gradient_clip_norm", 2.0),
    ("unlabeled_ramp_fraction", 0.2),
    ("mimic_weight", 0.2),
    ("max_teacher_dropout", 0.4),
    ("mimic_weight", True),
    ("student_lr", float("nan")),
    ("teacher_lr_head", float("inf")),
])
def test_fuse_training_scalars_are_locked_and_numeric(field, value):
    config = default_config(synthetic=True)
    config["training"][field] = value
    with pytest.raises(ValueError, match="FUSE training scalar"):
        normalize_config(config)
