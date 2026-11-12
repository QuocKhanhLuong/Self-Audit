from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from self_audit.candidate_a3d.data import synthetic_fixture
from self_audit.candidate_q3d.config import canonical_config, load_config
from self_audit.candidate_q3d.diagnostics import (
    _codebook_diagnostics,
    _pooled_teacher_outputs,
    _spearman_report,
    run_dev_diagnostics,
)
from self_audit.candidate_q3d.engine import (
    _atomic_torch_save,
    _build_models,
    _data_identity_without_labels,
    _load_data,
    _object_sha256,
    _rng_state,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_bundle(root: Path, *, arm: str = "v1", mismatch_o1_teacher: bool = False):
    fixture = synthetic_fixture(
        root / "fixture", seed=91, n_labeled=2, n_unlabeled=2,
        n_evaluation=3, shape_tzyx=(2, 8, 17, 19),
    )
    dev_path = Path(fixture["evaluation"])
    dev = json.loads(dev_path.read_text())
    for patient in dev["patients"]:
        patient["ed_frame"], patient["es_frame"] = 0, 1
    dev_path.write_text(json.dumps(dev, sort_keys=True) + "\n")
    dev_sha = _digest(dev_path)

    # Force explicit missing-RV exclusions for one synthetic DEV patient.
    first = dev["patients"][0]
    for frame_record in first["masks"].values():
        mask_path = dev_path.parent / frame_record["path"]
        mask = np.load(mask_path, allow_pickle=False)
        mask[mask == 1] = 0
        np.save(mask_path, mask)
        frame_record["sha256"] = _digest(mask_path)
    dev_path.write_text(json.dumps(dev, sort_keys=True) + "\n")
    dev_sha = _digest(dev_path)

    config = canonical_config(arm, dataset="acdc", protocol="A", k=2, seed=42)
    config["runtime"].update({"synthetic": True, "device": "cpu"})
    config["dataset"]["manifests"]["train"] = {
        "path": "fixture/train.json",
        "sha256": fixture["train_sha256"], "status": "READY",
    }
    config["dataset"]["manifests"]["development"] = {
        "path": "fixture/development.json", "sha256": dev_sha, "status": "READY",
    }
    config_path = root / "q3d.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    config = load_config(config_path)

    teacher, student, tokenizer, _ = _build_models(
        config, device=torch.device("cpu"), teacher=None, student=None, tokenizer=None,
    )
    if tokenizer is not None:
        torch.manual_seed(117)
        with torch.no_grad():
            tokenizer.C0.copy_(F.normalize(torch.randn_like(tokenizer.C0), dim=1))
            tokenizer.is_initialized.fill_(True)
            tokenizer.is_frozen.fill_(True)
        tokenizer.reconstructor = None
        tokenizer.anatomy_head = None
        student.install_codebook({"codes": tokenizer.C0, **tokenizer.artifact_metadata()})
    teacher.eval().requires_grad_(False)
    if tokenizer is not None:
        tokenizer.eval().requires_grad_(False)
    student.eval().requires_grad_(False)

    data = _load_data(config, None)
    current_teacher_state = teacher.state_dict()
    best_teacher_state = current_teacher_state
    if mismatch_o1_teacher:
        best_teacher_state = {key: value.clone() if torch.is_tensor(value) else value
                              for key, value in current_teacher_state.items()}
        tensor_key = next(key for key, value in best_teacher_state.items() if torch.is_tensor(value))
        best_teacher_state[tensor_key] = best_teacher_state[tensor_key] + 1.0
    state = {
        "schema": "self_audit.candidate_q3d.resume.v3",
        "config_sha256": _object_sha256(config),
        "data_identity": _data_identity_without_labels(data),
        "completed_stages": ["T", *(["Q"] if config["tokenizer"]["enabled"] else []),
                             *(["P"] if config["physiology"]["enabled"] else []), "J"],
        "gate": {"status": "PASS", "passed": True},
        "best_teacher_state": best_teacher_state,
        "teacher_state": current_teacher_state,
        "best_student_state": student.state_dict(),
        "student_state": student.state_dict(),
        "tokenizer_state": tokenizer.state_dict() if tokenizer is not None else None,
        "rng": _rng_state(),
    }
    checkpoint = root / "research.pt"
    checkpoint_sha = _atomic_torch_save(state, checkpoint)
    return config, config_path, checkpoint, checkpoint_sha, dev_path, dev_sha, (teacher, student, tokenizer)


def test_synthetic_cli_and_api_produce_report_only_dev_diagnostics(tmp_path: Path):
    torch.set_num_threads(1)
    config, config_path, checkpoint, checkpoint_sha, dev, dev_sha, modules = _fixture_bundle(tmp_path)
    output = tmp_path / "diagnostics.json"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/q3d_diagnostics.py"),
         "--config", str(config_path), "--checkpoint", str(checkpoint),
         "--checkpoint-sha256", checkpoint_sha, "--device", "cpu", "--out", str(output)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "COMPLETED"
    assert report["scope"] == "development_only"
    assert report["test_masks_opened"] is False
    assert report["training_or_checkpoint_selection_influence"] is False
    assert report["teacher_predictions_reused_from_label_free_first_pass"] is True
    assert report["teacher_student_hard_code_agreement"]["frame_count"] == 6
    assert report["teacher_student_hard_code_agreement"]["status"] == "COMPUTED"
    assert report["tokenizer_gate"]["assignment_count"] > 0
    exclusions = report["teacher_rv_metric_correlations"]["w_pat_vs_teacher_rv_hd95_exclusions"]
    assert exclusions["excluded_phase_count"] >= 2
    assert any("reference_rv_absent" in reason for reason in exclusions["by_reason"])
    assert set(report["teacher_rv_metric_correlations"]) >= {
        "w_pat_vs_teacher_rv_dice", "w_pat_vs_teacher_rv_hd95_mm",
    }
    assert all(item["status"].startswith("NOT_RUN")
               for item in report["teacher_rv_metric_correlations"].values()
               if isinstance(item, dict) and "status" in item)

    api = run_dev_diagnostics(
        *modules[:1], modules[2], modules[1], config=config,
        development_manifest=dev, development_sha256=dev_sha,
        train_patient_ids=("train_labeled_000", "train_labeled_001",
                           "train_unlabeled_000", "train_unlabeled_001"),
        protocol="A", device="cpu", synthetic=True,
    )
    assert api["teacher_state_sha256"] == report["teacher_state_sha256"]
    assert api["tokenizer_gate"] == report["tokenizer_gate"]
    with pytest.raises(ValueError, match="differs from config identity"):
        run_dev_diagnostics(
            *modules[:1], modules[2], modules[1], config=config,
            development_manifest=dev, development_sha256=dev_sha,
            train_patient_ids=("train_labeled_000", "train_labeled_001"),
            protocol="B", device="cpu", synthetic=True,
        )


def test_checkpoint_sidecar_tamper_and_deployment_rejection(tmp_path: Path):
    _, config_path, checkpoint, checkpoint_sha, _, _, _ = _fixture_bundle(tmp_path)
    with checkpoint.open("ab") as stream:
        stream.write(b"tamper")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/q3d_diagnostics.py"),
         "--config", str(config_path), "--checkpoint", str(checkpoint),
         "--checkpoint-sha256", checkpoint_sha, "--out", str(tmp_path / "bad.json")],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "sidecar mismatch" in result.stderr

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    _, clean_config, clean_checkpoint, _, _, _, _ = _fixture_bundle(clean_root)
    clean_checkpoint.with_suffix(".pt.sha256").unlink()
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/q3d_diagnostics.py"),
         "--config", str(clean_config), "--checkpoint", str(clean_checkpoint),
         "--out", str(clean_root / "missing-sidecar.json")],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "requires its .sha256 sidecar" in result.stderr
    clean_checkpoint.with_suffix(".pt.sha256").write_text("")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/q3d_diagnostics.py"),
         "--config", str(clean_config), "--checkpoint", str(clean_checkpoint),
         "--out", str(clean_root / "empty-sidecar.json")],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "sidecar is malformed" in result.stderr

    export = tmp_path / "student-export.pt"
    torch.save({"schema": "q3d.student_export.v1"}, export)
    export.with_suffix(".pt.sha256").write_text(f"{_digest(export)}  {export.name}\n")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/q3d_diagnostics.py"),
         "--config", str(config_path), "--checkpoint", str(export),
         "--out", str(tmp_path / "deployment.json")],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "research resume checkpoint" in result.stderr


def test_role_train_overlap_and_spearman_guards(tmp_path: Path):
    torch.set_num_threads(1)
    config, _, _, _, dev, dev_sha, modules = _fixture_bundle(tmp_path)
    test_manifest = json.loads(Path(dev).read_text())
    test_manifest["role"] = "test"
    test_path = tmp_path / "test-role.json"
    test_path.write_text(json.dumps(test_manifest, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="development/external_frozen"):
        run_dev_diagnostics(
            *modules[:1], modules[2], modules[1], config=config,
            development_manifest=test_path, development_sha256=_digest(test_path),
            train_patient_ids=("not-a-development-patient",), protocol="A", synthetic=True,
        )
    with pytest.raises(ValueError, match="overlap with TRAIN"):
        run_dev_diagnostics(
            *modules[:1], modules[2], modules[1], config=config,
            development_manifest=dev, development_sha256=dev_sha,
            train_patient_ids=("evaluation_000",), protocol="A", synthetic=True,
        )
    assert _spearman_report([{"w_pat": 0.1, "rv_dice": 0.8}, {"w_pat": 0.2, "rv_dice": 0.7}])["rho"] is None
    assert _spearman_report([
        {"w_pat": 0.1, "rv_dice": 0.8}, {"w_pat": 0.2, "rv_dice": 0.7},
        {"w_pat": 0.3, "rv_dice": 0.6},
    ]) == {"status": "COMPUTED", "n": 3, "rho": -1.0, "p_value": 0.0}


def test_codebook_metrics_use_exact_counts_and_soft_class_ownership():
    codes = [torch.tensor([[[[0, 0, 1, 1]]]])]
    classes = torch.tensor([[[[[0.0, 0.0, 1.0, 1.0]]],
                              [[[1.0, 1.0, 0.0, 0.0]]],
                              [[[0.0, 0.0, 0.0, 0.0]]],
                              [[[0.0, 0.0, 0.0, 0.0]]]]])
    report = _codebook_diagnostics(
        codes, [classes], 2, min_perplexity=2.0, max_dead_fraction=0.0,
        min_class_codes=1, dead_threshold=0.001,
    )
    assert report["perplexity"] == pytest.approx(2.0)
    assert report["dead_code_fraction"] == 0.0
    assert report["class_code_coverage"]["RV"] == 1
    assert report["class_code_coverage"]["LV"] == 0
    assert report["class_code_ownership_threshold_strictly_greater_than"] == 0.5

    tied = _codebook_diagnostics(
        [torch.zeros((1, 1, 1, 1), dtype=torch.long)],
        [torch.tensor([[[[[0.0]]], [[[0.5]]], [[[0.5]]], [[[0.0]]]]])],
        2, min_perplexity=0.0, max_dead_fraction=1.0,
        min_class_codes=0, dead_threshold=0.001,
    )
    assert tied["class_code_coverage"]["RV"] == 0
    assert tied["class_code_coverage"]["MYO"] == 0


def test_odd_17x19_native_grid_phase_pooling_and_c1_cli(tmp_path: Path):
    torch.set_num_threads(1)
    logits = torch.randn((1, 4, 8, 17, 19), dtype=torch.float32)
    pooled, confidence = _pooled_teacher_outputs(
        logits, (2, 2), student_phase=((1, 16, 16), (0, 0, 0)),
    )
    assert pooled.shape == (1, 4, 8, 2, 2)
    assert confidence.shape == (1, 1, 8, 2, 2)

    _config, config_path, checkpoint, checkpoint_sha, _, _, _ = _fixture_bundle(tmp_path, arm="c1")
    output = tmp_path / "c1-diagnostics.json"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/q3d_diagnostics.py"),
         "--config", str(config_path), "--checkpoint", str(checkpoint),
         "--checkpoint-sha256", checkpoint_sha, "--device", "cpu", "--out", str(output)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["tokenizer_gate"] == {
        "status": "NOT_APPLICABLE", "reason": "C1_has_no_tokenizer_or_codebook",
    }
    assert report["teacher_student_hard_code_agreement"]["status"] == "NOT_APPLICABLE"
    assert len(report["physiology"]["patients"]) == 3


def test_o1_uses_current_teacher_state(tmp_path: Path):
    torch.set_num_threads(1)
    from self_audit.candidate_q3d.engine import _state_sha256

    _config, config_path, checkpoint, checkpoint_sha, _, _, modules = _fixture_bundle(
        tmp_path, arm="o1", mismatch_o1_teacher=True,
    )
    expected_current_sha = _state_sha256(modules[0])
    output = tmp_path / "o1-diagnostics.json"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/q3d_diagnostics.py"),
         "--config", str(config_path), "--checkpoint", str(checkpoint),
         "--checkpoint-sha256", checkpoint_sha, "--device", "cpu", "--out", str(output)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["teacher_state_source"] == "teacher_state"
    assert report["teacher_state_sha256"] == expected_current_sha
