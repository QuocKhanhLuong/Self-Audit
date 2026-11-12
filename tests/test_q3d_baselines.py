from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from self_audit.candidate_a3d.data import synthetic_fixture
from self_audit.candidate_q3d.baselines import (
    evaluate_teacher_supervised_baseline,
    run_a3d_teacher_free,
    run_teacher_supervised,
)
from self_audit.candidate_q3d.config import canonical_config
from self_audit.model import digest


def _smoke_config(
    arm: str, fixture: dict[str, object], *, updates: int, protocol: str = "A",
) -> dict[str, object]:
    config = canonical_config(arm, dataset="acdc", protocol=protocol, k=2, seed=17)
    config["runtime"]["synthetic"] = True
    config["runtime"]["device"] = "cpu"
    config["runtime"]["smoke_update_overrides"] = {"T": None, "Q": None, "P": None, "J": None}
    stage = "T" if arm == "teacher_supervised" else "J"
    config["runtime"]["smoke_update_overrides"][stage] = updates
    config["dataset"]["manifests"]["train"] = {
        "path": str(fixture["train"].resolve()),
        "sha256": fixture["train_sha256"],
        "status": "READY",
    }
    config["dataset"]["manifests"]["development"] = {
        "path": str(fixture["evaluation"].resolve()),
        "sha256": fixture["evaluation_sha256"],
        "status": "READY",
    }
    if protocol == "B":
        config["dataset"]["manifests"]["teacher_train"] = {
            "path": str(fixture["teacher_train"].resolve()),
            "sha256": fixture["teacher_train_sha256"],
            "status": "READY",
        }
    return config


def _protocol_b_fixture(root: Path) -> dict[str, object]:
    fixture = synthetic_fixture(
        root, seed=37, n_labeled=3, n_unlabeled=0,
        n_evaluation=1, shape_tzyx=(2, 8, 32, 32),
    )
    all_train = json.loads(Path(fixture["train"]).read_text(encoding="utf-8"))
    student_train = copy.deepcopy(all_train)
    student_train["labeled"] = all_train["labeled"][:2]
    unlabeled_record = copy.deepcopy(all_train["labeled"][2])
    unlabeled_record.pop("masks")
    student_train["unlabeled"] = [unlabeled_record]
    student_train["label_budget"] = 2
    student_path = root / "student_k2_train.json"
    student_path.write_text(json.dumps(student_train, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    student_sha256 = hashlib.sha256(student_path.read_bytes()).hexdigest()
    return {
        "train": student_path,
        "train_sha256": student_sha256,
        "evaluation": fixture["evaluation"],
        "evaluation_sha256": fixture["evaluation_sha256"],
        "teacher_train": fixture["train"],
        "teacher_train_sha256": fixture["train_sha256"],
    }


def test_a3d_teacher_free_runs_public_engine_with_exact_k_only_budget(tmp_path: Path) -> None:
    fixture = synthetic_fixture(
        tmp_path / "a3d-data", seed=29, n_labeled=2, n_unlabeled=1,
        n_evaluation=1, shape_tzyx=(2, 8, 32, 32),
    )
    config = _smoke_config("a3d_teacher_free", fixture, updates=4)

    result = run_a3d_teacher_free(
        config, tmp_path / "a3d-run", device="cpu", smoke=True,
    )

    assert result["evidence"] == "SOFTWARE_ONLY"
    assert result["protocol"] == "A"
    assert result["label_budget_k"] == 2
    assert result["seed"] == 17
    assert result["logical_updates"] == 4
    assert result["optimizer_updates"] == 4
    assert result["optimizer_update_counts"]["student_unlabeled"] == 0
    assert result["training_forward_counts"]["unlabeled"] == 0
    assert result["trainer_report"]["unlabeled_usage"].startswith("UNUSED")
    assert result["trainer_report"]["unlabeled_image_reads"] == []
    assert Path(result["checkpoint_manifest_path"]).is_file()
    assert (tmp_path / "a3d-run" / "BASELINE_REPORT.json").is_file()


def test_teacher_supervised_runs_cpu_t_only_updates_for_protocol_b_k_pool(tmp_path: Path) -> None:
    fixture = _protocol_b_fixture(tmp_path / "teacher-data")
    config = _smoke_config("teacher_supervised", fixture, updates=2, protocol="B")

    result = run_teacher_supervised(
        config, tmp_path / "teacher-run", device="cpu", smoke=True,
    )

    assert result["evidence"] == "SOFTWARE_ONLY"
    assert result["protocol"] == "B"
    assert result["baseline_supervision_protocol"] == "K_only"
    assert result["teacher_train_manifest_used"] is False
    assert result["label_budget_k"] == 2
    assert result["labeled_patient_ids"] == ["train_labeled_000", "train_labeled_001"]
    assert result["stages"] == ["T"]
    assert result["tokenizer_enabled"] is False
    assert result["student_artifact"] is False
    assert result["logical_updates"] == 2
    assert result["optimizer_updates"] == 2
    assert result["training_forward_counts"]["unlabeled"] == 0
    assert result["evaluation"]["kind"] == "teacher_supervised_region_baseline"
    assert result["evaluation"]["separate_from_student_export"] is True
    assert result["independent_baseline"] == "NOT_RUN"
    assert result["stage_contract"] == "T_ONLY_STANDALONE"
    assert Path(result["checkpoint_path"]).is_file()
    assert (tmp_path / "teacher-run" / "BASELINE_REPORT.json").is_file()

    evaluation = evaluate_teacher_supervised_baseline(
        result["checkpoint_path"],
        result["checkpoint_sha256"],
        config,
        fixture["evaluation"],
        fixture["evaluation_sha256"],
        tmp_path / "teacher-evaluation",
        device="cpu",
        split="dev",
    )
    assert evaluation["artifact_kind"] == "teacher_supervised_region_baseline"
    assert evaluation["student_export"] is False
    assert evaluation["checkpoint_schema"] == "self_audit.candidate_q3d.resume.v3"
    assert evaluation["parent_protocol"] == "B"
    assert evaluation["baseline_supervision_protocol"] == "K_only"
    assert evaluation["teacher_train_manifest_used"] is False
    assert evaluation["label_budget_k"] == 2
    assert evaluation["evaluation_manifest_sha256"] == fixture["evaluation_sha256"]
    assert evaluation["independent_baseline"] == "NOT_RUN"
    assert evaluation["stage_contract"] == "T_ONLY_STANDALONE"
    assert (tmp_path / "teacher-evaluation" / "BASELINE_EVALUATION.json").is_file()

    config_path = tmp_path / "teacher-q3d-config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    cli = subprocess.run(
        [
            sys.executable, str(root / "evaluate.py"),
            "--checkpoint", result["checkpoint_path"],
            "--checkpoint-sha256", result["checkpoint_sha256"],
            "--config", str(config_path),
            "--data", str(fixture["evaluation"]), "--data-sha256", fixture["evaluation_sha256"],
            "--out", str(tmp_path / "teacher-cli-evaluation"), "--device", "cpu",
        ],
        cwd=root, text=True, capture_output=True, check=False,
    )
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["evaluation"]["role"] == "development"


def test_shared_t_artifact_evaluates_as_teacher_supervised_without_tokenizer_sidecar(
    tmp_path: Path,
) -> None:
    """A full-run T export is a valid zero-J teacher baseline input."""

    import self_audit.candidate_q3d.engine as engine
    from self_audit.candidate_q3d.baselines import evaluate_teacher_supervised_baseline
    from self_audit.candidate_a3d.data import synthetic_fixture
    from train import _manufactured_teacher

    fixture = synthetic_fixture(
        tmp_path / "shared-t-data", seed=71, n_labeled=2, n_unlabeled=1,
        n_evaluation=1, shape_tzyx=(2, 8, 32, 32),
    )
    producer = _smoke_config("v1", fixture, updates=1)
    producer["runtime"]["smoke_update_overrides"]["T"] = 1
    t_result = engine.run_training(
        producer, tmp_path / "producer", device="cpu", smoke=True,
        teacher=_manufactured_teacher(producer), stage="T",
        synthetic_gate_override=True,
    )
    reference = t_result["stage_artifacts"]["T"]

    consumer = _smoke_config("teacher_supervised", fixture, updates=1)
    consumer["runtime"]["smoke_update_overrides"]["T"] = 1
    evaluation = evaluate_teacher_supervised_baseline(
        reference["checkpoint_path"], reference["checkpoint_sha256"], consumer,
        fixture["evaluation"], fixture["evaluation_sha256"],
        tmp_path / "evaluation", device="cpu", split="dev",
    )
    assert evaluation["independent_baseline"] is False
    assert evaluation["shared_teacher_stage"] == "T"
    assert evaluation["stage_contract"] == "T0J"
    payload = __import__("torch").load(
        reference["checkpoint_path"], map_location="cpu", weights_only=True,
    )
    assert payload.get("tokenizer_state") is None
    assert any(row.get("event") == "development_selection" for row in payload["metrics"])


def test_a3d_baseline_export_runs_q3d_cli_on_phase_pinned_test(tmp_path: Path) -> None:
    fixture = synthetic_fixture(
        tmp_path / "a3d-test-data", seed=43, n_labeled=2, n_unlabeled=1,
        n_evaluation=1, shape_tzyx=(2, 8, 32, 32),
    )
    test_payload = json.loads(Path(fixture["evaluation"]).read_text(encoding="utf-8"))
    test_payload["role"] = "test"
    test_payload["scope"] = "synthetic_phase_pinned_test"
    test_payload["patients"][0]["phases"] = {"ED": 0, "ES": 1}
    test_manifest = tmp_path / "a3d-test-data" / "test.json"
    test_manifest.write_text(json.dumps(test_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    test_sha256 = digest(test_manifest)
    config = _smoke_config("a3d_teacher_free", fixture, updates=4)
    config["dataset"]["manifests"]["test"] = {
        "path": str(test_manifest.resolve()), "sha256": test_sha256, "status": "READY",
    }

    trained = run_a3d_teacher_free(config, tmp_path / "a3d-test-run", device="cpu", smoke=True)
    checkpoint = Path(trained["checkpoint_path"])
    export_dir = tmp_path / "a3d-test-export"
    from self_audit.candidate_a3d.io import export_student

    exported = export_student(checkpoint, trained["checkpoint_manifest_sha256"], export_dir)
    export_manifest = export_dir / "MANIFEST.json"
    config_path = tmp_path / "a3d-q3d-config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_dir = tmp_path / "a3d-q3d-test-evaluation"
    root = Path(__file__).resolve().parents[1]
    cli = subprocess.run(
        [
            sys.executable, str(root / "evaluate.py"),
            "--checkpoint", str(export_dir),
            "--checkpoint-sha256", exported["manifest_sha256"],
            "--config", str(config_path),
            "--data", str(test_manifest), "--data-sha256", test_sha256,
            "--out", str(output_dir), "--device", "cpu", "--split", "test",
        ],
        cwd=root, text=True, capture_output=True, check=False,
    )
    assert cli.returncode == 0, cli.stderr
    result = json.loads(cli.stdout)
    assert result["role"] == result["split"] == "test"
    assert result["manifest_sha256"] == test_sha256
    assert result["artifact_kind"] == "a3d_teacher_free_student"
    assert result["student_only_inference"] is True
    assert result["checkpoint_manifest_sha256"] == exported["manifest_sha256"]
    from self_audit.candidate_a3d.io import source_inventory
    current_source_hash = hashlib.sha256(
        json.dumps(source_inventory(), sort_keys=True).encode()
    ).hexdigest()
    assert result["source_inventory_sha256"] == current_source_hash
    assert result["patient_macro"]["fg_dice"] is not None
    assert result["per_patient"]["evaluation_000"]["clinical"]["errors"]
    metrics = json.loads((output_dir / "Q3D_METRICS.json").read_text(encoding="utf-8"))
    assert metrics["role"] == "test"
    assert metrics["per_patient"]["evaluation_000"]["clinical"]["errors"]
    report_path = tmp_path / "baseline_test_adapter.json"
    source_paths = (
        "self_audit/candidate_q3d/baselines.py", "evaluate.py", "tests/test_q3d_baselines.py",
        "self_audit/candidate_a3d/k1.py", "self_audit/candidate_a3d/io.py",
        "self_audit/candidate_q3d/evaluation.py",
    )
    source_hashes = {
        relative: digest(root / relative) for relative in source_paths
    }
    report_path.write_text(json.dumps({
        "schema": "self_audit.candidate_q3d.a3d_baseline_test_adapter.v1",
        "status": "PASS_SOFTWARE_ONLY_SYNTHETIC",
        "test": "test_a3d_baseline_export_runs_q3d_cli_on_phase_pinned_test",
        "workflow": "actual Q3D baseline runner -> unchanged A3D student export -> repository evaluate.py CLI -> Q3D evaluate_manifest",
        "checkpoint_export": {
            "checkpoint_manifest_sha256": trained["checkpoint_manifest_sha256"],
            "student_export_manifest_sha256": exported["manifest_sha256"],
            "source_inventory_sha256": result["source_inventory_sha256"],
            "source_inventory_matches_current_checkout": result["source_inventory_sha256"] == current_source_hash,
            "checkpoint_schema": result["checkpoint_schema"],
        },
        "evaluation": {
            "role": result["role"], "split": result["split"],
            "manifest_sha256": test_sha256,
            "phase_pins": {"ED": 0, "ES": 1},
            "patient_macro": result["patient_macro"],
            "per_patient_clinical_errors": result["per_patient"]["evaluation_000"]["clinical"]["errors"],
        },
        "source_sha256": source_hashes,
        "evidence_limits": {
            "data": "manufactured synthetic fixture only",
            "patient_metrics": "NOT_ESTABLISHED",
            "gpu_training_or_evaluation": "NOT_RUN",
            "clinical_evidence": "NOT_ESTABLISHED",
            "efficacy_or_SOTA": "NOT_ESTABLISHED",
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert json.loads(report_path.read_text())["status"] == "PASS_SOFTWARE_ONLY_SYNTHETIC"
