"""Actual serialized-model integration; no fake model or predictor substitution."""
import json
import copy
from pathlib import Path

import pytest

from scripts import run_context_diagnostic as context_cli
from scripts import run_gradient_bridge_diagnostic as bridge_cli
from self_audit.diagnostic_fixture import build_diagnostic_fixture
from self_audit import gradient_bridge_runtime as runtime


def test_real_fixture_lifecycle_freezes_both_before_gt_and_preserves_batch_scope(tmp_path, monkeypatch):
    fixture = build_diagnostic_fixture(tmp_path / "inputs")
    round_dir = tmp_path / "round"
    a_path, a_sha = fixture["context_config_path"], fixture["context_config_sha256"]
    b_path, b_sha = fixture["gradient_bridge_config_path"], fixture["gradient_bridge_config_sha256"]
    masks = set()
    for manifest_key in ("train_manifest_path", "val_manifest_path"):
        path = Path(fixture[manifest_key])
        manifest = json.loads(path.read_text())
        for patient in manifest.get("patients", []) + manifest.get("labeled", []):
            masks.update((path.parent / record["path"]).resolve() for record in patient["masks"].values())
    assert len(masks) == 3
    opened = []
    original_open = Path.open
    scoring = False

    def guard(path, *args, **kwargs):
        if path.resolve() in masks:
            assert scoring, "predict/fit read a label payload"
            assert (round_dir / "VAL_EXPOSURE.json").is_file()
            assert (round_dir / "predictions-A.frozen.json").is_file()
            assert (round_dir / "predictions-B.frozen.json").is_file()
            assert path.parent.name == "development", "fit/scorer read TRAIN mask"
            opened.append(str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    for command in ("preflight", "lock", "predict"):
        argv = [command, "--config", a_path, "--config-sha256", a_sha, "--round-dir", str(round_dir)]
        if command != "preflight":
            argv += ["--stage", "A"]
        assert context_cli.main(argv) == 0
    for command in ("preflight", "fit-p", "lock-b", "predict"):
        assert bridge_cli.main([command, "--config", b_path, "--config-sha256", b_sha,
                                "--round-dir", str(round_dir)]) == 0
    assert opened == []
    receipt = json.loads((round_dir / "B/PREDICTION_B.json").read_text())
    assert receipt["D0_parity"]["mismatches"] == 0
    assert receipt["D0_parity"]["slices"] == 6
    assert len(receipt["gradient_diagnostics"]) == 6
    batches = receipt["batch_diagnostics"]
    assert len(batches) == receipt["cost"]["gradient_arm_batches"] == 4
    assert sum(len(item["sample_ids"]) for item in batches) == 6
    assert all(set(item["per_arm_kd"]) == set(runtime.ARMS) for item in batches)
    assert all(set(loss) == {"student", "bridge"} for item in batches for loss in item["per_arm_kd"].values())
    assert all("kd_denominator" not in row for row in receipt["gradient_diagnostics"])
    assert receipt["h_dependence"]["patient_ids"][0] != receipt["h_dependence"]["patient_ids"][1]

    # A mutated output must fail before GT access. Restore exact bytes only in
    # this manufactured negative test, then exercise the normal score order.
    shard = Path(receipt["shards"][0]["path"])
    content = shard.read_bytes()
    shard.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(ValueError, match="frozen prediction changed"):
        runtime.score_stage(b_path, round_dir, expected_config_sha256=b_sha)
    assert not (round_dir / "VAL_EXPOSURE.json").exists()
    shard.write_bytes(content)
    scoring = True
    assert context_cli.main(["score", "--stage", "A", "--config", a_path, "--config-sha256", a_sha,
                             "--round-dir", str(round_dir)]) == 0
    assert bridge_cli.main(["score", "--config", b_path, "--config-sha256", b_sha,
                            "--round-dir", str(round_dir)]) == 0
    assert opened
    assert bridge_cli.main(["report", "--round-dir", str(round_dir)]) == 0
    report = json.loads((round_dir / "REPORT.json").read_text())
    assert report["status"] == "FIXTURE_ONLY"
    assert report["patient_results_included"] is False
    assert report["B_result"]["ground_truth_opened_after_authorization"] is True
    assert "rho_scores" not in report["B_result"]
    ledger = json.loads((round_dir / "VAL_EXPOSURE.json").read_text())
    assert ledger["first_score_stage"] == "A"
    assert ledger["both_protocols_locked_before_first_GT"]
    with pytest.raises(ValueError, match="precede"):
        runtime.fit_p_stage(b_path, round_dir, expected_config_sha256=b_sha)


def test_runtime_json_writer_preserves_real_patient_row_lists(tmp_path):
    path = tmp_path / "PATIENT_ROWS_B.json"
    rows = [{"patient_id": "fixture-a"}, {"patient_id": "fixture-b"}]
    runtime._write_once(path, rows)
    assert json.loads(path.read_text()) == rows
    with pytest.raises(FileExistsError):
        runtime._write_once(path, rows)


def test_cli_help_and_unfilled_protocol_block_before_creating_round(tmp_path, capsys):
    with pytest.raises(SystemExit) as exited:
        bridge_cli.main(["--help"])
    assert exited.value.code == 0
    assert "fixture-smoke" in capsys.readouterr().out
    config = runtime.ROOT / "configs/gradient_bridge_historical_val18.template.json"
    assert bridge_cli.main(["preflight", "--config", str(config), "--round-dir", str(tmp_path / "round")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"
    assert not (tmp_path / "round").exists()


def test_wrong_payload_and_u_mask_metadata_fail_closed(tmp_path):
    fixture = build_diagnostic_fixture(tmp_path / "inputs")
    bad = {**fixture["teacher_spec"], "payload_sha256": "0" * 64}
    with pytest.raises(ValueError, match="payload|hash"):
        runtime._artifact_static(bad, "teacher", Path(fixture["gradient_bridge_config_path"]))
    train = json.loads(Path(fixture["train_manifest_path"]).read_text())
    train["unlabeled"][0]["masks"] = {"0": {"path": "MUST_NOT_OPEN.npy", "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="mask|label|forbidden"):
        runtime._canonical_train(train)


def test_exploratory_lineage_cannot_omit_teacher_label_budget(tmp_path):
    fixture = build_diagnostic_fixture(tmp_path / "inputs")
    config = copy.deepcopy(fixture["gradient_bridge_config"])
    config.update(profile="exploratory", synthetic_fixture=False)
    train = json.loads(Path(fixture["train_manifest_path"]).read_text())
    val = json.loads(Path(fixture["val_manifest_path"]).read_text())
    train["synthetic"] = val["synthetic"] = False
    lids = [p["id"] for p in train["labeled"]]
    uids = [p["id"] for p in train["unlabeled"]]
    identity = {"payload_sha256": config["teacher"]["payload_sha256"],
                "manifest_sha256": config["teacher"]["manifest_sha256"],
                "inference_state": "student", "train_manifest_sha256": fixture["train_manifest_sha256"],
                "labeled_ids": lids, "label_budget": len(lids)}
    teacher = {"synthetic": False}
    student = {"synthetic": False, "lineage": {
        "teacher_identity": identity, "data_sha256": fixture["train_manifest_sha256"],
        "lids": lids, "uids": uids, "label_budget": len(lids)}}
    runtime._validate_lineage(config, train, fixture["train_manifest_sha256"], val, teacher, student)
    identity.pop("label_budget")
    with pytest.raises(ValueError, match="label budgets"):
        runtime._validate_lineage(config, train, fixture["train_manifest_sha256"], val, teacher, student)
    identity["label_budget"] = len(lids)
    identity.pop("labeled_ids")
    with pytest.raises(ValueError, match="labeled IDs"):
        runtime._validate_lineage(config, train, fixture["train_manifest_sha256"], val, teacher, student)


def test_a_only_fixture_report_never_claims_patient_results(tmp_path):
    round_dir = tmp_path / "round"
    assert context_cli.main(["fixture-smoke", "--round-dir", str(round_dir)]) == 0
    report = runtime.report_stage(round_dir)
    assert report["status"] == "FIXTURE_ONLY"
    assert report["synthetic_fixture"] is True
    assert report["patient_results_included"] is False
    assert report["stages"]["B"]["score_status"] is None


def test_cli_nonfinite_has_distinct_invalid_status(monkeypatch, capsys, tmp_path):
    from self_audit.gradient_bridge import InvalidGradientDiagnostic

    def invalid(*args, **kwargs):
        raise InvalidGradientDiagnostic("non-finite bridge gradient")

    monkeypatch.setattr(bridge_cli, "predict_stage", invalid)
    assert bridge_cli.main(["predict", "--config", "unused.json", "--round-dir", str(tmp_path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "INVALID_DIAGNOSTIC"
    assert result["reason"] == "INVALID_NONFINITE"
