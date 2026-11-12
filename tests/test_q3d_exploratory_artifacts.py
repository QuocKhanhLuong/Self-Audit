"""Exploratory quality-waiver propagation through Q/P stage artifacts and student export.

Real-path checks use ``runtime.synthetic=False`` payloads that mirror what the
engine writes for a real run, so they run on CPU without teacher weights. The
end-to-end test uses the synthetic fixture only to produce genuine stage files
and a genuine engine checkpoint; its results are software evidence only.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np
import torch

from self_audit.candidate_q3d import stages
from self_audit.candidate_q3d.export import (
    EXPORT_SCHEMA,
    export_checkpoint,
    load_export,
)
from self_audit.candidate_q3d.quality import WAIVER_EVENTS_KEY, WAIVER_STATE_KEY, quality_waiver
from self_audit.candidate_q3d.student import CandidateQ3DStudent

REASON = "overnight exploratory full pipeline"
REAL = {"runtime": {"synthetic": False}}
SYNTHETIC = {"runtime": {"synthetic": True}}


def _policy(reason: str = REASON) -> dict:
    policy = quality_waiver(reason)
    assert policy is not None
    return policy


def _failed_gate() -> dict:
    return {
        "status": "NO_GO",
        "passed": False,
        "checks": {"perplexity": False, "dead_code_fraction": True,
                   "RV_codes": True, "MYO_codes": True, "LV_codes": True},
        "observed": {"perplexity": 3.5, "dead_code_fraction": 0.1,
                     "class_code_coverage": {"RV": 4, "MYO": 4, "LV": 4}},
        "thresholds": {"min_perplexity": 8.0, "max_dead_code_fraction": 0.5, "min_class_codes": 2},
    }


def _real_q_payload(policy: dict | None = None, *, scope: str = "EXPLORATORY_TRAIN_EXECUTION") -> dict:
    """Mirror the engine's real-run Q state after an explicit tokenizer waiver."""
    policy = copy.deepcopy(policy or _policy())
    gate = _failed_gate()
    return {
        "gate": copy.deepcopy(gate),
        "gate_override": {
            "status": "EXPLORATORY_QUALITY_WAIVER",
            "continuation_allowed": True,
            "evidence_scope": scope,
            "original_gate": copy.deepcopy(gate),
            "waiver": copy.deepcopy(policy),
        },
        "synthetic_gate_override_used": False,
        "exploratory_tokenizer_waiver_used": True,
        "tokenizer_artifact": {
            "version": "q3d_teacher_codebook.v1", "sha256": "a" * 64, "size": 4, "dimension": 4,
            "initialized": True, "frozen": True, "train_heads_present": False,
            "gate": copy.deepcopy(gate),
            WAIVER_STATE_KEY: copy.deepcopy(policy),
            "exploratory": True,
        },
        WAIVER_STATE_KEY: copy.deepcopy(policy),
        "exploratory": True,
    }


def _validate_q(config: dict, payload: dict, request: dict | None) -> None:
    stages._validate_tokenizer_gate(
        config, payload, explicit_override_requested=False, request_policy=request,
    )


# --------------------------------------------------------------------------
# Q tokenizer gate (real path)
# --------------------------------------------------------------------------


def test_real_failed_q_gate_enters_only_with_the_exact_explicit_policy() -> None:
    payload = _real_q_payload()
    _validate_q(REAL, payload, _policy())
    # The measured gate stays NO_GO; validation never rewrites it.
    assert payload["gate"]["status"] == "NO_GO" and payload["gate"]["passed"] is False

    with pytest.raises(RuntimeError, match="requires the same explicit --q3d-exploratory-quality-waiver"):
        _validate_q(REAL, payload, None)
    with pytest.raises(RuntimeError, match="differs from the explicitly supplied waiver"):
        _validate_q(REAL, payload, _policy("a different reason"))


def test_default_strict_real_q_gate_still_stops() -> None:
    payload = _real_q_payload()
    for key in ("gate_override", WAIVER_STATE_KEY, "exploratory", "exploratory_tokenizer_waiver_used"):
        payload.pop(key)
    for key in ("gate", WAIVER_STATE_KEY, "exploratory"):
        payload["tokenizer_artifact"].pop(key)
    with pytest.raises(RuntimeError, match="tokenizer gate failed"):
        _validate_q(REAL, payload, None)
    # A request policy alone never excuses a payload that did not record its waiver.
    with pytest.raises(RuntimeError, match="tokenizer gate failed"):
        _validate_q(REAL, payload, _policy())


@pytest.mark.parametrize("tamper", [
    "forged_pass", "pass_status_on_failed", "synthetic_scope_on_real", "synthetic_combined",
    "metadata_missing_policy", "metadata_gate_rewritten", "metadata_not_frozen",
    "uninitialized_codebook", "override_gate_rewritten", "override_policy_differs",
    "payload_policy_missing", "marker_without_policy", "policy_without_marker",
])
def test_real_q_waiver_rejects_forged_or_inconsistent_records(tamper: str) -> None:
    payload = _real_q_payload()
    if tamper == "forged_pass":
        payload["gate"] = {**payload["gate"], "status": "PASS", "passed": True}
    elif tamper == "pass_status_on_failed":
        payload["gate"]["status"] = "PASS"
        payload["gate_override"]["original_gate"]["status"] = "PASS"
        payload["tokenizer_artifact"]["gate"]["status"] = "PASS"
    elif tamper == "synthetic_scope_on_real":
        payload["gate_override"]["evidence_scope"] = "SOFTWARE_ONLY_SYNTHETIC"
    elif tamper == "synthetic_combined":
        payload["synthetic_gate_override_used"] = True
    elif tamper == "metadata_missing_policy":
        payload["tokenizer_artifact"].pop(WAIVER_STATE_KEY)
    elif tamper == "metadata_gate_rewritten":
        payload["tokenizer_artifact"]["gate"] = {**payload["gate"], "status": "PASS", "passed": True}
    elif tamper == "metadata_not_frozen":
        payload["tokenizer_artifact"]["frozen"] = False
    elif tamper == "uninitialized_codebook":
        for record in (payload["gate"], payload["gate_override"]["original_gate"],
                       payload["tokenizer_artifact"]["gate"]):
            record.update({"status": "NOT_READY", "missing": ["initialized_codebook"]})
    elif tamper == "override_gate_rewritten":
        payload["gate_override"]["original_gate"] = {"status": "PASS", "passed": True}
    elif tamper == "override_policy_differs":
        payload["gate_override"]["waiver"] = _policy("other")
    elif tamper == "payload_policy_missing":
        payload.pop(WAIVER_STATE_KEY)
        payload.pop("exploratory")
    elif tamper == "marker_without_policy":
        payload.pop(WAIVER_STATE_KEY)
    elif tamper == "policy_without_marker":
        payload.pop("exploratory")
    with pytest.raises((RuntimeError, ValueError)):
        _validate_q(REAL, payload, _policy())


def test_synthetic_waiver_requires_synthetic_scope_and_keeps_synthetic_override_distinct() -> None:
    _validate_q(SYNTHETIC, _real_q_payload(scope="SOFTWARE_ONLY_SYNTHETIC"), _policy())
    with pytest.raises(RuntimeError):
        _validate_q(SYNTHETIC, _real_q_payload(), _policy())
    synthetic_override = {
        "gate": _failed_gate(),
        "gate_override": {"status": "SYNTHETIC_TEST_OVERRIDE", "continuation_allowed": True},
        "synthetic_gate_override_used": True,
        "tokenizer_artifact": {"initialized": True, "frozen": True},
    }
    stages._validate_tokenizer_gate(SYNTHETIC, synthetic_override, explicit_override_requested=True)
    # The synthetic override is never reachable for a real config, waiver or not.
    with pytest.raises(RuntimeError):
        stages._validate_tokenizer_gate(
            REAL, synthetic_override, explicit_override_requested=True, request_policy=_policy(),
        )


def test_passing_q_gate_under_policy_is_marked_but_not_rewritten() -> None:
    payload = {
        "gate": {"status": "PASS", "passed": True},
        "tokenizer_artifact": {"initialized": True, "frozen": True},
        WAIVER_STATE_KEY: _policy(),
        "exploratory": True,
    }
    _validate_q(REAL, payload, _policy())
    with pytest.raises(RuntimeError, match="explicit"):
        _validate_q(REAL, payload, None)
    strict_pass = {"gate": {"status": "PASS", "passed": True}}
    _validate_q(REAL, strict_pass, None)
    _validate_q(REAL, strict_pass, _policy())


# --------------------------------------------------------------------------
# P physiology support gate (real path)
# --------------------------------------------------------------------------


def _real_p_payload(policy: dict | None = None) -> dict:
    return {
        "physiology_support_gate": {
            "context": "P physiology cache", "status": "NO_GO_U_ESS_GATE", "passed": False,
            "mean_w_pat": 0.2, "ess": 1.1, "ess_fraction": 0.1,
            "override_used": True, "override_scope": "exploratory_quality_waiver",
            "override_policy": "exploratory_low_ess_relative_normalized",
        },
        "synthetic_gate_override_used": False,
        WAIVER_STATE_KEY: copy.deepcopy(policy or _policy()),
        "exploratory": True,
    }


def test_real_p_waiver_requires_policy_and_keeps_measured_no_go() -> None:
    payload = _real_p_payload()
    stages._validate_physiology_gate(payload, REAL, request_policy=_policy())
    with pytest.raises(RuntimeError, match="explicit"):
        stages._validate_physiology_gate(payload, REAL, request_policy=None)
    with pytest.raises(RuntimeError, match="differs"):
        stages._validate_physiology_gate(payload, REAL, request_policy=_policy("x"))

    for tamper in ("no_policy", "pass_forged", "status_rewritten", "synthetic", "no_override"):
        bad = _real_p_payload()
        gate = bad["physiology_support_gate"]
        if tamper == "no_policy":
            bad.pop(WAIVER_STATE_KEY)
            bad.pop("exploratory")
        elif tamper == "pass_forged":
            gate["passed"] = True
        elif tamper == "status_rewritten":
            gate["status"] = "PASS_POSITIVE_U_SUPPORT"
        elif tamper == "synthetic":
            bad["synthetic_gate_override_used"] = True
        elif tamper == "no_override":
            gate["override_used"] = False
        with pytest.raises((RuntimeError, ValueError)):
            stages._validate_physiology_gate(bad, REAL, request_policy=_policy())

    strict = {"physiology_support_gate": {"status": "NO_GO_U_ESS_GATE", "passed": False,
                                          "override_used": False, "override_scope": None}}
    with pytest.raises(RuntimeError, match="physiology gate failed"):
        stages._validate_physiology_gate(strict, REAL, request_policy=_policy())


def test_manifest_content_hash_seals_the_checkpoint_waiver() -> None:
    policy = _policy()
    sealed = {"exploratory_quality_waiver_sha256": stages._canonical_sha256(policy)}
    stages._validate_waiver_content_hash("Q", sealed, policy)
    stages._validate_waiver_content_hash("Q", {}, None)
    with pytest.raises(ValueError, match="does not seal"):
        stages._validate_waiver_content_hash("Q", {}, policy)
    with pytest.raises(ValueError, match="does not seal"):
        stages._validate_waiver_content_hash("Q", sealed, _policy("other"))
    with pytest.raises(ValueError, match="does not carry"):
        stages._validate_waiver_content_hash("P", sealed, None)


# --------------------------------------------------------------------------
# Student export / cold load (real path; no synthetic branch exists here)
# --------------------------------------------------------------------------


def _student_config() -> dict:
    return {
        "schema": "q3d.config.v1",
        "arm": "V1",
        "model": {"student": {
            "channels": [4, 4, 4, 4, 4], "in_channels": 1, "num_classes": 4,
            "attention": {"support": "uniform"},
            "vq": {"size": 4, "dim": 4, "tau": 0.1},
            "residual_vq": {"enabled": False},
        }},
        "tokenizer": {"codebook_size": 4, "codebook_dim": 4, "tau": 0.1},
    }


def _codebook_metadata() -> dict:
    codes = torch.eye(4, dtype=torch.float32).contiguous()
    return {
        "codes": codes, "version": "q3d_teacher_codebook.v1",
        "sha256": hashlib.sha256(codes.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "size": 4, "dimension": 4, "initialized": True, "frozen": True,
    }


def _write_checkpoint(path: Path, *, policy: dict | None, tamper: str | None = None) -> Path:
    config = _student_config()
    codebook = _codebook_metadata()
    model = CandidateQ3DStudent(config, codebook=codebook).to(dtype=torch.float32).eval()
    state = {key: (dict(value) if key == "_extra_state" else value.detach().cpu().contiguous())
             for key, value in model.state_dict().items()}
    meta = {key: codebook[key] for key in ("version", "sha256", "size", "dimension", "initialized", "frozen")}
    payload: dict = {"config": config, "student_state": state}
    if policy is None:
        payload.update({"gate": {"status": "PASS", "passed": True}, "tokenizer_artifact": meta})
    else:
        q = _real_q_payload(policy)
        q["tokenizer_artifact"].update(meta)
        payload.update(q)
        payload["physiology_support_gate"] = _real_p_payload(policy)["physiology_support_gate"]
        payload["physiology_support_gate"]["patient_weights"] = {"P001": 0.2}
        payload["physiology_support_gates"] = [copy.deepcopy(payload["physiology_support_gate"])]
        payload["j_kd_supported_branches"] = 0
        payload["exploratory_zero_distillation"] = True
        payload[WAIVER_EVENTS_KEY] = [{"event": "tokenizer_gate_waived", "exploratory": True,
                                       "waiver_reason": policy["reason"]}]
    if tamper == "forged_pass":
        payload["gate"] = {"status": "PASS", "passed": True}
    elif tamper == "synthetic":
        payload["synthetic_gate_override_used"] = True
    elif tamper == "metadata_policy_without_state":
        payload.pop(WAIVER_STATE_KEY)
        payload.pop("exploratory")
    elif tamper == "metadata_gate_rewritten":
        payload["tokenizer_artifact"]["gate"] = {"status": "PASS", "passed": True}
    torch.save(payload, path)
    return path


def test_waived_export_carries_marker_and_measured_failed_gates(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "waived.pt", policy=_policy())
    with pytest.raises(RuntimeError, match="requires the same explicit"):
        export_checkpoint(checkpoint, tmp_path / "a.pt", code_commit="test")
    with pytest.raises(RuntimeError, match="differs"):
        export_checkpoint(checkpoint, tmp_path / "a.pt", code_commit="test",
                          exploratory_quality_waiver="wrong reason")

    artifact_path, manifest, _model = export_checkpoint(
        checkpoint, tmp_path / "student.pt", code_commit="test",
        exploratory_quality_waiver=REASON,
    )
    assert manifest["exploratory"] is True
    assert manifest[WAIVER_STATE_KEY] == _policy()
    measured = manifest["measured_quality_gates"]
    assert measured["tokenizer_gate"]["status"] == "NO_GO"
    assert measured["tokenizer_gate"]["passed"] is False
    assert measured["physiology_support_gate"]["status"] == "NO_GO_U_ESS_GATE"
    assert "patient_weights" not in measured["physiology_support_gate"]
    assert measured["kd_support"] == {"j_kd_supported_branches": 0, "exploratory_zero_distillation": True}
    assert manifest["exploratory_quality_waiver_events"][0]["event"] == "tokenizer_gate_waived"
    assert manifest["software_only"] is True

    raw = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert set(raw) == {"schema", "config", "state_dict", "metadata"}
    assert raw["schema"] == EXPORT_SCHEMA
    assert raw["metadata"]["exploratory"] is True
    assert raw["metadata"]["measured_quality_gates"]["tokenizer_gate"]["passed"] is False

    with pytest.raises(RuntimeError, match="requires the same explicit"):
        load_export(artifact_path)
    with pytest.raises(RuntimeError, match="differs"):
        load_export(artifact_path, exploratory_quality_waiver="wrong reason")
    _loaded, metadata = load_export(
        artifact_path, expected_sha256=manifest["artifact_sha256"],
        exploratory_quality_waiver=REASON,
    )
    assert metadata["exploratory"] is True
    assert metadata[WAIVER_STATE_KEY]["reason"] == REASON


def test_default_export_is_unchanged_and_rejects_unneeded_waiver(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "strict.pt", policy=None)
    artifact_path, manifest, _model = export_checkpoint(checkpoint, tmp_path / "s.pt", code_commit="test")
    assert "exploratory" not in manifest and "measured_quality_gates" not in manifest
    raw = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert "exploratory" not in raw["metadata"]
    load_export(artifact_path)
    with pytest.raises(ValueError, match="not exploratory"):
        load_export(artifact_path, exploratory_quality_waiver=REASON)
    with pytest.raises(ValueError, match="not exploratory"):
        export_checkpoint(checkpoint, tmp_path / "s2.pt", code_commit="test",
                          exploratory_quality_waiver=REASON)


@pytest.mark.parametrize("tamper", [
    "forged_pass", "synthetic", "metadata_policy_without_state", "metadata_gate_rewritten",
])
def test_export_rejects_forged_pass_or_synthetic_waiver(tmp_path: Path, tamper: str) -> None:
    checkpoint = _write_checkpoint(tmp_path / "bad.pt", policy=_policy(), tamper=tamper)
    with pytest.raises(ValueError):
        export_checkpoint(checkpoint, tmp_path / "bad-out.pt", code_commit="test",
                          exploratory_quality_waiver=REASON)


def test_cold_load_rejects_tampered_exploratory_metadata(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "waived.pt", policy=_policy())
    artifact_path, _manifest, _model = export_checkpoint(
        checkpoint, tmp_path / "student.pt", code_commit="test", exploratory_quality_waiver=REASON,
    )
    raw = torch.load(artifact_path, map_location="cpu", weights_only=True)
    stripped = copy.deepcopy(raw)
    stripped["metadata"].pop(WAIVER_STATE_KEY)
    torch.save(stripped, tmp_path / "stripped.pt")
    with pytest.raises(ValueError, match="without its waiver policy"):
        load_export(tmp_path / "stripped.pt")
    unmarked = copy.deepcopy(raw)
    unmarked["metadata"]["exploratory"] = False
    torch.save(unmarked, tmp_path / "unmarked.pt")
    with pytest.raises(ValueError, match="exploratory=True"):
        load_export(tmp_path / "unmarked.pt", exploratory_quality_waiver=REASON)


# --------------------------------------------------------------------------
# End-to-end stage artifacts + engine checkpoint export (synthetic fixture)
# --------------------------------------------------------------------------


def test_waived_stage_artifacts_and_engine_export_end_to_end(tmp_path: Path) -> None:
    import self_audit.candidate_q3d.engine as engine
    from self_audit.candidate_q3d.config import validate_config
    from tests.test_q3d_stage_artifacts import _synthetic_config_and_fixture
    from train import _manufactured_teacher

    if "exploratory_quality_waiver" not in engine.run_training.__code__.co_varnames:
        pytest.skip("engine waiver API is not integrated in this checkout")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config, _fixture = _synthetic_config_and_fixture(tmp_path)
        config["runtime"]["synthetic_gate_override"] = False
        config = validate_config(config)
        result = engine.run_training(
            config, tmp_path / "run", device="cpu", smoke=True,
            teacher=_manufactured_teacher(config), exploratory_quality_waiver=REASON,
        )
        assert result["status"] == "COMPLETED"
        assert result["tokenizer_gate"]["passed"] is False
        assert result["synthetic_gate_override_used"] is False
        refs = result["stage_artifacts"]
        assert "exploratory" not in refs["T"]
        for stage in ("Q", "P"):
            assert refs[stage]["exploratory"] is True
            assert refs[stage][WAIVER_STATE_KEY] == _policy()
            manifest = json.loads(Path(refs[stage]["manifest_path"]).read_text(encoding="utf-8"))
            assert manifest["content_hashes"]["exploratory_quality_waiver_sha256"] == \
                stages._canonical_sha256(_policy())
            # The waiver is never part of the sealed source config.
            assert manifest["source_config"] == stages.stage_source_config(config, stage)
            assert WAIVER_STATE_KEY not in json.dumps(manifest["source_config"])
        q_payload = torch.load(refs["Q"]["checkpoint_path"], map_location="cpu", weights_only=False)
        assert q_payload["gate"]["passed"] is False
        assert q_payload["tokenizer_artifact"]["gate"] == q_payload["gate"]
        assert q_payload["tokenizer_artifact"][WAIVER_STATE_KEY] == _policy()
        assert q_payload["tokenizer_artifact"]["frozen"] is True

        artifacts = {stage: refs[stage] for stage in ("T", "Q", "P")}
        teacher = _manufactured_teacher(config)
        with pytest.raises(RuntimeError, match="explicit --q3d-exploratory-quality-waiver"):
            engine.run_training(config, tmp_path / "import-strict", device="cpu", smoke=True,
                                teacher=teacher, stage="J", stage_artifacts=artifacts)
        with pytest.raises(RuntimeError, match="differs"):
            engine.run_training(config, tmp_path / "import-wrong", device="cpu", smoke=True,
                                teacher=_manufactured_teacher(config), stage="J",
                                stage_artifacts=artifacts, exploratory_quality_waiver="wrong")
        imported = engine.run_training(
            config, tmp_path / "import-waived", device="cpu", smoke=True,
            teacher=_manufactured_teacher(config), stage="J", stage_artifacts=artifacts,
            exploratory_quality_waiver=REASON,
        )
        assert imported["status"] == "COMPLETED"
        assert imported["stage_artifacts"]["Q"]["exploratory"] is True

        checkpoint = Path(result["checkpoint"])
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with pytest.raises(RuntimeError, match="requires the same explicit"):
            export_checkpoint(checkpoint, tmp_path / "e2e-strict.pt",
                              expected_checkpoint_sha256=checkpoint_sha,
                              config_path=config_path, code_commit="test")
        artifact_path, manifest, _model = export_checkpoint(
            checkpoint, tmp_path / "e2e.pt", expected_checkpoint_sha256=checkpoint_sha,
            config_path=config_path, code_commit="test", exploratory_quality_waiver=REASON,
        )
        assert manifest["engine_config_sha256"] == result["config_sha256"]
        assert manifest["exploratory"] is True
        assert manifest["measured_quality_gates"]["tokenizer_gate"]["passed"] is False
        load_export(artifact_path, exploratory_quality_waiver=REASON)
        with pytest.raises(RuntimeError):
            load_export(artifact_path)
    finally:
        torch.set_num_threads(previous_threads)


def test_exploratory_consumer_clis_preserve_policy_and_failed_gates(tmp_path: Path) -> None:
    """Exercise all student consumer CLIs on real artifact bytes, using CPU fixtures."""
    from scripts import q3d_export, q3d_infer, q3d_profile

    checkpoint = _write_checkpoint(tmp_path / "checkpoint.pt", policy=_policy())
    artifact = tmp_path / "student.pt"
    volume = tmp_path / "volume.npz"
    shape = (2, 16, 16)
    np.savez(volume, image=np.zeros((1, 1, *shape), dtype=np.float32),
             shape_zyx=np.asarray(shape), spacing_zyx=np.ones(3),
             affine_zyx=np.eye(4), context=np.asarray("full_volume"))
    waiver = ["--q3d-exploratory-quality-waiver", REASON]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        export_args = ["--checkpoint", str(checkpoint), "--output", str(artifact),
                       "--code-commit", "fixture"]
        with pytest.raises(RuntimeError, match="explicit"):
            q3d_export.main(export_args)
        assert q3d_export.main(export_args + waiver) == 0
        inference = tmp_path / "prediction.npz"
        infer_args = ["--artifact", str(artifact), "--input", str(volume),
                      "--output", str(inference)]
        with pytest.raises(RuntimeError, match="explicit"):
            q3d_infer.main(infer_args)
        assert q3d_infer.main(infer_args + waiver) == 0
        with np.load(inference, allow_pickle=False) as prediction:
            metadata = json.loads(str(prediction["exploratory_metadata"].item()))
            assert metadata["exploratory"] is True
            assert metadata[WAIVER_STATE_KEY] == _policy()
            assert metadata["measured_quality_gates"]["tokenizer_gate"]["passed"] is False
        profile = tmp_path / "profile.json"
        profile_args = ["--artifact", str(artifact), "--output", str(profile),
                        "--input", str(volume), "--warmup", "0", "--repeats", "1"]
        with pytest.raises(RuntimeError, match="explicit"):
            q3d_profile.main(profile_args)
        assert q3d_profile.main(profile_args + waiver) == 0
        report = json.loads(profile.read_text())
        assert report["exploratory"] is True
        assert report[WAIVER_STATE_KEY] == _policy()
        assert report["measured_quality_gates"]["tokenizer_gate"]["status"] == "NO_GO"
        # Finalizing an already profiled export must forward the reason both to
        # load_export and the fresh-process cold-load verifier.
        assert q3d_export.main(export_args + waiver + ["--cost-report", str(profile)]
                               + [arg for _ in range(3) for arg in ("--verify-volume", str(volume))]) == 0
        manifest = json.loads(artifact.with_suffix(".pt.manifest.json").read_text())
        assert manifest["cold_load"]["status"] == "PASSED_BITWISE_THREE_NATIVE_VOLUMES"
        assert manifest["cold_load"]["exploratory"] is True
    finally:
        torch.set_num_threads(previous_threads)


def test_evaluate_cli_requires_reason_and_persists_exploratory_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    import sys
    import evaluate
    from self_audit.candidate_q3d import evaluation

    checkpoint = _write_checkpoint(tmp_path / "checkpoint.pt", policy=_policy())
    artifact, manifest, _ = export_checkpoint(
        checkpoint, tmp_path / "student.pt", code_commit="fixture",
        exploratory_quality_waiver=REASON,
    )
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"schema": "self_audit.data.v2", "role": "training",
                                    "labeled": [{"id": "TRAIN1"}], "unlabeled": []}))
    output = tmp_path / "evaluation"
    calls = []

    def fake_evaluate(model, data, sha, out, **kwargs):
        calls.append(kwargs)
        Path(out).mkdir()
        return {"patient_macro": {"fg_dice": 0.5}, "evidence_status": "FIXTURE_ONLY"}

    monkeypatch.setattr(evaluation, "evaluate_manifest", fake_evaluate)
    args = ["evaluate.py", "--checkpoint", str(artifact), "--checkpoint-sha256",
            manifest["artifact_sha256"], "--data", "fixture-only", "--data-sha256", "a" * 64,
            "--out", str(output), "--device", "cpu", "--train-data", str(training),
            "--train-data-sha256", hashlib.sha256(training.read_bytes()).hexdigest()]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(RuntimeError, match="explicit"):
        evaluate.main()
    assert calls == [] and not output.exists()
    monkeypatch.setattr(sys, "argv", args + ["--q3d-exploratory-quality-waiver", REASON])
    evaluate.main()
    assert calls[0]["train_patient_ids"] == ("TRAIN1",)
    report = json.loads((output / "Q3D_METRICS.json").read_text())
    assert report["evidence_status"] == "FIXTURE_ONLY"
    assert report[WAIVER_STATE_KEY] == _policy()
    assert report["measured_quality_gates"]["tokenizer_gate"]["passed"] is False
    assert report["exploratory"] is True
    assert json.loads(capsys.readouterr().out)["exploratory"] is True
