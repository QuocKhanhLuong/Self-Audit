"""Tests for exact SAME-RUN resume, atomic checkpoints, and weight-change verification."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch

from self_audit.data import TrainingData, synthetic_fixture, digest, write_json
from self_audit.engine import DEFAULT, train, load_checkpoint, save_checkpoint, validate_config, source_digest
from self_audit.model import build_model


def resume_recipe(method="ema_full", scope="head", steps=6, warmup_steps=2, save_every=2):
    c = copy.deepcopy(DEFAULT)
    c.update(method=method, update_scope=scope, steps=steps, warmup_steps=warmup_steps,
             save_every=save_every, pseudo_threshold=0.25, ema_decay=0.9,
             body_lr=0.001, head_lr=0.03, seed=42)
    c["model"]["backbone"] = "tiny_test"
    return c


def test_exact_cpu_uninterrupted_vs_resumed_parity_crossing_warmup(tmp_path):
    """Test bit-exact parity between uninterrupted and interrupted-then-resumed runs crossing warmup."""
    manifest, sha = synthetic_fixture(tmp_path / "data")
    c = resume_recipe(steps=6, warmup_steps=2, save_every=2)

    # 1. Uninterrupted run
    out_a = tmp_path / "run_uninterrupted"
    report_a = train(c, manifest, sha, out_a, device="cpu", synthetic=True)
    assert report_a["status"] == "COMPLETED"
    assert report_a["steps"] == 6

    # 2. Interrupted run stopped after step 3 (crosses warmup at step 2)
    out_b = tmp_path / "run_resumed"
    report_b_stop = train(c, manifest, sha, out_b, device="cpu", synthetic=True, stop_after_step=3)
    assert report_b_stop["status"] == "INTERRUPTED"
    assert report_b_stop["step"] == 3
    step3_dir = out_b / "step_000003"
    assert step3_dir.is_dir()
    step3_sha = digest(step3_dir / "MANIFEST.json")
    step3_manifest = json.loads((step3_dir / "MANIFEST.json").read_text())
    assert step3_manifest["resume"] == "SUPPORTED"
    assert step3_manifest["lr_scheduler"] is None
    assert step3_manifest["scaler"] is None

    # 3. Resume run B from step 3 to step 6 in the same output directory
    report_b_final = train(c, manifest, sha, out_b, device="cpu", synthetic=True,
                           resume=str(step3_dir), resume_sha256=step3_sha)
    assert report_b_final["status"] == "COMPLETED"
    assert report_b_final["steps"] == 6

    # 4. Compare reports
    # Timing/provenance hashes differ; compare actual trajectory state below.
    assert report_a["supervised_updates"] == report_b_final["supervised_updates"]
    assert report_a["pseudo_accepted"] == report_b_final["pseudo_accepted"]
    assert report_a["ema_updates"] == report_b_final["ema_updates"]
    assert report_a["seen_supervised"] == report_b_final["seen_supervised"]
    assert report_a["seen_audit"] == report_b_final["seen_audit"]
    assert report_a["seen_unlabeled"] == report_b_final["seen_unlabeled"]
    assert report_a["label_reads_outside_budget"] == report_b_final["label_reads_outside_budget"]

    # 5. Compare student and teacher weights
    payload_a = torch.load(out_a / "final" / "weights.pt", weights_only=True)
    payload_b = torch.load(out_b / "final" / "weights.pt", weights_only=True)
    for k in payload_a["student"]:
        assert torch.equal(payload_a["student"][k], payload_b["student"][k]), f"Student weight mismatch at {k}"
    for k in payload_a["teacher"]:
        assert torch.equal(payload_a["teacher"][k], payload_b["teacher"][k]), f"Teacher weight mismatch at {k}"

    assert torch.equal(payload_a["torch_rng"], payload_b["torch_rng"])
    assert payload_a["py_rng"] == payload_b["py_rng"]
    assert len(payload_a["cuda_rng"]) == len(payload_b["cuda_rng"])
    for a, b in zip(payload_a["cuda_rng"], payload_b["cuda_rng"]):
        assert torch.equal(a, b)
    ma = json.loads((out_a / "final/MANIFEST.json").read_text())
    mb = json.loads((out_b / "final/MANIFEST.json").read_text())
    assert ma["numpy_rng"] == mb["numpy_rng"]
    assert ma["numpy_global_rng"] == mb["numpy_global_rng"]
    assert mb["elapsed_seconds"] > step3_manifest["elapsed_seconds"]

    # 6. Compare optimizer states
    opts_a = payload_a["optimizers"]
    opts_b = payload_b["optimizers"]
    assert len(opts_a) == len(opts_b)
    for i, (oa, ob) in enumerate(zip(opts_a, opts_b)):
        assert len(oa["state"]) == len(ob["state"])
        for pid in oa["state"]:
            sa = oa["state"][pid]
            sb = ob["state"][pid]
            for sk in sa:
                if isinstance(sa[sk], torch.Tensor):
                    assert torch.equal(sa[sk], sb[sk]), f"Optimizer {i} state tensor mismatch at {sk}"
                else:
                    assert sa[sk] == sb[sk], f"Optimizer {i} state scalar mismatch at {sk}"

    # 7. Compare logs
    lines_a = [json.loads(line) for line in (out_a / "steps.jsonl").read_text().splitlines() if line.strip()]
    lines_b = [json.loads(line) for line in (out_b / "steps.jsonl").read_text().splitlines() if line.strip()]
    assert len(lines_a) == 6
    assert len(lines_b) == 6
    for ra, rb in zip(lines_a, lines_b):
        assert ra["step"] == rb["step"]
        assert ra["supervised_loss"] == pytest.approx(rb["supervised_loss"], abs=1e-12)
        assert ra["gradient_norm"] == pytest.approx(rb["gradient_norm"], abs=1e-12)
        if "pseudo_loss" in ra:
            assert ra["pseudo_loss"] == pytest.approx(rb["pseudo_loss"], abs=1e-12)
            assert ra["pseudo_gradient_norm"] == pytest.approx(rb["pseudo_gradient_norm"], abs=1e-12)
        assert ra["supervised"] == rb["supervised"]
        assert ra["audit"] == rb["audit"]
        assert ra["unlabeled"] == rb["unlabeled"]


def test_verify_encoder_detail_fuse_head_weight_changes(tmp_path):
    """Verify actual encoder, detail, fuse, head weight and optimizer updates without extra training/RNG draws."""
    manifest, sha = synthetic_fixture(tmp_path / "data")
    c = resume_recipe(steps=1, warmup_steps=1, save_every=1)

    # Initial reference model
    torch.manual_seed(c["seed"])
    init_model, _ = build_model(c["model"], synthetic=True)
    init_sd = {k: v.clone() for k, v in init_model.state_dict().items()}

    out = tmp_path / "run_weights"
    train(c, manifest, sha, out, device="cpu", synthetic=True)

    final_payload = torch.load(out / "final" / "weights.pt", weights_only=True)
    final_sd = final_payload["student"]

    # Verify every partition actually changed from initial state
    partitions = {
        "encoder": [k for k in final_sd if k.startswith("encoder.") and "weight" in k],
        "decoder": [k for k in final_sd if k.startswith(("lateral.", "up.", "refine.")) and "weight" in k],
        "detail": [k for k in final_sd if k.startswith("detail.") and "weight" in k],
        "fuse": [k for k in final_sd if k.startswith("fuse.") and "weight" in k],
        "head": [k for k in final_sd if k.startswith("head.") and "weight" in k],
    }

    for name, keys in partitions.items():
        assert len(keys) > 0, f"Expected parameter keys for {name}"
        total_diff = sum(torch.norm(final_sd[k] - init_sd[k]).item() for k in keys)
        assert total_diff > 0, f"Expected non-zero weight movement for {name}, got total diff {total_diff}"

    # Verify actual AdamW body state; head SGD intentionally has no momentum
    opts = final_payload["optimizers"]
    body_opt = opts[0]
    head_opt = opts[1]
    assert len(body_opt["state"]) > 0, "Body optimizer should have tracked states"
    for pstate in body_opt["state"].values():
        assert pstate["step"].item() > 0
        assert torch.isfinite(pstate["exp_avg"]).all()
        assert torch.isfinite(pstate["exp_avg_sq"]).all()
    # Single-Z manufactured data has zero adjacent-slice differences, hence
    # zero input-adapter moments are correct. Actual learned blocks move above.
    assert sum(v['exp_avg'].abs().sum().item() for v in body_opt['state'].values()) > 0
    assert head_opt['state'] == {}


def test_crash_consistent_log_recovery(tmp_path):
    """Test log truncation and append recovery after a simulated mid-step crash."""
    manifest, sha = synthetic_fixture(tmp_path / "data")
    c = resume_recipe(steps=4, warmup_steps=1, save_every=2)

    out = tmp_path / "run_crash"
    # Run to step 2 and stop
    train(c, manifest, sha, out, device="cpu", synthetic=True, stop_after_step=2)
    step2_dir = out / "step_000002"
    step2_sha = digest(step2_dir / "MANIFEST.json")

    # Simulate crash artifacts: trailing corrupt/duplicate step in steps.jsonl and a FAILURE.json
    steps_file = out / "steps.jsonl"
    corrupt_line = json.dumps({"step": 3, "status": "crashed_halfway"}) + "\n"
    with steps_file.open("a") as f:
        f.write(corrupt_line)
    (out / "FAILURE.json").write_text(json.dumps({"status": "FAILED", "error": "SimulatedCrash()"}))

    # Resume from step 2
    report = train(c, manifest, sha, out, device="cpu", synthetic=True,
                   resume=str(step2_dir), resume_sha256=step2_sha)
    assert report["status"] == "COMPLETED"
    assert report["steps"] == 4
    assert not (out / "FAILURE.json").exists(), "active failure marker should be archived on resume"
    assert len(list(out.glob("FAILURE.before_resume_*.json"))) == 1
    assert len(list(out.glob("steps.before_resume_*.jsonl"))) == 1

    rows = [json.loads(line) for line in steps_file.read_text().splitlines() if line.strip()]
    assert [r["step"] for r in rows] == [1, 2, 3, 4], "Log must be truncated back to saved step and cleanly appended"


def test_reject_not_supported_old_checkpoint(tmp_path):
    """Reject checkpoints marked with resume: NOT_SUPPORTED."""
    manifest, sha = synthetic_fixture(tmp_path / "data")
    c = resume_recipe(steps=2, warmup_steps=1, save_every=1)
    out = tmp_path / "run_old"
    train(c, manifest, sha, out, device="cpu", synthetic=True, stop_after_step=1)
    step1_dir = out / "step_000001"
    mp = step1_dir / "MANIFEST.json"
    m = json.loads(mp.read_text())
    m["resume"] = "NOT_SUPPORTED"
    write_json(mp, m)
    new_sha = digest(mp)

    with pytest.raises(ValueError, match="NOT_SUPPORTED"):
        train(c, manifest, sha, out, device="cpu", synthetic=True,
              resume=str(step1_dir), resume_sha256=new_sha)


def test_mismatched_resume_rejections(tmp_path):
    """Strict rejection of tampered manifest, payload, recipe, data, and output directory."""
    manifest, sha = synthetic_fixture(tmp_path / "data")
    c = resume_recipe(steps=4, warmup_steps=1, save_every=2)
    out = tmp_path / "run_mismatch"
    train(c, manifest, sha, out, device="cpu", synthetic=True, stop_after_step=2)
    step2_dir = out / "step_000002"
    step2_sha = digest(step2_dir / "MANIFEST.json")

    # 1. Manifest hash mismatch
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        train(c, manifest, sha, out, device="cpu", synthetic=True,
              resume=str(step2_dir), resume_sha256="0" * 64)

    # 2. Tampered weights.pt
    corrupt_out = tmp_path / "corrupt_run"
    corrupt_step = corrupt_out / "step_000002"
    import shutil
    shutil.copytree(step2_dir, corrupt_step)
    (corrupt_step / "weights.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        train(c, manifest, sha, corrupt_out, device="cpu", synthetic=True,
              resume=str(corrupt_step), resume_sha256=digest(corrupt_step / "MANIFEST.json"))

    # 3. Output directory mismatch
    other_out = tmp_path / "other_out"
    other_out.mkdir()
    with pytest.raises(ValueError, match="must reside within the specified --out directory"):
        train(c, manifest, sha, other_out, device="cpu", synthetic=True,
              resume=str(step2_dir), resume_sha256=step2_sha)

    # 4. Recipe mismatch (e.g. modified body_lr)
    c_modified = copy.deepcopy(c)
    c_modified["body_lr"] = 0.05
    with pytest.raises(ValueError, match="recipe config does not match"):
        train(c_modified, manifest, sha, out, device="cpu", synthetic=True,
              resume=str(step2_dir), resume_sha256=step2_sha)

    # 5. Data hash mismatch
    manifest2, _ = synthetic_fixture(tmp_path / "data2")
    m2_obj = json.loads(manifest2.read_text())
    m2_obj["labeled"][0]["id"] = "different_patient"
    write_json(manifest2, m2_obj)
    sha2 = digest(manifest2)
    with pytest.raises(ValueError, match="checkpoint data sha256 mismatch"):
        train(c, manifest2, sha2, out, device="cpu", synthetic=True,
              resume=str(step2_dir), resume_sha256=step2_sha)


def test_atomic_checkpoint_creation_and_legacy_load(tmp_path):
    """Test atomic checkpoint saving and legacy load_checkpoint raw-student compatibility."""
    manifest, sha = synthetic_fixture(tmp_path / "data")
    c = resume_recipe(steps=2, warmup_steps=1, save_every=1)
    out = tmp_path / "run_atomic"
    report = train(c, manifest, sha, out, device="cpu", synthetic=True)

    # Verify no temporary folders remain
    tmps = list(out.glob(".tmp_*"))
    assert len(tmps) == 0, f"Temporary directories lingered: {tmps}"

    # Verify load_checkpoint compatibility
    final_dir = out / "final"
    model, meta = load_checkpoint(final_dir, report["final_manifest_sha256"], device="cpu")
    assert meta["schema"] == "self_audit.checkpoint.v2"
    assert meta["prediction_model"] == "student"
    assert meta["resume"] == "SUPPORTED"
    assert meta["lr_scheduler"] is None
    assert meta["scaler"] is None
    dummy = torch.rand(1, 3, 32, 32)
    pred = model(dummy)
    assert pred.shape == (1, 4, 32, 32)
    assert torch.isfinite(pred).all()


def test_cli_train_resume(tmp_path):
    """Test CLI train.py execution with --stop-after-step and --resume/--resume-sha256."""
    manifest, sha = synthetic_fixture(tmp_path / "data")
    c = resume_recipe(steps=4, warmup_steps=1, save_every=2)
    cfg_path = tmp_path / "config.json"
    write_json(cfg_path, c)
    out = tmp_path / "run_cli"

    python_exe = sys.executable

    # Phase 1: stop after step 2
    cmd1 = [
        python_exe, "train.py",
        "--config", str(cfg_path),
        "--data", str(manifest),
        "--data-sha256", sha,
        "--out", str(out),
        "--device", "cpu",
        "--synthetic",
        "--stop-after-step", "2"
    ]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    assert res1.returncode == 0, f"CLI Phase 1 failed: {res1.stderr}"
    rep1 = json.loads(res1.stdout)
    assert rep1["status"] == "INTERRUPTED"
    assert rep1["step"] == 2

    step2_dir = out / "step_000002"
    step2_sha = rep1["checkpoint_manifest_sha256"]

    # Phase 2: resume to completion
    cmd2 = [
        python_exe, "train.py",
        "--config", str(cfg_path),
        "--data", str(manifest),
        "--data-sha256", sha,
        "--out", str(out),
        "--device", "cpu",
        "--synthetic",
        "--resume", str(step2_dir),
        "--resume-sha256", step2_sha
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert res2.returncode == 0, f"CLI Phase 2 failed: {res2.stderr}"
    rep2 = json.loads(res2.stdout)
    assert rep2["status"] == "COMPLETED"
    assert rep2["steps"] == 4
    assert (out / "final" / "MANIFEST.json").is_file()


@pytest.mark.parametrize('field', ['torch_rng', 'cuda_rng', 'py_rng', 'optimizers'])
def test_missing_payload_resume_state_fails_closed(tmp_path, field):
    manifest, sha = synthetic_fixture(tmp_path / 'data')
    c = resume_recipe(steps=3, warmup_steps=1, save_every=1)
    out = tmp_path / 'run'
    train(c, manifest, sha, out, device='cpu', synthetic=True, stop_after_step=1)
    checkpoint = out / 'step_000001'
    payload = torch.load(checkpoint / 'weights.pt', weights_only=True)
    del payload[field]
    torch.save(payload, checkpoint / 'weights.pt')
    meta = json.loads((checkpoint / 'MANIFEST.json').read_text())
    meta['payload_sha256'] = digest(checkpoint / 'weights.pt')
    write_json(checkpoint / 'MANIFEST.json', meta)
    before = (out / 'steps.jsonl').read_bytes()
    with pytest.raises(ValueError, match='missing required resume payload state'):
        train(c, manifest, sha, out, device='cpu', synthetic=True,
              resume=checkpoint, resume_sha256=digest(checkpoint / 'MANIFEST.json'))
    assert (out / 'steps.jsonl').read_bytes() == before


def test_checkpoint_overwrite_is_refused_without_deleting_payload(tmp_path):
    manifest, sha = synthetic_fixture(tmp_path / 'data')
    c = resume_recipe(steps=2, warmup_steps=1, save_every=1)
    out = tmp_path / 'run'
    train(c, manifest, sha, out, device='cpu', synthetic=True)
    target = out / 'final'
    before = digest(target / 'weights.pt')
    with pytest.raises(FileExistsError, match='checkpoint already exists'):
        save_checkpoint(target, None, None, [], c, None, 2, {}, None)
    assert digest(target / 'weights.pt') == before


def test_resume_older_checkpoint_refuses_trajectory_rollback(tmp_path):
    manifest, sha = synthetic_fixture(tmp_path / 'data')
    c = resume_recipe(steps=4, warmup_steps=1, save_every=1)
    out = tmp_path / 'run'
    train(c, manifest, sha, out, device='cpu', synthetic=True, stop_after_step=2)
    checkpoint = out / 'step_000001'
    with pytest.raises(ValueError, match='newer checkpoint exists'):
        train(c, manifest, sha, out, device='cpu', synthetic=True,
              resume=checkpoint, resume_sha256=digest(checkpoint / 'MANIFEST.json'))
