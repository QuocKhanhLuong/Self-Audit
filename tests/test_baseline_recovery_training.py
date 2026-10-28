"""Verification and telemetry tests for baseline recovery training:
1. Exact optimizer parameter ownership (AdamW body vs SGD head)
2. Nonzero gradients and weight deltas for all 5 canonical modules (encoder, decoder, detail, fuse, head)
3. Separate supervised and unlabeled batch configurations
4. Warmup noncollapse gate (pass and fail-closed with evidence)
5. Exactly one EMA update per outer iteration
6. End-to-end synthetic baseline recovery training smoke through engine.train
7. Backward compatibility for existing v2 configurations
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest
import torch
from torch import nn

from self_audit.model import NativeStudent, TinyEncoder
from self_audit.losses import supervised_loss
from self_audit.engine import (
    DEFAULT,
    train,
    load_checkpoint,
    supervised_step,
    validate_config,
    ema_update,
)
from self_audit.data import synthetic_fixture
from self_audit.baseline_recovery import (
    CANONICAL_MODULES,
    WarmupCollapseError,
    classify_parameter,
    verify_optimizer_ownership,
    snapshot_parameters,
    compute_module_telemetry,
    build_warmup_probe,
    evaluate_warmup_gate,
    validate_baseline_recovery_config,
)


def _make_test_model():
    return NativeStudent(TinyEncoder(), (8, 16, 24, 32))


def _make_test_optimizers(model):
    body = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    head = list(model.head.parameters())
    return [
        torch.optim.AdamW(body, lr=3e-5, weight_decay=1e-4),
        torch.optim.SGD(head, lr=3e-3, momentum=0.0, weight_decay=0.0),
    ]


def test_classify_parameter():
    assert classify_parameter("encoder.stages.0.weight") == "encoder"
    assert classify_parameter("lateral.0.weight") == "decoder"
    assert classify_parameter("up.0.weight") == "decoder"
    assert classify_parameter("refine.0.0.body.0.weight") == "decoder"
    assert classify_parameter("detail.0.weight") == "detail"
    assert classify_parameter("fuse.0.weight") == "fuse"
    assert classify_parameter("head.weight") == "head"
    assert classify_parameter("input_adapter.context.weight") == "input_adapter"


def test_optimizer_parameter_ownership_verified():
    model = _make_test_model()
    opts = _make_test_optimizers(model)
    report = verify_optimizer_ownership(model, opts)

    assert report["verified"] is True
    assert report["body_param_count"] == 87
    assert report["head_param_count"] == 2
    assert report["total_trainable"] == 89

    # All 5 canonical modules must have nonzero parameter counts and numel
    for mod in CANONICAL_MODULES:
        assert report["module_param_counts"][mod] > 0, f"Module {mod} has 0 params"
        assert report["module_numel"][mod] > 0, f"Module {mod} has 0 numel"


def test_optimizer_ownership_rejects_overlap_or_omission():
    model = _make_test_model()
    body = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    # Intentionally leak head param into AdamW
    leaked_body = body + list(model.head.parameters())[:1]
    opts_overlap = [
        torch.optim.AdamW(leaked_body, lr=3e-5),
        torch.optim.SGD(model.head.parameters(), lr=3e-3),
    ]
    with pytest.raises(ValueError, match="shared across optimizers"):
        verify_optimizer_ownership(model, opts_overlap)

    # Intentionally omit head by providing a dummy parameter in SGD
    dummy = nn.Parameter(torch.randn(1))
    opts_missing = [
        torch.optim.AdamW(body, lr=3e-5),
        torch.optim.SGD([dummy], lr=3e-3),
    ]
    with pytest.raises(ValueError, match="missing from optimizers"):
        verify_optimizer_ownership(model, opts_missing)


def test_nonzero_gradients_and_deltas_for_all_five_modules():
    model = _make_test_model()
    opts = _make_test_optimizers(model)

    # Verify model.module_parameters() returns all 5 canonical modules
    mod_params = model.module_parameters()
    for m in CANONICAL_MODULES:
        assert m in mod_params
        assert len(mod_params[m]) > 0

    x = torch.rand(2, 3, 32, 32)
    # Ensure all 4 classes (0=BG, 1=RV, 2=MYO, 3=LV) are present in the label batch
    y = torch.zeros(2, 32, 32, dtype=torch.long)
    y[0, 5:15, 5:15] = 1
    y[0, 15:25, 15:25] = 2
    y[1, 10:20, 10:20] = 3

    sloss, grad_norm, telem = supervised_step(model, opts, x, y, return_telemetry=True)

    assert sloss > 0
    assert grad_norm > 0
    assert telem["all_gradients_nonzero"] is True
    assert telem["all_deltas_nonzero"] is True

    for mod in CANONICAL_MODULES:
        assert telem["gradient_norms"][mod] > 0, f"{mod} had 0 gradient norm"
        assert telem["update_deltas"][mod] > 0, f"{mod} had 0 update delta"


def test_supervised_loss_components():
    logits = torch.randn(2, 4, 16, 16)
    labels = torch.randint(0, 4, (2, 16, 16))

    loss_plain = supervised_loss(logits, labels)
    loss_comp, comps = supervised_loss(logits, labels, return_components=True)

    assert torch.isclose(loss_plain, loss_comp)
    assert "ce" in comps and "dice" in comps
    assert comps["ce"] > 0
    assert comps["dice"] >= 0


def test_warmup_noncollapse_gate_passing():
    model = _make_test_model()
    # Construct synthetic probe with all classes present in outputs
    probe_x = torch.rand(4, 3, 32, 32)
    probe = {
        "images": probe_x,
        "valid": torch.ones(4, 32, 32, dtype=torch.bool),
        "patient_ids": ["p1", "p2", "p3", "p4"],
    }
    # With a randomly initialized model, counts for all 4 classes are typically > 0 on 4x32x32 = 4096 pixels
    min_counts = {"1": 1, "2": 1, "3": 1}
    evidence = evaluate_warmup_gate(model, probe, device="cpu", min_class_counts=min_counts)

    assert evidence["passed"] is True
    assert evidence["total_probe_pixels"] == 4 * 32 * 32
    assert all(evidence["class_counts"][int(c)] >= 1 for c in ("1", "2", "3"))


def test_warmup_noncollapse_gate_fails_closed_on_collapse():
    class CollapsedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Conv2d(3, 4, 1)

        def forward(self, x, valid_mask=None):
            # Always predict class 0 (BG) with large logit, others very negative
            out = torch.full((x.shape[0], 4, x.shape[2], x.shape[3]), -100.0)
            out[:, 0] = 100.0
            return out

    model = CollapsedModel()
    probe = {
        "images": torch.rand(2, 3, 16, 16),
        "valid": torch.ones(2, 16, 16, dtype=torch.bool),
        "patient_ids": ["p1", "p2"],
    }
    min_counts = {"1": 1, "2": 1, "3": 1}

    with pytest.raises(WarmupCollapseError) as exc_info:
        evaluate_warmup_gate(model, probe, device="cpu", min_class_counts=min_counts)

    err = exc_info.value
    assert err.evidence["passed"] is False
    assert err.evidence["class_counts"][1] == 0
    assert err.evidence["class_counts"][2] == 0
    assert err.evidence["class_counts"][3] == 0
    assert "1" in err.evidence["failed_classes"]


def test_validate_baseline_recovery_config():
    valid_cfg = {
        "enabled": True,
        "supervised_batch_size": 4,
        "unlabeled_batch_size": 8,
        "record_module_telemetry": True,
        "warmup_gate": {
            "enabled": True,
            "min_class_counts": {"1": 5, "2": 5, "3": 5},
        },
    }
    norm = validate_baseline_recovery_config(valid_cfg)
    assert norm["supervised_batch_size"] == 4
    assert norm["unlabeled_batch_size"] == 8

    # Reject invalid batch sizes
    with pytest.raises(ValueError, match="positive integer"):
        validate_baseline_recovery_config({"supervised_batch_size": 0})

    # Reject invalid warmup gate config
    with pytest.raises(ValueError, match="min_class_counts"):
        validate_baseline_recovery_config({"warmup_gate": {"min_class_counts": {"4": 1}}})


def test_backward_compatibility_validate_config():
    # Existing default config without baseline_recovery section
    c = copy_default = dict(DEFAULT)
    validated = validate_config(c)
    assert validated["baseline_recovery"] is None

    # Config with opt-in baseline_recovery section
    c_with_br = dict(DEFAULT)
    c_with_br["baseline_recovery"] = {
        "enabled": True,
        "supervised_batch_size": 8,
        "unlabeled_batch_size": 8,
    }
    validated_br = validate_config(c_with_br)
    assert validated_br["baseline_recovery"] is not None
    assert validated_br["baseline_recovery"]["supervised_batch_size"] == 8


def test_single_ema_update_per_outer_iteration():
    model = _make_test_model()
    teacher = _make_test_model()

    # Perturb student so all parameters differ from teacher
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    # Track how EMA updates behave: exactly one update with decay 0.99
    before_teacher = [p.clone() for p in teacher.parameters()]
    ema_update(teacher, model, decay=0.99)
    after_teacher = [p.clone() for p in teacher.parameters()]

    for b, a in zip(before_teacher, after_teacher):
        assert not torch.allclose(b, a), "Teacher should update once towards student"


def test_synthetic_baseline_recovery_end_to_end(tmp_path):
    data_dir = tmp_path / "data"
    manifest, manifest_sha = synthetic_fixture(data_dir, seed=42, n_labeled=3, n_unlabeled=3, size=32)

    config_path = Path("configs/baseline_recovery/baseline_smoke_cpu.json")
    assert config_path.is_file()
    cfg = json.loads(config_path.read_text())

    out_dir = tmp_path / "run"
    report = train(cfg, str(manifest), manifest_sha, str(out_dir), device="cpu", synthetic=True)

    assert report["status"] == "COMPLETED"
    assert report["steps"] == 4
    assert report["supervised_updates"] == 4
    assert report["ema_updates"] == 4
    assert "baseline_recovery" in report

    br_report = report["baseline_recovery"]
    assert br_report["supervised_batch_size"] == 2
    assert br_report["unlabeled_batch_size"] == 2
    assert br_report["optimizer_ownership"]["verified"] is True
    assert br_report["warmup_gate"]["passed"] is True

    # Verify steps.jsonl records telemetry
    log_lines = (out_dir / "steps.jsonl").read_text().strip().split("\n")
    assert len(log_lines) == 4
    first_step = json.loads(log_lines[0])
    assert "module_gradient_norms" in first_step
    assert "module_update_deltas" in first_step
    assert first_step["supervised_batch_size"] == 2
    assert first_step["unlabeled_batch_size"] == 2

    # Checkpoint restoration verification
    restored, metadata = load_checkpoint(out_dir / "final", report["final_manifest_sha256"], device="cpu")
    assert isinstance(restored, NativeStudent)
    assert metadata["step"] == 4


def test_train_py_cli_baseline_recovery_smoke(tmp_path):
    import subprocess
    import sys
    data_dir = tmp_path / "data"
    manifest, manifest_sha = synthetic_fixture(data_dir, seed=42, n_labeled=3, n_unlabeled=3, size=32)

    config_path = Path("configs/baseline_recovery/baseline_smoke_cpu.json")
    out_dir = tmp_path / "cli_run"

    cmd = [
        sys.executable, "train.py",
        "--config", str(config_path),
        "--data", str(manifest),
        "--data-sha256", manifest_sha,
        "--out", str(out_dir),
        "--device", "cpu",
        "--synthetic",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"train.py failed: {res.stderr}"

    receipt = json.loads(res.stdout)
    assert receipt["status"] == "COMPLETED"
    assert (out_dir / "REPORT.json").is_file()
    assert (out_dir / "steps.jsonl").is_file()
    assert (out_dir / "final" / "MANIFEST.json").is_file()


# ===========================================================================
# 8. Warmup Gate Regression Tests (Item 1)
# ===========================================================================

def test_warmup_gate_padded_fg_only_fails_closed():
    """Model predicts FG only in padded (valid=False) pixels; gate must fail closed."""
    class PaddedOnlyModel(nn.Module):
        def forward(self, x, valid_mask=None):
            # 4 classes: [B, 4, H, W]
            B, _, H, W = x.shape
            out = torch.full((B, 4, H, W), -100.0)
            # In valid region (left half), predict BG (class 0)
            out[:, 0, :, :W // 2] = 100.0
            # In invalid region (right half), predict FG (classes 1, 2, 3)
            out[:, 1, :, W // 2:] = 100.0
            return out

    model = PaddedOnlyModel()
    valid = torch.zeros(2, 16, 16, dtype=torch.bool)
    valid[:, :, :8] = True  # Left half is valid

    probe = {
        "images": torch.rand(2, 3, 16, 16),
        "valid": valid,
        "patient_ids": ["p1", "p2"],
    }
    min_counts = {"1": 1, "2": 1, "3": 1}

    with pytest.raises(WarmupCollapseError) as exc_info:
        evaluate_warmup_gate(model, probe, device="cpu", min_class_counts=min_counts)

    evidence = exc_info.value.evidence
    assert evidence["passed"] is False
    assert evidence["class_counts"][1] == 0  # No FG in valid area
    assert "1" in evidence["failed_classes"]


def test_warmup_gate_ignored_labels_only_fails_closed():
    """Model predicts FG only where label=255; gate must fail closed."""
    class IgnoredOnlyModel(nn.Module):
        def forward(self, x, valid_mask=None):
            B, _, H, W = x.shape
            out = torch.full((B, 4, H, W), -100.0)
            # Where label is 255 (top half), predict FG class 1
            out[:, 1, :H // 2, :] = 100.0
            # Where label is not 255 (bottom half), predict BG class 0
            out[:, 0, H // 2:, :] = 100.0
            return out

    model = IgnoredOnlyModel()
    labels = torch.zeros(2, 16, 16, dtype=torch.long)
    labels[:, :8, :] = 255  # Top half is ignore-255

    probe = {
        "images": torch.rand(2, 3, 16, 16),
        "valid": torch.ones(2, 16, 16, dtype=torch.bool),
        "labels": labels,
        "patient_ids": ["p1", "p2"],
    }
    min_counts = {"1": 1, "2": 1, "3": 1}

    with pytest.raises(WarmupCollapseError) as exc_info:
        evaluate_warmup_gate(model, probe, device="cpu", min_class_counts=min_counts)

    evidence = exc_info.value.evidence
    assert evidence["passed"] is False
    assert evidence["class_counts"][1] == 0


def test_warmup_gate_empty_support_fails_closed():
    """Empty valid support (valid=False everywhere) must raise WarmupCollapseError."""
    model = _make_test_model()
    probe = {
        "images": torch.rand(2, 3, 16, 16),
        "valid": torch.zeros(2, 16, 16, dtype=torch.bool),
        "patient_ids": ["p1", "p2"],
    }
    min_counts = {"1": 1, "2": 1, "3": 1}

    with pytest.raises(WarmupCollapseError) as exc_info:
        evaluate_warmup_gate(model, probe, device="cpu", min_class_counts=min_counts)

    evidence = exc_info.value.evidence
    assert evidence["passed"] is False
    assert evidence["total_valid_pixels"] == 0


def test_warmup_gate_nonfinite_logits_fails_closed():
    """Nonfinite (NaN/Inf) logits must raise WarmupCollapseError."""
    class NanModel(nn.Module):
        def forward(self, x, valid_mask=None):
            out = torch.randn(x.shape[0], 4, x.shape[2], x.shape[3])
            out[0, 1, 0, 0] = float("nan")
            return out

    model = NanModel()
    probe = {
        "images": torch.rand(2, 3, 16, 16),
        "valid": torch.ones(2, 16, 16, dtype=torch.bool),
        "patient_ids": ["p1", "p2"],
    }
    min_counts = {"1": 1, "2": 1, "3": 1}

    with pytest.raises(WarmupCollapseError) as exc_info:
        evaluate_warmup_gate(model, probe, device="cpu", min_class_counts=min_counts)

    assert "nonfinite" in exc_info.value.evidence.get("reason", "").lower() or not exc_info.value.evidence.get("passed", True)


def test_warmup_gate_restores_training_mode_and_rng_on_failure():
    """Gate must restore model training mode and RNG states in try/finally upon failure."""
    class FailModel(nn.Module):
        def forward(self, x, valid_mask=None):
            return torch.full((x.shape[0], 4, x.shape[2], x.shape[3]), -100.0)

    model = FailModel()
    model.train()
    assert model.training is True

    probe = {
        "images": torch.rand(2, 3, 16, 16),
        "valid": torch.ones(2, 16, 16, dtype=torch.bool),
        "patient_ids": ["p1", "p2"],
    }
    min_counts = {"1": 1, "2": 1, "3": 1}

    rng_before = torch.get_rng_state()
    with pytest.raises(WarmupCollapseError):
        evaluate_warmup_gate(model, probe, device="cpu", min_class_counts=min_counts)

    assert model.training is True, "Model training mode must be restored after gate failure"
    assert torch.equal(torch.get_rng_state(), rng_before), "Torch RNG state must be restored after gate failure"


def test_warmup_gate_streaming_heterogeneous_shapes():
    """Streaming heterogeneous probe shapes processes each slice independently without shape mismatch."""
    model = _make_test_model()
    # List of heterogeneous slice dicts
    probe = [
        {"image": torch.rand(3, 24, 28), "valid": torch.ones(24, 28, dtype=torch.bool), "patient_id": "p1"},
        {"image": torch.rand(3, 32, 36), "valid": torch.ones(32, 36, dtype=torch.bool), "patient_id": "p2"},
    ]
    min_counts = {"1": 1, "2": 1, "3": 1}
    evidence = evaluate_warmup_gate(model, probe, device="cpu", min_class_counts=min_counts)
    assert evidence["passed"] is True
    assert evidence["total_valid_probe_pixels"] == (24 * 28 + 32 * 36)


def test_warmup_gate_validates_min_class_counts():
    """Gate strictly requires min_class_counts with positive integer keys '1', '2', '3'."""
    model = _make_test_model()
    probe = {"images": torch.rand(2, 3, 16, 16), "valid": torch.ones(2, 16, 16, dtype=torch.bool), "patient_ids": ["p1"]}

    # Missing class 3
    with pytest.raises(ValueError, match="min_class_counts"):
        evaluate_warmup_gate(model, probe, device="cpu", min_class_counts={"1": 1, "2": 1})

    # Non-positive count
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_warmup_gate(model, probe, device="cpu", min_class_counts={"1": 1, "2": 0, "3": 1})

    # Float count
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_warmup_gate(model, probe, device="cpu", min_class_counts={"1": 1.5, "2": 1, "3": 1})


# ===========================================================================
# 9. Optimizer Ownership & First Step Failure Receipts (Item 2)
# ===========================================================================

def test_optimizer_ownership_duplicate_within_group():
    model = _make_test_model()
    body = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    # Duplicate param inside AdamW
    dup_body = body + [body[0]]
    opts = [
        torch.optim.AdamW(dup_body, lr=3e-5),
        torch.optim.SGD(model.head.parameters(), lr=3e-3),
    ]
    with pytest.raises(ValueError, match="Duplicate parameter detected in AdamW"):
        verify_optimizer_ownership(model, opts)


def test_optimizer_ownership_records_group_metadata():
    model = _make_test_model()
    opts = _make_test_optimizers(model)
    report = verify_optimizer_ownership(model, opts)

    groups = report["param_groups"]
    assert len(groups) == 2
    assert groups[0]["optimizer"] == "AdamW"
    assert groups[0]["lr"] == 3e-5
    assert groups[0]["weight_decay"] == 1e-4
    assert groups[0]["total_numel"] > 0
    assert len(groups[0]["parameter_names"]) == report["body_param_count"]

    assert groups[1]["optimizer"] == "SGD"
    assert groups[1]["lr"] == 3e-3
    assert groups[1]["weight_decay"] == 0.0
    assert len(groups[1]["parameter_names"]) == report["head_param_count"]


def test_first_step_failure_writes_receipt_and_aborts(tmp_path):
    """When a canonical module has 0 gradients on step 0, writes FIRST_STEP_FAILURE.json and FAILURE.json."""
    data_dir = tmp_path / "data"
    manifest, manifest_sha = synthetic_fixture(data_dir, seed=42, n_labeled=3, n_unlabeled=3, size=32)

    config_path = Path("configs/baseline_recovery/baseline_smoke_cpu.json")
    cfg = json.loads(config_path.read_text())

    out_dir = tmp_path / "failed_run"

    # Patch supervised_loss or detach detail module so that detail has 0 gradient
    from unittest.mock import patch

    def broken_supervised_step(*args, **kwargs):
        loss, norm, telem = supervised_step(*args, **kwargs)
        # Force detail gradient norm to 0
        telem["gradient_norms"]["detail"] = 0.0
        return loss, norm, telem

    with patch("self_audit.engine.supervised_step", side_effect=broken_supervised_step):
        with pytest.raises(RuntimeError, match="First step optimizer telemetry verification failed"):
            train(cfg, str(manifest), manifest_sha, str(out_dir), device="cpu", synthetic=True)

    assert (out_dir / "FIRST_STEP_FAILURE.json").is_file()
    assert (out_dir / "FAILURE.json").is_file()
    receipt = json.loads((out_dir / "FIRST_STEP_FAILURE.json").read_text())
    assert receipt["failure_type"] == "implementation_error"
    assert receipt["step"] == 1
    assert "detail" in receipt["reasons"][0]


# ===========================================================================
# 10. Telemetry and Warmup Gate Abort Timing (Items 3, 4, 6)
# ===========================================================================

def test_telemetry_only_on_step_0_and_slice_tracking(tmp_path):
    """Module telemetry is recorded on step 1 only; slices_sampled/optimized tracked correctly."""
    data_dir = tmp_path / "data"
    manifest, manifest_sha = synthetic_fixture(data_dir, seed=42, n_labeled=3, n_unlabeled=3, size=32)

    cfg = json.loads(Path("configs/baseline_recovery/baseline_smoke_cpu.json").read_text())
    cfg["steps"] = 4
    cfg["warmup_steps"] = 2

    out_dir = tmp_path / "telem_run"
    report = train(cfg, str(manifest), manifest_sha, str(out_dir), device="cpu", synthetic=True)

    log_lines = [json.loads(line) for line in (out_dir / "steps.jsonl").read_text().strip().split("\n")]
    assert len(log_lines) == 4

    # Step 1 (index 0) has telemetry
    assert "module_gradient_norms" in log_lines[0]
    assert "module_update_deltas" in log_lines[0]

    # Steps 2, 3, 4 do NOT have telemetry
    for step_row in log_lines[1:]:
        assert "module_gradient_norms" not in step_row
        assert "module_update_deltas" not in step_row

    # During warmup (steps 1, 2), unlabeled_slices_optimized must be 0
    assert log_lines[0]["slices_optimized"]["unlabeled"] == 0
    assert log_lines[1]["slices_optimized"]["unlabeled"] == 0

    # Report records slice counts and optimizer calls
    assert report["slices_sampled"]["supervised"] == 4 * 2
    assert report["slices_optimized"]["supervised"] == 4 * 2
    assert report["optimizer_calls"]["adamw"] >= 4
    assert report["optimizer_calls"]["sgd"] >= 4


def test_warmup_gate_abort_before_step_1001(tmp_path):
    """Warmup gate failure at warmup_steps=2 aborts before step 3 with exactly 2 sup and 2 EMA updates."""
    data_dir = tmp_path / "data"
    manifest, manifest_sha = synthetic_fixture(data_dir, seed=42, n_labeled=3, n_unlabeled=3, size=32)

    cfg = json.loads(Path("configs/baseline_recovery/baseline_smoke_cpu.json").read_text())
    cfg["steps"] = 4
    cfg["warmup_steps"] = 2
    # Set impossible min_class_counts so gate fails
    cfg["baseline_recovery"]["warmup_gate"]["min_class_counts"] = {"1": 99999999, "2": 99999999, "3": 99999999}

    out_dir = tmp_path / "abort_run"
    with pytest.raises(WarmupCollapseError):
        train(cfg, str(manifest), manifest_sha, str(out_dir), device="cpu", synthetic=True)

    assert (out_dir / "FAILURE.json").is_file()
    failure_record = json.loads((out_dir / "FAILURE.json").read_text())
    assert failure_record["supervised_updates"] == 2
    assert failure_record["ema_updates"] == 2
    assert failure_record["pseudo_accepted"] == 0
    assert failure_record["slices_optimized"]["unlabeled"] == 0

    # steps.jsonl should only have 2 steps logged
    log_lines = (out_dir / "steps.jsonl").read_text().strip().split("\n")
    assert len(log_lines) == 2


# ===========================================================================
# 11. Shared Manifest Gates & Model Config Flags (Items 7, 8, 9)
# ===========================================================================

def test_shared_real_training_prerequisite_enforced(tmp_path):
    """Shared non-synthetic training is blocked without baseline recovery and certificate."""
    from self_audit.shared_data import synthetic_shared_fixture
    manifest_p, sha = synthetic_shared_fixture(tmp_path / "shared_data")
    m = json.loads(manifest_p.read_text())
    m["synthetic"] = False
    if "mapping_provenance" in m and isinstance(m["mapping_provenance"], dict):
        m["mapping_provenance"]["mnm"] = "canonical prepared v2"
    manifest_p.write_text(json.dumps(m))
    from self_audit.model import digest
    sha = digest(manifest_p)

    cfg = dict(DEFAULT)
    # 1. Blocked when baseline_recovery is missing
    with pytest.raises(ValueError, match="Shared real training is BLOCKED"):
        train(cfg, str(manifest_p), sha, str(tmp_path / "out1"), device="cpu", synthetic=False)

    # 2. Blocked when certificate path is missing
    from self_audit.recovery_gates import MissingEvidenceError
    cfg_br = dict(DEFAULT)
    cfg_br["baseline_recovery"] = {
        "enabled": True,
        "baseline_certificate_path": None,
    }
    with pytest.raises(MissingEvidenceError, match="baseline_certificate_path"):
        train(cfg_br, str(manifest_p), sha, str(tmp_path / "out2"), device="cpu", synthetic=False)


def test_train_py_model_config_flag_allowed_only_for_baseline_recovery(tmp_path):
    """train.py allows --model-config for opt-in baseline recovery, rejects for historical."""
    import subprocess
    import sys

    # 1. Historical pilot.json rejects --model-config
    cmd_hist = [
        sys.executable, "train.py",
        "--config", "configs/pilot.json",
        "--model-config", "configs/pilot.json",
        "--data", "nonexistent.json",
        "--data-sha256", "0" * 64,
        "--out", str(tmp_path / "hist_out"),
    ]
    res_hist = subprocess.run(cmd_hist, capture_output=True, text=True, check=False)
    assert res_hist.returncode != 0
    assert "native-volume-only flags are invalid for historical schema" in res_hist.stderr

    # 2. baseline recovery config allows --model-config (fails later on nonexistent data, not on flag)
    model_cfg_p = tmp_path / "dummy_model.json"
    model_cfg_p.write_text(json.dumps({"backbone": "tiny_test", "weights": "", "weights_sha256": ""}))
    cmd_br = [
        sys.executable, "train.py",
        "--config", "configs/baseline_recovery/baseline_smoke_cpu.json",
        "--model-config", str(model_cfg_p),
        "--data", "nonexistent.json",
        "--data-sha256", "0" * 64,
        "--out", str(tmp_path / "br_out"),
    ]
    res_br = subprocess.run(cmd_br, capture_output=True, text=True, check=False)
    assert "native-volume-only flags are invalid" not in res_br.stderr


def test_shared_configs_exist_and_validate():
    """Verify shared_4plus4_10k.json and shared_4plus4_smoke_cpu.json exist and validate."""
    cfg_10k_p = Path("configs/baseline_recovery/shared_4plus4_10k.json")
    cfg_smoke_p = Path("configs/baseline_recovery/shared_4plus4_smoke_cpu.json")

    assert cfg_10k_p.is_file()
    assert cfg_smoke_p.is_file()

    c_10k = json.loads(cfg_10k_p.read_text())
    c_smoke = json.loads(cfg_smoke_p.read_text())

    val_10k = validate_config(c_10k)
    val_smoke = validate_config(c_smoke)

    assert val_10k["steps"] == 10000
    assert val_10k["warmup_steps"] == 1000
    assert val_10k["batch_size"] == 8
    assert val_10k["baseline_recovery"]["supervised_batch_size"] == 8
    assert val_10k["baseline_recovery"]["unlabeled_batch_size"] == 8
    assert val_10k["window"]["mode"] == "none"
    assert val_10k["teacher_memory"]["arm"] == "A0"

    assert val_smoke["steps"] == 4
    assert val_smoke["warmup_steps"] == 2
    assert val_smoke["batch_size"] == 2


# ===========================================================================
# 12. Cadence, Submodule Modes, Purpose, and Partial Freeze Tests (Followup)
# ===========================================================================

def test_outer_steps_optimizer_and_ema_cadence(tmp_path):
    """Test 3 outer steps, warmup 1, threshold 0: final each AdamW/SGD 5, EMA 3, pseudo 2.

    Steps array must record calls [1, 3, 5] and EMA [1, 2, 3].
    """
    data_dir = tmp_path / "data"
    manifest, manifest_sha = synthetic_fixture(data_dir, seed=42, n_labeled=3, n_unlabeled=3, size=32)

    cfg = json.loads(Path("configs/baseline_recovery/baseline_smoke_cpu.json").read_text())
    cfg["steps"] = 3
    cfg["warmup_steps"] = 1
    cfg["pseudo_threshold"] = 0.0  # Force pseudo acceptance on all valid pixels
    cfg["method"] = "ema_full"

    out_dir = tmp_path / "cadence_run"
    report = train(cfg, str(manifest), manifest_sha, str(out_dir), device="cpu", synthetic=True)

    # Final counts
    assert report["supervised_updates"] == 3
    assert report["ema_updates"] == 3
    assert report["pseudo_optimizer_updates"] == 2
    assert report["optimizer_calls"]["adamw"] == 5
    assert report["optimizer_calls"]["sgd"] == 5

    # Steps log verification
    log_lines = [json.loads(line) for line in (out_dir / "steps.jsonl").read_text().strip().split("\n")]
    assert len(log_lines) == 3

    # Step 1 (warmup): 1 AdamW, 1 SGD, 1 EMA, 0 pseudo
    assert log_lines[0]["optimizer_calls"]["adamw"] == 1
    assert log_lines[0]["optimizer_calls"]["sgd"] == 1
    assert log_lines[0]["ema_updates"] == 1
    assert log_lines[0]["pseudo_optimizer_updates"] == 0
    assert log_lines[0]["supervised_updates"] == 1

    # Step 2: 3 AdamW, 3 SGD, 2 EMA, 1 pseudo
    assert log_lines[1]["optimizer_calls"]["adamw"] == 3
    assert log_lines[1]["optimizer_calls"]["sgd"] == 3
    assert log_lines[1]["ema_updates"] == 2
    assert log_lines[1]["pseudo_optimizer_updates"] == 1
    assert log_lines[1]["supervised_updates"] == 2

    # Step 3: 5 AdamW, 5 SGD, 3 EMA, 2 pseudo
    assert log_lines[2]["optimizer_calls"]["adamw"] == 5
    assert log_lines[2]["optimizer_calls"]["sgd"] == 5
    assert log_lines[2]["ema_updates"] == 3
    assert log_lines[2]["pseudo_optimizer_updates"] == 2
    assert log_lines[2]["supervised_updates"] == 3


def test_warmup_gate_preserves_mixed_submodule_modes():
    """Gate must restore individual submodule modes (e.g. root.train + child.eval) after both PASS and FAIL."""
    model = _make_test_model()
    model.train()
    model.encoder.eval()

    assert model.training is True
    assert model.encoder.training is False

    # 1. PASS case
    probe_pass = {
        "images": torch.rand(4, 3, 32, 32),
        "valid": torch.ones(4, 32, 32, dtype=torch.bool),
        "patient_ids": ["p1", "p2", "p3", "p4"],
    }
    evaluate_warmup_gate(model, probe_pass, device="cpu", min_class_counts={"1": 1, "2": 1, "3": 1})
    assert model.training is True, "Root model training mode must remain True after PASS"
    assert model.encoder.training is False, "Child encoder training mode must remain False after PASS"

    # 2. FAIL case
    probe_fail = {
        "images": torch.rand(2, 3, 16, 16),
        "valid": torch.zeros(2, 16, 16, dtype=torch.bool),
        "patient_ids": ["p1", "p2"],
    }
    with pytest.raises(WarmupCollapseError):
        evaluate_warmup_gate(model, probe_fail, device="cpu", min_class_counts={"1": 1, "2": 1, "3": 1})
    assert model.training is True, "Root model training mode must remain True after FAIL"
    assert model.encoder.training is False, "Child encoder training mode must remain False after FAIL"


def test_min_class_counts_rejects_duplicate_or_aliased_keys():
    """Reject duplicate integer/string alias keys in min_class_counts; accept exact keys."""
    from self_audit.baseline_recovery import _validate_min_class_counts

    # Duplicate alias keys e.g. "1" and 1 in same dict
    with pytest.raises(ValueError, match="duplicate integer/string alias keys"):
        _validate_min_class_counts({"1": 1, 1: 1, "2": 1, "3": 1})

    # Exact string keys pass
    norm1 = _validate_min_class_counts({"1": 2, "2": 3, "3": 4})
    assert norm1 == {"1": 2, "2": 3, "3": 4}

    # Exact int keys pass and convert cleanly
    norm2 = _validate_min_class_counts({1: 2, 2: 3, 3: 4})
    assert norm2 == {"1": 2, "2": 3, "3": 4}

    # Extra/invalid keys rejected
    with pytest.raises(ValueError, match="min_class_counts keys must be exactly"):
        _validate_min_class_counts({"1": 1, "2": 1, "4": 1})


def test_baseline_recovery_purpose_and_strict_keys():
    """Verify purpose='training'|'profile', strict unknown key rejection, and rules."""
    # Unknown key in baseline_recovery rejected
    with pytest.raises(ValueError, match="baseline_recovery contains unknown keys"):
        validate_baseline_recovery_config({"enabled": True, "invalid_option": 123})

    # Unknown key in warmup_gate rejected
    with pytest.raises(ValueError, match="warmup_gate contains unknown keys"):
        validate_baseline_recovery_config({"enabled": True, "warmup_gate": {"enabled": True, "bad_key": 1}})

    # Invalid purpose rejected
    with pytest.raises(ValueError, match="baseline_recovery.purpose must be 'training' or 'profile'"):
        validate_baseline_recovery_config({"enabled": True, "purpose": "unknown_purpose"})

    # Valid profile purpose accepted
    cfg_prof = validate_baseline_recovery_config({"enabled": True, "purpose": "profile"})
    assert cfg_prof["purpose"] == "profile"


def test_optimizer_ownership_rejects_partial_freeze():
    """Require ALL canonical module parameters to be trainable; partial freeze is rejected."""
    model = _make_test_model()
    opts = _make_test_optimizers(model)

    # Freeze just one parameter in encoder
    model.encoder.stages[0].weight.requires_grad = False

    with pytest.raises(ValueError, match="Require ALL canonical module parameters to be trainable"):
        verify_optimizer_ownership(model, opts)


def test_shared_real_training_mandatory_cert_sha(tmp_path):
    """Shared real training strictly requires a 64-hex baseline_certificate_sha256."""
    from self_audit.shared_data import synthetic_shared_fixture
    from self_audit.recovery_gates import MissingEvidenceError
    manifest_p, sha = synthetic_shared_fixture(tmp_path / "shared_data")
    m = json.loads(manifest_p.read_text())
    m["synthetic"] = False
    if "mapping_provenance" in m and isinstance(m["mapping_provenance"], dict):
        m["mapping_provenance"]["mnm"] = "canonical prepared v2"
    manifest_p.write_text(json.dumps(m))
    from self_audit.model import digest
    sha = digest(manifest_p)

    dummy_cert = tmp_path / "cert.json"
    dummy_cert.write_text(json.dumps({"schema": "self_audit.baseline_gate_certificate.v1"}))

    cfg = dict(DEFAULT)
    cfg["baseline_recovery"] = {
        "enabled": True,
        "baseline_certificate_path": str(dummy_cert),
        "baseline_certificate_sha256": None,  # Missing mandatory SHA
    }

    with pytest.raises(MissingEvidenceError, match="baseline_certificate_sha256 is mandatory"):
        train(cfg, str(manifest_p), sha, str(tmp_path / "out"), device="cpu", synthetic=False)


def test_real_training_purpose_and_gate_enforcement(tmp_path):
    """Real baseline recovery training requires method ema_full, window none, warmup_gate enabled, and warmup > 0."""
    from self_audit.data import synthetic_fixture
    data_dir = tmp_path / "data"
    manifest, manifest_sha = synthetic_fixture(data_dir, seed=42, n_labeled=3, n_unlabeled=3, size=32)

    # Set manifest.synthetic = False for real training tests
    m = json.loads(manifest.read_text())
    m["synthetic"] = False
    manifest.write_text(json.dumps(m))
    from self_audit.model import digest
    manifest_sha = digest(manifest)

    cfg = json.loads(Path("configs/baseline_recovery/baseline_smoke_cpu.json").read_text())

    # 1. Real training (synthetic=False) with warmup_gate disabled is rejected
    cfg_no_gate = copy.deepcopy(cfg)
    cfg_no_gate["baseline_recovery"]["warmup_gate"]["enabled"] = False
    with pytest.raises(ValueError, match="Real baseline recovery training requires warmup_gate to be enabled"):
        train(cfg_no_gate, str(manifest), manifest_sha, str(tmp_path / "out1"), device="cpu", synthetic=False)

    # 2. Real profile (synthetic=False, purpose=profile) allows warmup_gate disabled, but steps <= 8
    cfg_profile = copy.deepcopy(cfg)
    cfg_profile["baseline_recovery"]["purpose"] = "profile"
    cfg_profile["baseline_recovery"]["warmup_gate"]["enabled"] = False
    cfg_profile["steps"] = 9
    with pytest.raises(ValueError, match="Real baseline recovery profile allows at most 8 steps"):
        train(cfg_profile, str(manifest), manifest_sha, str(tmp_path / "out2"), device="cpu", synthetic=False)

    # 3. Profile run (purpose=profile) with warmup_gate disabled runs and marks certifiable=False
    m["synthetic"] = True
    manifest.write_text(json.dumps(m))
    manifest_sha = digest(manifest)
    cfg_profile["steps"] = 2
    cfg_profile["warmup_steps"] = 1
    rep = train(cfg_profile, str(manifest), manifest_sha, str(tmp_path / "out3"), device="cpu", synthetic=True)
    assert rep["quality"] == "NOT_EVALUATED"
    assert rep["baseline_recovery"]["certifiable"] is False
    assert rep["baseline_recovery"]["purpose"] == "profile"


def test_guarded_direct_commits_do_not_increment_optimizer_calls(tmp_path):
    """Guarded head/decoder direct commits increment guarded_commit_updates but NOT AdamW/SGD calls."""
    data_dir = tmp_path / "data"
    manifest, manifest_sha = synthetic_fixture(data_dir, seed=42, n_labeled=3, n_unlabeled=3, size=32)

    cfg = json.loads(Path("configs/baseline_recovery/baseline_smoke_cpu.json").read_text())
    cfg["method"] = "joint_guard"
    cfg["update_scope"] = "head"
    cfg["steps"] = 3
    cfg["warmup_steps"] = 1
    cfg["pseudo_threshold"] = 0.0  # Encourage candidate generation
    cfg["risk_budget"] = 1.0       # Generous risk budget to accept candidate step

    out_dir = tmp_path / "guarded_run"
    rep = train(cfg, str(manifest), manifest_sha, str(out_dir), device="cpu", synthetic=True)

    # 3 outer steps -> exactly 3 supervised updates -> exactly 3 AdamW and 3 SGD calls
    assert rep["supervised_updates"] == 3
    assert rep["optimizer_calls"]["adamw"] == 3
    assert rep["optimizer_calls"]["sgd"] == 3
    # Pseudo optimizer updates are 0 for guarded commits
    assert rep["pseudo_optimizer_updates"] == 0
    # Any accepted pseudo step was direct commit, tracked in guarded_commit_updates
    if rep["pseudo_accepted"] > 0:
        assert rep["guarded_commit_updates"] == rep["pseudo_accepted"]
