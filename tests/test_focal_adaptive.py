"""Tests for opt-in focal loss and train_plateau learning rate scheduler."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from self_audit.data import synthetic_fixture, digest, write_json
from self_audit.engine import DEFAULT, train, load_checkpoint, save_checkpoint, validate_config, supervised_step
from self_audit.losses import (supervised_loss, focal_loss, validate_supervised_loss_config,
                               FOCAL_DICE_DEFAULT)
from self_audit.lr_schedule import (validate_lr_scheduler_config, validate_scheduler_state_dict,
                                    TrainPlateauScheduler, DEFAULT_PLATEAU)


# ==============================================================================
# 1. Focal loss tests
# ==============================================================================

def test_gamma0_equals_cross_entropy():
    torch.manual_seed(101)
    logits = torch.randn(2, 4, 16, 16)
    labels = torch.randint(0, 4, (2, 16, 16))
    labels[0, :4, :4] = 255  # Include ignore pixels

    # Pure focal_loss vs cross_entropy
    fl0 = focal_loss(logits, labels, gamma=0.0)
    ce = F.cross_entropy(logits, labels, ignore_index=255)
    assert torch.equal(fl0, ce)

    # Full supervised_loss with focal_dice (gamma=0, weights=1) vs default CE+Dice
    focal_cfg = {"name": "focal_dice", "gamma": 0.0, "focal_weight": 1.0, "dice_weight": 1.0}
    loss_focal = supervised_loss(logits, labels, config=focal_cfg)
    loss_default = supervised_loss(logits, labels)
    assert torch.equal(loss_focal, loss_default)


def test_ignored_pixel_zero_effect_and_gradient():
    torch.manual_seed(202)
    logits = torch.randn(1, 4, 8, 8, requires_grad=True)
    labels = torch.randint(0, 4, (1, 8, 8))
    labels[:, 4:, :] = 255  # Bottom half ignored

    focal_cfg = {"name": "focal_dice", "gamma": 2.0, "focal_weight": 1.0, "dice_weight": 1.0}
    loss = supervised_loss(logits, labels, config=focal_cfg)
    loss.backward()

    # Gradients on ignored pixels must be strictly zero
    assert torch.equal(logits.grad[:, :, 4:, :], torch.zeros_like(logits.grad[:, :, 4:, :]))
    # Gradients on valid pixels must not be all zero
    assert logits.grad[:, :, :4, :].abs().sum() > 0

    # Perturbing ignored pixels must have zero effect on the loss value
    with torch.no_grad():
        logits_perturbed = logits.clone()
        logits_perturbed[:, :, 4:, :] += 1000.0
        loss_perturbed = supervised_loss(logits_perturbed, labels, config=focal_cfg)
        assert torch.equal(loss, loss_perturbed)


def test_extreme_logits_finite():
    # Extreme logits (+/- 1e4)
    logits = torch.tensor([[[[-1e4, 1e4], [0.0, -1e4]],
                            [[1e4, -1e4], [0.0, 1e4]],
                            [[0.0, 0.0], [1e4, 0.0]],
                            [[-1e4, 0.0], [0.0, -1e4]]]], requires_grad=True)
    labels = torch.tensor([[[0, 1], [2, 3]]])

    focal_cfg = {"name": "focal_dice", "gamma": 2.0, "focal_weight": 1.0, "dice_weight": 1.0}
    loss = supervised_loss(logits, labels, config=focal_cfg)
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)

    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert not torch.isnan(logits.grad).any()


def test_empty_and_malformed_loss_config_rejected():
    logits = torch.randn(1, 4, 8, 8)
    labels = torch.full((1, 8, 8), 255)

    # Empty labeled support (all 255)
    with pytest.raises(ValueError, match="empty labeled support"):
        focal_loss(logits, labels, gamma=2.0)
    with pytest.raises(ValueError, match="empty labeled support"):
        supervised_loss(logits, labels)

    # Invalid ontology: label 4 is invalid
    bad_labels = torch.zeros((1, 8, 8), dtype=torch.long)
    bad_labels[0, 0, 0] = 4
    with pytest.raises(ValueError, match="labels must have values in {0, 1, 2, 3} or 255"):
        focal_loss(logits, bad_labels, gamma=2.0)
    with pytest.raises(ValueError, match="labels must have values in {0, 1, 2, 3} or 255"):
        supervised_loss(logits, bad_labels)

    # Shape mismatch
    with pytest.raises(ValueError, match="native logits and labels must have identical spatial shape"):
        supervised_loss(logits, torch.zeros((1, 4, 4), dtype=torch.long))

    # Channel != 4
    with pytest.raises(ValueError, match="native logits and labels must have identical spatial shape"):
        supervised_loss(torch.randn(1, 3, 8, 8), torch.zeros((1, 8, 8), dtype=torch.long))

    # Config validation rejections
    with pytest.raises(ValueError):
        validate_supervised_loss_config({"name": "unknown"})
    with pytest.raises(ValueError):
        validate_supervised_loss_config({"name": "focal_dice", "gamma": -1.0, "focal_weight": 1.0, "dice_weight": 1.0})
    with pytest.raises(ValueError):
        validate_supervised_loss_config({"name": "focal_dice", "gamma": float("nan"), "focal_weight": 1.0, "dice_weight": 1.0})
    with pytest.raises(ValueError):
        validate_supervised_loss_config({"name": "focal_dice", "gamma": 2.0, "focal_weight": 0.0, "dice_weight": 0.0})
    with pytest.raises(ValueError):
        validate_supervised_loss_config({"name": "focal_dice", "gamma": 2.0, "focal_weight": 1.0})  # Missing dice_weight
    with pytest.raises(ValueError):
        validate_supervised_loss_config("not a dict")

    # Mode restriction: focal_dice requires ema_full with window mode none
    bad_cfg = copy.deepcopy(DEFAULT)
    bad_cfg["method"] = "joint_guard"
    bad_cfg["supervised_loss"] = copy.deepcopy(FOCAL_DICE_DEFAULT)
    with pytest.raises(ValueError, match="supervised_loss focal_dice requires method ema_full with window mode none"):
        validate_config(bad_cfg)

    bad_cfg2 = copy.deepcopy(DEFAULT)
    bad_cfg2["method"] = "ema_full"
    bad_cfg2["window"]["mode"] = "fixed"
    bad_cfg2["supervised_loss"] = copy.deepcopy(FOCAL_DICE_DEFAULT)
    with pytest.raises(ValueError, match="supervised_loss focal_dice requires method ema_full with window mode none"):
        validate_config(bad_cfg2)


# ==============================================================================
# 2. LR scheduler tests
# ==============================================================================

def test_lr_scheduler_config_validation():
    with pytest.raises(ValueError):
        validate_lr_scheduler_config("not a dict")
    with pytest.raises(ValueError):
        validate_lr_scheduler_config({"name": "cosine"})
    with pytest.raises(ValueError):
        validate_lr_scheduler_config({**DEFAULT_PLATEAU, "factor": 1.5})
    with pytest.raises(ValueError):
        validate_lr_scheduler_config({**DEFAULT_PLATEAU, "patience": -1})
    with pytest.raises(ValueError):
        validate_lr_scheduler_config({**DEFAULT_PLATEAU, "min_lr_factor": 0.0})
    with pytest.raises(ValueError):
        validate_lr_scheduler_config({**DEFAULT_PLATEAU, "threshold_mode": "invalid"})

    # Mode restriction
    bad_cfg = copy.deepcopy(DEFAULT)
    bad_cfg["method"] = "joint_guard"
    bad_cfg["lr_scheduler"] = copy.deepcopy(DEFAULT_PLATEAU)
    with pytest.raises(ValueError, match="lr_scheduler train_plateau requires method ema_full with window mode none"):
        validate_config(bad_cfg)


def test_plateau_warmup_timing_and_reductions():
    p1 = torch.nn.Parameter(torch.zeros(1))
    p2 = torch.nn.Parameter(torch.zeros(1))
    opt_body = torch.optim.AdamW([p1], lr=1e-3)
    opt_head = torch.optim.SGD([p2], lr=1e-1)
    opts = [opt_body, opt_head]

    sched_cfg = {
        "name": "train_plateau",
        "interval_steps": 2,
        "factor": 0.5,
        "patience": 1,
        "threshold": 0.0001,
        "threshold_mode": "rel",
        "cooldown": 0,
        "min_lr_factor": 0.25,
        "eps": 1e-8,
    }
    scheduler = TrainPlateauScheduler(opts, sched_cfg, 1e-3, 1e-1)

    # 1. Warmup timing: warmup_steps = 3
    # During warmup steps 0, 1, 2: observe should be ignored
    for step in range(3):
        scheduler.observe(1.0, is_warmup=True)
        stepped, _, _ = scheduler.step_if_interval_complete()
        assert not stepped
        assert scheduler.metric_count == 0
        assert scheduler.metric_sum == 0.0

    # 2. Post-warmup accumulation
    # Step 3: accumulate 1st loss
    scheduler.observe(2.0, is_warmup=False)
    stepped, _, _ = scheduler.step_if_interval_complete()
    assert not stepped
    assert scheduler.metric_count == 1

    # Step 4: accumulate 2nd loss -> interval complete (2 steps)
    scheduler.observe(2.0, is_warmup=False)
    stepped, metric, lr_reduced = scheduler.step_if_interval_complete()
    assert stepped
    assert metric == pytest.approx(2.0)
    assert not lr_reduced  # First interval establishes baseline best=2.0
    assert scheduler.metric_count == 0
    assert scheduler.num_updates == 1
    assert opt_body.param_groups[0]["lr"] == 1e-3
    assert opt_head.param_groups[0]["lr"] == 1e-1

    # Interval 2 (steps 5, 6): stagnant loss 2.0 -> bad_epochs = 1 (patience is 1, not yet exceeded)
    scheduler.observe(2.0, is_warmup=False)
    scheduler.step_if_interval_complete()
    scheduler.observe(2.0, is_warmup=False)
    stepped, metric, lr_reduced = scheduler.step_if_interval_complete()
    assert stepped
    assert not lr_reduced
    assert scheduler.sched_body.num_bad_epochs == 1
    assert opt_body.param_groups[0]["lr"] == 1e-3

    # Interval 3 (steps 7, 8): stagnant loss 2.0 -> bad_epochs > patience -> LR reduction!
    scheduler.observe(2.0, is_warmup=False)
    scheduler.step_if_interval_complete()
    scheduler.observe(2.0, is_warmup=False)
    stepped, metric, lr_reduced = scheduler.step_if_interval_complete()
    assert stepped
    assert lr_reduced
    assert opt_body.param_groups[0]["lr"] == pytest.approx(0.0005)
    assert opt_head.param_groups[0]["lr"] == pytest.approx(0.05)
    # Ratio between head and body LR must be strictly preserved
    assert opt_head.param_groups[0]["lr"] / opt_body.param_groups[0]["lr"] == pytest.approx(100.0)

    # 3. Minimum LR enforcement (min_lr_factor = 0.25 => min_body = 0.00025, min_head = 0.025)
    # Trigger another reduction (intervals 4 and 5)
    for _ in range(4):
        scheduler.observe(2.0, is_warmup=False)
        scheduler.step_if_interval_complete()
    assert opt_body.param_groups[0]["lr"] == pytest.approx(0.00025)
    assert opt_head.param_groups[0]["lr"] == pytest.approx(0.025)

    # Subsequent stagnant intervals cannot reduce below minimum LR
    for _ in range(4):
        scheduler.observe(2.0, is_warmup=False)
        scheduler.step_if_interval_complete()
    assert opt_body.param_groups[0]["lr"] == pytest.approx(0.00025)
    assert opt_head.param_groups[0]["lr"] == pytest.approx(0.025)


# ==============================================================================
# 3. Checkpoint and Resume Strict Validation Tests
# ==============================================================================

def test_resume_state_missing_or_tampered_rejected():
    state = {
        "body": {
            "_last_lr": [0.001], "best": float("inf"), "cooldown": 0, "cooldown_counter": 0,
            "default_min_lr": 1e-5, "eps": 1e-8, "factor": 0.5, "last_epoch": 0,
            "min_lrs": [1e-5], "mode": "min", "mode_worse": float("inf"),
            "num_bad_epochs": 0, "patience": 5, "threshold": 0.0001, "threshold_mode": "rel",
        },
        "head": {
            "_last_lr": [0.1], "best": float("inf"), "cooldown": 0, "cooldown_counter": 0,
            "default_min_lr": 1e-3, "eps": 1e-8, "factor": 0.5, "last_epoch": 0,
            "min_lrs": [1e-3], "mode": "min", "mode_worse": float("inf"),
            "num_bad_epochs": 0, "patience": 5, "threshold": 0.0001, "threshold_mode": "rel",
        },
        "metric_sum": 0.0,
        "metric_count": 0,
        "num_updates": 0,
    }
    cfg = copy.deepcopy(DEFAULT_PLATEAU)
    opts = [{"param_groups": [{"lr": 0.001}]}, {"param_groups": [{"lr": 0.1}]}]

    # Valid initial state passes
    validate_scheduler_state_dict(state, config=cfg, step=2, warmup_steps=2,
                                  initial_body_lr=0.001, initial_head_lr=0.1, payload_optimizers=opts)

    # 1. Missing key in top-level state
    bad_state = copy.deepcopy(state)
    del bad_state["metric_sum"]
    with pytest.raises(ValueError, match="keys mismatch"):
        validate_scheduler_state_dict(bad_state, config=cfg)

    # 2. Negative metric_count
    bad_state = copy.deepcopy(state)
    bad_state["metric_count"] = -1
    with pytest.raises(ValueError, match="invalid metric_count"):
        validate_scheduler_state_dict(bad_state, config=cfg)

    # 3. Nonfinite metric_sum
    bad_state = copy.deepcopy(state)
    bad_state["metric_sum"] = float("nan")
    with pytest.raises(ValueError, match="invalid metric_sum"):
        validate_scheduler_state_dict(bad_state, config=cfg)

    # 4. Tampered factor
    bad_state = copy.deepcopy(state)
    bad_state["body"]["factor"] = 0.2
    with pytest.raises(ValueError, match="factor changed"):
        validate_scheduler_state_dict(bad_state, config=cfg)

    # 5. last_epoch != num_updates
    bad_state = copy.deepcopy(state)
    bad_state["body"]["last_epoch"] = 1
    with pytest.raises(ValueError, match="last_epoch"):
        validate_scheduler_state_dict(bad_state, config=cfg)

    # 6. Step / update count inconsistency
    # At step 10, warmup 2, interval 2: expected updates = (10-2)//2 = 4, metric_count = 0
    with pytest.raises(ValueError, match="inconsistent with step"):
        validate_scheduler_state_dict(state, config=cfg, step=10, warmup_steps=2)

    # 7. Mismatched optimizer LR vs scheduler last LR
    bad_opts = [{"param_groups": [{"lr": 0.0005}]}, {"param_groups": [{"lr": 0.1}]}]
    with pytest.raises(ValueError, match="mismatched LR between body optimizer"):
        validate_scheduler_state_dict(state, config=cfg, step=2, warmup_steps=2,
                                      initial_body_lr=0.001, initial_head_lr=0.1, payload_optimizers=bad_opts)

    # 8. NaN best
    bad_state = copy.deepcopy(state)
    bad_state["body"]["best"] = float("nan")
    with pytest.raises(ValueError, match="best is NaN"):
        validate_scheduler_state_dict(bad_state, config=cfg)

    # 9. Coordinator regression: body.best = -inf
    bad_state = copy.deepcopy(state)
    bad_state["body"]["best"] = float("-inf")
    with pytest.raises(ValueError, match="best must be exactly \\+inf"):
        validate_scheduler_state_dict(bad_state, config=cfg)

    # 10. Coordinator regression: body._last_lr[0] = NaN
    bad_state = copy.deepcopy(state)
    bad_state["body"]["_last_lr"] = [float("nan")]
    with pytest.raises(ValueError, match="_last_lr must contain finite positive float"):
        validate_scheduler_state_dict(bad_state, config=cfg)

    # 11. Coordinator regression: metric_count = 0 with metric_sum = 1.0
    bad_state = copy.deepcopy(state)
    bad_state["metric_count"] = 0
    bad_state["metric_sum"] = 1.0
    with pytest.raises(ValueError, match="metric_sum must be 0.0 when metric_count is 0"):
        validate_scheduler_state_dict(bad_state, config=cfg)

    # 12. mode_worse != +inf
    bad_state = copy.deepcopy(state)
    bad_state["body"]["mode_worse"] = 0.0
    with pytest.raises(ValueError, match="mode_worse must be \\+inf"):
        validate_scheduler_state_dict(bad_state, config=cfg)

    # 13. default_min_lr mismatch
    bad_state = copy.deepcopy(state)
    bad_state["body"]["default_min_lr"] = 999.0
    with pytest.raises(ValueError, match="default_min_lr"):
        validate_scheduler_state_dict(bad_state, config=cfg, step=2, warmup_steps=2,
                                      initial_body_lr=0.001, initial_head_lr=0.1, payload_optimizers=opts)

    # 14. Typed integer counters (reject float/bool/str)
    for key in ("num_updates", "metric_count"):
        bad_state = copy.deepcopy(state)
        bad_state[key] = 0.0
        with pytest.raises(ValueError, match=f"invalid {key}"):
            validate_scheduler_state_dict(bad_state, config=cfg)

    for key in ("last_epoch", "num_bad_epochs", "cooldown_counter"):
        bad_state = copy.deepcopy(state)
        bad_state["body"][key] = 0.0
        with pytest.raises(ValueError, match=f"invalid {key}"):
            validate_scheduler_state_dict(bad_state, config=cfg)


@pytest.mark.parametrize("mutation,match", [
    ("best_neg_inf", "best must be exactly \\+inf"),
    ("last_lr_nan", "_last_lr must contain finite positive float"),
    ("metric_sum_nonzero_empty", "metric_sum must be 0.0 when metric_count is 0"),
    ("last_lr_out_of_bounds_high", "outside \\[min_lr="),
    ("last_lr_out_of_bounds_low", "outside \\[min_lr="),
    ("mode_worse_not_inf", "mode_worse must be \\+inf"),
])
def test_parameterized_warmup_state_rejections(mutation, match):
    state = {
        "body": {
            "_last_lr": [0.001], "best": float("inf"), "cooldown": 0, "cooldown_counter": 0,
            "default_min_lr": 1e-5, "eps": 1e-8, "factor": 0.5, "last_epoch": 0,
            "min_lrs": [1e-5], "mode": "min", "mode_worse": float("inf"),
            "num_bad_epochs": 0, "patience": 5, "threshold": 0.0001, "threshold_mode": "rel",
        },
        "head": {
            "_last_lr": [0.1], "best": float("inf"), "cooldown": 0, "cooldown_counter": 0,
            "default_min_lr": 1e-3, "eps": 1e-8, "factor": 0.5, "last_epoch": 0,
            "min_lrs": [1e-3], "mode": "min", "mode_worse": float("inf"),
            "num_bad_epochs": 0, "patience": 5, "threshold": 0.0001, "threshold_mode": "rel",
        },
        "metric_sum": 0.0,
        "metric_count": 0,
        "num_updates": 0,
    }
    cfg = copy.deepcopy(DEFAULT_PLATEAU)
    opts = [{"param_groups": [{"lr": 0.001}]}, {"param_groups": [{"lr": 0.1}]}]

    if mutation == "best_neg_inf":
        state["body"]["best"] = float("-inf")
    elif mutation == "last_lr_nan":
        state["body"]["_last_lr"] = [float("nan")]
    elif mutation == "metric_sum_nonzero_empty":
        state["metric_sum"] = 1.0
    elif mutation == "last_lr_out_of_bounds_high":
        state["body"]["_last_lr"] = [0.05]
        opts[0]["param_groups"][0]["lr"] = 0.05
    elif mutation == "last_lr_out_of_bounds_low":
        state["body"]["_last_lr"] = [1e-7]
        opts[0]["param_groups"][0]["lr"] = 1e-7
    elif mutation == "mode_worse_not_inf":
        state["body"]["mode_worse"] = 1.0

    with pytest.raises(ValueError, match=match):
        validate_scheduler_state_dict(state, config=cfg, step=2, warmup_steps=2,
                                      initial_body_lr=0.001, initial_head_lr=0.1, payload_optimizers=opts)


# ==============================================================================
# 4. Integration: Uninterrupted vs Resumed Bit-Exact Parity
# ==============================================================================

def test_uninterrupted_vs_resumed_parity_focal_adaptive(tmp_path):
    """Verify bit-exact parity between uninterrupted and interrupted-then-resumed runs with focal loss and train_plateau."""
    manifest, sha = synthetic_fixture(tmp_path / "data")

    c = copy.deepcopy(DEFAULT)
    c.update(
        method="ema_full",
        update_scope="head",
        steps=8,
        warmup_steps=2,
        save_every=2,
        pseudo_threshold=0.25,
        ema_decay=0.9,
        body_lr=0.001,
        head_lr=0.03,
        seed=42,
    )
    c["model"]["backbone"] = "tiny_test"
    c["supervised_loss"] = {
        "name": "focal_dice",
        "gamma": 2.0,
        "focal_weight": 1.0,
        "dice_weight": 1.0,
    }
    c["lr_scheduler"] = {
        "name": "train_plateau",
        "interval_steps": 2,
        "factor": 0.5,
        "patience": 1,
        "threshold": 0.0001,
        "threshold_mode": "rel",
        "cooldown": 0,
        "min_lr_factor": 0.1,
        "eps": 1e-8,
    }

    # 1. Uninterrupted Run
    out_uninterrupted = tmp_path / "run_uninterrupted"
    report_un = train(c, manifest, sha, out_uninterrupted, device="cpu", synthetic=True)
    assert report_un["status"] == "COMPLETED"
    assert report_un["steps"] == 8
    assert report_un["scheduler_updates"] == 3  # (8 - 2) // 2 = 3 updates

    # 2. Interrupted Run stopped at step 5 (middle of monitoring interval: steps 3, 4 was update 1; step 5 is middle of interval 2)
    out_resumed = tmp_path / "run_resumed"
    report_stop = train(c, manifest, sha, out_resumed, device="cpu", synthetic=True, stop_after_step=5)
    assert report_stop["status"] == "INTERRUPTED"
    assert report_stop["step"] == 5

    step5_dir = out_resumed / "step_000005"
    assert step5_dir.is_dir()
    step5_sha = digest(step5_dir / "MANIFEST.json")
    step5_m = json.loads((step5_dir / "MANIFEST.json").read_text())
    assert step5_m["lr_scheduler"]["metric_count"] == 1
    assert step5_m["lr_scheduler"]["num_updates"] == 1

    # 3. Resume Run from step 5 to completion (step 8)
    report_res = train(c, manifest, sha, out_resumed, device="cpu", synthetic=True,
                       resume=str(step5_dir), resume_sha256=step5_sha)
    assert report_res["status"] == "COMPLETED"
    assert report_res["steps"] == 8
    assert report_res["scheduler_updates"] == 3

    # 4. Bit-exact payload comparison
    payload_un = torch.load(out_uninterrupted / "final/weights.pt", weights_only=True)
    payload_res = torch.load(out_resumed / "final/weights.pt", weights_only=True)

    for k in payload_un["student"]:
        assert torch.equal(payload_un["student"][k], payload_res["student"][k]), f"student mismatch at {k}"
    for k in payload_un["teacher"]:
        assert torch.equal(payload_un["teacher"][k], payload_res["teacher"][k]), f"teacher mismatch at {k}"
    assert torch.equal(payload_un["torch_rng"], payload_res["torch_rng"])
    assert payload_un["py_rng"] == payload_res["py_rng"]

    # Scheduler state dict equality
    sched_un = payload_un["lr_scheduler"]
    sched_res = payload_res["lr_scheduler"]
    assert sched_un["num_updates"] == sched_res["num_updates"]
    assert sched_un["metric_count"] == sched_res["metric_count"]
    assert sched_un["metric_sum"] == pytest.approx(sched_res["metric_sum"], abs=1e-12)
    assert sched_un["body"]["_last_lr"] == sched_res["body"]["_last_lr"]
    assert sched_un["head"]["_last_lr"] == sched_res["head"]["_last_lr"]

    # 5. Log lines comparison
    lines_un = [json.loads(x) for x in (out_uninterrupted / "steps.jsonl").read_text().splitlines() if x.strip()]
    lines_res = [json.loads(x) for x in (out_resumed / "steps.jsonl").read_text().splitlines() if x.strip()]
    assert len(lines_un) == 8
    assert len(lines_res) == 8
    for ru, rr in zip(lines_un, lines_res):
        assert ru["step"] == rr["step"]
        assert ru["supervised_loss"] == pytest.approx(rr["supervised_loss"], abs=1e-12)
        assert ru["body_lr"] == pytest.approx(rr["body_lr"], abs=1e-12)
        assert ru["head_lr"] == pytest.approx(rr["head_lr"], abs=1e-12)
        assert ru["next_body_lr"] == pytest.approx(rr["next_body_lr"], abs=1e-12)
        assert ru["next_head_lr"] == pytest.approx(rr["next_head_lr"], abs=1e-12)
        assert ru["scheduler_updated"] == rr["scheduler_updated"]
        if ru["monitored_metric"] is not None:
            assert ru["monitored_metric"] == pytest.approx(rr["monitored_metric"], abs=1e-12)


def test_old_defaults_unchanged(tmp_path):
    """Old default configs without supervised_loss or lr_scheduler remain 100% backward compatible."""
    c = copy.deepcopy(DEFAULT)
    assert "supervised_loss" not in c
    assert "lr_scheduler" not in c
    validated = validate_config(c)
    assert "supervised_loss" not in validated
    assert "lr_scheduler" not in validated

    manifest, sha = synthetic_fixture(tmp_path / "data")
    c["model"]["backbone"] = "tiny_test"
    c["steps"] = 3
    c["warmup_steps"] = 1

    out = tmp_path / "run_default"
    rep = train(c, manifest, sha, out, device="cpu", synthetic=True)
    assert rep["status"] == "COMPLETED"

    final_payload = torch.load(out / "final/weights.pt", weights_only=True)
    assert final_payload["lr_scheduler"] is None

    final_manifest = json.loads((out / "final/MANIFEST.json").read_text())
    assert final_manifest["lr_scheduler"] is None

    lines = [json.loads(x) for x in (out / "steps.jsonl").read_text().splitlines() if x.strip()]
    assert "body_lr" not in lines[0]
    assert "lr_scheduler" not in lines[0]
