"""Comprehensive tests for Wave 4 runtime portability and shared loader hardening.

Covers:
- Canonical nonpersistent worker settings for both configs (self_audit_full and self_audit_full_mnms)
- validate_checkpoint_finite_state API contract (reusable for direct resume)
- Finite-state negative cases: model, optimizer, scheduler, scaler tensors and numeric scalars
- Fail-closed behavior on load_checkpoint (no sanitization)
- CPU staging of weights_only checkpoint payloads in bind_evaluation_checkpoint and bind_existing_evaluation_state
- Preserving model target device without GPU allocation of unused optimizer/RNG state
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from self_audit.provenance import state_digest
from self_audit.training._utils import (
    _finite_tree,
    bind_evaluation_checkpoint,
    bind_existing_evaluation_state,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_finite_state,
)
from self_audit.training.unified_config import load_unified_config


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# 1. Canonical nonpersistent worker configurations
# ---------------------------------------------------------------------------

def test_canonical_configs_nonpersistent_workers() -> None:
    """Both self_audit_full and self_audit_full_mnms must configure nonpersistent workers.

    This ensures exact augmented-worker resume is feasible without failing strict
    execution-config comparison or dataloader worker RNG restrictions, while preserving
    data distribution, prefetch factor, and pin_memory settings.
    """
    for config_path in ("configs/self_audit_full.yaml", "configs/self_audit_full_mnms.yaml"):
        cfg = load_unified_config(config_path)
        dl = cfg.dataset.dataloader
        assert dl.persistent_workers is False, f"{config_path} persistent_workers must be False"
        assert dl.num_workers == 2, f"{config_path} num_workers must be 2"
        assert dl.pin_memory is True, f"{config_path} pin_memory must be True"
        assert dl.prefetch_factor == 2, f"{config_path} prefetch_factor must be 2"


# ---------------------------------------------------------------------------
# 2. validate_checkpoint_finite_state API contract
# ---------------------------------------------------------------------------

def test_validate_checkpoint_finite_state_valid_payload() -> None:
    """Valid payload with model, optimizer, scheduler, scaler passes cleanly."""
    model = _ToyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    scaler = torch.amp.GradScaler("cpu", enabled=True)

    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
    }
    # Must not raise
    validate_checkpoint_finite_state(payload)


def test_validate_checkpoint_finite_state_non_mapping_raises() -> None:
    """validate_checkpoint_finite_state requires a mapping payload."""
    with pytest.raises(ValueError, match="Checkpoint payload must be a mapping"):
        validate_checkpoint_finite_state("not_a_mapping")  # type: ignore[arg-type]


def test_validate_checkpoint_finite_state_accepts_nn_module() -> None:
    """validate_checkpoint_finite_state accepts nn.Module directly under 'model'."""
    model = _ToyModel()
    validate_checkpoint_finite_state({"model": model})

    # When model has NaN in a parameter, it must fail
    with torch.no_grad():
        model.fc1.weight[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Non-finite"):
        validate_checkpoint_finite_state({"model": model})


# ---------------------------------------------------------------------------
# 3. Finite-state negative cases: Model tensors
# ---------------------------------------------------------------------------

def test_finite_validation_model_nan_tensor(tmp_path: Path) -> None:
    """NaN tensor in model state strictly raises FloatingPointError."""
    model = _ToyModel()
    with torch.no_grad():
        model.fc1.weight[0, 0] = float("nan")

    # In reusable validator
    with pytest.raises(FloatingPointError, match="Non-finite tensor in model"):
        validate_checkpoint_finite_state({"model": model.state_dict()})

    # In load_checkpoint (even without model passed)
    clean_model = _ToyModel()
    ckpt_path = tmp_path / "nan_model.pt"
    # Construct raw payload
    raw_payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "epoch": 1,
        "global_step": 10,
        "optimizer_step": 10,
    }
    torch.save(raw_payload, ckpt_path)
    with pytest.raises(FloatingPointError, match="Non-finite tensor in model"):
        load_checkpoint(ckpt_path)


def test_finite_validation_model_inf_tensor(tmp_path: Path) -> None:
    """Inf tensor in model state strictly raises FloatingPointError."""
    model = _ToyModel()
    with torch.no_grad():
        model.fc2.bias[0] = float("inf")

    with pytest.raises(FloatingPointError, match="Non-finite tensor in model"):
        validate_checkpoint_finite_state({"model": model.state_dict()})

    ckpt_path = tmp_path / "inf_model.pt"
    torch.save(
        {"format_version": 1, "model": model.state_dict(), "epoch": 0, "global_step": 0, "optimizer_step": 0},
        ckpt_path,
    )
    with pytest.raises(FloatingPointError, match="Non-finite tensor in model"):
        load_checkpoint(ckpt_path)


# ---------------------------------------------------------------------------
# 4. Finite-state negative cases: Optimizer tensors and numeric scalars
# ---------------------------------------------------------------------------

def test_finite_validation_optimizer_nan_tensor(tmp_path: Path) -> None:
    """NaN tensor in optimizer state dictionary strictly raises FloatingPointError."""
    model = _ToyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()

    opt_state = opt.state_dict()
    # Inject NaN into an Adam momentum buffer
    first_param_id = next(iter(opt_state["state"].keys()))
    opt_state["state"][first_param_id]["exp_avg"][0, 0] = float("nan")

    payload = {"model": model.state_dict(), "optimizer": opt_state}
    with pytest.raises(FloatingPointError, match="Non-finite tensor in optimizer"):
        validate_checkpoint_finite_state(payload)

    ckpt_path = tmp_path / "nan_opt_tensor.pt"
    torch.save({"format_version": 1, "model": model.state_dict(), "optimizer": opt_state}, ckpt_path)
    # Rejection occurs even when optimizer=None
    with pytest.raises(FloatingPointError, match="Non-finite tensor in optimizer"):
        load_checkpoint(ckpt_path)


def test_finite_validation_optimizer_nan_lr_scalar(tmp_path: Path) -> None:
    """NaN float in optimizer param_groups strictly raises FloatingPointError."""
    model = _ToyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt_state = opt.state_dict()
    opt_state["param_groups"][0]["lr"] = float("nan")

    payload = {"model": model.state_dict(), "optimizer": opt_state}
    with pytest.raises(FloatingPointError, match="Non-finite value in optimizer"):
        validate_checkpoint_finite_state(payload)

    ckpt_path = tmp_path / "nan_opt_lr.pt"
    torch.save({"format_version": 1, "model": model.state_dict(), "optimizer": opt_state}, ckpt_path)
    with pytest.raises(FloatingPointError, match="Non-finite value in optimizer"):
        load_checkpoint(ckpt_path)


def test_finite_validation_optimizer_inf_eps_scalar(tmp_path: Path) -> None:
    """Inf float in optimizer param_groups strictly raises FloatingPointError."""
    model = _ToyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
    opt_state = opt.state_dict()
    opt_state["param_groups"][0]["eps"] = float("inf")

    payload = {"model": model.state_dict(), "optimizer": opt_state}
    with pytest.raises(FloatingPointError, match="Non-finite value in optimizer"):
        validate_checkpoint_finite_state(payload)


def test_finite_validation_optimizer_nan_tuple_scalar() -> None:
    """NaN in tuple element within optimizer param_groups (e.g. betas) strictly raises."""
    model = _ToyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_state = opt.state_dict()
    opt_state["param_groups"][0]["betas"] = (float("nan"), 0.999)

    payload = {"model": model.state_dict(), "optimizer": opt_state}
    with pytest.raises(FloatingPointError, match="Non-finite value in optimizer"):
        validate_checkpoint_finite_state(payload)


# ---------------------------------------------------------------------------
# 5. Finite-state negative cases: Scheduler and Scaler numeric scalars
# ---------------------------------------------------------------------------

def test_finite_validation_scheduler_nan_scalar(tmp_path: Path) -> None:
    """NaN numeric scalar in scheduler state strictly raises FloatingPointError."""
    model = _ToyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    sched_state = sched.state_dict()
    sched_state["gamma"] = float("nan")

    payload = {"model": model.state_dict(), "scheduler": sched_state}
    with pytest.raises(FloatingPointError, match="Non-finite value in scheduler"):
        validate_checkpoint_finite_state(payload)

    ckpt_path = tmp_path / "nan_sched.pt"
    torch.save({"format_version": 1, "model": model.state_dict(), "scheduler": sched_state}, ckpt_path)
    with pytest.raises(FloatingPointError, match="Non-finite value in scheduler"):
        load_checkpoint(ckpt_path)


def test_finite_validation_scheduler_inf_base_lrs(tmp_path: Path) -> None:
    """Inf numeric scalar in scheduler base_lrs list strictly raises FloatingPointError."""
    model = _ToyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    sched_state = sched.state_dict()
    sched_state["base_lrs"] = [float("inf")]

    payload = {"model": model.state_dict(), "scheduler": sched_state}
    with pytest.raises(FloatingPointError, match="Non-finite value in scheduler"):
        validate_checkpoint_finite_state(payload)


def test_finite_validation_scaler_nan_scale(tmp_path: Path) -> None:
    """NaN scale scalar in GradScaler state strictly raises FloatingPointError."""
    model = _ToyModel()
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    scaler_state = scaler.state_dict()
    scaler_state["scale"] = float("nan")

    payload = {"model": model.state_dict(), "scaler": scaler_state}
    with pytest.raises(FloatingPointError, match="Non-finite value in scaler"):
        validate_checkpoint_finite_state(payload)

    ckpt_path = tmp_path / "nan_scaler.pt"
    torch.save({"format_version": 1, "model": model.state_dict(), "scaler": scaler_state}, ckpt_path)
    with pytest.raises(FloatingPointError, match="Non-finite value in scaler"):
        load_checkpoint(ckpt_path)


def test_finite_validation_scaler_inf_growth_factor() -> None:
    """Inf growth_factor in GradScaler state strictly raises FloatingPointError."""
    model = _ToyModel()
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    scaler_state = scaler.state_dict()
    scaler_state["growth_factor"] = float("inf")

    payload = {"model": model.state_dict(), "scaler": scaler_state}
    with pytest.raises(FloatingPointError, match="Non-finite value in scaler"):
        validate_checkpoint_finite_state(payload)


# ---------------------------------------------------------------------------
# 6. _finite_tree coverage: ndarray and complex numbers
# ---------------------------------------------------------------------------

def test_finite_tree_numpy_ndarray_nan() -> None:
    """_finite_tree correctly detects NaN inside np.ndarray."""
    arr = np.array([1.0, np.nan, 3.0])
    with pytest.raises(FloatingPointError, match="Non-finite ndarray in test.arr"):
        _finite_tree(arr, "test.arr")


def test_finite_tree_complex_nan() -> None:
    """_finite_tree correctly detects NaN inside complex numbers."""
    c = complex(1.0, float("nan"))
    with pytest.raises(FloatingPointError, match="Non-finite value in test.c"):
        _finite_tree(c, "test.c")


# ---------------------------------------------------------------------------
# 7. CPU staging in bind_evaluation_checkpoint & bind_existing_evaluation_state
# ---------------------------------------------------------------------------

def test_bind_evaluation_checkpoint_cpu_staging(tmp_path: Path) -> None:
    """bind_evaluation_checkpoint stages full weights_only payload on CPU.

    Verifies:
    - load_checkpoint is called with map_location="cpu"
    - Model weights are loaded strictly and model remains on target device
    - Binding returns valid digest and CheckpointBinding
    """
    model = _ToyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    ckpt_path = tmp_path / "phase_c_best.pt"
    save_checkpoint(ckpt_path, model, optimizer=opt, scaler=scaler, epoch=1, global_step=5)

    eval_model = _ToyModel()
    # Mutate eval_model to verify it gets updated by binding
    with torch.no_grad():
        eval_model.fc1.weight.add_(1.0)
    assert state_digest(eval_model) != state_digest(model)

    load_calls: list[dict[str, Any]] = []
    original_load = load_checkpoint

    def tracked_load(*args: Any, **kwargs: Any) -> dict[str, Any]:
        load_calls.append({"args": args, "kwargs": kwargs})
        return original_load(*args, **kwargs)

    with patch("self_audit.training._utils.load_checkpoint", side_effect=tracked_load):
        binding = bind_evaluation_checkpoint(
            eval_model,
            [("best", ckpt_path)],
            map_location=torch.device("cuda:0"),
        )

    # Verify staging on CPU was requested despite target device argument
    assert len(load_calls) == 1
    assert load_calls[0]["kwargs"].get("map_location") == "cpu"
    assert load_calls[0]["kwargs"].get("restore_rng") is False
    assert all(p.device.type == "cpu" for p in eval_model.parameters())

    # Verify model now matches checkpoint
    assert state_digest(eval_model) == state_digest(model)
    assert binding.state_digest == state_digest(model)
    assert binding.restored == ("model",)
    assert binding.fallback_used is False


def test_bind_existing_evaluation_state_cpu_staging(tmp_path: Path) -> None:
    """bind_existing_evaluation_state stages full weights_only payload on CPU without mutating model.

    Verifies:
    - load_checkpoint is called with map_location="cpu" despite target device argument
    - Model is untouched and restored=()
    - Live digest equals checkpoint digest
    """
    model = _ToyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "phase_c_best.pt"
    save_checkpoint(ckpt_path, model, optimizer=opt, epoch=2, global_step=20)

    load_calls: list[dict[str, Any]] = []
    original_load = load_checkpoint

    def tracked_load(*args: Any, **kwargs: Any) -> dict[str, Any]:
        load_calls.append({"args": args, "kwargs": kwargs})
        return original_load(*args, **kwargs)

    with patch("self_audit.training._utils.load_checkpoint", side_effect=tracked_load):
        binding = bind_existing_evaluation_state(
            model,
            [("best", ckpt_path)],
            map_location=torch.device("cuda:0"),
        )

    assert len(load_calls) == 1
    assert load_calls[0]["kwargs"].get("map_location") == "cpu"
    assert load_calls[0]["kwargs"].get("restore_rng") is False
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert binding.restored == ()
    assert binding.state_digest == state_digest(model)


def test_bind_evaluation_checkpoint_preserves_target_device_request(tmp_path: Path) -> None:
    """Explicit target device specification via map_location is preserved on model."""
    model = _ToyModel()
    ckpt_path = tmp_path / "best.pt"
    save_checkpoint(ckpt_path, model)

    eval_model = _ToyModel()
    # If caller specifies map_location="cpu", model remains on CPU
    binding = bind_evaluation_checkpoint(
        eval_model,
        [("best", ckpt_path)],
        map_location="cpu",
    )
    for p in eval_model.parameters():
        assert p.device.type == "cpu"
    assert binding.state_digest == state_digest(model)


# ---------------------------------------------------------------------------
# 8. load_checkpoint entrypoint map_location
# ---------------------------------------------------------------------------

def test_load_checkpoint_map_location_cpu(tmp_path: Path) -> None:
    """Direct entrypoint load_checkpoint with map_location='cpu' returns CPU tensors."""
    model = _ToyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer=opt)

    loaded = load_checkpoint(ckpt_path, map_location="cpu")
    for key, val in loaded["model"].items():
        assert val.device.type == "cpu"
    if "optimizer" in loaded and "state" in loaded["optimizer"]:
        for state_item in loaded["optimizer"]["state"].values():
            for k, v in state_item.items():
                if torch.is_tensor(v):
                    assert v.device.type == "cpu"
