"""Independent AMP / RNG / resume runtime review regressions (parallel Claude review).

Scope of this file is deliberately narrow and complements
``tests/test_runtime_checkpoint.py`` instead of duplicating it:

- An **enabled** ``torch.amp.GradScaler`` on CPU, so a non-empty scaler
  ``state_dict`` actually travels through
  ``save_checkpoint``/``load_checkpoint`` under ``weights_only=True``.
  On the canonical bf16 configuration ``build_grad_scaler`` always returns a
  *disabled* scaler, so the scaler leg of the checkpoint contract is otherwise
  never exercised anywhere in the suite.
- The torch 2.4.1 contract that a disabled scaler serializes to ``{}`` and that
  an *enabled* scaler refuses to load ``{}``; that pair is the root cause of the
  cross-device resume failure documented in the accompanying review report.
- The CUDA RNG capture/restore branches, driven through mocks so that the
  device-count, non-tensor, missing-state and CPU-staging branches run on a
  CPU-only machine. ``test_runtime_checkpoint.py::test_cuda_rng_state_handling``
  only reaches the "CUDA unavailable" branch locally, and its
  ``does not match available devices`` regex no longer matches the message the
  implementation raises.
- Checkpoint payloads are staged on CPU on the *write* side, which is what makes
  a CPU ``map_location`` on the read side sufficient.

No production code is modified or asserted-as-buggy here: every assertion is a
contract that must keep holding after the Wave 4 best/last and resume changes.

Reference environment: Python 3.10.21 / torch 2.4.1 (CPU).
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from src.self_audit.training._utils import (
    _restore_rng_state,
    _rng_state,
    autocast_context,
    build_grad_scaler,
    load_checkpoint,
    resolve_amp,
    save_checkpoint,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
) -> None:
    """Take one real optimizer step so optimizer/scheduler/scaler state is non-empty."""

    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(8, 4)).pow(2).mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()


def _enabled_cpu_scaler() -> Any:
    """An *enabled* CPU GradScaler.

    ``build_grad_scaler`` intentionally never returns this (it enables scaling
    only for CUDA + float16), which is exactly why the non-empty scaler payload
    needs its own regression here.
    """

    return torch.amp.GradScaler("cpu", enabled=True)


# ---------------------------------------------------------------------------
# 1. Non-empty optimizer + scheduler + scaler checkpoint roundtrip
# ---------------------------------------------------------------------------

def test_enabled_cpu_scaler_full_state_roundtrips_under_weights_only(tmp_path: Path) -> None:
    """A non-empty optimizer/scheduler/scaler checkpoint survives a weights_only roundtrip."""

    torch.manual_seed(0)
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = _enabled_cpu_scaler()

    assert scaler.is_enabled()
    _train_one_step(model, optimizer, scheduler, scaler)

    saved_scaler = dict(scaler.state_dict())
    saved_scheduler = dict(scheduler.state_dict())
    # Guard the premise of this test: all three payloads must be non-empty.
    assert saved_scaler, "enabled CPU GradScaler must expose a non-empty state_dict"
    assert optimizer.state_dict()["state"], "optimizer must carry per-parameter state"
    saved_exp_avg = optimizer.state_dict()["state"][0]["exp_avg"].clone()

    path = save_checkpoint(
        tmp_path / "full.pt",
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
        global_step=1,
        optimizer_step=1,
    )

    # The file itself must be safe-loadable without any pickle fallback.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert {"model", "optimizer", "scheduler", "scaler", "rng_state"} <= set(raw)
    assert raw["scaler"] == saved_scaler
    # Optimizer integer state keys must not be stringified by normalization.
    assert list(raw["optimizer"]["state"]) == [0, 1, 2, 3]
    assert all(isinstance(key, int) for key in raw["optimizer"]["state"])

    target = _TinyModel()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1, gamma=0.5)
    target_scaler = _enabled_cpu_scaler()

    load_checkpoint(
        path,
        model=target,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        scaler=target_scaler,
    )

    assert target_scaler.state_dict() == saved_scaler
    assert target_scheduler.state_dict() == saved_scheduler
    assert target_scheduler.get_last_lr() == scheduler.get_last_lr()
    restored_exp_avg = target_optimizer.state_dict()["state"][0]["exp_avg"]
    assert torch.equal(restored_exp_avg, saved_exp_avg)
    for (name, expected), (_, observed) in zip(
        model.state_dict().items(), target.state_dict().items()
    ):
        assert torch.equal(expected, observed), f"model tensor {name} did not roundtrip"

    # A second step from the restored state must match the uninterrupted step.
    _train_one_step(model, optimizer, scheduler, scaler)
    _train_one_step(target, target_optimizer, target_scheduler, target_scaler)
    assert target_scaler.state_dict() == scaler.state_dict()
    for name, expected in model.state_dict().items():
        assert torch.equal(expected, target.state_dict()[name]), f"divergence after resume in {name}"


def test_saved_checkpoint_stages_every_training_tensor_on_cpu(tmp_path: Path) -> None:
    """Model/optimizer/scaler tensors are written to disk on CPU, whatever the live device."""

    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = _enabled_cpu_scaler()
    _train_one_step(model, optimizer, scheduler, scaler)

    path = save_checkpoint(tmp_path / "cpu_staged.pt", model, optimizer=optimizer, scaler=scaler)
    raw = torch.load(path, map_location="cpu", weights_only=True)

    def _assert_cpu(value: Any, where: str) -> None:
        if torch.is_tensor(value):
            assert value.device.type == "cpu", f"{where} was not staged on CPU"
        elif isinstance(value, dict):
            for key, child in value.items():
                _assert_cpu(child, f"{where}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _assert_cpu(child, f"{where}[{index}]")

    for section in ("model", "optimizer", "scaler", "rng_state"):
        _assert_cpu(raw[section], section)


# ---------------------------------------------------------------------------
# 2. Disabled-scaler payload versus an enabled scaler on resume (torch contract)
# ---------------------------------------------------------------------------

def test_disabled_scaler_serializes_empty_and_enabled_scaler_refuses_it(tmp_path: Path) -> None:
    """The disabled/enabled scaler asymmetry that breaks a cross-device resume.

    ``build_grad_scaler`` returns a *disabled* scaler for every non-CUDA device
    and for bfloat16, and a disabled scaler's ``state_dict`` is ``{}``.  The
    checkpoint therefore stores ``scaler == {}``, which is present and not
    ``None`` -- so an "is the scaler state missing?" guard accepts it -- while
    torch's *enabled* ``load_state_dict`` rejects an empty source outright.
    The mirror case silently drops the scale and growth tracker.
    """

    disabled = build_grad_scaler(enabled=True, device=torch.device("cpu"), dtype=torch.bfloat16)
    assert not disabled.is_enabled()
    assert disabled.state_dict() == {}

    model = _TinyModel()
    path = save_checkpoint(tmp_path / "disabled_scaler.pt", model, scaler=disabled)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    # Present, not None, and empty: the shape that defeats a presence-only check.
    assert "scaler" in raw and raw["scaler"] is not None and raw["scaler"] == {}

    with pytest.raises(RuntimeError, match="source state dict is empty"):
        _enabled_cpu_scaler().load_state_dict(raw["scaler"])

    # Mirror direction: a disabled scaler swallows a populated payload silently.
    populated = _enabled_cpu_scaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    _train_one_step(model, optimizer, scheduler, populated)
    assert populated.state_dict()["_growth_tracker"] == 1

    disabled_target = build_grad_scaler(
        enabled=True, device=torch.device("cpu"), dtype=torch.bfloat16
    )
    disabled_target.load_state_dict(populated.state_dict())
    assert disabled_target.state_dict() == {}, "disabled scaler discards restored scale silently"


# ---------------------------------------------------------------------------
# 3. Mocked CUDA RNG capture / restore branches
# ---------------------------------------------------------------------------

def _fake_cuda_states(count: int) -> list[torch.Tensor]:
    return [
        torch.arange(16 * index, 16 * (index + 1), dtype=torch.uint8) for index in range(count)
    ]


def test_cuda_rng_capture_stages_states_on_cpu_and_preserves_values() -> None:
    """``_rng_state`` copies every CUDA generator state to CPU without mutating it."""

    states = _fake_cuda_states(2)
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "get_rng_state_all", return_value=[tensor.clone() for tensor in states]
    ):
        captured = _rng_state()

    assert len(captured["cuda"]) == 2
    for expected, observed in zip(states, captured["cuda"]):
        assert observed.device.type == "cpu"
        assert torch.equal(expected, observed)


def test_cuda_rng_restore_passes_cpu_tensors_for_matching_device_count() -> None:
    """Restore hands ``set_rng_state_all`` exactly one CPU tensor per available device."""

    states = _fake_cuda_states(2)
    received: list[list[torch.Tensor]] = []

    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "device_count", return_value=2
    ), patch.object(torch.cuda, "set_rng_state_all", side_effect=received.append):
        _restore_rng_state({"cuda": [tensor.clone() for tensor in states]}, exact_cuda=True)

    assert len(received) == 1
    delivered = received[0]
    assert len(delivered) == 2
    for expected, observed in zip(states, delivered):
        assert observed.device.type == "cpu"
        assert torch.equal(expected, observed)


@pytest.mark.parametrize(
    ("saved", "devices"),
    [(2, 1), (1, 2), (0, 1)],
)
def test_cuda_rng_restore_refuses_wrong_state_count(saved: int, devices: int) -> None:
    """A state/device count mismatch is refused, naming both counts, before any restore."""

    def _must_not_restore(_states: Any) -> None:  # pragma: no cover - guard
        raise AssertionError("set_rng_state_all must not run on a count mismatch")

    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "device_count", return_value=devices
    ), patch.object(torch.cuda, "set_rng_state_all", side_effect=_must_not_restore):
        with pytest.raises(ValueError) as excinfo:
            _restore_rng_state({"cuda": _fake_cuda_states(saved)}, exact_cuda=True)

    message = str(excinfo.value)
    assert str(saved) in message and str(devices) in message, message
    assert "CUDA" in message.upper()


def test_cuda_rng_restore_refuses_non_tensor_and_non_sequence_states() -> None:
    """Malformed CUDA state containers are rejected with the offending type named."""

    def _must_not_restore(_states: Any) -> None:  # pragma: no cover - guard
        raise AssertionError("set_rng_state_all must not run on malformed state")

    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "device_count", return_value=1
    ), patch.object(torch.cuda, "set_rng_state_all", side_effect=_must_not_restore):
        with pytest.raises(TypeError, match="must be a Tensor"):
            _restore_rng_state({"cuda": ["not-a-tensor"]})
        with pytest.raises(TypeError, match="list or tuple"):
            _restore_rng_state({"cuda": {"device0": torch.zeros(8, dtype=torch.uint8)}})
        with pytest.raises(ValueError, match="cannot be None"):
            _restore_rng_state({"cuda": None})


def test_cuda_rng_restore_exact_requires_present_cuda_state() -> None:
    """Exact GPU resume refuses a checkpoint that carries no CUDA RNG state."""

    cpu_only_state = {"torch": torch.get_rng_state()}

    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "device_count", return_value=1
    ):
        with pytest.raises(ValueError, match="Missing 'cuda' RNG state"):
            _restore_rng_state(cpu_only_state, exact_cuda=True)

    # Without ``exact_cuda`` the same payload is accepted and the CUDA stream is
    # left untouched: callers that resume on GPU must therefore pass exact_cuda.
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "device_count", return_value=1
    ), patch.object(
        torch.cuda,
        "set_rng_state_all",
        side_effect=AssertionError("must not restore an absent state"),
    ):
        _restore_rng_state(cpu_only_state)


def test_cuda_rng_restore_without_cuda_reports_unavailable_for_exact_resume() -> None:
    """Requesting exact CUDA resume on a CPU-only host fails loudly, not silently."""

    with patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            _restore_rng_state({"cuda": _fake_cuda_states(1)}, exact_cuda=True)


def test_checkpoint_roundtrip_restores_cuda_rng_through_mocked_devices(tmp_path: Path) -> None:
    """A mocked-CUDA checkpoint carries CPU-staged CUDA RNG state end to end."""

    states = _fake_cuda_states(1)
    model = _TinyModel()

    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "get_rng_state_all", return_value=[tensor.clone() for tensor in states]
    ):
        path = save_checkpoint(tmp_path / "cuda_rng.pt", model)

    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert [tensor.device.type for tensor in raw["rng_state"]["cuda"]] == ["cpu"]

    received: list[list[torch.Tensor]] = []
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "device_count", return_value=1
    ), patch.object(torch.cuda, "set_rng_state_all", side_effect=received.append):
        load_checkpoint(path, model=_TinyModel(), exact_cuda=True)

    assert len(received) == 1
    assert torch.equal(received[0][0], states[0])
    assert received[0][0].device.type == "cpu"


def test_python_numpy_torch_streams_continue_exactly_after_reload(tmp_path: Path) -> None:
    """Host RNG streams resume bit-exactly, including a cached NumPy gaussian."""

    random.seed(20260910)
    np.random.seed(20260910)
    torch.manual_seed(20260910)
    # Leave a cached gaussian in both Python and NumPy so ``has_gauss`` matters.
    random.gauss(0.0, 1.0)
    float(np.random.normal())

    path = save_checkpoint(tmp_path / "streams.pt", _TinyModel())
    expected = (
        [random.gauss(0.0, 1.0) for _ in range(4)],
        [float(np.random.normal()) for _ in range(4)],
        torch.randn(4).tolist(),
    )

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    load_checkpoint(path, model=_TinyModel())

    observed = (
        [random.gauss(0.0, 1.0) for _ in range(4)],
        [float(np.random.normal()) for _ in range(4)],
        torch.randn(4).tolist(),
    )
    assert observed == expected


# ---------------------------------------------------------------------------
# 4. Target bf16 AMP API surface, checked against the local torch 2.4.1 APIs
# ---------------------------------------------------------------------------

def test_bf16_amp_resolution_and_scaler_policy_on_cpu() -> None:
    """The bf16 AMP path used by the canonical configs resolves without a scaler.

    This documents the API contract only. Nothing here measures GPU behaviour;
    it asserts that on torch 2.4.1 bfloat16 autocast is available for the
    configured device type and that gradient scaling stays disabled for
    bfloat16, which is the invariant the CUDA target relies on.
    """

    cpu = torch.device("cpu")
    assert resolve_amp({"amp": True, "amp_dtype": "bfloat16"}, cpu) == (True, torch.bfloat16)
    assert resolve_amp({"amp": "auto", "amp_dtype": "bfloat16"}, cpu) == (False, torch.bfloat16)
    with pytest.raises(ValueError, match="CPU AMP requires amp_dtype=bfloat16"):
        resolve_amp({"amp": True, "amp_dtype": "float16"}, cpu)

    scaler = build_grad_scaler(enabled=True, device=cpu, dtype=torch.bfloat16)
    assert not scaler.is_enabled(), "bfloat16 must never enable gradient scaling"

    model = _TinyModel()
    with autocast_context(enabled=True, device=cpu, dtype=torch.bfloat16):
        out = model(torch.randn(2, 4))
    assert out.dtype is torch.bfloat16
    assert model(torch.randn(2, 4)).dtype is torch.float32
