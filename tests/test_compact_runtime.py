import os

import pytest
import torch

from self_audit.compact_runtime import _DeterministicBilinear, compact_bilinear, deterministic_runtime
from self_audit.compact_training import DEFAULT_CONFIG, validate_compact_config


def flags():
    return (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


def test_default_leaves_runtime_and_rng_untouched():
    before, rng = flags(), torch.get_rng_state().clone()
    with deterministic_runtime():
        assert flags() == before
    assert flags() == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_enabled_is_strict_and_restores_on_failure(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    before, rng = flags(), torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="sentinel"):
        with deterministic_runtime(True):
            assert flags() == (True, False, True, False, ":4096:8")
            raise RuntimeError("sentinel")
    assert flags() == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_late_cuda_setup_fails_closed(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        with deterministic_runtime(True):
            pytest.fail("late configuration must not run")


@pytest.mark.parametrize("value", [1, "true", None])
def test_config_rejects_non_boolean_runtime(value):
    with pytest.raises(ValueError, match="deterministic_algorithms"):
        validate_compact_config({**DEFAULT_CONFIG, "deterministic_algorithms": value})


def test_conflicting_workspace_rejected(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="4096"):
        with deterministic_runtime(True):
            pytest.fail("conflicting policy must not run")


@pytest.mark.parametrize("shape,size", [((2, 3, 5, 7), (10, 14)), ((1, 4, 3, 4), (7, 9))])
def test_bilinear_backward_matches_reference_exactly_on_cpu(shape, size):
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    ref = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
    actual = _DeterministicBilinear.apply(x, size)
    grad = torch.randn_like(ref)
    assert torch.equal(ref, actual)
    expected_gradient = torch.autograd.grad(ref, x, grad)[0]
    actual_gradient = torch.autograd.grad(actual, x, grad)[0]
    assert torch.equal(expected_gradient, actual_gradient)
    assert torch.autograd.gradcheck(lambda v: _DeterministicBilinear.apply(v, size), (x,))


def test_default_bilinear_is_unchanged():
    x = torch.randn(1, 2, 3, 4, requires_grad=True)
    reference = torch.nn.functional.interpolate(x, size=(6, 8), mode="bilinear", align_corners=False)
    actual = compact_bilinear(x, (6, 8))
    assert torch.equal(reference, actual)
    assert type(reference.grad_fn) is type(actual.grad_fn)
