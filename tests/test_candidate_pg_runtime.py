import os

import pytest
import torch

from self_audit.candidate_pg.runtime import configure_runtime


@pytest.mark.parametrize("device,warn_only,status", [
    ("cpu", False, "STRICT_CPU"),
    ("cuda:0", True, "BEST_EFFORT_GPU_WARN_ONLY"),
])
def test_runtime_matches_fuse_policy_without_initializing_cuda(monkeypatch, device, warn_only, status):
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda enabled, **kw: calls.append((enabled, kw)))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "test-before")
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    flags = configure_runtime(device, ":4096:8")
    assert calls == [(True, {"warn_only": warn_only})]
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert flags["determinism_status"] == status
    assert flags["deterministic_warn_only"] is warn_only
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cuda.matmul.allow_tf32
