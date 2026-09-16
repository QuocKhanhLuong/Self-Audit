"""Exact norm arithmetic, timing boundary policy, and journal crash safety."""
import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from self_audit_maskfree.runtime import TimingAccumulator, global_grad_norm, append_jsonl, JSONLWriter

@pytest.mark.parametrize("mode,count", [("production", 0), ("diagnostic", 2)])
def test_cuda_timing_policy_without_cuda_hardware(mode, count):
    acc = TimingAccumulator.empty(mode=mode)
    acc.device = torch.device("cuda:0")
    with patch("torch.cuda.synchronize") as sync:
        with pytest.raises(ValueError):
            with acc.stage("failing_stage"):
                raise ValueError("deliberate")
    assert sync.call_count == count
    assert acc.calls["failing_stage"] == 1
    assert acc.as_dict()["cuda_synchronized"] == bool(count)


def test_invalid_timing_mode_rejected():
    with pytest.raises(ValueError):
        TimingAccumulator.empty(mode="typo")


def test_global_grad_norm_matches_original_loop_exactly():
    gen = torch.Generator().manual_seed(21)
    params = [SimpleNamespace(grad=torch.randn(size, generator=gen) * scale)
              for size, scale in [(23, 1e-9), (65, 1e3), (31, 0.1), (5, 7)]]
    params.insert(2, SimpleNamespace(grad=None))
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm(2).item() ** 2)
    assert global_grad_norm(iter(params)) == float(total ** 0.5)
    assert global_grad_norm([]) == 0.0
    assert not torch.isfinite(torch.tensor(global_grad_norm([SimpleNamespace(grad=torch.tensor([float("nan")]))])))


def test_writer_exact_bytes_order_and_bound(tmp_path):
    old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    rows = [{"index": i, "text": "é" * (80 if i == 4 else 2)} for i in range(9)]
    writer = JSONLWriter(new, max_resident_bytes=100)
    for row in rows:
        append_jsonl(old, row)
        writer.append(row)
        assert writer.buffer_bytes <= 100
    with patch("os.fsync", wraps=__import__("os").fsync) as sync:
        offset = writer.flush()
    assert sync.call_count == 2  # file plus newly created directory entry
    assert offset == new.stat().st_size
    assert old.read_bytes() == new.read_bytes()
    writer.close()
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.append({"lost": True})


def test_failed_fsync_poisoned_writer_cannot_commit_or_replay(tmp_path):
    path = tmp_path / "lineage.jsonl"
    writer = JSONLWriter(path)
    writer.append({"record": 1})
    with patch("os.fsync", side_effect=OSError("disk failure")):
        with pytest.raises(OSError, match="disk failure"):
            writer.flush()
    written = path.read_bytes()
    with pytest.raises(RuntimeError, match="failed"):
        writer.flush()
    writer.close()
    assert writer.handle.closed and path.read_bytes() == written


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_writer_rejects_invalid_bound(tmp_path, value):
    with pytest.raises(ValueError):
        JSONLWriter(tmp_path / "invalid.jsonl", max_resident_bytes=value)
