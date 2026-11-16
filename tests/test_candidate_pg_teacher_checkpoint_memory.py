"""CPU evidence for PG activation recomputation; no GPU capacity claim."""
from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn

from self_audit.candidate_pg.teacher_mednext import _PGAnisotropicMedNeXt, MedNeXtPGTeacher, MEDNEXT_DOWN_STRIDES
from tests.test_candidate_pg_models import _NativeMedNeXtFixture, _teacher_config, _geometry


class _ExpandedBlock(nn.Module):
    """One native-block-sized unit, with expansion hidden inside forward."""
    def __init__(self, channels=8):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
                                    nn.GroupNorm(channels, channels), nn.Conv3d(channels, channels * 8, 1),
                                    nn.GELU(), nn.Dropout(0.2), nn.Conv3d(channels * 8, channels, 1))

    def forward(self, value):
        return self.layers(value)


def test_checkpoint_reduces_saved_activations_with_gradient_optimizer_rng_parity():
    torch.manual_seed(31)
    block = nn.Sequential(*[_ExpandedBlock() for _ in range(8)])
    source = copy.deepcopy(block.state_dict())
    image = torch.rand(1, 8, 2, 16, 16)  # No input requires_grad: parameters must still train.
    results = []
    for recompute in (False, True):
        block.load_state_dict(source)
        optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        saved = {}
        parameter_storage = {p.untyped_storage().data_ptr() for p in block.parameters()}

        def pack(value):
            storage = value.untyped_storage()
            if storage.data_ptr() not in parameter_storage:
                saved[storage.data_ptr()] = storage.nbytes()
            return value

        torch.manual_seed(37)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            output = _PGAnisotropicMedNeXt._run_module(block, image) if recompute else block(image)
        saved_bytes = sum(saved.values())
        output.square().mean().backward()
        gradients = [p.grad.clone() for p in block.parameters()]
        rng = torch.get_rng_state().clone()
        optimizer.step()
        results.append((output.detach(), gradients, copy.deepcopy(block.state_dict()), rng, saved_bytes, copy.deepcopy(optimizer.state_dict())))
    reference, checked = results
    torch.testing.assert_close(checked[0], reference[0], atol=0, rtol=0)
    for actual, expected in zip(checked[1], reference[1]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for name in checked[2]:
        torch.testing.assert_close(checked[2][name], reference[2][name], atol=0, rtol=0)
    assert torch.equal(checked[3], reference[3])
    for key, state in checked[5]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, reference[5]["state"][key][name], atol=0, rtol=0)
    assert checked[4] < reference[4] * 0.2
    print(f"saved_nonparameter_storage_bytes: direct={reference[4]} checkpoint={checked[4]}")


@pytest.mark.parametrize("bf16", [False, True])
def test_native_teacher_fp_dropout_recompute_parity(bf16):
    torch.manual_seed(41)
    core = _PGAnisotropicMedNeXt(_NativeMedNeXtFixture(), MEDNEXT_DOWN_STRIDES)
    core.dec_block_1 = nn.Sequential(core.dec_block_1, _ExpandedBlock(64), _ExpandedBlock(64))
    teacher = MedNeXtPGTeacher(_teacher_config(), model=core).train()
    original_runner = teacher.encoder._run_module
    image = torch.rand(1, 1, 4, 16, 16)
    shape = tuple(image.shape[-3:])
    results = []
    keys = tuple(teacher.state_dict())
    initial = copy.deepcopy(teacher.state_dict())
    for recompute in (False, True):
        teacher.load_state_dict(initial, strict=True)
        teacher.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
        teacher.encoder._run_module = original_runner if recompute else lambda module, value: module(value)
        generator = torch.Generator().manual_seed(43)
        torch.manual_seed(47)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            result = teacher(image, _geometry(shape), skip_dropout=0.5, dropout_generator=generator)
        outputs = [result[name] for name in ("region_logits", "boundary_logits", "sdf_all")] + list(result["features"])
        sum(value.float().square().mean() for value in outputs).backward()
        results.append(([value.detach().clone() for value in outputs],
                        {name: p.grad.clone() if p.grad is not None else None for name, p in teacher.named_parameters()},
                        generator.get_state().clone(), torch.get_rng_state().clone()))
        optimizer.step()
        results[-1] += (copy.deepcopy(teacher.state_dict()), copy.deepcopy(optimizer.state_dict()))
    for actual, expected in zip(results[1][0], results[0][0]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for name, actual in results[1][1].items():
        expected = results[0][1][name]
        if actual is None or expected is None:
            assert actual is expected
        else:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(results[1][2], results[0][2])
    assert torch.equal(results[1][3], results[0][3])
    for name in keys:
        torch.testing.assert_close(results[1][4][name], results[0][4][name], atol=0, rtol=0)
    for key, state in results[1][5]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, results[0][5]["state"][key][name], atol=0, rtol=0)
    assert tuple(teacher.state_dict()) == keys


def test_checkpoint_bypassed_for_eval_and_no_grad(monkeypatch):
    import torch.utils.checkpoint

    def unexpected(*args, **kwargs):
        raise AssertionError("checkpoint reached outside training")

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", unexpected)
    module = nn.Linear(4, 4).eval()
    image = torch.ones(1, 4, requires_grad=True)
    _PGAnisotropicMedNeXt._run_module(module, image)
    module.train()
    with torch.no_grad():
        _PGAnisotropicMedNeXt._run_module(module, image)
    module.requires_grad_(False)
    _PGAnisotropicMedNeXt._run_module(module, image.detach())


def test_s2_oom_context_keeps_error_and_prevents_optimizer_step(tmp_path, monkeypatch):
    from self_audit.candidate_pg.teacher_training import run_teacher_stage
    from tests.test_candidate_pg_teacher_training import _TinyTeacher, _config
    from tests.test_candidate_pg_teacher_stream_memory import _data

    failure = torch.OutOfMemoryError("injected second-strong OOM")

    class FailingTeacher(_TinyTeacher):
        calls = 0

        def forward(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 5:  # weak A/B, labeled, s1, s2
                raise failure
            return super().forward(*args, **kwargs)

    def forbidden_step(*args, **kwargs):
        raise AssertionError("optimizer must not step after a failed stream")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    config = _config(1)
    config["teacher"]["semi"]["lambda_u"]["warmup_updates_labeled_only"] = 0
    data = _data()
    with pytest.raises(torch.OutOfMemoryError) as captured:
        run_teacher_stage(config, tmp_path, data, data["labeled"], data["labeled"], teacher=FailingTeacher())
    assert captured.value is failure
    record = json.loads((tmp_path / "S2/failures.jsonl").read_text())
    assert record["stream"] == "s2" and record["phase"] == "forward" and record["update"] == 1
    assert record["native_shape"] == record["padded_shape"] == [2, 64, 64]
    assert all(record[key] for key in ("labeled_key", "u_anchor_key", "u_partner_key"))
    assert not (tmp_path / "S2/teacher_latest.pt").exists()
