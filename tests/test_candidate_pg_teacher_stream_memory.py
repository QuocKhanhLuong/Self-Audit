"""S2 stream scheduling: combined-objective parity and checkpoint continuity."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from self_audit.candidate_pg import teacher_training as training
from tests.test_candidate_pg_teacher_training import _TinyTeacher, _config, _frame


class _GraphLifetime(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, counters):
        ctx.counters = counters
        counters["live"] += 1
        counters["peak"] = max(counters["peak"], counters["live"])
        return value

    @staticmethod
    def backward(ctx, gradient):
        ctx.counters["live"] -= 1
        return gradient, None


class _TrackedTeacher(_TinyTeacher):
    def __init__(self, *, bf16=False):
        super().__init__()
        self.bf16 = bf16
        self.counters = {"live": 0, "peak": 0}
        with torch.no_grad():
            self.region.bias[1] = 4.0  # Ensure active confident U losses.
        self.body.register_forward_hook(self._mark_graph)

    def _mark_graph(self, _module, _args, output):
        return _GraphLifetime.apply(output, self.counters) if torch.is_grad_enabled() else output

    def forward(self, *args, **kwargs):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.bf16):
            return super().forward(*args, **kwargs)


def _data():
    return {
        "labeled": [_frame(0, labeled=True, patient="L0", shape=(2, 64, 64))],
        "unlabeled": [_frame(1, labeled=False, patient="U0", shape=(2, 64, 64)), _frame(2, labeled=False, patient="U1", shape=(2, 64, 64))],
    }


def _equal_state(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _equal_state(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _equal_state(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("views", [1, 2])
def test_streamed_backward_matches_combined_objective(tmp_path, monkeypatch, bf16, views):
    config = _config(1)
    config["teacher"]["semi"]["lambda_u"]["warmup_updates_labeled_only"] = 0
    config["augmentation"]["n_strong_views"] = views
    data = _data()
    torch.manual_seed(19)
    initial = _TrackedTeacher(bf16=bf16).state_dict()
    original_backward = training._backward_loss
    original_clip = torch.nn.utils.clip_grad_norm_
    results = []
    for combined in (True, False):
        model = _TrackedTeacher(bf16=bf16)
        model.load_state_dict(initial)
        pending, gradients = [], []

        def defer(loss):
            pending.append(loss)

        def capture_clip(parameters, *args, **kwargs):
            parameters = list(parameters)
            if combined:
                # Independent combined-objective reference: retain every
                # weighted stream graph and perform one backward at the end.
                sum(pending).backward()
                pending.clear()
            gradients.extend(p.grad.detach().clone() if p.grad is not None else None for p in parameters)
            return original_clip(parameters, *args, **kwargs)

        monkeypatch.setattr(training, "_backward_loss", defer if combined else original_backward)
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_clip)
        out = tmp_path / str(combined)
        training.run_teacher_stage(config, out, data, data["labeled"], data["labeled"], teacher=model)
        payload = torch.load(out / "S2/teacher_latest.pt", weights_only=False)
        diag = json.loads((out / "S2/diag.jsonl").read_text())
        results.append((model, gradients, payload, diag))
    reference, streamed = results
    assert reference[0].counters == {"live": 0, "peak": views + 2}
    assert streamed[0].counters == {"live": 0, "peak": 1}
    tolerance = {"atol": 2e-3, "rtol": 1e-2} if bf16 else {"atol": 1e-6, "rtol": 1e-5}
    max_gradient_error = 0.0
    for a, b in zip(streamed[1], reference[1]):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, **tolerance)
            max_gradient_error = max(max_gradient_error, (a-b).abs().max().item())
    max_parameter_error = 0.0
    for name, value in streamed[2]["model"].items():
        expected = reference[2]["model"][name]
        torch.testing.assert_close(value, expected, atol=1e-6, rtol=1e-5)
        max_parameter_error = max(max_parameter_error, (value-expected).abs().max().item())
    for key in ("rng", "fp_generator_state", "sampler", "augmentation", "threshold", "update", "config_sha256"):
        _equal_state(streamed[2][key], reference[2][key])
    assert streamed[3]["confident_fraction"] > 0
    assert streamed[3]["loss_u"] > 0
    for key in ("loss_sup", "loss_u", "loss_total"):
        assert streamed[3][key] == reference[3][key]
    assert len(streamed[2]["optimizer"]["state"]) == len(reference[2]["optimizer"]["state"])
    for parameter, value in streamed[2]["optimizer"]["state"].items():
        assert value["step"].item() == 1
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(value[name], reference[2]["optimizer"]["state"][parameter][name], **tolerance)
    print(f"bf16={bf16} views={views} max_gradient_abs_error={max_gradient_error:.9g} max_parameter_abs_error={max_parameter_error:.9g}")


def test_resume_800_matches_uninterrupted_postwarmup_and_rows(tmp_path):
    config = _config(802)
    data = _data()
    torch.manual_seed(23)
    initial = _TrackedTeacher().state_dict()

    def model():
        result = _TrackedTeacher()
        result.load_state_dict(initial)
        return result

    def evaluate(*args, **kwargs):
        return {"fg_dice": 0.5, "per_class": {}, "per_patient": {}}

    full, resumed = tmp_path / "full", tmp_path / "resumed"
    training.run_teacher_stage(config, full, data, data["labeled"], data["labeled"], teacher=model(), evaluator=evaluate)
    training.run_teacher_stage(config, resumed, data, data["labeled"], data["labeled"], teacher=model(), evaluator=evaluate, max_updates=800)
    training.run_teacher_stage(config, resumed, data, data["labeled"], data["labeled"], teacher=model(), evaluator=evaluate, resume=True)
    a = torch.load(full / "S2/teacher_latest.pt", weights_only=False)
    b = torch.load(resumed / "S2/teacher_latest.pt", weights_only=False)
    for key in ("model", "optimizer", "rng", "fp_generator_state", "sampler", "augmentation", "threshold", "update", "config_sha256", "epoch_loss_count", "epoch_loss_sum"):
        _equal_state(a[key], b[key])
    rows = [json.loads(line) for line in (resumed / "S2/epochs.jsonl").read_text().splitlines()]
    assert len(rows) == 30
    assert len({(r["epoch"], r["split"], r["path"]) for r in rows}) == 30
    assert b["update"] == 802
    for value in b["optimizer"]["state"].values():
        assert value["step"].item() == 802


def test_nonfinite_stream_does_not_backward():
    value = torch.tensor(float("nan"), requires_grad=True)
    with pytest.raises(FloatingPointError):
        training._backward_loss(value)
    assert value.grad is None
