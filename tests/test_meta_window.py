import copy

import pytest
import torch
from torch import nn
from torch.func import functional_call

from self_audit.model import build_model
from self_audit.meta_window import (
    _safe_sqrt,
    _functional_step_with_states,
    _masked_pseudo_ce,
    _model_parameter_index,
    _present_foreground_dice,
    _snapshot_parameters,
    frozen_geometry,
    geometry_named_parameters,
    meta_geometry_gradients,
)


class _Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([0.17, -0.11]))

    def forward(self, x):
        return x * (1.0 + self.scale.view(1, 2, 1, 1))


class _ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.window = nn.Module()
        self.window.geometry = _Geometry()
        self.body = nn.Conv2d(2, 2, 1, bias=False)
        self.head = nn.Conv2d(2, 4, 1)

    def forward(self, x, valid_mask=None):
        return self.head(self.body(self.window.geometry(x)))


def _make_model_and_optimizers(seed=3, *, dtype=torch.float32):
    torch.manual_seed(seed)
    model = _ToyStudent().to(dtype=dtype)
    body = [p for n, p in model.named_parameters()
            if not n.startswith("head.") and not n.startswith("window.geometry.")]
    optimizers = [
        torch.optim.AdamW(body, lr=0.013, betas=(0.83, 0.97), eps=1e-7, weight_decay=2e-4),
        torch.optim.SGD(model.head.parameters(), lr=0.027, momentum=0.0, weight_decay=0.0),
    ]
    return model, optimizers


def _inputs(*, dtype=torch.float32):
    torch.manual_seed(7)
    strong = torch.rand(2, 2, 3, 4, dtype=dtype)
    py = torch.tensor([[[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]],
                       [[3, 2, 1, 0], [0, 1, 2, 3], [1, 0, 3, 2]]])
    valid = torch.ones_like(py, dtype=torch.bool)
    xa = torch.rand(2, 2, 3, 4, dtype=dtype)
    ya = py.clone()
    return strong, py, valid, xa, ya


def test_zero_adam_second_moment_has_finite_mixed_derivative():
    value = torch.tensor([0.0, 1.0], requires_grad=True)
    root = _safe_sqrt(value).sum()
    first = torch.autograd.grad(root, value, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), value)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert first[0] == 0 and second[0] == 0


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            _assert_nested_equal(x, y)
    else:
        assert left == right


def test_geometry_names_and_frozen_context_preserves_input_gradient():
    model, _ = _make_model_and_optimizers()
    names = geometry_named_parameters(model)
    assert list(names) == ["window.geometry.scale"]
    names["window.geometry.scale"].grad = torch.ones_like(names["window.geometry.scale"])
    inputs = torch.rand(2, 2, 3, 4, requires_grad=True)
    with frozen_geometry(model):
        assert not names["window.geometry.scale"].requires_grad
        model(inputs).sum().backward()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        assert names["window.geometry.scale"].grad is None
    assert names["window.geometry.scale"].requires_grad
    assert names["window.geometry.scale"].grad is None


def test_meta_gradient_is_finite_and_does_not_mutate_live_state():
    model, optimizers = _make_model_and_optimizers()
    strong, py, valid, xa, ya = _inputs()
    parameter_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer_before = copy.deepcopy([optimizer.state_dict() for optimizer in optimizers])
    gradients, diagnostics = meta_geometry_gradients(model, optimizers, strong, py, valid, xa, ya, clip_norm=5.)
    assert diagnostics["status"] == "ok"
    assert diagnostics["accepted_pixels"] == valid.numel()
    assert diagnostics["outer_foreground"] == [1, 2, 3]
    assert torch.isfinite(torch.tensor(diagnostics["reference_outer_dice_loss"]))
    assert torch.isfinite(torch.tensor(diagnostics["predicted_dice_loss_reduction"]))
    assert diagnostics["geometry_parameter_count"] == len(geometry_named_parameters(model))
    assert diagnostics["geometry_grad_norm"] > 0
    assert all(torch.isfinite(value).all() for value in gradients.values())
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, parameter_before[name], rtol=0, atol=0)
    for current, before in zip((optimizer.state_dict() for optimizer in optimizers), optimizer_before, strict=True):
        _assert_nested_equal(current, before)


def test_empty_supports_skip_without_a_fake_gradient():
    model, optimizers = _make_model_and_optimizers()
    strong, py, valid, xa, ya = _inputs()
    empty = torch.zeros_like(valid)
    gradients, diagnostics = meta_geometry_gradients(model, optimizers, strong, py, empty, xa, ya)
    assert diagnostics["status"] == "skip_no_pseudo_support"
    assert diagnostics["accepted_pixels"] == 0
    assert gradients == {}
    all_background = torch.zeros_like(ya)
    gradients, diagnostics = meta_geometry_gradients(model, optimizers, strong, py, valid, xa, all_background)
    assert diagnostics["status"] == "skip_no_outer_foreground"
    assert gradients == {}


@pytest.mark.parametrize("warm", [False, True])
def test_functional_adamw_sgd_and_clip_match_real_step_without_state_mutation(warm):
    model, optimizers = _make_model_and_optimizers(seed=12, dtype=torch.float64)
    strong, _, _, _, _ = _inputs(dtype=torch.float64)
    if warm:
        # Initialize AdamW's moments, so this also checks a nonzero step state.
        loss = model(strong).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    parameter_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer_before = copy.deepcopy([optimizer.state_dict() for optimizer in optimizers])
    named, geometry, owners = _model_parameter_index(model, optimizers)
    theta_names = [name for name in named if name not in geometry]
    theta = _snapshot_parameters(model, theta_names, requires_grad=None)
    torch.manual_seed(19)
    gradients = {name: 10. * torch.randn_like(value) for name, value in theta.items()}
    virtual, optimizer_diagnostics = _functional_step_with_states(theta, gradients, owners, named, clip_norm=5.)
    assert float(optimizer_diagnostics["clip_scale"]) < 1.0

    reference = copy.deepcopy(model)
    body = [p for n, p in reference.named_parameters()
            if not n.startswith("head.") and not n.startswith("window.geometry.")]
    reference_optimizers = [
        torch.optim.AdamW(body, lr=0.013, betas=(0.83, 0.97), eps=1e-7, weight_decay=2e-4),
        torch.optim.SGD(reference.head.parameters(), lr=0.027, momentum=0.0, weight_decay=0.0),
    ]
    for destination, source in zip(reference_optimizers, optimizer_before, strict=True):
        destination.load_state_dict(copy.deepcopy(source))
    for name, parameter in reference.named_parameters():
        if name in gradients:
            parameter.grad = gradients[name].clone()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 5., error_if_nonfinite=True)
    for optimizer in reference_optimizers:
        optimizer.step()
    for name, parameter in reference.named_parameters():
        if name in virtual:
            torch.testing.assert_close(virtual[name], parameter, rtol=1e-11, atol=1e-12, msg=name)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, parameter_before[name], rtol=0, atol=0)
    for current, before in zip((optimizer.state_dict() for optimizer in optimizers), optimizer_before, strict=True):
        _assert_nested_equal(current, before)


def _outer_objective_at_geometry(model, optimizers, strong, py, valid, xa, ya,
                                 geometry_value, reference_geometry, *, clip_norm=5.):
    named, geometry, owners = _model_parameter_index(model, optimizers)
    theta_names = [name for name in named if name not in geometry]
    theta = _snapshot_parameters(model, theta_names, requires_grad=None)
    psi = {name: value.detach().clone() for name, value in geometry_value.items()}
    state = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    state.update(theta)
    state.update(psi)
    inner_logits = functional_call(model, state, (strong.detach(),), kwargs={"valid_mask": valid}, strict=False)
    inner_loss, _ = _masked_pseudo_ce(inner_logits, py.detach(), valid.detach())
    inner_names = [name for name, value in theta.items() if value.requires_grad]
    inner_grads = torch.autograd.grad(inner_loss, [theta[name] for name in inner_names], allow_unused=True)
    all_grads = {name: None for name in theta}
    all_grads.update(dict(zip(inner_names, inner_grads, strict=True)))
    virtual, _ = _functional_step_with_states(theta, all_grads, owners, named, clip_norm=clip_norm)
    outer_state = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    outer_state.update(virtual)
    outer_state.update({name: value.detach() for name, value in reference_geometry.items()})
    outer_logits = functional_call(model, outer_state, (xa.detach(),), kwargs={"valid_mask": None}, strict=False)
    outer, _, _ = _present_foreground_dice(outer_logits, ya, None)
    return outer


@pytest.mark.parametrize("clip_norm", [5., 0.01])
def test_directional_hypergradient_matches_finite_difference_with_outer_geometry_fixed(clip_norm):
    model, optimizers = _make_model_and_optimizers(seed=21, dtype=torch.float64)
    strong, py, valid, xa, ya = _inputs(dtype=torch.float64)
    gradients, diagnostics = meta_geometry_gradients(model, optimizers, strong, py, valid, xa, ya, clip_norm=clip_norm)
    assert diagnostics["status"] == "ok"
    geometry = geometry_named_parameters(model)
    reference = {name: parameter.detach().clone() for name, parameter in geometry.items()}
    name = next(iter(geometry))
    direction = torch.zeros_like(reference[name])
    direction[0] = 1.
    assert gradients[name][0].abs() > 1e-7
    epsilon = 2e-4
    plus = {key: value + (epsilon * direction if key == name else 0.) for key, value in reference.items()}
    minus = {key: value - (epsilon * direction if key == name else 0.) for key, value in reference.items()}
    plus_loss = _outer_objective_at_geometry(model, optimizers, strong, py, valid, xa, ya, plus, reference,
                                             clip_norm=clip_norm)
    minus_loss = _outer_objective_at_geometry(model, optimizers, strong, py, valid, xa, ya, minus, reference,
                                              clip_norm=clip_norm)
    finite_difference = (plus_loss - minus_loss) / (2 * epsilon)
    torch.testing.assert_close(gradients[name][0], finite_difference, rtol=3e-4, atol=2e-8)


def test_native_student_adaptive_window_double_hypergradient_and_transaction_integrity():
    torch.manual_seed(41)
    config = {
        "backbone": "tiny_test",
        "source_repo": "",
        "source_commit": "",
        "weights": "",
        "weights_sha256": "",
    }
    model, _ = build_model(config, synthetic=True, window={"mode": "adaptive", "dim": 8})
    model = model.double()
    model.train()
    geometry = dict(geometry_named_parameters(model))
    body = [parameter for name, parameter in model.named_parameters()
            if not name.startswith("head.") and name not in geometry]
    optimizers = [
        torch.optim.AdamW(body, lr=0.003, weight_decay=1e-4),
        torch.optim.SGD(model.head.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0),
    ]
    strong = torch.rand(1, 3, 32, 32, dtype=torch.float64)
    xa = torch.rand(1, 3, 32, 32, dtype=torch.float64)
    pattern = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]])
    py = pattern.repeat(8, 8).unsqueeze(0)
    ya = py.clone()
    valid = torch.ones_like(py, dtype=torch.bool)
    parameter_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    buffer_before = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    modes_before = {id(module): module.training for module in model.modules()}
    optimizer_before = copy.deepcopy([optimizer.state_dict() for optimizer in optimizers])
    rng_before = torch.get_rng_state().clone()

    gradients, diagnostics = meta_geometry_gradients(
        model, optimizers, strong, py, valid, xa, ya,
        input_valid=valid, audit_valid=valid, clip_norm=5.,
    )
    assert diagnostics["status"] == "ok"
    name = "window.geometry.offset.bias"
    assert name in gradients
    assert torch.isfinite(gradients[name]).all()
    # Index 2 is the x-radius logit in the actual adaptive window geometry.
    assert gradients[name][2].abs() > 1e-8

    reference = {key: parameter.detach().clone() for key, parameter in geometry.items()}
    direction = torch.zeros_like(reference[name])
    direction[2] = 1.
    epsilon = 2e-4
    plus = {key: value + (epsilon * direction if key == name else 0.) for key, value in reference.items()}
    minus = {key: value - (epsilon * direction if key == name else 0.) for key, value in reference.items()}
    plus_loss = _outer_objective_at_geometry(model, optimizers, strong, py, valid, xa, ya, plus, reference)
    minus_loss = _outer_objective_at_geometry(model, optimizers, strong, py, valid, xa, ya, minus, reference)
    finite_difference = (plus_loss - minus_loss) / (2 * epsilon)
    torch.testing.assert_close(gradients[name][2], finite_difference, rtol=2e-3, atol=1e-9)

    for current_name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, parameter_before[current_name], rtol=0, atol=0)
        assert parameter.grad is None
    for current_name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, buffer_before[current_name], rtol=0, atol=0)
    assert {id(module): module.training for module in model.modules()} == modes_before
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    for current, before in zip((optimizer.state_dict() for optimizer in optimizers), optimizer_before, strict=True):
        _assert_nested_equal(current, before)


def test_non_plain_optimizer_is_rejected_instead_of_approximated():
    model, _ = _make_model_and_optimizers()
    body = [p for n, p in model.named_parameters()
            if not n.startswith("head.") and not n.startswith("window.geometry.")]
    bad = [torch.optim.AdamW(body, lr=1e-3, amsgrad=True), torch.optim.SGD(model.head.parameters(), lr=1e-2)]
    strong, py, valid, xa, ya = _inputs()
    with pytest.raises(ValueError, match="plain"):
        meta_geometry_gradients(model, bad, strong, py, valid, xa, ya)


def test_omitted_trainable_theta_is_rejected_for_full_update_contract():
    model, _ = _make_model_and_optimizers()
    bad = [torch.optim.SGD(model.head.parameters(), lr=1e-2, momentum=0.0)]
    strong, py, valid, xa, ya = _inputs()
    with pytest.raises(ValueError, match="omit trainable theta"):
        meta_geometry_gradients(model, bad, strong, py, valid, xa, ya)
