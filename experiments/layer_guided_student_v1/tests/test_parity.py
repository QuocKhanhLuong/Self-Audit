import copy
from pathlib import Path

import pytest
import torch
from torch import nn

import layer_upgrade as port


class TinyEncoder(nn.Module):
    channels = (8, 16, 32, 64)

    def __init__(self):
        super().__init__()
        widths = (3, *self.channels)
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    widths[i],
                    widths[i + 1],
                    4 if i == 0 else 2,
                    stride=4 if i == 0 else 2,
                )
                for i in range(4)
            ]
        )

    def forward(self, x):
        result = []
        for layer in self.layers:
            x = layer(x)
            result.append(x)
        return result


def identical(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_port_bytes_equal_verified_oracle(oracle):
    root = Path(port.__file__).parent
    reference = Path(oracle.__file__).parent
    for name in (
        "__init__.py",
        "layers.py",
        "losses.py",
        "model.py",
        "guarded.py",
        "refine.py",
    ):
        assert (root / name).read_bytes() == (reference / name).read_bytes()


@pytest.mark.parametrize("hw", [(32, 32), (47, 53), (64, 96), (65, 97)])
def test_complete_output_and_gradient_parity(oracle, hw):
    a = (
        port.LayerGuidedStudent(TinyEncoder(), decoder_channels=(16, 24, 32, 40))
        .double()
        .eval()
    )
    b = (
        oracle.LayerGuidedStudent(TinyEncoder(), decoder_channels=(16, 24, 32, 40))
        .double()
        .eval()
    )
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x = torch.rand(1, 3, *hw, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    ao, bo = a(x), b(y)
    assert set(ao) == set(bo)
    for key in ao:
        identical(ao[key], bo[key])
    for outputs in (ao, bo):
        (
            outputs["logits"].square().mean() + outputs["features"].square().mean()
        ).backward()
    identical(x.grad, y.grad)
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other
        if p.grad is None:
            assert q.grad is None
        else:
            identical(p.grad, q.grad)


def test_input_loss_and_detach_parity(oracle):
    image = torch.rand(2, 3, 13, 17, dtype=torch.double)
    models = [
        port.CenterDifferenceInput().double(),
        oracle.CenterDifferenceInput().double(),
    ]
    outputs = [m(image) for m in models]
    identical(*outputs)
    for x in outputs:
        x.square().mean().backward()
    identical(models[0].context.weight.grad, models[1].context.weight.grad)
    assert models[0].context.weight.grad.norm() > 0
    mass = torch.rand(2, 4, 3, 5, dtype=torch.double, requires_grad=True)
    support = mass.sum(1)
    logits = torch.randn_like(mass, requires_grad=True)
    target_a, valid_a = port.evidence_distribution(mass, support)
    target_b, valid_b = oracle.evidence_distribution(mass, support)
    identical(target_a, target_b)
    identical(valid_a, valid_b)
    for fun in ("soft_ce", "set_loss", "masked_feature_loss"):
        target = target_a if fun != "set_loss" else target_a > 0.1
        if fun == "set_loss":
            target[:, 0] = True
        weights = valid_a if fun == "masked_feature_loss" else support
        la = getattr(port, fun)(logits, target, weights)
        lb = getattr(oracle, fun)(logits, target, weights)
        identical(la, lb)
        identical(
            torch.autograd.grad(la, logits, retain_graph=True)[0],
            torch.autograd.grad(lb, logits, retain_graph=True)[0],
        )
    assert mass.grad is None


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [[0.0, 0.0]],
        [[1.0, 0.0]],
        [[1.0, 0.0], [-1.0, 0.0]],
        [[1.0, 2.0], [2.0, 4.0]],
    ],
)
def test_projection_degenerate_guard_parity(oracle, rows):
    direction = torch.tensor([-2.0, 1.0], dtype=torch.double)
    guards = torch.tensor(rows, dtype=torch.double).reshape(-1, 2)
    a, ar = port.project_direction(direction, guards)
    b, br = oracle.project_direction(direction, guards)
    identical(a, b)
    assert ar == br


@pytest.mark.parametrize("case", ["accepted", "rejected", "exception"])
def test_guard_actual_updates_and_rollback_parity(oracle, case):
    results = []
    for module in (port, oracle):
        p = nn.Parameter(torch.zeros(1, dtype=torch.double))
        unselected = nn.Parameter(torch.ones(1, dtype=torch.double))
        calls = 0

        def guards():
            nonlocal calls
            calls += 1
            if case == "exception" and calls > 1:
                raise RuntimeError("frozen witness failed")
            return [(p - 1).square().sum() if case == "accepted" else p.square().sum()]

        def objective():
            return (p - unselected).square().sum()

        if case == "exception":
            with pytest.raises(RuntimeError, match="frozen witness"):
                module.guarded_sgd_step([p], objective, guards, step_size=0.1)
            receipt = {"restored": True}
        else:
            receipt = module.guarded_sgd_step(
                [p], objective, guards, step_size=0.1, max_backtracks=1, atol=0
            )
        assert p.grad is None and unselected.grad is None and unselected.item() == 1
        if case != "accepted":
            assert p.item() == 0
        results.append((p.detach(), receipt))
    identical(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


@pytest.mark.parametrize("kind", ["informative", "wrong", "uniform"])
def test_frozen_energy_value_gradient_and_proposal_parity(oracle, kind):
    image = torch.rand(2, 3, dtype=torch.double)
    edges, weights = port.image_grid_graph(image)
    oe, ow = oracle.image_grid_graph(image)
    identical(edges, oe)
    identical(weights, ow)
    unary = torch.zeros(6, 4, dtype=torch.double)
    if kind != "uniform":
        unary[:, 0 if kind == "informative" else 1] = -8
    anchors = torch.full_like(unary, 0.25)
    logits = torch.zeros_like(unary)
    results = []
    for module in (port, oracle):
        energy = module.FrozenImageEnergy(
            unary, edges, weights, anchors, torch.zeros(6), lam=0.1, beta=0
        )
        p = logits.softmax(-1).requires_grad_()
        identical(energy.gradient(p), torch.autograd.grad(energy.value(p), p)[0])
        proposal, receipt = module.refine_distribution(logits, energy, steps=3)
        assert all(row["inequality_slack"] >= -1e-10 for row in receipt["trace"])
        with pytest.raises(ValueError):
            module.refine_distribution(logits, energy, eta=2 / energy.lipschitz_bound())
        results.append((proposal, receipt))
    identical(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]
