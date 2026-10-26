import torch

from self_audit.volume.fusion import CrossZResidual, SkipFusion


def test_cross_z_enabled_neighbor_influence_and_off_control():
    torch.manual_seed(3)
    x = torch.zeros(1, 2, 3, 4, 4)
    x[:, :, 1] = 1.0
    mix = CrossZResidual(2, mode="mix")
    with torch.no_grad():
        mix.mixer.weight.zero_()
        mix.mixer.bias.zero_()
        # Kernel index 0 reads the preceding Z slice for center index 1.
        mix.mixer.weight[:, :, 0, 0, 0] = 0.5
    changed = x.clone()
    changed[:, :, 0] = 2.0
    assert not torch.allclose(mix(x)[:, :, 1], mix(changed)[:, :, 1])
    off = CrossZResidual(2, mode="off")
    assert torch.allclose(off(x)[:, :, 1], off(changed)[:, :, 1])


def test_delayed_skip_stop_gradient_and_shapes():
    skip = torch.randn(1, 4, 2, 5, 7, requires_grad=True)
    context = torch.randn(1, 3, 2, 5, 7, requires_grad=True)
    fusion = SkipFusion(4, 3, 6, mode="delayed", stop_gradient=True)
    out = fusion(skip, context)
    assert out.shape == (1, 6, 2, 5, 7)
    out.square().mean().backward()
    assert skip.grad is None
    assert context.grad is not None and torch.isfinite(context.grad).all()


def test_fusion_rejects_grid_mismatch():
    fusion = SkipFusion(2, 2, 2)
    try:
        fusion(torch.zeros(1, 2, 2, 3, 3), torch.zeros(1, 2, 2, 4, 3))
    except ValueError as exc:
        assert "spatial grids" in str(exc)
    else:
        raise AssertionError("grid mismatch must fail closed")


def test_matched_additive_is_nonlinear_and_capacity_close_to_delayed():
    torch.manual_seed(5)
    skip = torch.randn(1, 8, 2, 5, 7)
    context = torch.randn(1, 8, 2, 5, 7)
    delayed = SkipFusion(8, 8, 8, mode="delayed")
    additive = SkipFusion(8, 8, 8, mode="matched_additive")
    d = delayed.capacity_report()
    a = additive.capacity_report()
    assert d["parameters"] > 0 and a["parameters"] > 0
    assert abs(a["parameters"] - d["parameters"]) / d["parameters"] < 0.1
    assert not torch.allclose(additive(skip, context), additive(skip, torch.zeros_like(context)))


def test_frozen_intervention_disables_skip_projection_gradients():
    skip = torch.randn(1, 4, 2, 5, 7, requires_grad=True)
    context = torch.randn(1, 4, 2, 5, 7, requires_grad=True)
    fusion = SkipFusion(4, 4, 4, mode="matched_additive")
    fusion.set_frozen_intervention(True)
    fusion(skip, context).square().mean().backward()
    assert all(parameter.grad is None for parameter in fusion.skip_proj.parameters())


def test_skip_zero_init_alias_only_zeros_terminal_projection():
    fusion = SkipFusion(4, 4, 4, mode="legacy", zero_init=True)
    upstream = [fusion.skip_proj, fusion.context_proj]
    assert all(torch.count_nonzero(parameter).item() > 0 for module in upstream for parameter in module.parameters())
    assert torch.count_nonzero(fusion.residual_projection.weight).item() == 0
    assert torch.count_nonzero(fusion.residual_projection.bias).item() == 0


def test_branch_intervention_changes_nonzero_residual_and_restores_without_state_mutation():
    torch.manual_seed(31)
    fusion = SkipFusion(4, 4, 4, mode="legacy", zero_init=True)
    skip = torch.randn(1, 4, 2, 5, 7)
    context = torch.randn(1, 4, 2, 5, 7)
    target = torch.randn(1, 4, 2, 5, 7)
    optimizer = torch.optim.SGD(fusion.parameters(), lr=0.05)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (fusion(skip, context) - target).square().mean().backward()
        optimizer.step()
    assert torch.count_nonzero(fusion.residual_projection.weight).item() > 0
    state_before = {name: value.detach().clone() for name, value in fusion.state_dict().items()}
    requires_grad_before = {name: parameter.requires_grad for name, parameter in fusion.named_parameters()}
    normal = fusion(skip, context).detach()
    with fusion.branch_intervention("skip"):
        intervened = fusion(skip, context).detach()
    restored = fusion(skip, context).detach()
    assert not torch.allclose(normal, intervened)
    assert torch.equal(normal, restored)
    assert all(torch.equal(state_before[name], value) for name, value in fusion.state_dict().items())
    assert all(requires_grad_before[name] == parameter.requires_grad for name, parameter in fusion.named_parameters())


def test_persistent_frozen_intervention_scope_blocks_added_skip_only():
    skip = torch.randn(1, 4, 2, 5, 7, requires_grad=True)
    context = torch.randn(1, 4, 2, 5, 7, requires_grad=True)
    fusion = SkipFusion(4, 4, 4, mode="legacy")
    fusion.set_frozen_intervention(True, branch="skip")
    fusion(skip, context).square().mean().backward()
    assert skip.grad is None
    assert context.grad is not None and torch.isfinite(context.grad).all()
    fusion.clear_frozen_intervention()
    assert all(parameter.requires_grad for parameter in fusion.skip_proj.parameters())
