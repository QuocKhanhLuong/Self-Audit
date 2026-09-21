"""Software contracts only; synthetic inputs do not establish anatomy quality."""
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from self_audit_pseudolabel.system_v3 import (
    AdaptiveAnnotationStudent, CinePseudoTeacher, UNKNOWN, pseudo_supervision_loss,
)


def cine():
    generator = torch.Generator().manual_seed(41)
    return tuple(torch.rand(2, 3, 33, 35, generator=generator) for _ in range(3))


def teacher():
    torch.manual_seed(17)
    return CinePseudoTeacher(width=8, appearance_dim=16, motion_dim=8, fused_dim=24, k=4)


def test_teacher_simplexes_finite_backward_and_odd_spatial_shapes():
    model = teacher()
    out = model(*cine())
    assert out['soft_label'].shape == (2, 4, 33, 35)
    assert out['region_prob'].shape == (2, 4, 9, 9)
    for key, dim in [('region_prob', 1), ('semantic_prob', -1), ('soft_label', 1)]:
        value = out[key]
        assert torch.isfinite(value).all()
        assert torch.allclose(value.sum(dim), torch.ones_like(value.sum(dim)), atol=1e-6)
    # An arbitrary differentiable diagnostic objective, NOT a teacher training recipe.
    diagnostic = -out['soft_label'][:, 2].clamp_min(1e-6).log().mean()
    diagnostic.backward()
    for module in [model.appearance, model.motion, model.fuse, model.regions, model.semantic]:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
    assert not out['pseudo_label'].requires_grad
    assert not out['valid'].requires_grad
    assert out['semantic_prob'].shape[-1] == 4  # UNKNOWN is not a fifth class.


def test_no_seed_reconstruction_does_not_train_semantic_head():
    model = teacher()
    inputs = cine()
    with patch('torch.nn.functional.cross_entropy', side_effect=AssertionError('unexpected semantic CE')):
        out = model(*inputs)
        F.mse_loss(out['reconstruction'], inputs[1][:, 1:2]).backward()
    assert any(p.grad is not None for p in model.appearance.parameters())
    for module in [model.motion, model.fuse, model.regions, model.semantic]:
        assert all(p.grad is None for p in module.parameters())


def test_abstention_and_semantic_evidence_intervention():
    model = teacher().eval()
    inputs = cine()
    with torch.no_grad():
        ordinary = model(*inputs)
        zero = model(*inputs, evidence_logits=torch.zeros(2, 4, 4))
        assert torch.equal(ordinary['soft_label'], zero['soft_label'])
        strict = model(*inputs, min_prob=1.0, min_margin=1.0)
        assert not strict['valid'].any()
        assert torch.all(strict['pseudo_label'] == UNKNOWN)
        for semantic_class in range(4):
            evidence = torch.full((2, 4, 4), -40.0)
            evidence[..., semantic_class] = 40.0
            out = model(*inputs, evidence_logits=evidence)
            assert out['valid'].all()
            assert torch.all(out['pseudo_label'] == semantic_class)


def test_confident_regions_with_conflicting_names_abstain_at_pixel_level():
    model = teacher().eval()
    evidence = torch.full((2, 4, 4), -40.0)
    evidence[:, torch.arange(4), torch.arange(4)] = 40.0
    handle = model.regions.register_forward_hook(lambda m, args, out: torch.full_like(out, 0.25))
    try:
        with torch.no_grad():
            out = model(*cine(), evidence_logits=evidence)
        assert torch.allclose(out['soft_label'], torch.full_like(out['soft_label'], 0.25))
        assert not out['valid'].any()
        assert torch.all(out['pseudo_label'] == UNKNOWN)
    finally:
        handle.remove()


@pytest.mark.parametrize('profile,turns,dw_calls', [('compact', 0, 0), ('balanced', 1, 1), ('accurate', 2, 3)])
def test_profile_exact_recurrence_shared_features_and_depth(profile, turns, dw_calls):
    torch.manual_seed(5)
    model = AdaptiveAnnotationStudent(width=8, window_k=4).eval()
    encoder_calls, refiner_args, dw_args = [], [], []
    handles = [
        model.encoder.register_forward_hook(lambda m, a, o: encoder_calls.append(o)),
        model.refiner.register_forward_pre_hook(lambda m, a, kw: refiner_args.append((a, kw)), with_kwargs=True),
        model.refiner.refinement_block.register_forward_pre_hook(lambda m, a, kw: dw_args.append(kw), with_kwargs=True),
    ]
    try:
        with torch.no_grad():
            out = model(cine()[1], profile=profile, return_metadata=True)
        assert len(encoder_calls) == 1
        assert len(refiner_args) == turns
        assert len(dw_args) == dw_calls
        assert len(out['stages']) == turns + 1
        assert out['stages'][0] is out['a0_logits']
        for index, (args, kwargs) in enumerate(refiner_args):
            assert args[0] is encoder_calls[0]
            assert args[1] is out['stages'][index]
            assert kwargs['previous_audit_evidence'] is None
            assert kwargs['turn_index'] == index
        assert [(k['turn_index'], k['iteration_index']) for k in dw_args] == [(0, 0), (1, 0), (1, 1)][:dw_calls]
        if profile == 'compact':
            assert out['final_logits'] is out['a0_logits']
    finally:
        for handle in handles:
            handle.remove()


def test_feature_only_is_audit_invariant_but_annotation_conditioned():
    model = AdaptiveAnnotationStudent(width=8, window_k=4).eval()
    with torch.no_grad():
        features = model.encoder(cine()[1])
        logits = F.interpolate(model.a0_head(features), (33, 35), mode='bilinear', align_corners=False)
        plain = model.refiner(features, logits, return_metadata=True)
        audit = model.refiner(features, logits, previous_audit_evidence=torch.randn(2, 3, 33, 35), return_metadata=True)
        changed = logits.clone()
        changed[:, 1] += 2
        intervention = model.refiner(features, changed, return_metadata=True)
    assert torch.equal(plain.candidate_logits, audit.candidate_logits)
    assert torch.equal(plain.window_metadata['coordinates'], audit.window_metadata['coordinates'])
    assert not torch.allclose(plain.window_metadata['coordinates'], intervention.window_metadata['coordinates'])


def test_final_loss_reaches_a0_encoder_and_window_generator():
    model = AdaptiveAnnotationStudent(width=8, window_k=4)
    out = model(cine()[1], profile='accurate')
    out['a0_logits'].retain_grad()
    target = torch.zeros(2, 33, 35, dtype=torch.long)
    loss = pseudo_supervision_loss(out, target, torch.ones_like(target, dtype=torch.bool), a0_weight=0)
    loss.backward()
    assert out['a0_logits'].grad.abs().sum() > 0
    for module in [model.encoder, model.a0_head, model.refiner.refinement_block.generator]:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0


def test_profiles_share_a0_and_are_deterministic_on_cpu():
    model = AdaptiveAnnotationStudent(width=8, window_k=4).eval()
    inputs = cine()[1]
    with torch.no_grad():
        outputs = [model(inputs, profile=p) for p in ('compact', 'balanced', 'accurate')]
        repeat = model(inputs, profile='accurate')
    assert all(torch.equal(outputs[0]['a0_logits'], x['a0_logits']) for x in outputs)
    assert torch.equal(repeat['final_logits'], outputs[-1]['final_logits'])
    assert not torch.equal(outputs[0]['final_logits'], outputs[1]['final_logits'])


@pytest.mark.parametrize('unknown_valid', [False, True])
def test_unknown_255_is_excluded_even_when_valid_flag_is_true(unknown_valid):
    logits = torch.randn(1, 4, 4, 4, requires_grad=True)
    target = torch.full((1, 4, 4), UNKNOWN, dtype=torch.long)
    target[:, 0, 0] = 2
    valid = torch.full_like(target, unknown_valid, dtype=torch.bool)
    valid[:, 0, 0] = True
    loss = pseudo_supervision_loss({'final_logits': logits, 'a0_logits': logits}, target, valid)
    expected = 1.25 * F.cross_entropy(logits[:, :, 0, 0], target[:, 0, 0])
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :, 1:, :]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 0, 1:]) == 0


def test_all_unknown_returns_differentiable_zero():
    logits = torch.randn(1, 4, 4, 4, requires_grad=True)
    target = torch.full((1, 4, 4), UNKNOWN, dtype=torch.long)
    loss = pseudo_supervision_loss({'final_logits': logits, 'a0_logits': logits}, target, torch.ones_like(target, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0
