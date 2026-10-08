"""CQA software/mechanism tests. Manufactured partitions are NOT learned anatomy."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import binary_fill_holes, label

from self_audit_pseudolabel.cqa_v3 import (
    CQAConfig, audit_partition, audit_components, relation_distillation_loss,
)
from self_audit_pseudolabel.losses_v3 import BootstrapLossConfig


def fixture_partition(open_ring=False):
    y, x = np.mgrid[-32:32, -32:32]
    radius = np.sqrt(x*x+y*y)
    ring = (radius >= 12) & (radius < 20)
    cavity = radius < 12
    ids = np.ones((64, 64), dtype=np.int64)
    ids[ring & (x < 0)] = 2
    ids[ring & (x >= 0)] = 3
    ids[cavity & (x < 0)] = 4
    ids[cavity & (x >= 0)] = 5
    image = np.zeros((64, 64), dtype=np.float64)
    image[ring], image[cavity] = .4, 1.
    if open_ring:
        seam = ring & (abs(x) <= 1) & (y < 0)
        ids[seam] = 0
        image[seam] = 0.
    image = (image-image.mean())/image.std()
    return ids, np.stack([image]*3)


def enclosure_count(ids):
    masks = [ids == i for i in np.unique(ids) if i]
    n = 0
    for wall in masks:
        if wall[0].any() or wall[-1].any() or wall[:, 0].any() or wall[:, -1].any():
            continue
        holes = binary_fill_holes(wall) & ~wall
        if label(wall)[1] != 1 or label(holes)[1] != 1:
            continue
        for cavity in masks:
            if label(cavity)[1] != 1 or (binary_fill_holes(cavity) & ~cavity).any():
                continue
            overlap = (holes & cavity).sum()
            if overlap/max(1, cavity.sum()) >= .8 and overlap/max(1, holes.sum()) >= .8:
                n += 1
    return n


def test_fragmented_ring_and_cavity_are_assembled_without_gt():
    ids, images = fixture_partition()
    out, relations, metrics = audit_partition(ids, images)
    assert enclosure_count(ids) == 0
    assert enclosure_count(out) == 1
    assert metrics['cqa_merge_relations'] == 2
    assert metrics['cqa_keep_relations'] >= 2
    assert len(np.unique(out)) == 3
    assert {row[2] for row in relations} == {0, 1}


def test_open_ring_void_cannot_be_sealed():
    ids, images = fixture_partition(open_ring=True)
    out, _, _ = audit_partition(ids, images)
    assert np.array_equal(out == 0, ids == 0)
    assert enclosure_count(out) == 0


def test_flat_and_noisy_images_do_not_supply_merges():
    ids, images = fixture_partition()
    for view in (np.zeros_like(images), np.random.default_rng(7).normal(size=images.shape)):
        out, relations, metrics = audit_partition(ids, view)
        assert metrics['cqa_merge_relations'] == 0
        assert not any(row[2] == 1 for row in relations)
        assert len(np.unique(out)) == len(np.unique(ids))


def test_counterexample_in_one_neighbor_vetoes_merge():
    ids, images = fixture_partition()
    images[2, ids == 3] += 2.
    _, relations, _ = audit_partition(ids, images)
    assert (2, 3, 1) not in relations


@pytest.mark.parametrize('seed', range(8))
def test_anonymous_atom_renumbering_preserves_physical_quotient(seed):
    ids, images = fixture_partition()
    expected, _, _ = audit_partition(ids, images)
    rng = np.random.default_rng(seed)
    mapping = np.r_[0, rng.permutation(np.arange(1, 6))]
    actual, _, _ = audit_partition(mapping[ids], images)
    assert np.array_equal(actual, expected)


def test_missing_witness_rejected():
    ids, images = fixture_partition()
    with pytest.raises(ValueError, match='witness'):
        audit_partition(ids, images[:1])


def test_disabled_audit_is_identity():
    ids, images = fixture_partition()
    out, relations, _ = audit_partition(ids, images, CQAConfig(enabled=False))
    assert np.array_equal(out, ids)
    assert relations == []


def test_budget_limits_are_honored():
    ids, images = fixture_partition()
    _, _, metrics = audit_partition(ids, images, CQAConfig(max_merges=1, max_edges=3))
    assert metrics['cqa_merge_relations'] <= 1
    assert metrics['cqa_edges'] <= 3
    assert metrics['cqa_edges_truncated'] > 0


def relation_fixture():
    ids = torch.tensor([[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 3, 3]]])
    logits = torch.randn(1, 5, 3, 4, generator=torch.Generator().manual_seed(42), requires_grad=True)
    relations = torch.tensor([[0, 1, 2, 1], [0, 2, 3, 0]])
    return logits, ids, relations


def test_relation_loss_supplies_finite_nonzero_gradients():
    logits, ids, relations = relation_fixture()
    loss = relation_distillation_loss(logits.softmax(1), ids, relations)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_relation_loss_invariant_to_prototype_permutation():
    logits, ids, relations = relation_fixture()
    q = logits.softmax(1)
    reference = relation_distillation_loss(q, ids, relations)
    changed = relation_distillation_loss(q[:, [4, 1, 3, 0, 2]], ids, relations)
    assert torch.allclose(reference, changed, atol=1e-6)


def test_only_positive_targets_cannot_train_all_merge_solution():
    logits, ids, relations = relation_fixture()
    loss = relation_distillation_loss(logits.softmax(1), ids, relations[:1])
    loss.backward()
    assert loss.item() == 0
    assert logits.grad.abs().sum() == 0


def test_empty_relations_are_differentiable_zero():
    logits, ids, _ = relation_fixture()
    loss = relation_distillation_loss(logits.softmax(1), ids, torch.empty(0, 4, dtype=torch.long))
    loss.backward()
    assert loss.item() == 0 and logits.grad is not None


def test_invalid_relation_does_not_silently_train():
    logits, ids, relations = relation_fixture()
    relations[0, 1] = 0
    with pytest.raises(ValueError, match='relation'):
        relation_distillation_loss(logits.softmax(1), ids, relations)


@pytest.mark.parametrize('kwargs', [dict(merge_risk_max=-1), dict(max_edges=0),
    dict(max_atoms_per_group=1), dict(min_real_neighbors=0), dict(enabled=1),
    dict(keep_risk_min=.001), dict(keep_boundary_min=.01), dict(distill_weight=float('nan'))])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        CQAConfig(**kwargs)


def test_loss_config_opt_in_and_json_roundtrip():
    assert BootstrapLossConfig().counterfactual_audit is None
    config = BootstrapLossConfig(counterfactual_audit=asdict(CQAConfig()))
    assert BootstrapLossConfig(**json.loads(json.dumps(asdict(config)))) == config
    with pytest.raises((ValueError, TypeError)):
        BootstrapLossConfig(counterfactual_audit={'unrecognized_key': True})


def test_transitive_drift_is_compared_with_finest_atoms():
    ids = np.tile(np.repeat(np.arange(1, 7), 4), (16, 1)).astype(np.int64)
    # Local differences pass, but merging a long chain must not accumulate drift.
    image = ids.astype(float) * .08
    images = np.stack([image]*3)
    out, _, metrics = audit_partition(ids, images, CQAConfig(merge_risk_max=.002,
        keep_risk_min=.01, merge_boundary_max=.1, keep_boundary_min=.2))
    assert len(np.unique(out)) > 1
    assert metrics['cqa_chain_vetoes'] > 0


def test_balanced_distillation_reduces_its_fixed_relational_objective():
    logits, ids, relations = relation_fixture()
    opt = torch.optim.Adam([logits], lr=.05)
    start = float(relation_distillation_loss(logits.softmax(1), ids, relations).detach())
    for _ in range(25):
        loss = relation_distillation_loss(logits.softmax(1), ids, relations)
        opt.zero_grad(); loss.backward(); opt.step()
    final = float(relation_distillation_loss(logits.softmax(1), ids, relations).detach())
    assert final < start


def adapter_fixture():
    from self_audit_pseudolabel.system_v3 import CinePseudoTeacher, pool_regions
    ids, images = fixture_partition()
    ids = torch.from_numpy(ids)[None]
    torch.manual_seed(7)
    teacher = CinePseudoTeacher(width=4, appearance_dim=8, motion_dim=4, fused_dim=8, k=6)
    image = torch.tensor(images[0], dtype=torch.float32)[None, None].repeat(1, 3, 1, 1)
    batch = {name: image.clone() for name in ('prev', 'cur', 'nxt')}
    batch.update(t=torch.tensor([1]), num_frames=torch.tensor([3]))
    logits = (torch.nn.functional.one_hot(ids, 6).permute(0, 3, 1, 2).float()*6).requires_grad_()
    q = logits.softmax(1)
    assign = torch.nn.functional.one_hot(ids, 6).permute(0, 3, 1, 2).float()
    fused = torch.randn(1, 8, 64, 64, requires_grad=True)
    motion = torch.zeros(1, 4, 64, 64)
    features = pool_regions(assign, fused, image[:, 1:2], motion)
    semantic = teacher.semantic(features)
    base = dict(region_prob=assign, anonymous_region_prob=q, region_features=features,
        semantic_logits=semantic, semantic_prob=semantic.softmax(-1),
        component_pixel_region=ids, component_pixel_eligible=ids > 0,
        component_confidence=q.detach().max(1).values,
        component_eligible=torch.tensor([[False, True, True, True, True, True]]),
        fused_features=fused, motion_features=motion)
    return base, batch, teacher, logits


def test_repooling_preserves_confidence_graph_and_unknown_contract():
    base, batch, teacher, logits = adapter_fixture()
    result = audit_components(base, batch, teacher)
    assert result['cqa_metrics']['cqa_merge_relations'] == 2
    assert result['anonymous_region_prob'] is base['anonymous_region_prob']
    assert result['component_confidence'] is base['component_confidence']
    assert torch.equal(result['component_pixel_eligible'], base['component_pixel_eligible'])
    assert enclosure_count(result['component_pixel_region'][0].numpy()) == 1
    # A quotient is a candidate, not permission to emit a named label.
    decoded = teacher.decode_evidence(result, (64, 64))
    assert torch.all(decoded['pseudo_label'] == 255)
    loss = relation_distillation_loss(result['anonymous_region_prob'],
                                    result['cqa_atomic_ids'], result['cqa_relations'])
    loss.backward()
    assert logits.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.semantic.parameters())


def test_semantic_head_cannot_certify_its_own_merges():
    base, batch, teacher, _ = adapter_fixture()
    one = audit_components(base, batch, teacher)
    for p in teacher.semantic.parameters():
        with torch.no_grad(): p.fill_(10.)
    changed = dict(base, semantic_logits=torch.randn_like(base['semantic_logits'])*100)
    two = audit_components(changed, batch, teacher)
    assert torch.equal(one['component_pixel_region'], two['component_pixel_region'])
    assert torch.equal(one['cqa_relations'], two['cqa_relations'])


def test_repeated_adapter_calls_are_pure_and_deterministic():
    base, batch, teacher, _ = adapter_fixture()
    originals = {k: v.clone() for k, v in base.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        first = audit_components(base, batch, teacher)
        second = audit_components(base, batch, teacher)
    for k, v in originals.items():
        assert torch.equal(base[k], v)
    assert torch.equal(first['component_pixel_region'], second['component_pixel_region'])
    assert first['cqa_metrics'] == second['cqa_metrics']


def test_replicated_temporal_endpoint_is_not_a_witness():
    base, batch, teacher, _ = adapter_fixture()
    batch.update(t=torch.tensor([0]), num_frames=torch.tensor([1]))
    out = audit_components(base, batch, teacher)
    assert out['cqa_metrics']['cqa_no_genuine_neighbor_samples'] == 1
    assert out['cqa_metrics']['cqa_merge_relations'] == 0
    assert torch.equal(out['component_pixel_region'], base['component_pixel_region'])


def test_actual_endpoint_uses_only_existing_neighbor():
    base, batch, teacher, _ = adapter_fixture()
    batch.update(t=torch.tensor([0]), num_frames=torch.tensor([3]))
    batch['prev'] = torch.full_like(batch['prev'], float('nan'))
    # Replicated/non-genuine previous endpoint is deliberately not read.
    out = audit_components(base, batch, teacher)
    assert out['cqa_metrics']['cqa_merge_relations'] == 2


def test_unsupported_pixel_cannot_be_promoted_to_close_a_wall():
    base, batch, teacher, _ = adapter_fixture()
    base['component_pixel_eligible'][0, 32, 32] = False
    with pytest.raises(ValueError, match='unsupported'):
        audit_components(base, batch, teacher)


def test_disconnected_anonymous_atom_is_not_treated_as_one_object():
    ids, images = fixture_partition()
    ids[1:3, 1:3] = 3
    with pytest.raises(ValueError, match='four-connected'):
        audit_partition(ids, images)


def test_multi_sample_adapter_keeps_sample_ownership():
    base, batch, teacher, _ = adapter_fixture()
    base = {k: torch.cat([v, v], 0) if isinstance(v, torch.Tensor) else v for k, v in base.items()}
    batch = {k: torch.cat([v, v], 0) for k, v in batch.items()}
    batch['num_frames'][1] = 1
    batch['t'][1] = 0
    result = audit_components(base, batch, teacher)
    assert result['cqa_metrics']['cqa_merge_relations'] == 2
    assert set(result['cqa_relations'][:, 0].tolist()) == {0}
    assert torch.equal(result['component_pixel_region'][1], base['component_pixel_region'][1])
    assert result['region_prob'].shape[:2] == (2, 6)


def test_prohibited_reference_keys_are_not_read():
    base, batch, teacher, _ = adapter_fixture()
    class ImageOnlyBatch(dict):
        def __getitem__(self, key):
            if key in ('gt', 'mask', 'label', 'reference'):
                raise AssertionError('reference read')
            return super().__getitem__(key)
        def get(self, key, *args):
            if key in ('gt', 'mask', 'label', 'reference'):
                raise AssertionError('reference read')
            return super().get(key, *args)
    assert audit_components(base, ImageOnlyBatch(batch), teacher)['cqa_metrics']['cqa_merge_relations'] == 2


def test_ineligible_component_must_not_be_promoted():
    base, batch, teacher, _ = adapter_fixture()
    base['component_eligible'][0, 2] = False
    with pytest.raises(ValueError, match='ineligible'):
        audit_components(base, batch, teacher)


def test_nonfinite_inherited_confidence_rejected():
    base, batch, teacher, _ = adapter_fixture()
    base['component_confidence'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='confidence'):
        audit_components(base, batch, teacher)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_coassignment_is_finite(dtype):
    logits, ids, relations = relation_fixture()
    q = logits.softmax(1).to(dtype)
    loss = relation_distillation_loss(q, ids, relations)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
