"""Synthetic physical-partition tests, not evidence of clinical accuracy."""
from itertools import permutations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from self_audit_pseudolabel.components_v3 import component_regions
from self_audit_pseudolabel.evidence import build_region_evidence
from self_audit_pseudolabel.system_v3 import CinePseudoTeacher, UNKNOWN


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)


class LocalTeacher(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.region_dim = dim+3
        self.semantic = nn.Linear(self.region_dim, 4)
        nn.init.zeros_(self.semantic.weight)
        nn.init.zeros_(self.semantic.bias)

    decode_evidence = CinePseudoTeacher.decode_evidence


def base_for(labels, *, confidence=.95, k=None, requires_grad=False):
    if labels.ndim == 2:
        labels = labels[None]
    b,h,w = labels.shape
    k = k or max(int(labels.max())+1, 2)
    hard = F.one_hot(labels,k).permute(0,3,1,2).float()
    q = hard*confidence+(1-hard)*((1-confidence)/(k-1))
    q.requires_grad_(requires_grad)
    fused = torch.stack([torch.arange(h*w).reshape(h,w),
                         -torch.arange(h*w).reshape(h,w)],0).float()
    fused = fused[None].repeat(b,1,1,1).requires_grad_(requires_grad)
    motion = torch.zeros(b,2,h,w, requires_grad=requires_grad)
    image = torch.zeros(b,3,h,w, requires_grad=requires_grad)
    return {'region_prob':q, 'fused_features':fused, 'motion_features':motion,
            'region_features':torch.randn(b,k,5),
            'semantic_logits':torch.randn(b,k,4),
            'semantic_prob':torch.softmax(torch.randn(b,k,4),-1)}, image, LocalTeacher()


def ring_labels(speck_size=2):
    labels = torch.zeros(24,24,dtype=torch.long)
    labels[7:18,7:18] = 1
    labels[9:16,9:16] = 2
    labels[2:2+speck_size,2:2+speck_size] = 1
    return labels


def test_disconnected_same_prototype_gets_independent_local_evidence():
    base, image, teacher = base_for(ring_labels())
    old, old_diag = build_region_evidence(base['region_prob'],image[:,1:2],base['motion_features'])
    assert old_diag[0]['enclosures'] == []
    out = component_regions(base,image,teacher)
    ids = out['component_pixel_region'][0]
    wall, speck, cavity = (int(ids[y,x]) for y,x in ((7,7),(2,2),(10,10)))
    assert wall != speck and wall and speck
    assert out['component_source_id'][0,wall] == out['component_source_id'][0,speck] == 1
    ev, diag = build_region_evidence(out['region_prob'],image[:,1:2],base['motion_features'])
    assert any(a == wall and b == cavity for a,b,_ in diag[0]['enclosures'])
    assert not any(speck in (a,b) for a,b,_ in diag[0]['enclosures'])
    decoded = teacher.decode_evidence(out,(24,24), evidence_logits=ev,
                                      evidence_valid=out['component_eligible'])
    assert decoded['pseudo_label'][0,7,7] == 2  # local MYO evidence
    assert decoded['pseudo_label'][0,10,10] == 3
    assert torch.all(decoded['pseudo_label'][0,2:4,2:4] == UNKNOWN)


def test_tiny_distant_speck_is_explicitly_ineligible_void():
    base,image,teacher = base_for(ring_labels(speck_size=1))
    out = component_regions(base,image,teacher)
    assert out['region_prob'][0,0,2,2] == 1
    assert out['component_sizes'][0,0] == 1
    assert not out['component_eligible'][0,0]
    assert out['component_confidence'][0,2,2] == 0
    assert not out['component_pixel_eligible'][0,2,2]
    # Even deliberately confident void evidence cannot override its mask.
    ev = torch.zeros_like(out['semantic_logits']); ev[:,:,2] = 20
    result = teacher.decode_evidence(out,(24,24),evidence_logits=ev,
                                     evidence_valid=out['component_eligible'])
    assert result['pseudo_label'][0,2,2] == UNKNOWN
    assert result['pseudo_label'][0,7,7] == 2


def test_hard_partition_does_not_erase_pixel_or_regional_uncertainty():
    labels = torch.zeros(4,8,dtype=torch.long); labels[:,4:] = 1
    base,image,teacher = base_for(labels)
    base['region_prob'][0,:,1,1] = torch.tensor([.55,.45])
    base['region_prob'][0,0,:,4:] = .45
    base['region_prob'][0,1,:,4:] = .55
    out = component_regions(base,image,teacher)
    left,right = out['component_pixel_region'][0,0,[0,7]].tolist()
    assert torch.all(out['region_prob'].sum(1) == 1)
    assert out['component_eligible'][0,left]
    assert not out['component_eligible'][0,right]
    assert out['component_region_confidence'][0,left] == pytest.approx(.95)
    assert right == 0
    assert out['component_region_confidence'][0,right] == 0
    assert out['component_pixel_region'][0,1,1] == 0
    assert out['component_confidence'][0,1,1] == 0
    assert out['component_pixel_confidence'][0,0,1,1] == pytest.approx(.55)
    assert out['component_sizes'][0].tolist() == [17,15]
    assert not out['component_pixel_eligible'][0,1,1]
    assert out['component_pixel_eligible'][0,0,0]
    assert not out['component_pixel_eligible'][0,:,4:].any()


def test_padding_channels_are_zero_and_ineligible():
    labels = torch.zeros(2,6,6,dtype=torch.long)
    labels[0,:,3:] = 1
    base,image,teacher = base_for(labels)
    out = component_regions(base,image,teacher)
    assert out['region_prob'].shape == (2,3,6,6)
    assert out['component_source_id'][1].tolist() == [-1,0,-1]
    assert out['component_eligible'][1].tolist() == [False,True,False]
    assert out['component_sizes'][1].tolist() == [0,36,0]
    assert torch.equal(out['region_prob'][1,2],torch.zeros(6,6))
    assert torch.equal(out['region_features'][1,2],torch.zeros(5))
    assert torch.equal(out['semantic_logits'][1,2],torch.zeros(4))
    assert torch.equal(out['component_region_confidence'][1,[0,2]],torch.zeros(2))


def test_fragmentation_and_component_cap_fail_closed_without_morphology():
    y,x = torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
    labels = (y+x)%2
    base,image,teacher = base_for(labels)
    empty = component_regions(base,image,teacher)  # every component too small
    assert empty['region_prob'].shape == (1,1,8,8)
    assert not empty['component_eligible'].any()
    assert not empty['component_pixel_eligible'].any()
    assert empty['component_count'].item() == 64
    assert empty['component_dropped_count'].item() == 64
    out = component_regions(base,image,teacher,min_region_pixels=1,max_components=3)
    assert out['region_prob'].shape[1] == 4
    assert out['component_pixel_region'].flatten().tolist() == [1,2,3]+[0]*61
    assert out['component_eligible'].tolist() == [[False,True,True,True]]
    assert out['component_dropped_count'].item() == 61
    assert (out['region_prob'][:,0] == 1).sum() == 61
    assert (out['component_confidence'] > 0).sum() == 3


def test_anonymous_id_permutation_preserves_partition_confidence_and_decoding():
    labels = ring_labels()
    base,image,teacher = base_for(labels)
    original = component_regions(base,image,teacher,max_components=3)
    ev,_ = build_region_evidence(original['region_prob'],image[:,1:2],base['motion_features'])
    decoded = teacher.decode_evidence(original,(24,24),evidence_logits=ev,
                                      evidence_valid=original['component_eligible'])
    for perm in permutations(range(3)):
        changed = {**base, 'region_prob':base['region_prob'][:,list(perm)]}
        out = component_regions(changed,image,teacher,max_components=3)
        for key in ('region_prob','region_features','semantic_logits','component_eligible',
                    'component_region_confidence','component_confidence',
                    'component_pixel_eligible','component_pixel_region','component_sizes'):
            assert torch.equal(out[key],original[key]), key
        moved,_ = build_region_evidence(out['region_prob'],image[:,1:2],base['motion_features'])
        result = teacher.decode_evidence(out,(24,24),evidence_logits=moved,
                                         evidence_valid=out['component_eligible'])
        assert torch.equal(result['pseudo_label'],decoded['pseudo_label'])


def test_exact_argmax_ties_abstain_without_channel_order_tiebreaking():
    labels = torch.zeros(6,6,dtype=torch.long)
    base,image,teacher = base_for(labels,confidence=.5)
    for q in (base['region_prob'],base['region_prob'].flip(1)):
        out = component_regions({**base,'region_prob':q},image,teacher,min_probability=0)
        assert out['region_prob'].shape[1] == 1
        assert not out['component_eligible'].any()
        assert not out['component_pixel_eligible'].any()
        assert not out['component_confidence'].any()


def test_pooling_and_gradients_are_component_local_and_anonymous_q_is_retained():
    base,image,teacher = base_for(ring_labels(),requires_grad=True)
    q = base['region_prob']; fused = base['fused_features']
    out = component_regions(base,image,teacher)
    for key in ('region_prob','region_features','semantic_logits','semantic_prob'):
        assert out['anonymous_'+key] is base[key]
    assert base['region_prob'] is q
    ids = out['component_pixel_region'][0]
    speck = int(ids[2,2]); mask = ids == speck
    assert torch.allclose(out['region_features'][0,speck,:2],fused[0,:,mask].mean(1))
    assert out['region_features'][0,speck,-1].detach().item() == pytest.approx(4/576)
    assert not out['region_prob'].requires_grad
    assert not out['component_confidence'].requires_grad
    # Training one component cannot backpropagate into a remote occurrence of
    # its anonymous prototype; unsupervised q remains independently trainable.
    with torch.no_grad():
        teacher.semantic.weight.fill_(1)
    out = component_regions(base,image,teacher)
    out['semantic_logits'][0,speck].sum().backward()
    assert fused.grad[0,:,mask].abs().sum() > 0
    assert fused.grad[0,:,~mask].abs().sum() == 0
    assert q.grad is None
    assert teacher.semantic.weight.grad.abs().sum() > 0
    out['anonymous_region_prob'].square().sum().backward()
    assert q.grad.abs().sum() > 0


def test_constant_and_noisy_inputs_remain_finite_without_inventing_anatomy():
    for seed in range(8):
        labels = torch.randint(0,4,(20,20),generator=torch.Generator().manual_seed(seed))
        base,image,teacher = base_for(labels)
        if seed%2:
            image = torch.randn(image.shape,generator=torch.Generator().manual_seed(seed+100))
        out = component_regions(base,image,teacher,max_components=8)
        for value in out.values():
            if isinstance(value,torch.Tensor):
                assert torch.isfinite(value).all()
        assert torch.all(out['region_prob'].sum(1) == 1)
        # Representation alone grants no semantic permission, even on confident
        # random assignments or spatially constant images.
        decoded = teacher.decode_evidence(out,(20,20))
        assert torch.all(decoded['pseudo_label'] == UNKNOWN)


def test_explicit_assignment_memory_bound_reduces_capacity(monkeypatch):
    import self_audit_pseudolabel.components_v3 as module
    labels = torch.zeros(4,8,dtype=torch.long); labels[:,2:4] = 1; labels[:,6:] = 1
    base,image,teacher = base_for(labels)
    monkeypatch.setattr(module,'MAX_ASSIGNMENT_ELEMENTS',96)
    out = component_regions(base,image,teacher)
    assert out['region_prob'].numel() <= 96
    assert out['region_prob'].shape[1] == 3
    assert out['component_dropped_count'].item() == 2


@pytest.mark.parametrize('kwargs',[
    {'min_region_pixels':0}, {'min_region_pixels':True},
    {'max_components':0}, {'max_components':257}, {'max_components':1.5},
    {'min_probability':-1}, {'min_probability':1.1}, {'min_probability':float('nan')},
    {'min_probability':True},
])
def test_invalid_options_fail_explicitly(kwargs):
    base,image,teacher = base_for(torch.zeros(4,4,dtype=torch.long))
    with pytest.raises(ValueError):
        component_regions(base,image,teacher,**kwargs)


@pytest.mark.parametrize('bad', ['nan','unnormalized','negative','shape','feature_nan'])
def test_invalid_inputs_fail_explicitly(bad):
    base,image,teacher = base_for(torch.zeros(4,4,dtype=torch.long))
    if bad == 'nan': base['region_prob'][0,0,0,0] = float('nan')
    if bad == 'unnormalized': base['region_prob'] *= .5
    if bad == 'negative': base['region_prob'][0,:,0,0] = torch.tensor([1.1,-.1])
    if bad == 'shape': base['fused_features'] = base['fused_features'][:,:,:-1]
    if bad == 'feature_nan': base['fused_features'][0,0,0,0] = float('nan')
    with pytest.raises(ValueError):
        component_regions(base,image,teacher)


def test_double_adaptation_is_rejected():
    base,image,teacher = base_for(torch.zeros(4,4,dtype=torch.long))
    out = component_regions(base,image,teacher)
    with pytest.raises(ValueError,match='twice'):
        component_regions(out,image,teacher)


def test_low_precision_large_component_keeps_finite_original_confidence():
    # 65,536 pixels exceed float16's largest finite integer. Pooling sums must
    # not turn this perfectly supported region into zero-confidence or NaN.
    labels = torch.zeros(256,256,dtype=torch.long)
    base,image,teacher = base_for(labels)
    base['fused_features'].zero_()
    base = {key:value.half() for key,value in base.items()}
    image = image.half(); teacher = teacher.half()
    out = component_regions(base,image,teacher)
    assert out['component_eligible'].tolist() == [[False,True]]
    assert out['component_region_confidence'][0,1].item() == pytest.approx(.95,abs=.001)
    for value in out.values():
        if isinstance(value,torch.Tensor):
            assert torch.isfinite(value).all()


def test_four_connected_partition_does_not_join_diagonal_pixels():
    labels = torch.tensor([[1,0],[0,1]])
    base,image,teacher = base_for(labels)
    out = component_regions(base,image,teacher,min_region_pixels=1)
    assert out['component_count'].item() == 4
    assert out['component_pixel_region'].flatten().tolist() == [1,2,3,4]


def test_memory_and_grid_limits_fail_before_dense_allocation(monkeypatch):
    import self_audit_pseudolabel.components_v3 as module
    base,image,teacher = base_for(torch.zeros(4,4,dtype=torch.long))
    monkeypatch.setattr(module,'MAX_GRID_PIXELS',15)
    with pytest.raises(ValueError,match='memory bound'):
        component_regions(base,image,teacher)
    monkeypatch.setattr(module,'MAX_GRID_PIXELS',16)
    monkeypatch.setattr(module,'MAX_ASSIGNMENT_ELEMENTS',16)
    with pytest.raises(ValueError,match='capacity'):
        component_regions(base,image,teacher)


def test_uncertain_radial_seam_cannot_close_wall_or_unlock_readiness():
    from scipy.ndimage import binary_fill_holes
    from self_audit_pseudolabel.bootstrap_v3 import (
        BootstrapConfig, BootstrapReadiness, assess_bootstrap_candidates,
    )
    from self_audit_pseudolabel.evolution import accepted_region_mask

    labels = ring_labels()
    base,image,teacher = base_for(labels)
    image[:,:,labels == 1] = .2
    image[:,:,labels == 2] = 1.
    batch = {'cur':image, 'prev':image.clone(), 'nxt':image.clone(),
             'patient_id':['p1'], 't':torch.tensor([1]), 'z':torch.tensor([0]),
             'num_frames':torch.tensor([3])}
    cfg = BootstrapConfig(min_train_samples=2,min_train_patients=2,
                          min_successful_updates=0,min_region_pixels=4)

    def assess(output):
        evidence,diagnostics = build_region_evidence(
            output['region_prob'],image[:,1:2],base['motion_features'])
        valid = accepted_region_mask(evidence.softmax(-1)) & output['component_eligible']
        support = assess_bootstrap_candidates(batch,output['region_prob'],
                                               evidence,valid,diagnostics,cfg)
        return support,diagnostics

    closed = component_regions(base,image,teacher)
    assert assess(closed)[0].sample_eligible.tolist() == [True]
    # The four-pixel radial seam has an unchanged hard winning anonymous ID
    # and the overall wall remains well above the regional mean threshold.
    seam = torch.zeros_like(labels,dtype=torch.bool); seam[7:9,11:13] = True
    base['region_prob'][0,:,seam] = torch.tensor([.245,.51,.245])[:,None]
    original_q = base['region_prob'].clone()
    hard_wall = base['region_prob'][0].argmax(0) == 1
    assert binary_fill_holes(hard_wall.numpy()).sum() > int(hard_wall.sum())
    assert base['region_prob'][0].max(0).values[hard_wall].mean() > .7

    opened = component_regions(base,image,teacher)
    assert torch.equal(opened['anonymous_region_prob'],original_q)
    assert torch.all(opened['component_pixel_region'][0,seam] == 0)
    assert not opened['component_pixel_eligible'][0,seam].any()
    assert not opened['component_confidence'][0,seam].any()
    wall_id = int(opened['component_pixel_region'][0,7,7])
    wall = opened['component_pixel_region'][0] == wall_id
    assert not (binary_fill_holes(wall.numpy()) & ~wall.numpy()).any()
    assert torch.allclose(opened['region_features'][0,wall_id,:2],
                          base['fused_features'][0,:,wall].mean(1))
    support,diagnostics = assess(opened)
    assert diagnostics[0]['enclosures'] == []
    assert not support.sample_eligible.any()
    assert not support.region_eligible.any()
    assert support.masks == [None]
    readiness = BootstrapReadiness(cfg,train_patient_ids=['p1','p2'])
    for patient in ['p1','p2']*3:
        batch['patient_id'] = [patient]
        assert not readiness.observe(batch,support)
    assert not readiness.ready
    assert readiness.qualified_samples == 0
