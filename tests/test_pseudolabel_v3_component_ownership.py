"""Upsampling cannot grant a pixel evidence from another physical component."""
import math

import pytest
import torch
from torch.nn import functional as F

from self_audit_pseudolabel.bootstrap_v3 import BootstrapConfig
from self_audit_pseudolabel.components_v3 import component_regions
from self_audit_pseudolabel.consistency import consistency_gate
from self_audit_pseudolabel.trainer_v3 import ProgressiveTeacherTrainer
from test_pseudolabel_v3_components import base_for


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)


def setup_square():
    labels = torch.zeros(8, 8, dtype=torch.long)
    labels[3:5, 3:5] = 1
    base, image, teacher = base_for(labels)
    out = component_regions(base, image, teacher)
    trainer = ProgressiveTeacherTrainer(teacher, None)
    ids = F.interpolate(out['component_pixel_region'][:, None].float(),
                        (32, 32), mode='nearest')[:, 0].long()
    inner = int(out['component_pixel_region'][0, 3, 3])
    outer = int(out['component_pixel_region'][0, 0, 0])
    return out, trainer, ids, inner, outer


def test_upsampling_does_not_borrow_evidence_inside_rejected_component():
    base, trainer, ids, inner, outer = setup_square()
    accepted = base['component_eligible'].clone()
    accepted[:, inner] = False
    evidence = torch.zeros_like(base['semantic_logits'])
    evidence[:, :, 2] = 20
    prediction = trainer._decode(base, (32, 32), evidence, accepted)
    assert prediction['valid'][ids == outer].any()
    assert not prediction['region_valid'][0, inner]
    assert not prediction['valid'][ids == inner].any()
    assert (prediction['pseudo_label'][ids == inner] == 255).all()


def test_upsampling_does_not_rename_accepted_component_from_neighbor():
    base, trainer, ids, inner, outer = setup_square()
    evidence = torch.full_like(base['semantic_logits'], -20)
    evidence[:, outer, 2] = 20
    evidence[:, inner, 2] = math.log(.295)
    evidence[:, inner, 3] = math.log(.705)
    prediction = trainer._decode(base, (32, 32), evidence, base['component_eligible'])
    assert prediction['region_valid'][0, inner]
    assert ((ids == inner) & prediction['valid']).any()
    assert ((ids == outer) & prediction['valid']).any()
    own_class = evidence.argmax(-1).gather(1, ids.flatten(1)).reshape_as(ids)
    assert torch.equal(prediction['pseudo_label'][prediction['valid']],
                       own_class[prediction['valid']])

    # Native export's temporal stage may only revoke support. It must not
    # restore invalid corners or move a supported pixel into another class.
    probabilities = prediction['soft_label'].repeat(3, 1, 1, 1, 1)
    validity = prediction['valid'].repeat(3, 1, 1, 1)
    filtered_prob, filtered_valid = consistency_gate(probabilities, validity)
    assert torch.equal(filtered_prob, probabilities)
    assert not (filtered_valid & ~validity).any()
    expected_class = own_class.repeat(3, 1, 1, 1)
    assert torch.equal(filtered_prob.argmax(2)[filtered_valid], expected_class[filtered_valid])


def test_prototype_bootstrap_cannot_combine_uncertain_nonowners_into_support():
    labels = torch.zeros(8, 8, dtype=torch.long)
    base, _, teacher = base_for(labels, k=3)
    base['region_prob'][:] = torch.tensor([.4, .3, .3])[None, :, None, None]
    base['semantic_logits'].zero_()
    evidence = torch.zeros_like(base['semantic_logits'])
    evidence[:, :, 2] = 20
    accepted = torch.tensor([[False, True, True]])
    legacy = ProgressiveTeacherTrainer(teacher, None)
    experimental = ProgressiveTeacherTrainer(teacher, None,
        bootstrap_config=BootstrapConfig(), train_patient_ids=['a', 'b'],
        component_config={'mode': 'prototype'})
    # This is a decode-unit fixture with supplied upstream evidence. Two
    # accepted nonowners together have mass .6, but no pixel has confident q.
    before = legacy._decode(base, (8, 8), evidence, accepted)
    assert before['valid'].all()  # The legacy recipe remains numerically intact.
    after = experimental._decode(base, (8, 8), evidence, accepted)
    assert not after['valid'].any()
    assert (after['pseudo_label'] == 255).all()


def test_prototype_bootstrap_requires_the_confident_pixel_owner_to_be_accepted():
    labels = torch.zeros(8, 8, dtype=torch.long)
    labels[3:5, 3:5] = 1
    base, _, teacher = base_for(labels)
    base['semantic_logits'].zero_()
    evidence = torch.zeros_like(base['semantic_logits'])
    evidence[:, :, 2] = 20
    accepted = torch.tensor([[True, False]])
    trainer = ProgressiveTeacherTrainer(teacher, None,
        bootstrap_config=BootstrapConfig(), train_patient_ids=['a', 'b'],
        component_config={'mode': 'prototype'})
    owners = F.interpolate(labels[None, None].float(), (32, 32), mode='nearest')[:, 0].long()
    prediction = trainer._decode(base, (32, 32), evidence, accepted)
    assert prediction['valid'][owners == 0].any()
    assert not prediction['valid'][owners == 1].any()
    assert (prediction['pseudo_label'][owners == 1] == 255).all()
