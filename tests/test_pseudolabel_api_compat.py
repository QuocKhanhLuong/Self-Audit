"""Guided inference API remains compatible without leaking evidence into seed loss."""
import math
import torch
from self_audit_pseudolabel.system_v3 import CinePseudoTeacher
from self_audit_pseudolabel.losses_v3 import teacher_loss

def test_guided_probability_alias_and_raw_seed_loss_are_separate():
    torch.set_num_threads(1); torch.manual_seed(11)
    model=CinePseudoTeacher(width=8,appearance_dim=16,motion_dim=8,fused_dim=24,k=8)
    with torch.no_grad():
        model.semantic[-1].weight.zero_(); model.semantic[-1].bias.zero_()
    x=torch.rand(1,3,16,16); evidence=torch.zeros(1,8,4); evidence[...,3]=10
    out=model(x,x,x,evidence_logits=evidence)
    assert torch.all(out["semantic_prob"].argmax(-1)==3)
    assert torch.equal(out["semantic_prob"],out["guided_semantic_prob"])
    assert torch.allclose(out["raw_semantic_prob"],torch.full_like(out["raw_semantic_prob"],.25))
    losses=teacher_loss(out,x,evidence,torch.ones(1,8,dtype=torch.bool))
    assert abs(losses["semantic_seed"].item()-math.log(4))<1e-6
    losses["semantic_seed"].backward()
    assert model.semantic[-1].bias.grad.abs().sum()>0
