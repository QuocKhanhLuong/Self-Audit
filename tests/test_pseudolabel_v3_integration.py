import torch
from self_audit_pseudolabel.system_v3 import CinePseudoTeacher,AdaptiveAnnotationStudent,pseudo_supervision_loss
from self_audit_pseudolabel.losses_v3 import registration_loss
from self_audit_pseudolabel.adaptive import AdaptiveRuntime,RuntimeBudget

def _cine(batch=1,h=32,w=32):
    torch.manual_seed(11); cur=torch.rand(batch,3,h,w)
    return torch.roll(cur,-1,-1),cur,torch.roll(cur,1,-1)

def test_registration_teacher_shapes_and_backward():
    teacher=CinePseudoTeacher(width=8,appearance_dim=16,motion_dim=8,fused_dim=24,k=8)
    prev,cur,nxt=_cine(); out=teacher(prev,cur,nxt)
    assert out["flow_prev"].shape==(1,2,8,8)
    assert out["warped_prev"].shape==(1,1,32,32)
    assert out["region_features"].shape[:2]==(1,8)
    photo,smooth=registration_loss(out,cur); loss=photo+.05*smooth+out["reconstruction"].mean()*0
    assert torch.isfinite(loss); loss.backward()

def test_evidence_hook_can_create_accepted_semantic_seed():
    teacher=CinePseudoTeacher(width=8,appearance_dim=16,motion_dim=8,fused_dim=24,k=8)
    prev,cur,nxt=_cine(); base=teacher(prev,cur,nxt)
    evidence=torch.zeros_like(base["semantic_logits"]); evidence[...,3]=10.0
    out=teacher(prev,cur,nxt,evidence_logits=evidence,min_prob=.7,min_margin=.2)
    assert bool((out["semantic_prob"].argmax(-1)==3).all())
    assert bool(out["region_valid"].all())

def test_unknown_pixels_are_not_supervised():
    model=AdaptiveAnnotationStudent(width=32,window_k=4)
    _,cur,_=_cine(); out=model(cur,profile="compact")
    target=torch.zeros(1,32,32,dtype=torch.long); valid=torch.zeros_like(target,dtype=torch.bool)
    loss=pseudo_supervision_loss(out,target,valid)
    assert float(loss.detach())==0.0

def test_adaptive_runtime_encodes_once():
    model=AdaptiveAnnotationStudent(width=32,window_k=4).eval(); _,cur,_=_cine()
    calls={"n":0}
    def hook(*_): calls["n"]+=1
    handle=model.encoder.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out=AdaptiveRuntime(model,RuntimeBudget(max_profile="balanced"))(cur)
    finally:
        handle.remove()
    assert calls["n"]==1
    assert out["profile"] in {"compact","balanced"}
    assert out["final_logits"].shape==(1,4,32,32)
