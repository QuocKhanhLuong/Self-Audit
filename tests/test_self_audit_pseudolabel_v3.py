import numpy as np
import torch
import nibabel as nib

from self_audit_pseudolabel.data_v3 import discover_acdc_full_cine,CineSliceDataset
from self_audit_pseudolabel.evidence import build_region_evidence
from self_audit_pseudolabel.evolution import PrototypeBank
from self_audit_pseudolabel.losses_v3 import prototype_information_loss,registration_loss
from self_audit_pseudolabel.consistency import consistency_gate
from self_audit_pseudolabel.system_v3 import CinePseudoTeacher,AdaptiveAnnotationStudent,pseudo_supervision_loss
from self_audit_pseudolabel.adaptive import AdaptiveRuntime,RuntimeBudget,choose_profile

def _cine(batch=1,h=32,w=32):
    torch.manual_seed(11); cur=torch.rand(batch,3,h,w)
    return torch.roll(cur,-1,-1),cur,torch.roll(cur,1,-1)

def test_full_cine_loader_never_requires_gt(tmp_path):
    patient=tmp_path/"patient001"; patient.mkdir()
    arr=np.random.RandomState(0).randn(20,18,3,4).astype(np.float32)
    nib.save(nib.Nifti1Image(arr,np.eye(4)),patient/"patient001_4d.nii.gz")
    (patient/"Info.cfg").write_text("ED: 1\nES: 3\n",encoding="utf-8")
    records=discover_acdc_full_cine(tmp_path)
    ds=CineSliceDataset(records,cache_records=1); sample=ds[0]
    assert sample["cur"].shape==(3,18,20)
    assert sample["ed_index"]==1 and sample["es_index"]==3

def test_registration_teacher_shapes_and_backward():
    teacher=CinePseudoTeacher(width=8,appearance_dim=16,motion_dim=8,fused_dim=24,k=8)
    prev,cur,nxt=_cine(); out=teacher(prev,cur,nxt)
    assert out["flow_prev"].shape==(1,2,8,8)
    assert out["warped_prev"].shape==(1,1,32,32)
    photo,smooth=registration_loss(out,cur); loss=photo+.05*smooth
    assert torch.isfinite(loss); loss.backward()

def test_image_evidence_ring_and_background():
    h=w=32; labels=torch.zeros(1,h,w,dtype=torch.long)
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing="ij"); r=((yy-16)**2+(xx-16)**2).float().sqrt()
    labels[0][(r>=7)&(r<=9)]=1; labels[0][r<7]=2
    q=torch.nn.functional.one_hot(labels,4).permute(0,3,1,2).float()
    ev,_=build_region_evidence(q,torch.zeros(1,1,h,w),torch.zeros(1,2,h,w),patient_left_axis=["-x"])
    assert ev[0,0,0]>0 and ev[0,1,2]>0 and ev[0,2,3]>0

def test_evidence_hook_can_create_seed_but_unknown_stays_nonclass():
    teacher=CinePseudoTeacher(width=8,appearance_dim=16,motion_dim=8,fused_dim=24,k=8)
    prev,cur,nxt=_cine(); base=teacher(prev,cur,nxt)
    evidence=torch.zeros_like(base["semantic_logits"]); evidence[...,3]=10
    out=teacher(prev,cur,nxt,evidence_logits=evidence,min_prob=.7,min_margin=.2)
    assert bool((out["semantic_prob"].argmax(-1)==3).all())
    assert bool(out["region_valid"].all())

def test_prototype_bank_and_anti_collapse():
    bank=PrototypeBank(4,3).to("cpu")
    feat=torch.tensor([[[1.,0,0],[0,1,0]]]); prob=torch.tensor([[[.9,.05,.03,.02],[.3,.3,.2,.2]]]); valid=torch.tensor([[True,True]])
    bank.update(feat,prob,valid,min_confidence=.8)
    assert bank.counts[0]==1 and bank.counts[1:].sum()==0
    c=torch.zeros(1,4,8,8); c[:,0]=1
    d=torch.zeros_like(c); d[:,0,:4,:4]=1; d[:,1,:4,4:]=1; d[:,2,4:,:4]=1; d[:,3,4:,4:]=1
    assert prototype_information_loss(d)<prototype_information_loss(c)

def test_consistency_gate_rejects_temporal_flip():
    prob=torch.zeros(3,1,4,2,2); prob[0,:,1]=1; prob[1,:,2]=1; prob[2,:,1]=1
    valid=torch.ones(3,1,2,2,dtype=torch.bool)
    _,v=consistency_gate(prob,valid,temporal_weight=1.0,slice_weight=0.0,min_agreement=.5)
    assert not bool(v[1].any())

def test_dynamic_window_profiles_and_unknown_masking():
    model=AdaptiveAnnotationStudent(width=32,window_k=4)
    _,cur,_=_cine()
    compact=model(cur,profile="compact"); balanced=model(cur,profile="balanced")
    assert len(compact["stages"])==1 and len(balanced["stages"])==2
    target=torch.zeros(1,32,32,dtype=torch.long); valid=torch.zeros_like(target,dtype=torch.bool)
    assert float(pseudo_supervision_loss(compact,target,valid).detach())==0.0

def test_adaptive_runtime_encodes_once_and_respects_cap():
    model=AdaptiveAnnotationStudent(width=32,window_k=4).eval(); _,cur,_=_cine(); calls={"n":0}
    handle=model.encoder.register_forward_hook(lambda *_: calls.__setitem__("n",calls["n"]+1))
    try:
        with torch.no_grad(): out=AdaptiveRuntime(model,RuntimeBudget(max_profile="balanced"))(cur)
    finally:
        handle.remove()
    assert calls["n"]==1 and out["profile"] in {"compact","balanced"}
    certain=torch.tensor([[[[8.]], [[-8.]], [[-8.]], [[-8.]]]])
    assert choose_profile(certain,RuntimeBudget(max_profile="accurate"))=="compact"
