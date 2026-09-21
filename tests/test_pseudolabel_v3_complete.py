import numpy as np
import torch
from self_audit_pseudolabel.data_v3 import triplet_z
from self_audit_pseudolabel.evidence import build_region_evidence
from self_audit_pseudolabel.evolution import PrototypeBank
from self_audit_pseudolabel.losses_v3 import prototype_information_loss
from self_audit_pseudolabel.adaptive import RuntimeBudget,choose_profile

def test_triplet_z_boundary_replication():
    v=np.arange(3*4*5,dtype=np.float32).reshape(3,4,5); a=triplet_z(v,0)
    assert np.array_equal(a[0],v[0]) and np.array_equal(a[1],v[0]) and np.array_equal(a[2],v[1])

def test_evidence_finds_background_and_ring_enclosure():
    h=w=32; labels=torch.zeros(1,h,w,dtype=torch.long)
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing="ij"); r=((yy-16)**2+(xx-16)**2).float().sqrt()
    labels[0][(r>=7)&(r<=9)]=1; labels[0][r<7]=2
    q=torch.nn.functional.one_hot(labels,4).permute(0,3,1,2).float()
    ev,_=build_region_evidence(q,torch.zeros(1,1,h,w),torch.zeros(1,2,h,w),patient_left_axis=["-x"])
    assert ev[0,0,0]>0 and ev[0,1,2]>0 and ev[0,2,3]>0

def test_prototype_bank_only_updates_confident():
    bank=PrototypeBank(4,3).to("cpu")
    feat=torch.tensor([[[1.,0,0],[0,1,0]]]); prob=torch.tensor([[[.9,.05,.03,.02],[.3,.3,.2,.2]]]); valid=torch.tensor([[True,True]])
    bank.update(feat,prob,valid,min_confidence=.8)
    assert bank.counts[0]==1 and bank.counts[1:].sum()==0

def test_anti_collapse_prefers_diverse_confident_prototypes():
    c=torch.zeros(1,4,8,8); c[:,0]=1
    d=torch.zeros_like(c); d[:,0,:4,:4]=1; d[:,1,:4,4:]=1; d[:,2,4:,:4]=1; d[:,3,4:,4:]=1
    assert prototype_information_loss(d)<prototype_information_loss(c)

def test_controller_respects_cap():
    certain=torch.tensor([[[[8.]], [[-8.]], [[-8.]], [[-8.]]]]); uncertain=torch.zeros(1,4,1,1)
    assert choose_profile(certain,RuntimeBudget(max_profile="accurate"))=="compact"
    assert choose_profile(uncertain,RuntimeBudget(max_profile="balanced"))=="balanced"
