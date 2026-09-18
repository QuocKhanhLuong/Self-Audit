import hashlib
import random
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cardiac_benchmark.config import load_primary_config
from cardiac_benchmark.dfc_runner import DFCConfig, MyNet, loss_terms, run_dfc, stop_after_update
from cardiac_benchmark.provenance import derive_sample_seed, seed_everything

class ReferenceMyNet(nn.Module):
    """Independent test-only transcription of untouched demo.py::MyNet."""
    def __init__(self, input_dim, c=100, n=2):
        super().__init__(); self.conv1=nn.Conv2d(input_dim,c,3,1,1); self.bn1=nn.BatchNorm2d(c)
        self.conv2=nn.ModuleList(); self.bn2=nn.ModuleList()
        for _ in range(n-1): self.conv2.append(nn.Conv2d(c,c,3,1,1)); self.bn2.append(nn.BatchNorm2d(c))
        self.conv3=nn.Conv2d(c,c,1,1,0); self.bn3=nn.BatchNorm2d(c)
    def forward(self,x):
        x=self.conv1(x); x=F.relu(x); x=self.bn1(x)
        for i in range(len(self.conv2)): x=self.conv2[i](x); x=F.relu(x); x=self.bn2[i](x)
        return self.bn3(self.conv3(x))

def ref_loss(out):
    o=out[0].permute(1,2,0).contiguous().view(-1,out.shape[1]); grid=o.reshape(out.shape[2],out.shape[3],out.shape[1]); target=o.max(1).indices
    return o, nn.CrossEntropyLoss()(o,target), nn.L1Loss()(grid[1:]-grid[:-1],torch.zeros_like(grid[1:])), nn.L1Loss()(grid[:,1:]-grid[:,:-1],torch.zeros_like(grid[:,1:])), target

def reference_headless(x, seed, max_iter, min_labels):
    """Separate direct-loop harness from the pinned script's headless branch."""
    seed_everything(seed); model=ReferenceMyNet(1); model.train(); opt=optim.SGD(model.parameters(),lr=.1,momentum=.9); forwards=0
    for _ in range(max_iter):
        opt.zero_grad(); out=model(x); forwards += 1; flat,sim,row,col,target=ref_loss(out); count=len(torch.unique(target)); (sim+row+col).backward(); opt.step()
        if count <= min_labels: break
    out=model(x); forwards += 1
    partition=out[0].permute(1,2,0).contiguous().view(-1,100).max(1).indices.detach().numpy().reshape(x.shape[-2:]).astype('<i4')
    return partition, forwards, {k:v.clone() for k,v in model.state_dict().items()}

def test_primary_config_and_seed_golden_vectors():
    cfg=load_primary_config("baseline/DFC/config/cardiac/dfc_direct_2d_minl3.yaml"); assert cfg.maxIter == 1000 and cfg.nChannel == 100
    with pytest.raises(ValueError): DFCConfig(nChannel=4)
    assert derive_sample_seed("ACDC:p001:f00:z01")["sample_seed"] == 3414101426319949208
    assert derive_sample_seed("a\\b")["seed_payload"] == '["dfc-sample-seed-v1",42,"a\\\\b"]'

def test_architecture_initialization_losses_and_one_step_parity():
    assert sum(p.numel() for p in MyNet(1).parameters()) == 101800
    assert sum(p.numel() for p in MyNet(3).parameters()) == 103600
    seed_everything(77); reference=ReferenceMyNet(1)
    seed_everything(77); wrapped=MyNet(1)
    for a,b in zip(reference.state_dict().values(),wrapped.state_dict().values()): assert torch.equal(a,b)
    x=torch.linspace(-1,1,35,dtype=torch.float32).reshape(1,1,5,7); reference.train(); wrapped.train()
    rr=reference(x); wr=wrapped(x); ro,rs,rrw,rco,rt=ref_loss(rr); wo,ws,wrw,wco,wt=loss_terms(wr)
    for a,b in ((ro,wo),(rs,ws),(rrw,wrw),(rco,wco),(rs+rrw+rco,ws+wrw+wco)): assert torch.equal(a,b)
    ra=optim.SGD(reference.parameters(),lr=.1,momentum=.9); wa=optim.SGD(wrapped.parameters(),lr=.1,momentum=.9)
    (rs+rrw+rco).backward(); (ws+wrw+wco).backward()
    for a,b in zip(reference.parameters(),wrapped.parameters()): assert torch.equal(a.grad,b.grad)
    ra.step(); wa.step()
    for a,b in zip(reference.state_dict().values(),wrapped.state_dict().values()): assert torch.equal(a,b)
    for (a,sa),(b,sb) in zip(ra.state.items(),wa.state.items()): assert torch.equal(sa["momentum_buffer"],sb["momentum_buffer"])

def test_final_forward_determinism_independence_and_stop_order():
    config=DFCConfig(profile_id="fixture-bounded",maxIter=2,minLabels=0)
    a=torch.arange(35,dtype=torch.float32).reshape(1,1,5,7)/35; b=torch.flip(a,[-1])
    seed=derive_sample_seed("fixture:A")["sample_seed"]
    first=run_dfc(a,config,seed); _=run_dfc(b,config,derive_sample_seed("fixture:B")["sample_seed"]); after=run_dfc(a,config,seed)
    reference_map, reference_forwards, reference_state = reference_headless(a,seed,2,0)
    assert first.forward_count == first.update_count + 1 == 3
    assert first.stop_reason == "max_iter" and np.array_equal(first.raw_cluster_map,after.raw_cluster_map)
    assert np.array_equal(first.raw_cluster_map, reference_map) and first.forward_count == reference_forwards
    assert first.pre_stop_active_count == len(first.pre_stop_active_ids) and first.final_active_count == len(first.final_active_ids)
    # threshold is sampled pre-update but still gets an update, demonstrated on high threshold.
    hit=run_dfc(a,DFCConfig(profile_id="fixture-bounded",maxIter=4,minLabels=100),seed)
    assert hit.update_count == 1 and hit.forward_count == 2 and hit.stop_reason.startswith("pre_update")
    # Controlled label-count traces make the pre-update/post-update ordering explicit.
    assert stop_after_update(2,3) and stop_after_update(3,3) and not stop_after_update(5,3)
    assert [stop_after_update(n,3) for n in [5,3]] == [False,True]  # 5 -> 3: the second update stops.
