"""Reproducible mathematical examples; manufactured, not segmentation evidence."""
import json
from pathlib import Path
import torch
from layer_upgrade.layers import compose_logits,block_mean
from layer_upgrade.losses import soft_ce
from layer_upgrade.guarded import guarded_sgd_step
from layer_upgrade.refine import FrozenImageEnergy,refine_distribution


def main():
    torch.set_num_threads(2);torch.manual_seed(20261015)
    a=torch.randn(1,4,3,4,dtype=torch.double,requires_grad=True)
    r=torch.randn(1,4,12,16,dtype=torch.double,requires_grad=True)
    logits=compose_logits(a,r.tanh(),4)
    q=torch.randn_like(a).softmax(1)
    loss=soft_ce(block_mean(logits,4),q,torch.ones_like(q[:,0]))
    ga,gr=torch.autograd.grad(loss,(a,r))
    p=torch.nn.Parameter(torch.zeros(2,dtype=torch.double))
    def guards():return [.5*(p[0]+1).square(),.5*(p[0]-p[1]-1).square()]
    def obj():return guards()[0]
    guarded=guarded_sgd_step([p],obj,guards,step_size=.2,atol=1e-12)
    p0=torch.tensor([[.05,.8,.1,.05],[.7,.2,.05,.05]],dtype=torch.double)
    energy=FrozenImageEnergy(
        torch.tensor([[-12.,0.,0.,0.],[0.,-12.,0.,0.]],dtype=torch.double),
        torch.tensor([[0],[1]]),torch.ones(1),
        torch.tensor([[0.,1.,0.,0.],[1.,0.,0.,0.]]),torch.ones(2),lam=.2,beta=.3)
    proposal,receipt=refine_distribution(p0.log(),energy,steps=3)
    results=dict(
        kind='mathematical_tensor_checks_not_patient_evidence',
        patient_arrays_read=False,pretrained_weights_loaded=False,
        coarse_detail=dict(max_identity_error=float((block_mean(logits,4)-a).abs().max().detach()),
                           detail_gradient_max=float(gr.abs().max()),coarse_gradient_norm=float(ga.norm())),
        selected_parameter_guard=guarded,
        image_refinement=dict(initial=p0.tolist(),final=proposal.tolist(),receipt=receipt,
                              interpretation='deliberately informative manufactured unary corrects wrong anchors; no real-image validation'))
    Path('PROBE_RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
