"""Manufactured weighted-gradient identity. No project imports or patient data."""
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
results=[]
for seed in [0, 1, 7, 19, 43]:
    torch.manual_seed(seed)
    x=torch.randn(5,3,dtype=torch.float64)
    theta=(.2*torch.randn(3,4,dtype=torch.float64)).requires_grad_()
    weights=torch.tensor([.03,.07,.15,.25,.5],dtype=torch.float64)[:,None]
    reference=torch.randn(5,4,dtype=torch.float64).softmax(-1)
    probabilities=(x@theta).softmax(-1)
    energy=(weights*(probabilities-reference).square()).sum()/2
    probability_gradient,=torch.autograd.grad(energy,probabilities,retain_graph=True)
    direct,=torch.autograd.grad(energy,theta)
    metric_gradient=probability_gradient.detach()/weights
    tangent=metric_gradient-metric_gradient.mean(-1,keepdim=True)
    tau=.001
    proposal=probabilities.detach()-tau*tangent
    assert (proposal>0).all()
    assert torch.allclose(proposal.sum(-1),torch.ones(5,dtype=torch.float64))
    p=(x@theta).softmax(-1)
    transfer=(weights*(p-proposal).square()).sum()/(2*tau)
    copied,=torch.autograd.grad(transfer,theta)
    gap=(direct-copied).abs().max().item()
    assert gap<1e-12
    wrong_tangent=probability_gradient.detach()-probability_gradient.detach().mean(-1,keepdim=True)
    wrong_proposal=probabilities.detach()-tau*wrong_tangent
    naive=(weights*((x@theta).softmax(-1)-wrong_proposal).square()).sum()/(2*tau)
    wrong_gradient,=torch.autograd.grad(naive,theta)
    wrong_gap=(direct-wrong_gradient).abs().max().item()
    assert wrong_gap>1e-6
    displaced=(theta.detach()-.1*direct).requires_grad_()
    p2=(x@displaced).softmax(-1)
    e2=(weights*(p2-reference).square()).sum()/2
    t2=(weights*(p2-proposal).square()).sum()/(2*tau)
    de2,=torch.autograd.grad(e2,displaced,retain_graph=True)
    dt2,=torch.autograd.grad(t2,displaced)
    stale_gap=(de2-dt2).abs().max().item()
    assert stale_gap>1e-6
    results.append(dict(seed=seed,same_point_max_gradient_error=gap,wrong_metric_max_error=wrong_gap,stale_origin_max_error=stale_gap))
receipt=dict(scope="CPU manufactured gradient check; not training/anatomical evidence",torch=torch.__version__,dtype="float64",checks=results,status="PASS")
Path(__file__).with_suffix('.json').write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
