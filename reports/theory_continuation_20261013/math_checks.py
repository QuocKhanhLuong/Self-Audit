"""Tiny mathematical/source checks only: no dataset, optimizer, or training run."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/tmp/self-audit-scientific-20261012-d5a45f3')


def solve(a, b):
    """Exact Gauss-Jordan elimination, multiple right-hand sides."""
    n, k = len(a), len(b[0])
    m = [list(a[i]) + list(b[i]) for i in range(n)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if m[i][j])
        m[j], m[pivot] = m[pivot], m[j]
        scale = m[j][j]
        m[j] = [x / scale for x in m[j]]
        for i in range(n):
            if i != j:
                scale = m[i][j]
                m[i] = [x - scale*y for x,y in zip(m[i], m[j])]
    return [r[n:n+k] for r in m]


def entropy(q):
    return -sum(float(p)*math.log(float(p)) for p in q if p)


def harmonic_example(path_length):
    # Three boundary leaves [0,1,2], hub 3, dangling path 4,... .
    n = 4 + path_length
    w = [[F(i == j) for j in range(n)] for i in range(n)]
    for j, value in enumerate([F(4,5), F(1,10), F(1,10)]):
        w[3][j] = w[j][3] = value
    for j in range(3, n-1):
        w[j][j+1] = w[j+1][j] = F(1)
    p = [[x/sum(row) for x in row] for row in w]
    u = list(range(3,n))
    h = solve([[F(i == j)-p[i][j] for j in u] for i in u],
              [[p[i][j] for j in range(3)] for i in u])
    assert all(row == [F(4,5),F(1,10),F(1,10)] for row in h)
    strength = F(49,50)
    votes = [[F(1,150),F(1,150),F(1,150),F(49,50)],
             [F(0),F(1),F(0),F(0)], [F(0),F(1),F(0),F(0)]]
    r = [[(1-strength)/4+strength*x for x in row] for row in votes]
    q = [sum(h[0][j]*r[j][c] for j in range(3)) for c in range(4)]
    lower = (1-strength)/4+strength*F(4,5)*F(49,50)
    assert lower == F(19333,25000) == q[3]
    rho = float(strength)*(1-entropy(q)/math.log(4))
    assert q[3] >= F(7,10) and q[3]-max(q[:3]) >= F(1,5) and rho >= .05
    return dict(path_length=path_length, farthest_anchor_distance=path_length+1,
                hitting=[str(x) for x in h[0]], q=[str(x) for x in q],
                correct_probability=float(lower), reliability=rho,
                added_nodes=len(u), imperfect_good_anchor=True,
                corrupt_anchor=True, cross_class_edge=True)


alpha, beta = F(7,20), F(13,20)
spatial = (alpha/4+beta*F(4,5))/(1-beta*F(1,5))
temporal_self_only = (alpha/4+beta*F(6,7))/(1-beta*F(1,7))
return_mass = alpha*(1+beta/7+beta*beta/7)
neutral_cap = 1-F(3,4)*return_mass
assert spatial == F(81,116) and spatial < F(7,10)
assert temporal_self_only > F(7,10)
assert neutral_cap == F(22313,32000)
result = dict(kind='mathematical_sanity_checks_not_segmentation_evidence',
              training_run=False, datasets_read=False, optimizer_steps=0,
              exact_bounds=dict(distance_1=str(F(1,4)+F(3,4)*beta),
                                distance_2=str(F(1,4)+F(3,4)*beta**2),
                                spatial_cap=str(spatial), spatial_float=float(spatial),
                                temporal_self_loop_only=float(temporal_self_only),
                                temporal_return_cap=str(neutral_cap),
                                temporal_return_float=float(neutral_cap)),
              harmonic=[harmonic_example(d) for d in [1,3,7]])

# Evidence-state collision and duplicated-witness confidence inflation.
u = [F(1,4)]*4
for strength in [F(0),F(1,2),F(99,100)]:
    q = [(1-strength)/4+strength*x for x in u]
    assert q == u
    assert abs(float(strength)*(1-entropy(q)/math.log(4))) < 1e-14
duplicates = []
for r in [1,2,3]:
    s = 1-math.exp(-.4*r)
    q = [(1-s)/4]*3+[(1-s)/4+s]
    rho = s*(1-entropy(q)/math.log(4))
    duplicates.append(dict(identical_votes=r, mass=.4*r, strength=s,
                           maximum=max(q), reliability=rho,
                           accepted=max(q)>=.7 and rho>=.05))
assert not duplicates[0]['accepted'] and duplicates[-1]['accepted']
result['evidence_states'] = dict(uniform_collision=True, duplicate_votes=duplicates)

# Execute only existing small operators; no model fitting or data loading.
import numpy as np
from scipy import sparse
import torch
sys.path.insert(0, str(SOURCE/'src'))
from self_audit_guidance.core import anchored_solve
from self_audit_guidance.student import guidance_loss, predict_cine
p = sparse.csr_matrix(np.array([[0,1,0],[0,0,1],[0,0,1]],float))
r0 = np.full((3,4),.25);r0[2] = [0,0,0,1]
q, receipt = anchored_solve(p,r0,alpha=.35,tolerance=1e-12)
assert np.max(np.abs(q[:,3]-np.array([.566875,.7375,1.]))) < 1e-11
result['actual_solver_chain'] = dict(q=q.tolist(), receipt=receipt)

target = torch.tensor([[[[.10,.05]],[[.15,.05]],[[.15,.10]],[[.60,.80]]]])
logits = torch.zeros_like(target,requires_grad=True)
feature = torch.ones((1,2,1,2),requires_grad=True)
teacher = torch.ones_like(feature)
w = torch.tensor([[[.3,.6]]])
fl, sl, active = guidance_loss(dict(features=feature,logits=logits), teacher,target,w,True)
sl.backward()
assert active and target.max(1).values[0,0,0] < .7
assert logits.grad[0,:,0,0].abs().sum() > 0
assert logits.grad[0,1,0,0] != 0  # RV can be disallowed as target argmax, but CE still has its channel.
result['actual_soft_ce'] = dict(active=active, hard_rejected_pixel_has_gradient=True,
                              gradient=logits.grad.tolist(), feature_loss=float(fl.detach()),
                              semantic_loss=float(sl.detach()))

class ConstantStudent:
    def __init__(self): self.calls=0
    def __call__(self,x):
        self.calls+=1
        z=torch.log(torch.tensor([.05,.05,.10,.80])).reshape(1,4,1,1)
        return dict(logits=z.expand(len(x),4,*x.shape[-2:]))

model = ConstantStudent()
cine = np.zeros((1,1,2,2),np.float32)
ck = dict(status='REPRESENTATION_ONLY',config=dict(min_probability=.7,min_margin=.2),
          allowed_classes=[False,False,False,True])
a=predict_cine(model,ck,cine);assert (a==255).all() and model.calls==0
ck['status']='PARTIAL_SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED'
b=predict_cine(model,ck,cine);assert (b==3).all()
ck['allowed_classes']=[True,False,False,False]
c=predict_cine(model,ck,cine);assert (c==255).all()
result['actual_export'] = dict(representation_status_veto=True,permitted_class_export=True,
                             unpermitted_class_veto=True,bank_reliability_argument_absent=True)

# Loss-grid nullspace: bilinear 2x2 -> 1x1 averages the four logits.
target_coarse=torch.tensor([.1,.1,.1,.7]).reshape(1,4,1,1)
base=target_coarse.log().expand(1,4,2,2).clone()
perturbed=base.clone()
perturbed[0,1]=perturbed[0,1]+20*torch.tensor([[1.,-1.],[-1.,1.]])
feat=torch.ones((1,2,1,1));weight=torch.ones((1,1,1))
_, loss_base, _=guidance_loss(dict(features=feat,logits=base),feat,target_coarse,weight,True)
_, loss_perturbed, _=guidance_loss(dict(features=feat,logits=perturbed),feat,target_coarse,weight,True)
assert abs(float(loss_base-loss_perturbed)) < 1e-6
assert (base.argmax(1)==3).all() and (perturbed.argmax(1)==1).sum()==2
assert (perturbed.softmax(1)[:,1]>.99).sum()==2
result['loss_grid_nullspace'] = dict(base_semantic_loss=float(loss_base),
                                   perturbed_semantic_loss=float(loss_perturbed),
                                   changed_native_pixels=2,total_native_pixels=4,
                                   coarse_target_equal=True,network_realizability_claimed=False)

q=np.array([.1,.1,.2,.6]);p=np.array([.15,.15,.2,.5])
kl=float((q*np.log(q/p)).sum());delta=float(abs(q-p).max())
assert kl+1e-14 >= 2*delta**2
result['kl_coordinate_check'] = dict(kl=kl,coordinate_difference=delta,
                                   lower_bound=2*delta**2)
result['environment'] = dict(python=sys.version.split()[0],numpy=np.__version__,torch=torch.__version__,device='cpu')
(ROOT/'MATH_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
