"""Manufactured CPU-only mechanism checks; no patient data or trained weights."""
from __future__ import annotations
import copy, hashlib, json, math, platform, sys
from pathlib import Path
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[4]
sys.path.insert(0,str(REPO/'experiments/layer_guided_student_v1'))
from layer_upgrade.guarded import project_direction, guarded_sgd_step

def scalar(x): return float(x.detach()) if isinstance(x,torch.Tensor) else float(x)
def kl(q,p): return q*math.log(q/p)+(1-q)*math.log((1-q)/(1-p))
result={'scope':'MANUFACTURED_CPU_ONLY_NOT_PATIENT_NOT_ANATOMY','torch':torch.__version__,'python':platform.python_version(),'seed':0}

# A. Same shared logit at two contradictory strata. This deliberately constructs
# a representation collision, not a claim that current features collide.
q_bg,q_rv=.02,.98
p0,p1=.5,1/(1+math.exp(1))
a0,a1=kl(q_bg,p0),kl(q_bg,p1)
r0,r1=kl(q_rv,p0),kl(q_rv,p1)
result['A_total_KL_vs_RV']={'mass':[.99,.01],'BG_KL':[a0,a1],'RV_KL':[r0,r1],'total_KL':[.99*a0+.01*r0,.99*a1+.01*r1],'shared_logit':[0,-1]}
assert result['A_total_KL_vs_RV']['total_KL'][1]<result['A_total_KL_vs_RV']['total_KL'][0] and r1>r0
# Equalizing sample weights cannot recover separability from identical features.
result['A_balanced_collision']={'balanced_optimal_probability':.5,'RV_proxy_KL_at_optimum':kl(q_rv,.5),'global_gradient_at_logit_zero':.5*(.5-q_bg)+.5*(.5-q_rv)}
# Distinct logits do not conflict; imbalance only slows rare-stratum movement.
logits=torch.zeros(2,requires_grad=True); targets=torch.tensor([q_bg,q_rv]); masses=torch.tensor([.99,.01])
loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none')*masses).sum(); g,=torch.autograd.grad(loss,logits)
p_after=torch.sigmoid(logits.detach()-g)
result['A_independent_logits']={'gradient':g.tolist(),'RV_KL_before':r0,'RV_KL_after':kl(q_rv,float(p_after[1]))}
assert result['A_independent_logits']['RV_KL_after']<r0

# B1. First-step AdamW adaptive scaling violates a feasible raw direction even
# without weight decay or momentum history.
x=torch.nn.Parameter(torch.zeros(2)); guard_grad=torch.tensor([2.,1.]); candidate=torch.tensor([-1.,3.])
d,receipt=project_direction(candidate,guard_grad[None,:])
opt=torch.optim.AdamW([x],lr=.1,betas=(0.,0.),eps=1e-8,weight_decay=0.)
before=scalar(.5*(guard_grad@x+1)**2); x.grad=d.clone(); opt.step(); after=scalar(.5*(guard_grad@x+1)**2)
result['B1_adam_scaling']={'raw_guard_dot_direction':scalar(guard_grad@d),'direction':d.tolist(),'actual_displacement':x.detach().tolist(),'guard_before':before,'guard_after':after,'projection':receipt}
assert scalar(guard_grad@d)>0 and after>before

# B2. Inherited first moment can point opposite a currently safe gradient.
x=torch.nn.Parameter(torch.tensor(1.)); opt=torch.optim.AdamW([x],lr=.1,weight_decay=0.)
x.grad=torch.tensor(-10.); opt.step()
with torch.no_grad(): x.fill_(1.)
before=scalar(x*x/2); x.grad=torch.tensor(1.); opt.step(); after=scalar(x*x/2)
result['B2_momentum']={'raw_gradient':1.,'guard_before':before,'guard_after':after,'actual_parameter':scalar(x)}
assert after>before

# B3. Decoupled decay itself can oppose a proxy descent.
x=torch.nn.Parameter(torch.tensor(1.)); opt=torch.optim.AdamW([x],lr=.1,betas=(0.,0.),weight_decay=10.)
before=scalar((x-2)**2/2); x.grad=torch.tensor(-1.); opt.step(); after=scalar((x-2)**2/2)
result['B3_decay']={'guard_before':before,'guard_after':after,'actual_parameter':scalar(x)}
assert after>before

# B4. Parameter-only rollback does not undo optimizer transition.
def init():
 p=torch.nn.Parameter(torch.tensor(1.)); o=torch.optim.AdamW([p],lr=.1,weight_decay=.01); p.grad=torch.tensor(.7);o.step();return p,o
base,base_opt=init(); saved_param=base.detach().clone();saved_state=copy.deepcopy(base_opt.state_dict())
paths={}
for mode in ['control','parameter_only','full_state']:
 p,o=init()
 if mode!='control':
  p.grad=torch.tensor(-10.);o.step()
  with torch.no_grad():p.copy_(saved_param)
  if mode=='full_state':o.load_state_dict(copy.deepcopy(saved_state))
 p.grad=torch.tensor(.2);o.step();paths[mode]={'parameter':scalar(p),'state_step':scalar(o.state[p]['step'])}
result['B4_rollback']=paths
assert abs(paths['control']['parameter']-paths['full_state']['parameter'])<1e-12
assert abs(paths['control']['parameter']-paths['parameter_only']['parameter'])>1e-3

# C. Opposed nonzero guard gradients admit only zero in one dimension.
d,receipt=project_direction(torch.tensor([1.]),torch.tensor([[1.],[-1.]]))
result['C_opposed_guards']={'direction':d.tolist(),'receipt':receipt};assert d.abs().max()<1e-12
# Strict finite-step guard at a wrong proxy minimizer forbids correction even
# though its zero derivative imposes no linearized constraint.
x=torch.nn.Parameter(torch.tensor(0.))
receipt=guarded_sgd_step([x],lambda:(x-1)**2/2,lambda:[x*x/2],step_size=.1,max_backtracks=8,atol=0.)
result['C_wrong_proxy_lock']={'receipt':receipt,'parameter':scalar(x),'true_loss':scalar((x-1)**2/2)}
assert not receipt['accepted'] and scalar(x)==0.
delta=.1;result['C_slack_escape']={'delta':delta,'proxy_increase':delta*delta/2,'true_loss_before':.5,'true_loss_after':(delta-1)**2/2}

# D. Feature-only batch moves shared encoder and semantic output, despite no
# semantic-head parameter update. Target distribution is frozen at baseline.
e=torch.nn.Parameter(torch.tensor(1.)); h=torch.nn.Parameter(torch.tensor(1.)); opt=torch.optim.AdamW([e,h],lr=.1,weight_decay=0.)
p_before=scalar(torch.sigmoid(e*h)); loss=(e-2)**2/2;opt.zero_grad(set_to_none=True);loss.backward();head_grad_none=h.grad is None;opt.step();p_after=scalar(torch.sigmoid(e*h))
result['D_inactive_semantic']={'head_grad_is_none':head_grad_none,'head_after':scalar(h),'encoder_after':scalar(e),'RV_probability_before':p_before,'RV_probability_after':p_after,'frozen_target_KL_after':kl(p_before,p_after)}
assert head_grad_none and scalar(h)==1. and p_before!=p_after

# E. Smooth noisy-gradient certificate with known manufactured constants.
# R(theta)=.5 ||theta-a||^2, exact L=1. Proxy g differs by a known bounded vector.
torch.manual_seed(0);cases=[]
for i in range(12):
 theta=torch.randn(3);a=torch.randn(3);g_true=theta-a;noise=.05*torch.randn(3);g_hat=g_true+noise;delta=noise.norm();d=g_hat; margin=g_hat@d-delta*d.norm()
 eta=min(.2,scalar(margin/(d@d))) if margin>0 else 0.
 step=-eta*d;true_change=scalar(.5*(theta+step-a).square().sum()-.5*(theta-a).square().sum())
 upper=scalar(g_hat@step+delta*step.norm()+.5*step.square().sum())
 cases.append({'eta':eta,'delta':scalar(delta),'true_change':true_change,'bound':upper,'valid':true_change<=upper+1e-12})
assert all(c['valid'] and c['bound']<=1e-12 for c in cases)
result['E_known_noise_bound']={'cases':cases,'all_valid':True,'note':'delta and L known only because manufactured quadratic truth is supplied'}

# F. Conditional feature stability: bounded head plus bounded representation
# drift bound logit change; feature cosine alone allows arbitrary rescaling.
w=torch.tensor([2.,-1.]); f=torch.tensor([1.,1.]);f2=10*f
result['F_cosine_vs_logit']={'cosine_distance':scalar(1-torch.nn.functional.cosine_similarity(f[None],f2[None])),'logit_before':scalar(w@f),'logit_after':scalar(w@f2)}
assert abs(result['F_cosine_vs_logit']['cosine_distance'])<1e-12 and scalar(w@f2)!=scalar(w@f)
result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(ROOT/'RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
