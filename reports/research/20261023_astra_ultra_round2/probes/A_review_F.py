"""Independent reconstruction of F's saved and rounded actual-generator counterexample."""
from pathlib import Path
import copy,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from self_audit.control import make_candidates,controlled_head_step
from self_audit.evaluation import class_metrics,confusion_matrix
torch.set_num_threads(1)
BASE=Path(__file__).resolve().parents[1];E=BASE/"evidence"
raw=json.loads((E/"F_search_result.json").read_text())["found"]
tensor=lambda x,dtype=torch.float64:torch.tensor(x,dtype=dtype)
W=tensor([[-.974,.475],[-1.134,1.224],[.775,-1.788],[.742,.170]])[:,:,None,None]
b=tensor([.292,.263,-.359,-.007])
u=tensor([[2.407,-2.527],[-.645,2.180]])[None,:,None,:]
a=tensor([[-.577,.654,.175,.377,.037,.902],[-1.160,-.233,1.231,-.200,.005,-.015]])[None,:,None,:]
y=tensor([[[0,1,1,2,3,0]]],torch.int64)
eta=9.603;r=eta/.001
liftW=torch.cat([W/r**.5,torch.zeros(4,1,1,1,dtype=torch.float64)],1)
lift=lambda z:torch.cat([z*r**.5,torch.full((1,1,1,z.shape[-1]),(r-1)**.5,dtype=torch.float64)],1)
variants=[("original_selected",tensor(raw["weight"]),tensor(raw["bias"]),tensor(raw["unlabeled_features"]),tensor(raw["audit_features"]),tensor(raw["labels"],torch.int64),raw["step_size"]),
          ("rounded_independent",W,b,u,a,y,eta),("lifted_independent",liftW,b,lift(u),lift(a),y,.001)]
results={};effects={}
for name,w,bias,uf,af,labels,step in variants:
    h=torch.nn.Conv2d(w.shape[1],4,1).double()
    with torch.no_grad():h.weight.copy_(w);h.bias.copy_(bias)
    teacher=copy.deepcopy(h).eval().requires_grad_(False)
    assert all(torch.equal(p,q) for p,q in zip(h.parameters(),teacher.parameters()))
    snapshot={k:v.detach().clone() for k,v in h.state_dict().items()}
    q=teacher(uf).softmax(1).detach();before=h(af).argmax(1).detach().numpy()
    ds,ids,info=make_candidates(h,uf,q,.7,step)
    assert ids==["pseudo_class_1","pseudo_class_2"] and info["accepted_pixels"]==2
    conf,py=q.max(1);valid=conf>=.7
    g=torch.autograd.grad((F.cross_entropy(h(uf),py,reduction="none")*valid).sum()/valid.sum(),tuple(h.parameters()))
    summed=tuple(sum(d[k] for d in ds) for k in range(2))
    err=max(float((summed[k]+step*g[k]).abs().max()) for k in range(2))
    assert err<1e-11 and all(torch.equal(v,snapshot[k]) for k,v in h.state_dict().items())
    effects[name]=torch.stack([F.conv2d(af,*d) for d in ds])
    joint=copy.deepcopy(h);scalar=copy.deepcopy(h)
    jr=controlled_head_step(joint,af,labels,ds,ids,method="joint_guard",ridge=1e-6,budget=0.,tolerance=1e-8,backtracks=8)
    sr=controlled_head_step(scalar,af,labels,[summed],["aggregate"],method="independent_guard",ridge=2e-6,budget=0.,tolerance=1e-8,backtracks=8)
    metric=lambda pred:class_metrics(confusion_matrix(labels.numpy(),pred))[1]
    bm=metric(before);jm=metric(joint(af).argmax(1).detach().numpy());sm=metric(scalar(af).argmax(1).detach().numpy())
    assert jr["positive_pixels"]==[[2],[1],[1]] and jr["scale"]>0 and jm["dice"]<bm["dice"]
    assert all(new<old-1e-8 for new,old in zip(jr["after"],jr["before"]))
    results[name]=dict(step=step,source_ids=ids,proposal=info,gradient_sum_error=err,student_state_unchanged_by_generation=True,
                       teacher_copy_exact=True,before_rv=bm,after_joint_rv=jm,after_scalar_rv=sm,joint=jr,scalar=sr,
                       max_abs_feature=float(max(uf.abs().max(),af.abs().max())))
lift_error=float((effects["rounded_independent"]-effects["lifted_independent"]).abs().max())
assert lift_error<1e-11
result=dict(kind="INDEPENDENT_SYNTHETIC_CPU_REVIEW_OF_F_GENERATED_DIRECTIONS",script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            original_F_search_sha256=hashlib.sha256((E/"F_search_result.json").read_bytes()).hexdigest(),
            scipy_not_rerun_search=True,torch=torch.__version__,numpy=np.__version__,results=results,
            lifting_max_source_logit_effect_error=lift_error,scope="Canonical affine generator/control seams; no native feature or EMA-history reachability claim.")
(E/"A_review_F.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":"PASS","lift_error":lift_error,"variants":{k:{"before":v["before_rv"]["dice"],"after":v["after_joint_rv"]["dice"],"scale":v["joint"]["scale"],"gradient_error":v["gradient_sum_error"]} for k,v in results.items()}},indent=2))
