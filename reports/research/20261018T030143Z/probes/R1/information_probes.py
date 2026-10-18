"""Manufactured finite arithmetic only; no patient data, models, or training."""
import itertools
import json
from pathlib import Path

def ham(a,b):
    return sum(x!=y for x,y in zip(a,b))/len(a)

y0,y1=(1,3),(3,1)
risks=[(ham(p,y0),ham(p,y1)) for p in itertools.product(range(4),repeat=2)]
assert min(a+b for a,b in risks)==1
assert min(max(a,b) for a,b in risks)==0.5

u=[0.25]*4
q=[0.4,0.2,0.2,0.2]
z1=[(v-(1-0.2)*u[k])/0.2 for k,v in enumerate(q)]
z2=q.copy()
assert all(abs(a-b)<1e-12 for a,b in zip(z1,[1,0,0,0]))
assert all(abs((1-0.2)*u[k]+0.2*z1[k]-q[k])<1e-12 for k in range(4))

good=[0.1,0,0,0.9]
bad=[0,1,0,0]
t=[0.8*good[k]+0.2*bad[k] for k in range(4)]
beta,eta,delta=0.2,0.1,0.1
L=(1-beta)*(1-eta)
p=t.copy(); p[3]-=delta; p[1]+=delta
assert 2*L-1>2*delta
assert max(range(4),key=t.__getitem__)==3
assert max(range(4),key=p.__getitem__)==3
naive=[(4*good[k]+100*bad[k])/104 for k in range(4)]
assert max(range(4),key=naive.__getitem__)==1

true_z=[0.7,0.1,0.1,0.1]; h,hh=0.2,0.18
true_q=[(1-h)*u[k]+h*true_z[k] for k in range(4)]
hat_q=true_q.copy();hat_q[0]+=0.001;hat_q[1]-=0.001
hat_z=[(hat_q[k]-(1-hh)*u[k])/hh for k in range(4)]
err=max(abs(hat_z[k]-true_z[k]) for k in range(4))
bound=(0.001+abs(hh-h)*max(abs(true_z[k]-u[k]) for k in range(4)))/hh
assert err<=bound+1e-12
# A support estimate error of 0.01 at true support 0.02 can leave the simplex.
small_z=[0.8,0.1,0.05,0.05]
small_q=[0.98*u[k]+0.02*small_z[k] for k in range(4)]
invalid_z=[(small_q[k]-0.99*u[k])/0.01 for k in range(4)]
assert min(invalid_z)<0 and max(invalid_z)>1

# A deterministic candidate set containing a wrong label has conditional
# ambiguity probability 1 at that X,Y, regardless of cohort size.
S={1,3}; true_label=1; wrong_label=3
assert wrong_label in S and true_label in S
conditional_wrong_inclusion=1.0

report={
 'scope':'MANUFACTURED_ARITHMETIC_NOT_PATIENT_EVIDENCE',
 'two_world_min_risk_sum':min(a+b for a,b in risks),
 'two_world_minimax_hamming':min(max(a,b) for a,b in risks),
 'same_Q_different_support':{'Q':q,'h_first':0.2,'Z_first':z1,'h_second':1,'Z_second':z2},
 'robust_pooling':{'target':t,'perturbed_student':p,'L':L,'lower_margin':2*L-1,'required_transfer':2*delta},
 'duplicate_wrong_source':{'naive_target':naive,'naive_argmax':1,'lineage_weight_capped_argmax':3},
 'inverse_support_sensitivity':{'actual_error':err,'bound':bound,'invalid_small_support_inverse':invalid_z},
 'deterministic_partial_label_wrong_inclusion':conditional_wrong_inclusion,
 'assertions':'PASS'
}
Path(__file__).with_name('probe_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
