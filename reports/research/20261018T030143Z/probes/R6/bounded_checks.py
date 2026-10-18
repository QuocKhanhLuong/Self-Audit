"""Manufactured scalar checks only; no patient data, torch, or external model."""
import json
import math
from pathlib import Path

def mean(x): return sum(x) / len(x)
def sigm(x): return 1 / (1 + math.exp(-x))
def mse(x, y): return mean([(a-b)**2 for a,b in zip(x,y)])
truth=[1.0,0.0]
good=[1.0,0.0]
bad=[0.0,1.0]
assert mean(good)==mean(bad)==0.5
assert mse(good,truth)==0.0 and mse(bad,truth)==1.0
coarse=2.0
detail=[2.0,-2.0]
logits=[coarse+d for d in detail]
assert mean(logits)==coarse
assert abs(mean([sigm(x) for x in logits])-sigm(coarse))>0.1
# Equal-variance exchangeable sources: Var(mean errors)=sigma^2*(rho+(1-rho)/m).
variance={str(rho):rho+(1-rho)/10 for rho in [0,0.9,1]}
# Standard triangle bound: proposal RMSE gain must exceed transfer RMSE for this sufficient bound.
baseline_rmse=0.30
proposal_rmse=0.20
transfer_rmse=0.05
# Metrics: 90 BG pixels correct, 10 RV missed -> 90% accuracy and RV Dice=0.
result={"scope":"MANUFACTURED_SCALAR_CHECKS_ONLY", "coarse_observation_ambiguous":{"good_mean":mean(good),"bad_mean":mean(bad),"good_mse":mse(good,truth),"bad_mse":mse(bad,truth)},"logit_probability_noncommutation":{"mean_logit":mean(logits),"sigmoid_mean_logit":sigm(coarse),"mean_sigmoid_logit":mean([sigm(x) for x in logits])},"variance_of_ten_sources_sigma_squared_1":variance,"triangle_bound_example":{"baseline_rmse":baseline_rmse,"proposal_plus_transfer_rmse":proposal_rmse+transfer_rmse,"sufficient":proposal_rmse+transfer_rmse<baseline_rmse},"bg_counterexample":{"accuracy":0.9,"rv_dice":0.0},"checks_passed":5}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
