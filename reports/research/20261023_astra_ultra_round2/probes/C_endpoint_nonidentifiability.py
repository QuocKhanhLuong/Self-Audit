#!/usr/bin/env python3
"""Exact finite probability counterexample, NOT NativeStudent/pseudo-gradient realization."""
from fractions import Fraction as F
import json
from pathlib import Path


def probs(u, v):
    delta = F(1, 100)
    return [[1 - x - 2 * delta, x, delta, delta] for x in (u, v)]


def risks(pair):
    u, v = pair
    # One positive and one negative: .5 outer Brier times .5 mass per side.
    return [((1-u)**2 + v**2)/4, F(1, 20000), F(1, 20000)]


def hard_confusion(pair):
    ps = probs(*pair)
    assert all(sum(p) == 1 and min(p) > 0 for p in ps)
    cm = [[0] * 5 for _ in range(4)]
    for y, p in zip([1, 0], ps):
        cm[y][max(range(4), key=lambda j: p[j])] += 1
    return cm


def accepted(before, after, tolerance=F(0)):
    a, b = risks(before), risks(after)
    return all(v <= u+tolerance for u,v in zip(a,b)) and sum(b)/3 <= sum(a)/3+tolerance


S, A, H = (F(4,5), F(1,10)), (F(9,10), F(1,20)), (F(3,5), F(1,5))
assert hard_confusion(S) == hard_confusion(A) == hard_confusion(H)
assert accepted(S,A) and not accepted(S,H)
trials=[]
for i in range(8):
    t = F(1, 2**i)
    point=(S[0]-F(1,5)*t,S[1]+F(1,10)*t)
    assert risks(point)[0] == F(1,80)*(1+t)**2
    assert not accepted(S,point,F(1,100000000))
    trials.append({"scale": str(t), "RV_risk": str(risks(point)[0]), "accepted_at_tolerance_1e_minus8": False})

K = 8
harmful=[(S[0]-F(1,5)*F(j,K),S[1]+F(1,10)*F(j,K)) for j in range(K+1)]
beneficial=[(S[0]+F(1,10)*F(j,K),S[1]-F(1,20)*F(j,K)) for j in range(K+1)]
assert all(not accepted(a,b) for a,b in zip(harmful,harmful[1:]))
assert all(accepted(a,b) for a,b in zip(beneficial,beneficial[1:]))
assert harmful[0] == beneficial[0] == S
# Both unguarded histories can end in exactly S after an unobserved labeled step.
harmful.append(S);beneficial.append(S)
assert harmful[-1] == beneficial[-1] == S
assert all(hard_confusion(p)==hard_confusion(S) for p in harmful+beneficial)

result={"status":"MANUFACTURED_ALGEBRAIC_IDENTIFIABILITY_COUNTEREXAMPLE",
        "scope":"Exact balanced one-vs-rest Brier arithmetic on valid4-class probabilities; no patient or actual NativeStudent candidate-generation evidence.",
        "risk_definition":"R_RV=((1-u)^2+v^2)/4; R_MYO=R_LV=1/20000; labels RV andBG; absent-class probabilities1/100.",
        "points":{name:{"RV_probabilities":[str(x) for x in p],"risks":[str(x) for x in risks(p)],"mean_risk":str(sum(risks(p))/3),"confusion":hard_confusion(p)} for name,p in [('S',S),('A',A),('H',H)]},
        "S_to_A_guard_accepts":True,"S_to_H_guard_accepts":False,"H_direction_backtracks":trials,
        "same_initial_and_final_soft_prediction_histories":{"pseudo_updates_per_history":K,"history1_harmful_pseudo_count":K,"history2_harmful_pseudo_count":0,"same_start":"S","same_final_after_labeled_compensation":"S","all_hard_confusions_equal":True},
        "conclusion":"Endpoint hard counts, even identical endpoint probabilities plus aggregate pseudo-step counts, do not determine prior audit-risk signs or would-be finite guard decisions without transaction data.",
        "limitations":["The compensating labeled transition is possible in an unconstrained probability trajectory, not claimed to be produced by the fixed training code.","No claim these probability paths are actual per-class pseudo-CE gradient directions or the sampled patient optimizer trajectory.","The probability map is manufactured; source-shaped realization is a separate question."]}
out=Path(__file__).resolve().parents[1]/'evidence'/'C_endpoint_nonidentifiability.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
