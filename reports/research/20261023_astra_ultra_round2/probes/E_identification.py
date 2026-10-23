#!/usr/bin/env python3
"""Manufactured sequential identification probe; no production imports/data/model.

Affine four-class probability toy, identity supervised reference, handcrafted
initial source directions and a declared EMA-coupled second proposal map.
This is NOT a simulation of the v2 CE candidate generator or an anatomy result.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random

RIDGE = .001
DECAY = .5
METHODS = ("P", "S", "J")
SOURCE_SHA = "637f38ced217fe1db2347a87924695533056d8f0"
EXPERIMENT_SHA = "8960b6baac4c6600553d900c303f192c2fcfa4f0"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def probabilities(theta, support):
    value = theta[0 if support == "A" else 1]
    p = [.98-value, value, .01, .01]
    assert min(p) >= -1e-12 and abs(sum(p)-1) < 1e-12
    return p


def risks(theta, support):
    p = probabilities(theta, support)
    # Each manufactured audit support has one RV-positive observation.
    return [.5*(p[1]-1)**2, .5*p[2]**2, .5*p[3]**2]


def candidate_map(state, step, teacher_override=None):
    teacher = state["teacher"] if teacher_override is None else teacher_override
    if step == 0:
        return [[.35, -.35], [-.15, .35]], {"map": "HANDCRAFTED_INITIAL", "pseudo_class": None}
    q = probabilities(teacher, "B")
    pseudo = max(range(4), key=q.__getitem__)
    # Explicit abstract feedback response, not claimed to be a CE gradient.
    direction = [.05, .05 if pseudo == 1 else -.05]
    return [direction], {"map": "EMA_ARGMAX_RESPONSE", "pseudo_class": pseudo, "teacher_B": q}


def objective(theta, directions, alpha):
    trial = [theta[k]+sum(a*d[k] for a, d in zip(alpha, directions)) for k in range(2)]
    return sum(risks(trial, "A"))/3 + .5*RIDGE*sum(a*a for a in alpha)


def exact_box_oracle(theta, directions):
    """Enumerate faces; free solution is rank-one ridge, m <= 2 here.

    The other two risk groups are constant. All enumerated candidates are also
    checked against the finite zero-budget risk guard. No SLSQP is used.
    """
    m = len(directions)
    v = [d[0] for d in directions]
    r = 1-theta[0]
    candidates = []
    for face in itertools.product((-1, 0, 1), repeat=m):
        free = [i for i, f in enumerate(face) if f == -1]
        alpha = [0. if f == -1 else float(f) for f in face]
        residual = r-sum(v[i]*alpha[i] for i in range(m) if i not in free)
        denom = 3*RIDGE+sum(v[i]*v[i] for i in free)
        for i in free:
            alpha[i] = residual*v[i]/denom
        if any(a < -1e-12 or a > 1+1e-12 for a in alpha):
            continue
        alpha = [min(1., max(0., a)) for a in alpha]
        trial = [theta[k]+sum(a*d[k] for a, d in zip(alpha, directions)) for k in range(2)]
        if all(x <= y+1e-12 for x,y in zip(risks(trial,"A"),risks(theta,"A"))):
            candidates.append((objective(theta,directions,alpha),alpha))
    assert candidates
    return min(candidates,key=lambda item:item[0])[1]


def transaction(state, step, method, teacher_override=None):
    """Pure function: state and candidate inputs are never mutated."""
    before_hash = digest(state)
    theta = state["student"]
    directions, info = candidate_map(state,step,teacher_override)
    m = len(directions)
    if method == "P":
        alpha = [1.]*m
    elif method == "S":
        v = sum(d[0] for d in directions)
        # Strict restriction alpha=a*1: ridge m*RIDGE; mean over 3 groups.
        a = min(1.,max(0.,v*(1-theta[0])/(v*v+3*m*RIDGE)))
        alpha = [a]*m
    else:
        alpha = exact_box_oracle(theta,directions)
    trial = [theta[k]+sum(a*d[k] for a,d in zip(alpha,directions)) for k in range(2)]
    before = risks(theta,"A")
    after = risks(trial,"A")
    guarded = method != "P"
    accepted = not guarded or all(x <= y+1e-12 for x,y in zip(after,before))
    if not accepted:trial=theta[:]
    assert digest(state)==before_hash
    return {"method":method,"step":step,"reference_state_sha256":before_hash,
            "candidate_sha256":digest(directions),"directions":directions,"candidate_info":info,
            "alpha":alpha,"scalar_ridge":m*RIDGE if method=="S" else None,
            "accepted":accepted,"guarded":guarded,"student_after":trial,
            "audit_A_before":before,"audit_A_after":risks(trial,"A"),
            "shadow_B_before":risks(theta,"B"),"shadow_B_after":risks(trial,"B"),
            "effective_delta":[x-y for x,y in zip(trial,theta)],
            "finite_trial_count":1,"jvp_count":0,"oracle":"EXACT_AFFINE_TOY"}


def initial_state():
    return {"student":[.5,.5],"teacher":[.5,.5],"buffers":{"sentinel":7},
            "optimizer":{"steps":0,"momentum":[0.,0.]},"ema_updates":0,
            "reference":"IDENTITY_TO_ISOLATE_EXTRA_TRANSACTION"}


def commit(state, outcome):
    state["student"]=outcome["student_after"][:]
    state["teacher"]=[DECAY*t+(1-DECAY)*s for t,s in zip(state["teacher"],state["student"])]
    state["ema_updates"]+=1


def live_hash(state, rng):
    return digest({"state":state,"rng":rng.getstate()})


def reference_replay(order=METHODS, repeats=1):
    state=initial_state(); rng=random.Random(20261023); rows=[]
    for step in range(2):
        exogenous_draw=rng.random() # one scheduled live draw, regardless of shadows
        before=live_hash(state,rng); shadows={}
        for _ in range(repeats):
            for method in order:
                # A shadow receives copies, including an isolated RNG if it needs one.
                shadow_state=copy.deepcopy(state)
                shadow_rng=random.Random(); shadow_rng.setstate(rng.getstate())
                shadow_rng.random() # deliberately consume shadow randomness
                shadows[method]=transaction(shadow_state,step,method)
                assert live_hash(state,rng)==before
        live=transaction(state,step,"P")
        commit(state,live)
        rows.append({"step":step,"exogenous_draw":exogenous_draw,"pre_shadow_live_sha256":before,
                     "shadow_live_unchanged":True,"shadows":shadows,"committed_method":"P",
                     "ema_updates":state["ema_updates"],"post_commit_live_sha256":live_hash(state,rng)})
    return {"final_state":state,"final_live_sha256":live_hash(state,rng),"rows":rows}


def independent_policy(method,freeze_teacher=False):
    state=initial_state(); rows=[]
    for step in range(2):
        override=[.5,.5] if freeze_teacher and step==1 else None
        outcome=transaction(state,step,method,override)
        commit(state,outcome); rows.append(outcome)
    a,b=risks(state["student"],"A")[0],risks(state["student"],"B")[0]
    return {"method":method,"freeze_teacher_at_step1":freeze_teacher,"rows":rows,
            "final_state":state,"final_RV_A":a,"final_RV_B":b,"equal_support_RV_risk":(a+b)/2}


def main():
    reference=reference_replay()
    no_shadow=reference_replay(repeats=0)
    checks=[]
    for order in itertools.permutations(METHODS):
        for repeat in (1,3):
            check=reference_replay(order,repeat)
            assert check["final_live_sha256"]==no_shadow["final_live_sha256"]
            checks.append({"order":list(order),"repeats":repeat,"same_live_endpoint":True})
    policies={m:independent_policy(m) for m in METHODS}
    frozen={m:independent_policy(m,True) for m in METHODS}
    for row in reference["rows"]:
        assert len({x["candidate_sha256"] for x in row["shadows"].values()})==1
        assert len({x["reference_state_sha256"] for x in row["shadows"].values()})==1
    assert reference["final_state"]["ema_updates"]==2
    assert reference["final_state"]["buffers"]==initial_state()["buffers"]
    assert reference["final_state"]["optimizer"]==initial_state()["optimizer"]
    first=reference["rows"][0]["shadows"]
    assert first["J"]["alpha"]==[1.,0.] and first["S"]["alpha"]==[1.,1.]
    gain_a=first["S"]["audit_A_after"][0]-first["J"]["audit_A_after"][0]
    gain_b=first["S"]["shadow_B_after"][0]-first["J"]["shadow_B_after"][0]
    assert gain_a>0 and gain_b<0
    policy_gain_a=policies["S"]["final_RV_A"]-policies["J"]["final_RV_A"]
    policy_gain_b=policies["S"]["final_RV_B"]-policies["J"]["final_RV_B"]
    assert policy_gain_a>0 and policy_gain_b<0
    assert policies["J"]["rows"][1]["candidate_info"]["pseudo_class"]==0
    assert policies["S"]["rows"][1]["candidate_info"]["pseudo_class"]==1
    assert reference["rows"][1]["shadows"]["J"]["candidate_info"]["pseudo_class"]==1
    assert policies["J"]["final_RV_B"]>frozen["J"]["final_RV_B"]
    result={"schema":"astra.E.identification.synthetic.v1","status":"PASS",
            "python":platform.python_version(),"production_imports":False,"patient_gpu_execution":False,
            "canonical_source_context_sha":SOURCE_SHA,"experiment_context_sha":EXPERIMENT_SHA,
            "probe_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model":"affine four-class probabilities; identity supervised reference; handcrafted proposal map",
            "candidate_generator_parity":"NOT_CLAIMED; no v2 CE or softmax/JVP reproduction",
            "ridge":RIDGE,"ema_decay":DECAY,"reference_replay":reference,
            "shadow_invariance":{"no_shadow_endpoint_sha256":no_shadow["final_live_sha256"],"cases":checks,
                                 "orders":6,"repeat_settings":2,"all_passed":True},
            "independent_policies":policies,"teacher_frozen_intervention":frozen,
            "contrast_units":"RV half-Brier; corresponding three-group mean differences are divided by three",
            "B_support_role":"Toy input/feature support enters teacher proposal map; manufactured B labels/scores are score-only; not untouched held-out data",
            "contrasts":{"first_transaction_J_advantage_A":gain_a,
                         "first_transaction_J_advantage_B":gain_b,
                         "final_policy_J_advantage_A":policy_gain_a,
                         "final_policy_J_advantage_B":policy_gain_b,
                         "final_policy_J_advantage_equal_support":policies["S"]["equal_support_RV_risk"]-policies["J"]["equal_support_RV_risk"],
                         "EMA_feedback_extra_J_B_risk":policies["J"]["final_RV_B"]-frozen["J"]["final_RV_B"]},
            "interpretation":"Logical nonidentification counterexample; not patient frequency or source failure."}
    out=Path(__file__).resolve().parents[1]/"evidence"/"E_identification.json"
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"receipt":str(out),"contrasts":result["contrasts"],
                      "invariance_cases":len(checks)},indent=2))


if __name__=="__main__":main()
