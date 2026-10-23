"""Independent review checks for E's affine toy; no writes to E artifacts."""
from pathlib import Path
import copy
import hashlib
import itertools
import json
import runpy

HERE = Path(__file__).resolve().parents[1]
TARGET = HERE / "probes/E_identification.py"
ns = runpy.run_path(str(TARGET))
probabilities, commit = ns["probabilities"], ns["commit"]


def main():
    branches=[]
    for method in ["P","S","J"]:
        for frozen in [False,True]:
            state=ns["initial_state"]()
            for step in range(2):
                override=[.5,.5] if frozen and step==1 else None
                directions, info=ns["candidate_map"](state,step,override)
                corners=[]
                for alpha in itertools.product([0.,1.],repeat=len(directions)):
                    theta=[state["student"][k]+sum(a*d[k] for a,d in zip(alpha,directions)) for k in range(2)]
                    ps={support:probabilities(theta,support) for support in ["A","B"]}
                    assert all(min(p)>=0 and abs(sum(p)-1)<1e-15 for p in ps.values())
                    corners.append({"alpha":alpha,"theta":theta,"probabilities":ps})
                before=copy.deepcopy(state)
                result=ns["transaction"](state,step,method,override)
                assert state==before
                teacher_before=state["teacher"][:]
                commit(state,result)
                ema_checks={}
                for support in ["A","B"]:
                    lhs=probabilities(state["teacher"],support)
                    rhs=[ns["DECAY"]*a+(1-ns["DECAY"])*b for a,b in
                         zip(probabilities(teacher_before,support),probabilities(state["student"],support))]
                    error=max(abs(a-b) for a,b in zip(lhs,rhs))
                    assert error < 2e-16
                    ema_checks[support]=error
                branches.append({"method":method,"frozen_teacher_intervention":frozen,"step":step,
                                 "all_box_corners":corners,"candidate_info":info,
                                 "parameter_vs_probability_ema_max_error":ema_checks})
    policies={m:ns["independent_policy"](m) for m in ["P","S","J"]}
    teacher_states={"initial":[.5,.5],"after_J":[.675,.325],"after_S":[.6,.5]}
    decisions={}
    for label,teacher in teacher_states.items():
        q=probabilities(teacher,"B")
        ordered=sorted(q,reverse=True)
        margin=ordered[0]-ordered[1]
        assert margin>0
        decisions[label]={"probabilities":q,"argmax":max(range(4),key=q.__getitem__),"margin":margin}
    tied=ns["initial_state"]();tied["teacher"]=[.5,.49]
    tie_info=ns["candidate_map"](tied,1)[1]
    assert tie_info["pseudo_class"]==0
    output={"status":"PASS","scope":"E affine toy only; no source softmax/CE or patient claim",
            "reviewed_probe_sha256":hashlib.sha256(TARGET.read_bytes()).hexdigest(),
            "review_probe_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "branches":branches,"teacher_argmax_decisions":decisions,
            "exact_tie_policy":"Python max over class indices chooses first index, BG=0",
            "exact_tie_check":tie_info,
            "final_J_B_risk":policies["J"]["final_RV_B"],
            "final_S_B_risk":policies["S"]["final_RV_B"],
            "note":"All coefficient-box corners are simplex-valid, hence all affine interpolants are valid."}
    out=HERE/"evidence/B_review_E.json"
    out.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":"PASS","branches":len(branches),"output":str(out)}))


if __name__=="__main__": main()
