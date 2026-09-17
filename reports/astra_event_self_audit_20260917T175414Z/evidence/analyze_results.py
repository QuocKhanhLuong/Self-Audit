"""Reproduce descriptive diagnostics from immutable experiment JSON. No training."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(5901)
def read(name):
    return json.loads((ROOT / name).read_text())
def finite_rows(d):
    return [r for r in d["rows"] if r["true_delta"] is not None]
runs = {(s,p): read(f"acdc_bounded/seed{s}_{p}.json") for s in range(3)
        for p in ["none","always","periodic","random","entropy","learned","unguided"]}
pids = sorted(r["patient_id"] for r in runs[0,"none"]["final"]["patients"])
def vector(s,p):
    a={r["patient_id"]:r["final_dice"] for r in runs[s,p]["final"]["patients"]}
    assert sorted(a)==pids
    return np.array([a[i] for i in pids])
paired={}
for p,c in [("learned","none"),("learned","random"),("learned","entropy"),("always","unguided")]:
    values=np.mean([vector(s,p)-vector(s,c) for s in range(3)],axis=0)
    boot=values[rng.integers(0,len(pids),(10000,len(pids)))].mean(1)
    paired[p+"_minus_"+c]={"mean":float(values.mean()),"patient_bootstrap_95":np.quantile(boot,[.025,.975]).tolist(),
       "patients":{i:float(v) for i,v in zip(pids,values)},
       "scope":"Resample 20 patients after averaging three seeds per patient. Conditional on these seeds, compact actor and protocol; not seed uncertainty or equivalence."}
frozen={}
for name in ["seed0","seed1","seed2","seed0_fresh_value"]:
    d=read(f"acdc_interventions_{name}.json");rows=finite_rows(d);allrows=d["rows"]
    policies={}
    oracle=np.array([r["true_delta"]*r["actions"]["offline_matched_oracle"] for r in rows])
    for p in rows[0]["actions"]:
        z=np.array([r["actions"][p] for r in rows]);delta=np.array([r["true_delta"] for r in rows]);kept=z*delta
        policies[p]={"calls_all_376_states":sum(r["actions"][p] for r in allrows),
            "calls_valid_dice_states":int(z.sum()),"mean_retained_state_delta":float(kept.mean()),
            "mean_true_delta_per_call":float(delta[z].mean()) if z.any() else None,
            "budget_matched_oracle_regret":float((oracle-kept).mean())}
    v=np.array([r["predicted_value"] for r in allrows]);target=np.array([r["rf_delta"]-.001 for r in allrows])
    frozen[name]={"policies":policies,"proxy_target_mse":float(np.mean((v-target)**2)),
       "value_mean":float(v.mean()),"current_proxy_target_mean":float(target.mean()),
       "current_proxy_net_positive_fraction":float(np.mean(target>0)),
       "scope":"Frozen slice-state descriptive values; zero/empty Dice states excluded only from GT diagnostics. Oracle is at most the learned per-batch cardinality, not forced to spend on negative actions."}
counts={}
for p in ["none","always","periodic","random","entropy","learned","unguided"]:
    entries=[]
    for s in range(3):
        d=runs[s,p]; c=d["counters"];setup=d["setup"]
        uses_rf=p not in ("none","unguided"); uses_v=p=="learned"
        # Each warm RF or value example first computes A0. Each value twin adds one q call and one read.
        rf_n=setup["rf_examples"] if uses_rf else 0
        v_n=setup["trigger_probe_examples"] if uses_v else 0
        invocation=c["policy_audit_calls"]+c["probe_audit_calls"]+v_n
        q_train=rf_n+c["rf_train_examples"]
        reads=setup["actor_examples"]+rf_n+2*v_n+c["policy_read_calls"]+c["probe_read_calls"]
        entries.append({"seed":s,"training_audit_invocations_including_setup":invocation,
          "auditor_forwards_for_optimization_including_setup":q_train,"dynamic_reads_including_setup":reads,
          "teacher_energy_evaluations_including_setup":rf_n+c["rf_train_examples"]+2*(v_n+c["probe_audit_calls"]),
          "audit_invocations_per_960_supervised_examples":invocation/960,
          "actor_updates":120,"rf_updates_including_setup":(setup["rf_updates"] if uses_rf else 0)+c["rf_updates"],
          "trigger_updates_including_setup":(setup["trigger_updates"] if uses_v else 0)+c["trigger_updates"],
          "postsetup_positive_rf_probes":sum(v>0 for r in d["training"] for v in r["rf_deltas"]),
          "postsetup_negative_rf_probes":sum(v<0 for r in d["training"] for v in r["rf_deltas"]),
          "postsetup_zero_rf_probes":sum(v==0 for r in d["training"] for v in r["rf_deltas"]),
          "minimum_actor_grad":min(r["actor_grad_norm"] for r in d["training"]),
          "minimum_window_grad":min(r["window_grad_norm"] for r in d["training"]),
          "scheduled_validation_seconds":sum(r["evaluation_seconds"] for r in d["curve"]),
          "final_twins_validation_seconds":d["final"]["evaluation_seconds"],
          "data_seconds_postsetup":sum(r["data_seconds"] for r in d["training"]),
          "actor_probe_seconds_postsetup":sum(r["actor_and_probe_seconds"] for r in d["training"]),
          "slow_seconds_postsetup":sum(r["slow_update_seconds"] for r in d["training"])})
    counts[p]=entries
out={"paired_patient_comparisons":paired,"frozen_diagnostics":frozen,"cost_ledger":counts,
     "cost_note":"Counts are example-level module calls, not FLOPs or seconds; optimizer audit forwards separated from invocation. Diagnostic-only score checks and validation are separate. Shared setup is logically charged to each standalone policy. No CUDA."}
(ROOT/"analysis_details.json").write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({"paired":{k:{x:v[x] for x in ["mean","patient_bootstrap_95"]} for k,v in paired.items()},"frozen":frozen},indent=2))
