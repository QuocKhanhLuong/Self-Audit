"""Manufactured report tests only: these are not patient or GPU evidence."""
import copy
import numpy as np
import pytest
from self_audit.compact_acceptance import assess_compact_deployment


def evidence():
    export = {"schema":"self_audit.compact.export.v1", "synthetic":False,
              "manifest_sha256":"a"*64, "payload_sha256":"b"*64,
              "lineage":{"data_sha256":"e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4"}}
    quality = {"synthetic":False,"export_manifest_sha256":"a"*64,
               "export_payload_sha256":"b"*64,"train_manifest_sha256":"e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4",
               "evaluation_source_manifest_sha256":"22a9ea63821f266bd59e810bcd8f9a6958e6f92cc363e4a360b5ab9d4eb1e25c",
               "coverage":{"complete":True,"expected_frames":{"p":[0]},"scored_frames":{"p":[0]}},
               "evaluation_manifest_sha256":"d"*64,"frozen_predictions_sha256":"e"*64,
               "recipe_sha256":"f"*64,"patients":{"p":{}},"class_first_macro4":1.,
               "frames":[{"patient":"p","frame":0,"confusion":np.pad(np.eye(4,dtype=int)*10,((0,0),(0,1))).tolist()}]}
    base = {"status":"COMPLETED","synthetic":False,"evaluation_manifest_sha256":"d"*64,
            "artifact_manifest_sha256":"1"*64,"device":"cuda:0","precision":"float32",
            "workload":{"cases":["p"],"batch":1},"io_policy":"warm OS cache, masks written",
            "latency":{"steady":{"e2e_median_seconds_per_volume":2.}},
            "memory":{"cuda_peak_allocated_mb":100.,"cuda_peak_reserved_mb":120.,
                      "peak_inference_process_tree_ram_mb":600.},
            "model_size":{"deployed_weights_bytes":10000}}
    small = copy.deepcopy(base)
    small.update(artifact_manifest_sha256="a"*64, artifact_payload_sha256="b"*64)
    small["latency"]["steady"]["e2e_median_seconds_per_volume"] = 1.
    small["memory"] = {k:v/2 for k,v in small["memory"].items()}
    small["model_size"]["deployed_weights_bytes"] = 1000
    bench = {"schema":"self_audit.compact.benchmark.v1","compact_student":small,
             "baseline_student":base,"reference_identity":{"verified":True,"deploy_manifest_sha256":"1"*64,
             "source_manifest_sha256":"03c1730f88941ada898bed7fcf1b9f13cdd8f08bcce4390539fec665c82e97de","inference_state":"student"}}
    first = quality["frames"][0]
    pids = ["p"] + [f"p{i}" for i in range(1,50)]
    quality["patients"] = {pid:{} for pid in pids}
    quality["frames"] = [{**copy.deepcopy(first), "patient":pid} for pid in pids]
    quality["coverage"].update(expected_frames={pid:[0] for pid in pids}, scored_frames={pid:[0] for pid in pids})
    return export, quality, bench


def test_complete_manufactured_evidence_exercises_gate_only():
    assert assess_compact_deployment(*evidence())["status"] == "GO"


@pytest.mark.parametrize("mutation",["synthetic","cpu","missing_reference","wrong_export","different_cases","partial_classes"])
def test_missing_or_mismatched_evidence_cannot_pass(mutation):
    e,q,b = evidence()
    if mutation == "synthetic": e["synthetic"] = True
    elif mutation == "cpu": b["compact_student"]["device"] = b["baseline_student"]["device"] = "cpu"
    elif mutation == "missing_reference": b["baseline_student"] = {"status":"NOT_RUN"}
    elif mutation == "wrong_export": b["compact_student"]["artifact_manifest_sha256"] = "2"*64
    elif mutation == "different_cases": b["baseline_student"]["workload"] = {"cases":["other"]}
    else:
        for row in q["frames"]: row["confusion"][3] = [0]*5
    assert assess_compact_deployment(e,q,b)["status"] == "BLOCKED"


def test_failed_resource_measurement_is_no_go_not_missing_evidence():
    e,q,b = evidence()
    b["compact_student"]["latency"]["steady"]["e2e_median_seconds_per_volume"] = 3.
    r = assess_compact_deployment(e,q,b)
    assert r["status"] == "NO_GO"
    assert r["gates"]["e2e_seconds_per_volume"] is False


def test_macro4_is_recomputed_not_trusted():
    e,q,b = evidence()
    q["class_first_macro4"] = .99
    assert assess_compact_deployment(e,q,b)["status"] == "BLOCKED"


@pytest.mark.parametrize("mutation", ["different_split", "missing_frame", "partial_predictions"])
def test_subset_or_different_protocol_cannot_satisfy_historical_acceptance(mutation):
    e,q,b=evidence()
    if mutation == "different_split": q["evaluation_source_manifest_sha256"]="0"*64
    elif mutation == "missing_frame": q["coverage"]["expected_frames"]["p"].append(1)
    else: q["coverage"]["complete"]=False
    assert assess_compact_deployment(e,q,b)["status"] == "BLOCKED"
