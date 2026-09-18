#!/usr/bin/env python3
"""Bounded real-image component profile of unchanged maskfree code (CPU).

Not a 150-epoch throughput benchmark and not a matched scientific comparison.
No optimizer/checkpoint/export or multiprocessing worker benchmark is implied.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_audit_maskfree.data.discovery import discover_dataset
from self_audit_maskfree.data.dataset import build_training_unit
from self_audit_maskfree.models import make_models
from self_audit_maskfree.losses import producer_loss, student_loss
from self_audit_maskfree.hypotheses import generate_bank
from self_audit_maskfree.auditor import audit_bank
from self_audit_maskfree.observation import ObservationModel
from self_audit_nogt.data import ReadFirewall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    guard = ReadFirewall()
    with guard:
        start = time.perf_counter()
        manifest = discover_dataset(a.images, "acdc", seed=42, protocol="spatial_predictive", depth_axis=2)
        discovery_seconds = time.perf_counter() - start
        models = make_models(seed=17)
        forward_counts = {}
        for model_name in ("producer", "student_no_audit", "student_audited"):
            def count_forward(module, inputs, output, name=model_name):
                forward_counts[name] = forward_counts.get(name, 0) + 1
            models[model_name].register_forward_hook(count_forward)
        observation = ObservationModel(execution_device=None)
        ids = [r["unit_id"] for r in manifest["records"]]
        chosen = [ids[i] for i in np.linspace(0, len(ids) - 1, 8, dtype=int)]
        rows = []
        for index, unit_id in enumerate(chosen[:1] + chosen):
            forward_counts.clear()
            row = {"unit_id": unit_id, "warmup": index == 0}
            start = time.perf_counter()
            unit = build_training_unit(manifest, unit_id, image_size=224, seed=42)
            row["data_prepare"] = time.perf_counter() - start
            context, support = unit.fitting.context[None], unit.fitting.support[None]
            start = time.perf_counter()
            models["producer"].zero_grad(set_to_none=True)
            loss, _ = producer_loss(models["producer"], context, support,
                                     generator=torch.Generator().manual_seed(17))
            loss.backward()
            row["producer_3forward_backward"] = time.perf_counter() - start
            start = time.perf_counter()
            with torch.no_grad():
                features = models["producer"](context)["features"][0].detach()
            row["feature_forward"] = time.perf_counter() - start
            start = time.perf_counter()
            bank = generate_bank(unit.fitting, features, seed=17)
            row["bank_generation"] = time.perf_counter() - start
            start = time.perf_counter()
            audit = audit_bank(bank, unit.fitting, unit.selection, observation)
            row["audit"] = time.perf_counter() - start
            for name, target, validity in [("student_no_audit", audit.initial, audit.initial.validity),
                                            ("student_audited", audit.selected, audit.validity)]:
                model = models[name]
                model.zero_grad(set_to_none=True)
                start = time.perf_counter()
                value, _ = student_loss(model(context), target.probabilities[None].detach(), validity[None].detach())
                if (validity > 0).any():
                    value.backward()
                row[name] = time.perf_counter() - start
                row[name + "_has_valid"] = bool((validity > 0).any())
            row["bank_traces"] = [h.metadata.get("ontology_trace", {}) for h in bank]
            row["measured_forward_counts"] = dict(forward_counts)
            rows.append(row)
    keys = ["data_prepare", "producer_3forward_backward", "feature_forward", "bank_generation",
            "audit", "student_no_audit", "student_audited"]
    medians = {k: float(np.median([r[k] for r in rows[1:]])) for k in keys}
    result = {"kind": "real-image bounded leaf profile, not whole training throughput",
              "device": "cpu", "resolution": 224, "batch_size": 1, "threads": 4,
              "repeats": 8, "warmup": 1, "models": models["parameter_counts"],
              "discovery_seconds": discovery_seconds, "median_seconds": medians, "rows": rows,
              "worker_ipc": "NOT MEASURED: workers=0", "optimizer_export": "NOT MEASURED in this leaf profile",
              "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    guard.save(a.out.with_name("maskfree_profile_file_reads.json"))
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    main()
