#!/usr/bin/env python3
"""Post-freeze development analysis only; never a method or selection dependency."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.report / "evidence"
    arms = ["C0", "C1_cache_s17", "C1_cache_s29", "C1_direct_s17", "C1_direct_s29"]
    evaluations = {arm: read(evidence / (arm + "_evaluation.json")) for arm in arms}
    anchor = {p["patient"]: p["foreground_mean_dice"] for p in evaluations["C0"]["patients"]}
    metrics, curves, deltas, costs = [], [], [], {}
    preparation = read(evidence / "preparation.json")["seconds"]
    anchor_time = read(evidence / "runs/C0/timing.json")["generation_seconds"]
    for arm in arms:
        timing = read(evidence / "runs" / arm / "timing.json")
        evaluation = evaluations[arm]
        final = evaluation["summary"][-1]
        step = final["step"]
        patients = [p for p in evaluation["patients"] if p["step"] == step]
        patient_delta = np.array([p["foreground_mean_dice"] - anchor[p["patient"]] for p in patients])
        image = timing.get("export_summary") or timing["checkpoints"][-1]
        conditional_fg = np.array([p["conditional_dice_known_pixels_ONLY"][1:] for p in patients], dtype=float)
        conditional_fg_mean = float(np.nanmean(conditional_fg)) if np.isfinite(conditional_fg).any() else None
        row = {"arm": arm, "patients": len(patients), "foreground_dice": final["foreground_mean_dice"],
               "rv_dice": final["class_dice"][1], "myo_dice": final["class_dice"][2], "lv_dice": final["class_dice"][3],
               "rv_coverage": final["class_coverage"][1], "myo_coverage": final["class_coverage"][2],
               "lv_coverage": final["class_coverage"][3],
               "conditional_fg_dice_known_only": conditional_fg_mean,
               "valid_non_bg_image_fraction": image["valid_non_bg_fraction"],
               "unknown_fraction_resized": image["unknown_fraction"],
               "anonymous_agreement": image["anonymous_agreement"],
               "partition_ari_volume_mean": final["partition_ari_volume_mean"],
               "warm_p50_ms": timing["latency"]["p50"] * 1000,
               "warm_p90_ms": timing["latency"]["p90"] * 1000,
               "max_host_rss_mib": timing["max_rss_bytes"] / 2**20,
               "harm_vs_C0_patients": int((patient_delta < -1e-12).sum()),
               "improved_vs_C0_patients": int((patient_delta > 1e-12).sum()),
               "tied_vs_C0_patients": int((np.abs(patient_delta) <= 1e-12).sum()),
               "mean_delta_vs_C0": float(patient_delta.mean())}
        metrics.append(row)
        for p, delta in zip(patients, patient_delta):
            deltas.append({"arm": arm, "patient": p["patient"], "final_fg_dice": p["foreground_mean_dice"],
                           "C0_fg_dice": anchor[p["patient"]], "delta": float(delta)})
        train_rows = []
        if arm != "C0":
            with (evidence / "runs" / arm / "curve.csv").open() as f:
                train_rows = list(csv.DictReader(f))
        train_by_step = {int(r["step"]): r for r in train_rows}
        for s in evaluation["summary"]:
            n = int(s["step"].split("_")[-1])
            r = train_by_step.get(n, {})
            im = timing["export_summary"] if arm == "C0" else next(c for c in timing["checkpoints"] if c["step"] == n)
            curves.append({"arm": arm, "step": n, "examples_seen": 4*n,
                           "train_seconds": float(r.get("train_seconds", 0)),
                           "elapsed_before_validation_seconds": float(r.get("elapsed_seconds", 0)),
                           "foreground_dice": s["foreground_mean_dice"],
                           "anonymous_agreement": im["anonymous_agreement"],
                           "valid_non_bg_fraction": im["valid_non_bg_fraction"],
                           "unknown_fraction": im["unknown_fraction"]})
        if arm == "C0":
            cost = {"preparation_seconds": preparation, "generation_all_1902_seconds": anchor_time,
                    "development_naming_seconds": image["naming_seconds"],
                    "development_native_export_seconds": image["native_export_seconds"]}
            cost["measured_stage_sum_seconds"] = sum(cost.values())
        else:
            initial = timing["checkpoints"][0]
            initial_seconds = sum(initial[k] for k in ("forward_seconds", "naming_seconds", "native_export_seconds"))
            cost = {"shared_preparation_seconds": preparation, "shared_anchor_cache_seconds": anchor_time,
                    "initial_development_export_measured_components_seconds": initial_seconds,
                    "train_only_seconds": timing["train_seconds"],
                    "train_plus_later_validation_export_seconds": timing["train_validation_export_seconds"],
                    "optimizer_updates": timing["updates"], "examples_seen": timing["examples"],
                    "training_examples_per_train_second": timing["examples"] / timing["train_seconds"]}
            cost["measured_stage_sum_seconds"] = preparation + anchor_time + initial_seconds + timing["train_validation_export_seconds"]
            own = [c for c in curves if c["arm"] == arm]
            cost["normalized_step_auc_development_dice"] = float(np.trapezoid([c["foreground_dice"] for c in own],
                                                                                  [c["step"] for c in own]) / timing["updates"])
        costs[arm] = cost
    interventions = read(evidence / "ontology_interventions.json")["real_interventions"]
    gap = {"eligible": len(interventions),
           "lost_enclosure": sum("R3_enclosure" not in r["after"]["rules"] for r in interventions),
           "median_partition_agreement": float(np.median([r["partition_agreement"] for r in interventions])),
           "median_named_agreement": float(np.median([r["named_agreement"] for r in interventions])),
           "min_named_agreement": min(r["named_agreement"] for r in interventions)}
    ontology = read(evidence / "C0_ontology_rows.json")
    unresolved_reasons = Counter(note for r in ontology if r["unresolved"] for note in r["trace"].get("notes", []))
    selection = read(evidence / "SELECTION_BEFORE_GT.json")
    assert all(e["evaluated_unix"] > selection["time_unix"] for e in evaluations.values())
    sequences_match = {}
    for seed in (17, 29):
        hashes = []
        for mode in ("cache", "direct"):
            with (evidence / "runs" / f"C1_{mode}_s{seed}" / "curve.csv").open() as f:
                hashes.append([r["sample_index_hash"] for r in csv.DictReader(f)])
        sequences_match[str(seed)] = hashes[0] == hashes[1]
    result = {"selection_before_all_GT_evaluations": True, "matched_example_sequences": sequences_match,
              "metrics": metrics, "costs": costs, "real_gap": gap,
              "unresolved_notes_counts": dict(unresolved_reasons),
              "cost_scope": "Stage sums exclude process startup, diagnostic benchmark, evaluator and some bookkeeping; shared prep/cache charged once per independent arm, not twice in suite.",
              "auc_scope": "Posthoc descriptive development AUC, not selected threshold or checkpoint; no convergence certification."}
    (evidence / "aggregate.json").write_text(json.dumps(result, indent=2))
    write_csv(evidence / "metrics.csv", metrics)
    write_csv(evidence / "patient_deltas.csv", deltas)
    write_csv(evidence / "learning_curves.csv", curves)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for arm in arms[1:]:
        rows = [r for r in curves if r["arm"] == arm]
        for axis, key in zip(ax, ("foreground_dice", "anonymous_agreement", "valid_non_bg_fraction")):
            axis.plot([r["step"] for r in rows], [r[key] for r in rows], marker=".", label=arm)
            axis.set_xlabel("Optimizer updates (B4)")
    ax[0].axhline(metrics[0]["foreground_dice"], color="black", linestyle="--", label="C0")
    ax[0].set_ylabel("Native patient-mean foreground Dice")
    ax[1].set_ylabel("Anonymous C0 agreement")
    ax[2].set_ylabel("Valid non-BG / full image FOV")
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(0, 1)
    ax[2].set_ylim(bottom=0)
    ax[1].legend(fontsize=7)
    fig.suptitle("ACDC development, CPU224, 1024-update pilot; no convergence claim")
    fig.savefig(evidence / "learning_curves.png", dpi=180)
    fig.savefig(evidence / "learning_curves.pdf")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
