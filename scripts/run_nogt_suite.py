#!/usr/bin/env python3
"""One command reproduces the locked image-only suite, then optional frozen evaluation.

All method runs finish and selection is written before any reference evaluator starts.
Never reuses partial output, changes a user checkout or invokes a remote machine.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def select_baseline(runs: Path, seeds=(17, 29)):
    c0 = json.loads((runs / "C0/timing.json").read_text())
    c0_coverage = c0["export_summary"]["valid_non_bg_fraction"]
    c0_latency = c0["latency"]["p50"]
    checks = {}
    for mode in ("cache", "direct"):
        checks[mode] = []
        for seed in seeds:
            d = json.loads((runs / f"C1_{mode}_s{seed}/timing.json").read_text())
            s = d["checkpoints"][-1]
            checks[mode].append({"seed": seed, "agreement": s["anonymous_agreement"],
                   "agreement_pass": s["anonymous_agreement"] >= .90,
                   "coverage": s["valid_non_bg_fraction"],
                   "coverage_pass": s["valid_non_bg_fraction"] >= .95 * c0_coverage,
                   "latency": d["latency"]["p50"], "latency_pass": d["latency"]["p50"] <= .80 * c0_latency})
    admissible = all(r["agreement_pass"] and r["coverage_pass"] and r["latency_pass"] for r in checks["cache"])
    chosen = "C1-cache" if admissible else "C0"
    if admissible and all(r["coverage_pass"] and r["latency_pass"] and
                         r["agreement"] >= c["agreement"] - .02 for r, c in zip(checks["direct"], checks["cache"])):
        chosen = "C1-direct"
    return {"selected_baseline": chosen, "time_unix": time.time(), "status": "PROVISIONAL",
            "selection_data": "image-only engineering admission, NOT semantic correctness",
            "gt_metrics_read": False, "checks": checks,
            "c0_coverage": c0_coverage, "c0_latency": c0_latency,
            "rule": "locked EXPERIMENT_PROTOCOL.md; candidate must pass both seeds"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/nogt_baseline_224.json"))
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run the locked C0 baseline only: raw images to frozen native outputs; no training.")
    parser.add_argument("--evaluate-references", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config.resolve()
    args.work.mkdir(parents=True, exist_ok=False)
    cache, runs = args.work / "cache", args.work / "runs"
    runs.mkdir()
    def run(stage, out, extra=()):
        cmd = [sys.executable, str(root / "scripts/run_nogt_baseline.py"), stage,
               "--config", str(config), "--cache", str(cache), "--out", str(out), *map(str, extra)]
        subprocess.run(cmd, cwd=root, check=True)
    run("prepare", runs / "prepare", ["--raw", args.raw.resolve(), "--split", args.split.resolve()])
    run("anchor", runs / "C0")
    cfg = json.loads(config.read_text())
    run_dirs = [runs / "C0"]
    if args.baseline_only:
        result = {"selected_baseline": "C0", "status": "PROVISIONAL",
                  "time_unix": time.time(), "gt_metrics_read": False,
                  "selection_data": "fixed baseline replay; no new candidate selection"}
    else:
        run("train", runs / "pilot", ["--mode", "cache", "--seed", 17, "--updates", 8])
        for mode in ("cache", "direct"):
            for seed in cfg["seeds"]:
                run_dir = runs / f"C1_{mode}_s{seed}"
                run("train", run_dir, ["--mode", mode, "--seed", seed])
                run_dirs.append(run_dir)
        result = select_baseline(runs, cfg["seeds"])
    result["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    (args.work / "SELECTION_BEFORE_GT.json").write_text(json.dumps(result, indent=2))
    subprocess.run([sys.executable, str(root / "scripts/diagnose_nogt_ontology.py"),
                    "--cache", str(cache), "--out", str(args.work / "ontology_interventions.json")], check=True)
    if args.evaluate_references:
        for run_dir in run_dirs:
            subprocess.run([sys.executable, str(root / "scripts/evaluate_nogt_frozen.py"),
                            "--run", str(run_dir), "--references", str(args.evaluate_references.resolve()),
                            "--out", str(args.work / (run_dir.name + "_evaluation.json"))], check=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
