#!/usr/bin/env python3
"""Cross-arm comparative analysis and transaction diagnostics for transfer-utility pilot.

Ingests runs/ across all four arms (plus optional conditional 5th arm), parses TRAIN_LOG.jsonl
and evaluations/ (steps 1000, 1500, 2000), evaluates screening decision rule, and outputs
both comprehensive JSON and markdown summary tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_transfer_utility_pilot import (
    analyze_checkpoint_evaluations,
    analyze_transaction_diagnostics,
    evaluate_screening_decision,
    read_json,
    write_json,
    SCREENING_DELTA_THRESHOLD,
    ALL_CLASSES,
    FOREGROUND_CLASSES,
)


def format_markdown_report(report: Dict[str, Any]) -> str:
    """Format analysis report into a human-readable GitHub-flavored markdown document."""
    lines: List[str] = []
    lines.append("# Scalar Transfer-Utility Pilot: Comparative Analysis")
    lines.append("")
    screening = report.get("screening", {})
    verdict = screening.get("verdict", "UNKNOWN")
    lines.append(f"**Pilot Screening Verdict:** `{verdict}`")
    lines.append(f"**Threshold Required:** > `{SCREENING_DELTA_THRESHOLD:.3f}` over EACH plain_kd, original, and larger_tie")
    if screening.get("synthetic"):
        lines.append("> [!NOTE] Synthetic validation smoke only; synthetic data cannot certify clinical PILOT_GO.")
    lines.append("")

    # 1. Final Macro4 Summary Table
    lines.append("## 1. Final Performance at Step 2000 (Exposed VAL18)")
    lines.append("")
    lines.append("| Arm | Selector | Status | Class-First Macro4 | FG Macro Dice | BG Dice | RV Dice | MYO Dice | LV Dice | RV Prec | RV Recall |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    arm_results = report.get("arm_results", {})
    for arm_name, arm_data in arm_results.items():
        evals = arm_data.get("evaluations", {})
        final_eval = evals.get("step_002000") or evals.get("final") or {}
        macro4 = final_eval.get("class_first_macro4")
        fg_macro = final_eval.get("class_first_foreground_macro")
        dices = final_eval.get("class_dices", {})
        bg = dices.get("BG")
        bg_str = f"{bg:.4f}" if isinstance(bg, (int, float)) else "N/A"
        rv = dices.get("RV")
        myo = dices.get("MYO")
        lv = dices.get("LV")
        rv_p = final_eval.get("rv_precision")
        rv_r = final_eval.get("rv_recall")

        m4_str = f"{macro4:.4f}" if isinstance(macro4, (int, float)) else "N/A"
        fg_str = f"{fg_macro:.4f}" if isinstance(fg_macro, (int, float)) else "N/A"
        rv_str = f"{rv:.4f}" if isinstance(rv, (int, float)) else "N/A"
        myo_str = f"{myo:.4f}" if isinstance(myo, (int, float)) else "N/A"
        lv_str = f"{lv:.4f}" if isinstance(lv, (int, float)) else "N/A"
        rv_p_str = f"{rv_p:.4f}" if isinstance(rv_p, (int, float)) else "N/A"
        rv_r_str = f"{rv_r:.4f}" if isinstance(rv_r, (int, float)) else "N/A"

        selector_name = arm_data.get("selector", "none")
        status = arm_data.get("status", "UNKNOWN")
        lines.append(f"| `{arm_name}` | `{selector_name}` | `{status}` | {m4_str} | {fg_str} | {bg_str} | {rv_str} | {myo_str} | {lv_str} | {rv_p_str} | {rv_r_str} |")

    lines.append("")

    # 2. Screening Decision Details
    lines.append("## 2. Screening Decision Rule Breakdown")
    lines.append("")
    deltas = screening.get("deltas", {})
    crit = screening.get("criteria", {})
    lines.append(f"- Utility vs Plain KD: delta = **{deltas.get('vs_plain_kd', 'N/A')}** (Passed > 0.002: `{crit.get('exceeds_plain_kd_by_002')}`)")
    lines.append(f"- Utility vs Original: delta = **{deltas.get('vs_scalar_guard_original', 'N/A')}** (Passed > 0.002: `{crit.get('exceeds_original_by_002')}`)")
    lines.append(f"- Utility vs Larger Tie: delta = **{deltas.get('vs_scalar_guard_larger_tie', 'N/A')}** (Passed > 0.002: `{crit.get('exceeds_larger_tie_by_002')}`)")
    lines.append(f"- Conditional 5th Arm Triggered: `{screening.get('should_run_conditional_5th_arm')}`")
    lines.append("")

    # 3. Transaction Diagnostics Table
    lines.append("## 3. Transaction Diagnostics (Audit Selection Behaviour)")
    lines.append("")
    lines.append("| Arm | Total Steps | Warmup | No Cand | U Steps | Accepted Nonzero | Zero Displ. | Rejected All | U Accept Rate | Wall Time (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for arm_name, arm_data in arm_results.items():
        diag = arm_data.get("diagnostics", {})
        if diag.get("status") == "COMPLETED":
            tot = diag.get("total_steps", 0)
            warmup = diag.get("warmup_steps", 0)
            no_cand = diag.get("no_candidate_steps", 0)
            u_steps = diag.get("u_attempted_steps", 0)
            tx = diag.get("transaction_outcomes", {})
            acc = tx.get("accepted_nonzero", 0)
            zero = tx.get("zero_displacement", 0)
            rej = tx.get("rejected_all", 0)
            u_acc_rate = tx.get("u_acceptance_rate")
            u_rate_str = f"{u_acc_rate*100:.1f}%" if u_acc_rate is not None else "N/A"
            wall_s = diag.get("latencies", {}).get("total_wall_seconds")
            wall_str = f"{wall_s:.1f}" if wall_s is not None else "N/A"
            lines.append(f"| `{arm_name}` | {tot} | {warmup} | {no_cand} | {u_steps} | {acc} | {zero} | {rej} | {u_rate_str} | {wall_str} |")
        else:
            lines.append(f"| `{arm_name}` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    lines.append("")

    # 4. Alpha Histogram
    lines.append("## 4. Alpha Scale Selection Frequencies")
    lines.append("")
    lines.append("| Arm | Scale 1.0 | 0.5 | 0.25 | 0.125 | 0.0625 | 0.03125 | 0.015625 | 0.0078125 | 0.0 (Zero/Rej) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for arm_name, arm_data in arm_results.items():
        diag = arm_data.get("diagnostics", {})
        if diag.get("status") == "COMPLETED":
            hist = diag.get("alpha_histogram", {})
            s1 = hist.get("1.0", 0)
            s2 = hist.get("0.5", 0)
            s3 = hist.get("0.25", 0)
            s4 = hist.get("0.125", 0)
            s5 = hist.get("0.0625", 0)
            s6 = hist.get("0.03125", 0)
            s7 = hist.get("0.015625", 0)
            s8 = hist.get("0.0078125", 0)
            s0 = hist.get("0.0", 0)
            lines.append(f"| `{arm_name}` | {s1} | {s2} | {s3} | {s4} | {s5} | {s6} | {s7} | {s8} | {s0} |")

    lines.append("")
    return "\n".join(lines)


def run_analysis(pilot_dir: Path | str, out_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    """Analyze a pilot directory containing runs/ for each arm."""
    p_dir = Path(pilot_dir).resolve()
    runs_dir = p_dir / "runs" if (p_dir / "runs").is_dir() else p_dir

    arm_names = [
        "plain_kd",
        "scalar_guard_original",
        "scalar_guard_larger_tie",
        "scalar_guard_cross_patient_utility",
        "scalar_guard_same_batch_control",
    ]

    arm_results: Dict[str, Any] = {}
    for arm in arm_names:
        arm_p = runs_dir / arm
        if not arm_p.is_dir():
            continue

        manifest_file = arm_p / "ARM_MANIFEST.json"
        arm_rec = read_json(manifest_file) if manifest_file.is_file() else {"status": "UNKNOWN"}

        # Extract diagnostics from TRAIN_LOG.jsonl if present
        log_file = arm_p / "TRAIN_LOG.jsonl"
        if not log_file.is_file():
            log_file = arm_p / "steps.jsonl"
        if log_file.is_file():
            arm_rec["diagnostics"] = analyze_transaction_diagnostics(log_file)

        # Ingest evaluations
        eval_base = arm_p / "evaluations"
        if eval_base.is_dir():
            eval_map = {}
            for step_dir in eval_base.iterdir():
                if step_dir.is_dir():
                    eval_map[step_dir.name] = {**arm_rec.get("evaluations", {}).get(step_dir.name, {}), **analyze_checkpoint_evaluations(step_dir)}
            arm_rec["evaluations"] = eval_map

        arm_results[arm] = arm_rec

    screening = evaluate_screening_decision(arm_results)
    # Compare checkpoints observed at or before the same training-time horizon.
    # No interpolation, extra updates, or retrospective hyperparameter changes.
    timelines = {}
    for arm, record in arm_results.items():
        log = runs_dir / arm / "TRAIN_LOG.jsonl"
        if not log.is_file():
            continue
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        elapsed, points = 0.0, []
        for row in rows:
            seconds = row.get("timings", {}).get("step_elapsed_seconds")
            if seconds is None:
                points = []
                break
            elapsed += seconds
            tag = f"step_{row['step'] + 1:06d}"
            if tag in record.get("evaluations", {}):
                points.append({"step": row['step'] + 1, "training_seconds": elapsed,
                    "Macro4": record['evaluations'][tag].get("class_first_macro4")})
        timelines[arm] = points
    complete_times = [v[-1]['training_seconds'] for v in timelines.values() if v]
    horizon = min(complete_times) if len(complete_times) == len(arm_results) and complete_times else None
    equal_time = {arm: next((p for p in reversed(points) if p['training_seconds'] <= horizon), None)
                  if horizon is not None else None for arm, points in timelines.items()}

    report = {
        "schema": "self_audit.compact.transfer_utility.analysis.v1",
        "pilot_dir": str(p_dir),
        "screening": screening,
        "arm_results": arm_results,
        "quality_over_training_time": timelines,
        "equal_training_time": {"horizon_seconds": horizon, "observed_checkpoints": equal_time,
            "scope": "sum measured optimizer-step time including audit/probe; cold setup, checkpoint I/O and evaluation reported separately; no interpolation"},
        "costs": {arm: {"marginal_train_export_eval_seconds": record.get("train_export_eval_wall_seconds"),
            "teacher_and_cache_cost": record.get("training_report", {}).get("teacher_costs", "NOT_KNOWN"),
            "profile": record.get("training_report", {}).get("profile", {})} for arm, record in arm_results.items()},
    }

    out_p = Path(out_dir).resolve() if out_dir else p_dir
    write_json(out_p / "ANALYSIS_REPORT.json", report)
    md_content = format_markdown_report(report)
    (out_p / "ANALYSIS_SUMMARY.md").write_text(md_content)

    report["plots"] = make_plots(report, out_p)
    write_json(out_p / "ANALYSIS_REPORT.json", report)
    return report


def make_plots(report, out):
    """Render actual saved evaluations and transaction distributions only."""
    if not report.get("arm_results"):
        return {"status": "NOT_RUN", "reason": "No completed run evidence"}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "NOT_RUN", "reason": "matplotlib unavailable"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for arm, rec in report["arm_results"].items():
        points = [(int(tag.removeprefix("step_")), e.get("class_first_macro4"))
                  for tag, e in rec.get("evaluations", {}).items() if tag.startswith("step_")]
        points = sorted((x,y) for x,y in points if y is not None)
        if points:
            axes[0].plot(*zip(*points), marker="o", label=arm)
        time_points = report.get("quality_over_training_time", {}).get(arm, [])
        if time_points:
            axes[1].plot([p['training_seconds'] for p in time_points], [p['Macro4'] for p in time_points], marker='o')
        hist = rec.get("diagnostics", {}).get("alpha_histogram", {})
        keys = [str(x) for x in (0., 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.)]
        if hist:
            axes[2].plot(range(len(keys)), [hist.get(k,0) for k in keys], marker='o')
    axes[0].set(xlabel='Optimizer steps', ylabel='VAL Macro4')
    axes[1].set(xlabel='Measured training seconds', ylabel='VAL Macro4')
    axes[2].set(xlabel='Selected scalar', ylabel='Transactions')
    axes[2].set_xticks(range(len(keys)), keys, rotation=60)
    if axes[0].lines:
        axes[0].legend(fontsize=6)
    fig.tight_layout()
    target = out / 'VAL_AND_SELECTION.png'
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return {"status": "CREATED", "path": str(target)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze transaction diagnostics, evaluation trajectories, and screening rule for transfer-utility study."
    )
    parser.add_argument(
        "--pilot-dir",
        required=True,
        help="Path to pilot root or runs directory",
    )
    parser.add_argument(
        "--out",
        help="Optional destination directory for analysis artifacts",
    )
    args = parser.parse_args()

    report = run_analysis(args.pilot_dir, args.out)
    print(format_markdown_report(report))
    verdict = report.get("screening", {}).get("verdict")
    return 0 if verdict in ("PILOT_GO", "SYNTHETIC_PASS") else 2


if __name__ == "__main__":
    sys.exit(main())
