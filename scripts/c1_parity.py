#!/usr/bin/env python3
"""Part C1: metric parity and significance of a student arm vs the frozen M3HL teacher on val10 and test20.

Both models are scored with the SAME evaluators on the native grid: (a) the SSL4MIS-compatible evaluator (medpy, per volume, HD95 in
voxels) and (b) our native 3D metric (HD95 in mm).  T-M3HL predictions come from the slice-wise wrapper (per-slice 256x256, bilinear back
to the native grid).  Per patient the ED and ES frames are averaged; paired differences (student - teacher) get a Wilcoxon signed-rank test
and a 10,000-sample bootstrap 95% CI of the mean difference (seed 42).  Verdict "supported" only if the CI excludes 0 and val10 has the
same sign; otherwise "inconclusive".  test20 is opened here for reporting only (never for selection).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w2s_common import load_config, load_teacher  # noqa: E402
from w2s_ssl4mis_eval import compat_metrics, native_metrics, predict_records  # noqa: E402


def per_patient_compat(records: list[dict]) -> dict[str, dict[str, float]]:
    """Mean over classes per volume (SSL4MIS protocol), then mean over the patient's frames."""
    from self_audit.candidate_pg.evaluation_ssl4mis_compat import evaluate_volumes

    rows = evaluate_volumes([{"patient": r["patient"], "frame": r["frame"], "target": r["target"].numpy(), "prediction": r["prediction"].numpy()} for r in records])["per_volume"]
    by: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        by.setdefault(row["patient"], []).append({k: float(np.mean([row[c][k] for c in ("RV", "MYO", "LV")])) for k in ("dice", "jaccard", "hd95", "asd")})
    return {p: {k: float(np.mean([v[k] for v in vs])) for k in vs[0]} for p, vs in by.items()}


def paired(a: dict[str, float], b: dict[str, float], seed: int = 42, boots: int = 10000) -> dict:
    from scipy.stats import wilcoxon

    ids = sorted(set(a) & set(b))
    diff = np.array([a[i] - b[i] for i in ids])
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(boots)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    try:
        p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"n": len(ids), "mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)], "wilcoxon_p": p, "ci_excludes_0": bool(lo > 0 or hi < 0)}


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-config", type=Path, required=True, help="any M3HL arm config (pins the teacher checkpoint)")
    ap.add_argument("--student-config", type=Path, required=True)
    ap.add_argument("--student-run", type=Path, required=True)
    ap.add_argument("--engine-arm", default="A2")
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from self_audit.candidate_pg import engine
    from self_audit.candidate_pg.student_pg import PGStudent

    device = torch.device(args.device)
    tcfg = load_config(args.teacher_config)
    teacher, _ = load_teacher(tcfg, args.student_run, device)
    scfg = load_config(args.student_config)
    student = PGStudent(scfg)
    ckpt = torch.load(args.student_run / f"resume_{args.engine_arm}.pt", map_location="cpu", weights_only=False)
    student.load_state_dict(ckpt["best_student"], strict=True)
    student.to(device).eval()
    bundle = engine._load_data(scfg, synthetic=False)["objects"]
    result: dict = {"student_epoch": int(ckpt["best_epoch"]), "splits": {}}
    for name, data in (("val10", bundle.val), ("test20", bundle.test)):
        frames = [data.load(k, labeled=True) for k in data.labeled_keys()]
        rt = predict_records(teacher, frames, device, kind="teacher")
        rs = predict_records(student, frames, device, kind="student")
        nat_t, nat_s = native_metrics(rt)["per_patient"], native_metrics(rs)["per_patient"]
        cmp_t, cmp_s = per_patient_compat(rt), per_patient_compat(rs)
        sec = {"compat_dice": paired({p: v["dice"] for p, v in cmp_s.items()}, {p: v["dice"] for p, v in cmp_t.items()}),
               "compat_hd95_vox": paired({p: v["hd95"] for p, v in cmp_s.items()}, {p: v["hd95"] for p, v in cmp_t.items()}),
               "native_fg_dice": paired({p: v["fg_dice"] for p, v in nat_s.items()}, {p: v["fg_dice"] for p, v in nat_t.items()}),
               "native_fg_hd95_mm": paired({p: v["fg_hd95"] for p, v in nat_s.items()}, {p: v["fg_hd95"] for p, v in nat_t.items()}),
               "means": {"teacher_compat_dice": float(np.mean([v["dice"] for v in cmp_t.values()])), "student_compat_dice": float(np.mean([v["dice"] for v in cmp_s.values()])),
                         "teacher_native_fg_dice": float(np.mean([v["fg_dice"] for v in nat_t.values()])), "student_native_fg_dice": float(np.mean([v["fg_dice"] for v in nat_s.values()]))}}
        result["splits"][name] = sec
        print(name, json.dumps({k: (round(v["mean_diff"], 4), [round(x, 4) for x in v["ci95"]]) for k, v in sec.items() if k != "means"}), flush=True)

    def verdict(metric: str, better: str) -> str:
        t, v = result["splits"]["test20"][metric], result["splits"]["val10"][metric]
        same_sign = (t["mean_diff"] > 0) == (v["mean_diff"] > 0)
        good = (t["mean_diff"] > 0) if better == "higher" else (t["mean_diff"] < 0)
        return "supported" if t["ci_excludes_0"] and same_sign and good else ("supported in the other direction" if t["ci_excludes_0"] and same_sign else "inconclusive")

    result["verdict"] = {"compat_dice": verdict("compat_dice", "higher"), "native_fg_dice": verdict("native_fg_dice", "higher"),
                         "compat_hd95_vox": verdict("compat_hd95_vox", "lower"), "native_fg_hd95_mm": verdict("native_fg_hd95_mm", "lower")}
    lines = ["# Parity and significance: student vs frozen T-M3HL (same evaluators)", "",
             f"Student: `{args.student_run.name}` val10-selected epoch {result['student_epoch']}. Differences are student - teacher, per patient (ED/ES averaged), n = 20.",
             "n = 20 patients, one seed: seed variance is not estimated. 'Supported' needs a bootstrap CI that excludes 0 and the same sign on val10.", ""]
    for name in ("val10", "test20"):
        lines += [f"## {name}", "", "| metric | mean diff | 95% bootstrap CI | Wilcoxon p |", "|---|---|---|---|"]
        for k, v in result["splits"][name].items():
            if k != "means":
                lines.append(f"| {k} | {v['mean_diff']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['wilcoxon_p']:.3f} |")
        m = result["splits"][name]["means"]
        lines += ["", f"Means: teacher compat Dice {m['teacher_compat_dice']:.4f}, student {m['student_compat_dice']:.4f}; native FG Dice teacher {m['teacher_native_fg_dice']:.4f}, student {m['student_native_fg_dice']:.4f}.", ""]
    lines += ["## Verdict (test20 CI excludes 0 and val10 has the same sign)", ""] + [f"- {k}: **{v}**" for k, v in result["verdict"].items()]
    args.out_md.write_text("\n".join(lines) + "\n")
    args.out_md.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
