#!/usr/bin/env python3
"""Three checks to run before writing up S0-slicessm (val10 for anything that needs the model; stored per-patient results for the test20 statistics).

1. Metric parity: native FG Dice and SSL4MIS-compatible Dice are the same number by construction (both are the mean over RV/MYO/LV of the per-volume class Dice, averaged
   over a patient's two frames and then patients).  Verified here against an independent numpy implementation on val10 predictions; HD95 differs (mm vs voxels), proving the
   two evaluators are separate code paths.
2. Paired per-patient bootstrap (10,000 resamples, seed 42) and Wilcoxon of the student against T-M3HL, from the stored per-volume tables (no re-inference): the official-path
   teacher (``T_M3HL_OFFICIAL.json``) and the 3D-wrapper teacher (``W_FINAL.json``), on test20 with the val10 sign next to it.
3. SliceAlign: trained weights vs zero init, predicted offsets (px of the stage-3 grid and native px) on val10, and val10 Dice with the module bypassed (fusion weights zeroed =
   exact identity) against the trained model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def indep_dice_jaccard(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    d, j = [], []
    for c in (1, 2, 3):
        a, b = pred == c, gt == c
        inter, ua, ub = float((a & b).sum()), float(a.sum()), float(b.sum())
        d.append(0.0 if ua + ub == 0 else 2 * inter / (ua + ub))
        j.append(0.0 if ua + ub == 0 else inter / (ua + ub - inter))
    return float(np.mean(d)), float(np.mean(j))


def patient_table_compat(rows) -> dict[str, dict[str, float]]:
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["patient"], []).append({k: float(np.mean([r[c][k] for c in ("RV", "MYO", "LV")])) for k in ("dice", "jaccard", "hd95", "asd")})
    return {p: {k: float(np.mean([v[k] for v in vs])) for k in vs[0]} for p, vs in by.items()}


def patient_table_official(per_case: list[dict]) -> dict[str, dict[str, float]]:
    by: dict[str, list] = {}
    for r in per_case:
        pid = r["case"].split("_frame")[0]
        by.setdefault(pid, []).append({k: float(np.mean([r[c][k] for c in ("RV", "MYO", "LV")])) for k in ("dice", "jaccard", "hd95", "asd")})
    return {p: {k: float(np.mean([v[k] for v in vs])) for k in vs[0]} for p, vs in by.items()}


def paired(a: dict[str, float], b: dict[str, float], seed: int = 42, boots: int = 10000) -> dict:
    from scipy.stats import wilcoxon

    ids = sorted(set(a) & set(b))
    diff = np.array([a[i] - b[i] for i in ids])
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(boots)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    return {"n": len(ids), "mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)], "wilcoxon_p": p, "ci_excludes_0": bool(lo > 0 or hi < 0)}


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", type=Path, required=True, help="S0slicessm run dir (resume_A2.pt, FINAL_RESULT.json, TEST20_RESULT.json)")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--t-m3hl", type=Path, required=True, help="T-M3HL run dir (T_M3HL_OFFICIAL.json, W_FINAL.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from w2s_common import load_config
    from w2s_ssl4mis_eval import compat_metrics, native_metrics, predict_records
    from self_audit.candidate_pg import engine
    from self_audit.candidate_pg.student_pg import build_student

    device = torch.device(a.device)
    cfg = load_config(a.config)
    student = build_student(cfg)
    ckpt = torch.load(a.arm_dir / "resume_A2.pt", map_location="cpu", weights_only=False)
    student.load_state_dict(ckpt["best_student"], strict=True)
    student.to(device).eval()
    bundle = engine._load_data(cfg, synthetic=False)["objects"]
    val = [bundle.val.load(k, labeled=True) for k in bundle.val.labeled_keys()]
    L = ["# Checks before writing up S0-slicessm", ""]

    # ---- 1. metric parity (val10)
    rec = predict_records(student, val, device, kind="student")
    nat, cmp_ = native_metrics(rec), compat_metrics(rec)
    ind = [indep_dice_jaccard(r["prediction"].numpy(), r["target"].numpy()) for r in rec]
    per_patient_ind: dict[str, list] = {}
    for r, v in zip(rec, ind):
        per_patient_ind.setdefault(r["patient"], []).append(v)
    ind_dice = float(np.mean([np.mean([x[0] for x in vs]) for vs in per_patient_ind.values()]))
    ind_jc = float(np.mean([np.mean([x[1] for x in vs]) for vs in per_patient_ind.values()]))
    cm = cmp_["mean"]
    ratio = float(np.mean([cmp_["per_volume"][i]["LV"]["hd95"] for i in range(len(rec))]) / max(nat["LV"]["hd95_mm"], 1e-9))
    L += ["## 1. Metric parity (val10, val10-selected checkpoint)", "",
          f"- native FG Dice (patient macro): {nat['fg_dice']:.10f}", f"- SSL4MIS-compatible mean Dice (medpy): {cm['dice']:.10f}", f"- independent numpy Dice (per class, ED/ES per patient): {ind_dice:.10f}",
          f"- max |native - compat| = {abs(nat['fg_dice'] - cm['dice']):.2e}; max |independent - compat| = {abs(ind_dice - cm['dice']):.2e}",
          f"- Jaccard: compat {cm['jaccard']:.10f}, independent numpy {ind_jc:.10f}",
          f"- HD95 differs by construction: native {nat['fg_hd95_mm']:.3f} mm, compat {cm['hd95']:.3f} voxels (LV compat/native ratio {ratio:.2f}, between 1/spacing and 1).",
          "- Why Dice is identical: `evaluation._frame_metrics` defines `fg_dice` as the mean of the three per-class Dice values of a frame, and the compat evaluator takes the mean over the same three classes per volume; "
          "averaging the two frames of a patient and then patients equals the average over all volumes when every patient has exactly two frames. It is an algebraic identity, not a shared code path "
          "(the HD95 values prove the two evaluators ran). If a patient lacked one frame or a class were empty, the two would differ.", ""]

    # ---- 2. paired statistics from stored tables
    arm_t = json.loads((a.arm_dir / "TEST20_RESULT.json").read_text())
    arm_f = json.loads((a.arm_dir / "FINAL_RESULT.json").read_text())
    stu_test = patient_table_compat(arm_t["metrics"]["test20"]["ssl4mis_compat"]["per_volume"])
    stu_val = patient_table_compat(arm_f["metrics"]["val10"]["ssl4mis_compat"]["per_volume"])
    off = json.loads((a.t_m3hl / "T_M3HL_OFFICIAL.json").read_text())["lists"]
    wrap = json.loads((a.t_m3hl / "W_FINAL.json").read_text())["metrics"]
    teachers = {"official-path T-M3HL (h5, SSL4MIS min-max input, voxel metrics)": (patient_table_official(off["test"]["per_case"]), patient_table_official(off["val"]["per_case"])),
                "3D-wrapper T-M3HL (our normalisation, same evaluator as the student)": (patient_table_compat(wrap["test20"]["ssl4mis_compat"]["per_volume"]), patient_table_compat(wrap["val10"]["ssl4mis_compat"]["per_volume"]))}
    L += ["## 2. Paired per-patient comparison with T-M3HL (student - teacher; bootstrap 10,000, seed 42; stored per-volume results)", ""]
    for name, (t_test, t_val) in teachers.items():
        L += [f"### vs {name}", "", "| metric | split | n | mean diff | 95% bootstrap CI | Wilcoxon p |", "|---|---|---|---|---|---|"]
        verdict = {}
        for metric, label in (("dice", "Dice"), ("jaccard", "Jaccard"), ("hd95", "HD95 (voxels)"), ("asd", "ASD (voxels)")):
            for split, s, t in (("test20", stu_test, t_test), ("val10", stu_val, t_val)):
                r = paired({p: v[metric] for p, v in s.items()}, {p: v[metric] for p, v in t.items()})
                verdict[(metric, split)] = r
                L.append(f"| {label} | {split} | {r['n']} | {r['mean_diff']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {r['wilcoxon_p']:.3f} |")
        sup = [m for m in ("dice", "hd95") if verdict[(m, "test20")]["ci_excludes_0"] and (verdict[(m, "test20")]["mean_diff"] > 0) == (verdict[(m, "val10")]["mean_diff"] > 0)]
        L += ["", f"Verdict (CI on test20 excludes 0 and val10 has the same sign): {'supported for ' + ', '.join(sup) if sup else 'inconclusive'}. n = 20 test patients (10 val), one seed.", ""]

    # ---- 3. SliceAlign
    align = getattr(student, "align3", None)
    L += ["## 3. SliceAlign after training", ""]
    if align is None:
        L += ["This arm has no SliceAlign module.", ""]
    else:
        norms = {n: float(p.detach().norm()) for n, p in align.named_parameters() if n in ("fuse.weight", "fuse.bias", "offset.2.weight", "offset.2.bias")}
        L += ["Zero-initialised parameters (init = 0): " + ", ".join(f"`{k}` norm {v:.4f}" for k, v in norms.items()), ""]
        flows = []
        hook = align.register_forward_hook(lambda m, i, o: flows.append(o[1].detach().float().cpu()) if o[1] is not None else None)
        with torch.no_grad():
            for f in val:
                student(f.image.to(device), f.geometry)
        hook.remove()
        mag = torch.cat([torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2).flatten() for x in flows]) if flows else torch.zeros(1)  # volumes differ in size: flatten each, then concatenate
        L += [f"Predicted offsets between adjacent slices on val10 (stage-3 cells, 1 cell = 8 native px; cap 4 cells): mean |offset| {mag.mean():.4f} cells ({mag.mean() * 8:.3f} px), "
              f"p95 {np.percentile(mag.numpy(), 95):.4f} cells, max {mag.max():.4f} cells; fraction of positions above 0.05 cell: {(mag > 0.05).float().mean():.3f}.", ""]
        saved = (align.fuse.weight.detach().clone(), align.fuse.bias.detach().clone())
        with torch.no_grad():
            align.fuse.weight.zero_(); align.fuse.bias.zero_()
        rec_off = predict_records(student, val, device, kind="student")
        with torch.no_grad():
            align.fuse.weight.copy_(saved[0]); align.fuse.bias.copy_(saved[1])
        n_off, n_on = native_metrics(rec_off), nat
        L += ["Bypass test at inference (fusion weights zeroed, so SliceAlign is exactly the identity), val10:", "",
              "| | trained | SliceAlign bypassed |", "|---|---|---|", f"| FG Dice | {n_on['fg_dice']:.4f} | {n_off['fg_dice']:.4f} |", f"| HD95 mm | {n_on['fg_hd95_mm']:.2f} | {n_off['fg_hd95_mm']:.2f} |", "",
              "This only measures reliance of the trained network on the module; it is not a retrained ablation (the rest of the network adapted to it). A real claim needs a `align = false` arm trained with the same recipe.", ""]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
