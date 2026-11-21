#!/usr/bin/env python3
"""Turn tpc_offline.json into TPC_OFFLINE.md (tables, pre-registered kill criteria, GO/NO-GO) and one figure per experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SIGNALS = ["tpc", "s_flip", "s_area", "s_topo", "s_tta"]
BASELINES = ["one_minus_conf", "entropy", "one_minus_margin", "slice_position"]


def boot(values_by_patient: dict[str, list[float]], n=1000, seed=42):
    rng = np.random.default_rng(seed)
    pats = sorted(values_by_patient)
    means = [np.nanmean([x for p in rng.choice(pats, len(pats)) for x in values_by_patient[p]]) for _ in range(n)]
    return float(np.nanmean([x for p in pats for x in values_by_patient[p]])), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def agg(recs, variant, tag, name):
    by = {}
    for r in recs:
        by.setdefault(r["patient"], []).append(r["variants"][variant][tag][name])
    return boot(by)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads(a.json.read_text())
    a.fig_dir.mkdir(parents=True, exist_ok=True)
    L = ["# TPC-Gate offline analysis on frozen T-M3HL (pre-registered; no training)", "",
         "val10 chooses q only; U63 ground truth is a diagnostic (never selection/training); test20 not used. Signals/errors/flags live in the aligned frame",
         "(translation of slices), so Dice/HD95 here are not identical to unaligned values. n = 20 val10 and 126 U63 volumes (ED/ES of 63 patients), one teacher, one seed.", ""]
    go = {}
    # ---- E0
    L += ["## E0 error map", ""]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for split in ("val10", "U63"):
        recs = d[split]
        err, roi, e99 = sum(r["e0"]["err"] for r in recs), sum(r["e0"]["roi"] for r in recs), sum(r["e0"]["err_conf99"] for r in recs)
        terc = [(sum(r["e0"]["terciles"][k]["err"] for r in recs), sum(r["e0"]["terciles"][k]["roi"] for r in recs)) for k in range(3)]
        rates = [e / max(n, 1) for e, n in terc]
        cd = {c: float(np.mean([r["e0"]["class_dice"][c] for r in recs])) for c in ("1", "2", "3")}
        L += [f"**{split}**: error rate in ROI {err / max(roi, 1):.4f}; errors with c_T >= 0.99: {e99 / max(err, 1):.1%}; error rate base/mid/apex terciles "
              f"{rates[0]:.4f} / {rates[1]:.4f} / {rates[2]:.4f}; class Dice RV {cd['1']:.3f} MYO {cd['2']:.3f} LV {cd['3']:.3f}.", ""]
        ax[0].bar([f"{split}\n{n}" for n in ("base", "mid", "apex")], rates)
        go[f"E0_{split}"] = (e99 / max(err, 1) >= 0.30) or (max(rates) > 1.5 * max(min(rates), 1e-9))
    ax[0].set_ylabel("error rate in ROI"); ax[0].set_title("E0: by slice position")
    ax[1].bar(["val10", "U63"], [sum(r["e0"]["err_conf99"] for r in d[s]) / max(sum(r["e0"]["err"] for r in d[s]), 1) for s in ("val10", "U63")])
    ax[1].axhline(0.30, ls="--"); ax[1].set_title("errors at c_T >= 0.99 (kill < 0.30 and no position concentration)")
    fig.tight_layout(); fig.savefig(a.fig_dir / "E0.png", dpi=130); plt.close(fig)
    L += [f"E0 GO (enough errors at high confidence or concentrated by position): val10 **{'GO' if go['E0_val10'] else 'NO-GO'}**, U63 **{'GO' if go['E0_U63'] else 'NO-GO'}**.", ""]
    # ---- E1
    L += ["## E1 AUROC for 'teacher voxel is wrong' (patient-bootstrap 95% CI; ROI = 5 px around predicted FG)", ""]
    best = {}
    for split in ("val10", "U63"):
        L += [f"### {split}", "", "| signal | variant | overall | boundary band | inside c_T >= 0.99 |", "|---|---|---|---|---|"]
        scores = {}
        for variant in ("generic", "cardiac"):
            for name in SIGNALS + BASELINES:
                o, b_, c = (agg(d[split], variant, tag, name) for tag in ("auroc", "band", "conf99"))
                scores[(variant, name)] = (o[0], b_[0], c[0])
                if variant == "generic" or name in ("tpc", "s_topo", "s_flip", "s_area"):
                    L.append(f"| {name} | {variant if name in SIGNALS else '-'} | {o[0]:.3f} [{o[1][0]:.3f}, {o[1][1]:.3f}] | {b_[0]:.3f} | {c[0]:.3f} |")
        tpc_best = max(((v, n) for (v, n) in scores if n in ("tpc", "s_flip", "s_area", "s_topo")), key=lambda k: scores[k][0])
        base_best = max(((v, n) for (v, n) in scores if n in BASELINES), key=lambda k: scores[k][0])
        delta = scores[tpc_best][0] - scores[base_best][0]
        c99 = scores[tpc_best][2]
        best[split] = (tpc_best, base_best, delta, c99)
        L += ["", f"Best TPC-family signal: {tpc_best[1]} ({tpc_best[0]}) AUROC {scores[tpc_best][0]:.3f}; best baseline: {base_best[1]} {scores[base_best][0]:.3f}; "
                  f"delta {delta:+.3f}; best TPC inside c_T >= 0.99: {c99:.3f}.", ""]
        go[f"E1_{split}"] = delta >= 0.05 and c99 >= 0.65
        L += [f"E1 {split}: **{'GO' if go[f'E1_{split}'] else 'NO-GO'}** (kill if delta < 0.05 or AUROC inside c_T >= 0.99 < 0.65).", ""]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        names = SIGNALS + BASELINES
        ax.bar(names, [scores[("generic", n)][0] for n in names]); ax.set_ylim(0.4, 1); ax.set_title(f"E1 AUROC overall ({split}, generic)"); plt.xticks(rotation=30)
        fig.tight_layout(); fig.savefig(a.fig_dir / f"E1_{split}.png", dpi=130); plt.close(fig)
    # ---- E2
    L += ["## E2 oracle headroom (GT replaces the teacher label at flagged voxels; TPC variant = best TPC family on val10 per E1)", ""]
    variant = best["val10"][0][0]
    gains = {}
    for split in ("val10", "U63"):
        L += [f"### {split}", "", "| q | flagged fraction of ROI | Dice before | Dice after | gain | HD95 mm before | after | gain |", "|---|---|---|---|---|---|---|---|"]
        for q in ("0.8", "0.9", "0.95"):
            rows = [r["variants"][variant]["flag"][q] for r in d[split]]
            m = lambda k: float(np.mean([x[k] for x in rows]))
            gains[(split, q)] = (m("dice_after") - m("dice_before"), m("hd_before") - m("hd_after"))
            L.append(f"| {q} | {m('flag_frac'):.3f} | {m('dice_before'):.4f} | {m('dice_after'):.4f} | {gains[(split, q)][0] * 100:+.2f} pt | {m('hd_before'):.2f} | {m('hd_after'):.2f} | {gains[(split, q)][1]:+.2f} |")
        L.append("")
    qsel = max(("0.8", "0.9", "0.95"), key=lambda q: gains[("val10", q)][0])
    for split in ("val10", "U63"):
        g = gains[(split, qsel)]
        go[f"E2_{split}"] = not (g[0] < 0.005 and g[1] < 0.5)
    L += [f"q chosen on val10 (max Dice gain): **{qsel}**. E2 val10 **{'GO' if go['E2_val10'] else 'NO-GO'}**, U63 **{'GO' if go['E2_U63'] else 'NO-GO'}** (kill if gain < 0.5 Dice pt and < 0.5 mm HD95).", ""]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, split in enumerate(("val10", "U63")):
        ax.bar(np.arange(3) + 0.4 * i, [gains[(split, q)][0] * 100 for q in ("0.8", "0.9", "0.95")], 0.4, label=split)
    ax.set_xticks(np.arange(3) + 0.2); ax.set_xticklabels(["q 0.80", "q 0.90", "q 0.95"]); ax.axhline(0.5, ls="--"); ax.legend(); ax.set_ylabel("oracle Dice gain (pt)")
    fig.tight_layout(); fig.savefig(a.fig_dir / "E2.png", dpi=130); plt.close(fig)
    # ---- E3
    L += ["## E3 student feasibility (flagged voxels where student and teacher disagree; existing checkpoints)", "", "| split | student | disagreeing flagged voxels | student right | teacher right |", "|---|---|---|---|---|"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    e3 = {}
    names = sorted({k[3:] for r in d["val10"] for k in r["variants"][variant]["flag"][qsel] if k.startswith("e3_")})
    for i, split in enumerate(("val10", "U63")):
        for j, n in enumerate(names):
            rows = [r["variants"][variant]["flag"][qsel][f"e3_{n}"] for r in d[split]]
            tot = sum(x["n"] for x in rows)
            sr = sum(x["student_right"] for x in rows) / max(tot, 1)
            tr = sum(x["teacher_right"] for x in rows) / max(tot, 1)
            e3[(split, n)] = sr
            L.append(f"| {split} | {n} | {tot} | {sr:.1%} | {tr:.1%} |")
            ax.bar(i * (len(names) + 1) + j, sr * 100, label=n if i == 0 else None)
    ax.axhline(55, ls="--"); ax.set_ylabel("student right (%)"); ax.legend(); fig.tight_layout(); fig.savefig(a.fig_dir / "E3.png", dpi=130); plt.close(fig)
    for split in ("val10", "U63"):
        go[f"E3_{split}"] = any(e3[(split, n)] >= 0.55 for n in names)
    L += ["", f"E3: override **{'GO' if go['E3_val10'] and go['E3_U63'] else 'NO-GO (drop-only)'}** (kill override if the student is right < 55% on both val10 and U63 for every student checkpoint).", ""]
    L += ["## GO / NO-GO summary", "", "| experiment | val10 | U63 |", "|---|---|---|"]
    for e in ("E0", "E1", "E2", "E3"):
        L.append(f"| {e} | {'GO' if go[f'{e}_val10'] else 'NO-GO'} | {'GO' if go[f'{e}_U63'] else 'NO-GO'} |")
    a.out_md.write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
