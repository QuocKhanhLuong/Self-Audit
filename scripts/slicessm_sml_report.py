#!/usr/bin/env python3
"""SLICESSM_SML.md: S/M/L width ablation table from the three finished arms (S = the existing S0-slicessm run)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=Path, required=True)
    ap.add_argument("--m", type=Path, required=True)
    ap.add_argument("--l", type=Path, required=True)
    ap.add_argument("--s1-s", type=Path, required=True, help="S1_RECEIPT.json of S1-SS (S)")
    ap.add_argument("--s1-m", type=Path, required=True)
    ap.add_argument("--s1-l", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    load = lambda p: json.loads(p.read_text())
    rows = []
    for tag, d, s1 in (("S", a.s, a.s1_s), ("M", a.m, a.s1_m), ("L", a.l, a.s1_l)):
        f, t, r = load(d / "FINAL_RESULT.json"), load(d / "TEST20_RESULT.json"), load(s1)
        v = f["metrics"]["val10"]["native"]
        rows.append({"tag": tag, "params": f["complexity"]["parameters"], "gflops": f["complexity"]["flops"] / 1e9, "lat": f["complexity"].get("latency_gpu_s"),
                     "sel": f["selected_epoch"], "vd": v["fg_dice"], "vh": v["fg_hd95_mm"], "vfd": f["final_epoch_metrics"]["val10"]["native"]["fg_dice"],
                     "td": t["metrics"]["test20"]["native"]["fg_dice"], "th": t["metrics"]["test20"]["native"]["fg_hd95_mm"],
                     "tfd": t["final_epoch_metrics"]["test20"]["native"]["fg_dice"], "cos_tr": r.get("cosine_train"), "cos_ho": r.get("cosine_holdout"),
                     "isc": f["inter_slice_consistency_val10"]["ratio"]})
    L = ["# SliceSSM width ablation S / M / L (UniMatch, no teacher, 150 epochs, seed 42; one run per size)", "",
         "S = the finished S0-slicessm (widths 24,48,96,160,192). M = x1.5 (40,72,144,240,288), L = x2 (48,96,192,320,384); depth, blocks and every other option are identical;",
         "the 3 M parameter cap is lifted for M and L. Each size has its own S1 (CineMA distillation) because a wider student cannot load the S1-SS artifact. val10 selects the epoch;",
         "test20 was opened once per arm after selection and decides nothing. n = 10 val and 20 test patients, one seed: differences of a few tenths of a Dice point are within noise.", "",
         "| | S | M | L |", "|---|---|---|---|",
         "| parameters (deployed) | " + " | ".join(f"{r['params'] / 1e6:.2f} M" for r in rows) + " |",
         "| FLOPs (10x256x232) | " + " | ".join(f"{r['gflops']:.0f} G" for r in rows) + " |",
         "| GPU latency | " + " | ".join(f"{r['lat'] * 1000:.1f} ms" if r["lat"] else "-" for r in rows) + " |",
         "| S1 cosine train / holdout | " + " | ".join(f"{r['cos_tr']:.3f} / {r['cos_ho']:.3f}" if r["cos_tr"] else "-" for r in rows) + " |",
         "| selected epoch | " + " | ".join(str(r["sel"]) for r in rows) + " |",
         "| val10 Dice (selected) | " + " | ".join(f"{r['vd']:.4f}" for r in rows) + " |",
         "| val10 HD95 mm (selected) | " + " | ".join(f"{r['vh']:.2f}" for r in rows) + " |",
         "| val10 Dice (final epoch) | " + " | ".join(f"{r['vfd']:.4f}" for r in rows) + " |",
         "| test20 Dice (selected) | " + " | ".join(f"{r['td']:.4f}" for r in rows) + " |",
         "| test20 HD95 mm (selected) | " + " | ".join(f"{r['th']:.2f}" for r in rows) + " |",
         "| test20 Dice (final epoch) | " + " | ".join(f"{r['tfd']:.4f}" for r in rows) + " |",
         "| inter-slice consistency pred / GT (val10) | " + " | ".join(f"{r['isc']:.3f}" for r in rows) + " |", ""]
    best = max(rows, key=lambda r: r["vd"])
    L.append(f"Best val10 Dice: **{best['tag']}** ({best['vd']:.4f}). Reading: whether capacity helps is judged on val10 first; test20 is reported for completeness.")
    a.out.write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
