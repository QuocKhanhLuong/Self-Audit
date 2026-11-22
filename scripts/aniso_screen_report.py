#!/usr/bin/env python3
"""ANISO_SCREEN.md: the pre-registered screening decision on val10 only (test20 is never read).

Keep AnisoStudent iff HD95 (mm) <= the UNETR++-lite HD95 and Dice >= UNETR++-lite - 0.005 (0.5 Dice points); also report the inter-slice consistency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", type=Path, required=True)
    ap.add_argument("--aniso", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    u, n = (json.loads((d / "FINAL_RESULT.json").read_text()) for d in (a.unet, a.aniso))
    if u["test20_opened"] or n["test20_opened"] or "test20" in u["metrics"] or "test20" in n["metrics"]:
        raise SystemExit("screening FINAL results must not contain test20")
    g = lambda r: r["metrics"]["val10"]["native"]
    du, dn = g(u)["fg_dice"], g(n)["fg_dice"]
    hu, hn = g(u)["fg_hd95_mm"], g(n)["fg_hd95_mm"]
    keep = hn <= hu and dn >= du - 0.005
    gf = lambda r: r["final_epoch_metrics"]["val10"]["native"]
    L = ["# AnisoStudent screening (val10 only; test20 not opened)", "",
         "Two UniMatch arms without teacher, seed 42, 150 epochs, val10-selected checkpoint. The S1 inits differ: UNETR++-lite uses the existing S1 (1,933 cine frames incl. the val/test patients),",
         "AnisoStudent uses S1-ANISO (ED/ES of the 70 TRAIN patients only, 140 volumes). Architecture and S1 data are therefore confounded in this screen. One seed, n = 10 patients.", "",
         "| | UNETR++-lite | AnisoStudent |", "|---|---|---|",
         f"| parameters (deployed) | {u['complexity']['parameters']:,} | {n['complexity']['parameters']:,} |",
         f"| FLOPs (native val input) | {u['complexity']['flops'] / 1e9:.1f} G | {n['complexity']['flops'] / 1e9:.1f} G |",
         f"| selected epoch | {u['selected_epoch']} | {n['selected_epoch']} |",
         f"| val10 Dice (selected) | {du:.4f} | {dn:.4f} |", f"| val10 HD95 mm (selected) | {hu:.2f} | {hn:.2f} |",
         f"| val10 Dice (final epoch) | {gf(u)['fg_dice']:.4f} | {gf(n)['fg_dice']:.4f} |", f"| val10 HD95 mm (final epoch) | {gf(u)['fg_hd95_mm']:.2f} | {gf(n)['fg_hd95_mm']:.2f} |",
         f"| inter-slice consistency pred / GT | {u['inter_slice_consistency_val10']['ratio']:.3f} | {n['inter_slice_consistency_val10']['ratio']:.3f} |", "",
         f"Decision rule (pre-registered): keep AnisoStudent iff HD95 <= UNETR++-lite and Dice >= UNETR++-lite - 0.005. Dice diff {dn - du:+.4f}, HD95 diff {hn - hu:+.2f} mm.",
         f"**Decision: {'KEEP' if keep else 'DROP'} AnisoStudent** (on this confounded one-seed screen)."]
    a.out.write_text("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
