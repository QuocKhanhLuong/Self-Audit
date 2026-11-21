#!/usr/bin/env python3
"""Official-path evaluation of T-M3HL on an SSL4MIS list (val.list or test.list) of the h5 tree.

Reproduces ``external/M3HL/code/test_ACDC.py``: slice-wise ``zoom(order=0)`` to 256x256, argmax, zoom back, then per class medpy
``dc/jc/hd95/asd`` on ``prediction == c`` vs ``label == c`` (the official ``pred[pred > 0] = 1`` acts on already-boolean masks, so it is a
no-op), with the official rule that a class with an empty prediction scores 0 on all four metrics.  It works for any list, so the
selection split (val10) can be scored too.  HD95/ASD are in voxels (no voxelspacing), as in the official code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def official_case_metrics(pred: np.ndarray, gt: np.ndarray, cls: int) -> tuple[float, float, float, float]:
    from medpy import metric

    p, g = (pred == cls), (gt == cls)
    if p.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (float(metric.binary.dc(p, g)), float(metric.binary.jc(p, g)), float(metric.binary.hd95(p, g)), float(metric.binary.asd(p, g)))


def main() -> int:
    import h5py
    import torch
    from scipy.ndimage import zoom

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="h5 tree (data/ACDC)")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--m3hl-source", type=Path, required=True)
    ap.add_argument("--lists", nargs="+", default=["val", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.out.is_file():
        print(f"{args.out} exists; evaluated once, refusing to overwrite", flush=True)
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from self_audit.candidate_pg.teacher_m3hl import _import_unet_hl

    net = _import_unet_hl(args.m3hl_source)(in_chns=1, class_num=4)
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    net = net.to(args.device).eval()
    result = {"schema": "self_audit.m3hl_official_eval.v1", "checkpoint": str(args.checkpoint), "units": "hd95/asd in voxels (no voxelspacing)", "lists": {}}
    for name in args.lists:
        cases = sorted(c.strip().split(".")[0] for c in (args.root / f"{name}.list").read_text().split() if c.strip())
        rows = []
        for case in cases:
            with h5py.File(args.root / "data" / f"{case}.h5", "r") as f:
                image, label = f["image"][:], f["label"][:]
            pred = np.zeros_like(label)
            for i in range(image.shape[0]):
                x, y = image[i].shape
                sl = zoom(image[i], (256 / x, 256 / y), order=0)
                inp = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).float().to(args.device)
                with torch.no_grad():
                    out = net(inp)
                    out = out[0] if len(out) > 1 else out
                    out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0).cpu().numpy()
                pred[i] = zoom(out, (x / 256, y / 256), order=0)
            per = {c: official_case_metrics(pred, label, c) for c in (1, 2, 3)}
            rows.append({"case": case, **{n: dict(zip(("dice", "jaccard", "hd95", "asd"), per[c])) for c, n in ((1, "RV"), (2, "MYO"), (3, "LV"))}})
        mean = {n: {k: float(np.mean([r[n][k] for r in rows])) for k in ("dice", "jaccard", "hd95", "asd")} for n in ("RV", "MYO", "LV")}
        overall = {k: float(np.mean([mean[n][k] for n in mean])) for k in ("dice", "jaccard", "hd95", "asd")}
        result["lists"][name] = {"volumes": len(rows), "per_class": mean, "mean": overall, "per_case": rows}
        print(f"[m3hl-official] {name}: Dice {overall['dice']:.4f} Jaccard {overall['jaccard']:.4f} HD95 {overall['hd95']:.2f} ASD {overall['asd']:.2f} (voxels)", flush=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
