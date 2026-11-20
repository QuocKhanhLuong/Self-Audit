#!/usr/bin/env python3
"""Phase 0 oracles (no teacher, no training): O1 pooled-code oracle, O2 shape-atom oracle on VAL18.

O1: pool one-hot GT to the stage-3 grid (1,8,8) as class fractions (valid-aware), upsample trilinearly to native,
    argmax.  Also the mixed-cell fraction (no class > 0.7), over all cells and over cells containing MYO.
O2: k-means (K=128) on TRAIN L_K one-hot GT patches (patch from --patch; shape_atoms.patch_split/assemble_full_res), replace each VAL18 patch by its nearest centroid,
    argmax, reassemble.
Both use GT only for measurement; the k-means uses TRAIN L_K masks only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

PATCH = (2, 8, 8)  # default; --patch overrides (the shape-atom stage uses the student stage-3 grid, 1x8x8)


def onehot(t: torch.Tensor) -> torch.Tensor:
    return F.one_hot(t.long(), 4).permute(3, 0, 1, 2).float()  # [4,D,H,W]


def pooled_oracle(target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """O1 on one volume ``[D,H,W]``: returns the prediction and mixed-cell statistics."""

    from self_audit.candidate_q3d.alignment import pool_native_cells

    d, h, w = target.shape
    phase = ((1, 8, 8), (0.0, 0.0, 0.0))
    frac = pool_native_cells(onehot(target)[None], (d, h, w), phase)  # [1,4,Dg,Hg,Wg]
    gd, gh, gw = frac.shape[-3:]
    # place cell centres at voxel i*s and interpolate back with trilinear (align_corners=True maps centres exactly)
    pos = [torch.linspace(0, (n - 1) / s, n) for n, s in ((d, 1), (h, 8), (w, 8))]
    grid = torch.stack(torch.meshgrid(*[2 * (p / max(g - 1, 1)) - 1 if g > 1 else torch.zeros_like(p) for p, g in zip(pos, (gd, gh, gw))], indexing="ij"), dim=-1)
    sample = F.grid_sample(frac, grid[None][..., [2, 1, 0]], mode="bilinear", align_corners=True)
    prediction = sample[0].argmax(0)
    peak = frac[0].max(0).values
    has_myo = frac[0, 2] > 0
    stats = {"cells": int(peak.numel()), "mixed_cells": int((peak <= 0.7).sum()), "myo_cells": int(has_myo.sum()),
             "myo_mixed_cells": int(((peak <= 0.7) & has_myo).sum())}
    return prediction, stats


def kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    centroids = x[torch.randperm(x.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centroids).argmin(1)
        for j in range(k):
            members = x[assign == j]
            if members.numel():
                centroids[j] = members.mean(0)
    return centroids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--patch", type=int, nargs=3, default=list(PATCH))
    ap.add_argument("--only", choices=("O1", "O2", "both"), default="both")
    args = ap.parse_args()
    from self_audit.candidate_pg import evaluation, shape_atoms as sa
    patch = tuple(args.patch)
    from self_audit.candidate_pg.data import open_pg_data
    from self_audit.candidate_pg.config import validate_config

    cfg = validate_config(json.loads(args.config.read_text()))
    bundle = open_pg_data(cfg)
    train = [bundle.train.load(k, labeled=True) for k in bundle.train.labeled_keys()]
    val = [bundle.val.load(k, labeled=True) for k in bundle.val.labeled_keys()]
    tgt = lambda f: torch.as_tensor(f.target).long().reshape(-1, *f.target.shape[-2:]) if f.target.ndim == 3 else torch.as_tensor(f.target)[0].long()
    centroids = kmeans(torch.cat([sa.patch_split(tgt(f), patch) for f in train]), args.k)
    recs = {"O1": [], "O2": []}
    mixed = {"cells": 0, "mixed_cells": 0, "myo_cells": 0, "myo_mixed_cells": 0}
    for f in val:
        t = tgt(f)
        p1 = None
        if args.only != "O2":
            p1, st = pooled_oracle(t)
            for k, v in st.items():
                mixed[k] += v
        rows = sa.patch_split(t, patch)
        p2 = sa.assemble_full_res(centroids[torch.cdist(rows, centroids).argmin(1)], tuple(t.shape), patch)
        base = {"patient": str(getattr(f, "patient_id", "?")), "frame": int(getattr(f, "frame_index", 0)), "target": t, "geometry": f.geometry}
        if p1 is not None:
            recs["O1"].append({**base, "prediction": p1})
        recs["O2"].append({**base, "prediction": p2})
    result = {name: {k: v for k, v in evaluation.evaluate_records(r, path="student_only").items() if k != "per_patient"} for name, r in recs.items() if args.only in ("both", name)}
    result["patch"] = list(patch)
    if args.only != "O2":
        result["O1_mixed_cell_fraction"] = {"all": mixed["mixed_cells"] / max(1, mixed["cells"]), "myo": mixed["myo_mixed_cells"] / max(1, mixed["myo_cells"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in ("O1", "O2", "patch")}, default=str)[:1500])


if __name__ == "__main__":
    main()
