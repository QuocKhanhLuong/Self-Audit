#!/usr/bin/env python3
"""Part C2: offline TPC-Gate analysis on the frozen M3HL teacher (inference only, batch 1, no_grad). Pre-registered; see reports/m3hl-teacher-20261121/TPC_OFFLINE.md.

Data: val10 (20 volumes; the only set used to choose q) and U63 (126 ED/ES volumes; ground truth is read from the raw ACDC files as a
DIAGNOSTIC only, never for selection or training).  test20 is not opened.  All signals live in the aligned frame (see tpc_signals); errors, ROI
and flagged sets use the same aligned frame, so metrics are consistent but not identical to unaligned-frame Dice/HD95.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w2s_common import load_config, load_teacher  # noqa: E402

TTA = ["hflip", "vflip", "hvflip", "rot90", "rot180", "rot270", "gamma0.7", "gamma1.4"]
Q_GRID = (0.80, 0.90, 0.95)


def tta_apply(x, name):
    import torch

    if name == "hflip": return x.flip(-1)
    if name == "vflip": return x.flip(-2)
    if name == "hvflip": return x.flip(-1).flip(-2)
    if name.startswith("rot"): return torch.rot90(x, int(name[3:]) // 90, dims=(-2, -1))
    return x.clamp(0, 1) ** float(name[5:])


def tta_invert(p, name):
    import torch

    if name == "hflip": return p.flip(-1)
    if name == "vflip": return p.flip(-2)
    if name == "hvflip": return p.flip(-1).flip(-2)
    if name.startswith("rot"): return torch.rot90(p, -(int(name[3:]) // 90), dims=(-2, -1))
    return p


def load_gt_raw(raw: Path, pid: str, frame_no: int):
    import nibabel as nib

    path = raw / pid / f"{pid}_frame{frame_no:02d}_gt.nii"
    return np.asarray(nib.load(str(path)).dataobj).transpose(2, 1, 0).astype(np.int64)


def volume_signals(prob, pred, gt, tta_dis, cardiac):
    """All signals in the aligned frame for one volume; returns dict of arrays plus pred/gt/roi/valid/conf."""
    from self_audit.candidate_pg import tpc_signals as t

    base = t.baselines(prob)
    extras = (gt, base["one_minus_conf"], base["entropy"], base["one_minus_margin"], base["slice_position"], tta_dis)
    ap, valid, (agt, a_conf, a_ent, a_mar, a_pos, a_tta) = t.align(pred, cardiac=cardiac, extra=extras)
    roi = t.region_of_interest(ap, valid)
    sig = t.tpc_score(ap, valid, roi, cardiac=cardiac)
    sig.update({"one_minus_conf": a_conf, "entropy": a_ent, "one_minus_margin": a_mar, "slice_position": a_pos, "s_tta": a_tta})
    return sig, ap, agt, roi, valid


def boundary_band(pred, radius=2):
    from scipy import ndimage as ndi

    edge = np.zeros(pred.shape, dtype=bool)
    for c in (1, 2, 3):
        m = pred == c
        for z in range(pred.shape[0]):
            if m[z].any():
                edge[z] |= m[z] ^ ndi.binary_erosion(m[z])
    return np.stack([ndi.binary_dilation(e, iterations=radius) if e.any() else e for e in edge])


def fg_dice(pred, gt):
    ds = []
    for c in (1, 2, 3):
        a, b = pred == c, gt == c
        s = a.sum() + b.sum()
        ds.append(1.0 if s == 0 else 2.0 * (a & b).sum() / s)
    return float(np.mean(ds))


def hd95_mm(pred, gt, spacing):
    from medpy import metric

    vals = []
    for c in (1, 2, 3):
        a, b = pred == c, gt == c
        vals.append(float(metric.binary.hd95(a, b, voxelspacing=tuple(spacing))) if a.any() and b.any() else 0.0)
    return float(np.mean(vals))


def main() -> int:
    import torch

    from self_audit.candidate_pg import engine, tpc_signals as t
    from self_audit.candidate_pg.student_pg import PGStudent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-config", type=Path, required=True)
    ap.add_argument("--students", nargs="+", required=True, help="name=config.json:run_dir:engine_arm")
    ap.add_argument("--raw", type=Path, required=True, help="raw ACDC training folder (U63 diagnostic ground truth)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="debug: volumes per set")
    args = ap.parse_args()
    device = torch.device(args.device)
    cfg = load_config(args.teacher_config)
    teacher, _ = load_teacher(cfg, Path("."), device)
    bundle = engine._load_data(cfg, synthetic=False)["objects"]
    students = {}
    for spec in args.students:
        name, rest = spec.split("=", 1)
        cpath, run, arm = rest.split(":")
        scfg = load_config(Path(cpath))
        st = PGStudent(scfg)
        ck = torch.load(Path(run) / f"resume_{arm}.pt", map_location="cpu", weights_only=False)
        st.load_state_dict(ck["best_student"], strict=True)
        students[name] = st.to(device).eval()
    source_frames = json.loads(Path(cfg["data"]["manifests"]["train"]["path"]).read_text())["source_frames"]
    jobs = [("val10", bundle.val, [(k, True) for k in bundle.val.labeled_keys()]),
            ("U63", bundle.train, [(k, False) for k in bundle.train.unlabeled_keys()])]
    store: dict = {}
    for split, data, keys in jobs:
        for key, labeled in keys[: args.limit]:
            frame = data.load(key, labeled=labeled)
            image = frame.image.to(device)
            geom = {k: v for k, v in frame.geometry.items()}
            with torch.no_grad():
                prob = teacher.probabilities(image)
                refp = prob.argmax(1)
                votes = []
                for name in TTA:
                    pv = tta_invert(teacher.probabilities(tta_apply(image, name)), name)
                    votes.append((pv.argmax(1) != refp).float())
                tta_dis = torch.stack(votes).mean(0)[0].cpu().numpy()
                preds = {n: engine._extract_logits(s(image, geom)).argmax(1)[0].cpu().numpy() for n, s in students.items()}
            prob_np, pred = prob[0].cpu().numpy(), refp[0].cpu().numpy()
            gt = frame.target[0].numpy() if labeled else load_gt_raw(args.raw, key[0], source_frames[key[0]]["ED" if key[1] == 0 else "ES"])
            assert gt.shape == pred.shape, (key, gt.shape, pred.shape)
            rec = {"patient": key[0], "spacing": [float(v) for v in geom["spacing_zyx"]], "variants": {}}
            for variant in ("generic", "cardiac"):
                sig, ap_, agt, roi, valid = volume_signals(prob_np, pred, gt, tta_dis, variant == "cardiac")
                wrong = (ap_ != agt)
                band = boundary_band(ap_)
                conf99 = (sig["one_minus_conf"] <= 0.01)
                out = {"auroc": {}, "band": {}, "conf99": {}}
                for name, s in sig.items():
                    for tag, region in (("auroc", roi), ("band", roi & band), ("conf99", roi & conf99)):
                        out[tag][name] = t.auroc(s[region], wrong[region]) if region.any() else float("nan")
                flagged_q = {}
                for q in Q_GRID:
                    thr = np.quantile(sig["tpc"][roi], q) if roi.any() else 2.0
                    flag = roi & (sig["tpc"] >= thr) & (sig["tpc"] > 0)
                    fixed = np.where(flag, agt, ap_)
                    flagged_q[q] = {"dice_before": fg_dice(ap_, agt), "dice_after": fg_dice(fixed, agt), "hd_before": hd95_mm(ap_, agt, rec["spacing"]),
                                    "hd_after": hd95_mm(fixed, agt, rec["spacing"]), "flag_frac": float(flag.sum() / max(roi.sum(), 1))}
                    # E3: where student and teacher disagree inside the flag set, who is right
                    for n, sp in preds.items():
                        if sp is None: continue
                        aligned_sp = t.apply_shifts(sp, t.slice_shifts(pred, (3,) if variant == "cardiac" else None))
                        dis = flag & (aligned_sp != ap_)
                        flagged_q[q][f"e3_{n}"] = {"n": int(dis.sum()), "student_right": int((dis & (aligned_sp == agt)).sum()), "teacher_right": int((dis & (ap_ == agt)).sum())}
                out["flag"] = {str(q): v for q, v in flagged_q.items()}
                rec["variants"][variant] = out
            e0 = {"err": int(((pred != gt) & t.region_of_interest(pred, np.ones(pred.shape, bool))).sum()), "roi": int(t.region_of_interest(pred, np.ones(pred.shape, bool)).sum())}
            wr = (pred != gt) & t.region_of_interest(pred, np.ones(pred.shape, bool))
            e0["err_conf99"] = int((wr & (prob_np.max(0) >= 0.99)).sum())
            d = pred.shape[0]
            terc = np.minimum((np.arange(d) * 3 // max(d, 1)), 2)
            e0["terciles"] = [{"err": int(wr[terc == k].sum()), "roi": int(t.region_of_interest(pred, np.ones(pred.shape, bool))[terc == k].sum()),
                               "dice": fg_dice(pred[terc == k], gt[terc == k]) if (terc == k).any() else None} for k in range(3)]
            e0["class_dice"] = {c: (2.0 * ((pred == c) & (gt == c)).sum() / max(((pred == c).sum() + (gt == c).sum()), 1)) for c in (1, 2, 3)}
            rec["e0"] = e0
            store.setdefault(split, []).append(rec)
        print(f"[tpc] {split}: {len(store[split])} volumes", flush=True)
    args.out.write_text(json.dumps(store, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
