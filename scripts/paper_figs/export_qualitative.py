#!/usr/bin/env python3
"""Export one comparable slice per dataset for the qualitative figure (SASNet vs other models).

Figures only: forward passes of trained, validation-selected checkpoints; no training, no change of model code.

Spec (JSON)::

    {"datasets": [
       {"name": "ACDC", "slug": "acdc", "config": "<config.json>", "split": "test", "colors": "cardiac",
        "models": [{"label": "SASNet", "run": "<run dir>", "ckpt": "resume_A2.pt", "config": "<optional own config>"},
                   {"label": "AnisoStudent", "run": "...", "ckpt": "resume_A2.pt", "config": "..."}]}],
     "out": "reports/paper-figures-20261124/qualitative"}

The first model of every dataset is the reference.  Fixed selection rule: the patient whose ED 3D Dice of the reference
model is closest to the median over the split (ties: lowest id); slice = the middle one of the slices that contain
ground-truth foreground; crop = square box around the GT foreground of the whole stack + 15 % margin.  Every model is
run on that same patient, so the columns of the figure are directly comparable.  A model that is missing (no
checkpoint yet) can be left out of the spec and added later by re-running the script.

Usage (GPU machine, repository root)::

    python scripts/paper_figs/export_qualitative.py --spec spec.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

COLORS = {"cardiac": {1: (0xE5, 0x39, 0x35), 2: (0x43, 0xA0, 0x47), 3: (0x1E, 0x88, 0xE5)},  # RV, MYO, LV
          "la": {3: (0x8E, 0x24, 0xAA)}}  # LA is stored in slot 3
ALPHA, OUT_PX, MARGIN = 0.55, 400, 0.15


def dice_present(target, prediction, classes) -> float:
    """3D Dice averaged over the classes that occur in the ground truth."""
    import numpy as np

    vals = []
    for c in classes:
        a, b = target == c, prediction == c
        if a.sum() == 0:
            continue
        vals.append(2.0 * float((a & b).sum()) / float(a.sum() + b.sum()))
    return float(np.mean(vals))


def overlay(gray_u8, label, colors):
    import numpy as np

    rgb = np.stack([gray_u8] * 3, axis=-1).astype("float32")
    for c, color in colors.items():
        m = label == c
        rgb[m] = (1.0 - ALPHA) * rgb[m] + ALPHA * np.asarray(color, dtype="float32")
    return rgb.round().astype("uint8")


def load_model(cfg_path, run, ckpt_name, device):
    import torch
    from w2s_common import load_config
    from self_audit.candidate_pg.student_pg import build_student

    cfg = load_config(Path(cfg_path))
    ckpt_path = Path(run) / ckpt_name
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("best_student") is None:
        raise RuntimeError(f"{ckpt_path}: no best_student")
    student = build_student(cfg)
    student.load_state_dict(ckpt["best_student"], strict=True)
    return student.to(device).eval(), ckpt_path


def main() -> int:
    import numpy as np
    import torch
    from PIL import Image
    from export_fig_slices import square_crop, to_uint8
    from self_audit.candidate_pg import engine
    from w2s_common import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    device = torch.device(args.device)
    out_root = Path(spec["out"])

    for ds in spec["datasets"]:
        colors = COLORS[ds.get("colors", "cardiac")]
        classes = sorted(colors)
        cfg = load_config(Path(ds["config"]))
        bundle = engine._load_data(cfg, synthetic=False)["objects"]
        part = bundle.test if ds.get("split", "test") == "test" else bundle.val
        by_patient: dict[str, list[int]] = {}
        for patient, frame in part.labeled_keys():
            by_patient.setdefault(str(patient), []).append(int(frame))
        ed_key = {p: (p, min(f)) for p, f in by_patient.items()}

        def predict(student, patient):
            frame = part.load(ed_key[patient], labeled=True)
            image, target, geometry = engine._frame_tensors(frame, device, labeled=True)
            pred = engine._extract_logits(student(image, geometry)).argmax(1)[0].cpu().numpy().astype("int64")
            return image[0, 0].cpu().numpy().astype("float32"), target[0].cpu().numpy().astype("int64"), pred

        models = ds["models"]
        ref_model, _ = load_model(models[0].get("config", ds["config"]), models[0]["run"], models[0].get("ckpt", "resume_A2.pt"), device)
        with torch.no_grad():
            rows = []
            for patient in sorted(ed_key):
                _, tgt, pred = predict(ref_model, patient)
                rows.append((patient, dice_present(tgt, pred, classes)))
        median = float(np.median([d for _, d in rows]))
        patient, ref_dice = min(rows, key=lambda r: (abs(r[1] - median), r[0]))
        del ref_model

        preds, hashes, dices = [], [], []
        img = tgt = None
        for m in models:
            net, ck = load_model(m.get("config", ds["config"]), m["run"], m.get("ckpt", "resume_A2.pt"), device)
            with torch.no_grad():
                img, tgt, pred = predict(net, patient)
            preds.append(pred)
            dices.append(dice_present(tgt, pred, classes))
            hashes.append(hashlib.sha256(ck.read_bytes()).hexdigest())
            del net

        fg = np.where((tgt > 0).any(axis=(1, 2)))[0]
        z = int(fg[len(fg) // 2])
        _, height, width = img.shape
        y0, x0, side = square_crop(tgt > 0, MARGIN, height, width)
        crop = lambda a: a[y0:y0 + side, x0:x0 + side]  # noqa: E731
        gray = to_uint8(crop(img[z]))
        out = out_root / ds["slug"]
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(gray).resize((OUT_PX, OUT_PX), Image.BICUBIC).save(out / "in.png")
        Image.fromarray(overlay(gray, crop(tgt[z]), colors)).resize((OUT_PX, OUT_PX), Image.NEAREST).save(out / "gt.png")
        for i, pr in enumerate(preds):
            Image.fromarray(overlay(gray, crop(pr[z]), colors)).resize((OUT_PX, OUT_PX), Image.NEAREST).save(out / f"pred_{i}.png")
        meta = {"name": ds["name"], "split": ds.get("split", "test"), "patient": patient, "frame_index_ed": ed_key[patient][1],
                "n_patients": len(rows), "reference": models[0]["label"], "reference_dice": ref_dice, "split_median_dice": median,
                "models": [{"label": m["label"], "run": m["run"], "ckpt_sha256": h, "patient_dice_3d": d}
                           for m, h, d in zip(models, hashes, dices)],
                "slice_index": z, "crop_box_yx_side": [y0, x0, side], "overlay": {"alpha": ALPHA, "classes": {str(k): "#%02X%02X%02X" % v for k, v in colors.items()}},
                "selection_rule": "reference-model ED 3D Dice closest to split median; ties lowest id; middle GT-foreground slice",
                "note": "test masks are used for patient selection, cropping and the GT column only; no metric is reported from this script"}
        (out / "META.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[qualitative] {ds['name']}: {patient} ED {ed_key[patient][1]} slice {z} ref Dice {ref_dice:.4f} (median {median:.4f}); " + ", ".join(f"{m['label']} {d:.4f}" for m, d in zip(models, dices)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
