#!/usr/bin/env python3
"""Export real ACDC slices and the real SliceSSM prediction for the paper figures (Fig 1 and Fig 4).

Figures only: no training, no change of model code, no re-run of a reported evaluation.  The script reads the test split
once to run the val10-selected student (``best_student`` of ``resume_A2.pt``) on every ED frame, picks ONE patient by a
fixed rule and writes 15 PNG files plus ``META.json``.

Selection rule (fixed in advance, not a search for a good-looking case): among the test20 patients, take the patient
whose 3D mean Dice over RV/MYO/LV at the ED frame is closest to the median over test20 (ties: lowest patient id).

ED frame: the smaller ``frame_index`` of the two labelled frames of a patient (in ACDC, ED precedes ES in the cardiac
cycle).  Both indices are written to META.json.

Usage (GPU machine, repository root)::

    python scripts/paper_figs/export_fig_slices.py \
        --config <run>/config.json --run <run dir with resume_A2.pt> --out reports/paper-figures-20261124/slices
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CLASS_COLORS = {1: (0xE5, 0x39, 0x35), 2: (0x43, 0xA0, 0x47), 3: (0x1E, 0x88, 0xE5)}  # RV, MYO, LV
ALPHA = 0.55
OUT_PX = 400
N_SLICES = 5
MARGIN = 0.15
TOLERANCE = 0.01  # chosen patient must be within +-1 Dice point of the test20 median


def patient_dice(target, prediction) -> float:
    """3D Dice averaged over the three foreground classes (RV, MYO, LV) of one frame."""
    import numpy as np

    vals = []
    for c in (1, 2, 3):
        a, b = target == c, prediction == c
        denom = a.sum() + b.sum()
        vals.append(1.0 if denom == 0 else 2.0 * float((a & b).sum()) / float(denom))
    return float(np.mean(vals))


def square_crop(foreground, margin: float, height: int, width: int) -> tuple[int, int, int]:
    """Square box (y0, x0, side) around the in-plane foreground bounding box of the whole stack plus ``margin``."""
    import numpy as np

    ys, xs = np.where(foreground.any(axis=0))
    cy, cx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
    side = int(round(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * (1.0 + 2.0 * margin)))
    side = min(side, height, width)
    y0 = int(round(cy - side / 2.0))
    x0 = int(round(cx - side / 2.0))
    y0 = max(0, min(y0, height - side))
    x0 = max(0, min(x0, width - side))
    return y0, x0, side


def to_uint8(stack, lo_hi=(1.0, 99.0)):
    import numpy as np

    lo, hi = np.percentile(stack, lo_hi)
    return (np.clip((stack - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255.0).round().astype("uint8")


def upscale(img, size: int, nearest: bool):
    from PIL import Image

    return Image.fromarray(img).resize((size, size), Image.NEAREST if nearest else Image.BICUBIC)


def overlay(gray_u8, label):
    """RGB uint8: grayscale slice with the label map blended at ALPHA (class colours)."""
    import numpy as np

    rgb = np.stack([gray_u8] * 3, axis=-1).astype("float32")
    for c, color in CLASS_COLORS.items():
        m = label == c
        rgb[m] = (1.0 - ALPHA) * rgb[m] + ALPHA * np.asarray(color, dtype="float32")
    return rgb.round().astype("uint8")


def main() -> int:
    import numpy as np
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="run directory that holds resume_A2.pt (SliceSSM S0, seed 42)")
    ap.add_argument("--out", type=Path, required=True, help="output directory (gets in_k.png, out_k.png, shift_k.png, META.json)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", choices=("test", "val"), default="test", help="test20 (default) or val10; the same rule applies")
    ap.add_argument("--seed", type=int, default=0, help="seed of the injected per-slice shifts of shift_k.png")
    args = ap.parse_args()

    from w2s_common import load_config  # noqa: E402
    from self_audit.candidate_pg import engine
    from self_audit.candidate_pg.slicessm_student import apply_slice_shifts, sample_slice_shifts
    from self_audit.candidate_pg.student_pg import build_student

    cfg, device = load_config(args.config), torch.device(args.device)
    ckpt_path = args.run / "resume_A2.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("best_student") is None:
        raise RuntimeError("checkpoint has no best_student (the run has not produced a val10-selected model)")
    student = build_student(cfg)
    if type(student).__name__ != "SliceSSMStudent":
        raise RuntimeError(f"config builds {type(student).__name__}, not SliceSSMStudent")
    student.load_state_dict(ckpt["best_student"], strict=True)
    student.to(device).eval()

    bundle = engine._load_data(cfg, synthetic=False)["objects"]
    part = bundle.test if args.split == "test" else bundle.val
    keys = list(part.labeled_keys())
    by_patient: dict[str, list[int]] = {}
    for patient, frame in keys:
        by_patient.setdefault(str(patient), []).append(int(frame))
    ed_key = {p: (p, min(frames)) for p, frames in by_patient.items()}

    cache, rows = {}, []
    with torch.no_grad():
        for patient in sorted(ed_key):
            frame = part.load(ed_key[patient], labeled=True)
            image, target, geometry = engine._frame_tensors(frame, device, labeled=True)
            logits = engine._extract_logits(student(image, geometry))
            pred = logits.argmax(1)[0].cpu().numpy().astype("int64")
            tgt = target[0].cpu().numpy().astype("int64")
            img = image[0, 0].cpu().numpy().astype("float32")
            rows.append((patient, patient_dice(tgt, pred)))
            cache[patient] = (img, tgt, pred)
    dices = np.array([d for _, d in rows])
    median = float(np.median(dices))
    patient, dice = min(rows, key=lambda r: (abs(r[1] - median), r[0]))
    if abs(dice - median) > TOLERANCE:
        raise RuntimeError(f"chosen patient Dice {dice:.4f} is not within +-{TOLERANCE} of the median {median:.4f}")

    img, tgt, pred = cache[patient]
    depth, height, width = img.shape
    fg_slices = np.where((tgt > 0).any(axis=(1, 2)))[0]
    pick = [int(fg_slices[int(round(i))]) for i in np.linspace(0, len(fg_slices) - 1, N_SLICES)]  # base (low z index) to apex
    y0, x0, side = square_crop(tgt > 0, MARGIN, height, width)
    crop = lambda a: a[..., y0:y0 + side, x0:x0 + side]  # noqa: E731
    img_c, pred_c = crop(img[pick]), crop(pred[pick])
    img_u8 = to_uint8(img_c)

    args.out.mkdir(parents=True, exist_ok=True)
    stack = torch.from_numpy(img_c.astype("float32"))[None, None]  # [1,1,5,side,side] native crop
    gen = torch.Generator().manual_seed(args.seed)
    shifts = sample_slice_shifts(1, N_SLICES, 8, generator=gen)  # +-8 px in native pixels (as in training)
    shifted, _ = apply_slice_shifts(stack, shifts)
    # same percentile window for the shifted stack as for the clean one, so the only change is the shift
    lo, hi = np.percentile(img_c, (1.0, 99.0))
    shifted_u8 = (np.clip((shifted[0, 0].numpy() - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 255.0).round().astype("uint8")

    from PIL import Image

    for k in range(N_SLICES):
        upscale(img_u8[k], OUT_PX, nearest=False).save(args.out / f"in_{k}.png")
        rgb = overlay(img_u8[k], pred_c[k])
        Image.fromarray(rgb).resize((OUT_PX, OUT_PX), Image.NEAREST).save(args.out / f"out_{k}.png")
        upscale(shifted_u8[k], OUT_PX, nearest=False).save(args.out / f"shift_{k}.png")

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {
        "patient": patient, "frame_index_ed": ed_key[patient][1], "frames_of_patient": sorted(by_patient[patient]),
        "ed_rule": "smaller frame_index of the patient's two labelled frames", "split": args.split,
        "selection_rule": "patient whose ED 3D mean Dice (RV/MYO/LV) is closest to the split median; ties: lowest id",
        "patient_dice": dice, "split_median_dice": median, "abs_diff_to_median": abs(dice - median), "n_patients": len(rows),
        "per_patient_dice": {p: d for p, d in rows},
        "slice_indices": pick, "slice_order": "k=0 base (lowest z index with GT foreground) .. k=4 apex",
        "crop_box_yx_side": [y0, x0, side], "crop_rule": f"GT foreground box of the whole stack + {int(MARGIN * 100)} % margin, square",
        "intensity": "1st/99th percentile of the cropped 5-slice stack (shift_k uses the same window)",
        "overlay": {"alpha": ALPHA, "RV": "#E53935", "MYO": "#43A047", "LV": "#1E88E5"},
        "shifts_px_dy_dx": shifts[0].tolist(), "shift_seed": args.seed, "shift_max_px": 8,
        "image_px": OUT_PX, "checkpoint": str(ckpt_path), "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "checkpoint_best_epoch": int(ckpt.get("best_epoch", -1)), "checkpoint_best_val_fg": float(ckpt.get("best_val", float("nan"))),
        "config": str(args.config), "git_hash": git,
        "note": "the test split is read only to run the model once for this figure; test masks are used for patient selection and cropping only",
    }
    (args.out / "META.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[fig-slices] patient {patient} ED frame {ed_key[patient][1]} Dice {dice:.4f} (median {median:.4f}, n={len(rows)}); slices {pick}; crop {y0},{x0},{side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
