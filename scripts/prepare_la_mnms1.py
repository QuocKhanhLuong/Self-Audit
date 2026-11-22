#!/usr/bin/env python3
"""Verified inventory + hash-pinned preparation receipts for M&Ms-1 (official layout) and the LA set (MSD Task02 Heart, King's College London).

Nothing is resampled, cropped or rewritten: raw NIfTI files stay where they are.  For every subject this records the sha256 of the image and label,
shape, voxel spacing (zooms, mm), the label values present, and (M&Ms) the ED/ES frames from the official CSV cross-checked against the frames that
carry a mask.  Geometry problems are collected per subject and written to the receipt (``status`` is FAILED when any subject is invalid).  M&Ms
Unlabeled subjects ship with masks in this copy; they are recorded but flagged ``labels_must_not_be_used`` (they are the unlabeled pool).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(image_path: Path, label_path: Path | None, *, classes: tuple[int, ...]) -> tuple[dict, list[str]]:
    import nibabel as nib

    problems: list[str] = []
    img = nib.load(str(image_path))
    shape, zooms = tuple(int(v) for v in img.shape), tuple(float(v) for v in img.header.get_zooms())
    if not np.isfinite(zooms).all() or any(z <= 0 for z in zooms[:3]):
        problems.append(f"invalid voxel spacing {zooms}")
    row = {"image": str(image_path), "image_sha256": sha256_file(image_path), "shape": list(shape), "zooms_mm": list(zooms)}
    if label_path is not None:
        lab = nib.load(str(label_path))
        data = np.asarray(lab.dataobj)
        values = sorted(int(v) for v in np.unique(data))
        row.update({"label": str(label_path), "label_sha256": sha256_file(label_path), "label_shape": list(lab.shape), "labels_present": values})
        if tuple(lab.shape) != shape:
            problems.append(f"label shape {tuple(lab.shape)} != image shape {shape}")
        if not set(values) <= set(classes):
            problems.append(f"unexpected label values {values}")
        if tuple(float(v) for v in lab.header.get_zooms()[:3]) != zooms[:3]:
            problems.append("label spacing differs from image spacing")
        if data.ndim == 4:
            frames = [t for t in range(data.shape[3]) if data[..., t].any()]
            row["frames_with_mask"] = frames
    return row, problems


def prepare_mnms1(raw: Path, out: Path) -> dict:
    csv_path = next(raw.glob("M&M/*diagnosis_opendataset.csv"))
    meta = {}
    for rec in csv.reader(csv_path.read_text().splitlines()):
        if len(rec) >= 8 and rec[1] and rec[1][0] == "A" or (len(rec) >= 8 and rec[1][:1] in "AB"):
            meta[rec[1]] = {"vendor_name": rec[2], "vendor": rec[3], "centre": rec[4], "ed": int(rec[5]), "es": int(rec[6]),
                            "pathology": rec[8] if len(rec) > 8 else None}
    roles = {"train_labeled": raw / "Training/Labeled", "train_unlabeled": raw / "Training/Unlabeled", "val": raw / "M&M/Validation", "test": raw / "Testing"}
    subjects, problems_all = {}, {}
    for role, folder in roles.items():
        for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
            sid = sub.name
            image, label = sub / f"{sid}_sa.nii.gz", sub / f"{sid}_sa_gt.nii.gz"
            row, problems = describe(image, label if label.exists() else None, classes=(0, 1, 2, 3))
            row["role"] = role
            m = meta.get(sid)
            if m is None:
                problems.append("subject missing from the official CSV")
            else:
                row["csv"] = m
                if "frames_with_mask" in row and sorted(row["frames_with_mask"]) != sorted({m["ed"], m["es"]}):
                    problems.append(f"mask frames {row['frames_with_mask']} differ from CSV ED/ES {(m['ed'], m['es'])}")
                if len(row["shape"]) == 4 and not (0 <= m["ed"] < row["shape"][3] and 0 <= m["es"] < row["shape"][3]):
                    problems.append("CSV ED/ES outside the cine")
            if role == "train_unlabeled" and "label" in row:
                row["labels_must_not_be_used"] = True
            if problems:
                problems_all[sid] = problems
            subjects[sid] = row
    counts = {r: sum(1 for s in subjects.values() if s["role"] == r) for r in roles}
    vendors = {}
    for s in subjects.values():
        vendors.setdefault(s["role"], {}).setdefault(s.get("csv", {}).get("vendor", "?"), 0)
        vendors[s["role"]][s.get("csv", {}).get("vendor", "?")] += 1
    receipt = {"schema": "self_audit.mnms1.preparation.v1", "status": "FAILED" if problems_all else "READY", "counts": counts, "vendors_by_role": vendors,
               "csv": {"path": str(csv_path), "sha256": sha256_file(csv_path), "subjects": len(meta)}, "problems": problems_all, "subjects": subjects,
               "note": "raw files unchanged; roles follow the official Training/Labeled, Training/Unlabeled, Validation, Testing folders"}
    write(out / "PREPARATION_RECEIPT.json", receipt)
    return receipt


def prepare_la(raw: Path, out: Path) -> dict:
    meta = json.loads((raw / "dataset.json").read_text())
    subjects, problems_all = {}, {}
    for role, folder in (("train", "imagesTr"), ("test", "imagesTs")):
        for image in sorted((raw / folder).glob("la_*.nii")):
            sid = image.stem
            label = raw / "labelsTr" / image.name if role == "train" else None
            row, problems = describe(image, label if label and label.exists() else None, classes=(0, 1))
            row["role"] = role
            if role == "train" and label is None or (role == "train" and not label.exists()):
                problems.append("missing label")
            if problems:
                problems_all[sid] = problems
            subjects[sid] = row
    counts = {r: sum(1 for s in subjects.values() if s["role"] == r) for r in ("train", "test")}
    listed = {Path(t["image"]).name.replace(".nii.gz", "") for t in meta["training"]}
    if listed != {s for s, r in subjects.items() if r["role"] == "train"}:
        problems_all["dataset.json"] = ["training list differs from imagesTr"]
    receipt = {"schema": "self_audit.la_msd_heart.preparation.v1", "status": "FAILED" if problems_all else "READY", "counts": counts, "problems": problems_all, "subjects": subjects,
               "dataset": {"name": meta.get("name"), "licence": meta.get("licence"), "numTraining": meta.get("numTraining"), "numTest": meta.get("numTest"),
                           "sha256": sha256_file(raw / "dataset.json"), "extension_note": "dataset.json lists .nii.gz; the shipped files are .nii"},
               "note": "MSD Task02 Heart (left atrium, King's College London): 20 labelled training volumes + 10 test volumes WITHOUT labels. Not the 100-volume 2018 Atrial Segmentation Challenge set."}
    write(out / "PREPARATION_RECEIPT.json", receipt)
    return receipt


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mnms1", type=Path, required=True)
    ap.add_argument("--la", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    for name, fn, raw in (("MnM1", prepare_mnms1, a.mnms1), ("LA_MSD_Heart", prepare_la, a.la)):
        r = fn(raw, a.out / name)
        print(name, r["status"], r["counts"], f"problems: {len(r['problems'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
