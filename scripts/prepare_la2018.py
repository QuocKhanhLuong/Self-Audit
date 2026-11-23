#!/usr/bin/env python3
"""Prepare the 2018 Atrial Segmentation Challenge (Utah) LA set in the UA-MT / M3HL h5 format, with a hash-pinned receipt.

* Raw NRRD files are never modified.  Output goes to ``<out>/data/<case>/mri_norm2.h5`` (``image`` float32, ``label`` uint8).
* Conversion = UA-MT ``code/dataloaders/la_heart_processing.py`` (read ``lgemri.nrrd``/``laendo.nrrd`` with pynrrd, label = (laendo == 255), crop x/y around the label
  box with margins 10-20 px plus padding up to the 112 x 112 output size, whole-volume z-score).  The upstream script draws the margins with an unseeded
  ``np.random``; here they are drawn with seed 42 per case in sorted order and the exact box is recorded, so the conversion is reproducible.  z is not cropped (as upstream).
* Split = the upstream ``train.list`` (80) and ``test.list`` (20) of the 100-volume "Training Set"; the lists must partition it exactly.  The 54-volume "Testing Set" is
  inventoried (hashes, shapes, spacing, label values) but not converted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUTPUT_SIZE = (112, 112, 80)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def spacing_of(header: dict) -> list[float]:
    dirs = np.asarray(header["space directions"], dtype=float)
    return [float(np.linalg.norm(dirs[i])) for i in range(3)]


def crop_box(label: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int]:
    w, h, _ = label.shape
    nz = np.nonzero(label)
    minx, maxx, miny, maxy = nz[0].min(), nz[0].max(), nz[1].min(), nz[1].max()
    px, py = max(OUTPUT_SIZE[0] - (maxx - minx), 0) // 2, max(OUTPUT_SIZE[1] - (maxy - miny), 0) // 2
    minx = max(minx - int(rng.integers(10, 20)) - px, 0)
    maxx = min(maxx + int(rng.integers(10, 20)) + px, w)
    miny = max(miny - int(rng.integers(10, 20)) - py, 0)
    maxy = min(maxy + int(rng.integers(10, 20)) + py, h)
    # the upstream z margins are drawn (and ignored); keep the same number of draws per case so the stream matches the upstream order
    rng.integers(5, 10), rng.integers(5, 10)
    return int(minx), int(maxx), int(miny), int(maxy)


def main() -> int:
    import h5py
    import nrrd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True, help="folder holding 'Training Set' and 'Testing Set'")
    ap.add_argument("--lists", type=Path, required=True, help="folder with the upstream train.list and test.list")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    train_ids = [x.strip() for x in (a.lists / "train.list").read_text().split() if x.strip()]
    test_ids = [x.strip() for x in (a.lists / "test.list").read_text().split() if x.strip()]
    all_cases = sorted(p.name for p in (a.raw / "Training Set").iterdir() if p.is_dir())
    problems: list[str] = []
    if sorted(train_ids + test_ids) != all_cases or set(train_ids) & set(test_ids):
        problems.append("train.list + test.list do not partition the Training Set")
    rng = np.random.default_rng(42)
    role = {**{c: "train" for c in train_ids}, **{c: "test" for c in test_ids}}
    cases, bad = {}, {}
    for case in all_cases:
        folder = a.raw / "Training Set" / case
        img_path, lab_path = folder / "lgemri.nrrd", folder / "laendo.nrrd"
        image, ih = nrrd.read(str(img_path))
        label_raw, lh = nrrd.read(str(lab_path))
        values = sorted(int(v) for v in np.unique(label_raw))
        issues = []
        if image.shape != label_raw.shape:
            issues.append(f"shape {image.shape} != label shape {label_raw.shape}")
        if not set(values) <= {0, 255}:
            issues.append(f"unexpected label values {values}")
        sp = spacing_of(ih)
        if not np.isfinite(sp).all() or min(sp) <= 0:
            issues.append(f"invalid spacing {sp}")
        label = (label_raw == 255).astype(np.uint8)
        if not label.any():
            issues.append("empty label")
        if issues:
            bad[case] = issues
            continue
        box = crop_box(label, rng)
        x0, x1, y0, y1 = box
        norm = ((image - np.mean(image)) / np.std(image)).astype(np.float32)
        target = a.out / "data" / case / "mri_norm2.h5"
        target.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(target, "w") as f:
            f.create_dataset("image", data=norm[x0:x1, y0:y1], compression="gzip")
            f.create_dataset("label", data=label[x0:x1, y0:y1], compression="gzip")
        cases[case] = {"role": role.get(case), "image_sha256": sha256_file(img_path), "label_sha256": sha256_file(lab_path), "shape": list(image.shape),
                       "spacing_mm": sp, "label_values_raw": values, "crop_xy": [x0, x1, y0, y1], "h5": str(target), "h5_sha256": sha256_file(target),
                       "cropped_shape": [x1 - x0, y1 - y0, int(image.shape[2])]}
    testing = {}
    for folder in sorted(p for p in (a.raw / "Testing Set").iterdir() if p.is_dir()):
        image, ih = nrrd.read(str(folder / "lgemri.nrrd"))
        label_raw, _ = nrrd.read(str(folder / "laendo.nrrd"))
        testing[folder.name] = {"image_sha256": sha256_file(folder / "lgemri.nrrd"), "label_sha256": sha256_file(folder / "laendo.nrrd"), "shape": list(image.shape),
                                "spacing_mm": spacing_of(ih), "label_values_raw": sorted(int(v) for v in np.unique(label_raw))}
    for sid, ps in bad.items():
        problems.append(f"{sid}: {'; '.join(ps)}")
    receipt = {"schema": "self_audit.la2018.preparation.v1", "status": "FAILED" if problems else "READY", "problems": problems,
               "counts": {"train": sum(1 for c in cases.values() if c["role"] == "train"), "test": sum(1 for c in cases.values() if c["role"] == "test"), "official_testing_set_inventory_only": len(testing)},
               "lists_sha256": {n: sha256_file(a.lists / f"{n}.list") for n in ("train", "test")}, "output_size": list(OUTPUT_SIZE), "crop_seed": 42,
               "cases": cases, "official_testing_set": testing,
               "note": "UA-MT conversion with seed-42 margins; lists from yulequan/UA-MT data/{train,test}.list (80/20 partition of the 100-volume Training Set)"}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "PREPARATION_RECEIPT.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    for n in ("train", "test"):
        (a.out / f"{n}.list").write_text((a.lists / f"{n}.list").read_text())
    print(receipt["status"], receipt["counts"], f"problems: {len(problems)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
