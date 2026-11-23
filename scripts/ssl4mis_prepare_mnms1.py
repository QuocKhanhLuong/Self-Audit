#!/usr/bin/env python3
"""Materialize M&Ms-1 (10% labels) for the PG engine in the SSL4MIS manifest schema: L15 / U135 (ED+ES) / val34 / test136.

* Labeled patients: exactly the L15 of ``prepared/MnM1_h5_L15/H5_RECEIPT.json`` (the split the M3HL teacher used, seed 42).
* U: the other 135 ``Training/Labeled`` patients, IMAGES ONLY (their mask files are never opened).
* val = official ``Validation`` (34), test = official ``Testing`` (136); ``Training/Unlabeled`` (25) is not used (no ED/ES).
* ED/ES frames from the official CSV (as in the h5 preparation); classes remapped M&Ms -> ACDC order (1 LV, 3 RV -> 3 LV, 1 RV).
* Images: the repo's ``normalize_cine`` (global cine percentiles 0.5/99.5, clip [0,1]); arrays ``[2, D, H, W]`` in (z, y, x).
* Geometry: native NIfTI geometry, nothing cropped or resampled.  M&Ms-1 headers declare unknown spatial units; the zooms are
  declared millimetres (recorded per patient with provenance), the same treatment as the ACDC pixdim amendment.
Writes the split lock, the three manifests and ``PREPARATION_RECEIPT.json`` into a new output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pg_prepare_acdc import _geometry_audit, sha256_file  # noqa: E402
from self_audit.candidate_fuse.data import _CLASSES  # noqa: E402
from self_audit.preparation import normalize_cine  # noqa: E402

SCHEMA = "self_audit.ssl4mis.data.v1"
LOCK_SCHEMA = "self_audit.ssl4mis_split_lock.v1"
MNMS_TO_ACDC = np.array([0, 3, 2, 1], dtype=np.uint8)
EXPECTED = {"labeled": 15, "unlabeled": 135, "val": 34, "test": 136}
UNIT_PROVENANCE = ("M&Ms-1 NIfTI headers declare unknown spatial units; zooms are millimetres per the M&Ms-1 challenge data "
                   "description (in-plane about 0.85-1.45 mm, slice thickness 5-10 mm); declared mm for every patient.")


def prepare(source_receipt: Path, h5_receipt: Path, out: Path) -> dict:
    import nibabel as nib

    if out.exists():
        raise SystemExit(f"output must be a new directory: {out}")
    src = json.loads(source_receipt.read_text())
    h5 = json.loads(h5_receipt.read_text())
    if src.get("status") != "READY" or h5.get("status") not in (None, "READY"):
        raise SystemExit("M&Ms-1 receipts are not READY")
    subjects = src["subjects"]
    roles = {r: sorted(s for s, v in subjects.items() if v["role"] == r) for r in ("train_labeled", "val", "test")}
    l15 = sorted(h5["labeled_patients"])
    if len(l15) != EXPECTED["labeled"] or not set(l15) <= set(roles["train_labeled"]):
        raise SystemExit("L15 of the h5 receipt must be 15 Training/Labeled patients")
    split = {"train_l": l15, "train_u": sorted(set(roles["train_labeled"]) - set(l15)), "val": roles["val"], "test": roles["test"]}
    for key, name in (("train_u", "unlabeled"), ("val", "val"), ("test", "test")):
        if len(split[key]) != EXPECTED[name]:
            raise SystemExit(f"{key} has {len(split[key])} patients, expected {EXPECTED[name]}")
    out.mkdir(parents=True)
    arrays = out / "arrays"
    arrays.mkdir()
    lock = {"schema": LOCK_SCHEMA, "dataset": "mnms1", "seed": 42, "source": {"h5_receipt": str(h5_receipt), "h5_receipt_sha256": sha256_file(h5_receipt)},
            "labeled": {"patients": split["train_l"]}, "unlabeled": {"patients": split["train_u"]},
            "val": {"patients": split["val"]}, "test": {"patients": split["test"]}}
    lock_path = out / "SPLIT_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")

    needs_masks = set(split["train_l"]) | set(split["val"]) | set(split["test"])
    records, geometry, frames_of = {}, {}, {}
    for pid in sorted(sum(split.values(), [])):
        meta = subjects[pid]
        img_path = Path(meta["image"])
        if sha256_file(img_path) != meta["image_sha256"]:
            raise SystemExit(f"image differs from the verified receipt: {pid}")
        nii = nib.load(str(img_path), mmap=False)
        raw = np.asarray(nii.dataobj)
        shape = list(reversed(tuple(int(v) for v in nii.header.get_data_shape()[:3])))
        geom, audit = _geometry_audit(pid, img_path, {"shape_zyx": shape}, {pid: {"unit": "mm", "provenance": UNIT_PROVENANCE}})
        full = normalize_cine(raw, "XYZT")  # [T, Z, Y, X]
        ed, es = int(meta["csv"]["ed"]), int(meta["csv"]["es"])
        stack = np.ascontiguousarray(np.stack([full[ed], full[es]]).astype(np.float32))
        if tuple(stack.shape[1:]) != tuple(geom["shape_zyx"]):
            raise SystemExit(f"image/header shape mismatch for {pid}")
        image = arrays / f"{pid}_edes.npy"
        np.save(image, stack, allow_pickle=False)
        rec = {"id": pid, "image": str(image), "image_sha256": sha256_file(image)}
        if pid in needs_masks:
            lab_path = Path(meta["label"])
            if sha256_file(lab_path) != meta["label_sha256"]:
                raise SystemExit(f"mask differs from the verified receipt: {pid}")
            lab4 = np.asarray(nib.load(str(lab_path), mmap=False).dataobj)
            masks = {}
            for idx, frame in ((0, ed), (1, es)):
                raw_lab = np.asarray(lab4[..., frame], dtype=np.float64)
                lab = np.rint(raw_lab)  # some M&Ms masks are float64 with values like 1.0000000196; integer labels within 1e-3
                if float(np.abs(raw_lab - lab).max()) > 1e-3:
                    raise SystemExit(f"{pid}: non-integer mask values at CSV frame {frame}")
                if not lab.any() or not np.isin(lab, (0, 1, 2, 3)).all():
                    raise SystemExit(f"{pid}: empty or invalid mask at CSV frame {frame}")
                arr = MNMS_TO_ACDC[lab.transpose(2, 1, 0).astype(np.int64)]
                if tuple(arr.shape) != tuple(geom["shape_zyx"]):
                    raise SystemExit(f"{pid}: mask shape mismatch")
                dst = arrays / f"{pid}_t{idx}_mask.npy"
                np.save(dst, np.ascontiguousarray(arr, dtype=np.uint8), allow_pickle=False)
                masks[str(idx)] = {"path": str(dst), "sha256": sha256_file(dst), "source_path": str(lab_path),
                                   "source_sha256": meta["label_sha256"], "source_frame": frame, "transpose": [2, 1, 0],
                                   "class_remap": "mnms_to_acdc [0,3,2,1]"}
            rec.update({"phases": {"ED": 0, "ES": 1}, "masks": masks})
        records[pid] = rec
        frames_of[pid] = {"ED": ed, "ES": es, "vendor": meta["csv"]["vendor"]}
        geometry[pid] = {"shape": geom["shape_zyx"], "spacing_zyx": geom["spacing_zyx_mm"], "affine_zyx": geom["affine_zyx"],
                         "context": "full_volume", "provenance": {"source_geometry_audit": audit, "unit_declaration": UNIT_PROVENANCE}}
        print(f"[prepare-mnms1] {pid} ED {ed} ES {es} shape {shape} spacing {geom['spacing_zyx_mm']}", flush=True)

    def base(role: str, ids: list[str]) -> dict:
        return {"schema": SCHEMA, "role": role, "dataset": "mnms1", "synthetic": False, "class_mapping": _CLASSES,
                "expected_counts": EXPECTED, "lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
                "normalization": "global cine percentiles 0.5/99.5 over the full cine, clip [0,1]; only ED/ES frames stored",
                "geometry": {"by_patient": {p: geometry[p] for p in ids}}, "source_frames": {p: frames_of[p] for p in ids}}

    manifests = {"train": base("train", split["train_l"] + split["train_u"])}
    manifests["train"].update({"labeled": [records[p] for p in split["train_l"]],
                               "unlabeled": [{"id": p, "image": records[p]["image"], "image_sha256": records[p]["image_sha256"]} for p in split["train_u"]],
                               "patients": split["train_u"], "labeled_patients": split["train_l"]})
    for role in ("val", "test"):
        manifests[role] = base(role, split[role])
        manifests[role]["patients"] = [records[p] for p in split[role]]
    refs = {}
    for role, manifest in manifests.items():
        path = out / f"{role}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        refs[role] = {"path": str(path), "sha256": sha256_file(path)}
    receipt = {"schema": "self_audit.ssl4mis.materialization.v1", "status": "READY", "dataset": "mnms1",
               "lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)}, "manifests": refs,
               "counts": {k: len(v) for k, v in split.items()}, "split_ids": split, "source_frames": frames_of,
               "source_receipt": {"path": str(source_receipt), "sha256": sha256_file(source_receipt)},
               "mask_policy": "masks read only for L15, val34 and test136; U135 mask files never opened",
               "unit_declaration": UNIT_PROVENANCE}
    (out / "PREPARATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-receipt", type=Path, required=True, help="prepared/MnM1/PREPARATION_RECEIPT.json")
    ap.add_argument("--h5-receipt", type=Path, required=True, help="prepared/MnM1_h5_L15/H5_RECEIPT.json (L15 identity)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = prepare(a.source_receipt.resolve(), a.h5_receipt.resolve(), a.out.resolve())
    print(json.dumps({"status": r["status"], "counts": r["counts"], "manifests": r["manifests"]}, indent=1))


if __name__ == "__main__":
    main()
