#!/usr/bin/env python3
"""Build the W2S-Match VAL20 manifest: the prepared VAL18 plus the two documented duplicate patients.

The seed-42 split (``splits/acdc_patient_split_seed42.json``) has 80 train / 20 val patients.  The PG preparer
excludes patient055 and patient076 (documented exact-voxel duplicates) from VAL, giving VAL18.  W2S-Match
selects on VAL20 and keeps duplicates in place, so this script materializes exactly those two patients
with the preparer's own helpers and writes ``val20.json`` = VAL18 records + the two new records.

Nothing under the source prepared directory or the raw tree is modified; output must be a new directory.
Raw cine/mask bytes are checked against the verified download inventory before use.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pg_prepare_acdc import (  # noqa: E402
    EXCLUDED_DUPLICATE_IDS, _cine_file, _geometry_audit, _mask_path, _phase_info, _raw_dirs, sha256_file,
)
from self_audit.preparation import normalize_cine  # noqa: E402
from self_audit.candidate_q3d.geometry_repair import ACDC_PIXDIM_NATIVE_V1  # noqa: E402

UNIT_PROVENANCE = "User-authorized ACDC 2026-11-12 pixdim amendment: unknown spatial units declared millimetres for acdc_pixdim_native_v1."


def build(prepared: Path, raw_root: Path, out: Path, files_receipt: Path, split_path: Path) -> dict:
    import nibabel as nib

    if out.exists():
        raise SystemExit(f"output must be a new directory: {out}")
    val = json.loads((prepared / "val.json").read_text())
    train = json.loads((prepared / "train.json").read_text())
    split = json.loads(split_path.read_text())["splits"]
    ids = lambda part: sorted(r if isinstance(r, str) else r.get("patient_id", r.get("id")) for r in (part["patients"] if isinstance(part, dict) and "patients" in part else part))
    val20, train80 = ids(split["val"]), ids(split["train"])
    have = sorted(r["id"] for r in val["patients"])
    missing = sorted(set(val20) - set(have))
    if sorted(EXCLUDED_DUPLICATE_IDS) != missing or set(have) - set(val20):
        raise SystemExit(f"VAL18 -> VAL20 difference {missing} is not the documented duplicate pair")
    if sorted(train["patients"]) != train80:
        raise SystemExit("prepared TRAIN80 differs from the seed-42 train list")
    inventory = {row["relative_path"]: row for row in json.loads(files_receipt.read_text())}
    training = _raw_dirs(raw_root, "training")

    def verified(path: Path) -> str:
        rel = path.relative_to(raw_root).as_posix()
        digest = sha256_file(path)
        if inventory[rel]["sha256"] != digest:
            raise SystemExit(f"raw file differs from the verified inventory: {rel}")
        return digest

    out.mkdir(parents=True)
    arrays = out / "arrays"
    arrays.mkdir()
    records, geometry, audits = [], {}, {}
    for pid in missing:
        cine = _cine_file(training[pid], pid)
        verified(cine)
        hdr = nib.load(str(cine), mmap=False).header
        shape = list(reversed(tuple(int(v) for v in hdr.get_data_shape()[:3])))
        decl = {} if hdr.get_xyzt_units()[0] not in {None, "unknown", ""} else {pid: {"unit": "mm", "provenance": UNIT_PROVENANCE}}
        geom, audit = _geometry_audit(pid, cine, {"shape_zyx": shape}, decl)
        array = normalize_cine(np.asarray(nib.load(str(cine), mmap=False).dataobj), "XYZT")
        if tuple(array.shape[1:]) != tuple(geom["shape_zyx"]):
            raise SystemExit(f"image/header shape mismatch for {pid}")
        image = arrays / f"{pid}_cine.npy"
        np.save(image, array, allow_pickle=False)
        phases = _phase_info(training[pid] / "Info.cfg", int(array.shape[0]))
        masks = {}
        for frame in phases.values():
            mp = _mask_path(training[pid], pid, frame)
            source_sha = verified(mp)
            labels = np.asarray(nib.load(str(mp), mmap=False).dataobj)
            if labels.ndim != 3 or tuple(labels.transpose(2, 1, 0).shape) != tuple(geom["shape_zyx"]) or not np.isin(labels, (0, 1, 2, 3)).all():
                raise SystemExit(f"invalid evaluation mask for {pid} frame {frame}")
            dst = arrays / f"{pid}_t{frame:03d}_mask.npy"
            np.save(dst, labels.transpose(2, 1, 0).astype(np.uint8, copy=False), allow_pickle=False)
            masks[str(frame)] = {"path": str(dst), "sha256": sha256_file(dst), "source_path": str(mp),
                                 "source_sha256": source_sha, "transpose": [2, 1, 0]}
        records.append({"id": pid, "image": str(image), "image_sha256": sha256_file(image),
                        "phases": {k: int(v) for k, v in phases.items()}, "masks": masks})
        geometry[pid] = {"shape": geom["shape_zyx"], "spacing_zyx": geom["spacing_zyx_mm"], "affine_zyx": geom["affine_zyx"],
                         "context": "full_volume",
                         "provenance": {"source_geometry_audit": audit, "pixdim_policy": ACDC_PIXDIM_NATIVE_V1}}
        audits[pid] = audit
    manifest = copy.deepcopy(val)
    manifest["patients"] = sorted(val["patients"] + records, key=lambda r: r["id"])
    manifest["geometry"]["by_patient"].update(geometry)
    manifest["w2s_val20"] = {"base_manifest": str(prepared / "val.json"), "base_manifest_sha256": sha256_file(prepared / "val.json"),
                             "added_patients": missing, "reason": "seed-42 VAL20 keeps the documented duplicate patients"}
    path = out / "val20.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    receipt = {"schema": "self_audit.w2s_val20.v1", "status": "READY", "val20": {"path": str(path), "sha256": sha256_file(path)},
               "base_val18": {"path": str(prepared / "val.json"), "sha256": sha256_file(prepared / "val.json")},
               "prepared_dir": str(prepared), "patients": [r["id"] for r in manifest["patients"]], "added": missing,
               "files_receipt": str(files_receipt), "files_receipt_sha256": sha256_file(files_receipt),
               "split": {"path": str(split_path), "sha256": sha256_file(split_path)},
               "train80_equals_seed42_train": True, "k8_subset_of_train80": None, "geometry_audit_added": audits}
    k8 = set(json.loads((prepared / "PREPARATION_RECEIPT.json").read_text())["labeled_ids"])
    receipt["k8_subset_of_train80"] = k8 <= set(train80)
    if not receipt["k8_subset_of_train80"]:
        raise SystemExit("K8 is not a subset of TRAIN80")
    (out / "VAL20_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--files-receipt", type=Path, required=True, help="verified download inventory (list of relative_path/sha256)")
    ap.add_argument("--split", type=Path, default=ROOT / "splits/acdc_patient_split_seed42.json")
    a = ap.parse_args()
    r = build(a.prepared.resolve(), a.raw.resolve(), a.out.resolve(), a.files_receipt.resolve(), a.split.resolve())
    print(json.dumps({k: r[k] for k in ("status", "val20", "added", "k8_subset_of_train80")}, indent=1))


if __name__ == "__main__":
    main()
