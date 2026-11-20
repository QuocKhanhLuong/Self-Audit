#!/usr/bin/env python3
"""Materialize the SSL4MIS ACDC split (L7 / U63 / val10 / test20) from pristine raw ACDC and the split lock.

Per patient only the ED and ES cine frames are stored (float32 ``[2,D,H,W]``), normalized with the repo's own
``normalize_cine`` (global cine percentiles 0.5/99.5 over the full cine, same preprocessing as pg-k8 and S1), so the
network inputs match what S1 saw.  Masks are read only for L7, val10 and test20; U63 mask files are never touched.
Raw bytes are checked against the verified download inventory.  Output must be a new directory.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pg_prepare_acdc import _cine_file, _geometry_audit, _mask_path, _phase_info, _raw_dirs, sha256_file  # noqa: E402
from self_audit.candidate_fuse.data import _CLASSES  # noqa: E402
from self_audit.candidate_q3d.geometry_repair import ACDC_PIXDIM_NATIVE_V1  # noqa: E402
from self_audit.preparation import normalize_cine  # noqa: E402

SCHEMA = "self_audit.ssl4mis.data.v1"
UNIT_PROVENANCE = "User-authorized ACDC 2026-11-12 pixdim amendment: unknown spatial units declared millimetres for acdc_pixdim_native_v1."


def prepare(lock_path: Path, raw_root: Path, out: Path, files_receipt: Path) -> dict:
    import nibabel as nib

    if out.exists():
        raise SystemExit(f"output must be a new directory outside the raw tree: {out}")
    lock = json.loads(lock_path.read_text())
    if lock.get("schema") != "self_audit.ssl4mis_split_lock.v1":
        raise SystemExit("unexpected split lock schema")
    roles = {"train_l": lock["labeled"]["patients"], "train_u": lock["unlabeled"]["patients"],
             "val": lock["val"]["patients"], "test": lock["test"]["patients"]}
    all_ids = sum(roles.values(), [])
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != 100:
        raise SystemExit("lock patient sets must be disjoint and cover 100 patients")
    inventory = {row["relative_path"]: row for row in json.loads(files_receipt.read_text())}
    training = _raw_dirs(raw_root, "training")
    if set(all_ids) - set(training):
        raise SystemExit(f"patients missing from raw training folder: {sorted(set(all_ids) - set(training))[:5]}")

    def verified(path: Path) -> str:
        rel, digest = path.relative_to(raw_root).as_posix(), sha256_file(path)
        if inventory[rel]["sha256"] != digest:
            raise SystemExit(f"raw file differs from the verified inventory: {rel}")
        return digest

    out.mkdir(parents=True)
    arrays = out / "arrays"
    arrays.mkdir()
    needs_masks = set(roles["train_l"]) | set(roles["val"]) | set(roles["test"])
    records, geometry, audits, source_frames, voxel_hashes = {}, {}, {}, {}, {}
    for pid in sorted(all_ids):
        cine = _cine_file(training[pid], pid)
        verified(cine)
        nii = nib.load(str(cine), mmap=False)
        raw_array = np.asarray(nii.dataobj)
        voxel_hashes[pid] = hashlib.sha256(np.ascontiguousarray(raw_array).tobytes()).hexdigest()
        shape = list(reversed(tuple(int(v) for v in nii.header.get_data_shape()[:3])))
        decl = {} if nii.header.get_xyzt_units()[0] not in {None, "unknown", ""} else {pid: {"unit": "mm", "provenance": UNIT_PROVENANCE}}
        geom, audit = _geometry_audit(pid, cine, {"shape_zyx": shape}, decl)
        full = normalize_cine(raw_array, "XYZT")
        if tuple(full.shape[1:]) != tuple(geom["shape_zyx"]):
            raise SystemExit(f"image/header shape mismatch for {pid}")
        phases = _phase_info(training[pid] / "Info.cfg", int(full.shape[0]))
        ed, es = int(phases["ED"]), int(phases["ES"])
        if not ed < es:
            raise SystemExit(f"{pid}: ED {ed} does not precede ES {es}; frame01/frame02 naming would be ambiguous")
        stack = np.ascontiguousarray(np.stack([full[ed], full[es]]).astype(np.float32))
        image = arrays / f"{pid}_edes.npy"
        np.save(image, stack, allow_pickle=False)
        rec = {"id": pid, "image": str(image), "image_sha256": sha256_file(image)}
        source_frames[pid] = {"ED": ed, "ES": es}
        if pid in needs_masks:
            masks = {}
            for idx, frame in ((0, ed), (1, es)):
                mp = _mask_path(training[pid], pid, frame)
                source_sha = verified(mp)
                labels = np.asarray(nib.load(str(mp), mmap=False).dataobj)
                if labels.ndim != 3 or tuple(labels.transpose(2, 1, 0).shape) != tuple(geom["shape_zyx"]) or not np.isin(labels, (0, 1, 2, 3)).all():
                    raise SystemExit(f"invalid mask for {pid} frame {frame}")
                dst = arrays / f"{pid}_t{idx}_mask.npy"
                np.save(dst, labels.transpose(2, 1, 0).astype(np.uint8, copy=False), allow_pickle=False)
                masks[str(idx)] = {"path": str(dst), "sha256": sha256_file(dst), "source_path": str(mp), "source_sha256": source_sha,
                                   "source_frame": frame, "transpose": [2, 1, 0]}
            rec.update({"phases": {"ED": 0, "ES": 1}, "masks": masks})
        records[pid] = rec
        geometry[pid] = {"shape": geom["shape_zyx"], "spacing_zyx": geom["spacing_zyx_mm"], "affine_zyx": geom["affine_zyx"],
                         "context": "full_volume", "provenance": {"source_geometry_audit": audit, "pixdim_policy": ACDC_PIXDIM_NATIVE_V1}}
        audits[pid] = audit
        print(f"[prepare] {pid} ED {ed} ES {es} shape {shape}", flush=True)

    def base(role: str, ids: list[str]) -> dict:
        return {"schema": SCHEMA, "role": role, "dataset": "acdc", "synthetic": False, "class_mapping": _CLASSES,
                "lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
                "normalization": "global cine percentiles 0.5/99.5 over the full cine, clip [0,1]; only ED/ES frames stored",
                "geometry": {"by_patient": {p: geometry[p] for p in ids}},
                "source_frames": {p: source_frames[p] for p in ids}}

    manifests = {}
    train = base("train", roles["train_l"] + roles["train_u"])
    train.update({"labeled": [records[p] for p in sorted(roles["train_l"])],
                  "unlabeled": [{"id": p, "image": records[p]["image"], "image_sha256": records[p]["image_sha256"]} for p in sorted(roles["train_u"])],
                  "patients": sorted(roles["train_u"]), "labeled_patients": sorted(roles["train_l"])})
    manifests["train"] = train
    for role in ("val", "test"):
        manifest = base(role, roles[role])
        manifest["patients"] = [records[p] for p in sorted(roles[role])]
        manifests[role] = manifest
    refs = {}
    for role, manifest in manifests.items():
        path = out / f"{role}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        refs[role] = {"path": str(path), "sha256": sha256_file(path)}
    groups = collections.defaultdict(list)
    for pid, digest in voxel_hashes.items():
        groups[digest].append(pid)
    duplicates = [sorted(g) for g in groups.values() if len(g) > 1]
    split_of = {p: role for role, ids in roles.items() for p in ids}
    cross = [g for g in duplicates if len({"train" if split_of[p].startswith("train") else split_of[p] for p in g}) > 1]
    receipt = {"schema": "self_audit.ssl4mis.materialization.v1", "status": "READY", "lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
               "manifests": refs, "counts": {k: len(v) for k, v in roles.items()}, "split_ids": roles,
               "raw_inventory": {"files_receipt": str(files_receipt), "sha256": sha256_file(files_receipt)},
               "duplicate_groups_exact_voxel": duplicates, "duplicate_groups_crossing_train_vs_val_test": cross,
               "source_frames": source_frames, "geometry_policy": ACDC_PIXDIM_NATIVE_V1, "geometry_audit": audits,
               "mask_policy": "masks read only for L7, val10 and test20; U63 mask files never opened"}
    (out / "PREPARATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock", type=Path, default=ROOT / "configs/candidate_pg/ssl4mis_split_lock.json")
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--files-receipt", type=Path, required=True)
    a = ap.parse_args()
    r = prepare(a.lock.resolve(), a.raw.resolve(), a.out.resolve(), a.files_receipt.resolve())
    print(json.dumps({"status": r["status"], "counts": r["counts"], "manifests": r["manifests"],
                      "duplicates": r["duplicate_groups_exact_voxel"], "crossing": r["duplicate_groups_crossing_train_vs_val_test"]}, indent=1))


if __name__ == "__main__":
    main()
