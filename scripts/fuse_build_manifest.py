#!/usr/bin/env python3
"""Build strict, new Candidate FUSE manifests from prepared native ACDC arrays.

The source TRAIN manifest supplies all 80 TRAIN cine arrays and repaired
``acdc_pixdim_native_v1`` geometry. Original NIfTI files supply exact Info.cfg
ED/ES indices and every TRAIN target is reconstructed from its original label
volume using transpose (2,1,0), without interpolation. All TRAIN labels must
exist. Existing files are read only; output must be a new directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

# Permit the documented ``python scripts/fuse_build_manifest.py`` invocation
# when the repository root is not already on PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from self_audit.candidate_fuse.data import _CLASSES, _geometry, _sha


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _phase_info(path: Path, count: int):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\s*(ED|ES)\s*:\s*(\d+)\s*", line, flags=re.I)
        if match:
            key, value = match.group(1).upper(), int(match.group(2)) - 1
            if key in values or value < 0 or value >= count:
                raise ValueError(f"invalid or duplicate {key} frame in {path}")
            values[key] = value
    if set(values) != {"ED", "ES"} or values["ED"] == values["ES"]:
        raise ValueError(f"{path} must declare distinct ED/ES frames")
    return values


def _split(path: Path):
    split = json.loads(path.read_text(encoding="utf-8"))
    # Accept only an explicit full cohort. The historical ACDC split lacks a
    # test cohort and has VAL20, so it must not be silently reused for FUSE.
    nested = split.get("splits", split)
    aliases = {"train": ("train",), "development": ("development", "val"), "test": ("test",)}
    result = {}
    for role, candidates in aliases.items():
        source = next((key for key in candidates if key in nested), None)
        if source is None:
            raise ValueError(f"locked split is missing {role} patient IDs")
        entry = nested[source]
        result[role] = list(entry["patients"] if isinstance(entry, dict) else entry)
    all_ids = sum(result.values(), [])
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("TRAIN/development/TEST patient sets overlap")
    expected = {"train": 80, "development": 18, "test": 50}
    if {role: len(ids) for role, ids in result.items()} != expected:
        raise ValueError("FUSE split must be exactly TRAIN80/VAL18/TEST50")
    return result


def build(source_train, source_development, source_test, source_split, original_root, out):
    sources = {"train": Path(source_train).resolve(), "development": Path(source_development).resolve(),
               "test": Path(source_test).resolve()}
    original_roots = original_root if isinstance(original_root, dict) else {"train": original_root, "development": original_root, "test": original_root}
    original_roots = {role: Path(value).resolve() for role, value in original_roots.items()}
    out = Path(out).resolve()
    if out.exists() or any(out == root or out.is_relative_to(root) for root in original_roots.values()):
        raise ValueError("output must be a new directory outside original source")
    split = _split(Path(source_split).resolve())
    train_manifest = json.loads(sources["train"].read_text(encoding="utf-8"))
    dev_manifest = json.loads(sources["development"].read_text(encoding="utf-8"))
    test_manifest = json.loads(sources["test"].read_text(encoding="utf-8"))
    if train_manifest.get("synthetic") is not False or train_manifest.get("dataset") != "acdc":
        raise ValueError("real ACDC source TRAIN manifest required")
    if set(r["id"] for r in train_manifest.get("labeled", []) + train_manifest.get("unlabeled", [])) != set(split["train"]):
        raise ValueError("prepared TRAIN source differs from locked patient split")
    if set(r["id"] for r in dev_manifest.get("patients", [])) != set(split["development"]):
        raise ValueError("prepared development source differs from locked split")
    if set(r["id"] for r in test_manifest.get("patients", [])) != set(split["test"]):
        raise ValueError("prepared TEST source differs from locked split")
    source_geometries = {
        "train": train_manifest.get("geometry", {}).get("by_patient", {}),
        "development": dev_manifest.get("geometry", {}).get("by_patient", {}),
        "test": test_manifest.get("geometry", {}).get("by_patient", {}),
    }
    images = {}
    geometries = {}

    def image_for(pid, source_record):
        source = (sources["train"] if pid in split["train"] else
                  sources["development"] if pid in split["development"] else sources["test"])
        path = Path(source_record["image"])
        if not path.is_absolute(): path = source.parent / path
        digest = _sha(path)
        if digest != source_record.get("image_sha256"):
            raise ValueError(f"prepared image SHA mismatch for {pid}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 4 or arr.shape[0] < 3 or not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
            raise ValueError(f"invalid native prepared cine for {pid}")
        source_role = "train" if pid in split["train"] else "development" if pid in split["development"] else "test"
        geometry_raw = source_geometries[source_role].get(pid)
        if not isinstance(geometry_raw, dict) or geometry_raw.get("schema") != "self_audit.geometry.native_zyx.v1":
            raise ValueError(f"missing repaired pixdim geometry for {pid}")
        geometry = _geometry({"shape": geometry_raw["shape_zyx"], "spacing_zyx": geometry_raw["spacing_zyx_mm"],
                              "affine_zyx": geometry_raw["affine_zyx"], "context": "full_volume"})
        geometry["provenance"] = {"source_schema": geometry_raw["schema"], "source_geometry": geometry_raw}
        if tuple(arr.shape[1:]) != tuple(geometry["shape"]):
            raise ValueError(f"native image/geometry shape mismatch for {pid}")
        # The normalized cache arrays are immutable source artifacts. Keeping
        # their resolved absolute paths avoids redundant multi-gigabyte copies.
        rec = {"id": pid, "image": str(path.resolve()), "image_sha256": _sha(path)}
        images[pid], geometries[pid] = (arr, rec), geometry
        return rec

    prepared_by_role = {
        "development": {r["id"]: r for r in dev_manifest.get("patients", [])},
        "test": {r["id"]: r for r in test_manifest.get("patients", [])},
    }
    # Use TRAIN's normalized native arrays for every train patient, including those
    # formerly called U; no existing split role or mask inventory is trusted.
    train_by_id = {r["id"]: r for r in train_manifest.get("labeled", []) + train_manifest.get("unlabeled", [])}
    all_records = {"train": [], "development": [], "test": []}
    for pid in split["train"] + split["development"] + split["test"]:
        role = "train" if pid in split["train"] else "development" if pid in split["development"] else "test"
        src = train_by_id.get(pid) if role == "train" else prepared_by_role[role].get(pid)
        if src is None:
            raise ValueError(f"prepared {role} source is missing locked patient {pid}")
        all_records[role].append(image_for(pid, src))

    # Preflight every source annotation before creating output so a partial
    # original-data inventory never leaves a deceptively usable build folder.
    import nibabel as nib
    original_hashes = {}
    source_masks = {}
    for role in ("train", "development", "test"):
        for rec in all_records[role]:
            pid = rec["id"]
            role = "train" if pid in split["train"] else "development" if pid in split["development"] else "test"
            folder = original_roots[role] / pid
            phases = _phase_info(folder / "Info.cfg", images[pid][0].shape[0])
            source_masks[pid] = {}
            for phase, frame in phases.items():
                matches = list(folder.glob(f"{pid}_frame{frame+1:02d}_gt.nii*"))
                if len(matches) != 1:
                    raise ValueError(f"missing or duplicate original {role} {phase} mask for {pid}")
                source_masks[pid][frame] = matches[0]
                original_hashes[str(matches[0])] = _sha(matches[0])
    out.mkdir(parents=True)
    labeled = []
    unlabeled = []
    for rec in all_records["train"]:
        pid = rec["id"]
        cine = images[pid][0]
        folder = original_roots["train"] / pid
        phases = _phase_info(folder / "Info.cfg", cine.shape[0])
        mask_entries = {}
        for phase, frame in phases.items():
            src_mask = source_masks[pid][frame]
            source_hash = original_hashes[str(src_mask)]
            nii = nib.load(str(src_mask))
            labels_xyz = np.asarray(nii.dataobj)
            if labels_xyz.ndim != 3 or tuple(labels_xyz.shape) != tuple(cine.shape[1:][::-1]):
                raise ValueError(f"original mask shape does not match exact XYZ/ZYX transpose for {pid}")
            if not np.isin(labels_xyz, [0, 1, 2, 3]).all():
                raise ValueError(f"unknown native ACDC label in {src_mask}")
            mask = labels_xyz.transpose(2, 1, 0).astype(np.uint8, copy=False)
            if not np.array_equal(mask.transpose(2, 1, 0), labels_xyz):
                raise ValueError("exact mask transpose check failed")
            dst = out / f"{pid}_t{frame:03d}_mask.npy"
            np.save(dst, mask)
            mask_entries[str(frame)] = {"path": str(dst), "sha256": _sha(dst), "source_path": str(src_mask),
                                        "source_sha256": source_hash, "source_axes": ["X", "Y", "Z"],
                                        "array_axes": ["Z", "Y", "X"], "transpose": [2, 1, 0]}
        labeled.append({**rec, "phases": phases, "masks": mask_entries})
        unlabeled.append({k: rec[k] for k in ("id", "image", "image_sha256")})

    role_manifests = {}
    for role in ("train", "development", "test"):
        records = all_records[role]
        if role != "train":
            output_records = []
            for rec in records:
                pid = rec["id"]; role_root = "development" if pid in split["development"] else "test"
                folder = original_roots[role_root] / pid; phases = _phase_info(folder / "Info.cfg", images[pid][0].shape[0]); masks = {}
                for phase, frame in phases.items():
                    src = source_masks[pid][frame]; source_hash = original_hashes[str(src)]; raw = np.asarray(nib.load(str(src)).dataobj)
                    arr = raw.transpose(2, 1, 0).astype(np.uint8, copy=False)
                    dst = out / f"{pid}_t{frame:03d}_mask.npy"; np.save(dst, arr); original_hashes[str(src)] = source_hash
                    masks[str(frame)] = {"path": str(dst), "sha256": _sha(dst), "source_path": str(src),
                                         "source_sha256": source_hash, "source_axes": ["X", "Y", "Z"],
                                         "array_axes": ["Z", "Y", "X"], "transpose": [2, 1, 0]}
                output_records.append({**rec, "phases": phases, "masks": masks})
        else: output_records = None
        manifest = {"schema": "self_audit.candidate_fuse.data.v1", "role": role, "synthetic": False,
                    "dataset": "acdc", "class_mapping": _CLASSES,
                    "split": {"train": split["train"], "development": split["development"], "test": split["test"]},
                    "source_split": str(Path(source_split).resolve()), "source_split_sha256": _sha(Path(source_split).resolve()),
                    "source_manifests": {r: {"path": str(p), "sha256": _sha(p)} for r, p in sources.items()},
                    "source_original_mask_sha256": original_hashes,
                    "geometry": {"by_patient": {r["id"]: geometries[r["id"]] for r in records}}}
        if role == "train": manifest.update(patients=split["train"], labeled=labeled, unlabeled=unlabeled)
        else: manifest["patients"] = output_records
        path = out / f"{role}.json"; _write_json(path, manifest)
        role_manifests[role] = {"path": str(path), "sha256": _sha(path)}
    _write_json(out / "BUILD_RECEIPT.json", {"schema": "self_audit.candidate_fuse.build_receipt.v1",
        "outputs": role_manifests, "source_split_sha256": _sha(Path(source_split).resolve()),
        "original_mask_sha256": original_hashes, "mask_reads": len(original_hashes), "u_mask_reads": 0,
        "axis_transform": "original XYZ to array ZYX transpose (2,1,0), no interpolation"})
    return role_manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--source-development", required=True)
    parser.add_argument("--source-test", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--original-root", help="common root containing TRAIN/DEV/TEST patient folders")
    parser.add_argument("--original-train-root")
    parser.add_argument("--original-development-root")
    parser.add_argument("--original-test-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    roots = ({"train": args.original_train_root, "development": args.original_development_root,
              "test": args.original_test_root} if args.original_train_root else args.original_root)
    if roots is None or (isinstance(roots, dict) and any(value is None for value in roots.values())):
        parser.error("supply --original-root or all three role-specific original roots")
    print(json.dumps(build(args.source_train, args.source_development, args.source_test, args.split,
                           roots, args.out), indent=2))


if __name__ == "__main__":
    main()
