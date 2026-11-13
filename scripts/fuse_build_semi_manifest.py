#!/usr/bin/env python3
"""Build a metadata-only FUSE semi-supervised TRAIN v2 manifest.

The two inputs are the full-label FUSE TRAIN v1 manifest and the locked
``train.q3d-pixdim.json`` K8 manifest. This script reads JSON metadata and
NumPy image headers only. It never opens a mask array or NIfTI file and never
copies image, mask, or geometry payloads.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


DEFAULT_SOURCE_TRAIN = "/root/mount_data/prepared/ACDC/fuse-final-20261113/train.json"
DEFAULT_LOCKED_K8 = "/root/mount_data/prepared/ACDC/k8/train.q3d-pixdim.json"
DEFAULT_OUT = "/root/mount_data/prepared/ACDC/fuse-k8-20261113"
LOCKED_K8_SHA256 = "60dd11747570f3ac183e8636050a3ac3d73651cb43b37deba9ff7422b442a433"
_CLASSES = {"0": "BG", "1": "RV", "2": "MYO", "3": "LV"}
_LOCKED_K8_CLASSES = {"0": "background", "1": "RV", "2": "MYO", "3": "LV", "255": "ignore"}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return value, _sha_bytes(raw)


def _require_source_shapes(train: dict, locked: dict, *, synthetic: bool):
    if (train.get("schema") != "self_audit.candidate_fuse.data.v1"
            or train.get("role") != "train" or train.get("synthetic") is not synthetic
            or train.get("dataset") != ("synthetic" if synthetic else "acdc")):
        raise ValueError("source must be the expected full-label FUSE TRAIN v1 manifest")
    if train.get("class_mapping") != _CLASSES:
        raise ValueError("full-label source has an unexpected class mapping")
    if not isinstance(train.get("split"), dict) or not isinstance(train["split"].get("train"), list):
        raise ValueError("full-label source has no locked TRAIN split")

    expected_top = {"schema", "role", "synthetic", "dataset", "class_mapping",
                    "label_seed", "label_budget", "evaluation_ids", "geometry",
                    "labeled", "unlabeled"}
    # This is the observed Q3D K8 source interface. A changed interface must be
    # reviewed explicitly instead of inferred from similarly named fields.
    if set(locked) != expected_top:
        raise ValueError("locked K8 source keys differ from the reviewed train.q3d-pixdim interface")
    if (locked.get("schema") != "self_audit.data.v2" or locked.get("role") != "training"
            or locked.get("synthetic") is not synthetic or locked.get("dataset") != train["dataset"]
            or locked.get("class_mapping") != (_CLASSES if synthetic else _LOCKED_K8_CLASSES)
            or locked.get("label_seed") != 42):
        raise ValueError("locked K8 source identity, dataset, class mapping, or seed mismatch")
    if (not isinstance(locked.get("label_budget"), int)
            or isinstance(locked["label_budget"], bool)
            or locked["label_budget"] != len(locked.get("labeled", []))):
        raise ValueError("locked K8 label_budget does not match its L records")
    if not synthetic and locked["label_budget"] != 8:
        raise ValueError("real ACDC semi-supervised build requires locked L8")


def _records_by_id(records, label):
    if not isinstance(records, list):
        raise ValueError(f"{label} records must be a list")
    result = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError(f"{label} contains an invalid record")
        if record["id"] in result:
            raise ValueError(f"{label} contains duplicate patient IDs")
        result[record["id"]] = record
    return result


def _image_only(record):
    if set(record) != {"id", "image", "image_sha256"}:
        raise ValueError("full-label TRAIN image record has unexpected keys")
    # Construct a fresh allowlisted object so no hidden mask reference can
    # survive into U, even when the input record is augmented in the future.
    return {key: record[key] for key in ("id", "image", "image_sha256")}


def _mask_hashes(record):
    masks = record.get("masks")
    if not isinstance(masks, dict) or not masks:
        raise ValueError("locked L records must carry a non-empty mask map")
    result = {}
    for frame, mask in masks.items():
        if not isinstance(frame, str) or not isinstance(mask, dict) or set(mask) != {"path", "sha256"}:
            raise ValueError("locked K8 mask entries differ from the reviewed path/SHA interface")
        if not isinstance(mask["path"], str) or not mask["path"]:
            raise ValueError("locked K8 mask path must be a non-empty string")
        result[frame] = mask["sha256"]
    return result


def build(source_train, locked_k8, out, *, expected_locked_sha256=LOCKED_K8_SHA256,
          synthetic=False, replace_existing=False):
    """Write one new TRAIN manifest and receipt without reading any masks."""
    source_path = Path(source_train).resolve()
    locked_path = Path(locked_k8).resolve()
    out_dir = Path(out).resolve()
    if source_path == locked_path or out_dir == source_path.parent or out_dir == locked_path.parent:
        raise ValueError("output must be a new directory separate from source manifests")
    train, source_sha = _read_json(source_path)
    locked, locked_sha = _read_json(locked_path)
    if expected_locked_sha256 is not None and locked_sha != expected_locked_sha256:
        raise ValueError("locked K8 source manifest SHA256 mismatch")
    _require_source_shapes(train, locked, synthetic=synthetic)
    if out_dir.exists():
        if not replace_existing:
            raise FileExistsError(f"output directory already exists: {out_dir}")
        receipt_path = out_dir / "BUILD_RECEIPT.json"
        existing_manifest = out_dir / "train.json"
        if (not receipt_path.is_file() or not existing_manifest.is_file()
                or {p.name for p in out_dir.iterdir()} != {"BUILD_RECEIPT.json", "train.json"}):
            raise ValueError("existing output is not an owned FUSE semi build")
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (previous.get("schema") != "self_audit.candidate_fuse.semi_build_receipt.v1"
                or previous.get("manifest") != str(existing_manifest)
                or previous.get("manifest_sha256") != _sha_file(existing_manifest)
                or previous.get("full_label_source") != str(source_path)
                or previous.get("full_label_source_sha256") != source_sha
                or previous.get("locked_l_source") != str(locked_path)
                or previous.get("locked_l_source_sha256") != locked_sha):
            raise ValueError("existing output receipt does not match these exact source artifacts")

    source_l = _records_by_id(train.get("labeled"), "full-label source L")
    source_u = _records_by_id(train.get("unlabeled"), "full-label source U")
    locked_l = _records_by_id(locked.get("labeled"), "locked K8 L")
    locked_u = _records_by_id(locked.get("unlabeled"), "locked K8 U")
    source_ids = set(source_l)
    if source_ids != set(source_u) or source_ids != set(train["split"]["train"]):
        raise ValueError("full-label source L/U records differ from its locked TRAIN split")
    if len(source_ids) != len(train["split"]["train"]):
        raise ValueError("full-label source TRAIN IDs must be unique")
    if set(locked_l) & set(locked_u) or set(locked_l) | set(locked_u) != source_ids:
        raise ValueError("locked K8 L/U IDs must exactly partition the source TRAIN set")
    locked_ids = sorted(locked_l)
    if not locked_ids:
        raise ValueError("locked K8 L set is empty")
    for pid in locked_ids:
        source_record, lock_record = source_l.get(pid), locked_l[pid]
        if source_record is None:
            raise ValueError(f"locked L patient is absent from full-label source: {pid}")
        if set(source_record) != {"id", "image", "image_sha256", "phases", "masks"}:
            raise ValueError("full-label source L record keys differ from the reviewed FUSE v1 interface")
        if set(lock_record) != {"id", "image", "image_sha256", "masks"}:
            raise ValueError("locked K8 L record keys differ from the reviewed interface")
        if source_record.get("image_sha256") != lock_record.get("image_sha256"):
            raise ValueError(f"locked L image SHA differs from full-label source for {pid}")
        source_masks = source_record.get("masks")
        if not isinstance(source_masks, dict):
            raise ValueError(f"full-label source L masks are malformed for {pid}")
        if _mask_hashes(lock_record) != {
                frame: entry.get("sha256") for frame, entry in source_masks.items()
                if isinstance(entry, dict)}:
            raise ValueError(f"locked L mask hashes differ from full-label source for {pid}")
        phases = source_record.get("phases")
        if not isinstance(phases, dict) or set(phases) != {"ED", "ES"}:
            raise ValueError(f"full-label source L lacks its ED/ES frame map for {pid}")
        if set(source_masks) != {str(phases["ED"]), str(phases["ES"])}:
            raise ValueError(f"full-label source L masks are not exactly ED/ES for {pid}")
    for pid in locked_u:
        if set(locked_u[pid]) != {"id", "image", "image_sha256"}:
            raise ValueError("locked K8 U record keys differ from the reviewed interface")
        if locked_u[pid].get("image_sha256") != source_u[pid].get("image_sha256"):
            raise ValueError(f"locked K8 U image SHA differs from full-label source for {pid}")

    # Keep the FUSE list order; only selected patient records retain masks.
    labeled = [deepcopy(source_l[pid]) for pid in source_l if pid in set(locked_ids)]
    # Strip masks from every record before any consumer or validation can see it.
    unlabeled = [_image_only(source_u[pid]) for pid in source_u]
    manifest = deepcopy(train)
    manifest["schema"] = "self_audit.candidate_fuse.data.v2"
    manifest["labeled"] = labeled
    manifest["unlabeled"] = unlabeled
    manifest["labeled_contract"] = {
        "patient_ids": locked_ids,
        "patient_ids_sha256": _sha_bytes(json.dumps(
            locked_ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")),
        "source_manifest": str(locked_path),
        "source_manifest_sha256": locked_sha,
    }
    source_inventory = train.get("source_original_mask_sha256")
    if not isinstance(source_inventory, dict):
        raise ValueError("full-label source is missing its original mask hash inventory")
    retained_inventory = {}
    for record in labeled:
        for entry in record["masks"].values():
            source_mask_path = entry.get("source_path")
            source_mask_sha = entry.get("source_sha256")
            if not isinstance(source_mask_path, str) or not source_mask_path:
                raise ValueError("selected L mask lacks source path provenance")
            if source_inventory.get(source_mask_path) != source_mask_sha:
                raise ValueError("selected L mask source hash differs from source inventory")
            retained_inventory[source_mask_path] = source_mask_sha
    manifest["source_original_mask_sha256"] = retained_inventory

    # Read image headers only to produce frame-count evidence. Never touch any
    # mask or NIfTI payload; all referenced arrays remain at their source paths.
    total_train_frames = 0
    unlabeled_frames = 0
    for pid, record in source_u.items():
        raw_image = Path(record["image"])
        image_path = raw_image if raw_image.is_absolute() else source_path.parent / raw_image
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        if image.ndim != 4 or image.shape[0] < 3:
            raise ValueError(f"invalid image array header for {pid}")
        count = int(image.shape[0])
        total_train_frames += count
        unlabeled_frames += count - 2 if pid in locked_l else count

    manifest_path = out_dir / "train.json"
    receipt_path = out_dir / "BUILD_RECEIPT.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = _sha_bytes(manifest_bytes)
    receipt = {
        "schema": "self_audit.candidate_fuse.semi_build_receipt.v1",
        "dataset": train["dataset"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "full_label_source": str(source_path),
        "full_label_source_sha256": source_sha,
        "locked_l_source": str(locked_path),
        "locked_l_source_sha256": locked_sha,
        "locked_l_patient_ids": locked_ids,
        "locked_l_patient_ids_sha256": manifest["labeled_contract"]["patient_ids_sha256"],
        "train_patient_count": len(source_ids),
        "labeled_patient_count": len(locked_ids),
        "labeled_frame_count": sum(len(r["phases"]) for r in labeled),
        "train_frame_count": total_train_frames,
        "unlabeled_frame_count": unlabeled_frames,
        "unlabeled_patient_count": len(source_ids),
        "mask_array_reads": 0,
        "nifti_reads": 0,
        "non_l_mask_references_in_output": 0,
        "original_mask_inventory_retained": len(retained_inventory),
        "payload_copies": 0,
    }
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    return {"manifest": str(manifest_path), "sha256": manifest_sha,
            "receipt": str(receipt_path), "locked_l_ids": locked_ids,
            "locked_l_ids_sha256": receipt["locked_l_patient_ids_sha256"],
            "counts": {key: receipt[key] for key in (
                "train_patient_count", "labeled_patient_count", "labeled_frame_count",
                "train_frame_count", "unlabeled_frame_count", "unlabeled_patient_count")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", default=DEFAULT_SOURCE_TRAIN)
    parser.add_argument("--locked-k8", default=DEFAULT_LOCKED_K8)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--expected-locked-sha256", default=LOCKED_K8_SHA256)
    parser.add_argument("--synthetic", action="store_true", help="permit small software-only L sets")
    parser.add_argument("--replace-existing-output", action="store_true",
                        help="replace only a receipt-verified prior output from these exact sources")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.source_train, args.locked_k8, args.out,
                           expected_locked_sha256=args.expected_locked_sha256,
                           synthetic=args.synthetic,
                           replace_existing=args.replace_existing_output), indent=2))


if __name__ == "__main__":
    main()
