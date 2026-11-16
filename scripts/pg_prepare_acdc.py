#!/usr/bin/env python3
"""Prepare native PG ACDC inputs from pristine raw data and committed locks.

The command audits pristine original CINE headers under raw/training and
raw/testing, derives native pixdim geometry through the existing Q3D repair
routine, and materializes the exact locked TRAIN80/VAL18/TEST50 manifests and
K8-only training masks. Raw source bytes are never modified. Cine arrays are
normalized with the existing FUSE routine; source and derived lock hashes are
kept separately in the relocation lineage.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.candidate_fuse.data import _CLASSES  # noqa: E402
from self_audit.preparation import normalize_cine  # noqa: E402
from self_audit.candidate_q3d.geometry_repair import (  # noqa: E402
    ACDC_PIXDIM_NATIVE_V1, _patient_geometry, validate_geometry_record,
)

DEFAULT_RAW = Path("/root/dataset/ACDC/raw")
DEFAULT_OUT = Path("/root/dataset/prepared/ACDC/pg-k8-20261116")
DEFAULT_VAL_RECEIPT = ROOT / "reports/fuse-semi-20261113/track-A/PATIENT_METRICS.json"
DEFAULT_TEST_RECEIPT = ROOT / "reports/fuse-semi-20261113/track-A/TEST_ONCE/TEST_TRANSACTION.json"
DEFAULT_L8_RECEIPT = ROOT / "reports/fuse-acdc-k8-aug-20261114/PHASE0_L8.json"
DEFAULT_EXCLUSION_RECEIPT = ROOT / "reports/automatic_pilot_pro4000/FINAL_REPORT.md"
DEFAULT_RAW_INVENTORY = Path("/root/dataset/receipts/ACDC_FILES.json")
DEFAULT_SOURCE_INVENTORY = Path("/root/dataset/receipts/ACDC_SOURCE.json")
DEFAULT_DOWNLOAD_RECEIPT = Path("/root/dataset/receipts/ACDC_DOWNLOAD.json")
EXCLUDED_DUPLICATE_IDS = ["patient055", "patient076"]
K8_SOURCE_SHA256 = "60dd11747570f3ac183e8636050a3ac3d73651cb43b37deba9ff7422b442a433"
L8_IDS = ["patient007", "patient010", "patient011", "patient024", "patient026", "patient041", "patient077", "patient099"]
EXPECTED = {"train": 80, "development": 18, "test": 50}
EXPECTED_MANIFEST_SHA = {
    "train": "ef7562a251e5c30622ab00dfa46a5b62e0c4552cc0aac06b24caa3e63ff23363",
    "development": "e3421779441afcab8f8feadbd293bedad6c00877730eb3c7151fa6a15667e88b",
    "test": "84f24ca7a50e4516548cc16037b6159eaed2eb1ea07d94287e8305894a951297",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path, expected: str | None = None) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"source manifest SHA256 mismatch: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"source manifest must be a JSON object: {path}")
    return value, digest


def _role_ids(manifest: dict[str, Any], role: str) -> list[str]:
    key = "patients" if role != "train" else "unlabeled"
    records = manifest.get(key)
    if role == "train":
        lrec = manifest.get("labeled")
        if not isinstance(lrec, list) or not isinstance(records, list):
            raise ValueError("source TRAIN requires L and U records")
        records = lrec + records
    if not isinstance(records, list):
        raise ValueError(f"source {role} records are malformed")
    ids = [r.get("id") if isinstance(r, dict) else None for r in records]
    if any(not isinstance(pid, str) or not pid for pid in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"source {role} patient IDs must be nonempty and unique")
    return ids


def _split_identity(train: dict[str, Any], val: dict[str, Any], test: dict[str, Any]) -> dict[str, list[str]]:
    split = train.get("split")
    if not isinstance(split, dict):
        raise ValueError("TRAIN source has no authoritative fixed split map")
    try:
        result = {"train": list(split["train"]), "development": list(split["development"]), "test": list(split["test"])}
    except (KeyError, TypeError):
        raise ValueError("TRAIN source split must declare train/development/test IDs") from None
    if {k: len(v) for k, v in result.items()} != EXPECTED or any(len(v) != len(set(v)) for v in result.values()):
        raise ValueError("source split is not exact TRAIN80/VAL18/TEST50")
    if any(set(result[a]) & set(result[b]) for a, b in (("train", "development"), ("train", "test"), ("development", "test"))):
        raise ValueError("TRAIN/VAL/TEST source splits overlap")
    if set(EXCLUDED_DUPLICATE_IDS) & (set(result["train"]) | set(result["development"]) | set(result["test"])):
        raise ValueError("documented duplicate-image patients must remain excluded from all splits")
    l_ids = [r.get("id") for r in train.get("labeled", [])]
    u_ids = [r.get("id") for r in train.get("unlabeled", [])]
    if set(l_ids) != set(L8_IDS) or set(u_ids) != set(result["train"]) or not set(l_ids).issubset(u_ids):
        raise ValueError("source TRAIN L/U records do not preserve exact locked L8 and TRAIN80 identity")
    if set(_role_ids(val, "development")) != set(result["development"]):
        raise ValueError("source VAL records do not match locked VAL split")
    if set(_role_ids(test, "test")) != set(result["test"]):
        raise ValueError("source TEST records do not match locked TEST split")
    return result


def _receipt_split(val_receipt: Path, test_receipt: Path, l8_receipt: Path, training_ids: list[str], testing_ids: list[str], *, excluded_ids: list[str] = EXCLUDED_DUPLICATE_IDS) -> tuple[dict[str, list[str]], dict[str, Any]]:
    val_obj, val_sha = _load(val_receipt)
    test_obj, test_sha = _load(test_receipt)
    l8_obj, l8_sha = _load(l8_receipt)
    try:
        val_ids = sorted(val_obj["splits"]["VAL18"]["models"]["fusion"])
        test_ids = sorted(test_obj["metrics"]["fusion"]["per_patient"])
        l8_ids = sorted(l8_obj["metrics"]["L8_TRAIN_UNAUGMENTED"]["fusion"]["per_patient"])
    except (KeyError, TypeError):
        raise ValueError("committed FUSE receipts do not expose patient identities") from None
    train_ids = sorted(set(training_ids) - set(val_ids) - set(excluded_ids))
    if (len(train_ids), len(val_ids), len(test_ids), len(l8_ids)) != (EXPECTED["train"], EXPECTED["development"], EXPECTED["test"], len(L8_IDS)):
        raise ValueError("receipt and raw folder identities do not establish TRAIN80/VAL18/TEST50/L8")
    if set(training_ids) != set(train_ids) | set(val_ids) | set(excluded_ids):
        raise ValueError("raw ACDC folder patient IDs differ from committed split receipts and documented duplicate exclusions")
    if set(testing_ids) != set(test_ids):
        raise ValueError("raw ACDC test folder patient IDs differ from committed split receipts")
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise ValueError("receipt-derived TRAIN/VAL/TEST patient lists overlap")
    if not set(l8_ids).issubset(train_ids) or l8_ids != L8_IDS:
        raise ValueError("receipt-derived L8 identity differs from the locked K8 patient list")
    return {"train": train_ids, "development": val_ids, "test": test_ids}, {
        "val_receipt": {"path": str(val_receipt), "sha256": val_sha},
        "test_receipt": {"path": str(test_receipt), "sha256": test_sha},
        "l8_receipt": {"path": str(l8_receipt), "sha256": l8_sha},
        "source_k8_sha256": K8_SOURCE_SHA256,
    }


def _raw_dirs(root: Path, folder: str) -> dict[str, Path]:
    path = root / folder
    if not path.is_dir():
        raise FileNotFoundError(f"raw ACDC {folder} directory missing: {path}")
    return {p.name: p for p in path.iterdir() if p.is_dir() and re.fullmatch(r"patient\d{3}", p.name)}


def _verify_raw_inventory(raw_root: Path, files_receipt: Path, source_receipt: Path, download_receipt: Path) -> dict[str, Any]:
    verified_obj = json.loads(files_receipt.read_text(encoding="utf-8"))
    verified_sha = sha256_file(files_receipt)
    source_obj = json.loads(source_receipt.read_text(encoding="utf-8"))
    if not isinstance(verified_obj, list) or not isinstance(source_obj, list) or len(verified_obj) != len(source_obj):
        raise ValueError("ACDC verified/source inventory receipts must be same-length file lists")
    verified = {str(row.get("relative_path")): row for row in verified_obj if isinstance(row, dict)}
    source = {str(row.get("Path")): row for row in source_obj if isinstance(row, dict)}
    if set(verified) != set(source) or len(verified) != len(verified_obj):
        raise ValueError("ACDC verified/source inventories have different or duplicate paths")
    source_sha = sha256_file(source_receipt)
    download_obj, download_sha = _load(download_receipt)
    if download_obj.get("status") != "DOWNLOADED_VERIFIED" or download_obj.get("dataset") != "ACDC":
        raise ValueError("ACDC download receipt is not DOWNLOADED_VERIFIED")
    if download_obj.get("source_inventory_sha256") != source_sha:
        raise ValueError("ACDC download receipt source inventory SHA mismatch")
    if Path(download_obj.get("files_receipt", "")).resolve() != files_receipt.resolve():
        raise ValueError("ACDC download receipt does not bind the verified files inventory")
    raw_files = []
    for folder in ("training", "testing"):
        for file_path in (raw_root / folder).rglob("*"):
            if file_path.is_file() and file_path.name.endswith((".nii", ".nii.gz")):
                raw_files.append(file_path)
    expected_raw = {rel for rel in verified if rel.endswith((".nii", ".nii.gz"))}
    actual_raw = {p.relative_to(raw_root).as_posix(): p for p in raw_files}
    if set(actual_raw) != expected_raw:
        raise ValueError("raw ACDC NIfTI paths differ from the verified Drive inventory")
    for rel, path in actual_raw.items():
        row, src_row = verified[rel], source[rel]
        if src_row.get("Size") != path.stat().st_size:
            raise ValueError(f"raw ACDC file size differs from source inventory: {rel}")
        if "_4d.nii" in path.name:
            actual = sha256_file(path)
            if actual != row.get("sha256") or path.stat().st_size != row.get("bytes"):
                raise ValueError(f"raw ACDC CINE differs from verified download receipt: {rel}")
    return {"files_receipt": {"path": str(files_receipt), "sha256": verified_sha},
            "source_receipt": {"path": str(source_receipt), "sha256": source_sha},
            "download_receipt": {"path": str(download_receipt), "sha256": download_sha},
            "nifti_file_count": len(actual_raw), "nifti_bytes": sum(p.stat().st_size for p in raw_files),
            "_verified_entries": verified}


def _geometry_audit(pid: str, cine_path: Path, header_geometry: dict[str, Any] | None,
                    declarations: dict[str, dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    geometry, row = _patient_geometry(pid, cine_path, header_geometry or {}, declarations, ACDC_PIXDIM_NATIVE_V1)
    if row.get("errors") or not geometry:
        raise ValueError(f"invalid NIfTI geometry for {pid}: {row.get('errors')}")
    geometry = validate_geometry_record(geometry)
    row["effective_spacing_zyx_mm"] = geometry["spacing_zyx_mm"]
    row["effective_affine_zyx"] = geometry["affine_zyx"]
    row["original_affine_xyz"] = row.get("selected_affine_xyz")
    row["effective_affine_xyz"] = row.get("reconstructed_affine_xyz")
    return geometry, row


def _cine_file(folder: Path, pid: str) -> Path:
    matches = sorted(p for p in (folder / f"{pid}_4d.nii", folder / f"{pid}_4d.nii.gz") if p.is_file())
    if len(matches) != 1:
        raise ValueError(f"expected exactly one raw CINE for {pid}; got {len(matches)}")
    return matches[0]


def _phase_info(path: Path, frames: int) -> dict[str, int]:
    from scripts.fuse_build_manifest import _phase_info as parse_info
    return parse_info(path, frames)


def _mask_path(folder: Path, pid: str, frame: int) -> Path:
    stem = folder / f"{pid}_frame{frame + 1:02d}_gt.nii"
    matches = sorted(p for p in (stem, Path(str(stem) + ".gz")) if p.is_file())
    if len(matches) != 1:
        raise ValueError(f"expected one ED/ES mask for {pid} frame {frame}; got {len(matches)}")
    return matches[0]


def prepare(source_train: str | Path | None, source_val: str | Path | None, source_test: str | Path | None,
            raw_root: str | Path, out: str | Path, *,
            locked_k8: str | Path | None = None,
            val_receipt: str | Path = DEFAULT_VAL_RECEIPT, test_receipt: str | Path = DEFAULT_TEST_RECEIPT,
            l8_receipt: str | Path = DEFAULT_L8_RECEIPT, mask_inventory: str | Path | None = ROOT / "reports/fuse-rebuild-20261113/BUILD_RECEIPT.json",
            expected_train_sha: str = EXPECTED_MANIFEST_SHA["train"],
            expected_val_sha: str = EXPECTED_MANIFEST_SHA["development"],
            expected_test_sha: str = EXPECTED_MANIFEST_SHA["test"],
            expected_k8_sha: str = "60dd11747570f3ac183e8636050a3ac3d73651cb43b37deba9ff7422b442a433",
            raw_inventory: str | Path = DEFAULT_RAW_INVENTORY,
            source_inventory: str | Path = DEFAULT_SOURCE_INVENTORY,
            download_receipt: str | Path = DEFAULT_DOWNLOAD_RECEIPT,
            exclusion_receipt: str | Path = DEFAULT_EXCLUSION_RECEIPT,
            unknown_unit_provenance: str = "") -> dict[str, Any]:
    """Create new relocated PG manifests and arrays; all source trees read-only."""
    sources = {role: (Path(value).resolve() if value else None) for role, value in (("train", source_train), ("development", source_val), ("test", source_test))}
    lock_path = Path(locked_k8).resolve() if locked_k8 else None
    raw, output = Path(raw_root).resolve(), Path(out).resolve()
    lineage_paths = [p for p in sources.values() if p is not None] + ([lock_path] if lock_path else []) + [Path(val_receipt).resolve(), Path(test_receipt).resolve(), Path(l8_receipt).resolve(), Path(raw_inventory).resolve(), Path(source_inventory).resolve(), Path(download_receipt).resolve(), Path(exclusion_receipt).resolve()] + ([Path(mask_inventory).resolve()] if mask_inventory else [])
    if output.exists() or any(output == p or output.is_relative_to(p) for p in (raw, *lineage_paths)):
        raise ValueError("output must be a new directory outside source/raw roots")
    manifests, hashes = {}, {}
    for role, path in sources.items():
        if path is None:
            continue
        manifests[role], hashes[role] = _load(path, {"train": expected_train_sha, "development": expected_val_sha, "test": expected_test_sha}[role])
        m = manifests[role]
        if m.get("schema") not in {"self_audit.candidate_fuse.data.v1", "self_audit.candidate_fuse.data.v2"} or m.get("synthetic") is not False or m.get("dataset") != "acdc":
            raise ValueError(f"source {role} must be real ACDC FUSE v1")
        actual_role = role
        if m.get("role") != actual_role:
            raise ValueError(f"source {role} manifest role mismatch")
    locked = None
    original_k8_sha = expected_k8_sha or K8_SOURCE_SHA256
    lock_sha = original_k8_sha
    if lock_path is not None:
        locked, lock_sha = _load(lock_path, expected_k8_sha or K8_SOURCE_SHA256)
        original_k8_sha = lock_sha
        if locked.get("schema") != "self_audit.data.v2" or locked.get("role") != "training" or locked.get("synthetic") is not False or locked.get("label_seed") != 42:
            raise ValueError("source K8 lock has unexpected schema, role, synthetic flag, or seed")
        locked_l = locked.get("labeled")
        lock_ids = sorted(r.get("id") for r in locked_l) if isinstance(locked_l, list) else []
        if lock_ids != sorted(L8_IDS) or locked.get("label_budget") != len(L8_IDS):
            raise ValueError("source K8 lock patient identity differs from committed L8 receipt")
        if any(not isinstance(r.get("masks"), dict) or len(r["masks"]) != 2 for r in locked_l):
            raise ValueError("locked K8 records must carry exactly two ED/ES mask references")
    source_lock_path = lock_path
    raw_training, raw_testing = _raw_dirs(raw, "training"), _raw_dirs(raw, "testing")
    raw_inventory_evidence = _verify_raw_inventory(raw, Path(raw_inventory).resolve(), Path(source_inventory).resolve(), Path(download_receipt).resolve())
    raw_inventory_entries = raw_inventory_evidence.pop("_verified_entries")
    if "train" in manifests:
        split = _split_identity(manifests["train"], manifests["development"], manifests["test"])
        receipt_lineage = {"source_k8_sha256": original_k8_sha}
    else:
        split, receipt_lineage = _receipt_split(Path(val_receipt).resolve(), Path(test_receipt).resolve(), Path(l8_receipt).resolve(), sorted(raw_training), sorted(raw_testing))
        exclusion_text = Path(exclusion_receipt).read_text(encoding="utf-8")
        if any(pid[-3:] not in exclusion_text for pid in EXCLUDED_DUPLICATE_IDS):
            raise ValueError("exclusion receipt does not document both duplicate-image patients")
        receipt_lineage["duplicate_image_exclusion_receipt"] = {"path": str(Path(exclusion_receipt).resolve()), "sha256": sha256_file(Path(exclusion_receipt).resolve())}
        hashes["committed_split_receipts"] = receipt_lineage
    if not set(L8_IDS).issubset(split["train"]):
        raise ValueError("locked L8 IDs are not a subset of TRAIN80")
    if locked is not None and "train" in manifests:
        source_l = {r["id"]: r for r in manifests["train"].get("labeled", [])}
        lock_l = {r["id"]: r for r in locked_l}
        for pid in L8_IDS:
            src_masks = source_l[pid].get("masks", {})
            locked_masks = lock_l[pid].get("masks", {})
            if set(src_masks) != set(locked_masks) or any(src_masks[k].get("source_sha256", src_masks[k].get("sha256")) != locked_masks[k].get("sha256") for k in src_masks):
                raise ValueError(f"source L8 mask identities differ from locked K8 manifest for {pid}")

    if set(split["train"]) - set(raw_training) or set(split["development"]) - set(raw_training) or set(split["test"]) - set(raw_testing):
        raise ValueError("pristine raw ACDC folder IDs do not match locked source splits")

    # Audit every CINE header before creating output. `_patient_geometry` opens
    # NIfTI header metadata only; only L8 training masks are subsequently loaded.
    geom: dict[str, dict[str, Any]] = {}
    audit: dict[str, dict[str, Any]] = {}
    source_records: dict[str, dict[str, Any]] = {}
    train_records = {r["id"]: r for r in manifests.get("train", {}).get("unlabeled", [])}
    train_records.update({r["id"]: r for r in manifests.get("train", {}).get("labeled", [])})
    dev_records = {r["id"]: r for r in manifests.get("development", {}).get("patients", [])}
    test_records = {r["id"]: r for r in manifests.get("test", {}).get("patients", [])}
    declarations = {}
    if unknown_unit_provenance:
        declarations = {pid: {"unit": "mm", "provenance": unknown_unit_provenance} for pid in split["train"] + split["development"] + split["test"]}
    for role, ids, dirs in (("train", split["train"], raw_training), ("development", split["development"], raw_training), ("test", split["test"], raw_testing)):
        for pid in ids:
            cine = _cine_file(dirs[pid], pid)
            cached = train_records.get(pid) if role == "train" else dev_records.get(pid) if role == "development" else test_records.get(pid)
            entry = manifests.get("train", {}).get("geometry", {}).get("by_patient", {}).get(pid) if role == "train" else manifests.get(role, {}).get("geometry", {}).get("by_patient", {}).get(pid)
            if entry is None:
                import nibabel as nib
                hdr = nib.load(str(cine), mmap=False).header
                xyz = tuple(int(v) for v in hdr.get_data_shape()[:3])
                entry = {"shape_zyx": list(reversed(xyz))}
            unit_declarations = declarations
            if not declarations:
                import nibabel as nib
                if nib.load(str(cine), mmap=False).header.get_xyzt_units()[0] in {None, "unknown", ""}:
                    unit_declarations = {pid: {"unit": "mm", "provenance": "User-authorized ACDC 2026-11-12 pixdim amendment: unknown spatial units declared millimetres for acdc_pixdim_native_v1."}}
            header_geom, row = _geometry_audit(pid, cine, entry, unit_declarations)
            geom[pid], audit[pid], source_records[pid] = header_geom, row, cached

    mask_source_hashes = {}
    inventory_sha = None
    if mask_inventory is not None:
        inventory_obj, inventory_sha = _load(Path(mask_inventory).resolve())
        mask_source_hashes = inventory_obj.get("original_mask_sha256", {})
        if not isinstance(mask_source_hashes, dict):
            raise ValueError("committed original-mask inventory is malformed")
    receipt_lineage["original_mask_inventory"] = {"path": str(Path(mask_inventory).resolve()) if mask_inventory else None, "sha256": inventory_sha}
    # All role annotations are enumerated from Info.cfg; payloads are touched
    # only for TRAIN L8 and evaluation preparation for VAL/TEST.
    prepared_frames = {}
    for role, ids, dirs in (("train", split["train"], raw_training), ("development", split["development"], raw_training), ("test", split["test"], raw_testing)):
        for pid in ids:
            source_record = source_records[pid]
            cine = _cine_file(dirs[pid], pid)
            if source_record is not None:
                image_path = Path(source_record["image"])
                src_manifest = sources[role]
                if not image_path.is_absolute(): image_path = src_manifest.parent / image_path
                if sha256_file(image_path) != source_record.get("image_sha256"):
                    raise ValueError(f"prepared source image hash mismatch for {pid}")
                array = np.load(image_path, mmap_mode="r", allow_pickle=False)
            else:
                import nibabel as nib
                nii = nib.load(str(cine), mmap=False)
                array = normalize_cine(np.asarray(nii.dataobj), "XYZT")
            if array.ndim != 4 or tuple(array.shape[1:]) != tuple(geom[pid]["shape_zyx"]):
                raise ValueError(f"prepared/raw image shape/header mismatch for {pid}")
            if role == "train" and pid not in L8_IDS:
                # Do not consult or enumerate any TRAIN non-L mask file names.
                prepared_frames[pid] = {"image": image_path if source_record is not None else None, "array": np.array(array, copy=True) if source_record is None else None, "source_cine": cine, "record": source_record, "phases": None, "masks": {}}
                continue
            if role == "train" and source_record is not None:
                phase_map = source_record.get("phases")
                if not isinstance(phase_map, dict) or set(phase_map) != {"ED", "ES"}:
                    raise ValueError(f"locked L8 source lacks ED/ES for {pid}")
                phases = {name: int(value) for name, value in phase_map.items()}
            else:
                phases = _phase_info(dirs[pid] / "Info.cfg", int(array.shape[0]))
            masks = {}
            if role == "train":
                for frame in phases.values():
                    mp = _mask_path(dirs[pid], pid, frame)
                    # Header and labels validated before writing any payload.
                    import nibabel as nib
                    nii = nib.load(str(mp), mmap=False)
                    labels = np.asarray(nii.dataobj)
                    if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
                        raise ValueError(f"fractional or nonfinite mask labels for {pid} frame {frame}")
                    if labels.ndim != 3 or tuple(labels.shape) != tuple(array.shape[1:][::-1]) or not np.isin(labels, [0, 1, 2, 3]).all():
                        raise ValueError(f"invalid L8 native mask shape/labels for {pid}")
                    expected_raw_sha = next((v for k, v in mask_source_hashes.items() if Path(k).name == mp.name), None)
                    actual_raw_sha = sha256_file(mp)
                    if mask_inventory is not None and mp.name.endswith(".nii") and expected_raw_sha != actual_raw_sha:
                        raise ValueError(f"L8 raw mask bytes differ from committed source inventory for {pid} frame {frame}")
                    raw_rel = mp.relative_to(raw).as_posix()
                    if raw_inventory_entries[raw_rel].get("sha256") != actual_raw_sha:
                        raise ValueError(f"L8 raw mask bytes differ from verified download receipt for {pid} frame {frame}")
                    masks[frame] = (mp, labels.transpose(2, 1, 0).astype(np.uint8, copy=True))
            prepared_frames[pid] = {"image": image_path if source_record is not None else None, "array": np.array(array, copy=True) if source_record is None else None, "source_cine": cine, "record": source_record, "phases": phases, "masks": masks}

    # Validate raw/source dimensions and mask coverage before opening output.
    if set(prepared_frames) != set(sum((split[k] for k in split), [])):
        raise ValueError("prepared patient inventory differs from exact split")
    output.mkdir(parents=True)
    arrays_dir = output / "arrays"
    arrays_dir.mkdir()
    relocated: dict[str, dict[str, Any]] = {}
    for role, ids in split.items():
        for pid in ids:
            item = prepared_frames[pid]
            src = item["image"]
            dst = arrays_dir / f"{pid}_cine.npy"
            if src is not None:
                shutil.copyfile(src, dst)
                if sha256_file(dst) != sha256_file(src):
                    raise IOError(f"byte-preserving cine relocation failed for {pid}")
                source_array_sha = sha256_file(src)
            else:
                np.save(dst, item["array"], allow_pickle=False)
                source_array_sha = sha256_file(item["source_cine"])
            relocated[pid] = {"path": str(dst), "sha256": sha256_file(dst), "source_path": str(src or item["source_cine"]), "source_sha256": source_array_sha, "relocation_mode": "byte_copy" if src is not None else "raw_normalized"}

    output_manifest_paths = {"train": output / "train.json", "development": output / "val.json", "test": output / "test.json"}
    final_hashes = {}
    lock_path = output / "locked_k8_derived.json"
    for role, ids in split.items():
        base = copy.deepcopy(manifests[role]) if role in manifests else {"schema": "self_audit.candidate_fuse.data.v2" if role == "train" else "self_audit.candidate_fuse.data.v1", "role": "train" if role == "train" else role, "synthetic": False, "dataset": "acdc", "class_mapping": _CLASSES, "split": {"train": split["train"], "development": split["development"], "test": split["test"]}}
        geom_map = {}
        records = []
        for pid in ids:
            item = prepared_frames[pid]
            src_record = item["record"]
            geom_map[pid] = {"shape": geom[pid]["shape_zyx"], "spacing_zyx": geom[pid]["spacing_zyx_mm"], "affine_zyx": geom[pid]["affine_zyx"], "context": "full_volume",
                             "provenance": {"source_geometry_audit": audit[pid], "pixdim_policy": ACDC_PIXDIM_NATIVE_V1}}
            ir = {"id": pid, "image": relocated[pid]["path"], "image_sha256": relocated[pid]["sha256"]}
            if role == "train" and pid in L8_IDS:
                phase_map = {k: int(v) for k, v in item["phases"].items()}
                masks = {}
                for frame, (source_mask, mask_array) in item["masks"].items():
                    dst = arrays_dir / f"{pid}_t{frame:03d}_mask.npy"
                    np.save(dst, mask_array, allow_pickle=False)
                    masks[str(frame)] = {"path": str(dst), "sha256": sha256_file(dst), "source_path": str(source_mask), "source_sha256": sha256_file(source_mask), "transpose": [2, 1, 0]}
                records.append({**ir, "phases": phase_map, "masks": masks})
            elif role == "train":
                records.append(ir)
            else:
                phase_map = {k: int(v) for k, v in item["phases"].items()}
                masks = {}
                for frame in phase_map.values():
                    source_mask = _mask_path((_raw_dirs(raw, "training") if role == "development" else raw_testing)[pid], pid, frame)
                    # Evaluation-mask path intentionally handles VAL/TEST legacy arrays.
                    import nibabel as nib
                    labels = np.asarray(nib.load(str(source_mask), mmap=False).dataobj)
                    expected_raw_sha = next((v for k, v in mask_source_hashes.items() if Path(k).name == source_mask.name), None)
                    if mask_inventory is not None and source_mask.name.endswith(".nii") and expected_raw_sha != sha256_file(source_mask):
                        raise ValueError(f"evaluation raw mask bytes differ from committed source inventory for {pid} frame {frame}")
                    raw_rel = source_mask.relative_to(raw).as_posix()
                    if raw_inventory_entries[raw_rel].get("sha256") != sha256_file(source_mask):
                        raise ValueError(f"evaluation raw mask bytes differ from verified download receipt for {pid} frame {frame}")
                    if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
                        raise ValueError(f"fractional or nonfinite evaluation mask labels for {pid}")
                    if labels.ndim != 3 or tuple(labels.transpose(2, 1, 0).shape) != tuple(geom[pid]["shape_zyx"]) or not np.isin(labels, [0, 1, 2, 3]).all():
                        raise ValueError(f"invalid native evaluation mask shape/labels for {pid}")
                    dst = arrays_dir / f"{pid}_t{frame:03d}_mask.npy"
                    np.save(dst, labels.transpose(2, 1, 0).astype(np.uint8, copy=False), allow_pickle=False)
                    masks[str(frame)] = {"path": str(dst), "sha256": sha256_file(dst), "source_path": str(source_mask), "source_sha256": sha256_file(source_mask), "transpose": [2, 1, 0]}
                records.append({**ir, "phases": phase_map, "masks": masks})
        base["geometry"] = {"by_patient": geom_map}
        base["relocation_lineage"] = {"source_manifest": str(sources[role]) if sources[role] else None, "source_manifest_sha256": hashes.get(role), "source_k8_lock": str(source_lock_path) if source_lock_path else None, "source_k8_lock_sha256": original_k8_sha, "historical_source_k8_sha256": original_k8_sha, "derived_k8_lock": str(lock_path), "derived_k8_lock_sha256": lock_sha, "source_images": {pid: {"source_path": relocated[pid]["source_path"], "source_sha256": relocated[pid]["source_sha256"]} for pid in ids}, "committed_receipts": receipt_lineage, "raw_root": str(raw), "arrays_byte_preserved": all(relocated[pid]["relocation_mode"] == "byte_copy" for pid in ids), "array_modes": sorted({relocated[pid]["relocation_mode"] for pid in ids})}
        if role == "train":
            base["labeled"] = [r for r in records if r["id"] in L8_IDS]
            base["unlabeled"] = [{"id": r["id"], "image": r["image"], "image_sha256": r["image_sha256"]} for r in records]
            base["patients"] = ids
            source_labeled_records = []
            for pid in sorted(L8_IDS):
                frame_info = records[[r["id"] for r in records].index(pid)]
                source_labeled_records.append({"id": pid, "image": frame_info["image"], "image_sha256": frame_info["image_sha256"], "masks": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in frame_info["masks"].items()}})
            derived_lock = {"schema": "self_audit.data.v2", "role": "training", "synthetic": False, "dataset": "acdc", "label_seed": 42, "label_budget": 8, "labeled": source_labeled_records, "source_lock_sha256": lock_sha, "source_lock_patient_ids": sorted(L8_IDS)}
            lock_path.write_text(json.dumps(derived_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            lock_sha = sha256_file(lock_path)
            base["relocation_lineage"]["derived_k8_lock_sha256"] = lock_sha
            base["labeled_contract"] = {"patient_ids": sorted(L8_IDS), "patient_ids_sha256": hashlib.sha256(json.dumps(sorted(L8_IDS), separators=(",", ":")).encode()).hexdigest(), "source_manifest": str(lock_path), "source_manifest_sha256": lock_sha}
        else:
            base["patients"] = records
        path = output_manifest_paths[role]
        path.write_text(json.dumps(base, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        final_hashes[role] = sha256_file(path)

    manifest_refs = {"train" if role == "train" else "val" if role == "development" else "test": {"path": str(output_manifest_paths[role]), "sha256": final_hashes[role]} for role in output_manifest_paths}
    artifact_refs = [dict(role=role, path=pin["path"], sha256=pin["sha256"]) for role, pin in manifest_refs.items()]
    artifact_refs.append({"role": "source_lock", "path": str(lock_path), "sha256": lock_sha})
    receipt = {"schema": "self_audit.candidate_pg.materialization.v1", "status": "READY", "historical_source_k8_sha256": original_k8_sha, "split_lock_sha256": "cdec2b85c549a1cf9a19eea8790b7d6dbe155fdbe7a4798b60315412cbead639", "raw_source": str(raw), "raw_inventory": raw_inventory_evidence,
               "output": str(output), "source_manifest_hashes": {"train": hashes.get("train"), "val": hashes.get("development"), "test": hashes.get("test")}, "locked_k8_source": {"path": str(source_lock_path) if source_lock_path else None, "sha256": original_k8_sha, "patient_ids": sorted(L8_IDS)},
               "artifacts": artifact_refs,
               "derived_k8_lock": {"path": str(lock_path), "sha256": lock_sha},
               "historical_source_lock": {"path": str(source_lock_path) if source_lock_path else None, "sha256": original_k8_sha, "patient_ids": sorted(L8_IDS)},
               "source_lock": {"path": str(lock_path), "sha256": lock_sha, "patient_ids": sorted(L8_IDS)},
               "manifests": manifest_refs,
               "split_ids": {"train": split["train"], "val": split["development"], "test": split["test"]}, "labeled_ids": sorted(L8_IDS),
               "evidence_pins": {"val_receipt": receipt_lineage.get("val_receipt"), "test_receipt": receipt_lineage.get("test_receipt"), "l8_receipt": receipt_lineage.get("l8_receipt"), "exclusions": receipt_lineage.get("duplicate_image_exclusion_receipt"), "original_mask_inventory": receipt_lineage.get("original_mask_inventory")},
               "loader_compatibility": {"historical_sha_pins_match": False, "required_action": "write these derived hashes into the explicit PG runtime manifest refs and make open_pg_data validate them plus source lineage; do not bypass hash checks", "historical_source_sha256": EXPECTED_MANIFEST_SHA},
               "counts": {"train": len(split["train"]), "val": len(split["development"]), "test": len(split["test"]), "labeled_train_patients": len(L8_IDS), "labeled_train_masks": 2 * len(L8_IDS), "train_non_l8_mask_reads": 0},
               "normalization": "global cine percentiles 0.5/99.5, clip [0,1], transpose XYZT to TZYX float32; raw NIfTI bytes untouched",
               "geometry_policy": ACDC_PIXDIM_NATIVE_V1, "geometry_audit": audit}
    (output / "PREPARATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", type=Path, help="optional existing hash-pinned FUSE semi-K8 TRAIN manifest")
    parser.add_argument("--source-val", type=Path, help="optional existing hash-pinned FUSE VAL18 manifest")
    parser.add_argument("--source-test", type=Path, help="optional existing hash-pinned FUSE TEST50 manifest")
    parser.add_argument("--locked-k8", type=Path, help="optional original K8 manifest; committed receipt lock is default")
    parser.add_argument("--locked-k8-sha256", default=K8_SOURCE_SHA256, help="expected source K8 manifest SHA256")
    parser.add_argument("--val-receipt", type=Path, default=DEFAULT_VAL_RECEIPT)
    parser.add_argument("--test-receipt", type=Path, default=DEFAULT_TEST_RECEIPT)
    parser.add_argument("--l8-receipt", type=Path, default=DEFAULT_L8_RECEIPT)
    parser.add_argument("--mask-inventory", type=Path, default=ROOT / "reports/fuse-rebuild-20261113/BUILD_RECEIPT.json")
    parser.add_argument("--raw-inventory", type=Path, default=DEFAULT_RAW_INVENTORY)
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--download-receipt", type=Path, default=DEFAULT_DOWNLOAD_RECEIPT)
    parser.add_argument("--exclusion-receipt", type=Path, default=DEFAULT_EXCLUSION_RECEIPT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--unknown-unit-provenance", default="", help="explicit mm declaration provenance if original NIfTI units are unknown")
    args = parser.parse_args(argv)
    sources = (args.source_train, args.source_val, args.source_test)
    if any(sources) and not all(sources):
        parser.error("supply all three source manifests or none")
    receipt = prepare(*sources, args.raw_root, args.out, locked_k8=args.locked_k8,
                      val_receipt=args.val_receipt, test_receipt=args.test_receipt, l8_receipt=args.l8_receipt, mask_inventory=args.mask_inventory,
                      expected_k8_sha=args.locked_k8_sha256, raw_inventory=args.raw_inventory, source_inventory=args.source_inventory,
                      download_receipt=args.download_receipt,
                      exclusion_receipt=args.exclusion_receipt, unknown_unit_provenance=args.unknown_unit_provenance)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
