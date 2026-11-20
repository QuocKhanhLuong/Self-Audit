"""Candidate PG data adapters and patient-balanced resumable sampler."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterator, Mapping

from ..candidate_fuse.data import Frame, FuseData, open_fuse_data
from .config import (
    K8_SOURCE_SHA256,
    SPLIT_LOCK_SHA256,
    TEST_MANIFEST_SHA256,
    TRAIN_MANIFEST_SHA256,
    VAL_MANIFEST_SHA256,
)


FrameKey = tuple[str, int]
_LOCKED_L8 = (
    "patient007", "patient010", "patient011", "patient024",
    "patient026", "patient041", "patient077", "patient099",
)
_MATERIALIZATION_SCHEMA = "self_audit.candidate_pg.materialization.v1"
_SPLIT_EVIDENCE = {
    "reports/fuse-semi-20261113/track-A/PATIENT_METRICS.json": "70f1d9b2ca1c6d93e2ab42ff69344b9721cd4fe89ed4170cf6f72474aa3a2816",
    "reports/fuse-semi-20261113/track-A/TEST_ONCE/TEST_TRANSACTION.json": "0f196a32e8d21bc6541907b954c44a4fc59c2f0395a5b87dd7f32ce1f7b36359",
    "reports/fuse-acdc-k8-aug-20261114/PHASE0_L8.json": "7ba19db0a896ce1169680e62dad1548a970af37fdf5c374cfb23ea40a4087db5",
    "reports/automatic_pilot_pro4000/FINAL_REPORT.md": "5ff62527c51541dc514cdeb5ecc00738c8112bd4fcdf8b419aeaebaa1e304017",
}
_TRAIN_SPLIT_SOURCE_SHA256 = "bc42574017276a6defc668bf533330fd41a64a5e90a75ea1b8958fc9e469265a"
_ORIGINAL_MASK_INVENTORY_SHA256 = "4c69ae6030242f7aeae903bdfaa1a6df6d8cf11d87a3e9fd9dd2978e67b7f3c6"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"PG {label} is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"PG {label} must be a JSON object")
    return dict(value)


def _path_hash_entries(value: Any, base: Path) -> dict[Path, str]:
    """Collect explicit path/SHA pairs from the inventory's documented JSON tree."""
    found: dict[Path, str] = {}
    if isinstance(value, Mapping):
        raw_path = value.get("path")
        digest = value.get("sha256")
        if isinstance(raw_path, str) and isinstance(digest, str):
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            found[candidate.resolve()] = digest
        for child in value.values():
            found.update(_path_hash_entries(child, base))
    elif isinstance(value, list):
        for child in value:
            found.update(_path_hash_entries(child, base))
    return found


def _source_pairs(value: Any):
    if isinstance(value, Mapping):
        if isinstance(value.get("source_path"), str) and isinstance(value.get("source_sha256"), str):
            yield value["source_path"], value["source_sha256"]
        for child in value.values():
            yield from _source_pairs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _source_pairs(child)


def _locked_split_identity() -> dict[str, list[str]]:
    """Reconstruct exact split membership from hash-pinned FUSE/K8 receipts."""
    paths = {rel: _resolve_path(rel) for rel in _SPLIT_EVIDENCE}
    if any(path is None for path in paths.values()):
        raise ValueError("PG committed ACDC split evidence paths are unavailable")
    for rel, path in paths.items():
        if not path.is_file() or _file_sha256(path) != _SPLIT_EVIDENCE[rel]:
            raise ValueError(f"PG committed split evidence changed: {rel}")
    val_receipt = _read_json_object(paths["reports/fuse-semi-20261113/track-A/PATIENT_METRICS.json"], "committed VAL receipt")
    test_receipt = _read_json_object(paths["reports/fuse-semi-20261113/track-A/TEST_ONCE/TEST_TRANSACTION.json"], "committed TEST receipt")
    l8_receipt = _read_json_object(paths["reports/fuse-acdc-k8-aug-20261114/PHASE0_L8.json"], "committed L8 receipt")
    split_source = _read_json_object(_resolve_path("splits/acdc_patient_split_seed42.json"), "committed TRAIN split source")
    if _file_sha256(_resolve_path("splits/acdc_patient_split_seed42.json")) != _TRAIN_SPLIT_SOURCE_SHA256:
        raise ValueError("PG committed TRAIN split source changed")
    report = paths["reports/automatic_pilot_pro4000/FINAL_REPORT.md"].read_text(encoding="utf-8")
    if "Two image-duplicate training IDs (055, 076) were already excluded by the reviewed split." not in report:
        raise ValueError("PG committed report no longer documents the reviewed duplicate exclusions")
    train = split_source.get("train_patients")
    try:
        val = sorted(val_receipt["splits"]["VAL18"]["models"]["fusion"])
        test = sorted(test_receipt["metrics"]["fusion"]["per_patient"])
        l8 = sorted(l8_receipt["metrics"]["L8_TRAIN_UNAUGMENTED"]["fusion"]["per_patient"])
    except (KeyError, TypeError):
        raise ValueError("PG committed receipts do not expose their locked patient IDs") from None
    if not isinstance(train, list) or len(train) != 80 or len(set(train)) != 80:
        raise ValueError("PG committed ACDC TRAIN split source has changed membership")
    if len(val) != 18 or len(set(val)) != 18 or len(test) != 50 or len(set(test)) != 50:
        raise ValueError("PG committed FUSE VAL/TEST receipts have changed patient membership")
    if l8 != list(_LOCKED_L8):
        raise ValueError("PG committed K8 receipt has changed L8 membership")
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("PG committed ACDC TRAIN/VAL/TEST memberships overlap")
    if not set(_LOCKED_L8).issubset(train) or {"patient055", "patient076"} & set(val):
        raise ValueError("PG committed split differs from reviewed L8 or duplicate exclusions")
    return {
        "train": sorted(train),
        "development": val,
        "test": test,
    }


def _validate_materialization(config: Mapping[str, Any], role: str, path: Path, manifest_sha: str) -> None:
    """Verify a config-pinned, real-only relocated manifest receipt and identity."""
    data_cfg = config["data"]
    materialization = data_cfg.get("materialization")
    if materialization is None:
        return
    if bool(config.get("runtime", {}).get("synthetic", False)):
        raise ValueError("PG materialization lineage cannot be used for synthetic data")
    receipt_path = _resolve_path(materialization["receipt"])
    if receipt_path is None or not receipt_path.is_file():
        raise FileNotFoundError("PG materialization receipt does not exist")
    if _file_sha256(receipt_path) != materialization["sha256"]:
        raise ValueError("PG materialization receipt SHA256 mismatch")
    receipt = _read_json_object(receipt_path, "materialization receipt")
    if (receipt.get("schema") != _MATERIALIZATION_SCHEMA
            or receipt.get("status") != "READY"
            or receipt.get("historical_source_k8_sha256") != K8_SOURCE_SHA256
            or receipt.get("split_lock_sha256") != SPLIT_LOCK_SHA256):
        raise ValueError("PG materialization receipt is not READY or does not descend from locked K8/split contracts")
    expected_normalization = "global cine percentiles 0.5/99.5, clip [0,1], transpose XYZT to TZYX float32; raw NIfTI bytes untouched"
    if receipt.get("normalization") != expected_normalization or receipt.get("geometry_policy") != "acdc_pixdim_native_v1":
        raise ValueError("PG materialization normalization or native geometry policy changed")
    for key in ("locked_k8_source", "historical_source_lock"):
        historical_lock = receipt.get(key)
        if not isinstance(historical_lock, Mapping) or historical_lock.get("sha256") != K8_SOURCE_SHA256:
            raise ValueError("PG materialization receipt lost the historical K8 source lock identity")
    raw_inventory_ref = receipt.get("raw_inventory")
    if not isinstance(raw_inventory_ref, Mapping):
        raise ValueError("PG materialization receipt must pin the raw byte inventory")
    raw_root = _resolve_path(receipt.get("raw_source"))
    if raw_root is None or not raw_root.is_dir():
        raise FileNotFoundError("PG raw ACDC source root does not exist")
    raw_file_ref = raw_inventory_ref.get("files_receipt")
    raw_source_ref = raw_inventory_ref.get("source_receipt")
    raw_download_ref = raw_inventory_ref.get("download_receipt")
    for label, ref in (("verified raw inventory", raw_file_ref), ("source raw inventory", raw_source_ref),
                       ("download receipt", raw_download_ref)):
        if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
            raise ValueError(f"PG materialization receipt has no {label} hash pin")
        ref_path = _resolve_path(ref.get("path"))
        if ref_path is None or not ref_path.is_file() or _file_sha256(ref_path) != ref.get("sha256"):
            raise ValueError(f"PG {label} is missing or changed")
    raw_file_rows = json.loads(_resolve_path(raw_file_ref["path"]).read_text(encoding="utf-8"))
    raw_source_rows = json.loads(_resolve_path(raw_source_ref["path"]).read_text(encoding="utf-8"))
    raw_download = _read_json_object(_resolve_path(raw_download_ref["path"]), "ACDC verified download receipt")
    if raw_download.get("status") != "DOWNLOADED_VERIFIED" or raw_download.get("dataset") != "ACDC":
        raise ValueError("PG raw ACDC download receipt is not verified")
    if (raw_download.get("source_inventory_sha256") != raw_source_ref["sha256"]
            or _resolve_path(raw_download.get("files_receipt")) != _resolve_path(raw_file_ref["path"])):
        raise ValueError("PG raw download receipt does not bind its source and file inventories")
    if not isinstance(raw_file_rows, list) or not isinstance(raw_source_rows, list) or len(raw_file_rows) != len(raw_source_rows):
        raise ValueError("PG raw inventory receipts must contain file lists")
    inventoried_raw: dict[Path, str] = {}
    source_rows_by_path: dict[str, Mapping[str, Any]] = {}
    for row in raw_source_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("Path"), str):
            raise ValueError("PG source raw inventory row is malformed")
        if row["Path"] in source_rows_by_path:
            raise ValueError("PG source raw inventory contains duplicate paths")
        source_rows_by_path[row["Path"]] = row
    for row in raw_file_rows:
        if (not isinstance(row, Mapping) or not isinstance(row.get("relative_path"), str)
                or not isinstance(row.get("sha256"), str) or not isinstance(row.get("md5"), str)):
            raise ValueError("PG verified raw inventory row is malformed")
        source_row = source_rows_by_path.get(row["relative_path"])
        source_hashes = source_row.get("Hashes", {}) if isinstance(source_row, Mapping) else {}
        if (source_row is None or source_row.get("Size") != row.get("bytes")
                or source_hashes.get("md5", "").lower() != row["md5"].lower()
                or (source_hashes.get("sha256") and source_hashes["sha256"].lower() != row["sha256"].lower())):
            raise ValueError("PG verified raw file row differs from original Drive inventory")
        inventoried_raw[(raw_root / row["relative_path"]).resolve()] = row["sha256"]
    if set(source_rows_by_path) != {Path(path).relative_to(raw_root).as_posix() for path in inventoried_raw}:
        raise ValueError("PG source and verified raw inventory paths differ")

    evidence_pins = receipt.get("evidence_pins")
    if not isinstance(evidence_pins, Mapping):
        raise ValueError("PG materialization receipt must pin committed split evidence")
    expected_evidence_keys = {"val_receipt", "test_receipt", "l8_receipt", "exclusions", "original_mask_inventory"}
    if not expected_evidence_keys.issubset(evidence_pins):
        raise ValueError("PG materialization receipt omits committed split evidence")
    actual_evidence: dict[str, str] = {}
    for key in expected_evidence_keys - {"original_mask_inventory"}:
        evidence = evidence_pins[key]
        if not isinstance(evidence, Mapping) or not {"path", "sha256"}.issubset(evidence):
            raise ValueError("PG split evidence reference is malformed")
        evidence_path = _resolve_path(evidence.get("path"))
        if evidence_path is None or not evidence_path.is_file() or _file_sha256(evidence_path) != evidence.get("sha256"):
            raise ValueError("PG committed split evidence is missing or changed")
        try:
            relative = evidence_path.relative_to(_repo_root()).as_posix()
        except ValueError:
            raise ValueError("PG split evidence must reference committed repository files") from None
        actual_evidence[relative] = evidence["sha256"]
    expected_evidence = {
        "reports/fuse-semi-20261113/track-A/PATIENT_METRICS.json": _SPLIT_EVIDENCE["reports/fuse-semi-20261113/track-A/PATIENT_METRICS.json"],
        "reports/fuse-semi-20261113/track-A/TEST_ONCE/TEST_TRANSACTION.json": _SPLIT_EVIDENCE["reports/fuse-semi-20261113/track-A/TEST_ONCE/TEST_TRANSACTION.json"],
        "reports/fuse-acdc-k8-aug-20261114/PHASE0_L8.json": _SPLIT_EVIDENCE["reports/fuse-acdc-k8-aug-20261114/PHASE0_L8.json"],
        "reports/automatic_pilot_pro4000/FINAL_REPORT.md": _SPLIT_EVIDENCE["reports/automatic_pilot_pro4000/FINAL_REPORT.md"],
    }
    if actual_evidence != expected_evidence:
        raise ValueError("PG materialization receipt cites different committed split evidence")
    mask_inventory_ref = evidence_pins["original_mask_inventory"]
    if not isinstance(mask_inventory_ref, Mapping) or not {"path", "sha256"}.issubset(mask_inventory_ref):
        raise ValueError("PG receipt does not pin the committed original mask inventory")
    mask_inventory_path = _resolve_path(mask_inventory_ref.get("path"))
    if (mask_inventory_path is None or not mask_inventory_path.is_file()
            or mask_inventory_ref.get("sha256") != _ORIGINAL_MASK_INVENTORY_SHA256
            or _file_sha256(mask_inventory_path) != _ORIGINAL_MASK_INVENTORY_SHA256):
        raise ValueError("PG committed original mask inventory is missing or changed")
    original_mask_inventory = _read_json_object(mask_inventory_path, "original ACDC mask inventory")
    original_mask_hashes = original_mask_inventory.get("original_mask_sha256")
    if not isinstance(original_mask_hashes, Mapping):
        raise ValueError("PG committed original mask inventory has no source mask identities")
    original_mask_by_name = {Path(name).name: digest for name, digest in original_mask_hashes.items()}

    artifact_refs = receipt.get("artifacts")
    if not isinstance(artifact_refs, (list, Mapping)) or not artifact_refs:
        raise ValueError("PG materialization receipt must pin produced artifact hashes")
    artifact_entries = _path_hash_entries(artifact_refs, receipt_path.parent)
    if not artifact_entries:
        raise ValueError("PG materialization artifact list has no path/SHA256 entries")
    for artifact_path, digest in artifact_entries.items():
        if not artifact_path.is_file() or _file_sha256(artifact_path) != digest:
            raise ValueError(f"PG materialization artifact is missing or changed: {artifact_path}")

    split = receipt.get("split_ids")
    if not isinstance(split, Mapping) or set(split) != {"train", "development", "test"}:
        if isinstance(split, Mapping) and set(split) == {"train", "val", "test"}:
            split = {"train": split["train"], "development": split["val"], "test": split["test"]}
        else:
            raise ValueError("PG materialization receipt must declare the complete fixed split")
    expected_sizes = {"train": 80, "development": 18, "test": 50}
    normalized_split: dict[str, list[str]] = {}
    for split_role, size in expected_sizes.items():
        ids = split[split_role]
        if (not isinstance(ids, list) or len(ids) != size
                or any(not isinstance(pid, str) or not pid for pid in ids)
                or len(ids) != len(set(ids))):
            raise ValueError(f"PG materialization {split_role} split identity is invalid")
        normalized_split[split_role] = ids
    if normalized_split != _locked_split_identity():
        raise ValueError("PG materialization split differs from the committed fixed TRAIN80/VAL18/TEST50 IDs")
    split_sets = [set(normalized_split[name]) for name in expected_sizes]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("PG materialization split memberships overlap")
    labeled_ids = receipt.get("labeled_ids")
    if not isinstance(labeled_ids, list) or labeled_ids != list(_LOCKED_L8) or not set(labeled_ids).issubset(split_sets[0]):
        raise ValueError("PG materialization L8 identities differ from the locked Track-A L8")

    source_lock_ref = receipt.get("source_lock")
    if not isinstance(source_lock_ref, Mapping) or not {"path", "sha256"}.issubset(source_lock_ref):
        raise ValueError("PG materialization receipt has no exact rebuilt source lock reference")
    source_lock_path = _resolve_path(source_lock_ref.get("path"))
    if source_lock_path is None or not source_lock_path.is_file():
        raise FileNotFoundError("PG rebuilt source K8 lock does not exist")
    source_lock_sha = source_lock_ref.get("sha256")
    if not isinstance(source_lock_sha, str) or _file_sha256(source_lock_path) != source_lock_sha:
        raise ValueError("PG rebuilt source K8 lock SHA256 mismatch")
    lock = _read_json_object(source_lock_path, "rebuilt K8 source lock")
    if (lock.get("schema") != "self_audit.data.v2" or lock.get("role") != "training"
            or lock.get("synthetic") is not False or lock.get("dataset") != "acdc"
            or lock.get("label_seed") != 42 or lock.get("source_lock_sha256") != K8_SOURCE_SHA256
            or lock.get("source_lock_patient_ids") != list(_LOCKED_L8)):
        raise ValueError("PG rebuilt K8 lock does not preserve the historical source and seed contract")
    derived_lock_ref = receipt.get("derived_k8_lock")
    if (not isinstance(derived_lock_ref, Mapping)
            or _resolve_path(derived_lock_ref.get("path")) != source_lock_path
            or derived_lock_ref.get("sha256") != source_lock_sha):
        raise ValueError("PG materialization source lock and derived K8 lock references disagree")
    lock_labeled = lock.get("labeled")
    if (not isinstance(lock_labeled, list) or len(lock_labeled) != 8
            or sorted(r.get("id") for r in lock_labeled if isinstance(r, Mapping)) != list(_LOCKED_L8)
            or lock.get("label_budget") != 8):
        raise ValueError("PG rebuilt source K8 lock does not contain exact L8 identities")
    lock_masks: dict[str, dict[str, str]] = {}
    for record in lock_labeled:
        masks = record.get("masks")
        if not isinstance(masks, Mapping) or len(masks) != 2:
            raise ValueError("PG rebuilt source K8 lock must contain exactly two L8 masks per patient")
        lock_masks[record["id"]] = {
            str(frame): str(mask.get("sha256"))
            for frame, mask in masks.items() if isinstance(mask, Mapping)
        }
        if len(lock_masks[record["id"]]) != 2:
            raise ValueError("PG rebuilt source K8 lock contains a malformed L8 mask record")

    manifest_refs = receipt.get("manifests")
    if not isinstance(manifest_refs, Mapping) or set(manifest_refs) != {"train", "val", "test"}:
        raise ValueError("PG materialization receipt must pin all three relocated manifests")
    cfg_role = {"train": "train", "val": "val", "test": "test"}[role]
    role_ref = manifest_refs[cfg_role]
    if not isinstance(role_ref, Mapping) or set(role_ref) != {"path", "sha256"}:
        raise ValueError(f"PG materialization {cfg_role} manifest pin is malformed")
    configured_path, configured_sha = _manifest_ref(config, cfg_role)
    receipt_manifest_path = _resolve_path(role_ref.get("path"))
    if (receipt_manifest_path != path or configured_path is None or configured_sha != role_ref.get("sha256")
            or _resolve_path(configured_path) != path or role_ref.get("sha256") != manifest_sha):
        raise ValueError(f"PG {cfg_role} manifest is not the exact receipt-pinned materialization")

    manifest = _read_json_object(path, f"{cfg_role} manifest")
    if manifest.get("split") != normalized_split:
        raise ValueError(f"PG {cfg_role} split differs from the receipt-pinned fixed split")
    geometry_audit = receipt.get("geometry_audit")
    geometry_map = manifest.get("geometry", {}).get("by_patient") if isinstance(manifest.get("geometry"), Mapping) else None
    if not isinstance(geometry_audit, Mapping) or not isinstance(geometry_map, Mapping):
        raise ValueError(f"PG {cfg_role} manifest or receipt has no native geometry audit")
    role_split_name = {"train": "train", "val": "development", "test": "test"}[role]
    role_ids = set(normalized_split[role_split_name])
    if set(geometry_map) != role_ids:
        raise ValueError(f"PG {cfg_role} native geometry inventory differs from its locked split")
    for pid, geometry in geometry_map.items():
        provenance = geometry.get("provenance") if isinstance(geometry, Mapping) else None
        if (not isinstance(provenance, Mapping)
                or provenance.get("pixdim_policy") != receipt.get("geometry_policy")
                or provenance.get("source_geometry_audit") != geometry_audit.get(pid)):
            raise ValueError(f"PG {cfg_role} native geometry differs from its hash-pinned raw-header audit for {pid}")
    record_list = manifest.get("labeled", []) if role == "train" else manifest.get("patients", [])
    mask_records = [mask for record in record_list if isinstance(record, Mapping)
                    for mask in (record.get("masks", {}) or {}).values()]
    for mask in mask_records:
        if not isinstance(mask, Mapping) or not isinstance(mask.get("source_path"), str) or not isinstance(mask.get("source_sha256"), str):
            raise ValueError(f"PG {cfg_role} manifest contains a mask without raw source identity")
        if original_mask_by_name.get(Path(mask["source_path"]).name) != mask["source_sha256"]:
            raise ValueError(f"PG {cfg_role} raw mask differs from the committed original mask inventory")
    if role == "train":
        ids = manifest.get("labeled_contract", {}).get("patient_ids") if isinstance(manifest.get("labeled_contract"), Mapping) else None
        actual_ids = sorted(r.get("id") for r in manifest.get("labeled", []) if isinstance(r, Mapping))
        if ids != list(_LOCKED_L8) or actual_ids != list(_LOCKED_L8):
            raise ValueError("PG relocated TRAIN manifest does not preserve exact locked L8 identities")
        if manifest.get("labeled_contract", {}).get("source_manifest_sha256") != source_lock_sha:
            raise ValueError("PG relocated TRAIN L8 contract does not pin the rebuilt K8 source lock")
        for record in manifest["labeled"]:
            expected_masks = lock_masks.get(record["id"], {})
            masks = record.get("masks", {})
            if (set(masks) != set(expected_masks)
            or any(masks[frame].get("sha256") != expected_masks[frame] for frame in expected_masks)):
                raise ValueError(f"PG relocated L8 mask bytes differ from rebuilt K8 source lock for {record['id']}")

    lineage = manifest.get("relocation_lineage")
    if not isinstance(lineage, Mapping) or lineage.get("derived_k8_lock_sha256") != source_lock_sha:
        raise ValueError(f"PG {cfg_role} manifest has no checked derived K8 relocation lineage")
    if lineage.get("source_k8_lock_sha256") != K8_SOURCE_SHA256:
        raise ValueError(f"PG {cfg_role} manifest has a different historical K8 source identity")
    committed = lineage.get("committed_receipts")
    evidence = receipt.get("evidence_pins", {})
    lineage_evidence_matches = (
        isinstance(committed, Mapping)
        and committed.get("source_k8_sha256") == K8_SOURCE_SHA256
        and committed.get("val_receipt") == evidence.get("val_receipt")
        and committed.get("test_receipt") == evidence.get("test_receipt")
        and committed.get("l8_receipt") == evidence.get("l8_receipt")
        and committed.get("duplicate_image_exclusion_receipt") == evidence.get("exclusions")
        and committed.get("original_mask_inventory") == evidence.get("original_mask_inventory")
    )
    if (lineage.get("historical_source_k8_sha256") != K8_SOURCE_SHA256
            or lineage.get("raw_root") != str(raw_root)
            or not lineage_evidence_matches):
        raise ValueError(f"PG {cfg_role} manifest relocation lineage does not match the receipt")
    if lineage.get("source_manifest_sha256") not in (None, receipt.get("source_manifest_hashes", {}).get(cfg_role)):
        raise ValueError(f"PG {cfg_role} manifest parent source hash differs from receipt")
    source_images = lineage.get("source_images")
    if not isinstance(source_images, Mapping) or set(source_images) != role_ids:
        raise ValueError(f"PG {cfg_role} manifest must pin raw CINE identities for every patient")
    source_pairs = list(_source_pairs(manifest)) + list(_source_pairs(source_images))
    if not source_pairs:
        raise ValueError(f"PG {cfg_role} manifest has no source-byte identities")
    for raw_path, raw_sha in source_pairs:
        resolved_raw = _resolve_path(raw_path)
        if resolved_raw is None or inventoried_raw.get(resolved_raw) != raw_sha:
            raise ValueError(f"PG {cfg_role} source bytes are not recorded in the pinned raw inventory")
        if (not resolved_raw.is_relative_to(raw_root) or resolved_raw.is_symlink()
                or not resolved_raw.is_file() or _file_sha256(resolved_raw) != raw_sha):
            raise ValueError(f"PG {cfg_role} raw source bytes changed: {resolved_raw}")
        relative_raw = resolved_raw.relative_to(raw_root).as_posix()
        drive_hashes = source_rows_by_path[relative_raw].get("Hashes", {})
        if _file_md5(resolved_raw).lower() != drive_hashes.get("md5", "").lower():
            raise ValueError(f"PG {cfg_role} raw source bytes differ from Drive MD5: {resolved_raw}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_ref(config: Mapping[str, Any], role: str) -> tuple[str | None, str | None]:
    data = config.get("data", {})
    refs = data.get("manifests") if isinstance(data, Mapping) else None
    if isinstance(refs, Mapping) and role in refs:
        value = refs[role]
        if isinstance(value, Mapping):
            return value.get("path"), value.get("sha256")
        if isinstance(value, (str, Path)):
            return str(value), None
    # Compatibility with the FUSE-style six manifest fields.  This adapter is
    # intentionally read-only and never invents a path or SHA.
    aliases = {
        "train": ("train_manifest", "train_manifest_sha256"),
        "val": ("val_manifest", "val_manifest_sha256"),
        "test": ("test_manifest", "test_manifest_sha256"),
    }
    path_key, sha_key = aliases[role]
    return data.get(path_key), data.get(sha_key)


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path).expanduser()
    return value if value.is_absolute() else (_repo_root() / value).resolve()


@dataclass(frozen=True)
class PGData:
    """Three role-specific FUSE views used by the PG engine."""

    train: FuseData
    val: FuseData
    test: FuseData

    @property
    def development(self) -> FuseData:
        return self.val

    def __getitem__(self, role: str) -> FuseData:
        if role in {"val", "development"}:
            return self.val
        if role in {"train", "test"}:
            return getattr(self, role)
        raise KeyError(role)

    def __iter__(self) -> Iterator[str]:
        return iter(("train", "val", "test"))

    def keys(self) -> tuple[str, str, str]:
        return ("train", "val", "test")


def open_pg_data(config: Mapping[str, Any]) -> PGData:
    """Open the exact Track-A K8 manifests through the FUSE loader.

    The TRAIN manifest is the semi-K8 v2 contract, whose U records contain no
    mask references.  This function never calls ``load`` while opening data;
    therefore a TRAIN U frame (and in particular a non-L mask) cannot be read
    as a side effect of constructing the PG bundle.
    """

    if not isinstance(config, Mapping):
        raise TypeError("PG config must be a mapping")
    data_cfg = config.get("data")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("PG config is missing data")
    if data_cfg.get("ssl4mis"):  # SSL4MIS-split campaign: L7 / U63 (ED+ES) / val10 / test20
        from .ssl4mis_data import open_ssl4mis_data

        return open_ssl4mis_data(config)
    if data_cfg.get("split_lock") != "configs/candidate_fuse/semi_k8.json":
        raise ValueError("PG must reuse the exact FUSE semi-K8 split lock")
    if data_cfg.get("split_lock_sha256") not in {None, SPLIT_LOCK_SHA256}:
        raise ValueError("PG split lock SHA256 does not match configs/candidate_fuse/semi_k8.json")
    if data_cfg.get("source_k8_sha256") not in {None, K8_SOURCE_SHA256}:
        raise ValueError("PG source K8 SHA256 does not match the reviewed lock")
    if data_cfg.get("u_partner_policy") != "different_patient" or data_cfg.get("u_sampling") != "patient_balanced":
        raise ValueError("PG requires patient-balanced different-patient U sampling")
    if data_cfg.get("native_geometry") is not True:
        raise ValueError("PG requires native geometry")

    runtime = config.get("runtime", {})
    synthetic = bool(runtime.get("synthetic", False)) if isinstance(runtime, Mapping) else False
    expected_real_sha = {
        "train": TRAIN_MANIFEST_SHA256,
        "val": VAL_MANIFEST_SHA256,
        "test": TEST_MANIFEST_SHA256,
    }
    views: dict[str, FuseData] = {}
    for role, fuse_role in (("train", "train"), ("val", "development"), ("test", "test")):
        path_value, sha_value = _manifest_ref(config, role)
        path = _resolve_path(path_value)
        if path is None or sha_value is None:
            raise ValueError(f"PG {role} manifest path and SHA256 are required")
        materialized_real = not synthetic and data_cfg.get("materialization") is not None
        if not synthetic and not materialized_real and sha_value != expected_real_sha[role]:
            raise ValueError(f"PG {role} manifest SHA256 is not the locked Track-A receipt")
        if not path.is_file():
            raise FileNotFoundError(f"PG {role} manifest does not exist: {path}")
        if materialized_real:
            _validate_materialization(config, role, path, sha_value)
        views[role] = open_fuse_data(path, sha_value, role=fuse_role, synthetic=synthetic)

    train = views["train"]
    if not isinstance(train, FuseData):
        raise TypeError("PG requires the ACDC FuseData TRAIN adapter")
    # Real runs must expose the reviewed K8 budget.  Tiny synthetic fixtures
    # are allowed to use fewer patients for software-only tests.
    if not synthetic and len(getattr(train, "_labeled_ids", ())) != 8:
        raise ValueError("real PG TRAIN manifest must contain exactly locked L8")
    if set(train.labeled_keys()) & set(train.unlabeled_keys()):
        raise ValueError("PG L/U frame identity overlap")
    return PGData(train=train, val=views["val"], test=views["test"])


class PGSampler:
    """Checkpointable one-L/two-U, patient-balanced sampler.

    Each epoch contains one update per U patient.  The first U patient is a
    shuffled permutation and the second is the same permutation cyclically
    shifted by one, guaranteeing different patients while using every patient
    exactly once as an anchor.  Frame choice within each patient is random but
    checkpointed.  L frames are independently shuffled and cycled as needed.
    """

    SCHEMA = "self_audit.candidate_pg.sampler.v1"

    def __init__(self, data: PGData | FuseData, *, seed: int = 42, steps_per_epoch: int | None = None):
        train = data.train if isinstance(data, PGData) else data
        if not isinstance(train, FuseData) or train.role != "train":
            raise ValueError("PGSampler requires TRAIN FuseData or PGData")
        if type(seed) is not int:
            raise TypeError("PGSampler seed must be an integer")
        self.data = train
        self.seed = seed
        self.py_rng = random.Random(seed)
        self.epoch_rng = random.Random(seed ^ 0x5047_5352)
        self.lkeys = tuple(train.labeled_keys())
        self.ukeys = tuple(train.unlabeled_keys())
        self._u_by_patient: dict[str, tuple[FrameKey, ...]] = {}
        for key in self.ukeys:
            self._u_by_patient.setdefault(key[0], tuple())
        for pid in tuple(self._u_by_patient):
            self._u_by_patient[pid] = tuple(key for key in self.ukeys if key[0] == pid)
        self._patients = tuple(sorted(self._u_by_patient))
        if len(self._patients) < 2:
            raise ValueError("PGSampler needs at least two U patients for different-patient pairs")
        if any(not self._u_by_patient[pid] for pid in self._patients):
            raise ValueError("every U patient must have at least one image-only frame")
        self.steps_per_epoch = int(steps_per_epoch or len(self._patients))
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.epoch = 0
        self.update = 0
        self._l_stream: list[FrameKey] = []
        self._l_cursor = 0
        self._anchor_patients: list[str] = []
        self._partner_patients: list[str] = []
        self._u_cursor = 0
        self._new_epoch()

    @property
    def joint_epoch_size(self) -> int:
        return self.steps_per_epoch

    @property
    def epoch_size(self) -> int:
        return self.steps_per_epoch

    def _new_epoch(self) -> None:
        self._l_stream = list(self.lkeys)
        self.py_rng.shuffle(self._l_stream)
        self._l_cursor = 0
        self._anchor_patients = list(self._patients)
        self.epoch_rng.shuffle(self._anchor_patients)
        # A one-place cyclic shift is a deterministic derangement for N >= 2.
        self._partner_patients = self._anchor_patients[1:] + self._anchor_patients[:1]
        self._u_cursor = 0

    def _next_labeled(self) -> FrameKey:
        if not self._l_stream or self._l_cursor >= len(self._l_stream):
            self._l_stream = list(self.lkeys)
            self.py_rng.shuffle(self._l_stream)
            self._l_cursor = 0
        key = self._l_stream[self._l_cursor]
        self._l_cursor += 1
        return key

    def _sample_frame(self, pid: str) -> FrameKey:
        frames = self._u_by_patient[pid]
        return frames[self.epoch_rng.randrange(len(frames))]

    def next_step(self) -> tuple[FrameKey, FrameKey, FrameKey]:
        if self._u_cursor >= self.steps_per_epoch:
            self.epoch += 1
            self._new_epoch()
        anchor = self._anchor_patients[self._u_cursor % len(self._anchor_patients)]
        partner = self._partner_patients[self._u_cursor % len(self._partner_patients)]
        if anchor == partner:  # defensive if a future policy changes the shift
            raise RuntimeError("PG sampler produced same-patient U pair")
        sample = (self._next_labeled(), self._sample_frame(anchor), self._sample_frame(partner))
        self._u_cursor += 1
        self.update += 1
        return sample

    def next_pair(self) -> tuple[FrameKey, FrameKey, FrameKey]:
        """Alias retained for engine code that calls the FUSE sampler API."""

        return self.next_step()

    def next_batch(self) -> tuple[FrameKey, FrameKey, FrameKey]:
        return self.next_step()

    def next(self) -> tuple[FrameKey, FrameKey, FrameKey]:
        return self.next_step()

    def __next__(self) -> tuple[FrameKey, FrameKey, FrameKey]:
        return self.next_step()

    def __iter__(self) -> "PGSampler":
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "seed": self.seed,
            "epoch": self.epoch,
            "update": self.update,
            "py_rng": self.py_rng.getstate(),
            "epoch_rng": self.epoch_rng.getstate(),
            "l_stream": list(self._l_stream),
            "l_cursor": self._l_cursor,
            "anchor_patients": list(self._anchor_patients),
            "partner_patients": list(self._partner_patients),
            "u_cursor": self._u_cursor,
            "steps_per_epoch": self.steps_per_epoch,
        }

    @staticmethod
    def _tuples(values: Any, label: str) -> list[tuple[str, int]]:
        if not isinstance(values, list):
            raise ValueError(f"sampler {label} must be a list")
        out: list[tuple[str, int]] = []
        for value in values:
            if not isinstance(value, (list, tuple)) or len(value) != 2 or not isinstance(value[0], str) or type(value[1]) is not int:
                raise ValueError(f"sampler {label} contains an invalid frame key")
            out.append((value[0], value[1]))
        return out

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != self.SCHEMA:
            raise ValueError("invalid PG sampler checkpoint schema")
        if int(state.get("seed")) != self.seed or int(state.get("steps_per_epoch")) != self.steps_per_epoch:
            raise ValueError("PG sampler checkpoint seed/epoch size mismatch")
        l_stream = self._tuples(state.get("l_stream"), "l_stream")
        if len(l_stream) != len(self.lkeys) or set(l_stream) != set(self.lkeys) or len(set(l_stream)) != len(l_stream):
            raise ValueError("PG sampler checkpoint labeled permutation differs from dataset")
        anchors = list(state.get("anchor_patients", [])); partners = list(state.get("partner_patients", []))
        if anchors != list(self._patients) and set(anchors) != set(self._patients):
            raise ValueError("PG sampler checkpoint anchor patients differ from dataset")
        if len(anchors) != len(self._patients) or len(set(anchors)) != len(anchors) or set(anchors) != set(self._patients):
            raise ValueError("PG sampler checkpoint anchor permutation is invalid")
        if len(partners) != len(anchors) or len(set(partners)) != len(partners) or set(partners) != set(self._patients):
            raise ValueError("PG sampler checkpoint partner permutation is invalid")
        if any(a == b for a, b in zip(anchors, partners)):
            raise ValueError("PG sampler checkpoint contains same-patient U pair")
        l_cursor = int(state.get("l_cursor")); u_cursor = int(state.get("u_cursor"))
        if not 0 <= l_cursor <= len(l_stream) or not 0 <= u_cursor <= self.steps_per_epoch:
            raise ValueError("PG sampler checkpoint cursor is invalid")
        self.py_rng.setstate(state["py_rng"])
        self.epoch_rng.setstate(state["epoch_rng"])
        self.epoch = int(state.get("epoch", 0)); self.update = int(state.get("update", 0))
        self._l_stream = l_stream; self._l_cursor = l_cursor
        self._anchor_patients = anchors; self._partner_patients = partners; self._u_cursor = u_cursor


__all__ = ["Frame", "FrameKey", "FuseData", "PGData", "PGSampler", "open_pg_data"]
