#!/usr/bin/env python3
"""Create metadata-only, role-frozen native M&Ms v3 manifests.

This command reads adjudication/L8 receipts and source container inventories.
It does not invoke nibabel or decode voxel payloads, including TEST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from self_audit.candidate_fuse.data import _sha

SCHEMA = "self_audit.candidate_fuse.data.v3"
CLASS_MAPPING = {"0": 0, "1": 3, "2": 2, "3": 1}
_CONTAINER_SHA: dict[Path, str] = {}


def _source(path: Path, member: str | None, *, expected_container_sha: str | None = None):
    path = path.resolve(strict=True)
    container_sha = expected_container_sha or _CONTAINER_SHA.get(path)
    if container_sha is None:
        container_sha = _sha(path)
        _CONTAINER_SHA[path] = container_sha
    if zipfile.is_zipfile(path):
        if not member:
            raise ValueError("ZIP source requires a member path")
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(member)
        member_sha = None
        if expected_container_sha is None:
            # Non-TEST ZIPs may record a compressed-member digest. TEST must
            # remain metadata-only and is bound by its pinned archive SHA,
            # exact member path, size, and CRC instead.
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo(member)
                offset, compressed_size = info.header_offset, info.compress_size
            with path.open("rb") as stream:
                stream.seek(offset)
                header = stream.read(30)
                if len(header) != 30 or header[:4] != b"PK\x03\x04":
                    raise ValueError(f"invalid ZIP local header for {member}")
                import struct
                name_len, extra_len = struct.unpack_from("<HH", header, 26)
                stream.seek(name_len + extra_len, 1)
                compressed = stream.read(compressed_size)
            if len(compressed) != compressed_size:
                raise ValueError(f"truncated ZIP member {member}")
            member_sha = hashlib.sha256(compressed).hexdigest()
        return {"path": str(path), "member": member, "container_sha256": container_sha,
                "member_sha256": member_sha, "member_size": info.file_size,
                "crc32": f"{info.CRC:08x}"}
    if member is not None and path.name != Path(member).name:
        # Direct-file roots are explicit; member remains the raw inventory path.
        pass
    return {"path": str(path), "member": member, "container_sha256": container_sha,
            "member_sha256": container_sha, "member_size": path.stat().st_size,
            "crc32": f"{_crc_file(path):08x}"}


def _crc_file(path: Path) -> int:
    import zlib
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xFFFFFFFF


def prepare_manifests(*, adjudicated: Path, l8_lock: Path, train_source: Path,
                      development_source: Path, test_source: Path, out_dir: Path,
                      mask_mapping_receipt: Path | None = None):
    adjudicated = adjudicated.resolve(strict=True)
    geometry_path = adjudicated / "geometry_manifest.json"
    per_patient_path = adjudicated / "per_patient.json"
    snapshot_path = adjudicated / "source_snapshot.json"
    summary_path = adjudicated / "SUMMARY.json"
    lock_path = l8_lock.resolve(strict=True)
    mask_mapping_path = (mask_mapping_receipt or lock_path.with_name("MASK_MAPPING_RECEIPT.json")).resolve(strict=True)
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "GEOMETRY_READY" or summary.get("blockers"):
        raise ValueError("M&Ms geometry adjudication is not ready")
    geometry = json.loads(geometry_path.read_text())
    snapshot = json.loads(snapshot_path.read_text())
    rows = json.loads(per_patient_path.read_text())
    lock = json.loads(lock_path.read_text())
    if (_sha(geometry_path) != summary.get("geometry_manifest_sha256")
            or _sha(snapshot_path) != summary.get("snapshot_sha256")
            or _sha(per_patient_path) != summary.get("per_patient_sha256")):
        raise ValueError("adjudicated geometry/source receipts do not match SUMMARY SHA pins")
    mask_mapping = json.loads(mask_mapping_path.read_text())
    if geometry.get("schema") != "self_audit.fuse_mnms_geometry_receipt.v1":
        raise ValueError("unexpected geometry receipt schema")
    if lock.get("schema") != "self_audit.fuse_mnms_target_l8_lock.v1" or lock.get("seed") != 42:
        raise ValueError("unexpected target L8 lock")
    if lock.get("geometry_manifest_sha256") != _sha(geometry_path):
        raise ValueError("L8 lock does not pin the supplied geometry receipt")
    if _sha(lock_path) != "ab239da264645bf82e246b8db6bb55dc5f3314260ef2dcccadc6c9e49bd579f4":
        raise ValueError("target L8 lock SHA does not match the pinned receipt")
    l8 = lock.get("selected_ids")
    if not isinstance(l8, list) or len(l8) != 8 or l8 != sorted(set(l8)):
        raise ValueError("invalid locked L8 IDs")
    if (mask_mapping.get("schema") != "self_audit.fuse_mnms_l8_mask_mapping_receipt.v1"
            or mask_mapping.get("l8_lock_sha256") != _sha(lock_path)
            or mask_mapping.get("selected_ids") != l8
            or mask_mapping.get("mask_reads_outside_l8") != 0
            or mask_mapping.get("source_to_canonical_class_map") != CLASS_MAPPING):
        raise ValueError("L8 mask mapping receipt does not match pinned lock and class map")
    mask_rows = {r["patient_id"]: r for r in mask_mapping.get("records", [])}
    if set(mask_rows) != set(l8):
        raise ValueError("mask mapping receipt must contain exactly L8")
    out_dir.mkdir(parents=True, exist_ok=False)
    split = {"train": lock["patient_ids_by_split"]["train"],
             "development": lock["patient_ids_by_split"]["val"],
             "test": lock["patient_ids_by_split"]["test"]}
    if tuple(map(len, (split["train"], split["development"], split["test"]))) != (175, 34, 136):
        raise ValueError("official TRAIN175/VAL34/TEST136 membership is required")
    row_by_id = {r["patient_id"]: r for r in rows}
    source_by_id = {r["patient_id"]: r for r in snapshot["records"]}
    expected_source_splits = {"train": set(split["train"]), "val": set(split["development"]),
                              "test": set(split["test"])}
    if (snapshot.get("source_identities_sha256") != lock.get("source_identities_sha256")
            or {role: {pid for pid, row in source_by_id.items() if row.get("split") == role}
                for role in expected_source_splits} != expected_source_splits):
        raise ValueError("source inventory identity digest or split membership differs from L8 lock")
    geometry_ids = {r.get("patient_id") for r in geometry.get("records", [])
                    if r.get("status") == "GEOMETRY_READY"}
    if geometry.get("source_identities_sha256") != lock.get("source_identities_sha256") or geometry_ids != set(source_by_id):
        raise ValueError("geometry receipt identity or patient inventory differs from L8 lock")
    geom_sha, lock_sha = _sha(geometry_path), _sha(lock_path)
    roots = {"train": train_source.resolve(strict=True), "development": development_source.resolve(strict=True),
             "test": test_source.resolve(strict=True)}
    if not roots["train"].is_dir() or not roots["development"].is_dir():
        raise ValueError("TRAIN and VAL sources must be unpacked directories; only TEST remains a ZIP")
    test_archive_sha = summary.get("artifact_sha256", {}).get("test_archive")
    if not isinstance(test_archive_sha, str) or len(test_archive_sha) != 64 or any(
            char not in "0123456789abcdef" for char in test_archive_sha):
        raise ValueError("SUMMARY must contain a lowercase test_archive SHA256 pin")
    if not roots["test"].is_file() or not zipfile.is_zipfile(roots["test"]):
        raise ValueError("TEST preparation requires the source ZIP; TEST files are not traversed or decoded")
    if _sha(roots["test"]) != test_archive_sha:
        raise ValueError("TEST source archive SHA differs from mandatory SUMMARY test_archive pin")
    outputs = {}
    for role in ("train", "development", "test"):
        patients = []
        labeled, unlabeled = [], []
        for pid in split[role]:
            row, source_row = row_by_id[pid], source_by_id[pid]
            source_id = source_row["source_identity"]
            raw_path = source_id.get("member", source_row["path"])
            root = roots[role]
            if root.is_dir():
                image_path = (root / raw_path).resolve(strict=True)
                image_source = _source(image_path, raw_path)
                if source_id.get("source_sha256") and source_id.get("member") is None and image_source["member_sha256"] != source_id["source_sha256"]:
                    raise ValueError(f"source file SHA differs from adjudication for {pid}")
            else:
                image_path = root
                image_source = _source(image_path, raw_path,
                                       expected_container_sha=test_archive_sha if role == "test" else None)
                if role == "test" and source_id.get("source_sha256") != test_archive_sha:
                    raise ValueError("TEST source identity differs from adjudicated archive pin")
            frame_count = int(source_row["shape"][3])
            md = row["metadata"]
            phases = None
            if md.get("phase_status") == "KNOWN":
                phases = {"ED": int(md["ED"]), "ES": int(md["ES"])}
            rec = {"id": pid, "image": image_source, "frame_count": frame_count, "phases": phases}
            if pid in l8 or role != "train":
                mask_path_member = raw_path.replace("_sa.nii.gz", "_sa_gt.nii.gz")
                if image_path.is_dir():
                    raise ValueError("unexpected directory source resolution")
                if pid in l8:
                    mapped = {int(m["frame"]): m for m in mask_rows[pid]["masks"]}
                    if set(mapped) != set(phases.values()):
                        raise ValueError(f"pinned mask mapping must contain exact ED/ES frames for {pid}")
                    mapped_name = Path(mapped[phases["ED"]]["mask_path"]).name
                    mask_path_member = str(Path(raw_path).parent / mapped_name)
                if root.is_dir():
                    mask_path = (root / mask_path_member).resolve(strict=True)
                    mask_source = _source(mask_path, mask_path_member)
                    if pid in l8 and mask_source["member_sha256"] != mapped[phases["ED"]]["mask_sha256"]:
                        raise ValueError(f"L8 mask bytes differ from pinned receipt for {pid}")
                else:
                    mask_source = _source(root, mask_path_member,
                                          expected_container_sha=test_archive_sha if role == "test" else None)
                    if pid in l8:
                        raise ValueError("locked L8 masks must be provided through the TRAIN source directory")
                if phases is None:
                    raise ValueError(f"labeled evaluation patient lacks ED/ES metadata: {pid}")
                rec["masks"] = {str(frame): mask_source for frame in phases.values()}
                if role == "train":
                    labeled.append(rec)
            if role == "train":
                unlabeled.append({k: rec[k] for k in ("id", "image", "frame_count", "phases")})
            else:
                patients.append(rec)
        manifest = {"schema": SCHEMA, "role": role, "synthetic": False,
                    "dataset_release": "original_2020_open345", "class_mapping": CLASS_MAPPING,
                    "split": split, "patients": [*split[role]],
                    "geometry_receipt": {"path": str(geometry_path), "sha256": geom_sha},
                    "l8_lock": {"path": str(lock_path), "sha256": lock_sha},
                    "mask_mapping_receipt": {"path": str(mask_mapping_path), "sha256": _sha(mask_mapping_path)},
                    "source_inventory": {"path": str(snapshot_path), "sha256": _sha(snapshot_path)},
                    "source_inventory_sha256": snapshot["source_identities_sha256"]}
        if role == "train":
            manifest.update(labeled=labeled, unlabeled=unlabeled)
        else:
            manifest["patients"] = patients
        out = out_dir / f"{role}.json"
        out.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        outputs[role] = {"path": str(out), "sha256": _sha(out)}
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudicated", type=Path, required=True, help="reports/fuse-mnms-geometry-20261113/adjudicated")
    parser.add_argument("--l8-lock", type=Path, required=True, help="reports/fuse-mnms-geometry-20261113/l8/L8_LOCK.json")
    parser.add_argument("--mask-mapping-receipt", type=Path, help="pinned reports/fuse-mnms-geometry-20261113/l8/MASK_MAPPING_RECEIPT.json")
    parser.add_argument("--train-source", type=Path, required=True, help="unpacked raw root containing Training/Labeled/... source files")
    parser.add_argument("--development-source", type=Path, required=True, help="unpacked raw root containing Validation/... source files")
    parser.add_argument("--test-source", type=Path, required=True, help="official TEST source ZIP (never unpacked during preparation)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_manifests(adjudicated=args.adjudicated, l8_lock=args.l8_lock,
                                      train_source=args.train_source, development_source=args.development_source,
                                      test_source=args.test_source, out_dir=args.out_dir,
                                      mask_mapping_receipt=args.mask_mapping_receipt), indent=2))


if __name__ == "__main__":
    main()
