"""Raw-partition bundle sealing and GT-free evaluator-skeleton verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .provenance import canonical_bytes, sha256_file, write_json


class FreezeError(ValueError):
    pass


def seal_raw_bundle(partitions: list[dict[str, Any]], output_dir: str | Path, *, required_sample_ids: set[str]) -> dict[str, Any]:
    seen = {entry["sample_id"] for entry in partitions}
    if seen != required_sample_ids:
        raise FreezeError(f"bundle samples differ: missing={required_sample_ids-seen}, unexpected={seen-required_sample_ids}")
    entries = []
    for item in sorted(partitions, key=lambda value: value["sample_id"]):
        entry = dict(item)
        if entry["status"] == "success":
            entry["raw_file_hash"] = sha256_file(entry["raw_partition_path"])
        entries.append(entry)
    bundle = {"schema_version": "cuts.cardiac.p0.raw-bundle.v1", "entries": entries,
              "sample_ids": sorted(required_sample_ids)}
    bundle["bundle_hash"] = hashlib.sha256(canonical_bytes(bundle)).hexdigest()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "raw_bundle.json", bundle)
    return bundle


def verify_raw_bundle(bundle_path: str | Path, *, expected_sample_ids: set[str] | None = None) -> dict[str, Any]:
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    supplied_hash = bundle.pop("bundle_hash", None)
    if supplied_hash != hashlib.sha256(canonical_bytes(bundle)).hexdigest():
        raise FreezeError("raw bundle index hash mismatch")
    actual = {entry["sample_id"] for entry in bundle["entries"]}
    declared = set(bundle["sample_ids"])
    if actual != declared or (expected_sample_ids is not None and actual != expected_sample_ids):
        raise FreezeError("missing or unexpected raw bundle samples")
    for entry in bundle["entries"]:
        if entry["status"] == "success" and sha256_file(entry["raw_partition_path"]) != entry["raw_file_hash"]:
            raise FreezeError(f"raw partition mutation: {entry['sample_id']}")
    bundle["bundle_hash"] = supplied_hash
    return bundle


def evaluator_skeleton(bundle_path: str | Path, output_dir: str | Path, *, expected_sample_ids: set[str] | None = None) -> dict[str, Any]:
    """Verify sealed prediction artifacts before any future GT mount is allowed."""
    bundle = verify_raw_bundle(bundle_path, expected_sample_ids=expected_sample_ids)
    receipt = {"schema_version": "cuts.cardiac.p0.evaluator-skeleton.v1", "bundle_hash": bundle["bundle_hash"],
               "verified_samples": len(bundle["entries"]), "gt_opened": False, "metrics_written": False}
    write_json(Path(output_dir) / "verification_receipt.json", receipt)
    return receipt
