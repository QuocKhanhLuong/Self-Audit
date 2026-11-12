"""Hash-bound selector evidence for restored manifests, without rewriting them."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

_SHA = re.compile(r"[0-9a-f]{64}")
_SELECTOR = "sha256(seed:patient_id) over source labeled IDs"


def _read_pin(path: str, expected: str) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(expected, str) or not _SHA.fullmatch(expected):
        raise ValueError("external lineage requires explicit SHA-256 pins")
    source = Path(path).expanduser().resolve(strict=True)
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("external lineage file SHA-256 mismatch")
    value = json.loads(payload)
    if (not isinstance(value, dict) or value.get("schema") != "self_audit.data.v2"
            or value.get("role") != "training" or value.get("synthetic") is not False):
        raise ValueError("external lineage requires real training manifests")
    return value, {"path": str(source), "sha256": expected}


def _records(value: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    records, labeled = {}, set()
    for role in ("labeled", "unlabeled"):
        rows = value.get(role)
        if not isinstance(rows, list):
            raise ValueError("external lineage requires explicit L/U lists")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("external lineage patient record must be an object")
            pid = row.get("id")
            if not isinstance(pid, str) or not pid or pid in records:
                raise ValueError("external lineage has invalid, duplicate or overlapping patient IDs")
            if not isinstance(row.get("image_sha256"), str) or not _SHA.fullmatch(row["image_sha256"]):
                raise ValueError("external lineage patient image hash is missing")
            if role == "unlabeled" and row.get("masks"):
                raise ValueError("external lineage U record exposes masks")
            if role == "labeled":
                _mask_hashes(row)
                labeled.add(pid)
            records[pid] = row
    return records, labeled


def _mask_hashes(record: Mapping[str, Any]) -> dict[str, str]:
    masks = record.get("masks")
    if not isinstance(masks, Mapping) or not masks:
        raise ValueError("external lineage labeled record must have mask hashes")
    hashes = {}
    for frame, item in masks.items():
        sha = item.get("sha256") if isinstance(item, Mapping) else None
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ValueError("external lineage mask hash is missing")
        hashes[str(frame)] = sha
    return hashes


def _selected(eligible: list[str], seed: int, k: int) -> list[str]:
    if type(k) is not int or not 0 < k <= len(eligible):
        raise ValueError("external lineage K exceeds the original eligible cohort")
    return sorted(sorted(eligible, key=lambda pid: hashlib.sha256(
        f"{seed}:{pid}".encode()).hexdigest())[:k])


def load_external_lineage(spec: Mapping[str, Any], *, dataset: str, seed: int) -> dict[str, Any]:
    """Verify historical selector plus its exact source bytes, without arrays."""
    keys = {"reference_path", "reference_sha256", "source_path", "source_sha256"}
    if not isinstance(spec, Mapping) or set(spec) != keys:
        raise ValueError("external lineage requires reference/source paths and SHA-256 pins together")
    reference, reference_pin = _read_pin(spec["reference_path"], spec["reference_sha256"])
    source, source_pin = _read_pin(spec["source_path"], spec["source_sha256"])
    for value in (source, reference):
        if str(value.get("dataset", "")).lower() != dataset.lower():
            raise ValueError("external lineage dataset mismatch")
    lineage = reference.get("fewshot")
    if (not isinstance(lineage, Mapping) or lineage.get("selection") != _SELECTOR
            or type(lineage.get("seed")) is not int or lineage["seed"] != seed
            or lineage.get("source_sha256") != source_pin["sha256"]):
        raise ValueError("external lineage reference selector/seed/source pin mismatch")
    source_records, source_labeled = _records(source)
    eligible = lineage.get("eligible_labeled_ids")
    if (not isinstance(eligible, list) or not eligible
            or any(not isinstance(pid, str) for pid in eligible)
            or len(set(eligible)) != len(eligible) or set(eligible) != source_labeled):
        raise ValueError("external lineage eligible IDs differ from original source L")
    context = {"source": source, "records": source_records, "eligible": eligible,
               "seed": seed, "lineage": dict(lineage),
               "pins": {"reference": reference_pin, "source": source_pin}}
    validate_external_view(reference, k=lineage.get("k"), context=context)
    if sorted(lineage.get("selected_ids", [])) != _selected(eligible, seed, lineage["k"]):
        raise ValueError("external lineage historical selected IDs do not reproduce the selector")
    return context


def validate_external_view(value: Mapping[str, Any], *, k: int, context: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce a K view from immutable IDs and hashes, ignoring relocated paths."""
    source = context["source"]
    for key in ("schema", "role", "synthetic", "dataset", "evaluation_ids", "class_mapping"):
        if value.get(key) != source.get(key):
            raise ValueError(f"external lineage view changed {key}")
    if value.get("label_budget") != k or value.get("label_seed") != context["seed"]:
        raise ValueError("external lineage view label budget/seed mismatch")
    rows, labeled = _records(value)
    if set(rows) != set(context["records"]):
        raise ValueError("external lineage view changed the TRAIN patient cohort")
    selected = _selected(context["eligible"], context["seed"], k)
    if labeled != set(selected):
        raise ValueError("external lineage view labeled IDs differ from reproduced selector")
    for pid, record in rows.items():
        original = context["records"][pid]
        if record["image_sha256"] != original["image_sha256"]:
            raise ValueError("external lineage view image hash mismatch")
        if pid in labeled and _mask_hashes(record) != _mask_hashes(original):
            raise ValueError("external lineage view mask hash mismatch")
    return {**context["lineage"], "k": k, "selected_ids": selected,
            "external_evidence": context["pins"], "validation": "REPRODUCED_FROM_HASH_PINNED_SOURCE"}
