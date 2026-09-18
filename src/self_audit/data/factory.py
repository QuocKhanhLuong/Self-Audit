"""Factory bridge from legacy training configs to audited cardiac adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .cardiac35d import Cardiac35DSliceDataset, CardiacUnit
from .cmr_multi import CMRMultiAdapter
from .cmrxmotion import CMRxMotionAdapter
from .splits import validate_subject_split


NEW_DATASET_NAMES = frozenset({"cmr_multi", "cmr_motion"})


def _source_config(config: Mapping[str, Any], dataset_name: str) -> dict[str, Any]:
    raw_sources = config.get("sources", {})
    if isinstance(raw_sources, Mapping) and dataset_name in raw_sources:
        source = raw_sources[dataset_name]
        if not isinstance(source, Mapping):
            raise ValueError(f"dataset source {dataset_name!r} must be a mapping")
        result = dict(source)
    else:
        result = {}
    result.setdefault("data_root", config.get("data_root"))
    if not result.get("data_root"):
        raise ValueError(f"data_root is required for dataset {dataset_name!r}")
    return result


def build_cardiac_adapter(dataset_name: str, source_config: Mapping[str, Any]):
    name = str(dataset_name).strip().lower()
    root = source_config.get("data_root")
    if not root:
        raise ValueError(f"data_root is required for dataset {name!r}")
    if name == "cmr_multi":
        return CMRMultiAdapter(
            root,
            require_confident=not bool(source_config.get("allow_manual_review", False)),
        )
    if name == "cmr_motion":
        return CMRxMotionAdapter(
            root,
            allow_affine_mismatch=bool(source_config.get("allow_affine_mismatch", False)),
        )
    raise ValueError(f"Unsupported cardiac adapter dataset {dataset_name!r}")


def _split_case_ids(path: str | Path, split: str) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"split manifest must contain an object: {path}")
    aliases = {
        "train": ("train", "training"),
        "val": ("val", "validation", "valid"),
        "test": ("test", "testing"),
    }
    canonical = str(split).strip().lower()
    candidates = aliases.get(canonical, (canonical,))
    values: list[Any] = []
    for alias in candidates:
        for key in (alias, f"{alias}_cases", f"{alias}_volumes"):
            raw = payload.get(key)
            if isinstance(raw, (list, tuple)):
                values.extend(raw)
        nested = payload.get("splits")
        if isinstance(nested, Mapping):
            raw = nested.get(alias)
            if isinstance(raw, Mapping):
                for key in ("cases", "volumes"):
                    if isinstance(raw.get(key), (list, tuple)):
                        values.extend(raw[key])
            elif isinstance(raw, (list, tuple)):
                values.extend(raw)
    if not values:
        raise ValueError(f"split manifest {path} contains no cases for split {split!r}")
    return {Path(str(value)).name.removesuffix(".nii.gz").removesuffix(".nii") for value in values}


def _filter_units(units: Sequence[CardiacUnit], manifest_path: str | Path | None, split: str) -> list[CardiacUnit]:
    if manifest_path is None:
        raise ValueError(
            f"a subject-level split manifest is required for audited cardiac dataset split {split!r}"
        )
    allowed = _split_case_ids(manifest_path, split)
    result: list[CardiacUnit] = []
    for unit in units:
        parent = str(unit.metadata.get("parent_case_id", unit.case_id))
        if unit.case_id in allowed or parent in allowed:
            result.append(unit)
    if not result:
        raise ValueError(f"split {split!r} has no matching supervised units")
    return result


def build_audited_cardiac_dataset(
    config: Mapping[str, Any],
    *,
    dataset_name: str,
    split: str,
    train: bool,
) -> Dataset:
    source = _source_config(config, dataset_name)
    adapter = build_cardiac_adapter(dataset_name, source)
    units = adapter.discover_units(split=split, supervised_only=True)
    manifest = source.get("split_manifest", config.get("split_manifest"))
    units = _filter_units(units, manifest, split)
    preprocessing = config.get("preprocessing", {})
    if not isinstance(preprocessing, Mapping):
        preprocessing = {}
    return Cardiac35DSliceDataset(
        adapter,
        units=units,
        image_size=config.get("image_size", 256),
        lower_percentile=float(preprocessing.get("clipping_min", 0.5)),
        upper_percentile=float(preprocessing.get("clipping_max", 99.5)),
        max_cache=int(config.get("max_cache", 2)),
        augment=bool(train and config.get("augment", False)),
    )


def validate_audited_dataset_splits(
    config: Mapping[str, Any],
    *,
    dataset_name: str,
) -> dict[str, Any]:
    """Validate audited subject manifests before a new dataset enters training."""

    source = _source_config(config, dataset_name)
    adapter = build_cardiac_adapter(dataset_name, source)
    if dataset_name == "cmr_multi":
        mapping = {
            pair.case_id: pair.subject_id
            for pair in adapter.pairs
            if adapter.infer_case(pair).accepted
        }
    else:
        records = adapter.audit()
        valid = {
            record.case_id
            for record in records
            if record.has_mask
            and not record.error
            and (record.affine_equal is True or adapter.allow_affine_mismatch)
        }
        mapping = {
            case.case_id: case.subject_id
            for case in adapter.cases
            if case.case_id in valid
        }
    manifest = source.get("split_manifest", config.get("split_manifest"))
    if manifest is None:
        raise ValueError(f"subject-level split_manifest is required for {dataset_name}")
    split_cases: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        if split == "test" and config.get("test_split", "test") is None:
            continue
        split_cases[split] = sorted(_split_case_ids(manifest, split))
    if "train" not in split_cases or "val" not in split_cases:
        raise ValueError(f"subject-level split manifest for {dataset_name} must contain train and val")
    declared = {case_id for values in split_cases.values() for case_id in values}
    unknown = sorted(declared.difference(mapping))
    missing = sorted(set(mapping).difference(declared))
    if unknown:
        raise ValueError(f"split manifest for {dataset_name} contains unknown cases: {unknown[:5]}")
    if missing:
        raise ValueError(f"split manifest for {dataset_name} omits audited cases: {missing[:5]}")
    validate_subject_split(split_cases, mapping)
    signature_payload = ";".join(
        f"{split}={','.join(split_cases[split])}" for split in sorted(split_cases)
    )
    signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
    return {
        "dataset": dataset_name,
        "validated": True,
        "strategy": "subject_manifest",
        "split_signature": signature,
        "membership_signature": signature,
        "cases": {split: len(values) for split, values in split_cases.items()},
        "case_counts": {split: len(values) for split, values in split_cases.items()},
        "patients": {
            split: len({mapping[case_id] for case_id in values})
            for split, values in split_cases.items()
        },
        "patient_counts": {
            split: len({mapping[case_id] for case_id in values})
            for split, values in split_cases.items()
        },
        "test_available": "test" in split_cases,
        "records": len(mapping),
        "effective_identities": split_cases,
        "effective_records": {
            split: [
                {"case_id": case_id, "subject_id": mapping[case_id], "split": split}
                for case_id in values
            ]
            for split, values in split_cases.items()
        },
    }
