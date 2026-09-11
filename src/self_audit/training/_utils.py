"""Small shared helpers for the three explicit training phases."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..provenance import (
    CheckpointBinding,
    checkpoint_producer,
    producer_provenance_record,
    resolve_model_identity,
    state_digest,
    verify_model_config,
)
from ..provenance import file_sha256 as _provenance_file_sha256
from .checkpoint_commit import (
    PUBLIC_BEST_NAME,
    SelectionError,
    resolve_best_reference,
)
from ..serialization import (
    CheckpointSerializationError,
    SAFE_TENSOR_DTYPES,
    atomic_save_torch,
    clean_wandb_payload,
    normalize_checkpoint_payload,
    normalize_metadata_tree,
)


# Keep the model constructor boundary explicit.  Training and data settings
# live beside ``model`` in the YAML files and must not accidentally become
# constructor kwargs (``image_size`` was previously forwarded this way).
MODEL_ARCHITECTURE_KEYS = frozenset(
    {
        "encoder_name",
        "pretrained_encoder",
        "encoder_allow_fallback",
        "shared_channels",
        "num_classes",
        "window_k",
        "max_turns",
    }
)
MODEL_RUNTIME_KEYS = frozenset({"tau_accept", "threshold", "t_max"})
MODEL_CONFIG_KEYS = MODEL_ARCHITECTURE_KEYS | MODEL_RUNTIME_KEYS
CHECKPOINT_FORMAT_VERSION = 1


def _require_integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Integral, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value}")
    return value


def _require_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _require_positive_real(value: Any, name: str) -> float:
    result = _require_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return result


def _require_nonnegative_real(value: Any, name: str) -> float:
    result = _require_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return result


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value


def validate_image_size(value: Any, *, default: int = 256) -> int | tuple[int, int]:
    """Validate the dataset raster size without making it a model setting."""

    value = default if value is None else value
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"image_size must be an integer or a pair, got {value!r}")
        return (
            _require_integer(value[0], "image_size[0]", minimum=1),
            _require_integer(value[1], "image_size[1]", minimum=1),
        )
    return _require_integer(value, "image_size", minimum=1)


def validate_depth_axis(value: Any, *, name: str = "depth_axis") -> int | None:
    """Validate an optional NumPy/PyTorch depth axis without guessing."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in (0, 1, 2):
        raise ValueError(f"{name} must be 0, 1, 2, or null, got {value!r}")
    return int(value)


def filter_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return only constructor kwargs from a model config.

    ``tau_accept``/``threshold``/``t_max`` are accepted as known inference
    settings for compatibility with the existing joint YAML, but are
    intentionally validated and discarded here.  Dataset ``image_size`` is
    never a model kwarg; putting it under ``model`` is an explicit error so a
    misspelled or misplaced size cannot silently change the data contract.
    """

    if not isinstance(config, Mapping):
        raise ValueError(f"model config must be a mapping, got {type(config).__name__}")
    unknown = set(config) - MODEL_CONFIG_KEYS
    if "image_size" in config:
        raise ValueError("image_size belongs to the dataset config, not model")
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"Unsupported model config key(s): {names}")

    if "encoder_name" in config and not isinstance(config["encoder_name"], str):
        raise ValueError("encoder_name must be a string")
    for key in ("pretrained_encoder", "encoder_allow_fallback"):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f"{key} must be a boolean")
    for key in ("shared_channels", "num_classes", "window_k", "max_turns"):
        if key in config:
            minimum = 2 if key == "num_classes" else 1
            _require_integer(config[key], key, minimum=minimum)
    for key in ("tau_accept", "threshold"):
        if key in config:
            _require_real(config[key], key)
    if "t_max" in config:
        _require_integer(config["t_max"], "t_max", minimum=0)

    return {key: config[key] for key in MODEL_ARCHITECTURE_KEYS if key in config}


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("PyYAML is required to load Self-Audit configs") from exc
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with open(path, encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a mapping, got {type(value).__name__}")
    return value


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python/NumPy/PyTorch and optionally request deterministic kernels."""

    seed = _require_integer(seed, "seed", minimum=0)
    _require_bool(deterministic, "deterministic")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def resolve_device(requested: str | None) -> torch.device:
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if str(requested).startswith("mps") and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def build_model_from_config(config: dict[str, Any], device: torch.device) -> nn.Module:
    from self_audit.models.self_audit_net import build_self_audit_net

    raw_model_cfg = config.get("model", {})
    if raw_model_cfg is None:
        raw_model_cfg = {}
    model_cfg = filter_model_config(raw_model_cfg)
    model_cfg.setdefault("num_classes", config.get("num_classes", 4))
    model_cfg.setdefault("encoder_name", "convnext_tiny")
    model_cfg.setdefault("pretrained_encoder", False)
    model_cfg.setdefault("encoder_allow_fallback", True)
    model_cfg.setdefault("shared_channels", 96)
    model_cfg.setdefault("window_k", 8)
    model_cfg.setdefault("max_turns", 3)
    if not isinstance(model_cfg["encoder_name"], str) or not model_cfg["encoder_name"].strip():
        raise ValueError("model.encoder_name must be a non-empty string")
    for key in ("shared_channels", "num_classes", "window_k", "max_turns"):
        value = model_cfg[key]
        minimum = 2 if key == "num_classes" else 1
        _require_integer(value, f"model.{key}", minimum=minimum)
    for key in ("pretrained_encoder", "encoder_allow_fallback"):
        if not isinstance(model_cfg[key], bool):
            raise ValueError(f"model.{key} must be a boolean, got {model_cfg[key]!r}")
    model = build_self_audit_net(**model_cfg)
    return model.to(device)


def get_model_parameter_summary(model: nn.Module) -> dict[str, Any]:
    """Calculate total, trainable, and per-module parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    modules = [
        ("Encoder (ConvNeXt)", getattr(model, "encoder", None)),
        ("FPN (Feature Pyramid)", getattr(model, "fpn", None)),
        ("Initial Head (A0)", getattr(model, "initial_head", None)),
        ("Annotation Expert (DW-Attn)", getattr(model, "annotation_expert", None)),
        ("Transition Auditor", getattr(model, "auditor", None)),
    ]

    breakdown = []
    for name, module in modules:
        if module is not None:
            m_total = sum(p.numel() for p in module.parameters())
            m_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            breakdown.append({
                "name": name,
                "total": m_total,
                "trainable": m_trainable,
                "frozen": m_total - m_trainable,
                "is_trainable": m_trainable > 0,
            })

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "breakdown": breakdown,
    }


def print_model_parameter_summary(model: nn.Module, *, title: str = "Model Parameter Summary") -> None:
    """Print a clean, structured table of parameter counts."""
    summary = get_model_parameter_summary(model)
    total = summary["total_parameters"]
    trainable = summary["trainable_parameters"]
    frozen = summary["frozen_parameters"]

    print("=" * 78)
    print(f" {title:^76}")
    print("=" * 78)
    print(f" {'Component':<32} {'Total Params':<16} {'Trainable':<16} {'Status':<12}")
    print("-" * 78)
    for item in summary["breakdown"]:
        status = "Trainable" if item["is_trainable"] else "Frozen"
        print(f" {item['name']:<32} {item['total']:>12,d}    {item['trainable']:>12,d}    {status:<12}")
    print("-" * 78)
    print(f" Total Parameters:     {total:>12,d} ({total / 1e6:.2f}M)")
    trainable_pct = (trainable / total * 100.0) if total > 0 else 0.0
    frozen_pct = (frozen / total * 100.0) if total > 0 else 0.0
    print(f" Trainable Parameters: {trainable:>12,d} ({trainable / 1e6:.2f}M - {trainable_pct:.1f}%)")
    print(f" Frozen Parameters:    {frozen:>12,d} ({frozen / 1e6:.2f}M - {frozen_pct:.1f}%)")
    print("=" * 78)


def build_patient_dataset(
    config: dict[str, Any],
    *,
    split: str,
    train: bool,
) -> torch.utils.data.Dataset:
    raw_ds = config.get("dataset", "acdc")
    if isinstance(raw_ds, Mapping):
        dataset_name = str(raw_ds.get("name", "acdc")).strip().lower()
        if "data_root" not in config and "data_root" in raw_ds:
            config = dict(config)
            config["data_root"] = raw_ds["data_root"]
    else:
        dataset_name = str(raw_ds).strip().lower()
    if dataset_name not in {"acdc", "mnms"}:
        raise ValueError(f"Unsupported dataset {dataset_name!r}")
    preprocessing = config.get("preprocessing", {})
    if not isinstance(preprocessing, Mapping):
        raise ValueError("preprocessing must be a mapping")
    preprocessing_kwargs = {
        "lower_percentile": preprocessing.get("clipping_min", 0.5),
        "upper_percentile": preprocessing.get("clipping_max", 99.5),
        "foreground_only": preprocessing.get("foreground_only", False),
    }
    if dataset_name == "mnms":
        try:
            from self_audit.data.mnms import DEFAULT_MNMS_TO_ACDC, MNMSClassMapping, MNMSDataset, is_mnms_binary_path
        except ImportError:
            from src.self_audit.data.mnms import DEFAULT_MNMS_TO_ACDC, MNMSClassMapping, MNMSDataset, is_mnms_binary_path

        data_root = config.get("data_root", "preprocessed_data/mnm")
        if is_mnms_binary_path(data_root):
            raise ValueError(f"M&Ms binary derivative dataset is incompatible with four-class contract: {data_root}")
        num_classes = int(config.get("num_classes", config.get("model", {}).get("num_classes", 4)))
        if num_classes != 4:
            raise ValueError(f"M&Ms external evaluation requires the four-class contract (num_classes=4), got {num_classes}")
        raw_mapping = config.get("raw_to_acdc", config.get("class_mapping", DEFAULT_MNMS_TO_ACDC))
        if not isinstance(raw_mapping, Mapping):
            raise ValueError("raw_to_acdc must be a mapping from raw M&Ms labels to ACDC labels")
        mapping = MNMSClassMapping(raw_mapping)
        kwargs: dict[str, Any] = {
            **preprocessing_kwargs,
            "data_root": data_root,
            "split": split,
            "image_size": validate_image_size(config.get("image_size")),
            "augment": bool(train and config.get("augment", False)),
            "class_mapping": mapping,
        }
        for key in ("depth_axis", "expected_slices", "max_cache"):
            if key in config:
                kwargs[key] = config[key]
        return MNMSDataset(**kwargs)

    try:
        from self_audit.data.acdc import ACDCDataset
    except ImportError:
        from src.self_audit.data.acdc import ACDCDataset

    data_root = config.get("data_root", "preprocessed_data/ACDC")
    kwargs: dict[str, Any] = {
        **preprocessing_kwargs,
        "data_root": data_root,
        "split": split,
        "image_size": validate_image_size(config.get("image_size")),
        "augment": bool(train and config.get("augment", True)),
    }
    for key in ("split_manifest", "seed", "depth_axis", "expected_slices", "max_cache"):
        if key in config:
            kwargs[key] = config[key]
    return ACDCDataset(**kwargs)


def build_data_loader(
    dataset: Dataset,
    config: Mapping[str, Any],
    *,
    device: torch.device,
    train: bool,
    batch_size: int | None = None,
) -> DataLoader:
    """Build a reliable DataLoader without invalid worker-only arguments."""

    if not isinstance(config, Mapping):
        raise ValueError("DataLoader config must be a mapping")
    workers = _require_integer(config.get("num_workers", 0), "num_workers", minimum=0)
    size = _require_integer(batch_size if batch_size is not None else config.get("batch_size", 1), "batch_size", minimum=1)
    pin_memory = bool(config.get("pin_memory", device.type == "cuda"))
    kwargs: dict[str, Any] = {
        "batch_size": size,
        "shuffle": bool(train),
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", False))
        if "prefetch_factor" in config:
            kwargs["prefetch_factor"] = _require_integer(config["prefetch_factor"], "prefetch_factor", minimum=1)
    return DataLoader(dataset, **kwargs)


def validate_dataset_splits(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate patient-disjoint splits before a training entrypoint starts.

    An explicitly configured manifest is authoritative: missing files, extra
    discovered cases, missing requested splits, or patient overlap fail
    immediately.  A deterministic patient split is used only when no manifest
    was configured.
    """

    if not isinstance(config, Mapping):
        raise ValueError("training config must be a mapping")
    raw_ds = config.get("dataset", "acdc")
    if isinstance(raw_ds, Mapping):
        dataset_name = str(raw_ds.get("name", "acdc")).lower()
        if "data_root" not in config and "data_root" in raw_ds:
            config = dict(config)
            config["data_root"] = raw_ds["data_root"]
    else:
        dataset_name = str(raw_ds).lower()
    if dataset_name not in {"acdc", "mnms"}:
        raise ValueError(f"Unsupported dataset {dataset_name!r}")
    if dataset_name == "mnms":
        try:
            from self_audit.data.common import (
                compute_split_signature,
                load_array,
                to_depth_first,
                validate_patient_split,
            )
            from self_audit.data.mnms import (
                DEFAULT_MNMS_TO_ACDC,
                MNMSClassMapping,
                discover_mnms_records,
                is_mnms_binary_path,
            )
        except ImportError:
            from src.self_audit.data.common import (
                compute_split_signature,
                load_array,
                to_depth_first,
                validate_patient_split,
            )
            from src.self_audit.data.mnms import (
                DEFAULT_MNMS_TO_ACDC,
                MNMSClassMapping,
                discover_mnms_records,
                is_mnms_binary_path,
            )

        data_root = config.get("data_root", "preprocessed_data/mnm")
        if is_mnms_binary_path(data_root):
            raise ValueError(
                f"M&Ms binary derivative dataset is incompatible with four-class contract: {data_root}"
            )

        num_classes = int(config.get("num_classes", config.get("model", {}).get("num_classes", 4)))
        if num_classes != 4:
            raise ValueError(
                f"M&Ms external evaluation requires the four-class contract (num_classes=4), got {num_classes}"
            )

        raw_mapping = config.get("raw_to_acdc", config.get("class_mapping", DEFAULT_MNMS_TO_ACDC))
        if not isinstance(raw_mapping, Mapping):
            raise ValueError("raw_to_acdc must be a mapping from raw M&Ms labels to ACDC labels")
        mapping = MNMSClassMapping(raw_mapping)
        depth_axis = validate_depth_axis(config.get("depth_axis"))

        has_native_splits = "train_split" in config or "val_split" in config
        if has_native_splits:
            train_split = str(config.get("train_split", "train"))
            val_split = str(config.get("val_split", "val"))
            test_split = str(config.get("test_split", "test")) if config.get("test_split") is not None else None

            splits_to_check: list[tuple[str, str]] = [("train", train_split), ("val", val_split)]
            if test_split is not None:
                aliases = {"test": ("test", "testing"), "testing": ("test", "testing")}.get(test_split.lower(), (test_split,))
                test_exists = any((Path(data_root) / a).exists() for a in aliases)
                if test_exists:
                    splits_to_check.append(("test", test_split))

            records_by_split: dict[str, list[VolumeRecord]] = {}
            case_counts: dict[str, int] = {}
            patient_counts: dict[str, int] = {}
            slice_counts: dict[str, int] = {}
            effective_identities: dict[str, list[str]] = {}
            effective_records: dict[str, list[dict[str, Any]]] = {}
            label_values: dict[str, list[int]] = {}
            mapped_label_values: dict[str, list[int]] = {}
            total_cohort_records = 0

            for canonical_split, requested_split in splits_to_check:
                records = discover_mnms_records(data_root, split=requested_split)
                if not records:
                    raise ValueError(f"M&Ms split {requested_split!r} ({canonical_split}) contains zero records")
                records_by_split[canonical_split] = records
                total_cohort_records += len(records)
                total_slices = 0
                labels_seen: set[int] = set()
                mapped_labels_seen: set[int] = set()
                descriptors: list[dict[str, Any]] = []

                for record in records:
                    volume, _ = load_array(record.image_path)
                    raw_mask, _ = load_array(record.mask_path)
                    volume_zhw = to_depth_first(volume, depth_axis=depth_axis)
                    mask_zhw = to_depth_first(raw_mask, depth_axis=depth_axis)
                    if volume_zhw.shape != mask_zhw.shape:
                        raise ValueError(
                            f"M&Ms volume/mask shape mismatch for {record.case_id}: "
                            f"{volume_zhw.shape} vs {mask_zhw.shape}"
                        )
                    if not np.isfinite(mask_zhw).all() or not np.equal(mask_zhw, np.floor(mask_zhw)).all():
                        raise ValueError(f"M&Ms mask labels for {record.case_id} must be finite integers")
                    labels_seen.update(int(value) for value in np.unique(mask_zhw))
                    mapped_mask = mapping.apply(mask_zhw)
                    mapped_labels_seen.update(int(value) for value in np.unique(mapped_mask))
                    total_slices += int(volume_zhw.shape[0])
                    descriptors.append({
                        "case_id": record.case_id,
                        "patient_id": record.patient_id,
                        "image_path": str(record.image_path),
                        "mask_path": str(record.mask_path),
                        "split": canonical_split,
                        "source_format": record.source_format,
                    })

                case_counts[canonical_split] = len(records)
                patient_counts[canonical_split] = len({record.patient_id for record in records})
                slice_counts[canonical_split] = total_slices
                effective_identities[canonical_split] = [r.case_id for r in records]
                effective_records[canonical_split] = descriptors
                label_values[canonical_split] = sorted(labels_seen)
                mapped_label_values[canonical_split] = sorted(mapped_labels_seen)

            # Enforce patient disjointness across splits
            validate_patient_split({s: [r.case_id for r in recs] for s, recs in records_by_split.items()})

            # Enforce no overlapping patient prefixes across splits
            split_keys = list(records_by_split.keys())
            for i in range(len(split_keys)):
                for j in range(i + 1, len(split_keys)):
                    s1, s2 = split_keys[i], split_keys[j]
                    pts1 = {r.patient_id for r in records_by_split[s1]}
                    pts2 = {r.patient_id for r in records_by_split[s2]}
                    for p1 in pts1:
                        for p2 in pts2:
                            if p1 == p2 or p1.startswith(p2) or p2.startswith(p1):
                                raise ValueError(
                                    f"Patient prefix overlap detected between {s1!r} and {s2!r}: {p1!r} vs {p2!r}"
                                )

            signature = compute_split_signature(records_by_split)
            return {
                "dataset": dataset_name,
                "validated": True,
                "strategy": "directory_splits",
                "splits": list(records_by_split.keys()),
                "split_signature": signature,
                "membership_signature": signature,
                "cases": case_counts,
                "case_counts": case_counts,
                "patients": patient_counts,
                "patient_counts": patient_counts,
                "slices": slice_counts,
                "slice_counts": slice_counts,
                "label_values": label_values,
                "mapped_label_values": mapped_label_values,
                "test_available": "test" in records_by_split,
                "records": total_cohort_records,
                "effective_identities": effective_identities,
                "effective_records": effective_records,
                "raw_to_acdc": dict(mapping.raw_to_acdc),
                "content_hash_duplicate_detection": {"executed": False, "status": "deferred"},
            }

        requested_split = str(config.get("test_split", config.get("split", "testing")))
        records = discover_mnms_records(data_root, split=requested_split)
        depth_axis = validate_depth_axis(config.get("depth_axis"))
        total_slices = 0
        labels_seen: set[int] = set()
        mapped_labels_seen: set[int] = set()
        descriptors: list[dict[str, Any]] = []
        for record in records:
            volume, _ = load_array(record.image_path)
            raw_mask, _ = load_array(record.mask_path)
            volume_zhw = to_depth_first(volume, depth_axis=depth_axis)
            mask_zhw = to_depth_first(raw_mask, depth_axis=depth_axis)
            if volume_zhw.shape != mask_zhw.shape:
                raise ValueError(
                    f"M&Ms volume/mask shape mismatch for {record.case_id}: "
                    f"{volume_zhw.shape} vs {mask_zhw.shape}"
                )
            if not np.isfinite(mask_zhw).all() or not np.equal(mask_zhw, np.floor(mask_zhw)).all():
                raise ValueError(f"M&Ms mask labels for {record.case_id} must be finite integers")
            labels_seen.update(int(value) for value in np.unique(mask_zhw))
            mapped_mask = mapping.apply(mask_zhw)
            mapped_labels_seen.update(int(value) for value in np.unique(mapped_mask))
            total_slices += int(volume_zhw.shape[0])
            descriptors.append({
                "case_id": record.case_id,
                "patient_id": record.patient_id,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path),
                "split": requested_split,
                "source_format": record.source_format,
            })
        canonical_split = "test"
        identities = [record.case_id for record in records]
        patient_count = len({record.patient_id for record in records})
        signature = compute_split_signature({canonical_split: records})
        case_counts = {canonical_split: len(records), requested_split: len(records)}
        patient_counts = {canonical_split: patient_count, requested_split: patient_count}
        slice_counts = {canonical_split: total_slices, requested_split: total_slices}
        label_values = {canonical_split: sorted(labels_seen), requested_split: sorted(labels_seen)}
        mapped_label_values = {
            canonical_split: sorted(mapped_labels_seen),
            requested_split: sorted(mapped_labels_seen),
        }
        effective_identities = {canonical_split: identities, requested_split: identities}
        effective_records = {canonical_split: descriptors, requested_split: descriptors}
        return {
            "dataset": dataset_name,
            "validated": True,
            "strategy": "explicit_split",
            "split": requested_split,
            "split_signature": signature,
            "membership_signature": signature,
            "cases": case_counts,
            "case_counts": case_counts,
            "patients": patient_counts,
            "patient_counts": patient_counts,
            "slices": slice_counts,
            "slice_counts": slice_counts,
            "label_values": label_values,
            "mapped_label_values": mapped_label_values,
            "test_available": True,
            "records": len(records),
            "effective_identities": effective_identities,
            "effective_records": effective_records,
            "raw_to_acdc": dict(mapping.raw_to_acdc),
            "content_hash_duplicate_detection": {"executed": False, "status": "deferred"},
        }
    try:
        from self_audit.data.acdc import discover_acdc_records, resolve_effective_acdc_splits
        from self_audit.data.common import load_array, to_depth_first
    except ImportError:
        from src.self_audit.data.acdc import discover_acdc_records, resolve_effective_acdc_splits
        from src.self_audit.data.common import load_array, to_depth_first

    data_root = config.get("data_root", "preprocessed_data/ACDC")
    records = discover_acdc_records(data_root)
    manifest_value = config.get("split_manifest")
    seed = int(config.get("seed", 42))

    effective = resolve_effective_acdc_splits(
        records,
        split_manifest=manifest_value,
        seed=seed,
        train_split=str(config.get("train_split", "train")),
        val_split=str(config.get("val_split", "val")),
        test_split=str(config.get("test_split", "test")),
    )

    configured_depth_axis = validate_depth_axis(config.get("depth_axis"))
    slice_counts: dict[str, int] = {}
    for name, split_records in effective.splits.items():
        total_slices = 0
        for record in split_records:
            volume, _ = load_array(record.image_path)
            axis = configured_depth_axis
            if axis is None and record.source_format == "nifti":
                axis = 2
            total_slices += int(to_depth_first(volume, depth_axis=axis).shape[0])
        slice_counts[name] = total_slices

    patient_counts = {
        name: len({record.patient_id for record in split_records})
        for name, split_records in effective.splits.items()
    }
    case_counts = {
        name: len(split_records)
        for name, split_records in effective.splits.items()
    }
    effective_identities = {
        name: [record.case_id for record in split_records]
        for name, split_records in effective.splits.items()
    }
    effective_descriptors = {
        name: [
            {
                "case_id": record.case_id,
                "patient_id": record.patient_id,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path),
                "split": record.split,
                "source_format": record.source_format,
            }
            for record in split_records
        ]
        for name, split_records in effective.splits.items()
    }

    return {
        "dataset": dataset_name,
        "validated": True,
        "strategy": effective.strategy,
        "split_signature": effective.signature,
        "cases": case_counts,
        "patients": patient_counts,
        "slices": slice_counts,
        "test_available": "test" in effective.splits,
        "records": len(records),
        "effective_identities": effective_identities,
        "effective_records": effective_descriptors,
        "content_hash_duplicate_detection": {"executed": False, "status": "deferred"},
    }


def _state_list(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        if value.ndim == 5:
            return list(value)
        return [value]
    return [item for item in list(value) if torch.is_tensor(item)]


def extract_annotation_states(output: Any, *, include_initial: bool = False) -> list[torch.Tensor]:
    """Extract refinement states, optionally prefixed with the explicit A0 state.

    ``states`` remains the historical refinement-only alias.  Callers that
    need the inference trajectory should use ``include_initial=True`` or
    :func:`extract_annotation_trajectory` so A0 is not accidentally skipped.
    """

    initial: torch.Tensor | None = None
    if isinstance(output, dict):
        initial = extract_initial_logits(output)
        for key in ("states", "refinement_logits", "intermediate_logits"):
            states = output.get(key)
            if states is not None:
                result = _state_list(states)
                if include_initial and initial is not None:
                    return [initial] + [state for state in result if state is not initial]
                return result
        for key in ("logits", "annotation_logits", "output"):
            value = output.get(key)
            if torch.is_tensor(value):
                result = [value]
                if include_initial and initial is not None and initial is not value:
                    return [initial, value]
                return result
    if torch.is_tensor(output):
        return [output]
    raise TypeError("Could not find annotation logits in model output")


def extract_annotation_trajectory(output: Any) -> list[torch.Tensor]:
    """Return ``[A0, A1, ..., A_T]`` without hard-coding the number of turns."""

    if isinstance(output, dict):
        for key in ("state_trace", "all_states", "annotation_trajectory"):
            value = output.get(key)
            if value is not None:
                states = _state_list(value)
                if states:
                    return states
    return extract_annotation_states(output, include_initial=True)


def adjacent_annotation_pairs(states: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build exactly the adjacent transitions represented by a state trace."""

    if len(states) < 2:
        return []
    return list(zip(states[:-1], states[1:]))


def extract_initial_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("initial_logits", "a0_logits", "logits", "annotation_logits", "output"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    if torch.is_tensor(output):
        return output
    raise TypeError("Could not find initial annotation logits in model output")


def encoder_head_optimizer(
    model: nn.Module,
    *,
    lr: float = 3e-4,
    encoder_lr: float = 3e-5,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Create separate parameter groups with a lower LR for pretrained features."""

    encoder_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    encoder = getattr(model, "encoder", None)
    encoder_ids = {id(p) for p in encoder.parameters()} if encoder is not None else set()
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (encoder_params if id(parameter) in encoder_ids else head_params).append(parameter)
    lr = _require_positive_real(lr, "lr")
    encoder_lr = _require_positive_real(encoder_lr, "encoder_lr")
    weight_decay = _require_nonnegative_real(weight_decay, "weight_decay")
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def build_adamw_optimizer(
    parameters: Iterable[nn.Parameter],
    *,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Build a validated AdamW optimizer for a single parameter group."""

    values = list(parameters)
    if not values:
        raise ValueError("AdamW requires at least one trainable parameter")
    if any(not isinstance(parameter, nn.Parameter) for parameter in values):
        raise TypeError("AdamW parameters must be torch.nn.Parameter instances")
    if any(not parameter.requires_grad for parameter in values):
        raise ValueError("AdamW parameter groups must contain only trainable parameters")
    lr = _require_positive_real(lr, "lr")
    weight_decay = _require_nonnegative_real(weight_decay, "weight_decay")
    if len(betas) != 2 or any(not 0.0 <= float(beta) < 1.0 for beta in betas):
        raise ValueError(f"betas must contain two values in [0, 1), got {betas!r}")
    eps = _require_positive_real(eps, "eps")
    return torch.optim.AdamW(values, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


def validate_accumulation_steps(value: Any) -> int:
    return _require_integer(value, "accumulation_steps", minimum=1)


def optimizer_steps_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    num_batches = _require_integer(num_batches, "num_batches", minimum=0)
    accumulation_steps = validate_accumulation_steps(accumulation_steps)
    return int(math.ceil(num_batches / accumulation_steps)) if num_batches else 0


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Schedule optimizer updates with linear warmup followed by cosine decay."""

    total_steps = _require_integer(total_steps, "total_steps", minimum=1)
    warmup_steps = _require_integer(warmup_steps, "warmup_steps", minimum=0)
    if warmup_steps > total_steps:
        raise ValueError("warmup_steps cannot exceed total_steps")
    min_lr_ratio = _require_real(min_lr_ratio, "min_lr_ratio")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(float(step) / float(warmup_steps), 1e-8)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((float(step) - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def build_training_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    *,
    num_batches: int,
    epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Build the configured update-based scheduler for a training loop."""

    if not isinstance(config, Mapping):
        raise ValueError("training scheduler config must be a mapping")
    name = str(config.get("scheduler", "cosine")).strip().lower()
    if name in {"", "none", "constant", "off"}:
        return None
    if name not in {"cosine", "warmup_cosine", "cosine_with_warmup"}:
        raise ValueError(f"Unsupported scheduler {name!r}; use cosine or none")
    epochs = _require_integer(epochs, "epochs", minimum=1)
    accumulation_steps = validate_accumulation_steps(config.get("accumulation_steps", 1))
    updates_per_epoch = optimizer_steps_per_epoch(num_batches, accumulation_steps)
    if updates_per_epoch == 0:
        raise ValueError("Cannot build a scheduler for an empty DataLoader")
    total_steps = updates_per_epoch * epochs
    if "warmup_steps" in config:
        warmup_steps = _require_integer(config["warmup_steps"], "warmup_steps", minimum=0)
    elif "warmup_epochs" in config:
        warmup_epochs = _require_nonnegative_real(config["warmup_epochs"], "warmup_epochs")
        warmup_steps = int(math.ceil(warmup_epochs * updates_per_epoch))
    else:
        warmup_ratio = _require_nonnegative_real(config.get("warmup_ratio", 0.0), "warmup_ratio")
        if warmup_ratio > 1.0:
            raise ValueError("warmup_ratio must be between 0 and 1")
        warmup_steps = int(math.ceil(total_steps * warmup_ratio))
    # Short smoke runs may have fewer updates than the production warmup
    # setting.  Clamp the schedule rather than crashing before the first step.
    warmup_steps = min(warmup_steps, total_steps)
    return build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=float(config.get("min_lr_ratio", 0.0)),
    )


def resolve_amp(config: Mapping[str, Any], device: torch.device) -> tuple[bool, torch.dtype]:
    """Resolve AMP policy, keeping CPU defaults deterministic and safe."""

    if not isinstance(config, Mapping):
        raise ValueError("AMP config must be a mapping")
    raw_enabled = config.get("amp", config.get("mixed_precision", False))
    if isinstance(raw_enabled, str):
        normalized = raw_enabled.strip().lower()
        if normalized == "auto":
            enabled = device.type == "cuda"
        elif normalized in {"true", "1", "yes", "on"}:
            enabled = True
        elif normalized in {"false", "0", "no", "off"}:
            enabled = False
        else:
            raise ValueError(f"Unsupported amp setting: {raw_enabled!r}")
    else:
        enabled = _require_bool(raw_enabled, "amp")
    if device.type not in {"cuda", "cpu"}:
        if enabled:
            raise ValueError(f"AMP is supported only on CUDA/CPU, got device {device}")
        return False, torch.float32
    dtype_name = str(config.get("amp_dtype", "float16" if device.type == "cuda" else "bfloat16")).lower()
    dtype = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported amp_dtype {dtype_name!r}")
    if device.type == "cpu" and enabled and dtype is torch.float16:
        raise ValueError("CPU AMP requires amp_dtype=bfloat16")
    return enabled, dtype


def build_grad_scaler(*, enabled: bool, device: torch.device, dtype: torch.dtype) -> Any:
    """Create a version-compatible scaler; bfloat16 does not need scaling."""

    scale_enabled = bool(enabled and device.type == "cuda" and dtype is torch.float16)
    try:
        return torch.amp.GradScaler("cuda", enabled=scale_enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older PyTorch
        return torch.cuda.amp.GradScaler(enabled=scale_enabled)


def autocast_context(*, enabled: bool, device: torch.device, dtype: torch.dtype) -> Iterator[None]:
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def is_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach()).all().item())
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError):
        return False


def _gradient_norm(model: nn.Module) -> float:
    gradients = [parameter.grad.detach().float() for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        return 0.0
    total = torch.stack([gradient.norm(2).pow(2) for gradient in gradients]).sum().sqrt()
    return float(total)


def finalize_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    pending_batches: int,
    accumulation_steps: int,
    grad_clip: float | None,
) -> tuple[bool, float]:
    """Unscale, normalize a partial accumulation, guard, and step once."""

    pending_batches = _require_integer(pending_batches, "pending_batches", minimum=1)
    accumulation_steps = validate_accumulation_steps(accumulation_steps)
    if pending_batches > accumulation_steps:
        raise ValueError("pending_batches cannot exceed accumulation_steps")
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer)
    if pending_batches < accumulation_steps:
        correction = float(accumulation_steps) / float(pending_batches)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
    if grad_clip is not None:
        grad_clip = _require_positive_real(grad_clip, "grad_clip")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
    else:
        norm = _gradient_norm(model)
    gradients_finite = all(
        is_finite(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    step_ok = math.isfinite(norm) and gradients_finite
    if step_ok:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
        if scheduler is not None:
            scheduler.step()
    elif scaler is not None and scaler.is_enabled():
        # Record the non-finite gradients so GradScaler reduces its scale; the
        # optimizer itself will skip the update.
        scaler.step(optimizer)
    if scaler is not None:
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return step_ok, norm


def _finite_tree(value: Any, name: str) -> None:
    if torch.is_tensor(value):
        if not is_finite(value):
            raise FloatingPointError(f"Non-finite tensor in {name}")
    elif isinstance(value, np.ndarray):
        if not bool(np.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite ndarray in {name}")
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"Non-finite value in {name}: {value}")
    elif isinstance(value, (complex, np.complexfloating)):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise FloatingPointError(f"Non-finite value in {name}: {value}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple, set)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{name}[{index}]")


def validate_checkpoint_finite_state(payload: Mapping[str, Any]) -> None:
    """Validate that model, optimizer, scheduler, and scaler states contain only finite values.

    Inspects tensors and numeric scalars across all present training state components
    ('model', 'state_dict', 'optimizer', 'scheduler', 'scaler').
    Raises FloatingPointError if any non-finite (NaN or Inf) tensor or scalar is detected.
    Does not sanitize or mutate any training state.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint payload must be a mapping, got {type(payload).__name__}")

    # Validate model / state_dict
    if "model" in payload and payload["model"] is not None:
        model_val = payload["model"]
        if isinstance(model_val, nn.Module):
            model_val = model_val.state_dict()
        _finite_tree(model_val, "model")
    elif "state_dict" in payload and payload["state_dict"] is not None:
        model_val = payload["state_dict"]
        if isinstance(model_val, nn.Module):
            model_val = model_val.state_dict()
        _finite_tree(model_val, "state_dict")
    elif payload and all(isinstance(k, str) and torch.is_tensor(v) for k, v in payload.items()):
        _finite_tree(payload, "model")

    # Validate optimizer, scheduler, scaler if present
    for component in ("optimizer", "scheduler", "scaler"):
        if component in payload and payload[component] is not None:
            _finite_tree(payload[component], component)


def _cpu_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            if value.dtype == getattr(torch, "uint32", None):
                raise CheckpointSerializationError(
                    f"Unsupported tensor dtype {value.dtype} at 'model.{key}': not supported for safe weights_only serialization"
                )
            result[str(key)] = value.detach().cpu().clone()
        elif isinstance(value, Mapping):
            result[str(key)] = _cpu_state_dict(value)
        else:
            result[str(key)] = value
    return result


def _rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch RNG states in a portable weights_only format."""
    np_state = np.random.get_state()
    # Lossless int64 NumPy MT keys for safe weights_only serialization on all torch versions
    # (avoids KeyError uint32 on torch <= 2.4.1 and avoids numpy reconstruct UnpicklingError)
    np_keys = torch.from_numpy(np_state[1].astype(np.int64))
    portable_numpy = (
        str(np_state[0]),
        np_keys,
        int(np_state[2]),
        int(np_state[3]),
        float(np_state[4]),
    )
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": portable_numpy,
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [value.clone().cpu() for value in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: Mapping[str, Any], *, exact_cuda: bool = False) -> None:
    """Restore Python, NumPy, and PyTorch RNG states with backward historical compatibility."""
    if not isinstance(state, Mapping):
        raise TypeError(f"rng_state must be a mapping, got {type(state).__name__}")
    if "python" in state:
        py_state = state["python"]
        if py_state is None:
            raise ValueError("Malformed 'python' RNG state: value cannot be None")
        if not isinstance(py_state, (tuple, list)):
            raise TypeError(f"Malformed 'python' RNG state: expected tuple or list, got {type(py_state).__name__}")
        if len(py_state) not in (3, 4):
            raise ValueError(f"Malformed 'python' RNG state: expected 3 or 4 elements, got {len(py_state)}")
        if isinstance(py_state, list):
            py_state = tuple(py_state)
        if isinstance(py_state[1], list):
            py_state = (py_state[0], tuple(py_state[1]), *py_state[2:])
        random.setstate(py_state)
    if "numpy" in state:
        raw_np = state["numpy"]
        if raw_np is None:
            raise ValueError("Malformed 'numpy' RNG state: value cannot be None")
        if isinstance(raw_np, (tuple, list)):
            if len(raw_np) != 5:
                raise ValueError(f"Malformed 'numpy' RNG state tuple: expected 5 elements, got {len(raw_np)}")
            algo, keys, pos, has_gauss, cached_gaussian = raw_np
        elif isinstance(raw_np, Mapping):
            algo = raw_np.get("algorithm", "MT19937")
            keys = raw_np.get("keys")
            if keys is None:
                raise ValueError("NumPy RNG state mapping missing 'keys'")
            pos = raw_np.get("pos", 0)
            has_gauss = raw_np.get("has_gauss", 0)
            cached_gaussian = raw_np.get("cached_gaussian", 0.0)
        else:
            raise TypeError(f"Malformed 'numpy' RNG state: expected tuple, list, or mapping, got {type(raw_np).__name__}")

        if str(algo) != "MT19937":
            raise ValueError(f"Unsupported NumPy RNG algorithm: {algo}")

        # Validate and extract keys safely through CPU NumPy without torch.uint32 comparison operators
        if torch.is_tensor(keys):
            if keys.ndim != 1 or keys.shape[0] != 624:
                raise ValueError(f"NumPy MT19937 keys must have length 624, got shape {tuple(keys.shape)}")
            keys_arr = keys.detach().cpu().numpy()
        elif isinstance(keys, np.ndarray):
            if keys.ndim != 1 or keys.shape[0] != 624:
                raise ValueError(f"NumPy MT19937 keys must have length 624, got shape {keys.shape}")
            keys_arr = keys
        elif isinstance(keys, (list, tuple)):
            if len(keys) != 624:
                raise ValueError(f"NumPy MT19937 keys must have length 624, got {len(keys)}")
            for idx, val in enumerate(keys):
                if isinstance(val, (bool, np.bool_)):
                    raise ValueError(f"NumPy MT19937 key at index {idx} cannot be boolean: {val!r}")
                if isinstance(val, (complex, np.complexfloating)):
                    raise ValueError(f"NumPy MT19937 key at index {idx} cannot be complex: {val!r}")
                if isinstance(val, (float, np.floating)):
                    if not math.isfinite(float(val)):
                        raise ValueError(f"NumPy MT19937 key at index {idx} must be finite: {val!r}")
                    if float(val) != math.floor(float(val)):
                        raise ValueError(f"NumPy MT19937 key at index {idx} cannot be fractional: {val!r}")
                    val = int(val)
                elif not isinstance(val, (int, np.integer)):
                    raise TypeError(f"NumPy MT19937 key at index {idx} must be an integer, got {type(val).__name__}")
                if val < 0 or val > 4294967295:
                    raise ValueError(f"NumPy MT19937 key at index {idx} outside uint32 range [0, 4294967295]: {val}")
            keys_arr = np.array(keys, dtype=np.uint32)
        else:
            raise TypeError(f"Unsupported type for NumPy MT19937 keys: {type(keys).__name__}")

        # Require keys NumPy dtype kind in integer ('i'), unsigned integer ('u'), or real float ('f')
        # Reject object ('O'), complex ('c'), bool ('b'), string ('U'/'S'), etc. before any comparison or cast
        if keys_arr.dtype.kind not in ("i", "u", "f"):
            raise ValueError(
                f"NumPy MT19937 keys must have integer or real float dtype, got kind '{keys_arr.dtype.kind}' ({keys_arr.dtype})"
            )

        if keys_arr.dtype.kind == "f":
            if not np.isfinite(keys_arr).all():
                raise ValueError("NumPy MT19937 keys contain non-finite values")
            if not np.equal(keys_arr, np.floor(keys_arr)).all():
                raise ValueError("NumPy MT19937 keys contain fractional values")

        if keys_arr.dtype == np.uint32:
            keys_np = keys_arr
        else:
            if (keys_arr < 0).any() or (keys_arr > 4294967295).any():
                raise ValueError("NumPy MT19937 keys contain values outside valid uint32 range [0, 4294967295]")
            keys_np = keys_arr.astype(np.uint32)

        if isinstance(pos, (bool, np.bool_)):
            raise ValueError(f"NumPy MT19937 pos cannot be boolean, got {pos!r}")
        if isinstance(pos, (float, np.floating)):
            if not math.isfinite(float(pos)):
                raise ValueError(f"NumPy MT19937 pos must be finite, got {pos!r}")
            if float(pos) != math.floor(float(pos)):
                raise ValueError(f"NumPy MT19937 pos cannot be fractional, got {pos!r}")
            pos = int(pos)
        elif isinstance(pos, (int, np.integer)):
            pos = int(pos)
        else:
            raise TypeError(f"NumPy MT19937 pos must be an integer, got {type(pos).__name__}")
        if pos < 0 or pos > 624:
            raise ValueError(f"NumPy MT19937 pos must be in [0, 624], got {pos}")

        if isinstance(has_gauss, (bool, np.bool_)):
            raise ValueError(f"NumPy has_gauss cannot be boolean, got {has_gauss!r}")
        if isinstance(has_gauss, (float, np.floating)):
            if not math.isfinite(float(has_gauss)):
                raise ValueError(f"NumPy has_gauss must be finite, got {has_gauss!r}")
            if float(has_gauss) != math.floor(float(has_gauss)):
                raise ValueError(f"NumPy has_gauss cannot be fractional, got {has_gauss!r}")
            has_gauss = int(has_gauss)
        elif isinstance(has_gauss, (int, np.integer)):
            has_gauss = int(has_gauss)
        else:
            raise TypeError(f"NumPy has_gauss must be an integer, got {type(has_gauss).__name__}")
        if has_gauss not in (0, 1):
            raise ValueError(f"NumPy has_gauss must be 0 or 1, got {has_gauss}")

        if isinstance(cached_gaussian, (bool, np.bool_)):
            raise ValueError(f"NumPy cached_gaussian cannot be boolean, got {cached_gaussian!r}")
        if not isinstance(cached_gaussian, (float, int, np.floating, np.integer)):
            raise TypeError(f"NumPy cached_gaussian must be a real number, got {type(cached_gaussian).__name__}")
        cached_gaussian = float(cached_gaussian)
        if not math.isfinite(cached_gaussian):
            raise ValueError(f"NumPy cached_gaussian must be finite, got {cached_gaussian}")

        np.random.set_state((str(algo), keys_np, pos, has_gauss, cached_gaussian))

    if "torch" in state:
        torch_state = state["torch"]
        if torch_state is None:
            raise ValueError("Malformed 'torch' RNG state: value cannot be None")
        if not torch.is_tensor(torch_state):
            raise TypeError(f"Malformed 'torch' RNG state: expected Tensor, got {type(torch_state).__name__}")
        torch.set_rng_state(torch_state.cpu())

    if exact_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA RNG restore requested but CUDA is not available")
    if torch.cuda.is_available():
        if "cuda" in state:
            cuda_states = state["cuda"]
            if cuda_states is None:
                raise ValueError("Malformed 'cuda' RNG state: value cannot be None")
            if not isinstance(cuda_states, (list, tuple)):
                raise TypeError(f"cuda RNG state must be a list or tuple of tensors, got {type(cuda_states).__name__}")
            dev_count = torch.cuda.device_count()
            if len(cuda_states) != dev_count:
                raise ValueError(f"Checkpoint contains {len(cuda_states)} CUDA RNG state(s), but {dev_count} device(s) are available")
            for idx, s in enumerate(cuda_states):
                if not torch.is_tensor(s):
                    raise TypeError(f"CUDA RNG state for device {idx} must be a Tensor, got {type(s).__name__}")
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
        elif exact_cuda:
            raise ValueError("Missing 'cuda' RNG state in checkpoint for exact GPU resume")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    optimizer_step: int = 0,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a portable, finite, weights-only-loadable checkpoint."""

    path = Path(path)
    epoch = _require_integer(epoch, "epoch", minimum=0)
    global_step = _require_integer(global_step, "global_step", minimum=0)
    optimizer_step = _require_integer(optimizer_step, "optimizer_step", minimum=0)
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    if extra is not None and not isinstance(extra, Mapping):
        raise ValueError("checkpoint extra must be a mapping")
    model_state = _cpu_state_dict(model.state_dict())
    # The producing revision and the digest of the bytes actually written are
    # stamped here, at save time.  Nothing downstream can reconstruct them
    # later without guessing, and a guess is exactly what W3 forbids.
    provenance = producer_provenance_record()
    provenance["state_digest"] = state_digest(model_state)
    provenance["model_identity"] = resolve_model_identity(model)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model_state,
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "rng_state": _rng_state(),
        "provenance": provenance,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if config is not None:
        if not isinstance(config, Mapping):
            raise ValueError("checkpoint config must be a mapping")
        payload["config"] = normalize_metadata_tree(dict(config), "config")
    if extra is not None:
        if not isinstance(extra, Mapping):
            raise ValueError("checkpoint extra must be a mapping")
        normalized_extra = normalize_metadata_tree(dict(extra), "extra")
        reserved = set(payload).intersection(normalized_extra)
        if reserved:
            raise ValueError(f"Checkpoint extra uses reserved key(s): {sorted(reserved)}")
        payload.update(normalized_extra)
    validate_checkpoint_finite_state(payload)
    # atomic_save_torch safely normalizes the payload to weights_only types and writes atomically
    return atomic_save_torch(payload, path)


def _extract_model_state(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping, got {type(payload).__name__}")
    if "model" in payload:
        model_state = payload["model"]
    elif "state_dict" in payload:
        model_state = payload["state_dict"]
    elif payload and all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()):
        model_state = payload
    else:
        raise ValueError("Checkpoint contains no model/state_dict tensor mapping")
    if not isinstance(model_state, Mapping) or not all(isinstance(key, str) for key in model_state):
        raise ValueError("Checkpoint model state must be a string-keyed mapping")
    return dict(model_state), dict(payload)


def _checkpoint_counter(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key, 0)
    return _require_integer(value, f"checkpoint.{key}", minimum=0)


def _model_architecture_signature(model: nn.Module) -> dict[str, int | str | bool]:
    expert = getattr(model, "annotation_expert", None)
    window = getattr(expert, "refinement_block", None)
    encoder = getattr(model, "encoder", None)
    return {
        "num_classes": int(getattr(model, "num_classes", getattr(expert, "num_classes", -1))),
        "shared_channels": int(getattr(getattr(model, "fpn", None), "out_channels", -1)),
        "window_k": int(getattr(window, "k", -1)),
        "encoder_name": str(getattr(encoder, "name", "")),
    }


def _validate_checkpoint_architecture(payload: Mapping[str, Any], model: nn.Module) -> None:
    config = payload.get("config")
    if not isinstance(config, Mapping):
        return
    configured = config.get("model", config)
    if not isinstance(configured, Mapping):
        return
    actual = _model_architecture_signature(model)
    mismatches: list[str] = []
    for key in ("num_classes", "shared_channels", "window_k", "encoder_name"):
        if key in configured and actual[key] not in {-1, ""} and str(configured[key]) != str(actual[key]):
            mismatches.append(f"{key}: checkpoint={configured[key]!r}, model={actual[key]!r}")
    if mismatches:
        raise ValueError(
            "Incompatible checkpoint architecture; refusing to load: "
            + "; ".join(mismatches)
        )


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
    exact_cuda: bool | None = None,
) -> dict[str, Any]:
    """Load a checkpoint with safe deserialization and optional state restore."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError(f"Unable to safely load checkpoint {path}: {exc}") from exc
    model_state, normalized = _extract_model_state(payload)
    normalized["model"] = model_state
    validate_checkpoint_finite_state(normalized)
    for key in ("epoch", "global_step", "optimizer_step"):
        _checkpoint_counter(normalized, key)
    if model is not None:
        _validate_checkpoint_architecture(normalized, model)
        model.load_state_dict(model_state, strict=strict)
    if optimizer is not None and "optimizer" in normalized:
        optimizer.load_state_dict(normalized["optimizer"])
    if scheduler is not None and "scheduler" in normalized:
        scheduler.load_state_dict(normalized["scheduler"])
    if scaler is not None and "scaler" in normalized:
        scaler.load_state_dict(normalized["scaler"])
    if restore_rng and isinstance(normalized.get("rng_state"), Mapping):
        if exact_cuda is None:
            if torch.cuda.is_available() and (
                (map_location is not None and "cuda" in str(map_location))
                or (model is not None and any(p.is_cuda for p in model.parameters()))
            ):
                exact_cuda = True
            else:
                exact_cuda = False
        _restore_rng_state(normalized["rng_state"], exact_cuda=exact_cuda)
    return normalized


def _select_checkpoint_candidate(candidates: Iterable[tuple[str, Any]]) -> tuple[int, str, Path, list[str]]:
    """Pick the first existing ``(role, path)`` pair, or refuse.

    Falling back to a later candidate is a *declared* choice reported through
    the index, never a silent substitution, and no candidate at all is an
    error rather than an evaluation of whatever weights happened to be live.
    """

    considered: list[str] = []
    selected: tuple[int, str, Path] | None = None
    for index, (role, raw_path) in enumerate(candidates):
        path = Path(raw_path)
        considered.append(f"{role}:{path}")
        if selected is None and path.is_file():
            selected = (index, str(role), path)
    if selected is None:
        raise FileNotFoundError(
            "No evaluation checkpoint available; considered: " + ", ".join(considered)
        )
    index, role, path = selected
    return index, role, path, considered


def _checkpoint_binding_from_payload(
    model: nn.Module,
    payload: Mapping[str, Any],
    *,
    path: Path,
    role: str,
    config: Mapping[str, Any] | None,
    fallback_used: bool,
    considered: list[str],
    restored: tuple[str, ...],
) -> CheckpointBinding:
    """Describe the binding between a checkpoint payload and the live model.

    The live digest must equal the digest of the state read off disk; that
    equality, not the file hash, is what proves the measured weights are the
    selected weights.  A payload whose recorded digest disagrees with its own
    tensors is refused as tampered.
    """

    identity = verify_model_config(model, config)
    file_digest = state_digest(payload["model"])
    live_digest = state_digest(model)
    if live_digest != file_digest:
        raise ValueError(
            f"Checkpoint {path} did not bind: live model state digest {live_digest} "
            f"differs from checkpoint state digest {file_digest}"
        )
    producer = checkpoint_producer(payload)
    recorded = producer.get("producer_state_digest")
    if recorded is not None and str(recorded) != file_digest:
        raise ValueError(
            f"Checkpoint {path} carries state digest {recorded} but its stored tensors "
            f"digest to {file_digest}; refusing to bind a tampered checkpoint"
        )
    return CheckpointBinding(
        path=path,
        role=role,
        checkpoint_sha256=_provenance_file_sha256(path),
        state_digest=live_digest,
        file_state_digest=file_digest,
        producer=producer,
        model_identity=identity,
        epoch=_checkpoint_counter(payload, "epoch"),
        global_step=_checkpoint_counter(payload, "global_step"),
        fallback_used=fallback_used,
        considered=tuple(considered),
        restored=restored,
    )


#: Key under which the trainer records the immutable selected-best reference in
#: ``last.pt``.  ``checkpoint_commit`` owns the reference format; this module
#: only reads it, and imports nothing from ``unified_trainer``, so no import
#: cycle is introduced (``checkpoint_commit`` is stdlib-only).
COMMITTED_BEST_REFERENCE_KEY = "best_reference"
LAST_CHECKPOINT_NAME = "last.pt"


def _require_committed_best_alias(path: Path) -> None:
    """Refuse to evaluate a public ``best.pt`` that is not the committed selection.

    The Wave 4 commit protocol writes the selected snapshot under
    ``selected_best/``, records a hash-verified reference to it in ``last.pt``,
    and only then republishes the public ``best.pt`` alias.  A crash between
    those last two steps leaves ``last.pt`` correct and ``best.pt`` stale --
    which is recoverable, but only by *resuming from* ``last.pt``: the resume
    path resolves the committed reference and republishes the alias.

    Evaluation must not perform that repair.  Evaluation is frozen: it measures
    the weights it was pointed at, so the honest response to a stale alias is to
    refuse and say what to run, never to quietly rewrite ``best.pt`` and then
    report a number for different weights than the operator asked about.

    Compatibility is deliberate:

    * A basename other than ``best.pt`` (the historical ``phase_c_best.pt`` and
      friends) is not part of this protocol and is left alone.
    * A standalone copied ``best.pt`` with no sibling ``last.pt`` still binds.
    * A ``last.pt`` that predates the protocol -- no
      ``best_reference`` key -- still binds.
    * An explicit ``best_reference: None`` means the run recorded that it had
      *no* committed selection, so a ``best.pt`` sitting next to it is
      unexplained and is refused.

    The sibling is read on CPU with ``weights_only=True`` and no pickle
    fallback, and that read is **fail-closed**: a ``last.pt`` that exists but
    cannot be safe-loaded raises, because unreadable bytes are not evidence
    that the reference key is absent.  A corrupted sibling could just as easily
    commit a selection this alias contradicts, which is precisely the case the
    guard exists to catch.  Only the *absence* of a sibling, or a readable
    sibling with no reference key, is historical compatibility.
    """

    if path.name != PUBLIC_BEST_NAME:
        return
    last_path = path.parent / LAST_CHECKPOINT_NAME
    if not last_path.is_file():
        return

    try:
        last_payload = torch.load(last_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        # Fail closed. Unreadable bytes are not evidence that the key is
        # absent: a sibling that cannot be safe-loaded could equally be a
        # corrupted last.pt that *does* commit a selection this alias
        # contradicts. Inferring "historical, no reference" from an unsafe load
        # would let exactly the stale alias this guard exists to catch through.
        raise ValueError(
            f"Refusing to bind {path}: sibling {last_path} exists but could not be safely read "
            f"with weights_only=True ({exc}), so the committed best selection cannot be verified. "
            "An unreadable sibling is not treated as one without a committed selection. Repair or "
            f"remove {last_path.name}, or evaluate a standalone copy of the checkpoint in a "
            "directory with no sibling last.pt."
        ) from exc

    if not isinstance(last_payload, Mapping):
        raise ValueError(
            f"Refusing to bind {path}: sibling {last_path} did not load as a mapping "
            f"({type(last_payload).__name__}), so the committed best selection cannot be verified."
        )
    if COMMITTED_BEST_REFERENCE_KEY not in last_payload:
        return

    reference = last_payload[COMMITTED_BEST_REFERENCE_KEY]
    if reference is None:
        raise ValueError(
            f"Refusing to bind {path}: sibling {last_path} explicitly records no committed best "
            f"selection ({COMMITTED_BEST_REFERENCE_KEY}=None), so the weights in {path.name} are "
            "not an accountable selection. Evaluate the last checkpoint instead, or resume the "
            "run so a best selection is committed."
        )

    try:
        resolve_best_reference(reference, path.parent)
    except SelectionError as exc:
        raise ValueError(
            f"Refusing to bind {path}: the committed best reference in {last_path} does not "
            f"verify ({exc}). The immutable snapshot it names must exist and match its recorded "
            "hash before its alias can be evaluated."
        ) from exc

    alias_digest = _provenance_file_sha256(path)
    committed_digest = str(reference["sha256"])
    if alias_digest != committed_digest:
        raise ValueError(
            f"Refusing to bind {path}: the public best alias has sha256 {alias_digest} but "
            f"{last_path} commits selected-best sha256 {committed_digest}. The alias is stale, "
            "which is what an interrupted alias publication leaves behind; resume from "
            f"{last_path.name}, which resolves the committed reference and republishes the alias. "
            "Evaluation will not repair it, because it must measure exactly the weights it was "
            "pointed at."
        )


def bind_evaluation_checkpoint(
    model: nn.Module,
    candidates: Iterable[tuple[str, Any]],
    *,
    map_location: str | torch.device | None = "cpu",
    config: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> CheckpointBinding:
    """Load the first available candidate and bind the evaluated state to it.

    ``candidates`` is an ordered sequence of ``(role, path)`` pairs, typically
    ``[("best", .../phase_c_best.pt), ("last", .../phase_c_last.pt)]``.  The
    first existing file wins; using the later one is a *declared* fallback
    (``fallback_used``), not a silent substitution.  With no candidate present
    the caller gets ``FileNotFoundError`` rather than an evaluation of whatever
    weights happened to be live.

    This is an evaluation-only load: the optimizer, scheduler, scaler and RNG
    state in the checkpoint are deliberately **not** restored, so binding a
    checkpoint cannot perturb the process's random stream.

    After loading, the digest of the live model is compared with the digest of
    the state read off disk.  They must be equal -- that equality, not the file
    hash, is what proves the measured weights are the selected weights.
    """

    index, role, path, considered = _select_checkpoint_candidate(candidates)
    _require_committed_best_alias(path)
    # Stage full checkpoint payload on CPU so calibration/external do not allocate unused optimizer/RNG state on GPU
    payload = load_checkpoint(
        path,
        map_location="cpu",
        strict=strict,
        restore_rng=False,
    )
    _validate_checkpoint_architecture(payload, model)
    model.load_state_dict(payload["model"], strict=strict)
    return _checkpoint_binding_from_payload(
        model,
        payload,
        path=path,
        role=role,
        config=config,
        fallback_used=index > 0,
        considered=considered,
        restored=("model",),
    )


def bind_existing_evaluation_state(
    model: nn.Module,
    candidates: Iterable[tuple[str, Any]],
    *,
    map_location: str | torch.device | None = "cpu",
    config: Mapping[str, Any] | None = None,
) -> CheckpointBinding:
    """Bind the checkpoint the live model *already is*, loading nothing.

    Used where a checkpoint's identity is needed but overwriting the live
    weights would change what the process is doing -- notably before Phase C
    starts, where loading a checkpoint would discard the Phase A/B weights the
    single-process pipeline is carrying.  The live state must already equal the
    checkpoint's; if it does not, that is an error the operator has to resolve,
    not something to paper over by loading.

    The full weights_only checkpoint payload is staged on CPU so calibration/external
    evaluators do not allocate unused optimizer or RNG state in GPU memory; the live
    model remains on its requested target device.

    ``restored`` is empty: no tensor, optimizer, scheduler or RNG state is
    touched.
    """

    index, role, path, considered = _select_checkpoint_candidate(candidates)
    _require_committed_best_alias(path)
    # Stage full checkpoint payload on CPU so calibration/external do not allocate unused optimizer/RNG state on GPU
    payload = load_checkpoint(path, map_location="cpu", restore_rng=False)
    live_digest = state_digest(model)
    file_digest = state_digest(payload["model"])
    if live_digest != file_digest:
        raise ValueError(
            f"Live model state does not match checkpoint {path}: live digest {live_digest} "
            f"differs from checkpoint digest {file_digest}. Resume from that checkpoint, or "
            "drop the option that requires the run to already be at those weights; this "
            "function will not overwrite the live weights to force a match."
        )
    return _checkpoint_binding_from_payload(
        model,
        payload,
        path=path,
        role=role,
        config=config,
        fallback_used=index > 0,
        considered=considered,
        restored=(),
    )


def verify_bound_state(model: nn.Module, binding: CheckpointBinding, *, boundary: str) -> str:
    """Re-check at a consumer boundary that the bound state is still live.

    Called before the cache collector, before calibration and before each
    diagnostic.  Any mutation of the module between binding and use -- a stray
    optimizer step, a reload, a dtype cast -- changes the digest and fails
    here instead of silently producing a measurement of something else.
    """

    observed = state_digest(model)
    if observed != binding.state_digest:
        raise ValueError(
            f"Model state changed after checkpoint binding at boundary {boundary!r}: "
            f"expected digest {binding.state_digest}, observed {observed} "
            f"(bound checkpoint: {binding.path})"
        )
    return observed


def checkpoint_progress(payload: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return completed epoch, global batch step, and optimizer step."""

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return (
        _checkpoint_counter(payload, "epoch"),
        _checkpoint_counter(payload, "global_step"),
        _checkpoint_counter(payload, "optimizer_step"),
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class WandbLogger:
    """Wrapper around Weights & Biases with safe local offline mode and graceful fallback.

    Run identity is reported, never guessed.  ``run_id`` and ``resume`` are
    optional, so every existing call site is unaffected, and what the wrapper
    *requested* is always kept separate from what actually took effect:

    * An explicit ``run_id`` is forwarded to ``wandb.init`` in any mode.
    * ``resume`` is forwarded **only** in ``online`` mode.  Offline and
      disabled modes have no backend to resume against, so a requested resume
      is recorded and explicitly reported as not performed -- reusing the same
      id offline does not continue a backend run, and this wrapper will not
      pretend it does.
    * ``actual_run_id`` comes from the id on the run object the SDK returned,
      and only when that really is a string.  It is never copied from the
      requested id and never invented.

    :attr:`identity_summary` renders all of that as a flat dict of
    ``str``/``bool``/``int``/``None`` values, so the trainer can persist it in
    checkpoint metadata or a JSON report.  No logger object and no exception
    object ever leaves through it.
    """

    #: Values ``wandb.init(resume=...)`` documents (docs.wandb.ai/ref/python/init):
    #: ``allow``, ``never``, ``must``, ``auto``.  ``True``/``False`` are
    #: documented as deprecated and are refused here, so a stale bool cannot
    #: quietly turn a resume into a fresh run.
    ALLOWED_RESUME_VALUES = ("allow", "never", "must", "auto")

    IDENTITY_SCHEMA_VERSION = 1

    def __init__(
        self,
        enabled: bool = False,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
        group: str | None = None,
        tags: Iterable[str] | None = None,
        config: Mapping[str, Any] | None = None,
        mode: str | None = None,
        dir: str | Path | None = None,
        run_id: str | None = None,
        resume: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.enabled_requested = bool(enabled)
        self.project = project or "self-audit"
        self.entity = entity
        self.run_name = run_name
        self.group = group
        self.tags = list(tags) if tags is not None else None
        self.config = clean_wandb_payload(dict(config)) if config is not None else {}
        self.mode = mode or "offline"
        self.dir = str(dir) if dir is not None else None
        self._run = None
        self.last_error: Exception | None = None
        self.failed_log_count: int = 0
        self.telemetry_errors: list[str] = []

        self.requested_run_id = self._validate_run_id(run_id)
        self.requested_resume = self._validate_resume(resume)
        self.actual_run_id: str | None = None
        self.effective_resume: str | None = None
        self.backend_resume_performed: bool = False
        self.resume_limitation: str | None = None
        self.sdk_available: bool | None = None
        self.sdk_reported_resumed: bool | None = None
        self.init_status: str = "not_attempted"
        self.init_error: str | None = None
        self.finish_status: str = "not_finished"
        self.finish_count: int = 0
        self._finished: bool = False

        if self.requested_resume is not None and not self._is_online:
            self.resume_limitation = (
                f"resume={self.requested_resume!r} was requested but mode={self.mode!r} has no "
                "backend run to resume; the value was not forwarded to wandb.init and reusing a "
                "run id locally does not continue a backend run"
            )

        if self.enabled:
            self._initialize()

    # -- identity helpers ---------------------------------------------------

    @property
    def _is_online(self) -> bool:
        return str(self.mode).strip().lower() == "online"

    @staticmethod
    def _validate_run_id(run_id: Any) -> str | None:
        if run_id is None:
            return None
        if isinstance(run_id, bool) or not isinstance(run_id, str):
            raise ValueError(
                f"run_id must be a string or None, got {type(run_id).__name__}: {run_id!r}"
            )
        value = run_id.strip()
        if not value:
            raise ValueError("run_id must not be blank; pass None to let W&B allocate one")
        return value

    @classmethod
    def _validate_resume(cls, resume: Any) -> str | None:
        if resume is None:
            return None
        if isinstance(resume, bool) or not isinstance(resume, str):
            raise ValueError(
                f"resume must be one of {list(cls.ALLOWED_RESUME_VALUES)} or None, got "
                f"{type(resume).__name__}: {resume!r} (wandb documents True/False as deprecated)"
            )
        value = resume.strip().lower()
        if value not in cls.ALLOWED_RESUME_VALUES:
            raise ValueError(
                f"resume must be one of {list(cls.ALLOWED_RESUME_VALUES)} or None, got {resume!r}"
            )
        return value

    @staticmethod
    def _reported_resumed(run: Any) -> bool | None:
        """Read ``run.resumed``, the SDK's own answer, only when it is a real bool.

        A matching run id is *not* evidence of a resume: ``resume="allow"``
        creates a new run with that id when none exists, and ``resume="auto"``
        starts fresh when there is nothing to recover.  ``run.resumed`` is the
        only explicit proof; anything else is unknown and is never reported as
        a resume.
        """

        candidate = getattr(run, "resumed", None)
        if isinstance(candidate, bool):
            return candidate
        return None

    @staticmethod
    def _reported_run_id(run: Any) -> str | None:
        """Read the id off the returned run, only when it is genuinely a string.

        A mocked or partial SDK returns something that is not a string; that is
        an unknown id, and it is reported as ``None`` rather than coerced.
        """

        candidate = getattr(run, "id", None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        return None

    @property
    def identity_summary(self) -> dict[str, Any]:
        """A flat, checkpoint-safe description of requested versus actual identity."""

        return {
            "schema_version": self.IDENTITY_SCHEMA_VERSION,
            "enabled_requested": bool(self.enabled_requested),
            "enabled_effective": bool(self.enabled),
            "mode": str(self.mode),
            "requested_run_id": self.requested_run_id,
            "actual_run_id": self.actual_run_id,
            "requested_resume": self.requested_resume,
            "effective_resume": self.effective_resume,
            "backend_resume_performed": bool(self.backend_resume_performed),
            "resume_limitation": self.resume_limitation,
            "sdk_available": self.sdk_available,
            "init_status": str(self.init_status),
            "init_error": self.init_error,
            "finish_status": str(self.finish_status),
            "finish_count": int(self.finish_count),
            "failed_log_count": int(self.failed_log_count),
            "telemetry_error_count": len(self.telemetry_errors),
        }

    def _disable_after_failure(self, status: str, message: str | None) -> None:
        self.init_status = status
        self.init_error = message
        self.enabled = False
        self._run = None
        self.effective_resume = None
        self.backend_resume_performed = False
        self.sdk_reported_resumed = None
        if self.requested_resume is not None and self.resume_limitation is None:
            self.resume_limitation = (
                f"resume={self.requested_resume!r} was requested but W&B initialization did not "
                f"succeed ({status}); no backend resume was performed"
            )

    def _initialize(self) -> None:
        try:
            import wandb
        except ImportError:
            print("[wandb] wandb is not installed. Disabling wandb logging. (Install with `pip install wandb`)")
            self.sdk_available = False
            self._disable_after_failure("sdk_missing", "wandb is not installed")
            return

        init = getattr(wandb, "init", None)
        if not callable(init):
            # A namespace package shadowing the SDK (for instance a local
            # ./wandb run-output directory on sys.path) imports cleanly but has
            # no init; that is a missing SDK, not a backend failure.
            print("[wandb] wandb module has no callable init. Disabling wandb logging.")
            self.sdk_available = False
            self._disable_after_failure(
                "sdk_missing", "imported wandb module exposes no callable init"
            )
            return

        self.sdk_available = True
        init_kwargs: dict[str, Any] = {
            "project": self.project,
            "entity": self.entity,
            "name": self.run_name,
            "group": self.group,
            "tags": self.tags,
            "config": self.config,
            "mode": self.mode,
            "dir": self.dir,
        }
        if self.requested_run_id is not None:
            init_kwargs["id"] = self.requested_run_id
        forwarded_resume: str | None = None
        if self.requested_resume is not None and self._is_online:
            forwarded_resume = self.requested_resume
            init_kwargs["resume"] = forwarded_resume

        try:
            self._run = init(**init_kwargs)
        except Exception as exc:
            self.last_error = exc
            self.failed_log_count += 1
            self.telemetry_errors.append(f"init: {exc}")
            print(f"[wandb] failed to initialize wandb ({exc}). Proceeding with wandb disabled.")
            self._disable_after_failure("failed", str(exc))
            return

        self.init_status = "ok"
        self.effective_resume = forwarded_resume
        self.actual_run_id = self._reported_run_id(self._run)
        # Only the SDK's own ``run.resumed`` proves a resume.  A matching id
        # proves nothing: resume="allow" creates a new run under that id when
        # none exists, and resume="auto" starts fresh with nothing to recover.
        self.sdk_reported_resumed = self._reported_resumed(self._run)
        self.backend_resume_performed = bool(
            forwarded_resume is not None and self.sdk_reported_resumed is True
        )
        if (
            forwarded_resume is not None
            and not self.backend_resume_performed
            and self.resume_limitation is None
        ):
            if self.sdk_reported_resumed is False:
                self.resume_limitation = (
                    f"resume={forwarded_resume!r} was forwarded to wandb.init but the SDK reports "
                    f"run.resumed=False: a new backend run was created (returned id "
                    f"{self.actual_run_id!r}), not a continuation of {self.requested_run_id!r}"
                )
            else:
                self.resume_limitation = (
                    f"resume={forwarded_resume!r} was forwarded to wandb.init but the SDK did not "
                    f"report run.resumed, so backend resume is unconfirmed (returned id "
                    f"{self.actual_run_id!r}); a matching run id is not evidence of a resume"
                )
        name_str = getattr(self._run, "name", self.run_name)
        print(f"[wandb] initialized: project={self.project} run={name_str} mode={self.mode}")

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            import wandb
            clean_metrics = clean_wandb_payload(dict(metrics))
            if clean_metrics:
                if step is not None:
                    wandb.log(clean_metrics, step=int(step))
                else:
                    wandb.log(clean_metrics)
        except Exception as exc:
            self.last_error = exc
            self.failed_log_count += 1
            self.telemetry_errors.append(f"log: {exc}")
            print(f"[wandb] warning: failed to log metrics ({exc})")

    def set_summary(self, summary_dict: Mapping[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            import wandb
            cleaned = clean_wandb_payload(dict(summary_dict))
            summary_target = getattr(self._run, "summary", None)
            if summary_target is not None:
                for key, val in cleaned.items():
                    summary_target[key] = val
        except Exception as exc:
            self.last_error = exc
            self.failed_log_count += 1
            self.telemetry_errors.append(f"set_summary: {exc}")
            print(f"[wandb] warning: failed to set summary ({exc})")

    def log_images(self, images: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            import wandb
            payload: dict[str, Any] = {}
            for key, val in images.items():
                if isinstance(val, (str, Path)) and Path(val).is_file():
                    payload[key] = wandb.Image(str(val))
                elif hasattr(val, "savefig"):
                    payload[key] = wandb.Image(val)
            if payload:
                if step is not None:
                    wandb.log(payload, step=int(step))
                else:
                    wandb.log(payload)
        except Exception as exc:
            self.last_error = exc
            self.failed_log_count += 1
            self.telemetry_errors.append(f"log_images: {exc}")
            print(f"[wandb] warning: failed to log images ({exc})")

    def finish(self, exit_code: int | None = None) -> None:
        """Close the run once.

        Idempotent: a second call is a no-op, so a lifecycle that finishes in
        both a normal path and a failure path cannot double-count telemetry
        failures or overwrite a recorded ``finish_status``.
        """

        if self._finished:
            return
        if self.enabled and self._run is not None:
            self._finished = True
            self.finish_count += 1
            try:
                import wandb
                if exit_code is not None:
                    wandb.finish(exit_code=int(exit_code))
                else:
                    wandb.finish()
                self.finish_status = "ok"
            except Exception as exc:
                self.last_error = exc
                self.telemetry_errors.append(f"finish: {exc}")
                self.finish_status = "failed"
                print(f"[wandb] warning: failed to finish run ({exc})")
            self._run = None


def add_wandb_and_tqdm_args(parser: argparse.ArgumentParser) -> None:
    """Add standard wandb, tqdm, and visualization CLI flags to an argument parser."""
    wandb_group = parser.add_argument_group("WandB and Progress Tracking")
    wandb_group.add_argument("--wandb", action="store_true", default=None, help="Enable Weights & Biases logging")
    wandb_group.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    wandb_group.add_argument("--wandb_project", default=None, help="WandB project name")
    wandb_group.add_argument("--wandb_entity", default=None, help="WandB entity/user/team")
    wandb_group.add_argument("--wandb_run_name", default=None, help="WandB run name")
    wandb_group.add_argument("--wandb_group", default=None, help="WandB group name")
    wandb_group.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default=None, help="WandB run mode (default: offline)")
    wandb_group.add_argument("--wandb_tags", default=None, help="Comma-separated tags for WandB")
    wandb_group.add_argument("--no_tqdm", action="store_true", help="Disable interactive tqdm progress bars")

    vis_group = parser.add_argument_group("Visual Inspection & Image Export")
    vis_group.add_argument("--visualize", action="store_true", help="Export visualization figures after validation/training")
    vis_group.add_argument("--vis_dir", default="reports/visualizations", help="Output directory for exported visualization figures")
    vis_group.add_argument("--vis_samples", type=int, default=4, help="Maximum number of validation samples to visualize")


def setup_wandb_logger(
    args: Any,
    config: Mapping[str, Any],
    *,
    phase: str,
    default_project: str = "self-audit",
) -> WandbLogger:
    """Helper to initialize WandbLogger from argparse args and YAML config."""
    wandb_cfg = config.get("wandb", {})
    if not isinstance(wandb_cfg, Mapping):
        wandb_cfg = {}

    cli_wandb = getattr(args, "wandb", None)
    cli_no_wandb = getattr(args, "no_wandb", False)
    if cli_no_wandb:
        enabled = False
    elif cli_wandb is not None and cli_wandb:
        enabled = True
    else:
        enabled = bool(wandb_cfg.get("enabled", False))

    project = getattr(args, "wandb_project", None) or wandb_cfg.get("project") or default_project
    entity = getattr(args, "wandb_entity", None) or wandb_cfg.get("entity")
    run_name = getattr(args, "wandb_run_name", None) or wandb_cfg.get("run_name")
    group = getattr(args, "wandb_group", None) or wandb_cfg.get("group") or f"phase_{phase}"
    mode = getattr(args, "wandb_mode", None) or wandb_cfg.get("mode") or "offline"

    raw_tags = getattr(args, "wandb_tags", None) or wandb_cfg.get("tags")
    if isinstance(raw_tags, str):
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    elif isinstance(raw_tags, Iterable):
        tags = [str(t) for t in raw_tags]
    else:
        tags = [f"phase_{phase}", str(config.get("dataset", "acdc"))]

    return WandbLogger(
        enabled=enabled,
        project=project,
        entity=entity,
        run_name=run_name,
        group=group,
        tags=tags,
        config=config,
        mode=mode,
    )
