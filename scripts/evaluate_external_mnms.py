#!/usr/bin/env python3
"""Evaluate a frozen ACDC-trained checkpoint on the M&Ms testing cohort.

This entrypoint deliberately has no training, calibration, or threshold-sweep
path.  The M&Ms result is an independent external evaluation with a fixed tau
provided by the CLI or the protocol config.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.audit.semantics import resolve_empty_policy, resolve_neutral_margin
from self_audit.data.common import load_array, to_depth_first
from self_audit.evaluation.cohort import evaluate_volume_cohort
from self_audit.evaluation.volume_inference import COMPARISON_MODES
from self_audit.training._utils import (
    bind_evaluation_checkpoint,
    build_model_from_config,
    build_patient_dataset,
    load_config,
    resolve_device,
    validate_dataset_splits,
    verify_bound_state,
)


EXTERNAL_SCHEMA_VERSION = 1
EVIDENCE_CLASS = "independent_external_evaluation"
PHASE_PARTITION_UNAVAILABLE = "unavailable_without_authoritative_metadata"
DEFAULT_MAPPING = {0: 0, 1: 3, 2: 2, 3: 1}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _normalize_external_config(
    config: Mapping[str, Any],
    *,
    data_root: str | Path | None,
    split: str | None,
) -> dict[str, Any]:
    """Overlay the protocol's ``external_test`` block onto factory settings."""

    if not isinstance(config, Mapping):
        raise ValueError("external protocol config must be a mapping")
    result = deepcopy(dict(config))
    if "dataset" in config and str(config["dataset"]).lower() != "mnms":
        raise ValueError(f"External evaluation only supports dataset='mnms', got {config['dataset']!r}")
    external = result.get("external_test", {})
    if not isinstance(external, Mapping):
        raise ValueError("external_test config must be a mapping")
    if "dataset" in external and str(external.get("dataset", "mnms")).lower() != "mnms":
        raise ValueError(f"external_test.dataset must be 'mnms', got {external['dataset']!r}")
    result.update({str(key): deepcopy(value) for key, value in external.items()})
    result["dataset"] = "mnms"
    resolved_split = str(split or external.get("split", "testing"))
    if resolved_split not in {"test", "testing"}:
        raise ValueError(f"Independent external evaluation requires the M&Ms test/testing split, got {resolved_split!r}")
    result["split"] = resolved_split
    result["test_split"] = resolved_split
    data_root_str = str(data_root or external.get("data_root", "preprocessed_data/mnm"))
    from self_audit.data.mnms import MNMSClassMapping, is_mnms_binary_path
    if is_mnms_binary_path(data_root_str):
        raise ValueError(f"M&Ms binary derivative path is not supported for external evaluation: {data_root_str}")
    result["data_root"] = data_root_str
    mapping = result.get("raw_to_acdc", DEFAULT_MAPPING)
    if not isinstance(mapping, Mapping):
        raise ValueError("external_test.raw_to_acdc must be a mapping")
    class_map = MNMSClassMapping(mapping)
    result["raw_to_acdc"] = dict(class_map.raw_to_acdc)
    result["class_mapping"] = dict(class_map.raw_to_acdc)
    result["num_classes"] = int(result.get("num_classes", result.get("model", {}).get("num_classes", 4)))
    if result["num_classes"] != 4:
        raise ValueError(f"External M&Ms evaluation requires num_classes=4, got {result['num_classes']}")
    return result


def _stored_grid(dataset: Any, *, depth_axis: int | None) -> tuple[list[int] | None, bool]:
    records = list(getattr(dataset, "records", []))
    if not records:
        return None, False
    grids: set[tuple[int, int]] = set()
    for record in records:
        first, _ = load_array(record.image_path)
        normalized = to_depth_first(first, depth_axis=depth_axis)
        grids.add((int(normalized.shape[-2]), int(normalized.shape[-1])))
    if len(grids) == 1:
        h, w = grids.pop()
        return [h, w], True
    return None, False


def _inspect_checkpoint_dataset(checkpoint_path: Path | str) -> tuple[str, str]:
    """Inspect checkpoint metadata for training dataset identity.

    Returns (training_dataset, evidence_class).
    Collects all declared source identities, requires exact normalized 'acdc',
    and rejects incompatible or contradictory claims.
    Historical checkpoints with missing metadata return ("unknown", "uncertified_historical_checkpoint").
    ACDC-trained checkpoints return ("acdc", EVIDENCE_CLASS).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Unable to read checkpoint {ckpt_path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping, got {type(payload).__name__}")

    declared_identities: set[str] = set()

    def _record_candidate(val: Any) -> None:
        if val is not None and not isinstance(val, (Mapping, list, tuple, set)):
            s = str(val).strip().lower()
            if s:
                declared_identities.add(s)

    # 1. Top-level keys
    for k in ("training_dataset", "dataset", "dataset_name"):
        if k in payload:
            _record_candidate(payload[k])

    # 2. Config mapping
    cfg = payload.get("config")
    if isinstance(cfg, Mapping):
        for k in ("training_dataset", "dataset_name"):
            if k in cfg:
                _record_candidate(cfg[k])
        ds_cfg = cfg.get("dataset")
        if isinstance(ds_cfg, Mapping):
            for k in ("name", "dataset_name", "dataset"):
                if k in ds_cfg:
                    _record_candidate(ds_cfg[k])
        elif ds_cfg is not None:
            _record_candidate(ds_cfg)

    # 3. Cohort descriptor mapping
    cohort = payload.get("cohort_descriptor")
    if isinstance(cohort, Mapping):
        for k in ("dataset", "dataset_name", "name"):
            if k in cohort:
                _record_candidate(cohort[k])

    # 4. Extra mapping (if present)
    extra = payload.get("extra")
    if isinstance(extra, Mapping):
        for k in ("training_dataset", "dataset", "dataset_name"):
            if k in extra:
                _record_candidate(extra[k])
        extra_cohort = extra.get("cohort_descriptor")
        if isinstance(extra_cohort, Mapping):
            for k in ("dataset", "dataset_name", "name"):
                if k in extra_cohort:
                    _record_candidate(extra_cohort[k])

    if not declared_identities:
        return "unknown", "uncertified_historical_checkpoint"

    if len(declared_identities) > 1:
        raise ValueError(
            f"Contradictory training dataset identities in checkpoint {checkpoint_path}: "
            f"{sorted(declared_identities)}"
        )

    declared = next(iter(declared_identities))
    if declared != "acdc":
        if "mnm" in declared:
            raise ValueError(
                f"Checkpoint {checkpoint_path} was trained on {declared!r}; "
                "cannot evaluate M&Ms-trained model as independent external evidence."
            )
        raise ValueError(
            f"Incompatible checkpoint training dataset {declared!r} in {checkpoint_path}; "
            "external M&Ms evaluation requires an ACDC-trained model."
        )

    return "acdc", EVIDENCE_CLASS


def run_external_evaluation(
    *,
    config: Mapping[str, Any] | str | Path,
    checkpoint: str | Path,
    data_root: str | Path | None = None,
    split: str | None = None,
    tau_accept: float | None = None,
    device: torch.device | str | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run frozen M&Ms evaluation and optionally write its JSON report."""

    training_dataset, evidence_class = _inspect_checkpoint_dataset(checkpoint)
    raw_config = load_config(config) if isinstance(config, (str, Path)) else dict(config)
    flat = _normalize_external_config(raw_config, data_root=data_root, split=split)
    resolved_split = str(flat["split"])
    validation = validate_dataset_splits(flat)
    if not validation.get("validated"):
        raise ValueError(f"M&Ms validation failed: {validation}")

    target_device = resolve_device(str(device) if device is not None else flat.get("device"))
    model = build_model_from_config(flat, target_device)
    binding = bind_evaluation_checkpoint(
        model,
        [("checkpoint", str(checkpoint))],
        map_location=target_device,
        config=flat,
    )
    model.eval()

    dataset = build_patient_dataset(flat, split=resolved_split, train=False)
    verify_bound_state(model, binding, boundary="external_mnms_before_inference")

    audit_config = flat.get("audit", {}) or {}
    if not isinstance(audit_config, Mapping):
        raise ValueError("audit config must be a mapping")
    if tau_accept is None:
        if "tau_accept" in audit_config:
            resolved_tau = float(audit_config["tau_accept"])
            tau_source = "config.audit.tau_accept"
        else:
            resolved_tau = 0.0
            tau_source = "default:0.0"
    else:
        resolved_tau = float(tau_accept)
        if not np.isfinite(resolved_tau):
            raise ValueError("tau_accept must be finite")
        tau_source = "cli:--tau_accept"
    if not np.isfinite(resolved_tau):
        raise ValueError("audit.tau_accept must be finite")
    neutral_margin = resolve_neutral_margin(audit_config.get("neutral_margin"))
    empty_policy = resolve_empty_policy(audit_config.get("empty_policy"))
    t_max = int(audit_config.get("t_max", flat.get("model", {}).get("max_turns", 3)))
    batch_size = int(flat.get("batch_size", 8))
    image_size = int(flat.get("image_size", 256))
    if t_max < 0 or batch_size < 1 or image_size < 1:
        raise ValueError("t_max must be non-negative, batch_size and image_size must be positive")
    volume = evaluate_volume_cohort(
        model,
        dataset,
        target_device,
        tau_accept=resolved_tau,
        t_max=t_max,
        neutral_margin=neutral_margin,
        empty_policy=empty_policy,
        image_size=image_size,
        num_classes=int(flat["num_classes"]),
        batch_size=batch_size,
    )
    verify_bound_state(model, binding, boundary="external_mnms_after_inference")

    records = list(getattr(dataset, "records", []))
    case_count = int(validation.get("case_counts", validation.get("cases", {})).get("test", len(records)))
    patient_count = int(validation.get("patient_counts", validation.get("patients", {})).get("test", len({r.patient_id for r in records})))
    binding_dict = binding.as_dict() if hasattr(binding, "as_dict") else dict(binding)
    grid, is_uniform = _stored_grid(dataset, depth_axis=flat.get("depth_axis"))
    payload: dict[str, Any] = {
        "external_schema_version": EXTERNAL_SCHEMA_VERSION,
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "evidence_class": evidence_class,
        "protocol": str(raw_config.get("protocol", "acdc_to_mnms_domain_shift")),
        "training_dataset": training_dataset,
        "dataset": "mnms",
        "split": resolved_split,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": case_count,
        "patient_count": patient_count,
        "cohort_signature": validation.get("membership_signature", validation.get("split_signature")),
        "cohort_validation": validation,
        "label_mapping": dict(validation.get("raw_to_acdc", flat["raw_to_acdc"])),
        "stored_grid": grid,
        "stored_grid_uniform": is_uniform,
        "network_input_grid": [image_size, image_size],
        "metric_space": "volume_resized",
        "native_dice_available": False,
        "phase_partition": PHASE_PARTITION_UNAVAILABLE,
        "phase_counts": {"ED": 0, "ES": 0, "unknown": case_count, "all": case_count},
        "tau_accept": resolved_tau,
        "tau_accept_source": tau_source,
        "neutral_margin": neutral_margin,
        "t_max": t_max,
        "empty_policy": empty_policy,
        "comparison_modes": list(COMPARISON_MODES),
        "oracle_accept_note": "oracle_accept is a non-deployable upper-bound diagnostic using ground truth.",
        "checkpoint_binding": binding_dict,
        "checkpoint_path": str(checkpoint),
        "live_state_digest": getattr(binding, "state_digest", binding_dict.get("state_digest")),
        "model_identity": getattr(binding, "model_identity", binding_dict.get("model_identity", {})),
        "metrics": volume,
    }
    safe_payload = _json_safe(payload)
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(safe_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return safe_payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen external M&Ms testing evaluation")
    parser.add_argument("--config", default="configs/self_audit_acdc_to_mnms.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", "--data_root", dest="data_root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--tau-accept", "--tau_accept", dest="tau_accept", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="reports/external_mnms.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    payload = run_external_evaluation(
        config=args.config,
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        tau_accept=args.tau_accept,
        device=args.device,
        output=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":  # pragma: no cover
    main()
