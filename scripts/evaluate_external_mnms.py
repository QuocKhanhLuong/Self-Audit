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

from self_audit.artifact_io import atomic_write_json, json_safe_artifact
from self_audit.audit.semantics import resolve_empty_policy, resolve_neutral_margin
from self_audit.data.common import load_array, to_depth_first
from self_audit.evaluation.cohort import evaluate_volume_cohort
from self_audit.evaluation.volume_inference import COMPARISON_MODES
from self_audit.provenance import (
    UNKNOWN,
    _mechanism_values_equal,
    verify_model_config,
)
from self_audit.training._utils import (
    CANDIDATE_C_KEYS,
    CANDIDATE_C_SOLVER_MODES,
    DEFAULT_WINDOW_MODE,
    bind_evaluation_checkpoint,
    build_model_from_config,
    build_patient_dataset,
    load_config,
    resolve_device,
    validate_candidate_c_settings,
    validate_dataset_splits,
    validate_window_mode,
    verify_bound_state,
)


EXTERNAL_SCHEMA_VERSION = 1
EVIDENCE_CLASS = "independent_external_evaluation"
PHASE_PARTITION_UNAVAILABLE = "unavailable_without_authoritative_metadata"
DEFAULT_MAPPING = {0: 0, 1: 3, 2: 2, 3: 1}


_json_safe = json_safe_artifact


def _normalize_external_config(
    config: Mapping[str, Any],
    *,
    data_root: str | Path | None,
    split: str | None,
    window_mode: str | None = None,
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
    if window_mode is not None:
        model_sec = result.setdefault("model", {})
        if isinstance(model_sec, Mapping):
            result["model"] = dict(model_sec)
        result["model"]["window_mode"] = validate_window_mode(window_mode)
    if "model" in result and isinstance(result["model"], Mapping) and "candidate_c" in result["model"]:
        c_cfg = result["model"]["candidate_c"]
        if c_cfg is not None:
            result["model"]["candidate_c"] = validate_candidate_c_settings(c_cfg)
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


def _load_checkpoint_payload(checkpoint_path: Path | str) -> Mapping[str, Any]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Unable to read checkpoint {ckpt_path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping, got {type(payload).__name__}")
    return payload


def _inspect_checkpoint_dataset(
    checkpoint_path: Path | str,
    payload: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Inspect checkpoint metadata for training dataset identity.

    Returns (training_dataset, evidence_class).
    Collects all declared source identities, requires exact normalized 'acdc',
    and rejects incompatible or contradictory claims.
    Historical checkpoints with missing metadata return ("unknown", "uncertified_historical_checkpoint").
    ACDC-trained checkpoints return ("acdc", EVIDENCE_CLASS).
    """
    if payload is None:
        payload = _load_checkpoint_payload(checkpoint_path)

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
    window_mode: str | None = None,
    device: torch.device | str | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run frozen M&Ms evaluation and optionally write its JSON report."""

    ckpt_payload = _load_checkpoint_payload(checkpoint)
    training_dataset, evidence_class = _inspect_checkpoint_dataset(checkpoint, payload=ckpt_payload)
    raw_config = load_config(config) if isinstance(config, (str, Path)) else dict(config)

    # Checkpoint declared model configuration and execution mechanism
    ckpt_cfg = ckpt_payload.get("config") if isinstance(ckpt_payload.get("config"), Mapping) else {}
    ckpt_model = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg.get("model"), Mapping) else {}
    ckpt_prov = ckpt_payload.get("provenance") if isinstance(ckpt_payload.get("provenance"), Mapping) else {}
    ckpt_prov_id = ckpt_prov.get("model_identity") if isinstance(ckpt_prov.get("model_identity"), Mapping) else {}
    ckpt_top_id = ckpt_payload.get("model_identity") if isinstance(ckpt_payload.get("model_identity"), Mapping) else {}

    # 1. Collect and validate all checkpoint execution mode declarations
    ckpt_mode_declarations: dict[str, str] = {}
    if "window_mode" in ckpt_model:
        ckpt_mode_declarations["checkpoint.config.model.window_mode"] = validate_window_mode(
            ckpt_model["window_mode"], name="checkpoint.config.model.window_mode"
        )
    if "window_mode" in ckpt_cfg:
        ckpt_mode_declarations["checkpoint.config.window_mode"] = validate_window_mode(
            ckpt_cfg["window_mode"], name="checkpoint.config.window_mode"
        )
    if "window_mode" in ckpt_payload:
        ckpt_mode_declarations["checkpoint.window_mode"] = validate_window_mode(
            ckpt_payload["window_mode"], name="checkpoint.window_mode"
        )
    if "window_mode" in ckpt_prov_id and ckpt_prov_id["window_mode"] != UNKNOWN:
        ckpt_mode_declarations["checkpoint.provenance.model_identity.window_mode"] = validate_window_mode(
            ckpt_prov_id["window_mode"], name="checkpoint.provenance.model_identity.window_mode"
        )
    if "window_mode" in ckpt_top_id and ckpt_top_id["window_mode"] != UNKNOWN:
        ckpt_mode_declarations["checkpoint.model_identity.window_mode"] = validate_window_mode(
            ckpt_top_id["window_mode"], name="checkpoint.model_identity.window_mode"
        )

    unique_ckpt_modes = set(ckpt_mode_declarations.values())
    if len(unique_ckpt_modes) > 1:
        details = ", ".join(f"{k}={v!r}" for k, v in sorted(ckpt_mode_declarations.items()))
        raise ValueError(f"Conflicting window_mode declarations in checkpoint: {details}")

    ckpt_window_mode: str | None = next(iter(unique_ckpt_modes)) if unique_ckpt_modes else None

    # 2. Collect and validate all checkpoint candidate_c settings declarations
    ckpt_c_declarations: dict[str, dict[str, Any]] = {}
    if "candidate_c" in ckpt_model:
        ckpt_c_declarations["checkpoint.config.model.candidate_c"] = validate_candidate_c_settings(
            ckpt_model["candidate_c"], name="checkpoint.config.model.candidate_c"
        )
    if "candidate_c" in ckpt_cfg:
        ckpt_c_declarations["checkpoint.config.candidate_c"] = validate_candidate_c_settings(
            ckpt_cfg["candidate_c"], name="checkpoint.config.candidate_c"
        )
    if "candidate_c" in ckpt_payload:
        ckpt_c_declarations["checkpoint.candidate_c"] = validate_candidate_c_settings(
            ckpt_payload["candidate_c"], name="checkpoint.candidate_c"
        )
    if "candidate_c_settings" in ckpt_prov_id and ckpt_prov_id["candidate_c_settings"] is not None:
        ckpt_c_declarations["checkpoint.provenance.model_identity.candidate_c_settings"] = validate_candidate_c_settings(
            ckpt_prov_id["candidate_c_settings"], name="checkpoint.provenance.model_identity.candidate_c_settings"
        )
    if "candidate_c_settings" in ckpt_top_id and ckpt_top_id["candidate_c_settings"] is not None:
        ckpt_c_declarations["checkpoint.model_identity.candidate_c_settings"] = validate_candidate_c_settings(
            ckpt_top_id["candidate_c_settings"], name="checkpoint.model_identity.candidate_c_settings"
        )

    if len(ckpt_c_declarations) > 1:
        first_loc, first_s = next(iter(ckpt_c_declarations.items()))
        for other_loc, other_s in list(ckpt_c_declarations.items())[1:]:
            for k in CANDIDATE_C_KEYS:
                if not _mechanism_values_equal(first_s[k], other_s[k]):
                    raise ValueError(
                        f"Conflicting candidate_c declarations in checkpoint: "
                        f"{first_loc}.{k}={first_s[k]!r} vs {other_loc}.{k}={other_s[k]!r}"
                    )

    ckpt_candidate_c: dict[str, Any] | None = (
        next(iter(ckpt_c_declarations.values())) if ckpt_c_declarations else None
    )

    # 3. Collect and validate evaluation config declarations
    config_model = raw_config.get("model", {}) if isinstance(raw_config.get("model"), Mapping) else {}

    config_mode_declarations: dict[str, str] = {}
    if "window_mode" in config_model:
        config_mode_declarations["config.model.window_mode"] = validate_window_mode(
            config_model["window_mode"], name="config.model.window_mode"
        )
    if "window_mode" in raw_config:
        config_mode_declarations["config.window_mode"] = validate_window_mode(
            raw_config["window_mode"], name="config.window_mode"
        )

    unique_config_modes = set(config_mode_declarations.values())
    if len(unique_config_modes) > 1:
        details = ", ".join(f"{k}={v!r}" for k, v in sorted(config_mode_declarations.items()))
        raise ValueError(f"Conflicting window_mode declarations in evaluation config: {details}")

    config_window_mode: str | None = next(iter(unique_config_modes)) if unique_config_modes else None

    cli_window_mode: str | None = None
    if window_mode is not None:
        cli_window_mode = validate_window_mode(window_mode, name="--window-mode")

    # Reject conflicts across CLI, evaluation config, and checkpoint
    if cli_window_mode is not None and ckpt_window_mode is not None and cli_window_mode != ckpt_window_mode:
        raise ValueError(
            f"CLI --window-mode {cli_window_mode!r} conflicts with checkpoint declared window_mode {ckpt_window_mode!r}"
        )
    if cli_window_mode is not None and config_window_mode is not None and cli_window_mode != config_window_mode:
        raise ValueError(
            f"CLI --window-mode {cli_window_mode!r} conflicts with config window_mode {config_window_mode!r}"
        )
    if config_window_mode is not None and ckpt_window_mode is not None and config_window_mode != ckpt_window_mode:
        raise ValueError(
            f"Config window_mode {config_window_mode!r} conflicts with checkpoint declared window_mode {ckpt_window_mode!r}"
        )

    # Requirement 4: Legacy checkpoints lacking mechanism settings
    if ckpt_window_mode is None:
        if cli_window_mode is not None and cli_window_mode != DEFAULT_WINDOW_MODE:
            raise ValueError(
                f"Cannot evaluate legacy checkpoint lacking execution mechanism metadata under non-default mode "
                f"{cli_window_mode!r}; source mechanism identity cannot be established."
            )
        if config_window_mode is not None and config_window_mode != DEFAULT_WINDOW_MODE:
            raise ValueError(
                f"Cannot evaluate legacy checkpoint lacking execution mechanism metadata under non-default mode "
                f"{config_window_mode!r}; source mechanism identity cannot be established."
            )

    resolved_window_mode = cli_window_mode or config_window_mode or ckpt_window_mode or DEFAULT_WINDOW_MODE

    # Requirement 3: Evaluation config candidate_c handling
    # Explicit candidate_c: null is NOT absence; only absent keys inherit source
    has_config_candidate_c = ("candidate_c" in config_model) or ("candidate_c" in raw_config)
    config_c_declarations: dict[str, dict[str, Any]] = {}
    if "candidate_c" in config_model:
        config_c_declarations["config.model.candidate_c"] = validate_candidate_c_settings(
            config_model["candidate_c"], name="config.model.candidate_c"
        )
    if "candidate_c" in raw_config:
        config_c_declarations["config.candidate_c"] = validate_candidate_c_settings(
            raw_config["candidate_c"], name="config.candidate_c"
        )

    if len(config_c_declarations) > 1:
        first_loc, first_s = next(iter(config_c_declarations.items()))
        for other_loc, other_s in list(config_c_declarations.items())[1:]:
            for k in CANDIDATE_C_KEYS:
                if not _mechanism_values_equal(first_s[k], other_s[k]):
                    raise ValueError(
                        f"Conflicting candidate_c declarations in evaluation config: "
                        f"{first_loc}.{k}={first_s[k]!r} vs {other_loc}.{k}={other_s[k]!r}"
                    )

    config_candidate_c: dict[str, Any] | None = (
        next(iter(config_c_declarations.values())) if config_c_declarations else None
    )

    if has_config_candidate_c and config_candidate_c is not None and ckpt_candidate_c is not None:
        for k in CANDIDATE_C_KEYS:
            if not _mechanism_values_equal(config_candidate_c[k], ckpt_candidate_c[k]):
                raise ValueError(
                    f"Config candidate_c.{k}={config_candidate_c[k]!r} conflicts with checkpoint candidate_c.{k}={ckpt_candidate_c[k]!r}"
                )

    # Requirement 4: Under non-default solver mode, checkpoint must establish candidate_c settings
    if resolved_window_mode in CANDIDATE_C_SOLVER_MODES and ckpt_candidate_c is None:
        raise ValueError(
            f"Cannot evaluate legacy checkpoint lacking candidate_c settings under non-default solver mode "
            f"{resolved_window_mode!r}; source mechanism identity cannot be established."
        )

    # Fallback only when absent in evaluation config
    resolved_candidate_c: dict[str, Any] | None
    if has_config_candidate_c:
        resolved_candidate_c = config_candidate_c
    else:
        resolved_candidate_c = ckpt_candidate_c

    flat = _normalize_external_config(
        raw_config,
        data_root=data_root,
        split=split,
        window_mode=resolved_window_mode,
    )
    if resolved_candidate_c is not None:
        flat.setdefault("model", {})["candidate_c"] = resolved_candidate_c

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

    # Requirement 1: Unconditional source-config and live model verification
    verify_model_config(model, flat)
    if isinstance(ckpt_payload.get("config"), Mapping):
        verify_model_config(model, ckpt_payload["config"])

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
    candidate_c_dict: dict[str, Any] | None = None
    if hasattr(model, "candidate_c") and model.candidate_c is not None:
        if hasattr(model.candidate_c, "as_dict"):
            candidate_c_dict = model.candidate_c.as_dict()
        elif isinstance(model.candidate_c, Mapping):
            candidate_c_dict = dict(model.candidate_c)

    payload: dict[str, Any] = {
        "external_schema_version": EXTERNAL_SCHEMA_VERSION,
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "evidence_class": evidence_class,
        "protocol": str(raw_config.get("protocol", "acdc_to_mnms_domain_shift")),
        "scientific_protocol_label": "acdc_frozen_to_mnms_external_v1",
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
        "window_mode": getattr(model, "window_mode", flat.get("model", {}).get("window_mode", DEFAULT_WINDOW_MODE)),
        "candidate_c_settings": candidate_c_dict,
        "metrics": volume,
    }
    safe_payload = _json_safe(payload)
    if output is not None:
        output_path = Path(output)
        atomic_write_json(output_path, payload, indent=2, sort_keys=True)
    return safe_payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen external M&Ms testing evaluation")
    parser.add_argument("--config", default="configs/self_audit_acdc_to_mnms.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", "--data_root", dest="data_root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--tau-accept", "--tau_accept", dest="tau_accept", type=float, default=None)
    parser.add_argument(
        "--window-mode",
        "--window_mode",
        dest="window_mode",
        default=None,
        help="Model execution mode override ('current', 'candidate_c', etc.)",
    )
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
        window_mode=args.window_mode,
        device=args.device,
        output=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":  # pragma: no cover
    main()
