#!/usr/bin/env python3
"""Evaluate existing Self-Audit checkpoints for headroom, audit value, and GT leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.evaluation.audit_decomposition import (
    evaluate_annotation_headroom,
    evaluate_audit_decomposition,
)
from self_audit.training._utils import (
    build_data_loader,
    build_model_from_config,
    build_patient_dataset,
    load_checkpoint,
    load_config,
    resolve_device,
    validate_dataset_splits,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an existing Self-Audit checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/self_audit_joint.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--tau_accept", type=float, default=None)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--neutral_margin", type=float, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--output", default="reports/audit_decomposition.json")
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.split_manifest is not None:
        config["split_manifest"] = args.split_manifest
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    validate_dataset_splits(config)
    device = resolve_device(args.device or config.get("device"))
    model = build_model_from_config(config, device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    val_dataset = build_patient_dataset(
        config,
        split=str(config.get("val_split", "val")),
        train=False,
    )
    val_loader = build_data_loader(
        val_dataset,
        config,
        device=device,
        train=False,
        batch_size=args.batch_size,
    )

    audit_cfg = dict(config.get("audit", {}))
    tau_accept = float(
        args.tau_accept if args.tau_accept is not None else audit_cfg.get("tau_accept", 0.0)
    )
    t_max = int(
        args.t_max
        if args.t_max is not None
        else audit_cfg.get("t_max", config.get("model", {}).get("max_turns", 3))
    )
    neutral_margin = float(
        args.neutral_margin
        if args.neutral_margin is not None
        else audit_cfg.get("neutral_margin", 0.005)
    )

    annotation = evaluate_annotation_headroom(
        model,
        val_loader,
        device,
        neutral_margin=neutral_margin,
        max_batches=args.max_val_batches,
        disable_tqdm=args.no_tqdm,
    )
    audit = evaluate_audit_decomposition(
        model,
        val_loader,
        device,
        tau_accept=tau_accept,
        t_max=t_max,
        neutral_margin=neutral_margin,
        max_batches=args.max_val_batches,
        disable_tqdm=args.no_tqdm,
    )

    payload = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "tau_accept": tau_accept,
        "t_max": t_max,
        "neutral_margin": neutral_margin,
        "annotation_headroom": annotation,
        "audit_decomposition": audit,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_safe(payload), indent=2), encoding="utf-8")

    print(
        f"A0={annotation.get('phase_a/a0_dice', float('nan')):.4f} "
        f"refine_gain={annotation.get('phase_a/total_refinement_gain', float('nan')):+.5f} "
        f"stage_headroom={annotation.get('phase_a/mean_stage_headroom', float('nan')):.5f}"
    )
    print(
        f"initial={audit.get('modes/initial_dice', float('nan')):.4f} "
        f"always={audit.get('modes/always_accept_dice', float('nan')):.4f} "
        f"self={audit.get('modes/self_audit_dice', float('nan')):.4f} "
        f"oracle={audit.get('modes/oracle_dice', float('nan')):.4f}"
    )
    print(
        f"audit_rescue={audit.get('modes/audit_rescue_vs_always', float('nan')):+.5f} "
        f"oracle_headroom={audit.get('modes/oracle_headroom', float('nan')):+.5f} "
        f"headroom_capture={audit.get('modes/headroom_capture_ratio', float('nan')):.3f} "
        f"GT_firewall={int(audit.get('gt_firewall/passed', 0.0))}"
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
