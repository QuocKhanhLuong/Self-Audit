#!/usr/bin/env python3
"""Canonical Self-Audit trainer: approved DINOv3 only; no legacy fallback."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHITECTURE = "layer_guided_convnext_tiny_v1"
IMPLEMENTATION = Path("experiments/layer_guided_student_v1")


def identity(root: Path | None = None) -> dict:
    root = ROOT if root is None else Path(root)
    value = json.loads((root / "ACTIVE_PIPELINE.json").read_text(encoding="utf-8"))
    expected = {"schema": "self_audit.active_pipeline.v1", "entrypoint": "train.py",
                "architecture": ARCHITECTURE, "model_class": "LayerGuidedStudent",
                "backbone": "dinov3_convnext_tiny", "pretrained_required": True,
                "training_phase": "coarse", "native_detail_objective": False,
                "semantic_target": "frozen_historical_q_reliability_v1",
                "guard": "off", "refiner": "off", "implementation": str(IMPLEMENTATION)}
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"ACTIVE_PIPELINE mismatch: {key}; use a clean current worktree")
    return value


def read_recipe(path: Path, allow_full: bool = False) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != "layer_guided.training_recipe.v1":
        raise ValueError("New layer-guided training recipe required; legacy configs are rejected")
    if config.get("architecture") != ARCHITECTURE:
        raise ValueError("Canonical trainer accepts only LayerGuidedStudent; no legacy/control fallback")
    if config.get("semantic_policy") != "frozen_historical_q_reliability_v1":
        raise ValueError("This version trains full Q; a Z/native-detail policy needs a versioned implementation")
    if config.get("guard") != "off" or config.get("refiner") != "off":
        raise ValueError("Guard/refiner are not enabled in the current training phase")
    if "max_steps" not in config:
        raise ValueError("Explicit max_steps is required")
    steps = config["max_steps"]
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("max_steps must be a positive integer or null")
    if (steps is None or steps > 128) and not allow_full:
        raise ValueError("Budget exceeds a bounded pilot: pass --allow-full-run explicitly")
    return config


def _load_runner(root: Path):
    """Reject cached modules from another checkout instead of mixing code paths."""
    package_root = (root / IMPLEMENTATION).resolve()
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in {"integration", "training_v1", "layer_upgrade", "self_audit_guidance"}:
            continue
        origin = getattr(module, "__file__", None)
        allowed = (root / "src").resolve() if name.startswith("self_audit_guidance") else package_root
        if origin is None or not Path(origin).resolve().is_relative_to(allowed):
            raise RuntimeError(f"Foreign module already loaded: {name}; start a fresh Python process")
    for directory in (root / "src", package_root):
        sys.path.insert(0, str(directory))
    module = importlib.import_module("training_v1.runner")
    if Path(module.__file__).resolve() != package_root / "training_v1/runner.py":
        raise RuntimeError("Resolved runner is not the canonical DINOv3 implementation")
    return module.run_training


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--describe", action="store_true", help="Print selected architecture without torch, weights or data")
    parser.add_argument("--phase", choices=("coarse",), help="Required for training; native-detail/self-learning is not yet integrated")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--bank-manifest-sha256")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--approval-sha256")
    parser.add_argument("--recipe", type=Path, default=ROOT / "configs/pilot.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-full-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        selected = identity()
        if args.describe:
            print(json.dumps(selected, indent=2, sort_keys=True))
            return 0
        missing = [name for name in ("phase", "bank", "bank_manifest_sha256", "approval", "approval_sha256", "out")
                   if getattr(args, name) is None]
        if missing:
            parser.error("required for training: " + ", ".join("--" + name.replace("_", "-") for name in missing))
        from scripts.check_repo import check_layout
        check_layout(ROOT)
        recipe = read_recipe(args.recipe, args.allow_full_run)
        # Check basic paths before importing torch or allocating GPU memory.
        if not args.bank.is_dir() or not (args.bank / "FROZEN.json").is_file():
            raise FileNotFoundError("Original frozen TRAIN bank is required; published metadata copies are not banks")
        if not args.approval.is_file():
            raise FileNotFoundError("Reviewed official DINOv3 approval receipt is required; no random fallback")
        if args.out.exists():
            raise FileExistsError("Output already exists; use a fresh directory")
        print("[SELF_AUDIT_SELECTED] " + json.dumps(selected, sort_keys=True), flush=True)
        print("[SELF_AUDIT_SCOPE] COARSE_ONLY; NATIVE_DETAIL_UNTRAINED; GUARD_OFF; REFINER_OFF", flush=True)
        runner = _load_runner(ROOT)
        result = runner(bank=args.bank, expected_bank_sha256=args.bank_manifest_sha256,
                        approval=args.approval, expected_approval_sha256=args.approval_sha256,
                        config=recipe, out=args.out, device=args.device)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"[SELF_AUDIT_FAILED] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print("[SELF_AUDIT_COMPLETE] " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
