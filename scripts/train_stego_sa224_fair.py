#!/usr/bin/env python
"""Train a fair STEGO-SA224 checkpoint on ACDC train images only."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STEGO_SRC = ROOT / "baseline" / "STEGO" / "src"
for extra in (ROOT / "src", STEGO_SRC):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from shared_benchmark.checkpoint_contract import SCHEMA_VERSION
from shared_benchmark.provenance import sha256_file
from shared_benchmark.spatial import grid_hash, load_self_audit_compat_224_grid_spec
from cardiac_benchmark.dataset import SA224_NORMALIZATION_VERSION
from cardiac_benchmark.provenance import classify_prior_checkpoint


def _env() -> dict[str, str]:
    pythonpath = [str(ROOT / "src"), str(STEGO_SRC)]
    current = str(Path.cwd())
    if current not in pythonpath:
        pythonpath.append(current)
    prior = [value for value in sys.path if value]
    pythonpath.extend(prior)
    return {**dict(), **__import__("os").environ, "PYTHONPATH": ":".join(dict.fromkeys(pythonpath))}


def _run(script: Path, overrides: list[str]) -> None:
    cmd = [sys.executable, str(script), *overrides]
    subprocess.run(cmd, cwd=STEGO_SRC, env=_env(), check=True)


def _find_last_checkpoint(checkpoint_root: Path) -> Path:
    search_roots = [checkpoint_root]
    if not checkpoint_root.is_absolute():
        search_roots.append(STEGO_SRC / checkpoint_root)
    candidates = []
    for root in search_roots:
        candidates.extend(root.glob("**/last.ckpt"))
    candidates = sorted({candidate.resolve(): candidate for candidate in candidates}.values(), key=lambda value: value.stat().st_mtime)
    if not candidates:
        all_ckpts = []
        for root in search_roots:
            all_ckpts.extend(root.glob("**/*.ckpt"))
        all_ckpts = sorted({candidate.resolve(): candidate for candidate in all_ckpts}.values(), key=lambda value: value.stat().st_mtime)
        if not all_ckpts:
            raise FileNotFoundError(f"No STEGO checkpoints were written under {checkpoint_root}")
        return all_ckpts[-1]
    return candidates[-1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_fair_prior_policy(
    checkpoint_path: Path | None,
    *,
    allow_stego_downstream_prior: bool,
) -> dict[str, Any]:
    prior = classify_prior_checkpoint(checkpoint_path)
    if prior["requires_explicit_override"] and not allow_stego_downstream_prior:
        raise ValueError(
            "STEGO-SA224-FAIR refuses downstream STEGO Lightning checkpoints by default "
            "because they contain out-of-domain projection-head state. Use a DINO teacher "
            "checkpoint or rerun with --allow-stego-downstream-prior to accept this "
            "adaptation explicitly."
        )
    if prior["requires_explicit_override"]:
        prior = {
            **prior,
            "fair_status": "intentional_override",
            "allowed_for_fair_mode": True,
            "override_flag": "--allow-stego-downstream-prior",
        }
    return prior


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "preprocessed_data" / "ACDC")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "splits" / "acdc_patient_split_seed42.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "stego" / "fair_sa224")
    parser.add_argument("--run-name", default="acdc_train_only")
    parser.add_argument("--pretrained-weights", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-freq", type=int, default=250)
    parser.add_argument("--experiment-name", default="fair_sa224")
    parser.add_argument("--allow-stego-downstream-prior", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prior_checkpoint = resolve_fair_prior_policy(
        args.pretrained_weights,
        allow_stego_downstream_prior=args.allow_stego_downstream_prior,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = args.output_dir / args.run_name
    pytorch_data_dir = scratch_root / "pytorch-data"
    checkpoint_root = scratch_root / "checkpoints"
    pytorch_data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    pretrained_value = "null" if args.pretrained_weights is None else str(args.pretrained_weights.resolve())
    common = [
        f"output_root={scratch_root}",
        f"pytorch_data_dir={pytorch_data_dir}",
        f"dataset_name=directory",
        f"dir_dataset_name=ACDC",
        f"dir_dataset_n_classes=4",
        f"dir_dataset_split_manifest={args.split_manifest.resolve()}",
        f"dir_dataset_data_root={args.data_root.resolve()}",
        f"res=224",
        f"batch_size={args.batch_size}",
        f"num_workers={args.num_workers}",
        f"pretrained_weights={pretrained_value}",
    ]

    _run(STEGO_SRC / "precompute_knns.py", common)
    _run(
        STEGO_SRC / "train_segmentation.py",
        [
            *common,
            f"experiment_name={args.experiment_name}",
            f"log_dir=stego_fair_sa224",
            f"max_steps={args.max_steps}",
            f"checkpoint_freq={args.checkpoint_freq}",
            "fair_train_mode=True",
            "train_linear_probe=False",
            "enable_validation=False",
        ],
    )

    raw_checkpoint = _find_last_checkpoint(checkpoint_root)
    final_checkpoint = args.output_dir / f"{args.run_name}.ckpt"
    if raw_checkpoint.resolve() != final_checkpoint.resolve():
        shutil.copy2(raw_checkpoint, final_checkpoint)
    checkpoint_sha = sha256_file(final_checkpoint)
    grid = load_self_audit_compat_224_grid_spec(ROOT)

    training_provenance = {
        "schema_version": "self_audit.stego_fair_training.v1",
        "baseline_name": "STEGO",
        "baseline_mode": "STEGO-SA224-FAIR",
        "dataset": "acdc",
        "run_name": args.run_name,
        "source_data_root": str(args.data_root.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "scratch_root": str(scratch_root.resolve()),
        "checkpoint_path": str(final_checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "knn_split": "train",
        "checkpoint_selection_policy": "last_without_dev_selection",
        "pretrained_weights": None if args.pretrained_weights is None else str(args.pretrained_weights.resolve()),
        "prior_checkpoint": prior_checkpoint,
        "config_overrides": {
            "fair_train_mode": True,
            "train_linear_probe": False,
            "enable_validation": False,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "checkpoint_freq": args.checkpoint_freq,
            "allow_stego_downstream_prior": bool(args.allow_stego_downstream_prior),
        },
        "training_tags": [
            "in_domain_acdc",
            "self_audit_normalized",
            "train_only_knn",
            "cluster_probe_inference",
        ],
    }
    contract = {
        "schema_version": SCHEMA_VERSION,
        "baseline_name": "STEGO",
        "baseline_mode": "STEGO-SA224-FAIR",
        "benchmark_tier": "fair",
        "checkpoint_sha256": checkpoint_sha,
        "dataset": "acdc",
        "split_policy_version": "self_audit.acdc.patient_split.v1",
        "shared_grid_hash": grid_hash(grid),
        "normalization_version": SA224_NORMALIZATION_VERSION,
        "training_data_schema": "self_audit.acdc.image_only.v1",
        "training_tags": training_provenance["training_tags"],
    }
    _write_json(args.output_dir / f"{args.run_name}.training_provenance.json", training_provenance)
    _write_json(args.output_dir / f"{args.run_name}.checkpoint_contract.json", contract)
    print(json.dumps({
        "checkpoint": str(final_checkpoint),
        "checkpoint_contract": str(args.output_dir / f"{args.run_name}.checkpoint_contract.json"),
        "training_provenance": str(args.output_dir / f"{args.run_name}.training_provenance.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
