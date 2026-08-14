#!/usr/bin/env python3
"""Run the non-training preflight for a Self-Audit configuration."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import platform
import sys
import tempfile

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.losses.annotation import annotation_loss
from self_audit.training._utils import (
    build_model_from_config,
    build_patient_dataset,
    load_config,
    move_batch,
    resolve_device,
    seed_everything,
)


def _finite_gradients(model: torch.nn.Module) -> tuple[bool, int]:
    count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        count += 1
        if not torch.isfinite(parameter.grad).all():
            return False, count
    return True, count


def _check_split_integrity(config: dict[str, object]) -> None:
    validator = None
    try:
        from self_audit.training._utils import validate_dataset_splits

        validator = validate_dataset_splits
    except ImportError:
        pass
    if validator is not None:
        validator(config)
        return
    # Compatibility fallback for a checkout where the helper has not yet been
    # installed: constructing train and validation datasets still validates the
    # configured manifest and patient IDs.
    train = build_patient_dataset(config, split=str(config.get("train_split", "train")), train=False)
    val = build_patient_dataset(config, split=str(config.get("val_split", "val")), train=False)
    train_patients = {str(item["patient_id"]) for item in train.records}
    val_patients = {str(item["patient_id"]) for item in val.records}
    overlap = sorted(train_patients & val_patients)
    if overlap:
        raise ValueError(f"Patient leakage between train and val: {overlap[:5]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow_pretrained", action="store_true", help="Permit external ConvNeXt weight loading")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    print(f"python={platform.python_version()}")
    print(f"pytorch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"timm_available={importlib.util.find_spec('timm') is not None}")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"gpu={torch.cuda.get_device_name(device)}")
        print(f"gpu_memory_mb={torch.cuda.get_device_properties(device).total_memory / 2**20:.1f}")

    try:
        config = load_config(args.config)
        seed_everything(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", False)))
        data_root = Path(str(config.get("data_root", "preprocessed_data/ACDC")))
        if not data_root.exists():
            raise FileNotFoundError(f"data_root does not exist: {data_root}")
        print(f"data_root={data_root}")
        _check_split_integrity(config)
        print("split_integrity=ok")

        model_config = dict(config)
        model_config["model"] = dict(config.get("model", {}))
        if not args.allow_pretrained:
            model_config["model"]["pretrained_encoder"] = False
            print("pretrained_encoder=false (preflight avoids external weight downloads)")
        device = resolve_device(args.device or config.get("device"))
        model = build_model_from_config(model_config, device)
        print(f"model_device={device}")
        dataset = build_patient_dataset(
            config,
            split=str(config.get("train_split", config.get("split", "train"))),
            train=False,
        )
        loader = DataLoader(dataset, batch_size=max(int(args.batch_size), 1), shuffle=False, num_workers=0)
        batch = move_batch(next(iter(loader)), device)
        print(f"batch_image_shape={tuple(batch['image'].shape)}")
        print(f"batch_mask_shape={tuple(batch['mask'].shape)}")
        model.train()
        model.zero_grad(set_to_none=True)
        output = model.forward_annotation(batch["image"]) if hasattr(model, "forward_annotation") else model(batch["image"])
        logits = output["logits"] if isinstance(output, dict) else output
        loss = annotation_loss(logits, batch["mask"])[0]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"preflight loss is non-finite: {float(loss.detach())}")
        loss.backward()
        finite, gradient_count = _finite_gradients(model)
        if not finite:
            raise FloatingPointError("preflight produced a non-finite gradient")
        print(f"loss={float(loss.detach()):.6f}")
        print(f"finite_gradients={finite} gradient_tensors={gradient_count}")

        output_path = Path(str(config.get("output", "weights/self_audit/preflight.pt")))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".preflight-", dir=output_path.parent, delete=True):
            pass
        print(f"checkpoint_dir_writable={output_path.parent}")
    except Exception as exc:
        print(f"PREFLIGHT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("PREFLIGHT_OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
