#!/usr/bin/env python3
"""Run ADNet 2D under the few-shot labeled-support benchmark contract.

This runner deliberately does not import ADNet's upstream TestDataset because
that dataset reads query labels during inference. Query GT belongs only in a
separate evaluator after these outputs have been sealed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shared_benchmark.adnet_fewshot import (  # noqa: E402
    ADNetContractError,
    ADNetQueryRecord,
    ADNetSupportClass,
    file_sha256,
    load_query_manifest,
    load_support_manifest,
    seal_adnet_outputs,
)
from shared_benchmark.artifacts import code_identity, repository_identity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--query-image-root", type=Path)
    parser.add_argument("--support-asset-root", type=Path)
    parser.add_argument("--adnet-root", type=Path, default=ROOT / "baseline" / "ADNet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--tie-rule", default="fail_closed", choices=("fail_closed",))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    support_manifest = load_support_manifest(args.support_manifest, asset_root=args.support_asset_root)
    query_manifest = load_query_manifest(args.query_manifest, image_root=args.query_image_root)
    if not args.checkpoint.is_file():
        raise ADNetContractError(f"checkpoint is missing: {args.checkpoint}")
    checkpoint_sha256 = file_sha256(args.checkpoint)

    random.seed(args.seed)
    np.random.seed(args.seed)

    model, torch, device = _load_model(args.adnet_root, args.checkpoint, args.device, args.seed)
    code = code_identity(
        [
            ROOT / "scripts" / "run_adnet_fewshot.py",
            ROOT / "src" / "shared_benchmark" / "adnet_fewshot.py",
            args.adnet_root / "main_inference.py",
            args.adnet_root / "models" / "fewshot_anom.py",
            args.adnet_root / "models" / "backbone" / "torchvision_backbones.py",
        ],
        repo_root=ROOT,
    )
    run_config = {
        "baseline_name": "ADNet",
        "baseline_mode": "ADNet-2D-FewShot-Support-v1",
        "device": str(device),
        "seed": int(args.seed),
        "tie_rule": args.tie_rule,
        "checkpoint_sha256": checkpoint_sha256,
        "support_manifest_sha256": support_manifest.sha256,
        "query_manifest_sha256": query_manifest.sha256,
        "repository_identity": repository_identity(ROOT),
        "adnet_root": str(args.adnet_root),
    }

    rows: list[dict[str, Any]] = []
    for record in query_manifest.records:
        class_predictions: dict[int, np.ndarray] = {}
        for support in support_manifest.supports:
            class_predictions[support.canonical_class_id] = _predict_class(
                torch=torch,
                model=model,
                device=device,
                query=record,
                support=support,
            )
        metadata = seal_adnet_outputs(
            args.output_root,
            sample_id=record.sample_id,
            class_predictions=class_predictions,
            support_manifest=support_manifest,
            query_manifest=query_manifest,
            query_record=record,
            checkpoint_sha256=checkpoint_sha256,
            code_identity=code,
            run_config=run_config,
            tie_rule=args.tie_rule,
        )
        rows.append({
            "sample_id": record.sample_id,
            "status": metadata["completion_status"],
            "scientific_payload_sha256": metadata["scientific_payload_sha256"],
        })

    print(json.dumps({
        "baseline": "ADNet",
        "mode": "ADNet-2D-FewShot-Support-v1",
        "results": rows,
    }, sort_keys=True))
    return 0


def _load_model(adnet_root: Path, checkpoint: Path, device_name: str, seed: int):
    if not adnet_root.is_dir():
        raise ADNetContractError(f"ADNet root is missing: {adnet_root}")
    if str(adnet_root) not in sys.path:
        sys.path.insert(0, str(adnet_root))
    try:
        import torch
        import torch.nn as nn
        from models.fewshot_anom import FewShotSeg
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ADNetContractError(f"cannot import ADNet runtime dependencies: {exc}") from exc

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ADNetContractError("requested CUDA device but torch.cuda.is_available() is false")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = FewShotSeg(use_coco_init=False)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ADNetContractError("checkpoint must be a PyTorch state_dict or contain state_dict")

    has_module_prefix = any(str(key).startswith("module.") for key in state)
    if device.type == "cuda" and has_module_prefix:
        model = nn.DataParallel(model)
        load_state = state
    else:
        load_state = {
            str(key)[len("module."):] if str(key).startswith("module.") else str(key): value
            for key, value in state.items()
        }
    model.to(device)
    if hasattr(model, "device"):
        model.device = device
    model.load_state_dict(load_state, strict=True)
    model.eval()
    return model, torch, device


def _predict_class(*, torch: Any, model: Any, device: Any, query: ADNetQueryRecord, support: ADNetSupportClass) -> np.ndarray:
    query_image = _read_adnet_image(query.image_path)
    support_image = _read_adnet_image(support.image_path)
    support_mask = _read_binary_mask(support.mask_path)
    if query_image.shape[-2:] != support_image.shape[-2:] or support_image.shape[0] != support_mask.shape[0]:
        raise ADNetContractError("query/support images and support mask must share the ADNet 2D grid")
    if max(support.slice_indices) >= support_image.shape[0]:
        raise ADNetContractError("support slice index outside support volume")

    query_tensor = torch.from_numpy(query_image).float().to(device)
    support_image_tensor = torch.from_numpy(support_image).float().to(device)
    support_mask_tensor = torch.from_numpy(support_mask).float().to(device)

    support_images = []
    support_masks = []
    for index in support.slice_indices:
        if float(support_mask_tensor[index].sum().item()) <= 0.0:
            raise ADNetContractError(f"support slice {index} is empty for class {support.class_name}")
        support_images.append(support_image_tensor[[index]])
        support_masks.append(support_mask_tensor[[index]])

    with torch.no_grad():
        output, _, _ = model([support_images], [support_masks], [query_tensor], train=False)
        prediction = output.argmax(dim=1).detach().cpu().numpy().astype(np.uint8, copy=False)
    return np.ascontiguousarray(prediction)


def _read_adnet_image(path: Path) -> np.ndarray:
    volume = _read_volume(path).astype(np.float32, copy=False)
    std = float(volume.std())
    if std <= 0.0 or not np.isfinite(std):
        raise ADNetContractError(f"image has invalid standard deviation: {path}")
    normalized = (volume - float(volume.mean())) / std
    return np.ascontiguousarray(np.stack(3 * [normalized], axis=1), dtype=np.float32)


def _read_binary_mask(path: Path) -> np.ndarray:
    mask = _read_volume(path)
    binary = np.asarray(mask != 0, dtype=np.float32)
    return np.ascontiguousarray(binary)


def _read_volume(path: Path) -> np.ndarray:
    name = path.name.lower()
    if name.endswith(".npy"):
        value = np.load(path, allow_pickle=False)
    elif name.endswith(".nii") or name.endswith(".nii.gz"):
        try:
            import SimpleITK as sitk
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ADNetContractError(f"SimpleITK is required to read {path.name}: {exc}") from exc
        value = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    else:
        raise ADNetContractError(f"unsupported ADNet volume container: {path.name}")
    array = np.asarray(value)
    if array.ndim != 3 or not np.issubdtype(array.dtype, np.number):
        raise ADNetContractError(f"ADNet volume must be numeric [Z,H,W]: {path}")
    if not np.isfinite(array).all():
        raise ADNetContractError(f"ADNet volume contains non-finite values: {path}")
    return np.ascontiguousarray(array)


if __name__ == "__main__":
    raise SystemExit(main())
