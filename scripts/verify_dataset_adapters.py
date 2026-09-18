"""Smoke-test audited adapters, DataLoader collation, and the current model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_audit.data.mixed import MixedCardiacDataset
from self_audit.training._utils import build_model_from_config, build_patient_dataset, resolve_device
from self_audit.training.unified_config import load_unified_config


def _model_smoke(model: torch.nn.Module, dataset, device: torch.device, batch_size: int) -> tuple[tuple[int, ...], str]:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    images = batch["image"].to(device)
    if images.ndim != 4 or images.shape[1] != 3:
        raise AssertionError(f"adapter emitted invalid model input {tuple(images.shape)}")
    with torch.no_grad():
        output = model.forward_annotation(images, turns=1)
    if not isinstance(output, dict):
        raise AssertionError("current model forward did not return its dict contract")
    return tuple(int(value) for value in images.shape), type(output).__name__


def run(config_path: str | Path, *, device_name: str = "cpu", num_workers: int = 0, batch_size: int = 1, allow_missing_sources: bool = False) -> dict:
    config = load_unified_config(config_path)
    legacy = config.to_legacy_dataset_config()
    legacy["num_workers"] = int(num_workers)
    legacy["batch_size"] = int(batch_size)
    legacy["image_size"] = min(int(legacy.get("image_size", 256)), 64)
    device = resolve_device(device_name)
    model_config = config.to_legacy_model_config()
    model_config["num_classes"] = 4
    model_config["model"]["num_classes"] = 4
    model_config["model"]["pretrained_encoder"] = False
    model_config["model"]["encoder_allow_fallback"] = True
    model_config["model"].pop("fallback", None)
    model = build_model_from_config(model_config, device).to(device).eval()
    results: dict[str, object] = {"config": str(config_path), "device": str(device), "sources": {}}

    if config.dataset.name == "mixed":
        source_datasets = {}
        missing = {}
        for source_name, source in config.dataset.sources.items():
            child = dict(legacy)
            child["dataset"] = source_name
            child["data_root"] = source["data_root"]
            child["split_manifest"] = source.get("split_manifest")
            child["sources"] = {source_name: dict(source)}
            try:
                source_datasets[source_name] = build_patient_dataset(child, split="train", train=False)
            except Exception as exc:
                if not allow_missing_sources:
                    raise
                missing[source_name] = str(exc)
        if missing:
            results["missing_sources"] = missing
        if len(source_datasets) < 2:
            raise RuntimeError("mixed smoke requires at least two available source datasets")
        dataset = MixedCardiacDataset(source_datasets)
        shape, output_type = _model_smoke(model, dataset, device, batch_size)
        results["mixed"] = {"available_sources": sorted(source_datasets), "input_shape": shape, "output_type": output_type}
        for source_name, source_dataset in source_datasets.items():
            shape, output_type = _model_smoke(model, source_dataset, device, batch_size)
            results["sources"][source_name] = {"samples": len(source_dataset), "input_shape": shape, "output_type": output_type}
    else:
        dataset = build_patient_dataset(legacy, split="train", train=False)
        shape, output_type = _model_smoke(model, dataset, device, batch_size)
        results["sources"][config.dataset.name] = {"samples": len(dataset), "input_shape": shape, "output_type": output_type}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allow-missing-sources", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        args.config,
        device_name=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        allow_missing_sources=args.allow_missing_sources,
    ), indent=2))


if __name__ == "__main__":
    main()
