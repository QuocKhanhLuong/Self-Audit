"""Bounded synthetic CPU/NIfTI source and validation equivalence probe.

This probe is deliberately self-contained so the report's diagnostic numbers
can be reproduced without touching a real cohort or running training.  It
creates one temporary NIfTI-1 volume, assigns the discovered units to ``dev``
only for this comparison, and uses local export/reference stubs for the
``observe_epoch`` call.  It prints one JSON record; temporary files are removed
when the process exits.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

from self_audit_maskfree import epoch_validation
from self_audit_maskfree.data import discover_dataset, load_full_input
from self_audit_maskfree.data.cache import (
    SourceDataCache,
    default_source_cache,
    manifest_digest,
    reset_default_source_cache,
)
from self_audit_maskfree.data.geometry import read_slice_stack


UTC = dt.timezone.utc
CACHE_BUDGET = 48 * 48 * 16 * 4 * 2
IMAGE_SIZE = 24
BATCH_SIZE = 5
SEED = 20260915


def _source_probe(manifest: dict[str, object], records: list[dict[str, object]], digest: str) -> dict[str, object]:
    native_load = nib.load
    baseline_loads = [0]

    def counted_baseline(path, *args, **kwargs):
        baseline_loads[0] += 1
        return native_load(path, *args, **kwargs)

    nib.load = counted_baseline
    try:
        started = time.perf_counter()
        baseline_stacks = [
            read_slice_stack(
                row["path"],
                depth_axis=int(row["depth_axis"]),
                slice_index=int(row["slice_index"]),
                frame_index=row.get("frame_index"),
                frame_axis=row.get("frame_axis"),
            )
            for row in records
        ]
        baseline_seconds = time.perf_counter() - started
    finally:
        nib.load = native_load

    cache = SourceDataCache(max_bytes=CACHE_BUDGET)
    cache_loads = [0]

    def counted_cache(path, *args, **kwargs):
        cache_loads[0] += 1
        return native_load(path, *args, **kwargs)

    nib.load = counted_cache
    try:
        started = time.perf_counter()
        cold_stacks = [
            cache.get_slice_stack(
                row,
                manifest_digest_value=digest,
                manifest_id=manifest["manifest_id"],
            )
            for row in records
        ]
        cold_seconds = time.perf_counter() - started
    finally:
        nib.load = native_load

    started = time.perf_counter()
    warm_stacks = [
        cache.get_slice_stack(
            row,
            manifest_digest_value=digest,
            manifest_id=manifest["manifest_id"],
        )
        for row in records
    ]
    warm_seconds = time.perf_counter() - started
    return {
        "baseline_uncached_seconds": baseline_seconds,
        "baseline_nib_load_calls": baseline_loads[0],
        "cache_cold_seconds": cold_seconds,
        "cache_warm_seconds": warm_seconds,
        "cache_nib_load_calls": cache_loads[0],
        "outputs_equal": all(np.array_equal(a, b) for a, b in zip(baseline_stacks, cold_stacks)),
        "warm_outputs_equal": all(np.array_equal(a, b) for a, b in zip(cold_stacks, warm_stacks)),
        "cache_stats_after_cold_warm": cache.stats(),
    }


class _Student(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(offset, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat((inputs[:, 0:1], inputs[:, 1:2] + self.offset, inputs[:, 2:3]), dim=1)


def _validation_probe(manifest: dict[str, object]) -> dict[str, object]:
    original_reference = epoch_validation._run_reference

    def reference_stub(*args):
        return {
            "epoch": 0,
            "freeze_id": "freeze-cache-equivalence",
            "status": "UNAVAILABLE",
            "available": False,
            "reason": "measurement stub; no reference data opened",
            "students": {},
        }

    def run(mode: str) -> tuple[dict[str, object], list[torch.Tensor], list[dict[str, object]], float]:
        run_root = Path(tempfile.mkdtemp(prefix=f"maskfree_probe_{mode}_"))
        run_manifest = copy.deepcopy(manifest)
        run_cache = default_source_cache() if mode == "cached" else None
        if run_cache is not None:
            reset_default_source_cache()
        loaded_values: list[torch.Tensor] = []
        exported: list[dict[str, object]] = []

        def cached_loader(current_manifest, unit_id, *, image_size, _manifest_digest_value=None):
            value = load_full_input(
                current_manifest,
                unit_id,
                image_size=image_size,
                cache=run_cache,
                _manifest_digest_value=_manifest_digest_value,
            )
            loaded_values.append(value[0].detach().clone())
            return value

        def uncached_loader(current_manifest, unit_id, *, image_size):
            value = load_full_input(current_manifest, unit_id, image_size=image_size, cache=None)
            loaded_values.append(value[0].detach().clone())
            return value

        def export_prediction(output, **kwargs):
            exported.append({
                "prediction_name": kwargs["prediction_name"],
                "record": copy.deepcopy(kwargs["record"]),
                "labels": kwargs["labels"].detach().cpu().numpy().copy(),
            })
            return {
                "kind": "volume",
                "prediction_name": kwargs["prediction_name"],
                "unit_id": kwargs["record"]["unit_id"],
            }

        def freeze_predictions(output, entries, checkpoints, **kwargs):
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            frozen = {
                "freeze_id": "freeze-cache-equivalence",
                "required_methods": list(kwargs["required_methods"]),
                "completeness": {"required": list(kwargs["required_methods"])},
                "predictions": list(entries),
            }
            (output / "freeze_manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
            return frozen

        config = SimpleNamespace(
            dataset="acdc",
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            total_epochs=1,
            epoch_reference_config=None,
            scientific_identity=lambda: {"measurement": "source-cache"},
        )
        torch.manual_seed(SEED)
        models = {"student_no_audit": _Student(0.0), "student_audited": _Student(0.5)}
        components = SimpleNamespace(
            load_full_input=cached_loader if mode == "cached" else uncached_loader,
            export_prediction=export_prediction,
            freeze_predictions=freeze_predictions,
            validate_freeze=lambda frozen, require_complete: None,
        )
        trainer = SimpleNamespace(
            paths=SimpleNamespace(root=run_root),
            run_id="cache-equivalence",
            manifest=run_manifest,
            config=config,
            source={"package": {"combined": "measurement-source"}},
            models=models,
            components=components,
            device=torch.device("cpu"),
            pseudo_label_version=lambda epoch: f"measurement-{epoch}",
        )
        started = time.perf_counter()
        result = epoch_validation.observe_epoch(trainer, 0)
        elapsed = time.perf_counter() - started
        return result, loaded_values, exported, elapsed

    epoch_validation._run_reference = reference_stub
    try:
        uncached_result, uncached_values, uncached_exports, uncached_seconds = run("uncached")
        cached_result, cached_values, cached_exports, cached_seconds = run("cached")
    finally:
        epoch_validation._run_reference = original_reference

    exports_equal = len(uncached_exports) == len(cached_exports)
    if exports_equal:
        for left, right in zip(uncached_exports, cached_exports):
            exports_equal = (
                left["prediction_name"] == right["prediction_name"]
                and left["record"] == right["record"]
                and np.array_equal(left["labels"], right["labels"])
            )
            if not exports_equal:
                break
    return {
        "uncached_seconds": uncached_seconds,
        "cached_seconds": cached_seconds,
        "uncached_status": uncached_result.get("status"),
        "cached_status": cached_result.get("status"),
        "loaded_image_values_equal": len(uncached_values) == len(cached_values)
        and all(torch.equal(a, b) for a, b in zip(uncached_values, cached_values)),
        "native_export_outputs_equal": exports_equal,
        "uncached_export_count": len(uncached_exports),
        "cached_export_count": len(cached_exports),
        "uncached_cache_delta": uncached_result.get("source_cache"),
        "cached_cache_delta": cached_result.get("source_cache"),
    }


def main() -> None:
    started = dt.datetime.now(UTC)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="maskfree_luna_data_") as tmp:
        root = Path(tmp) / "acdc"
        source = root / "training" / "patient001" / "patient001_sa.nii.gz"
        source.parent.mkdir(parents=True, exist_ok=True)
        native = np.arange(48 * 48 * 16, dtype=np.float32).reshape(48, 48, 16)
        image = nib.Nifti1Image(native, np.eye(4, dtype=np.float64))
        image.header.set_zooms((1.0, 1.0, 1.0))
        image.header.set_xyzt_units("mm", None)
        nib.save(image, str(source))
        manifest = discover_dataset(root, "acdc", seed=42)
        for record in manifest["records"]:
            record["split"] = "dev"
        records = sorted(manifest["records"], key=lambda row: str(row["unit_id"]))
        digest = manifest_digest(manifest)
        source_result = _source_probe(manifest, records, digest)
        validation_result = _validation_probe(manifest)
    ended = dt.datetime.now(UTC)
    print(json.dumps({
        "schema": "maskfree150.luna_data_validation_measurement.v1",
        "utc_start": started.isoformat(),
        "utc_end": ended.isoformat(),
        "fixture": {
            "format": "NIfTI-1",
            "shape": [48, 48, 16],
            "dtype": "float32",
            "records": len(records),
            "split_override": "all records assigned dev for bounded measurement",
        },
        "reproduction": {
            "cache_budget_bytes": CACHE_BUDGET,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "seed": SEED,
        },
        "source": source_result,
        "validation": validation_result,
        "scientific_boundary": "synthetic CPU/NIfTI and stubbed reference/export only; not a real-data, GPU, clinical, or production-throughput claim",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
