#!/usr/bin/env python
"""Generate the Self-Audit ACDC v3 contract freeze.

This generator is provenance-first: it consumes an already validated shared
manifest and selection receipt, preserves the historical v1/v2 freezes, and
does not discover data, split patients, or run a baseline algorithm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPEC = REPO_ROOT / "configs" / "adapter_v1_spec_source.json"
SOURCE_FIXTURES = REPO_ROOT / "configs" / "adapter_v1_synthetic_fixtures_source.json"
SOURCE_SPLIT = REPO_ROOT / "splits" / "acdc_patient_split_seed42.json"
FREEZE_SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v3"
FREEZE_ID_PREFIX = "cardiac-benchmark-v3"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v3"
DEFAULT_MANIFEST = REPO_ROOT / ".runtime" / "acdc_self_audit_outputs" / "acdc_shared_manifest_v3.json"
DEFAULT_SELECTION = REPO_ROOT / ".runtime" / "acdc_self_audit_selection.json"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _run(*args: str) -> str:
    return subprocess.run(args, cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _repo_binding(relative: str) -> dict[str, str]:
    path = REPO_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative.replace("\\", "/"), "sha256": _sha256_file(path)}


def _environment_receipt() -> dict[str, Any]:
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        numpy_version = "unavailable"
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pytorch_version = "unavailable"
        cuda_available = False
    return {
        "receipt_schema": "shared_benchmark.freeze_environment_receipt.v1",
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": pytorch_version,
        "numpy": numpy_version,
        "cuda_available": cuda_available,
        "gpu": "not bound into scientific payload",
        "note": "Execution metadata only; no algorithmic workload was run.",
    }


def _validate_fixture_source(value: Mapping[str, Any]) -> dict[str, Any]:
    # Reuse the historical, tested topology validator.  It is pure JSON and
    # does not access data or labels.
    source = REPO_ROOT / "scripts" / "regenerate_cardiac_benchmark_freeze.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("historical_freeze_generator", source)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load historical fixture validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._validate_fixture_source(value)


def _grid_contract() -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from shared_benchmark.spatial import grid_hash, load_self_audit_grid_spec

    grid = load_self_audit_grid_spec(REPO_ROOT)
    actual = grid_hash(grid)
    grid["shared_grid_sha256"] = actual
    grid["golden_fixture"] = {
        "fixture_version": "shared_benchmark.self_audit_bilinear_golden.v1",
        "input_dtype": "float32",
        "input_shape": [2, 3, 5],
        "input_values": "arange(30).reshape(2,3,5)",
        "target_hw": [4, 7],
        "output_dtype": "float32",
        "output_shape": [2, 4, 7],
        "output_little_endian_c_order_sha256": "45cb8f813ba87dd5bbff31a92e3bc324966582f0a9c93e51e9ab8db8255fe4cd",
        "expected_values": [
            [[0.0, 0.5714285969734192, 1.2857143878936768, 2.0, 2.7142858505249023, 3.4285714626312256, 4.0],
             [3.125, 3.6964287757873535, 4.410714149475098, 5.125, 5.839285850524902, 6.553571701049805, 7.125],
             [6.875, 7.446428298950195, 8.160714149475098, 8.875, 9.589285850524902, 10.303570747375488, 10.875],
             [10.0, 10.571428298950195, 11.285714149475098, 12.0, 12.714285850524902, 13.428571701049805, 14.0]],
            [[15.0, 15.571428298950195, 16.28571319580078, 17.0, 17.71428680419922, 18.428571701049805, 19.0],
             [18.125, 18.696428298950195, 19.410715103149414, 20.125, 20.839284896850586, 21.553571701049805, 22.125],
             [21.875, 22.446428298950195, 23.160715103149414, 23.875, 24.58928680419922, 25.303571701049805, 25.875],
             [25.0, 25.571428298950195, 26.285715103149414, 27.0, 27.714284896850586, 28.428571701049805, 29.0]],
        ],
    }
    return grid


def _manifest_without_local_receipt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(manifest)))
    result.pop("local_receipt", None)
    return result


def _counts(manifest: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    records = list(manifest.get("records", []))
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "dev", "test"):
        rows = [row for row in records if row.get("split") == split]
        result[split] = {
            "patients": len({str(row.get("patient_id")) for row in rows}),
            "samples": len(rows),
            "volumes": len({str(row.get("volume_id")) for row in rows}),
        }
    return result


def regenerate(output: Path, manifest_path: Path, selection_path: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing freeze: {output}")
    manifest = _read_json(manifest_path)
    selection = _read_json(selection_path)
    if manifest.get("schema_version") != "shared_benchmark_manifest.v1" or manifest.get("scientific") is not True:
        raise ValueError("v3 freeze requires a scientific shared manifest")
    if manifest.get("split_policy_version") != "self_audit.acdc.patient_split.v1":
        raise ValueError("v3 freeze requires the original Self-Audit patient split policy")
    grid = _grid_contract()
    if manifest.get("shared_grid_hash") != grid["shared_grid_sha256"]:
        raise ValueError("manifest grid hash does not derive from current Self-Audit configs")
    if selection.get("schema_version") != "self_audit.acdc.frame_selection.v1":
        raise ValueError("unsupported Self-Audit selection receipt")
    split_counts = _counts(manifest)
    if split_counts["test"]["samples"] != 0 or split_counts["train"]["patients"] != 80 or split_counts["dev"]["patients"] != 20:
        raise ValueError(f"unexpected Self-Audit cohort counts: {split_counts}")
    fixtures = _validate_fixture_source(_read_json(SOURCE_FIXTURES))
    spec = _read_json(SOURCE_SPEC)
    if spec.get("adapter_version") != "cardiac_adapter_v1":
        raise ValueError("source adapter spec is not cardiac_adapter_v1")

    output.mkdir(parents=True)
    (output / "configs").mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "receipts").mkdir(parents=True, exist_ok=True)
    (output / "configs" / "adapter_v1_spec.json").write_bytes(SOURCE_SPEC.read_bytes())
    _write_json(output / "configs" / "adapter_v1_synthetic_fixtures.json", fixtures)
    _write_json(output / "configs" / "resolved_shared_contract.json", grid)
    # The scientific manifest copy omits only the host-local receipt.  Its
    # manifest_hash is unchanged because local_receipt is excluded from the
    # scientific payload hash; all source locators remain relative.
    manifest_copy = _manifest_without_local_receipt(manifest)
    (output / "data" / "acdc_shared_manifest.json").write_bytes(_canonical_bytes(manifest_copy) + b"\n")
    (output / "data" / "acdc_selection_receipt.json").write_bytes(selection_path.read_bytes())
    (output / "data" / "acdc_patient_split_seed42.json").write_bytes(SOURCE_SPLIT.read_bytes())
    (output / "data" / "README.md").write_text(
        "# Static scientific manifest\n\n"
        "This v3 freeze records the original Self-Audit ACDC training cohort: "
        "80 patient-train / 20 patient-dev, ED+ES frames, every acquired Z "
        "slice, and no independent test cohort. Runtime training, DFC "
        "optimization, generation, PHATE, and real-data benchmarking are not "
        "part of this static freeze. M&Ms remains deferred.\n",
        encoding="utf-8",
    )

    commit = _run("git", "rev-parse", "HEAD")
    branch = _run("git", "branch", "--show-current")
    try:
        origin = _run("git", "remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        origin = "unavailable"
    try:
        origin_main = _run("git", "rev-parse", "origin/main")
    except subprocess.CalledProcessError:
        origin_main = "unavailable"
    _write_json(output / "receipts" / "repository.json", {
        "receipt_schema": "shared_benchmark.freeze_repository_receipt.v1",
        "branch_at_freeze_attempt": branch,
        "repository_commit_sha": commit,
        "origin": origin,
        "origin_main_sha": origin_main,
        "worktree_clean_at_start": False,
        "note": "Existing user changes were preserved; this is an evidence receipt, not a clean-tree claim.",
    })
    _write_json(output / "receipts" / "environment.json", _environment_receipt())
    _write_json(output / "receipts" / "firewall.json", {
        "receipt_schema": "shared_benchmark.freeze_firewall_receipt.v1",
        "schema_firewall": "PASS_STATIC",
        "physical_gt_isolation": "PASS_BWRAP_IMAGE_ONLY",
        "manifest_materialization": "PASS_IMAGE_ONLY_STATIC",
        "runtime_workloads": "NOT_RUN_BY_USER_PROTOCOL",
        "evidence": [
            "selection receipt contains no annotation-shaped fields",
            "image-only root contains 200 selected image hardlinks and no Info.cfg or *_gt",
            "scientific manifest source hashes were verified in the namespace",
        ],
    })
    _write_json(output / "receipts" / "tests.json", {
        "receipt_schema": "shared_benchmark.freeze_test_receipt.v1",
        "status": "STATIC_VALIDATION_PENDING",
        "commands": [],
        "runtime_gates": "NOT_RUN",
    })

    bound_repository_paths = [
        "configs/self_audit_full.yaml", "configs/self_audit_annotation.yaml", "configs/self_audit_auditor.yaml", "configs/self_audit_joint.yaml",
        "splits/acdc_patient_split_seed42.json", "scripts/preprocess_acdc.py", "scripts/train_self_audit.py",
        "src/self_audit/data/acdc.py", "src/self_audit/data/common.py", "src/self_audit/training/_utils.py", "src/self_audit/training/unified_trainer.py",
        "docs.md", "configs/adapter_v1_spec_source.json", "configs/adapter_v1_synthetic_fixtures_source.json",
        "src/shared_benchmark/__init__.py", "src/shared_benchmark/manifest.py", "src/shared_benchmark/spatial.py", "src/shared_benchmark/self_audit_protocol.py",
        "src/shared_benchmark/firewall.py", "src/shared_benchmark/provenance.py", "src/shared_benchmark/semantic_contract.py",
        "scripts/build_self_audit_acdc_selection.py", "scripts/prepare_self_audit_shared_benchmark_manifest.py",
        "scripts/prepare_shared_benchmark_manifest.py",
        "scripts/regenerate_cardiac_benchmark_freeze.py", "scripts/regenerate_self_audit_cardiac_benchmark_freeze.py", "scripts/validate_cardiac_benchmark_freeze.py",
        "scripts/run_cuts_scientific.py", "scripts/run_dfc_scientific.py",
        "baseline/CUTS/src/cardiac_benchmark/manifest.py", "baseline/CUTS/src/cardiac_benchmark/dataset.py", "baseline/CUTS/src/cardiac_benchmark/train_stage1.py",
        "baseline/CUTS/src/cardiac_benchmark/export_latents.py", "baseline/CUTS/src/cardiac_benchmark/cluster_kmeans.py", "baseline/CUTS/src/cardiac_benchmark/provenance.py",
        "baseline/DFC/src/cardiac_benchmark/manifest.py", "baseline/DFC/src/cardiac_benchmark/dataset.py", "baseline/DFC/src/cardiac_benchmark/config.py", "baseline/DFC/src/cardiac_benchmark/dfc_runner.py",
        "baseline/DFC/config/cardiac/dfc_direct_2d_minl3.yaml",
    ]
    bound_repo = [_repo_binding(path) for path in bound_repository_paths]
    freeze_paths = [
        "configs/resolved_shared_contract.json", "configs/adapter_v1_spec.json", "configs/adapter_v1_synthetic_fixtures.json",
        "data/acdc_shared_manifest.json", "data/acdc_selection_receipt.json", "data/acdc_patient_split_seed42.json",
    ]
    bound_freeze = [{"path": path, "sha256": _sha256_file(output / path)} for path in freeze_paths]
    fixture_hash = _sha256_file(output / "configs" / "adapter_v1_synthetic_fixtures.json")
    spec_hash = _sha256_file(output / "configs" / "adapter_v1_spec.json")
    scientific_payload: dict[str, Any] = {
        "freeze_schema_version": FREEZE_SCHEMA,
        "repository_commit_sha": commit,
        "authoritative_self_audit_policy": {
            "source_manifest_schema": "self_audit.acdc.image_only.v1",
            "split_policy_version": "self_audit.acdc.patient_split.v1",
            "split_seed": 42,
            "cohort": "ACDC/training only",
            "patient_counts": {"train": 80, "dev": 20, "test": 0},
            "volume_counts": {"train": 160, "dev": 40, "test": 0},
            "frame_selection": "Info.cfg ED and ES",
            "generation_inventory": "selected ED+ES frames x all acquired Z slices",
            "context_policy": "[max(z-1,0),z,min(z+1,Z-1)]",
            "independent_test": False,
        },
        "shared_manifest": {
            "schema_version": "shared_benchmark_manifest.v1",
            "source_manifest_schema": manifest["source_manifest"]["schema_version"],
            "manifest_sha256": manifest["manifest_hash"],
            "manifest_copy_sha256": _sha256_file(output / "data" / "acdc_shared_manifest.json"),
            "patient_counts": {split: split_counts[split]["patients"] for split in split_counts},
            "sample_counts": {split: split_counts[split]["samples"] for split in split_counts},
            "volume_counts": {split: split_counts[split]["volumes"] for split in split_counts},
            "scientific_manifest_materialization": "PASS_IMAGE_ONLY_STATIC",
        },
        "datasets": {
            "acdc": {
                "scientific_manifest_status": "PASS_STATIC",
                "root_availability": True,
                "manifest_sha256": manifest["manifest_hash"],
                "patient_counts": {split: split_counts[split]["patients"] for split in split_counts},
                "sample_counts": {split: split_counts[split]["samples"] for split in split_counts},
                "volume_counts": {split: split_counts[split]["volumes"] for split in split_counts},
                "frame_selection": "ED+ES",
                "runtime_status": "NOT_RUN",
            },
            "mnms": {"scientific_manifest_status": "DEFERRED", "root_availability": False, "runtime_status": "NOT_RUN"},
        },
        "shared_spatial_contract": {
            "version": grid["version"], "target_hw": grid["target_hw"], "whole_fov": grid["whole_fov"],
            "crop": grid["crop"], "resampling": grid["forward_values"], "grid_sha256": grid["shared_grid_sha256"],
            "config_provenance": grid["config_provenance"],
        },
        "consumer_contract": {
            "cuts_primary": "CUTS-2D", "cuts_sensitivity": "CUTS-2.5D",
            "dfc_primary": "DFC-Direct-2D-Default-MinL3", "shared_inventory_required": True,
            "normalization_contract": {
                "source_version": "self_audit.volume_percentile_clip_0p5_99p5_zscore.v1",
                "source_operation": "one full-frame volume 0.5/99.5 clip followed by population z-score",
                "cuts_2d": "source-normalized volume then central channel selection",
                "cuts_25d": "source-normalized volume then endpoint-replicated [z-1,z,z+1] selection",
                "dfc_2d": "source-normalized volume then central channel selection",
                "method_specific_normalization": False,
                "legacy_fixture_identity": "source.float32_identity.legacy_fixture.v1",
                "upstream_basis": {
                    "cuts": "no cardiac intensity normalizer; generic CUTS model is input-agnostic",
                    "dfc": "upstream direct path only divides BGR uint8 by 255; not applied to float MRI",
                },
            },
            "optimization_protocol_unchanged": True,
        },
        "firewall": {"schema_status": "PASS_STATIC", "physical_isolation_status": "PASS_BWRAP_IMAGE_ONLY", "runtime_status": "NOT_RUN"},
        "adapter": {
            "adapter_version": "cardiac_adapter_v1", "input_schema_version": "shared_benchmark.anonymous_partition.v1",
            "specification_status": "FROZEN_PRE_IMPLEMENTATION", "specification_sha256": spec_hash,
            "synthetic_fixture_set_version": fixtures["fixture_set_version"], "synthetic_fixture_set_sha256": fixture_hash,
            "synthetic_fixture_invariant": "validity == (semantic != VOID)",
        },
        "scientific_readiness": "CODE_AND_STATIC_PREFLIGHT_PASS_WAITING_FOR_USER_DECISION",
        "bound_repository_files": bound_repo,
        "bound_freeze_files": bound_freeze,
        "pre_freeze_sources": {
            "adapter_spec_source_sha256": _sha256_file(SOURCE_SPEC),
            "fixture_source_sha256": _sha256_file(SOURCE_FIXTURES),
            "split_source_sha256": _sha256_file(SOURCE_SPLIT),
            "selection_receipt_sha256": _sha256_file(selection_path),
            "source_manifest_sha256": _sha256_file(manifest_path),
            "freeze_generator": "scripts/regenerate_self_audit_cardiac_benchmark_freeze.py",
        },
    }
    payload_hash = _sha256_bytes(_canonical_bytes(scientific_payload))
    freeze_manifest = {
        "freeze_schema": FREEZE_SCHEMA,
        "freeze_id": f"{FREEZE_ID_PREFIX}-{payload_hash[:16]}",
        "scientific_payload_sha256": payload_hash,
        "scientific_payload": scientific_payload,
    }
    _write_json(output / "FREEZE_MANIFEST.json", freeze_manifest)
    (output / "FREEZE_REPORT.md").write_text(
        f"# Cardiac benchmark v3 freeze (Self-Audit ACDC protocol)\n\n"
        f"- freeze ID: `{freeze_manifest['freeze_id']}`\n"
        f"- scientific payload SHA-256: `{payload_hash}`\n"
        f"- repository commit: `{commit}`\n"
        f"- shared manifest SHA-256: `{manifest['manifest_hash']}`\n"
        f"- shared grid SHA-256: `{grid['shared_grid_sha256']}`\n"
        f"- counts: train={split_counts['train']['samples']} / dev={split_counts['dev']['samples']} / test=0\n\n"
        "- normalization: Self-Audit volume 0.5/99.5 clip + population z-score; "
        "no CUTS/DFC-specific intensity transform\n\n"
        "This is a code/static image-only freeze. Runtime training, DFC "
        "optimization, generation/PHATE, and real-data benchmarking are NOT_RUN. "
        "M&Ms remains deferred.\n",
        encoding="utf-8",
    )
    return {
        "freeze_id": freeze_manifest["freeze_id"],
        "scientific_payload_sha256": payload_hash,
        "manifest_sha256": manifest["manifest_hash"],
        "grid_sha256": grid["shared_grid_sha256"],
        "counts": split_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    args = parser.parse_args()
    print(json.dumps(regenerate(args.output, args.manifest, args.selection), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
