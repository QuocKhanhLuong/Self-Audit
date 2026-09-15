#!/usr/bin/env python
"""Build the image-only manifest and readiness inventory for one dataset.

Read-only with respect to the dataset root: it probes headers, never writes
there, and never opens a mask or an annotation sidecar.

Usage
-----
    python scripts/prepare_maskfree_data.py \
        --root /path/to/ACDC --dataset acdc --output runs/maskfree150/data

    # Record a NOT INSPECTED inventory when the root is not on this machine.
    python scripts/prepare_maskfree_data.py \
        --root /remote/ACDC --dataset acdc --output out --allow-missing-root

Outputs under ``--output``:

* ``manifest_<dataset>.json``  -- the frozen cohort, splits and partition ids;
* ``inventory_<dataset>.json`` -- readiness, limitations, unresolved metadata;
* ``inventory_<dataset>.md``   -- the same inventory for a human reader.

Exit codes: ``0`` usable, ``2`` root missing (unless ``--allow-missing-root``),
``3`` root inspected but no usable training units.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from self_audit_maskfree.progress import TerminalProgress, current_progress  # noqa: E402


def _atomic_write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _not_inspected(root: Path, dataset: str, reason: str) -> dict[str, Any]:
    return {
        "inventory_status": "NOT_INSPECTED",
        "dataset": dataset,
        "root": str(root),
        "reason": reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records_available": None,
        "resolved_protocol": None,
        "note": (
            "The dataset root is not present on this machine. No cohort, split, "
            "protocol or readiness claim is made. This is not an empty dataset."
        ),
    }


def _inventory(manifest: dict[str, Any], probe: dict[str, Any] | None) -> dict[str, Any]:
    records = manifest["records"]
    unresolved = {
        "records_without_native_affine": [
            r["unit_id"] for r in records if not r["native_grid_export"]
        ][:50],
        "records_without_valid_spacing": [
            r["unit_id"] for r in records if not r["spacing_valid"]
        ][:50],
        "records_without_orientation": [
            r["unit_id"] for r in records if r["orientation"] is None
        ][:50],
        "records_cine_eligible": [r["unit_id"] for r in records if r["cine_eligible"]][:50],
    }
    return {
        "inventory_status": manifest["readiness"]["inventory_status"],
        "dataset": manifest["dataset"],
        "root": manifest["root"],
        "manifest_id": manifest["manifest_id"],
        "schema_version": manifest["schema_version"],
        "contract_version": manifest["contract_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "resolved_protocol": manifest["resolved_protocol"],
        "protocol_reason": manifest["protocol_reason"],
        "requested_protocol": manifest["requested_protocol"],
        "split_provenance": manifest["split_provenance"],
        "readiness": manifest["readiness"],
        "limitations": manifest["limitations"],
        "duplicates_collapsed": manifest["duplicates"],
        "unresolved_metadata": unresolved,
        "unit_probe": probe,
        "supervision_ledger": {
            "mask_files_opened": 0,
            "annotation_sidecars_opened": 0,
            "diagnosis_labels_used": False,
            "manual_roi_used": False,
            "split_selected_by_annotation": False,
        },
    }


def _markdown(inventory: dict[str, Any]) -> str:
    lines = [
        f"# Mask-free data inventory — {inventory.get('dataset')}",
        "",
        f"- status: **{inventory.get('inventory_status')}**",
        f"- root: `{inventory.get('root')}`",
        f"- generated: {inventory.get('generated_at')}",
    ]
    if inventory.get("inventory_status") == "NOT_INSPECTED":
        lines += [
            f"- reason: {inventory.get('reason')}",
            "",
            inventory.get("note", ""),
            "",
        ]
        return "\n".join(lines) + "\n"

    readiness = inventory["readiness"]
    lines += [
        f"- manifest id: `{inventory['manifest_id']}`",
        f"- resolved protocol: **{inventory['resolved_protocol']}**",
        f"  - reason: {inventory['protocol_reason']}",
        "",
        "## Readiness",
        "",
        f"- image files seen: {readiness['n_image_files_seen']}",
        f"- source files seen: {readiness['n_source_files']}",
        f"- acquired volumes (source frame identities): {readiness['n_volumes']}",
        f"- acquired frames enumerated: {readiness['n_frames_enumerated']}",
        f"- counted units (every Z slice): {readiness['n_units']}",
        f"- counted records (unit alias): {readiness['n_records']}",
        f"- duplicate re-exports collapsed: {readiness['n_duplicates_collapsed']}",
        f"- patients: {readiness['n_patients']}",
        f"- records per split: {readiness['per_split_records']}",
        f"- patients per split: {readiness['per_split_patients']}",
        f"- source formats: {readiness['source_formats']}",
        f"- native geometry available: {readiness['native_geometry_available']}",
        f"- cine-eligible volume frames: {readiness['cine_eligible_records']}",
        f"- usable for training: {readiness['usable_for_training']}",
        "",
        "## Split provenance",
        "",
        f"- rule: `{inventory['split_provenance']['rule']}`",
        f"- official test membership preserved: {inventory['split_provenance']['official_test_membership']}",
        f"- selection inputs: {inventory['split_provenance']['selection_inputs']}",
        "",
        "## Supervision ledger",
        "",
    ]
    for key, value in inventory["supervision_ledger"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Limitations", ""]
    for item in inventory["limitations"]:
        lines.append(f"- **{item['code']}** — {item['detail']}")
    probe = inventory.get("unit_probe")
    if probe:
        lines += ["", "## Unit probe", ""]
        for key, value in probe.items():
            lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def _probe_unit(manifest: dict[str, Any], image_size: int, seed: int) -> dict[str, Any]:
    from self_audit_maskfree.data import ImageOnlyDataset
    for split in ("train", "dev", "test"):
        dataset = ImageOnlyDataset(manifest, split=split, image_size=image_size, seed=seed)
        if len(dataset) == 0:
            continue
        unit = dataset[0]
        return {
            "split": split,
            "unit_id": unit.record["unit_id"],
            "fitting_image_shape": list(unit.fitting.image.shape),
            "fitting_support_pixels": int(unit.fitting.support.sum()),
            "selection_support_pixels": int(unit.selection.support.sum()),
            "verify_reserved_pixels": unit.record["support_counts"]["verify_reserved"],
            "native_support_counts": unit.record["support_counts"]["native"],
            "partition_id": unit.record["partition_id"],
            "split_fingerprint": dataset.fingerprint(),
            "export_grid": unit.record["export_grid"],
        }
    return {"split": None, "reason": "no unit available to probe"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Image-only manifest and readiness inventory for the mask-free pipeline",
    )
    parser.add_argument("--root", required=True, help="dataset root, read-only")
    parser.add_argument("--dataset", required=True, choices=("acdc", "mnms"))
    parser.add_argument("--output", required=True, help="directory for manifest and inventory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocol", default="auto", choices=("auto", "spatial_predictive", "cine_predictive")
    )
    parser.add_argument("--depth-axis", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="skip loading one unit end to end after discovery",
    )
    parser.add_argument(
        "--allow-missing-root",
        action="store_true",
        help="record a NOT INSPECTED inventory and exit 0 when the root is absent",
    )
    parser.add_argument("--progress", choices=("compact", "verbose"), default=None,
                        help="console display; default compact tqdm (or MASKFREE_PROGRESS)")
    args = parser.parse_args(argv)

    with TerminalProgress(mode=args.progress) as progress:
        progress.attach(Path(args.output).expanduser() / "inventory_progress.jsonl")
        progress.update(dataset=args.dataset)
        return _execute(args)


def _execute(args: argparse.Namespace) -> int:
    progress = current_progress()
    with progress.stage("imports.load", modules="torch, numpy, nibabel, image-only discovery"):
        from self_audit_maskfree.data import DataRootError, discover_dataset, save_manifest
        from self_audit_maskfree.data.discovery import MixedStudyGeometryError

    output = Path(args.output).expanduser()
    root = Path(args.root).expanduser()

    try:
        with progress.stage("inventory.discover", root=str(root)):
            manifest = discover_dataset(
                root,
                args.dataset,
                seed=args.seed,
                protocol=args.protocol,
                depth_axis=args.depth_axis,
            )
    except DataRootError as exc:
        inventory = _not_inspected(root, args.dataset, str(exc))
        _atomic_write(
            output / f"inventory_{args.dataset}.json",
            json.dumps(inventory, indent=2),
        )
        _atomic_write(output / f"inventory_{args.dataset}.md", _markdown(inventory))
        print(f"NOT INSPECTED: {exc}")
        print(f"inventory written to {output / f'inventory_{args.dataset}.json'}")
        return 0 if args.allow_missing_root else 2
    except MixedStudyGeometryError as exc:
        inventory = {
            "inventory_status": "FAILED", "training_status": "NOT_STARTED",
            "dataset": args.dataset, "root": str(root),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__, "reason": str(exc),
            "geometry_conflict": getattr(exc, "details", None),
        }
        path = output / f"inventory_{args.dataset}.json"
        _atomic_write(path, json.dumps(inventory, indent=2) + "\n")
        _atomic_write(output / f"inventory_{args.dataset}.md",
                      "# Inventory FAILED; training NOT_STARTED\n\n" + str(exc) + "\n")
        progress.event("inventory.failed", **inventory, diagnostic_path=str(path))
        return 3

    with progress.stage("inventory.unit_probe", image_size=args.image_size, skipped=args.no_probe):
        probe = None if args.no_probe else _probe_unit(manifest, args.image_size, args.seed)
    inventory = _inventory(manifest, probe)

    with progress.stage("inventory.write", path=str(output)):
        manifest_path = save_manifest(manifest, output / f"manifest_{args.dataset}.json")
        _atomic_write(output / f"inventory_{args.dataset}.json", json.dumps(inventory, indent=2))
        _atomic_write(output / f"inventory_{args.dataset}.md", _markdown(inventory))

    readiness = manifest["readiness"]
    print(f"dataset={args.dataset} protocol={manifest['resolved_protocol']}")
    print(
        f"volumes={readiness['n_volumes']} frames={readiness['n_frames_enumerated']} "
        f"units={readiness['n_units']} patients={readiness['n_patients']} "
        f"duplicates_collapsed={readiness['n_duplicates_collapsed']} "
        f"splits={readiness['per_split_records']}"
    )
    if progress.mode == "verbose":
        for item in manifest["limitations"]:
            print(f"limitation: {item['code']}")
    print(f"manifest written to {manifest_path}")
    if not readiness["usable_for_training"]:
        print("NOT USABLE: no training units discovered")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
