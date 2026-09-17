"""Native development reference matching, confined to the post-freeze process."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from ..export import validate_freeze
from .freeze import sha256_file
from .metrics import write_json
from .native_reference_geometry import transfer_reference_indices
from .reference import ReferenceCase, ReferenceConfig, ReferenceConfigError, evaluate_reference

ARMS = ("student_no_audit", "student_audited")
METHODS = tuple(f"{arm}_full_input" for arm in ARMS)
LABEL_MAPS = {"acdc": {0: 0, 1: 1, 2: 2, 3: 3},
              "mnms": {0: 0, 1: 3, 2: 2, 3: 1}}
SCHEMA = "maskfree150.epoch_validation.v1"


def _bind_images(frozen: dict, image_manifest_path: Path) -> tuple[dict, list[dict]]:
    """Verify a hash-bound image-only dev cohort before any annotation access."""
    validate_freeze(frozen, require_complete=True)
    if set(frozen["completeness"]["required"]) != set(METHODS) or set(frozen["compared_methods"]) != set(METHODS):
        raise ReferenceConfigError("epoch validation requires exactly both full-input student arms")
    bindings = [row for row in frozen["checkpoints"] if row["name"] == "validation_image_manifest"]
    if (len(bindings) != 1 or Path(bindings[0]["path"]).resolve() != image_manifest_path.resolve() or
            sha256_file(image_manifest_path) != bindings[0]["sha256"]):
        raise ReferenceConfigError("image manifest is not the one bound to the epoch freeze")
    image_manifest = json.loads(image_manifest_path.read_text())
    if (image_manifest["dataset"] != frozen["dataset"] or
            image_manifest["resolved_protocol"] != frozen["protocol"] or
            image_manifest["manifest_id"] != frozen["manifest_id"]):
        raise ReferenceConfigError("epoch freeze/image manifest provenance mismatch")
    records = {row["unit_id"]: row for row in image_manifest["records"] if row["split"] == "dev"}
    expected = set(records)
    if not expected or set(frozen["required_unit_ids"]) != expected:
        raise ReferenceConfigError("epoch freeze does not cover exactly all development units")
    if any(set(frozen["method_units"].get(method, [])) != expected for method in METHODS):
        raise ReferenceConfigError("student development cohorts differ")
    train_patients = {row["patient_id"] for row in image_manifest["records"] if row["split"] == "train"}
    if train_patients & {row["patient_id"] for row in records.values()}:
        raise ReferenceConfigError("development patients overlap training")
    volumes: dict[str, dict] = {}
    # Join actual frozen volume IDs via unit IDs; never guess the exporter's
    # filename sanitization or infer a cine index from a mask filename.
    for entry in frozen["predictions"]:
        if entry["kind"] != "unit":
            if entry["split"] != "dev":
                raise ReferenceConfigError("non-development volume in epoch freeze")
            continue
        record = records[entry["unit_id"]]
        value = {**record, "frozen_volume_id": entry["volume_id"]}
        previous = volumes.setdefault(record["volume_id"], value)
        if previous["frozen_volume_id"] != value["frozen_volume_id"]:
            raise ReferenceConfigError("student volume identity differs across arms")
    frozen_volumes = {(r["patient_id"], r["volume_id"]) for r in frozen["predictions"] if r["kind"] == "volume"}
    if frozen_volumes != {(r["patient_id"], r["frozen_volume_id"]) for r in volumes.values()}:
        raise ReferenceConfigError("native volume cohort disagrees with the bound image manifest")
    return image_manifest, list(volumes.values())


def _sibling_mask(source: Path) -> Path | None:
    for suffix in (".nii.gz", ".nii"):
        if source.name.lower().endswith(suffix):
            stem = source.name[:-len(suffix)]
            candidates = [source.with_name(stem + "_gt" + extension) for extension in (".nii.gz", ".nii")]
            matches = [path for path in candidates if path.is_file()]
            if len(matches) > 1:
                raise ReferenceConfigError(f"ambiguous colocated masks for {source}")
            return matches[0] if matches else None
    return None


def _mnms_phases(root: Path, patients: set[str]) -> tuple[dict[str, set[int]], list[dict]]:
    """Read official zero-based ED/ES indices only in this isolated process."""
    phases: dict[str, set[int]] = {}
    provenance = []
    for path in sorted(root.rglob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = {"".join(c for c in name.lower() if c.isalnum()): name for name in (reader.fieldnames or [])}
            id_field = next((fields[key] for key in ("externalcode", "patientid", "patient", "subjectid", "subject", "id") if key in fields), None)
            if id_field is None or "ed" not in fields or "es" not in fields:
                continue
            used = False
            for row in reader:
                patient = str(row[id_field]).strip()
                if patient not in patients:
                    continue
                try:
                    values = [float(row[fields[key]]) for key in ("ed", "es")]
                    if any(not value.is_integer() or value < 0 for value in values):
                        raise ValueError("noninteger or negative ED/ES index")
                    indices = {int(value) for value in values}
                except (ValueError, TypeError, OverflowError) as error:
                    raise ReferenceConfigError(f"invalid M&Ms ED/ES metadata for {patient}: {path}") from error
                if patient in phases and phases[patient] != indices:
                    raise ReferenceConfigError(f"conflicting M&Ms ED/ES metadata for {patient}")
                phases[patient] = indices
                used = True
            if used:
                provenance.append({"path": str(path), "sha256": sha256_file(path), "frame_index_base": 0})
    return phases, provenance


def _auto_config(frozen: dict, manifest: dict, volumes: list[dict], output: Path) -> tuple[ReferenceConfig, dict]:
    import nibabel as nib

    dataset = frozen["dataset"]
    phases, metadata_files = ({}, [])
    if dataset == "mnms":
        phases, metadata_files = _mnms_phases(Path(manifest["root"]), {r["patient_id"] for r in volumes})
    duplicates: dict[str, list[dict]] = {}
    for duplicate in manifest.get("duplicates", []):
        duplicates.setdefault(duplicate["duplicate_of"], []).append(duplicate)
    cases = []
    skipped = []
    transfers = []
    source_hashes = {}
    mask_hashes = {}
    for record in sorted(volumes, key=lambda row: row["volume_id"]):
        source = Path(record["source_path"])
        patient = record["patient_id"]
        mask = _sibling_mask(source)
        mask_frame = None
        reason = None
        if len(record["shape"]) == 4:
            if dataset == "mnms" and mask is not None:
                indices = phases.get(patient)
                if indices is None:
                    reason = "M&Ms ED/ES metadata missing; sparse 4-D labels cannot define annotated frames"
                elif max(indices) >= record["shape"][3]:
                    raise ReferenceConfigError(f"M&Ms ED/ES metadata out of range for {source}")
                elif record["frame_index"] not in indices:
                    reason = "cine frame has no declared ED/ES annotation"
                else:
                    mask_frame = record["frame_index"]
            elif dataset == "acdc":
                matches = [(d, _sibling_mask(Path(d["source_path"])))
                           for d in duplicates.get(record["volume_id"], [])
                           if len(d.get("duplicate_geometry", {}).get("shape", [])) == 3]
                matches = [(d, p) for d, p in matches if p is not None]
                if len(matches) > 1:
                    raise ReferenceConfigError(f"ambiguous ACDC frame references for {record['volume_id']}")
                if matches:
                    duplicate, original_mask = matches[0]
                    mask_hashes[str(original_mask)] = sha256_file(original_mask)
                    mask, proof = transfer_reference_indices(
                        original_mask, duplicate["source_path"], record, output / "aligned_references")
                    transfers.append(proof)
                else:
                    reason = "no image-proven 3-D re-export with a colocated ACDC frame mask"
            else:
                reason = "no colocated native cine reference"
        if mask is None:
            reason = reason or "colocated native reference mask absent"
        if reason:
            skipped.append({"patient_id": patient, "volume_id": record["frozen_volume_id"], "reason": reason})
            continue
        if str(source) not in source_hashes:
            source_hashes[str(source)] = sha256_file(source)
        if source_hashes[str(source)] != record["source_hash"]:
            raise ReferenceConfigError(f"image source changed after discovery: {source}")
        mask_image = nib.load(str(mask))
        expected_shape = tuple(record["shape"]) if mask_frame is not None else tuple(record["native_shape"])
        if mask_image.shape != expected_shape:
            raise ReferenceConfigError(f"reference dimensions disagree with native image: {mask}")
        mask_hashes.setdefault(str(mask), sha256_file(mask))
        cases.append(ReferenceCase(patient_id=patient, volume_id=record["frozen_volume_id"],
                                   mask_path=str(mask), mask_frame_index=mask_frame,
                                   geometry_valid=True, note="native per-epoch development reference"))
    config = ReferenceConfig(dataset=dataset, split="dev", protocol=frozen["protocol"],
                             epoch=frozen["epoch"], cases=cases,
                             reference_label_map=LABEL_MAPS[dataset], allow_unmapped_volumes=True,
                             student_arms=(METHODS[1], METHODS[0]),
                             provenance="native colocated references; isolated post-epoch-freeze monitoring")
    return config, {"mode": "native_auto", "skipped_volumes": skipped, "index_transfers": transfers,
                    "metadata_files": metadata_files, "reference_sha256": mask_hashes,
                    "source_sha256": source_hashes}


def evaluate_epoch(freeze_path: str | Path, image_manifest_path: str | Path,
                   output_dir: str | Path, reference_config_path: str | Path | None = None) -> dict[str, Any]:
    freeze_path, image_manifest_path, output = Path(freeze_path), Path(image_manifest_path), Path(output_dir)
    frozen = json.loads(freeze_path.read_text())
    manifest, volumes = _bind_images(frozen, image_manifest_path)
    # The annotation boundary starts here, after physical freeze validation.
    if reference_config_path is None:
        config, matching = _auto_config(frozen, manifest, volumes, output)
    else:
        config = ReferenceConfig.load(reference_config_path)
        if config.dataset != frozen["dataset"] or config.split != "dev" or config.protocol != frozen["protocol"]:
            raise ReferenceConfigError("epoch reference config must match dataset/protocol and split=dev")
        if config.epoch is not None and config.epoch != frozen["epoch"]:
            raise ReferenceConfigError("explicit reference config epoch differs from the frozen snapshot")
        allowed = {(row["patient_id"], row["frozen_volume_id"]) for row in volumes}
        if any((case.patient_id, case.resolved_volume_key()) not in allowed for case in config.cases):
            raise ReferenceConfigError("reference cases must name exact frozen development volume IDs")
        config = replace(config, epoch=frozen["epoch"], student_arms=(METHODS[1], METHODS[0]))
        matching = {"mode": "explicit", "config_path": str(reference_config_path),
                    "config_sha256": sha256_file(reference_config_path), "skipped_volumes": []}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_reference_config.json", config.to_dict())
    write_json(output / "reference_matching.json", matching)
    report = evaluate_reference(frozen, config, output_dir=output)
    rows = {row["name"]: row for row in report["rows"]}
    students = {}
    for arm, method in zip(ARMS, METHODS):
        overlap = report.get("overlap", {}).get(method, {})
        dice = overlap.get("dice", {})
        def metric(name):
            entry = rows.get(f"{method}.{name}", {})
            return entry.get("value") if entry.get("available") else None
        students[arm] = {
            "dice": dice.get("macro_mean"), "iou": overlap.get("iou", {}).get("macro_mean"),
            "dice_per_class": {name: metric(f"dice.{name}") for name in ("RV", "MYO", "LV")},
            "hd95": metric("hd95"), "assd": metric("assd"),
            "patients": dice.get("patients_counted", 0), "volumes": overlap.get("volumes_counted", 0),
            "volumes_without_reference": len(overlap.get("patients_without_reference", [])),
        }
    available = all(student["dice"] is not None for student in students.values())
    reason = None if available else report.get("reason") or "no defined foreground reference Dice"
    if not available and matching.get("skipped_volumes"):
        reason = matching["skipped_volumes"][0]["reason"]
    summary = {"schema_version": SCHEMA, "epoch": frozen["epoch"], "freeze_id": frozen["freeze_id"],
               "status": "COMPLETED" if available else "UNAVAILABLE", "available": available,
               "reason": reason, "dataset": frozen["dataset"], "split": "dev", "students": students,
               "matching": matching, "reference_label_map": config.reference_label_map,
               "reference_metrics": str(output / "reference_metrics.json"),
               "checkpoint_selection": "none", "mask_access_after_complete_freeze": True}
    write_json(output / "epoch_validation.json", summary)
    return summary
