#!/usr/bin/env python
"""Build the checked-in Self-Audit ACDC ED/ES selection receipt.

This is a source-side, metadata-only step.  It reads the patient split and
``Info.cfg`` from the original ACDC training tree, but it never opens an image
or a ``*_gt`` file.  The resulting receipt is the only frame-selection input
mounted into the image-only manifest builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTION_SCHEMA = "self_audit.acdc.frame_selection.v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _patient_lists(split_path: Path) -> tuple[list[str], list[str], int]:
    payload = _read_json(split_path)
    if payload.get("dataset") != "ACDC" or payload.get("seed") != 42:
        raise ValueError("Self-Audit selection requires the checked-in ACDC seed-42 split")
    train = [str(value) for value in payload.get("train_patients", [])]
    dev = [str(value) for value in payload.get("val_patients", [])]
    if not train or not dev or set(train) & set(dev):
        raise ValueError("split manifest must contain disjoint train_patients and val_patients")
    expected = int(payload.get("n_patients", payload.get("num_patients", 0)))
    if expected != len(set(train) | set(dev)):
        raise ValueError("split manifest patient count does not match train/val membership")
    if expected != 100:
        raise ValueError(f"Self-Audit ACDC protocol expects 100 training patients, got {expected}")
    return train, dev, int(payload["seed"])


def _info_cfg(path: Path) -> dict[str, int]:
    # ACDC Info.cfg is a small ``KEY: value`` file, not a full ConfigParser
    # document.  Parsing only ED/ES avoids any dependency on patient labels.
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*(ED|ES)\s*:\s*(\d+)\s*$", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    if set(values) != {"ED", "ES"} or values["ED"] <= 0 or values["ES"] <= 0:
        raise ValueError(f"Info.cfg must declare positive ED and ES frames: {path}")
    return values


def _image_name(patient: str, frame: int, patient_dir: Path) -> str:
    for suffix in (".nii", ".nii.gz"):
        candidate = patient_dir / f"{patient}_frame{frame:02d}{suffix}"
        if candidate.is_file():
            return candidate.name
    raise FileNotFoundError(f"selected ED/ES image is missing for {patient}, frame {frame:02d}")


def build(raw_root: Path, split_path: Path, output: Path) -> dict[str, Any]:
    train, dev, seed = _patient_lists(split_path)
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split, patients in (("train", train), ("dev", dev)):
        for patient in patients:
            if not re.fullmatch(r"patient\d{3}", patient):
                raise ValueError(f"unexpected ACDC patient id: {patient!r}")
            patient_dir = raw_root / patient
            info = patient_dir / "Info.cfg"
            if not info.is_file():
                raise FileNotFoundError(info)
            frames = _info_cfg(info)
            for label in ("ED", "ES"):
                frame = frames[label]
                name = _image_name(patient, frame, patient_dir)
                relative = (Path("training") / patient / name).as_posix()
                # The receipt intentionally contains no mask path, mask hash,
                # label, or annotation-derived field.
                key = (patient, relative)
                if key in seen:
                    raise ValueError(f"duplicate selected frame: {key}")
                seen.add(key)
                rows.append({
                    "patient_id": patient,
                    "split": split,
                    "relative_path": relative,
                    "frame_index": frame,
                    "frame_label": label,
                })
    rows.sort(key=lambda row: (row["split"], row["patient_id"], row["relative_path"]))
    try:
        split_relative = str(split_path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        split_relative = str(split_path)
    payload: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "dataset": "acdc",
        "seed": seed,
        "source_cohort": "ACDC/training",
        "split_manifest": split_relative,
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "patient_split": {"train": len(train), "dev": len(dev), "test": 0, "independent_test": False},
        "frame_selection_rule": "Info.cfg ED and ES; exactly two rank-3 image frames per patient",
        "records": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="ACDC/training source directory")
    parser.add_argument("--split-manifest", type=Path, default=REPO_ROOT / "splits" / "acdc_patient_split_seed42.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.raw_root, args.split_manifest, args.output)
    counts = {split: sum(row["split"] == split for row in payload["records"]) for split in ("train", "dev")}
    print(json.dumps({"output": str(args.output), "patients": payload["patient_split"], "frames": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
