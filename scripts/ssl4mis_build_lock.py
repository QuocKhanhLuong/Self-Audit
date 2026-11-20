#!/usr/bin/env python3
"""Build configs/candidate_pg/ssl4mis_split_lock.json from the pinned SSL4MIS ACDC list files.

Rules reproduced from HiLab-git/SSL4MIS (code/train_mean_teacher_2D.py): ``patients_to_slices("ACDC", 7) == 136`` and the
labeled set is ``train_slices.list[:136]``; unlabeled data is the rest.  Here the labeled set must cover whole volumes,
which this script verifies.  The lock is pure list arithmetic (no raw data), so it can be built and tested anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

SSL4MIS_REPO = "HiLab-git/SSL4MIS"
LABELED_PATIENTS = 7
LABELED_SLICES = {"3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}  # patients_to_slices("ACDC", n)
LIST_FILES = ("train.list", "train_slices.list", "val.list", "test.list")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def patient_of(name: str) -> str:
    match = re.match(r"(patient\d{3})_frame(\d{2})", name)
    if not match:
        raise ValueError(f"unexpected volume name {name!r}")
    return match.group(1)


def volume_of(slice_name: str) -> str:
    return slice_name.rsplit("_slice_", 1)[0]


def build(lists_dir: Path, commit: str, labeled_patients: int = LABELED_PATIENTS) -> dict:
    train, slices, val, test = (read_list(lists_dir / name) for name in LIST_FILES)
    n_slices = LABELED_SLICES[str(labeled_patients)]
    head = slices[:n_slices]
    head_volumes = sorted({volume_of(s) for s in head})
    labeled = sorted({patient_of(v) for v in head_volumes})
    # whole-volume check: every slice of every head volume is inside the head, both frames present per patient
    all_slices_of = {v: [s for s in slices if volume_of(s) == v] for v in head_volumes}
    whole = all(len([s for s in head if volume_of(s) == v]) == len(all_slices_of[v]) for v in head_volumes)
    frames_ok = all({f"{p}_frame01", f"{p}_frame02"} <= set(head_volumes) for p in labeled)
    sets = {"train": sorted({patient_of(v) for v in train}), "val": sorted({patient_of(v) for v in val}),
            "test": sorted({patient_of(v) for v in test})}
    unlabeled = [p for p in sets["train"] if p not in labeled]
    disjoint = not (set(sets["train"]) & set(sets["val"]) or set(sets["train"]) & set(sets["test"]) or set(sets["val"]) & set(sets["test"]))
    cover = sorted(set(sets["train"]) | set(sets["val"]) | set(sets["test"])) == [f"patient{i:03d}" for i in range(1, 101)]
    checks = {"patients_disjoint": disjoint, "cover_patients_001_to_100": cover, "labeled_head_has_whole_volumes": whole,
              "labeled_patients_have_both_frames": frames_ok, "labeled_patient_count": len(labeled) == labeled_patients,
              "volumes_per_set": {"train": len(train), "val": len(val), "test": len(test)},
              "frame_names_only_01_02": {re.search(r"_frame(\d{2})", v).group(1) for v in train + val + test} == {"01", "02"}}
    checks["frame_names_only_01_02"] = bool(checks["frame_names_only_01_02"])
    if not all(v for k, v in checks.items() if k != "volumes_per_set") or checks["volumes_per_set"] != {"train": 140, "val": 20, "test": 40}:
        raise SystemExit(f"split checks failed: {checks}")
    return {
        "schema": "self_audit.ssl4mis_split_lock.v1",
        "source": {"repo": SSL4MIS_REPO, "commit": commit, "directory": "data/ACDC",
                   "list_sha256": {name: sha(lists_dir / name) for name in LIST_FILES}},
        "protocol": {"labeled_patients": labeled_patients, "labeled_rule": f"train_slices.list[:{n_slices}] = patients_to_slices('ACDC', {labeled_patients})",
                     "unlabeled": "ED/ES frames only of the remaining train patients",
                     "frame_name_rule": "frame01 = ED, frame02 = ES (positional; ED precedes ES for all 100 ACDC training patients)"},
        "labeled": {"patients": labeled, "volumes": head_volumes},
        "unlabeled": {"patients": unlabeled, "volumes": [f"{p}_frame0{i}" for p in unlabeled for i in (1, 2)]},
        "val": {"patients": sets["val"], "volumes": [v for v in sorted(val)]},
        "test": {"patients": sets["test"], "volumes": [v for v in sorted(test)]},
        "checks": checks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lists", type=Path, required=True, help="directory holding the four pinned .list files")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copy-lists-to", type=Path, help="copy the .list files into the repo for reproducibility")
    args = ap.parse_args()
    lock = build(args.lists, args.commit)
    if args.copy_lists_to:
        args.copy_lists_to.mkdir(parents=True, exist_ok=True)
        for name in LIST_FILES:
            shutil.copyfile(args.lists / name, args.copy_lists_to / name)
    args.out.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: L{len(lock['labeled']['patients'])} U{len(lock['unlabeled']['patients'])} "
          f"val{len(lock['val']['patients'])} test{len(lock['test']['patients'])}")


if __name__ == "__main__":
    main()
