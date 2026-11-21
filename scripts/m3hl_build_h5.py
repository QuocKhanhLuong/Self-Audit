#!/usr/bin/env python3
"""Build the SSL4MIS-format ACDC h5 tree that the official M3HL / SSL4MIS code reads, from our raw ACDC.

Layout written (identical to the SSL4MIS data release that ``external/M3HL`` expects)::

    <out>/train_slices.list  train.list  val.list  test.list      (copied from configs/candidate_pg/ssl4mis_lists)
    <out>/data/<case>.h5                  image [D,H,W] float32, label [D,H,W] uint8   (train, val and test volumes)
    <out>/data/slices/<case>_slice_<i>.h5 image [H,W]   float32, label [H,W]   uint8   (train volumes only; i is 1-based,
                                          as in the pinned train_slices.list, i.e. slice i = volume[i-1])

Preprocessing is SSL4MIS's own (``code/dataloaders/acdc_data_processing.py`` at 06df6047): the ED/ES frame is read
with SimpleITK (array order z,y,x) and min-max normalised **per 3D volume** to [0,1].  A case name ``patientNNN_frame01``
is the ED frame and ``frame02`` the ES frame (the repo's ED/ES convention, resolved through each patient's Info.cfg).

Differences from the SSL4MIS release, both deliberate:
  * label arrays of the 63 unlabeled TRAIN patients are written as zeros.  The official M3HL loop never uses unlabeled
    ground truth (``unl_label``/``l_label`` are only logged); keeping U63 masks out of the tree makes that structural.
  * the case list order comes from the pinned lock lists, not from a directory glob.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABELED_SLICES_7 = 136  # patients_to_slices("ACDC", 7) in SSL4MIS and M3HL


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def split_case(case: str) -> tuple[str, str]:
    match = re.fullmatch(r"(patient\d{3})_frame0([12])", case)
    if not match:
        raise ValueError(f"unexpected case name {case!r}")
    return match.group(1), {"1": "ED", "2": "ES"}[match.group(2)]


def check_lists(lists_dir: Path, lock: dict) -> dict:
    """Verify the lists against the lock; return per-role case lists."""
    train, val, test = (read_list(lists_dir / f"{n}.list") for n in ("train", "val", "test"))
    slices = read_list(lists_dir / "train_slices.list")
    labeled = lock["labeled"]["patients"]
    pat = lambda cases: sorted({split_case(c)[0] for c in cases})
    problems = []
    if pat(train) != sorted(lock["labeled"]["patients"] + lock["unlabeled"]["patients"]):
        problems.append("train.list patients differ from lock L7 + U63")
    if pat(val) != sorted(lock["val"]["patients"]):
        problems.append("val.list patients differ from the lock")
    if pat(test) != sorted(lock["test"]["patients"]):
        problems.append("test.list patients differ from the lock")
    slice_cases = {re.sub(r"_slice_\d+$", "", s) for s in slices}
    if slice_cases != set(train):
        problems.append("train_slices.list cases differ from train.list")
    if len(set(slices)) != len(slices):
        problems.append("train_slices.list has duplicates")
    head = slices[:LABELED_SLICES_7]
    head_patients = sorted({split_case(re.sub(r"_slice_\d+$", "", s))[0] for s in head})
    if head_patients != sorted(labeled):
        problems.append(f"first {LABELED_SLICES_7} slices belong to {head_patients}, not to L7 {sorted(labeled)}")
    labeled_total = sum(1 for s in slices if split_case(re.sub(r"_slice_\d+$", "", s))[0] in set(labeled))
    if labeled_total != LABELED_SLICES_7:
        problems.append(f"L7 has {labeled_total} slices in train_slices.list, expected {LABELED_SLICES_7}")
    return {"train": train, "val": val, "test": test, "slices": slices, "labeled": labeled,
            "problems": problems, "labeled_slice_count": labeled_total}


def default_loader(raw_root: Path, case: str, want_label: bool):
    import SimpleITK as sitk

    pid, phase = split_case(case)
    folder = raw_root / pid
    phases = _read_phases(folder / "Info.cfg")
    frame = phases[phase]
    image_path = folder / f"{pid}_frame{frame:02d}.nii"
    if not image_path.exists():
        image_path = folder / f"{pid}_frame{frame:02d}.nii.gz"
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
    label = None
    sources = [image_path]
    if want_label:
        label_path = image_path.with_name(image_path.name.replace(".nii", "_gt.nii", 1))
        label = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))).astype(np.uint8)
        sources.append(label_path)
    return image, label, frame, sources


def _read_phases(info_cfg: Path) -> dict:
    out = {}
    for line in info_cfg.read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() in ("ED", "ES"):
            out[key.strip()] = int(value.strip())
    if set(out) != {"ED", "ES"}:
        raise ValueError(f"cannot read ED/ES from {info_cfg}")
    return out


def minmax(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    low, high = image.min(), image.max()
    return ((image - low) / (high - low)).astype(np.float32)


def build(lists_dir: Path, lock: dict, raw_root: Path, out: Path, loader=default_loader) -> dict:
    import h5py

    info = check_lists(lists_dir, lock)
    if info["problems"]:
        raise SystemExit("list check failed: " + "; ".join(info["problems"]))
    if out.exists():
        raise SystemExit(f"output must be a new directory: {out}")
    (out / "data" / "slices").mkdir(parents=True)
    for name in ("train_slices", "train", "val", "test"):
        shutil.copyfile(lists_dir / f"{name}.list", out / f"{name}.list")
    labeled = set(lock["labeled"]["patients"])
    train_cases = set(info["train"])
    inventory, slice_count = {}, 0
    for role in ("train", "val", "test"):
        for case in info[role]:
            pid, _ = split_case(case)
            want_label = role != "train" or pid in labeled
            image, label, frame, sources = loader(raw_root, case, want_label)
            image = minmax(image)
            if label is None:
                label = np.zeros(image.shape, dtype=np.uint8)  # unlabeled TRAIN patient: no ground truth in the tree
            if image.shape != label.shape:
                raise SystemExit(f"image/label shape mismatch for {case}")
            if want_label and not np.isin(label, (0, 1, 2, 3)).all():
                raise SystemExit(f"labels outside 0..3 for {case}")
            with h5py.File(out / "data" / f"{case}.h5", "w") as f:
                f.create_dataset("image", data=image, compression="gzip")
                f.create_dataset("label", data=label, compression="gzip")
            if case in train_cases:
                for index in range(image.shape[0]):
                    with h5py.File(out / "data" / "slices" / f"{case}_slice_{index + 1}.h5", "w") as f:
                        f.create_dataset("image", data=image[index], compression="gzip")
                        f.create_dataset("label", data=label[index], compression="gzip")
                    slice_count += 1
            inventory[case] = {"role": role, "frame": frame, "shape": list(image.shape), "labels_read": want_label,
                               "sources": {Path(s).name: sha256_file(Path(s)) for s in sources if Path(s).exists()}}
    expected = {s for s in info["slices"]}
    written = {p.stem for p in (out / "data" / "slices").glob("*.h5")}
    receipt = {"schema": "self_audit.m3hl_h5.v1", "status": "READY" if written == expected else "FAILED",
               "slices_written": slice_count, "slices_listed": len(info["slices"]),
               "slice_files_equal_list": written == expected,
               "labeled_slice_count": info["labeled_slice_count"], "labeled_head": LABELED_SLICES_7,
               "lists_sha256": {n: sha256_file(lists_dir / f"{n}.list") for n in ("train_slices", "train", "val", "test")},
               "cases": inventory, "preprocessing": "per-volume min-max to [0,1] (SSL4MIS acdc_data_processing.py @06df6047)",
               "u63_labels": "zeros (never read by the official M3HL loss)"}
    (out / "H5_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock", type=Path, default=ROOT / "configs/candidate_pg/ssl4mis_split_lock.json")
    ap.add_argument("--lists", type=Path, default=ROOT / "configs/candidate_pg/ssl4mis_lists")
    ap.add_argument("--raw", type=Path, required=True, help="folder holding patientNNN/ (the raw ACDC training tree)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    receipt = build(args.lists, json.loads(args.lock.read_text()), args.raw, args.out)
    print(json.dumps({k: receipt[k] for k in ("status", "slices_written", "slices_listed", "slice_files_equal_list", "labeled_slice_count")}, indent=1))
    return 0 if receipt["status"] == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
