#!/usr/bin/env python3
"""M&Ms-1 in the SSL4MIS ACDC h5 layout (few-label, ED/ES), so the official M3HL / SSL4MIS code runs on it unmodified.

Roles: labeled = ``--n-labeled`` patients (default 15 = 10% of the 150 ``Training/Labeled`` patients, the SSL4MIS few-label setting) drawn with seed 42; the first
``labeled_slices`` entries of ``train_slices.list`` are exactly their ED+ES slices (SSL4MIS takes ``train_slices.list[:patients_to_slices(n)]`` as labeled; the real
slice total is recorded in the receipt and handed to the runner); U = the other ``Training/Labeled`` patients as IMAGES ONLY (label arrays are written as zeros); val = official ``Validation`` (34); test = official ``Testing`` (136).
``Training/Unlabeled`` (25) is not used: its ED/ES frames are not published (CSV 0,0) and finding them would require reading its masks.
Frames: ED = ``frame01``, ES = ``frame02`` from the official CSV.  Preprocessing = SSL4MIS: per-3D-volume min-max to [0,1]; array order (z, y, x) (the same
transposition as the ACDC preparation); classes remapped M&Ms -> ACDC order (M&Ms 1 LV, 3 RV  ->  ACDC 3 LV, 1 RV).  Slice files are 1-based like the pinned ACDC lists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

MNMS_TO_ACDC = np.array([0, 3, 2, 1], dtype=np.uint8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def pick_labeled(depths: dict[str, int], n: int, seed: int = 42) -> list[str]:
    ids = sorted(depths)
    rng = np.random.default_rng(seed)
    return sorted(ids[i] for i in rng.choice(len(ids), n, replace=False))


def minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return ((x - x.min()) / (x.max() - x.min())).astype(np.float32)


def main() -> int:
    import h5py
    import nibabel as nib

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True, help="M&Ms-1 PREPARATION_RECEIPT.json (roles, ED/ES, hashes)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-labeled", type=int, default=15)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"output must be a new directory: {a.out}")
    rec = json.loads(a.receipt.read_text())
    if rec["status"] != "READY":
        raise SystemExit("M&Ms-1 receipt is not READY")
    subjects = rec["subjects"]
    roles = {r: sorted(s for s, v in subjects.items() if v["role"] == r) for r in ("train_labeled", "val", "test")}
    depths = {s: subjects[s]["shape"][2] for s in roles["train_labeled"]}
    l7 = pick_labeled(depths, a.n_labeled)
    train_cases = [f"{s}_frame0{k}" for s in roles["train_labeled"] for k in (1, 2)]
    cases_of = lambda ids: [f"{s}_frame0{k}" for s in ids for k in (1, 2)]
    (a.out / "data" / "slices").mkdir(parents=True)
    inventory, slice_names = {}, {}
    for role in ("train_labeled", "val", "test"):
        for sid in roles[role]:
            meta = subjects[sid]
            img4 = np.asarray(nib.load(meta["image"]).dataobj)
            lab4 = np.asarray(nib.load(meta["label"]).dataobj)
            for k, frame in ((1, meta["csv"]["ed"]), (2, meta["csv"]["es"])):
                case = f"{sid}_frame0{k}"
                image = minmax(img4[..., frame].transpose(2, 1, 0))
                keep_label = role != "train_labeled" or sid in l7
                label = MNMS_TO_ACDC[lab4[..., frame].transpose(2, 1, 0).astype(np.int64)] if keep_label else np.zeros(image.shape, np.uint8)
                if keep_label and not lab4[..., frame].any():
                    raise SystemExit(f"{case}: empty mask at the CSV frame {frame}")
                with h5py.File(a.out / "data" / f"{case}.h5", "w") as f:
                    f.create_dataset("image", data=image, compression="gzip")
                    f.create_dataset("label", data=label, compression="gzip")
                if role == "train_labeled":
                    names = [f"{case}_slice_{i + 1}" for i in range(image.shape[0])]
                    slice_names[case] = names
                    for i, name in enumerate(names):
                        with h5py.File(a.out / "data" / "slices" / f"{name}.h5", "w") as f:
                            f.create_dataset("image", data=image[i], compression="gzip")
                            f.create_dataset("label", data=label[i], compression="gzip")
                inventory[case] = {"role": role, "labeled": bool(keep_label), "frame": int(frame), "shape": list(image.shape), "vendor": meta["csv"]["vendor"],
                                   "image_sha256": meta["image_sha256"], "label_sha256": meta["label_sha256"] if keep_label else None}
    rng = np.random.default_rng(42)
    head = [n for c in cases_of(l7) for n in slice_names[c]]
    rng.shuffle(head)
    rest = [n for c in train_cases if c.rsplit("_frame", 1)[0] not in l7 for n in slice_names[c]]
    rng.shuffle(rest)
    slices = head + rest
    assert len(set(slices)) == len(slices)
    for name, items in (("train", train_cases), ("train_slices", slices), ("val", cases_of(roles["val"])), ("test", cases_of(roles["test"]))):
        (a.out / f"{name}.list").write_text("\n".join(items) + "\n")
    receipt = {"schema": "self_audit.mnms1.h5.v1", "status": "READY", "labeled_patients": l7, "n_labeled": len(l7), "l7": l7, "l7_vendors": [subjects[s]["csv"]["vendor"] for s in l7], "labeled_slices": len(head),
               "slices": len(slices), "counts": {"train_patients": len(roles["train_labeled"]), "val_patients": len(roles["val"]), "test_patients": len(roles["test"])},
               "excluded": "Training/Unlabeled (25): ED/ES not published, masks must not be read", "class_mapping": "M&Ms [0,1,2,3] -> ACDC order [0,3,2,1]",
               "lists_sha256": {n: sha256_file(a.out / f"{n}.list") for n in ("train", "train_slices", "val", "test")}, "source_receipt": str(a.receipt),
               "source_receipt_sha256": sha256_file(a.receipt), "cases": inventory}
    (a.out / "H5_RECEIPT.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print("READY", receipt["counts"], "L7", l7, "slices", len(slices))
    return 0


if __name__ == "__main__":
    sys.exit(main())
