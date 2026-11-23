#!/usr/bin/env python3
"""Materialize LA 2018 (Utah) for the PG engine in the SSL4MIS manifest schema: L8 / U62 / val10 / test20, one volume per case.

* Source: the UA-MT/M3HL h5 conversion (``prepared/LA2018``: x/y crop around the label box with seeded margins, z not cropped,
  whole-volume z-score), so every case is exactly the volume the M3HL LA teacher saw.  Split = ``m3hl/data/LA_run/LA_SPLIT.json``
  (labeled8 / unlabeled62 / val10 / test20), the split of the M3HL LA run.  U masks are never read.
* Engine conventions: arrays ``[1, D, H, W]`` in (z, y, x) (h5 is (x, y, z)); intensities mapped to [0, 1] with per-volume
  percentiles 0.5/99.5 (monotone in the z-score); manifests declare ``stages = [0]`` (single frame).
* Labels: the PG student and evaluators are four-class (BG, RV, MYO, LV).  LA is stored in class slot 3 ("LV" slot); classes 1 and
  2 never occur.  Report the slot-3 metrics as the LA metrics.
* Spacing: the receipt records [1, 1, 1] from the NRRD headers; LA 2018 volumes are 0.625 mm isotropic per the challenge, which is
  declared here (recorded) so that HD95/ASD can be reported in mm.  Voxel-unit metrics (the usual LA protocol) are reported too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pg_prepare_acdc import sha256_file  # noqa: E402
from self_audit.candidate_fuse.data import _CLASSES  # noqa: E402

SCHEMA = "self_audit.ssl4mis.data.v1"
LOCK_SCHEMA = "self_audit.ssl4mis_split_lock.v1"
EXPECTED = {"labeled": 8, "unlabeled": 62, "val": 10, "test": 20}
LA_SLOT = 3
SPACING_MM = 0.625
SPACING_PROVENANCE = ("LA 2018 (Utah) NRRD headers give unit spacing; the challenge data are 0.625 mm isotropic, declared mm here "
                      "for every case so that HD95/ASD are in mm.  Voxel-unit metrics are reported next to them.")


def prepare(receipt_path: Path, split_path: Path, out: Path) -> dict:
    import h5py

    if out.exists():
        raise SystemExit(f"output must be a new directory: {out}")
    rec = json.loads(receipt_path.read_text())
    if rec.get("status") != "READY":
        raise SystemExit("LA 2018 receipt is not READY")
    sp = json.loads(split_path.read_text())
    test_ids = list(sp["official_test20"])  # upstream UA-MT test.list; LA_run/test.list holds val10 (M3HL validation)
    split = {"train_l": sorted(sp["labeled8"]), "train_u": sorted(sp["unlabeled62"]), "val": sorted(sp["val10"]),
             "test": sorted(test_ids)}
    for key, name in (("train_l", "labeled"), ("train_u", "unlabeled"), ("val", "val"), ("test", "test")):
        if len(split[key]) != EXPECTED[name]:
            raise SystemExit(f"{key} has {len(split[key])} cases, expected {EXPECTED[name]}")
    ids = sum(split.values(), [])
    if len(set(ids)) != len(ids) or not set(ids) <= set(rec["cases"]):
        raise SystemExit("split must be disjoint and inside the receipt's converted cases")
    out.mkdir(parents=True)
    arrays = out / "arrays"
    arrays.mkdir()
    lock = {"schema": LOCK_SCHEMA, "dataset": "la2018", "source": {"split": str(split_path), "split_sha256": sha256_file(split_path)},
            "labeled": {"patients": split["train_l"]}, "unlabeled": {"patients": split["train_u"]},
            "val": {"patients": split["val"]}, "test": {"patients": split["test"]}}
    lock_path = out / "SPLIT_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    needs_masks = set(split["train_l"]) | set(split["val"]) | set(split["test"])
    records, geometry = {}, {}
    for cid in sorted(ids):
        meta = rec["cases"][cid]
        h5_path = Path(meta["h5"])
        if sha256_file(h5_path) != meta["h5_sha256"]:
            raise SystemExit(f"h5 differs from the receipt: {cid}")
        with h5py.File(h5_path, "r") as f:
            img = np.asarray(f["image"][:], dtype=np.float32).transpose(2, 1, 0)
            lab = np.asarray(f["label"][:]).transpose(2, 1, 0) if cid in needs_masks else None
        lo, hi = np.percentile(img, [0.5, 99.5])
        img01 = np.clip((img - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)
        image = arrays / f"{cid}_vol.npy"
        np.save(image, np.ascontiguousarray(img01[None]), allow_pickle=False)
        r = {"id": cid, "image": str(image), "image_sha256": sha256_file(image)}
        if lab is not None:
            if not np.isin(lab, (0, 1)).all() or not lab.any():
                raise SystemExit(f"{cid}: LA label must be binary and non-empty")
            dst = arrays / f"{cid}_t0_mask.npy"
            np.save(dst, np.ascontiguousarray((lab > 0).astype(np.uint8) * LA_SLOT), allow_pickle=False)
            r.update({"phases": {"LGE": 0}, "masks": {"0": {"path": str(dst), "sha256": sha256_file(dst), "source_path": str(h5_path),
                                                              "source_sha256": meta["h5_sha256"], "transpose": [2, 1, 0], "la_class_slot": LA_SLOT}}})
        records[cid] = r
        shape = list(img.shape)
        geometry[cid] = {"shape": shape, "spacing_zyx": [SPACING_MM] * 3,
                         "affine_zyx": [[SPACING_MM, 0.0, 0.0, 0.0], [0.0, SPACING_MM, 0.0, 0.0], [0.0, 0.0, SPACING_MM, 0.0], [0.0, 0.0, 0.0, 1.0]],
                         "context": "full_volume", "provenance": {"spacing": SPACING_PROVENANCE, "crop_xy": meta.get("crop_xy")}}
        print(f"[prepare-la2018] {cid} shape {shape}", flush=True)

    def base(role: str, case_ids: list[str]) -> dict:
        return {"schema": SCHEMA, "role": role, "dataset": "la2018", "synthetic": False, "class_mapping": _CLASSES, "stages": [0],
                "expected_counts": EXPECTED, "la_class_slot": LA_SLOT, "lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
                "normalization": "UA-MT z-score volume mapped to [0,1] with per-volume percentiles 0.5/99.5",
                "geometry": {"by_patient": {c: geometry[c] for c in case_ids}}}

    manifests = {"train": base("train", split["train_l"] + split["train_u"])}
    manifests["train"].update({"labeled": [records[c] for c in split["train_l"]],
                               "unlabeled": [{"id": c, "image": records[c]["image"], "image_sha256": records[c]["image_sha256"]} for c in split["train_u"]],
                               "patients": split["train_u"], "labeled_patients": split["train_l"]})
    for role in ("val", "test"):
        manifests[role] = base(role, split[role])
        manifests[role]["patients"] = [records[c] for c in split[role]]
    refs = {}
    for role, manifest in manifests.items():
        path = out / f"{role}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        refs[role] = {"path": str(path), "sha256": sha256_file(path)}
    receipt = {"schema": "self_audit.ssl4mis.materialization.v1", "status": "READY", "dataset": "la2018",
               "lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)}, "manifests": refs,
               "counts": {k: len(v) for k, v in split.items()}, "split_ids": split,
               "source_receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
               "mask_policy": "masks read only for L8, val10 and test20; U62 masks never opened", "la_class_slot": LA_SLOT,
               "spacing_declaration": SPACING_PROVENANCE}
    (out / "PREPARATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True, help="prepared/LA2018/PREPARATION_RECEIPT.json")
    ap.add_argument("--split", type=Path, required=True, help="m3hl/data/LA_run/LA_SPLIT.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = prepare(a.receipt.resolve(), a.split.resolve(), a.out.resolve())
    print(json.dumps({"status": r["status"], "counts": r["counts"], "manifests": r["manifests"]}, indent=1))


if __name__ == "__main__":
    main()
