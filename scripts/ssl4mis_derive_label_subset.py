#!/usr/bin/env python3
"""Derive a smaller-label materialization (e.g. 5%) from an existing SSL4MIS-schema preparation without copying arrays.

The new labeled set must be a subset of the existing one; the removed labeled patients join U as IMAGE-ONLY records (their
masks are not referenced by the new train manifest).  val/test manifests are reused unchanged (same paths and hashes), so the
evaluation sets are identical to the source preparation.  Writes a new split lock, train manifest and receipt into ``--out``.
Used for ACDC 5% (L3 = patient038/050/099, the first 68 slices of the pinned SSL4MIS train_slices.list) and LA 5% (L4 = the
first four cases of the upstream train.list).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pg_prepare_acdc import sha256_file  # noqa: E402


def derive(prepared: Path, labeled: list[str], out: Path, note: str) -> dict:
    if out.exists():
        raise SystemExit(f"output must be a new directory: {out}")
    receipt = json.loads((prepared / "PREPARATION_RECEIPT.json").read_text())
    lock = json.loads(Path(receipt["lock"]["path"]).read_text())
    train = json.loads(Path(receipt["manifests"]["train"]["path"]).read_text())
    old_l = {r["id"]: r for r in train["labeled"]}
    labeled = sorted(set(labeled))
    if not labeled or not set(labeled) < set(old_l):
        raise SystemExit("the new labeled set must be a non-empty strict subset of the existing labeled set")
    moved = sorted(set(old_l) - set(labeled))
    u = sorted(train["patients"] + moved)
    out.mkdir(parents=True)
    new_lock = dict(lock)
    new_lock.update({"labeled": {"patients": labeled}, "unlabeled": {"patients": u},
                     "derived_from": {"lock": receipt["lock"], "note": note}})
    lock_path = out / "SPLIT_LOCK.json"
    lock_path.write_text(json.dumps(new_lock, indent=2, sort_keys=True) + "\n")
    expected = dict(train.get("expected_counts") or {"labeled": len(old_l), "unlabeled": len(train["patients"]), "val": 10, "test": 20})
    expected.update({"labeled": len(labeled), "unlabeled": len(u)})
    new_train = dict(train)
    new_train.update({"labeled": [old_l[p] for p in labeled],
                      "unlabeled": sorted(train["unlabeled"] + [{"id": p, "image": old_l[p]["image"], "image_sha256": old_l[p]["image_sha256"]}
                                                                for p in moved], key=lambda r: r["id"]),
                      "patients": u, "labeled_patients": labeled, "expected_counts": expected,
                      "lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)}, "derived_from": note})
    train_path = out / "train.json"
    train_path.write_text(json.dumps(new_train, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifests = dict(receipt["manifests"])
    manifests["train"] = {"path": str(train_path), "sha256": sha256_file(train_path)}
    split_ids = dict(receipt["split_ids"])
    split_ids.update({"train_l": labeled, "train_u": u})
    new_receipt = dict(receipt)
    new_receipt.update({"lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)}, "manifests": manifests,
                        "split_ids": split_ids, "counts": {k: len(v) for k, v in split_ids.items()},
                        "derived_from": {"receipt": str(prepared / "PREPARATION_RECEIPT.json"),
                                         "sha256": sha256_file(prepared / "PREPARATION_RECEIPT.json"), "moved_to_u_image_only": moved, "note": note}})
    (out / "PREPARATION_RECEIPT.json").write_text(json.dumps(new_receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return new_receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--labeled", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", required=True)
    a = ap.parse_args()
    r = derive(a.prepared.resolve(), a.labeled, a.out.resolve(), a.note)
    print(json.dumps({"counts": r["counts"], "manifests": r["manifests"]}, indent=1))


if __name__ == "__main__":
    main()
