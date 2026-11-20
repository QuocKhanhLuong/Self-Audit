#!/usr/bin/env python3
"""Import a completed S1 (student_init.pt + receipt) trained elsewhere (RTX 4070 server) into this run directory.

The artifact bytes are verified against the receipt's checkpoint_sha256 (and an optional expected hash). The receipt is
re-pointed to the local artifact path; the original is kept as S1_RECEIPT.imported_original.json and the provenance is
recorded in the receipt (``imported_from``) and in the DONE marker.  Only runs when S1.DONE.json is absent or stale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="folder holding student_init.pt and S1_RECEIPT.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--expect-sha256", default=None)
    args = ap.parse_args(argv)
    config_sha = sha(args.config)
    marker = args.out / "markers" / "S1.DONE.json"
    if marker.is_file() and json.loads(marker.read_text()).get("config_sha256") == config_sha:
        print("[import-s1] S1 DONE marker present, nothing to do")
        return 0
    receipt = json.loads((args.src / "S1_RECEIPT.json").read_text())
    ckpt = args.src / "student_init.pt"
    digest = sha(ckpt)
    if digest != receipt.get("checkpoint_sha256"):
        raise SystemExit(f"student_init.pt sha {digest} != receipt checkpoint_sha256")
    if args.expect_sha256 and digest != args.expect_sha256:
        raise SystemExit("student_init.pt sha differs from the expected hash")
    if receipt.get("status") != "COMPLETED":
        raise SystemExit(f"S1 receipt status {receipt.get('status')!r} is not COMPLETED")
    (args.out / "S1").mkdir(parents=True, exist_ok=True)
    dest = args.out / "S1" / "student_init.pt"
    shutil.copyfile(ckpt, dest)
    if sha(dest) != digest:
        raise SystemExit("copy verification failed")
    original = args.out / "S1_RECEIPT.imported_original.json"
    shutil.copyfile(args.src / "S1_RECEIPT.json", original)
    imported = dict(receipt)
    imported["imported_from"] = {"original_checkpoint_path": receipt.get("checkpoint_path"), "original_config_sha256": receipt.get("config_sha256"),
                                 "original_receipt_sha256": sha(original), "trained_on": "RTX 4070 server", "imported_unix": time.time()}
    imported["checkpoint_path"] = str(dest)
    (args.out / "markers").mkdir(parents=True, exist_ok=True)
    write_json(args.out / "S1_RECEIPT.json", imported)
    write_json(marker, {"stage": "S1", "config_sha256": config_sha, "imported": True, "elapsed_seconds": None,
                        "student_init_sha256": digest,
                        "artifact_sha256": {"S1_RECEIPT.json": sha(args.out / "S1_RECEIPT.json")}})
    print(f"[import-s1] imported S1 student_init.pt sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
