#!/usr/bin/env python3
"""Import a completed S1 (student_init.pt + receipt) or S2 (teacher_best.pt + S2_RECEIPT.json; --stage S2) trained elsewhere (RTX 4070 server) into this run directory.

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


CFG = {"S1": ("student_init.pt", "S1_RECEIPT.json", "S1"), "S2": ("teacher_best.pt", "S2_RECEIPT.json", "S2")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(CFG), default="S1")
    ap.add_argument("--src", type=Path, required=True, help="folder holding student_init.pt and S1_RECEIPT.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--expect-sha256", default=None)
    args = ap.parse_args(argv)
    ckpt_name, receipt_name, sub = CFG[args.stage]
    config_sha = sha(args.config)
    marker = args.out / "markers" / f"{args.stage}.DONE.json"
    if marker.is_file() and json.loads(marker.read_text()).get("config_sha256") == config_sha:
        print(f"[import] {args.stage} DONE marker present, nothing to do")
        return 0
    receipt = json.loads((args.src / receipt_name).read_text())
    ckpt = args.src / ckpt_name
    digest = sha(ckpt)
    if digest != receipt.get("checkpoint_sha256"):
        raise SystemExit(f"student_init.pt sha {digest} != receipt checkpoint_sha256")
    if args.expect_sha256 and digest != args.expect_sha256:
        raise SystemExit("student_init.pt sha differs from the expected hash")
    if receipt.get("status") != "COMPLETED":
        raise SystemExit(f"{args.stage} receipt status {receipt.get('status')!r} is not COMPLETED")
    (args.out / sub).mkdir(parents=True, exist_ok=True)
    dest = args.out / sub / ckpt_name
    shutil.copyfile(ckpt, dest)
    if sha(dest) != digest:
        raise SystemExit("copy verification failed")
    original = args.out / f"{receipt_name[:-5]}.imported_original.json"
    shutil.copyfile(args.src / receipt_name, original)
    imported = dict(receipt)
    imported["imported_from"] = {"source": "imported from server4070", "original_checkpoint_path": receipt.get("checkpoint_path"), "original_config_sha256": receipt.get("config_sha256"),
                                 "original_receipt_sha256": sha(original), "trained_on": "RTX 4070 server", "imported_unix": time.time()}
    for key in ("checkpoint", "checkpoint_path", "selected_checkpoint"):  # re-point only keys naming the imported artifact
        if key in imported and Path(str(imported[key])).name == ckpt_name:
            imported[key] = str(dest)
    imported.pop("latest_checkpoint", None)  # optimizer/resume checkpoint of the other machine is not imported
    (args.out / "markers").mkdir(parents=True, exist_ok=True)
    write_json(args.out / receipt_name, imported)
    write_json(marker, {"stage": args.stage, "config_sha256": config_sha, "imported": True, "source": "imported from server4070", "elapsed_seconds": None,
                        "checkpoint_sha256": digest,
                        "artifact_sha256": {receipt_name: sha(args.out / receipt_name)}})
    print(f"[import] imported {args.stage} {ckpt_name} sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
