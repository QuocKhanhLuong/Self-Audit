#!/usr/bin/env python3
"""Verify the reused S1 artifact for the SSL4MIS-split campaign and link S1 / teacher W receipts into arm run directories.

S1 is reused, never retrained.  Checks: encoder artifact hash vs its receipt (and an optional expected hash); the S1-relevant
config sections (cinema_init, runtime, student) equal the config S1 was trained under (``hf_revision`` pin ignored: arms load the
S1 artifact, not CineMA).  The overlap between the S1 image list and the new val10/test20 patients is *recorded*, not rejected:
S1 saw image-only frames of 6/10 val and 19/20 test patients (disclosed label-free exposure, see INSPECTION.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1-run", type=Path, required=True)
    ap.add_argument("--s1-config", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True, help="any resolved SSL4MIS arm config")
    ap.add_argument("--dest", type=Path, nargs="+", required=True)
    ap.add_argument("--w-run", type=Path)
    ap.add_argument("--teacher-dest", type=Path, nargs="*", default=[])
    ap.add_argument("--expect-sha256")
    args = ap.parse_args()
    receipt_path = args.s1_run / "S1_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    ckpt = Path(receipt["checkpoint_path"])
    digest = sha(ckpt)
    problems, checks = [], {"student_init_sha256": digest, "receipt_checkpoint_sha256": receipt.get("checkpoint_sha256")}
    if digest != receipt.get("checkpoint_sha256"):
        problems.append("student_init.pt sha differs from the S1 receipt")
    if args.expect_sha256 and not digest.startswith(args.expect_sha256):
        problems.append("student_init.pt sha differs from the expected hash")
    if receipt.get("status") != "COMPLETED":
        problems.append(f"S1 receipt status {receipt.get('status')!r}")
    a, b = json.loads(args.s1_config.read_text()), json.loads(args.config.read_text())
    strip = lambda v: {k: x for k, x in v.items() if k != "hf_revision"} if isinstance(v, dict) else v
    for section in ("cinema_init", "runtime", "student"):
        same = strip(a.get(section)) == strip(b.get(section))
        checks[f"config_section_equal:{section}"] = same
        if not same:
            problems.append(f"config section {section!r} differs from the S1 training config")
    checks["s1_hf_revision"] = a.get("cinema_init", {}).get("hf_revision")
    seen = {row["patient_id"] for row in json.loads((args.s1_run / "S1" / "data_list.json").read_text())}
    lock = json.loads(Path(b["data"]["ssl4mis"]["lock"]["path"]).read_text())
    overlap = {role: sorted(set(lock[role]["patients"]) & seen) for role in ("labeled", "unlabeled", "val", "test")}
    checks["s1_image_patients"] = len(seen)
    checks["s1_overlap_counts"] = {role: f"{len(ids)}/{len(lock[role]['patients'])}" for role, ids in overlap.items()}
    result = {"schema": "self_audit.w2s_ssl4mis.stage_link.v1", "status": "FAILED" if problems else "VERIFIED", "problems": problems,
              "checks": checks, "s1_overlap_patients": overlap, "s1_receipt": str(receipt_path), "s1_receipt_sha256": sha(receipt_path),
              "note": "S1 overlap with val10/test20 is label-free image exposure only; S1 is not retrained"}
    for dest in args.dest:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "S1_LINK.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if not problems:
            shutil.copyfile(receipt_path, dest / "S1_RECEIPT.json")
    if args.w_run and not problems:
        for dest in args.teacher_dest:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.w_run / "S2_RECEIPT.json", dest / "S2_RECEIPT.json")
    print(json.dumps({"status": result["status"], "problems": problems, **checks}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
