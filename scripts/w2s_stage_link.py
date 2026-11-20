#!/usr/bin/env python3
"""Verify the reused S1 artifact (and optionally link teacher W) for W2S-Match arm run directories.

S1 reuse is allowed only if: the encoder artifact hash matches its receipt, the S1 training image list is exactly
TRAIN80 of the W2S TRAIN manifest, and the W2S configs carry the same S1-relevant sections (cinema_init, runtime,
student, TRAIN manifest hash) as the config S1 was trained under.  Writes ``S1_LINK.json`` and copies the S1 receipt
into each destination run directory; with ``--w-run`` it also copies W's ``S2_RECEIPT.json`` into the destinations
that need the frozen teacher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1-run", type=Path, required=True, help="run directory holding S1_RECEIPT.json and S1/student_init.pt")
    ap.add_argument("--s1-config", type=Path, required=True, help="config S1 was trained under (Part A A4_SA_CONFIG.json)")
    ap.add_argument("--w2s-config", type=Path, required=True, help="any resolved W2S arm config")
    ap.add_argument("--dest", type=Path, nargs="+", required=True)
    ap.add_argument("--w-run", type=Path, help="teacher W run directory; copies S2_RECEIPT.json into --teacher-dest")
    ap.add_argument("--teacher-dest", type=Path, nargs="*", default=[])
    ap.add_argument("--expect-sha256", help="expected student_init.pt sha256")
    args = ap.parse_args()

    receipt_path = args.s1_run / "S1_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    ckpt = Path(receipt["checkpoint_path"])
    digest = sha(ckpt)
    checks = {"student_init_sha256": digest, "receipt_checkpoint_sha256": receipt.get("checkpoint_sha256")}
    problems = []
    if digest != receipt.get("checkpoint_sha256"):
        problems.append("student_init.pt sha differs from the S1 receipt")
    if args.expect_sha256 and not digest.startswith(args.expect_sha256):
        problems.append("student_init.pt sha differs from the expected hash")
    if receipt.get("status") != "COMPLETED":
        problems.append(f"S1 receipt status {receipt.get('status')!r}")

    a, b = json.loads(args.s1_config.read_text()), json.loads(args.w2s_config.read_text())
    for section in ("cinema_init", "runtime", "student"):
        # hf_revision only pins the CineMA download revision; arms load the S1 artifact, never CineMA
        strip = lambda value: {k: v for k, v in value.items() if k != "hf_revision"} if isinstance(value, dict) else value
        same = strip(a.get(section)) == strip(b.get(section))
        if section == "cinema_init":
            checks["s1_hf_revision"] = a.get(section, {}).get("hf_revision")
        checks[f"config_section_equal:{section}"] = same
        if not same:
            problems.append(f"config section {section!r} differs from the S1 training config")
    same_train = a["data"]["manifests"]["train"] == b["data"]["manifests"]["train"]
    checks["train_manifest_identical"] = same_train
    if not same_train:
        problems.append("TRAIN manifest differs from the S1 training config")

    data_list = json.loads((args.s1_run / "S1" / "data_list.json").read_text())
    s1_patients = {row["patient_id"] for row in data_list}
    train = json.loads(Path(b["data"]["manifests"]["train"]["path"]).read_text())
    train80 = set(train["patients"])
    checks.update({"s1_image_patients": len(s1_patients), "train80_patients": len(train80),
                   "s1_patients_equal_train80": s1_patients == train80})
    if s1_patients != train80:
        problems.append("S1 image list is not exactly TRAIN80")
    if re.search(r"patient(055|076)", json.dumps(sorted(s1_patients))):
        problems.append("S1 saw a VAL duplicate patient")

    result = {"schema": "self_audit.w2s_stage_link.v1", "status": "FAILED" if problems else "VERIFIED", "problems": problems,
              "checks": checks, "s1_receipt": str(receipt_path), "s1_receipt_sha256": sha(receipt_path)}
    for dest in args.dest:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "S1_LINK.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if not problems:
            shutil.copyfile(receipt_path, dest / "S1_RECEIPT.json")
    if args.w_run and not problems:
        w_receipt = args.w_run / "S2_RECEIPT.json"
        for dest in args.teacher_dest:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(w_receipt, dest / "S2_RECEIPT.json")
    print(json.dumps({"status": result["status"], "problems": problems, **{k: v for k, v in checks.items()}}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
