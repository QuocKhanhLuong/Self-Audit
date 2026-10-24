#!/usr/bin/env python3
"""Build a frozen paired-feature bank offline from TRAIN images only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.memory_bank import build_memory_bank, sha256_file, write_a1_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--native-manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--mode", choices=("teacher", "a5"), default="teacher")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-patients", type=int, help="bounded smoke build; marked incomplete")
    parser.add_argument("--max-timepoints", type=int, help="bounded smoke build; marked incomplete")
    parser.add_argument("--teacher-config", type=Path)
    parser.add_argument("--teacher-weights", type=Path)
    parser.add_argument("--teacher-source", type=Path)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--student-checkpoint-manifest-sha256")
    parser.add_argument("--write-a1-cache", action="store_true",
                        help="also save complete native-grid TRAIN probabilities (teacher mode only)")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory must be new; preserve prior artifacts")
    if args.max_patients is not None and args.max_patients < 1:
        parser.error("--max-patients must be positive")
    if args.max_timepoints is not None and args.max_timepoints < 1:
        parser.error("--max-timepoints must be positive")

    adapter = None
    if args.mode == "teacher":
        if args.write_a1_cache and not all((args.teacher_config, args.teacher_weights, args.teacher_source)):
            parser.error("teacher mode requires --teacher-config, --teacher-weights and --teacher-source")
        if not all((args.teacher_config, args.teacher_weights, args.teacher_source)):
            parser.error("teacher mode requires --teacher-config, --teacher-weights and --teacher-source")
        from self_audit.cinema_teacher import CineMATeacher
        adapter = CineMATeacher(args.teacher_config, args.teacher_weights,
                                source_root=args.teacher_source, device=args.device)
        if getattr(adapter, "synthetic", False):
            parser.error("synthetic adapter seams cannot write a patient memory bank")
    else:
        if args.write_a1_cache:
            parser.error("A5 uses only the frozen student and cannot produce external-teacher A1 targets")
        if args.teacher_config or args.teacher_weights or args.teacher_source:
            parser.error("A5 must not receive external teacher assets")
        if not args.student_checkpoint or not args.student_checkpoint_manifest_sha256:
            parser.error("A5 requires --student-checkpoint and its manifest SHA256")

    args.out.mkdir(parents=True)
    a1_cache_dir = args.out / "external_pseudo" if args.write_a1_cache else None
    result = build_memory_bank(
        args.native_manifest, args.native_manifest_sha256, mode=args.mode, adapter=adapter,
        student_checkpoint=args.student_checkpoint,
        student_manifest_sha256=args.student_checkpoint_manifest_sha256,
        device=args.device, max_patients=args.max_patients, max_timepoints=args.max_timepoints,
        collect_a1_cache=args.write_a1_cache, a1_cache_dir=a1_cache_dir,
    )
    from self_audit.teacher_memory import MemoryBank, save_bank
    payload = result["bank"]
    bank = MemoryBank(payload["endpoints"], payload["confidences"], payload["valid"], payload["metadata"])
    bank_path = args.out / "fixed_bank.npz"
    bank_sha256 = save_bank(bank, bank_path)
    cache_path = None
    if args.write_a1_cache:
        cache_path = write_a1_cache(args.out / "external_pseudo", result, bank_sha256=bank_sha256)
    receipt = {
        "schema": "self_audit.memory_bank_build.v1", "status": "BUILT",
        "mode": "A5_student_only" if args.mode == "a5" else "external_teacher_image_only",
        "bank": str(bank_path.resolve()), "bank_sha256": bank_sha256,
        "native_manifest_sha256": args.native_manifest_sha256,
        "coverage": result["coverage"],
        "coverage_status": "FULL_TRAIN" if result["coverage"]["complete"] else "BOUNDED_NOT_FULL_BANK",
        "a1_cache": str(cache_path.resolve()) if cache_path else None,
        "mask_paths_read": 0, "evaluation_ids_used": False,
    }
    receipt_path = args.out / "BUILD_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    receipt["receipt_sha256"] = sha256_file(receipt_path)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
