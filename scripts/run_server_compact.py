#!/usr/bin/env python3
"""Profile, lock, and run one server S0 through the canonical compact trainer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.server_runner import (
    backup_artifacts,
    create_protocol_lock,
    profile_batches,
    retry_backup,
    run_locked_experiment,
    sha256_file,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="measure canonical steps at batch 8/32/64")
    profile.add_argument("--out", required=True, type=Path)
    profile.add_argument("--train-manifest", type=Path)
    profile.add_argument("--train-sha256")
    profile.add_argument("--synthetic", action="store_true", help="generate a manufactured TRAIN fixture when no manifest is given")
    profile.add_argument("--device", default="cuda")
    profile.add_argument("--warmup-steps", type=int, default=2)
    profile.add_argument("--measured-steps", type=int, default=5)

    lock = commands.add_parser("lock", help="pin one batch 32/64 and the exact manifests")
    lock.add_argument("--profile", required=True, type=Path)
    lock.add_argument("--profile-sha256", required=True)
    lock.add_argument("--train-manifest", required=True, type=Path)
    lock.add_argument("--train-sha256", required=True)
    lock.add_argument("--val-manifest", required=True, type=Path)
    lock.add_argument("--val-sha256", required=True)
    lock.add_argument("--test-manifest", type=Path)
    lock.add_argument("--test-sha256")
    lock.add_argument("--batch-size", required=True, type=int, choices=(32, 64))
    lock.add_argument("--out", required=True, type=Path)

    run = commands.add_parser("run", help="run exactly one locked 10,000-step S0")
    run.add_argument("--lock", required=True, type=Path)
    run.add_argument("--lock-sha256", required=True)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--device", default="cuda")
    run.add_argument("--rclone-destination", required=True, help="existing remote parent; a fresh run-specific child is created")
    run.add_argument("--resume-checkpoint", type=Path)
    run.add_argument("--resume-sha256")
    run.add_argument("--retry-delay-seconds", type=float, default=15.0)

    backup = commands.add_parser("backup", help="retry a completed run's pending rclone backup")
    backup.add_argument("--run-root", required=True, type=Path)
    backup.add_argument("--rclone-destination", required=True)
    backup.add_argument("--device", default="cuda", help="device for any deferred final VAL/TEST completion")
    backup.add_argument("--retry-delay-seconds", type=float, default=15.0)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "profile":
        result = profile_batches(
            args.out,
            train_manifest=args.train_manifest,
            train_sha256=args.train_sha256,
            synthetic=args.synthetic,
            device=args.device,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
        )
        result["profile_file"] = str(args.out.resolve() / "PROFILE.json")
        result["profile_sha256"] = sha256_file(args.out.resolve() / "PROFILE.json")
    elif args.command == "lock":
        result = create_protocol_lock(
            args.out,
            profile_path=args.profile,
            profile_sha256=args.profile_sha256,
            train_manifest=args.train_manifest,
            train_sha256=args.train_sha256,
            val_manifest=args.val_manifest,
            val_sha256=args.val_sha256,
            test_manifest=args.test_manifest,
            test_sha256=args.test_sha256,
            batch_size=args.batch_size,
        )
        result["lock_file"] = str(args.out.resolve())
        result["lock_sha256"] = sha256_file(args.out.resolve())
    elif args.command == "run":
        result = run_locked_experiment(
            args.lock,
            args.lock_sha256,
            args.out,
            device=args.device,
            rclone_base=args.rclone_destination,
            resume_checkpoint=args.resume_checkpoint,
            resume_sha256=args.resume_sha256,
            retry_delay_seconds=args.retry_delay_seconds,
        )
        if result.get("status") == "UPLOAD_PENDING":
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
    else:
        result = retry_backup(
            args.run_root,
            rclone_base=args.rclone_destination,
            retry_delay_seconds=args.retry_delay_seconds,
            device=args.device,
        )
        if result.get("status") == "UPLOAD_PENDING":
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
