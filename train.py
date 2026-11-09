#!/usr/bin/env python3
"""Canonical A/A-3D training CLI; dimension tracks are explicit in config."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Candidate A: TRAIN labels, epoch metrics, student-only deployment")
    p.add_argument("--describe", action="store_true")
    p.add_argument("--config", default="configs/candidate_a/full.json")
    p.add_argument("--model-config", help="Explicit local backbone source and weight provenance JSON")
    p.add_argument("--teacher-checkpoint", help="Candidate A: verified initialization folder")
    p.add_argument("--teacher-checkpoint-sha256", help="Candidate A: MANIFEST.json SHA256")
    p.add_argument("--teacher-checkpoint-kind", choices=["candidate_a", "candidate_a3d", "native_v2"])
    p.add_argument("--data"); p.add_argument("--data-sha256"); p.add_argument("--out")
    p.add_argument("--validation-data", help="Disjoint development manifest, for epoch metrics only")
    p.add_argument("--validation-data-sha256", help="SHA256 of the development manifest")
    p.add_argument("--device", default="cuda"); p.add_argument("--allow-full-run", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="Manufactured CPU test, never patient evidence")
    args = p.parse_args()
    if args.describe:
        print(Path(__file__).with_name("ACTIVE_PIPELINE.json").read_text()); return
    if not all([args.data, args.data_sha256, args.out]):
        p.error("--data, --data-sha256 and --out are mandatory")
    if bool(args.validation_data) != bool(args.validation_data_sha256):
        p.error("validation requires both manifest and SHA256")
    c = json.loads(Path(args.config).read_text())
    if c.get("method") == "candidate_a3d":
        from self_audit.candidate_a3d.engine import train
        from self_audit.candidate_a3d.config import validate_config
    elif c.get("method") == "candidate_a":
        from self_audit.candidate_a.engine import train
        from self_audit.candidate_a.config import validate_config
    else:
        p.error("config.method must be candidate_a or candidate_a3d")
    if args.model_config:
        c["model"] = json.loads(Path(args.model_config).read_text())
    if args.teacher_checkpoint or args.teacher_checkpoint_sha256:
        if not (args.teacher_checkpoint and args.teacher_checkpoint_sha256):
            p.error("teacher initialization requires both path and manifest SHA256")
        c["teacher_init"] = {"kind": args.teacher_checkpoint_kind or c["method"], "path": args.teacher_checkpoint,
                             "sha256": args.teacher_checkpoint_sha256}
    c = validate_config(c)
    from self_audit.model import digest
    if digest(args.data) != args.data_sha256:
        p.error("training manifest hash mismatch")
    data = json.loads(Path(args.data).read_text())
    geometry = data.get("geometry", {}).get("by_patient", {})
    updates = c["epochs"] * sum(len(r["masks"]) * (
        1 if c["method"] == "candidate_a3d" else geometry[r["id"]]["shape_zyx"][0]
    ) for r in data["labeled"])
    if updates > 128 and not args.allow_full_run:
        p.error("more than 128 updates requires --allow-full-run")
    print(json.dumps(train(c, args.data, args.data_sha256, args.out, device=args.device,
                          synthetic=args.synthetic, validation_manifest=args.validation_data,
                          validation_sha256=args.validation_data_sha256), indent=2))


if __name__ == "__main__":
    main()
