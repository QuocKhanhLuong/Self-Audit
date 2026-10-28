#!/usr/bin/env python3
"""Canonical v2 CLI. Historical bank configs are never reinterpreted."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Self-Audit semi-supervised v2; explicit limited-label budget")
    p.add_argument("--describe", action="store_true")
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--model-config", help="Explicit local backbone source and weight provenance JSON")
    p.add_argument("--data"); p.add_argument("--data-sha256"); p.add_argument("--out")
    p.add_argument("--geometry"); p.add_argument("--geometry-sha256")
    p.add_argument("--resume", help="Hash-pinned native-volume checkpoint to continue")
    p.add_argument("--resume-sha256")
    p.add_argument("--steps", type=int, help="native-volume target outer steps (resume continuation)")
    p.add_argument("--device", default="cuda"); p.add_argument("--allow-full-run", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="Manufactured CPU test, never patient evidence")
    p.add_argument("--legacy-checkpoint", help="Explicit historical checkpoint compatibility verification")
    p.add_argument("--legacy-manifest-sha256"); p.add_argument("--legacy-source-sha256")
    p.add_argument("--phase", choices=["warm_start", "adaptation"], help="phase override for self_audit.automatic.v1")
    p.add_argument("--time-budget", type=float, help="time budget seconds for self_audit.automatic.v1")
    p.add_argument("--max-cache-bytes", type=int, help="max cache bytes for self_audit.automatic.v1")
    p.add_argument("--max-controller-workspace-bytes", type=int, help="max controller workspace bytes for self_audit.automatic.v1")
    args = p.parse_args()
    if args.describe:
        print(Path(__file__).with_name("ACTIVE_PIPELINE.json").read_text()); return
    c = json.loads(Path(args.config).read_text())
    # Apply explicit model provenance before schema dispatch so native-volume
    # configs validate the actual DINO/tiny override rather than a stale file.
    if args.model_config:
        override = json.loads(Path(args.model_config).read_text())
        if not isinstance(override, dict):
            p.error("--model-config must contain a JSON object")
        c["model"] = override
    is_volume = c.get("schema") == "self_audit.volume.v1"
    is_automatic = c.get("schema") == "self_audit.automatic.v1"
    is_compact = c.get("schema") == "self_audit.compact.v1"
    br_sec = c.get("baseline_recovery")
    is_baseline_recovery = isinstance(br_sec, dict) and br_sec.get("enabled", True)
    native_only = {
        "--geometry": args.geometry,
        "--geometry-sha256": args.geometry_sha256,
        "--resume": args.resume,
        "--resume-sha256": args.resume_sha256,
        "--steps": args.steps,
    }
    if not is_baseline_recovery:
        native_only["--model-config"] = args.model_config
    legacy_only = {
        "--legacy-checkpoint": args.legacy_checkpoint,
        "--legacy-manifest-sha256": args.legacy_manifest_sha256,
        "--legacy-source-sha256": args.legacy_source_sha256,
    }
    automatic_only = {
        "--phase": args.phase,
        "--time-budget": args.time_budget,
        "--max-cache-bytes": args.max_cache_bytes,
        "--max-controller-workspace-bytes": args.max_controller_workspace_bytes,
    }
    if is_volume:
        supplied = [flag for flag, value in legacy_only.items() if value is not None]
        if supplied:
            p.error(f"legacy-only flags are invalid for self_audit.volume.v1: {', '.join(supplied)}")
        auto_supplied = [flag for flag, value in automatic_only.items() if value is not None]
        if auto_supplied:
            p.error(f"automatic-only flags are invalid for self_audit.volume.v1: {', '.join(auto_supplied)}")
        if args.resume_sha256 and not args.resume:
            p.error("--resume-sha256 requires --resume")
    elif is_automatic:
        supplied = [flag for flag, value in legacy_only.items() if value is not None]
        if supplied:
            p.error(f"legacy-only flags are invalid for self_audit.automatic.v1: {', '.join(supplied)}")
        vol_only = {
            "--geometry": args.geometry,
            "--geometry-sha256": args.geometry_sha256,
        }
        vol_supplied = [flag for flag, value in vol_only.items() if value is not None]
        if vol_supplied:
            p.error(f"volume-geometry flags are invalid for self_audit.automatic.v1: {', '.join(vol_supplied)}")
        if args.resume and not args.resume_sha256:
            p.error("--resume-sha256 is mandatory when --resume is specified")
        if args.resume_sha256 and not args.resume:
            p.error("--resume-sha256 requires --resume")
    elif is_compact:
        supplied = [flag for flag, value in legacy_only.items() if value is not None]
        if supplied:
            p.error(f"legacy-only flags are invalid for self_audit.compact.v1: {', '.join(supplied)}")
        vol_only = {
            "--geometry": args.geometry,
            "--geometry-sha256": args.geometry_sha256,
            "--resume": args.resume,
            "--resume-sha256": args.resume_sha256,
        }
        vol_supplied = [flag for flag, value in vol_only.items() if value is not None]
        if vol_supplied:
            p.error(f"volume flags are invalid for self_audit.compact.v1: {', '.join(vol_supplied)}")
        auto_supplied = [flag for flag, value in automatic_only.items() if value is not None]
        if auto_supplied:
            p.error(f"automatic-only flags are invalid for self_audit.compact.v1: {', '.join(auto_supplied)}")
    else:
        supplied = [flag for flag, value in native_only.items() if value is not None]
        if supplied:
            p.error(f"native-volume-only flags are invalid for historical schema: {', '.join(supplied)}")
        auto_supplied = [flag for flag, value in automatic_only.items() if value is not None]
        if auto_supplied:
            p.error(f"automatic-only flags are invalid for historical schema: {', '.join(auto_supplied)}")
    if is_automatic:
        from self_audit.automatic import run, validate_automatic_config
        if args.phase:
            c["phase"] = args.phase
        if args.time_budget is not None:
            c["time_budget_seconds"] = args.time_budget
        if args.max_cache_bytes is not None:
            c["max_cache_bytes"] = args.max_cache_bytes
        if args.max_controller_workspace_bytes is not None:
            c["max_controller_workspace_bytes"] = args.max_controller_workspace_bytes
        effective_steps = args.steps if args.steps is not None else c.get("steps")
        if isinstance(effective_steps, bool) or not isinstance(effective_steps, int) or effective_steps < 1:
            p.error("automatic effective steps must be a positive integer")
        if not args.synthetic and not all([args.data, args.data_sha256]):
            p.error("automatic patient runs require --data/--data-sha256; no implicit data discovery")
        if not args.out:
            p.error("automatic runs require --out for a checkpoint receipt")
        try:
            c = validate_automatic_config(c, allow_full_run=args.allow_full_run, stop_steps=args.steps)
        except ValueError as exc:
            p.error(str(exc))
        print(json.dumps(run(c, output=args.out, device=args.device, synthetic=args.synthetic,
                             manifest=args.data, manifest_sha256=args.data_sha256,
                             steps=args.steps, resume=args.resume, resume_sha256=args.resume_sha256,
                             allow_full_run=args.allow_full_run), indent=2))
        return
    if is_volume:
        from self_audit.volume import train_volume, validate_volume_config
        effective_steps = args.steps if args.steps is not None else c.get("training", {}).get("steps")
        if isinstance(effective_steps, bool) or not isinstance(effective_steps, int) or effective_steps < 1:
            p.error("native-volume effective steps must be a positive integer")
        if effective_steps > 8 and not args.allow_full_run:
            p.error("native-volume effective steps > 8 requires --allow-full-run")
        if not args.synthetic and not all([args.data, args.data_sha256, args.geometry, args.geometry_sha256]):
            p.error("native-volume patient runs require --data/--data-sha256 and --geometry/--geometry-sha256; no implicit data discovery")
        if not args.out:
            p.error("native-volume runs require --out for a checkpoint receipt")
        try:
            c = validate_volume_config(c, allow_full_run=args.allow_full_run, stop_steps=args.steps)
        except ValueError as exc:
            p.error(str(exc))
        print(json.dumps(train_volume(c, out=args.out, device=args.device, synthetic=args.synthetic,
                                      manifest=args.data, manifest_sha256=args.data_sha256,
                                      geometry_path=args.geometry, geometry_sha256=args.geometry_sha256,
                                      steps=args.steps,
                                      resume=args.resume, resume_sha256=args.resume_sha256,
                                      allow_full_run=args.allow_full_run), indent=2))
        return
    if is_compact:
        from self_audit.compact_training import validate_compact_config
        from self_audit.engine import train
        effective_steps = args.steps if args.steps is not None else c.get("steps")
        if isinstance(effective_steps, bool) or not isinstance(effective_steps, int) or effective_steps < 1:
            p.error("compact effective steps must be a positive integer")
        if effective_steps > 128 and not args.allow_full_run:
            p.error("more than 128 updates requires --allow-full-run")
        if not args.synthetic and not all([args.data, args.data_sha256]):
            p.error("compact patient runs require --data/--data-sha256; no implicit data discovery")
        if not args.out:
            p.error("compact runs require --out for a checkpoint receipt")
        try:
            c = validate_compact_config(c, allow_full_run=args.allow_full_run, stop_steps=args.steps)
        except ValueError as exc:
            p.error(str(exc))
        print(json.dumps(train(
            c,
            args.data,
            args.data_sha256,
            args.out,
            device=args.device,
            synthetic=args.synthetic,
            stop_steps=args.steps,
            allow_full_run=args.allow_full_run,
        ), indent=2))
        return
    if not all([args.data, args.data_sha256, args.out]):
        p.error("--data, --data-sha256 and --out are mandatory; v1 --bank is unsupported")
    from self_audit.engine import train, validate_config
    if args.legacy_checkpoint:
        p.error("--legacy-checkpoint cannot verify then fresh-train; use load_legacy_checkpoint with explicit model/canonical hashes")
    c = validate_config(c)
    if not args.synthetic:
        # Prevent CLI bypass for shared real training without baseline recovery
        manifest_obj = json.loads(Path(args.data).read_text())
        is_shared_manifest = bool(manifest_obj.get("shared", False)) or len(manifest_obj.get("domains", [])) > 1
        if is_shared_manifest:
            br_cfg = c.get("baseline_recovery")
            if not isinstance(br_cfg, dict) or not br_cfg.get("enabled", True):
                p.error("shared real training requires baseline_recovery configuration to be present and enabled")
            cert_p = br_cfg.get("baseline_certificate_path")
            if not cert_p or not Path(cert_p).is_file():
                p.error(f"shared real training requires valid baseline_certificate_path: {cert_p}")
            cert_sha = br_cfg.get("baseline_certificate_sha256")
            if not cert_sha or len(cert_sha) != 64:
                p.error("shared real training requires mandatory baseline_certificate_sha256 (64 hex characters)")
            from self_audit.recovery_gates import block_shared_training_unless_baseline_gated
            try:
                block_shared_training_unless_baseline_gated(
                    cert_p,
                    expected_cert_sha=cert_sha,
                    shared_manifest_path=args.data,
                    shared_config=c,
                )
            except Exception as e:
                p.error(f"shared real training baseline gate check failed: {e}")
    if c["steps"] > 128 and not args.allow_full_run:
        p.error("more than 128 updates requires --allow-full-run")
    print(json.dumps(train(c, args.data, args.data_sha256, args.out, device=args.device, synthetic=args.synthetic), indent=2))


if __name__ == "__main__":
    main()
