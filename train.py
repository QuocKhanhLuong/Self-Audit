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
    p.add_argument("--device", default="cuda"); p.add_argument("--allow-full-run", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="Manufactured CPU test, never patient evidence")
    args = p.parse_args()
    if args.describe:
        print(Path(__file__).with_name("ACTIVE_PIPELINE.json").read_text()); return
    if not all([args.data, args.data_sha256, args.out]):
        p.error("--data, --data-sha256 and --out are mandatory; v1 --bank is unsupported")
    from self_audit.engine import train, validate_config
    c = json.loads(Path(args.config).read_text())
    if args.model_config:
        c["model"] = json.loads(Path(args.model_config).read_text())
    c = validate_config(c)
    if c["steps"] > 128 and not args.allow_full_run:
        p.error("more than 128 updates requires --allow-full-run")
    print(json.dumps(train(c, args.data, args.data_sha256, args.out, device=args.device, synthetic=args.synthetic), indent=2))


if __name__ == "__main__":
    main()
