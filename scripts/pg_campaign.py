#!/usr/bin/env python3
"""Local Candidate PG campaign supervisor (S1/S2/S3/A1-A4).

This command is intentionally foreground and local: it never SSHes, starts a
server job, downloads data, or mutates another track's ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/candidate_pg/pg_k8_base.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=("S1", "S2", "S3", "arms", "all"))
    parser.add_argument("--arm", choices=("A1", "A2", "A3", "A4"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    raw_config = json.loads(Path(args.config).read_text())
    if set(raw_config) == {"pg", "cbkd"}:
        file_arm = {(False, False): "A1", (True, False): "A2", (False, True): "A3", (True, True): "A4"}[
            (raw_config["pg"]["enabled"], raw_config["cbkd"]["enabled"])
        ]
        if args.arm and args.arm != file_arm:
            parser.error("--arm conflicts with the switch-only arm config")
        args.arm = file_arm
    args.stage = args.stage or ("arms" if args.arm else "all")
    if args.arm and args.stage != "arms":
        parser.error("a single PG arm requires stage arms after shared S1/S2/S3")
    from self_audit.candidate_pg.config import apply_arm, load_config
    config = load_config(args.config)
    if set(config) == {"pg", "cbkd"}:
        base = Path(__file__).resolve().parents[1] / "configs" / "candidate_pg" / "pg_k8_base.json"
        config = apply_arm(load_config(base), config)
    if config.get("runtime", {}).get("server"):
        parser.error("PG campaign is local-only; server execution is forbidden")
    from self_audit.candidate_pg.engine import run_training
    result = run_training(config, args.out, device=args.device, stage=args.stage,
                          resume=args.resume, max_updates=args.max_updates, arm=args.arm)
    # Durable RESULT.json/receipts contain the full structured campaign.  The
    # foreground campaign keeps stdout intentionally small for runner logs.
    print(f"[PG] stage={args.stage} status={result.get('status')} out={Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
