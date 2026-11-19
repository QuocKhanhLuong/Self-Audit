#!/usr/bin/env python3
"""Phase 2 timing probe: 100 updates of each stage type (S1, S2, S3-SA, A4-SA), nothing kept as a final artifact.

All warmups are set to 0 in the probe copy so the expensive paths (U streams, PG, CBKD-SA) are timed;
the real config is untouched.  Reports seconds/update (wall incl. one epoch evaluation, and the in-loop
step time from diag files) and peak CUDA memory per stage.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import torch
    from self_audit.candidate_pg.engine import run_training

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--updates", type=int, default=100)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    cfg = copy.deepcopy(cfg)
    cfg["pg"]["lambda_u"]["warmup_updates_labeled_only"] = 0
    cfg["teacher"]["semi"]["lambda_u"]["warmup_updates_labeled_only"] = 0
    cfg["cbkd"]["warmup_updates_labeled_only"] = 0
    cfg["cbkd"]["ramp_updates"] = 1
    report = {}
    for label, stage, arm in (("S1", "S1", None), ("S2", "S2", None), ("S3", "S3", None), ("A4", "arms", "A4")):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        run_training(cfg, args.out, device="cuda", stage=stage, resume=False, max_updates=args.updates, arm=arm)
        wall = time.time() - t0
        entry = {"wall_seconds": wall, "wall_s_per_update": wall / args.updates, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        diag = {"S2": args.out / "S2" / "diag.jsonl", "S3": args.out / "S3" / "diag.jsonl", "A4": args.out / "arms" / "A4" / "diag.jsonl"}.get(label)
        if diag is not None and diag.is_file():
            rows = [json.loads(line) for line in diag.read_text().splitlines() if line.strip()]
            secs = [r["seconds"] for r in rows if "seconds" in r]
            entry["step_s_per_update_median"] = sorted(secs)[len(secs) // 2] if secs else None
        report[label] = entry
        print(json.dumps({label: entry}), flush=True)
        (args.out / "PROBE.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
