#!/usr/bin/env python3
"""Synthetic CPU-only smoke for the Candidate PG orchestrator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()
    from self_audit.candidate_pg.config import default_config, validate_config
    from self_audit.candidate_pg.engine import run_training
    config = validate_config(default_config(synthetic=True))
    # The production recipe intentionally locks both PG/CBKD to 800
    # labelled-only warmup updates.  A one-update software smoke would never
    # execute those mechanisms, so this fixture explicitly opts into a
    # warmup-zero diagnostic override; the engine writes a durable receipt and
    # keeps the run labelled SOFTWARE_ONLY.
    config["pg"]["lambda_u"]["warmup_updates_labeled_only"] = 0
    config["cbkd"]["warmup_updates_labeled_only"] = 0
    config["teacher"]["semi"]["lambda_u"]["warmup_updates_labeled_only"] = 0
    # The tiny teacher fixture is intentionally near-uniform at initialization;
    # a zero threshold is an explicit smoke-only diagnostic override so the
    # UniMatch/F P streams execute instead of producing an empty confidence
    # mask.  The locked production clip remains [0.5, 0.95].
    config["teacher"]["semi"]["threshold"]["clip"] = [0.0, 0.0]
    config["runtime"]["synthetic_warmup_override"] = {
        "status": "APPLIED",
        "pg_lambda_u_warmup_updates_labeled_only": 0,
        "cbkd_warmup_updates_labeled_only": 0,
        "teacher_lambda_u_warmup_updates_labeled_only": 0,
        "teacher_threshold_clip": [0.0, 0.0],
        "reason": "exercise_s2_pg_cbkd_on_cpu_smoke",
    }
    result = run_training(config, args.out, device="cpu", stage="all", max_updates=max(1, args.updates))
    result["software_only"] = True
    result["real_runs"] = "NOT_RUN"
    # RESULT.json is the durable machine-readable receipt.  Keep stdout to a
    # single concise status line so callers cannot mistake console chatter for
    # the campaign artifact.
    result_path = Path(args.out) / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[PG] smoke status={result.get('status')} out={Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
