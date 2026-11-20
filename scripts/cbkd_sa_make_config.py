#!/usr/bin/env python3
"""Write the resolved A4-SA config from the real pg-k8 resolved config and the newly prepared manifests.

Differences from the A4 resolved config (reports/pg-server-20261116/PG_K8_CONFIG.json + A4 switches):
  * cbkd.variant = cbkd_sa, cbkd.warmup_updates_labeled_only = 1200, cbkd.ramp_updates = 400
  * data.manifests / data.materialization: the re-prepared data (hash-pinned; reported if they differ)
  * training.eval_paths = [student_only] (no fusion-path monitoring; approved cost cut)
  * training.eval_splits drops TEST50 (TEST50 is evaluated once, after selection, by cbkd_sa_finalize.py)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(prepared: Path, base_path: Path) -> dict:
    from self_audit.candidate_pg.config import validate_config

    cfg = json.loads(base_path.read_text())
    cfg = copy.deepcopy(cfg)
    receipt = json.loads((prepared / "PREPARATION_RECEIPT.json").read_text())
    for role, name in (("train", "train.json"), ("val", "val.json"), ("test", "test.json")):
        # The preparer pins the resolved path (on Modal a /__modal/volumes/<id>/... alias); the data loader
        # requires config path == receipt path, so use the receipt's own string.
        pinned = receipt["manifests"][role]
        assert pinned["sha256"] == sha(prepared / name), f"{name} differs from its receipt pin"
        cfg["data"]["manifests"][role] = {"path": pinned["path"], "sha256": pinned["sha256"]}
    cfg["data"]["materialization"] = {"receipt": str(prepared / "PREPARATION_RECEIPT.json"), "sha256": sha(prepared / "PREPARATION_RECEIPT.json")}
    cfg["pg"]["enabled"] = True
    cfg["cbkd"].update({"enabled": True, "variant": "cbkd_sa", "warmup_updates_labeled_only": 1200, "ramp_updates": 400})
    cfg["training"]["eval_splits"] = ["train_labeled", "VAL18"]
    cfg["training"]["eval_paths"] = ["student_only"]  # option 4 (approved): no per-epoch fusion-path evaluation
    return validate_config(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=ROOT / "reports/pg-server-20261116/PG_K8_CONFIG.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = build(args.prepared, args.base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
