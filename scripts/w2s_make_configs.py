#!/usr/bin/env python3
"""Write the resolved W2S-Match arm configs (W, S0, S2, S2arb) bound to the prepared, hash-pinned data.

Starts from ``candidate_pg.w2s_config.make_config`` (PG K8 base + the reviewed per-arm fields) and adds only
the run-environment bindings, identical for every arm:
  * data.manifests / data.materialization: the prepared ACDC receipt (as in cbkd_sa_make_config.py)
  * data.w2s_val20: the hash-pinned VAL20 manifest (VAL18 + patient055/076); selection and monitoring use it
  * training.eval_splits = [train_labeled, VAL20] (TEST50 is evaluated once, after selection, by the finalizers)
  * training.eval_paths = [student_only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(arm: str, prepared: Path, val20_dir: Path, tau_v: float) -> dict:
    from self_audit.candidate_pg.config import validate_config
    from self_audit.candidate_pg.w2s_config import make_config

    cfg = make_config(arm, tau_v=tau_v)
    receipt = json.loads((prepared / "PREPARATION_RECEIPT.json").read_text())
    for role, name in (("train", "train.json"), ("val", "val.json"), ("test", "test.json")):
        pinned = receipt["manifests"][role]
        assert pinned["sha256"] == sha(prepared / name), f"{name} differs from its receipt pin"
        cfg["data"]["manifests"][role] = {"path": pinned["path"], "sha256": pinned["sha256"]}
    cfg["data"]["materialization"] = {"receipt": str(prepared / "PREPARATION_RECEIPT.json"), "sha256": sha(prepared / "PREPARATION_RECEIPT.json")}
    val20 = json.loads((val20_dir / "VAL20_RECEIPT.json").read_text())
    assert val20["status"] == "READY" and val20["base_val18"]["sha256"] == sha(prepared / "val.json")
    assert val20["val20"]["sha256"] == sha(val20_dir / "val20.json")
    cfg["data"]["w2s_val20"] = {"path": val20["val20"]["path"], "sha256": val20["val20"]["sha256"]}
    cfg["training"]["eval_splits"] = ["train_labeled", "VAL20"]
    cfg["training"]["eval_paths"] = ["student_only"]
    return validate_config(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--val20-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tau-v", type=float, default=0.99, help="0.99, or 0.999 only if the saturation check says so")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for arm in ("W", "S0", "S2", "S2arb"):
        cfg = build(arm, args.prepared, args.val20_dir, args.tau_v)
        path = args.out_dir / f"w2s_{arm}.json"
        path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path} sha256 {sha(path)[:16]}")


if __name__ == "__main__":
    main()
