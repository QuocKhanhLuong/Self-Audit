#!/usr/bin/env python3
"""Evaluate the frozen teacher W once: val10 (selection split) and test20, native and SSL4MIS-compatible metrics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w2s_common import load_config, load_teacher, write_once  # noqa: E402
from w2s_ssl4mis_eval import compat_metrics, native_metrics, predict_records  # noqa: E402


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = args.run / "W_FINAL.json"
    if out.is_file():
        print("[w-final] W_FINAL.json exists; test20 is evaluated once, refusing to re-evaluate", flush=True)
        return 0
    from self_audit.candidate_pg import engine

    cfg, device = load_config(args.config), torch.device(args.device)
    teacher, artifact = load_teacher(cfg, args.run, device)
    bundle = engine._load_data(cfg, synthetic=False)["objects"]
    result = {"schema": "self_audit.w2s_ssl4mis.teacher_final.v1", "teacher_artifact": str(artifact), "selection_split": "val10", "metrics": {}}
    for name, data in (("val10", bundle.val), ("test20", bundle.test)):
        frames = [data.load(k, labeled=True) for k in data.labeled_keys()]
        records = predict_records(teacher, frames, device, kind="teacher")
        result["metrics"][name] = {"native": native_metrics(records), "ssl4mis_compat": compat_metrics(records)}
        print(f"[w-final] {name} FG {result['metrics'][name]['native']['fg_dice']}", flush=True)
    write_once(out, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
