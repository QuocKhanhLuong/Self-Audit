#!/usr/bin/env python3
"""Evaluate frozen teacher W once: VAL20 (selection split), derived VAL18, TEST50 and TEST49 (no patient118)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w2s_common import DUPLICATE_OF_VAL, ROOT, load_config, load_teacher, subset_mean, write_once  # noqa: E402


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = args.run / "W_FINAL.json"
    if out.is_file():
        print("[w-final] W_FINAL.json exists; TEST50 is evaluated once, refusing to re-evaluate", flush=True)
        return 0
    from self_audit.candidate_pg import engine, teacher_training

    cfg, device = load_config(args.config), torch.device(args.device)
    teacher, artifact = load_teacher(cfg, args.run, device)
    bundle = engine._load_data(cfg, synthetic=False)["objects"]
    metrics = {}
    for name, data in (("VAL20", bundle.val), ("TEST50", bundle.test)):
        metrics[name] = teacher_training._evaluate_teacher(teacher, data, name, device=device, epoch=0, update=0)
        print(f"[w-final] {name} FG {metrics[name].get('fg_dice')}", flush=True)
    val18 = set(bundle.val.patients) - set(bundle.val.extra_patients)
    test = metrics["TEST50"].get("per_patient", {})
    write_once(out, {"schema": "self_audit.w2s.teacher_final.v1", "teacher_artifact": str(artifact), "metrics": metrics,
                     "VAL18_from_VAL20_patients": subset_mean(metrics["VAL20"].get("per_patient", {}), val18),
                     "TEST49_without_patient118": subset_mean(test, set(test) - {DUPLICATE_OF_VAL})})
    return 0


if __name__ == "__main__":
    sys.exit(main())
