#!/usr/bin/env python3
"""W2S saturation check (no training): confidence of frozen W on predicted-FG voxels of U; picks tau_v (§5)."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w2s_common import ROOT, clean, load_config, load_teacher, write_once  # noqa: E402


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="teacher W run directory (S2_RECEIPT.json)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.run / "SATURATION.json"
    if out.is_file():
        print(f"[saturation] {out} exists; not recomputing", flush=True)
        return 0
    from self_audit.candidate_pg import engine
    from self_audit.candidate_pg.training_step import _teacher_output
    from self_audit.candidate_pg.w2s_checks import SaturationAccumulator

    cfg, device = load_config(args.config), torch.device(args.device)
    teacher, artifact = load_teacher(cfg, args.run, device)
    data = engine._load_data(cfg, synthetic=False)
    keys = sorted(data["unlabeled_keys"])
    chosen = random.Random(42).sample(keys, min(args.frames, len(keys)))
    train = data["objects"].train
    acc = SaturationAccumulator()
    with torch.no_grad():
        for i, key in enumerate(chosen):
            frame = train.load(key, labeled=False)
            image, _, geometry = engine._frame_tensors(frame, device, labeled=False)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = _teacher_output(teacher, image, geometry)["region_logits"]
            acc.add(logits.float().softmax(1))
            if (i + 1) % 50 == 0:
                print(f"[saturation] {i + 1}/{len(chosen)}", flush=True)
    report = acc.report()
    report.update({"frames": len(chosen), "teacher_artifact": str(artifact), "selection": "random.Random(42).sample of U frame keys (min(requested, available); U63 ED/ES gives 126 frames, fewer than the 200 asked for in the W2S prompt)",
                   "note": "frames are single U cine frames (each a full 3D volume at one time point)"})
    write_once(out, report)
    print(f"[saturation] overall fraction {report['overall_fraction']:.4f} -> tau_v {report['tau_v']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
