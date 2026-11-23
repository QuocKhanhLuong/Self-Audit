#!/usr/bin/env python3
"""Open test20 exactly once for a screening arm that was finalized with ``--val-only``.

The arm's ``FINAL_RESULT.json`` must exist with ``test20_opened = false``; its val10-selected checkpoint (``best_student``
in ``resume_<arm>.pt``) is evaluated on test20 with the same evaluators as ``w2s_finalize_ssl4mis.py``, next to the final
(last-update) checkpoint.  val10 is recomputed and must match FINAL_RESULT (same checkpoint, same evaluator).  The result
is written once to ``TEST20_RESULT.json``; an existing file is never overwritten, so test20 cannot be re-opened.
Selection was made on val10 before this script runs; test20 plays no role in any decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w2s_common import load_config, split_names, write_once  # noqa: E402
from w2s_ssl4mis_eval import compat_metrics, native_metrics, predict_records  # noqa: E402


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arm", choices=("A2",), default="A2")
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = args.run / "TEST20_RESULT.json"
    if out.is_file():
        print(f"[test20] {out} exists; test20 is opened once, refusing to re-evaluate", flush=True)
        return 0
    final = json.loads((args.run / "FINAL_RESULT.json").read_text())
    if final.get("test20_opened") is not False:
        raise SystemExit("FINAL_RESULT must come from a --val-only finalization (test20_opened == false)")

    from self_audit.candidate_pg import engine
    from self_audit.candidate_pg.student_pg import build_student

    cfg, device = load_config(args.config), torch.device(args.device)
    VN, TN = split_names(cfg)
    ckpt = torch.load(args.run / f"resume_{args.arm}.pt", map_location="cpu", weights_only=False)
    if int(ckpt["best_epoch"]) != int(final["selected_epoch"]):
        raise SystemExit("checkpoint best_epoch differs from FINAL_RESULT selected_epoch")
    student = build_student(cfg)
    student.load_state_dict(ckpt["best_student"], strict=True)
    student.to(device).eval()
    bundle = engine._load_data(cfg, synthetic=False)["objects"]
    val = [bundle.val.load(k, labeled=True) for k in bundle.val.labeled_keys()]
    test = [bundle.test.load(k, labeled=True) for k in bundle.test.labeled_keys()]

    val_native = native_metrics(predict_records(student, val, device, kind="student"))
    ref = float(final["metrics"][VN]["native"]["fg_dice"])
    if abs(float(val_native["fg_dice"]) - ref) > 1e-4:
        raise SystemExit(f"val10 recomputation {val_native['fg_dice']:.6f} differs from FINAL_RESULT {ref:.6f}")
    records = predict_records(student, test, device, kind="student")
    metrics = {TN: {"native": native_metrics(records), "ssl4mis_compat": compat_metrics(records)}}
    final_metrics = None
    if ckpt.get("student") is not None:
        student.load_state_dict(ckpt["student"], strict=True)
        rec_last = predict_records(student, test, device, kind="student")
        final_metrics = {TN: {"native": native_metrics(rec_last), "ssl4mis_compat": compat_metrics(rec_last)}}
    write_once(out, {"schema": "self_audit.w2s_ssl4mis.test20_open.v1", "arm": args.name, "selected_epoch": int(ckpt["best_epoch"]),
                     "selection": "val10 student-only FG Dice (made before this script)", "val10_recomputed_fg": float(val_native["fg_dice"]),
                     "metrics": metrics, "final_epoch_metrics": final_metrics, "test20_opened": True, "split_names": [VN, TN]})
    t = metrics[TN]
    print(f"[test20] {args.name} {TN} FG {t['native']['fg_dice']:.4f} HD95 {t['native']['fg_hd95_mm']:.2f} mm | SSL4MIS-compat Dice "
          f"{t['ssl4mis_compat']['mean']['dice']:.4f} HD95(vox) {t['ssl4mis_compat']['mean']['hd95']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
