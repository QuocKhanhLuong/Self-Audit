#!/usr/bin/env python3
"""Final step of a student arm: test20 exactly once on the val10-selected checkpoint, export, cold-load check, complexity."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w2s_common import load_config, write_once  # noqa: E402
from w2s_ssl4mis_eval import compat_metrics, complexity, inter_slice_consistency, native_metrics, predict_records  # noqa: E402


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arm", choices=("A2", "A4"), required=True, help="engine arm: A2 = S0 / S1kd / S2arb, A4 = S2main (codebook)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-only", action="store_true", help="screening arms: never open test20 (val10 metrics + inter-slice consistency only)")
    ap.add_argument("--early-stop-ok", action="store_true", help="evaluate a run that the user stopped before its planned update count (recorded in FINAL_RESULT)")
    args = ap.parse_args()
    final = args.run / "FINAL_RESULT.json"
    if final.is_file():
        print("[final] FINAL_RESULT.json exists; test20 is evaluated once, refusing to re-evaluate", flush=True)
        return 0
    from self_audit.candidate_pg import engine, shape_atoms
    from self_audit.candidate_pg.export import export_student, load_student_export, verify_cold_load
    from self_audit.candidate_pg.student_pg import build_student

    cfg, device = load_config(args.config), torch.device(args.device)
    ckpt = torch.load(args.run / f"resume_{args.arm}.pt", map_location="cpu", weights_only=False)
    planned = int(cfg["training"]["updates"])
    early = int(ckpt["update"]) != planned
    if ckpt.get("best_student") is None or (early and not args.early_stop_ok):
        raise RuntimeError("arm has not finished (resume update != training.updates); not evaluating (pass --early-stop-ok for a deliberately stopped run)")
    student = build_student(cfg)
    aniso = type(student).__name__ in ("AnisoStudent", "SliceSSMStudent")
    slicessm = type(student).__name__ == "SliceSSMStudent"
    if args.arm == "A4":
        receipt = engine._read_receipt(args.run / "S3_RECEIPT.json")
        student.attach_code_branch(shape_atoms.StudentAtomBranch(shape_atoms.load_shape_atoms(receipt["artifact"])))
    student.load_state_dict(ckpt["best_student"], strict=True)
    student.to(device).eval()
    bundle = engine._load_data(cfg, synthetic=False)["objects"]
    val = [bundle.val.load(k, labeled=True) for k in bundle.val.labeled_keys()]
    test = [] if args.val_only else [bundle.test.load(k, labeled=True) for k in bundle.test.labeled_keys()]  # --val-only never reads test20
    metrics = {}
    splits = (("val10", val),) if args.val_only else (("val10", val), ("test20", test))
    inter_slice = None
    for name, frames in splits:
        records = predict_records(student, frames, device, kind="student")
        metrics[name] = {"native": native_metrics(records), "ssl4mis_compat": compat_metrics(records)}
        if name == "val10":
            inter_slice = inter_slice_consistency(records)
    # also the final (last-update) checkpoint, reported next to the val10-selected one (selection is never made on test20)
    final_metrics = None
    if ckpt.get("student") is not None:
        student.load_state_dict(ckpt["student"], strict=True)
        final_metrics = {}
        for name, frames in splits:
            records = predict_records(student, frames, device, kind="student")
            final_metrics[name] = {"native": native_metrics(records), "ssl4mis_compat": compat_metrics(records)}
        student.load_state_dict(ckpt["best_student"], strict=True)
    student.cpu()
    volumes = [(f.image if f.image.ndim == 5 else f.image.unsqueeze(0), dict(f.geometry)) for f in val[:3]]
    if aniso:
        from self_audit.candidate_pg.aniso_student import export_aniso_student, load_aniso_export
        if slicessm:
            from self_audit.candidate_pg.slicessm_student import export_slicessm_student as export_aniso_student  # noqa: F811
            from self_audit.candidate_pg.slicessm_student import load_slicessm_export as load_aniso_export  # noqa: F811

        (args.run / "exports").mkdir(parents=True, exist_ok=True)
        export = export_aniso_student(student, args.run / "exports" / f"{args.name}.pt", epoch=int(ckpt["best_epoch"]))
        deployed = load_aniso_export(export["path"])
        bad = [k for k in deployed.state_dict() if any(t in k for t in ("code_branch", "sdf_head", "aux_heads", "stage3_tap", "teacher"))]
        if bad:
            raise RuntimeError(f"export contains non-deployment keys: {bad[:5]}")
        student.eval()
        with torch.no_grad():
            diffs = [float((student(x, g)["logits"] - deployed(x, g)["logits"]).abs().max()) for x, g in volumes]
        cold = {"status": "PASS" if max(diffs) < 1e-4 else "FAIL", "max_abs_logit_diff": max(diffs)}
    else:
        export = export_student(student, cfg, args.run / "exports" / f"{args.name}.pt", epoch=int(ckpt["best_epoch"]),
                                metadata={"arm": args.name, "selected_epoch": int(ckpt["best_epoch"]), "selection": "val10 student-only FG Dice",
                                          "test_policy": "test20 once after selection"})
        deployed = load_student_export(export["path"])
        bad = [k for k in deployed.state_dict() if any(t in k for t in ("code_branch", "P_S", "D_A", "teacher", "fusions."))]
        if bad:
            raise RuntimeError(f"export contains non-deployment keys: {bad[:5]}")
        cold = verify_cold_load(student, export["path"], volumes)
    cx = complexity(deployed.to(device).eval(), val[0], device)
    write_once(final, {"schema": "self_audit.w2s_ssl4mis.final.v1", "arm": args.name, "selected_epoch": int(ckpt["best_epoch"]),
                       "best_val_fg": float(ckpt["best_val"]), "metrics": metrics, "final_epoch_metrics": final_metrics,
                       "stopped_early": {"update": int(ckpt["update"]), "planned_updates": planned, "epochs_done": int(ckpt["update"]) / int(cfg["training"]["updates_per_epoch"])} if early else None, "epochs": cfg["training"].get("epochs", int(cfg["training"]["updates"]) // int(cfg["training"]["updates_per_epoch"])), "export": export, "cold_load": cold, "complexity": cx,
                       "inter_slice_consistency_val10": inter_slice, "test20_opened": not args.val_only})
    if args.val_only:
        print(f"[final] {args.name} val10 FG {metrics['val10']['native']['fg_dice']:.4f} HD95 {metrics['val10']['native']['fg_hd95_mm']:.2f} mm "
              f"| inter-slice consistency pred/GT {inter_slice['ratio']:.3f} (test20 not opened)", flush=True)
    else:
        print(f"[final] {args.name} val10 FG {metrics['val10']['native']['fg_dice']:.4f} | test20 FG {metrics['test20']['native']['fg_dice']:.4f} "
              f"| test20 SSL4MIS-compat Dice {metrics['test20']['ssl4mis_compat']['mean']['dice']:.4f} HD95(vox) {metrics['test20']['ssl4mis_compat']['mean']['hd95']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
