#!/usr/bin/env python3
"""Final step of a W2S student arm: TEST50 exactly once on the VAL20-selected checkpoint, export, cold-load check."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w2s_common import DUPLICATE_OF_VAL, ROOT, load_config, subset_mean, write_once  # noqa: E402


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arm", choices=("A2", "A4"), required=True, help="engine arm: A2 = S0 / S2-arb, A4 = S2 (codebook)")
    ap.add_argument("--name", required=True, help="W2S arm name for the export (S0, S2, S2arb)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    final = args.run / "FINAL_RESULT.json"
    if final.is_file():
        print("[final] FINAL_RESULT.json exists; TEST50 is evaluated once, refusing to re-evaluate", flush=True)
        return 0
    from self_audit.candidate_pg import engine, shape_atoms
    from self_audit.candidate_pg.export import export_student, load_student_export, verify_cold_load
    from self_audit.candidate_pg.student_pg import PGStudent

    cfg, device = load_config(args.config), torch.device(args.device)
    ckpt = torch.load(args.run / f"resume_{args.arm}.pt", map_location="cpu", weights_only=False)
    if ckpt.get("best_student") is None or int(ckpt["update"]) != int(cfg["training"]["updates"]):
        raise RuntimeError("arm has not finished (resume update != training.updates); not evaluating")
    student = PGStudent(cfg)
    if args.arm == "A4":
        receipt = engine._read_receipt(args.run / "S3_RECEIPT.json")
        student.attach_code_branch(shape_atoms.StudentAtomBranch(shape_atoms.load_shape_atoms(receipt["artifact"])))
    student.load_state_dict(ckpt["best_student"], strict=True)
    student.to(device).eval()
    bundle = engine._load_data(cfg, synthetic=False)["objects"]
    val = [bundle.val.load(k, labeled=True) for k in bundle.val.labeled_keys()]
    test = [bundle.test.load(k, labeled=True) for k in bundle.test.labeled_keys()]
    metrics = {"VAL20": engine._evaluate(student, val, device, path="student_only"),
               "TEST50": engine._evaluate(student, test, device, path="student_only")}
    val18 = set(bundle.val.patients) - set(bundle.val.extra_patients)
    derived = {"VAL18_from_VAL20_patients": subset_mean(metrics["VAL20"]["per_patient"], val18),
               "TEST49_without_patient118": subset_mean(metrics["TEST50"]["per_patient"], set(metrics["TEST50"]["per_patient"]) - {DUPLICATE_OF_VAL})}
    student.cpu()
    export = export_student(student, cfg, args.run / "exports" / f"{args.name}.pt", epoch=int(ckpt["best_epoch"]),
                            metadata={"arm": args.name, "selected_epoch": int(ckpt["best_epoch"]), "selection": "VAL20 student-only FG Dice",
                                      "test_policy": "TEST50 once after selection"})
    deployed = load_student_export(export["path"])
    bad = [k for k in deployed.state_dict() if any(t in k for t in ("code_branch", "P_S", "D_A", "teacher", "fusions."))]
    if bad:
        raise RuntimeError(f"export contains non-deployment keys: {bad[:5]}")
    volumes = [(f.image if f.image.ndim == 5 else f.image.unsqueeze(0), dict(f.geometry)) for f in val[:3]]
    cold = verify_cold_load(student, export["path"], volumes)
    write_once(final, {"schema": "self_audit.w2s.final.v1", "arm": args.name, "selected_epoch": int(ckpt["best_epoch"]),
                       "best_val_fg": float(ckpt["best_val"]), "metrics": metrics, "derived": derived, "export": export, "cold_load": cold})
    print(f"[final] {args.name} VAL20 FG {metrics['VAL20']['fg_dice']:.4f} | TEST50 FG {metrics['TEST50']['fg_dice']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
