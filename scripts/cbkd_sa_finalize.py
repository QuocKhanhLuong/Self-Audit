#!/usr/bin/env python3
"""Final step of the A4-SA pipeline.

Loads the VAL18-selected ``best_student`` from ``resume_A4.pt``, evaluates VAL18 and TEST50 (TEST50 exactly
once, after selection), exports the student-only artifact, cold-load verifies it and writes
``FINAL_RESULT.json``.  TEST50 is never used for any decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    out = args.out
    final = out / "FINAL_RESULT.json"
    if final.is_file():
        print("[final] FINAL_RESULT.json exists; TEST50 is evaluated once, refusing to re-evaluate", flush=True)
        return 0
    from self_audit.candidate_pg import engine, shape_atoms
    from self_audit.candidate_pg.config import validate_config
    from self_audit.candidate_pg.data import open_pg_data
    from self_audit.candidate_pg.export import export_student, load_student_export, verify_cold_load
    from self_audit.candidate_pg.student_pg import PGStudent

    cfg = validate_config(json.loads(args.config.read_text()))
    device = torch.device(args.device)
    ckpt = torch.load(out / "resume_A4.pt", map_location="cpu", weights_only=False)
    if ckpt.get("best_student") is None or int(ckpt["update"]) != int(cfg["training"]["updates"]):
        raise RuntimeError("A4 has not finished (resume_A4.pt update != training.updates); not evaluating")
    receipt = json.loads((out / "S3_RECEIPT.json").read_text())
    student = PGStudent(cfg)
    student.attach_code_branch(shape_atoms.StudentAtomBranch(shape_atoms.load_shape_atoms(receipt["artifact"])))
    student.load_state_dict(ckpt["best_student"], strict=True)
    student.to(device).eval()
    bundle = open_pg_data(cfg)
    val = [bundle.val.load(k, labeled=True) for k in bundle.val.labeled_keys()]
    test = [bundle.test.load(k, labeled=True) for k in bundle.test.labeled_keys()]
    metrics = {"VAL18": engine._evaluate(student, val, device, path="student_only"),
               "TEST50": engine._evaluate(student, test, device, path="student_only")}
    student.cpu()
    export = export_student(student, cfg, out / "exports" / "A4_SA.pt", epoch=int(ckpt["best_epoch"]),
                            metadata={"arm": "A4-SA", "selected_epoch": int(ckpt["best_epoch"]), "selection": "VAL18 student_only fg_dice"})
    deployed = load_student_export(export["path"])
    bad = [k for k in deployed.state_dict() if any(t in k for t in ("code_branch", "P_S", "D_A", "teacher", "fusions.", "mimic_heads."))]
    if bad:
        raise RuntimeError(f"export contains non-deployment keys: {bad[:5]}")
    volumes = [(f.image if f.image.ndim == 5 else f.image.unsqueeze(0), dict(f.geometry)) for f in val[:3]]
    cold = verify_cold_load(student, export["path"], volumes)
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
            return value.item()
        return value
    payload = {"schema": "self_audit.cbkd_sa_a4.final.v1", "selected_epoch": int(ckpt["best_epoch"]), "best_val_fg": float(ckpt["best_val"]),
               "metrics": clean(metrics), "export": clean(export), "cold_load": clean(cold)}
    tmp = final.with_name(".FINAL_RESULT.json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(final)
    print(f"[final] VAL18 FG {metrics['VAL18']['fg_dice']:.4f} | TEST50 FG {metrics['TEST50']['fg_dice']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
