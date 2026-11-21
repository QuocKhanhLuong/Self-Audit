#!/usr/bin/env python3
"""Write the resolved SSL4MIS-split W2S configs: W (teacher) and the student arms S0, S1kd, S2arb, S2main.

Starts from ``w2s_config.make_config`` (PG base + reviewed W2S fields) and only adds the run bindings, identical for every arm:
  * data.manifests: the receipt-pinned SSL4MIS materialization (train L7+U63, val10, test20); data.ssl4mis = {lock, receipt}
  * data.k_labeled 7, ED/ES-only U, forbid VAL10/TEST20 in training; selection and monitoring on val10 (student-only)
  * training.eval_splits = [train_labeled, VAL10]; training.eval_paths = [student_only]  (test20 is evaluated once, by the finalizers)
Student arms differ only in ``pg.target_mode`` and ``cbkd.*``; ``S1kd`` is UniMatch with teacher pseudo-labels (no arbiter, no codebook).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARMS = ("W", "S0", "S1kd", "S2arb", "S2main")
# written only on request (--only), so the configs of the running arms stay byte-identical; M3*: M3HL teacher (campaign m3hl-teacher-20261121)
EXTRA_ARMS = ("A4SA", "S1kdM3", "S2arbM3", "S2M3")
ENGINE_ARM = {"S0": "A2", "S1kd": "A2", "S2arb": "A2", "S2main": "A4", "A4SA": "A4", "S1kdM3": "A2", "S2arbM3": "A2", "S2M3": "A4"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(arm: str, prepared: Path, lock: Path, tau_v: float, m3hl_checkpoint: Path | None = None, epochs: int | None = None) -> dict:
    from self_audit.candidate_pg.config import validate_config
    from self_audit.candidate_pg.w2s_config import make_config

    base_arm = {"S1kd": "S0", "S2main": "S2", "A4SA": "W", "S1kdM3": "S0", "S2arbM3": "S2arb", "S2M3": "S2"}.get(arm, arm)
    cfg = make_config(base_arm, tau_v=tau_v)
    if arm == "A4SA":
        # A4-SA (PG fusion target + CBKD-SA) exactly as scripts/cbkd_sa_make_config.py: only the split / selection fields below
        # differ from the old A4-SA config; the teacher threshold is the fixed 0.95 of the W2S teacher (clip [0.95, 0.95]).
        cfg["pg"]["enabled"] = True
        cfg["cbkd"].update({"enabled": True, "variant": "cbkd_sa", "warmup_updates_labeled_only": 1200, "ramp_updates": 400})
    if arm in ("S1kd", "S1kdM3"):
        cfg["pg"]["target_mode"] = "teacher"
    if arm.endswith("M3"):
        # M3HL teacher: identical to the MedNeXt-teacher arm except pg.teacher_source/pg.m3hl and (S2M3) cbkd.variant
        if m3hl_checkpoint is None:
            raise SystemExit("M3HL arms need --m3hl-checkpoint")
        cfg["pg"]["teacher_source"] = "m3hl"
        cfg["pg"]["m3hl"] = {"checkpoint": {"path": str(m3hl_checkpoint), "sha256": sha(m3hl_checkpoint)}, "source_root": "external/M3HL", "source_commit": "19151b3"}
        if arm == "S2M3":
            cfg["cbkd"]["variant"] = "cbkd_sa_out"
    if epochs is not None and arm != "W":
        # student epoch budget: total updates follow updates_per_epoch (fraction-defined schedules scale; absolute onsets such as the
        # labeled-only warm-up at 800 and the CBKD onset at 1200 are absolute update counts and are NOT rescaled)
        cfg["training"]["epochs"] = int(epochs)
        cfg["training"]["updates"] = int(epochs) * int(cfg["training"]["updates_per_epoch"])
    receipt_path = prepared / "PREPARATION_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    for role in ("train", "val", "test"):
        pinned = receipt["manifests"][role]
        assert pinned["sha256"] == sha(Path(pinned["path"])), f"{role} manifest differs from its receipt pin"
        cfg["data"]["manifests"][role] = dict(pinned)
    assert receipt["lock"]["sha256"] == sha(lock)
    cfg["data"]["ssl4mis"] = {"lock": {"path": str(lock), "sha256": sha(lock)}, "receipt": {"path": str(receipt_path), "sha256": sha(receipt_path)}}
    cfg["data"]["k_labeled"] = 7
    cfg["data"]["unlabeled"] = "ed_es_frames_of_remaining_train_patients"
    cfg["data"]["forbid_splits_in_training"] = ["VAL10", "TEST20"]
    cfg["training"]["selection"] = {"split": "VAL10", "path": "student_only", "metric": "fg_dice_patient_macro"}
    cfg["training"]["eval_splits"] = ["train_labeled", "VAL10"]
    cfg["training"]["eval_paths"] = ["student_only"]
    cfg["teacher"]["select"] = "VAL10 FG of teacher region head"
    return validate_config(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--lock", type=Path, default=ROOT / "configs/candidate_pg/ssl4mis_split_lock.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tau-v", type=float, default=0.99, help="0.99, or 0.999 only if the saturation check says so")
    ap.add_argument("--m3hl-checkpoint", type=Path, help="unet_best_model.pth of T-M3HL (needed for the M3 arms)")
    ap.add_argument("--epochs", type=int, help="student epoch budget (default: the config default of 100 epochs = 8000 updates)")
    ap.add_argument("--only", nargs="*", help="write only these arms (default: the five standard arms)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for arm in (a.only or ARMS):
        path = a.out_dir / f"ssl4mis_{arm}.json"
        path.write_text(json.dumps(build(arm, a.prepared.resolve(), a.lock.resolve(), a.tau_v, a.m3hl_checkpoint, a.epochs), indent=2, sort_keys=True) + "\n")
        print(f"wrote {path} sha256 {sha(path)[:16]}")


if __name__ == "__main__":
    main()
