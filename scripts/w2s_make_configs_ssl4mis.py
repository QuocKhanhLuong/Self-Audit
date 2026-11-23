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
EXTRA_ARMS = ("A4SA", "S1kdM3", "S2arbM3", "S2M3", "S2arbM3E150", "S0unet150", "S0aniso", "S0slicessm", "S1kdM3ss", "S0slicessmM", "S0slicessmL")  # S0slicessmM/L: width ablation (S = S0slicessm); S0slicessm/S1kdM3ss: SliceSSMStudent arms (no teacher / M3HL KD); S0unet150/S0aniso: architecture screening arms (UniMatch, no teacher); S2arbM3E150: S2-m3hl-arb rerun at 150 epochs (its own run dir; the 100-epoch S2arbM3 is untouched)
ENGINE_ARM = {"S0": "A2", "S1kd": "A2", "S2arb": "A2", "S2main": "A4", "A4SA": "A4", "S1kdM3": "A2", "S2arbM3": "A2", "S2M3": "A4", "S2arbM3E150": "A2", "S0unet150": "A2", "S0aniso": "A2", "S0slicessm": "A2", "S1kdM3ss": "A2", "S0slicessmM": "A2", "S0slicessmL": "A2"}


ANISO_CINEMA_REVISION = "b1251ee50423bceeca84c080782fc3bc7756dea6"  # the revision the existing S1 was distilled from


# S = the existing S0slicessm widths (24, 48, 96, 160, 192; 2.53 M); M = x1.5 and L = x2, rounded to multiples of 8
SLICESSM_WIDTHS = {"S0slicessmM": (40, 72, 144, 240, 288), "S0slicessmL": (48, 96, 192, 320, 384)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(arm: str, prepared: Path, lock: Path, tau_v: float, m3hl_checkpoint: Path | None = None, epochs: int | None = None) -> dict:
    from self_audit.candidate_pg.config import validate_config
    from self_audit.candidate_pg.w2s_config import make_config

    base_arm = {"S1kd": "S0", "S2main": "S2", "A4SA": "W", "S1kdM3": "S0", "S2arbM3": "S2arb", "S2M3": "S2", "S2arbM3E150": "S2arb", "S0unet150": "S0", "S0aniso": "S0", "S0slicessm": "S0", "S1kdM3ss": "S0", "S0slicessmM": "S0", "S0slicessmL": "S0"}.get(arm, arm)
    cfg = make_config(base_arm, tau_v=tau_v)
    if arm == "A4SA":
        # A4-SA (PG fusion target + CBKD-SA) exactly as scripts/cbkd_sa_make_config.py: only the split / selection fields below
        # differ from the old A4-SA config; the teacher threshold is the fixed 0.95 of the W2S teacher (clip [0.95, 0.95]).
        cfg["pg"]["enabled"] = True
        cfg["cbkd"].update({"enabled": True, "variant": "cbkd_sa", "warmup_updates_labeled_only": 1200, "ramp_updates": 400})
    if arm == "S0aniso":
        # AnisoStudent: same recipe as S0, only the architecture differs; the CineMA revision is pinned (the S1-ANISO stage downloads it)
        cfg["student"]["arch"] = "aniso"
        cfg["cinema_init"]["hf_revision"] = ANISO_CINEMA_REVISION
    if arm in ("S0slicessm", "S1kdM3ss", "S0slicessmM", "S0slicessmL"):
        # SliceSSMStudent (B1 physical-step z-SSM + B2 aligned slice fusion, <= 3 M parameters): same recipe as S0 / S1kdM3,
        # only the architecture and the shift-recovery auxiliary differ; CineMA revision pinned as for the aniso arm
        cfg["student"]["arch"] = "slicessm"
        cfg["student"]["slicessm"] = {"spacing_mode": "physical", "spacing_ref_mm": 10.0, "align": True, "ssm_blocks": [1, 2], "state": 8,
                                      "shift_aux": {"weight": 0.1, "max_px": 8, "start_update": 0}}
        cfg["cinema_init"]["hf_revision"] = ANISO_CINEMA_REVISION
        if arm in SLICESSM_WIDTHS:  # width ablation: only the encoder/decoder widths change (and the 3 M parameter cap is lifted)
            cfg["student"]["slicessm"]["channels"] = list(SLICESSM_WIDTHS[arm])
            cfg["student"]["slicessm"]["parameter_cap"] = None
    if arm in ("S1kd", "S1kdM3", "S1kdM3ss"):
        cfg["pg"]["target_mode"] = "teacher"
    if arm.endswith("M3") or arm in ("S2arbM3E150", "S1kdM3ss"):
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
