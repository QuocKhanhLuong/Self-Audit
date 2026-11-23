#!/usr/bin/env python3
"""SliceSSMStudent screening pipeline (resumable), modelled on aniso_pipeline.py: S1-SS distillation, then the S0-slicessm
UniMatch arm (no teacher, compare with S0-aniso / S0-unetrpp-150) and the S1kd-M3HL-slicessm arm (compare with S1-kd M3HL),
both 150 epochs, val10 only (test20 is never read).

Steps: SSCONFIG -> S1SS -> SSLINK -> S0SS -> S0SSFINAL -> S1KDSS -> S1KDSSFINAL.  M3HL_CHECKPOINT (env) is required for SSCONFIG.
The original AnisoStudent notes follow.
* S1-ANISO distils CineMA into AnisoStudent with the existing S1 code (taps cinema_conv_1_8 -> E3 and cinema_tokens_1_16 -> E4; every encoder
  stage below is trained through them) with the same 10,000-update budget, but on the ED/ES frames of the SSL4MIS TRAIN set only (L7 + U63 images,
  140 volumes): the SSL4MIS data object holds no other cine frames.  The existing S1 used 1,933 cine frames of 80 patients (including the new val/test patients).
* S0-unetrpp-150 uses the existing S1 artifact; S0-aniso uses S1-ANISO.  Both 150 epochs, seed 42, UniMatch without teacher, val10 selection.
  FINAL runs with --val-only: test20 is never read.
Env: SELF_AUDIT_ROOT (required), MAX_HOURS, CAP_HOURS_<STEP>, S1_RUN / S1_CONFIG / S1_SHA256 as the other pipelines.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cbkd_sa_pipeline import run_stage, sha_file, write_json  # noqa: E402

PY = sys.executable
ALL = ["SSCONFIG", "S1SS", "SSLINK", "S0SS", "S0SSFINAL", "S1KDSS", "S1KDSSFINAL"]
ARMS = {"S0SS": ("S0slicessm", "S0-slicessm"), "S1KDSS": ("S1kdM3ss", "S1kd-M3HL-slicessm")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", default=" ".join(ALL))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    args = ap.parse_args(argv)
    root = Path(os.environ["SELF_AUDIT_ROOT"])
    # SS_RUN / SS_DATASET / SS_LOCK reuse the pipeline for M&Ms-1 (dataset mnms1, lock = <prepared>/SPLIT_LOCK.json)
    run = root / "runs" / os.environ.get("SS_RUN", "slicessm-screen-20261122")
    dataset = os.environ.get("SS_DATASET", "acdc")
    prep = Path(os.environ.get("PREP", root / "prepared/ACDC/ssl4mis-20261120"))
    cfgdir = run / "configs"
    cfg = lambda arm: cfgdir / f"ssl4mis_{arm}.json"
    dirs = {"S0slicessm": run / "S0slicessm", "S1kdM3ss": run / "S1kdM3ss", "S1": run / "S1-SS"}
    scripts, dev = ROOT / "scripts", args.device
    max_hours = float(os.environ["MAX_HOURS"]) if os.environ.get("MAX_HOURS") else None
    deadline = time.time() + max_hours * 3600 if max_hours else None

    def link_s1() -> list[str]:
        for arm in os.environ.get("SS_ARMS", "S0slicessm S1kdM3ss").split():
            dirs[arm].mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dirs["S1"] / "S1_RECEIPT.json", dirs[arm] / "S1_RECEIPT.json")
        return [PY, "-c", "print('S1-SS receipt linked into both SliceSSM arms')"]

    def make_configs() -> list[str]:
        ckpt = os.environ.get("M3HL_CHECKPOINT")
        if not ckpt:
            raise SystemExit("M3HL_CHECKPOINT is required (the S1kd-M3HL-slicessm arm uses the frozen M3HL teacher)")
        arms = os.environ.get("SS_ARMS", "S0slicessm S1kdM3ss").split()
        extra = ["--dataset", dataset] + (["--lock", os.environ["SS_LOCK"]] if os.environ.get("SS_LOCK") else [])
        return [PY, str(scripts / "w2s_make_configs_ssl4mis.py"), "--prepared", str(prep), "--out-dir", str(cfgdir), "--tau-v", "0.99",
                "--only", *arms, "--epochs", "150", "--m3hl-checkpoint", ckpt, *extra]

    spec: dict[str, tuple] = {
        "SSCONFIG": (make_configs, None, [cfg(a) for a in os.environ.get("SS_ARMS", "S0slicessm S1kdM3ss").split()]),
        # S1 (CineMA distillation) is shared: the student architecture is identical in both arms
        "S1SS": (lambda: [PY, str(scripts / "pg_campaign.py"), "--device", dev, "--config", str(cfg("S0slicessm")), "--out", str(dirs["S1"]), "--stage", "S1", "--resume"],
                 cfg("S0slicessm"), [dirs["S1"] / "S1_RECEIPT.json"]),
        "SSLINK": (link_s1, None, [dirs[a] / "S1_RECEIPT.json" for a in os.environ.get("SS_ARMS", "S0slicessm S1kdM3ss").split()]),
    }
    for step, (arm, name) in ARMS.items():
        spec[step] = ((lambda arm=arm: [PY, str(scripts / "pg_campaign.py"), "--device", dev, "--config", str(cfg(arm)), "--out", str(dirs[arm]), "--stage", "arms", "--arm", "A2"]),
                      cfg(arm), [dirs[arm] / "resume_A2.pt"])
        spec[step + "FINAL"] = ((lambda arm=arm, name=name: [PY, str(scripts / "w2s_finalize_ssl4mis.py"), "--config", str(cfg(arm)), "--run", str(dirs[arm]), "--arm", "A2",
                                                              "--name", name, "--device", dev, "--val-only"]), cfg(arm), [dirs[arm] / "FINAL_RESULT.json"])
    wanted = [s for s in args.steps.replace(",", " ").split() if s]
    unknown = [s for s in wanted if s not in spec]
    if unknown:
        raise SystemExit(f"unknown steps {unknown}")
    run.mkdir(parents=True, exist_ok=True)
    for step in [s for s in ALL if s in wanted]:
        thunk, cfg_path, arts = spec[step]
        marker = run / "markers" / f"{step}.DONE.json"
        cfg_sha = sha_file(cfg_path) if cfg_path and cfg_path.is_file() else None
        if marker.is_file() and json.loads(marker.read_text()).get("config_sha256") == cfg_sha:
            print(f"[slicessm] {step}: DONE marker present, skipping", flush=True)
            continue
        cap = float(os.environ[f"CAP_HOURS_{step}"]) if os.environ.get(f"CAP_HOURS_{step}") else None
        print(f"[slicessm] {step}: start {time.strftime('%F %T')} (cap {cap} h, max {max_hours} h)", flush=True)
        t0 = time.time()
        code, reason = run_stage(step, thunk(), cap, deadline, args.poll_seconds)
        if code != 0:
            status = "ABORTED_BUDGET" if reason else "FAILED"
            write_json(run / "markers" / f"{step}.{status}.json", {"step": step, "reason": reason or f"exit {code}", "exit_code": code,
                                                                  "elapsed_seconds": time.time() - t0, "config_sha256": cfg_sha})
            print(f"[slicessm] {step}: {status} ({reason or code}); re-run the same command to resume", flush=True)
            return 3 if reason else 1
        write_json(marker, {"step": step, "config_sha256": cfg_sha, "elapsed_seconds": time.time() - t0,
                            "artifact_sha256": {str(p): sha_file(p) for p in arts if p.is_file()}})
        print(f"[slicessm] {step}: DONE in {time.time() - t0:.0f}s", flush=True)
    print(f"[slicessm] finished {time.strftime('%F %T')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
