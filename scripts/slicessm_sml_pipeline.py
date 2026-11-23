#!/usr/bin/env python3
"""SliceSSM width ablation (S/M/L), resumable.  S = the finished S0-slicessm (widths 24,48,96,160,192); M and L widen every stage (x1.5, x2).

One size per invocation (``--size M`` or ``--size L``; run both in parallel).  Steps: SMCONFIG -> S1SM -> SMLINK -> S0SM -> S0SMFINAL -> T20SM.
Everything but the widths (and the lifted 3 M parameter cap) is the S0-slicessm recipe: UniMatch without teacher, 150 epochs, seed 42, val10 selection, the same
shift-recovery auxiliary.  A wider student cannot load the S1-SS artifact, so S1 (CineMA distillation, 10,000 updates, same ED/ES data of the TRAIN patients) is
re-run per size.  FINAL is val10 only; test20 is opened once afterwards (``w2s_open_test20.py``) for the report, not for any decision.
Env: SELF_AUDIT_ROOT (required), MAX_HOURS, CAP_HOURS_<STEP>.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cbkd_sa_pipeline import run_stage, sha_file, write_json  # noqa: E402

PY = sys.executable
ALL = ["SMCONFIG", "S1SM", "SMLINK", "S0SM", "S0SMFINAL", "T20SM"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=("M", "L"), required=True)
    ap.add_argument("--steps", default=" ".join(ALL))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    args = ap.parse_args(argv)
    arm = f"S0slicessm{args.size}"
    name = f"S0-slicessm-{args.size}"
    root = Path(os.environ["SELF_AUDIT_ROOT"])
    run = root / "runs" / "slicessm-sml-20261123"
    prep = Path(os.environ.get("PREP", root / "prepared/ACDC/ssl4mis-20261120"))
    cfgdir = run / "configs"
    cfg_path = cfgdir / f"ssl4mis_{arm}.json"
    arm_dir, s1_dir = run / arm, run / f"S1-SS-{args.size}"
    scripts, dev = ROOT / "scripts", args.device
    max_hours = float(os.environ["MAX_HOURS"]) if os.environ.get("MAX_HOURS") else None
    deadline = time.time() + max_hours * 3600 if max_hours else None

    def link() -> list[str]:
        arm_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s1_dir / "S1_RECEIPT.json", arm_dir / "S1_RECEIPT.json")
        return [PY, "-c", f"print('S1-SS-{args.size} receipt linked into {arm}')"]

    spec = {
        "SMCONFIG": (lambda: [PY, str(scripts / "w2s_make_configs_ssl4mis.py"), "--prepared", str(prep), "--out-dir", str(cfgdir), "--tau-v", "0.99", "--only", arm, "--epochs", "150"],
                     None, [cfg_path]),
        "S1SM": (lambda: [PY, str(scripts / "pg_campaign.py"), "--device", dev, "--config", str(cfg_path), "--out", str(s1_dir), "--stage", "S1", "--resume"], cfg_path, [s1_dir / "S1_RECEIPT.json"]),
        "SMLINK": (link, None, [arm_dir / "S1_RECEIPT.json"]),
        "S0SM": (lambda: [PY, str(scripts / "pg_campaign.py"), "--device", dev, "--config", str(cfg_path), "--out", str(arm_dir), "--stage", "arms", "--arm", "A2"], cfg_path, [arm_dir / "resume_A2.pt"]),
        "S0SMFINAL": (lambda: [PY, str(scripts / "w2s_finalize_ssl4mis.py"), "--config", str(cfg_path), "--run", str(arm_dir), "--arm", "A2", "--name", name, "--device", dev, "--val-only"],
                      cfg_path, [arm_dir / "FINAL_RESULT.json"]),
        "T20SM": (lambda: [PY, str(scripts / "w2s_open_test20.py"), "--config", str(cfg_path), "--run", str(arm_dir), "--arm", "A2", "--name", name, "--device", dev],
                  cfg_path, [arm_dir / "TEST20_RESULT.json"]),
    }
    wanted = [s for s in args.steps.replace(",", " ").split() if s]
    unknown = [s for s in wanted if s not in spec]
    if unknown:
        raise SystemExit(f"unknown steps {unknown}")
    (run / "markers").mkdir(parents=True, exist_ok=True)
    for step in [s for s in ALL if s in wanted]:
        thunk, cpath, arts = spec[step]
        marker = run / "markers" / f"{args.size}.{step}.DONE.json"
        cfg_sha = sha_file(cpath) if cpath and cpath.is_file() else None
        if marker.is_file() and json.loads(marker.read_text()).get("config_sha256") == cfg_sha:
            print(f"[sml-{args.size}] {step}: DONE marker present, skipping", flush=True)
            continue
        cap = float(os.environ[f"CAP_HOURS_{step}"]) if os.environ.get(f"CAP_HOURS_{step}") else None
        print(f"[sml-{args.size}] {step}: start {time.strftime('%F %T')} (cap {cap} h, max {max_hours} h)", flush=True)
        t0 = time.time()
        code, reason = run_stage(step, thunk(), cap, deadline, args.poll_seconds)
        if code != 0:
            status = "ABORTED_BUDGET" if reason else "FAILED"
            write_json(run / "markers" / f"{args.size}.{step}.{status}.json", {"step": step, "reason": reason or f"exit {code}", "exit_code": code, "elapsed_seconds": time.time() - t0})
            print(f"[sml-{args.size}] {step}: {status} ({reason or code}); re-run the same command to resume", flush=True)
            return 3 if reason else 1
        write_json(marker, {"step": step, "config_sha256": cfg_sha, "elapsed_seconds": time.time() - t0, "artifact_sha256": {str(p): sha_file(p) for p in arts if p.is_file()}})
        print(f"[sml-{args.size}] {step}: DONE in {time.time() - t0:.0f}s", flush=True)
    print(f"[sml-{args.size}] finished {time.strftime('%F %T')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
