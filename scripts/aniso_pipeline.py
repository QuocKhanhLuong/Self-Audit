#!/usr/bin/env python3
"""AnisoStudent screening pipeline (resumable): S1-ANISO distillation, then two equal-schedule UniMatch arms, val10 only.

Steps: ANCONFIG -> ANLINK -> S1ANISO -> ANLINK2 -> S0UNET150 -> S0UNET150FINAL -> S0ANISO -> S0ANISOFINAL -> ANREPORT.
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
ALL = ["ANCONFIG", "ANLINK", "S1ANISO", "ANLINK2", "S0UNET150", "S0UNET150FINAL", "S0ANISO", "S0ANISOFINAL", "ANREPORT"]
ARMS = {"S0UNET150": ("S0unet150", "S0-unetrpp-150"), "S0ANISO": ("S0aniso", "S0-aniso")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", default=" ".join(ALL))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    args = ap.parse_args(argv)
    root = Path(os.environ["SELF_AUDIT_ROOT"])
    run = root / "runs" / "aniso-screen-20261122"
    w2s = root / "runs" / "w2s-ssl4mis-20261120"
    prep = Path(os.environ.get("PREP", root / "prepared/ACDC/ssl4mis-20261120"))
    cfgdir = run / "configs"
    cfg = lambda arm: cfgdir / f"ssl4mis_{arm}.json"
    dirs = {"S0unet150": run / "S0unet150", "S0aniso": run / "S0aniso", "S1": run / "S1-ANISO"}
    scripts, dev = ROOT / "scripts", args.device
    max_hours = float(os.environ["MAX_HOURS"]) if os.environ.get("MAX_HOURS") else None
    deadline = time.time() + max_hours * 3600 if max_hours else None

    def link_s1_existing() -> list[str]:
        def go() -> None:
            dirs["S0unet150"].mkdir(parents=True, exist_ok=True)
            for name in ("S1_RECEIPT.json", "S1_LINK.json"):  # the existing S1 artifact, same receipts as the finished S0 arm
                shutil.copyfile(w2s / "S0" / name, dirs["S0unet150"] / name)
        go()
        return [PY, "-c", "print('existing S1 receipts linked into S0unet150')"]

    def link_s1_aniso() -> list[str]:
        dirs["S0aniso"].mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dirs["S1"] / "S1_RECEIPT.json", dirs["S0aniso"] / "S1_RECEIPT.json")
        return [PY, "-c", "print('S1-ANISO receipt linked into S0aniso')"]

    spec: dict[str, tuple] = {
        "ANCONFIG": (lambda: [PY, str(scripts / "w2s_make_configs_ssl4mis.py"), "--prepared", str(prep), "--out-dir", str(cfgdir), "--tau-v", "0.99",
                              "--only", "S0unet150", "S0aniso", "--epochs", "150"], None, [cfg("S0unet150"), cfg("S0aniso")]),
        "ANLINK": (link_s1_existing, None, [dirs["S0unet150"] / "S1_RECEIPT.json"]),
        "S1ANISO": (lambda: [PY, str(scripts / "pg_campaign.py"), "--device", dev, "--config", str(cfg("S0aniso")), "--out", str(dirs["S1"]), "--stage", "S1", "--resume"],
                    cfg("S0aniso"), [dirs["S1"] / "S1_RECEIPT.json"]),
        "ANLINK2": (link_s1_aniso, None, [dirs["S0aniso"] / "S1_RECEIPT.json"]),
        "ANREPORT": (lambda: [PY, str(scripts / "aniso_screen_report.py"), "--unet", str(dirs["S0unet150"]), "--aniso", str(dirs["S0aniso"]),
                              "--out", str(ROOT / "reports/m3hl-teacher-20261121/ANISO_SCREEN.md")], None, [ROOT / "reports/m3hl-teacher-20261121/ANISO_SCREEN.md"]),
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
            print(f"[aniso] {step}: DONE marker present, skipping", flush=True)
            continue
        cap = float(os.environ[f"CAP_HOURS_{step}"]) if os.environ.get(f"CAP_HOURS_{step}") else None
        print(f"[aniso] {step}: start {time.strftime('%F %T')} (cap {cap} h, max {max_hours} h)", flush=True)
        t0 = time.time()
        code, reason = run_stage(step, thunk(), cap, deadline, args.poll_seconds)
        if code != 0:
            status = "ABORTED_BUDGET" if reason else "FAILED"
            write_json(run / "markers" / f"{step}.{status}.json", {"step": step, "reason": reason or f"exit {code}", "exit_code": code,
                                                                  "elapsed_seconds": time.time() - t0, "config_sha256": cfg_sha})
            print(f"[aniso] {step}: {status} ({reason or code}); re-run the same command to resume", flush=True)
            return 3 if reason else 1
        write_json(marker, {"step": step, "config_sha256": cfg_sha, "elapsed_seconds": time.time() - t0,
                            "artifact_sha256": {str(p): sha_file(p) for p in arts if p.is_file()}})
        print(f"[aniso] {step}: DONE in {time.time() - t0:.0f}s", flush=True)
    print(f"[aniso] finished {time.strftime('%F %T')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
