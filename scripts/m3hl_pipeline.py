#!/usr/bin/env python3
"""Resumable M3HL-teacher pipeline (campaign m3hl-teacher-20261121) on the SSL4MIS split, seed 42 for the student stages.

T-M3HL itself (official ``ACDC_M3HL_train.py``, defaults, seed 1337) is trained beforehand in ``<run>/T-M3HL`` (tmux window flowB).
Steps: M3WAIT (training reached 30000 iterations) -> M3OFFICIAL (official-path val/test metrics) -> M3CONFIG -> M3LINK -> M3EVAL (native +
SSL4MIS-compat val10/test20 of the frozen teacher through the 3D wrapper) -> M3SAT -> M3CONFIG2 -> S2ARBM3 -> S1KDM3 -> M3CB (CB-out) -> S2M3,
each arm followed by its FINAL step.  Env: SELF_AUDIT_ROOT (required), MAX_HOURS, CAP_HOURS_<STEP>, S1_RUN, S1_CONFIG, S1_SHA256 (as the
W2S pipeline).  Markers: ``<run>/markers/<STEP>.DONE.json`` (config hash + artifact hashes); re-run the same command to resume.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cbkd_sa_pipeline import run_stage, sha_file, write_json  # noqa: E402

PY = sys.executable
ALL = ["M3WAIT", "M3OFFICIAL", "M3CONFIG", "M3LINK", "M3EVAL", "M3SAT", "M3CONFIG2", "S2ARBM3", "S2ARBM3FINAL", "S1KDM3", "S1KDM3FINAL", "M3CB", "S2M3", "S2M3FINAL"]
ARMS = {"S2ARBM3": ("S2arbM3", "S2-m3hl-arb", "A2"), "S1KDM3": ("S1kdM3", "S1-kd-m3hl", "A2"), "S2M3": ("S2M3", "S2-m3hl", "A4")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", default=" ".join(ALL))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    args = ap.parse_args(argv)
    root = Path(os.environ["SELF_AUDIT_ROOT"])
    run = root / "runs" / "m3hl-teacher-20261121"
    tdir = run / "T-M3HL"
    h5 = Path(os.environ.get("M3HL_H5", root / "m3hl/data/ACDC"))
    prep = Path(os.environ.get("PREP", root / "prepared/ACDC/ssl4mis-20261120"))
    s1_run = Path(os.environ.get("S1_RUN", root / "runs/cbkd-sa-a4-20261119"))
    s1_cfg = Path(os.environ.get("S1_CONFIG", s1_run / "A4_SA_CONFIG.json"))
    s1_sha = os.environ.get("S1_SHA256", "c8195fb4a79ee665b169da36a492c99192eb33a1962c42079404107c1d2a01bf")
    ckpt = tdir / "model/M3HL/ACDC_M3HL_7_labeled/self_train/unet_best_model.pth"
    m3src = ROOT / "external/M3HL"
    cfgdir = run / "configs"
    cfg = lambda arm: cfgdir / f"ssl4mis_{arm}.json"
    dirs = {a: run / a for a in ("S2arbM3", "S1kdM3", "S2M3")}
    scripts, dev = ROOT / "scripts", args.device
    max_hours = float(os.environ["MAX_HOURS"]) if os.environ.get("MAX_HOURS") else None
    deadline = time.time() + max_hours * 3600 if max_hours else None
    fallback = {"value": False}
    # S2arbM3 finished at 100 epochs: its config file is never rewritten (it is only created if missing)
    m3_arms = ["S1kdM3", "S2M3"] + ([] if cfg("S2arbM3").is_file() else ["S2arbM3"])

    def tau_v() -> str:
        return str(json.loads((tdir / "SATURATION.json").read_text())["tau_v"])

    def make_cfgs(v: str) -> list[str]:
        return [PY, str(scripts / "w2s_make_configs_ssl4mis.py"), "--prepared", str(prep), "--out-dir", str(cfgdir), "--tau-v", v,
                "--m3hl-checkpoint", str(ckpt), "--only", *m3_arms,
                *(["--epochs", os.environ["STUDENT_EPOCHS"]] if os.environ.get("STUDENT_EPOCHS") else [])]

    def wait() -> list[str]:
        log = tdir / "train_stdout.log"
        done = log.is_file() and any(line.startswith("iteration 30000") for line in log.read_text(errors="ignore").splitlines()[-400:])
        if not done or not ckpt.is_file():
            raise SystemExit("T-M3HL has not finished 30000 iterations (or unet_best_model.pth is missing); nothing is launched")
        return [PY, "-c", "print('T-M3HL training finished')"]

    def link() -> list[str]:
        return [PY, str(scripts / "w2s_stage_link_ssl4mis.py"), "--s1-run", str(s1_run), "--s1-config", str(s1_cfg), "--config", str(cfg("S2M3")),
                "--dest", *(str(dirs[a]) for a in ("S2arbM3", "S1kdM3", "S2M3")), "--expect-sha256", s1_sha]

    spec: dict[str, tuple] = {
        "M3WAIT": (wait, None, [ckpt]),
        "M3OFFICIAL": (lambda: [PY, str(scripts / "m3hl_official_eval.py"), "--root", str(h5), "--checkpoint", str(ckpt), "--m3hl-source", str(m3src),
                                "--out", str(tdir / "T_M3HL_OFFICIAL.json"), "--device", dev], None, [tdir / "T_M3HL_OFFICIAL.json"]),
        "M3CONFIG": (lambda: make_cfgs("0.99"), None, [cfg(a) for a in m3_arms]),
        "M3LINK": (link, None, [dirs["S2M3"] / "S1_LINK.json"]),
        "M3EVAL": (lambda: [PY, str(scripts / "w2s_eval_teacher_ssl4mis.py"), "--config", str(cfg("S2arbM3")), "--run", str(tdir), "--device", dev],
                   cfg("S2arbM3"), [tdir / "W_FINAL.json"]),
        "M3SAT": (lambda: [PY, str(scripts / "w2s_saturation.py"), "--config", str(cfg("S2arbM3")), "--run", str(tdir), "--device", dev],
                  cfg("S2arbM3"), [tdir / "SATURATION.json"]),
        "M3CONFIG2": (lambda: make_cfgs(tau_v()), None, [cfg(a) for a in m3_arms]),
        "M3CB": (lambda: [PY, str(scripts / "pg_campaign.py"), "--device", dev, "--config", str(cfg("S2M3")), "--out", str(dirs["S2M3"]), "--stage", "S3", "--resume"],
                 cfg("S2M3"), [dirs["S2M3"] / "S3_RECEIPT.json"]),
    }
    for step, (arm, name, eng) in ARMS.items():
        spec[step] = ((lambda arm=arm, eng=eng: [PY, str(scripts / "pg_campaign.py"), "--device", dev, "--config", str(cfg(arm)), "--out", str(dirs[arm]), "--stage", "arms", "--arm", eng]),
                      cfg(arm), [dirs[arm] / f"resume_{eng}.pt"])
        spec[step + "FINAL"] = ((lambda arm=arm, name=name, eng=eng: [PY, str(scripts / "w2s_finalize_ssl4mis.py"), "--config", str(cfg(arm)), "--run", str(dirs[arm]),
                                                                       "--arm", eng, "--name", name, "--device", dev]), cfg(arm), [dirs[arm] / "FINAL_RESULT.json"])
    wanted = [s for s in args.steps.replace(",", " ").split() if s]
    unknown = [s for s in wanted if s not in spec]
    if unknown:
        raise SystemExit(f"unknown steps {unknown}")
    run.mkdir(parents=True, exist_ok=True)
    for step in [s for s in ALL if s in wanted]:
        if step in ("S2M3", "S2M3FINAL") and fallback["value"]:
            print(f"[m3hl] {step}: skipped (CB-out fallback rule fired: q_T FG Dice > 0.05 below raw M3HL, or perplexity < 10)", flush=True)
            continue
        thunk, cfg_path, arts = spec[step]
        marker = run / "markers" / f"{step}.DONE.json"
        cfg_sha = sha_file(cfg_path) if cfg_path and cfg_path.is_file() else None
        if marker.is_file() and json.loads(marker.read_text()).get("config_sha256") == cfg_sha:
            print(f"[m3hl] {step}: DONE marker present, skipping", flush=True)
            if step == "M3CB" and (dirs["S2M3"] / "CB_DECISION.json").is_file():
                fallback["value"] = not json.loads((dirs["S2M3"] / "CB_DECISION.json").read_text())["use_codebook"]
            continue
        cap = float(os.environ[f"CAP_HOURS_{step}"]) if os.environ.get(f"CAP_HOURS_{step}") else None
        print(f"[m3hl] {step}: start {time.strftime('%F %T')} (cap {cap} h, max {max_hours} h)", flush=True)
        t0 = time.time()
        code, reason = run_stage(step, thunk(), cap, deadline, args.poll_seconds)
        if code != 0:
            status = "ABORTED_BUDGET" if reason else "FAILED"
            write_json(run / "markers" / f"{step}.{status}.json", {"step": step, "reason": reason or f"exit {code}", "exit_code": code,
                                                                  "elapsed_seconds": time.time() - t0, "config_sha256": cfg_sha})
            print(f"[m3hl] {step}: {status} ({reason or code}); re-run the same command to resume", flush=True)
            return 3 if reason else 1
        if step == "M3CB":
            check = subprocess.run([PY, str(scripts / "w2s_cb_check.py"), str(dirs["S2M3"])])
            fallback["value"] = check.returncode == 3
            if check.returncode not in (0, 3):
                raise SystemExit("codebook check failed")
            arts = arts + [dirs["S2M3"] / "CB_DECISION.json"]
        write_json(marker, {"step": step, "config_sha256": cfg_sha, "elapsed_seconds": time.time() - t0,
                            "artifact_sha256": {str(p): sha_file(p) for p in arts if p.is_file()}})
        print(f"[m3hl] {step}: DONE in {time.time() - t0:.0f}s", flush=True)
    print(f"[m3hl] finished {time.strftime('%F %T')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
