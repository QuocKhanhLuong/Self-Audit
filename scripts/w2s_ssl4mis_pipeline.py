#!/usr/bin/env python3
"""Resumable W2S-Match pipeline on the SSL4MIS split (seed 42): CONFIG -> LINK -> W -> WFINAL -> SAT -> CONFIG2 -> LINKW -> CB -> arms.

Env: SELF_AUDIT_ROOT (required), MAX_HOURS (total wall clock), CAP_HOURS_<STEP> (projected hours of a step; a step is stopped with
``ABORTED_BUDGET`` at 1.3x its cap), S1_RUN / S1_CONFIG (the run holding the reused S1 artifact and the config it was trained under),
S1_SHA256 (expected hash prefix of student_init.pt).  Every finished step writes ``<run>/markers/<STEP>.DONE.json`` (config hash plus
artifact hashes) and is skipped on rerun; training steps resume from their last checkpoint (written every 500 updates by the engine).
``--steps`` selects steps, which is how the work is split between the 4070 server and Modal.  Test20 is evaluated once per arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cbkd_sa_pipeline import OVERRUN, run_stage, sha_file, write_json  # noqa: E402

PY = sys.executable
ALL = ["CONFIG", "LINK", "W", "WFINAL", "SAT", "CONFIG2", "LINKW", "CB", "S2MAIN", "S2MAINFINAL", "S0", "S0FINAL", "S1KD", "S1KDFINAL", "S2ARB", "S2ARBFINAL"]
ARM_OF = {"S2MAIN": ("S2main", "S2main", "A4"), "S0": ("S0", "S0", "A2"), "S1KD": ("S1kd", "S1kd", "A2"), "S2ARB": ("S2arb", "S2arb", "A2")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", default=" ".join(ALL), help="space or comma separated subset of: " + " ".join(ALL))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    args = ap.parse_args(argv)
    root = Path(os.environ["SELF_AUDIT_ROOT"])
    run = root / "runs" / "w2s-ssl4mis-20261120"
    prep = Path(os.environ.get("PREP", root / "prepared/ACDC/ssl4mis-20261120"))
    s1_run = Path(os.environ.get("S1_RUN", root / "runs/cbkd-sa-a4-20261119"))
    s1_cfg = Path(os.environ.get("S1_CONFIG", s1_run / "A4_SA_CONFIG.json"))
    s1_sha = os.environ.get("S1_SHA256", "c8195fb4a79ee665b169da36a492c99192eb33a1962c42079404107c1d2a01bf")
    cfgdir = run / "configs"
    cfg = lambda arm: cfgdir / f"ssl4mis_{arm}.json"
    dirs = {"W": run / "W", "S0": run / "S0", "S1kd": run / "S1kd", "S2arb": run / "S2arb", "S2main": run / "S2main"}
    scripts = ROOT / "scripts"
    dev = args.device
    max_hours = float(os.environ["MAX_HOURS"]) if os.environ.get("MAX_HOURS") else None
    deadline = time.time() + max_hours * 3600 if max_hours else None
    fallback = {"value": False}

    def tau_v() -> str:
        return str(json.loads((dirs["W"] / "SATURATION.json").read_text())["tau_v"])

    def make_cfgs(v: str) -> list[str]:
        return [PY, str(scripts / "w2s_make_configs_ssl4mis.py"), "--prepared", str(prep), "--out-dir", str(cfgdir), "--tau-v", v]

    def link(with_teacher: bool) -> list[str]:
        cmd = [PY, str(scripts / "w2s_stage_link_ssl4mis.py"), "--s1-run", str(s1_run), "--s1-config", str(s1_cfg), "--config", str(cfg("S2main")),
               "--dest", *(str(dirs[a]) for a in ("S0", "S1kd", "S2arb", "S2main")), "--expect-sha256", s1_sha]
        if with_teacher:
            cmd += ["--w-run", str(dirs["W"]), "--teacher-dest", *(str(dirs[a]) for a in ("S1kd", "S2arb", "S2main"))]
        return cmd

    def campaign(arm: str, out: Path, *extra: str) -> list[str]:
        return [PY, str(scripts / "pg_campaign.py"), "--device", dev, "--config", str(cfg(arm)), "--out", str(out), *extra]

    def final(arm: str, name: str, eng: str) -> list[str]:
        return [PY, str(scripts / "w2s_finalize_ssl4mis.py"), "--config", str(cfg(arm)), "--run", str(dirs[arm]), "--arm", eng, "--name", name, "--device", dev]

    # step -> (command thunk, config path for the hash, artifacts)
    spec: dict[str, tuple] = {
        "CONFIG": (lambda: make_cfgs("0.99"), None, [cfg(a) for a in ("W", "S0", "S1kd", "S2arb", "S2main")]),
        "LINK": (lambda: link(False), None, [dirs["S0"] / "S1_LINK.json"]),
        "W": (lambda: campaign("W", dirs["W"], "--stage", "S2", *(["--resume"] if (dirs["W"] / "S2/teacher_latest.pt").is_file() else [])), cfg("W"), [dirs["W"] / "S2_RECEIPT.json"]),
        "WFINAL": (lambda: [PY, str(scripts / "w2s_eval_teacher_ssl4mis.py"), "--config", str(cfg("W")), "--run", str(dirs["W"]), "--device", dev], cfg("W"), [dirs["W"] / "W_FINAL.json"]),
        "SAT": (lambda: [PY, str(scripts / "w2s_saturation.py"), "--config", str(cfg("W")), "--run", str(dirs["W"]), "--device", dev], cfg("W"), [dirs["W"] / "SATURATION.json"]),
        "CONFIG2": (lambda: make_cfgs(tau_v()), None, [cfg("S2main")]),
        "LINKW": (lambda: link(True), None, [dirs["S2main"] / "S2_RECEIPT.json"]),
        "CB": (lambda: campaign("S2main", dirs["S2main"], "--stage", "S3", "--resume"), cfg("S2main"), [dirs["S2main"] / "S3_RECEIPT.json"]),
    }
    for step, (arm, name, eng) in ARM_OF.items():
        spec[step] = ((lambda arm=arm, eng=eng: campaign(arm, dirs[arm], "--stage", "arms", "--arm", eng)), cfg(arm), [dirs[arm] / f"resume_{eng}.pt"])
        spec[step + "FINAL"] = ((lambda arm=arm, name=name, eng=eng: final(arm, name, eng)), cfg(arm), [dirs[arm] / "FINAL_RESULT.json"])

    wanted = [s for s in args.steps.replace(",", " ").split() if s]
    unknown = [s for s in wanted if s not in spec]
    if unknown:
        raise SystemExit(f"unknown steps {unknown}")
    run.mkdir(parents=True, exist_ok=True)
    for step in [s for s in ALL if s in wanted]:
        if step in ("S2MAIN", "S2MAINFINAL") and fallback["value"]:
            print(f"[pipeline] {step}: skipped (codebook fallback rule fired; S2-arb is the main arm)", flush=True)
            continue
        thunk, cfg_path, arts = spec[step]
        marker = run / "markers" / f"{step}.DONE.json"
        cfg_sha = sha_file(cfg_path) if cfg_path and cfg_path.is_file() else None
        if marker.is_file() and json.loads(marker.read_text()).get("config_sha256") == cfg_sha:
            print(f"[pipeline] {step}: DONE marker present, skipping", flush=True)
            if step == "CB" and (dirs["S2main"] / "CB_DECISION.json").is_file():
                fallback["value"] = not json.loads((dirs["S2main"] / "CB_DECISION.json").read_text())["use_codebook"]
            continue
        cap = float(os.environ[f"CAP_HOURS_{step}"]) if os.environ.get(f"CAP_HOURS_{step}") else None
        print(f"[pipeline] {step}: start {time.strftime('%F %T')} (cap {cap} h, max {max_hours} h)", flush=True)
        t0 = time.time()
        code, reason = run_stage(step, thunk(), cap, deadline, args.poll_seconds)
        if code != 0:
            status = "ABORTED_BUDGET" if reason else "FAILED"
            write_json(run / "markers" / f"{step}.{status}.json", {"step": step, "reason": reason or f"exit {code}", "exit_code": code,
                                                                  "elapsed_seconds": time.time() - t0, "config_sha256": cfg_sha})
            print(f"[pipeline] {step}: {status} ({reason or code}); re-run the same command to resume", flush=True)
            return 3 if reason else 1
        if step == "CB":
            check = subprocess.run([PY, str(scripts / "w2s_cb_check.py"), str(dirs["S2main"])])
            fallback["value"] = check.returncode == 3
            if check.returncode not in (0, 3):
                raise SystemExit("codebook check failed")
            arts = arts + [dirs["S2main"] / "CB_DECISION.json"]
        write_json(marker, {"step": step, "config_sha256": cfg_sha, "elapsed_seconds": time.time() - t0,
                            "artifact_sha256": {str(p): sha_file(p) for p in arts if p.is_file()}})
        print(f"[pipeline] {step}: DONE in {time.time() - t0:.0f}s", flush=True)
    print(f"[pipeline] finished {time.strftime('%F %T')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
