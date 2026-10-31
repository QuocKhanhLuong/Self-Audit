#!/usr/bin/env python3
"""Bounded four-arm smoke and exact replay, on manufactured data only."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from self_audit.compact_artifacts import compact_training_source_sha256, digest


def exact_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact_equal(x, y) for x, y in zip(a, b))
    return a == b


def replay_rng(state):
    """Exclude only observed durations, never sampler state or coverage counters."""
    result = copy.deepcopy(state)
    probe = result.get("probe_rng_state")
    if isinstance(probe, dict):
        probe.pop("total_sampling_seconds", None)
        probe.pop("total_cache_seconds", None)
    return result


def verify(out, device):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = compact_training_source_sha256()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    commands = []
    receipt = {"status": "RUNNING", "device": device, "synthetic": True,
               "source_sha256": source, "commands": commands,
               "excluded_observational_fields": ["probe_rng_state.total_sampling_seconds",
                                                  "probe_rng_state.total_cache_seconds"],
               "patient_quality": "NOT_RUN"}

    def run(name, args):
        cmd = [sys.executable, *map(str, args)]
        start = time.monotonic()
        with (out / (name + ".log")).open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=600)
        commands.append({"name": name, "argv": cmd, "exit_code": result.returncode,
                         "seconds": time.monotonic() - start,
                         "log_sha256": digest(out / (name + ".log"))})
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {out / (name + '.log')}")

    try:
        if str(device).startswith("cuda"):
            occupants = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            if occupants:
                raise RuntimeError("GPU occupied; verification will not overlap existing jobs")
        smoke = out / "smoke"
        run("four_arm_smoke", ["scripts/smoke_compact_pipeline.py", "--transfer-utility",
                              "--deterministic-algorithms", "--device", device, "--out", smoke])
        data, config = smoke / "train_data/train.json", smoke / "cross_patient_utility.json"
        common = ["train.py", "--config", config, "--data", data,
                  "--data-sha256", digest(data), "--synthetic", "--device", device]
        run("segment1", [*common, "--stop-after-step", "2", "--out", out / "segment1"])
        checkpoint = out / "segment1/final"
        run("segment2", [*common, "--resume", checkpoint, "--resume-sha256",
                         digest(checkpoint / "MANIFEST.json"), "--out", out / "segment2"])
        uninterrupted = torch.load(smoke / "cross_patient_utility/final/weights.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(out / "segment2/final/weights.pt", map_location="cpu", weights_only=True)
        for key in ("student", "optimizer"):
            if not exact_equal(uninterrupted[key], resumed[key]):
                raise AssertionError(f"bitwise replay differs: {key}")
        if not exact_equal(replay_rng(uninterrupted["rng"]), replay_rng(resumed["rng"])):
            raise AssertionError("RNG/sampler state or coverage differs")
        rows = [[json.loads(line) for line in path.read_text().splitlines()]
                for path in (smoke / "cross_patient_utility/TRAIN_LOG.jsonl", out / "segment2/TRAIN_LOG.jsonl")]
        if len(rows[0]) != 4 or len(rows[1]) != 4:
            raise AssertionError("resume did not retain the full four-step trajectory")
        for first, second in zip(*rows):
            for key in ("data_schedule_sha256", "status", "scale", "realized_displacement_norm"):
                if first[key] != second[key]:
                    raise AssertionError(f"selector replay differs: step {first['step']} {key}")
        if source != compact_training_source_sha256():
            raise AssertionError("source changed during verification")
        receipt.update(status="PASS", exact_weights_optimizer_rng=True,
                       exact_selector_decisions=True,
                       smoke_report_sha256=digest(smoke / "SMOKE_REPORT.json"))
    except BaseException as exc:
        receipt.update(status="FAIL", error=repr(exc))
        raise
    finally:
        (out / "VERIFICATION.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(out / "VERIFICATION.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    verify(args.out, args.device)
