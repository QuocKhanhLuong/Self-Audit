#!/usr/bin/env python3
"""Hypergradient-first, three-arm CPU smoke on manufactured data only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from self_audit.data import synthetic_fixture, write_json
from self_audit.engine import source_digest, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New folder outside patient data")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "self_audit.window_smoke.v1", "status": "RUNNING", "synthetic": True,
              "device": "cpu", "patient_accuracy": "NOT_EVALUATED", "source_sha256": source_digest(),
              "arms": {}, "protocol": {"steps": 3, "warmup_steps": 1, "batch_size": 1,
                                         "crop_size": 40, "source_image_size": 32,
                                         "pseudo_threshold": .25, "seed": 42}}
    try:
        # No pilot training occurs unless the numerical gate completes first.
        gate = subprocess.run([sys.executable, str(ROOT / "scripts/check_window_hypergradient.py"),
                               "--out", str(out / "hypergradient.json")], cwd=ROOT,
                              text=True, capture_output=True)
        (out / "hypergradient.log").write_text(gate.stdout + gate.stderr)
        gate.check_returncode()
        gate_report = json.loads((out / "hypergradient.json").read_text())
        if gate_report["status"] != "PASS" or gate_report["source_sha256"] != source_digest():
            raise RuntimeError("hypergradient gate failed or runtime source changed")
        report["hypergradient"] = gate_report
        manifest, sha = synthetic_fixture(out / "data")
        histories = {}
        for arm in ("baseline", "fixed", "meta"):
            config = json.loads((ROOT / f"configs/window_pilot_{arm}.json").read_text())
            config.update(steps=3, warmup_steps=1, batch_size=1, crop_size=40,
                          pseudo_threshold=.25, body_lr=.001, head_lr=.03, save_every=100)
            config["model"]["backbone"] = "tiny_test"
            config["window"]["dim"] = 4
            result = train(config, manifest, sha, out / arm, device="cpu", synthetic=True)
            report["arms"][arm] = result
            histories[arm] = [json.loads(s) for s in (out / arm / "steps.jsonl").read_text().splitlines()]
            assert result["ema_updates"] == 3 and result["pseudo_accepted"] == 2
            assert result["label_reads_outside_budget"] == []
        for rows in zip(*(histories[a] for a in ("baseline", "fixed", "meta")), strict=True):
            for role in ("supervised", "audit", "unlabeled"):
                assert rows[0][role] == rows[1][role] == rows[2][role]
        assert report["arms"]["meta"]["geometry_updates"] == 2
        report["meta_diagnostics"] = [r["meta_window"] for r in histories["meta"] if "meta_window" in r]
        report["matched_role_schedule"] = True
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        write_json(out / "WINDOW_SMOKE.json", report)
        print(json.dumps({"status": report["status"], "receipt": str(out / "WINDOW_SMOKE.json"),
                          "patient_accuracy": "NOT_EVALUATED"}))


if __name__ == "__main__":
    main()
