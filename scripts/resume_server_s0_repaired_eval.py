#!/usr/bin/env python3
"""Resume a pinned S0 with an explicit, metadata-only evaluator repair.

The original training worktree, protocol, optimizer and RNG checkpoint stay
unchanged. Only the evaluator's parent-manifest metadata assignment is replaced;
the original source and executed repaired bytes are both retained and hashed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


OLD = '    metrics["evaluation_source_manifest_sha256"] = manifest_data.get("geometry_provenance", {}).get("parent_manifest_sha256", actual_manifest_sha)\n'
NEW = '''    provenance = manifest_data.get("geometry_provenance")
    metrics["evaluation_source_manifest_sha256"] = (
        provenance.get("parent_manifest_sha256", actual_manifest_sha)
        if isinstance(provenance, dict) else actual_manifest_sha
    )
'''


def repaired_bytes(original: bytes) -> bytes:
    text = original.decode("utf-8")
    if text.count(OLD) != 1:
        raise ValueError("expected exactly one known evaluator metadata bug; refusing broader repair")
    return text.replace(OLD, NEW, 1).encode("utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--lock", required=True, type=Path)
    p.add_argument("--lock-sha256", required=True)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rclone-destination", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    source, out = a.source.resolve(), a.out.resolve()
    if not out.is_dir():
        raise ValueError("repair can only resume an existing run")
    with (out / "EVALUATION_REPAIR.lock").open("a") as process_lock:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, str(source))
        from self_audit import server_runner as runner
        from self_audit import compact_inference as inference
        if Path(runner.__file__).resolve() != source / "self_audit/server_runner.py":
            raise ValueError("wrong imported training source")
        lock = runner.verify_protocol_lock(a.lock, a.lock_sha256, root=source)
        if runner.sha256_file(out / "RUN_LOCK.json") != a.lock_sha256:
            raise ValueError("existing run lock differs")
        latest = runner._latest_checkpoint(out)
        if latest is None or latest[:2] != (a.checkpoint.resolve(), a.checkpoint_sha256):
            raise ValueError("repair requires the latest checkpoint and exact hash")
        if str(a.device).startswith("cuda"):
            occupied = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
            ).strip()
            if occupied:
                raise RuntimeError("GPU occupied; preserving other jobs")
        original_path = source / "self_audit/compact_inference.py"
        original = original_path.read_bytes()
        if hashlib.sha256(original).hexdigest() != lock["source"]["files"]["self_audit/compact_inference.py"]:
            raise ValueError("original evaluator does not match the locked source")
        repaired = repaired_bytes(original)
        repair_dir = out / "evaluation_repair"
        repair_dir.mkdir(exist_ok=True)
        executed = repair_dir / "compact_inference_repaired.py"
        if executed.exists() and executed.read_bytes() != repaired:
            raise ValueError("existing repair bytes differ")
        executed.write_bytes(repaired)
        receipt = {
            "schema": "self_audit.s0.evaluation_metadata_repair.v1",
            "scope": "parent-manifest provenance metadata only; prediction and metric calculations unchanged",
            "training_source": lock["source"], "protocol_lock_sha256": a.lock_sha256,
            "original_evaluator_sha256": hashlib.sha256(original).hexdigest(),
            "executed_evaluator_sha256": hashlib.sha256(repaired).hexdigest(),
            "wrapper_sha256": runner.sha256_file(__file__),
            "resume_checkpoint": str(a.checkpoint.resolve()),
            "resume_checkpoint_manifest_sha256": a.checkpoint_sha256,
            "resume_checkpoint_payload_sha256": runner.sha256_file(a.checkpoint / "weights.pt"),
            "training_recipe_changed": False, "training_source_changed": False,
        }
        receipt_path = repair_dir / f"REPAIR_step_{latest[2]:06d}.json"
        if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
            raise ValueError("existing repair receipt differs")
        runner.atomic_json(receipt_path, receipt)
        spec = importlib.util.spec_from_file_location("self_audit._s0_repaired_inference", executed)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        inference.run_two_phase_evaluation = module.run_two_phase_evaluation
        # Complete the failed checkpoint's scheduled VAL before continuing; the
        # original failed prediction directory is retained as a separate attempt.
        runner.evaluate_checkpoint(a.checkpoint, a.checkpoint_sha256, lock=lock,
                                   run_root=out, device=a.device)
        result = runner.run_locked_experiment(
            a.lock, a.lock_sha256, out, device=a.device,
            rclone_base=a.rclone_destination,
            resume_checkpoint=a.checkpoint, resume_sha256=a.checkpoint_sha256, root=source,
        )
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result.get("status") == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
