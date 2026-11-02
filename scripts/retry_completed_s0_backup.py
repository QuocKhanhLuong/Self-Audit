#!/usr/bin/env python3
"""Retry backup/evaluation of the completed, pinned S0 without any trainer call.

Keep its historical source and immutable training checkpoint. Record the exact
metadata-only evaluator repair already used during S0 and a duplicate-key fix
for the old post-TEST backup status writer. Neither repair changes predictions,
losses, optimizer state or checkpoint selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

# Pure source-text helper; importing it does not import a training package.
try:
    from .resume_server_s0_repaired_eval import repaired_bytes
except ImportError:  # direct CLI invocation
    from resume_server_s0_repaired_eval import repaired_bytes


OLD_STATUS = '''            run_root, "COMPLETED", **common, backup=result, best_step=best["step"],
            best_macro4=best["val_macro4"], test_status="COMPLETED",
'''
NEW_STATUS = '''            run_root, "COMPLETED", **{**common, "best_step": best["step"],
                "best_macro4": best["val_macro4"]}, backup=result, test_status="COMPLETED",
'''


def repair_retry_source(source: str) -> str:
    if source.count(OLD_STATUS) != 1:
        raise ValueError("expected exactly one legacy backup status site")
    return source.replace(OLD_STATUS, NEW_STATUS, 1)


def require_completed_run(status: dict, latest: tuple | None, expected_sha: str) -> None:
    if (status.get("status") != "UPLOAD_PENDING" or status.get("training_complete") is not True
            or latest is None or latest[2] != 10000):
        raise ValueError("backup-only recovery requires an already completed 10,000-step run")
    if (latest[1] != expected_sha or status.get("final_checkpoint_manifest_sha256") != expected_sha
            or Path(status["final_checkpoint"]).resolve() != Path(latest[0]).resolve()):
        raise ValueError("completed checkpoint identity differs from explicit recovery pin")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--final-manifest-sha256", required=True)
    parser.add_argument("--rclone-destination", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    source, out = args.source.resolve(), args.out.resolve()
    if not out.is_dir():
        raise ValueError("backup recovery cannot create a new run")
    with (out / "EVALUATION_REPAIR.lock").open("a") as process_lock:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, str(source))
        from self_audit import server_runner as runner, compact_inference as inference
        from self_audit.compact_artifacts import load_exact_compact_checkpoint
        if Path(runner.__file__).resolve() != source / "self_audit/server_runner.py":
            raise ValueError("backup recovery imported a different source")
        lock = runner.verify_protocol_lock(out / "RUN_LOCK.json", args.lock_sha256, root=source)
        status = json.loads((out / "RUN_STATUS.json").read_text())
        latest = runner._latest_checkpoint(out)
        require_completed_run(status, latest, args.final_manifest_sha256)
        if status.get("protocol_lock_sha256") != args.lock_sha256:
            raise ValueError("completed run lock differs")
        model, payload, _ = load_exact_compact_checkpoint(
            latest[0], latest[1], device="cpu", expected_config=lock["config"])
        del model, payload
        if args.device.startswith("cuda"):
            occupied = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
            if occupied:
                raise RuntimeError("GPU occupied; refusing concurrent evaluation")

        original_path = source / "self_audit/compact_inference.py"
        original = original_path.read_bytes()
        if hashlib.sha256(original).hexdigest() != lock["source"]["files"]["self_audit/compact_inference.py"]:
            raise ValueError("original evaluator differs from pinned S0 source")
        eval_bytes = repaired_bytes(original)
        retry_before = inspect.getsource(runner.retry_backup)
        retry_after = repair_retry_source(retry_before)
        repair_dir = out / "evaluation_repair"
        repair_dir.mkdir(exist_ok=True)
        eval_path = repair_dir / "compact_inference_repaired.py"
        retry_path = repair_dir / "retry_backup_repaired.py"
        for path, content in ((eval_path, eval_bytes), (retry_path, retry_after.encode())):
            if path.exists() and path.read_bytes() != content:
                raise ValueError("existing audited repair bytes differ")
            if not path.exists():
                path.write_bytes(content)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        runner.atomic_json(repair_dir / f"BACKUP_RETRY_{stamp}.json", {
            "schema": "self_audit.s0.completed_backup_repair.v1", "training_steps_to_run": 0,
            "protocol_lock_sha256": args.lock_sha256, "final_manifest_sha256": latest[1],
            "final_weights_sha256": runner.sha256_file(latest[0] / "weights.pt"),
            "training_source": lock["source"], "wrapper_sha256": runner.sha256_file(__file__),
            "original_evaluator_sha256": hashlib.sha256(original).hexdigest(),
            "executed_evaluator_sha256": hashlib.sha256(eval_bytes).hexdigest(),
            "original_retry_function_sha256": hashlib.sha256(retry_before.encode()).hexdigest(),
            "executed_retry_function_sha256": hashlib.sha256(retry_after.encode()).hexdigest(),
            "changes": ["parent-manifest metadata type handling", "post-TEST status duplicate kwargs"],
            "backup_before_remaining_evaluation": True,
        })
        spec = importlib.util.spec_from_file_location("self_audit._s0_completed_eval_repair", eval_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        inference.run_two_phase_evaluation = module.run_two_phase_evaluation
        exec(compile(retry_after, str(retry_path), "exec"), runner.__dict__)
        result = runner.retry_backup(out, rclone_base=args.rclone_destination, device=args.device)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result.get("status") == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
