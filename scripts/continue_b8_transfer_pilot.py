#!/usr/bin/env python3
"""Persistent, fail-closed B8 backup/evaluation -> four-arm CPTU continuation.

This does not launch or restart a teacher. The separate B8 runner owns its one
authorized run. A failed gate leaves evidence and stops this continuation.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_b8_reproduction import atomic_json, sha256


def teacher_gate(root, expected_plan_sha):
    root = Path(root)
    state = json.loads((root / "STATUS.json").read_text())
    if state.get("plan_sha256") != expected_plan_sha:
        raise ValueError("teacher run plan identity differs")
    if state["status"] in ("BLOCKED", "UPLOAD_PENDING"):
        raise RuntimeError("teacher pipeline stopped: " + state["status"])
    if state["status"] != "TEACHER_COMPLETED_REMOTE_VERIFIED":
        return False
    if state.get("quality_gate") != "GO":
        raise RuntimeError("teacher quality gate NO_GO; compact expansion prohibited")
    identity = json.loads((root / "TEACHER_IDENTITY.json").read_text())
    if identity.get("quality_gate") != "GO" or identity.get("inference_state") != "student":
        raise ValueError("final teacher identity is not a qualified raw student")
    proof = json.loads((root / "REMOTE_WEIGHTS_VERIFIED.json").read_text())
    if proof.get("files") != identity.get("remote_files"):
        raise ValueError("teacher identity does not match remote readback proof")
    return True


def completed_teacher_for_backup(root):
    """A backup retry can never authorize a fresh or partial teacher training run."""
    root = Path(root)
    report = json.loads((root / "training/REPORT.json").read_text())
    manifest = json.loads((root / "training/final/MANIFEST.json").read_text())
    if (report.get("status") != "COMPLETED" or report.get("steps") != 10000
            or manifest.get("step") != 10000
            or manifest.get("config", {}).get("method") != "ema_full"
            or sha256(root / "training/final/weights.pt") != manifest.get("payload_sha256")):
        raise ValueError("backup retry requires an intact completed B8 teacher; training is prohibited")
    return manifest["payload_sha256"]


def main(plan_path, expected_sha):
    plan_path = Path(plan_path).resolve()
    if sha256(plan_path) != expected_sha:
        raise ValueError("continuation plan SHA mismatch")
    plan = json.loads(plan_path.read_text())
    if subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() != plan["source_commit"]:
        raise ValueError("continuation source SHA mismatch")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("continuation source is dirty")
    for item in plan["pinned_files"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("pinned continuation asset changed: " + item["path"])
    out = Path(plan["out"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "LOCKED_PLAN.json").exists():
            raise RuntimeError("continuation already started; inspect preserved state instead of rerunning arms")
        shutil.copyfile(plan_path, out / "LOCKED_PLAN.json")
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        env["PATH"] = str(Path(plan["python"]).parent) + os.pathsep + env.get("PATH", "")
        commands = []

        def status(value, **extra):
            atomic_json(out / "STATUS.json", {"status": value, "time": time.time(),
                        "plan_sha256": expected_sha, **extra})
            print(json.dumps({"status": value, **extra}), flush=True)

        def run(name, args, timeout):
            if subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() != plan["source_commit"]:
                raise RuntimeError("source changed after continuation was locked")
            if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
                raise RuntimeError("source was edited after continuation was locked")
            cmd = [plan["python"], *map(str, args)]
            started = time.monotonic()
            status(name.upper())
            timed_out = False
            with (out / (name + ".log")).open("w") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        code = proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        code = proc.wait()
            commands.append({"name": name, "argv": cmd, "exit_code": code, "timed_out": timed_out,
                             "seconds": time.monotonic() - started,
                             "log_sha256": sha256(out / (name + ".log"))})
            atomic_json(out / "COMMANDS.json", commands)
            if code or timed_out:
                raise RuntimeError(f"{name} failed; inspect {out / (name + '.log')}")

        try:
            status("WAITING_FOR_TEACHER_BACKUP_AND_EVALUATION")
            deadline = time.monotonic() + 6 * 3600
            backup_retry = plan.get("backup_retry")
            if backup_retry is not None:
                if (type(backup_retry.get("max_attempts")) is not int
                        or not 1 <= backup_retry["max_attempts"] <= 3
                        or type(backup_retry.get("delay_seconds")) is not int
                        or not 60 <= backup_retry["delay_seconds"] <= 900):
                    raise ValueError("backup retry budget must be 1..3 attempts with 60..900 seconds delay")
                pinned_paths = {str(Path(x["path"]).resolve()) for x in plan["pinned_files"]}
                if any(str(Path(backup_retry[k]).resolve()) not in pinned_paths
                       for k in ("runner", "teacher_plan")):
                    raise ValueError("backup runner and teacher plan must be hash-pinned")
            retry_count = 0
            while True:
                if time.monotonic() > deadline:
                    raise RuntimeError("teacher wait limit exceeded; no new teacher run launched")
                state = json.loads((Path(plan["teacher_run"]) / "STATUS.json").read_text())
                if state.get("plan_sha256") != plan["teacher_plan_sha256"]:
                    raise ValueError("teacher run plan identity differs")
                if state.get("status") == "UPLOAD_PENDING" and backup_retry is not None:
                    if retry_count >= backup_retry["max_attempts"]:
                        raise RuntimeError("bounded backup retries exhausted; weights retained; no retraining")
                    completed_teacher_for_backup(plan["teacher_run"])
                    if sha256(backup_retry["teacher_plan"]) != plan["teacher_plan_sha256"]:
                        raise ValueError("backup retry teacher plan differs")
                    # Evaluation can follow a successful backup in this runner.
                    # Wait for any other CUDA job; never stop or overlap it.
                    occupied = subprocess.check_output(
                        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
                    if occupied:
                        status("WAITING_FOR_GPU_BEFORE_BACKUP_RETRY")
                        time.sleep(30)
                        continue
                    retry_count += 1
                    try:
                        run(f"backup_retry_{retry_count}", [backup_retry["runner"],
                            "--plan", backup_retry["teacher_plan"], "--plan-sha256", plan["teacher_plan_sha256"],
                            "--resume"], 3600)
                    except RuntimeError:
                        after = json.loads((Path(plan["teacher_run"]) / "STATUS.json").read_text())
                        if after.get("status") != "UPLOAD_PENDING":
                            raise
                        status("WAITING_FOR_DRIVE_QUOTA", attempt=retry_count,
                               maximum=backup_retry["max_attempts"])
                        time.sleep(backup_retry["delay_seconds"])
                    continue
                if teacher_gate(plan["teacher_run"], plan["teacher_plan_sha256"]):
                    break
                time.sleep(30)
            teacher = Path(plan["teacher_run"])
            receipt = teacher / "TEACHER_IDENTITY.json"
            from self_audit.compact_teacher import validate_reproduction_receipt, verify_reproduction_source
            identity = validate_reproduction_receipt(
                receipt, expected_receipt_sha256=sha256(receipt),
                checkpoint_dir=teacher / "training/final", raw_student_path=teacher / "raw_student/raw_student.pt")
            verify_reproduction_source(plan["teacher_source"], expected_commit=identity["source_commit"],
                                       expected_closure=identity["source_sha256"])
            atomic_json(out / "TEACHER_BINDING.json", {"receipt_sha256": sha256(receipt), "identity": identity})
            training_report_path = teacher / "training/REPORT.json"
            training_report = json.loads(training_report_path.read_text())
            if training_report.get("status") != "COMPLETED" or training_report.get("steps") != 10000:
                raise RuntimeError("teacher cost report is not from the completed locked run")
            derived_receipt = json.loads(receipt.read_text())
            derived_receipt["parent_receipt_sha256"] = sha256(receipt)
            derived_receipt["source_training_cost"] = {
                "status": "MEASURED_BY_TRAINER", "steps": 10000,
                "elapsed_seconds": training_report["elapsed_seconds"],
                "source": str(training_report_path), "report_sha256": sha256(training_report_path),
                "scope": "canonical trainer elapsed; setup/download/upload/evaluation costs separate"}
            receipt = out / "CPTU_TEACHER_RECEIPT.json"
            atomic_json(receipt, derived_receipt)
            identity = validate_reproduction_receipt(receipt, expected_receipt_sha256=sha256(receipt),
                checkpoint_dir=teacher / "training/final", raw_student_path=teacher / "raw_student/raw_student.pt")
            run("check_repo", ["scripts/check_repo.py"], 120)
            run("pytest", ["-m", "pytest", "-q"], 900)
            run("cuda_replay", ["scripts/verify_compact_resume.py", "--out", out / "cuda_verification",
                                "--device", "cuda:0"], 1200)
            if json.loads((out / "cuda_verification/VERIFICATION.json").read_text())["status"] != "PASS":
                raise RuntimeError("CUDA replay verification did not pass")
            if shutil.disk_usage(out).free < 24 * 1024**3:
                raise RuntimeError("insufficient free disk for bounded FP32 teacher cache and retained outputs")
            cache = out / "teacher_cache/CACHE_MANIFEST.json"
            run("teacher_cache", ["scripts/build_compact_teacher_cache.py", "--data-manifest", plan["data"],
                "--data-sha256", plan["data_sha256"], "--output-dir", cache.parent,
                "--checkpoint-dir", teacher / "training/final", "--reproduction-receipt", receipt,
                "--reproduction-receipt-sha256", sha256(receipt), "--reproduction-source", plan["teacher_source"],
                "--precision", "float32", "--device", "cuda:0", "--batch-size", "8"], 3600)
            from self_audit.transfer_preflight import derive_reproduction_protocol
            protocol = derive_reproduction_protocol(json.loads(Path(plan["base_protocol"]).read_text()), identity, sha256(cache))
            if protocol.get("runtime_policy", {}).get("deterministic_algorithms") is not True:
                raise RuntimeError("derived arms do not match the verified deterministic runtime")
            protocol_path = out / "PROTOCOL.json"
            atomic_json(protocol_path, protocol)
            run("four_arm_pilot", ["scripts/run_transfer_utility_pilot.py", "--mode", "all",
                "--protocol", protocol_path, "--matrix", plan["matrix"], "--out", out / "pilot",
                "--device", "cuda:0", "--data", plan["data"], "--data-sha256", plan["data_sha256"],
                "--val-manifest", plan["val"], "--val-manifest-sha256", plan["val_sha256"],
                "--teacher-checkpoint", teacher / "training/final",
                "--teacher-checkpoint-sha256", identity["checkpoint_manifest_sha256"],
                "--teacher-payload-sha256", identity["checkpoint_payload_sha256"],
                "--teacher-cache", cache, "--teacher-cache-sha256", sha256(cache),
                "--reproduction-receipt", receipt, "--reproduction-receipt-sha256", sha256(receipt),
                "--reproduction-source", plan["teacher_source"]], 37000)
            report = json.loads((out / "pilot/TRANSFER_UTILITY_PILOT_REPORT.json").read_text())
            if report["status"] not in ("PILOT_GO", "NO_GO"):
                raise RuntimeError("pilot lacks completed empirical decision")
            status("PILOT_COMPLETED", result=report["status"], report=str(out / "pilot/TRANSFER_UTILITY_PILOT_REPORT.json"))
        except BaseException as exc:
            status("BLOCKED", error=repr(exc))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    main(args.plan, args.plan_sha256)
