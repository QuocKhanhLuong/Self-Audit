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
from self_audit.compact_teacher import (
    BACKUP_DEFERRED_AUTHORIZATION_INSTRUCTION,
    BACKUP_DEFERRED_AUTHORIZATION_POLICY,
    BACKUP_DEFERRED_BUDGET,
    BACKUP_DEFERRED_POLICY,
    BACKUP_DEFERRED_QUALITY_GATE,
    BACKUP_DEFERRED_RECEIPT_SCHEMA,
    EXPLORATORY_TEACHER_AUTHORIZATION_SCHEMA,
    EXPLORATORY_TEACHER_BUDGET,
    EXPLORATORY_TEACHER_USER_INSTRUCTION,
    REPRODUCTION_RECEIPT_SCHEMA,
)


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


def backup_deferred_opt_in_enabled(state, allow_backup_deferred):
    """Return true only for UPLOAD_PENDING plus the explicit CLI opt-in."""
    return allow_backup_deferred is True and isinstance(state, dict) and state.get("status") == "UPLOAD_PENDING"


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


def _historical_eval_summary(path, domain):
    expected_patients = {"val18": 18, "test50": 50}[domain]
    metrics = json.loads(Path(path).read_text())
    patients = metrics.get("patients")
    if not isinstance(patients, dict) or len(patients) != expected_patients:
        raise ValueError(f"{domain} evaluation must contain exactly {expected_patients} patients")
    if any(not isinstance(pid, str) or not pid.strip() for pid in patients):
        raise ValueError(f"{domain} evaluation contains invalid patient IDs")
    macro = metrics.get("macro_patient")
    if not isinstance(macro, dict):
        raise ValueError(f"{domain} evaluation lacks historical macro_patient results")
    class_dice = {}
    for cls in ("BG", "RV", "MYO", "LV"):
        row = macro.get(cls)
        value = row.get("dice") if isinstance(row, dict) else None
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError(f"{domain} evaluation has invalid macro_patient.{cls}.dice")
        class_dice[cls] = float(value)
    return {
        "class_dice": class_dice,
        "Macro4": sum(class_dice.values()) / 4,
        "foreground": sum(class_dice[k] for k in ("RV", "MYO", "LV")) / 3,
        "patients": expected_patients,
    }


def build_backup_deferred_receipt(
    plan, out, *, exploratory_authorization_path=None, exploratory_authorization_sha256=None,
):
    """Derive local-only B8 evidence from pinned artifacts and actual evaluator outputs."""
    teacher = Path(plan["teacher_run"]).resolve()
    teacher_plan_sha = plan["teacher_plan_sha256"]
    locked_plan_path = teacher / "LOCKED_PLAN.json"
    if not locked_plan_path.is_file() or sha256(locked_plan_path) != teacher_plan_sha:
        raise ValueError("backup-deferred mode requires the exact locked teacher plan")
    locked_plan = json.loads(locked_plan_path.read_text())
    config_path = Path(locked_plan["config"]).resolve()
    state = json.loads((teacher / "STATUS.json").read_text())
    report_path = teacher / "training/REPORT.json"
    report = json.loads(report_path.read_text())
    final = teacher / "training/final"
    manifest_path = final / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    raw_path = teacher / "raw_student/raw_student.pt"
    if state.get("status") != "UPLOAD_PENDING" or state.get("plan_sha256") != teacher_plan_sha:
        raise ValueError("backup-deferred mode requires teacher STATUS=UPLOAD_PENDING for the locked plan")
    if report.get("status") != "COMPLETED" or report.get("steps") != 10000:
        raise ValueError("backup-deferred mode requires the existing completed 10,000-step teacher")
    if (manifest.get("step") != 10000 or manifest.get("config", {}).get("method") != "ema_full"
            or manifest.get("synthetic", False) is not False
            or sha256(final / "weights.pt") != manifest.get("payload_sha256")):
        raise ValueError("backup-deferred mode requires an intact real full-EMA B8 checkpoint")
    if locked_plan.get("source_commit") != plan.get("teacher_source_commit", locked_plan.get("source_commit")):
        raise ValueError("continuation teacher source commit disagrees with the locked teacher plan")
    if not raw_path.is_file():
        raise FileNotFoundError(f"local raw-student export is missing: {raw_path}")

    authorization_path = Path(out) / "BACKUP_DEFERRED_AUTHORIZATION.json"
    eval_status_path = Path(out) / "EVAL_STATUS.json"
    if not authorization_path.is_file() or not eval_status_path.is_file():
        raise FileNotFoundError("backup-deferred authorization and completed evaluator status are required")
    eval_status = json.loads(eval_status_path.read_text())
    evaluation_files = {}
    metrics = {}
    for domain in ("val18", "test50"):
        evaluation_dir = teacher / f"{domain}_evaluation"
        metrics_path = evaluation_dir / "METRICS.json"
        recipe_path = evaluation_dir / "RECIPE.json"
        frozen_path = evaluation_dir / "PREDICTIONS_FROZEN.json"
        evaluation_plan = locked_plan.get("evaluation", {}).get(domain, {})
        evaluation_manifest_path = Path(evaluation_plan.get("path", "")).resolve()
        evaluation_manifest_sha256 = evaluation_plan.get("sha256")
        summary = _historical_eval_summary(metrics_path, domain)
        status_summary = eval_status.get("metrics", {}).get(domain, {})
        metrics_sha = sha256(metrics_path)
        if (status_summary.get("metrics_path") != str(metrics_path.resolve())
                or status_summary.get("metrics_sha256") != metrics_sha
                or status_summary.get("patients") != summary["patients"]
                or status_summary.get("class_dice") != summary["class_dice"]
                or abs(float(status_summary.get("Macro4", -1)) - summary["Macro4"]) > 1e-12
                or abs(float(status_summary.get("foreground", -1)) - summary["foreground"]) > 1e-12):
            raise ValueError(f"{domain} evaluator summary does not match the actual METRICS.json")
        metrics[domain] = summary
        if (not recipe_path.is_file() or not frozen_path.is_file()
                or not evaluation_manifest_path.is_file()
                or not isinstance(evaluation_manifest_sha256, str)
                or sha256(evaluation_manifest_path) != evaluation_manifest_sha256):
            raise ValueError(f"{domain} RECIPE, frozen predictions, or locked evaluation manifest is missing/mismatched")
        evaluation_files[domain] = {
            "path": str(metrics_path.resolve()), "sha256": metrics_sha,
            "recipe_path": str(recipe_path.resolve()), "recipe_sha256": sha256(recipe_path),
            "frozen_path": str(frozen_path.resolve()), "frozen_sha256": sha256(frozen_path),
            "evaluation_manifest_path": str(evaluation_manifest_path),
            "evaluation_manifest_sha256": evaluation_manifest_sha256,
        }
    expected_quality = "GO" if metrics["test50"]["Macro4"] >= 0.87 else "NO_GO"
    if (eval_status.get("status") != "COMPLETED"
            or eval_status.get("backup_status") != "UPLOAD_PENDING"
            or eval_status.get("quality_gate") != expected_quality):
        raise ValueError("completed VAL/TEST evaluation status or unchanged quality gate is inconsistent")

    exploratory_authorization = None
    if (exploratory_authorization_path is None) != (exploratory_authorization_sha256 is None):
        raise ValueError("exploratory authorization path and SHA256 must be supplied together")
    if exploratory_authorization_path is not None:
        exploratory_path = Path(exploratory_authorization_path).resolve()
        if not exploratory_path.is_file() or sha256(exploratory_path) != exploratory_authorization_sha256:
            raise ValueError("explicit exploratory authorization file/hash mismatch")
        authorization = json.loads(exploratory_path.read_text())
        if (authorization.get("schema") != EXPLORATORY_TEACHER_AUTHORIZATION_SCHEMA
                or authorization.get("user_instruction") != EXPLORATORY_TEACHER_USER_INSTRUCTION
                or authorization.get("purpose") != "EXPLORATORY_CPTU_PILOT"
                or authorization.get("teacher_plan_sha256") != teacher_plan_sha
                or authorization.get("budget") != EXPLORATORY_TEACHER_BUDGET):
            raise ValueError("explicit exploratory authorization does not bind the locked teacher/four-arm budget")
        exploratory_authorization = {
            "path": str(exploratory_path), "sha256": exploratory_authorization_sha256.lower(),
        }

    identity = {
        "schema": REPRODUCTION_RECEIPT_SCHEMA,
        "new_reproduction": True,
        "inference_state": "student",
        "original_payload_recovered": False,
        "checkpoint_payload_sha256": manifest["payload_sha256"],
        "raw_student_sha256": sha256(raw_path),
        "checkpoint_manifest_sha256": sha256(manifest_path),
        "source_commit": locked_plan["source_commit"],
        "source_sha256": manifest["source_sha256"],
        "label_budget": 8,
        "labeled_ids": manifest["labeled_ids"],
        "metrics": metrics,
        "quality_gate": expected_quality,
        "historical_reference": {"Macro4": 0.8800492397526454, "foreground": 0.8412102397257094},
        "test50_status": "EXPOSED_DEVELOPMENT; final checkpoint only; no tuning",
        "source_training_cost": {
            "status": "MEASURED_BY_TRAINER", "steps": 10000,
            "elapsed_seconds": report["elapsed_seconds"],
            "source": str(report_path.resolve()), "report_sha256": sha256(report_path),
            "scope": "canonical trainer elapsed; setup/download/upload/evaluation costs separate",
        },
        "backup_deferred": {
            "schema": BACKUP_DEFERRED_RECEIPT_SCHEMA,
            "policy": BACKUP_DEFERRED_POLICY,
            "status": "UPLOAD_PENDING",
            "remote_verified": False,
            "teacher_run": str(teacher),
            "teacher_plan_sha256": teacher_plan_sha,
            "teacher_config": {"path": str(config_path), "sha256": sha256(config_path)},
            "authorization": {
                "path": str(authorization_path.resolve()), "sha256": sha256(authorization_path),
            },
            "evaluation_status": {
                "path": str(eval_status_path.resolve()), "sha256": sha256(eval_status_path),
            },
            "evaluation_files": evaluation_files,
        },
    }
    if exploratory_authorization is not None:
        identity["backup_deferred"]["exploratory_authorization"] = exploratory_authorization
    return identity


def main(
    plan_path, expected_sha, *, allow_backup_deferred=False,
    allow_exploratory_teacher=False, exploratory_authorization_path=None,
    exploratory_authorization_sha256=None,
):
    if allow_exploratory_teacher and not allow_backup_deferred:
        raise ValueError("--allow-exploratory-teacher also requires --allow-backup-deferred")
    if allow_exploratory_teacher != (exploratory_authorization_path is not None):
        raise ValueError("--allow-exploratory-teacher requires an explicit authorization file")
    if allow_exploratory_teacher != (exploratory_authorization_sha256 is not None):
        raise ValueError("--allow-exploratory-teacher requires its explicit authorization SHA256")
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
            status("WAITING_FOR_TEACHER_BACKUP_AND_EVALUATION_OR_EXPLICIT_LOCAL_POLICY")
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
            deferred_identity = None
            while True:
                if time.monotonic() > deadline:
                    raise RuntimeError("teacher wait limit exceeded; no new teacher run launched")
                state = json.loads((Path(plan["teacher_run"]) / "STATUS.json").read_text())
                if state.get("plan_sha256") != plan["teacher_plan_sha256"]:
                    raise ValueError("teacher run plan identity differs")
                if backup_deferred_opt_in_enabled(state, allow_backup_deferred):
                    deferred_identity = build_backup_deferred_receipt(
                        plan, out,
                        exploratory_authorization_path=(exploratory_authorization_path
                                                       if allow_exploratory_teacher else None),
                        exploratory_authorization_sha256=(exploratory_authorization_sha256
                                                          if allow_exploratory_teacher else None),
                    )
                    if deferred_identity["quality_gate"] == "NO_GO" and not allow_exploratory_teacher:
                        raise RuntimeError("teacher quality gate NO_GO; explicit exploratory authorization is required")
                    break
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
            from self_audit.compact_teacher import validate_reproduction_receipt, verify_reproduction_source
            if deferred_identity is None:
                source_receipt = teacher / "TEACHER_IDENTITY.json"
                identity = validate_reproduction_receipt(
                    source_receipt, expected_receipt_sha256=sha256(source_receipt),
                    checkpoint_dir=teacher / "training/final", raw_student_path=teacher / "raw_student/raw_student.pt")
                derived_receipt = json.loads(source_receipt.read_text())
                derived_receipt["parent_receipt_sha256"] = sha256(source_receipt)
                training_report_path = teacher / "training/REPORT.json"
                training_report = json.loads(training_report_path.read_text())
                if training_report.get("status") != "COMPLETED" or training_report.get("steps") != 10000:
                    raise RuntimeError("teacher cost report is not from the completed locked run")
                derived_receipt["source_training_cost"] = {
                    "status": "MEASURED_BY_TRAINER", "steps": 10000,
                    "elapsed_seconds": training_report["elapsed_seconds"],
                    "source": str(training_report_path), "report_sha256": sha256(training_report_path),
                    "scope": "canonical trainer elapsed; setup/download/upload/evaluation costs separate"}
            else:
                derived_receipt = deferred_identity
                status("VALIDATING_LOCAL_TEACHER_EVIDENCE", backup_status="UPLOAD_PENDING",
                       remote_verified=False, quality_gate=derived_receipt["quality_gate"])
            receipt = out / "CPTU_TEACHER_RECEIPT.json"
            atomic_json(receipt, derived_receipt)
            identity = validate_reproduction_receipt(receipt, expected_receipt_sha256=sha256(receipt),
                checkpoint_dir=teacher / "training/final", raw_student_path=teacher / "raw_student/raw_student.pt")
            verify_reproduction_source(plan["teacher_source"], expected_commit=identity["source_commit"],
                                       expected_closure=identity["source_sha256"])
            if deferred_identity is not None:
                status(
                    "EXPLORATORY_TEACHER_VALIDATED_NO_GO" if identity["quality_gate"] == "NO_GO"
                    else "LOCAL_TEACHER_VALIDATED",
                    backup_status=identity["backup_status"], remote_verified=False,
                    quality_gate=identity["quality_gate"],
                    exploratory_authorized=identity["exploratory_authorized"],
                )
            atomic_json(out / "TEACHER_BINDING.json", {
                "receipt_sha256": sha256(receipt), "identity": identity,
                "backup_status": identity["backup_status"],
                "remote_verified": identity["remote_drive_verified"],
            })
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
    parser.add_argument(
        "--allow-backup-deferred", action="store_true",
        help="explicitly permit the hash-verified local teacher when Drive status is UPLOAD_PENDING; quality gates stay mandatory",
    )
    parser.add_argument(
        "--allow-exploratory-teacher", action="store_true",
        help="with explicit authorization only, permit this exact teacher at NO_GO for the locked four-arm exploratory pilot",
    )
    parser.add_argument("--exploratory-authorization")
    parser.add_argument("--exploratory-authorization-sha256")
    args = parser.parse_args()
    if args.allow_exploratory_teacher and not args.allow_backup_deferred:
        parser.error("--allow-exploratory-teacher requires --allow-backup-deferred")
    if args.allow_exploratory_teacher and not (args.exploratory_authorization and args.exploratory_authorization_sha256):
        parser.error("--allow-exploratory-teacher requires --exploratory-authorization and --exploratory-authorization-sha256")
    if not args.allow_exploratory_teacher and (args.exploratory_authorization or args.exploratory_authorization_sha256):
        parser.error("exploratory authorization arguments require --allow-exploratory-teacher")
    main(
        args.plan, args.plan_sha256, allow_backup_deferred=args.allow_backup_deferred,
        allow_exploratory_teacher=args.allow_exploratory_teacher,
        exploratory_authorization_path=args.exploratory_authorization,
        exploratory_authorization_sha256=args.exploratory_authorization_sha256,
    )
