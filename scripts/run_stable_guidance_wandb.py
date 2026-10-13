#!/usr/bin/env python3
"""Track the unmodified full guidance CLI in one parent-owned W&B run.

Logs are observations only: no metric can change training, select a checkpoint,
or open reference masks. Tracking files live outside the frozen stage folders.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


class Progress:
    def __init__(self, output: Path):
        self.output = output
        self.stage = "starting"
        self.steps_per_epoch = None
        self.bank_patients = 0
        self.inventory_patients = 0
        self.export_patients = 0

    def observe(self, line: str) -> dict:
        if line.startswith("[STAGE] "):
            self.stage = line.split(maxsplit=2)[1]
            self.export_patients = 0
            if self.stage == "train":
                files = sorted(self.output.glob("PREFLIGHT_*.json"))
                if files:
                    report = json.loads(files[-1].read_text())
                    self.steps_per_epoch = report["checks"]["native_cine_inventory"]["train_steps_per_epoch"]
            return {"pipeline/stage": self.stage}
        if line.startswith("[GUIDANCE_METRIC] "):
            row = json.loads(line.partition("] ")[2])
            values = {f"train/{k}": v for k, v in row.items()}
            if self.steps_per_epoch is not None:
                values["train/global_step"] = (row["epoch"] - 1) * self.steps_per_epoch + row["step"]
            return values
        if line.startswith("[STUDENT] "):
            row = ast.literal_eval(line.partition("] ")[2])
            return {f"epoch/{k}": v for k, v in row.items()}
        if line.startswith("[PREFLIGHT/DATA] "):
            self.inventory_patients += 1
            return {"preflight/inventoried_patients": self.inventory_patients}
        if line.startswith("[PREFLIGHT] "):
            return {"preflight/status": line.split()[1]}
        if line.startswith("[GUIDANCE] patient="):
            return {"bank/current_patient": line.split()[1].split("=", 1)[1]}
        if line.startswith("[SSL] "):
            match = re.search(r"frame=(\d+)/(\d+) tiles=(\d+)", line)
            if match:
                frame, total, tiles = map(int, match.groups())
                if frame == 1 or frame == total or frame % 5 == 0:
                    return {"teacher/frame": frame, "teacher/frames": total, "teacher/tiles": tiles}
        if line.startswith("[BANK] "):
            match = re.search(r"\[BANK\] (\S+) (\S+) classes=(\[[^\]]+\]) residual=(\S+)", line)
            if match:
                pid, status, counts, residual = match.groups()
                self.bank_patients += 1
                values = {"bank/completed_patients": self.bank_patients, "bank/patient_id": pid,
                          "bank/status": status, "bank/solver_residual": float(residual)}
                values.update({f"bank/class_{i}_pixels": n for i, n in enumerate(ast.literal_eval(counts))})
                return values
        if line.startswith("[EXPORT] "):
            self.export_patients += 1
            return {f"{self.stage}/completed_patients": self.export_patients}
        if line.startswith("[FULL] "):
            return {"pipeline/status": line.partition("] ")[2].strip()}
        return {}


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-dir", required=True)
    parser.add_argument("--wandb-project", default="self-audit-stable-guidance")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded = args.pipeline_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if "--out" not in forwarded:
        parser.error("pass the full pipeline arguments after --, including --out")
    output = Path(forwarded[forwarded.index("--out") + 1]).resolve()
    tracking = Path(args.tracking_dir).resolve()
    if tracking == output or output in tracking.parents:
        parser.error("tracking-dir must be outside the scientific output folder")
    tracking.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-u", str(repo / "scripts/run_stable_guidance_full.py"), *forwarded]
    # An explicitly requested online run must initialize before consuming GPU time.
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_run_name, mode=args.wandb_mode, dir=str(tracking),
                     config={"pipeline": "stable_ssl_guidance_full", "command": command,
                             "gt_used_for_training": False,
                             "checkpoint_selection": "fixed_last_epoch_no_reference_feedback"},
                     tags=["self-audit", "stable-ssl-guidance"])
    run.define_metric("train/global_step")
    run.define_metric("train/*", step_metric="train/global_step")
    run.define_metric("epoch/epoch")
    run.define_metric("epoch/*", step_metric="epoch/epoch")
    state = dict(id=run.id, url=run.url, project=run.project, entity=run.entity,
                 mode=args.wandb_mode, status="running", telemetry_errors=0,
                 output=str(output), command=command, started_at=time.time())
    write_json(tracking / "WANDB_RUN.json", state)
    print("[WANDB]", run.url, flush=True)
    progress = Progress(output)
    process = None
    exit_code = 1

    def forward_signal(signum, _frame):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signum)

    old_handlers = {sig: signal.signal(sig, forward_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (tracking / "console.log").open("a", buffering=1) as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, start_new_session=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                try:
                    metrics = progress.observe(line)
                    if metrics:
                        run.log(metrics)
                        state.update(last_event=metrics, last_event_at=time.time(), stage=progress.stage)
                        write_json(tracking / "WANDB_RUN.json", state)
                except Exception as exc:
                    # Only telemetry failures are caught; the child exit remains authoritative.
                    state["telemetry_errors"] += 1
                    state["last_telemetry_error_type"] = type(exc).__name__
                    write_json(tracking / "WANDB_RUN.json", state)
            exit_code = process.wait()
        status_path = output / "RUN_STATUS.json"
        final_status = json.loads(status_path.read_text()) if status_path.exists() else {}
        try:
            run.summary["pipeline"] = final_status
            run.define_metric("evaluation/epoch")
            recipe = json.loads((output / "RECIPE.json").read_text())
            final_metrics = {"evaluation/epoch": recipe["epochs"]}
            for split in ("val", "test"):
                metric_file = output / f"evaluation_{split}" / "METRICS.json"
                if metric_file.exists():
                    from publish_guidance_metrics import evaluation_scalars
                    report = json.loads(metric_file.read_text())
                    run.summary[f"evaluation_{split}"] = report["summary"]
                    run.define_metric(f"{split}/*", step_metric="evaluation/epoch")
                    final_metrics.update(evaluation_scalars(report, split))
            if len(final_metrics) > 1:
                run.log({k: v for k, v in final_metrics.items() if v is not None})
                run.summary.update(final_metrics)
                run.summary["evaluation/auroc_status"] = "unavailable: hard-label exports; no transition auditor"
                write_json(tracking / "FINAL_METRICS.json", {"metrics": final_metrics})
        except Exception as exc:
            state["telemetry_errors"] += 1
            state["last_telemetry_error_type"] = type(exc).__name__
        state["pipeline_status"] = final_status.get("status")
        completed = str(state["pipeline_status"]).startswith(("COMPLETED", "PREFLIGHT_PASSED"))
        state["status"] = "finished" if completed and exit_code in (0, 2) else "failed"
        return exit_code if exit_code >= 0 else 128 - exit_code
    finally:
        # Never leave a child pipeline behind if the tracking parent raises.
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        state.update(exit_code=exit_code, finished_at=time.time())
        if state["status"] == "running":
            state["status"] = "failed"
        write_json(tracking / "WANDB_RUN.json", state)
        run.finish(exit_code=0 if state["status"] == "finished" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
