#!/usr/bin/env python3
"""One locked B8 run: train, export, verified Drive backup, then evaluation.

Run in tmux. An interrupted run is resumed explicitly, never replaced by a new
seed/run. No patient data arrays are uploaded. Drive verification reads every
payload byte back, independently of rclone's upload return code.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def command(argv, log, cwd=None):
    with Path(log).open("a") as stream:
        stream.write(json.dumps({"argv": list(map(str, argv)), "time": time.time()}) + "\n")
        stream.flush()
        subprocess.run(list(map(str, argv)), cwd=cwd, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


class DriveBackup:
    def __init__(self, remote, parent_id, log, attempts=3):
        self.remote = remote.rstrip("/")
        self.log = Path(log)
        self.attempts = attempts
        self.flags = ["--drive-root-folder-id", parent_id, "--retries", "1",
                      "--low-level-retries", "2", "--timeout", "60s", "--tpslimit", "2"]

    def readback(self, name):
        h = hashlib.sha256()
        size = 0
        with self.log.open("ab") as errors:
            proc = subprocess.Popen(["rclone", "cat", self.remote + "/" + name,
                                     *self.flags], stdout=subprocess.PIPE, stderr=errors)
            try:
                for chunk in iter(lambda: proc.stdout.read(8 << 20), b""):
                    h.update(chunk)
                    size += len(chunk)
            finally:
                proc.stdout.close()
            if proc.wait() != 0:
                raise RuntimeError("rclone remote readback failed")
        return size, h.hexdigest()

    def upload_verified(self, path, name):
        path = Path(path)
        size, digest = path.stat().st_size, sha256(path)
        last = None
        for attempt in range(self.attempts):
            try:
                command(["rclone", "copyto", path, self.remote + "/" + name,
                         *self.flags], self.log)
                remote_size, remote_digest = self.readback(name)
                if (remote_size, remote_digest) != (size, digest):
                    raise RuntimeError("remote bytes differ from local payload")
                raw = subprocess.check_output(["rclone", "lsjson", self.remote + "/" + name,
                                               "--stat", *self.flags], text=True)
                metadata = json.loads(raw)
                if metadata["Size"] != size or not metadata.get("ID"):
                    raise RuntimeError("missing Drive size/file ID")
                return {"file": name, "bytes": size, "sha256": digest,
                        "drive_id": metadata["ID"], "readback": "SHA256_MATCH"}
            except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, KeyError) as exc:
                last = exc
                if attempt + 1 < self.attempts:
                    time.sleep(15 * (attempt + 1))
        raise RuntimeError(f"UPLOAD_PENDING: {name}: {last}")


def validate_plan(plan):
    for item in plan["pinned_files"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"pinned file changed: {item['path']}")
    source = Path(plan["source"])
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
                                     "--untracked-files=no"], text=True).strip()
    if head != plan["source_commit"] or dirty:
        raise ValueError("training source commit/clean state mismatch")
    original = json.loads(Path(plan["historical_config"]).read_text())
    actual = json.loads(Path(plan["config"]).read_text())
    restored = json.loads(json.dumps(actual))
    for key in ("source_repo", "weights"):
        restored["model"][key] = original["model"][key]
    if original != restored or actual["method"] != "ema_full" or actual["steps"] != 10000:
        raise ValueError("B8 recipe changed beyond local backbone paths")
    if sha256(actual["model"]["weights"]) != actual["model"]["weights_sha256"]:
        raise ValueError("pretrained encoder hash mismatch")
    manifest = json.loads(Path(plan["data"]).read_text())
    ids = sorted(r["id"] for r in manifest["labeled"])
    expected = ["patient007", "patient010", "patient011", "patient024", "patient026",
                "patient041", "patient077", "patient099"]
    if ids != expected or len(manifest["unlabeled"]) != 72:
        raise ValueError("not historical ACDC L8/U72")
    if any("masks" in r or "mask" in r for r in manifest["unlabeled"]):
        raise ValueError("U manifest exposes masks")


def export_raw(source, checkpoint, destination):
    """Executed in a fresh process using the pinned historical source only."""
    sys.path.insert(0, str(Path(source).resolve()))
    import torch
    from self_audit.engine import load_checkpoint
    torch.set_num_threads(2)
    checkpoint, destination = Path(checkpoint), Path(destination)
    destination.mkdir(exist_ok=True)
    manifest = json.loads((checkpoint / "MANIFEST.json").read_text())
    model, _ = load_checkpoint(checkpoint, sha256(checkpoint / "MANIFEST.json"), "cpu")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    target = destination / "raw_student.pt"
    temporary = destination / "raw_student.pt.tmp"
    torch.save(state, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    restored = torch.load(target, map_location="cpu", weights_only=True)
    model.load_state_dict(restored, strict=True)
    if set(restored) != set(state) or any(not torch.equal(state[k], restored[k]) for k in state):
        raise RuntimeError("raw student export reload differs")
    atomic_json(destination / "RAW_STUDENT.json", {
        "schema": "self_audit.b8.raw_student.v1", "model_class": "NativeStudent",
        "inference_state": "student", "payload_sha256": sha256(target),
        "source_checkpoint_payload_sha256": manifest["payload_sha256"],
        "source_checkpoint_manifest_sha256": sha256(checkpoint / "MANIFEST.json"),
        "architecture": manifest["config"]["model"], "window": manifest["config"]["window"],
        "preprocessing": {"normalization": "per_cine_image_only_percentiles_0.5_99.5",
                          "axes": "TZYX", "context": "3 adjacent z slices with edge replication",
                          "class_map": {"0": "BG", "1": "RV", "2": "MYO", "3": "LV"}},
        "labeled_ids": manifest["labeled_ids"], "label_budget": manifest["label_budget"],
        "reload": "STRICT_STATE_EQUAL", "source_sha256": manifest["source_sha256"],
        "deployment_note": "Frozen KD teacher raw state; pinned NativeStudent/DINO source and initialization assets included separately. Not compact deployment."})


def run(plan_path, expected_sha, resume=False, preflight_only=False):
    plan_path = Path(plan_path).resolve()
    if sha256(plan_path) != expected_sha:
        raise ValueError("plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    root = Path(plan["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pinned = root / "LOCKED_PLAN.json"
        if pinned.exists() and pinned.read_bytes() != plan_path.read_bytes():
            raise ValueError("existing run belongs to another plan")
        if not pinned.exists():
            shutil.copyfile(plan_path, pinned)
        state = root / "STATUS.json"
        def status(value, **extra):
            atomic_json(state, {"status": value, "time": time.time(), "plan_sha256": expected_sha, **extra})
            print(json.dumps({"status": value, **extra}), flush=True)
        backup = DriveBackup(plan["drive_remote"], plan["drive_parent_id"], root / "rclone.log")
        try:
            status("PREFLIGHT")
            probe = root / "WRITE_PROBE.json"
            atomic_json(probe, {"plan_sha256": expected_sha, "purpose": "B8 backup write/readback gate"})
            proof = backup.upload_verified(probe, probe.name)
            atomic_json(root / "DRIVE_WRITE_VERIFIED.json", proof)
            if preflight_only:
                status("READY_NOT_STARTED")
                return
            report = root / "training/REPORT.json"
            final = root / "training/final"
            if not report.exists():
                cmd = [plan["python"], "train.py", "--config", plan["config"], "--data", plan["data"],
                       "--data-sha256", plan["data_sha256"], "--out", str(root / "training"),
                       "--device", "cuda", "--allow-full-run"]
                if (root / "training").exists():
                    if not resume:
                        raise RuntimeError("existing training requires explicit same-run --resume")
                    candidates = sorted((root / "training").glob("step_*/MANIFEST.json"))
                    if not candidates:
                        raise RuntimeError("no resumable checkpoint; do not start a replacement run")
                    latest = candidates[-1]
                    cmd.extend(["--resume", str(latest.parent), "--resume-sha256", sha256(latest)])
                elif (root / "TRAIN_STARTED.json").exists():
                    raise RuntimeError("started run lost its output; refusing fresh replacement")
                status("TRAINING")
                if not (root / "TRAIN_STARTED.json").exists():
                    atomic_json(root / "TRAIN_STARTED.json", {"time": time.time(), "command": cmd})
                command(cmd, root / "train.log", plan["source"])
            result = json.loads(report.read_text())
            if result["status"] != "COMPLETED" or result["steps"] != 10000:
                raise RuntimeError("training incomplete; no final checkpoint claim")
            metadata = json.loads((final / "MANIFEST.json").read_text())
            if metadata["step"] != 10000 or sha256(final / "weights.pt") != metadata["payload_sha256"]:
                raise RuntimeError("final checkpoint hash/step mismatch")
            status("EXPORTING_BEFORE_EVALUATION")
            command([plan["python"], str(Path(__file__).resolve()), "--export-source", plan["source"],
                     "--checkpoint", final, "--export-out", root / "raw_student"], root / "export.log")
            files = [(final / "weights.pt", "full_checkpoint/weights.pt"),
                     (final / "MANIFEST.json", "full_checkpoint/MANIFEST.json"),
                     (root / "raw_student/raw_student.pt", "raw_student/raw_student.pt"),
                     (root / "raw_student/RAW_STUDENT.json", "raw_student/RAW_STUDENT.json"),
                     (pinned, "LOCKED_PLAN.json"), (Path(plan["config"]), "CONFIG.json"),
                     (Path(plan["data"]), "TRAIN_MANIFEST.json"), (report, "TRAIN_REPORT.json")]
            files += [(Path(x["path"]), "provenance/" + x["name"]) for x in plan["backup_files"]]
            sums = root / "SHA256SUMS.txt"
            sums.write_text("".join(f"{sha256(path)}  {name}\n" for path, name in files))
            files.append((sums, sums.name))
            status("UPLOADING_WEIGHTS")
            verified = [backup.upload_verified(path, name) for path, name in files]
            atomic_json(root / "REMOTE_WEIGHTS_VERIFIED.json", {"files": verified, "remote": plan["drive_remote"]})
            backup.upload_verified(root / "REMOTE_WEIGHTS_VERIFIED.json", "REMOTE_WEIGHTS_VERIFIED.json")
            status("WEIGHTS_VERIFIED_EVALUATING")
            metrics = {}
            for domain in ("val18", "test50"):
                item = plan["evaluation"][domain]
                destination = root / (domain + "_evaluation")
                if not (destination / "METRICS.json").exists():
                    if destination.exists():
                        raise RuntimeError("incomplete frozen evaluation retained; inspect before resuming")
                    command([plan["python"], plan["evaluator"], "--checkpoint", final,
                             "--checkpoint-sha256", sha256(final / "MANIFEST.json"),
                             "--data", item["path"], "--data-sha256", item["sha256"],
                             "--out", destination, "--device", "cuda", "--batch-size", "1", "--threads", "2"],
                            root / (domain + ".log"), plan["source"])
                m = json.loads((destination / "METRICS.json").read_text())
                classes = {k: m["macro_patient"][k]["dice"] for k in ("BG", "RV", "MYO", "LV")}
                metrics[domain] = {"class_dice": classes, "Macro4": sum(classes.values()) / 4,
                                   "foreground": sum(classes[k] for k in ("RV", "MYO", "LV")) / 3,
                                   "patients": len(m["patients"])}
                for filename in ("METRICS.json", "RECIPE.json", "PREDICTIONS_FROZEN.json"):
                    backup.upload_verified(destination / filename, domain + "/" + filename)
            identity = {"schema": "self_audit.b8.reproduction.v1", "inference_state": "student",
                        "new_reproduction": True, "original_payload_recovered": False,
                        "checkpoint_payload_sha256": metadata["payload_sha256"],
                        "raw_student_sha256": sha256(root / "raw_student/raw_student.pt"),
                        "checkpoint_manifest_sha256": sha256(final / "MANIFEST.json"),
                        "source_commit": plan["source_commit"], "source_sha256": metadata["source_sha256"],
                        "labeled_ids": metadata["labeled_ids"], "label_budget": 8,
                        "metrics": metrics, "remote_files": verified,
                        "historical_reference": {"Macro4": .8800492397526454, "foreground": .8412102397257094},
                        "quality_gate": "GO" if metrics["test50"]["Macro4"] >= .87 else "NO_GO",
                        "test50_status": "EXPOSED_DEVELOPMENT; final checkpoint only; no tuning"}
            atomic_json(root / "TEACHER_IDENTITY.json", identity)
            for path in [root / "TEACHER_IDENTITY.json", root / "train.log", root / "training/steps.jsonl"]:
                backup.upload_verified(path, path.name)
            handover = Path(plan["handover"])
            current = json.loads(handover.read_text())
            shutil.copyfile(handover, root / ("HANDOVER.before." + sha256(handover) + ".json"))
            current["b8_reproduction_20261031"] = {"status": "COMPLETED_REMOTE_VERIFIED", "local": str(root),
                                                  "identity": str(root / "TEACHER_IDENTITY.json"), **identity}
            atomic_json(handover, current)
            backup.upload_verified(handover, "HANDOVER.json")
            status("TEACHER_COMPLETED_REMOTE_VERIFIED", quality_gate=identity["quality_gate"])
        except Exception as exc:
            status("UPLOAD_PENDING" if "UPLOAD_PENDING" in str(exc) else "BLOCKED", error=str(exc))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan"); parser.add_argument("--plan-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--export-source"); parser.add_argument("--checkpoint"); parser.add_argument("--export-out")
    args = parser.parse_args()
    if args.export_source:
        export_raw(args.export_source, args.checkpoint, args.export_out)
    elif args.plan and args.plan_sha256:
        run(args.plan, args.plan_sha256, args.resume, args.preflight_only)
    else:
        parser.error("--plan and --plan-sha256 required")
