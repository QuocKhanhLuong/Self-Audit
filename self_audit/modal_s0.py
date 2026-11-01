"""S0 execution shell around the canonical compact trainer; no new learning rule."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time

from .asset_staging import atomic_json, sha256, source_digest
from .modal_data_prep import L8, OLD_STAGING_SHA


def budget_plan(credit_usd, spent_this_round_usd=0.0):
    """Conservative whole-container bound, with CPU/storage/egress reserve."""
    if not all(math.isfinite(x) and x >= 0 for x in (credit_usd, spent_this_round_usd)):
        raise ValueError("credit and spent amount must be finite, nonnegative")
    remaining = min(15.0 - spent_this_round_usd, credit_usd)
    reserve = 2.0
    # CPU=2 maximum, memory=16GiB maximum: observed rate < $2.33/h.
    # $3/h also reserves idle/cold execution overhead; at most two hours.
    timeout = min(7200, math.floor((remaining - reserve) / 3.0 * 3600))
    if timeout < 900:
        raise ValueError("BLOCKED_CREDIT: insufficient remaining credit including reserve")
    return {"cap_usd": min(15.0, credit_usd + spent_this_round_usd),
            "remaining_credit_user_reported_usd": credit_usd,
            "spent_this_round_usd": spent_this_round_usd,
            "reserve_usd": reserve, "container_rate_upper_usd_per_hour": 3.0,
            "function_timeout_seconds": timeout, "training_seconds": timeout - 600,
            "max_container_cost_usd": timeout / 3600 * 3.0,
            "gpu": "A100-40GB", "max_containers": 1, "retries": 0}


def s0_config(training_seconds=None):
    from .compact_training import validate_compact_config
    path = Path(__file__).resolve().parents[1] / "configs/fresh_s0/supervised_wide_seed42.json"
    cfg = json.loads(path.read_text())
    return validate_compact_config(cfg, allow_full_run=True)


def checked_data(namespace):
    base = Path(namespace)
    r = json.loads((base / "DATA_PREP.json").read_text())
    if r["status"] != "PASS" or not r.get("persistence_verified"):
        raise ValueError("BLOCKED_DATA_PREP: requires verified fresh-container PASS")
    assert sha256("/assets/receipts/STAGING.json") == OLD_STAGING_SHA
    assert r["labeled_ids"] == L8 and r["train_count"] == 80 and r["val_count"] == 18
    assert r["unlabeled_mask_reads"] == r["unlabeled_mask_downloads"] == 0
    assert sha256(r["train_manifest"]) == r["train_manifest_sha256"]
    assert sha256(r["val_manifest"]) == r["val_manifest_sha256"]
    for f in r["files"]:
        assert sha256(f["path"]) == f["sha256"]
    return r


def atomic_save(save, commit, *args, **kwargs):
    """Publish a complete hash-verified directory; partial saves never look resumable."""
    out = Path(args[4])
    if out.exists():
        raise FileExistsError(out)
    temp = out.with_name("." + out.name + ".partial")
    if temp.exists():
        raise FileExistsError(f"interrupted save retained for inspection: {temp}")
    altered = list(args)
    altered[4] = temp
    manifest_sha = save(*altered, **kwargs)
    assert sha256(temp / "MANIFEST.json") == manifest_sha
    manifest = json.loads((temp / "MANIFEST.json").read_text())
    assert sha256(temp / "weights.pt") == manifest["payload_sha256"]
    for name in ("weights.pt", "MANIFEST.json"):
        with (temp / name).open("rb") as f:
            os.fsync(f.fileno())
    os.replace(temp, out)
    commit()
    print(json.dumps({"checkpoint": str(out), "step": kwargs["step"],
                      "sha256": manifest["payload_sha256"], "volume_committed": True}), flush=True)
    return manifest_sha


class BudgetBoundary(RuntimeError):
    pass


@contextmanager
def checkpoint_io(commit, after_save=None):
    from . import compact_training as trainer
    original = trainer.save_compact_checkpoint
    def save(*args, **kwargs):
        result = atomic_save(original, commit, *args, **kwargs)
        if after_save is not None:
            after_save(Path(args[4]), kwargs["step"])
        return result
    trainer.save_compact_checkpoint = save
    try:
        yield
    finally:
        trainer.save_compact_checkpoint = original


def progress(log_path, stop):
    """Read-only telemetry; never touches PyTorch or sampler RNG."""
    done = 0
    started = time.monotonic()
    while not stop.wait(10):
        try:
            with Path(log_path).open() as f:
                rows = f.readlines()
            row = json.loads(rows[-1]) if rows else None
            if row and int(row["step"]) + 1 > done:
                done = int(row["step"]) + 1
                elapsed = time.monotonic() - started
                print(json.dumps({"progress": f"{done}/10000", "loss": row["supervised_loss"],
                    "eta_seconds": (10000 - done) * elapsed / done,
                    "gpu_memory": row.get("cuda_memory")}), flush=True)
        except (OSError, json.JSONDecodeError):
            continue


def cpu_smoke(namespace, git_sha, config, commit):
    import torch
    from .compact_training import train_compact
    data = checked_data(namespace)
    torch.set_num_threads(2)
    out = Path(namespace) / "s0_cpu_smoke"
    with checkpoint_io(commit):
        report = train_compact(config, data["train_manifest"], data["train_manifest_sha256"],
                              out, device="cpu", stop_after_step=2, allow_full_run=True)
    summary = report["warm_start_summary"]
    for field in ("initial_gradient_norms", "initial_weight_changes"):
        for module in ("encoder", "decoder", "head"):
            assert summary[field][module] > 0, (field, module)
    assert report["optimizer_calls"]["adamw"] == 2 and report["optimizer_calls"]["sgd"] == 0
    assert report["seen_audit"] == [] and report["seen_probe"] == []
    receipt = {"status": "PASS", "git_sha": git_sha, "source_sha256": source_digest("/repo"),
        "config": config, "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "initial_state_sha256": report["initialization"]["initial_state_sha256"],
        "train_manifest_sha256": data["train_manifest_sha256"],
        "report_sha256": sha256(out / "REPORT.json"), "warm_start_summary": summary,
        "supervised_steps": 2, "used_for_s0_initialization": False}
    atomic_json(Path(namespace) / "CPU_SMOKE.json", receipt)
    commit()
    return receipt


def run_s0(namespace, git_sha, config, budget, commit, resume=None, resume_sha256=None):
    import torch
    from .compact_training import train_compact, _state_sha256
    from .compact_model import build_compact_model
    from .compact_artifacts import export_compact_student, load_compact_export
    from .compact_inference import run_two_phase_evaluation
    assert "RCLONE_CONFIG_TEXT" not in os.environ
    data = checked_data(namespace)
    base = Path(namespace)
    smoke = json.loads((base / "CPU_SMOKE.json").read_text())
    assert smoke["status"] == "PASS" and smoke["config"] == config
    assert smoke["git_sha"] == git_sha and smoke["source_sha256"] == source_digest("/repo")
    assert smoke["train_manifest_sha256"] == data["train_manifest_sha256"]
    assert torch.cuda.device_count() == 1 and "A100" in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    initial = build_compact_model(config["model"])
    assert _state_sha256(initial) == smoke["initial_state_sha256"]
    del initial
    out = base / "s0_train"
    resume_step = 0
    if resume is not None:
        resume = Path(resume)
        assert resume.is_relative_to(base) and resume_sha256 == sha256(resume / "MANIFEST.json")
        resume_meta = json.loads((resume / "MANIFEST.json").read_text())
        resume_step = resume_meta["step"]
        assert 0 < resume_step < 10000
        assert resume_meta["initial_state_sha256"] == smoke["initial_state_sha256"]
        out = base / f"s0_resume_{resume_step:06d}"
    if out.exists():
        raise ValueError("S0 already started: never restart from scratch in this namespace")
    lock = {"schema": "self_audit.fresh_s0.lock.v1", "git_sha": git_sha,
        "source_sha256": source_digest("/repo"), "config": config, "budget": budget,
        "config_sha256": smoke["config_sha256"],
        "initial_state_sha256": smoke["initial_state_sha256"], "data_prep_sha256": sha256(base / "DATA_PREP.json"),
        "train_manifest_sha256": data["train_manifest_sha256"],
        "val_manifest_sha256": data["val_manifest_sha256"], "historical_reproduction": False,
        "teacher": None, "initialization": "random_seed42_kaiming_normal", "deployment": "raw_student",
        "evaluation": "final_step10000_only_export_reload_VAL18_no_TEST",
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
    lock_path = base / "S0_LOCK.json"
    if lock_path.exists():
        original_lock = json.loads(lock_path.read_text())
        if resume is None:
            raise ValueError("S0 launch lock exists; explicit checkpoint resume required")
        for key in ("git_sha", "source_sha256", "initial_state_sha256", "train_manifest_sha256", "config"):
            assert original_lock[key] == lock[key]
        atomic_json(base / f"S0_RESUME_{resume_step:06d}.json", dict(lock, resume=str(resume), resume_sha256=resume_sha256))
    else:
        assert resume is None
        atomic_json(lock_path, lock)
    atomic_json(base / "S0_STATUS.json", {"status": "RUNNING", "app_id": os.environ.get("MODAL_APP_ID"), "steps": 0})
    commit()
    stop = threading.Event()
    watcher = threading.Thread(target=progress, args=(out / "TRAIN_LOG.jsonl", stop), daemon=True)
    watcher.start()
    started = time.monotonic()
    def cost_boundary(path, step):
        elapsed = time.monotonic() - started
        next_segment_seconds = elapsed / max(step - resume_step, 1) * min(1000, 10000 - step)
        if step < 10000 and elapsed + 1.5 * next_segment_seconds > budget["training_seconds"]:
            raise BudgetBoundary(str(path))
    try:
        with checkpoint_io(commit, cost_boundary):
            report = train_compact(config, data["train_manifest"], data["train_manifest_sha256"],
                                  out, device="cuda", allow_full_run=True,
                                  resume=resume, resume_sha256=resume_sha256)
        assert report["initialization"]["initial_state_sha256"] == smoke["initial_state_sha256"]
        if report["steps"] != 10000:
            status = {"status": report["status"], "steps": report["steps"],
                      "checkpoint": report["final_checkpoint_path"], "resume_required": True}
        else:
            # Backup and commit the real payload BEFORE any VAL reference is opened.
            export = base / "s0_export"
            export_sha = export_compact_student(out / "final", report["final_manifest_sha256"], export)
            exported, _ = load_compact_export(export, export_sha, device="cpu")
            del exported
            backup = base / "s0_backup"
            backup.mkdir(exist_ok=False)
            shutil.copytree(out / "final", backup / "full_checkpoint")
            shutil.copytree(export, backup / "exported_student")
            files = sorted(p for p in backup.rglob("*") if p.is_file())
            sums = "".join(f"{sha256(p)}  {p.relative_to(backup)}\n" for p in files)
            (backup / "SHA256SUMS.txt").write_text(sums)
            atomic_json(base / "S0_STATUS.json", {"status": "WEIGHTS_BACKED_UP_EVAL_PENDING", "steps": 10000,
                "export_manifest_sha256": export_sha, "backup": str(backup)})
            commit()
            metrics = run_two_phase_evaluation(export_path=export, export_sha256=export_sha,
                manifest_path=data["val_manifest"], manifest_sha256=data["val_manifest_sha256"],
                train_manifest_path=data["train_manifest"], train_manifest_sha256=data["train_manifest_sha256"],
                out_dir=base / "s0_val18", device="cuda", batch_size=8, normalization="prepared")
            status = {"status": "COMPLETED", "steps": 10000, "export_manifest_sha256": export_sha,
                "metrics": metrics, "backup": str(backup), "checkpoint": report["final_checkpoint_path"]}
        atomic_json(base / "S0_STATUS.json", status)
        return status
    except BudgetBoundary as exc:
        path = Path(str(exc))
        meta = json.loads((path / "MANIFEST.json").read_text())
        status = {"status": "STOPPED_COST_BOUNDARY", "steps": meta["step"], "checkpoint": str(path),
                  "checkpoint_manifest_sha256": sha256(path / "MANIFEST.json"), "resume_required": True}
        atomic_json(base / "S0_STATUS.json", status)
        return status
    except Exception as exc:
        atomic_json(base / "S0_FAILURE.json", {"status": "FAILED", "type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        stop.set()
        watcher.join(timeout=12)
        commit()
