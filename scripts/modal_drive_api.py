"""Direct Drive API CPU staging; no rclone binary in the image."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("selfaudit-drive-api")
volume = modal.Volume.from_name("selfaudit-assets", create_if_missing=False)
base_image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("google-api-python-client==2.201.0", "google-auth-httplib2==0.4.4", "google-auth==2.62.0"))
image = (base_image
    .add_local_file(ROOT / "self_audit/__init__.py", "/repo/self_audit/__init__.py")
    .add_local_file(ROOT / "self_audit/asset_staging.py", "/repo/self_audit/asset_staging.py")
    .add_local_file(ROOT / "self_audit/drive_api_staging.py", "/repo/self_audit/drive_api_staging.py"))
training_image = (base_image
    .pip_install("torch==2.14.0", "numpy==2.4.6", "scipy>=1.13,<2", "nibabel==5.4.2", "psutil>=5.9,<7")
    .add_local_dir(ROOT / "self_audit", "/repo/self_audit", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file(ROOT / "prepare_acdc.py", "/repo/prepare_acdc.py")
    .add_local_file(ROOT / "train.py", "/repo/train.py")
    .add_local_dir(ROOT / "configs/fresh_s0", "/repo/configs/fresh_s0")
    .add_local_file(ROOT / "splits/acdc_reviewed_locked_train80_val18.json", "/repo/splits/acdc_reviewed_locked_train80_val18.json")
    .add_local_file(ROOT / "results/compact_acdc_l8_20261029/experiment_20261028/val18_native.json", "/repo/historical_val18_native.json"))


@app.function(image=image, cpu=(1,1), memory=(1024,1024), timeout=240, retries=0,
    single_use_containers=True, max_containers=1, volumes={"/assets": volume},
    secrets=[modal.Secret.from_name("selfaudit-rclone")])
def stage_smoke(namespace: str):
    import sys
    sys.path.insert(0, "/repo")
    from self_audit.drive_api_staging import smoke
    result = smoke(namespace)
    volume.commit()
    return result


@app.function(image=image, cpu=(1,1), memory=(1024,1024), timeout=120, retries=0,
    single_use_containers=True, volumes={"/assets": volume})
def verify_smoke(namespace: str):
    import sys
    sys.path.insert(0, "/repo")
    volume.reload()
    from self_audit.drive_api_staging import verify_smoke as verify
    result = verify(namespace)
    volume.commit()
    return result


@app.function(image=training_image, cpu=(2,2), memory=(8192,8192), timeout=3600, retries=0,
    single_use_containers=True, max_containers=1, volumes={"/assets": volume},
    secrets=[modal.Secret.from_name("selfaudit-rclone")])
def stage_data(namespace: str):
    import sys
    sys.path.insert(0, "/repo")
    volume.reload()
    from self_audit.drive_api_staging import transfer_and_prepare
    return transfer_and_prepare(namespace, volume.commit)


@app.function(image=training_image, cpu=(2,2), memory=(8192,8192), timeout=1800, retries=0,
    single_use_containers=True, volumes={"/assets": volume})
def verify_data(namespace: str):
    import sys
    sys.path.insert(0, "/repo")
    volume.reload()
    from self_audit.modal_data_prep import verify_fresh_container
    result = verify_fresh_container(namespace)
    volume.commit()
    return result


@app.function(image=training_image, cpu=(2,2), memory=(8192,8192), timeout=1800, retries=0,
    single_use_containers=True, volumes={"/assets": volume})
def smoke_s0(namespace: str, git_sha: str, config: dict):
    import sys
    sys.path.insert(0, "/repo")
    volume.reload()
    from self_audit.modal_s0 import cpu_smoke
    return cpu_smoke(namespace, git_sha, config, volume.commit)


@app.function(image=training_image, cpu=(2,2), memory=(16384,16384), gpu="A100-40GB",
    timeout=7200, retries=0, max_containers=1, scaledown_window=2, single_use_containers=True,
    volumes={"/assets": volume})
def train_s0(namespace: str, git_sha: str, config: dict, budget: dict,
             resume: str = "", resume_sha256: str = ""):
    import sys
    sys.path.insert(0, "/repo")
    volume.reload()
    from self_audit.modal_s0 import run_s0
    return run_s0(namespace, git_sha, config, budget, volume.commit,
                  resume=resume or None, resume_sha256=resume_sha256 or None)


@app.function(image=image, cpu=(0.125,0.125), memory=(256,256), timeout=10800, retries=0,
    max_containers=1, single_use_containers=True, volumes={"/assets": volume})
def pipeline(namespace: str, secret_name: str, git_sha: str, config: dict, budget: dict):
    """Remote controller survives a detached client's disconnect; no credentials attached."""
    import os, sys
    sys.path.insert(0, "/repo")
    from self_audit.asset_staging import atomic_json
    from self_audit.drive_api_staging import historical_kd_prerequisites
    base = Path(namespace)
    base.mkdir(parents=True, exist_ok=True)
    # O_EXCL: a second submission cannot launch a duplicate S0.
    with (base / "API_PIPELINE_LAUNCH.json").open("x") as f:
        json.dump({"git_sha": git_sha, "budget": budget, "app_id": os.environ.get("MODAL_APP_ID")}, f)
    volume.commit()
    kd = historical_kd_prerequisites()
    status = {"DATA_PREP": "RUNNING", "S0": "BLOCKED_DATA_PREP", "S1": kd["status"],
              "S2": kd["status"], "KD_prerequisites": kd, "git_sha": git_sha}
    def record():
        atomic_json(base / "PIPELINE_STATUS.json", status)
        volume.commit()
        print(json.dumps(status), flush=True)
    record()
    try:
        secrets = [modal.Secret.from_name(secret_name)]
        copied = stage_smoke.with_options(secrets=secrets).remote(namespace)
        if copied["status"] != "COPIED_PENDING_FRESH_CONTAINER":
            status.update(DATA_PREP="BLOCKED", error=copied.get("error")); record(); return status
        if verify_smoke.remote(namespace)["status"] != "PASS":
            status["DATA_PREP"] = "BLOCKED_SMOKE_PERSISTENCE"; record(); return status
        staged = stage_data.with_options(secrets=secrets).remote(namespace)
        if staged["status"] != "PREPARED_PENDING_FRESH_CONTAINER":
            status.update(DATA_PREP="BLOCKED", error=staged.get("error")); record(); return status
        verified = verify_data.remote(namespace)
        if verified["status"] != "PASS":
            status["DATA_PREP"] = "BLOCKED_PERSISTENCE"; record(); return status
        status.update(DATA_PREP="READY", S0="CPU_SMOKE"); record()
        smoke_result = smoke_s0.remote(namespace, git_sha, config)
        if smoke_result["status"] != "PASS":
            status["S0"] = "BLOCKED_CPU_SMOKE"; record(); return status
        status["S0"] = "RUNNING"; record()
        trained = train_s0.with_options(timeout=budget["function_timeout_seconds"]).remote(namespace, git_sha, config, budget)
        status["S0"] = trained["status"]
        status["S0_result"] = trained
        status["DRIVE_BACKUP"] = "DRIVE_BACKUP_PENDING"
        record()
        return status
    except Exception as exc:
        status["S0"] = "BLOCKED_EXECUTION" if status["DATA_PREP"] == "READY" else "BLOCKED_DATA_PREP"
        if status["DATA_PREP"] == "RUNNING":
            status["DATA_PREP"] = "BLOCKED"
        status["error_type"] = type(exc).__name__
        record()
        return status


@app.local_entrypoint()
def main(namespace: str, action: str = "smoke", secret_name: str = "selfaudit-rclone",
         credit_usd: float = 0.0, spent_this_round_usd: float = 0.0,
         resume: str = "", resume_sha256: str = ""):
    p = Path(namespace)
    if not p.is_relative_to("/assets/fresh_s0") or ".." in p.parts or p == Path("/assets/fresh_s0"):
        raise ValueError("Use an isolated namespace below /assets/fresh_s0")
    if action == "smoke":
        receipt = stage_smoke.with_options(secrets=[modal.Secret.from_name(secret_name)]).remote(namespace)
        print(json.dumps(receipt, indent=2))
        if receipt["status"] == "COPIED_PENDING_FRESH_CONTAINER":
            print(json.dumps(verify_smoke.remote(namespace), indent=2))
    elif action in ("pipeline", "resume"):
        import subprocess, sys
        sys.path.insert(0, str(ROOT))
        from self_audit.modal_s0 import budget_plan, s0_config
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
            raise ValueError("Commit source before launching the locked pipeline")
        budget = budget_plan(credit_usd, spent_this_round_usd)
        if action == "resume":
            if not resume or not resume_sha256:
                raise ValueError("Explicit checkpoint directory and manifest SHA256 are required for resume")
            call = train_s0.with_options(timeout=budget["function_timeout_seconds"]).spawn(
                namespace, git_sha, s0_config(), budget, resume, resume_sha256)
        else:
            if resume or resume_sha256:
                raise ValueError("Use action=resume; never relaunch the pipeline over an existing S0")
            call = pipeline.spawn(namespace, secret_name, git_sha, s0_config(), budget)
        print(json.dumps({"function_call_id": call.object_id, "namespace": namespace}), flush=True)
        print(json.dumps(call.get(), indent=2))
    else:
        raise ValueError("action must be smoke, pipeline or resume")
