"""Fresh ACDC/S0 workflow, independent of historical teacher restoration."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("selfaudit-fresh-s0")
volume = modal.Volume.from_name("selfaudit-assets", create_if_missing=False)
cpu_image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("rclone")
    .pip_install("torch==2.14.0", "numpy==2.4.6", "scipy>=1.13,<2", "nibabel==5.4.2", "psutil>=5.9,<7")
    .add_local_dir(ROOT/"self_audit", "/repo/self_audit", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file(ROOT/"prepare_acdc.py", "/repo/prepare_acdc.py")
    .add_local_file(ROOT/"train.py", "/repo/train.py")
    .add_local_file(ROOT/"configs/fresh_s0/acdc_l8_historical_train.json", "/repo/configs/fresh_s0/acdc_l8_historical_train.json")
    .add_local_file(ROOT/"configs/fresh_s0/supervised_wide_seed42.json", "/repo/configs/fresh_s0/supervised_wide_seed42.json")
    .add_local_file(ROOT/"splits/acdc_reviewed_locked_train80_val18.json", "/repo/splits/acdc_reviewed_locked_train80_val18.json")
    .add_local_file(ROOT/"results/compact_acdc_l8_20261029/experiment_20261028/val18_native.json", "/repo/historical_val18_native.json"))


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=3600, max_containers=1,
              volumes={"/assets": volume}, secrets=[modal.Secret.from_name("selfaudit-rclone")])
def prepare_data(namespace: str, observed_inventory: bool = False):
    import sys
    sys.path.insert(0, "/repo")
    from self_audit.modal_data_prep import prepare_on_modal
    try:
        return prepare_on_modal(namespace, use_observed_inventory=observed_inventory)
    finally:
        volume.commit()


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=1800, single_use_containers=True,
              volumes={"/assets": volume})
def verify_data(namespace: str):
    import sys
    sys.path.insert(0, "/repo")
    volume.reload()
    from self_audit.modal_data_prep import verify_fresh_container
    result = verify_fresh_container(namespace)
    volume.commit()
    return result


@app.function(image=cpu_image, cpu=(2, 2), memory=(8192, 8192), timeout=1800,
              single_use_containers=True, volumes={"/assets": volume})
def smoke_s0(namespace: str, git_sha: str, config: dict):
    import sys
    sys.path.insert(0, "/repo")
    volume.reload()
    from self_audit.modal_s0 import cpu_smoke
    return cpu_smoke(namespace, git_sha, config, volume.commit)


@app.function(image=cpu_image, cpu=(2, 2), memory=(16384, 16384), gpu="A100-40GB",
              timeout=7200, retries=0, max_containers=1, scaledown_window=2,
              single_use_containers=True, volumes={"/assets": volume})
def train_s0(namespace: str, git_sha: str, config: dict, budget: dict,
             resume: str = "", resume_sha256: str = ""):
    import sys
    sys.path.insert(0, "/repo")
    volume.reload()
    from self_audit.modal_s0 import run_s0
    return run_s0(namespace, git_sha, config, budget, volume.commit,
                  resume=resume or None, resume_sha256=resume_sha256 or None)


@app.local_entrypoint()
def main(namespace: str, action: str = "prepare", observed_inventory: bool = False,
         credit_usd: float = 0.0, spent_this_round_usd: float = 0.0,
         resume: str = "", resume_sha256: str = ""):
    if action not in {"prepare", "verify", "smoke", "train"}:
        raise ValueError("action must be prepare, verify, smoke or train")
    if action == "prepare":
        print(json.dumps(prepare_data.remote(namespace, observed_inventory), indent=2))
    result = verify_data.remote(namespace)
    print(json.dumps(result, indent=2))
    if action in {"smoke", "train"}:
        if result["status"] != "PASS":
            raise ValueError("DATA_PREP is not PASS; no GPU allocated")
        import subprocess
        import sys
        sys.path.insert(0, str(ROOT))
        from self_audit.modal_s0 import budget_plan, s0_config
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
            raise ValueError("commit the isolated worktree before locking a real S0 run")
        budget = budget_plan(credit_usd, spent_this_round_usd)
        config = s0_config(budget["training_seconds"])
        if action == "smoke":
            print(json.dumps(smoke_s0.remote(namespace, git_sha, config), indent=2))
        else:
            # Validate smoke on CPU before allocating any GPU, including its source/config identity.
            vol_smoke = json.loads(b"".join(volume.read_file(namespace.removeprefix("/assets/") + "/CPU_SMOKE.json")))
            from self_audit.asset_staging import source_digest
            assert vol_smoke["status"] == "PASS" and vol_smoke["git_sha"] == git_sha
            assert vol_smoke["config"] == config and vol_smoke["source_sha256"] == source_digest(ROOT)
            print(json.dumps({"budget": budget, "git_sha": git_sha}), flush=True)
            print(json.dumps(train_s0.with_options(timeout=budget["function_timeout_seconds"]).remote(
                namespace, git_sha, config, budget, resume, resume_sha256), indent=2))
