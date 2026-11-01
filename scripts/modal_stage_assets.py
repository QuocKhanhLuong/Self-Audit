#!/usr/bin/env python3
"""CPU-only Modal staging. The Drive Secret is scoped to stage_assets alone."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("selfaudit-asset-staging")
volume = modal.Volume.from_name("selfaudit-assets", create_if_missing=True, version=2)
image = (modal.Image.debian_slim(python_version="3.11").apt_install("rclone")
         .add_local_file(ROOT / "self_audit/asset_staging.py", "/opt/staging/asset_staging.py"))


@app.function(image=image, cpu=1, memory=1024, timeout=3600, max_containers=1,
              volumes={"/assets": volume}, secrets=[modal.Secret.from_name("selfaudit-rclone")])
def stage_assets(action: str = "discover", request: dict | None = None):
    import sys
    sys.path.insert(0, "/opt/staging")
    from asset_staging import stage
    result = stage("/assets", action, request or {})
    volume.commit()
    return result


@app.function(image=image, cpu=1, memory=1024, timeout=1800,
              volumes={"/assets": volume}, single_use_containers=True)
def verify_assets():
    import os
    import sys
    sys.path.insert(0, "/opt/staging")
    from asset_staging import atomic_json, verify_files, sha256
    if "RCLONE_CONFIG_TEXT" in os.environ or "RCLONE_REMOTE_NAME" in os.environ:
        raise RuntimeError("verification container must not receive Drive Secret")
    volume.reload()
    p = Path("/assets/receipts/STAGING.json")
    receipt = json.loads(p.read_text())
    failures = verify_files("/assets", receipt)
    container_id = os.environ["MODAL_TASK_ID"]
    new_container = container_id != receipt["stage_container"]
    receipt["persistence_verified"] = new_container and not failures
    receipt["verification"] = {"container": container_id, "new_container": new_container,
                               "files_read_and_hashed": len(receipt["files"]), "failures": failures,
                               "drive_secret_present": False}
    if failures or not new_container:
        receipt["status"] = "BLOCKED"
    atomic_json(p, receipt)
    volume.commit()
    return {"status": receipt["status"], "missing": receipt["missing"],
            "persistence_verified": receipt["persistence_verified"], "verification": receipt["verification"],
            "receipt_sha256": sha256(p), "files": receipt["files"], "error": receipt.get("error")}


@app.local_entrypoint()
def main(action: str = "discover", request_file: str = ""):
    request = json.loads(Path(request_file).read_text()) if request_file else {}
    if action == "inventory":
        print(json.dumps(verify_assets.remote(), indent=2))
    elif action == "batch":
        for operation in request["actions"]:
            print(json.dumps(stage_assets.remote(operation["action"], operation["request"]), indent=2))
        print(json.dumps(verify_assets.remote(), indent=2))
    else:
        print(json.dumps(stage_assets.remote(action, request), indent=2))
        print(json.dumps(verify_assets.remote(), indent=2))
