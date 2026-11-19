"""Modal launcher for Self-Audit W2S-Match (A100-40GB).

Usage (from the repo root on the Mac):
  modal run scripts/modal_app.py::pull_drive --folder-id <DRIVE_FOLDER_ID> --dest raw/ACDC
  modal run scripts/modal_app.py::cpu --cmd "python -m pytest -q"
  modal run --detach scripts/modal_app.py::gpu --cmd "<training command>"
  modal volume get self-audit-mount runs/w2s-match-20261119 ./runs-from-modal

The volume is mounted at /root/mount_data, the same path the old vast server
used, so existing configs (/root/mount_data/prepared, /root/mount_data/runs)
keep working.
"""

import os
import subprocess
import threading
from pathlib import Path

import modal

REPO_LOCAL = os.environ.get("SELF_AUDIT_REPO", str(Path(__file__).resolve().parents[1]))
REPO = "/root/Self-Audit"
MOUNT = "/root/mount_data"

vol = modal.Volume.from_name("self-audit-mount", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "rclone", "zstd", "unzip", "rsync")
    .pip_install(
        "torch>=2.10,<3",
        "numpy>=1.26,<3",
        "scipy>=1.13,<2",
        "nibabel>=5.2,<6",
        "pytest>=8,<10",
        "huggingface_hub",
        "safetensors",
        "pyyaml",
        "omegaconf",
    )
    # MedNeXt teacher factory (module `nnunet_mednext`, used by candidate_a3d/teacher.py)
    .pip_install("git+https://github.com/MIC-DKFZ/MedNeXt.git")
    # CineMA source model (`cinema.mae.mae.get_model`, used by candidate_pg/cinema_init.py)
    .pip_install("git+https://github.com/mathpluscode/CineMA.git")
    .env({"HF_HOME": f"{MOUNT}/hf", "PYTHONPATH": REPO, "PYTHONUNBUFFERED": "1"})
    # Code is attached at container start (no image rebuild when code changes).
    .add_local_dir(
        REPO_LOCAL,
        REPO,
        ignore=[".git", "**/__pycache__", "**/.pytest_cache", ".venv", "runs", "**/*.pt", "**/*.zip"],
    )
)

app = modal.App("self-audit-w2s", image=image)


def _sh(cmd: str, cwd: str = REPO) -> None:
    """Run a shell command, committing the volume every 10 minutes and at exit."""
    print(f"$ {cmd}", flush=True)
    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(600):
            try:
                vol.commit()
            except Exception as exc:  # keep training alive if a commit fails
                print(f"[modal] volume commit warning: {exc}", flush=True)

    threading.Thread(target=committer, daemon=True).start()
    try:
        code = subprocess.run(["bash", "-lc", cmd], cwd=cwd).returncode
    finally:
        stop.set()
        vol.commit()
    if code:
        raise SystemExit(code)


@app.function(
    volumes={MOUNT: vol},
    secrets=[modal.Secret.from_name("rclone-gdrive")],
    cpu=2,
    timeout=4 * 3600,
)
def pull_drive(folder_id: str, dest: str) -> None:
    """Copy one Google Drive folder (by ID) into /root/mount_data/<dest>. CPU only."""
    target = f"{MOUNT}/{dest}"
    _sh(
        f"rclone copy gdrive: '{target}' --drive-root-folder-id {folder_id} "
        f"--transfers 8 --checkers 16 --stats 30s --stats-one-line",
        cwd="/",
    )
    _sh(f"du -sh '{target}'; find '{target}' -type f | wc -l", cwd="/")


@app.function(volumes={MOUNT: vol}, cpu=4, memory=16384, timeout=6 * 3600)
def cpu(cmd: str) -> None:
    """Cheap CPU container: tests, data preparation, unzip, analysis."""
    _sh(cmd)


@app.function(
    volumes={MOUNT: vol},
    gpu="A100-40GB",
    cpu=4,
    memory=24576,
    timeout=24 * 3600,
    # retries=0 on purpose: a retry re-runs the same command, which is only safe
    # once the training command resumes from its last checkpoint.
)
def gpu(cmd: str) -> None:
    """A100-40GB container for training. Run with `modal run --detach`."""
    _sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    _sh(cmd)
