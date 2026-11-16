#!/usr/bin/env python3
"""Provision and verify the external source dependencies used by Candidate PG.

The default command is read-only. ``--install`` explicitly clones/checks out
the two source repositories and installs their Python requirements without
touching the installed torch distribution. ``--download-cinema`` explicitly
downloads exactly the two allowed Hub files into the normal HF cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MEDNEXT_URL = "https://github.com/MIC-DKFZ/MedNeXt.git"
MEDNEXT_REVISION = "0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba"
CINEMA_URL = "https://github.com/mathpluscode/CineMA.git"
CINEMA_GITHUB_COMMIT = "c10daa1d93f0ea28d8b9ad9206b0f673d25805c1"
HF_REPO = "mathpluscode/CineMA"
# Independently queried from https://huggingface.co/api/models/mathpluscode/CineMA
# (HTTP 200, API field "sha"). Hub and GitHub revisions are distinct. Do not
# infer either pin from the other; this value is passed only to the Hub loader.
CINEMA_HF_REVISION = "b1251ee50423bceeca84c080782fc3bc7756dea6"
HF_CONFIG = "pretrained/config.yaml"
HF_WEIGHTS = "pretrained/cinema.safetensors"
# File metadata from the Hub tree API at CINEMA_HF_REVISION. The config OID
# is Git blob SHA-1, not a guessed/raw SHA-256; compute and record raw SHA-256
# for the actual file bytes separately. The weight LFS oid is its SHA-256.
CONFIG_GIT_BLOB = "8a465e51889187cf1056fc4feaef9a0182feabe4"
CONFIG_SIZE = 348
WEIGHTS_SHA256 = "a841b7e9300d60ea8feb73e3a7d8c60ddedc1d1d49ec1049ebfcfa337c5d5590"
WEIGHTS_SIZE = 502749000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_output(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def clone_pin(url: str, revision: str, path: Path) -> None:
    if path.exists() and not (path / ".git").exists():
        raise RuntimeError(f"refusing to replace non-git source path: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", url, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", revision], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True)
    head = git_output(path, "rev-parse", "HEAD")
    if head != revision:
        raise RuntimeError(f"checkout mismatch at {path}: {head} != {revision}")


def verify_checkout(path: Path, url: str, revision: str) -> dict[str, Any]:
    if not path.is_dir():
        return {"path": str(path), "status": "MISSING", "expected_revision": revision}
    head = git_output(path, "rev-parse", "HEAD")
    origin = git_output(path, "remote", "get-url", "origin")
    dirty = bool(git_output(path, "status", "--porcelain", "--untracked-files=all"))
    return {
        "path": str(path), "url": origin, "expected_url": url,
        "revision": head, "expected_revision": revision,
        "clean": not dirty,
        "status": "READY" if head == revision and origin.rstrip("/") == url.rstrip("/") and not dirty else "MISMATCH",
    }


def resolve_hf_file(filename: str, *, download: bool) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        if download:
            raise RuntimeError("install huggingface_hub before --download-cinema") from None
        return None
    try:
        path = hf_hub_download(
            repo_id=HF_REPO, filename=filename, revision=CINEMA_HF_REVISION,
            local_files_only=not download,
        )
    except Exception:
        if download:
            raise
        return None
    return Path(path).resolve()


def verify_cinema(*, download: bool) -> dict[str, Any]:
    files: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    for name in (HF_CONFIG, HF_WEIGHTS):
        path = resolve_hf_file(name, download=download)
        if path is None:
            files[name] = {"status": "MISSING", "sha256": None}
            continue
        digest = sha256_file(path)
        expected = CONFIG_GIT_BLOB if name == HF_CONFIG else WEIGHTS_SHA256
        expected_size = CONFIG_SIZE if name == HF_CONFIG else WEIGHTS_SIZE
        actual_size = path.stat().st_size
        # The config is a Git blob whose object ID is not the raw SHA-256.
        # A Hub-cache object includes its source commit in the path, so bind its
        # raw bytes in the receipt even when checking the upstream blob ID.
        if name == HF_CONFIG:
            git_blob = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0" + path.read_bytes()).hexdigest()
            ok = git_blob == expected and actual_size == expected_size
            files[name] = {
                "path": str(path), "size_bytes": actual_size, "sha256": digest,
                "git_blob_sha1": git_blob,
                "expected_git_blob_sha1": expected,
                "expected_size_bytes": expected_size,
                "status": "READY" if ok else "HASH_MISMATCH",
            }
        else:
            ok = digest == expected and actual_size == expected_size
            files[name] = {
                "path": str(path), "size_bytes": actual_size, "sha256": digest,
                "expected_sha256": expected,
                "expected_size_bytes": expected_size,
                "status": "READY" if ok else "HASH_MISMATCH",
            }
        artifacts.append({
            "source": f"hf://{HF_REPO}@{CINEMA_HF_REVISION}/{name}",
            "path": str(path), "sha256": digest,
            "expected_source_oid": expected,
            "source_oid_algorithm": "git-blob-sha1" if name == HF_CONFIG else "sha256",
            "verified": bool(ok),
        })
    return {
        "repo": HF_REPO, "revision": CINEMA_HF_REVISION,
        "revision_source": "Hugging Face model API field sha",
        "files": files, "artifacts": artifacts,
    }


def install_python_dependencies(*, cinema_repo: Path) -> None:
    # Install only missing non-torch runtime dependencies. Never let pip
    # resolve/reinstall torch, CUDA, or any package pinned by requirements.txt.
    required = {
        "huggingface-hub": "huggingface_hub",
        "safetensors": "safetensors",
        "PyYAML": "yaml",
        "omegaconf": "omegaconf",
        "monai": "monai",
        "timm": "timm",
        "einops": "einops",
    }
    missing = []
    for distribution, module in required.items():
        try:
            importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", *missing], check=True)
    # CineMA is imported directly from its pinned source tree; do not pip-install
    # its unconstrained dependency file, which can replace torch/CUDA packages.
    del cinema_repo


def build_receipt(root: Path, *, download_cinema: bool) -> dict[str, Any]:
    mednext_path = root / "external" / "MedNeXt"
    cinema_path = root / "external" / "CineMA"
    mednext = verify_checkout(mednext_path, MEDNEXT_URL, MEDNEXT_REVISION)
    cinema_code = verify_checkout(cinema_path, CINEMA_URL, CINEMA_GITHUB_COMMIT)
    cinema_weights = verify_cinema(download=download_cinema)
    all_file_statuses = [item.get("status") for item in cinema_weights["files"].values()]
    ready = mednext["status"] == "READY" and cinema_code["status"] == "READY" and all(
        value == "READY" for value in all_file_statuses
    )
    return {
        "schema": "self_audit.candidate_pg.source_setup.v1",
        "status": "READY" if ready else "NOT_READY",
        "root": str(root.resolve()),
        "sources": {"mednext": mednext, "cinema_code": cinema_code, "cinema_pretrained": cinema_weights},
        "artifacts": cinema_weights["artifacts"],
        "allowed_cinema_files_only": [HF_CONFIG, HF_WEIGHTS],
        "config_binding": {
            "teacher.source_repo": "external/MedNeXt (relative to Candidate PG repository root)",
            "teacher.source_commit": MEDNEXT_REVISION,
            "teacher.weights": None,
            "teacher.weights_sha256": None,
            "cinema_init.hf_repo": HF_REPO,
            "cinema_init.hf_revision": CINEMA_HF_REVISION,
            "cinema_init.hf_revision_source": "Hugging Face API sha, queried independently of GitHub source commit",
            "cinema_init.files": [HF_WEIGHTS, HF_CONFIG],
            "cinema_init.source_paths": "omit; real loader uses pinned HF cache",
            "environment": {
                "PYTHONPATH": "prepend external/CineMA so cinema.mae.mae imports from the pinned checkout",
                "HF_HOME": "optional persistent Hugging Face cache root",
                "MEDNEXT_SOURCE_ROOT": "not consumed by loader; teacher.source_repo remains external/MedNeXt",
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="PG repository root")
    parser.add_argument("--install", action="store_true", help="clone pinned source checkouts and install missing non-torch dependencies")
    parser.add_argument("--download-cinema", action="store_true", help="download only the two pinned CineMA files")
    parser.add_argument("--receipt", type=Path, help="write JSON receipt (default: reports/pg-server-20261116/SOURCE_READY.json)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.install:
        clone_pin(MEDNEXT_URL, MEDNEXT_REVISION, root / "external" / "MedNeXt")
        clone_pin(CINEMA_URL, CINEMA_GITHUB_COMMIT, root / "external" / "CineMA")
        install_python_dependencies(cinema_repo=root / "external" / "CineMA")
    receipt = build_receipt(root, download_cinema=args.download_cinema)
    target = args.receipt or root / "reports" / "pg-server-20261116" / "SOURCE_READY.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
