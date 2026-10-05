#!/usr/bin/env python3
"""Clone/pin and audit the upstream ADNet source tree.

This script intentionally keeps ADNet as an external upstream checkout under
``baseline/ADNet`` instead of vendoring the full source into this repository.
It verifies the checkout against ``reports/ADNET_SOURCE_INDEX_2026-10-01.json``
and writes a receipt that can be attached to the GPU audit run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shared_benchmark.adnet_fewshot import atomic_write_json, file_sha256  # noqa: E402
from shared_benchmark.provenance import sha256_json  # noqa: E402


DEFAULT_INDEX = ROOT / "reports" / "ADNET_SOURCE_INDEX_2026-10-01.json"
DEFAULT_RECEIPT = ROOT / "reports" / "adnet_source_setup_receipt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adnet-root", type=Path, default=ROOT / "baseline" / "ADNet")
    parser.add_argument("--source-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--remote")
    parser.add_argument("--commit")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--force-checkout", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = setup_adnet_source(
        adnet_root=args.adnet_root,
        source_index_path=args.source_index,
        receipt_path=args.receipt,
        remote=args.remote,
        commit=args.commit,
        skip_clone=args.skip_clone,
        force_checkout=args.force_checkout,
    )
    print(json.dumps({
        "status": receipt["status"],
        "adnet_root": receipt["adnet_root"],
        "commit": receipt["commit"],
        "receipt": str(args.receipt),
    }, sort_keys=True))
    return 0


def setup_adnet_source(
    *,
    adnet_root: Path,
    source_index_path: Path,
    receipt_path: Path,
    remote: str | None = None,
    commit: str | None = None,
    skip_clone: bool = False,
    force_checkout: bool = False,
) -> dict[str, Any]:
    source_index = _load_source_index(source_index_path)
    expected_remote = str(remote or source_index["remote"])
    expected_commit = str(commit or source_index["commit"])
    adnet_root = adnet_root.resolve()

    setup_actions: list[dict[str, Any]] = []
    if not adnet_root.exists():
        if skip_clone:
            raise RuntimeError(f"ADNet source root is missing and --skip-clone was set: {adnet_root}")
        adnet_root.parent.mkdir(parents=True, exist_ok=True)
        setup_actions.append(_run(["git", "clone", expected_remote, str(adnet_root)], cwd=ROOT))
    elif not (adnet_root / ".git").is_dir():
        raise RuntimeError(f"ADNet source root exists but is not a git checkout: {adnet_root}")

    current_commit = _git(adnet_root, "rev-parse", "HEAD")
    if current_commit != expected_commit:
        dirty = _git(adnet_root, "status", "--porcelain")
        if dirty and not force_checkout:
            raise RuntimeError(
                "ADNet checkout is dirty and not at the expected commit. "
                "Commit/clean it or rerun with --force-checkout."
            )
        if not _commit_available(adnet_root, expected_commit):
            setup_actions.append(_run(["git", "fetch", "origin", expected_commit], cwd=adnet_root))
        checkout_cmd = ["git", "checkout", expected_commit]
        if force_checkout:
            checkout_cmd.insert(2, "--force")
        setup_actions.append(_run(checkout_cmd, cwd=adnet_root))

    audit = audit_adnet_source_tree(adnet_root=adnet_root, source_index=source_index)
    receipt = {
        "schema": "adnet.source_setup_receipt.v1",
        "status": "READY",
        "adnet_root": str(adnet_root),
        "source_index": str(source_index_path.resolve()),
        "source_index_sha256": file_sha256(source_index_path),
        "source_index_scientific_sha256": sha256_json(source_index),
        "remote": expected_remote,
        "commit": expected_commit,
        "git_remote_origin": _git(adnet_root, "config", "--get", "remote.origin.url", allow_failure=True),
        "git_dirty": bool(_git(adnet_root, "status", "--porcelain")),
        "setup_actions": setup_actions,
        "audit": audit,
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


def audit_adnet_source_tree(*, adnet_root: Path, source_index: Mapping[str, Any]) -> dict[str, Any]:
    expected_files = source_index.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise RuntimeError("ADNet source index requires a non-empty files list")
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    verified = 0
    for row in expected_files:
        if not isinstance(row, Mapping):
            raise RuntimeError("ADNet source index file entries must be objects")
        relative = str(row.get("path", ""))
        expected_sha = str(row.get("sha256", ""))
        if not relative or not expected_sha:
            raise RuntimeError("ADNet source index file entry requires path and sha256")
        path = adnet_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        observed_sha = file_sha256(path)
        if observed_sha != expected_sha:
            mismatched.append({
                "path": relative,
                "expected_sha256": expected_sha,
                "observed_sha256": observed_sha,
            })
            continue
        verified += 1
    commit = _git(adnet_root, "rev-parse", "HEAD")
    expected_commit = str(source_index.get("commit", ""))
    status = "READY" if not missing and not mismatched and commit == expected_commit else "FAILED"
    audit = {
        "schema": "adnet.source_tree_audit.v1",
        "status": status,
        "expected_commit": expected_commit,
        "observed_commit": commit,
        "expected_file_count": int(source_index.get("file_count", len(expected_files))),
        "verified_file_count": verified,
        "missing": missing,
        "mismatched": mismatched,
    }
    if status != "READY":
        raise RuntimeError(f"ADNet source audit failed: {audit}")
    return audit


def _load_source_index(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read ADNet source index: {path}") from exc
    if not isinstance(value, dict) or value.get("schema") != "adnet.source_index.v1":
        raise RuntimeError("unknown ADNet source index schema")
    if not isinstance(value.get("remote"), str) or not isinstance(value.get("commit"), str):
        raise RuntimeError("ADNet source index requires remote and commit")
    return value


def _commit_available(root: Path, commit: str) -> bool:
    result = _run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root, check=False)
    return result["returncode"] == 0


def _git(root: Path, *args: str, allow_failure: bool = False) -> str:
    result = _run(["git", *args], cwd=root, check=not allow_failure)
    return str(result["stdout"]).strip()


def _run(command: list[str], *, cwd: Path, check: bool = True) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=str(cwd), check=False, capture_output=True, text=True)
    result = {
        "command": command,
        "cwd": str(cwd),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed: {command}\n{completed.stderr}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
