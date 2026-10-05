from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from setup_adnet_source import audit_adnet_source_tree
from shared_benchmark.adnet_fewshot import file_sha256


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=str(root), check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _fixture_repo(tmp_path: Path) -> tuple[Path, dict]:
    repo = tmp_path / "ADNet"
    repo.mkdir()
    (repo / "models").mkdir()
    (repo / "models" / "fewshot_anom.py").write_text("class FewShotSeg: pass\n", encoding="utf-8")
    (repo / "main_train.py").write_text("print('train')\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    files = []
    for relative in ("main_train.py", "models/fewshot_anom.py"):
        path = repo / relative
        files.append({"path": relative, "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
    index = {
        "schema": "adnet.source_index.v1",
        "remote": "file://" + str(repo),
        "source_root": "baseline/ADNet",
        "commit": commit,
        "file_count": len(files),
        "files": files,
    }
    return repo, index


def test_audit_adnet_source_tree_accepts_indexed_git_tree(tmp_path: Path):
    repo, index = _fixture_repo(tmp_path)
    result = audit_adnet_source_tree(adnet_root=repo, source_index=index)
    assert result["status"] == "READY"
    assert result["verified_file_count"] == 2


def test_audit_adnet_source_tree_rejects_hash_mismatch(tmp_path: Path):
    repo, index = _fixture_repo(tmp_path)
    index = json.loads(json.dumps(index))
    index["files"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="source audit failed"):
        audit_adnet_source_tree(adnet_root=repo, source_index=index)
