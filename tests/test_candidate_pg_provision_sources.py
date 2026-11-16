"""Offline tests for the Candidate PG source provisioning CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path



SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pg_provision_sources.py"
SPEC = importlib.util.spec_from_file_location("pg_provision_sources", SCRIPT)
assert SPEC and SPEC.loader
pgsrc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pgsrc)


def test_hub_revision_is_separate_from_verified_github_commit():
    assert pgsrc.CINEMA_GITHUB_COMMIT == "c10daa1d93f0ea28d8b9ad9206b0f673d25805c1"
    assert pgsrc.CINEMA_HF_REVISION == "b1251ee50423bceeca84c080782fc3bc7756dea6"
    assert pgsrc.CINEMA_GITHUB_COMMIT != pgsrc.CINEMA_HF_REVISION


def _git_repo(path: Path, url: str, *, revision: str) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "fixture.txt").write_text("source fixture\n")
    subprocess.run(["git", "-C", str(path), "add", "fixture.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    # Rewrite the commit identity fixture so verification can use an arbitrary
    # pinned 40-hex revision without a remote operation.
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    assert len(actual) == 40 and len(revision) == 40
    subprocess.run(["git", "-C", str(path), "remote", "add", "origin", url], check=True)


def test_sha256_and_checkout_verification(tmp_path):
    path = tmp_path / "repo"
    revision = "a" * 40
    _git_repo(path, "https://example.invalid/source.git", revision=revision)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    result = pgsrc.verify_checkout(path, "https://example.invalid/source.git", actual)
    assert result["status"] == "READY"
    assert result["clean"] is True
    assert pgsrc.verify_checkout(path, "https://wrong.invalid/source.git", actual)["status"] == "MISMATCH"


def test_cinema_verifies_only_two_allowed_files(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  size: base\n", encoding="utf-8")
    weights = tmp_path / "cinema.safetensors"
    weights.write_bytes(b"weights fixture")
    monkeypatch.setattr(pgsrc, "CONFIG_GIT_BLOB", "0" * 40)
    monkeypatch.setattr(pgsrc, "WEIGHTS_SHA256", hashlib.sha256(weights.read_bytes()).hexdigest())
    monkeypatch.setattr(pgsrc, "WEIGHTS_SIZE", weights.stat().st_size)
    cache = {pgsrc.HF_CONFIG: config, pgsrc.HF_WEIGHTS: weights}
    monkeypatch.setattr(pgsrc, "resolve_hf_file", lambda filename, *, download: cache[filename])
    verified = pgsrc.verify_cinema(download=False)
    assert set(verified["files"]) == {pgsrc.HF_CONFIG, pgsrc.HF_WEIGHTS}
    assert verified["files"][pgsrc.HF_WEIGHTS]["status"] == "READY"
    assert verified["files"][pgsrc.HF_CONFIG]["status"] == "HASH_MISMATCH"


def test_cinema_ready_hashes_and_download_opt_in_are_bound(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config_bytes = b"official config fixture\n"
    config.write_bytes(config_bytes)
    weights = tmp_path / "cinema.safetensors"
    weights.write_bytes(b"official weight fixture")
    blob = hashlib.sha1(b"blob " + str(len(config_bytes)).encode() + b"\0" + config_bytes).hexdigest()
    monkeypatch.setattr(pgsrc, "CONFIG_GIT_BLOB", blob)
    monkeypatch.setattr(pgsrc, "WEIGHTS_SHA256", hashlib.sha256(weights.read_bytes()).hexdigest())
    monkeypatch.setattr(pgsrc, "CONFIG_SIZE", config.stat().st_size)
    monkeypatch.setattr(pgsrc, "WEIGHTS_SIZE", weights.stat().st_size)
    calls = []
    cache = {pgsrc.HF_CONFIG: config, pgsrc.HF_WEIGHTS: weights}

    def resolve(name, *, download):
        calls.append((name, download))
        return cache[name]

    monkeypatch.setattr(pgsrc, "resolve_hf_file", resolve)
    verified = pgsrc.verify_cinema(download=True)
    assert all(item["status"] == "READY" for item in verified["files"].values())
    assert calls == [(pgsrc.HF_CONFIG, True), (pgsrc.HF_WEIGHTS, True)]
    assert verified["revision"] == pgsrc.CINEMA_HF_REVISION
    assert len(verified["artifacts"]) == 2
    assert {item["path"] for item in verified["artifacts"]} == {str(config.resolve()), str(weights.resolve())}
    assert all(item["verified"] and len(item["sha256"]) == 64 for item in verified["artifacts"])


def test_receipt_not_ready_without_sources_or_weight_files(tmp_path):
    receipt = pgsrc.build_receipt(tmp_path, download_cinema=False)
    assert receipt["status"] == "NOT_READY"
    assert receipt["config_binding"]["teacher.source_repo"] == "external/MedNeXt (relative to Candidate PG repository root)"
    assert receipt["config_binding"]["cinema_init.source_paths"].startswith("omit")
    assert receipt["config_binding"]["cinema_init.hf_revision"] == pgsrc.CINEMA_HF_REVISION
    assert receipt["config_binding"]["cinema_init.hf_revision"] == receipt["sources"]["cinema_pretrained"]["revision"]
    assert receipt["artifacts"] == receipt["sources"]["cinema_pretrained"]["artifacts"]


def test_cli_dry_run_writes_receipt_and_returns_not_ready(tmp_path, capsys):
    receipt = tmp_path / "receipt.json"
    code = pgsrc.main(["--root", str(tmp_path), "--receipt", str(receipt)])
    assert code == 2
    parsed = json.loads(receipt.read_text(encoding="utf-8"))
    assert parsed["schema"] == "self_audit.candidate_pg.source_setup.v1"
    assert parsed["status"] == "NOT_READY"
    assert "NOT_READY" in capsys.readouterr().out
