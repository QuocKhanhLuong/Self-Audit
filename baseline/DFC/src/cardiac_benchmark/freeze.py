"""Hash-bound raw-artifact freeze; fixture and scientific namespaces never mix."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from .output import AttemptLedger
from .provenance import canonical_json, file_sha256, sha256_json

class FreezeError(ValueError): pass


def create_freeze(root: str | Path, *, scope: str, expected_sample_ids: list[str], entries: list[Mapping[str, Any]], manifest_hash: str, config_hash: str, code_identity: str, environment_id: str, config_path: str | None = None) -> dict[str, Any]:
    if scope not in {"fixture", "scientific"}: raise FreezeError("invalid scope")
    ledger = AttemptLedger(expected_sample_ids)
    for entry in entries: ledger.terminal(str(entry["sample_id"]), str(entry["status"]), **{k:v for k,v in entry.items() if k not in {"sample_id","status"}})
    ledger.assert_complete()
    config_receipt = {"config_hash": config_hash}
    if config_path is not None:
        config = Path(config_path)
        if not config.is_file(): raise FreezeError("bound config file is missing")
        config_receipt.update({"path": str(config), "file_hash": file_sha256(config)})
    freeze = {"freeze_version": "dfc-raw-freeze-v1", "scope": scope, "fixture_freeze": scope == "fixture", "expected_sample_ids": sorted(expected_sample_ids), "manifest_hash": manifest_hash, "config": config_receipt, "code_identity": code_identity, "environment_id": environment_id, "entries": ledger.attempts}
    freeze["scientific_payload_hash"] = sha256_json(freeze)
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    (root / "freeze_manifest.json").write_bytes(canonical_json(freeze))
    return freeze


def validate_freeze(freeze: Mapping[str, Any], root: str | Path, *, scientific: bool = False) -> None:
    if scientific and (freeze.get("scope") != "scientific" or freeze.get("fixture_freeze")):
        raise FreezeError("fixture freeze can never validate as scientific")
    payload = dict(freeze); observed = payload.pop("scientific_payload_hash", None)
    if observed != sha256_json(payload): raise FreezeError("freeze metadata mutation detected")
    config = freeze.get("config", {})
    if config.get("path") and file_sha256(config["path"]) != config.get("file_hash"):
        raise FreezeError("bound config file was mutated")
    ledger = AttemptLedger(list(freeze.get("expected_sample_ids", [])))
    root = Path(root)
    for entry in freeze.get("entries", []):
        ledger.terminal(entry["sample_id"], entry["status"], **{k:v for k,v in entry.items() if k not in {"sample_id","status","attempt_id"}})
        if entry["status"] == "success":
            part = entry.get("partition", {}); raw = root / entry.get("artifact_dir", "") / part.get("path", "")
            if not raw.is_file() or file_sha256(raw) != part.get("file_hash"): raise FreezeError("raw map missing or mutated")
            metadata = entry.get("metadata", {})
            if metadata:
                meta = root / entry.get("artifact_dir", "") / metadata.get("path", "")
                if not meta.is_file() or file_sha256(meta) != metadata.get("file_hash"):
                    raise FreezeError("metadata missing or mutated")
    ledger.assert_complete()


def can_resume(existing: Mapping[str, Any], required: Mapping[str, Any]) -> bool:
    keys = {"sample_id", "manifest_hash", "scope", "source_hash", "input_hash", "config_hash", "sample_seed", "code_identity", "environment_id"}
    return existing.get("status") == "success" and all(existing.get(k) == required.get(k) for k in keys) and existing.get("scope") == required.get("scope")
