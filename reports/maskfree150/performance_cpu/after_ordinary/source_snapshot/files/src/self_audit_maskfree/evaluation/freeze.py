"""Freeze-manifest validation shared by verification and the isolated evaluator.

W7 owns the freeze schema and its single authoritative validator,
``self_audit_maskfree.export.validate_freeze``.  This module only adapts that
validator's successful result into the receipt shape consumed by the data and
evaluation layers; it has no fallback validator that could accept a weaker or
different freeze representation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class FreezeValidationError(RuntimeError):
    """Raised when a frozen prediction set cannot be trusted."""


CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Stream a file's SHA256 so a large export does not have to fit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_freeze_manifest(path: str | Path) -> dict[str, Any]:
    """Read a freeze manifest from disk without validating it."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FreezeValidationError(f"freeze manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise FreezeValidationError("freeze manifest must be a JSON object")
    # Keep the decoded object semantically identical to the producer payload.
    # The authoritative validator hashes the original mapping, so adding a
    # convenience key here would invalidate its manifest-id check. W7
    # manifests carry their own root; older callers can pass root= explicitly.
    return manifest


def _hash_entries(section: Any, label: str) -> list[tuple[str, str]]:
    """Normalize the two shapes a hash section may take into (path, sha256) pairs."""
    if section is None:
        return []
    entries: list[tuple[str, str]] = []
    if isinstance(section, Mapping):
        items: Iterable[Any] = section.items()
        for path, value in items:
            if isinstance(value, Mapping):
                digest = value.get("sha256")
            else:
                digest = value
            if not isinstance(digest, str):
                raise FreezeValidationError(f"{label} entry {path!r} has no sha256 string")
            entries.append((str(path), digest))
        return entries
    if isinstance(section, Sequence) and not isinstance(section, (str, bytes)):
        for entry in section:
            if not isinstance(entry, Mapping):
                raise FreezeValidationError(f"{label} entries must be objects with path and sha256")
            path = entry.get("path")
            digest = entry.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                raise FreezeValidationError(f"{label} entry is missing path or sha256")
            entries.append((path, digest))
        return entries
    raise FreezeValidationError(f"{label} must be a mapping or a list of objects")


def validate_freeze_manifest(
    manifest: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    required_methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Re-hash every claimed file and confirm the compared method set is complete.

    Parameters
    ----------
    manifest:
        A manifest produced by W7's ``freeze_predictions`` (or any mapping with
        ``files``/``checkpoints`` hash sections).
    root:
        Base directory for relative paths. Defaults to the manifest's own
        ``output_dir``/``root``, then to the manifest file's directory.
    required_methods:
        Method names that must all appear in the frozen comparison set. The
        freeze must enumerate *every* compared method, otherwise a later
        reference evaluation could quietly compare a subset.

    Returns a receipt describing exactly what was verified. Raises
    :class:`FreezeValidationError` on any missing file, hash mismatch, or missing
    method.
    """
    if not isinstance(manifest, Mapping):
        raise FreezeValidationError("freeze manifest must be a mapping")

    base = root
    if base is None:
        base = manifest.get("output_dir") or manifest.get("root")
    base_path = Path(base) if base is not None else Path.cwd()

    try:  # W7 owns the only accepted freeze validator.
        from ..export import validate_freeze as external_validator  # type: ignore
    except Exception as error:  # pragma: no cover - package layout failure
        raise FreezeValidationError(
            "authoritative export.validate_freeze is unavailable; refusing to use a fallback"
        ) from error
    try:
        external_validator(manifest)
    except FreezeValidationError:
        raise
    except Exception as error:
        # Keep the evaluation-facing exception stable while preserving the
        # owner's validation message and exception as the cause.
        raise FreezeValidationError(str(error)) from error

    # W7 already validated this canonical flat map and every claimed byte.
    # Retain a compact receipt: repeating the entire corpus inventory in every
    # unit result would create quadratic report size on a full cine dataset.
    files = manifest.get("files", {})
    if not isinstance(files, Mapping) or not files:
        raise FreezeValidationError("freeze manifest lists no frozen files")
    method_names = sorted(map(str, manifest.get("compared_methods", [])))
    if required_methods is not None:
        missing = sorted(set(map(str, required_methods)) - set(method_names))
        if missing:
            raise FreezeValidationError(
                "freeze does not enumerate every compared method; missing: " + ", ".join(missing)
            )

    return {
        "validated": True,
        "freeze_id": str(manifest.get("freeze_id") or ""),
        "dataset": manifest.get("dataset"),
        "protocol": manifest.get("protocol"),
        "epoch": manifest.get("epoch"),
        "root": str(base_path),
        "files_checked": len(files),
        "methods": method_names,
        "external_validator_used": True,
    }
