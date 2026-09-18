"""Shared schema and source-lineage firewall for generation artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from self_audit_maskfree.data.firewall import forbidden_reason


FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "gt", "mask", "masks", "label", "labels", "annotation", "annotations",
        "groundtruth", "ground_truth", "dice", "hungarian", "oracle", "reference", "scribble",
    }
)
FORBIDDEN_PATH_TOKENS = (
    "mask", "label", "annotation", "groundtruth", "ground_truth", "gt", "reference", "scribble",
)


class FirewallError(ValueError):
    """Raised when a generation manifest exposes non-image-only evidence."""


def _forbidden_key(key: object) -> bool:
    normalized = str(key).lower()
    return normalized in FORBIDDEN_EXACT_KEYS or (
        normalized.endswith(("_path", "_root", "_file"))
        and any(token in normalized for token in FORBIDDEN_PATH_TOKENS)
    )


def validate_image_only_value(value: Any, *, where: str = "manifest") -> None:
    """Reject GT-shaped fields recursively, including innocently nested ones."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _forbidden_key(key):
                raise FirewallError(f"forbidden generation field at {where}.{key}")
            validate_image_only_value(child, where=f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_image_only_value(child, where=f"{where}[{index}]")


def validate_image_only_source_locator(locator: str) -> None:
    """Apply FreeMask's path-shaped annotation exclusion to a source locator."""
    reason = forbidden_reason(Path(locator))
    if reason is not None:
        raise FirewallError(f"source locator is not image-only: {reason}")


def validate_image_only_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate fields and every shared source locator before generation."""
    validate_image_only_value(manifest)
    for record in manifest.get("records", []):
        source = record.get("source", {})
        locator = source.get("locator")
        if not isinstance(locator, str) or not locator:
            raise FirewallError("shared record has no source locator")
        validate_image_only_source_locator(locator)
