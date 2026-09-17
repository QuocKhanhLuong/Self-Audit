"""Data-layer supervision firewall (W3).

Every byte that enters the mask-free pipeline passes through this module. The
rule it enforces is narrow and mechanical: *this package never opens a manual
segmentation mask, and never opens an annotation sidecar that encodes a
cohort-selection decision made from masks or diagnosis*.

That second clause matters as much as the first. ACDC's ``Info.cfg`` contains
the ED/ES frame indices and the pathology group. Using it to pick which frames
exist would silently import an annotation-derived cohort rule into an
"image-only" pipeline, so the file is forbidden here even though it is not a
mask.

The checks are path-shaped, which is a real limitation: a mask stored under an
innocuous name would not be caught. The mitigation is that discovery only ever
*enumerates* files it selected itself, and the loaders refuse any path that did
not come from discovery.
"""
from __future__ import annotations

from pathlib import Path

#: Directory names that hold manual annotation in ACDC / M&Ms style trees.
FORBIDDEN_DIR_NAMES: frozenset[str] = frozenset(
    {
        "mask",
        "masks",
        "label",
        "labels",
        "gt",
        "ground_truth",
        "groundtruth",
        "annotation",
        "annotations",
        "segmentation",
        "segmentations",
    }
)

#: Exact file names that encode an annotation-derived cohort decision.
FORBIDDEN_FILE_NAMES: frozenset[str] = frozenset({"info.cfg", "diagnosis.csv"})

#: Substrings in a file name that mark a manual annotation volume.
FORBIDDEN_NAME_TOKENS: tuple[str, ...] = (
    "_gt.",
    "_gt_",
    "_mask",
    "_label",
    "_seg.",
    "_seg_",
    "_manual",
    "groundtruth",
    "ground_truth",
)

#: Image containers this data layer is allowed to read.
ALLOWED_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii", ".npy")


class MaskAccessError(PermissionError):
    """Raised when the mask-free data layer is asked to open forbidden data."""


def _lower_name(path: Path) -> str:
    return path.name.lower()


def forbidden_reason(path: str | Path) -> str | None:
    """Return why ``path`` is forbidden to this package, or ``None`` if allowed.

    The whole path is inspected, not just the leaf, so a legitimately named
    volume that lives inside ``masks/`` is still rejected.
    """
    p = Path(path)
    for part in p.parts[:-1]:
        if part.lower() in FORBIDDEN_DIR_NAMES:
            return f"path traverses annotation directory {part!r}"
    name = _lower_name(p)
    if name in FORBIDDEN_FILE_NAMES:
        return f"{p.name} is an annotation sidecar (cohort/diagnosis metadata)"
    if p.is_dir() and name in FORBIDDEN_DIR_NAMES:
        return f"{p.name} is an annotation directory"
    for token in FORBIDDEN_NAME_TOKENS:
        if token in name:
            return f"file name contains annotation token {token!r}"
    return None


def is_image_only_path(path: str | Path) -> bool:
    return forbidden_reason(path) is None


def assert_image_only(path: str | Path) -> Path:
    """Gate every read. Raises :class:`MaskAccessError` for annotation paths."""
    reason = forbidden_reason(path)
    if reason is not None:
        raise MaskAccessError(f"mask-free data layer refused {path}: {reason}")
    return Path(path)


def has_allowed_suffix(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in ALLOWED_SUFFIXES)
