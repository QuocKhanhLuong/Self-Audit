"""Immutable best-snapshot selection and public alias commit primitives.

Per-file atomicity is not a transaction: writing a new ``best.pt`` and then
failing to write ``last.pt`` leaves the previous ``last.pt`` pointing at bytes
that no longer exist anywhere (audit plan R9).  This module supplies the
filesystem half of a recoverable commit protocol that removes that window.

The protocol the trainer is expected to run, in this order:

1. Save the epoch's candidate with the ordinary ``save_checkpoint`` to a
   **unique, never-reused** path ``<checkpoint_dir>/selected_best/epoch_<n>_<uuid>.pt``.
2. Build a reference to it with :func:`make_best_reference`.
3. Embed that reference in the ``last.pt`` payload and commit ``last.pt``
   **first**.
4. Only then call :func:`publish_best_alias` to move the public ``best.pt``
   alias onto the selected snapshot.

What each failure point costs:

* Failure before step 3 leaves the old ``last.pt`` referencing its own
  untouched snapshot; the orphan candidate is inert bytes.
* Failure during step 3 likewise leaves the old ``last.pt`` intact, because
  the trainer commits it with the existing atomic writer.
* Failure during or after step 4 leaves ``last.pt`` already carrying a
  verified immutable reference, so the next resume resolves that reference and
  repairs the public alias by calling :func:`publish_best_alias` again.

Recovery therefore never depends on in-process rollback, which cannot run
under ``SIGKILL`` or power loss -- it depends only on durable bytes plus a
hash.

Design constraints this module holds to:

* Stdlib only.  It imports neither :mod:`torch` nor
  ``self_audit.training._utils`` nor ``self_audit.training.unified_trainer``,
  so it introduces no circular dependency and, crucially, **no second torch
  serialization policy**: it treats checkpoints as opaque bytes that
  ``save_checkpoint`` already produced and validated.
* Fail closed.  Every reference field is validated before the filesystem is
  touched, and the snapshot hash is verified before any consumer loads it.
* Nothing is ever deleted.  Snapshot garbage collection is deliberately out of
  scope; an old snapshot may still be referenced by an older ``last.pt``.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "SELECTED_BEST_DIRNAME",
    "PUBLIC_BEST_NAME",
    "SelectionError",
    "ReferenceValidationError",
    "SnapshotIntegrityError",
    "AliasPublishError",
    "RelocationConflictError",
    "REFERENCE_FIELDS",
    "make_best_reference",
    "resolve_best_reference",
    "publish_best_alias",
    "relocate_best_reference",
    "best_alias_matches_reference",
]

SCHEMA_VERSION = 1
SELECTED_BEST_DIRNAME = "selected_best"
PUBLIC_BEST_NAME = "best.pt"

REFERENCE_FIELDS = ("schema_version", "path", "sha256", "epoch", "metric")

_HASH_CHUNK = 1 << 20
_HEX_DIGITS = frozenset("0123456789abcdef")


class SelectionError(Exception):
    """Base class for every failure raised by this module."""


class ReferenceValidationError(SelectionError):
    """A reference is malformed, or names a path outside the snapshot store."""


class SnapshotIntegrityError(SelectionError):
    """A snapshot is missing, is not a regular file, or does not match its hash."""


class AliasPublishError(SelectionError):
    """The public ``best.pt`` alias could not be published safely."""


class RelocationConflictError(SelectionError):
    """The relocation target already holds different or unrelated bytes."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Hash a file in chunks, without loading a whole checkpoint into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_directory(value: Any, name: str) -> Path:
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
    else:
        raise ReferenceValidationError(
            f"{name} must be a path, got {type(value).__name__}"
        )
    if not path.is_dir():
        raise ReferenceValidationError(f"{name} is not an existing directory: {path}")
    return path.resolve()


def _require_regular_file(path: Path, description: str) -> None:
    """Refuse anything that is not a plain file, symlinks included.

    A symlink is refused rather than followed: the whole point of an immutable
    snapshot is that the reference plus the hash fully determine the bytes, and
    a link is a second, mutable indirection that a hash cannot pin.
    """

    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise SnapshotIntegrityError(f"{description} does not exist: {path}") from exc
    except OSError as exc:
        raise SnapshotIntegrityError(f"{description} is not accessible: {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise SnapshotIntegrityError(
            f"{description} is a symbolic link, which cannot be pinned by hash: {path}"
        )
    if not stat.S_ISREG(info.st_mode):
        raise SnapshotIntegrityError(f"{description} is not a regular file: {path}")


def _require_epoch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReferenceValidationError(
            f"epoch must be an int, got {type(value).__name__}: {value!r}"
        )
    if value < 1:
        raise ReferenceValidationError(f"epoch must be a positive integer, got {value}")
    return int(value)


def _require_metric(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceValidationError(
            f"metric must be a finite real number, got {type(value).__name__}: {value!r}"
        )
    metric = float(value)
    if not math.isfinite(metric):
        raise ReferenceValidationError(f"metric must be finite, got {value!r}")
    return metric


def _require_digest(value: Any) -> str:
    if not isinstance(value, str):
        raise ReferenceValidationError(
            f"sha256 must be a string, got {type(value).__name__}"
        )
    if len(value) != 64 or not set(value) <= _HEX_DIGITS:
        raise ReferenceValidationError(
            f"sha256 must be 64 lowercase hex characters, got {value!r}"
        )
    return value


def _relative_snapshot_path(value: Any) -> PurePosixPath:
    """Validate the stored path *as text*, before it can touch the filesystem."""

    if not isinstance(value, str):
        raise ReferenceValidationError(
            f"path must be a string, got {type(value).__name__}"
        )
    if not value:
        raise ReferenceValidationError("path must not be empty")
    if "\\" in value:
        raise ReferenceValidationError(f"path must use forward slashes, got {value!r}")
    if "\x00" in value:
        raise ReferenceValidationError("path must not contain a null byte")
    # Validate the *raw* segments: PurePosixPath silently drops a "." segment,
    # so normalizing first would let "selected_best/./x.pt" through.
    if any(segment in ("", ".", "..") for segment in value.split("/")):
        raise ReferenceValidationError(
            f"path must not contain empty, '.' or '..' segments, got {value!r}"
        )
    relative = PurePosixPath(value)
    if relative.is_absolute():
        raise ReferenceValidationError(f"path must be relative, got {value!r}")
    parts = relative.parts
    if len(parts) < 2 or parts[0] != SELECTED_BEST_DIRNAME:
        raise ReferenceValidationError(
            f"path must name a file under {SELECTED_BEST_DIRNAME}/, got {value!r}"
        )
    return relative


def _require_snapshot_store(checkpoint_dir: Path) -> Path:
    """Return the snapshot store root, refusing a store that is not really there.

    Containment of a snapshot is checked against the *resolved* store, so the
    store root itself has to be a genuine directory inside ``checkpoint_dir``.
    Otherwise the store can be moved out of the checkpoint root and replaced by
    a symlink to its new location: every relative path still resolves "inside
    the store", and every hash still matches, while the bytes actually live
    outside the directory the operator pointed at. Refuse that here, once, for
    every entry point.

    A store that does not exist yet is allowed: a relocation destination
    legitimately gets its store created by the copy.
    """

    store = checkpoint_dir / SELECTED_BEST_DIRNAME
    if not os.path.lexists(store):
        return store
    info = os.lstat(store)
    if stat.S_ISLNK(info.st_mode):
        raise ReferenceValidationError(
            f"snapshot store {store} is a symbolic link; the immutable store must be a real "
            f"directory inside {checkpoint_dir} so that a reference cannot be redirected outside it"
        )
    if not stat.S_ISDIR(info.st_mode):
        raise ReferenceValidationError(f"snapshot store {store} is not a directory")
    if os.path.realpath(store) != str(store):
        raise ReferenceValidationError(
            f"snapshot store {store} resolves outside the checkpoint directory "
            f"({os.path.realpath(store)})"
        )
    return store


def _contained_target(relative: PurePosixPath, checkpoint_dir: Path) -> Path:
    """Join a validated relative path and prove the result stays in the store.

    The textual checks above already reject ``..`` and absolute paths; this
    second check is against the *resolved* path, so a symlinked intermediate
    directory cannot smuggle the target outside the snapshot store either.
    """

    store = _require_snapshot_store(checkpoint_dir)
    candidate = checkpoint_dir.joinpath(*relative.parts)
    resolved_store = os.path.realpath(store)
    resolved_parent = os.path.realpath(candidate.parent)
    if resolved_parent != resolved_store and not resolved_parent.startswith(
        resolved_store + os.sep
    ):
        raise ReferenceValidationError(
            f"path escapes the snapshot store: {relative} resolves outside {store}"
        )
    return Path(resolved_parent) / candidate.name


def _validated_fields(reference: Mapping[str, Any]) -> tuple[PurePosixPath, str, int, float]:
    if not isinstance(reference, Mapping):
        raise ReferenceValidationError(
            f"reference must be a mapping, got {type(reference).__name__}"
        )
    keys = set(reference)
    expected = set(REFERENCE_FIELDS)
    missing = sorted(expected - keys)
    if missing:
        raise ReferenceValidationError(f"reference is missing field(s): {missing}")
    unknown = sorted(keys - expected)
    if unknown:
        raise ReferenceValidationError(f"reference carries unknown field(s): {unknown}")

    schema_version = reference["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ReferenceValidationError(
            f"schema_version must be an int, got {type(schema_version).__name__}"
        )
    if schema_version != SCHEMA_VERSION:
        raise ReferenceValidationError(
            f"unsupported reference schema_version {schema_version}, expected {SCHEMA_VERSION}"
        )

    relative = _relative_snapshot_path(reference["path"])
    digest = _require_digest(reference["sha256"])
    epoch = _require_epoch(reference["epoch"])
    metric = _require_metric(reference["metric"])
    return relative, digest, epoch, metric


def _reference_dict(
    relative: PurePosixPath, digest: str, epoch: int, metric: float
) -> dict[str, str | int | float]:
    return {
        "schema_version": SCHEMA_VERSION,
        "path": relative.as_posix(),
        "sha256": digest,
        "epoch": epoch,
        "metric": metric,
    }


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync so a rename survives a power loss."""

    fd = None
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        os.fsync(fd)
    except (OSError, AttributeError):
        pass
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _discard_temporary(temporary: Path | None) -> None:
    """Remove a partial temporary without ever masking the primary error."""

    if temporary is None:
        return
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _copy_verified(source: Path, target: Path, expected_digest: str) -> None:
    """Copy ``source`` onto ``target`` atomically, proving the bytes are identical.

    The digest of what actually landed in the temporary is recomputed and
    compared against ``expected_digest``: if the source changed while it was
    being read -- exactly what an immutable snapshot must never do -- the copy
    is discarded and the existing target is left alone.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            digest = hashlib.sha256()
            with open(source, "rb") as reader:
                while True:
                    chunk = reader.read(_HASH_CHUNK)
                    if not chunk:
                        break
                    digest.update(chunk)
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        written = digest.hexdigest()
        if written != expected_digest:
            raise SnapshotIntegrityError(
                f"Snapshot bytes changed while copying {source}: expected sha256 "
                f"{expected_digest}, read {written}; refusing to publish"
            )
        os.replace(temporary, target)
        temporary = None
    except BaseException:
        _discard_temporary(temporary)
        raise
    _fsync_directory(target.parent)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_best_reference(
    snapshot_path: str | os.PathLike[str],
    checkpoint_dir: str | os.PathLike[str],
    *,
    epoch: int,
    metric: float,
) -> dict[str, str | int | float]:
    """Describe an immutable selected-best snapshot as a portable reference.

    The returned mapping holds only ``str``/``int``/``float`` values, so it can
    be embedded verbatim in a ``last.pt`` payload under ``weights_only=True``
    and in a JSON report.

    ``metric`` must be finite: an undefined or non-finite validation metric can
    never improve the best, so it must never reach a selection reference.  The
    snapshot itself is read to hash it and is otherwise untouched.
    """

    directory = _require_directory(checkpoint_dir, "checkpoint_dir")
    validated_epoch = _require_epoch(epoch)
    validated_metric = _require_metric(metric)

    if not isinstance(snapshot_path, (str, os.PathLike)):
        raise ReferenceValidationError(
            f"snapshot_path must be a path, got {type(snapshot_path).__name__}"
        )
    snapshot = Path(snapshot_path)
    _require_regular_file(snapshot, "Selected-best snapshot")

    store = _require_snapshot_store(directory)
    resolved_store = os.path.realpath(store)
    resolved_parent = os.path.realpath(snapshot.parent)
    if resolved_parent != resolved_store and not resolved_parent.startswith(
        resolved_store + os.sep
    ):
        raise ReferenceValidationError(
            f"Selected-best snapshot {snapshot} is not inside {store}; a snapshot "
            "outside the immutable store cannot be referenced"
        )

    absolute = Path(resolved_parent) / snapshot.name
    relative = PurePosixPath(
        os.path.relpath(absolute, directory).replace(os.sep, "/")
    )
    # Re-run the textual path checks on what will actually be stored, so a
    # reference this function emits is always one ``resolve_best_reference``
    # accepts.
    relative = _relative_snapshot_path(relative.as_posix())
    digest = _sha256_file(absolute)
    return _reference_dict(relative, digest, validated_epoch, validated_metric)


def resolve_best_reference(
    reference: Mapping[str, Any], checkpoint_dir: str | os.PathLike[str]
) -> Path:
    """Validate a reference and return the verified snapshot path.

    Field shapes and the path are checked before the filesystem is touched,
    then the snapshot's hash is recomputed and compared.  The hash is therefore
    verified *before* any consumer deserializes the file.

    This function deliberately says nothing about the checkpoint's contents.
    Run identity, execution config, state digest and selection metadata are
    payload-level facts the caller must check after its own safe load.
    """

    directory = _require_directory(checkpoint_dir, "checkpoint_dir")
    relative, digest, _epoch, _metric = _validated_fields(reference)
    target = _contained_target(relative, directory)
    _require_regular_file(target, "Referenced selected-best snapshot")
    observed = _sha256_file(target)
    if observed != digest:
        raise SnapshotIntegrityError(
            f"Selected-best snapshot {target} does not match its reference: expected "
            f"sha256 {digest}, computed {observed}; refusing to use a modified snapshot"
        )
    return target


def publish_best_alias(
    reference: Mapping[str, Any], checkpoint_dir: str | os.PathLike[str]
) -> Path:
    """Point the public ``best.pt`` alias at the verified immutable snapshot.

    The snapshot is resolved and hash-verified, streamed into a temporary in
    the same directory as ``best.pt``, re-hashed, flushed, fsynced, and only
    then moved into place with :func:`os.replace`.  A failure at any step
    leaves the previous ``best.pt`` byte-identical and removes the temporary,
    and a cleanup failure never masks the original error.

    Calling this again with the same reference is safe and is exactly how a
    resume repairs an alias that a crash left stale after ``last.pt`` had
    already committed.
    """

    directory = _require_directory(checkpoint_dir, "checkpoint_dir")
    _relative, digest, _epoch, _metric = _validated_fields(reference)
    _require_snapshot_store(directory)
    source = resolve_best_reference(reference, directory)
    alias = directory / PUBLIC_BEST_NAME

    # Never publish through a link: os.replace would swap the link itself, so
    # the operator's intent is ambiguous and the alias contract is not ours.
    if alias.is_symlink():
        raise AliasPublishError(
            f"Refusing to publish over the symbolic link {alias}: the public best "
            "alias must be a regular file"
        )
    if alias.exists() and not alias.is_file():
        raise AliasPublishError(f"Public best alias {alias} is not a regular file")

    try:
        _copy_verified(source, alias, digest)
    except SelectionError:
        raise
    except OSError as exc:
        raise AliasPublishError(
            f"Failed to publish best alias {alias} from {source}: {exc}"
        ) from exc
    return alias


def relocate_best_reference(
    reference: Mapping[str, Any],
    source_dir: str | os.PathLike[str],
    target_dir: str | os.PathLike[str],
) -> dict[str, str | int | float]:
    """Copy a verified snapshot into another checkpoint directory, atomically.

    The snapshot keeps its relative path, so the returned reference is
    identical to the input and is valid for ``target_dir``.  The source is left
    in place and re-verified afterwards, so a relocation can never be the thing
    that invalidates the directory being relocated *from*.

    A target that already holds byte-identical content is accepted as already
    relocated.  Anything else -- different bytes, a directory, a symlink --
    raises :class:`RelocationConflictError` rather than being overwritten.
    Publishing the public alias in the target directory stays a separate,
    explicit :func:`publish_best_alias` call.
    """

    source_directory = _require_directory(source_dir, "source_dir")
    target_directory = _require_directory(target_dir, "target_dir")
    _require_snapshot_store(target_directory)
    relative, digest, epoch, metric = _validated_fields(reference)
    source = resolve_best_reference(reference, source_directory)

    # The destination must be contained the same way the source is: a plain
    # joinpath writes through a symlinked intermediate directory (a target
    # ``selected_best/nested`` pointing outside the checkpoint root), so the
    # copy would land outside the directory the operator named.
    target = _contained_target(relative, target_directory)
    if os.path.lexists(target):
        # The link/regular-file check comes first: a symlink pointing at the
        # source would otherwise pass a "same file" shortcut and be silently
        # accepted as an already-relocated snapshot.
        info = os.lstat(target)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RelocationConflictError(
                f"Relocation target {target} exists and is not a regular file; "
                "refusing to replace an unrelated entry"
            )
        if os.path.realpath(target) == os.path.realpath(source):
            # Literally the same file: nothing to copy.
            return _reference_dict(relative, digest, epoch, metric)
        existing = _sha256_file(target)
        if existing != digest:
            raise RelocationConflictError(
                f"Relocation target {target} already holds different bytes: expected "
                f"sha256 {digest}, found {existing}; refusing to clobber it"
            )
        return _reference_dict(relative, digest, epoch, metric)

    try:
        _copy_verified(source, target, digest)
    except SelectionError:
        raise
    except OSError as exc:
        raise SnapshotIntegrityError(
            f"Failed to relocate snapshot {source} to {target}: {exc}"
        ) from exc

    # The copy must be exact, and the source must have survived it untouched.
    landed = _sha256_file(target)
    if landed != digest:
        raise SnapshotIntegrityError(
            f"Relocated snapshot {target} has sha256 {landed}, expected {digest}"
        )
    surviving = _sha256_file(source)
    if surviving != digest:
        raise SnapshotIntegrityError(
            f"Source snapshot {source} changed during relocation: expected sha256 "
            f"{digest}, computed {surviving}"
        )
    return _reference_dict(relative, digest, epoch, metric)


def best_alias_matches_reference(
    reference: Mapping[str, Any], checkpoint_dir: str | os.PathLike[str]
) -> bool:
    """Report whether ``best.pt`` already holds the referenced snapshot's bytes.

    Lets a resume decide whether the alias needs repairing without loading
    anything.  A missing or non-regular alias is simply ``False``; a malformed
    reference still raises, because an unreadable reference is never an
    "already fine" answer.
    """

    directory = _require_directory(checkpoint_dir, "checkpoint_dir")
    _relative, digest, _epoch, _metric = _validated_fields(reference)
    _require_snapshot_store(directory)
    alias = directory / PUBLIC_BEST_NAME
    if alias.is_symlink() or not alias.is_file():
        return False
    return _sha256_file(alias) == digest
