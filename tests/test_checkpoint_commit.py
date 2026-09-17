"""Tests for the immutable selected-best / public-alias commit primitives.

These exercise the *filesystem* half of the Wave 4 commit protocol only: byte
fixtures stand in for checkpoints, because nothing in
``src/self_audit/training/checkpoint_commit.py`` interprets checkpoint content.
Pairing the protocol with real models, resume and calibration is the main
trainer work.

Reference environment: Python 3.10.21 / torch 2.4.1 (CPU), single-threaded.
"""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.self_audit.training.checkpoint_commit import (
    PUBLIC_BEST_NAME,
    SCHEMA_VERSION,
    SELECTED_BEST_DIRNAME,
    AliasPublishError,
    ReferenceValidationError,
    RelocationConflictError,
    SnapshotIntegrityError,
    best_alias_matches_reference,
    make_best_reference,
    publish_best_alias,
    relocate_best_reference,
    resolve_best_reference,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _store(checkpoint_dir: Path) -> Path:
    path = checkpoint_dir / SELECTED_BEST_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_snapshot(checkpoint_dir: Path, epoch: int, payload: bytes, token: str = "aaaa") -> Path:
    """Write a uniquely named snapshot, mimicking epoch_<epoch>_<uuid>.pt."""

    path = _store(checkpoint_dir) / f"epoch_{epoch}_{token}.pt"
    path.write_bytes(payload)
    return path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _temp_leftovers(directory: Path) -> list[str]:
    return sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.name.startswith(".") and entry.name.endswith(".tmp")
    )


@pytest.fixture()
def checkpoint_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "run"
    _store(directory)
    return directory


# ---------------------------------------------------------------------------
# 1. make_best_reference / resolve_best_reference happy path
# ---------------------------------------------------------------------------

def test_make_reference_preserves_expected_sha_and_snapshot_bytes(checkpoint_dir: Path) -> None:
    """The reference records the exact snapshot digest and never rewrites the file."""

    payload = b"selected-best-epoch-3-payload"
    snapshot = _write_snapshot(checkpoint_dir, 3, payload)
    before = snapshot.stat()

    reference = make_best_reference(snapshot, checkpoint_dir, epoch=3, metric=0.8125)

    assert reference == {
        "schema_version": SCHEMA_VERSION,
        "path": f"{SELECTED_BEST_DIRNAME}/epoch_3_aaaa.pt",
        "sha256": _sha(payload),
        "epoch": 3,
        "metric": 0.8125,
    }
    # weights_only-safe and JSON-safe: only str/int/float leaves.
    assert all(isinstance(value, (str, int, float)) for value in reference.values())
    assert snapshot.read_bytes() == payload
    assert snapshot.stat().st_size == before.st_size

    resolved = resolve_best_reference(reference, checkpoint_dir)
    assert resolved.read_bytes() == payload
    assert resolved == snapshot.resolve()


def test_make_reference_accepts_nested_snapshot_and_integer_metric(checkpoint_dir: Path) -> None:
    """Sub-directories under the store are allowed; an int metric becomes a float."""

    nested = _store(checkpoint_dir) / "interval_1"
    nested.mkdir()
    snapshot = nested / "epoch_9_bbbb.pt"
    snapshot.write_bytes(b"nested")

    reference = make_best_reference(snapshot, checkpoint_dir, epoch=9, metric=1)

    assert reference["path"] == f"{SELECTED_BEST_DIRNAME}/interval_1/epoch_9_bbbb.pt"
    assert isinstance(reference["metric"], float) and reference["metric"] == 1.0
    assert resolve_best_reference(reference, checkpoint_dir).read_bytes() == b"nested"


# ---------------------------------------------------------------------------
# 2. make_best_reference input validation
# ---------------------------------------------------------------------------

def test_make_reference_refuses_snapshot_outside_the_store(checkpoint_dir: Path) -> None:
    """A checkpoint that is not in the immutable store cannot be referenced."""

    stray = checkpoint_dir / "last.pt"
    stray.write_bytes(b"not-a-snapshot")
    with pytest.raises(ReferenceValidationError, match="not inside"):
        make_best_reference(stray, checkpoint_dir, epoch=1, metric=0.5)


def test_make_reference_refuses_missing_symlinked_and_non_file_snapshots(
    checkpoint_dir: Path,
) -> None:
    """Only an existing regular file can be pinned by hash."""

    store = _store(checkpoint_dir)
    with pytest.raises(SnapshotIntegrityError, match="does not exist"):
        make_best_reference(store / "absent.pt", checkpoint_dir, epoch=1, metric=0.5)

    real = _write_snapshot(checkpoint_dir, 1, b"real")
    link = store / "link.pt"
    link.symlink_to(real)
    with pytest.raises(SnapshotIntegrityError, match="symbolic link"):
        make_best_reference(link, checkpoint_dir, epoch=1, metric=0.5)

    directory = store / "epoch_2_dir.pt"
    directory.mkdir()
    with pytest.raises(SnapshotIntegrityError, match="not a regular file"):
        make_best_reference(directory, checkpoint_dir, epoch=2, metric=0.5)


@pytest.mark.parametrize("epoch", [0, -1, True, 1.0, "3", None])
def test_make_reference_refuses_non_positive_or_non_integer_epoch(
    checkpoint_dir: Path, epoch: Any
) -> None:
    snapshot = _write_snapshot(checkpoint_dir, 1, b"payload")
    with pytest.raises(ReferenceValidationError, match="epoch"):
        make_best_reference(snapshot, checkpoint_dir, epoch=epoch, metric=0.5)


@pytest.mark.parametrize(
    "metric", [float("nan"), float("inf"), float("-inf"), True, "0.5", None]
)
def test_make_reference_refuses_non_finite_or_non_real_metric(
    checkpoint_dir: Path, metric: Any
) -> None:
    """An undefined metric can never improve the best, so it never reaches a reference."""

    snapshot = _write_snapshot(checkpoint_dir, 1, b"payload")
    with pytest.raises(ReferenceValidationError, match="metric"):
        make_best_reference(snapshot, checkpoint_dir, epoch=1, metric=metric)


def test_make_reference_refuses_missing_checkpoint_directory(tmp_path: Path) -> None:
    with pytest.raises(ReferenceValidationError, match="not an existing directory"):
        make_best_reference(tmp_path / "x.pt", tmp_path / "absent", epoch=1, metric=0.5)


# ---------------------------------------------------------------------------
# 3. resolve_best_reference: malformed references, traversal, hash corruption
# ---------------------------------------------------------------------------

def _valid_reference(checkpoint_dir: Path) -> dict[str, Any]:
    snapshot = _write_snapshot(checkpoint_dir, 4, b"resolvable")
    return dict(make_best_reference(snapshot, checkpoint_dir, epoch=4, metric=0.75))


def test_resolve_refuses_non_mapping_and_wrong_field_sets(checkpoint_dir: Path) -> None:
    reference = _valid_reference(checkpoint_dir)

    with pytest.raises(ReferenceValidationError, match="must be a mapping"):
        resolve_best_reference(["not", "a", "mapping"], checkpoint_dir)

    incomplete = {key: value for key, value in reference.items() if key != "sha256"}
    with pytest.raises(ReferenceValidationError, match=r"missing field\(s\): \['sha256'\]"):
        resolve_best_reference(incomplete, checkpoint_dir)

    extended = dict(reference, best_epoch=4)
    with pytest.raises(ReferenceValidationError, match=r"unknown field\(s\): \['best_epoch'\]"):
        resolve_best_reference(extended, checkpoint_dir)


@pytest.mark.parametrize("schema_version", [0, 2, "1", True, None])
def test_resolve_refuses_unsupported_schema_version(
    checkpoint_dir: Path, schema_version: Any
) -> None:
    reference = dict(_valid_reference(checkpoint_dir), schema_version=schema_version)
    with pytest.raises(ReferenceValidationError, match="schema_version"):
        resolve_best_reference(reference, checkpoint_dir)


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "deadbeef",
        "g" * 64,
        "A" * 64,  # uppercase hex is not the canonical form we emit
        None,
        1234,
    ],
)
def test_resolve_refuses_malformed_digest(checkpoint_dir: Path, digest: Any) -> None:
    reference = dict(_valid_reference(checkpoint_dir), sha256=digest)
    with pytest.raises(ReferenceValidationError, match="sha256"):
        resolve_best_reference(reference, checkpoint_dir)


@pytest.mark.parametrize(
    "path",
    [
        f"/etc/{SELECTED_BEST_DIRNAME}/epoch_1_x.pt",
        f"{SELECTED_BEST_DIRNAME}/../last.pt",
        f"{SELECTED_BEST_DIRNAME}/../../outside/epoch_1_x.pt",
        f"../{SELECTED_BEST_DIRNAME}/epoch_1_x.pt",
        f"{SELECTED_BEST_DIRNAME}/./epoch_1_x.pt",
        "last.pt",
        f"other_dir/{SELECTED_BEST_DIRNAME}/epoch_1_x.pt",
        SELECTED_BEST_DIRNAME,
        "",
        f"{SELECTED_BEST_DIRNAME}\\epoch_1_x.pt",
        f"{SELECTED_BEST_DIRNAME}/epoch_1_x.pt\x00",
        None,
    ],
)
def test_resolve_refuses_path_escape_and_malformed_paths(
    checkpoint_dir: Path, path: Any
) -> None:
    """Textual path validation runs before the filesystem is touched."""

    reference = dict(_valid_reference(checkpoint_dir), path=path)
    with pytest.raises(ReferenceValidationError, match="path"):
        resolve_best_reference(reference, checkpoint_dir)


def test_resolve_refuses_traversal_through_a_symlinked_store(tmp_path: Path) -> None:
    """Containment is re-checked against the resolved path, not just the text."""

    directory = tmp_path / "run"
    store = _store(directory)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "epoch_1_x.pt").write_bytes(b"outside-bytes")
    (store / "escape").symlink_to(outside, target_is_directory=True)

    reference = {
        "schema_version": SCHEMA_VERSION,
        "path": f"{SELECTED_BEST_DIRNAME}/escape/epoch_1_x.pt",
        "sha256": _sha(b"outside-bytes"),
        "epoch": 1,
        "metric": 0.5,
    }
    with pytest.raises(ReferenceValidationError, match="escapes the snapshot store"):
        resolve_best_reference(reference, directory)


def test_resolve_refuses_a_corrupted_snapshot(checkpoint_dir: Path) -> None:
    """A snapshot whose bytes changed is refused before any consumer loads it."""

    snapshot = _write_snapshot(checkpoint_dir, 5, b"original-bytes")
    reference = make_best_reference(snapshot, checkpoint_dir, epoch=5, metric=0.9)
    resolve_best_reference(reference, checkpoint_dir)  # sanity: currently valid

    snapshot.write_bytes(b"tampered-bytes")
    with pytest.raises(SnapshotIntegrityError, match="does not match its reference"):
        resolve_best_reference(reference, checkpoint_dir)


def test_resolve_refuses_missing_or_relinked_snapshot(checkpoint_dir: Path) -> None:
    snapshot = _write_snapshot(checkpoint_dir, 6, b"vanishing")
    reference = make_best_reference(snapshot, checkpoint_dir, epoch=6, metric=0.6)

    snapshot.unlink()
    with pytest.raises(SnapshotIntegrityError, match="does not exist"):
        resolve_best_reference(reference, checkpoint_dir)

    decoy = checkpoint_dir / "decoy.pt"
    decoy.write_bytes(b"vanishing")
    snapshot.symlink_to(decoy)
    with pytest.raises(SnapshotIntegrityError, match="symbolic link"):
        resolve_best_reference(reference, checkpoint_dir)


# ---------------------------------------------------------------------------
# 4. publish_best_alias
# ---------------------------------------------------------------------------

def test_publish_alias_copies_verified_bytes_and_is_idempotent(checkpoint_dir: Path) -> None:
    payload = b"published-best-payload"
    snapshot = _write_snapshot(checkpoint_dir, 7, payload)
    reference = make_best_reference(snapshot, checkpoint_dir, epoch=7, metric=0.71)

    assert best_alias_matches_reference(reference, checkpoint_dir) is False

    alias = publish_best_alias(reference, checkpoint_dir)

    assert alias == checkpoint_dir / PUBLIC_BEST_NAME
    assert alias.read_bytes() == payload
    assert not alias.is_symlink()
    assert _temp_leftovers(checkpoint_dir) == []
    assert best_alias_matches_reference(reference, checkpoint_dir) is True
    # The immutable snapshot is untouched by publication.
    assert snapshot.read_bytes() == payload

    # Republishing the same reference is a no-op in effect, not an error.
    publish_best_alias(reference, checkpoint_dir)
    assert alias.read_bytes() == payload
    assert _temp_leftovers(checkpoint_dir) == []


def test_publish_alias_replaces_a_previous_best_and_keeps_both_snapshots(
    checkpoint_dir: Path,
) -> None:
    """Advancing the alias never deletes the snapshot the old alias pointed at."""

    old_snapshot = _write_snapshot(checkpoint_dir, 1, b"old-best", token="old1")
    old_reference = make_best_reference(old_snapshot, checkpoint_dir, epoch=1, metric=0.4)
    publish_best_alias(old_reference, checkpoint_dir)

    new_snapshot = _write_snapshot(checkpoint_dir, 2, b"new-best", token="new2")
    new_reference = make_best_reference(new_snapshot, checkpoint_dir, epoch=2, metric=0.6)
    publish_best_alias(new_reference, checkpoint_dir)

    assert (checkpoint_dir / PUBLIC_BEST_NAME).read_bytes() == b"new-best"
    # No garbage collection: the older selection is still resolvable, which is
    # what keeps an older last.pt's reference valid.
    assert resolve_best_reference(old_reference, checkpoint_dir).read_bytes() == b"old-best"


@pytest.mark.parametrize("failing", ["fsync", "replace"])
def test_publish_alias_failure_leaves_old_alias_valid_and_no_temporary(
    checkpoint_dir: Path, failing: str
) -> None:
    """A durability or rename failure must not damage the alias already in place."""

    old_snapshot = _write_snapshot(checkpoint_dir, 1, b"old-best", token="old1")
    old_reference = make_best_reference(old_snapshot, checkpoint_dir, epoch=1, metric=0.4)
    publish_best_alias(old_reference, checkpoint_dir)
    alias = checkpoint_dir / PUBLIC_BEST_NAME
    alias_bytes_before = alias.read_bytes()

    new_snapshot = _write_snapshot(checkpoint_dir, 2, b"new-best", token="new2")
    new_reference = make_best_reference(new_snapshot, checkpoint_dir, epoch=2, metric=0.6)

    target = f"src.self_audit.training.checkpoint_commit.os.{failing}"
    with patch(target, side_effect=OSError(28, "No space left on device")):
        with pytest.raises(AliasPublishError, match="Failed to publish best alias"):
            publish_best_alias(new_reference, checkpoint_dir)

    # The old alias is byte-identical and still matches its own reference.
    assert alias.read_bytes() == alias_bytes_before
    assert best_alias_matches_reference(old_reference, checkpoint_dir) is True
    assert best_alias_matches_reference(new_reference, checkpoint_dir) is False
    assert _temp_leftovers(checkpoint_dir) == []
    # Both snapshots survive, so a later retry can still succeed.
    assert resolve_best_reference(new_reference, checkpoint_dir).read_bytes() == b"new-best"

    publish_best_alias(new_reference, checkpoint_dir)
    assert alias.read_bytes() == b"new-best"


def test_publish_alias_failure_cleanup_error_does_not_mask_the_primary_error(
    checkpoint_dir: Path,
) -> None:
    """A failing temp unlink must not replace the real failure with its own."""

    snapshot = _write_snapshot(checkpoint_dir, 3, b"payload")
    reference = make_best_reference(snapshot, checkpoint_dir, epoch=3, metric=0.5)

    with patch(
        "src.self_audit.training.checkpoint_commit.os.replace",
        side_effect=OSError(5, "Input/output error"),
    ), patch(
        "src.self_audit.training.checkpoint_commit.os.unlink",
        side_effect=OSError(1, "Operation not permitted"),
    ):
        with pytest.raises(AliasPublishError, match="Input/output error"):
            publish_best_alias(reference, checkpoint_dir)

    assert not (checkpoint_dir / PUBLIC_BEST_NAME).exists()


def test_publish_alias_fails_closed_when_snapshot_changes_mid_copy(
    checkpoint_dir: Path,
) -> None:
    """Bytes are re-hashed after the copy; a snapshot that mutated is refused."""

    snapshot = _write_snapshot(checkpoint_dir, 8, b"stable-bytes")
    reference = make_best_reference(snapshot, checkpoint_dir, epoch=8, metric=0.8)
    old_snapshot = _write_snapshot(checkpoint_dir, 1, b"old-best", token="old1")
    publish_best_alias(
        make_best_reference(old_snapshot, checkpoint_dir, epoch=1, metric=0.1),
        checkpoint_dir,
    )

    real_sha = __import__("src.self_audit.training.checkpoint_commit", fromlist=["_sha256_file"])
    original = real_sha._sha256_file
    calls: list[Path] = []

    def _mutate_after_verification(path: Path) -> str:
        calls.append(Path(path))
        digest = original(path)
        # Simulate the snapshot being rewritten between verification and copy.
        if len(calls) == 1 and Path(path) == snapshot.resolve():
            snapshot.write_bytes(b"mutated")
        return digest

    with patch.object(real_sha, "_sha256_file", side_effect=_mutate_after_verification):
        with pytest.raises(SnapshotIntegrityError, match="changed while copying"):
            publish_best_alias(reference, checkpoint_dir)

    assert (checkpoint_dir / PUBLIC_BEST_NAME).read_bytes() == b"old-best"
    assert _temp_leftovers(checkpoint_dir) == []


def test_publish_alias_refuses_symlinked_or_non_regular_public_target(
    checkpoint_dir: Path,
) -> None:
    """The public alias must be a regular file, never a link or a directory."""

    snapshot = _write_snapshot(checkpoint_dir, 9, b"payload")
    reference = make_best_reference(snapshot, checkpoint_dir, epoch=9, metric=0.9)

    alias = checkpoint_dir / PUBLIC_BEST_NAME
    alias.symlink_to(snapshot)
    with pytest.raises(AliasPublishError, match="symbolic link"):
        publish_best_alias(reference, checkpoint_dir)
    alias.unlink()

    alias.mkdir()
    with pytest.raises(AliasPublishError, match="not a regular file"):
        publish_best_alias(reference, checkpoint_dir)


def test_publish_alias_refuses_a_corrupted_or_malformed_reference(
    checkpoint_dir: Path,
) -> None:
    """Publication inherits every resolve-time gate; nothing unverified is copied."""

    snapshot = _write_snapshot(checkpoint_dir, 4, b"good")
    reference = make_best_reference(snapshot, checkpoint_dir, epoch=4, metric=0.5)
    snapshot.write_bytes(b"corrupted")

    with pytest.raises(SnapshotIntegrityError, match="does not match its reference"):
        publish_best_alias(reference, checkpoint_dir)
    assert not (checkpoint_dir / PUBLIC_BEST_NAME).exists()
    assert _temp_leftovers(checkpoint_dir) == []

    with pytest.raises(ReferenceValidationError, match="path"):
        publish_best_alias(dict(reference, path="../best.pt"), checkpoint_dir)


# ---------------------------------------------------------------------------
# 5. Crash between last.pt and the alias, then explicit recovery
# ---------------------------------------------------------------------------

def test_interrupted_alias_publication_is_repaired_from_the_reference(
    checkpoint_dir: Path,
) -> None:
    """The recoverable-commit scenario the protocol exists for.

    ``last.pt`` has already committed a verified reference to the new selection
    when the process dies before -- or during -- alias publication. The public
    alias is stale, but the reference resolves, so an explicit republish on
    resume repairs it. No in-process rollback is involved.
    """

    old_snapshot = _write_snapshot(checkpoint_dir, 1, b"epoch-1-weights", token="old1")
    old_reference = make_best_reference(old_snapshot, checkpoint_dir, epoch=1, metric=0.40)
    publish_best_alias(old_reference, checkpoint_dir)

    new_snapshot = _write_snapshot(checkpoint_dir, 2, b"epoch-2-weights", token="new2")
    new_reference = make_best_reference(new_snapshot, checkpoint_dir, epoch=2, metric=0.55)

    # last.pt commits the new reference first; the alias publish then dies.
    committed_last_reference = dict(new_reference)
    with patch(
        "src.self_audit.training.checkpoint_commit.os.replace",
        side_effect=OSError(4, "Interrupted system call"),
    ):
        with pytest.raises(AliasPublishError):
            publish_best_alias(new_reference, checkpoint_dir)

    alias = checkpoint_dir / PUBLIC_BEST_NAME
    assert alias.read_bytes() == b"epoch-1-weights", "stale alias, as expected after a crash"
    assert best_alias_matches_reference(committed_last_reference, checkpoint_dir) is False

    # Resume: resolve the committed reference and repair the alias explicitly.
    resolved = resolve_best_reference(committed_last_reference, checkpoint_dir)
    assert resolved.read_bytes() == b"epoch-2-weights"
    publish_best_alias(committed_last_reference, checkpoint_dir)

    assert alias.read_bytes() == b"epoch-2-weights"
    assert best_alias_matches_reference(committed_last_reference, checkpoint_dir) is True
    assert _temp_leftovers(checkpoint_dir) == []


def test_previous_last_reference_survives_a_failed_new_commit(checkpoint_dir: Path) -> None:
    """Audit-plan R9 regression: a failed new commit cannot orphan the old last.

    Under the old scheme, writing the new ``best.pt`` before ``last.pt`` meant a
    failure in between left the old ``last.pt`` describing bytes that had been
    overwritten. Here the old ``last.pt``'s reference names an immutable
    snapshot that a new selection never touches.
    """

    old_snapshot = _write_snapshot(checkpoint_dir, 3, b"committed-epoch-3", token="old3")
    old_last_reference = make_best_reference(old_snapshot, checkpoint_dir, epoch=3, metric=0.50)
    publish_best_alias(old_last_reference, checkpoint_dir)

    # A new epoch is selected and its snapshot is written, then everything after
    # that point fails -- alias publication included.
    new_snapshot = _write_snapshot(checkpoint_dir, 4, b"candidate-epoch-4", token="new4")
    new_reference = make_best_reference(new_snapshot, checkpoint_dir, epoch=4, metric=0.61)
    with patch(
        "src.self_audit.training.checkpoint_commit.os.replace",
        side_effect=OSError(28, "No space left on device"),
    ):
        with pytest.raises(AliasPublishError):
            publish_best_alias(new_reference, checkpoint_dir)

    # The old last.pt's selection is intact, byte-for-byte, and still published.
    assert resolve_best_reference(old_last_reference, checkpoint_dir).read_bytes() == (
        b"committed-epoch-3"
    )
    assert best_alias_matches_reference(old_last_reference, checkpoint_dir) is True


# ---------------------------------------------------------------------------
# 6. relocate_best_reference
# ---------------------------------------------------------------------------

def test_relocate_copies_snapshot_to_same_relative_path_and_preserves_source(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "old_run"
    target_dir = tmp_path / "new_run"
    _store(source_dir)
    target_dir.mkdir()

    payload = b"relocatable-weights"
    snapshot = _write_snapshot(source_dir, 5, payload, token="rel5")
    reference = make_best_reference(snapshot, source_dir, epoch=5, metric=0.66)

    relocated = relocate_best_reference(reference, source_dir, target_dir)

    assert relocated == dict(reference)
    copied = target_dir / SELECTED_BEST_DIRNAME / "epoch_5_rel5.pt"
    assert copied.read_bytes() == payload
    assert resolve_best_reference(relocated, target_dir) == copied.resolve()
    # The source directory is left fully usable.
    assert snapshot.read_bytes() == payload
    assert resolve_best_reference(reference, source_dir).read_bytes() == payload
    # Relocation does not publish; that stays an explicit call.
    assert not (target_dir / PUBLIC_BEST_NAME).exists()

    publish_best_alias(relocated, target_dir)
    assert (target_dir / PUBLIC_BEST_NAME).read_bytes() == payload


def test_relocate_is_idempotent_for_byte_identical_target(tmp_path: Path) -> None:
    source_dir = tmp_path / "old_run"
    target_dir = tmp_path / "new_run"
    _store(source_dir)
    target_dir.mkdir()

    snapshot = _write_snapshot(source_dir, 6, b"same-bytes", token="rel6")
    reference = make_best_reference(snapshot, source_dir, epoch=6, metric=0.7)

    first = relocate_best_reference(reference, source_dir, target_dir)
    second = relocate_best_reference(reference, source_dir, target_dir)
    assert first == second == dict(reference)
    assert (target_dir / SELECTED_BEST_DIRNAME / "epoch_6_rel6.pt").read_bytes() == b"same-bytes"


def test_relocate_refuses_conflicting_bytes_and_unrelated_entries(tmp_path: Path) -> None:
    """A target that already holds something else is never clobbered."""

    source_dir = tmp_path / "old_run"
    target_dir = tmp_path / "new_run"
    _store(source_dir)
    target_store = _store(target_dir)

    snapshot = _write_snapshot(source_dir, 7, b"authentic", token="rel7")
    reference = make_best_reference(snapshot, source_dir, epoch=7, metric=0.8)

    conflicting = target_store / "epoch_7_rel7.pt"
    conflicting.write_bytes(b"someone-elses-bytes")
    with pytest.raises(RelocationConflictError, match="different bytes"):
        relocate_best_reference(reference, source_dir, target_dir)
    assert conflicting.read_bytes() == b"someone-elses-bytes"
    assert snapshot.read_bytes() == b"authentic"

    conflicting.unlink()
    conflicting.symlink_to(snapshot)
    with pytest.raises(RelocationConflictError, match="not a regular file"):
        relocate_best_reference(reference, source_dir, target_dir)

    conflicting.unlink()
    conflicting.mkdir()
    with pytest.raises(RelocationConflictError, match="not a regular file"):
        relocate_best_reference(reference, source_dir, target_dir)


def test_relocate_refuses_a_corrupted_source_and_missing_target_dir(tmp_path: Path) -> None:
    source_dir = tmp_path / "old_run"
    target_dir = tmp_path / "new_run"
    _store(source_dir)
    target_dir.mkdir()

    snapshot = _write_snapshot(source_dir, 8, b"original", token="rel8")
    reference = make_best_reference(snapshot, source_dir, epoch=8, metric=0.9)
    snapshot.write_bytes(b"tampered")

    with pytest.raises(SnapshotIntegrityError, match="does not match its reference"):
        relocate_best_reference(reference, source_dir, target_dir)
    assert not (target_dir / SELECTED_BEST_DIRNAME).exists()

    snapshot.write_bytes(b"original")
    with pytest.raises(ReferenceValidationError, match="target_dir is not an existing directory"):
        relocate_best_reference(reference, source_dir, tmp_path / "absent")


def test_relocate_refuses_a_symlinked_nested_directory_in_the_destination(
    tmp_path: Path,
) -> None:
    """The destination is contained the same way the source is.

    The target's ``selected_best`` is a real directory, but its ``nested``
    sub-directory is a symlink pointing outside the checkpoint root. A plain
    join would write the copy through that link, outside the directory the
    operator named.
    """

    source_dir = tmp_path / "source_run"
    target_dir = tmp_path / "target_run"
    nested_source = _store(source_dir) / "nested"
    nested_source.mkdir()
    snapshot = nested_source / "e1.pt"
    snapshot.write_bytes(b"nested-escape-bytes")
    reference = make_best_reference(snapshot, source_dir, epoch=1, metric=0.5)
    assert reference["path"] == f"{SELECTED_BEST_DIRNAME}/nested/e1.pt"

    outside = tmp_path / "outside"
    outside.mkdir()
    (_store(target_dir) / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ReferenceValidationError, match="escapes the snapshot store"):
        relocate_best_reference(reference, source_dir, target_dir)

    assert list(outside.iterdir()) == [], "nothing may be written outside the checkpoint root"
    assert snapshot.read_bytes() == b"nested-escape-bytes"


def test_relocate_into_the_same_directory_is_a_verified_no_op(checkpoint_dir: Path) -> None:
    snapshot = _write_snapshot(checkpoint_dir, 9, b"in-place", token="rel9")
    reference = make_best_reference(snapshot, checkpoint_dir, epoch=9, metric=0.5)

    assert relocate_best_reference(reference, checkpoint_dir, checkpoint_dir) == dict(reference)
    assert snapshot.read_bytes() == b"in-place"
    assert _temp_leftovers(_store(checkpoint_dir)) == []


# ---------------------------------------------------------------------------
# 7. Snapshot store root itself must be genuine (reproduced STORE_ESCAPE_ACCEPTED)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entrypoint", ["make", "resolve", "publish", "relocate_destination", "alias_match"]
)
def test_symlinked_snapshot_store_is_refused_by_every_entrypoint(
    tmp_path: Path, entrypoint: str
) -> None:
    """A store moved out of the checkpoint root and symlinked back in is refused.

    Every relative path still resolves "inside the store" and every hash still
    matches, so containment relative to ``realpath(store)`` alone accepts it
    while the bytes live outside the directory the operator named. The store
    root has to be a real directory, checked at each entry point.
    """

    root = tmp_path / "run"
    _store(root)
    snapshot = _write_snapshot(root, 1, b"e1-bytes", token="e1")
    reference = make_best_reference(snapshot, root, epoch=1, metric=0.5)

    outside = tmp_path / "outside_store"
    shutil.move(str(root / SELECTED_BEST_DIRNAME), str(outside))
    (root / SELECTED_BEST_DIRNAME).symlink_to(outside, target_is_directory=True)
    # The escape is otherwise invisible: the bytes and the digest both still match.
    linked = root / SELECTED_BEST_DIRNAME / "epoch_1_e1.pt"
    assert linked.read_bytes() == b"e1-bytes"
    assert _sha(linked.read_bytes()) == reference["sha256"]

    if entrypoint == "make":
        call = lambda: make_best_reference(linked, root, epoch=1, metric=0.5)
    elif entrypoint == "resolve":
        call = lambda: resolve_best_reference(reference, root)
    elif entrypoint == "publish":
        call = lambda: publish_best_alias(reference, root)
    elif entrypoint == "alias_match":
        call = lambda: best_alias_matches_reference(reference, root)
    else:
        source_dir = tmp_path / "source_run"
        _store(source_dir)
        source_snapshot = _write_snapshot(source_dir, 1, b"e1-bytes", token="e1")
        source_reference = make_best_reference(source_snapshot, source_dir, epoch=1, metric=0.5)
        call = lambda: relocate_best_reference(source_reference, source_dir, root)

    with pytest.raises(ReferenceValidationError, match="snapshot store"):
        call()
    assert not (root / PUBLIC_BEST_NAME).exists()


# ---------------------------------------------------------------------------
# 8. Dependency isolation
# ---------------------------------------------------------------------------

def test_module_is_stdlib_only_and_imports_no_training_modules() -> None:
    """No torch, no _utils, no unified_trainer: no circular import, no second policy."""

    from src.self_audit.training import checkpoint_commit

    tree = ast.parse(Path(checkpoint_commit.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    assert imported == {
        "__future__",
        "collections.abc",
        "hashlib",
        "math",
        "os",
        "pathlib",
        "stat",
        "tempfile",
        "typing",
    }, f"unexpected imports: {sorted(imported)}"
    for forbidden in ("torch", "numpy", "self_audit"):
        assert not any(
            name == forbidden or name.startswith(f"{forbidden}.") for name in imported
        ), f"checkpoint_commit must not import {forbidden}"

    for attribute in ("torch", "np", "numpy"):
        assert not hasattr(checkpoint_commit, attribute)
    assert os.path.basename(checkpoint_commit.__file__) == "checkpoint_commit.py"
