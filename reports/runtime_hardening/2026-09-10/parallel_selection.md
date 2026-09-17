# Isolated immutable-selection helper for Wave 4

Worker: parallel Claude. Owned paths only —
`src/self_audit/training/checkpoint_commit.py` (new),
`tests/test_checkpoint_commit.py` (new), and this report.
`unified_trainer.py` and `_utils.py` were **not** edited; the main AGY integrates the API below.
No commits, no push, no config/recipe/architecture change.

Environment: `/private/tmp/self-audit-torch241/bin/python` (Python 3.10.21, torch 2.4.1 CPU),
`PYTHONPATH=src:.`, single-threaded.

## Test run

```
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_checkpoint_commit.py -q -p no:randomly
```

**66 passed in 0.51s.** One focused invocation, as briefed. Byte fixtures throughout: nothing in
the helper interprets checkpoint content, so a real payload would only slow the filesystem logic
down. Pairing the protocol with actual models, resume and calibration stays main-trainer work.

## Revision after Astra gate REVISE (store-escape)

Root reproduced `STORE_ESCAPE_ACCEPTED True`: create `root/selected_best/e1.pt`, take a reference,
move the `selected_best` directory out of the checkpoint root, symlink `root/selected_best` at its
new location, and `resolve_best_reference` happily returned the outside path.

Root cause: containment was proven against `os.path.realpath(store)`. When the store root *is* the
symlink, its realpath is the outside directory, so every relative path still resolves "inside the
store" and every hash still matches, while the bytes live outside the directory the operator named.
Comparing against the resolved store cannot detect a store that was itself moved.

Fix: a new internal `_require_snapshot_store(checkpoint_dir)` refuses a store root that is a
symlink, is not a directory, or does not equal its own realpath. It is called from
`make_best_reference`, from `_contained_target` (so `resolve_best_reference` and
`publish_best_alias` inherit it), from the **relocation destination**, and from
`best_alias_matches_reference` — not only from `make_best_reference`. A store that does not exist
yet is still allowed, because a relocation destination legitimately has its store created by the
copy.

The public API is unchanged; the new gate raises `ReferenceValidationError` like every other
reference-level refusal. Regression:
`test_symlinked_snapshot_store_is_refused_by_every_entrypoint`, one compact test parametrized over
`make`, `resolve`, `publish`, `relocate_destination` and `alias_match`. It asserts first that the
escape is otherwise invisible — the bytes read through the link and their digest both still match
the reference — so the test would pass against the old code only if the store check were missing.
No other scenarios were added.

## Second revision: nested destination escape on relocation

Root reproduced `RELOCATION_NESTED_ESCAPE_WRITTEN True`: make a reference for
`selected_best/nested/e1.pt` in the source; in the *target*, `selected_best` is a real directory
but `selected_best/nested` is a symlink pointing outside the checkpoint root. The source side was
already contained, but the destination was built with a plain
`target_directory.joinpath(*relative.parts)`, so the atomic copy was written through the link and
landed outside the directory the operator named.

Fix: the destination now goes through the same `_contained_target(relative, target_directory)` the
source uses, so the resolved parent must sit inside the target's snapshot store. One regression,
`test_relocate_refuses_a_symlinked_nested_directory_in_the_destination`, asserts the
`ReferenceValidationError`, that the outside directory stays empty, and that the source snapshot is
untouched. No other scenarios were added.

## Evaluation-time consumer of the protocol (`_utils.py`)

The reference the protocol commits is now also enforced where evaluation binds the canonical
alias. `_require_committed_best_alias(path)` runs at both `bind_evaluation_checkpoint` and
`bind_existing_evaluation_state` before binding. When the requested basename is `best.pt` and a
sibling `last.pt` carries `best_reference`, the sibling is safe-loaded on CPU
(`weights_only=True`, no pickle fallback), the reference is verified with
`resolve_best_reference`, and a `best.pt` whose hash differs from the committed selection is
refused with the recovery named: resume from `last.pt`, which resolves the committed reference and
republishes the alias. Evaluation never repairs the alias itself, because it must measure exactly
the weights it was pointed at. `_utils.py` imports only `checkpoint_commit` (stdlib-only) for this,
never `unified_trainer`, so no cycle is introduced.

Key name: `COMMITTED_BEST_REFERENCE_KEY = "best_reference"`, exported from `_utils.py`, at the top
level of the `last.pt` payload (pass it through `save_checkpoint(..., extra={...})`). The main AGY
is writing that exact key.

Compatibility is deliberate and tested: a non-canonical basename (the historical
`phase_c_best.pt`) is outside the protocol; a standalone copied `best.pt` with no sibling `last.pt`
binds; a `last.pt` predating the protocol — no `best_reference` key — binds. An explicit
`best_reference: None` means the run recorded that it had *no* committed selection, so a `best.pt`
next to it is unexplained and is refused.

One judgment call worth flagging: a sibling `last.pt` that cannot be safely loaded
(`weights_only=True` failing on legacy metadata, which audit item R2 makes plausible for
pre-hardening checkpoints) is reported on stdout as unverified and then skipped, rather than
refused. Refusing would break every historical run whose `last.pt` predates the recursive
normalization work, and loading it unsafely is not on the table. If root prefers fail-closed there,
it is a one-line change in `_require_committed_best_alias`.

Tests: `tests/test_canonical_best_alias.py`, **12 passed in 1.55s** on Python 3.10.21 / torch
2.4.1 with small `nn.Linear` checkpoints — committed alias binds and loads the actual weights
through both entrypoints; stale alias (last commits epoch 2, alias still epoch 1) refused at both
entrypoints with the recovery named; missing, corrupt and path-escaping references refused;
explicit `None` refused; standalone `best.pt`, key-less historical `last.pt` and
`phase_c_best.pt` all still bind. Every refusal test hashes the whole run directory before and
after and asserts nothing on disk changed, and one test spies on `torch.load` to assert the
guard's own read is `map_location="cpu"` with `weights_only=True`.

## The problem this closes

Audit plan **R9**: per-file atomicity is not a transaction. Writing the new `best.pt` and then
failing to write `last.pt` leaves the old `last.pt` naming a best whose bytes were already
overwritten, so resume cannot reproduce the selection it recorded. Wave 4 asks for a recoverable
commit protocol that does not depend on in-process rollback, which cannot run under `SIGKILL`.

The protocol these helpers support:

1. Trainer saves the epoch's candidate with the existing `save_checkpoint` to a unique, never
   reused path `<checkpoint_dir>/selected_best/epoch_<epoch>_<uuid>.pt`.
2. `make_best_reference(...)` describes it (relative path + sha256 + epoch + metric).
3. Trainer embeds that reference in the `last.pt` payload and commits **`last.pt` first**.
4. Trainer calls `publish_best_alias(...)` to move the public `best.pt` onto the snapshot.

Failure accounting:

| Crash point | State afterwards |
|---|---|
| Before step 3 | Old `last.pt` still references its own untouched snapshot; the new candidate is inert bytes. |
| During step 3 | Same — `last.pt` is committed with the existing atomic writer. |
| During or after step 4 | `last.pt` already carries a verified immutable reference; the next resume resolves it and repairs the public alias by calling `publish_best_alias` again. |

Recovery depends only on durable bytes plus a hash. Nothing is ever deleted: snapshot garbage
collection is deliberately out of scope, because an older `last.pt` may still reference an older
snapshot.

## API (confirmed accepted by root; `epoch >= 1` confirmed correct for one-based completed epochs)

Module constants: `SCHEMA_VERSION = 1`, `SELECTED_BEST_DIRNAME = "selected_best"`,
`PUBLIC_BEST_NAME = "best.pt"`, `REFERENCE_FIELDS`.

Exceptions, all under `SelectionError(Exception)`: `ReferenceValidationError`,
`SnapshotIntegrityError`, `AliasPublishError`, `RelocationConflictError`.

```python
make_best_reference(snapshot_path, checkpoint_dir, *, epoch: int, metric: float)
    -> dict[str, str | int | float]
resolve_best_reference(reference: Mapping[str, Any], checkpoint_dir) -> Path
publish_best_alias(reference: Mapping[str, Any], checkpoint_dir) -> Path
relocate_best_reference(reference: Mapping[str, Any], source_dir, target_dir)
    -> dict[str, str | int | float]
best_alias_matches_reference(reference: Mapping[str, Any], checkpoint_dir) -> bool
```

### `make_best_reference`

Returns exactly
`{"schema_version": 1, "path": "selected_best/...", "sha256": "<64 lowercase hex>", "epoch": int, "metric": float}`
— only `str`/`int`/`float` leaves, so it embeds verbatim in a `last.pt` payload under
`weights_only=True` and in a JSON report.

Requires an existing directory for `checkpoint_dir`; an existing **regular** snapshot file (a
symlink is refused, not followed, because a link is a second mutable indirection a hash cannot
pin); the snapshot inside `checkpoint_dir/selected_best` with no path escape; `epoch` an `int >= 1`
(`bool` refused); `metric` a finite real (`bool`, NaN and ±inf refused — an undefined metric can
never improve the best, so it must never reach a selection reference). Sub-directories under the
store are allowed. The snapshot is read to hash it and is otherwise untouched.

### `resolve_best_reference`

Fail-closed, in this order: exact key set (missing *and* unknown fields refused, so a hand-edited
reference cannot smuggle extra state), `schema_version == 1`, 64-char lowercase-hex digest,
textual path validation, resolved-path containment, regular-file check, then hash comparison.
**The hash is verified before any consumer deserializes the file.**

Path validation rejects: absolute paths, `..` and `.` segments (checked on the raw string, because
`PurePosixPath` silently normalizes `./` away), empty segments, backslashes, null bytes, a first
component other than `selected_best`, and a bare directory name. Containment is then re-checked
against `os.path.realpath`, so a symlinked intermediate directory inside the store cannot point
the target outside it either, and the store root itself must be a real directory inside
`checkpoint_dir` (see the revision section above).

Deliberately silent about content: run identity, execution config, state digest and selection
metadata are payload-level facts the trainer checks after its own safe load.

### `publish_best_alias`

Resolves and hash-verifies the snapshot, streams the bytes to a temporary in the **same directory**
as `best.pt`, re-hashes what actually landed and fails closed if the source changed in flight,
`flush()` + `os.fsync()`, refuses to proceed if `best.pt` is an existing symlink or non-regular
file, `os.replace`, then best-effort directory fsync. Any read/fsync/replace error leaves the
previous `best.pt` byte-identical and removes the temporary; a cleanup failure never masks the
primary error. Idempotent — republishing the same reference is exactly how resume repairs a stale
alias.

### `relocate_best_reference`

Verifies the snapshot in `source_dir`, atomically copies it to the **same relative path** under
`target_dir` (temp + fsync + `os.replace` + directory fsync), and returns a reference identical to
the input, valid for `target_dir`. The source is left in place and re-verified after the copy, so
relocation can never be what invalidates the directory being relocated from. A byte-identical
target is accepted as already relocated; different bytes, a symlink, or a directory raise
`RelocationConflictError` rather than being overwritten. `target_dir` must already exist.
**Alias publication stays a separate explicit call.**

### `best_alias_matches_reference`

Convenience so resume can decide whether a repair publish is needed without loading anything.
`True` only when `best.pt` is a regular file whose sha256 equals the reference; a missing or
non-regular alias is `False`; a malformed reference still raises.

## Dependency isolation

Imports are exactly `__future__`, `collections.abc`, `hashlib`, `math`, `os`, `pathlib`, `stat`,
`tempfile`, `typing`. The module imports neither `torch` nor `self_audit.training._utils` nor
`self_audit.training.unified_trainer`, so there is no circular dependency and — the important part
— **no second torch serialization policy**: checkpoints are opaque bytes that `save_checkpoint`
already produced and validated. `test_module_is_stdlib_only_and_imports_no_training_modules`
asserts the import set with `ast`, so a later accidental import fails the suite.

## Test coverage map

| Area | Tests |
|---|---|
| Expected SHA preserved, bytes not altered, weights_only-safe leaves, nested snapshot, int metric coerced | `test_make_reference_preserves_expected_sha_and_snapshot_bytes`, `test_make_reference_accepts_nested_snapshot_and_integer_metric` |
| `make` input gates | snapshot outside the store; missing / symlinked / directory snapshot; epoch in `{0, -1, True, 1.0, "3", None}`; metric in `{NaN, inf, -inf, True, "0.5", None}`; missing `checkpoint_dir` |
| Malformed references | non-mapping, missing field, unknown field, `schema_version` in `{0, 2, "1", True, None}`, digest in `{"", "deadbeef", "g"*64, "A"*64, None, 1234}` |
| Nested destination escape (second REVISE) | `test_relocate_refuses_a_symlinked_nested_directory_in_the_destination` |
| Store root escape (Astra REVISE) | `test_symlinked_snapshot_store_is_refused_by_every_entrypoint` — parametrized over `make`, `resolve`, `publish`, `relocate_destination`, `alias_match` |
| Path traversal / malformation | 12 parametrized cases: absolute, `selected_best/../last.pt`, `../selected_best/...`, `selected_best/./...`, bare `last.pt`, `other_dir/selected_best/...`, bare store name, empty, backslash, null byte, `None`; plus `test_resolve_refuses_traversal_through_a_symlinked_store` for a symlinked store sub-directory |
| Hash corruption | `test_resolve_refuses_a_corrupted_snapshot`, `test_resolve_refuses_missing_or_relinked_snapshot`, `test_publish_alias_refuses_a_corrupted_or_malformed_reference` |
| Failing fsync/replace leaves old alias valid, no temp | `test_publish_alias_failure_leaves_old_alias_valid_and_no_temporary` (parametrized over `os.fsync` and `os.replace` raising `OSError`), then a retry that succeeds |
| Cleanup does not mask the primary error | `test_publish_alias_failure_cleanup_error_does_not_mask_the_primary_error` (`os.replace` **and** `os.unlink` both failing; the I/O error is what surfaces) |
| Source identity across copy | `test_publish_alias_fails_closed_when_snapshot_changes_mid_copy` (snapshot rewritten between verification and copy; old alias intact) |
| Unsafe final target | `test_publish_alias_refuses_symlinked_or_non_regular_public_target` |
| Publish happy path / idempotence / no GC | `test_publish_alias_copies_verified_bytes_and_is_idempotent`, `test_publish_alias_replaces_a_previous_best_and_keeps_both_snapshots` |
| Interrupted alias then explicit recovery | `test_interrupted_alias_publication_is_repaired_from_the_reference` — `last.pt`'s reference committed, publish dies at `os.replace`, alias is stale, `best_alias_matches_reference` is `False`, resolve still returns the right bytes, explicit republish repairs it, no temp left |
| R9 regression | `test_previous_last_reference_survives_a_failed_new_commit` — a failed new commit leaves the old `last.pt`'s selection byte-identical and still published |
| Relocation | same relative path + source preserved + publish still separate; idempotent for identical bytes; refuses different bytes, a symlink and a directory; refuses a corrupted source and a missing `target_dir`; same-directory no-op |
| Dependency isolation | `test_module_is_stdlib_only_and_imports_no_training_modules` |

## Notes for the integrating AGY

- **Snapshot paths must be unique per selection.** The helpers verify but do not allocate names;
  reusing a path would mutate a referenced snapshot and every reference to it would then fail
  closed (correctly, but the run would stop). `epoch_<epoch>_<uuid4hex>.pt` is what the tests model.
- **Commit `last.pt` before publishing.** Nothing here enforces the ordering; it is what makes the
  failure table above true.
- **`epoch >= 1`** is enforced, per root's confirmation that completed checkpoint epochs are
  one-based. A selection at completed epoch 0 would need the bound relaxed.
- **Non-finite or absent metric.** `make_best_reference` refuses a non-finite metric, so the
  explicit "no selection" state must be represented by the *absence* of a reference in `last.pt`,
  not by a reference carrying a sentinel. That pairs with Wave 4's requirement that a finite
  `best_metric` with no verifiable selected state must not suppress future selection: no
  reference means no selected state, whatever `best_metric` says.
- **Relocation does not publish.** After `relocate_best_reference`, call `publish_best_alias`
  against the target directory explicitly; `best_alias_matches_reference` tells you whether that is
  still needed.
- **No garbage collection.** If snapshot pruning is ever wanted, it needs its own
  reference-liveness analysis over every retained `last.pt`; it is not in this module.
- The helper says nothing about payload contents. Run id, execution config, model/state digest and
  selection-metadata cross-consistency (`last.best_metric`/`best_epoch` agreeing with the
  referenced snapshot) remain the trainer's checks, before use and before alias repair.
