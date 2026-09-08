# W3.1 — producer checkpoint binding (worker report)

Date: 2026-09-08. Implementer: Claude Code via Orca (`task_15d1ae019ddc` / `ctx_8d1708d5bcfb`).
Scope: producer side only. W3.2 consumer redesign is **not** done here, and W3 is **not**
complete from this task.

## What was wrong

`scripts/train_self_audit.py` collected the validation transition cache from the *live*
last-epoch model, then independently selected `phase_c_best.pt` (falling back to
`phase_c_last.pt`) and handed that path to `save_calibration`, which hashed the file. When
best != last, the artifact named weights that were never measured. The two final
diagnostics ran on the same live model, so all four consumers shared the defect.

## What changed

### New: `src/self_audit/provenance.py`

Narrow, dependency-light helpers. No environment/credential/provider scanning; the only
external process consulted is `git` in the working tree.

- `state_digest(module | state_dict)` — deterministic SHA-256 over every parameter *and*
  buffer, sorted by key, framing key/shape/dtype/length before the raw bytes. Bytes are
  taken through a `uint8` view, never NumPy of the tensor's own dtype, so `bfloat16` and
  scalar integer buffers hash correctly.
- `file_sha256(path)` — strict file hash (missing file raises).
- `git_source_provenance()` — `git_sha`, whole-tree `dirty`, and `source_dirty` restricted
  to `src/scripts/configs`. A report-only commit therefore cannot read as a model-code
  change. Both flags are `None` (not `False`) when git is unavailable.
- `producer_provenance_record()` / `checkpoint_producer(payload)` — the producing revision
  is written at save time and read back at load time. A legacy checkpoint reports
  `producer_recorded=False` and `producer_git_sha="unknown"`; current HEAD is never
  substituted.
- `resolve_model_identity(model)` — read off live modules after construction/load, never
  from a filename or directory: `encoder_backend` distinguishes `timm` from
  `fallback_synthetic`, and `entropy_version` comes from the live
  `annotation_expert.entropy_version`.
- `verify_model_config(model, config)` — mismatch fails; a configured key the live model
  cannot report also fails (no silent unknown bypass).
- `preprocessing_descriptor(dataset|loader)` — the actual dataset's *recipe*: image size,
  depth axis, foreground-only, percentiles, augment/transform, normalization and input
  construction, plus the declared geometry metadata contract (axis orders, in-plane resize
  and effective-spacing semantics, spacing-units policy). The descriptor never indexes the
  dataset — fetching a sample would load ground truth, run the transform and consume RNG,
  and one sample cannot speak for a cohort. A dataset that is not this project's
  volume-slice pipeline is recorded as `unknown` rather than inheriting its guarantees.
  Cohort sizes are deliberately **excluded** so two disjoint cohorts processed identically
  share one recipe signature; per-record spacing availability is cohort content and is
  reported by `cohort_identity` over every record.
- `split_membership_descriptor` — reuses W2's `compute_split_signature`; labelled
  `coverage="membership_only_not_content_hash"`.
- `cohort_identity` — exact observed cohort. `covers_full_split` is `True` only when every
  record was iterated: `max_batches` truncation, a `Subset` (indices signature recorded),
  a subset sampler, or a `drop_last` remainder each make it `False` with a stated reason.
  An uncharacterisable sampler or custom batch sampler raises rather than being guessed.
  `_unwrap_dataset` composes nested `Subset` indices outer-to-inner and keeps a directly
  wrapped `Subset`'s indices (both cases were wrong in the first draft and are now
  regression-tested against the coordinator's exact probe).
- `build_lineage(...)` — checkpoint block (path, file SHA, state digest, producer, resolved
  model identity), evaluation-code revision recorded *separately* from the producing
  revision, semantics (entropy version, full versioned metric-contract definition resolved
  from the registry, metric space, neutral margin, t_max, fixed class mapping),
  preprocessing recipe, split membership, cohort.
- `CheckpointBinding` dataclass with `as_dict()`.

### `src/self_audit/training/_utils.py`

- `save_checkpoint` now stamps `payload["provenance"]`: schema version, producing git SHA,
  dirty/source-dirty, torch version, the digest of the saved tensors, and the resolved
  model identity. The payload stays `weights_only=True`-loadable.
- `bind_evaluation_checkpoint(model, [(role, path), ...])` — first existing candidate wins;
  a later candidate is a *declared* fallback (`fallback_used`), none present raises
  `FileNotFoundError`. Evaluation-only load: strict `load_state_dict`, and no optimizer,
  scheduler, scaler or RNG restore (the function takes no such parameters). After the load
  the live digest must equal the on-disk digest, and a stored provenance digest that
  disagrees with the stored tensors is rejected as tampered.
- `verify_bound_state(model, binding, boundary=...)` — recomputes the live digest at a
  consumer boundary and fails on any mutation.
- `load_checkpoint` is unchanged for resume/inspection, RNG restore included.

### `scripts/train_self_audit.py`

- The Phase-C tail was extracted verbatim into `run_post_training_calibration(...)`, which
  `main()` now calls; `bind_post_training_checkpoint(...)` is the small helper it calls
  first. Best selection criterion, training recipe, objectives and optimizer behaviour are
  unchanged.
- Order is now: select best (last = declared fallback) → strict load into the live model →
  digest verification → cache collection → calibration sweep → both diagnostics. Each of
  those four boundaries calls `verify_bound_state`.
- `save_calibration` receives `binding.path` — the checkpoint that was actually measured —
  and carries the lineage under `extra.lineage` (schema untouched; W3.2 owns promoting it
  to a validated field). The cache saved to `validation_transitions.pt` carries the same
  lineage.
- `report["checkpoint_binding"]` and `report["final_diagnostic"]["cohort"]` are new; the
  cohort record states that `--max_val_batches` truncates the diagnostics.

### `scripts/cache_validation_transitions.py`

Uses the same binding helper instead of `load_checkpoint` — this also removes an
unintended RNG restore on an evaluation-only path — verifies the bound state before
collection, and attaches lineage to the cache.

## Tests — `tests/test_checkpoint_binding.py` (30 new)

Digest: buffers, scalar int buffer, `bfloat16`, empty tensor, shape/dtype/key separation.
Provenance: new payload stamped and `weights_only` round-trips; legacy producer stays
`unknown` and is never current HEAD; source-dirty scope excludes `reports/`; identity read
live.
Binding: tiny best != last fixture binds best (live digest == best, != last); declared last
fallback; missing both raises; config mismatch and incompatible state dict fail; tampered
checkpoint rejected; RNG byte state provably unchanged across a bind while the legacy
resume path still restores it; `verify_bound_state` catches a post-binding mutation.
Honesty: recipe signature identical across disjoint cohorts and absent of cohort counts;
declared geometry semantics with per-record spacing availability in the cohort; the recipe
build provably never calls `__getitem__` and leaves the RNG byte state untouched; a generic
dataset claims no normalization; direct and nested `Subset` indices survive unwrapping
(`Subset(d,[2,4,6])` → `[2,4,6]`; `DataLoader(Subset(that,[1]))` → `[4]`); `Subset`,
`drop_last` remainder and truncation cannot claim full coverage; a shuffled full pass is
recorded as shuffled-but-complete; an unsupported sampler raises; metric contract resolved
to its full versioned definition; an unresolvable live attribute is not skipped.
Runner: `run_post_training_calibration` — the function `main()` calls — is exercised end to
end on a tiny synthetic model/loader with best != last, spying on the collector, the sweep
and both diagnostics; all four observe the *best* digest and never the last one, the
artifact names the bound checkpoint, and the cohort record marks the truncated diagnostics.

## Exact outputs

```
$ python -m pytest tests/test_checkpoint_binding.py -q
..............................                                           [100%]
30 passed in 6.39s

$ python -m pytest tests -q
222 passed, 1 warning in 14.00s

$ python -m compileall -q src scripts tests; echo compileall_rc=$?
compileall_rc=0

$ git diff --check; echo diff_check_rc=$?
diff_check_rc=0

$ python -c "... _unwrap_dataset probe ..."
[2, 4, 6]
[4]
```

The compile and diff-check exit codes above are the commands' own, not a pipeline tail's.
The single warning is pre-existing (`tests/test_self_audit_core.py:32`).

## Limits — read before claiming anything

- Producer side only. Consumers (`evaluation/threshold.py`, `scripts/calibrate_threshold.py`,
  `scripts/audit_checkpoint.py`) still do **not** strictly validate this lineage; that is
  W3.2. W3 is not complete.
- The calibration artifact schema is unchanged; lineage rides in `extra`.
- All fixtures are tiny and synthetic on CPU. No real checkpoint, no real bank, no medical
  validation, no Dice claim.
- Membership signatures prove membership, not content.
- Historical calibrations and checkpoints written before this change carry an unknown
  producer; they are not upgraded.
- No training was run, no commit or push was made, no old checkpoint was edited, and no
  architecture/objective/recipe/split change was made.

## Tooling note (disclosure)

`apply_patch` is not on this harness's `PATH` (`which apply_patch` → not found); the path
the coordinator supplied resolves to the Codex binary, which is that harness's editing
entrypoint rather than this one's. All corrections in the review round were made with the
structured file-edit tool (exact-match find/replace against the current file), not with
free-form shell/Python string rewrites; the earlier part of the session did use scripted
rewrites before the instruction arrived, and those edits are preserved rather than redone.
Shell commands are `rtk`-prefixed. No authored-patch fallback is outstanding: every
pending correction is already applied, tested and reported here.
