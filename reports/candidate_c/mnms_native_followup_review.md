# M&Ms native Candidate C execution — follow-up review after the joint-from-epoch-1 change

Date: 2026-09-12. Base commit: `bb639ec118af72dac0d5e24339a5e325e2d6d163` (`main`).
Scope: the native M&Ms execution path only — `scripts/run_acdc_mnms_candidate_c.sh`,
`configs/self_audit_joint_from_start_mnms.yaml`, the staged
`configs/self_audit_full_mnms.yaml`, `scripts/run_full_pipeline.sh`, dataset
resolution and `src/self_audit/data/mnms.py`. The external ACDC→M&Ms evaluator is
owned by another worker and was not reviewed.

**Status of the original audit: REVISE.** Its first pass reported PASS and filed
two real defects as non-blocking observations. That was wrong: both could
silently change what a native M&Ms run actually trains on, and neither is
excused by being pre-existing.

**Status of the fix pass: applied, gates green locally, root review pending.**
The two defects are fixed and pinned by tests, and the schedule/curriculum
properties the first pass verified still hold and are kept below. This is not a
self-issued PASS: the final verdict on these changes belongs to the root
reviewer, and nothing here has passed a full-suite or real-data gate. It is in
any case a software/configuration statement only — see "Evidence class".

**Subsequent root verdict:** PASS after actual diff review and independent
combined gates; see [final integration review](mnms_followup_integration_review.md).

This report does not repeat `reports/candidate_c/joint_from_start_review.md`; it
covers the M&Ms-specific surface that report treated only jointly with ACDC.

## Defects found and fixed

### D1 — A concurrent invocation could rewrite a pending M&Ms leg's config (fixed)

**Was:** `scripts/run_acdc_mnms_candidate_c.sh` generated its run configs at
`run_configs/{acdc,mnms}_candidate_c_${CURRICULUM}.yaml` — a path keyed only on
the curriculum, while run directories were keyed on `STAMP`. The M&Ms leg reads
its config only after the ACDC leg finishes, hours later. A second invocation
started in that window rewrote the file the first invocation's pending M&Ms leg
was about to read, so that leg trained under the second invocation's
`BATCH_SIZE`. Silent, and invisible in the first run's own banner.

**Falsified against the pre-fix script**, not argued from reading: the
`bb639ec` version of the runner was extracted to a scratch copy and driven with
a shim `python` that runs the real interpreter for config generation but, for
the ACDC training call, first runs a whole second invocation
(`BATCH_SIZE=2`) to completion, then records the training arguments and exits.
The outer invocation ran with `BATCH_SIZE=8`. Both invocations resolved to
`--config run_configs/mnms_candidate_c_joint_from_start.yaml`, and the config
the outer run's pending M&Ms leg received contained `batch_size: [2]` — the
interleaved invocation's value, not its own.

**Now:** each invocation allocates its own directory with `mktemp -d
"run_configs/${CURRICULUM}_${STAMP}_XXXXXX"`. A stamp alone was not enough — two
invocations can share a second — and `mktemp` fails rather than reuse an
existing path. The allocation is unconditional: there is deliberately **no**
environment override for it, because an override would reintroduce exactly the
sharing it prevents. Each generated config is created with an exclusive open
(`open(dst, "x")`), so the file is claimed and written in one step rather than
through an `exists()` check followed by a write, which left a window between the
two; the OS raises `FileExistsError` before any byte is written. Written configs
are `chmod 0444` and kept after the run as the provenance record of what was
executed. The chosen directory is printed in both banners. Batch size, CUDA
defaults, the `joint_from_start` default and the staged path are unchanged.

### D2 — An unpaired supervised volume silently shrank the cohort (fixed)

**Was:** `discover_mnms_records` skipped an image whose mask was missing
(`continue`) in the `volumes/`+`masks/` layout, while the paired-NIfTI fallback
raised for the mirror case. A partially copied M&Ms cohort therefore trained on
fewer cases than deployed, with no failure and no warning; the reverse case (a
mask with no image) was not detected at all. Worse, discovery built its two
sides with dictionary comprehensions keyed on the normalized case key, so two
files normalizing to one key — `case.npy` beside `case.npz` — collapsed into a
single entry and one of them disappeared before any pairing check could see it.

**Now:** `discover_mnms_records` takes a keyword-only `strict_pairing` flag
(`src/self_audit/data/mnms.py`), and the two sides are grouped by key
(`_group_volume_files`) so no file is hidden by the normalizer. With
`strict_pairing=True` the function raises `FileNotFoundError` listing whichever
of these it found:

* images with no mask, and masks with no image;
* several files sharing one normalized key, on either side;
* an image matching more than one candidate mask (`case` and `case_gt` both
  present as masks);
* one mask claimed by two images (`p` and `p_gt` both images, `p_gt` the only
  mask);
* a case id discovered more than once, when two pair directories or two split
  aliases (`train` beside `training`) both offer it — previously the second was
  dropped in silence.

The default stays `False` and its selection is byte-for-byte the old one:
grouping preserves `iterdir()` order and the tolerant caller takes the **last**
entry of a group, which is exactly the file the previous dictionary
comprehension kept, so external and evaluation callers see no semantic change
from this native fix. A test pins that parity directly against a dictionary
comprehension over the same directory. `validate_dataset_splits` passes
`strict_pairing=True`
only in the native `directory_splits` branch
(`src/self_audit/training/_utils.py`, the `has_native_splits` path); the
evaluation-only branch is untouched.

Mask-key matching precedence is the named constant `MASK_KEY_SUFFIXES =
("_gt", "_label", "_seg")`, applied after the bare image key. In tolerant mode
that precedence silently resolves a multi-candidate match; in strict mode a
multi-candidate match is an error rather than a silent choice.

The raw-to-ACDC mapping was deliberately **not** hard-locked to
`DEFAULT_MNMS_TO_ACDC`. An alternative raw encoding may be intentional, so
`class_mapping` stays configurable and is only required to be a
background-preserving bijection over `0..3`; a regression test pins that a
non-canonical mapping still validates.

## Properties verified (unchanged from the first pass)

### 1. The default native run is joint-from-epoch-1 for BOTH legs

`scripts/run_acdc_mnms_candidate_c.sh:42` sets
`CURRICULUM="${CURRICULUM:-joint_from_start}"`; lines 52–58 map that default to
`configs/self_audit_joint_from_start.yaml` (ACDC) **and**
`configs/self_audit_joint_from_start_mnms.yaml` (M&Ms). Both legs are generated
from one `pairs` list, so the datasets cannot diverge in curriculum by accident.
The enum is closed and exits 2 on an unknown value (lines 43–50), with the same
check repeated inside the generation block.

### 2. Feedback gates are live and both modules trainable from epoch 0 on M&Ms

`configs/self_audit_joint_from_start_mnms.yaml:139-157` declares one interval
`[0, 130)` with `trainable: "all"`, `objective: "retained_final_annotation"`,
`transition_population: "active_attempted"`, `rollout: "threshold_gate"`,
`annotation_weight: 1.0`, `audit_weight: 1.0`, `reset_optimizer: true` — exactly
the invariants `src/self_audit/training/schedule.py:184-194` enforces for that
objective, so a weakened M&Ms variant cannot load.
`src/self_audit/training/unified_trainer.py:1584-1587` sets
`rollout/feedback_gates = "active"` for a `threshold_gate` interval and
`rollout/joint_trainable = 1.0` for `trainable == "all"`, mirrored into the epoch
log at lines 1717–1720; both hold at global epoch 0.
`checkpoint.best_selection_min_epoch: 0` is accepted only because the gated
interval starts at 0 (`unified_config.py:891-908`).
`predicted_history_exposure: false` is inert for this objective.

### 3. Staged stays the default of the generic runner; the joint profile is the opt-in

This corrects the first pass, whose heading implied the generic runner needed an
opt-in to reach staged. It does not:

* `scripts/run_full_pipeline.sh:20` defaults to `configs/self_audit_full.yaml`
  (staged). Reaching a joint profile through it requires an explicit
  `--config configs/self_audit_joint_from_start{,_mnms}.yaml`, documented at
  lines 9–12 and 49–51. `scripts/run_full_pipeline.ps1:16` likewise defaults to
  the staged config.
* Only the native Candidate C runner defaults to `joint_from_start`, and there
  the staged curriculum is the explicit opt-in: `CURRICULUM=staged`
  (script lines 52–54), which alone re-enables the original
  `predicted_history_exposure=True` at weight `0.1` (lines 119, 138–140 of the
  generation block).
* The checked-in profile YAMLs declare `model.window_mode: "current"`; Candidate
  C exists only in the runner's generated run configs. README now states this.

### 4. The M&Ms dataset contract is the staged one, unchanged

A key-by-key comparison of `configs/self_audit_full_mnms.yaml` against
`configs/self_audit_joint_from_start_mnms.yaml`, excluding `training.schedule`,
reports exactly six differing keys — all schedule identity or selection:

```
checkpoint.best_selection_min_epoch: staged=120  joint=0
checkpoint.output_dir:               staged='weights/self_audit_full_mnms'   joint='weights/self_audit_joint_from_start_mnms'
experiment.name:                     staged='self_audit_full_mnms'           joint='self_audit_joint_from_start_mnms'
logging.report_dir:                  staged='reports/self_audit_full_mnms'   joint='reports/self_audit_joint_from_start_mnms'
logging.wandb.run_name:              staged='full-pipeline-mnms'             joint='joint-from-start-mnms'
logging.wandb.tags:                  +['joint_from_start']
```

`dataset.*`, `model.*`, optimizer, LR curve, losses, counterfactual settings,
calibration and diagnostics are identical to the canonical M&Ms baseline.

### 5. Dataset resolution, depth, splits, paired inputs, unknown labels

* **Root.** The M&Ms leg passes `--data_root preprocessed_data/mnm`, equal to
  `dataset.data_root` in the profile, so the override is a no-op rather than a
  redirect (`apply_overrides`, `unified_config.py:945-946`). Identical to the
  pre-change runner.
* **Splits.** `split_manifest: null` plus `train_split`/`val_split` selects the
  `directory_splits` strategy (`_utils.py`, `has_native_splits` branch).
  Configured `train`/`val`/`test` resolve through the alias table in
  `src/self_audit/data/mnms.py` to `training`/`validation`/`testing`. The
  optional `test` split is checked only when present on disk.
* **Paired inputs.** `_find_pair_dirs` accepts the deployed `volumes/`+`masks/`
  layout. Pairing is now strict for supervised splits — see D2. Note what this
  does and does not say: strict discovery guarantees that every supported file
  it *sees* is unambiguously paired, and it no longer lets the key normalizer
  hide a file, but it does not certify the contents of any volume; label and
  shape checks happen afterwards, per volume, in split validation.
* **Depth.** `depth_axis: 2` matches the deployed `(H, W, D)` arrays; validated
  and applied by `to_depth_first` before the shape and label checks.
* **Class mapping.** `{0:0, 1:3, 2:2, 3:1}` equals `DEFAULT_MNMS_TO_ACDC`
  (M&Ms LV/MYO/RV → ACDC RV/MYO/LV); `MNMSClassMapping.__post_init__` requires
  background→background and a bijection over `0..3`.
* **Unknown labels fail closed.** `MNMSClassMapping.apply` raises on any raw
  label outside the mapping and on non-integer or non-finite masks; the same
  check runs per volume during startup split validation.
* **Binary derivative rejected** at dataset construction, discovery, the builder
  and the split validator (`is_mnms_binary_path`).
* **Startup cost.** Split validation loads every train, val and test volume and
  mask once before epoch 0. Pre-existing staged behaviour, unchanged here. No
  cohort size is asserted in this report: no M&Ms tree exists in this worktree.

### 6. Checkpoint isolation; ACDC and M&Ms never mix training

Two separate `train_self_audit.py` processes with two separate configs, no
`--resume` in either, distinct generated config directories, experiment names,
run directories (`runs/$ACDC_RUN` vs `runs/$MNMS_RUN`), output/report directories
and W&B projects (`self-audit-acdc-candidate-c` vs `self-audit-mnms-candidate-c`).
Cross-dataset resume is independently blocked: `compare_cohort_descriptors`
treats `dataset_name`, `data_root`, `split_manifest`, `train_split`, `val_split`
and `class_mapping` as critical fields (`unified_trainer.py:237-245`), populated
from the live config in `get_cohort_descriptor` (`unified_trainer.py:846-860`),
with per-split membership signatures compared on top. `set -euo pipefail` plus
`| tee` means a failing ACDC leg aborts before the M&Ms leg starts.

## Changes made in this revision

| File | Change |
|---|---|
| `scripts/run_acdc_mnms_candidate_c.sh` | Unconditional per-invocation `mktemp -d` run-config directory with no environment override; generated configs created with an exclusive `open(dst, "x")` rather than an `exists()`-then-write sequence; `chmod 0444`; directory printed in the start and completion banners |
| `src/self_audit/data/mnms.py` | `strict_pairing` keyword on `discover_mnms_records`; `_group_volume_files` so the key normalizer cannot hide a file, in `iterdir()` order with last-wins tolerant selection so default/external discovery is unchanged; strict rejection of unpaired images, orphan masks, duplicate normalized keys, multi-candidate masks, one mask claimed by two images, and a case id discovered twice; `MASK_KEY_SUFFIXES` precedence constant |
| `src/self_audit/training/_utils.py` | Native `directory_splits` discovery calls `strict_pairing=True`; evaluation-only branch unchanged |
| `README.md` | Per-profile selection boundary; native M&Ms joint command; profile YAML is current-window while the runner selects Candidate C; per-entrypoint defaults stated explicitly (`train_self_audit.py` and `run_full_pipeline.sh` staged, native Candidate C runner joint) instead of a blanket claim; per-invocation run configs and strict pairing documented; external `--checkpoint` must name the frozen ACDC `best.pt` the source run actually produced, and `--tau-accept 0.0` is a stated fixed threshold, not the source run's calibrated tau (coordinator follow-up; no external code touched) |
| `tests/test_mnms_native_safety.py` (new) | 17 tests covering both defects; the runner tests execute inside a throwaway repository layout under `tmp_path` |

No core, loss, schedule or config-schema file was modified. No commit was made.
`tests/test_joint_from_start_configs.py` needed no change and was left alone.

## Tests

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_mnms_native_safety.py
# 19 passed in 4.18s

PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_mnms_native_safety.py tests/test_mnms_wave2.py \
  tests/test_joint_from_start_configs.py
# 56 passed in 7.46s

PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_mnms_external_flow.py
# 7 passed in 0.50s   (rerun after the tolerant last-wins change)
```

`tests/test_mnms_native_safety.py` is new and holds 19 collected tests: 16 on
cohort pairing and 3 on the runner. The 16 count each parameterized case
separately (4 ambiguity layouts, 4 mask suffixes), so they come from 10 test
functions. They are:

* `test_concurrent_invocations_cannot_rewrite_a_pending_mnms_leg` — runs the
  real runner script with a shim `python`: real config generation, no training.
  The shim runs a second complete invocation (`BATCH_SIZE=2`) inside the first
  invocation's ACDC leg, **with the same `STAMP`**, so only `mktemp` can keep
  the two apart. It asserts the first invocation's pending M&Ms leg still
  received `batch_size: 8`, the second `batch_size: 2`, that the two
  invocations allocated different directories, that both generated configs keep
  `window_mode: candidate_c`, the M&Ms class mapping and the single
  `trainable: all` / `threshold_gate` interval, that each directory holds exactly
  its two configs, and that those files are not writable.
* `test_each_leg_keeps_its_own_data_root_and_output_identity` — the ACDC leg gets
  `preprocessed_data/ACDC/training` and the M&Ms leg `preprocessed_data/mnm`,
  with distinct output, report and W&B identities, no `--resume`, and neither leg
  handed the other's config.
* `test_generated_run_config_directory_cannot_be_pointed_at_an_existing_path` —
  setting `RUN_CONFIG_DIR` in the environment does not redirect generation, the
  pre-created directory stays empty, and the script contains no `${RUN_CONFIG_DIR:-}`
  override.
* Pairing tests: missing mask, orphan mask, four parameterized ambiguity cases
  (duplicate image key, duplicate mask key, two candidate masks, one mask claimed
  by two images), a case id discovered under two split aliases, tolerant
  selection matching a dictionary comprehension over the same directory, the
  four accepted mask suffixes, native split validation
  failing on an unpaired training volume, native validation passing on a fully
  paired cohort, evaluation-only validation staying tolerant, and a non-canonical
  `class_mapping` still validating.

The three runner tests execute inside a throwaway repository layout under
`tmp_path` — a copy of the runner script and the four profile configs, plus a
symlink to `src` — so they never create, rewrite or delete anything in the
developer's `run_configs/` or `logs/`. Each of them additionally asserts that a
snapshot of those two real directories is identical before and after the run.
`tests/test_mnms_external_flow.py` was run as a blast-radius check on the shared
discovery function and the README, not as a review of the evaluator.

## Remaining items (design choices and one unrelated pre-existing gap)

These are recorded so the next reader does not re-litigate them. The first is a
pre-existing gap left alone by instruction; the other two are deliberate
interface decisions, **not** missing guards:

1. **Both legs hardcode `--device cuda`** with no environment override, unlike
   `BATCH_SIZE` and `CUDA_VISIBLE_DEVICES`. Pre-existing; out of scope here.
2. **A global M&Ms `class_mapping` is configuration, by design.** The strict
   loader enforces the identity map for ACDC only
   (`unified_config.py:875-876`), because an M&Ms tree may legitimately use a
   different raw encoding and the mapping is the place where the operator
   declares it. `MNMSClassMapping` still requires a background-preserving
   bijection over `0..3`, and a wrong-but-valid mapping is an operator error in
   configuration, not an unguarded code path. If it is ever to be caught
   automatically, the right mechanism is reporting the observed raw label
   histogram beside the declared mapping — not locking the mapping.
3. **`MNMSDataset`'s permissive default discovery is API design, not a hole.**
   Supervised training reaches it only after `validate_dataset_splits` has
   already rejected an ambiguous or unpaired native cohort, and evaluation
   callers legitimately need the tolerant behaviour for trees with unlabelled
   volumes. `strict_pairing` is opt-in precisely so those two audiences do not
   share one policy.

## Evidence class

All evidence here is static source reading, synthetic-fixture unit tests, and a
shimmed runner whose training command never executes — on macOS CPU.
`preprocessed_data/mnm` and `preprocessed_data/ACDC` do not exist in this
worktree, so no real M&Ms volume was read and no cohort or slice count was
observed; the deployed layout is quoted from the design spec, not re-verified.
No GPU was used: CUDA behaviour, RTX 4070 memory at `BATCH_SIZE=8` under the
gated joint rollout from epoch 0, and throughput remain unmeasured. Nothing here
is evidence that joint-from-epoch-1 produces better M&Ms segmentation, useful
early Auditor feedback, or any scientific result.
