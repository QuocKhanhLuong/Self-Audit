# Independent Exact-Resume Regressions for W4

**Owner:** Claude worker (took over from the stopped AGY terminal on task `ctx_ba884aae5d73`).
**Owned artifacts:** `tests/test_runtime_resume.py` (new), this report.
**Not touched:** any production source, any pre-existing test, any other report.

---

## 1. Executive summary

A single real 6-epoch tiny curriculum (2 `annotation_bootstrap`, 2 `auditor_counterfactual`,
2 `joint_finetune`) is trained uninterrupted, its committed checkpoint tree is snapshotted
after every epoch commit, and the run is then resumed from each of the five legitimate
completed-epoch checkpoints (positions 1–5). The resumed runs are compared against the
uninterrupted run on model weights, optimizer, scheduler, gradient scaler, Python/NumPy/PyTorch
RNG streams, step counters, selection lineage, per-group learning rates, and freeze masks.

**Result of the focused invocation: 31 passed, 5 failed.**

Exact resume of the *training state* holds today at every one of the five positions, including
across both interval boundaries and through a real shuffled, augmenting, non-persistent
`num_workers=2` DataLoader. The five failures are all one defect, on one axis: **the pipeline
report's completed-epoch history is dropped on resume.**

---

## 2. How the fixture is built

`tests/test_runtime_resume.py`

* **One baseline, parameterized resumes of its suffix.** The module-scoped `baseline` fixture
  trains the full 6-epoch schedule once. `resumed_runs` then resumes that same baseline from
  epoch 1, 2, 3, 4 and 5. There is no second baseline and no per-test retraining.
* **Legitimate committed states only — checkpoints and reports both.** For the baseline run,
  `unified_trainer.save_checkpoint` and `unified_trainer.atomic_write_json` are both wrapped. Each
  wrapper delegates to the real writer first and only then copies what was durably written:
  the checkpoint wrapper hard-links the *already-committed* output directory (`last.pt`, the
  `best.pt` alias, and the immutable `selected_best/` snapshots) into `snapshots/epoch_<n>/`, and
  the report wrapper copies the durable `pipeline_report.json` bytes into the snapshot directory
  named by that file's own `last_checkpoint_committed_epoch`. Committed checkpoints are written
  through an atomic replace, so their inodes are never mutated in place and a hard link preserves
  the exact committed bytes at no storage cost. **No report is reconstructed by filtering the final
  report's rows, and no status field is fabricated**; every artifact a resume reads is the artifact
  the trainer itself committed at that state. No counter is edited by hand.
* **No configuration extension on resume.** Each resumed trainer parses the identical config
  dictionary; only `checkpoint.output_dir` and `logging.report_dir` move, which is the one
  difference `compare_execution_configs` permits. `test_resume_does_not_extend_or_rewrite_the_execution_config`
  asserts the config signature is unchanged and re-runs `compare_execution_configs` on the saved
  payload directly.
* **Varied per-interval shape.** Batch sizes 2/3/2, loader cardinalities 3/2/3, accumulation
  steps 1/1/2, `augment` False/True/False, trainability `annotation`/`auditor`/`all`, rollout
  `propagate_no_audit`/`annotation_eval`/`threshold_gate`, and three different LR triples. A
  resume that silently restarts the schedule cannot coincidentally reproduce this.
* **Selection gating is real.** `checkpoint.best_selection_min_epoch = 4` (the minimum the config
  parser accepts for a `[4, 6)` gated interval), so the immutable `selected_best/` snapshot and
  the `best.pt` alias exist only from epoch 5 onward. Resume at position 5 therefore exercises the
  selected-best reference path, and positions 1–4 exercise the no-selection path.
* **Boundary learning rates are asserted against the real schedule.** The canonical curve has
  `warmup_epochs = 5`, so a freshly built interval scheduler emits `max(0 / warmup_steps, 1e-8)`
  at its first step, not the configured LR. `_boundary_warmup_factor` re-derives that multiplier
  from the config exactly as `setup_interval_optimizer_and_scheduler` does, and the assertion is
  `lr == configured_lr * factor`. The optimizer-reset check (the boundary optimizer state must
  differ from the previous interval's committed state) is retained alongside it.
* **Bounded checkpoints are never treated as resumable.**
  `test_max_steps_bounded_checkpoint_is_refused_for_resume` runs a `max_steps=1` bounded run and
  asserts `resume_from_checkpoint` rejects the resulting `last.pt`.
* **Encoder.** For these unit tests only, `timm.create_model` is made to raise *during model
  construction*, forcing the repository's own `_FallbackHierarchicalEncoder` (≈6 M parameters
  instead of ≈27.8 M) so that eleven committed checkpoint trees stay small. This is the
  dependency-light path the repository already ships for offline tests; canonical pretraining is
  untouched and the unpatched ConvNeXt CLI is covered by the separate smoke suite. No other
  monkeypatching is used — in particular **no RNG is mocked away and no assertion is weakened**.

Runtime for the whole file is ~30 s on CPU (torch 2.4.1, Python 3.10, `OMP_NUM_THREADS=1`).

---

## 3. DataLoader / sampler generator state — investigated, and it is fine

`build_data_loader` (`src/self_audit/training/_utils.py:365`) constructs `DataLoader` **without a
dedicated `generator=`**. That was the suspected divergence. It is not one, because of how PyTorch
falls back:

* With `generator=None` and `shuffle=True`, `RandomSampler` draws its permutation seed from the
  **global** torch RNG at each `__iter__`.
* With `generator=None` and `num_workers>0`, `_BaseDataLoaderIter` draws `_base_seed` from the
  **global** torch RNG, and each worker's torch/NumPy/Python seeds are derived deterministically
  from `_base_seed + worker_id`.

The global torch RNG *is* checkpointed (`_rng_state()` stores `python`, `numpy`, `torch`, and
`cuda` when available) and `resume_from_checkpoint` restores it **last**, after the model load and
after the optimizer/scheduler rebuild, so nothing between restore and the first batch consumes
from the restored stream. Consequently both the shuffle permutation and the worker seeds are
reproduced exactly.

This is confirmed empirically, not by inspection: the augmenting interval's train loader runs with
`shuffle=True`, `num_workers=2`, `persistent_workers=False`, and an augmentation in
`__getitem__` that draws from the worker's torch RNG. Resumes at epoch 2 (the boundary *into* that
interval) and epoch 3 (mid-interval) both reproduce the uninterrupted final model digest bit for
bit. `test_multi_worker_augmenting_loader_is_actually_exercised` asserts the loader really did run
with `num_workers=2` and a `RandomSampler`, and is written to report missing coverage explicitly
rather than degrade silently — on this platform (darwin, spawn start method) it **passes**, so the
coverage is real.

**No confirmed divergence on this axis. No early escalation was warranted.**

The existing `persistent_workers` guard in `resume_from_checkpoint` (refuse exact resume for an
augmenting interval backed by persistent workers) remains correct and is untouched: persistent
workers survive across epochs and their RNG genuinely cannot be re-derived from the global stream.

---

## 4. What passes (31 tests)

| Area | Coverage |
| --- | --- |
| Baseline integrity | 6 epochs, 3 intervals in order, `completed=True`, loader cardinalities `[3, 2, 3]`, selected best from the gated joint interval |
| Snapshot legitimacy | every snapshot has `resumable=True`, `incomplete_epoch=False`, `validation_complete=True`, full `rng_state`, real `loader_cardinality` |
| Multi-worker loader | augmenting interval really ran `num_workers=2`, non-persistent, `RandomSampler` |
| **Exact final state** (k = 1…5) | model, optimizer, scheduler, scaler, RNG (python/numpy/torch), counters, selection, per-group LRs, freeze mask — all identical to the uninterrupted run |
| State at the moment of resume (k = 1…5) | restored model + RNG match the committed epoch; `global_step`/`optimizer_step`/`completed_epochs`/`last_checkpoint_committed_epoch` are the committed values, not fabricated |
| Schedule continuity (k = 1…5) | interval index, freeze mask, and rollout mode belong to the epoch about to run; mid-interval resumes restore optimizer/scheduler/scaler and LRs exactly; boundary resumes rebuild them from the new interval's configured LRs **through that interval's own warmup** and do **not** carry the previous optimizer state |
| Commit schema | every snapshot's `last.pt` carries `role="last"`, a `wandb_identity` block, and selection metadata derived from the committed immutable reference (`best_reference.sha256 == best_checkpoint_hash`, epoch 5); pre-gate epochs 1–4 claim no selected best at all |
| Source quiescence | an explicit test fails if `src/`/`configs/` changed while the baseline trained, so a source-modification skip of the resume comparisons is never silent |
| Config lineage (k = 1…5) | config signature unchanged, schedule not extended, `compare_execution_configs` clean against the saved payload |
| Selected-best lineage (k = 5) | the immutable `selected_best/` snapshot's SHA-256 is retained in the relocated output directory |
| Calibration (k = 1…5) | a resumed, completed curriculum runs real post-training calibration and lands on the same `tau_accept` as the uninterrupted run |
| Bounded smoke | a `max_steps`-truncated `last.pt` is refused for resume |

The exact-state result is the important one: **`resume_from_checkpoint` + `train(start_epoch=…)`
is already byte-exact on the training state at all five positions.** W4's remaining work on this
axis is not "make resume exact"; it is the report defect below.

---

## 5. Confirmed failing premise (5 failures, one defect)

### D1 — the resumed run's final report drops the completed-epoch history

`test_resumed_final_report_preserves_the_completed_epoch_history[1..5]` — **5 failed.**

```
assert [5, 6] == [1, 2, 3, 4, 5, 6]      # resume at epoch 4
assert [6]    == [1, 2, 3, 4, 5, 6]      # resume at epoch 5
```

**Mechanism.** `UnifiedTrainer.__init__` initialises `self.report["epochs"] = []`.
`resume_from_checkpoint` restores weights, counters, selection lineage, cohort identity and RNG,
but never reads back the pipeline report that the interrupted run had already committed. `_train_impl`
then appends only epochs `k+1…6` and atomically overwrites `pipeline_report.json` with that
truncated list, together with `completed_epochs` reflecting only the resumed segment's tail.

**Why this matters, and why the test is written the way it is.** The test seeds the resumed run's
`report_dir` with the pipeline report the interrupted run had genuinely committed at epoch `k` — the
state a real interrupted run finds on disk. So the history is not hypothetical: it is present, it is
authoritative, and the resumed run overwrites it with a shorter one. For a research pipeline whose
per-epoch report *is* the record of the run, an interruption silently erases the training curve of
every epoch before the resume point. The checkpoint counters are correct; only the narrative record
is lost.

**Suggested shape of the fix (main W4 trainer owner's call — no production edit made here).**
On resume, load `report_dir/pipeline_report.json` when it exists, verify it belongs to the same
`run_id`/`config_signature` as the checkpoint, and rehydrate `self.report["epochs"]` with the rows
whose `epoch <= completed_epoch` before `_train_impl` appends. Rows beyond the committed epoch
(from an epoch that trained but never committed a checkpoint) must be discarded, not kept, so the
report cannot claim an epoch the checkpoint lineage does not.

Once that lands, all five parameterizations should pass with no change to the test.

---

## 6. Two environment notes for the coordinator

1. **Concurrent source edits break the resume premise, by design.**
   `resume_from_checkpoint` refuses a checkpoint whose `source_signature` differs from the live
   source content signature, and `source_content_signature()` hashes `src/` and `configs/`. During
   the first probe run, the parallel W4 implementation session wrote `unified_trainer.py`
   mid-baseline, which changed the signature between epoch 1 and epoch 2 and made resumes at 2–5
   fail with a source-signature mismatch. That is the guard working correctly, not a defect. The
   fixture therefore records the signature before and after the baseline and, if it changed,
   `pytest.skip`s the resume comparisons with an explicit reason instead of reporting a
   meaningless failure. **No assertion is relaxed** — the skip fires only when the working tree was
   mutated underneath the run, and a dedicated test
   (`test_source_tree_was_quiescent_while_the_baseline_trained`) **fails** in exactly that case, so
   the condition is always reported rather than quietly swallowed. In the invocation reported here
   the tree was quiescent: that test passes and nothing was skipped.
2. **Production moved during this task.** The commit protocol changed under me mid-session
   (`selected_best/` immutable snapshots plus a hash-verified reference in `last.pt`, and selection
   now additionally gated on `interval.rollout == "threshold_gate"`). The tests are written against
   the observable contract — committed files, payload flags, restored state — rather than internal
   symbol names, so they survived that change; but a re-run is worth doing after W4 settles.

---

## 7. Reproduction

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_runtime_resume.py -q
```

Latest invocation: `5 failed, 31 passed in 30.03s`.

---

## 8. Note on `parallel_portability.md`

The old AGY task asked for a correction to the bind sentence in
`reports/runtime_hardening/2026-09-10/parallel_portability.md`. That file already reads
"it leaves the model on its existing model device and does not move a CPU model to the requested
`map_location`", which is the corrected wording, so no edit was needed and none was made. No partial
`tests/test_runtime_resume.py` or `parallel_resume.md` existed on disk from the stopped AGY
terminals; there was no prior work to preserve.
