# Run the locked baseline

Use worktree `/Users/alvinluong/Self-Audit-nogt`, branch
`feat/no-gt-baseline-20260918`. Commit identity is in
`evidence/commit_identity.json`; exact experimental source hashes are in each
FROZEN.json. Runtime versions used are in `evidence/environment.json`.
The environment requires Python, torch, numpy, scipy and nibabel; tests also use
pytest and the report plot uses matplotlib. No sklearn, pretrained download or GPU
is required. No user training process or other checkout is touched.

From that worktree, this **single command** runs current image-only preparation,
C0 generation/naming, native export, frozen hashes and image-only resolver diagnostics:

```bash
rtk proxy python scripts/run_nogt_suite.py --baseline-only --raw /tmp/astra_event_acdc_training/training --split /tmp/astra_event_acdc_training/acdc_patient_split_seed42.json --work /tmp/astra_nogt_C0_reproduction
```

The output directory must not exist; the command refuses to overwrite it. Outputs
are `runs/C0/step_0000/*_named.nii.gz`, `*_valid.nii.gz`, native `.npz`, `FROZEN.json`,
timing/read ledgers and `SELECTION_BEFORE_GT.json`. It exports the development split;
it generates anonymous cache for train+development. All slices of each frame are used.
The provided cohort has the historical acquisition caveat in SUPERVISION_LEDGER.
This command reproduces that cohort; it does not erase its provenance.

Method invocation accepts no reference path. No evaluator runs unless the explicitly
separate optional `--evaluate-references` argument is supplied to the suite driver.
Evaluation then starts only after freeze and selection. To score an existing frozen
run independently (development feedback only):

```bash
rtk proxy python scripts/evaluate_nogt_frozen.py --run /tmp/astra_nogt_runs/C0 --references /tmp/astra_event_acdc_training/training --out /tmp/astra_nogt_C0_development_evaluation.json
```

Unknown is255, never BG. GT cluster matching is analysis-only and cannot rewrite
these exported masks. Reference evaluator imports no torch and participates in no
training computation graph. A changed frozen NPZ/checkpoint causes failure before
the first reference read. Do not tune the resolver or pick a checkpoint from these
development scores while still claiming the locked protocol.

## Reproduce all bounded controls

Omit `--baseline-only` and use a **new** work directory. This executes the discarded
8-step pilot then four sequential1024-update arms, fixed seeds17/29, B4, CPU4, no
full150-epoch launch. The per-arm wall cap is45minutes; failure remains INCOMPLETE.
Adding `--evaluate-references /tmp/astra_event_acdc_training/training` evaluates all
five runs after image-only selection, not during training. This is available for
reproduction, not a recommendation to spend another training budget on a weak teacher.

Scoped software verification:

```bash
rtk proxy python -m pytest -q tests/test_nogt_baseline.py
```

Recompute committed analysis tables/figures from saved raw metrics, without method
training or a new reference read:

```bash
rtk proxy python scripts/summarize_nogt_results.py --report reports/astra_nogt_baseline_20260918T042050Z
```

`summarize_nogt_results.py` consumes independent evaluator results; never import it
into the method. It is not a baseline selector. Primary run metrics are immutable;
aggregation/figures can be regenerated. No GPU/SSH call occurs in these commands.

## Operational limits

The whole dataset cache lives in host RAM/disk; small CNN parameter count is not a
peak-memory guarantee. Observed anchor RSS861MiB, neural process peaks about1.4GiB;
preparation transient RSS about1.23GiB on macOS. Recorded `ru_maxrss` bytes are
macOS-specific; Linux reports KiB, so convert before using that field on another OS.
Current low-level runner's config fields outside the locked recipe are not a sweep
interface; new resolution/weights/batch/iteration settings need a separate protocol.

Read-only NumPy-memmap-to-Torch conversion may warn in the naming adapter. The resolver
reads these image tensors; it does not mutate them. No warning was treated as numerical
or CUDA evidence. Benchmark clocks exclude several stages as explicitly documented.
Do not benchmark concurrent jobs or infer CUDA behavior from these CPU receipts.

Stop before student/history expansion if independent semantic utility remains this
poor. Any revised anatomical correspondence must start a new, development-informed
protocol; preserve a separate final test. M&Ms has not been used for tuning or scoring.
