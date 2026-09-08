# W3.1 review checklist — read before implementation completion

These refine the delegated producer scope; W3.2 consumers remain a separate task.

- Hash actual parameters AND buffers, including scalar integer buffers and bfloat16 tensors;
  no NumPy dtype conversion that fails on bfloat16. Include key/shape/dtype boundaries.
- Dataset signatures must come from actual loader.dataset effective records/preprocessing,
  not config-only guesses. Capture actual image_size/depth_axis/normalization/foreground-only
  settings and geometry semantics. Split membership hash does not prove content identity.
- Preserve unknown checkpoint-producing SHA on legacy checkpoints. Current evaluation SHA
  is not the producing SHA; source dirty/provenance reporting must be honest.
- Tiny best != last test must run the helper actually called by runner after Phase C;
  collector/calibration/both diagnostics see selected best parameter and buffer state.
- Ensure validation binding does not restore RNG/optimizer; test RNG byte-state invariant.
- Diagnostic max_batches/subset limits are protocol metadata, not interchangeable full-cohort
  measurements. Do not claim diagnostics at max_val_batches cover the entire validation set.
- Do not change Phase C best selection criterion or calibration objective during binding.
  Existing thresholds/calibration from old entropy semantics will be handled fail-closed in W3.2.

Keep helpers small; no heavyweight registry/system needed. Exact local focused/full outputs,
diff-check and compile remain mandatory. No training, real bank, commits or push here.

## Claude draft review, 2026-09-08 (not final approval)

Independent focused run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests/test_checkpoint_binding.py`:
**18 passed in 4.49s**. The actual extracted runner branch uses real cache/sweep/diagnostics
behind spies and observes best rather than last. This does not establish complete lineage correctness.

Additional review requests sent to dispatch `ctx_8d1708d5bcfb` in `msg_152831b6167e`:

- Resolve string metric contracts into complete versioned metadata (draft produced version `None`).
- Keep cohort counts out of preprocessing recipe signatures; disjoint cohort sizes are not recipe changes.
- Do not erase Subset/sampler/drop-last selection while claiming full-cohort coverage.
- Do not assign a known normalization recipe to an unknown generic dataset.
- Include applicable geometry/preprocessing semantics and verify live model attributes.
- Report direct compilation exit status, not the final `tail` command status from a pipeline.

Status: **REVISE pending worker response**, not W3.1 PASS and not Proposal 1 completion.

Independent draft subset probe: `d=TensorDataset(torch.arange(10)); s=Subset(d,[2,4,6])`.
`_unwrap_dataset(s)[1]` returned `None` (expected `[2,4,6]`);
`_unwrap_dataset(DataLoader(Subset(s,[1]),batch_size=1))` raised `IndexError`
(expected base indices `[4]`). Sent in `msg_bb82475d3b8c` with required regressions.
Also requested metadata-only geometry description instead of invoking `dataset[0]`,
which may load GT, apply transforms and alter RNG just to describe a recipe.
