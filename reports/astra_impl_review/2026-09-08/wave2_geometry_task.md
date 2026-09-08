# W2.3 — minimum geometry safety (TASK SPEC, not implemented)

Execute only after Astra dispatch. AGY owns necessary data/common.py, data/acdc.py,
preprocess_acdc.py, evaluation metrics/native helpers and focused tests. Read plan W2.3.
Preserve all work, no pixel recipe changes, training, commits/push, historical report edits.

## Revalidated evidence

VolumeSliceDataset.__getitem__ resizes image/mask but emits pre-resize spacing.
Preprocessor preserves orig_shape/orig_spacing and effective_spacing but not affine or
orientation/transform chain. evaluate_volume_native assumes inverse resize is sufficient
to align native target; array shape and spacing alone cannot establish that provenance.
compute_metrics can use nonunit spacing even when spacing_known=False and label distances
as pixel. These are measurement issues, not justification for new architecture.

## Minimum safe work

- Sample metadata retains original/source grid shape and spacing separately from network
  grid. For in-plane resize H,W -> H2,W2, effective spacing becomes
  (sz, sh*H/H2, sw*W/W2). Explicit source/network axis order. Keep image/mask pixel outputs
  unchanged. Unknown spacing remains unknown, never fabricated physical mm.
- Preserve real affine/orientation/raw shape and explicit resize chain in newly produced
  preprocessing metadata, with a geometry schema version. Validate image-mask geometry
  compatibility before preprocessing. Do not rewrite old datasets/metadata or claim old
  missing affine is recoverable from spacing.
- Existing-output skip path currently recomputes metadata from current raw inputs even
  when it leaves old arrays untouched. Do not stamp fresh geometry/resize provenance on
  skipped historical arrays. Preserve their original metadata or fail explicitly when it
  cannot be verified; test a changed target size with existing output. Do not overwrite
  historical datasets during this task.
- Unknown spacing with supplied nonunit values must not yield mislabeled pixel distances:
  either use unit grid for explicit pixel-only scoring or reject conflicting inputs.
- Physical-spacing public boundaries validate dimensionality, finite and strictly positive
  values before empty-mask early returns. Zero/negative/NaN spacing must not produce a
  physical result. Keep existing empty-class aggregation unchanged in this task.
- Native evaluation must not claim verified physical/native alignment from shape alone.
  Add a clear fail-closed guard for strict native-mm use without sufficient transform/grid
  metadata. If reliable inverse-geometry support requires large refactor, mark native
  reconstruction verification DEFERRED, do not invent an affine or claim native PASS.
- Do not treat arbitrary caller boolean `verified=True` as geometry evidence. Record what
  concrete geometry was checked and limitations. Non-axis-aligned/sheared physical grids
  not supported by spacing-only distance transforms must be rejected or explicitly deferred.
- Keep reporting distinctions between slice_proxy, volume_resized, volume_native and
  distance units. Synthetic bookkeeping tests do not establish raw ACDC correctness.

## Tests / PASS criteria

Anisotropic non-square two-resize spacing fixture; original metadata unchanged;
dataset pixel/mask parity before/after patch; unknown geometry cannot produce native-mm
claim; mismatched image-mask affine rejected where available; explicit pixel distances
independent of ignored physical spacing. Existing legitimate grid-metric tests maintained.
Actual default-DataLoader batches must still collate known/unknown spacing and multiple
cases with different source H/W. Avoid nested None or ragged transform lists in sample
dicts; rich metadata can remain record/dataset-side. Test the builder, not just ds[0].
Run focused -> all tests/ -> source compile, exact commands/output. Concise new report
wave2_geometry_worker.md, explicitly list native paths left DEFERRED and why.

No need to implement general reorientation/resampling or verify raw data unavailable here.
If safe minimal scope is unclear, ask Astra before adding a large geometry subsystem.
