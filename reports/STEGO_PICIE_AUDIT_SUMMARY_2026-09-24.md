# STEGO + PiCIE Audit Summary — 2026-09-24

This report consolidates the current fair-baseline audit state for the
`baseline/stego-picie-recovered` branch into a single handoff note.

## Scope

Covered in this summary:

- STEGO prior provenance hardening
- PiCIE fair-recipe alignment status
- regression and smoke evidence on the current branch
- remaining risks before full-budget training

Not covered here:

- final dev evaluation on full-ish checkpoints
- domain-shift claims
- M&Ms scientific execution

## Branch Decisions

### STEGO

- `STEGO-SA224-FAIR` now rejects downstream STEGO Lightning checkpoints by
  default.
- acceptable fair prior:
  - no local checkpoint, which defers to official DINO teacher loading
  - local DINO full checkpoint with top-level `teacher`
- downstream STEGO checkpoint use is only allowed through the explicit
  override flag:

```bash
--allow-stego-downstream-prior
```

- fair provenance now records the prior classification in
  `*.training_provenance.json`.

### PiCIE

- the fair trainer now follows the intended PiCIE family more closely:
  - two-view rendering
  - per-view mini-batch k-means
  - per-view pseudo-label assignment
  - frozen nonparametric classifiers rebuilt from centroids
  - within-view and across-view CE losses
- the remaining recipe differences are treated as explicit MRI-side
  adaptations and are written into `recipe_audit`.

## Smoke Evidence

### Regression gates

- STEGO focused cardiac tests: `11/11` pass
- PiCIE focused cardiac tests: `9/9` pass

### STEGO smoke

- verified prior:
  `checkpoints/stego/priors/dino_vitbase8_pretrain_full_checkpoint.pth`
- checkpoint format check:
  - top-level `teacher` present
  - no top-level `state_dict`
- fair smoke train completed and produced:
  - checkpoint
  - checkpoint contract
  - training provenance
- provenance confirms:
  - `prior_checkpoint.checkpoint_format = dino_teacher`
  - no downstream-prior override
- scientific smoke `--limit 1` completed with:
  - `RAW_COMPLETE`
  - `SEMANTIC_COMPLETE`

### PiCIE smoke

- fair smoke train completed and produced:
  - checkpoint
  - checkpoint contract
  - training provenance
- checkpoint and provenance both include `recipe_audit`
- key recipe status retained:
  - `classifier_update_path = matched`
  - `epoch_iteration_semantics = intentional_adaptation`
- scientific smoke `--limit 1` completed with:
  - `RAW_COMPLETE`
  - `SEMANTIC_COMPLETE`

## Current Quality Reading

The smoke runs confirm pipeline integrity, not scientific quality.

- STEGO smoke semantic output on the checked sample collapsed to `BG + VOID`
  with foreground Dice `0.0`.
- PiCIE smoke semantic output on the checked sample also collapsed to
  `BG + VOID` with foreground Dice `0.0`.
- this is consistent with the current diagnosis that the raw partitions carry
  weak signal but still fail the cardiac semantic topology resolver.

## Remaining Risks Before Full Train

### High

- PiCIE full training writes its final checkpoint only at the end of the run,
  so an interrupted job loses the run output.
- both fair training paths currently select the last/final checkpoint rather
  than a dev-selected checkpoint.
- the current semantic failure mode is still dominated by topology collapse to
  `BG/VOID`, so a full run may remain scientifically weak even if the pipeline
  stays clean.

### Medium

- STEGO will switch to multi-GPU DDP automatically if the target machine sees
  more than one GPU.
- STEGO fair inference still uses a central-slice-to-3-channel adaptation,
  which is a deliberate compat choice but remains less MRI-native than a true
  contextual image path.

## Bottom Line

- STEGO prior provenance is now clean enough to support a real fair run.
- PiCIE fair recipe is now usable enough to support a real fair run.
- the branch is ready for full-budget training from a pipeline and provenance
  standpoint.
- the main remaining uncertainty is model quality after meaningful training,
  not whether the branch can produce valid scientific artifacts.
