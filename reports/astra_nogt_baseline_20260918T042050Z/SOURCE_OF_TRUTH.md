# Source of truth

Selected baseline: **C0 PROVISIONAL**, deterministic image-only partition + frozen
explicit ontology, no neural training. This document identifies the implementation;
scientific status is governed by BASELINE_DECISION and the measured receipts.

* Worktree: `/Users/alvinluong/Self-Audit-nogt`.
* Branch: `feat/no-gt-baseline-20260918`.
* Fetched base `origin/main`: `826a11ca9c37fd024c9d7d83df594f8b051ec9ca`.
* Task implementation commit: recorded in `evidence/commit_identity.json` after the
  code-only commit; report commit is its descendant on this branch. No main merge/push.
* Existing main checkout remains at `605e62b5e11ce1fc792068813e9ec2b829cefd4f`;
  its unrelated dirty files are not part of this task.
* Existing event worktree remains at `9568217ffbdfc2abf4956eabad8e7c48c455627c`.

## Actual call path

| Responsibility | New source, symbol and line |
|---|---|
| Entry | `scripts/run_nogt_suite.py:main` → separate prepare/anchor/train subprocesses; `--baseline-only` selects prepare+anchor only |
| Method dispatcher | `src/self_audit_nogt/runner.py:main:245`; activates file firewall before any data operation |
| Data | `data.py:prepare_images:59`; image-only byte mirror → NIfTI → volume image percentiles → full-FOV resize/pad → manifest |
| Reference boundary | `data.py:ReadFirewall:20`; actual Python opens, reference filename refusal, saved reads |
| Feature/partition | `core.py:cluster_image:30`; descriptor[N,4], centers[4,4], nearest-center labels[H,W] for15 iterations |
| ID alignment | `core.py:canonicalize:20`; intensity-order groups, no GT matching |
| Anatomy | `core.py:name_partition:50` → unchanged `self_audit_maskfree/ontology.py:resolve_roles:558` |
| Validity/export | `runner.py:export_predictions:48`; UNKNOWN255 when validity0; cropped FOV to `data.py:native_volume:131` |
| Freeze | `runner.py:freeze:115`; source/config/split/checkpoint/prediction hashes and fixed semantic order |
| Post-freeze evaluator | `scripts/evaluate_nogt_frozen.py:main`; validates every NPZ and checkpoint hash before first reference read |
| Controls | `core.py:SmallAnnotator:63`, `cache_loss:98`, `direct_loss:81`; `runner.py:train:179` |
| Measurements | `runner.py:benchmark:126`; `scripts/profile_maskfree_leaves.py`; `scripts/summarize_nogt_results.py` is posthoc analysis only |

Current method never imports the independent evaluator or aggregation script. Images
are float32[B,1,224,224], image-FOV support bool[B,224,224]. C1 logits are
float32[B,4,224,224], anonymous targets uint8[B,224,224]; no current method tensor
comes from manual masks. FittingView gets only the unpadded image[1,H,W], all-true
image support[H,W] and repeated image features[3,H,W], no annotation/reference input.
Those features are a minimal container for the frozen naming API, not a learned
medical representation. No history persists between samples.

## Fixed identities

Configuration: `configs/nogt_baseline_224.json`.
SHA256 `bfe29e093fea13f64d43c0dcf912d3c6b08195630206bc5202289384203d8396`.
The implemented feature weights,15 iterations, network widths and direct-loss weights
are code constants matching that locked recipe; this is not a generic configuration
sweep engine. Changing code/config creates a new experimental identity.

Protocol SHA256 `7dac65d495be646b4428d4ef14083407220a3f8850edcfe8b1c11b2a67940a9c`,
locked `2026-09-18T04:30:16.738352+00:00`. `evidence/protocol_lock.json` is the receipt.
Split SHA256 `bc42574017276a6defc668bf533330fd41a64a5e90a75ea1b8958fc9e469265a`.
Every run's exact module/ontology/contract hash and frozen output hashes are in
`evidence/runs/<arm>/FROZEN.json`; the base commit identifies inherited dependencies.
`evidence/environment.json` records the actual Python/Torch/NumPy/SciPy/Nibabel host.

`evidence/SELECTION_BEFORE_GT.json` fixes C0 before all successful evaluator timestamps.
`evidence/aggregate.json` independently verifies that ordering and matched sample
sequences. Earlier development knowledge is disclosed; timestamps do not certify a
project that never inspected labels. No GT-selected checkpoint or output mapping exists.

## Artifacts and ownership

Raw data, image cache and binaries remain outside Git:

* `/tmp/astra_event_acdc_training/training` — original local-copy cohort.
* `/tmp/astra_nogt_cache224_v3` — image-only mirror, normalized images/supports,
  anonymous C0 cache, manifest and original split bytes.
* `/tmp/astra_nogt_runs/C0` and four C1 directories — frozen native NPZ/NIfTI;
  C1 `final.pt`, logs, per-step traces, timing and file reads.
* `/tmp/astra_nogt_c0_e2e_smoke` — separate synthetic CLI integration fixture.

Committed reports contain metrics, hashes, source provenance, timing arrays,
patient-level analysis, curves, test receipts and worker accounting. They contain
no raw image volume, mask or model checkpoint. Native NIfTI is created from the same
arrays as the hash-frozen NPZ; **the evaluator validates NPZ, not NIfTI file hashes**.
Do not replace a validated NPZ with an unchecked export in a downstream evaluation.

Original canonical, event, Candidate-C, REW and maskfree source/configs were not edited.
Only new namespace/config/scripts/test/report paths are staged. Before/after checkout
status and diff hashes are stored in `evidence/*_before.json` and `*_after.json`.
The original dirty work is preserved. Temporary artifacts need explicit retention
before `/tmp` cleanup; committed hashes cannot reconstruct deleted private inputs.
