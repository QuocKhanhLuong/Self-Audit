# Q3D ResEnc migration — delivery and verification

Implemented the replacement request: Q3D uses the pinned **nnFoundationCNN ResEnc-L pretrained encoder**, a new region decoder/head and the existing imported boundary block. The abandoned SCRATCH design is not part of this delivery. No patient training, dataset download, dataset mutation, or active-job change was performed. The checkpoint was downloaded only for the explicitly authorized inspection and remains outside Git.

## Verification on final implementation

- `python scripts/check_repo.py`: **PASS**, 13 Q3D recipes; [raw output](check_repo.json).
- `python -m pytest -q`: **511 passed**, including the opt-in real checkpoint strict-load and pinned nnssl factory parity tests. [Verbatim output](pytest-final.txt).
- `python scripts/smoke_q3d.py --out /tmp/self-audit-q3d-resenc-final-20261112 --all-arms`: **13/13 completed**; [receipt](smoke-all-arms.json). The synthetic tokenizer gate remains NOT_READY for short usage history/dead-code evidence; the explicitly synthetic override was used. No real-config gate was relaxed.
- Actual pretrained CPU preflight with `--run-erf --erf-samples 16`: checkpoint hash, 448 strict encoder keys, architecture and empirical phase pass. Overall `NOT_READY` refers only to missing TRAIN/GT diagnostics, not teacher provenance. [Receipt](preflight-runtime.json).
- Actual runtime ERF helper agrees with the independent Phase-0 probe to about 1e-6 for its mean offsets. [Runtime receipt](runtime-erf.json).
- [Required cleanup grep](cleanup-check.txt): empty, exit 1 means no matches. [Frozen baseline diff](baseline-check.txt): empty.
- The suite includes the clean repository-copy regression. Test execution does not rewrite tracked reports. The delivery worktree is committed clean; pre-existing unrelated untracked files in the main checkout are retained.

Exact pytest output:

```text
........................................................................ [ 14%]
........................................................................ [ 28%]
........................................................................ [ 42%]
........................................................................ [ 56%]
........................................................................ [ 70%]
........................................................................ [ 84%]
........................................................................ [ 98%]
.......                                                                  [100%]
511 passed in 199.71s (0:03:19)
```

The optional verification environment supplied `Q3D_NNFOUNDATION_CHECKPOINT`, `Q3D_NNSSL_REGISTRY` and `Q3D_NNFOUNDATION_WEIGHTS` pointing to the downloaded pinned checkpoint and source file outside Git. Normal recipes use the portable `${Q3D_NNFOUNDATION_WEIGHTS}` binding; never commit the weight file.

## Phase 0

[Full inspection and key/shape inventory](../q3d-resenc-phase0-20261112/README.md): HF revision `f443c09319988e833b3d9589bf95909e4443d7b2`, checkpoint SHA `ac262d3e8c226c79f9567fddc34d38e011284730ddf2294b5c6c319ce2f22bbd`, 409,579,442 bytes. Encoder strict load: **448 keys, missing=[], unexpected=[]**. Decoder checkpoint keys are excluded, including aliases.

TRAIN median declared ZYX spacing=(1,1,1), shape=(9,256,216); 52/80 unit-spacing records are not independently reconciled to raw NIfTI headers. Strides lock to `[(1,1,1),(2,2,2),(1,2,2),(1,2,2),(1,2,2),(1,2,2)]`, full padding multiple=(2,32,32). Stage-3 main phase step=(2,8,8), offset zero; max skip displacement=(0.5,3.5,3.5).

Sixteen random Z-score samples with actual replicate padding produce mean offsets (0.6721337438,1.9470438957,1.7222030163) voxels. All satisfy the predeclared absolute-mean threshold 2, so the locked correction is zero. Earlier incompatible-center/padding probes are preserved as NONCALIBRATION; no threshold was changed. S1 keeps its independent analytical stage-4 zero phase, explicitly not empirically calibrated, and remains NOT_PLANNED.

## Campaign

[PLAN](../q3d-resenc-plan-20261112/PLAN.json) and [acceptance/lineage proof](../q3d-resenc-plan-20261112/ACCEPTANCE.json). Validation command without `--execute` returns `PLAN_VALIDATED_NOT_RUN`, 15 jobs.

| Priority | K | T | Q | P | J |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 1 | 1 | 1 | v1, v1u, c1, c0 |
| 2 | 5 | 1 | 1 | 1 | v1 |
| 3 | 10 | 1 | 1 | 1 | v1 |
| Total | | 3 | 3 | 3 | 6 |

T=8,000 updates; Q=1,000; P=one pass; J=8,000. Runtime dependency traversal completes the four K8 J jobs before K5/K10. Shared producer source-config hashes are recorded; future real checkpoint hashes remain NOT_RUN. V1/V1u share T/Q/P and C1 shares T/P. `teacher_supervised@8` evaluates the shared T region head, 0 J, `independent_baseline=false`. No extra baseline training job is introduced.

K8 TRAIN SHA is exact. Existing selector reproduces L5 ⊂ L8 ⊂ L10 from locked eligible IDs; full K5/K10 manifest bytes and DEV bytes are still absent locally. No IDs, split, masks or manifest bytes were regenerated. All job blockers in the delivered PLAN are missing TRAIN/development manifests; real checkpoint provenance passes for every teacher-using recipe.

`a3d_teacher_free@8` is reference-only **NOT_RUN**, with no job. Its preserved handoff receipt SHA is `6aa1c4b8772b2c8f1a1962debfebc5dcb046f99eb922535fc0a307c7332395a8`, status `INFO_HANDOFF_ONLY_NOT_RUN`; a completed checkpoint and equivalent budget were not proven.

## Changes and review

[Complete changed/deleted file list](CHANGED_FILES.txt). Removed Q3D legacy teacher factories, source-checkout plumbing, feature hooks, two obsolete feature-source recipes and the old official-channel test; renamed the shared teacher-head baseline to `teacher_supervised`. A/A-3D source, baseline docs/tests and old reports are unchanged.

T warmup freezes only the encoder; optimizer groups deduplicate encoder aliases. T artifacts seal selected DEV patient-macro quality separately from the final learning curve. Student export cold-load remains isolated from all teacher modules and DNA/nnssl.

Claude reviewed teacher, config, runtime, cleanup, preflight and campaign workers, plus Phase-0 geometry; [dispositions](REVIEW_DISPOSITION.md) and the raw review files preserve findings from concurrent snapshots. The post-push independent review is recorded separately once completed.

## NOT_RUN and interpretation limits

Patient T/Q/P/J and DEV/TEST evaluation, GPU forward VRAM/latency, raw-header verification, L-only class-coverage/physiology diagnostics, and independent overlap audit are **NOT_RUN**. CUDA was unavailable in the local verification environment. The largest declared TRAIN volume is (8,512,428); no measured GPU memory number is invented.

Checkpoint provenance follows Table D7, not an independent source-patient audit; weights carry CC-BY-SA-4.0. Software, strict-load, random-input ERF and synthetic metric checks do not establish cardiac efficacy, SOTA, novelty, or seed variability.
