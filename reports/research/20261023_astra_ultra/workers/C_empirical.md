# C — empirical evidence and immutable Git audit

Status: independently challenged by worker F and revised, 2026-10-23. No training, GPU work, patient-file access, checkpoint loading, weight download, or remote runtime inspection occurred in this audit. Only Git-tracked source, metadata, aggregate metrics and receipts were read. All numerical checks below were executed locally with standard-library Python.

## Decision

The latest experimental branch contains four completed baseline recipes on one already exposed development split. It contains **no completed, scored full joint-guard experiment**. The conventional full-EMA recipe has the strongest observed endpoint among those four; the same-decoder-block unguarded recipe has materially worse RV Dice than supervised-only. Those observations justify testing whether the proposed guard can prevent harmful decoder updates, but do not establish that it does so, that joint allocation is useful, or that either mechanism improves unseen patients.

The 72-step joint-guard pilot did complete according to a tracked external receipt, including four accepted pseudo transactions. Calling the guard simply “never executed” would therefore be false. The precise boundary is: **bounded execution/capacity was reported; segmentation efficacy of the full guarded recipe was not measured**.

## Pinned identities and evidence levels

| Identity | Immutable reference | What it establishes |
|---|---|---|
| Canonical current main | `637f38ced217fe1db2347a87924695533056d8f0` | Semi-supervised v2 implementation and canonical contracts; not the source snapshot that produced the final B52 results |
| Earlier GPU experiment | `970f142a76f417c516db1c9f18f60a1dfb38d342` | Native-grid B16 pilots and a subsequently stopped supervised run |
| Fixed-crop source change | `7940ec2` | B52/C256 protocol and valid-padding changes used by completed runs |
| Latest experiment publication | `8960b6baac4c6600553d900c303f192c2fcfa4f0` | Final four-arm result tables/receipts plus packaging tools; not a fifth scientific result |
| Historical archive | `924265f7df604d3dca84dda49578b1a0cf28d821` | Frozen-Q, pre-semisupervised full50 training/evaluation; a different system and supervision regime |

Evidence vocabulary:

- **VERIFIED_LOCAL_SOURCE**: immutable Git contents and source fingerprint recomputation.
- **RECOMPUTED_TRACKED_AGGREGATES**: arithmetic on saved per-patient Dice/confusion counts; does not certify those counts against images or predictions.
- **REPORTED_EXTERNAL_GPU**: external execution, timing, memory, coverage, acceptance and artifact hashes recorded in tracked receipts. No current local GPU replay or payload verification.
- **UNAVAILABLE/NOT_RUN**: absent tracked evidence or explicitly cancelled experiments. Hashes naming absent payloads are not payload verification.

`source_digest()` hashes the sorted top-level `self_audit/*.py` files followed by `train.py`, with `basename:SHA256\n` records. It does **not** hash the entire Git tree, preparation script, configs, external DINO source, dependencies, runtime, dataset or initial parameter tensor. Recomputing that exact algorithm from Git blobs gives:

| Git tree | Recomputed source SHA256 |
|---|---|
| `637f38c` | `3cca8cf4925ab7e98515ee3d731e2d4d0336f72129238d9efdc457c2af9735cf` |
| `970f142` | `858d8b31f473ef2362a3f8e977f540d3e9c93afe36f602ef916d4ce09356c30a` |
| `7940ec2`, `8960b6b` | `6e12885e44b725c132a01437dc6179cbf524b94c51dd5a5f5254701eb78cf442` |

All four final measured receipts carry the last fingerprint. This is a positive identity check for that hashed source subset, not independent certification of the whole executed machine state. `PREFLIGHT.json` retains older source/current-run fields and `WEIGHTS_ACCESS.json` retains an early `patient_training_started:false`; they are mixed-stage receipts, not the final status authority. The completed-summary, final tables and execution policy supersede those status fields. Evidence: `8960b6b:self_audit/engine.py:54–57`, `results/logs/PREFLIGHT.json`, `results/logs/WEIGHTS_ACCESS.json`, `results/logs/COMPLETED_SUMMARY.json`; local [C_recomputation.json](../evidence/C_recomputation.json).

## Data, label budget, model and execution contract

All four final recipes declare L8, subset seed 42, initialization seed 42, 10,000 iterations, 1,000 warmup iterations, batch 52, crop 256, FP32 and fixed final **student** checkpoint. The source resets Torch and NumPy seeds before building a fresh model; the queue launches each method independently. There is no exact-resume path. Same seeded initialization is supported by source and matching configuration, but a shared initial-state tensor hash is not among the tracked final receipts. The queued JSON model fields are placeholders resolved by the external `/workspace/self-audit-v2/BACKBONE.json`; that effective file and final checkpoint config are not tracked here.

The backbone is DINOv3 ConvNeXt-Tiny with a native decoder. Receipt weight SHA256 is `21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54`; external DINO source commit is `6876159a11b4df116f30f667f8c9888617df0751`. Weights came from the explicitly recorded mirror. The official short filename hash and mirror LFS match are reported; an independently verified full official upstream digest is unavailable. These are pretrained initialization claims, not a named-anatomy oracle or complete upstream certification.

The experimental preflight reports NVIDIA RTX PRO 4000 Blackwell, Python3.12.15, Torch2.14.1+cu132 and CUDA availability. This is an external runtime receipt, not a local hardware measurement or a per-final-run environment attestation. Each final measured JSON omits a complete runtime/dependency manifest, so the source-subset hash must not be expanded into that claim.

The split contains 80 TRAIN, 18 development VAL and 50 TEST IDs. `patient055` and `patient076` are explicitly excluded; TRAIN IDs were preserved after review of duplicate-containing data. Split SHA256 is `b826eaf494760079aa559b0c0c12f7444951d779ebbb5ed291c0398f20bf835f`. L8/S42 contains `patient007,010,011,024,026,041,077,099`; its declared budget includes both supervised and audit labels, 16 reference frames and 154 labeled slices. The remaining 72 TRAIN patients provide image-only cine data. The data preparation table reports zero U-mask reads. The final common training-manifest digest is `e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4`. Patient-file content and access logs were not independently replayed in this task.

Batch 52 means 52 sampled slices/crops from **one patient per role**, not 52 independent patients. Supervised and audit patients differ within a transaction. Audit labels are TRAIN optimization data; rotating audit patients do not form a held-out risk sample. With `li=step%8` and `ui=step%72`, each U patient is permanently paired to one supervised patient; the audit patient is randomized. This phase locking is shared by arms and is a protocol limitation, not proof of leakage.

All arms load labeled, audit and U samples and generate the strong view every iteration, preserving the data RNG schedule. `supervised_extra` adds 9,000 extra labeled optimization calls after warmup: 19,000 labeled updates versus 10,000 for the other arms. Derived sample-processing counts are 520,000 supervised slices and 520,000 audit slices drawn for each arm, with 468,000 extra labeled slices actually optimized in `supervised_extra`; these are repeated draws from eight patients, not annotation counts. EMA-full/block report 9,000 accepted pseudo transactions, one per post-warmup iteration. Raw step records are unavailable to independently verify those counts.

Evidence: `8960b6b:results/configs/L8_S42_joint_guard.json:1–29`; equivalent configs for the four completed methods; `results/configs/PRIORITY_QUEUE.sh:24–33`; `self_audit/engine.py:82–96,115–171,201–215`; `self_audit/data.py:124–152`; `results/DATA_PREPARATION.csv:2–10`; `splits/acdc_reviewed_locked_train80_val18.json:1–168`; `results/logs/WEIGHTS_ACCESS.json:1–22`; `results/REPORT.md:29–35`.

## Completed outcomes

Each Dice is the mean across the 18 patients of each patient's mean Dice over reference frames **where that class is present**. A frame Dice is a 3D volume ratio pooling Z,H,W; the distinction here is averaging those ED/ES volume Dice versus pooling counts across ED/ES or across patients. The prediction is dense raw argmax; the unknown column is zero in the saved final confusions. The available patient metric rows can be averaged back to the headline values, but their mean-frame Dice cannot be reconstructed from the pooled-frame confusion alone.

| Method | RV Dice % | MYO Dice % | LV Dice % | Foreground macro %¹ | Loop/final-save h² | Peak allocated GiB² | Accepted pseudo² |
|---|---:|---:|---:|---:|---:|---:|---:|
| supervised | 60.7510 | 77.1315 | 88.5355 | 75.4727 | 2.58165 | 18.4350 | 0 |
| supervised_extra | 64.1921 | 77.1738 | 87.4644 | 76.2768 | 4.03410 | 18.4350 | 0 |
| ema_full | 73.6249 | 80.6737 | 91.4156 | 81.9047 | 4.38679 | 18.5273 | 9000 |
| ema_block | 52.2591 | 80.2877 | 87.8306 | 73.4592 | 13.47240 | 22.3207 | 9000 |

¹ Locally derived unweighted average of RV/MYO/LV patient macro Dice; not a previously preregistered endpoint. ² External execution receipts, locally unit-converted, not locally timed measurements. Timing starts after startup/data checks/model initialization and includes the loop plus final checkpoint save. It excludes GPU reservation/billing and does not represent optimized implementations of all baselines.

| Paired comparison | RV Δ pp (patients better/worse) | MYO Δ pp (better/worse) | LV Δ pp (better/worse) | Foreground Δ pp |
|---|---:|---:|---:|---:|
| ema_full − supervised | +12.8739 (17/1) | +3.5422 (12/6) | +2.8800 (16/2) | +6.4320 |
| ema_block − supervised | −8.4918 (5/13) | +3.1562 (14/4) | −0.7050 (9/9) | −2.0135 |
| ema_full − ema_block | +21.3657 (18/0) | +0.3860 (9/9) | +3.5850 (15/3) | +8.4456 |
| supervised_extra − supervised | +3.4411 (13/5) | +0.0423 (9/9) | −1.0712 (10/8) | +0.8041 |

The RV loss of block versus supervised is heterogeneous: patient deltas span −44.0856 to +9.4342 percentage points. Aggregate BG→RV false positives rise from 10,997 to 100,452; total RV FP rises from 17,125 to 105,066, while RV FN falls from 121,236 to 108,532. Thus the bad RV endpoint is not simply more RV misses; the saved confusion indicates substantial RV overprediction. This is a diagnostic description of those predictions, not a causal explanation of the training dynamics.

The strongest full-EMA foreground score still has collateral tradeoffs: BG→MYO FP rises from 22,574 for supervised to 25,081, although total MYO FP falls from 49,266 to 43,559 and MYO FN falls from 59,050 to 46,611. Higher mean Dice does not imply every error type or every patient improves.

These paired win counts/deltas are descriptive. One initialization seed and one subset seed were completed. No across-training-seed variance, selection-adjusted interval, independent confirmation or clinical claim follows from 18 already exposed patients. Pixel counts are not independent sample sizes.

Evidence: `8960b6b:results/RESULTS_MAIN.csv:1–5`, `RESULTS_ABLATION.csv:1–5`, `PATIENT_METRICS.csv:1–289`; `self_audit/evaluation.py:18–26,64–96`; `scripts/collect_acdc_results.py:42–68`. Arithmetic: [C_recomputation.json](../evidence/C_recomputation.json), generated by [C_recompute.py](../probes/C_recompute.py).

## What is and is not a matched comparison

The four methods share the declared data recipe, source fingerprint, subset/initialization seeds, final iteration budget and evaluation population. The source deliberately generates the same data and augmentation RNG draws for every arm. This supports a controlled recipe comparison, subject to missing runtime schedule/initial-state receipts.

However, EMA-full versus EMA-block changes more than access to encoder parameters. `ema_full` uses cross-entropy on all trainable parameters, clips total gradients at 5 and steps the existing AdamW body optimizer (`3e-5`) and SGD head optimizer (`3e-3`). Block control instead builds pseudo-class directions on decoder/head parameters, uses `pseudo_lr=1e-3` and commits the selected displacement directly. Adam moments, clipping, parameter scope and effective step scaling differ. Therefore the +8.4456 pp foreground difference is **not** a causal estimate of scope alone, allocation, or a guard.

`ema_block` also computes diagnostic audit responses/JVPs despite accepting all candidates. Its 5.2185× supervised time and 3.0711× full-EMA time are measurements of this instrumented implementation. They do not show that ordinary EMA intrinsically costs that much. Equal 10,000 iterations are neither equal labeled optimization nor equal compute. `supervised_extra` is a label-use control, not a measured equal-wall-time control.

`RESULTS_ABLATION.csv` lists `scope=decoder` even for EMA-full because the collector copies `cfg['update_scope']`; `engine.py` explicitly overrides the actual pseudo scope to full for that method. Reading the CSV scope literally would misdescribe the experiment.

Evidence: `8960b6b:self_audit/engine.py:133–145,168–200`; `self_audit/decoder_control.py:13–29`; `scripts/collect_acdc_results.py:57`; canonical `637f38c:docs/EXPERIMENTS.md:52–64`.

## Joint guard: completed pilots, incomplete efficacy test

| Receipt | Scope/budget | Reported outcome | Admissible inference |
|---|---|---|---|
| B16 head pilot | 32 steps, warmup 8, L8; no fixed crop | 12.0902 s, 5.7020 GiB, 0 accepted, 11 no-candidate iterations; 8 L and 32/72 U seen | Head code was exercised in a bounded reported GPU run; no coverage-complete or efficacy result |
| B16 decoder pilot | 32 steps, warmup 8, L8 | 14.4592 s, 11.7579 GiB, 0 accepted, 13 decoder JVPs; 8 L and 32/72 U seen | Decoder response path reported finite/executed, no quality result |
| C256/B52 decoder pilot | 72 steps, warmup 8, L8 | 158.8487 s; 4 accepted, 14 no-candidate post-warmup iterations, 64 JVPs; all 8 L/audit and 72 U seen; `quality=NOT_EVALUATED` | Coverage-complete bounded execution receipt and nonzero acceptance; no Dice endpoint |
| C256/B52 transaction profile | One selected L/audit/U transaction, fresh optimizer, checkpoint-derived model | 4 candidates; alpha `[0,1,1,1]`; actual risks decrease on all three current audit groups; 5.6819 s; 22.3163 GiB | One finite transaction can pass; cannot estimate training efficacy or population safety |
| Full guarded matrix | L4/8/16 × subset seeds 42/123/2026 | Five allocation/guard methods and other budget/seed runs cancelled; 77/81 total planned cells `NOT_RUN_USER_STOP` | No full-guard comparative patient endpoint |

For the 72-step pilot, locally derived acceptance is 4/64 = 6.25% of post-warmup iterations, or 4/(64−14) = 8% among iterations reported to have candidates. The remaining 46 candidate-bearing iterations are **not** evidence of nonlinear guard rejection specifically: raw steps are absent, so zero local allocation, numerical fail-closed behavior and failed finite trials cannot be separated. The pilot warmup of eight is also different from the 1,000-step full protocol. Its stall fraction cannot simply be projected to a later trained teacher.

Both B16 pilot completion receipts carry `858d8b31...`, matching the recomputed `970f142` source subset, and report zero finite-trial rejections; their zero accepted counts must not be called evidence that nonlinear backtracking rejected thirteen proposals. The B52 transaction profile uses source fingerprint `5d3a9fed...`, whereas the 72-step pilot/final runs use `6e12885...`. It is a separate diagnostic receipt. B54 also reports four candidates and PASS, with lower allocated peak than B52 for that sampled transaction. Thus B52 is the chosen operational recipe, not a proof of maximal feasible batch size or worst-case capacity. The early repeated-single-slice B16 profiling receipt likewise is not a full cine/batch training benchmark.

The native B16 full run was stopped after logged step 1,515 (350.4955 s) for user-requested VRAM reconfiguration, with step-1000 checkpoint retained externally. Fresh initialization was used for the changed B52 recipe; it was not an exact resume. The final stop policy explicitly says finish EMA-block, cancel all unstarted, and skip TEST inference. No scientific futility or superiority stopping boundary was used, and no evidence supports implying that joint guard failed a full run.

Evidence: `8960b6b:results/PILOT_SUMMARY.csv:1–3`; `results/logs/pilot_B16_head_32_REPORT.json`, `pilot_B16_decoder_32_REPORT.json`; `results/configs/PILOT_C256_B52_72.json:1–29`; `results/logs/PILOT_C256_B52_72_REPORT.json:1–118`; `results/logs/transaction_C256_B52.json:1–274`; `results/CROP_VRAM_PROFILE.csv:1–6`; `results/logs/STOPPED_NATIVE_B16.json:1–12`; `results/logs/EXECUTION_POLICY.json:1–8`; `results/EXPERIMENT_STATUS.csv:1–82`; `results/REPORT.md:19–25`.

## Checkpoint lineage and replay boundary

| Final student | Manifest SHA256 | Payload SHA256 |
|---|---|---|
| supervised | `a0f8e6174ddc4f001fd048935bd2a84996bb358639a8d9e5aa61031e4d233f7d` | `c3aebe731c7d9f2a191f81e20898038b91ef4813f48c0dfc03ce63b805716a82` |
| supervised_extra | `4faa990bc70a34c419b6dc3ec29739f9292d4679fef3a67f1a001ca1da727459` | `ebe19aa7fb39df447f3479e0adaab17744c44e52883b9127336f4c360c8ebce9` |
| ema_full | `9eec782e0d1aa20f99d47fb398197d4a5441812b3e9f3517c3e9fbba454c284a` | `edbce7e5da09e33ec46586b71fc5807102fe5b8849aee6eed7ad768257b2e89d` |
| ema_block | `6a473c54727e9d6b549ef52c1e39cafc0d0239cdf711e861d7bf6515727ffcc0` | `1fba9783c6fff82291eed135fcd286acda7b29c6dd238258abc21f9d3c6c57b9` |

The per-run measured JSON records evaluation-recipe and full-metrics digests too. Those digests differ by checkpoint as expected. The referenced checkpoint manifests, payloads, effective runtime configs, STARTUP/steps logs, frozen probability arrays and full v2 METRICS/RECIPE files are not in the tracked experimental tree. A whole-tree inventory found no tracked `.pt`, `.pth`, `.npy`, `.npz` or `.jsonl` payloads at `8960b6b`. Compact figure provenance merely repeats digests of externally retained artifacts; it is not a second independent measurement. Packaging/upload tools added in the final commit do not themselves prove a successful upload or verify current remote availability.

The strongest local verification is therefore internal arithmetic and source-subset identity: 2,016 confusion-derived fields; 576 presence/missing-HD95 conditions; 16 patient-mean→summary and 12 patient-mean→main matches; four seconds→hours matches; 72 paired reference-marginal equality checks; four source-digest matches. The count checks include TP/FP/FN, pooled-frame IoU, precision/recall and BG→MYO FP. They do **not** validate the original images, annotation correctness, reference-frame dice, checkpoint bytes, GPU execution or generalization.

Evidence: `8960b6b:results/logs/COMPLETED_SUMMARY.json:1–142`; `results/logs/L8_S42_*_MEASURED.json:1–34`; `results/REPORT.md:21–25,35`; `self_audit/engine.py:100–112`; [C_extraction_manifest.json](../evidence/C_extraction_manifest.json) and [C_recomputation.json](../evidence/C_recomputation.json).

## Exposed validation and metric denominator limits

VAL18 was already used in historical development and is explicitly designated development-exposed by both canonical protocol and experimental reports. Freezing a final checkpoint/predictions before reading reference masks for a particular evaluation protects that evaluation's execution order; it does not make previously exposed patients pristine confirmation. The four endpoint observations are useful development evidence.

The evaluator averages each patient's class Dice over reference frames with nonzero reference count; absent-reference frames are omitted for that class's Dice. Their false positives remain visible in pooled confusion and precision, but they are not penalized in the displayed mean-frame Dice. All final headline classes have 18 present patients; the compact CSV alone does not tell how many class-present frames contributed for each patient. Require per-frame support counts and an explicit absent-class endpoint before treating the current mean as comprehensive anatomical quality.

No tracked TEST endpoint or inference result exists, and the recorded policy cancels TEST inference; remote availability was not inspected. No physical HD95 is implemented/certified in this cache path. No independent external domain, multi-seed result, exact BCP/HCO reproduction, formal noninferiority result, or matched-compute learning curve is supported by the tracked artifacts. `sign` is a proposed internal control, not an implemented HCO replication.

Evidence: `637f38c:AGENTS.md:20–23`, `docs/EXPERIMENTS.md:32–37,66–83`; `8960b6b:self_audit/evaluation.py:70–95`; `results/REPORT.md:7,17–25,41`.

## Historical full50 is a negative control on overclaiming, not a v2 ablation

Archive `924265f` contains a reported RTX4070 full50 run of `layer_guided_convnext_tiny_v1`: batch one, 50 epochs, 1,010,750 optimizer steps, 314,900 semantic updates, 98,250.4722 s (27.2918 h), seed42, AdamW LR3e-4, historical frozen-Q/reliability semantic objective plus feature cosine objective. Guard/refiner were off. It was not trained with the v2 L8 dense supervision or native detail objective. The source plan records handoff `c4973553...` and external source `318628a9464e075fa2d4a7f63d4622c9ff27fb11`; archived training artifact collection occurred later. Those identities must not be collapsed into current main.

The subsequent frozen evaluation at `924265f:reports/dinov3_full50_val_20261019` scored 18 exposed patients/36 ED/ES frames. Raw Dice was RV18.3558%, MYO14.9129%, LV36.0623%, foreground macro23.1103%. Fixed probability0.7/margin0.2 gating yielded **zero coverage and zero full-domain gated Dice**. Native-minus-lifted-coarse logits and raw detail were reported zero. AUROC/AP values are reported but their pixel rankings cannot be recomputed from confusion counts. This is evidence against interpreting long training completion, low feature loss, or favorable ranking metrics as useful accepted named segmentation.

Unlike the latest v2 compact CSV, this archive includes per-frame raw/gated confusion. Local [C_historical_recompute.py](../probes/C_historical_recompute.py) independently reconstructs 288 frame Dice values, 72 coverage values, two pooled confusions and eight macro values from saved counts. This remains saved-count arithmetic, not raw array/reference replay. Full50 checkpoint manifest `fb9d830574845c6e0802c89b6ebe4f5647793326c6a613391595be847670ca85` points to untracked payload `76061e081896bf07f7a703570a86a0b913cab1d16f15ab27e1768c4f2faee654`.

Comparing full50's23.11% with v2's81.90% is **not** a matched improvement claim: dense-label access, targets, architecture, preprocessing, optimizer, step count, hardware and semantics all changed. The archive supplies cautionary historical evidence; only split metadata was explicitly reused for v2, not checkpoints/teacher banks.

Evidence: `924265f:reports/astra_dinov3_4070_20261019/summary.json:1–29`, `full50/CONFIG.json:1–27`, `PLAN.json:1–31`; `reports/dinov3_full50_val_20261019/REPORT.md:1–43`, `RECIPE.json:1–95`, `METRICS.json:1–3673`; local [C_historical_recomputation.json](../evidence/C_historical_recomputation.json).

## Falsification-first next step

No compute is authorized or launched by this recommendation. The user-requested stop remains the actual execution policy.

1. Recover and verify a minimal provenance bundle for the existing four runs: effective config, complete checkpoint manifest, initial-state hash if available, STARTUP/source/environment receipts, full per-frame METRICS and compact step statistics. Preserve the raw files externally. Verify final weights against stored digests before deriving new causal claims from them. This can resolve the current auditability limit without more training.
2. Freeze one exact experimental source and objective before any new authorized run. If preserving the existing protocol, use the verified `7940ec2`/`8960b6b` hashed source subset and unchanged crop/padding/denominator recipe. If changing sampling, optimizer compatibility or response normalization, give the new factor a new identity and rerun its matched controls. Do not attribute old results to current-main single-slice code.
3. First separate allocation from uniform shrinkage: compare J=`joint_guard`, S=one aggregate direction with `alpha=a·1` and the identical finite classwise guard, and P=`ema_block`. S must use scalar ridge `m*lambda` to match the joint objective restricted to that subspace; it is not available in the current CLI. A frozen same-state transaction replay can hold actual q and candidate directions identical. Fresh full training instead holds initialization, candidate-generation policy, threshold, denominator, data stream, decoder scope and pseudo scaling fixed; after the first different update, endogenous EMA teachers and actual candidates may diverge. That endpoint estimates the whole allocation policy, not a direct same-candidate effect at every iteration. `independent_guard` is an optional extra mechanism control, not a substitute for S. Include fresh supervised/extra-supervised/full-EMA controls according to the practical-utility claim and declare any omitted comparison out of scope. Use final fixed checkpoints and a new, untouched confirmation plan; do not choose thresholds by these already inspected VAL18 outcomes. Full-EMA remains a practical stronger comparator whose optimizer/scope confound is acknowledged.
4. Require a mechanism falsification gate before expanding budget/seeds: nontrivial accepted displacement, observed positive audit support, finite guard compliance, no sustained reject-all regime, and a distinction between candidate absence, zero allocation, numerical failure and finite-trial rejection. Record classwise candidate mass, alpha/displacement norms, local/actual risk error and cumulative classwise movement. Acceptance alone is not success; a guard that returns the supervised model is a null efficacy result.
5. Report paired patient Dice and classwise/absent-frame FP endpoints, repeated label subsets and initialization seeds, plus instrumented and optimized timing and an explicitly matched-compute labeled control. Predeclare patient-level uncertainty and a meaningful improvement/noninferiority threshold. A positive one-seed development result permits confirmation work, not a publication-level efficacy claim.

The strongest supported conclusion is that ordinary dense-label-initialized full EMA currently provides the best observed development endpoint among completed recipes, while unguarded decoder-only pseudo updates have a clear RV tradeoff. Whether joint allocation plus finite TRAIN Brier checks can improve that tradeoff remains an unmeasured scientific question.

## Reproduction and memory provenance

Run from this worktree:

```text
rtk proxy python reports/research/20261023_astra_ultra/probes/C_recompute.py
rtk proxy python reports/research/20261023_astra_ultra/probes/C_historical_recompute.py
```

Both completed successfully. No package installation or patient/runtime import was required. The coordinator separately verified canonical-main software tests; those results are not repeated here as empirical evidence.

Worker F independently verified the headline means and experimental source fingerprint, and challenged candidate identity across divergent EMA trajectories, the missing scalar control, frame-Dice geometry, and the scope of TEST absence. All four clarifications are incorporated above. F also computed an alternate equal-patient Dice endpoint pooling ED/ES confusion first; its ranking is unchanged, but that sensitivity does not reconstruct omitted frame-support information. See `../evidence/F_pooled_patient_sensitivity.json` and F's review for that independently owned calculation.

Memory was used only to identify historical evidence boundaries, then superseded with current immutable Git artifacts: `/Users/alvinluong/.codex/memories/MEMORY.md:25–40,352–360,408–409`; rollout IDs `01a13f0c-a763-7043-8bbb-e1fc27b00972`, `01a12e03-e80f-7390-93cd-a6d77afeba1a`. No unverified historical memory number is used as a current result.
