**Geometry attribution audit — 2026-10-14**

**Verdict.** The observed TRAIN-bank intervention is removal of the RV side prior from 28 patients without coded header geometry. It is not a real-data test of recovering oblique directions. All 52 coded patients have the same axis-aligned selected sform, and the new covector is a positive multiple of the old `+x` rule. Their reported hard counts are unchanged for every class and patient. The uncoded group loses 4,322 accepted RV pixels, gains 192 BG and 53 MYO pixels, and has unchanged LV counts. These are verified reductions of archived records, not verified anatomical corrections or a causal explanation for all downstream prediction changes.

The coded cohort is also not certified anatomy: every coded TRAIN image has `sform_code=2`, `qform_code=0`, selected affine `diag(-1,-1,1,1)`, and spatial unit `unknown`. A software-recognized coordinate system is not independent evidence that the file preserves patient laterality. No raw NIfTI, bank array, checkpoint, target mask, or TEST prediction array was loaded for this audit.

**Scope and evidence vocabulary.** Workspace HEAD was verified as `b937e3b746608725e43d3bc3cef8861666d152dc`. Baseline *training* source is `d5a45f31481f0a702a1f9d5d418889fe704e9105`; new-run base is `7d1c6a04064cda473d1eccfbed818392c37b1a37`, plus archived launcher patch SHA256 `ecbcd639fbaa010ac3f7fca9ac5ba3272278806ce92eecc8171c4833f70815ee`. Baseline forensic code was later run at `7d1c6a0`; that does not change the source that trained the baseline. All nine baseline source hashes in each of its bank/recipe/student receipts match the `d5a45f3` Git blobs.

`VERIFIED` means checked against the currently available source/JSON bytes or rerun CPU geometry checks. `REPORTED` means historical execution or array-derived measurements stored in those bytes, not reexecuted here. `DERIVED` means arithmetic or a stated conditional proof. `BLOCKED` identifies information absent from this package. JSON metadata agreement does not establish unseen payload identity. Counts and physical-coordinate checks never establish named anatomical accuracy.

In citations below, `E/` means `reports/stable_guidance_world_v2_20261014/`, `S/` means `E/executed_source/src/self_audit_guidance/`, `B` means `E/baseline_forensics/train_all/DIAGNOSTICS.json`, and `N` means `E/run_evidence/run/bank/FROZEN.json`. The audited code is the executed snapshot, not historical top-level `src/`.

**Reproducible checks.** Run from the workspace:

```sh
rtk proxy python -B reports/guidance_research_continuation_20261014/GEOMETRY_ATTRIBUTION_CHECK.py
```

The check passed and writes only same-prefix artifacts. [CHECKS](GEOMETRY_ATTRIBUTION_CHECKS.json) records all 110 evidence-file hashes, per-patient reductions, exact source/patch checks, recipe comparisons, and CPU covector checks. [PATIENTS](GEOMETRY_ATTRIBUTION_PATIENTS.csv) contains all 80 paired patient records. [SOURCE_DIFF](GEOMETRY_ATTRIBUTION_SOURCE_DIFF.txt) preserves diffs against both source SHAs. [VAL_STRATA](GEOMETRY_ATTRIBUTION_VAL_STRATA.json) contains reductions of already exposed VAL counts. The evidence tree hashes were identical before and after the check. No training, model forward pass, GPU, download, production edit, commit, push, merge, or threshold change was performed.

**Paired bank attribution — VERIFIED reductions, REPORTED historical arrays.**

Class order throughout is BG, RV, MYO, LV. The old hard counts come from `B.patients.<id>.classes[*].hard_accepted`, independently checked equal to each `recorded_bank_report.class_pixels`. New counts come from `N.patient_reports.<id>.class_pixels`, also checked equal to `E/run_evidence/run/bank/REPORT.json.patient_reports`.

| TRAIN stratum | Patients | Old hard pixels, BG/RV/MYO/LV | New hard pixels, BG/RV/MYO/LV | Delta |
|---|---:|---|---|---|
| CODED_WORLD_LEFT | 52 | 428926 / 877 / 32101 / 3711 | 428926 / 877 / 32101 / 3711 | 0 / 0 / 0 / 0 |
| NO_CODED_WORLD_GEOMETRY | 28 | 366263 / 4322 / 29611 / 3066 | 366455 / 0 / 29664 / 3066 | +192 / −4322 / +53 / 0 |
| Total | 80 | 795189 / 5199 / 61712 / 6777 | 795381 / 877 / 61765 / 6777 | +192 / −4322 / +53 / 0 |

Old class-support patient counts are `[62,26,55,47]`; new counts are `[62,14,55,47]`. All 14 remaining RV-support patients belong to the coded cohort. Twelve of the 28 uncoded patients had old accepted RV pixels; refusal applies to all 28, not only those twelve. The corresponding RV hypothesis counts change from 1,607 to zero in the uncoded cohort and remain 2,528 in the coded cohort. All per-patient wall/cavity hypothesis counts and graph node/spatial-edge/temporal-edge counts agree between old and new records. These equal counts do not prove identical feature arrays, graph edge weights, or pixel identities.

The entire bank contains 46,575,360 grid pixels. Accepted counts fall from 868,877 to 864,800, a net loss of 4,077; `UNKNOWN=255` rises from 45,706,483 to 45,710,560. The net is not −4,322 because BG/MYO counts also change. The package does not contain paired hard-label arrays, so “4,322 removed RV pixels” means the old uncoded RV count becomes zero. It does not identify whether each old RV pixel became UNKNOWN, BG, MYO, or LV. Likewise, a net increase of 192 BG pixels does not identify their previous labels.

Sixteen uncoded patients have at least one changed class count; 12 lose accepted RV. For example, `patient042` changes `[33652,3258,4166,599]` to `[33652,0,4173,599]`, and `patient064` changes `[74338,618,10706,517]` to `[74342,0,10713,517]`. `patient006` had zero accepted RV in both runs yet gains seven BG pixels. Thus the intervention affects soft evidence and the subsequent solve/reliability/gates, not merely deletion from an already generated hard-label image. See `GEOMETRY_ATTRIBUTION_CHECKS.json.strata` and `.patient_records`.

**What headers do and do not establish.**

| Recorded property | Coded TRAIN, n=52 | Uncoded TRAIN, n=28 |
|---|---|---|
| sform/qform codes | 2 / 0 in every patient | 0 / 0 in every patient |
| Selected form | sform | fallback |
| Coded sform | Exact same `diag(-1,-1,1,1)` | null |
| Coded qform | null | null |
| sform/qform agreement field | null; no pair to compare | null; no pair to compare |
| Spatial unit | unknown | mm |
| Old cardinal field | `+x` | `+x` |
| New covector `(y,x)` | `(0,W_native/48)`, positive x | `(0,0)`, abstain |

These fields are in `N.image_records.<id>` and `.world_left_geometry`. Reexecuting the archived pure `world_left_geometry` helper on each recorded affine/shape reproduces all its recorded fields. The old forensic geometry object equals the new object for all 80 patients, including the image SHA256. The selected affine equals the recorded coded sform for all 52 coded cases. No TRAIN case uses a qform-only selection, an sform/qform disagreement, a nonzero y component of the new covector, or an old cardinal-orientation rejection.

NiBabel selects a nonzero-coded sform first, then a nonzero-coded qform, then its fallback affine. Code 2 denotes an aligned coordinate system; it is not scanner-acquisition provenance. NiBabel's fallback assumes LAS voxel orientation and a centered origin rather than measuring the patient's laterality. These implementation semantics explain why old `read_cine` can return `+x` even when both codes are zero. [NiBabel primary documentation](https://nipy.org/nibabel/nifti_images.html#choosing-the-image-affine).

`S/io.py:73-89` computes the old cardinal field from `nii.affine` without checking codes. `S/experimental_geometry.py:15-38` refuses codes `(0,0)` but accepts a finite, nonsingular coded affine with in-plane left information. `:41-63` checks image hashes and recorded header consistency, reads sform/qform, and records disagreement; it does not reject every possible disagreement or verify the acquisition's clinical orientation. Here qform is absent, so dual-form agreement supplies no validation. The matrix and unknown units in all coded records are compatible with a convention or transformed image representation; this audit cannot establish their upstream meaning. Unknown units do not alone make a direction mathematically undefined, but they prevent treating these records as independently validated physical geometry.

For uncoded cases, metadata-derived voxel scaling and `mm` units do not identify anatomical left/right. The new refusal is faithful to its declared trust policy. Whether refusal deletes mostly correct or incorrect RV hypotheses is **BLOCKED**, as is whether the retained coded hypotheses refer to true RV.

**Mathematical attribution, with assumptions and proofs.**

1. **The covector implements selected-affine left differences within one slice.** Let a bank centroid be `(y_b,x_b)`, native sizes `(H,W)`, bank sizes `(h,w)`, and `s_y=H/h`, `s_x=W/w`. The pixel-center resize map is `x_n=s_x x_b+(s_x−1)/2`, `y_n=s_y y_b+(s_y−1)/2`. Let `A` be the selected voxel-to-world affine and let world-left coordinate be `−X`. For two centroids in the same z plane, subtraction cancels translation, the z term, and both constant pixel-center offsets. Therefore

   `Δ(−X) = −A[0,1] s_y Δy_b − A[0,0] s_x Δx_b`.

   This equals `v·Δ(y_b,x_b)` for the exact covector returned at `S/experimental_geometry.py:32-38`. Affine maps commute with finite averaging, so using region centroids gives the same difference as averaging mapped pixel centers. The proof requires the selected affine to describe the intended world frame and both regions to share z; it does not prove that the file's frame is anatomically authentic. Four CPU examples including a 45-degree oblique affine verify the algebra using the actual helper. The 45-degree example is rejected by the old 0.9 cardinal projection rule and accepted by the new helper; that is synthetic software evidence only.

2. **The coded cohort cannot demonstrate oblique-direction recovery.** Every coded affine has `A[0,0]=−1` and `A[0,1]=0`, hence `v=(0,W/48)` with `W/48>0`. For any two region centroids, `v·c_RV < v·c_LV` if and only if `x_RV < x_LV`, because multiplying a real inequality by a positive scalar preserves its direction. The old `+x` rule uses exactly that comparison (`S/core.py:251-265`). Thus the two side predicates are identical in exact arithmetic for these affines, and the old field was never empty. If image/features/candidates are identical and no floating comparison boundary changes, all coded votes, strength, fixed-graph solves, reliability and hard labels follow identically through the unchanged operations. Archived counts support that conditional conclusion; absent arrays and hashes prevent upgrading it to verified bitwise bank identity. Floating arithmetic at centroid ties is another reason to request per-array comparisons rather than infer them from counts.

3. **Refusal eliminates uncoded RV anchors without deleting all soft RV target mass.** With `v=(0,0)`, `S/core.py:258-259` skips the RV proposal loop after recording wall/cavity votes. No other code adds RV votes. At a node let nonnegative class votes be `V_c` and their sum be `m`. For `m>0`, the anchor is

   `R0_c = (1−exp(−m)) V_c/m + exp(−m)/4`;

   for `m=0`, all entries are `1/4`. If `V_RV=0`, then `R0_RV≤1/4` and `R0_c≥R0_RV` for every class. Let `P` be the fixed nonnegative row-stochastic graph and `α=.35`. Initialization is `Q0=R0`; each step is `Q(k+1)=αR0+(1−α)P Qk` (`S/core.py:147-168`). A convex combination of numbers at most `1/4` is at most `1/4`, so induction gives `Qk_RV≤1/4` for every node and iteration. Applying the same recurrence to `Qk_c−Qk_RV` preserves nonnegativity, proving RV cannot exceed BG; exact ties select lower-index BG in this implementation. In particular RV cannot pass the fixed 0.70 probability gate. These statements hold in exact arithmetic; the 0.45 gap to the gate is far larger than the solver's recorded numerical tolerance. The observed zero accepted uncoded RV count is therefore structurally expected, not an accuracy result.

   Yet `R0_RV=exp(−m)/4` is generally positive where `m` is finite, and unknown/unvoted nodes start uniform. Propagation and weighted cross-entropy can retain RV probability mass. Training uses `Q` and reliability, not hard labels as cross-entropy targets (`S/student.py:38-59,95-108`). Consequently zero hard RV support in one stratum is not zero RV supervision. Finite-precision saturation may make individual entries zero, so strict positivity everywhere is not claimed.

4. **Why other class counts can change.** Removing RV votes changes both `m` and the normalized four-class vote vector above. It therefore changes non-RV anchor entries even when their raw votes are untouched. It also changes the propagated strength, entropy-dependent reliability, and threshold decisions (`S/core.py:277-303`). Conditional on the same `P`, the fixed-point difference obeys

   `ΔQ = α Σ(k=0..∞) (1−α)^k P^k ΔR0`.

   Proof: a row-stochastic nonnegative matrix satisfies `||P X||max≤||X||max`; therefore `||(1−α)P||≤1−α<1`. The geometric operator series converges, multiplying it by `I−(1−α)P` telescopes to the identity, and applying the inverse to the subtracted fixed-point equations gives the displayed expression. The weights sum to one, so `||ΔQ||max≤||ΔR0||max`. This is a conditional stability result, not a bound on anatomical error or on changes in hard decisions. Neither hard gates nor the reliability function are fixed linear functionals of `Q`, so the expression does not assert any monotonic BG/MYO/LV count change. The observed +192/+53 counts are compatible with this coupled transformation; exact pixelwise attribution requires the missing arrays.

**Available Q summaries and their limits.**

The baseline forensic report contains full-bank per-patient Q summaries, distinct from its one selected model batch per patient. Aggregating `.classes` gives:

| Baseline statistic | BG | RV | MYO | LV |
|---|---:|---:|---:|---:|
| Informative R0 argmax pixels | 6,882,154 | 95,391 | 904,973 | 86,706 |
| Q argmax pixels | 44,614,861 | 173,642 | 1,692,285 | 94,572 |
| Positive-weight pixels with that Q argmax | 8,454,581 | 133,999 | 1,285,209 | 90,158 |
| Of those, hard UNKNOWN | 7,659,392 | 128,800 | 1,223,497 | 83,381 |

The new package lacks the bank `.npz` files and an analogous per-patient Q audit. It does contain full-epoch `per_class_target_mass`, computed as `Σ_i w_i Q_i,c` before batch activation masking in `S/experimental_diagnostics.py:11-56`. Both runs permit all four classes because every class has at least two support patients; shrinking RV support from 26 to 14 does not disable the RV permission. Each complete epoch visits every `(t,z)` once (`S/student.py:79-100`). Thus summing batch masses is algebraically the full-bank mass: the batches partition the pixel indices, and finite-sum distributivity gives `Σ_b Σ_i∈b w_i Q_i,c = Σ_i w_i Q_i,c`. This includes inactive batches and is not the normalized class gradient magnitude.

| Weighted Q target mass | BG | RV | MYO | LV |
|---|---:|---:|---:|---:|
| Old, full bank reported | 338048.138 | 84381.566 | 132990.976 | 85815.277 |
| New, epoch 1 reported | 337656.680 | 81770.887 | 132625.535 | 85458.247 |
| Difference | −391.459 | −2610.678 | −365.441 | −357.030 |
| Relative difference | −0.116% | −3.094% | −0.275% | −0.416% |

These are **DERIVED cross-reductions of REPORTED measurements**; baseline NumPy patientwise sums and new GPU batchwise sums differ in reduction order/grouping. New RV mass ranges only from 81770.887108 to 81770.887557 across the 50 epochs, as expected for a fixed bank. Hard RV count falls 83.131%, while weighted RV target mass falls about 3.094%. Neither percentage estimates how much useful RV anatomy was removed. All four weighted class masses fall despite higher BG/MYO hard counts. A complete interpretation needs per-patient Q, reliability and batch-denominator information. New raw RV vote mass is zero in the uncoded group and 35392.795215 in the coded group; raw votes, Q mass, positive-weight winning pixels, and accepted labels are different quantities.

**Exact source interventions and run comparability.**

Against `7d1c6a0`, every archived scientific module except `full.py` is byte-identical. The only other changed files are `scripts/run_stable_guidance_wandb.py` and two tests. Strictly replaying every unified-diff hunk in memory, checking the old context, reproduces all four archived files exactly. All launch `scientific_source_sha256` values match the archived files. This verifies the supplied patch's content rather than relying on its description. `GEOMETRY_ATTRIBUTION_SOURCE_DIFF.txt` includes both comparison bases; CHECKS `.patch_replayed_exactly_against_7d1c6a0` and `.executed_source_matches_launch_hashes` contain the receipts.

Against baseline training `d5a45f3`, the graph builder, solver, clustering routine, model architecture, loss, contexts, and prediction function have identical source segments. Their individual hashes are recorded under CHECKS `.unchanged_scientific_symbols`, avoiding unrelated file-level diagnostic/metadata changes. Teacher, input IO, and preflight modules are byte-identical. Changes are: optional covector anchoring and geometry trust policy; extra anatomy diagnostics; training diagnostic calculations and recording; optional evaluation-only ACDC frame/cine identity recovery; launcher forwarding of these options and recording/plotting their metadata. `S/core.py:193-307`, `S/pipeline.py:24-73`, `S/student.py:66-134`, and `S/full.py:93-268` identify the executed paths.

| Comparison item | Evidence and conclusion |
|---|---|
| Split and input identity | `B.resolved_recipe.splits` equals new `run/RECIPE.json.splits`: 80 TRAIN, 18 VAL, 50 official unlabelled TEST identities. Recipe split digest is the same. All TRAIN raw-image hashes in old header diagnostics equal new image records. Arrays were not reopened. |
| Seed and initialization | Both recipes specify seed 42; unchanged `GuidanceStudent` constructor runs after `torch.manual_seed(seed)` and creates a fresh model, with no baseline checkpoint initialization (`S/student.py:71-78`). Initial weight tensor hashes/RNG states are absent. Same specified seed is verified; bitwise identical initialization and trajectories are not. |
| Budget and optimizer | Both specify 50 epochs, batch size 8, width 32, learning rate 0.0003, four threads, 2,553 steps/epoch, 127,650 total. Same AdamW with weight decay 0.0001 and norm clipping at 5.0. |
| Loss and permission | Same cosine feature loss plus reliability-weighted soft CE, same activation test, semantic-head gradient suppression on inactive batches, and all-class permission. Semantic updates change 110,604 → 110,461 (−143), consistent with changed bank eligibility. Baseline count is historically recorded/replayed; new is summed from logs. |
| Teacher | Same adapter bytes; same pinned teacher hash `a841b7e9300d60ea8feb73e3a7d8c60ddedc1d1d49ec1049ebfcfa337c5d5590`, model revision and config digest in recipes. Frozen MAE, no segmentation fine-tuning by this adapter. Full upstream data/preprocessing lineage remains unaudited (`S/teacher.py:41-71`). Fresh extraction output arrays are not available for comparison. |
| Precision | Unchanged `.float()` feature/logit/loss computations; teacher feature cache uses float16, restored to float32 in training; graph solve float64 then Q/reliability float32 (`S/teacher.py:109-132`, `S/pipeline.py:46-55`, `S/core.py:151-160,298-300`, `S/student.py:38-58,95-99`). No autocast/scaler is added by the patch. Runtime TF32/backend settings and paired deterministic-kernel receipts are not established. |
| Geometry treatment | Old cardinal naming → new `world_left_v2`. On this TRAIN cohort, this is principally removal of uncoded RV votes. Coded side decisions are mathematically equivalent under the conditions proved above. |
| Diagnostic treatment | Enabled in the new run. Diagnostic terms are converted to detached CPU scalars, never added to the training loss; no RNG draw is introduced (`S/experimental_diagnostics.py:11-56`, `S/student.py:101-118`). This changes logging/workload but not the written objective. Timing changes are not pure geometry performance estimates. |
| Config and gates | Decoded config dictionaries are equal, including probability 0.70, margin 0.20, reliability 0.05. Config file digest differs, so byte identity is not asserted. No threshold was fitted here. |
| Checkpoint selection | Same fixed last epoch, no reference feedback. Final checkpoint hashes differ. Actual checkpoint payloads are absent from this report package. |
| Evaluation | New full launcher forwards `allow_verified_acdc_frame_grid`; seven VAL patients require exact original-frame/cine voxel correspondence. Baseline re-audited count report already contains those seven, so the presented 18-patient count comparison is not a comparison against only 11 aligned patients. This recovery is evaluation-only and is not orientation recovery used to name TRAIN structures. |

Matching seeds alone do not guarantee reproducibility across releases, platforms, or nondeterministic GPU kernels; explicit backend controls and state hashes are needed to strengthen that assertion. [PyTorch primary reproducibility documentation](https://docs.pytorch.org/docs/2.14/notes/randomness.html). No such assurance is inferred from the single seed here.

**Downstream evidence: reference-informed development, not causal accuracy attribution.**

The already exposed VAL count records show no new RV predictions in either evaluation alignment stratum. Reducing those records independently gives:

| VAL evaluation stratum | Patients | Old → new FG Dice (%) | Old RV predicted pixels / TP | New RV predicted pixels / TP |
|---|---:|---:|---:|---:|
| Direct affine alignment | 11 | 6.1440 → 6.6160 | 12,831 / 70 | 0 / 0 |
| Verified frame/cine index recovery | 7 | 2.3907 → 2.6368 | 2,762 / 44 | 0 / 0 |

Old RV predictions occur in 12 and 10 frames respectively; new RV predictions occur in zero of all 36 frames. Reference RV pixel totals match old/new in each stratum (169,253 and 61,028). Source keys are the two `val_counts/DIAGNOSTICS.json.patient_frame_counts`; derived details are in VAL_STRATA. These strata describe evaluation alignment, not independently verified TRAIN header-code groups; the VAL export image metadata does not contain the TRAIN-style sform/qform-code audit.

The published overall foreground Dice changes from 4.6844% to 5.0685%, while coverage falls from 0.6148% to 0.3668%. New RV precision is undefined, and the foreground macro precision changes its defined class set. These observations do not establish that removal improved pseudo-label precision, caused every missing RV output, or recovered geometry. No per-patient training-treatment randomization, unchanged-bank repeat, initial-state proof, ungated paired logits, or matched controlled downstream comparison is available. Shared student parameters also let changes from uncoded TRAIN patients affect every VAL patient; stratifying VAL alone cannot isolate that path.

The scientific distinction is therefore: **VERIFIED** header-policy/count association; **DERIVED** deterministic side-predicate equivalence and refusal mechanism under explicit fixed-input assumptions; **REPORTED** downstream trained outputs; **BLOCKED** correctness of removed/retained RV labels and quantitative causal attribution of student accuracy. TEST remains unscored/unexamined; `UNKNOWN=255` remains abstention rather than background. VAL exposure is declared reference-informed development and supplies no basis to lower abstention or choose new thresholds.

**Minimal additional information and useful checks, proposed only.**

- Obtain existing per-array digests for both banks (`image`, frozen `features`, `r0`, `q`, `reliability`, `labels`) and, where possible, graph operator digests. On the coded 52, compare arrays directly; on the uncoded 28, record full label-transition tables plus max/quantile Q changes, weighted class mass, support/entropy, and active-batch changes. This would distinguish identical counts from identical arrays and identify the destinations of old RV labels. No new training is necessary for this evidence request.
- Recover raw header exports and conversion/acquisition lineage with independently supported laterality: original DICOM orientation/direction-cosine records, a documented conversion transform, or equivalent immutable metadata. Explicitly inspect code 2/unknown-unit/diagonal-affine cases, qform-only cases, form disagreement, and absence of both codes. Merely writing nonzero codes into fallback headers would not supply the missing orientation evidence. Do not use masks or reference-based class matching to repair naming.
- Separate the three relevant groups prospectively: coded cardinal, coded genuinely oblique (especially old max projection below 0.9), and uncoded. The current TRAIN cohort contains no example of the second. Count actual side-predicate disagreements, anatomy hypotheses, and bank changes for that group before any claim of real-data oblique recovery. Synthetic rotations validate coordinate software but do not validate the dataset's anatomical naming.
- Request initial state dict/RNG hashes, exact runtime/library and TF32/determinism settings, frozen feature identities, and full per-patient diagnostic outputs before attributing training differences exclusively to geometry. A future authorized controlled comparison would need an unchanged-bank repeat and a fixed-array intervention with all other execution inputs pinned. It is not executed or recommended as an immediate GPU escalation by this report.
- Existing exposed VAL counts support paired fixed-cohort checks by evaluation alignment, with coverage, true positives, false positives and false negatives retained for every class. Add header-code strata only after obtaining admissible VAL image-header records. If frozen ungated logits already exist, use predetermined gate accounting to separate no RV argmax from RV rejected by probability/margin; do not tune/export lower thresholds. Keep all such analyses declared development-only, preserve an untouched later evaluation, and never score or tune on official TEST here.

**Remaining boundary.** The audit completes bank-count and source attribution within the provided evidence package. Pixelwise bank identity, trustworthy anatomical orientation, correctness of removed pseudo-labels, and isolated downstream causal effects remain unresolved because the required evidence is absent; no evidence was manufactured to close those gaps.
