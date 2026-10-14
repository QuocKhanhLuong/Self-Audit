**Independent provenance and VAL metric audit — 14 October 2026.** The saved counts reproduce the published metrics, but the mean Dice improvement is concentrated: 4 of 18 patients improve, 14 decline, the median paired change is −0.126854 percentage points, and omitting patient098 changes the mean delta from +0.384111 to −0.312415 points. The 29.084% versus 19.623% headline precision comparison changes both class and patient/frame support. A matched 15-patient, two-class, two-frame descriptive precision comparison is feasible and rises from 26.468211% to 31.290271%, conditional on both models making MYO and LV predictions in both frames; it does not rescue whole-cohort coverage, establish RV performance, or support a causal improvement claim.

The audit ran at repository SHA `b937e3b746608725e43d3bc3cef8861666d152dc` on branch `QuocKhanhLuong/guidance-results-research-20261014`; the initial tree was clean. All evidence paths below are relative to `reports/stable_guidance_world_v2_20261014/`. Executed scientific source references use `executed_source/`, never historical top-level `src`. The durable executable check is [PROVENANCE_METRICS_recompute.py](PROVENANCE_METRICS_recompute.py), and the machine-readable result is [PROVENANCE_METRICS_checks.json](PROVENANCE_METRICS_checks.json). Its `status=PASS` means **406 scoped text, hash, identity and arithmetic checks passed**, not model quality or execution reproduction. It uses only Python's standard library, published text files and local Git objects; no masks, images, weight files, prediction arrays, training, GPU work or TEST scoring were used.

Here **VERIFIED** means directly inspected bytes, source, or a check executed in this audit; **DERIVED** means arithmetic or a proof conditional on the saved counts/receipts; **REPORTED** means an earlier runtime claim preserved in the package; **BLOCKED** means the package lacks the payload or evidence needed for independent verification. Exposed VAL is reference-informed development. The new results are not an untouched validation confirmation, and no TEST outcome is inferred.

**Publication integrity is internally consistent.** VERIFIED: all 102 `MANIFEST.json.files` entries have the declared published SHA-256 and byte size, totaling 6,691,118 copied bytes; all are byte-identical to their versions at the pinned publication commit. There are 44 entries marked sanitized and 58 unchanged; every flag agrees with whether original and published hashes differ. The 8 package-level files outside that inventory are `.gitattributes`, `MANIFEST.json`, `PUBLICATION.json`, `README.txt`, `VALIDATION.json`, `comparison.csv`, `metrics.csv`, and `summary.json`; these also match pinned Git bytes. Thus the copied-file manifest is not a self-authenticating inventory of every package file, but the inspected Git snapshot also pins those eight files. Exact results are in `publication.files`, `publication.unlisted_files_pinned_by_git`, and `checks`.

VERIFIED: PUBLICATION declares five path placeholders, three redacted metadata categories, three nonduplicated W&B symlink aliases, one dereferenced core debug log, and 219 excluded binary artifacts. A bounded scan of all published text found no unredacted `/home/<name>` or `/Users/<name>` paths, full GPU UUID patterns, or long literal credential assignments matching the documented patterns. The sole email-pattern hit is the added `@pytest.mark.parametrize` decorator at `launcher_integration.patch:112`, not an email. This bounded scan is not a universal secret detector or a proof that every possible host identifier was removed. REPORTED: `PUBLICATION.original_run_artifacts_modified=false`, preservation of all numeric measurements, and the sanitization's exact relationship to the unavailable originals. The original versions of the 44 modified files are absent, so a transformation-level or numeric before/after audit of those files is BLOCKED. Existing numeric copies and cross-receipts can be checked for consistency, as done here.

SHA-256 is used to detect changes to bytes, as specified by [NIST FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final). The logical limitation here is stronger than a formatting caveat: matching `hash(published_bytes)` proves consistency with a declared published digest; comparing an embedded identifier with `MANIFEST.original_sha256` proves agreement between two declarations. Neither operation hashes an absent original file. A manifest could consistently describe a false execution history. No trusted signature, independent runtime attestation, or complete original artifact replay is supplied by these checks.

**Source identity was reconstructed, not assumed.** VERIFIED: the 23 preserved source/test files are byte-for-byte the result of base commit `7d1c6a04064cda473d1eccfbed818392c37b1a37` plus `run_evidence/launcher_integration.patch`. The checker applies every patch hunk in memory, verifies all deleted/context lines and hunk lengths against Git objects, then compares the resulting bytes with the published source. The four patch targets are `src/self_audit_guidance/full.py`, `scripts/run_stable_guidance_wandb.py`, `tests/test_self_audit_guidance_full.py`, and `tests/test_guidance_wandb_runner.py`; test copies use `.py.txt`. The patch SHA-256 is `ecbcd639fbaa010ac3f7fca9ac5ba3272278806ce92eecc8171c4833f70815ee`, matching `LAUNCH_MANIFEST.local_patch_sha256` and `summary.local_launcher_patch_sha256`. All 11 launched scientific source hashes match the snapshot, and recipe/bank/student source maps agree. See `source_identity.executed_source`, `source_identity.patch_targets`, and the `launch source`/`recipe source map verified` checks.

VERIFIED: all 27 baseline source-receipt entries—9 modules each in `baseline_forensics/train_all/DIAGNOSTICS.json.source_receipts.{bank,recipe,student}`—match SHA-256 hashes computed directly from Git commit `d5a45f31481f0a702a1f9d5d418889fe704e9105`. This verifies that the recorded baseline source identity is d5a45f3. `baseline_forensics/RESULTS.json.source_commit=7d1c6a0...` and `EXECUTION.json.source_commit` identify the later forensic tools, not baseline training. The distinction is essential. The original base source archive `680481d83987f2bf5a8c378b63c49d89145df55a6236d59afca10959ac6e0d4b` is absent: the compact 23-file snapshot does not verify the full archive, the installed CineMA source tree, dependency installations, or all runtime code.

**Configuration, cohorts and ordering agree across receipts.** VERIFIED: the new recipe, launcher CONFIG, bank config, embedded bank configuration, baseline resolved configuration, and baseline resolved recipe contain the same 22 configuration values. In particular, `min_probability=0.7`, `min_margin=0.2`, `min_reliability=0.05`, `grid=48`, and `alpha=0.35`. The baseline and new config byte hashes differ (`9dffc821...` versus `e9cf877b...`), but the preserved resolved values agree; a byte-hash change is not evidence of a numerical hyperparameter change. The new recipe explicitly records `orientation_policy=world_left_v2`, `diagnostics=true`, `allow_verified_acdc_frame_grid=true`, seed 42, width 32, batch size 8, learning rate 0.0003, and 50 epochs. Baseline resolved recipe uses the default cardinal policy and no diagnostic option. These are one-seed runs with a changed geometry policy and instrumentation, not repeated randomized evidence of an isolated causal effect.

VERIFIED: `SPLIT.json`, `EFFECTIVE_SPLIT.json`, recipe splits, preflight splits, bank locked splits, both prediction manifests and the baseline resolved recipe agree on 80 TRAIN, 18 VAL, and 50 official TEST identities; all 148 included IDs are distinct. All 148 preflight image hashes are also distinct. The 80 baseline TRAIN image hashes in `train_all/DIAGNOSTICS.json.patients.<id>.geometry.image_sha256` match the new bank. New bank and export image hashes match the preflight inventory. The archived source defines patient discovery from full-cine filenames, without Info.cfg or masks (`executed_source/src/self_audit_guidance/io.py:48-66`, `full.py:42-53`).

REPORTED with inspectable metadata: two original VAL cases were removed for byte-identical cine duplicates: patient076 duplicated TRAIN patient074 (`69487c77...`), and patient055 duplicated TEST patient118 (`ae245957...`), under `SPLIT.provenance.exclusions`. Both runs' preserved resolved splits already use the same 18-patient cohort; the new-versus-baseline improvement is not a 20-versus-18 arithmetic comparison. Because excluded raw images are absent, their duplicate relation cannot be independently rehashed here. Split identity alone also cannot certify absence of all latent subject duplication.

VERIFIED source contract: permissions come from hard-accepted TRAIN class support in at least two patients (`student.py:79-82`); native export uses checkpoint-stored probability/margin thresholds and permissions (`student.py:148-158`); final checkpoint is written after the fixed epoch loop (`student.py:120-133`). The archived full launcher freezes VAL and TEST predictions and writes the lock before invoking an evaluator (`full.py:205-239`), and official TEST bypasses evaluation (`full.py:228-231`). References are not passed to prepare/train/export commands. Current review does not modify or select naming, permissions, thresholds, or checkpoints using target masks. REPORTED runtime facts remain `gt_read=false`, `gt_used_for_training=false` and fixed-last-epoch selection; the source and logs are consistent with those facts, but they do not independently audit every historical process or upstream teacher's pretraining lineage.

VERIFIED receipt consistency: the launcher has exactly `preflight → prepare → train → export_val → export_test → evaluate_val`; all 50 `[STUDENT]` rows equal `student/training.json`; their steps sum to 127,650 and diagnostic active batches to 110,461. The baseline forensic receipt records 110,604 semantic updates. Archived JUnit files contain 122 and 120 successful cases respectively; those suites were not rerun. Preflight reports `PASS_EXECUTION_ONLY_QUALITY_UNVERIFIED`, inventory for 148 studies, a real-teacher check on patient057, and two student backward/reload steps with `semantic_quality_tested=false`. These are historical execution checks, not segmentation validation. The launcher does not forward `world_left_v2` to preflight (`full.py:170-174`); the policy is forwarded to actual bank preparation at `full.py:183-186`. Therefore the preflight's tiny bank solve cannot be cited as a validation of the full world-left geometry behavior.

**The checkpoint and lock form a consistent chain of original identifiers.** VERIFIED: all 21 checked links in `original_hash_bindings` equal the relevant `MANIFEST.files[].original_sha256`. Twelve do not equal the corresponding sanitized published-file digest; that is expected, documented transformation, not an unexplained hash failure. Selected original links are:

| Object | Recorded original SHA-256 | Binding evidence |
|---|---|---|
| source SPLIT | `4676e847bca5e4066594b217d62600a6162b34747e3c225da5df157171954b2b` | recipe and effective split provenance |
| effective split | `1631da7be8f7c959e3674ac1607c06e6b832c911141c6c202f8b129aec45481f` | preflight and bank source split |
| bank manifest | `04fb2d30df6c0e0ed235dcf851a491895661b7fd4763e6b89ce95ca6811accae` | student manifest |
| student manifest | `85340cf5f4502d974638edd08eaca030d4642c530b5bf1680d2c21b18e09971e` | prediction lock and both prediction manifests |
| selected `last.pt` | `90bc16bb9f89963b452355376673e3ba9aa24b38870e4595237d75e402ce873e` | student file manifest, both export manifests, summary |
| VAL prediction manifest | `9f5a38185ba159db70b5c17b949739af87c00a8343d823bb66865e4f27ca6e40` | prediction lock, evaluator manifest, METRICS |
| TEST prediction manifest | `b2c6e0b9e1fa35daae83f6206b1f48393abe6acc2b7b194b324c387e471d1f1d` | prediction lock only; no TEST scoring |
| recipe | `bed149d73bb0527c1dbcea8ff72e9b28e093149974dcb0c5893fd43d53a6092b` | prediction lock |
| VAL METRICS | `55f4c6fcb2716947ccbe2f0ee694c00f9f0aa98552155cd92aec3bad83f7fd9d` | evaluator file manifest and new count diagnostics |

The baseline recorded checkpoint is `516e569c9141e2d8edb00b598df5cf12c8a9a4741cf1ae0cca8cbf23d6d03353`, consistent between train-all forensics, its inventory and RESULTS. BLOCKED: neither checkpoint's weights are included; baseline original evaluation METRICS (`dc6e1d82dd34deec3d7a181b97ce9600de37af6bb2f05351cb7c5b7c9b68da6e`) is also absent. Baseline arithmetic therefore starts at the saved published diagnostic confusion counts, not an independently rehashed original evaluation file. New original METRICS is available only as a sanitized copy whose confusion counts agree exactly with the post-training diagnostics.

The three bank/VAL-prediction/TEST-prediction FROZEN copies have changed bytes, while student and evaluation FROZEN copies are unchanged. None is a runnable reproduction of the complete original artifact chain: weights and prediction arrays are missing, and the evaluation file's embedded original digest differs from its sanitized bytes. `io.verify` requires exact file inventory and exact payload digests (`io.py:101-110`), which these report-only folders cannot satisfy. Replacing placeholders or rewriting embedded hashes would create a new artifact identity, not restore the original evidence. The preserved copies prove the published declarations and their internal linkage; they do not prove omitted payloads, exact geometry recovery, accuracy of pseudo-labels, original artifacts remaining untouched, or that the experiment actually executed as described.

**The numerical estimand is preserved exactly.** DERIVED and VERIFIED numerically: both diagnostics contain 144 unique patient/frame/class count rows, representing 18 patients × 2 labeled frames × 4 classes. Corresponding patient/frame/class identities, reference volumes `tp+fn`, and aligned/recovery strata agree in both runs. All rows have nonnegative integer counts. These equal marginals do not prove that baseline and new used identical reference-mask bytes; the baseline original reference hashes are not available in the diagnostic table. New evaluator frame counts independently agree with new diagnostic counts.

For each class/frame, the checker computes Dice `2TP/(2TP+FP+FN)`, IoU `TP/(TP+FP+FN)`, precision `TP/(TP+FP)`, and recall `TP/(TP+FN)` from counts. A zero denominator returns JSON `null`, never zero or one. It averages defined frame values within each patient/class, then defined patient values within each class, then defined foreground classes. The archived count reducer states this at `executed_source/scripts/diagnose_stable_guidance.py:30-65`; whole-volume UNKNOWN behavior and class validation are at `evaluate.py:18-40,59-63`. Precision's denominator and undefined-zero-prediction case also follow the [primary scikit-learn metric documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.precision_score.html). No scikit-learn implementation was imported for this calculation.

Proof of recovered coverage: the reference is a partition into BG/RV/MYO/LV by the evaluator's validated class contract, so `Σc(TPc+FNc)` is the full labeled frame volume. Valid predictions are mutually exclusive class assignments; hence `Σc(TPc+FPc)` counts each accepted pixel once. Their ratio is frame coverage, and the remainder is UNKNOWN, not BG. Averaging those frame ratios within patient then across patients reproduces the recorded coverage to 1e−12. This does not pool pixels across patients. For all 36 frames and all four classes, Dice/IoU/recall denominators are defined. Thus, for any of these metrics with patient/class mean `m_pc`, the published foreground macro is `(1/3)Σc[(1/18)Σp m_pc] = (1/18)Σp[(1/3)Σc m_pc]` by distributivity of a finite sum. Consequently the mean of the paired patient foreground deltas equals the difference of the headline means. Precision cannot use that equality with run-varying class/patient/frame supports.

| Metric, percent | Baseline | New | Delta, percentage points |
|---|---:|---:|---:|
| foreground Dice | 4.684410 | 5.068521 | +0.384111 |
| foreground IoU | 2.634978 | 2.901769 | +0.266791 |
| foreground recall | 3.273159 | 3.322679 | +0.049520 |
| coverage, patient macro | 0.614758 | 0.366846 | −0.247912 |
| UNKNOWN, patient macro | 99.385242 | 99.633154 | +0.247912 |
| precision, changing support | 19.623489 | 29.084105 | +9.460615, not like-for-like |

Values and full patient/class/frame reductions are in `recomputed.{baseline,new}.overall`. Counts across the 36 frames sum to 17,734,144 reference pixels in each run. There are 106,129 versus 64,519 accepted pixels and 17,628,015 versus 17,669,625 UNKNOWN pixels. These totals would give pooled coverage 0.598444% versus 0.363812%; they are presented only to check count arithmetic and must not replace patient-macro coverage or create a pixel-IID inferential sample.

| Foreground class, percent | Baseline Dice → new | Baseline recall → new | Baseline precision → new |
|---|---:|---:|---:|
| RV | 0.117627 → 0 | 0.065435 → 0 | 3.392469 → undefined |
| MYO | 6.117321 → 6.829655 | 4.987301 → 4.791536 | 11.636822 → 17.060718 |
| LV | 7.818283 → 8.375908 | 4.766740 → 5.176502 | 43.841176 → 41.107491 |

Baseline RV has 114 TP and 15,479 FP; new RV has zero TP and FP in all 36 frames, with all 230,281 reference RV pixels counted as FN. Its Dice and recall therefore equal zero, while its precision is undefined. MYO has 11,683 → 11,666 TP and 59,382 → 28,263 FP; LV has 8,896 → 10,329 TP and 9,144 → 10,628 FP. These totals are secondary descriptions, not the macro precision denominators. BG precision is 100% on the few predicted BG pixels in each run, while its new patient-macro recall is only 0.022042%; this is another reason precision alone is insufficient.

| Precision support | Baseline defined patients / frames | New defined patients / frames |
|---|---:|---:|
| BG | 6 / 11 | 11 / 21 |
| RV | 11 / 22 | 0 / 0 |
| MYO | 18 / 36 | 17 / 34 |
| LV | 18 / 33 | 16 / 32 |

**The precision increase has a large class-omission component.** With `B_R=0.03392469086122546`, `B_M=0.11636822467468069`, `B_L=0.4384117593532065`, the baseline macro is `(B_R+B_M+B_L)/3=0.1962348916297042`. New precision is `(0.1706071778183496+0.41107491372902705)/2=0.2908410457736883`; RV contributes no defined value. Simply removing baseline RV raises the baseline summary to `(B_M+B_L)/2=0.2773899920139436` (27.738999%). Therefore the exact additive identity is:

`9.460615 pp = 8.115510 pp (omit baseline RV from the macro) + 1.345105 pp (change MYO/LV means on each run's own support)`.

Proof: add and subtract `(B_M+B_L)/2` in `P_new−P_baseline`, then multiply by 100. The second term still changes patient/frame supports, so it is not a fully matched comparison. Setting new RV precision to zero would invent a different metric and was not done.

Two conditional common-support calculations are justified by the available counts; neither needs masks or changed thresholds:

| Descriptive precision comparison | Support | Baseline % | New % | Delta pp |
|---|---|---:|---:|---:|
| MYO/LV only, original supports | run-specific defined patients/frames | 27.738999 | 29.084105 | +1.345105 |
| same defined frame/patient support within each class | MYO: 17 patients, 34 frames; LV: 16 patients, 31 frames | 24.571332 | 29.084105 | +4.512772 |
| same patients, classes and both frames | 15 patients, both MYO/LV predicted in both frames by both runs | 26.468211 | 31.290271 | +4.822060 |

Construction/proof: define `S_c={(p,f): TP_b+FP_b>0 and TP_n+FP_n>0}` for MYO/LV. Recompute each frame's precision on its own prediction set, average only identical `S_c` frames within each patient, then average the resulting patients for class c and the two classes. Both runs have exactly the same finite index sets at every averaging step. The second comparison therefore changes precision values, not averaging identities, although pixel sets can differ. The third further requires every MYO/LV count denominator to be positive in both annotated frames for both runs for the same patient. It excludes patient015, patient065 and patient070, leaving 15 patients and 60 class/frame observations per run; equal frame/class weights imply its macro delta equals the average of the 15 paired patient precision deltas by finite-sum distributivity. Those deltas have median +6.826645 pp, 9 positive and 6 negative. Exact eligible identities and values are in `precision_comparisons`.

Selection into these intersections depends on the two fitted models' predictions. They describe the retained prediction-support subgroup, not an intention-to-evaluate cohort. They cannot recover RV precision, compare the same accepted pixels, establish a threshold-independent ranking gain, or claim a population or causal gain. A new precision mean can equal a restricted mean accidentally—for example through zero-valued frames—so equality of summary values is not proof of identical supports.

**Geometry strata remain separate.** The 11 aligned patients and 7 patients with saved verified-frame recovery receipts partition the same 18 patients in both runs; each contributes exactly two frames. Recovery is an evaluation grid-correspondence category, not the TRAIN bank's coded/uncoded world-geometry grouping. The source checks cine/frame voxel identity and frame/reference shape/affine agreement without resampling (`evaluate.py:133-164,193-209`); this audit did not rerun those checks on arrays. Accordingly the stratum label is REPORTED, while the arithmetic on each recorded stratum is VERIFIED/DERIVED.

| Stratum | Metric, percent | Baseline | New | Delta pp |
|---|---|---:|---:|---:|
| aligned, n=11 | Dice | 6.144014 | 6.615970 | +0.471956 |
| aligned | IoU | 3.498177 | 3.836482 | +0.338305 |
| aligned | recall | 4.402630 | 4.277291 | −0.125339 |
| aligned | coverage | 0.772637 | 0.379018 | −0.393619 |
| recovery, n=7 | Dice | 2.390747 | 2.636816 | +0.246068 |
| recovery | IoU | 1.278521 | 1.432935 | +0.154413 |
| recovery | recall | 1.498276 | 1.822575 | +0.324299 |
| recovery | coverage | 0.366663 | 0.347718 | −0.018945 |

The raw stratum macro precisions are 19.226324% → 33.501826% aligned and 19.888651% → 21.921122% recovery, with the same undefined-RV warning and changing supports; they are not matched precision effects. For Dice/IoU/recall/coverage the overall delta equals `(11/18)Δ_aligned+(7/18)Δ_recovery`, because patients partition and the relevant means have full support. That equality follows by splitting the overall finite patient sum into the two disjoint subsets. The geometry-recovery receipt does not supply missing physical units; the new package reports 42 missing-prediction, 26 measured, and 40 unavailable-physical-geometry foreground frame/class HD95 statuses. No cohort HD95 is derived here.

**Patient tails reverse the impression of uniform improvement.** All changes below are percentage points; patient Dice averages RV/MYO/LV and then frames under the exact preserved definition. Full Dice/IoU/recall/coverage pairing is in `paired_patient_metrics`, with interpolated descriptive quantiles in `paired_delta_summaries` (quantile position `(n−1)p`, linear interpolation).

| Patient | Stratum | Baseline Dice % | New Dice % | Delta pp |
|---|---|---:|---:|---:|
| patient004 | recovery | 0.290354 | 0.201591 | −0.088764 |
| patient005 | recovery | 1.514944 | 1.029914 | −0.485030 |
| patient012 | aligned | 8.451433 | 11.013980 | +2.562547 |
| patient014 | aligned | 0.252095 | 0.050533 | −0.201562 |
| patient015 | aligned | 0.147110 | 0.005418 | −0.141691 |
| patient018 | aligned | 11.311368 | 10.222216 | −1.089152 |
| patient028 | recovery | 5.917229 | 9.381243 | +3.464014 |
| patient029 | recovery | 1.516040 | 0.942420 | −0.573620 |
| patient030 | recovery | 7.301355 | 6.453122 | −0.848233 |
| patient032 | recovery | 0.043213 | 0.409342 | +0.366129 |
| patient036 | aligned | 21.086576 | 20.575268 | −0.511308 |
| patient065 | recovery | 0.152097 | 0.040080 | −0.112017 |
| patient070 | aligned | 0.036321 | 0 | −0.036321 |
| patient082 | aligned | 0.389230 | 0.070917 | −0.318313 |
| patient087 | aligned | 0.104148 | 0 | −0.104148 |
| patient089 | aligned | 12.367806 | 5.277888 | −7.089918 |
| patient095 | aligned | 1.268837 | 1.165169 | −0.103668 |
| patient098 | aligned | 12.169232 | 24.394280 | +12.225048 |

DERIVED: Dice improves in only 2/11 aligned and 2/7 recovery patients. Overall mean delta is +0.384111 pp, median −0.126854 pp, interquartile interval [−0.504738, −0.049431] pp, 10th/90th percentiles [−0.920509, +2.832987] pp, minimum −7.089918 and maximum +12.225048 pp. IoU and recall likewise improve in 4/18 patients; recall's median delta is −0.095001 pp. Coverage falls in 15/18 patients, with median delta −0.097335 pp. Median patient Dice falls from 1.391890% to 0.986167%; the number below 1% Dice rises from 8 to 9, and zero-foreground-Dice patients rise from 0 to 2 (patient070 and patient087).

The largest positive delta, patient098, contributes `12.225048/18=0.679169` pp to the full mean, 176.816% of its net positive value. This contribution can exceed 100% because other patients have negative deltas. For any paired delta vector with sum T and n=18, removing patient i gives `(T−Δ_i)/(n−1)` by the definition of an arithmetic mean. Applying that exact identity gives −0.312415 pp without patient098; removing the two largest positive deltas (patient098, patient028) gives −0.548442 pp; removing the three largest (also patient012) gives −0.755841 pp. Across all leave-one-patient-out deletions, the mean ranges from −0.312415 to +0.823760 pp, the upper value excluding patient089. These are influence diagnostics, not valid reasons to delete inconvenient patients from the reported endpoint, and no trimmed result replaces the full-cohort result.

No standard error, confidence interval or significance test uses pixels as independent units. No inferential test is claimed at all: there are 18 paired patients from an exposed development cohort and one training seed per policy. Saved aggregate confusion counts cannot identify spatially matched prediction errors, ungated score rankings, anatomical error locations, or what the model would do at a different threshold. Conditional common-support precision gains coexist with very low Dice, lower coverage and complete RV abstention on labeled VAL frames. The supported conclusion is a reproducible, heterogeneous descriptive change with consistent published provenance; causal geometry attribution, absent-payload integrity, rerun geometry proofs and independent generalization remain BLOCKED by the available evidence and scope.

Reproduction command, from this workspace: `rtk proxy python reports/guidance_research_continuation_20261014/PROVENANCE_METRICS_recompute.py`. It writes only the same-prefix JSON file beside this report. Its explicit check list, own script SHA-256, evidence SHA, original/published hash pairs, full count-based reductions, common-support membership, patient deltas and leave-one-out values make the result reviewable without raw arrays. The primary repository evidence is the [pinned publication tree](https://github.com/QuocKhanhLuong/Self-Audit/tree/b937e3b746608725e43d3bc3cef8861666d152dc/reports/stable_guidance_world_v2_20261014); line citations above resolve inside its `executed_source` snapshot, and named JSON keys resolve inside that same pinned tree.
