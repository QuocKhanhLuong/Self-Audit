# Worker F — independent adversarial review

Status: complete and independently challenged by A; F cross-review of C is complete in `reviews/F_reviews_C.md`. Scope is current semi-supervised v2, not the archived no-GT method. No production/config/baseline edits, patient jobs, GPU work, downloads, commits or pushes were performed.

## Evidence identity

- Main source: `637f38ced217fe1db2347a87924695533056d8f0` (native, single-slice default).
- Experiment source/results: `8960b6baac4c6600553d900c303f192c2fcfa4f0` (C256/B52 experiment branch).
- I independently recomputed the experiment `source_digest()` from immutable Git blobs, obtaining `6e12885e44b725c132a01437dc6179cbf524b94c51dd5a5f5254701eb78cf442`, exactly matching all four retained measured receipts. This verifies the correspondence between the committed source and receipt strings; it does not rerun or authenticate missing payloads.
- Executed CPU probe: `probes/F_adversarial_probe.py`; full output: `evidence/F_probe_results.json`. Run command: `rtk proxy python reports/research/20261023_astra_ultra/probes/F_adversarial_probe.py`. Torch 2.14.0, NumPy 2.4.6, SciPy 1.17.1. No training or patient data is used.
- Source reads use exact `git show <SHA>:<path>` after Semble discovery, not worker summaries. General historical memory only informed the report-only scope; every scientific/source fact below was freshly inspected.

## High-confidence findings

### F1. A successful real guard can reduce every audited Brier risk while destroying audited Dice

**VERIFIED, manufactured CPU counterexample.** Main `self_audit/control.py:83-117` defines foreground one-versus-rest Brier, and `:205-220` checks only its actual finite trial values. With two RV-positive and two BG pixels, an affine 1x1 head and a constructed finite displacement, the actual `joint_guard` implementation chooses alpha=1, scale=1 and accepts. Risks change RV `0.1517625 -> 0.08865`, MYO `0.00125 -> 0.0008`, LV `0.00125 -> 0.0008`; RV argmax Dice changes `2/3 -> 0` on the same audit pixels. All classes meet zero-budget guards.

The candidate is deliberately constructed, not asserted to be produced by the EMA candidate generator or to resemble a patient run. It is nevertheless an exact counterexample to any implication from accepted actual-output Brier checks to nondecreasing Dice. This is consistent with the narrow wording in `docs/METHOD.md`, not an implementation violation. A proposed stronger Dice-safety claim fails before any expensive experiment. The negative result cannot be repaired by reporting local-linearized feasibility or more independent numerical solver checks.

**Falsification test:** run the saved probe and inspect both actual risk values and argmax Dice. A different guard objective would be a separately declared method.

### F2. The nominal round-robin coverage permanently couples every U patient to one labeled patient

**VERIFIED source + exhaustive schedule enumeration; performance harm NOT ESTABLISHED.** Main `self_audit/engine.py:142-149` uses `li=step % len(L)`, `ui=step % len(U)`, and chooses audit `ai` randomly among L excluding `li`. Experiment `engine.py:150-157` retains the same arithmetic. Experiment `self_audit/data.py:124-151` calls the sampler with the same patient index for every member of the B52 batch, so 52 crops do not break this coupling.

For the TRAIN80 budgets L=4,8,16, U=76,72,64 respectively, every U count is divisible by L. Consequently every U patient is always paired with exactly one supervised L patient over all 10,000 steps; that same L patient is never eligible to audit that U patient's proposals. Changing the training/audit RNG seed changes audit/crop draws but not this arithmetic relationship. A label-subset seed can change identities and the specific pairing, while the one-partner property persists at these fixed budgets. The saved probe enumerates each case. This is a representativeness/coverage design limitation, not label leakage, unequal schedules across current arms, or proof of lower accuracy. All patients being visited does not demonstrate all cross-patient audit pairings.

**Falsification test:** report the U-by-L supervised and audit co-occurrence matrices. Compare a predeclared, independently permuted cycle schedule with the current one, matched across all methods and holding crop/augmentation RNG recipes fixed. Worker E recommends prospectively independent balanced L/U permutations, a separate audit RNG excluding current L, separate crop/augmentation RNG streams, identical schedule hashes across arms, and saved incidence tables. This is a new primary-protocol factor requiring fresh matched utility references; the current CLI does not implement it. Source/synthetic incidence checks suffice to demonstrate the restriction. A full old-versus-new schedule efficacy ablation is only needed to claim a causal performance effect of the schedule change.

### F3. The committed headline metric is mean of available reference-present frame Dice, then equal-patient mean

**VERIFIED definition; effect on the retained results is not fully reconstructable.** Main `self_audit/evaluation.py:18-25` sets class Dice to null whenever GT count is zero, even if prediction FP count is positive. `:80-90` excludes null frame Dice from the within-patient average, then excludes patients with no available Dice from the patient macro. Experiment evaluation adds IoU but preserves these rules. Each frame confusion pools its full Z,H,W volume (`evaluation.py:72-74`), so the per-frame Dice is volumetric. The headline then averages those ratios; it is not Dice from jointly pooling ED+ES frames, and it is not an all-frame false-positive score.

The saved CPU probe has one perfectly predicted RV-positive frame and one reference-absent frame with only false-positive RV predictions. The reported mean of available frame Dice is 1.0, whereas pooled Dice is 2/3. FP counts remain visible in confusion statistics, so this is a declared denominator limitation rather than hidden deletion of evidence. The retained receipts show all 18 patients present for each foreground class; that does not establish that every reference frame was present. Committed patient-level confusion matrices pool frames, so they cannot establish how often frame exclusions occurred. Raw `METRICS.json` frame rows/probabilities are absent from Git.

A secondary reaggregation from committed patient TP/FP/FN gives mean pooled-per-patient foreground Dice of supervised 76.5916%, supervised_extra 77.7334%, ema_full 83.0635%, ema_block 74.7268% (`evidence/F_pooled_patient_sensitivity.json`). The rank is unchanged. This is a sensitivity endpoint, not a replacement for the reported primary endpoint, and cannot reconstruct missing frame-presence exclusions. When GT=0 and prediction count>0, the ordinary Dice ratio equals zero; marking it null is this implementation's endpoint policy rather than mathematical undefinedness.

**Falsification test:** report per-class total frames, reference-present frames, absent frames, and FP volume on absent frames; reproduce both the current endpoint and a predeclared pooled-per-patient endpoint. Do not retroactively replace the endpoint because it favors one method.

### F4. Current patient results cannot demonstrate joint allocation or guard efficacy; the block comparator is negative

**VERIFIED committed-table arithmetic; original patient execution REPORTED.** Experiment `results/REPORT.md:18-24` and `results/logs/EXECUTION_POLICY.json:2-7` explicitly state that all five allocation/guard arms were cancelled. There is no full joint-guard result to support an efficacy claim. I reaggregated every committed patient-class row in `results/PATIENT_METRICS.csv` and exactly recovered `results/RESULTS_MAIN.csv:2-5`.

| Retained arm | RV | MYO | LV | Derived mean foreground Dice |
|---|---:|---:|---:|---:|
| supervised | 60.7510% | 77.1315% | 88.5355% | 75.4727% |
| supervised_extra | 64.1921% | 77.1738% | 87.4644% | 76.2768% |
| ema_full | 73.6249% | 80.6737% | 91.4156% | 81.9047% |
| ema_block | 52.2591% | 80.2877% | 87.8306% | 73.4592% |

The derived foreground mean is an explicitly new arithmetic summary, not the stored primary endpoint. EMA-block versus supervised is RV -8.4918 points, MYO +3.1562, LV -0.7050; the mixed class outcome and lower mean must remain visible. One subset seed, one initialization seed and development-exposed VAL18 do not support a general ranking. No raw probability/checkpoint recomputation was possible from Git alone.

**Falsification test:** before attributing benefit to allocation, produce matched retained trajectories for `ema_block`, `independent_guard`, `joint_guard` and a label-use control from common initialization, with actual accepted displacement norms and per-class outcomes. Full clinical or generalization claims additionally require independent confirmation.

### F5. EMA-full versus EMA-block changes optimizer geometry and state, not just which parameters receive U gradients

**VERIFIED source.** Experiment `self_audit/engine.py:133-135,176-190` uses body AdamW at 3e-5 and head SGD at 0.003 for EMA-full, including an extra optimizer step and gradient clipping. Decoder EMA-block instead applies the direct unpreconditioned decoder/head displacement with `pseudo_lr=0.001`; `self_audit/decoder_control.py:22-46,56-57,68-79` bypasses the optimizer and leaves AdamW moments untouched by the U update. Main has the same distinction at `engine.py:127-129,166-180`. Full also permits U-to-encoder/input-adapter learning.

Thus the observed full-versus-block gap cannot isolate encoder adaptation, allocation, or optimizer choice. `supervised_extra` also adds labeled optimization steps and accesses the audit crop as another supervised batch, so it is a label-use control rather than exact compute equivalence. These distinctions are already partly acknowledged by the documentation; they must not be lost in a mechanism narrative. The 13.47h block runtime includes JVP/audit diagnostics even though alpha is fixed to one (`decoder_control.py:54-57`); it is not a measurement of an optimized ordinary SSL baseline.

**Falsification test:** for a scope ablation, match the pseudo-update optimizer/displacement law, learning-rate rule, clipping, teacher timing and state updates first; otherwise label it a combined method comparison. Avoid silently changing the established matched guard-versus-block convention.

### F6. Multiple class candidates do not supply independent semantic evidence

**VERIFIED source.** A single `teacher(xu)` prediction supplies all candidate labels (`engine.py:164-165` main; `:174-175` experiment). `control.py:140-158` and `decoder_control.py:24-46` partition its accepted argmax labels by class with one common denominator. The source IDs are unique labels for pieces of one pseudo loss, not independent teachers, observations or estimates of correctness. Joint cross terms can exploit interacting directions but cannot establish semantic correctness independently of the budgeted labels. Candidate errors share model, teacher history, patient image and threshold policy.

**Falsification test:** vary the teacher error pattern on a fixed manufactured labeled audit and verify whether the controller learns support-specific corrections or merely suppresses all candidates. An independent-ground-truth or independent-ensemble interpretation is invalid without new evidence.

## Integrity checks that survived this challenge

- The canonical preparation path chooses label subsets deterministically from IDs/seed before reading GT; all supervised/audit mask access is restricted to the same declared labeled pool. Main `prepare_acdc.py:49,64-85`; `self_audit/data.py:14-17,31-44,79-98`; analogous experiment lines `prepare_acdc.py:49,73-94`, `data.py:55-68,103-122`. U records are image-only, and sampling U does not consult ED/ES metadata or masks. I found no canonical U/VAL/TEST-mask training leak.
- Foreground-enriched slice sampling does use L masks (`data.py:88-89` main; `:112-113` experiment). This is legitimate charged supervision, but it makes the audit measure a sampled labeled optimization objective, not the full-cine evaluation distribution.
- Source IDs are unique and common accepted normalization is used. Direct pseudo commits are not fed through an extra AdamW transform in the guarded branch. Teacher moves once after the final student state. Rejected trials are functional. These properties support a bounded transaction-level implementation claim.
- Actual measured receipts agree on source, training-manifest and initialization-weight hashes. The recomputed source digest exactly matches the experiment source and differs in design from current main. Current checkpoint loading rejects source mismatch (`engine.py:99-102` main).
- Exposed VAL is explicitly development. Raw predictions and checkpoint payloads are not in Git, and the receipts explicitly lack independently certified official full-weight hash, TEST scoring and modern external SSL comparisons. No independent-confirmation conclusion follows.

## Suspicions and gaps, not demonstrated run failures

1. Audit overfitting: labels are repeatedly optimized through supervised loss, allocation and trial acceptance. Transaction-local disjointness is not a holdout, especially with only eight L patients and fixed U/L pairings. No generalization theorem or unexposed monitoring evidence repairs this. Magnitude of overfitting has not been measured.
2. Duplicate-content defense is strongest in canonical preparation; standalone evaluation checks IDs rather than content against training (`evaluation.py:36-38`). Metadata does not retain training image hashes for that comparison. A custom renamed duplicate manifest could bypass that boundary; no retained-run duplicate leakage has been shown.
3. The recorded 9,000 accepted pseudo transactions indicate a positive reported trial scale, not semantic correctness, improved Dice or even a lower bound on realized parameter displacement. Norms and no-op checks are needed for efficacy diagnostics.
4. A numerical convex solve may be valid yet stall under per-class zero budgets, correlated candidates or rare positives; the existing source guarantees neither nonzero progress nor long-term benefit. Do not tune budgets against the exposed VAL to rescue an idea and then call that confirmation.

## Cheapest decisive next experiment

Already executed: the exact CPU counterexample decisively rejects the strongest tempting claim, namely that audited Brier safety implies Dice safety.

Next **PROPOSED_NOT_RUN**, subject to separately approved patient/compute scope: use one existing frozen checkpoint and a fixed small list of budgeted-TRAIN transactions, with no live training/EMA changes. Replay `ema_block`, independent scalar guard and joint guard using identical candidates. Evaluate on the transaction audit patient and the other budgeted L patients; record classwise Brier, hard Dice, false positives, realized displacement norm, local-versus-actual risk error, candidate cross terms and accepted scale. Include the formerly excluded U/L partner. This cheaply tests whether allocation offers any nonzero useful motion and whether its observed protection transfers even within L. It is still optimization/development evidence, not unseen-patient efficacy.

Only if that replay survives should a predeclared, matched short training comparison test persistent gain under both the existing and independently shuffled patient schedule. Independent frozen confirmation remains necessary for a paper-level efficacy claim. Do not run the cancelled full matrix merely because the code now passes software checks.


## Independent challenge and disposition

A verified the F1 counterexample scope and F2 divisibility argument, then requested two refinements. Both are incorporated: the per-frame metric pools Z,H,W and is volumetric, while averaging across frames differs from ED+ES pooling; training/audit RNG seed changes do not remove phase locking, whereas a subset seed may change patient identities/mapping while preserving the one-partner property. F independently reconstructed the alternate pooled-per-patient endpoint, retaining unchanged rankings and the missing-frame-support caveat. See `reviews/A_reviews_F.md` for A's full challenge. F's review of C found two prospective-design ambiguities (same actual candidates under divergent EMA trajectories, and missing aggregate scalar control); C corrected both along with metric/TEST wording. These reciprocal reviews validate report bounds, not method efficacy.
