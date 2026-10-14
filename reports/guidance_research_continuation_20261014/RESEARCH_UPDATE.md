# Self-Audit research continuation after the published world-left-v2 run

**Decision: NO-GO for another full training run; GO for frozen, stagewise attribution using existing artifacts.** The current result does not establish trustworthy named RV/MYO/LV supervision or a benefit from an active self-audit mechanism. The executed mechanism is fixed stable guidance. The new evidence rules out some earlier explanations and makes two others testable: inadequate named teacher evidence, and a loss/export mismatch that is large relative to the very small accepted support.

## Identity and what was actually done

Remote refs were refreshed, an isolated worktree was created from current `origin/main`, and `git pull --ff-only origin main` completed with exit 0. Analyzed branch: `QuocKhanhLuong/guidance-results-research-20261014`. HEAD: **`b937e3b746608725e43d3bc3cef8861666d152dc`**, committed **2026-10-14T12:50:11+07:00**, “reports: publish world-left-v2 training logs and results.” Remote main was rechecked at that SHA. The earlier repair branch remains remotely available at `7d1c6a04064cda473d1eccfbed818392c37b1a37`; this main commit publishes evidence, not a top-level merge of that experimental source.

The new run is `stable_guidance_world_v2_acdc_4070_20261014_024155`, W&B `syevfplp`, recorded checkpoint SHA256 `90bc16bb9f89963b452355376673e3ba9aa24b38870e4595237d75e402ce873e`. Its executed source is base `7d1c6a0` plus launcher patch `ecbcd639fbaa010ac3f7fca9ac5ba3272278806ce92eecc8171c4833f70815ee`. The comparator remains the original run `stable_guidance_acdc_4070_20261013_222918`, trained at `d5a45f31481f0a702a1f9d5d418889fe704e9105`, checkpoint `516e569c9141e2d8edb00b598df5cf12c8a9a4741cf1ae0cca8cbf23d6d03353`.

Eight real Orca workers were launched with requested and effective model `gpt-6-astra`: provenance/metrics, learning mechanism, geometry, theory/prior art, independent adversarial review, evidence-bank instrumentation, paired-pilot implementation, and stagewise diagnostics. Workers were reused for corrections and code review. Their identities, outcomes and resource disposition are in [ORCHESTRATION.json](ORCHESTRATION.json). This was substantive completed source/research work, not a dispatch-only plan.

We inspected the published recipe, effective split, preflight, bank/student manifests and reports, all 50 epoch records, export locks, frame confusion counts, launcher logs/patch, and the **executed-source snapshot**. We independently reran text/hash/arithmetic and CPU geometry/mathematical checks. We did not execute the reported 50-epoch run, rerun a model checkpoint, read real patient arrays/masks, run real-data training, tune thresholds, or evaluate official TEST in this continuation. The binary payloads were excluded from the publication; their digests remain recorded identities, not locally rehashed originals. Original frozen artifacts and production code were unchanged. The inspected VAL is exposed and all resulting development is reference-informed.

[PROVENANCE_METRICS.md](PROVENANCE_METRICS.md) specifies every identity binding, source path/line, JSON key, stratum and null convention. Its 406 scoped checks pass; this verifies the published evidence's internal consistency and arithmetic, not an independently replayed experiment. All 102 manifest entries and eight additional package files match pinned Git bytes. The archived 120/122 test passes are historical receipts, not tests rerun in this session.

## Four findings that change the diagnosis

### 1. The apparent improvement is heterogeneous and loses an entire requested class

The same 18 patients and 36 labeled frames are retained. Recomputed frame → patient → class means are:

| Outcome | Baseline | World-left-v2 |
|---|---:|---:|
| Foreground macro Dice | 4.684410% | 5.068521% |
| Foreground recall | 3.273159% | 3.322679% |
| Patient-macro coverage | 0.614758% | 0.366846% |
| UNKNOWN | 99.385242% | 99.633154% |
| RV predicted pixels, 36 frames pooled | 15,593 | **0** |
| Patients with higher paired Dice | — | **4/18** |

**DERIVED from reported confusion counts.** The median paired Dice change is **−0.126854 percentage points**. Patient098 contributes +0.679169 points to the +0.384111-point full-cohort change; leave-one-out removal of that patient changes the mean delta to −0.312415 points. This is an influence diagnostic, not permission to exclude that patient. The full endpoint remains all 18 patients. Both the 11-patient aligned and seven-patient verified-recovery strata are retained; neither is a TRAIN orientation-authority stratum.

New RV precision is undefined because no RV is predicted. Thus headline foreground precision rises from 19.623489% to 29.084105% partly because the macro drops RV. Exactly **8.115510 of the 9.460615 percentage-point difference** comes from dropping baseline RV from its class average. A strict common-support calculation on 15 patients, MYO/LV and both frames does improve descriptively by 4.822060 points, but conditions on prediction by both models and excludes RV. It does not establish whole-cohort improvement or useful segmentation. See the complete paired table and derivations in [PROVENANCE_METRICS.md](PROVENANCE_METRICS.md).

**Claim changed:** “the repair improved the model” is too broad. The evidence supports a small, concentrated mean Dice change with reduced coverage and complete RV failure at export. No repeated-seed causal or untouched-cohort inference is available.

### 2. Geometry changed the authority of RV evidence, not oblique recovery in this cohort

**VERIFIED source; DERIVED paired records.** All 52 coded TRAIN cases use an axis-aligned `diag(-1,-1,1,1)` sform with code 2, no coded qform and unknown spatial units. Their new bank left covector is a positive multiple of the old `+x` rule. Every per-patient/class hard count is unchanged. No TRAIN case exercised recovery of a 30/45/60-degree left direction rejected by the old cardinal test.

The 28 uncoded cases lose all RV votes under the declared geometry-abstention policy. Their accepted RV total changes **4,322 → 0**; coded RV remains **877 → 877**. Total new RV hard support is 877 pixels in 14 patients, versus 5,199 in 26 patients before. BG/MYO counts change slightly because deleting votes changes all normalized target coordinates and downstream reliability, not just one hard-label channel.

A precise source-level consequence is stronger than the count observation: if a patient's RV vote is zero everywhere, then `R0_RV≤1/4`; nonnegative row-stochastic restart propagation preserves this inequality at every iteration. It cannot reach a 0.70 RV bank threshold. This is a **bank** result. Global student permissions remain open, and the shared student can still predict RV on that patient.

Full weighted RV target-channel mass falls only about **3.09%**, while hard RV count falls **83.13%**. Those quantities cannot be equated. Neither identifies whether removed labels were correct. A coded affine is also not independent validation of patient laterality; its conversion/acquisition provenance remains material. Complete proofs, source comparisons and geometry checks are in [GEOMETRY_ATTRIBUTION.md](GEOMETRY_ATTRIBUTION.md).

**Claim changed:** the geometric software defect is real on synthetic oblique transforms, but it is not the demonstrated cause of the original ACDC failure. Restoring unsupported fallback directions merely to recover coverage would not add trustworthy information.

### 3. The semantic channel operates, but its counters and channel masses overstate what they establish

The actual path is:

```
named votes → R0 → propagated Q/reliability → hard bank labels
                                         ↘
hard labels → global class permissions → soft Q/reliability loss
               → common-grid student logits → native logits → export gates
```

All four permissions pass in both runs. The new semantic counter is **110,461 / 127,650 eligible batches**, not a measurement of nonzero anatomical learning on every step. Baseline saved checkpoint probes have nonzero shared-encoder semantic gradients in all 80 selected batches; those are published historical forensic measurements, not model forwards rerun in this continuation or recovered original optimization gradients. One eligible batch has only `9.76e-14` total weight, below the `1e-8` denominator floor, illustrating why eligibility and signal strength differ.

The new last-epoch equal-batch values satisfy

```
CE = H + KL = 1.0138087404 = 0.9813062258 + 0.0325025146.
```

Entropy accounts for **96.794%** of CE. Active-batch mean KL falls from 0.073778 to 0.037581, while feature loss also falls. This supports optimization progress, not adequate class-specific fitting or anatomical improvement. The epoch average combines changing checkpoints and inactive zeros; it is not a final-checkpoint divergence under a patient/class measure.

The baseline bank supplies positive soft weight on **9,095,070 hard-UNKNOWN pixels**; for RV, 128,800 of 133,999 positive-weight RV-argmax pixels are hard UNKNOWN. A hard-bank expansion obstruction therefore does not imply zero soft supervision or prevent neural generalization. Conversely, `ΣwQ_RV>0` does not prove positive RV-favoring evidence: a BG-biased target `(.4,.2,.2,.2)` has positive RV mass while its RV coordinate lies below uniform and can push an RV logit downward. An additional foreground target makes the two-pixel counterexample obey the real batch-activation rule. The derivative and finite-difference check are explicit in [LEARNING_MECHANISM.md](LEARNING_MECHANISM.md) and [ROOT_MATH_CHECKS.json](ROOT_MATH_CHECKS.json).

**Claim changed:** global no-foreground absorption and “CE≈1 implies optimizer failure” do not explain these runs. Sparse or biased named information, rare-class fit and teacher bias remain open.

### 4. Native-grid instability is large on the accepted support, even when globally small

**DERIVED from the original checkpoint's 80 selected TRAIN batches; no reference scoring.** Native export accepts 215,428 of 35,335,936 pixels. The predetermined coarse-probability projection control accepts 121,350. Reconstructing gated-label disagreement from the saved fractions yields 119,330 pixels.

If accepted sets have sizes `a,b`, intersection `i`, and label disagreement `d`, then `d=a+b−2i+v`, where `0≤v≤i` counts different labels accepted by both. Hence
`ceil((a+b−d)/2)≤i≤min(a,b,a+b−d)`, together with ordinary set-size bounds. Applied to the saved batches, **43.67–49.53% of native-accepted pixels are absent from the projection's accepted set**. The small global disagreement percentage does not make this immaterial.

A complete constructive counterexample also shows identical coarse logits/CE with arbitrarily changed native predictions: add logit perturbations in the downsampling operator's null space. The executed objective does not observe them. This proves a missing spatial constraint, not that the network realizes every such perturbation.

**Claim changed:** resolution handling deserves attribution before further training. It does not follow that projection is anatomically better; it actually accepts fewer pixels in these baseline probes. The equivalent measurement for the new checkpoint and stagewise reference scoring remain absent. Details and full set/CE proofs: [LEARNING_MECHANISM.md](LEARNING_MECHANISM.md) and [THEORY_PRIOR_ART.md](THEORY_PRIOR_ART.md).


## Updated scope and local handoff

The user subsequently directed **local code and preflight only**, with a separate Codex on the 4070. The single current SSH attempt failed authentication before that direction; no further remote attempt is made. This does not invalidate the newly published original-checkpoint audits or completed world-left-v2 run. New tensor-level measurements are **NOT RUN HERE**, not a return to the superseded historical blocked-forensics conclusion.

The current primary handoff is [UPDATED_FAILURE_ATTRIBUTION.md](UPDATED_FAILURE_ATTRIBUTION.md), [TARGET_SIGNAL_AUDIT.md](TARGET_SIGNAL_AUDIT.md), [NEXT_EXPERIMENT_DECISION.md](NEXT_EXPERIMENT_DECISION.md), and the [local experimental runbook](../../experiments/guidance_signal_20261014/README.md). The isolated opt-in code pins the executed source, reconstructs evidence without entropy/reliability inversion, freezes all stage predictions before reference scoring, and gates a matched fixed-step Q→Z pilot on original-artifact identity and reproduction. Synthetic CPU optimizer steps test software contracts only.

[STAGEWISE_DIAGNOSTICS.csv](STAGEWISE_DIAGNOSTICS.csv) includes the available old/new gated confusion counts and explicitly missing stage rows. Missing raw/teacher/student probabilities, geometry strata or joint-region errors are blank/UNKNOWN, never estimated from marginals. Published precision nulls remain null. The old checkpoint's gradient probes remain historical selected-batch evidence. No new-checkpoint real gradient or anatomical result is claimed.

All new files remain **local, uncommitted and unpublished** on the isolated branch. Production source, frozen runs, thresholds, patient splits and prior research history are unchanged. No full-run authorization follows from software preflight.

Final software preflight: **85 tests passed**, exit0; [EXECUTION_RECEIPT.json](EXECUTION_RECEIPT.json). The subsequent independent code-review follow-up was interrupted by provider model capacity after delivering actionable findings and seven mathematical checks. The coordinator verified the corrections and final integrated suite; see [CODE_REVIEW.md](CODE_REVIEW.md). No completed independent post-fix signoff is claimed.
