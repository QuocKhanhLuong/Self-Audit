# Updated failure attribution: completed world-left-v2 run

Analyzed publication: `origin/main@b937e3b746608725e43d3bc3cef8861666d152dc`, 2026-10-14. This supersedes the older report's lack of published original-checkpoint forensics. The package now contains a completed 80-patient original-checkpoint audit and a completed new 50-epoch run. Those historical measurements are available; the additional stagewise tensors requested below have not been measured in this local continuation.

The new run is `stable_guidance_world_v2_acdc_4070_20261014_024155`, base `7d1c6a04064cda473d1eccfbed818392c37b1a37` plus launcher patch `ecbcd639fbaa010ac3f7fca9ac5ba3272278806ce92eecc8171c4833f70815ee`. New checkpoint: `90bc16bb9f89963b452355376673e3ba9aa24b38870e4595237d75e402ce873e`; old checkpoint: `516e569c9141e2d8edb00b598df5cf12c8a9a4741cf1ae0cca8cbf23d6d03353`. The executed source copy is authoritative for this analysis; its presence does not mean it was merged into production `src/`.

## Confirmed, refuted, and unresolved explanations

| Candidate explanation | Verdict | Decisive evidence and limit |
|---|---|---|
| Wrong semantic anchors are the primary failure | **UNRESOLVED** | End-to-end anatomy is poor; named R0/Q were not scored stagewise in the published evidence. A high-fidelity wrong teacher is possible. Readout rules and feature agreement do not establish anatomical correctness. |
| Target dilution / starvation of positive class evidence | **PARTLY CONFIRMED, causal primacy UNRESOLVED** | Targets explicitly contain a uniform mixture; 28 uncoded TRAIN patients have no new RV votes and cannot exceed Q_RV=1/4. RV hard support falls 83.13%, while total weighted RV channel mass falls only 3.09%. Positive channel mass is not favorable class evidence. Full M/h/Z and rare-location diagnostics remain to be executed. |
| Student cannot fit rare-class information | **UNRESOLVED** | All permissions open; online KL decreases. Mean KL can conceal RV-specific errors, low-weight regions and gradients. Need matched final-checkpoint diagnostics stratified by positive source evidence and patient; mean telemetry does not settle this. |
| Loss-to-native transfer matters | **CONFIRMED as baseline sensitivity; anatomical contribution UNRESOLVED** | For 80 selected old-checkpoint TRAIN blocks, 43.67–49.53% of native accepted support is absent in the fixed projection control. Coarse-loss null-space counterexample is exact. This is not a new-checkpoint measurement or proof that projection improves anatomy. |
| Global class permission closed / no semantic operation | **REFUTED for these runs** | New class-patient counts [62,14,55,47] permit all classes. 110461/127650 batches were eligible. Eligibility is not the number of nonzero semantic parameter changes. |
| CE near one means optimization failed | **REFUTED as an interpretation** | Epoch-50 CE=1.0138087404 = H 0.9813062258 + KL 0.0325025146. H is 96.794% of CE, not anatomical accuracy. |
| Persistent destructive gradient conflict | **NOT ESTABLISHED** | Old selected-checkpoint probes have mean cosine +0.1961 and 19/80 negative cosines; these are not random population, historical trajectory, or new-checkpoint gradients. |
| The old 45-degree cardinal rejection caused the observed TRAIN failure | **REFUTED for this cohort** | All 80 old TRAIN orientations are +x and no wall/cavity opportunity was skipped for missing cardinal orientation. Real oblique recovery was not exercised. Synthetic geometry correctness remains valid software evidence. |
| Header-policy change removed RV support | **CONFIRMED** | Coded 52: 877→877 hard RV pixels; uncoded 28: 4322→0. All coded affines are axis-aligned code-2 sforms. This is source support removal, not removed-seed accuracy or a complete downstream causal explanation. |
| New raw RV argmax is absent | **NOT MEASURED** | Only accepted RV is known absent in all 36 VAL frames. Gating could reject raw RV winners. Stagewise code preserves this distinction. |
| Larger architecture / active CQA mechanism was tested | **FALSE** | The run used fixed Q weighted CE plus feature cosine. Q→evidence-content targets, changed student context, recurrent relabeling and image-grounded annotation repair were not tested. |

The earliest established **support** obstruction is at readout/bank generation. The earliest **anatomical error** remains unlocalized. These are different claims.

## Preserved endpoint and denominator facts

Frame ratios are computed from TP/FP/FN, averaged over defined frames within patient/class, then patients/classes. The complete locked cohort is retained: 18 patients, 36 frames, 11 aligned and seven verified-recovery patients. No pixel-IID significance claim is made.

- Dice: 4.684410% → 5.068521%; coverage: 0.614758% → 0.366846%.
- Only 4/18 paired patient Dice values increase; median change is −0.126854 percentage points. Patient098 dominates the positive mean. Leave-one-out sensitivity is descriptive and does not change the reported cohort.
- New accepted RV: TP=0, FP=0, FN=230281. Precision remains **undefined**, not zero.
- Foreground macro precision 19.623489% → 29.084105% changes the class/patient/frame support. Dropping old RV alone accounts for 8.115510 percentage points of the headline difference.
- Correctly predicted class coverage `TP/(TP+FN)` equals class recall under this action definition. Coverage of *all accepted decisions on a true class*, including wrongly named decisions, requires a joint confusion/abstention record and is not inferred from class marginals.

See [PROVENANCE_METRICS.md](PROVENANCE_METRICS.md) and [STAGEWISE_DIAGNOSTICS.csv](STAGEWISE_DIAGNOSTICS.csv). Blank unmeasured-stage entries are intentional. They are not zero predictions or evidence of all-UNKNOWN behavior.

## The complete information path and the new target arm

Raw named votes are mixed with uniform uncertainty, propagated, and thresholded into hard bank labels. Hard labels determine global permissions. The loss still uses **soft Q and reliability** at many hard-UNKNOWN pixels. The native exporter thresholds student probabilities without the bank's reliability gate. Thus a hard-bank bound, soft-supervision amount, learned class prediction and accepted output are four different objects.

For nonnegative source evidence `R0=(1-s)u+sV`, fixed row-stochastic P and restart kernel K, exact content variables satisfy

```
M = K(sV), h = Ks, Q = (1-h)u + M, Z = M/h where h is reliably positive.
CE(Q,p) = (1-h) CE(u,p) + h CE(Z,p).
```

The uniform component can favor a faithful low-confidence student. Removing that component changes gradients; replacing CE(Q,p) by KL(Q||p) alone does not, because target entropy is detached and independent of student logits. A scalar multiplier scales the entire gradient rather than deleting the uniform component.

This identifies a justified **diagnostic ablation**, not a proved remedy. Z preserves the source class ordering in exact arithmetic when h>0; it does not discover the correct name. Wrong image-derived names can become confidently wrong. Missing raw vote histories cannot be reconstructed from R0, nor h from reliability divided by an almost-zero entropy deficit. The implemented experimental route replays named sources and the recorded graph, persists M/h/provenance, checks Q reproduction, and abstains when the numerical ratio is unsupported. [TARGET_SIGNAL_AUDIT.md](TARGET_SIGNAL_AUDIT.md) contains the proofs and counterexamples.

The Q-versus-Z pilot uses identical valid-support masks, original detached weight values and denominator, original Q-derived permissions/activation, identical initialization/order/budget and unchanged model/input/gates. It logs the untouched full-Q loss separately. Its shared-valid-support Q control is explicitly distinct from the published full-domain objective if any positive-weight positions are excluded. Substantial excluded weight or failed reproduction prevents the pilot.

## Current access and execution boundary

A fresh SSH attempt to configured alias 4070 returned exit 255 (authentication denied); saved Orca remote environments were empty. A bounded check of existing Self-Audit ZIP names found report-only archives, no original banks/checkpoints. This does not negate the published successful host audits.

The human then explicitly directed: **do not access 4070; code and preflight locally, with a separate Codex on that host.** No further host attempt is made. The remaining real-data stages are therefore **NOT RUN HERE / EXECUTION-HOST HANDOFF**, not an assertion that the original run or its completed audit is inaccessible everywhere.

Local opt-in code lives only under `experiments/guidance_signal_20261014/`. Production code, original configurations, split, old runs and published evidence remain unchanged. Software/synthetic preflight is distinct from original-artifact identity verification, real Q reproduction, CUDA execution and anatomical scoring. All new documents/code remain local and uncommitted/unpublished unless a later explicit publication instruction changes that status.

## Decision

Do not choose between anchor error, target dilution, rare-class fitting and native transfer using the existing aggregate loss or endpoint alone. Complete frozen stagewise attribution and the positive-evidence audit first. Only then run the bounded, single-factor Q→Z pilot if its identity and numerical checks pass. Annotation repair, rare-source reweighting, student-context changes and larger models remain conditional and unimplemented. The exposed VAL is development evidence throughout; no research-wide label-free selection or pristine-validation claim is made.

