# Teacher-memory pilot: prior art and pairing falsification

Audit date: 2026-10-24. Source baseline: `9341cc81fc046d61135b928f5ac51ebed19c19ed`.
Scope: literature and proposed experiment controls. This document does not certify
the implementation or report a patient experiment. Patient/GPU efficacy is **NOT RUN**
for this audit; implementation review remains separate.

The broad ingredients are established prior art. The narrow candidate is a **fixed
TRAIN-image bank of spatially paired, opposite-class boundary exemplars, read through
an antisymmetric uncertainty-gated logit residual**. This is a research hypothesis,
not a verified novelty claim. The sources below do not constitute an exhaustive
priority search.

## Primary-source comparison

| Source | Verified overlap | Difference in the proposed pilot; claim limit |
| --- | --- | --- |
| [ProC-KD, arXiv:2212.13180v1](https://arxiv.org/html/2212.13180v1), Sections III-A–C, Eqs. 1–6 | Learns teacher-derived prototypes; student cross-attention retrieves prototypes and adds a residual feature augmentation; uses embedding and prototype consistency/classification objectives. | Fixed observed boundary endpoint pairs and logit correction differ from learned prototypes and auxiliary distillation objectives. Prototype retrieval, attention and residual augmentation are not new ingredients. |
| [IC-KD, arXiv:2501.07040v1](https://arxiv.org/html/2501.07040v1), Section III-B, Eqs. 7–11, Algorithm 1 | Offline fixed-teacher feature memory; similarity retrieval from same/different classes; aggregated teacher predictions guide positive/negative distillation. The detailed retrieval equation uses a teacher query. | The proposed query is a student feature projection; retrieval couples two spatially associated endpoints and changes logits without an added matching loss. Offline teacher memory and positive/negative retrieval are prior art. |
| [BRPG, arXiv:2307.10097v2](https://arxiv.org/html/2307.10097v2), Sections 3.4–3.5, Eqs. 8–22 | Confidence-separated category memory, clustered boundary-refining prototypes, prototype classification/contrastive training, and ongoing prototype EMA updates. High-confidence threshold 0.8 appears explicitly. | BRPG addresses feature-space class boundaries through high/low-confidence clustering. The proposed bank stores actual spatially neighboring opposite-class endpoints and stays fixed. Confidence filtering and boundary-aware prototypes alone are not a contribution. |
| [Dual-Teacher Knowledge Distillation with SAM and Learnable Prototype Guidance for Semi-supervised 3D Medical Segmentation](https://papers.miccai.org/miccai-2026/0318-Paper3395.html), MICCAI 2026 | Official accepted-paper page verifies the exact title, authors Huo/Sang/Luo/Zhu/Xi, MICCAI 2026 and LNCS 16880. EMA teacher plus frozen SAM-Med3D; learnable prototype residual pseudo-label calibration and entropy-guided training. Author feedback describes class-bound, location-independent prototypes. | The exact reference exists. Its official page still lists the DOI as unavailable. The proposed frozen paired exemplar memory differs from its learnable prototype guidance. Dual teachers, external anatomical priors and prototype residual fusion are prior art. The comparison here uses the official abstract and author clarification, not an independently reproduced implementation. |

Additional nearby threats: [SimCVD](https://arxiv.org/abs/2108.06227) already combines
boundary-aware medical representation learning with pairwise similarity distillation;
[MONA](https://arxiv.org/abs/2209.13476) uses anatomical neighbor relations, and its
[author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC11903367/) describes FIFO
neighbor memory. These block broader claims about the first boundary/relational
medical-memory method. Their precise overlap with this paired residual requires a
separate full comparison before any publication-level priority claim.

## Exact independent-memory comparator A2

Use the **same stored endpoint vectors**, including any repeated endpoint occurrence,
the same projection `W`, six pair gains, base logits, detached gate, temperature and
training schedule. For an unordered semantic pair `(c,d)` containing `N` endpoint
pairs, define endpoint reliabilities `rho_c[i] > 0` and `rho_d[j] > 0`:

```text
s_c[i] = dot(q, u_c[i]); s_d[j] = dot(q, u_d[j]); tau = 0.2
a_c = softmax(s_c / (2*tau) + log(rho_c))
a_d = softmax(s_d / (2*tau) + log(rho_d))
r_A2 = sum_i sum_j a_c[i] * a_d[j] * tanh((s_c[i] - s_d[j]) / tau)
delta_A2 += beta_cd * stopgrad(4*p0[c]*p0[d]) * r_A2 * (e_c-e_d)
```

This integrates over independently retrieved endpoints, preserving `2N` vectors
while removing their observed correspondence. The paired A3 uses only `(i,i)` with
`softmax((s_c[i]+s_d[i])/(2*tau) + log(rho_c[i]*rho_d[i]))`. Both have the same
nonlinearity, temperature, information and trainable capacity. A2 evaluates at most
64 differences per class-pair when `N<=8`; disclose its compute relative to A3.
Pooling each class first and applying `tanh` afterward is a weaker control because
it changes the placement of the nonlinearity as well as pairing.

Reliability is not fully specified by the phrase "log(reliability)". Freeze a
factorized endpoint convention (for example `rho_c=sqrt(conf_c)` and
`rho_d=sqrt(conf_d)`) across A2/A3/A4, or use uniform reliability for their primary
comparison. Do not carry an old pair-specific scalar onto newly shuffled endpoints.
This formula is a review recommendation; the implemented protocol must record its
actual choice before execution.

## Controls and conclusions they can support

| Comparison | Question | Falsification or interpretation |
| --- | --- | --- |
| A3 vs A0 fixed-window full EMA | Does the complete memory addition help? | No improvement rules out demonstrated utility for this configuration; improvement alone does not identify pairing or external knowledge as its cause. |
| A3 vs A1 external pseudo-label CE | Is the memory route better than external supervision on the same EMA-eligible transactions? | Match external teacher checkpoint, TRAIN-image exposure, U inputs, update scope, accepted-pixel normalization and tuning budget. Under the frozen pilot decision, external CE is added only when the existing EMA U transaction runs; no external-only update occurs when EMA support is empty. Record candidate external support separately from applied external support. This is a restricted external-supervision comparator, not a pairing isolation. |
| A3 vs A2 independent memory | Does observed endpoint correspondence add value beyond the same marginal features? | Similar or better A2 performance removes support for pairing-specific benefit. Non-significance alone is not evidence of equivalence; use a prespecified meaningful-effect margin and patient-level uncertainty. |
| A3 vs A4 correspondence shuffle | Do actual spatial pair associations matter? | Permute `d` endpoints **across entries inside each semantic class-pair**, preserving class labels and endpoint reliabilities. An A3 benefit must survive repeated frozen shuffles. Swapping `c` and `d` inside the original pair is orientation invariance or label corruption, not this null. |
| A3 vs A5 student-derived memory | Is the external source needed? | Derive labels, positions and features from a declared frozen A0 student if claiming no external teacher. If external features remain, call this only a no-external-label/position control. Match dimension, normalization, vector counts and projection capacity; disclose any fixed projection. |
| A3 vs A6 random-feature bank | Do semantic feature directions matter beyond extra model capacity? | Preserve metadata, pair topology, confidence/gate, vector count, dimension and normalization; replace directions with seeded normalized noise. Comparable gains weaken the semantic-memory interpretation. |

With fewer than two entries, A4 cannot destroy correspondence: record an identity
shuffle and count that class-pair as uninformative for this test. Shuffling across
patients also changes patient association. Prefer a within-patient shuffle when
support permits; otherwise disclose this confound and do not call A4 a pure test of
spatial adjacency. A2 and A4 address related but distinct nulls.

## Mathematical checks and interpretation

For normalized endpoint/query vectors, `|r_cd|<=1`. With
`beta_cd=0.5*sigmoid(alpha_cd)` and valid probabilities, the proposed formula gives:

```text
sum_c delta[c] = 0
|delta[c]| <= beta_max * 4*p0[c]*(1-p0[c]) <= beta_max <= 0.5
beta_initial = 0.5*sigmoid(-4) = 0.00899310498...
```

These are algebraic bounds, not performance guarantees. Test them against the
implemented tensor layout. At initialization the per-class logit effect is small;
a short null pilot can reflect insufficient learning rather than an adequately
powered rejection. Report gain, gate, retrieval entropy and residual magnitude.
`tanh(x)=2*sigmoid(2*x)-1` is a conventional pairwise soft preference, not a new
theorem. The gate suppresses confident errors as well as confident correct pixels.
No boundary, Dice, calibration or anatomy guarantee follows from zero-sum logits.

## Review checklist before interpreting a patient pilot

- [ ] Record immutable model, weight, bank, source-list, split, seed and config hashes;
  bank evidence uses TRAIN images only, with no VAL/TEST/U mask access.
- [ ] Boundary endpoint labels/locations come from the declared predictions, never
  hidden masks; both endpoint confidences meet 0.8 and the exact 3 mm physical
  distance rule uses correct image spacing, resampling and coordinate transforms.
- [ ] Distinguish spatial adjacency across a predicted interface from merely having
  two different labels somewhere inside a radius. Freeze the chosen predicate.
- [ ] Include all six BG/RV/MYO/LV unordered pairs; report missing support explicitly;
  at most eight pairs each; patient balancing, deterministic selection and no
  synthetic filling of missing anatomical pairs.
- [ ] Confirm stride-4 `refine[0]` hook, unit normalization, finite reliability,
  correct softmax axis, detached gate, fixed buffers, and no online teacher call,
  bank mutation, teacher matching loss, meta loss or window adaptation.
- [ ] A2/A3/A4/A6 share stored vector marginals where applicable, parameter counts,
  initialization, labeled/U samples and update scope; A4 preserves semantic labels.
- [ ] Confirm A5 provenance field by field; do not label it teacher-free when any
  external teacher field or external pseudo-label selection survives.
- [ ] Predeclare primary patient-level contrasts, meaningful-effect margin, seeds
  and checkpoint selection. Preserve failed seeds, empty-class cases and absent
  pair support. Repeated slices from one patient are not independent replicates.
- [ ] Report classwise Dice, physical surface/boundary metrics and BG false positives;
  use frozen evaluation predictions. Evaluation GT can define a locked diagnostic
  boundary band but cannot feed bank construction, gates or tuning.
- [ ] Report all A0–A6 outcomes and confidence intervals. Exposed VAL/TEST results
  remain exploratory; use fresh held-out data for confirmation. No CPU/synthetic
  check can establish patient efficacy or novelty.

Machine-readable companion: [NOVELTY_AUDIT.json](../reports/teacher_memory_pilot/NOVELTY_AUDIT.json).
