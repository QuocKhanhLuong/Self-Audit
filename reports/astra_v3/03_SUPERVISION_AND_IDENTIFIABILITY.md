# Supervision and identifiability

**Current v3 does not contain a mechanism that grounds BG/RV/MYO/LV names.** Names come from an external semantic source, such as explicitly typed anatomical rules or previously labeled model weights; a four-row MLP does not create them.

## What is and is not identifiable

Write the final semantic layer as `z=W h+b`. For a class permutation P, use `W'=PW, b'=Pb`. Then `softmax(z')=P softmax(z)`. Reconstruction and class-symmetric clustering, entropy and consistency objectives have the same value, while the names change. Root's RV/LV permutation experiment has zero numerical equivariance error.

This is a statement about the scaffold and symmetric objectives. It is **not a theorem that image-only anatomy is universally impossible**. Typed asymmetric priors can associate a surrounding wall with MYO and its cavity with LV. Their correctness on pathology, base/apex and acquisition variation must be tested. They supply semantic knowledge even when they use no manual segmentation arrays. A permutation orbit has up to 24 class assignments; it need not consist of 24 isolated global optima, contrary to Worker B's stronger wording.

Anonymous segmentation answers which pixels belong together. Named segmentation additionally assigns a fixed clinical meaning. Hungarian mapping with evaluation masks is allowed only as explicitly labeled oracle partition analysis. It cannot produce the deployable named-mask score.

## Semantic-information and leakage ledger

| Channel | Current observation | Locked policy |
|---|---|---|
| Manual masks | No v3 loader, but supervised ACDC preprocessing loads masks and requires paired existence | Never use that preprocessing/training path for the proposed no-mask pipeline |
| `_gt` filename presence | Historical local ED/ES acquisition was selected by paired mask presence | Disclose provenance; new acquisition enumerates image-only allowlist independent of mask existence |
| Filename patient/phase | Useful grouping; no cavity/wall semantics | Patient split before slice/frame expansion; prohibit diagnosis-derived targets |
| ACDC Info.cfg | ED/ES and phase-count metadata; diagnosis/clinical fields also present | Allow only declared phase fields if needed; no diagnostic/measurement targets. Maskfree legacy firewall forbids the entire sidecar; new contract must explicitly differ |
| Orientation | Potential left/right cue; all 200 local images have unknown units and zoom/affine mismatch | No trusted physical-orientation cue from this copy; reacquire/verify acquisition geometry |
| Crop/normalization | Supervised loader resizes images/masks; crop provenance can hide GT | Full FOV aspect-preserving preprocessing; record every transform and image-only statistics |
| External weights | v3 defaults random; no pretrained weights loaded by scaffold | Hash and inventory every checkpoint, pretraining dataset, task and crop/selection dependency |
| Semantic evidence | Caller supplies `B,K,4` arbitrary tensor | Typed evidence producer, source hashes, per-cue validity/conflict records required before use |
| Thresholds | .70/.20 scaffold defaults, not calibrated precision | Freeze before evaluator opens GT; no post-evaluation retuning of the same identity |
| Pseudo selection | Reliability can prefer confident wrong labels | Measure class precision, class coverage, swaps and patient harm; confidence is not a guarantee |
| Checkpoint/round selection | No runner currently | Fixed epochs/checkpoint/round schedule; no GT-best checkpoint. Development stop/go is disclosed research feedback |
| Evaluation | H process ran after manifest freeze; root reran the same independent evaluator process | No evaluator imports of training/model code, no mask arrays returned upstream; hashes before and after |

The generator used an image-only mirror and a path access guard. That is an auditable local control, not a security proof against renamed masks or an independently compromised process. Hashes establish artifact identity; they cannot retroactively remove human development feedback or historical acquisition selection.

## Correct historical negative controls

Root read `/Users/alvinluong/Self-Audit-nogt/reports/astra_nogt_baseline_20260918T042050Z/{RESULTS,BASELINE_DECISION,SUPERVISION_LEDGER,RUNBOOK}.md`.

| Method | Historical foreground Dice |
|---|---:|
| C0 | .01623 |
| C1-cache seed17 | .02146 |
| C1-cache seed29 | .02751 |
| C1-direct | 0 |

ARI was also poor (approximately .02–.03 for non-direct variants). These are unsuccessful anatomy methods. Historical metrics pool ED/ES counts within each patient before Dice; the new audit uses phase-equal Dice. They are not a matched numerical comparison with Round-0.

Worker B accidentally substituted **Candidate-C numerical replay checks also named C0/C1** for these experiments. That evidence is rejected for this question. Reusing old topology rules does not overcome the negative results; C0 remains a falsification control, never a distillation target. The old reference-free Auditor has no demonstrated useful correction signal and is not reintroduced.

## Claims that must stay separate

| Setting | Masks in optimization | Other prior information | Honest claim |
|---|---|---|---|
| Scratch + explicit anatomy rules | None | Human-designed class rules | Manual-segmentation-mask-free, prior-guided; not semantic-knowledge-free |
| Same recipe + image-only SSL pretraining | None if full lineage verifies | Unlabeled images and acquisition/view priors | Mask-free SSL-initialized; not scratch |
| Mask-supervised external model | External masks influenced weights | Learned named anatomy | Target-dataset-label-free transfer only; violates strict no-manual-mask-anywhere setting |
| ImageNet classification pretraining | No segmentation masks necessarily | Human image class labels | Possibly segmentation-mask-free, but not label-free/scratch |

Worker B's blanket assertion that all pretrained classification weights violate “mask-free” is too broad. Conversely, self-supervised representations alone do not assign the four names. Strict scratch reaching .91 is unsupported under present evidence, not mathematically disproved in all possible prior-guided systems.
