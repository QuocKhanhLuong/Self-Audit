# Methods and code audit, 18 September 2026

No score in this report is a head-to-head result against our ACDC pilot. “Unsupervised
loss” does not imply image-only data preparation, named output, or label-free selection.
Astra read the primary methods and independently inspected the decisive code below.
Worker audit detail is in `workers/`; source/code contradictions remain explicit.

## Six closest systems

| Method / primary methods read | Input, initialization | Train and selection labels | Naming and loss | Per-image work / whole cost | Implication and unknowns |
|---|---|---|---|---|---|
| [CUTS, §3 and evaluation](https://arxiv.org/html/2209.11359v7) | Medical 2D images; learned patch encoder, no foundation model required | Contrastive + reconstruction training; **binary extraction in evaluation uses GT hint** | Anonymous diffusion-condensation partitions; local patch reconstruction and SSIM/proximity-mined contrast | Dense embedding, PHATE/diffusion condensation, multiple scales per image; not just encoder parameter count | Useful partition prior art, not a solution to GT-free BG/RV/MYO/LV naming. Full code/data selection history not reproduced. |
| [IIC, §3](https://arxiv.org/html/1807.06653) | Paired augmented images/patches; random network initialization | MI training has no semantic targets; released segmentation script has GT-matched evaluation and best checkpoint | Discrete joint-distribution MI; output clusters need correspondence | Shared trained network; paired training forwards and local spatial joint statistics; no test weight optimization | Marginal entropy favors balanced **clusters**, not equal cardiac class areas. Does not identify anatomy names. |
| [Ferreira et al., Methods and supplement](https://doi.org/10.1038/s41467-025-59451-5) | Echocardiography A2C/A4C/SAX; SAX HED initialized with ImageNet VGG weights | Automatically extracted weak labels, shape QC, self-learning; validation weak-label loss stopping; human visual tuning acknowledged | View-specific watershed/Hough/geometry and chamber relationship priors; U-Net learns weak labels | Multiple staged teachers/U-Nets, extraction/QC/recruitment, then trained inference | Closest conceptual bootstrapping system; **not an entirely scratch pipeline and not MRI segmentation**. Public examples differ from paper's stopping/parameters. |
| [SmooSeg, §3/§4 and Algorithm 1](https://arxiv.org/abs/2310.17874) | Frozen ImageNet-pretrained DINO ViT-S/8 | Training objective ignores GT; released validation uses GT Hungarian mIoU and top-1 checkpoint | Projector, local prototypes, EMA teacher prototypes; within/across-image smoothness plus teacher CE | Frozen ViT forward still costs compute; pairwise training and CRF evaluation; no test weight updates | Violates scratch requirement and released selection contract. Pseudo-label amortization itself is established. |
| [UnSegMedGAT, §II/III](https://arxiv.org/html/2411.01966v1) | Frozen DINO key features + GAT | Modularity loss; public notebook **GT crop before features and GT polarity selection** | Binary clusters, relaxed modularity + anti-collapse regularizer | Dense graph, 60 per-image epochs in paper; collective stage + per-image stage; optional bilateral solver | Small GAT does not mean cheap pipeline or GT-free deployment. Exact reported checkpoint/hyperparameter selection unknown. |
| [DiffSegLung, §3/4, 2026 preprint](https://arxiv.org/html/2605.11758) | CT HU and radiomics; 3D DDPM | Claims no annotations; descriptor discrimination study provenance unclear | GMM plus HU-based pathology assignment; diffusion + radiomic contrastive distillation | Paper specifies **250 DPM-Solver steps**, multi-time features, GMM and boundary refinement | HU physical naming cannot be transferred to MRI intensities. Code completeness, lung-mask provenance and text/table agreement unresolved. |

## Decisive official-code checks

**CUTS:** author repository directs to KrishnaswamyLab. Pinned
`27751160b8dc3304f57d3725e7e1028a576f6849`.
[`label_hint_seg`, lines 5–38](https://github.com/KrishnaswamyLab/CUTS/blob/27751160b8dc3304f57d3725e7e1028a576f6849/src/utils/segmentation.py#L5)
selects the cluster most common at GT foreground locations. Its actual callers are
[`generate_kmeans`:17–43](https://github.com/KrishnaswamyLab/CUTS/blob/27751160b8dc3304f57d3725e7e1028a576f6849/src/scripts_analysis/helper_generate_kmeans.py#L17)
and [`segment` / `segment_every_diffusion`:143–170](https://github.com/KrishnaswamyLab/CUTS/blob/27751160b8dc3304f57d3725e7e1028a576f6849/src/scripts_analysis/run_metrics.py#L143).
This does not invalidate anonymous clustering; it invalidates importing those binary
outputs as a no-GT anatomical-label generator. Evaluation hint and training label use
are different claims.

**IIC:** pinned `b7602b743552c28a4af3de238bf2539a3af0338e`.
[`IID_segmentation_loss`:14–83](https://github.com/xu-ji/IIC/blob/b7602b743552c28a4af3de238bf2539a3af0338e/code/utils/segmentation/IID_losses.py#L14)
aligns transforms, accumulates/symmetrizes joint probabilities, then minimizes negative
MI. [`segmentation_eval`:19–39](https://github.com/xu-ji/IIC/blob/b7602b743552c28a4af3de238bf2539a3af0338e/code/utils/segmentation/segmentation_eval.py#L19)
returns best accuracy; [`segmentation.py`:331–370](https://github.com/xu-ji/IIC/blob/b7602b743552c28a4af3de238bf2539a3af0338e/code/scripts/segmentation/segmentation.py#L331)
writes best.pytorch using it. GT Hungarian matching is in cluster_eval:193–227.
We do not equate the unsupervised objective with a GT-free released selection pipeline.

**Ferreira:** author-linked CardioML `784799faded0169a15cce58b58a7d600c5582bae`.
[`hed.py`:25–98](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/model_arch/hed.py#L25)
loads local VGG16 weights; binary provenance lacks a documented download hash.
[`util_seg.py`:156–245](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/utils/util_seg.py#L156)
implements watershed weak labels; view-specific QC and RV stretching are in the
subsequent scripts. Paper: weak-label validation-loss elbow and repeated self-learning.
Public notebooks: fixed example epochs and ordinary val_loss checkpoints; HED saves
every epoch. No manual-mask reader was found in these QC/checkpoint paths; undisclosed
private-data selection is UNKNOWN. Aggregate clinical priors and visual tuning (~20
images) are supervision and must be disclosed. CMR in this study is a measurement
comparator, not proof of label-free MRI segmentation.

**SmooSeg:** `1eecdec75247ce448574a1bbe2dc9f693be99712`.
[`modules.py`:6–63](https://github.com/mc-lan/SmooSeg/blob/1eecdec75247ce448574a1bbe2dc9f693be99712/modules.py#L6)
freezes and downloads DINO. [`train_segmentation.py`:264–283](https://github.com/mc-lan/SmooSeg/blob/1eecdec75247ce448574a1bbe2dc9f693be99712/train_segmentation.py#L264)
monitors `test/cluster/mIoU` on the supplied validation loader, then evaluates `best`.
The metric name alone is not proof that test patients tune the model; the actual
callback does establish label-informed validation selection. Training_step consumes
images only, although dataset wrappers read/return labels. Paper-to-released-checkpoint
lineage remains unknown.

**UnSegMedGAT:** `7f5a0e53bb11da7608c07af4f364140c3d6366cf`.
[`segment.py`:98–135](https://github.com/mudit-adityaja/UnSegMedGAT/blob/7f5a0e53bb11da7608c07af4f364140c3d6366cf/segment.py#L98)
calls `cts(image,true_mask)` before feature extraction. The notebook's `cts` finds
the foreground bounding box from `segmap==255`; this is not merely evaluation.
[`segment.py`:198–204](https://github.com/mudit-adityaja/UnSegMedGAT/blob/7f5a0e53bb11da7608c07af4f364140c3d6366cf/segment.py#L198)
chooses output versus complement by GT IoU. The notebook's 5 collective + 55 per-image
steps differ from paper settings. Stride-4 DINO at 224 gives 3025 nodes; one dense
float32 graph is ~36.6 MB (derived, not measured). These are compatibility concerns,
not accusations about unobserved experiments.

**DiffSegLung:** [author-name-associated repository](https://github.com/noureddinekhiati/DiffLungSeg)
at `eef2226f231ef11fe906ff41c459b9f89f7f20f4`; the inspected paper has no explicit
repository link, so exact author-release linkage is not fully established.
[`inference_volume_distill.py`:112–139](https://github.com/noureddinekhiati/DiffLungSeg/blob/eef2226f231ef11fe906ff41c459b9f89f7f20f4/inference_volume_distill.py#L112)
comments say 50 steps, consumes a lung mask, and calls an imported sampler whose
`diffusion_model/` directory is absent in this checkout. Comments do not verify actual
solver steps. GPU radiomics returns 25 dimensions while paper/README describes 34;
CPU and GPU paths may differ. README requires CT plus a lung mask; its source is
unknown. Paper §4.5 says no-warmup distillation regresses below HU-only, whereas Table
3 reports 77.6 versus 74.8. Treat effectiveness/reproducibility as unverified; do not
quote its runtime/score as evidence for our MRI pipeline.

## Other required families and what they do not solve

[Kim and Ye, Mumford–Shah loss, §II/III](https://arxiv.org/html/1904.02872) already
replace hard region indicators with network softmax and optimize regional fit and
boundary regularity; labels can be absent in its unsupervised variant. Our direct
control belongs to this family, not a new method. [W-Net, §3/4](https://arxiv.org/html/1711.08506)
already amortizes soft normalized-cut segmentation plus reconstruction, with CRF
and hierarchical merging. Neither objective alone identifies cardiac class names.

[RIM, equations 6–8](https://arxiv.org/html/1706.04008) uses current estimate,
likelihood gradient and GRU state, and trains losses across rollout steps against
known reconstruction targets. Memory can track optimization curvature/progression;
it does not manufacture anatomical semantics. [Learning to learn by gradient descent
by gradient descent](https://arxiv.org/abs/1606.04474) is related learned optimization,
not evidence that updating all segmentation weights per test image is efficient.
Teacher process/trajectory distillation transfers a teacher's update law; it is only
worth testing here after independent evidence of useful teacher corrections.

[ELR, Appendix F](https://arxiv.org/html/2007.00151) combines noisy-label cross entropy
with a regularizer toward an EMA prediction history indexed by training example.
This is history across optimization epochs, not recurrent annotation within an image.
[ADELE, §2 and Appendix B](https://arxiv.org/html/2110.03740) observes noisy-target IoU
deceleration per category, then corrects confident labels with multi-scale predictions.
Its medical experiments corrupt manual annotations; it does not establish semantic
discovery from scratch. Both motivate a cheap EMA control if C2 is ever admitted.

Current cardiac searches also found [PDFMSeg, 2026](https://doi.org/10.1016/j.neunet.2025.108329)
(sparse annotated training) and [2026 cross-sequence myocardial adaptation](https://doi.org/10.1109/JBHI.2025.3649765)
(labeled source domain). They are excluded from strict no-GT comparators. This is a
targeted search through the run date, not a proof no qualifying MRI method exists.

## Novelty decision

Clustering + one student is established amortization, including this repository's
existing maskfree students. Direct unsupervised region loss is established. Adding
RNN history to a teacher does not resolve a missing semantic target. **No novelty
claim is supported for the baseline implemented here.** A later contribution would
need to solve and independently validate image-only anatomical correspondence and
coverage at bounded total cost, while surviving simple classical and fixed-teacher
controls. Existing names and modules do not provide that evidence.
