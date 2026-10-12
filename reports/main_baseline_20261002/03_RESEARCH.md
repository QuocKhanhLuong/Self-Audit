# Research: strict no-GT cardiac segmentation, diffusion, and a small deployed model

Date checked: **2026-10-02**. Scope: ACDC/M&Ms, current Self-Audit pseudo-label v3, and fast/lightweight deployment. Status: **SOURCE-REVIEWED / PROPOSED; no training or performance measurement in this task**.

## Decision

**Do not promise Dice >=0.90 for the present strict image-only pipeline.** None of the nine primary research records checked below establishes patient-disjoint, named RV/MYO/LV segmentation at that target on ACDC and M&Ms under the user's full no-GT restriction. This is a bounded evidence finding, not an impossibility theorem.

The defensible next investigation is to establish whether independently generated anatomical seeds have useful class identity and coverage, then test a small one-pass student against simple, matched controls. Keep expensive temporal processing or graph operations offline if they earn their cost. Generative diffusion is not presently justified as the default teacher or deployed segmenter.

The coordinator relayed the user's clarification **“Giữ hoàn toàn no-GT”** during this review. Consequently, mask-supervised, limited-GT, scribble, few-shot-mask, and mask-trained-teacher methods below are **inadmissible comparators**, not authorized fallback implementations. Their successes establish the importance of supervision, not an attainable score for this system.

## Source and checkout provenance

| Item | Observed state and implication |
|---|---|
| Open worktree | `/Users/alvinluong/orca/workspaces/Self-Audit/astra-pseudolabel-v3-audit`, HEAD `234be0a9cc9d557d1263025f163ef28792da4423`; this older checkout is not current main. |
| Current-main audit target | `origin/main = c31825a7f90df47f9f9382a9c9595e43f7f71236`, following the coordinator's fetch; independently verified with `git rev-parse`. |
| Earlier tracking ref | `0605e3b1af56156c8bd5f5a6147b8e00dbfdf881` was inspected initially and superseded after fetch. Relevant changes to v3 and its runbook were inspected. |
| Source access | Semble located the v3 and CUTS paths first. Exact main files were read using `git show <SHA>:<path>`. The current main blobs for `system_v3.py`, `evidence.py`, `trainer_v3.py`, and `losses_v3.py` were verified identical to the initially read versions. |
| Ownership | Only this report is written. No code deletion, git mutation, model training, environment installation, dataset loading, or local quality/latency benchmark was performed. |

Current main must not be described using the old “classes only, no trainer” assessment:

- **IMPLEMENTED, source checked:** `src/self_audit_pseudolabel/system_v3.py` has separate offline teacher and deployment student paths, explicit UNKNOWN abstention without evidence, and validity-filtered student CE. `evidence.py:39` builds automatic class proposals using border/motion, enclosure, adjacency, and orientation rules. `trainer_v3.py:27` uses displacement fields for evidence; raw seeds supply the class identity and prototype history may not invent or rename it. `losses_v3.py:30` applies seed loss to raw neural probabilities.
- **IMPLEMENTED, source checked:** `pipeline_v3.py` has patient-split teacher training, freeze/export, and a frozen-pseudo-label student consumer. The latest-main delta adds logging and `scripts/run_full_pipeline_v3.py`, which runs teacher, frozen evaluation, and optional student training. `--train-student` is an opt-in flag; the script does not implement a Dice acceptance threshold or evaluate final student accuracy.
- **UNKNOWN scientifically:** anatomical seed correctness, deployed student quality, M&Ms transfer, and target-device speed. The current [v3 runbook](../../docs/pseudolabel_v3_review_runbook.md) explicitly distinguishes software integrity from real-data validation and discloses historical development-label feedback. Nonzero foreground seeds are not a semantic quality certificate.
- **Protocol distinction, source checked:** current main's `baseline/CUTS/ACDC_ORIGINAL_STYLE_PROTOCOL.md` separates its label-backed original-style loader from an image-only variant. Both retain a slice-level split and are explicitly not patient-disjoint evidence; image-only mode alone cannot repair that. Its persistent export produces an anonymous partition, while original GT-guided/best-level metrics are diagnostic oracles. Use the separate shared-benchmark patient manifest for an admissible comparison.
- **Scope distinction:** `reports/ACTIVE_MILESTONE_SCOPE.md` retires DFC/STEGO/PiCIE from the active milestone while preserving historical contracts; DSS-US/SGSCN native paper protocols have unresolved gates. This review does not turn their names, wrappers, or historical records into validated cardiac baselines, and does not authorize deleting dependencies.

Relative repository links above are navigation aids; source statements in this report refer to the pinned main SHA, not whichever branch a reader happens to have open.

## What “diffusion” means here

| Mechanism | Operation | Source of semantic identity | Relevance to a fast student |
|---|---|---|---|
| Generative/denoising diffusion | Learn to reverse progressive noise corruption; sample a clean output through a sequence of denoising network calls. | Image generation learns image statistics. Conditional mask generation additionally needs masks or another explicit source of mask targets. | An offline teacher is possible, but its target provenance and measured benefit must justify training/sampling cost. |
| Graph/label diffusion | Spread a fixed set of seed-class scores through an affinity graph, usually with seed anchoring. | Named seed columns already encode the classes; propagation does not define their meaning. | Potentially useful offline refinement; can be removed from deployed inference after verified distillation. |
| Diffusion condensation, as in CUTS | Repeatedly smooth/condense embeddings and merge nearby points to produce a hierarchy of partitions. | Produces anonymous groups; a separate rule is needed to identify RV, MYO, LV and background. | A source of candidate regions, not automatically a named segmentation teacher. |

For a denoising model, a standard forward corruption is

\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

The denoiser learns to predict noise or an equivalent clean-data quantity. In image-conditioned mask diffusion, `x0` is a mask and the input image conditions that denoiser. Image-only reconstruction of MRI does not by itself supply the missing named masks. These mechanisms follow [DDPM](https://arxiv.org/pdf/2006.11239) and the mask-conditioning formulation in [MedSegDiff, Section 2](https://proceedings.mlr.press/v227/wu24a/wu24a.pdf).

A classical label-propagation example uses an affinity matrix `W`, degree matrix `D`, normalized affinity `S=D^(-1/2) W D^(-1/2)`, and seed matrix `Y`:

\[
F^{(k+1)}=\alpha SF^{(k)}+(1-\alpha)Y, \quad 0<\alpha<1.
\]

The seed term resists smoothing away the initial labels. With zero initialization and no seeds, this iteration produces zero scores; with incorrect named seeds, it can spread their mistakes. This equation is from [Learning with Local and Global Consistency](https://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency.pdf), not a claim that current v3 implements that exact algorithm.

**Inference about v3:** its inspected teacher is a reconstruction/registration/prototype/heuristic-evidence system, not a DDPM noise-schedule sampler. Current CUTS code uses diffusion condensation in a separate baseline. Neither should be described as “generative diffusion” merely because an iterative operation or the word diffusion appears elsewhere in the repository.

**Identifiability falsifier:** anonymous clusters can be renamed without changing reconstruction or cluster structure. Named Dice changes under that renaming. A usable method must therefore explain where its named cardiac identity comes from. V3 now has explicit anatomical heuristics, so the question is their validity across anatomy and acquisition geometry, rather than the historical absence of any semantic anchor. Consistency, low entropy, or prototype stability cannot independently certify those heuristics.

## Nine directly checked primary research records

All records were accessed on **2026-10-02**. A record groups a paper with its official dataset/project implementation where relevant; bibliographic dates below are publication/version dates, not search-engine crawl dates. This is a focused review, not an exhaustive state-of-the-art survey. Paper scores are **LITERATURE-REPORTED**, never local measurements.

| ID | Primary source and date | Checked protocol/evidence | Limitation for this task |
|---|---|---|---|
| R1 | Bernard et al., **ACDC**, TMI **2018**: [organizer-hosted paper](https://www.creatis.insa-lyon.fr/Challenge/acdc/files/tmi_2018_bernard.pdf), [official dataset description](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). | Paper Section III describes 100 training and 50 testing patients with ED/ES reference volumes; target structures are RV, myocardium, and LV. The organizer page describes the full cine acquisition. | These are supervised benchmark/reference resources. The common 70/10/20 research split of the public training cohort is not the original 100/50 challenge protocol. Full-cine input is not equivalent to ED/ES-only input. |
| R2 | Campello et al., **M&Ms**, TMI **2021**, DOI `10.1109/TMI.2021.3090082`: [institutional full paper](https://ddd.uab.cat/pub/artpub/2021/270221/270221.pdf). | Sections II.B–C: training contains 75 labeled A, 75 labeled B, and 25 unlabeled C cases; validation contains 40 cases. Total release has 375 studies across four vendors and six hospitals. Original challenge disallows external data/pretrained models and weights unseen/unlabeled vendor performance more heavily. | “Unlabeled vendor” is not fully no-GT: source-vendor masks train the models. ACDC success cannot establish M&Ms transfer. Keep original challenge rules distinct from any new cross-dataset protocol. |
| R3 | Liu et al., **CUTS**, MICCAI **2024**; arXiv v7 **2024-06-25**: [conference PDF](https://papers.miccai.org/miccai-2024/paper/1569_paper.pdf), [version record](https://arxiv.org/abs/2209.11359), [official code](https://github.com/ChenLiu-1996/CUTS), [foreground-hint function](https://github.com/ChenLiu-1996/CUTS/blob/main/src/utils/segmentation.py). | Paper Sections 3–4: image-only contrastive/reconstruction embeddings followed by multigranular condensation; experiments cover fundus and brain MRI. Page 5 uses GT foreground pixels to select the binary foreground cluster; “diffusion-B” selects a best granularity. The official `label_hint_seg` accepts `label_true`. | Representation learning is image-only, but that semantic readout is inadmissible as a no-GT runtime rule. Persistent anonymous partitions remain useful controls; neither the paper nor that readout proves named ACDC/M&Ms performance. |
| R4 | Zhou et al., **Learning with Local and Global Consistency**, NIPS **2003**, proceedings volume 16: [paper](https://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency.pdf). | Algorithm in Section 2 combines normalized graph affinity with an initial label matrix, using the anchored iteration above. Demonstrations are semi-supervised classification, not cardiac MRI. | This is direct prior art for seeded graph propagation. It neither generates correct named seeds nor establishes cardiac accuracy or a deployment-speed result. |
| R5 | Ho, Jain, Abbeel, **Denoising Diffusion Probabilistic Models**, NeurIPS **2020**: [paper](https://arxiv.org/pdf/2006.11239), [version record](https://arxiv.org/abs/2006.11239). | Equations 4/11–12 and Algorithms 1–2 define noise corruption, denoiser training, and iterative sampling for image generation. | Primary source for the generative meaning of diffusion. It is not a no-GT cardiac segmentation study; image synthesis quality is not named anatomical Dice. |
| R6 | Wu et al., **MedSegDiff**, MIDL **2023**, PMLR volume 227 published **2024**: [proceedings record](https://proceedings.mlr.press/v227/wu24a.html), [full paper](https://proceedings.mlr.press/v227/wu24a/wu24a.pdf), [official repository](https://github.com/ImprintLab/MedSegDiff). | Section 2 starts from a segmentation label and learns image-conditioned denoising. Evaluations cover optic cup, brain tumor, and thyroid ultrasound. Official sampling examples use `diffusion_steps=1000` and `num_ensemble=5`; the README also documents a shorter solver setting. | Mask-target training is inadmissible here; these experiments do not establish ACDC/M&Ms performance. README settings are not measured latency or proof that a shorter sampler preserves quality. |
| R7 | Isensee et al., **nnU-Net**, Nature Methods **2021**: [publisher paper record/abstract](https://www.nature.com/articles/s41592-020-01008-z), [official repository](https://github.com/MIC-DKFZ/nnUNet). | Automatically configures preprocessing, architecture, training, and postprocessing; official code supports 2D and 3D workflows. Current v2 is a reimplementation, so an experiment must pin its version. | Strong supervised pipeline comparator, not a mask-free result. Full publisher text was subscription-limited in this access; no cardiac score or timing is inferred from its abstract. A small pseudo-label-trained U-Net is a different experimental condition. |
| R8 | Bai et al., **Bidirectional Copy-Paste (BCP)**, CVPR **2023**: [full paper](https://arxiv.org/pdf/2305.00673), [official repository](https://github.com/DeepMed-Lab-ECNU/BCP). | Sections 4.1/4.3 and Table 3: ACDC patient split 70/10/20, 2D U-Net, 256-square training slices. Paper reports Dice **87.59%** for 3 labeled + 67 unlabeled training cases and **88.84%** for 7 + 63; its fully supervised 70-case U-Net row reports **91.44%**. Test inference uses the student alone. | These low-label settings do not reach the requested target, and none is no-GT. Its table is protocol-specific, not a forecast for v3 or M&Ms. Paper describes surface distances in voxels, which must not be silently compared with millimetres. |
| R9 | Rahman et al., **EMCAD**, CVPR **2024**; arXiv v1 **2024-05-11**: [full paper and supplement](https://arxiv.org/html/2405.06880v1), [official repository](https://github.com/SLDGroup/EMCAD). | Table 3 reports ACDC PVT-EMCAD-B0 mean Dice **91.34%**: RV **89.37**, MYO **88.99**, LV **95.65**. Supplement uses 70/10/20 cases; training uses labeled masks and ImageNet-pretrained encoders. Table 1 lists full B0 at **3.92M parameters / 0.84G FLOPs**, with FLOPs specified at **256×256** in that table. | These complexity figures accompany the binary-task table, not a measured ACDC latency. The smaller decoder-only count must not be presented as full-model size. Mean >=0.90 does not mean every class >=0.90. Current official README redirects ACDC training/inference to CASCADE code; paper reproducibility needs that integration verified. |

Access notes: the web fetcher rejected the CVF BCP/EMCAD PDFs, so author arXiv versions were inspected. The CUTS conference PDF exceeded the web fetcher's size limit; it was downloaded into process memory and pages 1, 3–7 were text-inspected directly. The official CUTS GitHub function was separately read. A remote Semble index attempt timed out; no remote source-check claim depends on that failed index. No paper result was reproduced locally.

## Feasibility by supervision contract

| Contract | Status of Dice >=0.90 | Decision |
|---|---|---|
| **Strict image-only/no-GT**, including automatic anatomy priors | **UNKNOWN and unsupported for present v3.** The reviewed no-GT representation evidence does not close the named-semantic, patient-split, or cross-vendor gaps. | Active research track. First diagnose raw seed identity and dense output failure; do not start a long run based on reconstruction loss or confidence alone. |
| Limited labeled patients + unlabeled patients | R8 demonstrates useful learning but its quoted low-label settings miss the target. Other budgets/methods might succeed; this review supplies no guarantee. | **Inadmissible comparator under the confirmed user instruction.** Do not add scribbles or few-shot masks as a quiet repair. |
| Fully supervised one-pass model | R8/R9 establish that protocol-specific mean ACDC Dice above the target is possible without a generative sampler. Local reproduction, M&Ms transfer, and hardware speed remain unmeasured. | **Inadmissible comparator**, useful for interpreting the supervision gap; not a recommended change of task. |
| A teacher pretrained on segmentation masks, including external masks | A nominally “zero-shot” target-dataset evaluation still inherits mask supervision. | **Inadmissible.** A small student distilled from it does not restore no-GT status. |

GT may be opened only by an independent evaluator after the producer recipe, weights, mapping, thresholds, and output manifest are frozen. A scalar validation Dice used to choose an epoch, seed rule, teacher, crop, diffusion level, model, or stopping point is still GT feedback. Historical development feedback already disclosed in the main runbook cannot be erased by a new directory or a later freeze.

Use scratch initialization as the clean default for new image-only candidates. Any externally pretrained image-only encoder requires a provenance review for label exposure, source cohort overlap, and the exact pretraining objective; “self-supervised” in a model name is insufficient. Anatomical rules are explicit human priors and must be disclosed. They are not automatically manual segmentation annotations, but fitting them to validation masks would violate this contract.

## Prioritized admissible research path

All rows are **PROPOSED / NOT RUN**. They are a staged investigation, not authorization for long training.

| Priority | Candidate and purpose | First reason to reject or stop |
|---|---|---|
| 0 | **Input/metric audit.** Fix patient manifests, native axes, spatial spacing, label ontology, phase metadata, and full-cine availability. Freeze an untouched evaluation cohort and the producer's admissible inputs. | Spatial neighbours are passed as temporal frames; release-specific label maps are assumed; image enumeration depends on reference masks; evaluation leaks patients or drops failed cases. |
| 1 | **Simple fixed image-only seed baseline.** Compare the current raw heuristic evidence with intensity/edge/connected-component proposals plus explicitly declared anatomical rules. Export region IDs, named seeds, UNKNOWN, and support statistics. No learned student is needed to expose absent/incorrect class identity. | No foreground/class support, unstable class naming under valid coordinate transformations, or confidently wrong seeds in the frozen independent audit. Do not weaken thresholds to manufacture training data. |
| 2 | **Small one-pass segmentation student.** Use the same frozen training pseudo-labels for current compact A0 and a narrow 2D U-Net with skip connections; then a spatial 2.5-D variant if input provenance supports it. | Added capacity fits pseudo-label errors or loses thin-MYO/basal/apical detail. Current A0 is a low-resolution encoder plus upsampled head; a tiny decoder is a plausible information-preservation control, not a proven improvement. |
| 3 | **Offline regularization only after seed evidence exists.** Compare no propagation, a fixed local sparse graph, and motion-warp rejection. Keep seed classes anchored; do not fill unsupported classes just to improve coverage. | More smoothness/coverage without better frozen semantic outcomes, propagation across blood/myocardium edges, erroneous flow, or excessive whole-volume generation cost. |
| 4 | **Image-only representation learning if needed.** Compare a random/fixed encoder with reconstruction or contrastive pretraining, and CUTS-style persistent anonymous partitions with the same GT-free anatomical mapper. | Better clustering only after GT matching or choosing the best level; unchanged class identity; gains vanish when teacher seeds and compute are matched. |
| 5 | **Compression/refinement only after a useful dense baseline.** Try narrower widths, depthwise separable blocks, an exported compact-only network, and separately optional Dynamic Window profiles. | The model falls outside the quality/resource budget, or refined profiles fail to improve the paired outcome at their measured cost. |

The recommendation for a small U-Net is an engineering hypothesis supported by the value of simple one-pass comparators, not a transfer of supervised paper performance. Replacing ordinary convolutions with depthwise blocks can change actual hardware utilization; fewer operations are not a latency measurement.

Keep the teacher/student input contract explicit. Offline `prev/cur/nxt` refers to temporal context while each three-channel image can represent spatial slice context. A student intended for a single ED/ES volume must not require unavailable cine frames at deployment. Motion can enrich offline supervision, but a student's static image may not contain everything the teacher used. If raw cine is missing, report that dependency and evaluate an explicitly static variant; do not relabel z-neighbours as time.

## Teacher quality does not transfer by declaration

An admissible offline teacher may use full-cine registration, graphs, or a larger encoder trained only on admissible images and priors. The deployed student can be a single encoder/decoder pass. This separates **where compute is paid**, not **where semantic supervision comes from**.

Measure three different quantities after freezing the complete pipeline: teacher versus reference, student versus teacher, and student versus reference. High student/teacher agreement can merely copy a systematic mistake. Sparse accepted seeds can have high precision but leave much of the myocardium or RV unsupervised; teacher partial-label quality is not dense-output quality. Conversely, regularization could let a student outperform noisy labels, but that is an experimental result, not an assumption or a general bound.

Train the student only from training-patient frozen outputs and train-only statistics. Preserve UNKNOWN validity in every loss; never convert missing labels into background. Initially compare hard-label CE with a separately preregistered validity-masked soft-target variant. Treat teacher soft scores as uncalibrated unless independent evidence supports calibration. Distillation must not silently change seeds, thresholds, or teacher generation between student architectures.

For the existing `compact` profile, verify both executed operations and shipped parameters. Skipping a refiner's forward pass does not remove its registered weights from a saved model. Report full checkpoint size and the actually exported compact artifact separately; do not describe dormant weights as a deployment saving until an export has been checked.

## Matched controls and falsifiers

Freeze candidate definitions before reference access. The following are experimental comparisons, not a search over held-out GT to choose the winning no-GT model.

| Question | Matched controls | Falsifying observation |
|---|---|---|
| Do anatomical seeds supply real identity? | Raw seed generator; absent-evidence/all-UNKNOWN negative control; a fixed class-permutation diagnostic; image-derived connected-component baseline. Keep images and ontology fixed. | Confident semantic output exists without evidence, or naming is driven by class IDs/order instead of image/geometry evidence. |
| Is a graph useful? | Same seeds and embeddings with identity/no propagation, fixed local smoothing, and proposed graph; match stopping and graph budget. | Gain exists only after reference-guided level selection, or only by raising coverage while propagating wrong labels. |
| Does temporal context help? | Identical teacher with correct temporal neighbours, repeated center frame, and deliberately scrambled temporal order; test learned flow versus identity warp. | Shuffled time is equally good, or improved agreement conceals anatomy errors. Scrambled data are diagnostic controls, not valid training recipes. |
| Does the learned representation matter? | Same mapper and seed budget with raw image features, frozen random features, and image-only learned features. | Anonymous partition metrics improve but named Dice does not; only GT-based remapping produces a gain. |
| Does the student architecture matter? | Identical frozen pseudo-label manifest, patients, preprocessing, augmentations, optimizer-step budget, and random-seed set for A0, narrow U-Net, and spatial 2.5-D variant. | Claimed architecture gain vanishes after matching label quality or is paid for by undisclosed extra training data/teacher work. |
| Is Dynamic Window worth deploying? | Same trained encoder/A0 where protocol permits; compact, fixed refinement, and the proposed selector, with exact executed pass counts. | Average or tail cost rises without a paired semantic benefit; entropy is treated as correctness without calibration. |
| Are priors robust? | Coordinate transforms with correspondingly updated affine/orientation; plausible intensity perturbations; vendor/pathology and basal/mid/apical strata. | Correctly transformed inputs change anatomical identities, or the method succeeds only where a ring/left-right heuristic happens to hold. |

A “best level using GT,” per-image Hungarian remapping, mask-derived crop, or oracle seed is allowed only in a separately labeled **inadmissible diagnostic upper bound**. It cannot supply predictions, training targets, stopping decisions, or thresholds to the primary track. If an independent frozen evaluation refutes the method, reject its claim. Do not tune on that cohort and continue calling it untouched; any later study must disclose the feedback history and reserve genuinely unseen evaluation data.

## Patient splits and the exact Dice claim

**Proposed primary endpoint:** mean foreground Dice across the named RV/MYO/LV classes, computed on reconstructed native 3D ED and ES volumes, averaged within patient and then equally across patients. Exclude background from the headline mean. Write this aggregation into the evaluator contract before opening masks. The requested `>=0.90` is a goal, not an observed result or a per-class guarantee.

For class `c`, use `Dice_c = 2|P_c intersect G_c| / (|P_c| + |G_c|)`. Report each class and phase, patient distribution, median and lower tail, plus patient-bootstrap uncertainty. Do not bootstrap slices as independent subjects. Predeclare treatment of absent classes, missing outputs and empty predictions; record empty cases separately so background-only performance cannot inflate the result. A mean above the target must not be described as every patient or every class passing.

- **ACDC:** preserve patient identity across all slices, ED/ES and cine frames. If using a local split of the public training patients, name the manifest/hash and call it a local research split. Use the organizer's original split only if that is the actual protocol; R1 and the 70/10/20 paper experiments are not interchangeable. Existing development patients exposed in earlier project work are development, even if a new script calls them test.
- **M&Ms:** name release, vendor/site and patient/scan identities. Keep repeated scans from one patient together. Report within-vendor and held-out-vendor outcomes separately. An ACDC-trained model evaluated on untouched M&Ms is a cross-dataset test; adapting to M&Ms images first creates a distinct image-only adaptation condition. In the strict no-GT condition, M&Ms source masks also remain unavailable to training. Do not compare its raw mean directly to the original challenge's weighted ranking.
- **Preprocessing:** fit normalization/cropping decisions from training images only; no mask bounding boxes. Preserve the forward/inverse transform and affine/spacing. Score native volumes after inversion rather than treating a resized canvas as clinical geometry. Audit per-release semantic IDs and ED/ES indices explicitly.
- **Abstention:** report dense named Dice with UNKNOWN never counted as a correct named prediction, plus known fraction, per-class seed precision, foreground recall/support, and risk-versus-coverage as separate diagnostics. Do not score only accepted pixels for the headline dense target. An all-UNKNOWN output must not look like a successful segmenter.
- **Other outcomes:** HD95 and average symmetric surface distance in millimetres using native spacing, with missing-surface handling declared; volume bias and severe anatomical failures; per-pathology/site performance where metadata permits. Surface-distance units from R8 are not already physical millimetres.

The primary producer should be chosen by the preregistered research/compute hypothesis and image-only development rules, not by ranking models on evaluation Dice. Report all preregistered candidates and failures. Any supervised or oracle control stays in its separate diagnostic table and cannot justify selecting the admissible model.

## Speed, memory, and parameter measurement contract

No local values are claimed. The user-selected deployment target, relayed by the coordinator, is **NVIDIA RTX 4080 Super 16GB**. Actual target-device measurements, maximum acceptable latency, usable memory headroom, and whether the requirement is per slice or per volume remain **UNKNOWN**. The coordinator reports that this local Mac has no CUDA; any separately produced untrained CPU microbenchmark is a software/runtime envelope, not 4080 Super performance. Report a quality/resource curve rather than calling a network “real-time” before a latency requirement is fixed.

**PROPOSED initial engineering gate, not a user-specified budget or measured result:** preregister a one-pass student with at most **5 million shipped parameters**, **p95 network-only latency <=25 ms** for batch-1 `[1,3,256,256]` FP32 inference on that GPU, and **peak allocated inference memory <=2 GiB** including model/input/output residency. These are deliberately explicit shortlist criteria chosen for this investigation; no cited paper proves the conjunction with no-GT Dice >=0.90. Freeze these criteria before benchmarking; report failures rather than relaxing them after seeing results. Report reserved memory as well. This slice-level proposal does not set an end-to-end native-volume target, which still needs a representative volume-size and I/O workload. A different spatial size or reduced precision is a separate condition, not a silent replacement for this gate.

| Quantity | Required measurement |
|---|---|
| Deployed latency | Use the actual saved/exported student on the same RTX 4080 Super, with fixed backend/precision and other device load recorded. Report cold start separately, then warm synchronized p50/p95 latency, variability, and sample count. Start with batch 1; report throughput at other batches separately. Include slice dimensions and slices per volume. |
| Scope of timing | Separate network-only forward from end-to-end loading, normalization/resampling, CPU/GPU transfer, postprocessing, native-volume reconstruction, and output writing. Include ROI localization if needed. Distinguish device-event timing from synchronized wall-clock timing. |
| Executed work | Record encoder evaluations, windows, refinement passes, graph construction, denoising steps, samples/ensembles, test-time augmentation, and postprocessing. A one-pass label refers to execution, not the number of objects in a Python model. |
| Parameter/storage size | Count all and trainable parameters, full checkpoint bytes, and exported inference artifact bytes; include encoder, decoder, heads and any retained refiner. Decoder-only counts are not full-model counts. |
| Arithmetic | State input shape, batch, MAC/FLOP convention and profiler/operator coverage. Do not infer latency from FLOPs, especially for grid sampling, sparse graphs, depthwise kernels, and repeated small launches. |
| Memory | Reset peak counters before the defined scope; report allocated/reserved accelerator memory and host RSS separately, including model residency, inputs, working buffers and preprocessing. Report training optimizer/activation memory separately from inference. |
| Offline teacher cost | Report training wall time/device-hours, generation time per complete cine/volume, graph/flow memory, cache size, and failed/abstaining cases. A fast cached student does not make the original annotation-generation process cheap. |
| Reproducibility | Record code/config/checkpoint hashes, patient manifest and preprocessing identity, device, software versions, precision, thread settings, warmup/repetition policy and all fallback paths. Random-weight or synthetic timings remain software evidence, separate from the trained checkpoint's quality. |

Use the same input volumes and RTX 4080 Super when comparing A0, the small decoder, and optional refinement. Establish a declared reference precision, then evaluate any reduced-precision/export variant as its own paired quality/timing condition; do not assume numerical equivalence. Pair quality measurements with the exact checkpoint that was timed. For offline-versus-online cost, an illustrative accounting is `total_cost = teacher_training + frozen_label_generation + student_training + N * student_inference`; report the separate terms rather than inventing a break-even volume count.

## Completion and remaining evidence

**Completed:** latest-main-aware source review, nine directly checked primary research records, diffusion distinctions, strict no-GT admissibility analysis, falsifiers, and proposed split/metric/resource protocols. No model, checkpoint, or numerical performance result was produced.

**Still required before an accuracy/speed claim:** an admissible frozen seed/teacher output with provenance; patient-disjoint independent evaluation of named anatomy and coverage; evidence that a small dense student retains or improves useful quality; untouched M&Ms evaluation; and measurements on the intended deployment hardware. The immediate research decision is to test whether the automatic semantic evidence is useful at all, with a simple one-pass student as a later controlled experiment. Greater diffusion complexity and a target number cannot substitute for that evidence.
