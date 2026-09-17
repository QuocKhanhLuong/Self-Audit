# Union rf_prior — adversarial primary-paper audit: reference-free segmentation QC / RCA / self-supervised QC

Scope: adversarial audit of reference-free segmentation QC, reverse classification accuracy (RCA), self-supervised QC, and consistency/image-mask energies used to redirect attention. Previous SESV baseline (DOI 10.1109/TMI.2020.3025308) was NOT re-fetched full-text per coordinator instruction; no full text of it was read this dispatch.
Verification date: 2026-09-17/18 (this dispatch). All primary URLs below were accessed via live websearch; arXiv abstract/method text was retrieved and read. Word budget: report body kept under 1400 words per task.

## 1. Verified primary papers (5-7 closest)

P1. Valindria et al., "Reverse Classification Accuracy: Predicting Segmentation Performance in the Absence of Ground Truth," IEEE TMI 36(8):1597-1606, 2017. DOI 10.1109/TMI.2017.2665165; arXiv 1702.03407.
  - URL: https://doi.org/10.1109/tmi.2017.2665165 | https://ar5iv.labs.arxiv.org/html/1702.03407
  - Depth: abstract + method section (Sec. on RCA framework, Eq. for max-over-references proxy) read from ar5iv full text.
  - Method: train a reverse classifier on a single new image I using the predicted mask S_I as pseudo-GT; apply to a reference bank T of m images with GT; proxy score = max_k DSC on references.

P2. Cosarinsky, Billot, Mansilla, et al., "In-Context Reverse Classification Accuracy" / "Conformal In-Context RCA," arXiv 2503.04522 (v2 2025; v3 title adds "Conformal ... with Statistical Guarantees").
  - URL: https://arxiv.org/html/2503.04522v2 | https://arxiv.org/html/2503.04522v3 | code: https://github.com/mcosarinsky/Conformal-In-Context-RCA
  - Depth: abstract + method (Sec. II-B conformal; In-Context RCA Sec. on ICL/retrieval) read from arXiv HTML v2 and v3.
  - Method: replace trained reverse classifier with in-context segmenters (UniverSeg, SAM2/MedSAM2); retrieval-augmented (DINOv2/RAD-DINO + FAISS) reference selection; conformal prediction intervals over RCA scores. ~0.37-0.7 s vs ~1 min for atlas reverse classifier; SAM2 corr 0.74 on NuCLS.

P3. Robinson et al., "Real-Time Prediction of Segmentation Quality," MICCAI 2018, LNCS, pp. 578-585. DOI 10.1007/978-3-030-00937-3_66; arXiv 1806.06244.
  - URL: https://arxiv.org/pdf/1806.06244 | https://cris.fau.de/publications/262843323/ (DOI listing)
  - Depth: abstract + Method section (both experiments) read from arXiv PDF text.
  - Method: CNN predicts per-case DSC from image+segmentation pair. Exp 1: trained on 12,880 samples with GT DSC labels, MAE 0.03, 97% binary acc. Exp 2: no manual annotations; DSC labels obtained via reverse testing strategy (RCA-style), MAE 0.14, 91% acc; real-time (~<100 ms) enabling in-scanner feedback and "optimising image acquisition".

P4. SegAE, "Large-scale Label Assessment for Medical Segmentation," arXiv 2601.14406 (Jan 2026 preprint).
  - URL: https://arxiv.org/pdf/2601.14406
  - Depth: abstract + method (framework, loss Eq. 1, data synthesis stages 1-2) read from arXiv PDF.
  - Method: VLM (BiomedCLIP encoders) judges 2D image-mask pairs + class text; trained on 4M+ synthetic image-mask-DSC triples derived from intermediate checkpoints of a segmenter trained on DAP Atlas (142 structures); MSE + optimal-pair ranking loss. r=0.902 with GT DSC; 0.06 s per 3D mask; used to select active/semi-supervised samples and to audit public datasets.

P5. AutoQ-VIS, "Improving Unsupervised Video Instance Segmentation via Automatic Quality Assessment," arXiv 2508.19808 (2025 preprint).
  - URL: https://arxiv.org/html/2508.19808
  - Depth: abstract + method (quality predictor design, Eq. for Q_l, self-training rounds, DropLoss, threshold ablation) read from arXiv HTML.
  - Method: Mask-Scoring-R-CNN-style predictor of mask IoU; scores pseudo-labels in a self-training loop; fixed threshold filters which pseudo-labels enter training; closed-loop QC -> training-set redirection.

P6. Gaggion et al. (cited as [4] inside P2), "Unsupervised bias discovery in medical image segmentation," CIP Workshop (Springer), 2023 — verified only as citation inside P2's reference list and text (cited for segmentation-bias detection without GT); no primary full text read this dispatch. Included as the reference-free mask-competition/consistency line, not independently verified beyond P2's description. UNKNOWN: exact mechanism.

P7. RS-SQA, "Unsupervised quality assessment for remote sensing semantic segmentation based on VLM," arXiv 2502.13990 (Feb 2025).
  - URL: https://arxiv.org/pdf/2502.13990
  - Depth: abstract + method (dual-branch CLIP-RS + segmentation-feature fusion, quality head) read from arXiv PDF (partial; text layer noisy).
  - Method: dual-branch (VLM semantic features + segmenter intermediate features) predicts overall accuracy without labels; also recommends best model per image. Trained on RS-SQED dataset with OA labels from 8 segmentation methods.

## 2. Required comparison columns (best-effort per paper; "no"/"unknown" where the paper does not address it)

| # | Method | Labels used to train QC | Runtime reference bank / GT need | Intrinsic score | Can compare natural transitions | Modifies evidence acquisition | Predicts intervention value | Convergence claim |
|---|---|---|---|---|---|---|---|---|
| P1 | RCA (reverse classifier, atlas/forest/CNN) | none for the QC step itself (predicted mask is pseudo-GT) | YES: m reference images WITH GT required at runtime; score = max DSC over bank | max DSC over reference bank | no (per-image score only) | no | no | no; empirical correlation only |
| P2 | In-Context / Conformal RCA | none (frozen in-context segmenters; no QC training) | YES: labeled reference bank, retrieval-reduced but still GT-labeled | DSC point estimate + conformal interval | no | retrieval selects bank subset (evidence acquisition of references, not of actor reads) | no | conformal coverage guarantee under exchangeability; explicitly degrades under domain shift |
| P3 | Real-time QC CNN (Robinson 2018) | Exp1: GT DSC labels (12,880 samples); Exp2: RCA-derived DSC labels (no manual GT) | none at inference; Exp2 needs RCA machinery only at training-data generation | predicted DSC | no | no | no | no |
| P4 | SegAE | synthetic DSC labels vs GT masks from segmenter checkpoints | none at inference; GT used to synthesize training labels | predicted DSC (+rank loss) | no | no | indirectly: selects samples for active/semi-supervised training (resource allocation, not intervention value) | no |
| P5 | AutoQ-VIS quality predictor | threshold-binarized predicted-IoU vs matched GT inside synthetic video rounds | none at inference; GT used in synthetic pretraining | predicted IoU x confidence | no | yes but coarse: filters which pseudo-labels join the training set | implicitly: quality gate = expected contribution of a pseudo-label, not measured causal delta | no; empirical SOTA only |
| P6 | Unsupervised bias discovery (Gaggion 2023) | unknown (not read) | unknown | unknown | unknown | unknown | unknown | unknown |
| P7 | RS-SQA | OA labels from 8 methods on RS-SQED | none at inference | predicted OA | no | no | no | no |

## 3. Adversarial challenges applied

- Identifiability: P1/P2/P3 all predict DSC-like overlap with a bank; none can distinguish "mask consistent with anatomy" from "mask consistent with the reference population's bias." P2's own limitation section concedes exchangeability failure under domain shift (exact quote in my sources: "coverage can degrade"). None claims to identify semantic correctness intrinsically.
- Corruption ranking on wrong masks: P4 (SegAE) evaluates ranking of synthetic erosion/dilation corruptions (LCC 0.850/0.775 on BTCV) — i.e., it ranks corruptions of a given mask. No paper found that ranks corruptions of an image while the mask itself is wrong and reference-free. UNKNOWN in all 7.
- Shape/entropy shortcuts: none of the 7 addresses shape-prior or entropy shortcuts explicitly in the QC score; P4 mitigates with text-class conditioning but that is a prior, not an identifiability argument. No paper provides a counterfactual check that its score would not be minimized by a degenerate prior-shaped mask.
- Collusion: P5 is the closest to a collusion loop (QC predictor scores pseudo-labels produced by the same system it trains); it mitigates only via fixed thresholds and weight resets, with no theoretical guarantee. P1/P2 are structurally collusion-resistant only because the reference bank is externally labeled; they pay for that with the GT bank the ASTRA contract forbids.

## 4. Access log

- Read (abstract + method, primary): P1 ar5iv full HTML; P2 arXiv HTML v2+v3; P3 arXiv PDF; P4 arXiv PDF; P5 arXiv HTML; P7 arXiv PDF (partial).
- Citation-only (not independently verified): P6 via P2's reference list.
- Not attempted per coordinator instruction: SESV DOI 10.1109/TMI.2020.3025308 full text (IEEE paywall; not retried this dispatch).
- Failed accesses: none. All listed URLs returned content during this dispatch.
- IEEE Xplore page for P1 (7902121) was reached via search highlight but full text was read from the open ar5iv mirror instead.
