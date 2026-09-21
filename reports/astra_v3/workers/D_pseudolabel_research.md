# Astra Self-Audit v3: Primary-Source Pseudolabel Research Audit (Worker D2 Revision)

> **Root review status: investigation record, not accepted wholesale.** Read [root disagreement resolutions](../11_RED_TEAM.md) and the corresponding root report before using these claims. Worker completion indicates a delivered report, not scientific validation.

- **Audit Target Base SHA**: `d4f503b7107eca4e8ac85f37a5dd01ec50479666`
- **Orca Run ID**: `run_698eb3b460ef`
- **Task ID**: `task_f3b42a2bb8c9`
- **Dispatch ID**: `ctx_80f870229c9e`
- **Auditor**: AGY Worker D2 (Citation-Corrected Follow-Up)
- **Output Report**: `reports/astra_v3/workers/D_pseudolabel_research.md`
- **Evidence Directory**: `reports/astra_v3/workers/D_evidence/`
- **Prior Unverified Draft Preserved At**: `reports/astra_v3/workers/D_evidence/initial_report_unverified.md`

---

## 1. Executive Summary & Verification Findings

This revised audit evaluates primary literature and mathematical foundations for generating named cardiac pseudo-labels on short-axis cine CMR (ACDC and M&Ms) without manual segmentation masks in proposed training. Following root audit rejection of unverified assertions, all claims have been strictly reconciled against verified primary sources.

### Core Verified Findings:
1. **No Inspected Primary Source Demonstrates $\ge 0.88 - 0.90$ Named 3-Class Dice Without Manual Masks on ACDC/M&Ms**: Across all verified primary literature, we found **no supporting evidence** that an autonomous model trained without manual segmentation masks achieves named foreground Dice $\ge 0.88 - 0.90$ across Left Ventricle (LV), Myocardium (MYO), and Right Ventricle (RV) on ACDC or M&Ms.
2. **Deconstruction of External "Mask-Free" or "High-Dice" Claims**:
   - **Ferreira et al. (*Nature Communications* 2025 / arXiv:2210.04979)**: Successfully achieves Dice $\approx 0.89$ on Left Ventricular (LV) endocardial segmentation in echocardiography without manual segmentation masks. However, primary fulltext verification (PDF p. 5, lines 142–146) reveals that its edge detection backbone (HED) was **initialized with ImageNet pretrained weights**, meaning it is not strictly trained from scratch. Furthermore, the 0.89 Dice applies solely to the single LV chamber on ultrasound; other chamber measurements are evaluated via clinical volumetric correlations, not multi-class CMR segmentation Dice.
   - **CUTS (MICCAI 2024 / arXiv:2209.11359)**: Generates anonymous topological clusters, not named segmentations. Quantitative Dice evaluation in the paper relies on **post-hoc Hungarian bipartite matching or many-to-one mapping using ground-truth test masks**. It does not autonomously produce named anatomical labels at deployment.
   - **CineMA (arXiv:2506.00679)**: Uses self-supervised Masked Autoencoding (MAE) on 74,916 unlabeled cine CMR studies (UK Biobank), but its downstream ACDC/M&Ms segmentation results ($\sim 0.91$ Dice) are achieved exclusively through **supervised fine-tuning with manual target masks**.
   - **CM-TAPE (*Pattern Recognition*)**: The previously reported DOI was unverified. Secondary publisher indexing suggests DOI `10.1016/j.patcog.2026.114699`, but because the primary fulltext was inaccessible for end-to-end verification, its protocol, representations, checkpointing, and quantitative claims are categorized as **UNKNOWN**.
   - **MedSAM / Foundation Models**: Require manual bounding-box or point prompts at inference time and were pretrained on $>1.5\text{M}$ supervised medical masks.
3. **Supervision Boundary Disambiguation**: Target-dataset-label-free is not synonymous with an absence of anatomical or supervisory priors. Anatomical rules (e.g. concentricity), class text prompts, and source-modality annotations do not use target ground truth, but must be categorized separately from strict scratch learning.
4. **Historical Negative Controls**: Root independently verified historical C0/C1 negative controls in sibling worktree `/Users/alvinluong/Self-Audit-nogt/reports/astra_nogt_baseline_20260918T042050Z`. Candidate-C replay C0/C1 belongs to a separate execution namespace. Negative outcomes serve as empirical falsification controls.

---

## 2. Exhaustive Method Comparison Matrix (Primary-Source Verified)

*Note on Matrix Classification*:
- **Manual Masks Anywhere?**: Whether the training process uses manual segmentation masks at any stage (pretraining, source domain, or target).
- **Target Dataset Manual Masks?**: Whether manual masks from the target evaluation dataset (e.g. ACDC, M&Ms) are used during training or checkpoint selection.
- Fields marked **UNKNOWN** denote details not confirmed directly from accessible primary fulltext.

| Method | Primary Citation & Verified URL | Manual Masks Anywhere? | Target Dataset Manual Masks? | Pretraining Data & Model Class | Foundation Model Used? | Crop / GT Checkpoint Selection Protocol | Semantic Class Names at Inference? | Reported Metric & Dataset in Primary Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Label-Free Cardiac US** | Ferreira et al., *Nat. Commun.* 2025<br>[arXiv:2210.04979](https://arxiv.org/abs/2210.04979)<br>[Nat. Commun. s41467-025-59451-5](https://www.nature.com/articles/s41467-025-59451-5) | **No segmentation masks** (Uses ImageNet weights) | **No** | HED initialized with ImageNet weights (p5 lines 142-146) $\to$ UNet | No | Domain-based weak seed generation; checkpoint selection: UNKNOWN from paper text | **Yes** (LV endocardium/epicardium structurally defined) | **Dice $\approx 0.89$** on LV in echocardiography (external test set); other chambers via clinical metrics |
| **CUTS** | Liu, Amodio, Krishnaswamy et al., *MICCAI* 2024<br>[arXiv:2209.11359](https://arxiv.org/abs/2209.11359)<br>[GitHub: KrishnaswamyLab/CUTS](https://github.com/KrishnaswamyLab/CUTS) | **No** (in training) | **No** (in training) | Scratch (patch reconstruction + contrastive) | No | Patch/contrastive loss; **Evaluation requires test GT masks for Hungarian matching**; checkpoint selection: UNKNOWN | **NO (Anonymous partition)**; cluster indices have no semantic labels without oracle GT | Evaluated on brain MRI (ventricles) and retinal fundus under oracle Hungarian matching; CMR unverified |
| **DFC** | Kanezaki, *IEEE TIP* 2020<br>[DOI: 10.1109/TIP.2020.2993802](https://doi.org/10.1109/TIP.2020.2993802)<br>[GitHub: pytorch-unsupervised-segmentation-tip](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip) | **No** | **No** | Scratch (per-image transductive CNN) | No | Fixed stop criteria (`minLabels` threshold); no validation GT selection in base method | **NO (Anonymous partition)** | Evaluated on natural images (BSD500); medical evaluation uses Hungarian matching or scribbles |
| **CineMA** | Fu et al., 2024/2025<br>[arXiv:2506.00679](https://arxiv.org/abs/2506.00679)<br>[GitHub: mathpluscode/CineMA](https://github.com/mathpluscode/CineMA) | **YES** (for segmentation fine-tuning) | **YES** (fine-tuned on target ACDC/M&Ms) | Spatio-temporal MAE on 74,916 cine CMR studies (UK Biobank) | Yes (Cine Foundation) | Supervised fine-tuning loss on manual target masks; standard validation checkpointing | **Yes** (learned via supervised fine-tuning) | **Dice $\sim 0.91 - 0.92$** on ACDC/M&Ms **only after supervised fine-tuning** with manual masks |
| **CM-TAPE** | Bi et al., *Pattern Recognition*<br>(Secondary metadata: DOI 10.1016/j.patcog.2026.114699)<br>*Primary fulltext unavailable* | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** (Primary paper not accessible to verify protocol or numbers) |
| **Anatomical Position SSL** | Bai et al., *MICCAI* 2019<br>[arXiv:1907.02757](https://arxiv.org/abs/1907.02757) | **YES** (for downstream segmentation) | **YES** (fine-tuned on 1–5+ annotated subjects) | Pretext task: predicting anatomical coordinates | No | Pretext self-supervised loss; downstream supervised fine-tuning with small annotated cohorts | **Yes** (from few-shot manual supervision) | Outperforms scratch UNet in low-label regimes on CMR; requires manual masks |
| **Joint Motion & Seg.** | Qin et al., *MICCAI* 2018<br>[arXiv:1807.03175](https://arxiv.org/abs/1807.03175)<br>[GitHub: Joint_Motion_Seg](https://github.com/cq615/Joint_Motion_Seg) | **YES** (requires manual ED/ES masks) | **YES** (ED/ES manual masks on target sequences) | Motion estimation network + segmentation network | No | Semi-supervised tracking; propagates labeled ED/ES masks to unlabeled cine frames | **Yes** (propagated from labeled ED/ES frames) | Evaluated on cardiac cine sequences with ED/ES annotations |
| **MedSAM** | Ma et al., *Nat. Commun.* 2024<br>[arXiv:2304.12306](https://arxiv.org/abs/2304.12306)<br>[GitHub: bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) | **YES** ($>1.5\text{M}$ manual medical masks) | **No** (zero-shot transfer) | SA-1B + 1.57M medical mask fine-tuning | Yes (SAM) | **Requires manual bounding box prompt at inference**; box often derived from GT mask in benchmarks | **Yes** (prompt specifies object) | High Dice on diverse tasks when prompted with manual bounding boxes; prompt-free: UNKNOWN |
| **Persistent Homology Loss** | Clough et al., *IEEE TPAMI* 2020<br>[arXiv:1910.01877](https://arxiv.org/abs/1910.01877)<br>[GitHub: topological-loss](https://github.com/JamesClough/topological-loss) | **Topological term is mask-free** | **YES** (used alongside base supervised loss) | Topological regularizer during network training | No | Differentiable Betti-number penalty ($B_0=1, B_1=1$ for myocardium); combined with CE/Dice | **Requires base class supervision** | Improves MYO topological connectivity and Dice on CMR when trained with limited labels |

---

## 3. Critical Analysis of Specific Methods & Literature

### 3.1 Ferreira et al. (Nature Communications 2025 / arXiv:2210.04979)
- **Primary Source Analysis**:
  - Title: *"Self-supervised learning for label-free segmentation in cardiac ultrasound"*
  - Authors: Danielle L. Ferreira, Connor Lau, Zaynaf Salaymang, Rima Arnaout
  - Direct Verification (PDF p. 5, lines 142–146):
    > *"We used Holistically-Nested Edge Detection (HED) initialized with ImageNet weights to detect edges..."*
- **Implications for "Mask-Free" Claims**:
  1. *Supervision Source*: While the method is **free of manual segmentation masks**, it is **not strictly trained from scratch**: it utilizes standard ImageNet supervised pretraining for edge initialization.
  2. *Single-Chamber Scope*: The reported $\approx 0.89$ Dice applies specifically to the Left Ventricle (LV) in short-axis and apical echocardiographic views. Other cardiac chambers (e.g. Right Ventricle) are not segmented into a 3-class or 4-class discrete segmentation map with reported Dice; rather, they are assessed via secondary clinical surrogate measurements (e.g., ventricular length ratios).
  3. *Domain Transfer to CMR*: In short-axis cine CMR, the blood pool has high contrast, but the myocardium forms a thin annular ring and the RV forms an asymmetric crescent on the **patient's anatomical right**. Directly transferring an echocardiographic Hough circle pipeline to CMR without adapting for multi-chamber contrast and through-plane slice variations remains unproven in primary literature.

### 3.2 CM-TAPE Status (Pattern Recognition)
- **Correction & Status**:
  - In our initial draft, an unverified DOI was cited. Secondary bibliographic indexing indicates DOI `10.1016/j.patcog.2026.114699` (*Pattern Recognition*, available online 2026).
  - However, because the primary fulltext could not be accessed directly during this audit, all detailed architectural claims regarding CM-TAPE (such as specific 2D VLM transfer mechanisms, CLIP text prompt protocols, crop strategies, GT checkpoint selection, and quantitative DSC numbers) are classified as **UNKNOWN**.
  - **Audit Standard**: Secondary abstracts or search index snippets must not be presented as primary empirical confirmation.

### 3.3 CineMA Foundation Model (Fu et al., 2024/2025 / arXiv:2506.00679)
- **Primary Source Analysis**:
  - Pretraining: CineMA demonstrates that self-supervised masked autoencoding on 74,916 cine CMR studies produces rich spatio-temporal representations of cardiac dynamics.
  - **Crucial Limitation**: The paper does not provide an autonomous, mask-free segmentation mechanism. All published segmentation Dice scores on ACDC and M&Ms ($\approx 0.91 - 0.92$) were achieved via **supervised fine-tuning on manual training masks**.
  - Whether frozen CineMA encoder features can support unprompted, mask-free semantic segmentation via unsupervised clustering or biophysical adapters remains an open empirical question requiring dedicated experiments.

### 3.4 Anonymous Partitioning vs. Named Segmentation (CUTS and DFC)
- **Mathematical Distinction**:
  - **Anonymous Partitioning**: Partitions an image domain $\Omega$ into $K$ disjoint regions $\{S_1, \dots, S_K\}$ based on spatial or feature similarity. Permutation of region IDs is mathematically invariant.
  - **Named Anatomical Segmentation**: Requires a semantic mapping $f: \Omega \to \{\text{BG}, \text{RV}, \text{MYO}, \text{LV}\}$ with clinical identifiability.
- **The Hungarian Matching Fallacy**:
  - CUTS (Liu et al., MICCAI 2024) and DFC (Kanezaki, IEEE TIP 2020) output anonymous cluster IDs.
  - To report a Dice score, benchmark pipelines evaluate the clusters using bipartite matching (Hungarian algorithm) against ground-truth test annotations:
    $$\sigma^* = \arg\max_{\sigma \in \Pi} \text{Dice}(\sigma(S), Y_{\text{GT}})$$
  - This requires access to the ground-truth test masks $Y_{\text{GT}}$ at inference. At deployment on unannotated images, no oracle is available to assign cluster IDs to anatomical names. Thus, reporting Hungarian-matched Dice as "unsupervised segmentation" obscures the fact that the deployed pipeline cannot produce named masks autonomously.

---

## 4. Supervision Taxonomy: Disambiguating Mask and Prior Sources

To prevent conflation of distinct methodologies, priors and supervisory signals must be partitioned cleanly:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               SUPERVISION TAXONOMY                                     │
├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│ 1. TARGET-DATASET LABELS │ 2. EXTERNAL MASK SUPERVISION│ 3. WEAK / ANATOMICAL PRIORS   │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ • Manual masks from      │ • Manual masks from source  │ • Geometric invariants        │
│   ACDC or M&Ms           │   domain (e.g. MM-WHS CT in │   (concentric ring, lumen)    │
│ • Supervised fine-tuning │   cross-modality UDA)       │ • Temporal motion variance    │
│ • Early stopping /       │ • Foundation models trained │ • ImageNet weights (HED)      │
│   checkpoint selection on│   on masks (MedSAM, SAM-Med)│ • Text prompts to 2D VLMs     │
│   target validation GT   │ • Manual prompts at test    │   (class queries)             │
│ • Oracle Hungarian match │   time (bounding box/click) │ • Image-only SSL pretrain     │
│   against target test GT │                             │   (CineMA, raw CMR pixels)    │
└──────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

- **Target-dataset-label-free**: A model that uses zero target dataset masks during training or checkpoint selection. This category includes UDA, zero-shot foundation models, and biophysical seed engines.
- **Strictly manual-mask-free**: A model that never uses manual segmentation masks from any dataset (neither target nor external source domains).
- **Class-Name Priors**: Introducing class text queries, anatomical geometry, or orientation metadata resolves the permutation symmetry disproven in Worker B without requiring manual pixel masks.

---

## 5. Potential Practical Baseline Architecture (Hypothesis for Empirical Validation)

*Note: As established in Section 1, no primary literature proves that this architecture will achieve $\ge 0.88 - 0.90$ Dice on ACDC/M&Ms without manual masks. This represents a conditional engineering hypothesis based on available evidence.*

```
                      [ Raw Cine CMR Slices (2D+T) ]
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌─────────────────────────────────┐         ┌─────────────────────────────────┐
│  STAGE 0: Image-Only SSL        │         │  STAGE 1: Biophysical Seed      │
│  (CineMA-style Spatio-Temporal) │         │  Anatomical Prior Engine        │
│  • Pretrained on raw CMR pixels │         │  • Temporal motion variance     │
│  • Representation backbone      │         │  • Circular lumen search (LV)   │
│  • Features need empirical test │         │  • Concentric ring search (MYO) │
│                                 │         │  • Patient-right crescent (RV)  │
└────────────────┬────────────────┘         └────────────────┬────────────────┘
                 │                                           │
                 │ Feature Embeddings                        │ Weak Initial Seeds
                 │ f_app [B, D, H/4, W/4]                    │ Y_seed [B, H, W]
                 └─────────────────┬─────────────────────────┘
                                   │
                                   ▼
         ┌───────────────────────────────────────────────────┐
         │  STAGE 2: Prototypical Assignment & Dense Gating  │
         │  • Prototype centroids initialized from seeds     │
         │  • Softmax class assignment                       │
         │  • Dense Margin Gating: p(1) - p(2) >= tau        │
         │  • Rejection of ambiguous pixels: UNKNOWN = 255   │
         └─────────────────────────┬─────────────────────────┘
                                   │ Filtered Pseudo-Labels
                                   ▼
         ┌───────────────────────────────────────────────────┐
         │  STAGE 3: Progressive Student Self-Training       │
         │  • Student network trained on valid pixels        │
         │  • Early Learning Regularization (ELR)            │
         │  • Deformable cycle-consistency constraints       │
         └───────────────────────────────────────────────────┘
```

### 5.1 Biophysical Prior Grounding (Correcting Anatomical Coordinates)
1. **Cardiac ROI Localization**:
   In cine CMR, ventricular contraction creates localized temporal intensity variance over cardiac cycles. Temporal variance across frames provides an initial hypothesis for the cardiac region.
2. **Left Ventricle (LV, Class 3)**:
   In short-axis view, the LV cavity appears as a bright, roughly circular lumen enclosed by myocardium. Radial intensity profiles or circular transforms can seed the central LV pool.
3. **Myocardium (MYO, Class 2)**:
   The myocardium is an intermediate-intensity muscular wall enclosing the LV blood pool (concentric annular topology).
4. **Right Ventricle (RV, Class 1)**:
   The RV is an asymmetric blood pool situated on the **patient's anatomical right** (which corresponds to the viewer's left in standard radiological display), abutting the ventricular septum.
5. **Background (BG, Class 0)**:
   Thoracic wall, lungs, and peripheral structures contacting the field-of-view borders.

### 5.2 Mandatory Gating & Noise Resistance
1. **Dense Margin Gating**: As diagnosed by Worker A, region-level gating validates ambiguous 50/50 mixture pixels at boundaries. A dense pixel-level margin gate ($p_{(1)} - p_{(2)} \ge \tau_{\text{margin}}$, e.g. 0.20) must reject mixed boundary assignments as `UNKNOWN = 255`.
2. **Early Learning Regularization (ELR)**: Because biophysical seeds contain geometric approximations (e.g. trabeculations, irregular contours), student training requires regularizers like ELR (Liu et al., NeurIPS 2020) to prevent memorizing seed noise during later training epochs.

---

## 6. Seed Experiments, Falsifiers & Missing Evidence

### 6.1 Falsification Probes & Synthetic Controls
- **Synthetic Moving-Circle Toy Probe** (`reports/astra_v3/workers/D_evidence/synthetic_prior_falsifier.py`):
  - In our preliminary investigation, a toy numerical simulation demonstrated that unseeded linear heads produce uniform probabilities (~0.25) leading to 100% abstention under margin gates.
  - **Explicit Limitation**: This toy script is a **syntactic and algorithmic sanity check only**. It does not constitute evidence of real cardiac physiology, nor does it support quantitative claims regarding in vivo motion variance contrast.
- **Permutation Symmetry Falsifier**:
  - Evaluating an unseeded semantic MLP without external priors confirms that class assignments are completely ungrounded across random initializations.

### 6.2 Missing Local Evidence & Concrete Blockers
1. **Absence of 4D Cine Temporal Sequences Locally**:
   - As audited by Worker E, local storage at `/tmp/astra_event_acdc_training/training` contains only static ED and ES endpoints ($t_{\text{gap}} \approx 10$ frames).
   - Full 4D cine temporal sequences ($T \approx 28 - 40$ frames) are **absent on the workstation**.
   - *Impact*: True temporal motion estimation and temporal variance ROI extraction cannot be executed on local ACDC data until 4D cine NIfTI volumes are staged.
2. **Absence of Raw M&Ms Data**:
   - M&Ms raw image files are absent locally; paths in existing configs point to remote cluster mount locations.
3. **Unpinned Upstream Dependencies**:
   - Checked-in CUTS and DFC baselines do not record immutable upstream Git commit SHAs, leaving upstream bit-level reproducibility unresolved.
4. **Empirical Behavior of Self-Supervised Features**:
   - Masked autoencoders (CineMA) and convolutional encoders produce features that reflect appearance statistics; whether these features naturally separate complex myocardial boundaries without fine-tuning requires direct empirical benchmarking.

---

## 7. Correction Log & Primary URL Ledger

### 7.1 Detailed Correction Log

| Item # | Area | Initial Report Statement | Root Finding & Correction Applied |
| :--- | :--- | :--- | :--- |
| **1** | Ferreira Citation & Weights | Cited arXiv preprint without noting pretraining; implied pure scratch. | **Corrected**: Added verified citations ([arXiv:2210.04979](https://arxiv.org/abs/2210.04979) and [Nature Communications](https://www.nature.com/articles/s41467-025-59451-5)). Explicitly documented that HED uses ImageNet pretrained weights (PDF p. 5, lines 142–146); noted that 0.89 Dice applies solely to LV echo, with other chambers evaluated via clinical metrics. |
| **2** | CM-TAPE Protocol & Metrics | Asserted 76.2% DSC, CLIP transfer, and specific mechanisms under an unverified DOI. | **Corrected**: Removed invalid DOI. Noted secondary indexing points to `10.1016/j.patcog.2026.114699`, but because primary fulltext is inaccessible, marked entire protocol, representations, checkpointing, and numbers as **UNKNOWN**. |
| **3** | Bai Anatomical Position Link | Cited incorrect arXiv link. | **Corrected**: Updated to authoritative URL: [https://arxiv.org/abs/1907.02757](https://arxiv.org/abs/1907.02757) (MICCAI 2019). |
| **4** | LF-LVS & Speculative Dice | Included unverified LF-LVS URL and estimated per-method Dice ranges. | **Corrected**: Removed unsupported LF-LVS URL and speculative Dice intervals. Verified CUTS and DFC bibliographic citations. |
| **5** | Supervision Categorization | Blended target-label-free with general absence of priors. | **Corrected**: Explicitly separated "Manual masks anywhere" vs. "Target dataset manual masks", clarifying that class text prompts and anatomical rules do not use target GT. |
| **6** | Impossibility & Ceiling Claims | Claimed mathematical proof that no method can reach target, with speculative 0.78–0.84 scratch ceilings. | **Corrected**: Retracted universal impossibility claims and speculative numerical ceilings. Accurately stated that no inspected primary source provides supporting evidence of $\ge 0.88 - 0.90$ Dice on ACDC/M&Ms without manual masks. |
| **7** | Historical C0/C1 Controls | Implied historical Candidate-C oracle leakage across C0/C1. | **Corrected**: Clarified that root independently verified historical C0/C1 negative controls in `/Users/alvinluong/Self-Audit-nogt/reports/astra_nogt_baseline_20260918T042050Z`, and Candidate-C replay C0/C1 is a separate execution namespace. |
| **8** | Anatomy & Synthetic Evidence | Stated RV is on "patient-left"; claimed 10x–50x motion contrast based on toy synthetic script. | **Corrected**: Corrected anatomical positioning: RV is on the **patient's anatomical right**. Explicitly marked synthetic script as toy syntax/logic probe, not clinical physiological proof. Removed unsupported motion contrast multipliers. |
| **9** | Feature Boundary Assurances | Claimed SSL/MAE guarantees sharp anatomical boundaries. | **Corrected**: Clarified that CNN/MAE feature distributions require empirical validation on cine data; no automatic boundary sharpness guarantee exists. |
| **10** | Checkpoint Selection Defaults | Assumed standard validation loss checkpointing across methods. | **Corrected**: Marked all crop, prompt, and checkpoint selection protocols as **UNKNOWN** when not explicitly detailed in accessible primary sources. |

### 7.2 Primary URL & Identifier Ledger

- **Ferreira et al. (Label-Free Ultrasound)**:
  - arXiv Preprint: [https://arxiv.org/abs/2210.04979](https://arxiv.org/abs/2210.04979)
  - Nature Communications (2025): [https://www.nature.com/articles/s41467-025-59451-5](https://www.nature.com/articles/s41467-025-59451-5)
- **Bai et al. (Anatomical Position SSL)**:
  - arXiv / MICCAI 2019: [https://arxiv.org/abs/1907.02757](https://arxiv.org/abs/1907.02757)
- **CineMA (Cine Foundation Model)**:
  - arXiv Preprint: [https://arxiv.org/abs/2506.00679](https://arxiv.org/abs/2506.00679)
  - GitHub Code: [https://github.com/mathpluscode/CineMA](https://github.com/mathpluscode/CineMA)
- **CUTS (Multigranular Unsupervised Medical Segmentation)**:
  - arXiv / MICCAI 2024: [https://arxiv.org/abs/2209.11359](https://arxiv.org/abs/2209.11359)
  - GitHub Code: [https://github.com/KrishnaswamyLab/CUTS](https://github.com/KrishnaswamyLab/CUTS)
- **DFC (Differentiable Feature Clustering)**:
  - IEEE TIP 2020: [https://doi.org/10.1109/TIP.2020.2993802](https://doi.org/10.1109/TIP.2020.2993802)
  - GitHub Code: [https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip)
- **Qin et al. (Joint Motion & Segmentation)**:
  - arXiv / MICCAI 2018: [https://arxiv.org/abs/1807.03175](https://arxiv.org/abs/1807.03175)
  - GitHub Code: [https://github.com/cq615/Joint_Motion_Seg](https://github.com/cq615/Joint_Motion_Seg)
- **Clough et al. (Persistent Homology Topological Loss)**:
  - arXiv / IEEE TPAMI 2020: [https://arxiv.org/abs/1910.01877](https://arxiv.org/abs/1910.01877)
  - GitHub Code: [https://github.com/JamesClough/topological-loss](https://github.com/JamesClough/topological-loss)
- **Liu et al. (Early Learning Regularization - ELR)**:
  - arXiv / NeurIPS 2020: [https://arxiv.org/abs/2007.00151](https://arxiv.org/abs/2007.00151)
  - GitHub Code: [https://github.com/ShengLiu-sjtu/ELR](https://github.com/ShengLiu-sjtu/ELR)
- **MedSAM (Segment Anything in Medical Images)**:
  - arXiv / Nature Communications 2024: [https://arxiv.org/abs/2304.12306](https://arxiv.org/abs/2304.12306)
  - GitHub Code: [https://github.com/bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)
