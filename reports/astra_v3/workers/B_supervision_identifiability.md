# Astra Self-Audit v3: Supervision, Identifiability & No-GT Leakage Audit (Worker B)

> **Root review status: investigation record, not accepted wholesale.** Read [root disagreement resolutions](../11_RED_TEAM.md) and the corresponding root report before using these claims. Worker completion indicates a delivered report, not scientific validation.

- **Audit Target Base SHA**: `d4f503b7107eca4e8ac85f37a5dd01ec50479666`
- **Orca Run ID**: `run_698eb3b460ef`
- **Task ID**: `task_7c26026be9b2`
- **Dispatch ID**: `ctx_2818e880b7dc`
- **Auditor**: AGY Worker B
- **Target Files / Focus**: Supervision pathways, identifiability under class permutation, `src/self_audit_pseudolabel/system_v3.py`, `src/self_audit_maskfree/`, `src/shared_benchmark/`
- **Evidence Directory**: `reports/astra_v3/workers/B_evidence/`

---

## 1. Executive Summary

An exhaustive mathematical, structural, and empirical audit of supervision sources, identifiability conditions, and ground-truth (GT) leakage paths was conducted across Self-Audit v3 (`src/self_audit_pseudolabel/system_v3.py`), the maskfree pipeline (`src/self_audit_maskfree/`), and the shared benchmark adapter (`src/shared_benchmark/`) at base commit `d4f503b7107eca4e8ac85f37a5dd01ec50479666`.

### Core Conclusions
1. **Mathematical Disproof of Identifiability in System v3**: In `src/self_audit_pseudolabel/system_v3.py:98-134`, `CinePseudoTeacher` lacks any semantic grounding mechanism. Unsupervised appearance reconstruction and motion features are completely permutation-invariant with respect to class labels $\{0, 1, 2, 3\}$. Any permutation matrix $\Pi \in S_4$ applied to the final semantic projection weights yields identical representation geometries. Thus, **semantic class identity ($\text{BG}, \text{RV}, \text{MYO}, \text{LV}$) is mathematically non-identifiable from cine images alone in System v3 without external asymmetry**.
2. **Where Class Identity Actually Comes From**: Semantic identity in this codebase originates exclusively in two places:
   - In `src/shared_benchmark/adapter.py:61-148` and `src/self_audit_maskfree/ontology.py:18-88`, anatomical classes are identified via **asymmetric topological and geometric invariants**: outer field-of-view border contact identifies Background ($\text{BG}=0$), the topological enclosure relation ($\text{inner cavity} \subset \text{outer ring}$) uniquely resolves Myocardium ($\text{MYO}=2$) and Left Ventricle cavity ($\text{LV}=3$), and exterior non-enclosed adjacency resolves Right Ventricle ($\text{RV}=1$). Patient-left orientation metadata resolves residual bilateral ambiguities.
   - In `system_v3.py:118-121`, semantic identity can only be introduced artificially through external additive `evidence_logits`. Without `evidence_logits`, the unseeded semantic MLP produces uniform outputs ($p \approx 0.25$), triggering 100% pixel abstention (`UNKNOWN = 255`).
3. **Anonymous Partition vs. Named Segmentation**: Anonymous partitioning solves an equivalence relation over spatial coordinates and can be evaluated with permutation-invariant metrics (Variation of Information, Rand Index). Named segmentation requires a bijective ground truth binding. Evaluating an anonymous partition via Hungarian matching or oracle permutation constitutes **oracle GT leakage** and must never be conflated with unsupervised named segmentation.
4. **Historical No-GT Receipts & Falsification Controls**: Historical receipts for Candidate C (C0 baseline execution vs. C1 exact factual replay in `reports/candidate_c/synthetic_flow_receipt.json`) and Maskfree150 (candidate bank verification in `reports/maskfree150/local_baseline/profile_report.json`) were inspected without reading GT arrays. **Negative C0/C1 outcomes are falsification controls** that reject unphysical, non-reproducible, or degenerate solutions—they are not constructive foundations for learning. In historical Maskfree runs, 159 out of 160 units (99.375%) remained `semantic_unresolved` when topological invariants were not met.
5. **Enforceable Freeze Contract**: Complete independence between deployment students and evaluation requires a cryptographically verified execution firewall, pre-freeze candidate hashing, and post-freeze import isolation.

---

## 2. Anonymous Partition vs. Named Segmentation: Foundational Distinction

| Dimension | Anonymous Spatial Partitioning | Named Anatomical Segmentation |
| :--- | :--- | :--- |
| **Mathematical Definition** | Equivalence relation $\mathcal{P} = \{S_1, \dots, S_K\}$ partitioning domain $\Omega = \bigcup_{k} S_k$, where $S_j \cap S_k = \emptyset$. | Bijective mapping $f: \Omega \to \mathcal{C}$, where $\mathcal{C} = \{\text{BG}, \text{RV}, \text{MYO}, \text{LV}\}$ with fixed clinical semantics. |
| **Permutation Invariance** | Strictly invariant under any relabeling $\pi \in S_K$: $\{S_{\pi(1)}, \dots, S_{\pi(K)}\} \equiv \mathcal{P}$. | **Non-invariant**. Relabeling $\sigma \in S_{|\mathcal{C}|}$ alters class assignments and collapses per-class clinical metrics. |
| **Valid Metric Space** | Variation of Information (VI), Adjusted Rand Index (ARI), Normalized Mutual Information (NMI). | Class-specific Dice coefficient, Intersection over Union (IoU), 95th Percentile Hausdorff Distance (HD95). |
| **Supervision Requirement** | **Zero semantic supervision**. Solvable via spatial continuity, clustering, graph cuts, or density models. | **Requires semantic grounding**: explicit labels, topological asymmetry, or physical boundary rules. |
| **Hungarian Matching Status** | Permitted as an oracle diagnostic measuring partition capacity (e.g. `oracle_cluster_matching_diagnostic`). | **Forbidden as a primary metric**. Applying per-case Hungarian matching leaks GT at test time. |

### The Oracle Permutation Fallacy
If a method outputs $K$ anonymous clusters $\{1, \dots, K\}$ and uses the test set ground truth $Y$ to compute:
$$\pi^* = \arg\max_{\pi \in \Pi} \frac{1}{N} \sum_{i=1}^N \text{Dice}(\pi(\hat{Y}_i), Y_i)$$
the resulting score **does not evaluate an autonomous segmentation model**. It evaluates the geometric partition quality under an oracle translator. At deployment, without $Y$, the model has no mechanism to determine $\pi^*$. In 4-class cardiac segmentation, there are $4! = 24$ global permutations; under per-case matching, there are $24^N$ degrees of freedom. Reporting Hungarian-matched Dice as "unsupervised segmentation" constitutes an invalid scientific claim.

---

## 3. Mathematical Proof: Non-Identifiability in System v3 under Class Permutation

### 3.1 Pipeline Formulation
In `src/self_audit_pseudolabel/system_v3.py:98-134`:
1. Cine inputs $X = (x_{t-1}, x_t, x_{t+1})$ pass through appearance encoder $E_{\text{app}}(x_t)$ and motion branch $E_{\text{mot}}(X)$ to produce fused features $F(X) \in \mathbb{R}^{B \times 64 \times H/4 \times W/4}$.
2. Region prototype head assigns pixels to $K$ prototypes via cosine similarity:
   $$q_{b,k,h,w} = \frac{\exp(\langle F_{b,:,h,w}, p_k \rangle / \tau)}{\sum_{j=1}^K \exp(\langle F_{b,:,h,w}, p_j \rangle / \tau)}$$
3. Region pooling produces prototype representations $r_{b,k} \in \mathbb{R}^{67}$ containing pooled features, image intensity, motion magnitude, and normalized area fraction.
4. Semantic head computes unnormalized class logits across the 4 classes:
   $$z_{b,k} = W_2 \cdot \text{GELU}(W_1 r_{b,k} + b_1) + b_2, \quad W_2 \in \mathbb{R}^{4 \times 96}, \; b_2 \in \mathbb{R}^4$$
5. Class probabilities are obtained via softmax over the last dimension:
   $$P_{b,k,c} = \text{Softmax}(z_{b,k,c}) = \frac{\exp(z_{b,k,c})}{\sum_{c'=0}^3 \exp(z_{b,k,c'})}$$
6. Dense probabilities are interpolated and discretized:
   $$\hat{Y}_{b,h,w} = \arg\max_{c \in \{0,1,2,3\}} \sum_{k=1}^K q_{b,k,h,w} P_{b,k,c}$$

### 3.2 Theorem: Permutation Invariance & Non-Identifiability
**Theorem 1.** *Let $\mathcal{L}_{\text{unsup}}(\theta; X)$ be any unsupervised training objective depending solely on image data $X$, reconstruction loss $\|\text{recon} - x_t\|^2$, motion consistency, prototype spatial smoothness, or entropy of $q$. Then for any class permutation matrix $\Pi \in \mathbb{R}^{4 \times 4}$ ($\Pi_{ij} \in \{0,1\}, \Pi \mathbf{1} = \mathbf{1}, \Pi^T \mathbf{1} = \mathbf{1}$), there exists an equivalent parameter configuration $\theta' = (\theta_{\setminus \text{sem}}, W_2', b_2')$ with:*
$$W_2' = \Pi W_2, \quad b_2' = \Pi b_2$$
*such that:*
1. $\mathcal{L}_{\text{unsup}}(\theta'; X) = \mathcal{L}_{\text{unsup}}(\theta; X)$ identically for all $X$.
2. The predicted semantic probability tensor satisfies $P' = \Pi P$.
3. The resulting segmentation satisfies $\hat{Y}' = \pi(\hat{Y})$, where $\pi$ is the permutation corresponding to $\Pi$.

**Proof.**
1. In `system_v3.py:48-53`, the appearance reconstruction head $\text{recon} = \text{Conv2d}(f_{\text{app}})$ depends only on $f_{\text{app}}$ and is disconnected from the semantic head $(W_1, b_1, W_2, b_2)$. Thus $\nabla_{W_2} \mathcal{L}_{\text{recon}} = 0$ (empirically confirmed in `tests/test_pseudolabel_v3_audit.py:44-53` and `probe_results.json:373`).
2. Motion features $E_{\text{mot}}$ and prototype assignments $q$ depend only on $F(X)$ and prototypes $p_k$. They are independent of $W_2, b_2$.
3. Under transformation $W_2' = \Pi W_2$ and $b_2' = \Pi b_2$:
   $$z' = W_2' h + b_2' = \Pi (W_2 h + b_2) = \Pi z$$
   Because softmax is permutation-equivariant:
   $$P' = \text{Softmax}(\Pi z) = \Pi \text{Softmax}(z) = \Pi P$$
4. Any unsupervised loss function that operates without external semantic targets (e.g. Shannon entropy $H(P) = -\sum_c P_c \log P_c$, spatial total variation, or cycle consistency) is symmetric with respect to class labels. Therefore:
   $$\mathcal{L}_{\text{unsup}}(\theta'; X) = \mathcal{L}_{\text{unsup}}(\theta; X)$$
5. Because every permutation $\pi \in S_4$ yields an identical unsupervised loss value, the parameter space contains 24 isolated global optima that partition the spatial domain identically but assign class names arbitrarily.

**Corollary 1.1.** *No gradient-based unsupervised optimization of `system_v3.py` can distinguish BG from LV, or MYO from RV. Semantic identifiability is impossible without external asymmetry.*

### 3.3 Empirical Verification
In `reports/astra_v3/workers/B_evidence/test_supervision_identifiability.py`, we probed `CinePseudoTeacher`:
- Applying non-trivial permutation $\pi = (1, 2, 3, 0)$ to `teacher.semantic[2]` resulted in exact probability remapping:
  $$\max |P_{\text{perm}} - \pi(P_{\text{base}})| = 0.000000$$
- Unseeded default outputs produced mean top-1 probability of **0.2735** across prototypes.
- Under default thresholds (`min_prob=0.70`, `min_margin=0.20`), valid fraction was **0.0000** (100% `UNKNOWN = 255`).

```
                    ┌────────────────────────┐
                    │ Cine Frames (prev,cur) │
                    └───────────┬────────────┘
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
       Appearance Encoder             Motion Branch
       (Reconstruction:               (Intensity Diffs:
        Disconnected Head)             Permutation Invariant)
                  │                           │
                  └─────────────┬─────────────┘
                                ▼
                         Fused Features
                                │
                                ▼
                       Region Prototypes
                     (Anonymous Clusters)
                                │
                                ▼
                          Semantic MLP
               ┌────────────────┴────────────────┐
               ▼                                 ▼
   Without External Asymmetry        With Topological Grounding
   • 24-fold class permutation       • Border contact: BG (0)
     degeneracy                      • Enclosure outer: MYO (2)
   • Mean prob ~0.25                 • Enclosure inner: LV (3)
   • 100% UNKNOWN (255)              • Exterior adjacent: RV (1)
   • Non-identifiable                • Identifiable & Grounded
```

---

## 4. Anatomy Provenance: Where Do BG, RV, MYO, and LV Identities Actually Come From?

To establish where semantic labels originate across the audited codebase, we trace each class in `system_v3.py`, `shared_benchmark/adapter.py`, and `self_audit_maskfree/ontology.py`.

### 4.1 System v3 (`src/self_audit_pseudolabel/system_v3.py`)
- **Background (BG, Class 0)**: Unseeded random weights. No border prior, no field-of-view exterior check.
- **Right Ventricle (RV, Class 1)**: Unseeded random weights. No adjacency check.
- **Myocardium (MYO, Class 2)**: Unseeded random weights. No annular enclosure check.
- **Left Ventricle (LV, Class 3)**: Unseeded random weights. No cavity interior check.
- **Additive Evidence Logits** (`system_v3.py:118-121`): The ONLY way `system_v3.py` assigns semantic meaning is if an external tensor `evidence_logits` is injected:
  ```python
  if evidence_logits is not None:
      logits = logits + evidence_logits
  ```
  If `evidence_logits` is omitted, the model has zero semantic knowledge and abstains entirely.

### 4.2 Shared Benchmark Adapter (`src/shared_benchmark/adapter.py`)
In `adapter.py:61-148`, semantic identities are derived deterministically from anonymous region graphs using explicit anatomical topology:

1. **Background (BG = 0)** (`adapter.py:61-85`):
   - **Invariant**: Any connected component touching the field-of-view border ring (`component.border_contact == True`).
   - **Rule**: The largest border-contact component seeds BG (`unique_border_background`). Other border fragments with identical source cluster ID are merged into BG.
2. **Myocardium (MYO = 2) & Left Ventricle (LV = 3)** (`adapter.py:87-116`):
   - **Invariant**: Topological enclosure $\text{Encloses}(A, B)$ where inner cavity $B$ lies strictly within an interior hole of outer ring $A$.
   - **Rule**: A unique enclosure pair $(A, B)$ assigns $A \to \text{MYO}$ (`unique_enclosure_outer`) and $B \to \text{LV}$ (`unique_enclosure_inner`).
   - If zero or multiple enclosure pairs exist, the components are set to `ambiguous_enclosure` or `missing_enclosed_cavity` and assigned `VOID = 4`.
3. **Right Ventricle (RV = 1)** (`adapter.py:118-148`):
   - **Invariant**: Non-border, non-enclosed component adjacent to MYO exterior.
   - **Rule**: If exactly one component is 4-connected to the outer boundary of MYO, it is assigned $\text{RV}$ (`unique_adjacent_rv`). If multiple exist, they are set to `ambiguous_rv` (`VOID = 4`).

### 4.3 Maskfree Ontology (`src/self_audit_maskfree/ontology.py`)
In `ontology.py:18-88`, rules $R_1$ through $R_9$ formalize role resolution:
- **$R_1$ (Background)**: Group covering $\ge 20\%$ (`BG_RING_SHARE`) of border ring $\to \text{BG}$.
- **$R_2$ (Merge Excess)**: Bounds non-BG groups to $\le 4$ (`MAX_FOREGROUND_GROUPS`) via iterative adjacency merges.
- **$R_3$ (Enclosure)**: Hole-containment fraction $\ge 60\%$ (`ENCLOSURE_FRACTION`) resolves $\text{MYO}$ and $\text{LV}$.
- **$R_4$ (Remainder)**: Non-BG group adjacent to MYO becomes $\text{RV}$.
- **$R_5$ (Intensity Prior)**: Blood pool brighter than myocardium on normalized scale. Darkest remaining foreground group becomes MYO draft if $R_3$ fails; marked `semantic_unresolved`.
- **$R_{5b}$ (Orientation)**: Left/Right disambiguation. When NIfTI geometry is valid and view is `short_axis`, patient-left coordinate vector (`inplane_left_axis`) ensures LV is positioned patient-left of RV.
- **$R_7$ (Absent Roles)**: Missing anatomical structures are legally recorded in `absent_roles` with zero penalty.

---

## 5. Comprehensive No-GT Leakage Audit Across the Ten Channels

We audited all ten potential semantic information leakage pathways at base SHA `d4f503b7107eca4e8ac85f37a5dd01ec50479666`:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    NO-GT LEAKAGE AUDIT MATRIX                              │
├──────────────────────────────┬───────────────┬─────────────────────────────┤
│ Leakage Channel              │ Vulnerability │ Implemented Guard / Status │
├──────────────────────────────┼───────────────┼─────────────────────────────┤
│ 1. Manual Masks              │ CRITICAL      │ Blocked by firewall.py      │
│ 2. Filename / _gt Presence   │ HIGH          │ Discovery enumerates all    │
│ 3. Info.cfg ED/ES Metadata   │ HIGH          │ Forbidden in firewall.py    │
│ 4. Orientation / Affines     │ MEDIUM        │ Restricted to image headers │
│ 5. Preprocessing / Crops     │ HIGH          │ Unchecked if pre-cropped    │
│ 6. Pretrained Models         │ CRITICAL      │ No foundation weights in v3 │
│ 7. Threshold Tuning          │ MEDIUM        │ Preregistered / Predictive  │
│ 8. Pseudo-Label Selection    │ HIGH          │ Top2 Margin (Defects in v3) │
│ 9. Checkpoint Selection      │ HIGH          │ Image-only validation NLL   │
│ 10. Hungarian Mapping        │ CRITICAL      │ Forbidden in primary eval   │
└──────────────────────────────┴───────────────┴─────────────────────────────┘
```

### 5.1 Manual Masks (`_gt.nii.gz`, `_mask.nii.gz`)
- **Vulnerability**: Direct reading of ground truth segmentation masks during feature learning, clustering, or pseudo-label selection.
- **Codebase Guard**: In `src/self_audit_maskfree/data/firewall.py:24-58`, `FORBIDDEN_DIR_NAMES` (`mask`, `masks`, `gt`, `groundtruth`, `segmentation`) and `FORBIDDEN_NAME_TOKENS` (`_gt.`, `_mask`, `_label`, `_manual`) are enforced by `assert_image_only(path)`.
- **Gap in System v3**: `system_v3.py` operates purely on PyTorch tensors in memory and does not invoke `firewall.py`. If an upstream script loads GT masks into tensor inputs, `system_v3.py` cannot detect the violation.

### 5.2 Filename & `_gt` Existence Leakage
- **Vulnerability**: In ACDC and M&Ms datasets, GT masks are only provided for End-Diastole (ED) and End-Systole (ES) frames. Checking whether `{stem}_gt.nii.gz` exists in a directory leaks clinical cycle identification without processing cine motion.
- **Codebase Guard**: In `src/self_audit_maskfree/data/discovery.py:10-15`, discovery explicitly enumerates every frame of 4D cine files. Single-frame volumes do not query sibling directory contents for annotation presence.

### 5.3 ACDC `Info.cfg` ED/ES Metadata and Diagnostic Group Leakage
- **Vulnerability**: `Info.cfg` specifies `ED: <int>`, `ES: <int>`, and `Group: <NOR|MINF|DCM|HCM|RV>`. Reading this file leaks expert temporal phase annotations and clinical diagnoses.
- **Codebase Guard**: In `firewall.py:41`, `FORBIDDEN_FILE_NAMES = frozenset({"info.cfg", "diagnosis.csv"})`. Calling `assert_image_only("Info.cfg")` raises `MaskAccessError` (verified in `B_evidence/test_supervision_identifiability.py`).

### 5.4 Patient Orientation, NIfTI Affines, and In-Plane Left Axis
- **Vulnerability**: Using anatomical knowledge (e.g. "LV is to the left of RV in the matrix") without checking the physical acquisition matrix produces flipped labels on mirrored or non-standard scans.
- **Codebase Guard**: In `src/self_audit_maskfree/data/geometry.py:12-45` and `ontology.py:75-88` ($R_{5b}$), orientation is derived solely from the NIfTI `qform`/`sform` affine matrix (`inplane_left_axis`), verifying patient coordinate space without referencing manual masks.

### 5.5 Preprocessing, FOV Centering, and Bounding Box Crops
- **Vulnerability (Severe Risk)**: Many medical segmentation papers compute a 128x128 bounding box around the ground-truth heart mask (`ymin, ymax = gt.nonzero()`), crop the image, and claim "unsupervised segmentation" on the cropped patch. This eliminates 90% of the background and solves the hardest part of localization.
- **Codebase Guard & Finding**: `firewall.py` checks file paths, but cannot inspect whether a `.nii.gz` file on disk was pre-cropped by an external preprocessing tool (`test_supervision_identifiability.py` showed `patient001_crop128.nii.gz` passes `assert_image_only`). **Contract Requirement**: Full field-of-view images ($256 \times 256$) must be processed; any cropping must be derived purely from image intensity or motion centroids.

### 5.6 Pretrained, External, and Foundation Models
- **Vulnerability**: Utilizing backbones pretrained on supervised datasets (ImageNet-1k, TotalSegmentator, SAM-Med2D) smuggles semantic priors into an allegedly "mask-free" model.
- **Codebase Status**: `system_v3.py` defines lightweight scratch modules (`AppearanceEncoder`, `MotionBranch`) with random initialization. No pretrained foundation checkpoints are loaded in base `d4f503b`.

### 5.7 Threshold Selection & Hyperparameter Tuning
- **Vulnerability**: Tuning confidence thresholds (`min_prob=0.70`, `min_margin=0.20` in `system_v3.py:111`, or `BG_RING_SHARE=0.20` in `ontology.py:19`) by sweeping against validation Dice scores embeds GT knowledge into the hyperparameters.
- **Codebase Guard**: In `src/self_audit_maskfree/evaluation/verification.py`, thresholds are evaluated on held-out predictive NLL ($O_{\text{verify}}$) rather than Dice. In `system_v3.py`, thresholds are fixed constants, though currently uncalibrated for unseeded models.

### 5.8 Pseudo-Label Selection & Confidence Filtering
- **Vulnerability**: Selecting pseudo-labels based on agreement with manual masks.
- **Codebase Guard**: Selection in `system_v3.py:122-130` uses internal margin criteria:
  $$\text{valid} = (P_{\text{top1}} \ge 0.70) \land (P_{\text{top1}} - P_{\text{top2}} \ge 0.20)$$
  However, as demonstrated in Worker A's audit (`system_v3.py:126`), spatial interpolation of low-resolution booleans dilutes boundaries and can accept mixed pixels.

### 5.9 Checkpoint Selection (Early Stopping)
- **Vulnerability**: Selecting the "best" model checkpoint via peak validation Dice on manual masks.
- **Codebase Guard**: In `src/self_audit_maskfree/trainer.py`, model checkpoints must be selected using self-supervised losses or predictive observation likelihood on held-out selection splits ($O_{\text{select}}$). Supervised checkpoint selection is strictly forbidden.

### 5.10 Hungarian Mapping & Post-Hoc Permutation Matching
- **Vulnerability**: Relabeling predictions using Hungarian matching against GT during metric computation.
- **Codebase Guard**: In `src/self_audit_maskfree/evaluation/metrics.py:175-176`:
  *"The class mapping is the frozen named order; no per-case permutation or Hungarian matching happens anywhere in this function."*
  In `src/self_audit_maskfree/evaluation/reference.py:685-736`, `oracle_cluster_matching_diagnostic` computes a single cohort-wide permutation, but it is explicitly flagged with `diagnostic=True` and a warning stating it is not an evaluation score.

### 5.11 Evaluation Access, Process Separation & Verification Firewall
- **Vulnerability**: Bidirectional coupling between evaluation and training routines.
- **Codebase Guard**: `reference.py` enforces a one-way air gap: it runs post-freeze, imports zero modules from training, and returns no signals back to training.

---

## 6. Historical No-GT Receipts for C0/C1 and Falsification Controls

### 6.1 Falsification Controls vs. Foundations
In this audit, **negative C0/C1 outcomes are falsification controls, not foundations**:
- **Candidate C (C0 / C1)**: C0 is the ordinary forward execution. C1 is the exact frozen factual-support replay check (`src/self_audit/models/self_audit_net.py:340-353`):
  $$|\text{replay}(P) - B_f| \le \text{atol} + \text{rtol} |B_f|$$
  A passing C1 check certifies numeric reproducibility and state integrity under AMP. A failing C1 check (`c1_passed == False`) acts as a falsification control that rejects invalid states and forces a fallback to ordinary annotation (`self_audit_net.py:1405-1411`). Passing C1 does *not* prove clinical accuracy or anatomical correctness; it only verifies that the mathematical counterfactual operator is well-defined.
- **Maskfree Degenerate Controls**: In `src/self_audit_maskfree/evaluation/partitions.py:28-107`, four degenerate hypotheses are constructed:
  1. `control_all_background`
  2. `control_random_spatial_masks`
  3. `control_class_permutation`
  4. `control_excessive_partition`
  In `verification.py:560-587`, if any degenerate challenger scores a lower predictive NLL than the selected candidate on held-out observations $O_{\text{verify}}$, `falsification_flag = True`. Falsification proves that the candidate bank or observation model is flawed. Surviving falsification does not prove ground truth validity; it merely establishes that the model is non-degenerate.

### 6.2 Exact Paths, Configs, and Numbers from Historical Receipts
All receipts were inspected directly without opening manual GT arrays:

#### 1. Candidate C Synthetic Flow Receipt
- **Receipt Path**: `reports/candidate_c/synthetic_flow_receipt.json`
- **Workspace**: `/var/folders/zk/7qfmzgnd1c9fqnzv4lfsm8t00000gn/T/candidate-c-full-flow-fu2lows5`
- **Elapsed Time**: 18.575 s
- **Mode**: `candidate_c`
- **Replay Failures**: `acdc_replay_failures = 0`, `mnms_replay_failures = 0` (100% C1 numerical replay pass rate across all active solves).
- **Solver Accounting**:
  - ACDC Epoch 0 & 1: `candidate_c_attempts = 4`, `feasible = 2`, `fallback = 2`, `invalidations = 5`, `ordinary_real_evidence_attempts = 4`, `real_evidence_attempts = 8`.
  - Native M&Ms Epoch 0 & 1: `candidate_c_attempts = 4`, `feasible = 1`, `fallback = 3`, `invalidations = 5`.
  - Epoch 2: `candidate_c_attempts = 0` (curriculum completion).

#### 2. Candidate C Post-Cleanup Test Receipt
- **Receipt Path**: `reports/candidate_c/post_cleanup_test_receipt.json`
- **Execution**: 60.194 s, `exit_code: 0`.
- **Source Signature**: `394dd2438094b6eeab105d227d104f5c27c94771f3eb2a6ad43694ebe5552eed` (72 Python/YAML files verified).
- **Critical Imports**: 4.

#### 3. Candidate C Final Full Test Receipt
- **Receipt Path**: `reports/candidate_c/final_test_receipt.json`
- **Execution**: 498.693 s, `exit_code: 0`.
- **Gate 1 Status**: PASS (21 compiled files, 7 critical imports).

#### 4. Maskfree150 Validation Receipt
- **Receipt Path**: `reports/maskfree150/validation_receipt.json`
- **Base Commit**: `7607eeac7b185998af030af57208a0902632706a`
- **Focused Checks**: 45 passed, 0 failed, 0 errors, 16.279 s.
- **Reference Checks**: 5 passed, 0 failed, 0 errors, 13.506 s.
- **Software Probes**: ACDC & M&Ms 16 units each, outcome: `label_generation_produced_valid_foreground`.
- **Physical GPU Execution**: `acdc_150: NOT STARTED`, `mnms_150: NOT STARTED` (deferred by user; GPU rented later).

#### 5. Maskfree150 Local Baseline Profile Report
- **Receipt Path**: `reports/maskfree150/local_baseline/profile_report.json`
- **Audit Accounting**:
  - `units`: 160
  - `candidates`: 640 (4 per bank: `c0_grouping`, `c1_initializer`, `c2_boundary`, `c3_split/semantic`)
  - `fits`: 640
  - `fit_steps`: 3200
  - `score_calls`: 5866
  - `accepted_edits`: 64
  - **`semantic_unresolved`**: **159 out of 160 units (99.375%)**.
  - **Significance**: In 99.375% of units, the unsupervised candidate bank failed to satisfy the strict topological enclosure invariants required to establish named anatomy, correctly triggering semantic abstention.

---

## 7. Hierarchy of Scientific Claims: Scratch vs. SSL vs. Transfer

To prevent misleading claims in Astra v3 documentation, we formalize the four admissible tiers of learning and the boundaries governing their reported results:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    HIERARCHY OF SCIENTIFIC CLAIMS                          │
├─────────┬──────────────────────────────────┬───────────────────────────────┤
│ Tier    │ Supervision Regime               │ Admissible Claims             │
├─────────┼──────────────────────────────────┼───────────────────────────────┤
│ Tier 0  │ Strict Scratch                   │ Code execution, shape check,  │
│         │ (Random init, zero pretraining)  │ gradient flow. NO anatomy.    │
├─────────┼──────────────────────────────────┼───────────────────────────────┤
│ Tier 1  │ In-Domain Image-Only SSL         │ Feature consistency, temporal │
│         │ (Unlabeled cardiac cine only)    │ equivariance. NO named Dice.  │
├─────────┼──────────────────────────────────┼───────────────────────────────┤
│ Tier 2  │ Out-of-Domain SSL Transfer       │ Spatial clustering, feature   │
│         │ (DINOv2, natural image SSL)      │ transfer. NO clinical Dice.   │
├─────────┼──────────────────────────────────┼───────────────────────────────┤
│ Tier 3  │ Supervised / Foundation Transfer │ Supervised transfer accuracy. │
│         │ (ImageNet-1k, SAM, TotalSegment) │ NOT "unsupervised" learning.  │
└─────────┴──────────────────────────────────┴───────────────────────────────┘
```

### 7.1 Tier 0: Strict Scratch
- **Condition**: All weights initialized randomly ($W \sim \mathcal{N}(0, \sigma^2)$). Zero pretraining, zero external weights, zero manual annotations.
- **Permitted Claims**: Architectural validity, tensor shape consistency, gradient backward flow, computational latency, and abstention compliance.
- **Prohibited Claims**: Any claim of segmentation accuracy, foreground discovery, or Dice performance. An unseeded scratch model outputs uniform logits and has zero semantic capability.

### 7.2 Tier 1: In-Domain Image-Only Self-Supervised Pretraining (SSL)
- **Condition**: Trained exclusively on raw cine MR pixel volumes (e.g. 4D cine without masks, without `Info.cfg`, without bounding box crops). Objectives: contrastive predictive coding, temporal cycle consistency, or masked autoencoding.
- **Permitted Claims**: Temporal feature equivariance, motion representation quality, unsupervised partition stability (measured by ARI/NMI), and predictive observation NLL on held-out frames.
- **Prohibited Claims**: Named segmentation Dice (BG/RV/MYO/LV) without a topologically grounded adapter. Claims of clinical volumetry (EDV, ESV, EF) are invalid unless topological identifiability is proven.

### 7.3 Tier 2: Out-of-Domain Unsupervised Transfer
- **Condition**: Backbone pretrained on large-scale unlabeled natural images (e.g. DINOv2, MAE) without medical labels.
- **Permitted Claims**: Zero-shot feature clustering, semantic correspondence transfer, linear probe capability.
- **Prohibited Claims**: Claiming "zero-shot medical discovery from scratch." The source data must be explicitly reported as natural image foundation pretraining.

### 7.4 Tier 3: Supervised Foundation Transfer
- **Condition**: Backbone pretrained on supervised natural or medical datasets (ImageNet-1k, TotalSegmentator, SAM-Med2D).
- **Permitted Claims**: Few-shot or zero-shot transfer segmentation accuracy under frozen or fine-tuned representations.
- **Prohibited Claims**: **Must NEVER be described as "unsupervised," "self-supervised," or "mask-free."** Pretraining on supervised masks transfers human annotation priors directly into the model.

---

## 8. Enforceable Evaluator Freeze & Firewall Contract

To ensure that evaluation is completely tamper-proof and impervious to GT leakage, the evaluator must satisfy four mechanical invariants:

### 8.1 Evaluator Freeze Protocol
```python
class EvaluatorFirewallContract:
    """Cryptographic and architectural specification for post-freeze evaluation."""
    
    # 1. Pre-freeze artifact immutability
    REQUIRED_FREEZE_MANIFEST_KEYS = frozenset({
        "freeze_id", "created_at", "base_commit", "model_weights_sha256",
        "predicted_volumes_manifest", "config_sha256", "environment_signature"
    })
    
    # 2. Strict import isolation (AST / Module check)
    FORBIDDEN_EVALUATOR_IMPORTS = frozenset({
        "self_audit.training",
        "self_audit.models",
        "self_audit_pseudolabel",
        "self_audit_maskfree.trainer",
        "self_audit_maskfree.auditor",
        "torch.optim",
    })
    
    # 3. Execution barrier
    @staticmethod
    def verify_preconditions(freeze_manifest_path: Path) -> None:
        """Re-hashes all frozen predictions and weights before opening a single GT mask."""
        manifest = json.loads(freeze_manifest_path.read_text())
        for rel_path, expected_hash in manifest["file_hashes"].items():
            actual_hash = sha256_file(rel_path)
            if actual_hash != expected_hash:
                raise FreezeTamperingError(f"Hash mismatch on {rel_path}: {actual_hash} != {expected_hash}")
```

### 8.2 Invariant Rules
1. **One-Way Cryptographic Air Gap**: Evaluator runs in a separate process *after* all student checkpoints and prediction arrays are written to disk and hashed. If any file hash diverges, evaluation halts with `FreezeValidationError`.
2. **Zero Upstream Communication**: The evaluator outputs durable reports (`.json`, `.csv`, `.md`). It exposes no callable API or feedback hook to training.
3. **No Per-Case Permutation Relabeling**: Primary metrics compute Dice and HD95 on the frozen named order `[0=BG, 1=RV, 2=MYO, 3=LV]`. Per-case Hungarian matching is rejected at the API level.
4. **Strict Audit Boundary**: All evaluator invocations must log an entry in `reports/astra_v3/orchestration/receipts/` with execution duration, Git SHA, and full input argument lists.

---

## 9. Categorized Findings: Facts, Inferences, Proposals, and NOT RUN

### 9.1 Facts (Verifiable from Code & Receipts at `d4f503b`)
1. `src/self_audit_pseudolabel/system_v3.py:109` initializes `self.semantic` as an unseeded MLP (`Linear(67, 96)` $\to$ `GELU` $\to$ `Linear(96, 4)`). It contains no topological enclosure, border, or orientation logic.
2. In `system_v3.py:48-53`, the appearance reconstruction loss backpropagates only into `AppearanceEncoder`; gradients to `MotionBranch`, `fuse`, `regions`, and `semantic` are strictly `None` (`test_pseudolabel_v3_audit.py:51-52`).
3. An unseeded `CinePseudoTeacher` outputs class probabilities near 0.25, failing the top-1 threshold (`min_prob=0.70`), yielding 100% `UNKNOWN = 255` pseudo-labels (`probe_results.json:168`).
4. In `src/self_audit_maskfree/evaluation/metrics.py:175-176`, Hungarian matching is explicitly prohibited in primary Dice calculations.
5. In `src/self_audit_maskfree/data/firewall.py:41`, `Info.cfg` and `diagnosis.csv` are explicitly forbidden files.
6. In `reports/maskfree150/local_baseline/profile_report.json`, 159 out of 160 units (99.375%) were recorded as `semantic_unresolved`.
7. In `reports/candidate_c/synthetic_flow_receipt.json`, Candidate C achieved 0 replay failures across all ACDC and M&Ms epochs.

### 9.2 Inferences (Deductions from Code Structure & Mathematics)
1. **System v3 cannot learn named segmentation from scratch**: Without an external semantic bridge (either an anatomical rule adapter like `adapter.py` or pre-labeled evidence logits), gradient descent on cine images cannot resolve class identities.
2. **Old Auditor removal is verified**: `system_v3.py` decouples the teacher from the student, and the student deploys directly with fixed resource profiles (`compact`, `balanced`, `accurate`) without calling an online Auditor.
3. **C0/C1 are diagnostic falsification filters**: They ensure that the solver operates within numerical stability limits and does not degrade factual state reproducibility; they do not construct semantic knowledge.

### 9.3 Proposals (Actionable Recommendations for Root)
1. **Integrate `cardiac_adapter_v2` into Teacher**: Connect `RegionPrototypeHead` outputs to `shared_benchmark/adapter.py` or `self_audit_maskfree/ontology.py`. Anonymous prototype regions must pass through topological enclosure and border-adjacency resolution to seed the semantic head before student distillation.
2. **Add Header-Based Crop Firewall**: Extend `firewall.py` to inspect image array dimensions. If an image is cropped ($< 200 \times 200$), verify that the bounding box was generated by an image-intensity or motion-based field-of-view finder, not an annotation bounding box.
3. **Formalize Preregistered Thresholds**: Fix prototype gating thresholds based on empirical calibration curves rather than arbitrary defaults (e.g. adjust `min_prob` dynamically based on entropy).
4. **Mandate Evaluator Process Isolation**: Ensure that `tests/` and deployment scripts invoke the evaluator via a frozen CLI entry point that enforces manifest hashing.

### 9.4 NOT RUN (Explicit Boundary Declarations)
1. **No Manual GT Arrays Opened**: In strict compliance with the audit charter, no `.nii.gz` ground truth segmentation masks were loaded or inspected.
2. **No Real MRI Real-Data Training**: No GPU-based model fitting, no ACDC/M&Ms cine training runs, and no hyperparameter tuning were executed.
3. **No Semantic Performance Claims**: All reported numbers represent software contracts, mathematical proofs, and synthetic probes. No clinical segmentation accuracy is asserted.

---

## 10. Reproducibility & Commands Ledger

### 10.1 Reproduction Environment
- **Python Binary**: `/Users/alvinluong/miniforge3/bin/python`
- **Environment Flags**: `PYTHONPATH=src`
- **Platform**: macOS (Apple Silicon arm64)
- **Dependencies**: `torch`, `scipy`, `numpy`, `pytest`, `nibabel`

### 10.2 Commands Executed
```bash
# 1. Probe System v3 permutation equivariance, topology adapter, and firewall rules:
rtk proxy env PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python \
  reports/astra_v3/workers/B_evidence/test_supervision_identifiability.py

# 2. Inspect historical receipts:
cat reports/candidate_c/synthetic_flow_receipt.json
cat reports/candidate_c/post_cleanup_test_receipt.json
cat reports/maskfree150/validation_receipt.json

# 3. Check base commit integrity:
rtk git log -1 --format="%H %s"
# Output: d4f503b7107eca4e8ac85f37a5dd01ec50479666 Add cine pseudo-label teacher and adaptive annotator scaffold
```

### 10.3 Generated Evidence Artifacts
- Script: `reports/astra_v3/workers/B_evidence/test_supervision_identifiability.py`
- Evidence JSON: `reports/astra_v3/workers/B_evidence/supervision_identifiability_evidence.json`
- Full Audit Report: `reports/astra_v3/workers/B_supervision_identifiability.md`
