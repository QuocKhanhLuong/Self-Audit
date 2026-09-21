# Astra Self-Audit v3: Pseudolabel Code & Architecture Audit (Worker A)

> **Root review status: investigation record, not accepted wholesale.** Read [root disagreement resolutions](../11_RED_TEAM.md) and the corresponding root report before using these claims. Worker completion indicates a delivered report, not scientific validation.

- **Audit Target Base SHA**: `d4f503b7107eca4e8ac85f37a5dd01ec50479666`
- **Orca Run ID**: `run_698eb3b460ef`
- **Task ID**: `task_87f5a2436605`
- **Auditor**: AGY Worker A
- **Target File**: `src/self_audit_pseudolabel/system_v3.py`
- **Evidence Directory**: `reports/astra_v3/workers/A_evidence/`

---

## 1. Executive Summary

An exhaustive mathematical, architectural, and empirical code audit of `src/self_audit_pseudolabel/system_v3.py` at commit `d4f503b` was performed. The pipeline represents an offline pseudo-label teacher (`CinePseudoTeacher`) coupled with an adaptive online deployment student (`AdaptiveAnnotationStudent`).

**Core Finding**: The codebase is **syntactically and structurally coherent software**, executing without tensor crashes or gradient faults. However, as an unsupervised cardiac segmentation learner, **it completely lacks the supervisory and physical signal required to learn named semantic classes**. Without external `evidence_logits` or anatomical priors, the unseeded semantic head produces uniform class logits (~0.25), resulting in **100% pixel abstention (`UNKNOWN = 255`)** and zero training signal for downstream students. Furthermore, the motion branch measures raw pixel intensity differences rather than optical flow, the prototype head is permutation-unstable and prone to boundary mixture dilution, and the appearance reconstruction head is completely disconnected from semantic supervision.

---

## 2. Architectural Pipeline Trace

The offline pseudo-label generation path traces from input cine frames to the final discrete pseudo-label mask. Below is the exact tensor flow across each stage with shapes, softmax axes, resize modes, and gradient boundaries:

```
Input: prev, cur, nxt [B, 3, H, W]
  │
  ├──> AppearanceEncoder(cur) ───────────────────────────> f_app [B, 48, H/4, W/4]
  │       └─> recon head (1x1 conv) ──[bilinear]─────────> recon [B, 1, H, W] (isolated head)
  │
  ├──> MotionBranch(prev[:, 1:2], cur[:, 1:2], nxt[:, 1:2]) ─> f_mot [B, 16, H/4, W/4]
  │       (raw: [c-p, n-c, |c-p|, |n-c|] [B, 4, H, W])
  │
  ├──> Fusion: Conv2d(64, 64, 1) + GN + GELU ─────────────> fused [B, 64, H/4, W/4]
  │
  ├──> RegionPrototypeHead: cosine_sim(fused, p) / 0.1 ──> q [B, K, H/4, W/4]
  │       (Softmax axis: dim=1 across K prototypes)
  │
  ├──> pool_regions(q, fused, cur[:, 1:2], f_mot) ─────────> r [B, K, 67]
  │       (weighted pooled: feat[64], inten[1], mov[1], area[1])
  │
  ├──> Semantic Head: Linear(67, 96) -> GELU -> Linear(96, 4) ─> logits [B, K, 4]
  │       (+ evidence_logits if provided)
  │       (Softmax axis: dim=-1 across 4 classes) ────────> prob [B, K, 4]
  │
  ├──> Prototype Validity Gate: top1 >= 0.70 & (top1 - top2) >= 0.20 ─> valid_region [B, K] (bool)
  │
  ├──> Spatial Aggregation:
  │       dense_prob_low = einsum("bkhw,bkc->bchw", q, prob)  [B, 4, H/4, W/4]
  │       dense_valid_low = einsum("bkhw,bk->bhw", q, valid_region) >= 0.5 [B, H/4, W/4]
  │
  ├──> Upsampling:
  │       dense_prob = interpolate(dense_prob_low, [H, W], mode="bilinear", align_corners=False) [B, 4, H, W]
  │       dense_valid = interpolate(dense_valid_low.float(), [H, W], mode="nearest").bool() [B, H, W]
  │
  └──> Discretization:
          label = dense_prob.argmax(dim=1) [B, H, W]
          pseudo = torch.where(dense_valid, label, UNKNOWN) [B, H, W] (torch.long, detached)
```

### Exact Parameter & Operation Matrix

| Stage | Input Tensor(s) & Shapes | Operation / Module | Output Shape | Softmax Axis / Resize Mode | Gradient Detach / Flow |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | `prev, cur, nxt`: `[B, 3, H, W]` | Cine timeframe slices ($z-1, z, z+1$) | `[B, 3, H, W]` | None | Leaf tensors |
| **Appearance** | `cur`: `[B, 3, H, W]` | `AppearanceEncoder` (`e0`: stride 1, `e1`: stride 2, `e2`: stride 2, `out`: 1x1 conv) | `f_app`: `[B, 48, H/4, W/4]` | None | Full gradient flow to `cur` |
| **Reconstruction** | `f_app`: `[B, 48, H/4, W/4]` | `F.interpolate` + `nn.Conv2d(48, 1, 1)` | `recon`: `[B, 1, H, W]` | `bilinear`, `align_corners=False` | Head isolated from semantic loss |
| **Motion** | `p, c, n`: `[B, 1, H, W]` | Concat temporal diffs + Conv(stride 1) + Conv(stride 4) | `f_mot`: `[B, 16, H/4, W/4]` | None | Full gradient flow to `prev, cur, nxt` |
| **Fusion** | `cat([f_app, f_mot])`: `[B, 64, H/4, W/4]` | `nn.Conv2d(64, 64, 1)` + `GroupNorm` + `GELU` | `fused`: `[B, 64, H/4, W/4]` | None | Full gradient flow |
| **Prototypes** | `fused`: `[B, 64, H/4, W/4]`, `p`: `[K, 64]` | Cosine similarity / temperature ($T=0.1$) + Softmax | `q`: `[B, K, H/4, W/4]` | `dim=1` (across $K$ prototypes) | Gradients flow to `fused` and `self.p` |
| **Pooling** | `q, fused, cur[:, 1:2], f_mot` | `pool_regions`: mass-weighted contraction + area fraction | `r`: `[B, K, 67]` | None | Mass clamped to $\min 10^{-6}$ |
| **Semantic Head** | `r`: `[B, K, 67]` | MLP: `Linear(67, 96)` $\to$ `GELU` $\to$ `Linear(96, 4)` | `logits`: `[B, K, 4]` | None | Full gradient flow |
| **Evidence** | `logits`, `evidence_logits`: `[B, K, 4]` | `logits + evidence_logits` | `logits`: `[B, K, 4]` | None | Direct additive shift |
| **Region Prob** | `logits`: `[B, K, 4]` | `F.softmax` | `prob`: `[B, K, 4]` | `dim=-1` (across 4 classes) | Valid categorical distribution |
| **Region Gate** | `prob`: `[B, K, 4]` | Threshold: $\text{top1} \ge 0.70 \land \text{margin} \ge 0.20$ | `valid_region`: `[B, K]` | Boolean thresholding | Non-differentiable boolean mask |
| **Low-Res Dense** | `q`: `[B, K, H/4, W/4]`, `prob`: `[B, K, 4]` | `torch.einsum("bkhw,bkc->bchw", q, prob)` | `dense_prob_low`: `[B, 4, H/4, W/4]` | None (sums to 1 across $C$) | Differentiable contraction |
| **Low-Res Valid** | `q`: `[B, K, H/4, W/4]`, `valid_region`: `[B, K]` | `torch.einsum("bkhw,bk->bhw", q, valid_region) >= 0.5` | `dense_valid_low`: `[B, H/4, W/4]` | Boolean thresholding | Non-differentiable boolean mask |
| **High-Res Dense**| `dense_prob_low`: `[B, 4, H/4, W/4]` | `F.interpolate(..., size=[H, W])` | `dense_prob`: `[B, 4, H, W]` | `bilinear`, `align_corners=False` | Differentiable upsampling |
| **High-Res Valid**| `dense_valid_low`: `[B, H/4, W/4]` | `F.interpolate(..., size=[H, W])[:, 0].bool()` | `dense_valid`: `[B, H, W]` | `nearest` | Non-differentiable boolean mask |
| **Discretization**| `dense_prob, dense_valid` | `argmax(1)` and `torch.where(dense_valid, label, UNKNOWN)` | `pseudo`: `[B, H, W]` | Discretization | **Detached discrete integers (`torch.long`)** |

---

## 3. Deep Diagnostic Analysis

### 3.1 Prototype Collapse and Prototype Permutation
1. **Permutation Ambiguity**:
   In `RegionPrototypeHead` (`src/self_audit_pseudolabel/system_v3.py:74-83`), prototypes `self.p = nn.Parameter(torch.randn(k, dim) * 0.02)` have no inherent semantic ordering. The prototype assignment $q$ is permutation-equivariant: any permutation $\pi \in S_K$ of the rows of `self.p` merely permutes the prototype channels of $q$.
2. **Semantic MLP Disconnect**:
   The semantic head `self.semantic` (`system_v3.py:109`) is an MLP applied to $r \in \mathbb{R}^{B \times K \times 67}$ along the prototype dimension. Without fixed slot grounding or semantic labels, `self.semantic` cannot know whether prototype index $k$ corresponds to LV cavity, myocardium, or background. Across different random initializations (evaluated in `reports/astra_v3/workers/A_evidence/probe_results.json`), prototype spatial mass varies by up to 8x across seeds, with no consistency in which index binds to which physical structure.
3. **Collapse Risk**:
   Because `q = logits.softmax(1)` performs softmax across prototypes at each pixel, a prototype vector that aligns slightly better with global foreground features will monopolize probability mass across large spatial regions. Although `mass.clamp_min(1e-6)` prevents division-by-zero in `pool_regions`, inactive prototypes (mass $< 10^{-5}$) have their pooled descriptors scaled by $10^6$ of tiny residuals.

### 3.2 UNKNOWN Behavior and Abstention Contracts
1. **Unseeded Default Abstention**:
   In `system_v3.py:122-124`, `valid_region = (top2[..., 0] >= min_prob) & ((top2[..., 0] - top2[..., 1]) >= min_margin)`. With default `min_prob=0.70` and `min_margin=0.20`, an unseeded semantic MLP produces logits near zero and probabilities near uniform ($1/4 = 0.25$). Our empirical probe (`probe_results.json`) demonstrated:
   - Mean top-1 probability across prototypes: **0.2682**.
   - Fraction of valid regions: **0.0000** (0.0%).
   - Fraction of accepted pseudo-label pixels: **0.0000** (0.0%).
   - Fraction of `UNKNOWN` (255) pixels: **1.0000** (100.0%).
2. **Student Loss Annihilation**:
   In `pseudo_supervision_loss` (`system_v3.py:188-199`):
   ```python
   if not bool(valid.any()):
       return final.sum() * 0.0
   ```
   When the teacher produces 100% `UNKNOWN`, `valid.any()` is `False`. The loss returns `0.0`, resulting in zero parameter updates to the student annotator.
3. **Abstention Contract**:
   This behavior is structurally consistent with the contract ("It may abstain; UNKNOWN is never trainable"), but reveals that out-of-the-box, **the teacher is permanently silent and provides zero supervision**.

### 3.3 Semantic Mixture Construction and Mixture Dilution at Boundaries
1. **Mixture Construction**:
   Line 125 computes `dense_prob_low = torch.einsum("bkhw,bkc->bchw", q, prob)`. Because $\sum_k q_{b,k,h,w} = 1$ and $\sum_c prob_{b,k,c} = 1$, `dense_prob_low` is guaranteed to be a valid convex combination of prototype probabilities.
2. **Mixture Dilution Vulnerability (Architectural Defect)**:
   In line 126:
   ```python
   dense_valid_low = torch.einsum("bkhw,bk->bhw", q, valid_region.to(q.dtype)) >= 0.5
   ```
   Validity is evaluated at the *prototype level* (`valid_region`), and then summed across prototypes!
   Consider a boundary pixel between Myocardium and LV cavity:
   - Prototype $k_1$ (Myocardium): $prob_{k_1} = [0, 0, 1.0, 0] \implies \text{valid\_region} = \text{True}$.
   - Prototype $k_2$ (LV): $prob_{k_2} = [0, 0, 0, 1.0] \implies \text{valid\_region} = \text{True}$.
   - At the boundary: $q_{k_1} = 0.5, q_{k_2} = 0.5$.
   - The resulting dense class probability is $P(\text{MYO}) = 0.5, P(\text{LV}) = 0.5$ (top-2 margin is **0.000**).
   - However, $\sum_{k \in \text{valid}} q_k = 0.5 + 0.5 = 1.0 \ge 0.5$.
   - **The system flags this ambiguous 50/50 boundary pixel as valid (`dense_valid = True`)!**
   - Line 129 then applies `argmax(1)`, arbitrarily assigning the boundary pixel based on floating-point tie breaking. This proves that region-level validation permits severe boundary label noise.

### 3.4 Reconstruction Shortcut and Head Decoupling
1. **Reconstruction Head Architecture**:
   In `AppearanceEncoder` (`system_v3.py:48-53`), `recon = self.recon(F.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False))` passes 1/4-resolution features $f$ through a single linear 1x1 convolution `nn.Conv2d(48, 1, 1)`.
2. **Gradient Decoupling**:
   Our gradient probe (`probe_results.json`) confirms that `self.recon.weight.grad` and `self.recon.bias.grad` are **`None`** during pseudo-label generation and backpropagation.
3. **Shortcut Failure Mode**:
   If an auxiliary pixel reconstruction loss $\| \text{recon} - cur[:, 1:2] \|^2$ were applied to train `AppearanceEncoder`, the single 1x1 conv forces feature map $f$ to encode raw pixel intensities and high-frequency edge textures rather than invariant semantic representations. Furthermore, `recon` reconstructs only the center slice `cur[:, 1:2]`, ignoring temporal dynamics.

### 3.5 Pooled Statistics: Missing Spatial and Geometric Biases
In `pool_regions` (`system_v3.py:86-95`):
- Features concatenated: $r_k = [f_k \ (64), \ \text{intensity}_k \ (1), \ \text{motion}_k \ (1), \ \text{area}_k \ (1)] \in \mathbb{R}^{67}$.
- **Missing Inductive Biases**:
  1. **Spatial Coordinates / Centroid**: Cardiac MRI follows strict anatomical topology (LV is central; RV is antero-lateral; myocardium encloses LV). No normalized $(x, y)$ centroid or coordinate meshgrid is passed to $r_k$.
  2. **Higher-Order Moments / Texture**: Only the 1st moment (mean) of intensity and motion is pooled. Variance, contrast, and edge gradients within the region are discarded.
  3. **Geometric / Topological Descriptors**: Myocardium is an annular ring (genus 1 topology); LV blood pool is a disk (genus 0); RV is crescentic. Neither compactness, eccentricity, nor boundary perimeter is provided.
  4. **Slice Position / Z-Coordinate**: Basal slices exhibit the LV outflow tract; apical slices exhibit diminutive blood pool. No slice z-index or cardiac phase fraction is encoded.

### 3.6 Motion Branch: Pixel Differencing vs. True Velocity/Optical Flow
In `MotionBranch` (`system_v3.py:56-72`):
```python
p, c, n = prev[:, 1:2], cur[:, 1:2], nxt[:, 1:2]
raw = torch.cat([c - p, n - c, (c - p).abs(), (n - c).abs()], 1)
```
1. **Mathematical Nature**:
   This is **pure temporal pixel intensity differencing**, not optical flow or displacement vector tracking. There is no formulation of optical flow $I(x+u, y+v, t+1) \approx I(x, y, t)$, nor any deformation grid.
2. **Empirical Flaws Confirmed by Interventions**:
   - **Intensity Flicker as Spurious Motion**: When a stationary cardiac image undergoes a global 1.5x brightness change with zero displacement, `MotionBranch` produces a feature norm change of **26.06** relative to zero motion. High-contrast static boundaries appear as moving structures.
   - **Homogeneous Moving Blood Pool**: In true laminar blood flow or translation within the LV cavity, pixel intensities are homogeneous ($c \approx p \approx n$). Therefore, $c - p \approx 0$. The interior of moving blood exhibits **zero motion signal**!
   - **Temporal Direction Inversion**: Swapping `prev` and `nxt` alters motion features by **96.95% relative L2 difference** (`mean_abs_diff = 0.2977`), because `raw` channels $[c-p, n-c, |c-p|, |n-c|]$ are mapped asymmetrically by the first convolution layer.

### 3.7 No-Seed Semantic-Head Gradient: The Semantic Grounding Void
1. **The Disconnect**:
   The semantic head `self.semantic` is an MLP that maps $r_k \in \mathbb{R}^{67} \to \mathbb{R}^4$.
   In an unsupervised setting, how does the MLP know that class index 0 is Background, 1 is RV, 2 is Myocardium, and 3 is LV?
   It cannot. Soft clustering (k-means or prototype learning) can group pixels by appearance and motion similarity, but the assignment of cluster indices to named medical categories requires external supervision or an anatomical prior.
2. **Gradient Flow**:
   - `CinePseudoTeacher` outputs `pseudo` as discrete integers (`torch.long`) via `label = dense_prob.argmax(1)`.
   - In `pseudo_supervision_loss(outputs, target, valid)`, `target` is frozen (`pseudo.detach()`).
   - The student receives gradients to its parameters (`student_has_nonzero_grads = True`).
   - The teacher receives **zero gradient** (`teacher_grads_from_student_loss` is completely `None`).
   - If the teacher is offline and frozen from inception, and its weights are randomly initialized, **the semantic head has never received a gradient signal to bind classes 0..3 to anatomical structures**.

---

## 4. Empirical Evidence & Synthetic Interventions

All synthetic probes were executed on CPU using `/Users/alvinluong/miniforge3/bin/python` with `PYTHONPATH=src`. Full quantitative outputs are preserved in `reports/astra_v3/workers/A_evidence/probe_results.json`.

### 4.1 Motion Interventions
| Intervention | Metric | Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Zero Motion** (`cur == prev == nxt`) | Feature Norm | 2.8774 | Non-zero due to GroupNorm and convolution bias terms |
| **Zero Motion** | Spatial Std Dev | 0.0446 | Nearly constant spatial feature map |
| **Temporal Swap** (`nxt` $\leftrightarrow$ `prev`) | Relative $L_2$ Difference | 0.9695 | **96.95% feature distortion**; lacks forward/backward consistency |
| **Temporal Swap** | Mean Absolute Difference | 0.2977 | Significant magnitude shift |
| **Channel Shuffle** ($[c-p, n-c]$ flipped) | Mean Absolute Difference | 0.3798 | First-layer filters do not share forward/backward weights |
| **Intensity Flicker** (1.5x brightness, $\Delta x=0$) | $L_2$ Norm vs Zero Motion | 26.0619 | **Massive false motion detected** from pure photometric variation |

### 4.2 Appearance Z-Channel Permutations
| Intervention | Metric | Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Z-Flipped** ($[z-1, z, z+1] \to [z+1, z, z-1]$) | Relative $L_2$ Difference | 1.0179 | Architecture has no through-plane reflection symmetry |
| **Z-Scrambled** ($[z-1, z+1, z]$) | Relative $L_2$ Difference | 1.0471 | Center slice identity is encoded solely by channel ordering |
| **Reconstruction MSE** | Initial MSE vs `cur[:, 1:2]` | 1.0019 | Standard untuned random initialization error |

### 4.3 Evidence Logits & UNKNOWN Abstention Sweep
| Configuration | Evidence Logits | Thresholds (`min_prob`, `min_margin`) | Valid Fraction | UNKNOWN Fraction | Mean Max Prob |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Unseeded Default** | `None` | $p \ge 0.70, m \ge 0.20$ | **0.0000** | **1.0000 (100%)** | 0.2682 |
| **Relaxed Thresholds**| `None` | $p \ge 0.50, m \ge 0.10$ | **0.0000** | **1.0000 (100%)** | 0.2682 |
| **Random Evidence** | $N(0, 1)$ | $p \ge 0.70, m \ge 0.20$ | 0.0039 | 0.9961 (99.6%) | 0.3512 |
| **Strong Prior** | Cyclic +10.0 | $p \ge 0.70, m \ge 0.20$ | **1.0000** | **0.0000 (0%)** | 0.9999 |

*Empirical Proof*: Out-of-the-box, the teacher abstains on 100% of pixels. It cannot produce a single valid pseudo-label pixel without external `evidence_logits`.

### 4.4 Prototype Mass & Assignment Entropy Across Seeds
Evaluating 5 independent random seeds ($B=2, K=12, H=64, W=64$):
- Mean normalized assignment entropy: **0.7819** (where 1.0 is uniform over 12 prototypes).
- Spatial mass per prototype (Seed 1): spans from **6.78** (Prototype 3) to **52.46** (Prototype 6) — a **7.7x imbalance**.
- Spatial mass per prototype (Seed 42): Prototype 0 has mass **50.33**, whereas in Seed 1, Prototype 0 has mass **7.94**.
- *Conclusion*: Prototypes lack canonical semantic identities; indexing is random across initializations.

### 4.5 Boundary Mixture Dilution Verification
In a controlled synthetic scenario (`probe_boundary_mixture_and_dense_validity` in `test_proposals.py`):
- Prototype 0 is confident Myocardium ($prob = [0.1, 0.0, 0.9, 0.0]$).
- Prototype 1 is confident LV ($prob = [0.1, 0.0, 0.0, 0.9]$).
- A boundary pixel with $q_0 = 0.5, q_1 = 0.5$ produces dense probabilities $[0.1, 0.0, 0.45, 0.45]$ (top-2 margin = **0.0000**).
- Result: `dense_valid_low` evaluates to **`True`** (mixture dilution defect confirmed).
- Proposed dense margin gate correctly evaluates to **`False`** (rejection of ambiguous mixture).

### 4.6 Gradient Flow & Backpropagation Verification
- Student parameters: All weights and biases receive valid non-zero gradients under `pseudo_supervision_loss`.
- Teacher parameters: Receive **zero gradients** (`None`) from student supervision.
- Reconstruction head: Receives **zero gradients** (`None`) from any semantic or pseudo-label loss.

---

## 5. The Central Question: Coherent Software vs. Named Semantic Learning Missing Signal

> **Main Question**: Is the Self-Audit v3 system a coherent piece of software, or is named semantic learning missing its fundamental supervisory signal?

### Definitive Conclusion:
The system is **coherent software with a missing named semantic signal**.

1. **Software Coherence**:
   - The tensor contracts, dimensional matching, group normalization divisibility, multi-turn dynamic window integration, and resource profile configurations (`compact`, `balanced`, `accurate`) compile and execute flawlessly.
   - The separation between teacher (`CinePseudoTeacher`) and student (`AdaptiveAnnotationStudent`) is clean and adheres to self-training design patterns.
2. **Missing Semantic Signal**:
   - The teacher contains **no mechanism to ground anonymous appearance/motion clusters into named cardiac anatomy** (Background, RV, Myocardium, LV).
   - Because the semantic head is randomly initialized and detached from ground truth, it produces uniform probabilities (~0.25), triggering the reliability gate to mark **100% of pixels as UNKNOWN**.
   - `AdaptiveAnnotationStudent` then trains on a null target (`loss = final.sum() * 0.0`).
   - To make this software learn named cardiac anatomy, one of two missing signals must be introduced:
     1. An explicit **anatomical evidence engine** feeding `evidence_logits` (e.g. blood-pool brightness, myocardial wall thickness, circularity priors, ventricular topological ordering).
     2. A **sparse seed supervision mechanism** (e.g. 1-click or 1-slice anatomical anchors).

---

## 6. Concrete Proposals & Decisive Regression Tests

*(All proposed tests are verified in `reports/astra_v3/workers/A_evidence/test_proposals.py` without modifying root tests).*

### 6.1 Dense Margin Gating Proposal
**Problem**: Region-level gating validates ambiguous 50/50 mixture pixels at boundaries (`system_v3.py:126`).
**Fix**: Apply the margin check directly on the interpolated dense probability map:
```python
# Proposed replacement for lines 126-128:
dense_prob = F.interpolate(dense_prob_low, cur.shape[-2:], mode="bilinear", align_corners=False)
dense_top2 = dense_prob.topk(2, dim=1).values
dense_valid = (dense_top2[:, 0] >= min_prob) & ((dense_top2[:, 0] - dense_top2[:, 1]) >= min_margin)
```

### 6.2 Motion Invariance & Normalization Proposal
**Problem**: Pure brightness flicker generates massive spurious motion; time reversal distorts features by 97%.
**Fix**:
1. Symmetrize the temporal difference channels: pass forward difference $(n-c)$ and backward difference $(c-p)$ through a shared 1-channel weight branch.
2. Normalize differences by local image gradient: $\Delta I / (\|\nabla I\| + \epsilon)$ to distinguish true motion edges from illumination flicker.

### 6.3 Decisive Proposed Regression Tests
Implemented in `reports/astra_v3/workers/A_evidence/test_proposals.py` (5/5 passing):

1. `test_teacher_abstains_100_percent_without_evidence_seed`:
   Verifies that an unseeded teacher produces 100% `UNKNOWN` (255) rather than hallucinating bogus labels.
2. `test_strong_evidence_enables_valid_pseudolabels`:
   Verifies that providing valid `evidence_logits` successfully activates valid pseudo-labels across all 4 classes.
3. `test_motion_branch_is_pixel_difference_not_optical_flow`:
   Falsification test showing that pure brightness flicker induces $>1.0$ $L_2$ norm distortion in motion features.
4. `test_reconstruction_head_is_isolated_from_semantic_gradient`:
   Verifies that `appearance.recon` has `grad is None` under semantic supervision.
5. `test_boundary_mixture_dilution_vulnerability`:
   Demonstrates the region-gating flaw at 50/50 boundary transitions and confirms the dense margin gate fix.

---

## 7. Segregated Classifications

### Facts (Directly Verified in Code & Execution)
- Base commit `d4f503b7107eca4e8ac85f37a5dd01ec50479666` contains `src/self_audit_pseudolabel/system_v3.py` (199 lines).
- `CinePseudoTeacher` forward pass executes cleanly for all tested spatial dimensions ($16 \le H, W \le 256$).
- Motion branch computes $[c-p, n-c, |c-p|, |n-c|]$ without displacement estimation.
- Without `evidence_logits`, `CinePseudoTeacher` produces 100% `UNKNOWN` (255) masks.
- Boundary pixels composed of two high-confidence prototypes are marked `dense_valid = True` even when dense margin is 0.0.
- `pseudo_supervision_loss` returns `0.0` when `valid.any()` is False.
- Reconstruction head `recon` has no backward connection to semantic logits.

### Inferences (Logical & Domain Deductions)
- `CinePseudoTeacher` was designed as an empty scaffold awaiting an anatomical heuristic engine to populate `evidence_logits`.
- Without an external orientation/topology engine, self-training on this teacher will result in dead models (0 gradients).
- The student model's adaptive compute profiles (`compact`, `balanced`, `accurate`) cannot demonstrate empirical value until the pseudo-label source produces valid segmentation masks.

### Proposals (Architectural Recommendations)
- Replace region-level gating with dense probability margin gating.
- Implement an explicit anatomical evidence engine that injects topological priors (LV inside MYO, RV adjacent to LV) into `evidence_logits`.
- Symmetrize the motion difference channels to ensure time-reversal equivariance.
- Connect the reconstruction head to an explicit self-supervised pretraining loss or remove it to save memory.

### NOT RUN (Explicitly Bounded / Prohibited Tasks)
- **Full ACDC dataset evaluation**: NOT RUN (prohibited; no manual GT access, no GPU compute).
- **End-to-end student training**: NOT RUN (would train on null loss).
- **Unsupervised optical flow registration training**: NOT RUN (out of scope).

---

## 8. Commands & Reproducibility Evidence

To reproduce all findings, run the following commands from the repository root:

```bash
# 1. Verify environment and existing tests
PYTHONPATH=src /Users/alvinluong/miniforge3/bin/pytest tests/test_pseudolabel_system_v3.py

# 2. Run diagnostic probe suite (generates probe_results.json)
PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python reports/astra_v3/workers/A_evidence/probe_system_v3.py

# 3. Run proposed regression test suite (5 passing regression tests)
PYTHONPATH=src /Users/alvinluong/miniforge3/bin/pytest reports/astra_v3/workers/A_evidence/test_proposals.py
```

All raw numerical probe outputs are archived at:
`reports/astra_v3/workers/A_evidence/probe_results.json`.
