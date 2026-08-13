# Self-Audit — Locked Baseline Architecture

> Source of truth for the new Self-Audit baseline. The legacy SpecUMamba/S3R code remains in the repository but is not part of this baseline unless explicitly reused.

## 1. Research Task

The target task is **cardiac semantic annotation with self-auditing**.

The system must:

1. produce an initial semantic annotation,
2. refine it through a small number of stage-wise refinement steps,
3. audit whether each proposed transition improves or regresses the current annotation,
4. accept or reject that transition,
5. use audit evidence to guide later refinement.

Ground truth is available during training, but **not required during inference**.

The baseline is **not classical Mixture-of-Experts (MoE)**. There is no learned router choosing among multiple competing experts. The architectural description is:

- shared recurrent Annotation Expert,
- stage-conditioned dynamic-window attention,
- counterfactual transition-level self-auditing.

Primary novelty target: **Counterfactual Self-Audit of annotation transitions**.

Secondary architectural identity: **Audit-guided Stage-Conditioned Dynamic Window Attention** inside the shared Annotation Expert.

---

## 2. Dataset / Input Contract

Initial implementation target: ACDC cardiac cine MRI with four classes:

```text
0 Background
1 RV
2 MYO
3 LV
```

### Input mode

Use **2.5D** input:

```text
input  = [slice z-1, slice z, slice z+1]
target = label(slice z)
shape  = [B, 3, 256, 256]
```

Boundary handling:

```text
z = 0     -> [z0, z0, z1]
z = last  -> [z-1, z, z]
```

Normalization:

```text
volume-wise percentile clipping
-> volume-wise z-score normalization
```

All geometric augmentations must be shared across the three slices.

---

## 3. End-to-End Baseline

```text
ACDC 2.5D image X
        |
        v
ConvNeXt-Tiny encoder
ImageNet pretrained
        |
        v
multi-scale features F1..F4
        |
        v
lightweight FPN
        |
        v
shared feature H  [B, 96, H/4, W/4]
        |
        v
Initial Annotation Head
        |
        v
A^0 = initial 4-class logits
        |
        v
+------------------------------------------------------+
| Refinement Stage t                                   |
|                                                      |
| Shared Annotation Expert E_theta                     |
| inputs:                                              |
|   H                                                  |
|   current annotation A^t                             |
|   stage embedding                                    |
|   previous audit evidence C^(t-1), if available     |
|                                                      |
| core operator:                                       |
|   Stage-Conditioned Dynamic Window Attention         |
|                                                      |
| recurrent depth: L_t                                 |
|                                                      |
| output: candidate annotation A_tilde^(t+1)           |
|        |                                             |
|        v                                             |
| Counterfactual Audit Expert                          |
| compares A^t -> A_tilde^(t+1)                       |
| outputs:                                             |
|   C^t       local fix/unchanged/regress evidence     |
|   DeltaQ^t  global transition improvement score      |
|        |                                             |
|        v                                             |
| accept / reject transition                           |
|        |                                             |
|        v                                             |
| next state A^(t+1)                                   |
+------------------------------------------------------+
        |
        v
T = 3 refinement stages
        |
        v
Final semantic annotation A*
```

---

## 4. Shared Image Encoder

### Locked choice

```text
Backbone: ConvNeXt-Tiny
Initialization: ImageNet pretrained
Input: 3-channel 2.5D stack
Trainability: fine-tuned end-to-end
```

Implementation target:

```python
timm.create_model(
    "convnext_tiny",
    pretrained=True,
    features_only=True,
)
```

Expose hierarchical features:

```text
F1 : 1/4 resolution
F2 : 1/8
F3 : 1/16
F4 : 1/32
```

Reason for this baseline choice:

- strong dense-prediction backbone,
- hierarchical multi-scale output,
- compatible with existing `timm` stack,
- avoids contaminating the main contribution with a domain-specific foundation model.

CineMA can later be tested as an encoder ablation, but the baseline must not depend on an ACDC-finetuned CineMA checkpoint.

---

## 5. FPN + Initial Annotation Head

### FPN

Project all encoder scales to 96 channels:

```text
F1..F4
-> 1x1 projections
-> top-down FPN fusion
-> H = [B, 96, H/4, W/4]
```

### Initial annotation head

```text
H
-> 3x3 Conv
-> normalization + activation
-> 1x1 Conv(num_classes=4)
-> bilinear upsample x4
-> A^0 logits [B, 4, 256, 256]
```

The recurrent state remains **soft logits / probabilities**, not argmax masks.

Initial supervised objective:

```text
L_initial = Dice + CrossEntropy
```

---

## 6. Shared Annotation Expert

There is exactly **one shared Annotation Expert** reused at all refinement stages.

The expert does not re-predict the full annotation from scratch. It predicts a **residual correction** to the current annotation.

Inputs:

```text
H                   shared image feature
A^t                 current annotation logits/probabilities
E(A^t)              entropy / uncertainty map
C^(t-1)             previous audit evidence, when available
stage_embedding      stage identity
iteration_embedding  recurrent step identity within the stage
```

Output:

```text
DeltaA               annotation residual
g                    spatial update gate in [0,1]
A_tilde^(t+1)        candidate annotation
```

Update:

```text
A_tilde^(t+1) = A^t + g * DeltaA
```

---

## 7. Core Operator — Stage-Conditioned Dynamic Window Attention

This is the custom attention mechanism used inside the Annotation Expert.

The window/support is **not a fixed square Swin window** and is not merely shifted between layers.

For every query token, the model predicts the geometry of the support it wants to inspect.

### 7.1 Dynamic support generator

For query token `q_i`, build a descriptor from:

```text
local image feature
current annotation feature
uncertainty
stage embedding
audit evidence
```

The support generator predicts:

```text
Delta c_i     window-center displacement
r_x, r_y      anisotropic spatial extent
phi_i         optional orientation
Delta p_ik    K residual sampling offsets
```

Use a canonical normalized K-point template inside the predicted window, then transform it by the predicted center/scale/orientation and add residual offsets.

Initial baseline:

```text
K = 8 sampled support points per query
```

The support can therefore be:

- local or displaced,
- small or large,
- anisotropic,
- oriented along anatomy/boundaries,
- increasingly exploratory in later refinement stages.

### 7.2 Audit-guided window adaptation

Audit evidence from the previous stage conditions the support generator.

Intended behavior:

```text
low-suspicion area
-> compact/local support

high-suspicion area
-> larger/displaced/more exploratory support
```

The model learns this behavior; do not hard-code a different fixed window size for every pixel.

Stage embeddings provide a global stage prior, while the audit map provides local spatial guidance.

### 7.3 Sampling

Sample from `H` with PyTorch bilinear interpolation:

```python
torch.nn.functional.grid_sample
```

No custom CUDA operator in the baseline.

### 7.4 Sparse attention / aggregation

For the K sampled features:

```text
query projection Q_i
sampled key projections K_ik
sampled value projections V_ik
```

Compute sparse attention only over the sampled support:

```text
alpha_ik = softmax(Q_i dot K_ik / sqrt(d) + relative_geometry_bias)
z_i = sum_k alpha_ik V_ik
```

Fuse `z_i` with the current annotation state and predict `DeltaA` + gate `g`.

### 7.5 Shared recurrent depth

Use the same expert parameters `E_theta` repeatedly.

Locked baseline schedule:

```text
L = [1, 2, 3]

Stage 1 -> E_theta x1
Stage 2 -> E_theta x2
Stage 3 -> E_theta x3
```

The weights remain shared across stages and recurrent iterations.

Variable depth is an architectural design choice, not the main novelty claim.

Ablations later:

```text
[2,2,2]
[1,2,3]
[3,2,1]
```

---

## 8. Counterfactual Audit Expert

The auditor evaluates a **transition**, not an isolated segmentation confidence.

Its core question is:

> Did the proposed correction make the current annotation better or worse?

### Inputs

Use detached shared image evidence plus both states:

```text
stopgrad(H)
P_previous
P_candidate
DeltaP = P_candidate - P_previous
entropy_previous
entropy_candidate
```

`P` denotes softmax probabilities.

The audit loss must not update the encoder/Annotation Expert through these inputs.

### Architecture

At 1/4 resolution concatenate the audit inputs with `H`, then:

```text
1x1 projection -> 96 channels
-> residual block
-> residual block
-> residual block
-> shared auditor feature
```

Two output heads:

```text
Local Audit Head  -> C
Global Audit Head -> DeltaQ
```

### Local output

For every location classify transition evidence as:

```text
FIX
UNCHANGED
REGRESS
```

### Global output

```text
DeltaQ = predicted global improvement score
```

Positive means the candidate is expected to improve the current annotation; negative means regression.

---

## 9. Counterfactual Training Targets

GT is a **training-time oracle only**.

It is never an input to the auditor and is absent at inference.

### 9.1 Local transition target

Using GT `Y`:

```text
error_previous  = prediction_previous != Y
error_candidate = prediction_candidate != Y
```

Local label:

```text
FIX       : previous wrong, candidate correct
UNCHANGED : same correctness state
REGRESS   : previous correct, candidate wrong
```

Use three-class cross-entropy or focal cross-entropy for `C`.

### 9.2 Global transition target

Compute foreground mean Dice for RV/MYO/LV:

```text
DeltaD = Dice(candidate, Y) - Dice(previous, Y)
```

The auditor predicts the **sign/order of improvement**, not simply the absolute Dice score.

Use signed ranking loss:

```text
L_rank = softplus(-sign(DeltaD) * DeltaQ)
```

Weight samples by `abs(DeltaD)`.

For near-neutral transitions:

```text
abs(DeltaD) < epsilon
```

encourage:

```text
DeltaQ -> 0
```

---

## 10. Counterfactual Candidate Generator

Counterfactuals must be **local edits around real model predictions**.

Do not use `A+ = GT`.

Generate four families.

### Positive counterfactual

Partially repair a real model error toward GT:

```text
repair only 30-60% of a selected false-positive / false-negative / wrong-class region
```

### Negative counterfactual

Corrupt an area that was previously correct:

```text
boundary erosion
boundary dilation
small component shift
hole insertion
false island
wrong-class swap
```

### Hard / neutral counterfactual

Simultaneously improve one area and regress another so total quality change is small.

### On-policy transition

Always include real candidates generated by the current Annotation Expert:

```text
A^t -> A_tilde^(t+1)
```

The auditor must not learn only synthetic corruption artifacts.

Initial synthetic sampling mix:

```text
40% positive
40% negative
20% hard/neutral
```

Then progressively increase the fraction of on-policy transitions during auditor/joint training.

### Soft probability-space editing

Counterfactuals should remain prediction-like.

Do not create obviously hard one-hot synthetic masks when the network normally emits soft probabilities.

Use soft interpolation inside an edit region:

```text
P_cf = (1-M) * P_original
     + M * ((1-alpha) * P_original + alpha * P_target)

alpha ~ Uniform(0.3, 0.8)
```

---

## 11. Accept / Reject Self-Audit Loop

For every refinement stage:

```text
A^t
 |
 v
Annotation Expert
 |
 v
A_candidate
 |
 v
Counterfactual Auditor
 |
 v
DeltaQ
```

Baseline rule:

```text
DeltaQ > tau -> ACCEPT candidate
else         -> REJECT and retain A^t
```

Initial `tau = 0`.

Calibrate `tau` on the validation split after the auditor is trained.

The accepted or retained state becomes `A^(t+1)`.

Audit evidence `C^t` is passed to the next Annotation Expert stage regardless of global acceptance so the next stage knows where the attempted transition was helpful or harmful.

Baseline uses exactly:

```text
T = 3 refinement stages
```

No learned early-halting module in the first implementation.

---

## 12. Training Schedule

### Phase A — Annotation network pretraining

Train:

```text
ConvNeXt encoder
FPN
Initial Annotation Head
Shared Annotation Expert
```

Auditor disabled/frozen.

Supervise every stage:

```text
A^0, A^1, A^2, A^3
```

Loss:

```text
L_ann = 0.5 L(A^0)
      + 0.7 L(A^1)
      + 0.8 L(A^2)
      + 1.0 L(A^3)

L(A) = Dice + CrossEntropy
```

During this phase, always propagate the generated candidate state so the expert learns refinement dynamics before audit gating is enabled.

### Phase B — Auditor pretraining

Freeze annotation network.

Use cached/current model predictions plus synthetic and on-policy counterfactual transitions.

Train:

```text
L_audit = lambda_rank * L_rank
        + lambda_local * L_local
```

### Phase C — Joint fine-tuning

Fine-tune the full pipeline with small learning rates.

Important gradient rule:

```text
Audit loss must not backpropagate through H, A_previous, or A_candidate into the annotation network.
```

Use detached inputs for the auditor path.

Accept/reject decisions are also detached/non-differentiable in the baseline.

The annotation network is updated only by annotation supervision; the auditor is updated by transition supervision.

---

## 13. Optimizer / Runtime Baseline

```text
Optimizer: AdamW
```

Starting learning rates:

```text
ConvNeXt encoder  3e-5
FPN/head          3e-4
Annotation Expert 3e-4
Auditor           3e-4
```

Schedule:

```text
5-epoch warmup
-> cosine decay
```

Runtime:

```text
image size: 256 x 256
AMP: enabled
batch size target: 8
fallback: batch 4 + gradient accumulation
```

---

## 14. Technical Stack

Reuse the existing lightweight stack:

```text
Python 3.10
PyTorch
timm
MONAI
einops
NumPy / SciPy
nibabel / SimpleITK
Albumentations
PyYAML
W&B / TensorBoard
```

Do not require in the baseline:

```text
mamba-ssm
custom CUDA deformable attention
Detectron2
MMCV
SAM / MedSAM runtime dependencies
```

---

## 15. Evaluation Contract

Primary annotation metrics:

```text
Dice
HD95
ASSD
Precision
Recall
```

Evaluate four core systems:

```text
A. Initial annotation only
B. Refinement with every candidate always accepted
C. Refinement with GT-oracle accept/reject (analysis upper bound only)
D. Refinement with learned Counterfactual Auditor
```

Desired ordering:

```text
D > A
D > B
D approaches C
```

### Audit-specific metrics

Global:

```text
pairwise improve/regress accuracy
AUROC improve vs regress
AUPRC
correlation(DeltaQ, actual DeltaDice)
```

Local:

```text
F1(FIX)
F1(REGRESS)
AUPRC(regression pixels)
```

System-level safety:

```text
harmful acceptance rate
beneficial rejection rate
net Dice gain after auditing
```

---

## 16. Required Ablations

Self-audit contribution:

```text
initial annotation only
refinement always accept
confidence-based acceptance
candidate-only quality auditor
transition auditor
transition auditor + local evidence
transition auditor + local evidence + counterfactual training
```

Annotation Expert:

```text
fixed local attention / convolution
vs dynamic-window attention
vs audit-guided dynamic-window attention
```

Depth:

```text
[2,2,2]
vs [1,2,3]
vs [3,2,1]
```

Auditor training distribution:

```text
synthetic only
vs synthetic + on-policy transitions
```

---

## 17. Planned Code Layout

```text
src/
└── self_audit/
    ├── data/
    │   ├── acdc.py
    │   └── transforms.py
    │
    ├── models/
    │   ├── encoder.py
    │   ├── fpn.py
    │   ├── annotation_head.py
    │   ├── dynamic_window.py
    │   ├── annotation_expert.py
    │   ├── auditor.py
    │   └── self_audit_net.py
    │
    ├── audit/
    │   ├── counterfactual.py
    │   ├── targets.py
    │   └── gate.py
    │
    ├── losses/
    │   ├── annotation.py
    │   └── audit.py
    │
    └── training/
        ├── train_annotation.py
        ├── train_auditor.py
        └── finetune_joint.py

configs/
└── self_audit/
    └── acdc_baseline.yaml

tests/
└── self_audit/
```

Do not remove or rewrite legacy `src/models/s3r/` while the new Self-Audit baseline is being brought up.

---

## 18. Locked Baseline Summary

```text
Task
  cardiac semantic annotation + self-audit

Input
  2.5D [z-1,z,z+1], 256x256

Encoder
  ImageNet-pretrained ConvNeXt-Tiny

Shared feature decoder
  lightweight FPN, C=96

Initial state
  soft 4-class annotation logits A^0

Annotation Expert
  one shared recurrent expert
  residual annotation update
  Stage-Conditioned Dynamic Window Attention
  audit-guided support geometry
  K=8 support samples/query
  PyTorch grid_sample
  depth schedule [1,2,3]

Auditor
  one Counterfactual Audit Expert
  evaluates A_previous -> A_candidate
  local FIX/UNCHANGED/REGRESS map
  global DeltaQ improvement score
  detached shared image features

GT usage
  training-time oracle only for annotation supervision,
  counterfactual generation, and transition targets

Inference
  no GT
  annotate -> refine -> audit -> accept/reject x3

Main novelty
  transition-level counterfactual self-auditing

Supporting architectural novelty
  audit-guided stage-conditioned dynamic-window attention
```

This specification is the implementation baseline until an explicit later decision changes it.
