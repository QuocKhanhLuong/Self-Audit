# Self-Audit — Living Architecture & Technical Decisions

> This file is the source of truth for the refactored Self-Audit baseline. It is updated after each technical decision batch.

## 1. Research Task

The target task is **medical semantic annotation with self-auditing**, not a standalone segmentation pipeline.

The system must:

1. generate an initial semantic annotation,
2. refine the annotation through stage-wise reasoning,
3. audit whether a proposed correction improves or regresses the current annotation,
4. accept or reject the correction,
5. repeat for a small fixed number of refinement stages or halt.

Ground truth (GT) is available during training but is **not required during inference**.

## 2. Naming / Positioning

The refactored baseline is **not classical Mixture-of-Experts (MoE)** because there is no learned router choosing among multiple competing experts at a stage.

Preferred architectural description:

- stage-wise expert refinement,
- recurrent/shared annotation expert,
- counterfactual self-auditing.

The novelty target is the **transition-level counterfactual auditor**, not variable depth itself.

## 3. Baseline End-to-End Flow

```text
Input medical image X
        |
        v
Shared Image Encoder
        |
        v
Image feature representation H
        |
        v
Initial Annotation Head
        |
        v
Initial annotation A^0
        |
        v
+------------------------------------------------------+
| Refinement stage t                                   |
|                                                      |
|  Shared Annotation Expert                           |
|    inputs: H, A^t, stage context, previous audit    |
|    recurrent depth: L_t                             |
|          |                                           |
|          v                                           |
|    candidate annotation A_tilde^(t+1)               |
|          |                                           |
|          v                                           |
|  Counterfactual Audit Expert                        |
|    compares: A^t -> A_tilde^(t+1)                   |
|    outputs:                                          |
|      - local improvement/regression evidence C^t    |
|      - global improvement score DeltaQ^t            |
|          |                                           |
|          v                                           |
|    accept / reject candidate                         |
|          |                                           |
|          v                                           |
|    next state A^(t+1)                               |
+------------------------------------------------------+
        |
        v
Final semantic annotation A*
```

## 4. Module A — Shared Image Encoder

**Status:** TECH TBD

Responsibility:

- encode the input medical image into reusable image features `H`,
- provide shared visual evidence for both annotation and auditing,
- avoid duplicating a large backbone between the Annotation Expert and Audit Expert.

To decide:

- 2D / 2.5D / 3D input mode,
- backbone family,
- pretrained vs from-scratch initialization,
- which feature scales are exposed,
- frozen vs trainable behavior.

## 5. Module B — Initial Annotation Head

**Status:** TECH TBD

Responsibility:

- map `H` to the initial semantic annotation `A^0`,
- provide the starting state for recurrent refinement.

The initial annotation should be a lightweight proposal rather than a separate heavy model.

To decide:

- annotation representation,
- decoder/head architecture,
- logits/probability/state format,
- supervised training objective.

## 6. Module C — Shared Annotation Expert

**Status:** ARCHITECTURAL DIRECTION LOCKED; TECH TBD

There is one **shared Annotation Expert** reused across refinement stages.

It is not a different independent expert per stage.

Inputs conceptually include:

```text
H                  image features
A^t                current annotation state
stage embedding    which refinement stage is active
C^(t-1)            previous audit evidence, when available
```

Output:

```text
A_tilde^(t+1)      candidate annotation
```

### Stage-dependent depth

The Annotation Expert may use a different computational depth `L_t` at each stage.

Preferred implementation direction:

```text
single shared expert block E_theta
        |
        +-- repeated L_t times
```

rather than allocating a separate parameter-heavy expert for each stage.

Example depth schedules to ablate later:

```text
fixed:       [2, 2, 2]
increasing:  [1, 2, 3]
decreasing:  [3, 2, 1]
```

Variable depth is treated as an architectural choice, not the primary novelty claim.

To decide:

- internal expert block/operator,
- recurrent state representation,
- stage conditioning mechanism,
- audit-feedback injection mechanism,
- final depth schedule.

## 7. Module D — Counterfactual Audit Expert

**Status:** CORE MECHANISM LOCKED; TECH TBD

The auditor evaluates a **transition**, not merely an isolated annotation confidence.

Input concept:

```text
H
A_previous
A_candidate
DeltaA = A_candidate - A_previous
```

The core question is:

> Did this candidate correction actually improve the current annotation?

Outputs:

```text
C        local improvement / regression evidence
DeltaQ   global transition quality / improvement score
```

The auditor does not require GT during inference.

## 8. Counterfactual Audit Training

**Status:** MECHANISM LOCKED; TECH TBD

GT is used only as a **training-time oracle**.

For a current annotation `A`, construct nearby counterfactual candidates:

```text
A+   a locally improved candidate
A-   a locally degraded candidate
A?   optional hard/neutral candidate
```

GT determines relative quality during training, for example:

```text
quality(A+) > quality(A) > quality(A-)
```

The auditor learns to identify which transition is beneficial without receiving GT at inference.

Important design constraint:

- counterfactuals should be **local edits around the model prediction**,
- do not simply replace predictions with GT,
- avoid trivial leakage that lets the auditor learn synthetic shortcuts.

To decide:

- positive counterfactual generator,
- negative counterfactual generator,
- hard/neutral candidate strategy,
- local evidence supervision,
- ranking/preference objective,
- global audit score definition.

## 9. Module E — Accept / Reject Self-Audit Loop

**Status:** LOGIC LOCKED; TECH TBD

For each refinement stage:

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
 +--> improvement -> accept A_candidate
 |
 +--> regression  -> reject and retain A^t
```

Accepted or retained state becomes `A^(t+1)`.

The next stage receives previous audit evidence so that refinement can focus on suspicious regions.

To decide:

- hard threshold vs learned/calibrated acceptance,
- local vs global acceptance,
- maximum number of stages,
- early halting criterion,
- behavior under ambiguous audit scores.

## 10. Training Objectives

**Status:** HIGH-LEVEL ONLY

Keep the first implementation minimal.

Expected loss families:

```text
L_annotation   supervised annotation objective
L_rank         counterfactual preference / transition ranking
L_local        local improvement-regression evidence objective
```

Conceptually:

```text
L_total = L_annotation
        + lambda_rank * L_rank
        + lambda_local * L_local
```

Exact formulations remain TBD.

## 11. Inference Contract

Inference must not require GT.

```text
X
 -> encoder
 -> initial annotation
 -> [annotation refinement -> counterfactual audit -> accept/reject] x T
 -> final annotation
```

Expected inference outputs eventually include:

- final semantic annotation,
- audit evidence map(s),
- accepted/rejected transition trace,
- stage-wise audit score,
- optional uncertainty / abstention status.

## 12. Implementation Policy

We will finalize and implement the system module-by-module in this order:

1. Shared Image Encoder
2. Initial Annotation Head / annotation state representation
3. Shared recurrent Annotation Expert
4. Counterfactual candidate generation
5. Counterfactual Audit Expert
6. Audit training losses
7. Accept/reject refinement loop
8. End-to-end training schedule
9. Evaluation / ablations

After each decision batch, this `docs.md` must be updated before or alongside code implementation.

## 13. Repository Migration Note

The repository currently contains legacy SpecUMamba / S3R-Net cardiac segmentation code. Do not assume that all existing modules belong to the new Self-Audit baseline.

Legacy code should only be reused when a component is explicitly selected during the new technical design process.

---

### Current Decision Checkpoint

Locked:

- task = semantic annotation + self-audit,
- no classical MoE routing/pool in the refactored baseline,
- one shared Annotation Expert reused across stages,
- stage-dependent recurrent depth is allowed,
- one Counterfactual Audit Expert,
- auditor evaluates `A_previous -> A_candidate`,
- GT used as training-time oracle only for counterfactual supervision,
- GT-free audit at inference,
- accept/reject loop drives self-correction.

Next technical decision: **Shared Image Encoder**.
