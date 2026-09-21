# Self-Audit v3: pseudo teacher, adaptive annotator, and bounded system path

This is research code. It preserves prior C0/C1 evidence and makes **no ACDC >=0.91 claim** until an independent post-freeze evaluation supports it.

## 1. Offline pseudo-label teacher

```text
full cine MRI [T,Z,H,W]
  -> lazy image-only full-cine loader
  -> 2.5-D appearance encoder (z-1/z/z+1)
  +  unsupervised pairwise registration (t-1/t/t+1)
  -> K anonymous appearance/motion prototypes
  -> auditable region evidence:
       border / low motion
       topology enclosure
       adjacency
       native left-right orientation
       image-boundary support
  -> semantic seed head BG/RV/MYO/LV
  -> confidence + margin abstention
  -> class prototype bank (accepted seeds only)
  -> temporal + cross-slice consistency gate
  -> frozen named soft pseudo-labels + validity
```

Relevant files:

- `data_v3.py`: ACDC/M&Ms 4-D image-only discovery and lazy LRU loading.
- `system_v3.py`: appearance, registration-motion, prototypes, semantic teacher, deployment student.
- `evidence.py`: detached auditable region evidence. No mask input.
- `evolution.py`: confidence-gated semantic prototype bank.
- `consistency.py`: conservative time/slice verifier.
- `losses_v3.py`: reconstruction, anonymous anti-collapse, registration, semantic seed losses.
- `trainer_v3.py`: bounded progressive teacher loop.
- `freeze.py`: immutable NPZ hashes + FROZEN.json.
- `scripts/train_pseudolabel_v3.py`: image-only training and frozen export.
- `scripts/evaluate_pseudolabel_frozen.py`: separate post-freeze ACDC evaluator.

The teacher is offline. Its compute is not deployment inference latency.

## 2. Final annotation model

```text
ED/ES 2.5-D image
  -> one lightweight encoder pass
  -> A0 BG/RV/MYO/LV
  -> optional canonical Dynamic Window refinement
  -> final mask
```

`AdaptiveAnnotationStudent` learns only from frozen pseudo-label pixels with validity=1.
UNKNOWN=255 is never a fifth trainable semantic class.

`scripts/train_student_v3.py` verifies frozen pseudo-label hashes before training.

## 3. Dynamic Window and adaptive compute

The final annotator reuses the canonical `AnnotationExpert` with
`audit_conditioning="feature_only"`. No runtime Auditor is required.

Profiles:

- `compact`: A0 only.
- `balanced`: one Dynamic Window turn.
- `accurate`: two recurrent turns.

`AdaptiveRuntime` computes the encoder and A0 exactly once, then selects a profile from
A0 normalized entropy under an explicit hardware/profile cap. The thresholds are engineering
configuration and must be locked before evaluation; entropy is not correctness.

Matched-profile accuracy and latency must be reported together.

## 4. Supervision boundary

Proposed training code accepts images/phase metadata only.

The semantic evidence engine may use:
- image intensity/edges,
- unsupervised motion,
- topology,
- native orientation,
- neighboring slices/frames,
- accepted prototype history.

It may NOT use:
- segmentation masks,
- GT crops,
- GT cluster matching,
- GT checkpoint selection,
- post-hoc class permutation fed back into training.

ACDC Info.cfg ED/ES indices remain human-curated phase metadata and must be disclosed.

The independent evaluator is allowed to open references only after all pseudo outputs and hashes
are frozen.

## 5. What is implemented vs still unproven

Implemented in code:
- full-cine ACDC/M&Ms image-only loader;
- unsupervised registration-motion branch;
- anonymous anti-collapse prototypes;
- auditable semantic evidence;
- reliability/UNKNOWN;
- class prototype evolution;
- temporal/cross-slice consistency gate;
- frozen export;
- separate evaluator;
- frozen-pseudo student trainer;
- Dynamic Window resource profiles;
- automatic entropy + resource-cap controller.

Still unproven:
- real-data pseudo-label quality;
- semantic identifiability sufficient for RV/MYO/LV;
- convergence;
- ACDC >=0.91;
- M&Ms transfer;
- matched low-resource speed/accuracy;
- superiority of Dynamic Window over matched-compute ordinary refinement.

## 6. Immediate scientific gate

Do **not** run a long final-student experiment until the pseudo teacher is frozen and independently
scored. The next decision is TEACHER_READY vs TEACHER_NOT_READY based on named RV/MYO/LV
precision/coverage and patient-level harm, not on training loss or self-agreement.
