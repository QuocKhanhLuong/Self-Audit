# Self-Audit v3 system split: pseudo teacher vs final annotator

This branch is isolated research work. It does not modify the locked C0/C1 results and does not claim ACDC accuracy.

## 1. Offline pseudo-label path

```text
full cine MRI
  -> 2.5-D appearance encoder
  +  temporal motion branch
  -> K anonymous prototypes
  -> region-level semantic evidence
  -> high-precision BG/RV/MYO/LV seeds
  -> reliability gate / UNKNOWN
  -> freeze named soft pseudo-labels + validity
```

The pseudo-label path is a **teacher used only offline**. It is not the final deployment model. The current `system_v3.py` exposes `evidence_logits` so orientation/topology/cross-slice/cycle evidence can be added explicitly rather than silently invented.

## 2. Final annotation path

```text
ED/ES 2.5-D image
  -> lightweight encoder (one pass)
  -> A0 semantic logits
  -> optional shared Dynamic Window refinement
  -> final BG/RV/MYO/LV mask
```

The final model is `AdaptiveAnnotationStudent`. It trains from **frozen accepted pseudo-label pixels**; UNKNOWN pixels contribute no semantic loss.

## 3. Where Dynamic Window Attention belongs

Dynamic Window is **not** used to manufacture pseudo-label truth. It belongs in the final annotator as conditional compute.

The implementation reuses the existing `self_audit.models.annotation_expert.AnnotationExpert` with `audit_conditioning="feature_only"`. Therefore there is no runtime auditor in this scaffold, but the shared recurrent Dynamic Window block still reads the current image features + annotation state/entropy through the expert state.

Resource profiles are explicit:

- `compact`: A0 only, zero Dynamic Window turns.
- `balanced`: one refinement turn.
- `accurate`: two refinement turns; the existing expert uses stage-dependent internal depth.

This makes adaptation measurable: the same input/output contract is evaluated at different compute budgets. Accuracy and latency must be reported for the **same profile**.

## 4. Not implemented / not claimed

- No raw ACDC/M&Ms full-cine loader in this new namespace yet.
- No validated semantic evidence engine yet (native orientation, topology + image boundary, cross-slice and full-cycle correspondence are still research work).
- No long pseudo-label evolution loop, no prototype memory bank, no final training run.
- No claim of Dice >= 0.91 yet.
- No claim that random semantic logits are meaningful without external image-only evidence.

## Immediate scientific gate

Before long student training, freeze pseudo-label outputs and measure named RV/MYO/LV precision-versus-coverage with an independent evaluator. Dynamic Window and resource adaptation are downstream of a useful pseudo-label source, not substitutes for one.
