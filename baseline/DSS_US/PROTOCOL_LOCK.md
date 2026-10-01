# DSS-US native protocol lock

Required reproduction: CAMUS, paper 2408.02043v1, Tables 1 and 2.
Implementation: independent reimplementation. Reference source is unlicensed and
read-only; no upstream implementation, evaluator, configs or functions are copied.

The committed profiles enumerate every required paper row and are hash-locked.
JSON syntax in .yaml files is intentional (valid YAML, deterministic stdlib parser).

Verified facts: last-attention DINO keys; paper equations 1–6; Step I has 15
segments; Step II fits segment representations across the cohort; CRF sharpens
outputs. A pinned Apache-2.0 DINO backbone is a separate licensed dependency.

UNRESOLVED/BLOCKED_PROTOCOL:
- Exact CAMUS 500-image inventory, views and five-image ED/ES sampling.
- Complete row-specific preprocessing/affinity recipes, Gaussian scales and
  nearest-neighbor policies; example-figure weights are not row configuration.
- Exact eigenvector inclusion, normalization and clustering initialization settings.
- Exact crop/mask/position embeddings, morphology and Step II per-row weights.
- CRF recipe applicability to each row and exact published implementation parity.
- Step I per-image segment Dice pairing, unmatched/background and aggregation.
- Step II: the pinned evaluator defines both match branches (eval_utils.match ->
  hungarian_match for equal counts, majority_vote_exclusive at eval_utils.py:202
  otherwise). Unresolved: which segm_eval.py branch/config (eval_per_image,
  iou_thresh, void_label) produced each Table 2 row, label remapping equivalence,
  conflict/unmatched handling and aggregation. Keep Step II blocked; never
  substitute another rule. (Erratum 2026-10-01: an earlier revision wrongly said
  majority_vote_exclusive was undefined in the pinned source.)

Evidence audit 2026-10-01 (reports/dss_us_sgscn_paper_protocol_evidence_20261001.md):
the Step I native evaluator is resolved from the paper and the official per-image
evaluation path and is implemented independently (evaluation/track_b/matching.py,
step1.py); Step I Track B is EVALUATOR_READY. Producers stay BLOCKED_PROTOCOL: no
CAMUS row recipe, cohort list or CRF parameter correspondence exists in the paper or
anywhere in the official repository history. Step II stays blocked on its evaluated
stage and the label-consistency computation.

No full paper profile is executable until evidence resolves its gate. Implemented
equation kernels and synthetic tests do not certify full CAMUS reproduction.
NATIVE_TRACK_A_STATUS is independently BLOCKED_ADAPTER (CAMUS includes LA).

Sources: https://arxiv.org/html/2408.02043v1
https://github.com/alexaatm/UnsupervisedSegmentor4Ultrasound/tree/d4ac44c60df18b921c590796f6994a4c8ac0726c

PAPER_FAITHFUL_REIMPLEMENTATION profiles (`*_paper_faithful.yaml`, src/dss_us/paper_faithful.py)
compose the paper equations and row components with every unstated value required. They stay
blocked: the end-to-end CAMUS runner is not wired and the canonical environment has no dense-CRF
backend. See reports/dss_us_sgscn_paper_faithful_20261001.md.
