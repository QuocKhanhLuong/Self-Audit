# Source ledger

| Evidence | Location | Status |
|---|---|---|
| Required Step I / II profiles | Paper Tables 1 / 2 | VERIFIED_PAPER |
| Affinity and normalized Laplacian | Paper equations 1–6 | VERIFIED_PAPER |
| Step I Dice, Step II semantic evaluation | Paper Evaluation Methodology | VERIFIED_PAPER; detailed Step I specification UNRESOLVED |
| DINO key representation | Paper 2.1 | VERIFIED_PAPER |
| CAMUS exact sampled inventory | Paper Experimental Setup does not enumerate files | UNRESOLVED |
| Default configs (15-segment override, CRF, bbox fields) | Pinned reference configs | OFFICIAL_IMPLEMENTATION_DETAIL, not proven per-row paper recipe |
| Step II evaluator entrypoint and defaults | Pinned evaluation/segm_eval.py::main; configs/eval/defaults.yaml (eval_per_image=True, iou_thresh=0.0, void_label=0) | OFFICIAL_IMPLEMENTATION_DETAIL, not proven per paper row |
| Equal-count Hungarian IoU branch | Pinned evaluation/eval_utils.py::match (line 153) -> hungarian_match | OFFICIAL_IMPLEMENTATION_DETAIL |
| Unequal-count majority_vote_exclusive | Defined in pinned evaluation/eval_utils.py (line 202), called by match (line 172) | OFFICIAL_IMPLEMENTATION_DETAIL; paper-row applicability, remapping and aggregation UNRESOLVED/BLOCKED_PROTOCOL |

Erratum 2026-10-01: an earlier ledger revision recorded majority_vote_exclusive as
"called but missing". It is defined in the pinned file; the Step II gate remains
BLOCKED_PROTOCOL for the unresolved row-level evaluator details above.

Reference commit: d4ac44c60df18b921c590796f6994a4c8ac0726c.
Licensed DINO: facebookresearch/dino@7c446df5b9f45747937fb0d72314eb9f7b66930a,
Apache-2.0. Checkpoint weights must be separately hash-bound by the caller.
No official ACDC configuration was found in the inspected pinned reference.
This does not authorize adaptation before native verification readiness.
