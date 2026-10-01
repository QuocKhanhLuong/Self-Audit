# Source ledger

| Evidence | Location | Status |
|---|---|---|
| Required Step I / II profiles | Paper Tables 1 / 2 | VERIFIED_PAPER |
| Affinity and normalized Laplacian | Paper equations 1–6 | VERIFIED_PAPER |
| Step I Dice, Step II semantic evaluation | Paper Evaluation Methodology | VERIFIED_PAPER; detailed Step I specification UNRESOLVED |
| DINO key representation | Paper 2.1 | VERIFIED_PAPER |
| CAMUS exact sampled inventory | Paper Experimental Setup does not enumerate files | UNRESOLVED |
| Default configs (15-segment override, CRF, bbox fields) | Pinned reference configs | OFFICIAL_IMPLEMENTATION_DETAIL, not proven per-row paper recipe |
| Equal-count Hungarian IoU branch | Pinned evaluation/eval_utils.py::match | OFFICIAL_IMPLEMENTATION_DETAIL |
| Unequal-count majority_vote_exclusive | Called but missing in pinned file | UNRESOLVED/BLOCKED_PROTOCOL |

Reference commit: d4ac44c60df18b921c590796f6994a4c8ac0726c.
Licensed DINO: facebookresearch/dino@7c446df5b9f45747937fb0d72314eb9f7b66930a,
Apache-2.0. Checkpoint weights must be separately hash-bound by the caller.
No official ACDC configuration was found in the inspected pinned reference.
This does not authorize adaptation before native verification readiness.
