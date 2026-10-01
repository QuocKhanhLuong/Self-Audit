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
- Step II source refers to majority_vote_exclusive without defining it in the
  pinned evaluator utility. Keep that branch blocked; never substitute another rule.

No full paper profile is executable until evidence resolves its gate. Implemented
equation kernels and synthetic tests do not certify full CAMUS reproduction.
NATIVE_TRACK_A_STATUS is independently BLOCKED_ADAPTER (CAMUS includes LA).

Sources: https://arxiv.org/html/2408.02043v1
https://github.com/alexaatm/UnsupervisedSegmentor4Ultrasound/tree/d4ac44c60df18b921c590796f6994a4c8ac0726c
