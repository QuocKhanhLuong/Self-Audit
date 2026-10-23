# F — an actual hard-pseudo CE update can pass the guard and reduce same-audit Dice

Status: **VERIFIED_SYNTHETIC_CPU**, generated-direction loophole closed. This is a new executable case using the production candidate generator, not the round-one arbitrary displacement example. Parent HEAD `ccfde9bb99b6050e00ad94fbd6d57fcf9afc8b3f`; canonical source `637f38ced217fe1db2347a87924695533056d8f0`; current package/train fingerprint `3cca8cf4925ab7e98515ee3d731e2d4d0336f72129238d9efdc457c2af9735cf`. Production/config/baseline/round-one files were not changed. No patient, GPU, checkpoint, weights or Git-mutation work occurred.

## Result and exact scope

**Yes:** `make_candidates` can produce finite hard-pseudo CE gradient directions, with unique source IDs and the shared accepted denominator, for which the real `controlled_head_step(..., method='joint_guard')` decreases every actual audit Brier group yet decreases hard RV Dice on the same labeled audit pixels. Every foreground class has positive audit support: RV=2, MYO=1, LV=1. There are six audit pixels and two accepted U pixels, with distinct feature tensors for U and audit. No U reference mask is defined or used.

The teacher probability tensor is not manually assigned: an identical finite affine teacher copy generates it by softmax on U features. The affine head weights, U/audit features and audit labels are manufactured. Identical teacher/student state at this reference is a stipulated valid instantaneous input, not a demonstrated EMA training history. These are not native-backbone activations, cardiac data, or evidence about prevalence during patient training.

The original small-feature example uses pseudo step size approximately 9.603, well above the default 0.001. An exact augmented-feature representation reproduces its effective logit displacement at default 0.001, with large finite feature magnitude about 247.6. This is explicitly the same effective-step reparameterization, **not** evidence that typical default-LR activations or training trajectories exhibit the failure. A float32 replay also succeeds.

## Bounded discovery, with all attempts retained

`probes/F_realizable_search.py` fixed NumPy seed 20261023, a maximum of 2,000 trials, feature dimension 2, four U pixels, six audit pixels, audit labels `[0,1,1,2,3,0]`, and first-success stopping. It sampled finite weights/features and log-uniform pseudo step size in `[0.1,10]`. Each q came from the identical affine teacher copy; threshold 0.7, ridge 1e-6, zero risk budget, eight finite backtracks and tolerance 1e-8 match the canonical guard defaults.

The first qualifying example was **trial 292**. All 292 rows are saved in `evidence/F_search_trials.jsonl`:

| Outcome | Count |
|---|---:|
| Fewer than two accepted pseudo classes; no guard evaluation | 140 |
| No initial RV true positive; no guard evaluation | 88 |
| Guard evaluated but strict all-group Brier decrease plus RV Dice decrease did not both hold | 63 |
| First qualifying success; search stopped | 1 |

Thus 64 guard calls were evaluated under the qualifying support/initial-Dice filter. This selected first success supplies existence evidence only. Neither 1/292 nor 1/64 is a failure-prevalence estimate. No discarded alternative successful fixture or seed was selected. Full original tensors, q, actual generated directions, guard trials and search recipe are retained in `evidence/F_search_result.json`.

Postprocessing is separately recorded, not hidden as part of discovery: remove the two below-threshold U pixels (an exact denominator/direction-preserving reduction); apply one fixed rounding-to-three-decimals simplification; apply one exact feature representation change; replay the two rounded representations in float32. All five variants are reported. No iterative rounding-level search or optimization was performed. The final replay script asserts its success conditions.

## Small deterministic rounded fixture

The student is a biased 4-class affine 1x1 head `z=Wf+b`. Feature vectors are columns. Class order is BG,RV,MYO,LV.

```text
W = [[-0.974,  0.475],
     [-1.134,  1.224],
     [ 0.775, -1.788],
     [ 0.742,  0.170]]
b = [0.292, 0.263, -0.359, -0.007]

U = [[ 2.407, -2.527],
     [-0.645,  2.180]]
A = [[-0.577,  0.654, 0.175,  0.377, 0.037,  0.902],
     [-1.160, -0.233, 1.231, -0.200, 0.005, -0.015]]
y_A = [0, 1, 1, 2, 3, 0]

pseudo_lr=9.603, threshold=0.7, ridge=1e-6,
risk_budget=0, guard_tolerance=1e-8, backtracks=8.
```

The copied teacher yields the following U probabilities:

```text
q = [[0.00479054, 0.11828043],
     [0.00195310, 0.88112395],
     [0.72423811, 0.00000535],
     [0.26901825, 0.00059027]]
```

Both U pixels are accepted, one MYO and one RV. Canonical generator source IDs are `pseudo_class_1` and `pseudo_class_2`; class counts are `[0,1,1,0]`; both gradient terms divide by the same Z=2. The probe independently differentiates the complete accepted hard-CE loss and checks that the summed candidate displacement equals its ordinary SGD step. It does not supply any arbitrary update direction to the joint guard. Actual generator/guard code references: canonical `self_audit/control.py:132-159,162-228`.

## Actual finite outcomes

For the rounded float64 fixture:

| State/policy | Actual RV Brier | Actual MYO Brier | Actual LV Brier | Hard RV Dice |
|---|---:|---:|---:|---:|
| Frozen supervised reference | 0.130506632 | 0.145923704 | 0.167426301 | 0.666666667 |
| Joint guard | 0.123007156 | 0.136495376 | 0.166579875 | 0.500000000 |
| Exact aggregate scalar + same guard | 0.128000146 | 0.137628632 | 0.166673945 | 0.500000000 |
| Plain full aggregate | 0.132036293 | 0.135414643 | 0.229977962 | 0.666666667 |

Joint local coefficients are `[0.4264018107484825,0.12385989160017911]`. The true finite trial at scale 1 is rejected because LV risk is 0.168620371, above its reference. Scale 0.5 is accepted and gives the row above. The smallest actual group improvement is approximately 0.000846426, far larger than the 1e-8 acceptance tolerance. Thus this is not a tolerated risk regression or merely a linearized prediction.

The audit predictions change from `[2,2,1,2,0,3]` to `[2,2,1,2,1,2]`. RV TP stays 1 and FN stays 1, but FP increases from 0 to 1 (the true LV pixel becomes RV), so RV Dice falls from 2/3 to 1/2. All three group Brier risks still decrease. A probability-regression predicate can improve its balanced squared errors without preserving an argmax decision boundary.

The scalar comparator sums the **same** generated class directions, uses the canonical single-direction analytic `scalar_allocation` path via `independent_guard`, and retains the exact same finite predicate/backtracks. Its ridge is `m*lambda=2e-6`, the correct restriction of the joint regularizer to alpha=a·1. Its coefficient is 0.1075631712, and accepted scale 0.5. The scalar comparator also loses RV Dice, so this fixture is a limitation of transferring Brier protection to Dice, not evidence that joint allocation is uniquely defective or that scalar generally solves the problem. Plain's preserved RV Dice does not make plain better overall: its LV Brier increases substantially. No policy ranking follows from a selected existence example.

## Default raw LR, same effective logit path

Let eta0=9.603 and eta=0.001, r=eta0/eta=9603. For every U/audit feature f set

```text
f' = [sqrt(r) f, sqrt(r-1)]
W' = [W/sqrt(r), 0]
b' = b
```

Initial logits are unchanged. Because the affine bias is trained, a CE-gradient source changes an audit logit proportionally to `eta*(f_a·f_u+1)`. The lifted kernel satisfies

```text
eta * (f'_a · f'_u + 1) = eta0 * (f_a · f_u + 1).
```

Therefore every generated source's audit logit response and finite affine-head trial path are preserved. The replay measures a maximum audit logit-effect discrepancy of 1.33e-15 in float64. The representation has three features and maximum absolute feature 247.6331. The same joint alpha/scale and Dice loss result at the default raw pseudo LR, but it is deliberately the same large effective displacement represented differently. This demonstrates why raw LR alone is not a bound on function change; it does not establish realistic activation statistics or failure rates.

Both the small-feature/high-LR and lifted/default-LR rounded variants pass in float32 as well: same 2/3→1/2 RV Dice loss, all risks strictly decrease, joint scale 0.5. These dtype checks address this fixture's numerical robustness only; they are not a GPU/backend reproducibility claim.

## What this closes, and what remains unproved

Closed: round one's arbitrary-direction loophole. The new directions are emitted by the actual canonical hard-pseudo CE gradient generator from a valid finite teacher prediction, with the same model's distinct U/audit features, actual parameter bias gradients, common accepted denominator, unique canonical source IDs and all foreground audit classes present. Actual finite backtracking accepts the Dice-worsening commit. The proposition “canonical generated candidates plus actual classwise Brier protection imply same-audit Dice protection” is false.

Not established: any particular EMA training history reaching these weights/features, realistic backbone feature distributions, patient frequency, long-term harm or benefit, generalization, solver optimality, or superiority of J/S/P. The canonical docs already deny a Dice guarantee; this supports that limit rather than contradicting the documented transaction contract. Manufacturing labels/features to exhibit a counterexample is disclosed; they must not be presented as measured patient evidence.

The cheapest next mechanism check is now descriptive, not a stronger universal guarantee: on a separately authorized frozen provenance-complete patient replay, record exact generated directions, actual Brier, hard Dice, decision-margin crossings and effective update size. No such patient replay was authorized or run here. Do not tune thresholds or risk budgets using this selected fixture and imply that it resolves the outcome mismatch in general.

## Reproduce

```sh
rtk proxy python reports/research/20261023_astra_ultra_round2/probes/F_realizable_search.py
rtk proxy python reports/research/20261023_astra_ultra_round2/probes/F_realizable_replay.py
```

Discovery and all five fixed replays completed. The replay reads the saved exact search fixture, generates all directions anew, verifies summed-gradient parity, and writes complete actual trial probabilities/metrics to `evidence/F_realizable_replay.json`. CPU Torch 2.14.0, NumPy 2.4.6, SciPy 1.17.1. Search/replay receipts are synthetic artifacts. The completed numerical solver cross-review is reported separately in `reviews/F_reviews_A.md`.

## Independent review

A independently reconstructed the original four-U-pixel, rounded two-U-pixel and lifted default-LR fixtures and accepted the result within its affine-interface manufactured-existence scope. Independent full-CE/summed-candidate discrepancies were at most 2.22e-16, and the lifted audit logit-effect discrepancy was 1.33e-15. All three reproduced the accepted scale 0.5, strict all-group actual Brier decrease and RV Dice 2/3→1/2. See `reviews/A_reviews_F.md` (SHA-256 `66a48b1c9b9a9b389c62f53c89678808b90c6e4660663521c4e8673369d31b14`) and `evidence/A_review_F.json`. This independent replay does not establish a native-backbone feature realization or a full EMA training history.
