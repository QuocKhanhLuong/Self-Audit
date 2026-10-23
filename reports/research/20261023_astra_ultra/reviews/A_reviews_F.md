# A independent review of F

Reviewed `workers/F_adversarial.md`, `probes/F_adversarial_probe.py`, `evidence/F_probe_results.json`, and the added `evidence/F_pooled_patient_sensitivity.json`; checked the control, scheduling, and evaluation source at canonical `637f38c` and the already-inspected experimental diff at `8960b6b`. This review is read-only with respect to production and F's files. It does not rerun patient data or authenticate original training payloads.

**Verdict: the three requested challenges survive with two accepted wording corrections.** They establish a logical guarantee boundary and two protocol/endpoint facts. They do not establish a cause of the retained patient outcomes.

## 1. Brier-safe does not imply Dice-safe

F constructs strictly positive, finite four-class probabilities, embeds their logarithms in a real affine 1x1 head, and supplies the finite difference as one legal-shaped direction to `controlled_head_step`. The full production allocator and actual-trial guard choose alpha=1, scale=1. This is stronger than an abstract comparison of risk formulas: the implemented acceptance path permits the change. The initial RV argmax hits one of two positives; the final argmax hits neither. All three foreground Brier risks decrease, so the drop from Dice 2/3 to zero is compatible with every actual guard inequality.

The direction is deliberately constructed; F explicitly does not assert that the current EMA CE generator produces it. Therefore the precise result is a counterexample to an implication **from those Brier inequalities alone** to Dice safety. It does not prove how often a fully generated transaction violates Dice monotonicity, nor demonstrate a bug in a method that already disclaims Dice safety. F's wording now preserves this distinction. No change required.

No conflict exists with A's numerical liveness finding: a better numerical solver can make the stated Brier problem more faithfully solved without making Brier a Dice certificate. Numerical repair and scientific objective alignment are separate gates.

## 2. Round-robin pairing is permanently restricted at the declared budgets

Independent algebra: for a fixed U index `u`, visits occur at `t=u+k*N_U`. The supervised index is `(u+k*N_U) mod N_L`. It visits `N_L/gcd(N_L,N_U)` distinct L indices. At TRAIN80 budgets `(N_L,N_U)=(4,76),(8,72),(16,64)`, this equals one. Thus the same L index is always the supervised partner for that U index and, because `ai != li`, can never audit that U's proposals. The experimental batch sampler repeats the same patient index within its B52 batch; independent crops do not change this arithmetic.

This is a valid coverage limitation, not a leak or a mismatch between current method arms. It is also not proof of harm: all arms receive the same restricted schedule, and the numerical experiments have not compared alternate pairing policies. F says both explicitly.

Challenge sent to F: “changing seed” must mean training/audit RNG seed. A different label-subset seed can change patient membership and index identities, even though the one-partner property still holds at the same counts. F accepted and updated this wording. A follow-up control should record its independent scheduling RNG so that changed sampling order is not accidentally confounded with a wholly different augmentation stream.

## 3. The endpoint omits reference-absent frames from class Dice means

Source confirms `evaluation.py:23` returns null Dice whenever reference count is zero; `:85-90` excludes null values from within-patient and across-patient means. FP counts still appear in confusion statistics. Therefore the synthetic example's mean-available-frame Dice 1.0 versus pooled Dice 2/3 is correct, and the endpoint can hide FP-only frames **from this Dice average** while retaining them elsewhere.

Challenge sent to F: evaluation's per-frame confusion uses the complete `Z,H,W` mask (`evaluation.py:72-74`), so each frame metric is volumetric 3D Dice. The distinction is mean of available per-frame 3D ratios versus a ratio after pooling ED/ES volumes. F accepted and made this explicit. “Not pooled 3D” alone would have risked implying that the current endpoint is slice-wise Dice.

Second challenge: committed per-patient pooled counts already suffice to derive a pooled-per-patient sensitivity endpoint, although they cannot recover the number or identity of excluded frames. F added `F_pooled_patient_sensitivity.json`: the alternate derived foreground means remain ordered full EMA > extra-supervised > supervised > block EMA. This comparison is a sensitivity analysis of a different endpoint, not a replacement primary metric, not evidence that absent-frame omission occurred in these retained runs, and not a raw-prediction re-evaluation.

## 4. Additional checks and open limits

- F's full-versus-block optimizer attribution warning agrees with A's direct inspection. It should survive synthesis even when full EMA wins the development table.
- F's common-teacher evidence statement is correct: unique source IDs do not create statistically independent semantic evidence. They index class-partitioned terms of one hard-pseudo loss.
- The recomputed source hash can establish that a committed source snapshot matches receipt strings. It cannot establish that a missing weight payload actually came from that execution; F's qualification is appropriate.
- A's `ema_block` probe independently confirms removable computation: four JVPs and seven encoder forwards for an update equivalent to a single direct pseudo-SGD backward on the selected block. Consequently neither report should treat the retained block runtime as a lower bound on optimized block SSL cost.
- F's frozen replay proposal should add the global-ray scalar comparator and numerical-stability checks before interpreting rejection rates. The global-ray penalty must be declared: `M*lambda` gives the literal `alpha=beta*1` restriction of the existing vector objective.

Disposition: **ACCEPT WITH AMENDMENTS INCORPORATED**. No patient-outcome causal attribution or scientific efficacy certification is supplied by this review.
