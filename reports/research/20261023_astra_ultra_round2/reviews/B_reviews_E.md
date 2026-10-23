# B independent review of E's sequential-identification example

Reviewer `/root/astra_theory`. **Accept after the recorded clarifications.** The construction is a valid manufactured affine-probability example of a fixed-reference transaction contrast failing to identify a different-support or complete-policy outcome. It is not an implementation of the native softmax/CE pipeline.

## Snapshots reviewed

- Initial `workers/E_identification.md`: SHA256 `77d1c9c51c83bb312d340fc425a838ed389676c72504c0f4bd79367f8cc12f9a`.
- Initial `probes/E_identification.py`: SHA256 `ebd6984c77d7c668dc60383f9882481331a06a393b0ad5d666354d867d479217`.
- Revised report after B challenge/reply: SHA256 `a6fcc6719eae9da7ff0c74abbb10f6b507fa4a2b7eeb03ad49e3991012410522`.
- Revised E probe: SHA256 `c02f06b18fa8d34dc73a7a723dc85192db476a38f634efcb71ca7248c6766666`.
- Revised E receipt: SHA256 `45a96315cede772e729b62164c8a3c7f3b949fdddbe4d4c610638a049f9b8d10`.

Independent review executable: [`../probes/B_review_E.py`](../probes/B_review_E.py); receipt [`../evidence/B_review_E.json`](../evidence/B_review_E.json). It loads E's functions without running E's writer/main, never edits E artifacts, and writes only B-owned evidence. It was rerun after E's clarification changes. **PASS, 12 branches**: P/S/J, ordinary/frozen-teacher intervention, and two steps. No production model/data imports or patient/GPU execution.

## Checks and conclusions

**Simplex validity — verified.** The toy map is `(.98-theta_x, theta_x, .01, .01)`. B checked every coefficient-box corner for both supports in each branch; each probability is nonnegative and sums to one. Since the map and updates are affine, every intervening box coefficient is also simplex-valid. This is stronger than checking only the selected endpoint. Parameter EMA is a convex combination of admissible parameter states and remains valid too.

**Exact local choices — algebraically checked.** Initially the A residual after J is .15 and candidate A responses are `.35,-.15`; at J=(1,0), objective gradient is `(-.0165,.0075)` for ridge .001 and mean over three groups. These are the global convex-box optimum signs, and J is guard-feasible. The scalar derivative at a=1 is `-.018`, so S=(1,1) is exact. Both second-step one-source optima likewise saturate at 1 and pass the A guard. The face enumeration is appropriate for these fixtures because the box optimum is guard-feasible. It should not be advertised as a general constrained-QCQP oracle merely because it filters the finitely enumerated box candidates by a guard.

**Teacher update — verified and clarified.** E's `commit` averages the actual two-dimensional parameter coordinates, not a separately learned probability tensor. For an affine map P, `P(decay*phi+(1-decay)*theta)=decay*P(phi)+(1-decay)*P(theta)` exactly in real arithmetic. B checked both supports at all twelve branches; floating-point discrepancies were below `2e-16`. This equality generally fails for a nonlinear softmax network; E now explicitly says so. The report's teacher coordinates and subsequent argmax choices are consistent with its declared parameter EMA.

**Tie behavior — verified.** The BG/RV crossing is at theta_B=.49. The initial and S teacher value .50 gives RV a .02 margin; J's .325 gives BG a .33 margin. The demonstrated outcome is not a tie-breaking artifact. A separate exact-tie check at .49 confirms Python's first-index `max` selects BG, but this case is not visited by the claimed trajectories.

**State and proposal dependency — valid only with the declared toy boundary.** The second proposal map depends on the teacher's B prediction and the iteration phase; it does not read a B reference label or score. The iteration phase is part of the known exogenous schedule, so this stateful rule is well-defined. It is handcrafted rather than a consequence of a CE gradient. P's second state visits the RV branch, while actual J's state visits the BG branch; P replay cannot identify the unvisited J branch without another assumption or experiment.

**Sequential outcomes — independently reproduced.** J's final B RV half-Brier is .405; S's is .10125. The direction of the first same-A contrast and final B contrast differs. These values demonstrate nonidentification for the constructed dynamic system, not the magnitude or prevalence of native patient failures. E's state/RNG sentinel checks validate its tiny harness only; B did not reinterpret them as native CUDA or arbitrary-module purity tests.

## Sent challenges, replies and verified resolutions

### B-E-1: “B cannot enter the teacher/live state” conflicted with the construction

**Challenge:** Initial prose excluded B itself from teacher/live-state decisions and described B as an endpoint only. Yet step two explicitly evaluates the teacher on B and uses that prediction to choose the live proposal. A completely untouched support interpretation would be false.

**E reply:** Accepted the distinction. Exclusion concerns **B reference labels and measured scores**; B image/feature support is an allowed unlabeled input to the declared second proposal map.

**Verified resolution:** Revised report lines 17 and 39 explicitly call this a label-hidden, **transductive** toy outcome support, not an untouched held-out patient. This preserves the logical counterexample while removing an overstatement. It does not authorize U-mask access in the native pipeline. **RESOLVED.**

### B-E-2: Parameter EMA versus probability EMA

**Challenge:** The report needed to explain why its coordinate averaging represents parameter EMA and why probability averaging happens to coincide here.

**E reply:** Accepted; added the affine-map explanation and explicit nontransfer to softmax networks.

**Verified resolution:** Revised line 58 states parameter EMA, affine equivalence and the limitation; independent numerical checks pass. **RESOLVED.**

### B-E-3: Contrast units and tie interpretation

**Challenge:** Values such as .03375 are RV-channel half-Brier contrasts, whereas the allocation objective averages three foreground groups. They should not be read as three-group mean contrasts. The actual argmax margin also needed to be explicit.

**E reply:** Accepted; stated all example scalar contrasts use RV half-Brier, with three-group mean contrasts divided by three, and reported the nonzero teacher margins.

**Verified resolution:** Revised lines 54 and 58 contain both corrections. **RESOLVED.**

## Verdict boundary

The revised example is internally coherent and useful for separating an off-policy fixed-reference mechanism test from an on-policy training comparison. It does not prove that J is usually worse, that EMA is generally harmful, or that a particular native CE generator produces the handcrafted directions. B's separate exact-softmax controller examples and F's generated-candidate examples address different questions; none should be silently substituted for the others.
