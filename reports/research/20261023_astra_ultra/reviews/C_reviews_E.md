# C independent challenge of E experimental design

Reviewed file: `workers/E_design.md`, initial SHA256 `2ae68e1e5e2d885e26f4823c556ba09369b43141a64471f1ef653e6416c93536`, read 2026-10-23. This review uses immutable experimental source `8960b6baac4c6600553d900c303f192c2fcfa4f0` and C's local saved-table arithmetic. No patient compute or implementation change is authorized by this review.

Verdict on that snapshot: the proposed scalar restriction with `m*lambda`, source/config separation, numerical qualification before patient interpretation, new-schedule identity, exposed-VAL restrictions and absent-resume handling are sound. Two material phrasings need correction before synthesis; three design clarifications improve identifiability and stopping-rule interpretation. They were sent directly to E.

## Material findings

1. **Actual targets/candidates cannot remain identical across independent EMA training trajectories.** E:49 says all arms start with identical supervised references, q and directions; E:71/81 then describes independent fresh P/S/J training. The initial state can match, and the data stream/policy can match, but `engine.py:163–202` commits method-specific updates and then updates that arm's EMA. After divergence, theta_s, q, candidate classes and directions may differ even for identical images. Clarify two estimands: same-state frozen-transaction J/S/P comparison for direct allocation mechanics; full-training J−S policy contrast with teacher/candidate divergence as mediated outcomes. Do not quietly freeze a shared teacher, which would create another method.

2. **Observed unguarded time is not a guarded-run ceiling.** E:77 calls the reported13.47h/10k instrumented EMA-block cost a rough planning ceiling. Guarded methods add solver and potentially multiple finite-trial forwards; source `decoder_control.py` permits backtracking, and no full guarded timing is reported. Retain13.47h as an observed timing reference only. The report correctly requires a future explicit GPU-hour cap from profiling; no new run is currently authorized.

## Clarifications

3. **Name the absent-reference omission policy, not mathematical definedness.** E:32 says average “defined frame Dice.” In source `evaluation.py:23,85–91`, all gt=0 cases become None, including gt=0/pred>0 where ordinary Dice is mathematically0. Say reference-present frame Dice under the explicit omission policy. Each frame is a 3D ZHW volume. Retain absent-frame FP as a separate endpoint; the compact current CSV cannot reconstruct those missing frame supports.

4. **Crossing initialization seeds needs an explicit schedule seed rule.** E:85 proposes2 subset×2 training seeds. Existing `engine.py:125–128` uses the same seed for initialization and data RNG. E already marks separate schedule RNG as proposed infrastructure. To interpret initialization effects, hold the prospective data schedule fixed across initialization seeds inside each subset, with explicit independent initialization and schedule seed/hash. Otherwise label the second factor a joint training-randomness block, not isolated initialization variability. Sharing a schedule within method pairs is necessary but not sufficient to isolate initialization across blocks.

5. **No multi-source support makes an allocation test uninformative.** E:75 proposes no additional allocation study if S/J coincide on every multi-source transaction. Define the zero-multi-source case explicitly: no observed opportunity to identify allocation freedom under the recipe. It can justify resource triage but is not evidence that scalar and joint allocation are equivalent when multiple meaningful sources exist. Log source-class support before interpreting this gate.

## Scope agreement and residual limits

- Five fresh P/S/J/extra-supervised/full-EMA arms can support the stated J−S and practical comparisons against those controls. A sixth supervised-only run is not mandatory for a narrow J−S claim. However, the within-transaction supervised reference is not a completed supervised-only training trajectory; without a fresh matched supervised endpoint, do not claim the new whole policy beats supervised-only. Old supervised scores are context after schedule changes.
- S is the decisive simple comparator. Independent per-class allocation does not answer whether useful behavior is just aggregate shrinkage plus finite checking. A direct audit-gradient family is a valuable subsequent label-use control, not a replacement for same-direction S.
- The72-step pilot's4/64 acceptance says neither that the full1k-warmup recipe will stall nor that nonlinear backtracking caused46 candidate-bearing zero updates. E correctly separates causes and asks for moments/effective displacement.
- Proposed1pp continuation margins are explicitly resource decisions. They are not clinically meaningful thresholds, powered sample-size choices, or confirmation. Reusing exposed VAL18 for the screen and replication preserves development status; a separate frozen cohort is still required.
- New shuffled schedules, scalar arm, numerical repair, effective-delta diagnostics and shadow audits all require implementation/review. The current CLI illustration cannot execute the full proposal; E correctly labels it incomplete.

Source evidence and arithmetic are independently documented in `workers/C_empirical.md`, `evidence/C_recomputation.json` and `probes/C_recompute.py`. Review did not re-run patient inference, inspect remote jobs or treat report hashes as payload certification.

## Resolution

Final reviewed SHA256: `50ffc8ad2fa146cd072b2941f4d91c1f81251bcb1b9cad7a3c03bc8d8becd56b`. The intermediate revision `82d317cf657c8bf5bb468fd37894a6c4e5ac9d253bbc806a954be64b9616bb9d` was also read.

All five initial findings are resolved. E now distinguishes a frozen same-state comparison from total endogenous-EMA policy effects; labels13.47h as an observed timing reference; states the reference-present omission policy; fixes a separate schedule seed and preserves the schedule across initialization seeds; and treats zero multi-source opportunities as uninformative.

The revised Gate1 is one fresh P trajectory with functional P/S/J replay at each of144 post-warmup states, followed by only the P live commit and one EMA update. This is a smaller **conditional mechanism** experiment, not three on-policy pilots or a forecast of J/S efficacy. C additionally challenged two consequences of that revision: guarded-risk aborts must apply only to S/J accepted replay outcomes, since P is deliberately unguarded; and J replay acceptance must not be called a live J commit. The final snapshot resolves both. The later on-policy five-arm screen remains a separate proposed study with all costs and source changes declared.

Final disposition: **PASS for evidence-bounded research design, not implementation readiness, training authorization, statistical power, or empirical efficacy.** No unresolved material finding remains in the reviewed design snapshot. C's empirical report is also final after F's independent challenge and retains external-GPU versus locally-recomputed evidence labels.
