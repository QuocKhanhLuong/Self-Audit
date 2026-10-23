# B independent review of A

Reviewer: `/root/astra_theory`. Scope: mathematical interpretation, small-response numerical finding, minimum controls, and claim limits. This is not a second complete data/security audit.

## Reviewed snapshots

- Initial A draft: `workers/A_source.md`, SHA256 `9acea78a75df16452f2c0f15fb48a9bcd3954fee8a7e7feaf5d06d50e38889c9`.
- Intermediate v2, timing correction checked: SHA256 `d05fe43809f400c17d1d45d522af46e2e8f49abf379c061c5aa2f7001b456990`.
- Revised v3 after challenge/reply: SHA256 `25401b86cbab3ee8cdd01a6dfd595c1b9735bd372aff11ba68888b066b34366f`.
- A probe receipt inspected: `evidence/A_source_probes.json`, SHA256 `b375915b14297e14261bbc5be5e14a33deb231be4e441fd70e3381cad38595dd`.
- Relevant source pinned at `637f38ced217fe1db2347a87924695533056d8f0`: `self_audit/control.py:22-80,120-129,202-217`; `self_audit/decoder_control.py:48-79`; `self_audit/engine.py:166-192`.

## Independent challenge and result

**Primary finding upheld.** A's abstract scalar moments are legitimate squared-response moments, but residual r=1 would not itself establish a finite softmax-head failure. A already includes a CE-derived finite p=.25 fixture; B independently constructed a different finite affine-head direction and computed its moments through the production `response_moments`. At logit magnitude `1e-6`, the exact scalar coefficient is `.2291666141493176`, while the production joint solver returns zero with `NUMERICAL_FEASIBLE`; exact attainable regularized objective gain is `2.625867454e-8`. The report's distinction between an allocation/interpretation failure and an actual-guard safety breach is therefore correct. It is not evidence that a particular patient run encountered this defect.

Reproduction: `rtk proxy python reports/research/20261023_astra_ultra/probes/B_counterexamples.py`; see `evidence/B_counterexamples.json`, `physical_small_scale`. B did not rerun A's entire probe or the repository suite. A's normalized scalar routine was read directly at `probes/A_source_probes.py:25-36`; subtracting a constant and dividing objective, gradient, constraints and Jacobian by one positive scale preserves its zero-budget feasible optimization problem. Its successful scalar sweep is a useful diagnostic, not general validation of a production solver repair.

## Challenge/reply resolution ledger

### B-A-1 — Unqualified KKT / projected-gradient gate

**Challenge sent:** A's v1/v2 recommendation to record “KKT or projected-gradient residual” could be misread as a universal QCQP certificate. For zero-budget quadratic constraints, a true optimum may fail constraint qualification and admit no KKT multipliers. With `q_1(alpha)=alpha^2/2-alpha`, `q_2(alpha)=alpha^2/2`, box `[0,1]`, and positive ridge, the only feasible point is zero, but the objective derivative is `-1/2` and nonnegative inequality multipliers cannot cancel it. A projected gradient on the box alone also ignores the quadratic feasible set.

**A reply:** Accepted the objection; agreed to use exact scalar gaps and independent feasible comparators, reserve KKT residuals for verified constraint regularity, and reject a box-only residual as a QCQP certificate.

**Resolution checked:** A v3 replaces the unqualified wording at its numerical gate and records the caveat. **RESOLVED.** No change to the actual implementation is claimed.

### B-A-2 — Receipt timing mismatch

**Challenge sent:** Initial A text quoted `.1191 s` / `.00495 s`, while its current regenerated receipt contained `.131176... s` / `.005291... s`.

**A reply:** Acknowledged the rerun and updated text to the retained receipt `.1312 s` / `.00529 s`.

**Resolution checked:** Updated values at A `:65` agree after rounding. The report still correctly labels this a single tiny CPU diagnostic, with differing audit instrumentation and no GPU speedup claim. **RESOLVED.**

### B-A-3 — What the solver failure does and does not explain

**Challenge considered:** Could the zero result simply be the correct response to a class-risk conflict, or be an artifact of impossible moments?

**Evidence/reply:** Both A's CE-derived finite-head fixture and B's distinct finite-head construction have a nonzero analytic scalar feasible solution. A's equivalent numerical normalization recovers the latter kind of scalar optimum without changing budgets or penalty. A explicitly labels attribution to retained patient runs `NOT_RUN`.

**Resolution:** The existence of a numerical false stall is confirmed; its prevalence and anatomical impact remain unknown. **RESOLVED with that limitation retained.**

## Verdict

**Accept A's bounded numerical/source conclusions after the v3 corrections.** The most useful immediate action is a same-problem numerical repair and a scalar aggregate finite-guard control. The literal restricted joint objective uses scalar ridge `m*lambda`, which both reports now state. Preserve the strong full-EMA comparator as a package comparison because it also changes optimizer, learning rate, clipping and state evolution. No patient efficacy, clinical value, novel theorem, or blanket solver reliability follows from these CPU checks.
