# Controlled baseline comparison

MEASURED on real ACDC at 64x64, 20 complete validation patients, three seeds. All Dice values below are fractions, not percentages. Compute and budget qualifications are essential; these are underfit mechanism diagnostics, not leaderboard results.

| Policy | A0 Dice | Retained Dice | RV / MYO / LV Dice | Harm-patient rate | Final patient-mean audit fraction | CPU seconds |
|---|---:|---:|---|---:|---:|---:|
| B0 no audit | 0.1625 | 0.1625 | 0.1165 / 0.0247 / 0.3464 | 0.000 | 0.0000 | 16.74 |
| B1 always | 0.2421 | 0.2607 | 0.2443 / 0.1330 / 0.4047 | 0.250 | 1.0000 | 19.91 |
| B2 periodic | 0.1805 | 0.1947 | 0.1433 / 0.0603 / 0.3804 | 0.133 | 0.5071 | 19.16 |
| B3 random cap 0.5 | 0.1811 | 0.1964 | 0.1816 / 0.0252 / 0.3822 | 0.167 | 0.4999 | 19.45 |
| B4 entropy cap 0.5 | 0.2238 | 0.2386 | 0.2195 / 0.0558 / 0.4407 | 0.100 | 0.4986 | 19.35 |
| Proposed learned | 0.1819 | 0.1822 | 0.0738 / 0.0304 / 0.4426 | 0.000 | 0.0039 | 21.24 |
| Extra read, zero audit guidance | 0.2522 | 0.2669 | 0.2777 / 0.1219 / 0.4011 | 0.200 | 0 (one extra read) | 18.88 |

A0 is the initial annotation of **each policy's separately trained actor**, so between-row A0 differences are training effects. Harm means a patient's retained final Dice is below that same actor's A0, not below B0 and not a clinical safety outcome. A zero harm rate from almost always skipping is not evidence of a useful safe auditor. Class order follows ACDC: RV=1, myocardium=2, LV=3.

The raw evaluator's `audit_fraction` field records the extra-read decision for the unguided control; its value is 1, while its actual `audit_calls` is 0. The table above correctly reports zero audit invocation. No tile or local-region trigger exists in this baseline: selected slice guidance is dense. A “percentage of regions audited” claim would be undefined beyond slice occupancy.

## Matching and remaining confounds

All policies use identical physical/effective batch, 120 actor optimizer updates, 960 supervised examples, initial actor/Adam state, sample ordering, K, writer and random seeds. Periodic, random and entropy execute 320 policy audits over the 640 post-warmup examples; always executes 640. Learned executes 212/14/173 (mean 133), below the same cap. All audit policies additionally receive the same uniform exploration stream and 20 slow RF updates. Only learned receives value-head optimization. No/unguided incur no RF/probe setup or training expense.

Thus the long-form training comparison is **cap-matched, not realized-budget-matched** between learned and random/entropy. It tests whether the chosen system works, but it cannot attribute the learned policy's poorer result entirely to allocation quality. The frozen-state comparison in 10 fixes the realized cardinality; its fresh-head extension spends additional value-training compute and is not folded into this training table.

The extra-read control shares reader/writer and recurrence, uses zero guidance, and incurs no audit calls. It is the direct falsifier for “more reading, not better audit.” Its mean retained Dice is 0.2669 versus always-audit 0.2607. Seed-wise always-minus-extra-read differences are +0.02434, +0.00752 and -0.05057, so equivalence or universal superiority is not established. The present evidence does not support paying for the Auditor.

## Patient uncertainty, kept separate from seed uncertainty

The optional descriptive bootstrap averages the three seeds **within each patient first**, then resamples the 20 patients 10,000 times (seed 5901). It never treats 376 slices or 60 seed-patient rows as independent. Intervals below are conditional on these three fitted seeds, this compact model and this split; they omit training-seed uncertainty and are not equivalence tests.

| Paired retained Dice contrast | Mean | Patient-bootstrap 95% interval |
|---|---:|---|
| learned minus none | +0.01970 | [+0.00755, +0.03127] |
| learned minus random | -0.01411 | [-0.03552, +0.00655] |
| learned minus entropy | -0.05639 | [-0.07152, -0.04198] |
| always minus unguided | -0.00624 | [-0.01856, +0.00558] |

Per-patient records, all three seed values, and the bootstrap generator are retained. No result is selected by best seed.

## B5 and B6

**B5 supervised REW: IMPLEMENTED elsewhere, NOT RUN as a matched control here.** It has a different actor, three proposals, supervised outcomes and acceptance logic. The occupied remote training job cannot be treated as a fair comparator or an upper bound numerically. Preserve it as the next supervised control; do not import its success/failure to this direction.

**B6 oracle trigger: TESTED offline only.** Every frozen state has skip/audit outcomes. `offline_matched_oracle` chooses positive true-DeltaDice actions up to the learned per-batch cap. It uses GT only in evaluation and has no optimizer. Original collapsed policies yield a vacuous zero-budget comparison. The fresh-head seed-0 diagnostic creates a nonzero cap: oracle retained slice-DeltaDice 0.006138 versus learned 0.001846. Budget-matched regret is 0.004292. The raw `utility_regret` field compares with an unconstrained positive-action oracle; `analysis_details.json` separately records the matched-cap quantity so these are not conflated. Neither oracle is a feasible deployment policy.

The first ranking is therefore: ordinary extra reading and always-audit are competitive here; learned RF invocation is not validated. A simple entropy allocation is a serious threat. This is insufficient evidence to declare every learned trigger unnecessary.
