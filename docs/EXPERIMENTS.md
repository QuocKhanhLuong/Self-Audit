# Experiments and falsification

The delivered code adds experiments; it does NOT establish patient efficacy.
No patient masks/images or pretrained DINOv3 weights were available for execution
in this environment. Manufactured tests do not estimate clinical generalization.

## Stage 0: mechanical/analytical

`run_mechanisms.py`: compensation example, 50 seeded finite QCQPs, an affine
native-head compensation case. `pytest`: direct/JVP parity, finite acceptance,
rollback/state isolation, one EMA commit, native detail gradients, shared pseudo
denominator, all nine methods, decoder scope, label access, data split schemas,
checkpoint roundtrip and frozen prediction/evaluation separation.

`run_synthetic.py` runs 11 small CPU configurations: nine head-scope arms plus
plain and audited decoder scope. It exercises the complete loader/model/loss/
optimizer/candidate/JVP/solver/trial/EMA/checkpoint path on manufactured shapes.
Threshold .25, tiny model and synthetic LR are explicitly test-only; do not use
those values as a patient result or selected medical recipe.

## Stage 1: first usable real baseline

Prepare nested 4/8/16 labeled-patient subsets within the locked 80 TRAIN pool;
8 is the initial development budget. Every supervised/audit mask access counts
in that pool. All unselected patients use full cine images only; their ED/ES
identities and masks are not consulted. Report actual labeled volumes/slices.

First profile the 32-step head pilot using genuine approved DINO weights, then
profile decoder JVP on the same input. Do not silently downgrade the backbone,
disable the nonlinear audit, change labels or use synthetic fallback on OOM.
A bounded pilot needs enough steps to visit every declared patient; log actual
coverage. Full 10,000-step configs are hypotheses, not proven schedules.

Train supervised-only, then the ordinary full-EMA and same-block-EMA baselines.
Verify native supervised learning first. No user-supplied test labels are tuning
data. Checkpoint selection is fixed final step unless a separately budgeted
selection protocol is declared. Existing VAL18 remains development-exposed.

## Stage 2: matched method ablations

- `supervised`: only the supervised reference step.
- `supervised_extra`: another labeled update, measuring additional label-use value.
- `ema_full`: conventional two-stage pseudo learning with all trainable parameters.
- `ema_block`: same fixed candidate block, all candidates fully applied.
- `sign`: aggregate first-order alignment, no finite guard.
- `independent_guard`: solve source coefficients independently, guard the sum.
- `joint_no_class`: joint unconstrained local objective, same classwise actual guard.
- `joint_no_guard`: joint constrained local solve, no nonlinear guard.
- `joint_guard`: joint local class budgets plus actual nonlinear guard.

For matched allocation comparisons hold backbone, initialization, input, label
subset, patient order, target threshold, candidate denominator, pseudo LR and
scope fixed. Head versus decoder is its own experiment. Full-EMA is a stronger
scope comparator, not a factorially isolated allocation comparison. Plain arms
currently compute diagnostic audit responses too; reference timing includes that
instrumentation and is NOT an optimized latency comparison against those methods.

`plan.py` writes 81 configs (3 budgets x 3 label-subset seeds x 9 methods) and
command templates; it never starts jobs. Additional training seeds, patient
bootstrap uncertainty and matched-compute comparisons are required for publication.
Count all supervised updates, audit reads, JVPs, trial forwards, accepted/rejected
U updates, teacher calls, wall time, memory, pixels and patients. Equal optimizer
steps do not imply equal compute. Supervised-extra is not an exact wall-time match.

## Stage 3: outcomes, not just loss

Score frozen dense raw predictions using fixed BG/RV/MYO/LV channel meanings.
Report patient/frame Dice and confusion, especially BG->MYO FP and RV FN.
Absent-reference class cases are not inserted as perfect Dice; retain FP counts.
The optional gate is separate; UNKNOWN counts as a missed foreground pixel.
Evaluate classwise precision/recall, correct-class coverage and external domains
with a separately frozen protocol. Physical HD95 is not implemented in this
cache evaluator; add only with certified spacing, explicit empty conventions,
and tests. Do not call the current metrics a full clinical evaluation suite.

Audit risk diagnostics are on TRAIN, not this quality endpoint. Report absent
positive audit groups, population of source class candidates, local versus actual
risk prediction, acceptance/stall rate and classwise cumulative changes. Use
patient-level uncertainty, not millions of pixels as independent samples.

External BCP/HCO/modern SSL comparisons remain required and unrun. No adapter here
pretends to reproduce those papers. Make the protocol, annotation cost and data
access equivalent before comparing published numbers.
