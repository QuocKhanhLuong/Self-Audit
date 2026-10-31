# Method and theory: one compatible pipeline

Status: implemented research candidate, not proven publication novelty or patient
efficacy. Primary setting: a budgeted dense-labeled TRAIN subset L and image-only
U. Audit examples rotate within L and are patient-disjoint from the supervised
minibatch **within each transaction**, not a held-out statistical validation set.

The opt-in `self_audit.automatic.v1` pilot has a separate
[locked contract](AUTOMATIC_PILOT_CONTRACT.md) and
[runbook](AUTOMATIC_PILOT_RUNBOOK.md). It uses fixed complete TRAIN-L support,
frozen fixed-window features during head adaptation, and raw-student deployment.
Its finite Dice predicate protects only the extra U step relative to theta_sup.
The fixed volume support measure is a declared experimental factor, distinct
from the rotating single-slice protocol described below. Neither protocol
certifies EMA predictions or query-patient Dice.

## Questions and choices

1. Where do names/boundaries come from? Dense native CE + present-foreground Dice
   on L. A verified pretrained encoder is initialization, not a semantic oracle.
   No historical image-rule Q bank is needed.
2. Which U information is useful? An EMA copy proposes class labels on a clean
   view. A stronger intensity-only view preserves pixel coordinates. A fixed
   confidence threshold generates candidates, not a correctness certificate.
3. How should multiple candidates interact? Joint local function-response
   allocation, followed by actual per-class audit-risk checks on the update.
4. When may the teacher learn the result? Once, after the student state is final.
   Rejected proposals must not contaminate the next target or optimizer state.

Backbone/EMA/CE/Dice are standard choices, not contributions. A conventional
native decoder is used rather than enforcing an obsolete coarse constraint.
The earlier exact D/U split remains a historical experiment, not a hidden path.

## State and information timeline

At iteration t, draw supervised (Xs,ys), an audit slice (Xa,ya) from a different
budgeted TRAIN patient, and unlabeled Xu. A labeled-only AdamW-body/SGD-head step
produces theta_s. This is the counterfactual reference. The body includes the
input adapter, encoder and native decoder. Head SGD has no momentum or decay.

Freeze teacher phi_t while generating q=softmax(T_phi_t(Xu)). Partition accepted
pixels by pseudo argmax class. For source j, with a COMMON accepted denominator Z:

    L_U,j = sum_{i:accepted and class=j} CE(q_hard,i, p_theta(Xu,strong)_i) / Z
    d_j = -eta_U grad_B L_U,j | theta_s.

Thus sum d_j equals the ordinary SGD pseudo-step on the SAME parameter block B.
No source is duplicated; class identities give at most four canonical directions.
There is no inverse-frequency amplification or hidden U ground-truth selection.
Head and decoder scopes are distinct declared ablations. Full EMA updating all
parameters is a stronger comparator, not attributed to allocation alone.

Audit each foreground class one-versus-rest. A present class allocates half mass
to positives and half to negatives, patient-normalized. An absent class measures
false positives only, with positive count=0 reported. Ignore=255 has zero weight.
This explicitly observes BG-to-MYO errors; an interior-only audit would miss them.
A single slice remains limited support: it is not an audit of all patients/volumes.

## Local joint solve

For each group c define r_c=sqrt(W_c)(y_c-p_s,c),
V_c=sqrt(W_c) J_c [d_1 ... d_m]. The local squared-probability risk is

    E_c(alpha) = 1/2 ||r_c - V_c alpha||^2.

Solve for 0<=alpha_j<=1:

    min mean_c E_c(alpha) + lambda/2 ||alpha||^2
    subject to E_c(alpha) <= E_c(0) + epsilon_c for all c.

The groups include negative pixels. The objective is a local Brier proxy, not
Dice, confidence or a probability that a source is correct. Define A_c=V_c'V_c,
b_c=V_c'r_c and e_c=||r_c||^2/2, reducing the solve to small GxMxM moments.

### Existence and exact specializations

Alpha=0 is feasible. The box is compact, constraints are convex, and lambda>0
makes the objective strictly convex. Hence the finite mathematical problem has
one minimizer. This does not guarantee a nonzero accepted update. The numerical
SLSQP implementation is checked independently for feasibility and improvement
against zero, and fails closed to zero on solver failure. It is not a proof that
a floating-point solver found the exact optimum.

Without active constraints, stacked appropriately normalized V,r give
alpha=(V'V+lambda I)^(-1)V'r. This is classical ridge regression, not a new theorem.
For a single direction a_c=||v_c||^2,b_c=r_c'v_c, the group upper bound is
(b_c+sqrt(b_c^2+2a_c epsilon_c))/a_c when a_c>0. Intersect those intervals with
[0,1] and clamp mean(b)/(mean(a)+lambda). For a_c=0, b_c must also be0.
Clipping the vector ridge solution coordinatewise is NOT the multi-source solve.

Cross terms alpha_j alpha_k V_j'V_k retain useful compensation among sources.
The manufactured example V=[[.5,.15],[-.07,.02]], r=(.8,.1) has optimum
alpha=(2/7,1) at lambda=.001 and zero budgets. Risks change (.32,.005) to
(.128596939,.005). Independently discarding direction1 gives (.21125,.0032).
This proves an interaction can matter, not superiority on medical data.

The coefficient penalty is not invariant to duplicated columns. The implemented
fixed class grouping and duplicate-ID rejection are part of the contract.
Do not create multiple copies/crops of one source as independent candidates.

## Parameter response and finite-step compatibility

For a 1x1 affine head, candidate logit changes dz are exact on frozen features.
The probability derivative is dp=p*(dz-sum_c p_c dz_c). The head path uses this
formula; tests compare it to autograd. For the native decoder, torch.func JVPs
measure the actual parameter-to-probability response at theta_s in eval mode.
Native image features are not replaced by a theoretical infinite-width kernel.

Probability linearization has error R_c(alpha): actual risk differs from the
local risk by at most ||r_c-V_c alpha|| ||R_c|| + ||R_c||^2/2. Consequently a
feasible local solution alone is insufficient. Evaluate the EXACT function at

    theta_trial = theta_s + scale * sum_j alpha_j d_j,
    scale in {1,1/2,...}.

The full method accepts only finite trial predictions with actual group risks
<= reference + declared budget + numeric tolerance, and nonincreasing mean risk
up to tolerance. Otherwise backtrack, then retain theta_s if all trials fail.
All trials are functional; no optimizer.step is called and no candidate updates
EMA. Commit the selected displacement directly, then update EMA once.

The guarantee concerns the extra pseudo step on the CURRENT TRAIN audit measure.
The supervised step, later iterations, EMA output, unseen patients and Dice do
not inherit it. Positive budgets/tolerances permit corresponding small changes;
there is no unclaimed global debt bound. Pseudo directions touching decoder
parameters coexist with supervised AdamW state but are not sent through AdamW;
future optimizer effects are not certified. Both guarded/control arms use that
same split-step convention. All sources can be rejected, which is not efficacy.

## What self-learning means here

Labeled supervision trains the entire native segmentation model, including detail
features. EMA proposals provide extra updates to the selected head/decoder block.
In decoder scope, U can change the feature decoder and native detail path, not
only final probabilities. The encoder continues to learn from labeled examples;
U-to-encoder learning is represented by the separate full-EMA comparator.
No learned auditor, geometry naming, image refiner or recurrent controller is
required. These omissions remove incompatible objectives, not useful test data.

## Critical statistical boundaries

The audit set is optimization data. Rotating disjoint-in-step patients reduces
immediate same-batch reuse but does not create an unseen risk estimate. Missing
class positives, overfitting the labeled subset, zero-update stalls, teacher
bias, threshold sensitivity and long-term EMA drift must be measured. No true
external generalization guarantee follows. All audit labels are charged to L.

A small labeled subset may not represent anatomical diversity. Experiments must
use nested patient budgets and repeated subset seeds; compare extra labeled-only
optimization at reported compute, and retain standard stronger SSL baselines.
Final quality is evaluated from frozen predictions outside TRAIN, separately.

## Opt-in adaptive-window pilot

The separate [window pilot](WINDOW_PILOT.md) adds fixed or adaptive semantic
support at the final decoder fusion, with an optional update-mediated meta-Dice
geometry rule on top of full EMA. It does not alter the canonical defaults or
the guarded policy described above. The new method has no finite-step Dice
guarantee. Its initial evidence was CPU/manufactured numerical and software
verification. Subsequent matched GPU runs and frozen TEST50 evaluation are
recorded in `reports/window_pilot_test50_20261024/`; the Drive reference copy is
not independently certified as official challenge ground truth. These
measurements do not extend the guarded policy's guarantees to patient Dice.


## Opt-in compact fixed-teacher distillation

`self_audit.compact.v1` is an additive mode through the same `train.py` entrypoint
and `self_audit.engine` dispatch. See [the compact contract and runbook](COMPACT_STUDENT.md).
It uses a separately initialized, fully learned compact encoder/decoder, not the
synthetic `tiny_test` backbone and not a smaller head on DINO. The full-EMA-run
teacher is frozen and its selected state and complete target-label lineage must
be verified. The compact student has no EMA or prototype inference branch.

The new transaction is supervised AdamW -> post-supervised reference -> fixed
teacher targets -> class SGD displacement candidates -> finite scalar/vector
coefficient proposals -> actual whole-network functional trials -> one additive
commit or keep reference. No EMA update follows in this mode. The existing EMA
and Brier modes above keep their original meaning. Hard-Dice and absent-class FP
guards constrain only the extra U displacement on the declared TRAIN audit unit.
The guard does not certify supervised updates, future updates or query Dice.

Deploy only the independently reloadable compact export. Acceptance requires
real evaluation of that export with historical Macro4 >= .87 and matched-device
inference measurements showing lower seconds/volume, peak GPU memory, peak host
RAM and weights-only bytes than the pinned full-EMA-run reference. Foreground
and every class remain reported separately. Tests do not establish this gate.

The optional compact scalar selectors `larger_tie` and
`cross_patient_utility` are specified in [COMPACT_STUDENT.md](COMPACT_STUDENT.md)
and [TRANSFER_UTILITY_RUNBOOK.md](TRANSFER_UTILITY_RUNBOOK.md). The historical
`original` selector stays the default. The utility selector maximizes a fixed
teacher KD reduction on an independently sampled, transaction-disjoint TRAIN-U
patient among the same hard-Dice/FP-feasible scalar trials, including zero.
This changes the finite selection objective, not the teacher, candidate update
rule, guard or deployment architecture. It is a teacher transfer heuristic;
its value must be measured against original, larger-tie and plain KD on VAL.


Historical full-model B8 reproduction uses the separately pinned historical
source plus a reviewed checkpoint/resume extension. See
[B8 reproduction and backup](B8_REPRODUCTION_RUNBOOK.md). It is a new artifact
identity, with one fixed-budget run and verified Drive backup before evaluation;
archived Dice scores remain historical evidence.

The reproduced teacher is an explicit opt-in identity, bound to its new full
checkpoint, raw-student export, source closure, L8 lineage, measured training
cost and verified Drive readback. It cannot satisfy the old B8 hash pins.
`scripts/continue_b8_transfer_pilot.py` waits for this teacher's completed backup
and quality gate, then requires repository checks, exact manufactured CUDA
replay, a new TRAIN-U cache and the bounded profile before the locked pilot.

That continuation applies the same `deterministic_algorithms: true` runtime to
every compact arm. It retains the existing bilinear forward and architecture;
the unsupported CUDA atomic bilinear backward is evaluated by the deterministic
CPU derivative and transferred back. This is an explicit execution factor with
training overhead, included in measured costs. The runtime scope restores its
decoder dispatch and PyTorch flags on exit. Default recipes, the standalone
model definition and no-grad deployment remain unchanged. Numerical tests of
this backend and exact replay do not establish clinical quality or speed gains.
