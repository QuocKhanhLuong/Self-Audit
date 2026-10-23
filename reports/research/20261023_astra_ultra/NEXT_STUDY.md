# Next study: earn the extra allocation freedom

Status: **PROPOSED_NOT_RUN**. This document does not restart cancelled jobs or implement the new arms. The detailed protocol and exact current CLI limitations are in [E's design dossier](workers/E_design.md).

## Ordered gates

| Priority | Deliverable before advancing | Evidence now | Stop condition |
|---|---|---|---|
| 0 | Numerically qualified local solve across response scales; analytic scalar/tiny-vector references; explicit feasibility and objective-gap diagnostics | Small-response premature stopping reproduced by A and B on CPU, including physical affine-head moments | Feasible-but-avoidable zero/tiny step, state contamination, or inconsistent moments/support |
| 1 | Literal scalar restriction of joint allocation, with identical finite guard | Derived; new method not in current CLI | No parity with alpha=a·1, mismatched ridge/budgets/denominator, or changing pseudo directions |
| 2 | Fixed-source/config/schedule protocol and telemetry | Current main and experiment config semantics differ; B52 remains one patient per role; planner does not regenerate retained C256/B52 recipe | Silent fallback, source mismatch, uncharged labels, or unrecorded schedule/optimizer change |
| 3 | Bounded liveness pilot/replay on approved existing TRAIN data, only after scope is authorized | No new patient execution in this session; raw historical transaction moments unavailable in Git | No effective nonzero useful motion, solver errors mistaken for rejection, or scalar and joint identical on all informative transactions |
| 4 | Fixed-final P/S/J development comparison and practical reference methods | Four historical full baselines only; joint efficacy NOT_RUN | Joint does not beat the scalar or label-use control enough to justify compute |
| 5 | Replication over labeled subsets and training seeds, then independent frozen evaluation | Not available | Effect unstable across subsets/classes or only present on already exposed development data |

## Primary contrast

For a frozen transaction replay, hold theta_s, teacher logits q, unique nonempty class directions d_j, image views, audit weights, pseudo-loss denominator, parameter block, update rule, tolerance and budgets fixed. In full training, share initial student/teacher state, data schedule and candidate-generation policy. Actual supervised references, EMA targets and directions will diverge as the interventions change the trajectory; forcing identical evolving targets would be a different study.

- P: current `ema_block`, full aggregate D=sum_j d_j.
- S: **proposed** scalar coefficient a on D, then the same finite class guard.
- J: current `joint_guard` after numerical qualification, separate alpha_j coefficients.

The exact scalar restriction has A_c^S=1ᵀA_c1, b_c^S=1ᵀb_c and ridge **m·lambda** because ||a·1||²=m·a². Using lambda without this adjustment changes the objective and is a separate variant. The analytic scalar control is conventional optimization, not a novelty claim.

J−S asks whether extra allocation freedom helps. P−S asks about the combined effect of scalar selection and finite guarding. Retain `supervised_extra` for extra label-use value and `ema_full` as a stronger practical SSL package; neither is an exact scope-only or time-only control. A direct same-block audit-Brier gradient with the same finite guard is another proposed direction-family control; it is not implemented.

## Schedule and measure

Under current round-robin schedules, a U patient sees L/gcd(L,U) supervised partners over a full joint period. In TRAIN80 with L in {4,8,16}, U=80−L is divisible by L, leaving one partner and permanently excluding that partner from its audit role. Independently permuted L/U cycles are a proposed remedy to study, not a demonstrated accuracy improvement. A new schedule must be shared by all compared arms. Historical results then serve as context unless explicit compatibility is established.

The reported endpoint is patient/class mean of reference-present frame Dice; false-positive-only absent-reference frames are excluded by the evaluator's null policy. Preserve that historical endpoint, explicitly count excluded frames, and prospectively define any additional pooled-per-patient endpoint. Do not change metrics after seeing favorable rankings. Store full confusion and absent-class FP statistics. A 52-crop batch is not 52 independent patients.

## Minimal execution plan after qualification

E proposes a fresh 1,144-step liveness pilot: the retained 1,000-step warmup plus two 72-U-patient cycles. One P trajectory supplies frozen transaction states; P/S/J are evaluated from each identical state, but only P commits live parameters and advances EMA. This isolates local allocation mechanics and does not estimate the total J/S training-policy effects. Full policy comparisons later start fresh and let their endogenous teacher/candidate trajectories diverge. This pilot tests activity and support, not final accuracy. Current source has no exact resume, scalar arm, shadow audit, role-partition or compute-stop option; new JSON fields alone cannot implement them. No new GPU-hour cap is approved by this report.

If the pilot survives, use fixed-final 10,000-step endpoints. The proposed +1.0 percentage-point J−S continuation margin and no-class-drop-beyond1.0 point are engineering resource-triage rules in E's dossier, not clinically validated or power-derived thresholds. Freeze or explicitly replace them before observing new outcomes. Replicate only a surviving hypothesis across the proposed 2 subset × 2 training-seed block, and do not call four blocks a powered confirmatory study.

Record effective parameter displacement rather than scale>0 alone; source moments and solver residuals; local and actual risks; absent-positive groups; rejected trials; candidate pixels/class; all TRAIN/U/audit patient pairings; teacher/gradient/JVP/forward counts; startup and loop time; peak memory; and checkpoint/evaluation overhead. Optimize diagnostic overhead only with parameter/output parity evidence, and keep instrumentation cost visible.

## Research decision

Continue the bounded numerical/mechanism study. Defer architectural expansion and new full patient matrices. A useful paper claim would require evidence that joint compensation adds value beyond scalar shrinkage, extra labeled optimization and close existing methods. A rigorous negative characterization is also a valid outcome. Current data establish neither that joint guard works on patients nor that it cannot work.
