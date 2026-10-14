# Independent adversarial review — 14 October 2026

**Recommendation: NO-GO for further full training on the present justification; GO for bounded attribution using frozen existing artifacts.** The saved counts support a small, heterogeneous descriptive Dice change, loss of all RV output on scored VAL frames, and substantial baseline sensitivity to resolution handling. They do not establish useful named-anatomy recovery, real-data oblique recovery, or a causal account of the missing RV predictions. The continuation drafts mostly state those limitations correctly. The conditional mathematics is useful for identifying missing assumptions; it is not an operational no-GT certificate.

This is an AI-assisted working review for the requesting researcher/coordinator, not a journal decision, clinical review, or new model experiment. The user explicitly authorized internal review and retention of this report; no external manuscript submission or venue policy is involved. No personal conflict was identified; acquisition provenance and clinical laterality remain outside the evidence inspected. Public primary sources were consulted only by their public URLs, without sending repository content to search. Local intake declarations passed `READY_FOR_LOCAL_REVIEW`; this validates process declarations, not scientific merit.

Scope: checkout `b937e3b746608725e43d3bc3cef8861666d152dc`, branch `QuocKhanhLuong/guidance-results-research-20261014`. The working report directory was already untracked when this reviewer began. No production/configuration/prior-report edits, training, model execution, dataset/weight download, commit, push, or TEST scoring were performed. The relevant scientific source is the copied **executed_source** snapshot. `.codegraph/` was absent; all source reads used already-known paths supplied by the evidence and drafts, so no semantic code-discovery search was needed.

Path abbreviations: **R/** = `reports/guidance_research_continuation_20261014/`; **E/** = `reports/stable_guidance_world_v2_20261014/`; **S/** = `E/executed_source/src/self_audit_guidance/`. JSON keys below are exact locators. Draft hashes and input hashes are recorded in `R/ADVERSARIAL_REVIEW_CHECKS.json`; lines refer to the reviewed drafts and may move in later revisions.

Evidence labels used here:

- **PUBLISHED MEASUREMENT:** historical array/model measurements preserved in the supplied evidence, not reexecuted by this reviewer.
- **INDEPENDENT ARITHMETIC:** reductions recomputed from those records by this reviewer's separate standard-library script.
- **SOURCE/PROOF:** inspected executed source or a stated mathematical derivation.
- **ASSUMPTION / NOT VERIFIED:** absent payloads, anatomical premises, causal effects, or future measurements.

All five substantive drafts were read: `PROVENANCE_METRICS.md`, `GEOMETRY_ATTRIBUTION.md`, `LEARNING_MECHANISM.md`, `THEORY_PRIOR_ART.md`, and `PROBLEM_FORMULATION.md`, together with their available checks and `ROOT_MATH_CHECKS`. The synthesis `RESEARCH_UPDATE.md:1-85` was also checked as it became available. Existing worker PASS statuses were not treated as model validation. The independent check rehashes all 102 manifest-listed published files and recomputes decisive metric, geometry, overlap, and loss arithmetic directly from published records. It does not independently reproduce remote execution or authenticate omitted original payloads.

## Severity-ranked findings

### 1. HIGH — useful RV/MYO/LV recovery is not supported by the endpoint

**Location:** `R/PROVENANCE_METRICS.md:51-81,110-137`; `E/{baseline_forensics,run_evidence/post_training}/val_counts/DIAGNOSTICS.json.patient_frame_counts`; independent receipt `.paired` and `.metric_runs`.

**Observation / evidence:** mean foreground Dice rises from 4.684410% to 5.068521%, a **0.384111 percentage-point** difference. Only 4/18 patients improve and 14 decline; the median paired difference is −0.126854 points. Leaving out patient098 changes the descriptive mean difference to −0.312415 points. This is an influence diagnostic, not permission to omit that patient. Patient-macro coverage falls from 0.614758% to 0.366846%; UNKNOWN rises to 99.633154%. New RV has **TP=0, FP=0, FN=230281**, with zero predictions in all 36 scored frames. These are independently recomputed from published counts, not new image scoring.

The precision headline compares different estimands. The baseline foreground precision uses RV/MYO/LV, whereas new RV precision is undefined. Of the +9.460615-point headline increase, **8.115510 points** is exactly the arithmetic effect of dropping the baseline RV mean; the remaining +1.345105 points still uses changing patient/frame support. The draft's matched MYO/LV comparisons are legitimate conditional descriptions, but their selection depends on both fitted models predicting and excludes RV. They cannot establish whole-cohort recovery.

**Why it matters:** a higher average with complete class abstention and lower coverage cannot justify useful three-class recovery or another full training run. Nor can a one-seed exposed-VAL comparison isolate the geometry policy's causal effect.

**Requested action / corrected wording:** retain the full cohort, class-specific counts, undefined precision, and influence diagnostics. Use: **“On exposed VAL, saved-count mean Dice increased by 0.384 percentage points, with 14/18 patient declines, reduced coverage, and no accepted RV; useful named-anatomy recovery remains unestablished.”** The current provenance draft already meets this requirement. Do not convert missing RV precision to zero without defining a different metric, or present conditional MYO/LV precision as a replacement endpoint.

### 2. HIGH — the TRAIN intervention tests an uncoded-header refusal policy, not oblique recovery or certified laterality

**Location:** `R/GEOMETRY_ATTRIBUTION.md:3-5,25-66`; `E/run_evidence/run/bank/FROZEN.json.image_records.*.world_left_geometry` and `.patient_reports.*.class_pixels`; `S/experimental_geometry.py:15-38`; independent receipt `.geometry`.

**Observation / evidence:** independent reduction confirms 52 coded patients, all with `sform_code=2`, `qform_code=0`, unknown spatial units, and selected affine `diag(-1,-1,1,1)`. The new left covector is `(0,W_native/48)`, a positive multiple of the old +x rule. All 52 have unchanged per-patient hard class counts. The other 28 patients have codes `(0,0)` and are refused RV side information: RV count 4322→0, BG +192, MYO +53, LV unchanged. Refusal applies to 28 patients; only 12 had old accepted RV. Equal counts are not pixel/array identity.

**SOURCE/PROOF:** on the coded affines, scaling the x-side comparison by a positive constant preserves its exact-arithmetic ordering. Under identical feature/candidate inputs, the side predicate does not introduce a genuinely oblique case. Unknown units do not alone destroy a mathematical direction, but neither a coded affine nor a direction formula proves acquisition laterality. NiBabel can create an image from an arbitrary supplied affine with default codes `(2,0)`; its best-affine selection chooses a nonzero sform first. This is a concrete reason the code is a metadata declaration, not a clinical validation result. [NiBabel primary documentation](https://nipy.org/nibabel/nifti_images.html#default-sform-and-qform-codes).

**Why it matters:** “52 valid-header patients” risks conflating software acceptance with trustworthy anatomical orientation. No coded TRAIN case had the old cardinal rejection that the covector was designed to overcome. A synthetic rotation validates the coordinate formula only. Removing RV votes also changes normalization, entropy, propagated strength, and other class decisions; it is not simply deleting RV labels from a finished bank.

**Requested action / corrected wording:** use **“52 software-accepted coded-header patients with conditionally invariant side predicates; 28 uncoded patients whose RV prior was refused.”** Require existing acquisition/conversion provenance before treating laterality as anatomical authority. Preserve the statement that no real-data oblique recovery was empirically tested. No source fix or data acquisition is requested within this review.

### 3. HIGH — training telemetry does not establish sufficient local RV evidence or explain the disappearance causally

**Location:** `S/student.py:54-58,79-82,100-110`; `S/experimental_diagnostics.py:24-38,46-56`; `R/LEARNING_MECHANISM.md:66-80,92-136`; `E/run_evidence/run/student/training.json`; independent receipt `.epochs`.

**Observation / evidence:** all four global permissions remain open. The 110461 semantic “updates” are sums of active-batch indicators, versus 127650 optimizer steps; they are not measured nonzero parameter changes. Baseline final-checkpoint gradients are reported only for 80 selected eligible blocks, not the original historical steps or new checkpoint. Patient024 is a concrete witness: active=true with weight sum approximately `9.76e-14`, clamped denominator approximately `1e-8`, and tiny reported semantic gradient.

For batch activation `a_b`, denominator `d_b=max(sum w,1e-8)`, and probability targets, **CE_b=H_b+KL_b** holds with the same scaling on all terms, including inactive zeros. Last-epoch logged values are CE=1.0138087404, H=0.9813062258, KL=0.0325025146; H is 96.794% of CE. Dividing by active fraction gives the exact mean of the logged active-batch KL terms, 0.0375810325. It becomes a normalized mixture of per-batch probability measures only if active denominators are not floored. Those are different claims. The epoch also averages changing model states, so neither value is final-checkpoint KL.

Positive `sum w Q_RV` includes inactive batches and every soft target channel; it is not favorable RV contrast or a class gradient norm. The new sum near 81770.89 is therefore compatible with weak RV evidence. In each uncoded patient, no RV votes implies `R0_RV<=1/4`; patient-local stochastic restart preserves that ceiling, so bank RV cannot pass 0.7. A shared student can still predict RV; the bound is not a theorem that VAL RV must disappear.

**Concrete counterexample:** the owned checks use two positive-weight targets `(.4,.2,.2,.2)` and `(.1,.2,.5,.2)`. The second activates the batch. At uniform student probabilities, the first pixel's RV logit derivative is **+0.0123099922**, so gradient descent decreases that RV logit despite positive RV target mass. At `p=q`, KL is zero and neither target passes the 0.7 gate. This counterexample obeys the loss activation rule and can use entropy-compatible positive weights; it is synthetic arithmetic, not an observed patient.

**Requested action / corrected wording:** **“Logs establish semantic eligibility and decreasing online mismatch to uncertain targets; class-specific final-checkpoint fit, effective gradients, and the anatomical contribution of removed RV evidence remain unresolved.”** The learning draft largely says this correctly. Continue to distinguish raw vote mass, argmax support, hard acceptance, weighted channel mass, favorable contrast, gradient, and exported predictions.

### 4. HIGH — conditional expansion is valid accounting, but no operational anatomical guarantee follows

**Location:** `R/THEORY_PRIOR_ART.md:28-96,100-185` (revised draft); `R/PROBLEM_FORMULATION.md:25-36,49-61`; `S/core.py:147-168,295-303`.

**Proof review:** no weighted/unweighted substitution was found in Result 1. It explicitly normalizes by total weight, retains the low-weight exceptional region for unweighted statements, and limits the result to the common grid. The binary KL constant and Pinsker comparison are correct. Result 2 uses the actual restart family, defines contamination using **route-weighted strength relative to the target node's true class**, and includes correctly named anchors of other classes as leakage. The restart perturbation bound follows from the row total-variation bound on [0,1]-valued coordinates. The positive coverage/risk result correctly requires a fixed student, `w>=w0`, enough true-class teacher margin outside the bad subset, permission/status eligibility, and charging all accepted outputs outside its candidate region. Its accepted-set denominator is conservative but valid. The reliability gate is correctly kept separate from student export.

**Unverified premises:** actual anchor correctness, influence-weighted contamination, a usable region C and its bad mass β, final-checkpoint divergence on the claimed measure/grid, and generalization beyond fixed TRAIN nodes. Those quantities are not supplied by convergence, hashes, entropy, or agreement with the same teacher. Correctly naming the premises does not make them observable without additional admissible information. In particular, a high count of correct anchors would not imply small route contamination.

**Concrete corruption counterexample:** for the general row-stochastic family, let 99 good anchors be irrelevant to a neutral target and let the target route entirely to one absorbing, confidently wrong anchor. At α=.35, the target has wrong-class probability .7375, propagated strength .65, and reliability about .244872. It passes the fixed bank probability/margin/reliability gates despite only 1% of counted anchors being bad. Its influence-weighted contamination is one. This is a theorem-assumption witness, not a graph claimed to have been built by this repository.

**Operational vacuity:** at τ=.7 and δ=.1 the bound denominator κ is .0225824211. Even if one improperly substituted the logged active-batch mean KL .0375810325 as a normalized fixed-model divergence, the ratio is **1.66417**, yielding only the trivial upper bound one. The valid interpretation is stronger: the necessary final/native measure is not available, so the theorem cannot yet be numerically applied to the exported run at all. Synthetic nonvacuous examples establish logical possibility, not a measured certificate.

**Revised positive fixture accepted:** `THEORY_PRIOR_ART.md:154-179` adds a genuinely neutral node with one wrong source, row probabilities `(.02,.97,.01)`, and α=.35. Independently recomputing its exact kernel gives correct-coordinate probability .7181083333, margin .61984, propagated strength .637, and reliability .2214876105. This proves positive expansion is possible in the stated general stochastic family. It does **not** demonstrate an executable workaround: its target self-transition .02 violates the spatial-only repository bound `Pii>=.2`, whose neutral-node ceiling is .6982758621 below .7. The draft states that boundary explicitly; no temporal-graph or anatomical-anchor realizability was assumed. The revised active-batch discussion also now correctly distinguishes an unconditional recorded-scalar mean from normalization under the λ measure when denominators are floored.

**Requested action / corrected wording:** **“Conditional expansion is mathematically possible when enough correctly named influence survives contamination and transfer; current evidence does not verify those premises or certify a nonzero anatomical region.”** Retain the two-world identifiability limitation as conditional on indistinguishable admissible worlds, not as a claim that anatomical image inference is universally impossible. Avoid importing the older clamped-boundary proof into this restart operator.

### 5. MEDIUM — the 43.67–49.53% overlap finding is valid but has a narrow denominator and scope

**Location:** `R/ROOT_MATH_CHECKS.json.baseline_selected_batch_overlap`; `R/LEARNING_MECHANISM.md:138-162`; `E/baseline_forensics/train_all/DIAGNOSTICS.json.patients.*.batches[0]`; `E/executed_source/scripts/diagnose_stable_guidance.py:195-226`; independent receipt `.overlap` and `.overlap_rows`.

**INDEPENDENT ARITHMETIC:** all 80 recorded float32 disagreement fractions map uniquely to their reconstructed integer counts: `float32(d/N)` equals the recorded fraction and adjacent integer counts do not. The maximum residual `abs(N*f-round(N*f))` is .0001831055 pixel. These checks support reconstruction under the stated float32 mean semantics; they do not replace direct access to paired maps.

Let A and B be native/projected accepted sets, sizes a,b; I their intersection, size i; v the accepted-intersection pixels whose class labels differ; and d all gated-label disagreement, with rejected outputs both 255. Then

`d=a+b-2i+v`, `0<=v<=i`, and

`max(0,a+b-N,ceil((a+b-d)/2)) <= i <= min(a,b,a+b-d)`.

Checking endpoint feasibility per batch and summing gives a=215428, b=121350, d=119330, i in [108724,121350]. Thus **94078–106704 native accepted pixels are absent from projected acceptance**, or **43.670275–49.531166% of native accepted support**. The root's aggregate calculation agrees with these stronger per-batch calculations here; generally, global marginals need not give equally tight bounds.

The pooled disagreement is .337702% of all 35335936 native pixels; the equal-batch mean is .344966%. The ratio `119330/215428` is not the native-support error rate because d includes projected-only pixels and accepted pixels with differing class names. Neither route is anatomically scored by this diagnostic. Blocks are the first eligible lexicographic TZ block per TRAIN patient at the **baseline final checkpoint**, not a random sample, full TRAIN pass, new checkpoint, or VAL cohort.

**Requested action / corrected wording:** **“On 80 selected baseline TRAIN blocks, 43.67–49.53% of native accepted support is absent under the fixed diagnostic projection; this establishes resolution sensitivity, not anatomical improvement.”** Do not label these pixels false positives, recovered anatomy, or new-run errors, or adopt the projection as a new export policy on this evidence. The learning draft's current bound and qualifications are accepted.

### 6. MEDIUM — the named mechanisms already have strong prior art; the remaining gap is empirical semantic grounding

**Location:** `R/THEORY_PRIOR_ART.md`, “Closest primary results and the novelty they remove”; `R/PROBLEM_FORMULATION.md:47-69`.

**Independent primary-source spot check:** the main novelty exclusions are justified. Zhu et al. provide harmonic propagation and external classifier information through graph augmentation; Zhou et al. give restart propagation and explicitly discuss the row-stochastic variant. Neither is a new operator contribution here. [Zhu et al., 2003](https://pages.cs.wisc.edu/~jerryzhu/pub/zgl.pdf), [Zhou et al., 2003](https://proceedings.neurips.cc/paper/2506-learning-with-local-and-global-consistency.pdf).

Wei et al.'s Theorem 4.3 already gives conditional denoising of imperfect pseudolabels, with worst-class starting error, expansion/separation, and a population objective. The repository's weighted CE plus feature matching has not been shown to satisfy those premises or that objective. Prior art removes a broad “consistency can improve noisy labels” novelty claim without solving the present named-cardiac evidence problem. [Wei et al., 2021, Assumption 4.1 and Theorem 4.3](https://arxiv.org/pdf/2010.03622).

The draft also correctly separates labeled risk calibration, empirical confidence calibration, and class-conditional loss correction from label-free entropy weights. The inspected selective-classification theorem assumes labeled iid calibration data; temperature scaling fits validation NLL; unbiased label-noise correction requires the specified noise model/rates. None supplies absent RV ground truth or authorizes VAL threshold tuning here. [Geifman and El-Yaniv, 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf), [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), [Natarajan et al., 2013](https://dept.stat.lsa.umich.edu/~tewaria/research/natarajan13learning.pdf).

**Requested action:** preserve the draft's express refusal to claim theorem priority or standalone novelty for clamping, restart, consistency, rejection, or scalar calibration. Any future contribution must identify the admissible information that grounds names and demonstrate an improvement under matched controls. This review is a targeted prior-art check, not an exhaustive novelty clearance or independent reproof of every cited theorem.

### 7. MEDIUM/MINOR — precise examples and metric wording need source-compatible qualifiers

**7a — RESOLVED DURING REVIEW.** The initial `ROOT_MATH_CHECKS.py` soft-mass example used only `q=(.4,.2,.2,.2)`, whose BG argmax would make the actual semantic loss inactive. I sent the coordinator a specific correction. The root owner added a second positive-weight foreground pixel, recomputed the denominator, and narrowed the teacher-match statement to the first pixel. Current `ROOT_MATH_CHECKS.py:19-26,54-61` and JSON keys `.soft_mass_counterexample.batch_active`, `.batch_denominator`, and `.first_pixel_teacher_match_accepted` resolve the issue. The owned check supplies an independent active two-pixel version with both targets below .7. No other worker file was edited by this reviewer.

**7b — RESOLVED DURING REVIEW.** The initial `LEARNING_MECHANISM.md:78` said a uniform target has positive channel mass. With this bank's exact reliability `h*(1-H(Q)/log4)`, exactly uniform Q has weight zero and weighted channel mass zero. The owner corrected the current paragraph to say this explicitly and uses the nonuniform BG-biased active-batch example. I inspected the revised bytes and accept the correction. The uniform/nonuniform derivative decomposition and the conclusion about positive RV channel mass remain valid.

**7c — minor reporting clarification.** `PROBLEM_FORMULATION.md:34` says frame counts are aggregated within patient/class. Taken literally, pooling TP/FP/FN before taking the ratio would differ from the archived metric. Specify **“compute each frame's ratio, average defined frame ratios within patient/class, then average patients/classes under the preserved convention.”** `PROVENANCE_METRICS.md:47-49` already states this correctly. This is a wording ambiguity, not a demonstrated arithmetic error.

**7d — minor causal wording.** `PROBLEM_FORMULATION.md:9` calls the intervention a geometry repair and says valid numerical learning operated. Prefer “geometry trust-policy change” and “semantic eligibility plus online fitting progress were recorded.” The rest of the formulation appropriately leaves clinical geometry, gradient size, and causal attribution unproved.

**7e — minor synthesis provenance wording.** The initial `RESEARCH_UPDATE.md:61` called baseline probes “newly measured historical-checkpoint probes.” They are published measurements from the earlier forensic run, not newly executed model forwards in this continuation. Use that explicit timing to avoid confusing fresh arithmetic with fresh model measurements. This wording correction was sent to the coordinator. The reviewed synthesis's four quantitative findings and causal qualifications otherwise agree with this audit.

## Accepted claims and decision boundary

The drafts are accepted as an evidence-bounded continuation with the qualifications above. In particular: execution/source receipts and byte hashes are internally consistent; copied artifacts are not runnable originals; no permissive class gate explains RV loss; UNKNOWN remains distinct from BG; CE=H+KL is correctly scaled; grid mismatch is a real identifiability issue at the logit level; the support-overlap bounds are correct; and the theory does not silently turn weighted TRAIN fit into native or population accuracy.

The formulation's positive-coverage requirement prevents an all-reject solution from satisfying the declared objective, and its foreground obligations address BG dominance. Numerical coverage/risk obligations remain human choices to be fixed before a future confirmatory evaluation. They must not be chosen to make the exposed VAL look successful. A risk bound without verified named premises, or a guarantee on a GT-selected candidate region, is not an admissible no-GT deployment gate.

**NO-GO:** another full student run, larger backbone, threshold relaxation, CQA/attention module, or operational recovery claim justified by these measurements alone. This is a judgment about the current evidence, not a theorem that learning cannot improve a teacher. It does not endorse undoing the uncoded-header refusal: whether removed or retained RV anchors are anatomically correct is unresolved.

**GO:** a bounded, prespecified attribution study after the existing frozen arrays/checkpoints or richer saved diagnostics are made available in an authorized task. Keep the original gates and artifact identities. First compare both checkpoints on identical fixed TRAIN blocks for batch denominators, class-favorable target contrasts, final common-grid divergence, and exact native/projected joint label counts. Then, if separately authorized, localize the first semantic failure using frozen R0/Q/student outputs against exposed VAL references only inside the evaluator. Match patients/classes and include absent RV, coverage and confusion counts. Recover acquisition/conversion provenance for laterality and distinguish coded cardinal, genuinely oblique, and uncoded cases. These are unrun attribution requirements, not authorization to train or score official TEST.

The present task can finish without those payloads: the missing artifacts are scientific application blockers, not blockers to writing this review. Any subsequent model or threshold decisions remain reference-informed development, and a later generalization claim requires an evaluation insulated from that development.

## Verification receipt

Run `rtk proxy python3 -B reports/guidance_research_continuation_20261014/ADVERSARIAL_REVIEW_CHECKS.py` from the workspace. It writes only the owned sibling JSON. The successful status is **PASS_INDEPENDENT_ARITHMETIC_AND_CONDITIONAL_COUNTEREXAMPLES_ONLY**. Decisive checks include all 102 published manifest hashes/sizes, per-frame metric reductions, 18 paired patient changes, 52/28 geometry reductions, unique float32 disagreement-count reconstruction and feasible set endpoints for all 80 batches, CE decomposition for 50 epochs, an active soft-mass finite-difference witness, and the route-contamination counterexample. It imports no repository model, framework, image reader, or evaluator.

The generic peer-review linter was also tried and returned `REVISION_REQUIRED` because it expects separate author/editor journal headings, which this internal coordinator report does not use; it reported no lexical/abuse errors. That structural result is not counted as a passing scientific check or treated as a reason to invent an editor channel.

Reported measurements, independently recomputed arithmetic, proofs, and missing assumptions remain separate. No check here proves anatomical correctness, original remote execution, absent-payload integrity, acquisition laterality, causal improvement, or clinical readiness.
