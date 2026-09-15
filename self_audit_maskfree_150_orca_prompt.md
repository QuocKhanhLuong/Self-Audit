# Implementation and execution prompt — Mask-free anatomical label generation, 150 epochs

You are implementing and executing a research pipeline, not writing another research proposal.

## 1. Ownership, scope and execution authority

- Planner, architect, integrator and final reviewer: **Astra, requested XHigh reasoning profile**.
- Primary implementation workers: **Claude Opus through Orca**.
- Additional workers: **AGY through Orca when quota is available**; otherwise reassign their isolated tasks to available Opus workers.
- Repository: `QuocKhanhLuong/Self-Audit`, latest `main`.
- Reference inspected when drafting: `a3737081fe10c1e4c38ba3e2fbeaddc79f1c4e65`. Refresh HEAD and read repository instructions before editing.
- User workspace: `/home/linhdang/workspace/quockhanh_workspace/SpecMamba`.
- Intended hardware: the user's RTX 4070, previously physical GPU 1, 12 GB; Python 3.10, PyTorch 2.4.1/CUDA 12.1 environment `specmamba`. Verify actual hardware/runtime.
- Keep configs, outputs, reports and logs in the workspace, not `/home/aidev`.

The user authorizes implementing this new branch of functionality, focused validation, normal publication to `main` after review, and actual full runs on their accessible authorized GPU. Preserve existing jobs, checkpoints, data, and supervised baselines. Do not kill unrelated processes, spend money provisioning machines, force-push, or bypass permissions.

This task concerns a NEW **mask-free, anatomy-prior-guided** label-generation pipeline. It is NOT the existing supervised Candidate C pipeline with its losses renamed. Existing GT-trained FIX/REGRESS heads, old checkpoints, and GT-driven counterfactual generation are forbidden dependencies of the new primary method. Reuse generic geometry, runtime and serialization utilities only after checking their contracts. Candidate C's software-intervention idea is optional infrastructure, not a compulsory mechanism or novelty claim.

Produce working code and run it, not TODO implementations or smoke-only claims. If the real GPU/data cannot be accessed, finish the code and executable launcher, report the precise blocker, and mark full execution NOT STARTED. Never fabricate execution, GPU access, quota, model identities, or completion.

## 2. The scientific contract

Goal: from unlabeled cardiac images, generate anatomically named draft labels, identify insufficiently supported regions, and train a segmentation student without any manual mask influencing training or selection.

Primary workflow:

`unlabeled images -> dense self-supervised features -> competing anatomical partitions -> equal-budget observation fitting -> evidence-based selection/challenge -> pseudo-labels -> student -> frozen exports -> optional isolated reference evaluation`

Hypothesis being tested: predictive evidence on observations not used to construct a partition can improve generated labels beyond confidence, consistency, or fitting the observations already seen.

This is provisional. Do not describe predictive likelihood as segmentation accuracy, evidence margins as calibrated correctness probabilities, or implemented software as established novelty.

### Absolutely no mask supervision in the generating pipeline

No manual masks in losses, positive/negative construction, cropping, anatomical anchor fitting, pseudo-label filtering, threshold choice, model selection, or stopping. No SAM/MedSAM/GT-trained Self-Audit teacher or mask-trained atlas. Default initialization is random for a small model; any later self-supervised external checkpoint needs an explicit provenance profile. Do not inherit `pretrained_encoder: true` from existing configs.

Ground truth may exist ONLY in a separate final evaluation process after every compared method's predictions and checkpoint identities are frozen. If reference masks are absent, the pipeline must still generate labels, finish training, and report all available image-only metrics; reference metrics are unavailable, never zero or invented.

Anatomical class definitions and hand-specified priors are permitted but must be enumerated. Claim **no manual segmentation masks**, not absence of all prior information.

## 3. Data discovery and evidence separation

Do not ask the user to repeat dataset locations already known. Inspect their authorized workspace before making assumptions.

Known preprocessed image locations include `preprocessed_data/ACDC/training` and `preprocessed_data/mnm`. Raw full cine availability has NOT been confirmed. Current ACDC preprocessing selects ED/ES and requires masks; do not reuse that selection rule for a new image-only loader.

Implement image-only discovery and manifest creation, including:

- patient/study/case identity, dataset/site, file paths, native affine/orientation, spacing validity, axis semantics, available frame indices and times;
- patient-disjoint training, unlabeled-development and final-test memberships;
- no dependency on whether a mask exists, and no fake zero-mask field;
- duplicate ED/ES/full-cine acquisitions identified without counting the same acquisition twice;
- no train-time use of diagnosis labels, manual ROI boxes, annotation quality flags, or other hidden supervision.

Keep ACDC and M&Ms native experiments independent. Do not transfer ACDC weights to M&Ms native training. Preserve official held-out membership where available. Otherwise create a deterministic patient split from image identity only, freeze it, and record the limitation. Do not claim a previously repeatedly inspected validation cohort is a fresh test set.

### Two explicit observation protocols

1. `cine_predictive`: requires genuine multi-frame cine, geometry/time metadata and enough frames for a declared fitting/selection/verification partition. Predict withheld frames from fitting observations; do not compute registration/flow using withheld target images and call this prediction.
2. `spatial_predictive`: for paired 3-D volumes or insufficient cine, use masked spatial blocks with a guard band and documented spatial conditioning. It is a weaker spatial-appearance experiment, NOT a temporal/biomechanical experiment.

Auto-detect once, save the resolved protocol and reason, and keep profiles/results separate. Do not fabricate cine from ED/ES. Missing orientation can make semantic assignment unresolved; missing spacing prohibits millimeter metrics. Historical ED/ES-only preprocessing must be disclosed as a cohort-selection limitation.

### Three observation roles inside each study

- `O_fit`: input to the proposal generator and all per-hypothesis nuisance fitting.
- `O_select`: observations used to choose/teach hypotheses; repeated use is adaptive selection, not untouched validation.
- `O_verify`: never used for fitting, proposals, threshold adjustment, checkpoint selection or student supervision; checked only after the final protocol/models/predictions are locked.

Also keep patient-level final test studies out of all global representation/student fitting. Masked reconstruction training targets are self-supervision, not independent verification evidence. Do not mix these roles in reports.

Exclude withheld intensities from normalization statistics, crops, feature caches, motion estimation, augmentation inputs, nearest-neighbor retrieval and proposal initialization. Record evidence provenance. Only fitting images/metadata define those operations. Any use of selection observations is logged honestly as adaptive information consumption.

Image-only verification applies only to predictions constructed without O_verify. A conventional student output that consumed the complete image can be evaluated against sealed segmentation references, but cannot also be credited with predicting those same consumed intensities unseen. Keep fit-view predictive outputs and full-input deployment outputs in distinct manifests. Predicting a fully withheld cine frame requires transport/prediction from fitting frames, not segmenting the withheld frame first.

## 4. Minimal executable method

### 4.1 Representation producer

Use a small dense GroupNorm CNN/encoder-decoder, approximately 5M parameters or less for the first profile; measure the actual count. No custom CUDA or new giant backbone.

Implement dense contrastive learning with known augmentation correspondences, plus local masked reconstruction and geometric equivariance. Avoid dense all-pairs HW-by-HW similarity matrices; sample a bounded set of patches. Do not assume every other patient is a true anatomical negative. State the negative-sampling rule and false-negative limitation.

Producer training uses image-only objectives. In the primary controlled experiment, it receives no gradients from either downstream student and no GT or audited-student outputs. This makes a shared candidate bank fair between audited and unaudited students.

### 4.2 Anatomical hypotheses and semantic grounding

Generate a bounded bank of four candidates per study/slab, including the incumbent. Sources include image/feature grouping, an image-only anatomical initializer, bounded boundary changes, and split/merge or semantic alternatives. Reuse the SAME generated bank for selector comparisons. An edit type never defines whether an edit is beneficial.

Export semantic order `0=BG, 1=RV, 2=MYO, 3=LV`. Explicitly implement an image-only anatomical-role resolver in `ontology.py`, with machine-readable rules covering spatial orientation, enclosure/adjacency, study-level continuity, absent structures, contour convention and unsupported cases.

Do not pretend that four k-means clusters identify four anatomical classes. Do not choose cluster IDs by GT overlap or per-case Hungarian matching. Semantic permutations may leave appearance likelihood identical: the ontology resolver must expose this ambiguity instead of inventing a likelihood distinction.

Specify actual computational rules and all constants before a long run; no unspecified `assign_anatomy()` placeholder. Use full-study evidence without requiring every class in every slice. Priors must not enforce a normal-heart template or blindly delete pathological structures. If role assignment remains ambiguous, export the draft and alternative roles plus `semantic_unresolved`; do not mark it certified or silently map it to BG.

### 4.3 Observation model

Start with a constrained, inspectable region-conditioned appearance model: fixed-capacity Gaussian/Student-t mixtures with variance floors, a restricted smooth bias model, and explicit region-complexity terms. Background may have nuisance components but every compared candidate receives the same capacity and fitting budget.

For `cine_predictive`, add an explicitly parameterized smooth nonrigid temporal deformation model fit ONLY on `O_fit` and extrapolated/interpolated to withheld times. Do not assume rigid-heart motion or brightness constancy without reporting residual/model mismatch. The spatial profile must function without this optional module.

No unrestricted neural renderer, per-pixel free reconstruction parameters, target-image skip connection or copying target intensities. The partition must affect predictive distributions. Test mask-shuffle and all-background counterexamples.

For each hypothesis Y:

`eta_hat(Y) = argmin_eta [-log p(O_fit | Y, eta) + regularization(eta)]`

`J_select(Y) = normalized_predictive_NLL(O_select | O_fit, Y, eta_hat(Y)) + beta * complexity(Y)`

Record raw likelihood, priors, complexity and valid-observation denominators separately. All score terms, masks, noise models and optimization limits must be specified in the architecture contract. Candidate mask construction on withheld locations must come from fitting-side inference/transport, not fitting-side masks plus target-informed registration.

### 4.4 Evidence auditor and challenger

Use the directly computed evidence score as the primary auditor; do NOT start with a learned correctness head that needs GT.

Seek distinct, admissible alternative partitions under comparable fitting loss. Default budget: four total candidates, two correction rounds, at most five nuisance-fitting iterations per candidate, streamed one candidate at a time on a reduced fitting grid. These are provisional fixed budgets, not empirically optimal settings. Log actual forwards, fitting steps and latency.

Choose an alternative only under a preregistered improvement rule on `O_select`, after comparable fitting effort. Always retain incumbent/no-change. A rejection rejects that edit, not the segmentation-learning sample; alternative proposals and self-supervised learning may continue within the fixed budget. Do not inherit random-auditor reject-HALT starvation from supervised training.

Compare with genuinely different partitions. An empty challenger set means `search_inconclusive`, not infinite confidence. Record semantic alternatives even if likelihood ties. Do not call an evidence-improving edit FIX or an evidence-worsening edit true REGRESS.

Use regional competitor margins and reproducibility to weight pseudo-labels. These are evidence scores, not GT correctness certificates. Export a full draft mask, soft labels, validity/ambiguity map and alternative hypotheses. Unresolved regions are ignored/downweighted in training, never converted to background. Log class coverage and prevent hidden all-background or vanishing-coverage success. If no useful pseudo-labels survive, continue legal SSL work but report that label generation failed its objective.

### 4.5 Students and learning targets

Train two identical lightweight students, with identical initialization, architecture, optimizer, augmentation and update budgets:

- `student_no_audit`: labels from the shared initial hypothesis, without evidence-based selection.
- `student_audited`: labels selected by the full evidence auditor.

Keep student losses identical: CE/KL on generated soft labels with explicit validity weights, plus the same optional image-consistency term. Loss normalization uses valid support, not the full image when pixels are ignored. Log skipped/zero-valid batches.

The producer/bank is shared and independent of students; students do not feed back into it in v1. Report natural-coverage results AND a matched-coverage comparison to distinguish better labels from simply discarding more pixels. These are independent learners, not a student initialized from a supervised teacher.

## 5. Exactly one 150-epoch global timeline

Provide one trainer, one resolved config, one continuous global epoch counter per dataset, and one primary W&B run with producer/student sub-namespaces.

`global_epoch = 0 ... 149`

Every epoch visits each defined training sampling unit once, with a seeded sampler. Define the unit for full cine versus volumes and report patients, slices/slabs, batches, and the number of updates per component. Do not hide extra dataset passes as zero-cost preprocessing.

Every epoch performs image-only producer learning, bounded hypothesis generation/auditing and both student updates. Auditing starts at epoch 0; it is not deferred to epoch 100. A smooth pseudo-label weight ramp over the first 20 epochs is permitted and recorded, e.g. `0.05 + 0.95 * min(1, (epoch+1)/20)`. Do not restore public A/B/C stages or train an Auditor separately from GT.

Use a single global warmup/cosine schedule and saved optimizer counters. Do not add an unreported 150-epoch student stage afterwards. Both students train within the same 150-epoch timeline; their additional optimization cost MUST be reported. ACDC 150 plus independent M&Ms 150 means 300 dataset-global epochs, not 150 total across the two datasets.

Primary checkpoint is the valid final epoch-150 state. `best_unlabeled.pt` may be a secondary image-only selection checkpoint under a preregistered rule; it must never be selected by Dice. Always save `last.pt`, final component checkpoints, optimizer/scaler/RNG state, sampler state, evidence partitions and pseudo-label-version identity. Resume must restore the exact experiment, not migrate supervised or other-dataset checkpoints.

Default seed 42. No automatic multi-seed/grid-search explosion in this urgent execution. Provide seeds 43/44 as an explicit later plan, not claimed results.

## 6. Metrics: mandatory, with scope and denominators

### During the 150 epochs — no GT

- SSL contrastive/reconstruction/equivariance losses; feature variance and collapse indicators.
- Fitting and selection predictive NLL in declared units; separate prior/complexity terms.
- Candidate count, distinctness, no-change rate, accepted-edit rate, evidence improvement, candidate/fitter budget utilization, challenger failure and semantic-resolution rates.
- Pseudo-label coverage globally, per class, per patient; class occupancy, ignored fraction and class disappearance warnings.
- Valid spatial/temporal consistency measurements, cycle errors where measurable, and anatomical-prior violations. Not correctness metrics.
- Both student losses, actual optimizer steps, skipped batches, gradient norms, LR, runtime, peak VRAM and I/O cost.

### After locking everything — still image-only

Evaluate `O_verify` predictive NLL and selection-to-verification gap, observation perturbation sensitivity, annotation stability, ambiguity/coverage and degenerate-partition challenge outcomes. Report the finite challenger-search scope. Never recalibrate or refit on these results and still call them untouched.

### Optional isolated GT evaluator, only after ALL compared predictions are frozen

- Native-volume per-class Dice/IoU and patient macro mean; foreground-only aggregate. Report draft-label quality AND student quality separately.
- HD95/ASSD only with valid native geometry; explicitly label pixel-space alternatives. Count failed/undefined surface cases.
- Named semantic mapping correctness without per-case oracle relabeling. Cluster-matched metrics may appear only under `oracle_cluster_matching_diagnostic` and never drive selection.
- True initial-to-final FIX/REGRESS, previous-correct destruction, net quality change and harm, with exact denominators and provenance.
- Correlation of image-only evidence gain with true quality gain; quality-ranking AUROC/AUPRC only with sufficient defined positive/negative support. Report tie/neutral policy, counts and uncertainty.
- Coverage-versus-error curves at preregistered coverage levels 25/50/75/100%; no accuracy improvement claim without its coverage.
- Audited-minus-unaudited student differences with paired patient bootstrap intervals. Do not bootstrap slices as independent patients.

Metric contract: both-empty classes are excluded from the relevant aggregate and counted; one-empty errors count as errors; no undefined-to-zero conversion. A score margin is not a probability: no ECE/Brier/confidence-guarantee claim without a separately justified probability model and calibration protocol.

Every metric row needs dataset, split, protocol, epoch/checkpoint, population, sample count, availability, unit and contract version. Do not mix stale epoch-120 audit numbers with epoch-150 student numbers in W&B summary. Missing GT produces `available=false` with a reason; it must not block image-only export or fabricate Dice.

## 7. Minimal experiment suite — actually execute it

On the same frozen candidate banks for both datasets, run:

- E0: simple image-only region/appearance baseline with the same permitted anatomical rules.
- E1: self-supervised feature grouping + ontology, without audit.
- E2: choose from the shared bank by confidence/consistency only.
- E3: choose by fitting-side score only.
- E4: choose by withheld selection predictive evidence, without active challenge.
- E5: full predictive-evidence selection + active challenger.
- Negative controls: random same-budget selection, shuffled evidence across incompatible studies, all-background, spatially random masks, class permutations and excessive partitioning. These are diagnostics, not hidden training augmentations labeled good/bad.

Run selectors on identical candidate sets first. On a fixed bank, E4 and E5 may select the exact same argmin mask; this is expected if their only difference is ambiguity reporting. Do not invent a mask-quality gain from the audit label. Separately compare challenger-generated candidates against an equal-number/equal-cost random-edit candidate set; otherwise a larger search budget could explain the gain. Attribute any additional challenge benefit to this search comparison, and any selective-quality benefit to explicit coverage-aware evaluation.

E2–E5 are paired-bank policy comparisons, not claims that their separately trained end-to-end networks were compared. Actual label usefulness is tested by the two 150-epoch lockstep students. Do not run six additional 150-epoch trainings by default.

Use the published CUTS method as a literature baseline. The core suite MUST include an executable feature-grouping control now. An in-repo simplified control is named `cuts_inspired_control`, never misrepresented as a reproduced published CUTS result. Integrate exact external CUTS only if dependencies/license and label-free semantic selection are resolved; otherwise mark that literature reproduction pending, not completed.

A single-seed pilot is not SOTA/novelty proof. Report negative results honestly. Predefine outcomes that undermine the idea: evidence gain fails to predict quality gain; confidence/simple grouping matches audit; all-background wins the observation model; semantic roles remain unresolved; useful coverage collapses; or audited student does not improve at matched coverage/budget.

## 8. Orca orchestration — detailed bounded worker assignments

Discover the installed Orca interface and available providers. Do not invent command syntax. Record each actual task/session ID, requested and resolved provider/model, ownership, status, and returned diff. Target eight independent workstreams, only as many simultaneously as tools/quota support. Worker parallelism is for coding; one training job may own the 4070 at a time.

Before dispatch Astra writes `reports/maskfree150/architecture_contract.md` with exact APIs, dataclasses, tensor/axis semantics, score equations, observation-access permissions and accepted provisional constants. Each worker receives this file plus its complete task prompt. Workers must not be expected to remember this conversation. Freeze interfaces before dependent implementation; negotiate changes through Astra.

### W1 — Opus: predictive observation model

Own `src/self_audit_maskfree/observation.py` and focused observation tests.

Implement deterministic `fit(hypothesis, fitting_view)` and `score(fitted_hypothesis, scoring_view)` with a type-level separation of fitting and scoring data. Deliver static appearance likelihood first, valid cine extension second, equal-capacity fitting, variance floors and explicit residual units. Test target blindness, score sensitivity as positive control, partition sensitivity, and matched fitting budgets. No full-image target-informed flow; no shared mutable nuisance parameters across candidates.

### W2 — Opus: hypotheses, ontology and evidence auditor

Own `hypotheses.py`, `ontology.py`, `auditor.py` in the new package.

Implement four-candidate banks, bounded edit/search actions, real anatomical-role rules, semantic ambiguity, incumbent fallback, region margins and two-round budget. Consume W1's APIs; never train against GT. Deliver competing-class/partition examples, including tied likelihood, missing anatomy, and all-background. Export full draft plus validity/alternative records. No GT-shaped positive/negative edits or arbitrary class-ID assignment.

### W3 — AGY if available, otherwise Opus: ACDC/M&Ms image-only data

Own new `data/` and `scripts/prepare_maskfree_data.py`.

Inspect actual roots read-only. Build manifests, enforce patient split and observation roles, normalize from permitted observations only, preserve raw metadata, and separate cine/spatial protocols. Never require/read masks. Output dataset-readiness inventory, unresolved metadata and exact input contracts. Existing paired supervised loaders remain untouched. Test successful loading when masks are absent and when mask trees are forbidden.

### W4 — Opus: producer, students and losses

Own new `models.py` and `losses.py`.

Implement small random-init dense representation model, sampled local contrastive loss, masked reconstruction, equivariance, and two identical students. Specify known-positive correspondence and negative-sampling rules. Students train from detached generated labels only; preserve producer independence. Test finite gradients, all-invalid pseudo-label handling, no cross-arm gradients, same student initialization, and measured parameter/activation sizes.

### W5 — Opus: trainer and reproducibility

Own new `trainer.py`, `config.py`, `runtime.py` and `scripts/train_maskfree.py`.

Implement strict one-config 150-epoch timeline, smooth weights, both students, bank/cache versioning, epoch/update counters, exact resume and per-epoch reports. Reuse vetted atomic checkpoint/JSON utilities without importing supervised targets or stale metrics. Maintain new recipe/checkpoint identities. Save final epoch150, optional explicitly selected unlabeled checkpoint, pseudo-label lineage, RNG and observation-split state. Test partial runs cannot masquerade as completion.

### W6 — Opus or AGY: experiments and isolated evaluation

Own new `experiments.py`, `evaluation/`, `scripts/evaluate_maskfree_reference.py`.

Implement E0–E5 and negative controls, paired common-bank selection, coverage matching, native-volume metrics, true edit attribution and patient bootstrap. Ground-truth reading exists ONLY in this isolated evaluator. It must accept already frozen prediction manifests; it cannot return decisions into training. Produce JSON/CSV/Markdown results with explicit missing fields and scope. Test metric edge cases and stable fixed class mapping.

### W7 — AGY or Opus: launchers, exports and operator docs

Own new `export.py`, `scripts/run_maskfree_full.sh`, `scripts/run_maskfree_acdc_mnms.sh`, configs `maskfree_acdc_150.yaml` / `maskfree_mnms_150.yaml`, experiment manifest YAML and usage docs.

Integrate true native-grid masks, probabilities, validity and sidecars; no label-volume resizing without reverse geometry. Generate unique read-only run-local configs, workspace output paths and sequential independent dataset runs. Use W5's strict schema, no private alternative parser. Check completion artifacts before advancing; no resume/weights shared between datasets. Document actual CLI and readable logs. Do not edit other workers' implementation files.

### W8 — independent Opus/AGY reviewer: leakage and scientific red team

Read-only production access; own focused firewall tests and `red_team.md`.

Try to break the method: mask-path access, target-image leakage, label-based crop/split selection, semantic swaps, all-background, independent fitting of held-out motion, stale banks, circular confidence, unlimited verification reuse, misreported compute, GT-driven best checkpoint, and covered-pixel-only metric inflation. Reproduce material defects, send fixes to owners, and distinguish methodological weakness from a software bug. Do not accept other workers' prose instead of inspecting code.

Every dispatch must include: goal, owned files, exact callable interfaces, permitted inputs, prohibited dependencies, required artifacts, 1–3 focused checks, and PASS/REVISE/BLOCK acceptance criteria. Workers return actual diffs, exact commands and outputs; no independent push or schema redesign. Astra integrates and reviews. Missing quota triggers reassignment, not a false worker-completed status.

## 9. Fast gates and resource control

No repeated full repository suite per worker. Run focused checks, then one combined critical integration suite. No unbounded literature search or profiling project in this coding task.

Non-negotiable gates before full training:

1. Image-only loader succeeds without readable masks; GT mutation cannot affect training, proposals, selection or exports.
2. Altering `O_select` cannot change fitting-only proposal/nuisance construction before scoring; it CAN change its score. Altering sealed `O_verify` cannot change any trained state or selected draft.
3. Actual semantic resolver executes; labels are not GT-matched clusters. Ambiguous cases remain explicit.
4. Score cannot silently ignore the partition; candidate capacities and observation sets match.
5. Both dataset CLIs run bounded real-GPU smoke where available, including an actual fit/score/edit-or-rejection path, checkpoint/report and tiny export. A zero-attempt smoke does not test the auditor.
6. Tiny interruption/resume parity and old-artifact preservation.
7. Basic volume metrics and per-patient aggregation tests, isolated from the mask-free run.

Default requested physical batch is 8, accumulation 1. Stream candidate fitting; do not allocate all hypotheses' graphs together. Verify GPU identity, preferably select its UUID after checking `nvidia-smi`; do not accidentally use the unsupported 5070 Ti.

If batch8 fails the bounded probe, log the OOM and explicitly resolve microbatch4×accumulation2 or microbatch2×accumulation4 for effective batch8 BEFORE the real run. Record physical and effective batch separately. It is not numerically identical to batch8 for contrastive negatives: use a documented queue/memory scheme or label the contrastive-batch change honestly. No silent mid-run OOM retry, batch change, LR scaling or input-resolution change.

## 10. Full execution and publication

After focused gates pass, make logical commits, publish normally to `main` without force push, and run the reviewed immutable revision. If a normal push conflicts, reconcile changes safely; do not overwrite someone else's work.

Required new interfaces, to implement and verify before documenting as available:

```bash
# Full native ACDC and full independent native M&Ms, sequentially.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
RUN_FULL=1 TOTAL_EPOCHS=150 BATCH_SIZE=8 \
bash scripts/run_maskfree_acdc_mnms.sh
```

This command must perform inventory/preflight, two actual 150-epoch runs with the two student arms, final native-grid exports, E0–E5/negative-control reports, frozen verification, and optional isolated reference evaluation when its separate configuration provides legitimate masks. No external reference is required for the core pipeline to succeed.

Use a real local/remote execution tool or a persistent session only when available. Do not stop at printing a command when authorized GPU access works. Do not report a launched job as completed. Record observed PID/session, exact command, SHA, config hashes, dataset identities, current progress, exit status and artifact paths. If access fails or required metadata is absent, report the precise stage as BLOCKED; still deliver every working component and remaining launch commands.

Per-run outputs under `runs/maskfree150/<dataset>/<run_id>/` must include:

- resolved config, dependency/source/supervision provenance and data/evidence manifests;
- producer, audited-student and unaudited-student checkpoints, shared history and exact-resume state;
- versioned pseudo-labels, full draft masks, probabilities, validity and semantic alternatives;
- common candidate-bank records, fitting budgets and evidence traces;
- image-only epoch metrics and final verification metrics;
- experiment matrix/results, optional isolated reference metrics and comparison tables;
- atomic pipeline report, failure report when relevant, and final receipt.

Completion requires all 150 global epochs, readable final artifacts, exports and required experiment jobs. A skipped optional literature reproduction or missing GT evaluation is explicitly separate. Semantic failure/low coverage is a research result, not silently changed to success; runtime completion and scientific outcome have separate fields.

Final Astra response: actual implementation, actual worker identities, measured tests, exact run status for each dataset, available metrics/results, skipped work, supervision ledger, falsification outcomes, remaining risks, commit SHA and pull/run commands. Do not promise novelty, GT-free correctness guarantees, or a Dice improvement before measurement.

START NOW: inspect source/data/tools, freeze the minimal contract, dispatch workers through Orca, implement, run short gates, publish reviewed code, and execute the requested full pipeline on accessible authorized hardware. Do not end after the plan.

## Primary background references (not a novelty certificate)

Use these as existing baselines, not as claims of what this new pipeline achieves:

- CUTS, MICCAI 2024, official proceedings: https://papers.miccai.org/miccai-2024/186-Paper1569.html
- Unsupervised Multi-Object Segmentation by Predicting Probable Motion Patterns, NeurIPS 2022, official proceedings: https://proceedings.neurips.cc/paper_files/paper/2022/hash/0eaf2c04280c7fecc8b26762dd4ab6da-Abstract-Conference.html

The motion paper's rigid-object formulation is precedent, not a cardiac-motion model to copy blindly. If consulting full papers, verify method, inputs and limitations before implementing an adaptation; record distinctions in method documentation.
