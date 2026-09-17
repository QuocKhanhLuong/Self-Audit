# Primary sources and novelty attack

Search date: 17–18 September 2026. An initial table preceded implementation selection (`evidence/initial_prior_art_matrix.md`); this table expands it. This is a targeted adversarial review, not an exhaustive systematic review. M = method section read by Astra; A = abstract screened, limited claims; U = unresolved access/details. A “no” describes the inspected method, not every possible extension. 2026 preprints are provisional.

WHEN concerns computation allocation; RF means no segmentation-GT quality targets/reference masks during audit learning or scoring. WHERE concerns the segmenter's future read. VALUE is incremental audit-minus-skip utility. Training convergence and test-time efficiency are different claims.

| Work / reading depth | WHEN | Quality with no target GT / audit-training labels | WHERE next | Evidence acquisition, not just output? | VALUE | Convergence or efficiency | Medical | Remaining distinction |
|---|---|---|---|---|---|---|---|---|
| [ACT](https://arxiv.org/pdf/1603.08983), M §2 | Learned halt | Task-supervised, no QC | No | Recurrent computation | General cost, no audit contrast | Adaptive compute | No | RF-conditioned support, not halting itself |
| [PonderNet](https://arxiv.org/pdf/2107.05407), M §2 | Halting distribution | Supervised prediction | No | More pondering | No audit contrast | Adaptive depth | No | Learned allocation is established |
| [SACT](https://arxiv.org/abs/1612.02297), A | Spatial depth | No RF QC verified | Computation regions | Conditional depth | No verified | Inference compute | No verified | Region-wise computation is prior |
| [SkipNet](https://arxiv.org/abs/1711.09485), A | Layer routing | Supervised task / RL | No support coordinates | Network execution | No audit-specific target | Inference compute | No | Conditional execution is prior |
| [SelectiveNet](https://arxiv.org/abs/1901.09192), A | Abstention | Supervised selective risk | No | No | No | Risk–coverage | General | Abstaining does not acquire evidence |
| [Learning to defer](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf), A | Delegate to expert | Label/expert-error supervision | No | External expert | Deferral cost | Predictive risk/cost | General | Deferral interpretation is prior |
| [Metareasoning](https://people.eecs.berkeley.edu/~russell/papers/uai12-meta.pdf), M §2–3 | Value minus cost | Not segmentation QC | General computation actions | Abstract information gathering | Yes, general computation value | Decision utility/cost | No | Our value equation applies a known principle |
| [Event-triggered control](https://www.seas.ucla.edu/~tabuada/Papers/EventTriggered.pdf), intro | State-dependent execution | No QC | No | Control updates | Stability conditions | Control resource use | No | Plant stability does not imply semantic truth |
| [RAM](https://arxiv.org/pdf/1406.6247), M | Glimpse location | Task reward/supervision | Yes | Retinal observations | Policy reward | Limited-observation task learning | No | Where-to-look learning is established |
| [APPLE, ICLR 2026](https://arxiv.org/html/2505.06182v6), M §3 Eq2–5 | Active sensing | Task prediction objective | Yes | Environment observations/actions | General action value | Active-perception learning | No | Active perception + learned values is prior |
| [Deformable DETR](https://arxiv.org/abs/2010.04159), method previously inspected | Fixed layer schedule | Detection GT | Sparse learned offsets | Feature samples | No | Efficient attention | No | Existing operator family |
| [DAT](https://arxiv.org/abs/2201.00520), method previously inspected | Fixed architecture | Task supervision | Learned support | Feature samples | No | Recognition attention | No | Audit-conditioning must add causal utility |
| [RCA](https://arxiv.org/abs/1702.03407), M via primary validation | No | Target GT absent; **GT reference bank** | References | QC support, not actor | No | QC correlation | Yes | Violates the requested RF audit contract |
| [In-Context RCA v2](https://arxiv.org/html/2503.04522v2), M Eq1 | No | **GT reference scores** | Retrieves reference images | QC support, not actor read | No | QC speed | Yes | Reference retrieval already exists |
| [SQA-SAM](https://arxiv.org/pdf/2312.09899), M §3 | No | No task-specific QC labels; SAM mask-pretrained | Predicted-component prompts | Extra segmentation probes | No invocation-value learning | QC | Yes | Closest independent-agreement proxy |
| [CMF, 2026 preprint](https://arxiv.org/html/2608.09101v1), M §3 | No | Training-free image-text compatibility | Keep/erase views | Counterfactual scoring evidence | No invocation-value model | Mask fidelity | No | RF counterfactual assessment is prior |
| [MQ-Auditor, 2026 preprint](https://arxiv.org/html/2602.03892v1), M §3 | Post-segmentation actions | Runtime RF; GT mask/IoU/action training | No read controller verified | Assessment/refinement | No RF value target | Mask quality | Not specific | “Auditor” name is not novelty |
| [SegAE, 2026 preprint](https://arxiv.org/html/2601.14406v1), M §2 | Sample selection | **GT DSC targets** | No actor support | Dataset selection | No transition contrast | QC + active/semi-supervised selection | Yes | Quality-guided training is prior |
| [QG-SSL, 2026 preprint](https://arxiv.org/html/2606.01753v1), M §2 | Pseudo-label weighting | **GT DSC targets** | No | Regularizer/sample weights | No paired contrast | Segmentation learning; speed claim U | Yes | Frozen quality guidance already exists |
| [RBQE, 2026 preprint](https://arxiv.org/html/2609.10495v1), A | QC | Referee agreement; full training provenance U | No verified | Referee comparisons | No verified | Quality screening | Yes | Shared errors/empty agreement remain |
| [RefSeg](https://arxiv.org/abs/2207.00476), A; PDF access failed | Test reflection | Reconstruction compatibility; training GT provenance U | No dynamic read verified | Parameter/output correction | No verified | Test robustness | Yes | Reconstruction-driven correction is prior |
| [Self-reflective QC](https://www.sciencedirect.com/science/article/pii/S1361841522000779), A | QC | Reconstruction discrepancy; training contract U | No verified | No read control verified | No verified | QC | Yes | Do not claim first reconstruction auditor |
| [TENT](https://arxiv.org/abs/2006.10726), A | Adaptation | Entropy, not correctness | No | Normalization update | No | Test adaptation | General | Confidence baseline |
| [EATA](https://proceedings.mlr.press/v162/niu22a/niu22a.pdf), M §4 | **Optimizer-update selection** | Entropy/diversity, pseudo-label Fisher | No | Parameter updates | No twin value | Fewer backward passes | General | Sparse updates/noisy-feedback reduction is prior |
| [TTT](https://arxiv.org/abs/1909.13231), A | Test optimization | Self-supervised task | No | Parameter adaptation | No | Test adaptation | General | Self-supervised corrective learning is prior |
| [Vector Chan–Vese](https://www.math.ucla.edu/~lvese/PAPERS/JVCIR2000.pdf), M formulation | Energy optimization | Image region fit + regularity | No | Contour correction | No | Energy minimization | General | Our intrinsic teacher is classical |
| [Cardiac temporal consistency](https://arxiv.org/abs/2112.02102), A | Sequence correction | Shape/temporal prior; GT-free training U | No | Postprocessing | No verified | Temporal validity | Yes | Smoothness is not extent correctness |
| [Temporal self-supervision, 2026](https://arxiv.org/abs/2606.31785), A | Temporal regularization | Motion derivative constraints | No verified | Landmark/temporal correction | No verified | Temporal consistency | Echo | Does not establish MRI truth |

## Closest methods, read beyond their titles

Correction-specific controls complete the distinction from active sensing:

| Work / depth | WHEN | RF quality training | WHERE | Evidence vs output | Value of audit | Convergence scope | Medical / remaining distinction |
|---|---|---|---|---|---|---|---|
| [PointRend](https://arxiv.org/abs/1912.08193), A | Adaptive refinement locations | Supervised point prediction | Selected points | Fine/coarse features at selected sites | No | Efficient output rendering | General; uncertainty-based where-to-refine is prior |
| [FocalClick](https://arxiv.org/abs/2204.02574), A | User interaction | Supervised interactive correction | Target/focus crops | Yes, cropped evidence plus local output update | No learned RF invocation value | Inference FLOPs/interactions | General; local rereading/preservation is prior |
| [IteR-MRL](https://arxiv.org/pdf/1911.10334), method previously read | Iterative interactive actions | GT-based training rewards | Pixel agents/user hints | Primarily label/probability actions | Action value, not RF audit contrast | Interactive refinement | Medical; RL correction is prior |
| [ErrorNet](https://arxiv.org/pdf/1910.04814), method previously read | Correction pass | GT-trained error correction/prior | Error regions | Output correction | No | Segmentation correction | Medical; learned correction/error modeling is prior |
| [SAMRefiner](https://arxiv.org/html/2502.06756v2), M §3 previously inspected | Refinement schedule | No extra annotation for adaptation; SAM mask-pretrained | Multiple prompts from coarse mask | Prompted extra image interpretation | IoU adaptation, no learned invocation contrast | Refinement efficiency | General; self-boosted proposal ranking is prior |

PointRend/FocalClick are screened here at abstract level; no detailed loss or deployment claim is inferred beyond their stated mechanisms. PDF/HTML fetch failures for individual versions are recorded as access failures, not evidence that prior art lacks a mechanism.

In-Context RCA compares reverse predictions with reference-mask GT (v2 Eq1). A label-free target is not a GT-free audit objective. Astra independently verified this. Later conformal variants mentioned by Union are excluded from our guarantee claims because only v2 was independently verified here.

QG-SSL freezes a GT-trained quality predictor and uses differentiable quality regularization or detached pseudo-label weighting (§2.4). SegAE trains DSC prediction/ranking and uses quality for active/semi-supervised sample selection (§2). These defeat broad claims that independent quality-guided learning is new. Their “synthetic” quality targets still depend on reference labels.

SQA-SAM derives point/box prompts from predicted components and compares their SAM masks. Its §3.3 explicitly identifies semantic-label permutation and missed-object blind spots. These closely match the selected raw-feature energy's limitations. A new score name cannot remove non-identifiability.

CMF compares keep/erase image-text evidence with an independent frozen encoder. That is close RF counterfactual scoring, but not a learned predictor of whether an audit will improve a subsequent Annotation read. This difference is a hypothesis, not yet a contribution.

EATA filters unreliable/redundant examples from backward updates with entropy and prediction diversity. It addresses test adaptation efficiency. Our audit-invocation and optimizer-update schedules must therefore be measured separately; sparse updates alone cannot be the novelty claim.

## Astra's review of Union's report

The original report is preserved in `workers/union_rf_prior/report.md`. Training-data filtering is **not** equivalent to changing the actor's future evidence support; Astra rejects that broad classification in its AutoQ-VIS row. Reference DSC is not an intrinsic quality signal. Unverified Gaggion/RS-SQA/AutoQ-VIS details and reported numbers are not used in the final verdict. RCA, In-Context RCA and SegAE were independently checked. Full-text access failures remain UNKNOWN.

## Novelty conclusion

Dynamic supports, active perception, allocation, quality prediction, classical energies, detachment and replay already exist. Their combination alone is insufficient.

Potential contribution: an empirically valid, RF **action-specific contrast** predicts invocation value and causally redirects a shared Annotation reader, improving learning efficiency at matched examples, updates and budget. The current energy-distillation implementation is a diagnostic use of known parts. It must beat entropy, matched random, extra-read and no-audit controls before being positioned as a core method. No Q1 acceptance forecast or “first” claim is justified.

Not established: absolute correctness from consistency, optimal computation, clinical safety, external robustness, or faster convergence. A negative result on this proxy cannot prove that every reference-free signal is impossible.
