# Architecture decisions before measurement

| Question | Minimum choice | Reason and falsifier |
|---|---|---|
| Separate trigger/auditor? | Separate cheap scalar value head and spatial q head | A skip must avoid q forward; a shared full trunk would hide its cost. Audit-lite features can be added only if cheap features fail. |
| Global/local/hierarchical? | Per-slice trigger, spatial dense guidance | Avoid two nested allocation mechanisms. Patient remains evaluation unit. Hierarchy requires evidence of sparse useful regions. |
| State or transition? | State guidance; transition-aware value target | State plausibility alone does not measure action benefit. Twin contrast is action-specific. |
| Scalar or spatial output? | Spatial q; scalar trigger | Guidance must causally alter support. No separate directional head. |
| One candidate or multiple? | One audit candidate versus exact identity skip | Tests the original annotation/evidence idea without best-of-M confounding. |
| Where does audit enter? | Coordinate generator only | Fix-coordinate intervention isolates its path; no direct writer shortcut. |
| QKV or pair messages? | Existing QKV | Operator novelty is not needed for the question. Non-QKV adds a confound. |
| Shape corruption ranking? | Excluded | Erosion can repair oversegmentation. Corruption names cannot supply correctness labels. |
| Anchor | Fixed image filters | Fully auditable absence of segmentation pretraining labels. Weak semantics is a serious limitation, not concealed strength. |
| Actor–critic reward | Excluded | Supervised actor loss trains useful read/write behavior; detached RF guidance cannot be gamed by differentiating a score. |
| Replay | Seeded uniform recent observations | No priority or importance-weight complexity before usefulness is demonstrated. |
| Retained acceptance | No second critic gate | Trigger decides whether to spend one read; post-hoc acceptance would mix two questions. |
| Candidate C | Excluded | Historical restitution is a different action and value horizon. |
| Current REW | Separate supervised control | GT-derived outcomes make it ineligible as the new Auditor. |

## Signal survey and assumptions

Transformation consistency is a reproducibility constraint, not truth. Geometric warps need inverse alignment and valid support masks; rotations/flips need anatomy/label semantics preserved. Intensity, bias-field and noise perturbations need plausible ranges. Repeated forward cost must be counted. A consistently eroded myocardium can be perfectly equivariant.

Image compatibility has an observable image anchor and can test image swaps. Raw MRI intensity is not a unique semantic tissue identifier, varies with bias field and acquisition, and can make texture partitioning preferable to an anatomical boundary. Frozen self-supervised MRI encoders are an alternative only with provenance and a natural-transition validation gate; an arbitrary “foundation model” is not automatically GT-free.

Known corruption ranking is not defensible as universal good/bad supervision on a model prediction. If used later, train distinguishable invariances or corruption magnitude, or obtain scores from independent evidence; do not label all eroded masks bad. Hold out corruption families and test natural transitions. Metadata hiding does not eliminate visible artifact shortcuts.

Cross-slice consistency requires real geometry and z spacing, particularly thick-slice cardiac MRI. Smooth propagation can erase basal/apical structures. Cine temporal consistency requires motion correspondence and temporal identity; raw adjacent-frame pixel agreement is invalid during contraction. Motion-cycle consistency permits jointly consistent errors. The copied annotated ED/ES pairs do not provide a full cine benchmark, so temporal experiments are NOT RUN.

Disagreement/stability can identify epistemic instability but shared actor/teacher mistakes are invisible. SQA-SAM already tests external mask agreement, and its paper explicitly notes class-assignment and missed-object limitations ([method §3](https://arxiv.org/pdf/2312.09899)). Reconstruction compatibility can learn to synthesize convincing images from wrong masks; RefSeg is a close existing direction ([primary paper](https://arxiv.org/abs/2207.00476)). Neither guarantees true transition utility.

## Efficiency mechanisms that remain hypotheses

Selective reads may allocate supervised gradient to difficult states, avoid harmful extra reads, or simply save computation. Measure skip/guided gradient cosine, useful intervention density, and matched learning curves. Lower optimization frequency may reduce cost and stale-target chasing, but can also delay useful guidance. The factorial invocation schedule × optimizer-update schedule is required to distinguish those mechanisms; it is not an invitation to run a large grid immediately.
