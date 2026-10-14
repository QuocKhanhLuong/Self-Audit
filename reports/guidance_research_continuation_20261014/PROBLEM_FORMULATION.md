# Problem formulation after the completed world-left-v2 run

Source: `origin/main@b937e3b746608725e43d3bc3cef8861666d152dc` (2026-10-14). This is a continuation, not a new architecture proposal. References in the exposed 18-patient VAL cohort inform the diagnosis below; this is reference-informed development. The official unlabeled TEST is not an accuracy benchmark and was not scored or used for selection.

## What must be solved

Recover a useful amount of **correctly named RV, MYO and LV supervision from image-derived evidence whose semantic validity is uncertain**, then preserve that evidence through learning and selective export. The system must disclose which assumptions establish class meaning, when it has no information, and when its numerical confidence merely reflects repeated or biased sources.

The completed run narrows the question. Semantic permissions were open and soft-target learning operated. Semantic eligibility and proxy-fitting progress nevertheless coexisted with very low coverage and no exported RV on VAL. A geometry repair changed the availability of RV evidence in patients without coded world geometry; it did not empirically test oblique orientation recovery. Neither a stronger backbone nor an active CQA auditor has been tested by this run.

## Inputs, outputs, and information constraints

For patient `j`, observe a cine `X_j : Ω_j -> R`, with `Ω_j = {1,...,T_j} × {1,...,Z_j} × {1,...,H_j} × {1,...,W_j}`, and permitted metadata `M_j`. The latent field `Y_j : Ω_j -> C` has ontology `C={BG,RV,MYO,LV}`; MYO here has the benchmark's myocardium definition. An abstention action `⊥` (stored as 255) is separate from BG.

Admissible priors `Π` may describe topology, orientation, image appearance and motion, but each must name its source, coordinate convention, validity domain and failure modes. A pretrained representation `E` has a separate supervision/provenance ledger. Image-only target training does not mean that `Π` or `E` contains no upstream knowledge. A coded NIfTI transform supplies a usable coordinate map to the software; its clinical laterality must not be inferred from its mere presence.

Fit a recipe using TRAIN images/metadata only. At patient-specific inference, that patient's whole cine may be observed, including temporal neighbors. This is different from fitting hyperparameters or a shared bank with the held-out cohort. Proposals, graphs, readouts, permissions and student targets must be measurable functions of the permitted inputs, priors and training randomness. Reference masks must not enter those stages, checkpoint choice or threshold fitting. Freeze predictions and provenance before evaluation. The existing VAL is exposed; future claims of generalization need an evaluation protocol insulated from this development.

An algorithm returns a class distribution `p_j(v)`, an action `a_j(v) ∈ C∪{⊥}`, and provenance sufficient to distinguish source evidence, propagation, model approximation and selection. A probability simplex alone is not a ledger of what semantic information was supplied. Anonymous partition recovery is invariant to class-name permutations; named recovery is not. Partition stability and feature agreement cannot settle that difference.

## Objective and rejection contract

Define a reference distribution by drawing a patient uniformly from the evaluation population, then a location from a prespecified measure `ν_j` on that patient's cine. Spatial/temporal weighting and frame inclusion are fixed before scoring. This avoids treating millions of correlated pixels as millions of independent patients.

Let `A={a≠⊥}`. Define coverage `C_A=E[1_A]`, error mass `E_A=E[1_A 1_{a≠Y}]`, and selective error `R_A=E_A/C_A` when `C_A>0`; risk is undefined when coverage is zero. One admissible contract is

```
minimize E_A + λ(1-C_A),   0 < λ < 1,
subject to C_A ≥ κ > 0 and prespecified per-class reporting/coverage obligations.
```

Alternatively require `C_A≥κ` and `R_A≤ρ`. The human must choose `κ,ρ,λ` and class-specific obligations before the next confirmatory evaluation. They are not to be optimized on the inspected VAL. An overall contract alone permits a BG-only solution, so report true-class coverage `E[1_A | Y=c]`, correct-class recall, confusion counts and missing-class events for each foreground class. These reference-based quantities are evaluation outcomes, not no-GT operational gates.

This contract is distinct from the repository's reported foreground macro Dice and class precision. Preserve that reporting convention: compute each frame's ratio from counts, average defined frame ratios within patient/class, then average patients and classes, with undefined denominators retained. Do not replace it with pooled accuracy, and do not silently change the class set when comparing precision.

Positive coverage at bounded semantic risk is not guaranteed over arbitrary image-label worlds. Feasibility requires a nonzero region where permitted evidence distinguishes the named class with bounded ambiguity/corruption, and a way to retain that distinction through prediction and selection. If the model class permits the same `(X,M,Π,E)` with conflicting anatomical labels, no observed confidence can certify which world is true. The current dataset summaries do not independently establish the required semantic-error bound.

## Obstruction and conditional opportunity

Four different quantities must remain separate:

1. **Named evidence:** whether image/metadata sources favor the anatomically correct class, with known or assumed error.
2. **Transport:** whether aggregation/propagation preserves a correct class margin despite neutral and incorrect sources.
3. **Approximation:** whether a learned predictor matches a fixed target on the supervised grid and weighting measure.
4. **Selection:** whether the exported action has adequate coverage and bounded semantic error on the requested grid.

Classical propagation can address transport under assumptions. Proper soft-label losses can address approximation. Neither supplies the absent semantic premise. Abstention does not validate that premise either, and an unconstrained rejector can satisfy a superficial accuracy goal by predicting nothing.

The conditional positive result in [THEORY_PRIOR_ART.md](THEORY_PRIOR_ART.md) gives a concrete opportunity: enough influence from correctly biased sources can dominate bounded erroneous influence and produce a positive class margin and additional accepted support; sufficiently small student approximation error then preserves most of that support and bounds additional error. The graph/target parts can be computed from permitted observations. The correctness of the source partition and its corruption bound depend on latent anatomy or external knowledge. That bound is conditional evidence, not an operational label-free certificate.

The transfer result requires the same grid and weighting measure. Matching coarse logits cannot in general control unsupervised native oscillations. Small mean KL likewise cannot protect an unweighted rare class when its training weight is very small or zero.

## Two central questions and falsifiable hypotheses

**RQ1. Where does named anatomical information first become inadequate: source/readout, propagation, or learned prediction?**

**H1 (working diagnostic hypothesis, unproven):** the current failure is already present in the named teacher targets, rather than being fully explained by optimizer failure. Test this with frozen R0, Q, student coarse probabilities, native probabilities and a predetermined projection on the same patients. H1 is falsified as the primary explanation if Q meets the human's prespecified per-class coverage/risk contract while the coarse student fails it materially under a matched weighting/grid analysis. If R0 satisfies the contract but Q fails, narrow the failure to propagation instead of blaming source semantics. If neither has useful named foreground, another long student run cannot establish that the teacher is adequate.

**RQ2. Under explicitly bounded source error and leakage, can named support expand while its additional semantic error stays within a declared budget, and can learning/export preserve that improvement?**

**H2 (conditional recovery hypothesis):** for a fixed evidence ledger and fixed selection rule, patients/regions satisfying the positive theorem's source-influence and margin conditions admit nonzero expansion that survives an independently measured approximation/transfer budget. The theorem itself is mathematical. Its applicability to these images is empirical and can fail: insufficient admissible named evidence, large corruption/leakage, or no region with positive margin falsifies the practical premise. If observable witnesses claim those premises while frozen evaluation contradicts them, that falsifies the witness-to-correctness assumption, not the algebra. Compare against classical propagation before claiming a new mechanism.

These questions do not presuppose CQA, CineMA, an RNN, or a student relabel loop. Do not count agreement between a source and its own propagated/student descendants as independent validation.

## Minimum requirements and human choices

Any future solution must (i) declare how names enter; (ii) preserve source provenance and distinguish repeated correlated evidence from independent information; (iii) preserve weak/contradictory/missing evidence rather than reporting only normalized confidence; (iv) measure class-specific favorable target contrast, not only total channel mass; (v) bind approximation and selection to the same spatial domain or supply an explicit transfer bound; and (vi) preserve the prediction freeze, split and exposed-VAL status.

The human still chooses the permissible additional semantic information, the coverage/risk contract, whether the intended contribution is task characterization or a new grounding mechanism, and the final architecture. A metadata provenance repair or classical clamped diffusion comparator may be useful engineering; neither is a novelty claim. The leading research problem is the source-correctness premise plus safe support expansion, with grid transfer as a necessary downstream contract. Architecture selection is premature until the frozen stagewise diagnostic identifies the first anatomical failure.

