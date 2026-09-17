# Minimal diagnostic baseline selected by Astra

Supervised compact Annotation Expert with the unchanged QKV DynamicWindow leaf, plus a fixed image compatibility teacher, a small spatial Auditor that distills that teacher, and a cheap scalar value Trigger trained on sparse detached audit/skip twins. One optional refinement, exact skip, no candidate bank. Its purpose is to falsify the proposed chain, not assert a new core method.

The new namespace is `src/self_audit_event/`, on isolated branch `feat/event-triggered-reference-free-audit` at fetched main 874c0772c9b00a0fbc5e781ea8e457794160f654. No canonical/REW/C/mask-free implementation is overwritten. Root owns data, training, evaluation, tests, scripts and final review. Union was initially assigned seven core modules under `workers/implementation_contract.md`; after its explicit delivery failure and confirmed exit, Astra recovered and completed them. Final ownership and defects are recorded in 00_ORCHESTRATION and 08_IMPLEMENTATION_AUDIT.

| Item | Prespecified bounded setting |
|---|---|
| Data | Existing ACDC 80/20 patient manifest; all slices; no test data |
| Resolution | 64×64 for the first CPU falsification; no comparison with 128/224 production claims |
| Actor | 16 feature channels; K=8; supervised CE + foreground Dice |
| Batch | Physical/effective B8, accumulation 1, CPU workers 0, four Torch threads |
| Seeds | 0,1,2 |
| Common actor setup | 40 no-audit optimizer updates, identical sample stream |
| RF setup | 16 training-image-only updates, counted for audit policies |
| Value setup | 16 training-image-only twin updates, counted for learned policy |
| Comparison | 80 further actor updates, validation every 40 |
| Allocation | 50% cap; random/entropy select the same maximum count; report realized learned count |
| Exploration | 25% independent twin probes; separate from retained actions |
| Slow optimization | Every 4 actor steps; uniform buffer 256; expire age >8 |
| RF cost | 0.001 proxy units; no post-hoc threshold tuning |
| Thresholds | Dice 0.5,0.7,0.8; unreached = right-censored |
| Modes | none, always, periodic, random, entropy, learned, unguided extra-read |

The configuration is frozen before real-data results. Large-batch fits on a 4060 Ti do not determine this scientific recipe. No attempt is made to occupy its currently running REW GPU. Local CPU results cannot validate VRAM, 4090 throughput or late-training memory peaks.

All policies start from identical actor and, when relevant, q initialization and see the same supervised examples. Different realized audit counts remain a confound for learned-versus-50%-random; additionally compare frozen-state selections under equal cardinality. Always/periodic are deliberate cost controls, not claimed budget-matched baselines. B0 has no hidden RF training expense. The unguided extra-read control tests whether another pass alone suffices.

Auditor MSE convergence and RF ranking are software/mechanism evidence only. A positive novelty or accuracy claim requires natural-transition ΔQ/ΔDice agreement, causal guidance benefit, improvement over simple allocation at equal budget, and faster learning including setup/probe costs. If the intrinsic target fails, do not add architectural modules to disguise it.
