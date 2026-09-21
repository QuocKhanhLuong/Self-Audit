# Special-issue fit and novelty gate

**NO-GO for claiming a submission-ready adaptive cardiac annotation method from the current evidence.** The requested theme, “Adaptive and Scalable Vision Models in Dynamic and Resource-Constrained Environments,” is a plausible destination for a validated system. A title match is not evidence that its scientific requirements have been met.

The official [Pattern Recognition call-for-papers page](https://www.sciencedirect.com/journal/pattern-recognition/about/call-for-papers) returned403 during this audit. The exact call text, guest editors and deadline are **not independently verified**. Secondary deadline snippets are not treated as authoritative. The fit assessment below uses the theme supplied by the user, not an invented acceptance checklist or deadline promise.

## What would make the system relevant

| Theme | Current implementation/evidence | Necessary proof |
|---|---|---|
| Dynamic environments | Fixed manual profiles; no case or hardware controller | A frozen image-only selector improves an accuracy/latency or deadline frontier versus fixed and random choices at matched budget |
| Resource constraints | CPU and MPS microbenchmarks on one 24GB machine | Real 4-core/8GB CPU execution, a declared GPU target, cold/warm/tail latency and peak memory for the same checkpoint/profile that supplies accuracy |
| Scalable vision | Offline teacher/deployment separation is architecturally sensible | Account for teacher preparation cost, dataset size scaling, deployable weight footprint, and volume/UI/export costs |
| Adaptation across datasets | No M&Ms execution | Frozen ACDC→M&Ms zero-shot first, then separate native recipe and separately named image-only adaptation |
| End-to-end application | Contract specified, no new frontend built | Geometry round-trip tests, native-mask export, provenance and viewer/runtime validation without conflating UI with method novelty |

## Prior-art stress test

A region-prototype MLP followed by confidence filtering is not a new semantic-identification principle. Label-free ultrasound already combines domain priors, early learning and iterative pseudo labels; its modality and external LV result do not establish three-class MRI feasibility. [Ferreira et al.](https://arxiv.org/pdf/2210.04979).

Deformable attention already samples input-dependent locations, and DynamicViT already conditions inference computation on input. A fixed-K window and manual0/1/2-turn profiles therefore require a specific new mechanism and matched controls before a novelty claim. [DAT](https://arxiv.org/abs/2201.00520), [DynamicViT](https://proceedings.neurips.cc/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html).

An engineering integration can be useful without being a new learning method. The strongest conditional paper thesis would be: **typed semantic anchors plus conflict-aware temporal recruitment reduce named-anatomy swaps at controlled coverage, and this quality transfers to a small deployment model whose compute can be selected under explicit budgets.** Neither clause has passed its experiment. The first clause is currently blocked at the seed gate.

## Falsifiable hypothesis cards for the locked B1 baseline

| Hypothesis | Mechanism and discriminating control | Falsifier | Current status |
|---|---|---|---|
| H1: trusted anatomy/cine cues yield useful seeds | Freeze each cue and fixed combinations; compare per-class precision/coverage/swaps, not only agreement | Any required class fails.95 precision/.05 coverage or patient support gate | Dimensionless precursor failed; full-cine/physical version NOT RUN |
| H2: SSL improves early learning from admitted seeds | Identical scratch versus masked-image initialization, same anchors and downstream schedule; charge SSL cost | No independently measured named-Dice/coverage gain, or higher patient harm | NOT RUN; blocked by H1 |
| H3: one recruitment round improves semantics | Frozen-seed and random-recruitment controls at matched coverage; measure swaps and class harm | More agreement/coverage without better independent quality | NOT RUN; blocked by H1/H2 |
| H4: DW improves deployment frontier | Same admitted teacher and A0, ordinary CNN matched by measured latency and separate MAC control | CNN matches or exceeds DW; useful A0 headroom is lost | NOT RUN; worker toy comparison rejected |
| H5: routing helps under domain/resource shift | Fixed and random routing, same case set and measured budgets; first frozen ACDC→M&Ms | No matched-budget advantage or unacceptable tail error/latency | NOT RUN; data/controller absent |

If H4 fails, remove DW from the proposed final method. If H5 fails, describe fixed-profile deployment as engineering and remove adaptive-superiority claims. If H1 continues to fail, stop the method rather than investing in UI or long distillation. A negative, reproducible named-semantics result is scientifically preferable to a composite claim made from different models' Dice and latency.
