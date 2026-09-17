# Prior art and novelty gate

Initial gate, before implementation (2026-09-17). Astra chooses a falsification baseline, not a novelty claim. This matrix will be expanded with final source verification. `No` means the inspected formulation differs, not a literature-wide absence. RF-inference does not imply GT-free auditor training.

| Closest work | WHEN compute | QC without current GT | WHERE next evidence | Evidence vs output | Audit value | Convergence vs inference | Medical | Remaining distinction |
|---|---|---|---|---|---|---|---|---|
| [ACT](https://arxiv.org/pdf/1603.08983), method §2 | learned halting | no segmentation QC | no read support | recurrent compute | supervised ponder tradeoff | adaptive compute | no | GT-free audit target not supplied |
| [PonderNet](https://arxiv.org/pdf/2107.05407), §2 | halting distribution | no segmentation QC | no support guidance | recurrence | task loss + halting prior | compute allocation | no | not label-free intervention contrast |
| [Metalevel selection](https://people.eecs.berkeley.edu/~russell/papers/uai12-meta.pdf), §2–3 | expected computation value | general latent utility | selects computation | information acquisition | yes | decision quality/cost | no | value of audit is an application, not a new principle |
| [RAM](https://arxiv.org/abs/1406.6247), abstract | fixed-budget glimpses | no QC | learned glimpse location | evidence | task reward | visual compute | no | attention control is established |
| [In-Context RCA](https://arxiv.org/html/2503.04522v2), §2 | no event policy | current-case GT absent, labeled reference bank required | retrieves reference examples | QC evidence, not actor support | no paired intervention value | QC runtime | yes | violates this task's no-GT-reference audit contract |
| [CMF](https://arxiv.org/html/2608.09101v1), §3.3–3.4, 2026 preprint | no event policy | frozen image-text judge scores masks | keep/erase image views | evidence interrogation, then mask arbitration | relative fidelity | cross-domain training result, not convergence proof | no | closest label-free relative evidence threat; no audit-triggered support learning |
| [MQ-Auditor](https://arxiv.org/html/2602.03892v1), §3, 2026 preprint | downstream QC action | runtime only; GT-derived IoU/action targets | no dynamic read support shown | assesses provided masks | quality/action labels | QC/segmentation performance | no | not GT-free auditor training |
| [Chan–Vese vector model](https://www.math.ucla.edu/~lvese/PAPERS/JVCIR2000.pdf), original abstract/formulation | iterative energy optimization | intrinsic image partition energy | contour evolution | output boundary | energy change | optimization, not learned audit | general imaging | region fitting energy itself is not novel |

No broad claim survives: learned computation allocation, adaptive sensing, reference-free quality proxies, frozen independent judges and relative mask scoring all have prior art. The narrow unproven question is whether a **label-free intervention contrast** can schedule a spatial audit that changes a supervised actor's read support and improves its learning curve at matched examples/updates/total cost.

Implementation decision: test a transparent frozen-image compatibility proxy first. Do not hide semantic supervision in a learned quality target, assume every corruption worsens a mask, or add multiple judges to rescue an unvalidated proxy. Remaining survey and method limits will be recorded here before the final verdict.
