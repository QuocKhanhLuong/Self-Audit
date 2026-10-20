# Novelty assessment, checked 2026-10-20

## Claim to investigate, not a first-ever assertion

At fixed annotation budget and native student, **jointly allocate candidate
pseudo-updates using their class-specific function responses, then verify the
actual nonlinear step relative to a supervised-only reference before EMA commits
it**. The off-diagonal response terms permit useful compensation that independent
source filtering can discard. Dense audit negatives expose BG-to-MYO errors.

A native decoder, DINO pretraining, EMA, confidence thresholds, clean-set feedback,
ridge/QCQP optimization, and line search are NOT separately novel. A useful and
potentially publishable difference requires evidence that the combination offers
more than a smaller step, more labeled optimization, a stronger backbone, or
ordinary reweighting. Local mathematical feasibility is not scientific novelty.

## Primary collisions

| Prior | Already established | Distinction still requiring proof/evidence |
|---|---|---|
| Ren et al., ICML 2018, Learning to Reweight Examples | Clean-set meta-gradient weighting | Joint response curvature and classwise actual-step constraints, not another scalar weight |
| Ren/Yeh/Schwing, 2020, Not All Unlabeled Data are Equal | Learning U example weights using influence | Native group-risk interactions and paired supervised-reference transactions |
| Meta Pseudo Labels, CVPR 2021 | Labeled student feedback changes teacher | Our teacher is plain EMA; candidate acceptance is the intervention, not a novel teacher |
| HCO, MICCAI 2026 | Labeled-driven momentum accepts aligned U directions | Joint source compensation and finite class-specific output checks; sign-only is NOT HCO reproduction |
| GRAD-MATCH, ICML 2021; GLISTER, AAAI 2021 | Joint weighted gradient/subset selection against training/validation objectives | We measure probability responses of candidate displacements with per-class budgets; this alone may be incremental |
| ELN, CVPR 2022 | A learned network finds erroneous pseudo pixels | No extra learned auditor here; evaluate realized updates instead of classifying pixels |

Sources:
- https://proceedings.mlr.press/v80/ren18a.html
- https://arxiv.org/abs/2007.01293
- https://research.google/pubs/meta-pseudo-labels/
- https://papers.miccai.org/miccai-2026/0458-Paper0945.html
- https://proceedings.mlr.press/v139/killamsetty21a
- https://ojs.aaai.org/index.php/AAAI/article/view/16988
- https://cvlab.postech.ac.kr/research/ELN/

Verification scope: original publisher/author abstracts and accessible primary
pages checked. HCO's public abstract describes its trust hierarchy; its PDF fetch
failed in this session. No claim of full algorithm parity or exhaustive novelty
clearance follows. Implement and run an exact HCO comparator only after inspecting
its full algorithm. The reference sign baseline here is intentionally named sign.

## Integrated research questions

RQ1: Does joint allocation retain useful U learning while reducing class-specific
regression relative to a matched independent/sign/ordinary SSL update?
RQ2: Does this local effect survive nonlinear transfer, changing audit patients,
EMA feedback and external evaluation at the SAME annotation/compute accounting?

Positive mechanism test: the compensation example in METHOD.md; neural JVP and
actual transaction tests. Negative tests: numerical solver failure, absent-class
witnesses, wrong/unrepresentative audit labels, decoder nonlinearity, all-candidate
rejection, frozen teacher and source-label corruption. A proof on a TRAIN proxy
cannot answer RQ2. Rotating audit patients is optimization, not independent proof.

Kill criteria: no improvement over same-block plain SSL or extra supervised
compute; rare-class effect disappears with different labeled subsets; gains only
come from confidence/coverage; frequent reject-all makes U irrelevant; an exact
HCO/reweighting baseline matches it at matched budgets. Report these outcomes,
rather than adding another network to preserve a desired story.

Before venue selection, separately verify the venue's current AI-use/authorship
policy; changing supervision does not change the provenance of AI-assisted work.
