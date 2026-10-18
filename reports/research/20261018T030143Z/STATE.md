# Self-Audit parallel theory and method research

## Snapshot and authority

- Session: 20261018T030143Z; start 2026-10-18 03:01:43 UTC.
- Research worktree: /Users/alvinluong/Self-Audit-research-20261018 (detached).
- Pinned remote main: f00e4faa52204f8e6863f76a493684b68ed3a132, freshly fetched.
- Original worktree: /Users/alvinluong/Self-Audit, HEAD 92a60744efef85558d7c0226c84b7c49d47217ec; 236 existing modified/untracked file fingerprints preserved before research.
- Publication status: LOCAL_ONLY; no commit, push or merge authorized.
- Authority: current user Phase-I research request. Production is read-only.
- The user explicitly permits questioning/rejecting prior D/U hypotheses. ASTRA_NEXT's instruction to preserve D/U applies to the production contract; it does not preselect the research conclusion. No replacement is implemented in production in this session.
- Read entry documents: AGENTS.md, ACTIVE_PIPELINE.json, docs/CURRENT.md, docs/RUNBOOK.md, docs/prompts/ASTRA_NEXT.md.
- CodeGraph absent in this worktree. Use Semble for semantic discovery, then direct known-file reads. All shell commands start with rtk (typically rtk proxy).

## Shared information contract (given identically to all six workers)

Research question: How can a strong dense student learn named cardiac segmentation and useful native-resolution structure from uncertain coarse image-derived teacher guidance, and improve beyond that guidance without amplifying errors?

Let X denote raw full cine images; M observed acquisition metadata with validity/missingness; F pretrained features with explicit upstream lineage; O={BG,RV,MYO,LV}; A image/geometry-derived uncertain anchors with source identity and dependence; Q historical propagated full probabilities; h support when a justified source decomposition exists; Z conditional evidence only where its derivation/support is valid. Q is not silently replaced by Z. UNKNOWN=255 is abstention, not background or a fifth learned class. Distinguish missing, conflicting, and duplicated evidence.

Permitted training information: images and legitimate metadata, frozen TRAIN-bank distributions/reliability/features and their provenance, explicit image/anatomical priors, reviewed pretrained backbone. No target pixel masks, source semantic heads, scribbles or image-level labels may be smuggled into a no-task-annotation claim. If proposed as an alternative setting, label the additional supervision explicitly. Historical priors are not verified anatomical truth.

Inference: declare fixed single-cine/image-only inference versus target-domain adaptation. Per-image preprocessing differs from statistics fit across a target cohort. Do not introduce hidden teacher/target-cohort access.

Current documented path (coordinator verifying code/receipts): train.py -> experiments/layer_guided_student_v1; LayerGuidedStudent with official DINOv3 ConvNeXt-Tiny local loading path. Enabled recipe is full-view feature cosine plus historical full Q/reliability coarse CE, ordinary AdamW; native detail objective absent; guard/refiner off. Shared encoder paths exist. Presence of dormant source is not execution evidence. Old small-student results are not new-student results.

Production, configs, runtime, dependencies, immutable reference and historical runs remain untouched. No patient/GPU training, weight/dataset download, TEST/VAL tuning, automatic commit/push/merge or process termination. Tiny manufactured/symbolic probes only, inside this session directory; they are not patient evidence or proofs of general claims.

All candidate methods are PROPOSED_NOT_APPROVED. Agreement among workers cannot activate Phase II. The human selects the emphasis/design after the final decision brief.

## Independent investigation and review protocol

Six workers R1-R6 share only this contract and pinned source initially. Each must write an independent initial draft in its own dossier BEFORE reading another dossier. Preserve that draft verbatim thereafter, append dated revisions. Independent drafts are not statistically independent evidence.

Workers own only DOSSIERS/RN.md and probes/RN/. Coordinator owns STATE.md, shared ledgers, CROSS_REVIEW.jsonl, decision brief/handoff and probes/coordinator/. Workers are not alone in this worktree: do not overwrite/revert other agents' files. No worker may edit production.

Search current PRIMARY literature; verify title, authors/year/version/publication status, and precise theorem/section/equation or official code location for important claims. Include contrary prior art and simple controls. Record inaccessible content without inventing conclusions. Verify chat leads afresh. Keep source metadata/search notes in the dossier.

Evidence-card fields: ID; plain-language statement; status (CODE_FACT, EMPIRICAL_OBSERVATION, LITERATURE_RESULT, DERIVED_RESULT, COUNTEREXAMPLE, CONJECTURE, PROPOSED_NOT_APPROVED); scope; assumptions; formal claim with spaces/quantifiers/norms/constants; proof or missing step; counterexample; closest prior result; code/evidence references; implications; limitations; falsifiable prediction. Do not call classical lemmas novel. Separate output-space mathematics from network parameter updates; mark truth-dependent constants as latent, not operational certificates.

For each serious hypothesis include: claim -> nearest prior method/result -> shared mechanism -> differing supervision/assumptions -> unresolved gap -> simplest plausible baseline. Distinguish implementation defects, method-family limits, theory gaps and task-level gaps. Seek constructive positive results with noise/mis-specification, not only impossibility.

After drafts, coordinator deduplicates findings. Three distinct challenge-response-revision cycles follow:
1. Mechanism/mathematics (initial ring R1<->R4, R2<->R3, R5<->R6).
2. Prior-art/simpler replacement (rotated reviewers).
3. Failure in actual setting and no-GT kill criteria (rotated reviewers).

Each central surviving hypothesis needs >=2 non-author reviewers; theorem-level claims need mathematical and setting/prior-art review. Authors respond by correction, evidence, narrowing or withdrawal. Save actual records: round, claim ID, author, reviewer, objection, cited evidence/counterexample, author response, modification, remaining disagreement, verdict (SURVIVES/NARROW/NEEDS_EVIDENCE/REJECT/DUPLICATE). No fabricated execution or consensus. Do not skip an author response.

Use English in technical artifacts. AI-assisted literature search, mathematical drafting and agent review are disclosed; no human-read claim is implied. This is a scoped mechanism search, not an exhaustive systematic review or publication-readiness verdict.

## State map and execution ledger

Coordinator completed a bounded source inspection and current CPU suite. Later remote changes will be recorded as deltas, never mixed silently into this pinned analysis.

| Surface | Implemented | Enabled by canonical coarse recipe | Software evidence | Pretrained/patient/anatomical evidence at pinned snapshot |
| --- | --- | --- | --- | --- |
| Canonical train.py -> training_v1.runner | Yes; train.py:56-72 resolves and checks the actual module | Yes; --phase coarse only; native rejected | Current 153/153 tests and layout check below | No real-run receipt located |
| Official DINOv3 ConvNeXt-Tiny + LayerGuidedStudent | integration/artifacts.py:164-202 explicitly loads local pretrained weights and four stages | Required, with no random fallback | Mocked constructor/interface tests; full-size random reference smoke is separately labelled | Official-weight execution NOT VERIFIED in checked-in receipts |
| Feature cosine | runner.py:194-220; feature_project consumes encoder stage 3 in layer_upgrade/model.py:112 | Yes, full-view, fully native aligned support | Current synthetic tests | Patient execution NOT VERIFIED |
| Historical full-Q coarse CE | runner.py:430-474; training_v1/data.py:batch/prepare_targets | Yes, reliability/class permission and batch activation; no Q->Z | Current manufactured optimizer/checkpoint tests | Named anatomy accuracy NOT VERIFIED |
| Exact D/U detail factorization | layer_upgrade/model.py:78-114; layer_upgrade/layers.py:compose_logits | Forward computes detail, but no loss on native logits | Original operators/parity and zero-detail tests | No learned native-detail claim |
| Guard / frozen energy refiner | layer_upgrade/guarded.py, refine.py | OFF; no active call in runner's loss/backward/step | Isolated reference operators tested | No patient benefit demonstrated |
| Dense/gated prediction and strict reload | training_v1/checkpoint.py:load_checkpoint/predict_cine | Available API; no automatic evaluation in trainer | Synthetic save/load and image-only prediction | No new-student held-out anatomy receipt located |

Active gradient path: runner.py:433 calls model(images); lines 434-470 combine feature cosine with optional CE on coarse_logits; lines 473-493 zero gradients, backward, clip and ordinary AdamW step. Feature loss reaches the shared encoder/adapter and feature projection; coarse loss reaches encoder/decoder/coarse head. Neither loss depends on detail-only parameters. Since detail_head weight and bias initialize to zero (model.py:82), no detail-only gradient/update is created and its output remains zero. Shared-encoder interference remains possible; direct gradient isolation is not functional independence of all model outputs.

Current local checks (Python /Users/alvinluong/miniforge3/bin/python, CPU):
- rtk proxy env PYTHONDONTWRITEBYTECODE=1 python scripts/check_repo.py: exit 0, PASS_CANONICAL_LAYOUT.
- rtk proxy env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider --junitxml=reports/research/20261018T030143Z/probes/coordinator/current_tests.xml: exit 0, 153 passed in 3.46s.
- Test evidence is manufactured/software only; no datasets or pretrained weights downloaded/loaded by this research session.

Located checked-in receipts: reports/cleanup_20261017/LOCAL_CHECKS.json and reports/layer_level_upgrade_20261015/reference/{FULL_MODEL_CHECK,PROBE_RESULTS,MANIFEST}.json. The frozen historical bank metadata is reports/stable_guidance_main513ddda_b12_20261016/completed_run/run/bank/FROZEN.json. It is not a patient-array training run. The user previously reported a mirror weight download and host preparation, but no corresponding real DINO load/patient run/quality receipt was located in this snapshot; those execution claims remain UNVERIFIED, not false.

Only the explicitly identified historical question (what information/producer the old full-Q bank contains) may use its pinned metadata/archive source read-only. No legacy training surface is restored. Historical small-CNN quality numbers are not imported as new-model outcomes.

### Worker dispatch correction

The first six workers (/root/research_r1 through research_r6) were launched with inherited model/researcher-role settings. While they were still running, the user explicitly requested Astra ultra. All six were interrupted before completion; none count as completed independent drafts or reviews. Any partial dossier is retained under probes/coordinator/interrupted_drafts/.

Replacement workers are requested explicitly with model gpt-6-astra and reasoning_effort ultra, using the default role so a fixed researcher-role reasoning setting does not override ultra. Tool metadata confirms requested parameters/agent dispatch only; resolved backend model identity is not independently exposed. Replacement workers must not read interrupted drafts or other workers' first drafts before their own initial draft is saved.

| Track | Active worker | Responsibility | Requested model / effort |
| --- | --- | --- | --- |
| R1 | /root/astra_ultra_r1 | Information and semantic identifiability | gpt-6-astra / ultra |
| R2 | /root/astra_ultra_r2 | Layer operators and native observability | gpt-6-astra / ultra |
| R3 | /root/astra_ultra_r3 | Image-grounded correction evidence | gpt-6-astra / ultra |
| R4 | /root/astra_ultra_r4 | Distillation and feedback dynamics | gpt-6-astra / ultra |
| R5 | /root/astra_ultra_r5 | Optimization and rare-class transfer | gpt-6-astra / ultra |
| R6 | /root/astra_ultra_r6 | Prior-art collisions and falsification | gpt-6-astra / ultra |

All six replacement spawns succeeded and ran concurrently. The coordinator is /root, with no claimed model override. Independent first-draft completion/hashes and actual review completion will be recorded when observed.

Planned reviewer rotations (a plan, not evidence of completed review):
- Round 1: R1 reviews R4; R4 reviews R1; R2 reviews R3; R3 reviews R2; R5 reviews R6; R6 reviews R5.
- Round 2: R6 reviews R1; R1 reviews R2; R4 reviews R3; R5 reviews R4; R2 reviews R5; R3 reviews R6.
- Round 3: R5 reviews R1; R6 reviews R2; R1 reviews R3; R2 reviews R4; R3 reviews R5; R4 reviews R6.


## Independent draft sealing and review launch

All six requested Astra/ultra workers delivered independent first drafts. Exact hashes and byte-preserving coordinator snapshots are in `probes/coordinator/independent_manifest.json`. Initial deduplicated findings were written to FINDINGS_AND_THEORY.md before the coordinator's general Round-1 launch. R5 and R6 independently began their assigned reciprocal exchange after both had sealed their drafts, slightly before that general launch; no cross-reading before their own initial drafts was reported. This scheduling deviation is recorded rather than retroactively presenting simultaneous starts. It did not replace any review cycle.

The first round has produced substantive narrowings: region correctness is not strict improvement over a pointwise teacher; local inverse bounds need candidate localization and a fixed metric; hard graph constraints can change discontinuously under tiny edges; native output correction and an actual network update are separate claims; and a one-step detached proposal may exactly reproduce the direct-energy gradient. Subsequent records and dispositions are authoritative in CROSS_REVIEW.jsonl.

Two workers reported first probe imports without PYTHONDONTWRITEBYTECODE, which may have produced ignored Python bytecode in this new research worktree. No intentional production source edit occurred. Generated bytecode will be inventoried and relocated into the session at close; this does not authorize touching another checkout. Subsequent imports use bytecode suppression.


## Phase-I closure and explicit report-publication update

All six Astra/ultra workers completed independent drafts and three genuine review cycles. CROSS_REVIEW.jsonl has93 records:29 in Round1,30 in Round2,32 in Round3, and2 supplemental mathematical/source audits;18 main directed exchanges. Each of H06 and H11 received3 nonauthor reviewers. All retained central math has mathematical and setting/prior-art review; scoped coverage is in probes/coordinator/review_coverage.json. No reviewer agreement is treated as empirical independence or proof of anatomical assumptions.

Root consolidated nine theorem/finding cards,22 prior-art collision rows and14 hypothesis dispositions, retaining only2 central RQs. The final direction is conditional image-evidence correction followed by separately qualified transfer. A semantic-first direct-region alternative remains for human emphasis selection. Every method remains PROPOSED_NOT_APPROVED.

Fresh source suite153/153 passes; no real pretrained/patient/anatomy receipt was located. The same-point weighted-gradient probe passed five float64 cases with error≤2.93e−15; wrong-metric and stale-origin controls fail equality as expected. These are CPU/software/manufactured evidence only. The initial R3 fixture failure and torch-version typo were preserved/corrected by append; a coordinator JSON merge rejected a nonallowed verdict string and the reviewer corrected RETAIN_SCOPED to SURVIVES without altering its scope. No failure was relabeled as a patient success.

Closing integrity: pinned research HEAD and remote main both f00e4faa52204f8e6863f76a493684b68ed3a132 before report commit. No tracked production/reference changes. Original checkout HEAD92a60744efef85558d7c0226c84b7c49d47217ec and236 recorded file hashes unchanged. Six generated ignored bytecode files were relocated into probes/coordinator/generated_bytecode with hashes; no original checkout file was removed or modified. See integrity.json and bytecode_relocation.json.

**Authorization changed after research completion:** the human asked “xong chưa, cbi kết luận và đẩy report lên git đi”. This explicitly authorizes report commit/push; it does not activate PhaseII, patient training or a main merge. Publication target is the separate branch `research/theory-20261018-astra-ultra`. The allowlist contains consolidated reports, own probes, independent draft snapshots, review records and source metadata. Downloaded third-party PDFs/full code, bytecode, interrupted unfinished artifacts and original workspace fingerprints remain local; their locations in dossiers are local-only evidence references. Initial worker disclaimers describe their own execution and remain verbatim. Final remote commit verification is recorded in the local publication receipt and delivery message.

Publication formatting check: CSV CRLF line endings were normalized to LF. The exact preserved R6 initial snapshots each end with a blank line; that immutable-byte property is retained as a documented whitespace-check exception. All other staged files pass git diff --cached --check. No substantive draft text or original hash was changed.
