# DSS-US and SGSCN: implementation and execution plan

Revision 4 — 2026-10-01. **PLAN ONLY: no code implementation or execution.**

Scope: DSS-US and SGSCN only. Every frozen native or ACDC raw run has two
separately reported evaluation tracks. Native Track B preserves each paper's
evaluator; ACDC Track B uses one shared raw-ID-majority oracle rule for both
methods. See the [static audit](dss_us_sgscn_codebase_audit_20260930.md).

## Shared rules

- Reproduce DSS-US on CAMUS and SGSCN on PH2 and SYSU-US before adapting to ACDC.
  Preserve original architecture, losses, optimizer, preprocessing, update order,
  stopping rules, clustering and dataset-specific settings.
- Producers, training, inference and feature/clustering fitting cannot read,
  enumerate or probe GT. Use image-only inventories and isolated processes.
- Freeze anonymous raw integer maps [H,W] before any evaluation. Both tracks
  consume the exact same run/sample inventory and raw seal; neither reruns the
  producer, changes the partition, or selects a different seed/level.
- **Track A — GT-free prediction/mapping:** frozen raw -> frozen common semantic
  adapter -> sealed semantic output -> common benchmark metrics.
  GT is permitted only inside the separate metric evaluator.
- **Native Track B — GT-assisted/original:** the same frozen raw -> each paper's
  original matching/selection -> original metrics.
- **ACDC Track B — common GT-assisted oracle:** the same frozen raw -> shared
  raw_id_majority_vote_v1 mapping -> the same common benchmark metric
  contract used in Track A. Label these oracle diagnostics, not paper reproduction.
- GT-assisted results cannot tune the producer or select settings, samples,
  seeds, levels, thresholds or adapter profiles for Track A. Report all
  preregistered runs, including failures; no best-of-track or best-of-seed score.
- Keep baseline/DSS_US/ independent and paper-based, with no copied upstream
  source. Keep baseline/SGSCN/ isolated under GPL-3.0, with its own source,
  environment, provenance and outputs. Shared infrastructure exchanges files
  without importing GPL algorithm code.
- Shared sealing, adapter, metric and oracle utilities remain method-agnostic:
  operate on explicit dataset/class/grid contracts and run receipts, with no
  method-specific mapping, scientific defaults or algorithm imports.

Paths below are proposed modules unless described as existing. These phases
authorize planning only; their runs are future execution tasks.

## Phase 1 — Lock original protocols and both evaluation contracts

**Files/modules**

- baseline/DSS_US/{PROTOCOL_LOCK.md,SOURCE_LEDGER.md,REIMPLEMENTATION.md}.
- baseline/DSS_US/config/native/camus_step{1,2}_<paper_row>.yaml.
- baseline/SGSCN/{PROTOCOL_LOCK.md,UPSTREAM.md,LICENSE,PATCHES.md}.
- baseline/SGSCN/config/native/{ph2_paper,sysu_us_paper}.yaml.
- Per-package evaluation/track_b/spec.json; common Track A dataset/profile locks.
- reports/dss_us_sgscn_protocol_lock.md and native_data_availability.json.

**Work**

Pin papers, supplements, source references, dataset releases and exact native
image/fit/evaluation inventories. Audit whether an official ACDC recipe exists.
Record every scientific value with its source; unresolved paper/code differences
block the affected profile rather than being filled with guessed demo defaults.

For CAMUS lock the actual paper profiles:

- Step I: DSS baseline, Ours_proc, Ours_Aff, Ours_comb, with their paper CRF settings.
- Step II: DSS baseline; Ours_proc/Ours_Aff/Ours_comb + DSS_step2; and
  Ours_proc/Ours_Aff/Ours_comb + Ours_step2.

Recover per-row affinity, preprocessing, features, segment/semantic K, fit cohort
and postprocessing. Do not invent a generic “published affinity” profile.
[DSS-US paper](https://arxiv.org/html/2408.02043v1).

For SGSCN lock PH2 and SYSU-US separately, including dataset-specific learning
rates, enabled paper context loss, loss coefficients, optimizer, exact iteration/
stopping behavior, update order and final prediction pass. Resolve paper/demo
differences explicitly. Missing either native dataset means PARTIAL reproduction.
[SGSCN paper](https://arxiv.org/pdf/2107.04934),
[pinned official source](https://github.com/osmond332/Spatial_Guided_Self_Supervised_Clustering/tree/592efb6e72ceeef15c8be0630a4673eda5dce6f5).

Freeze native Track B evaluator details separately:

- DSS-US Step I: the original per-image segment Dice evaluation, exactly as
  supported by paper/source evidence. Lock segment/GT comparison, background/
  unmatched handling and aggregation only when evidenced. Any unresolved Step I
  detail stays UNRESOLVED/BLOCKED_PROTOCOL; do not assume Step II matching.
- DSS-US Step II: the locked original Hungarian/majority semantic evaluation,
  including its matching, ties, unmatched clusters and aggregation.
- SGSCN: the original paper max-overlap evaluator.

Never evaluate alternative rules and keep the higher result.

**Native Track A compatibility dependency:** the existing common adapter is
cardiac BG/RV/MYO/LV, while CAMUS includes LA and PH2/SYSU-US use lesion/tumor
schemas. Establish and freeze compatible GT-free dataset profiles under the
common adapter interface before native Track A execution, without native GT
access or score selection. Freeze common metric/class/VOID definitions alongside
them. If unavailable, set NATIVE_TRACK_A_STATUS = BLOCKED_ADAPTER; do not silently
rename cardiac labels or claim complete two-track evaluation. This does not block
the original producer or native Track B, and does not make native reproduction
PARTIAL. Existing ACDC adapter stays frozen.

Native producers retain native geometry. Any adapter-only coordinate view must
be predetermined, deterministic and hash-linked to the unchanged native raw;
return semantic predictions to the declared evaluation grid. No forced 224
native reproduction or geometry relabeling.

**Tests**

Configuration/source completeness; exact dataset/profile matrix; image-only
inventory checks; adapter class/grid compatibility; evaluator synthetic matching
tests; producer/evaluator and license/import boundaries.

**Artifacts**

Native protocol/config hashes; availability manifests; paper/code discrepancy
register; Track A adapter/spec and metric locks; Track B evaluator specs;
official-ACDC search record; preregistered seeds/profiles.

**Gate**

Required original producer protocols and native Track B evaluator contracts are
fully specified. Missing data or unresolved original settings have named
BLOCKED/PARTIAL status. An unsupported native adapter blocks only Track A
completion and must be resolved without GT-based tuning; it does not block
original native reproduction.

## Phase 2 — Reproduce native producers and freeze raw outputs

**Files/modules**

- DSS-US independent src/dss_us/{native_dataset,preprocessing,features,affinity,
  spectral,step1,segment_features,step2,crf}.py and scripts/run_native.py.
- SGSCN isolated upstream/, src/sgscn/{native_dataset,producer,raw_output}.py,
  scripts/run_native.py and its own environment lock.
- Each package: provenance.py, seal_raw.py and tests/native/.

**Work**

Run all locked CAMUS Step I/II profiles and both locked SGSCN native datasets.
Implement DSS-US independently from published equations and factual settings;
do not copy or translate upstream source. For SGSCN restrict patches to
necessary image-only I/O, device compatibility and reproducibility instrumentation,
preserving the locked mathematics and execution order.

Use image-only input mounts, correct original fit cohorts and unchanged method
preprocessing. Export actual native-size anonymous integer partitions. Freeze
every separately identified stage/profile/seed run before either evaluation
track. Preserve intermediate diagnostic outputs separately; do not use GT to
choose which stage enters a track.

**Tests**

GT-access tracing and deny-by-default input allowlists; DSS-US equation/stage
tests; SGSCN official behavior parity on synthetic inputs; integer/shape/alignment
checks; complete inventory; no overwrite; seal/tamper verification.

**Artifacts**

Raw maps; input/fit inventories; code/config/environment/checkpoint hashes;
seeds and solver settings; cluster counts, stopping/timing/memory metadata;
access logs; aggregate raw seal and RAW_COMPLETE receipt.

**Gate**

All declared samples verify and the raw seal is valid before evaluation.
Failures remain incomplete. Missing CAMUS blocks DSS-US native reproduction;
one missing SGSCN dataset is explicitly PARTIAL, never full reproduction.

## Phase 3 — Verify reproducibility and evaluate native runs in both tracks

**Files/modules**

- Each package: scripts/verify_native_run.py and tests/reproducibility/.
- Common scripts/evaluate_frozen_two_tracks.py and Track A dataset bindings.
- Per-package evaluation/track_b/{matching,metrics,run}.py; DSS-US keeps Step I
  per-image segment Dice and Step II semantic evaluation distinct.
- reports/native/<method>/<dataset>/<profile>/<seed>/{track_a,track_b}/.
- reports/native/<method>/status.json with separate reproduction/track statuses.

**Work**

Repeat each executable profile in fresh same-seed processes. Repeat the full
DSS-US fit cohort/pipeline; SGSCN additionally checks the locked update trajectory
on a predeclared representative subset. Seal repeat predictions before evaluation.
Measure, record and report same-seed nondeterminism; preserve every original and
repeat raw hash, seed, environment receipt, fit-state hash and repeat-run
diagnostic. Document inherent stochastic behavior without modifying the original
scientific algorithm to hide it. Hash differences alone do not make native
reproduction PARTIAL/BLOCKED when protocol execution is faithful and stochastic
behavior is inherent/documented.

For each frozen run:

1. Verify its raw seal.
2. If a compatible adapter is available, Track A runs the frozen common GT-free
   adapter with image/raw access only; seal semantic outputs, then launch separate
   common metric evaluation with GT. Otherwise record BLOCKED_ADAPTER with the
   same frozen raw reference, without preventing native Track B.
3. Track B mounts GT in its separate evaluator and applies the locked original
   paper rule to the same raw partitions, independently of Track A availability.
4. Verify equal raw seal and sample identities in both track records and
   unchanged raw hashes afterward.

DSS-US Step I uses the locked original per-image segment Dice evaluator; any
unresolved detail remains UNRESOLVED/BLOCKED_PROTOCOL. DSS-US Step II uses the
locked original Hungarian/majority semantic evaluator. SGSCN evaluates original
max-overlap on PH2/SYSU-US. Keep Track A results, Track B paper metrics and
reproduction agreement/disagreement in separate columns; do not merge them.

Report three separate statuses per required native dataset/profile and method:

- NATIVE_REPRODUCTION_STATUS: completeness of the original producer plus original
  paper Track B evaluation, with verified raw/provenance and measured repeat-run
  diagnostics. COMPLETE requires all required native experiments; missing original
  data or settings may yield PARTIAL/BLOCKED with reasons. Inherent/documented
  stochastic variation alone does not prevent COMPLETE.
- NATIVE_TRACK_A_STATUS: COMPLETE when compatible frozen GT-free adaptation and
  common metrics finish; BLOCKED_ADAPTER when no compatible native adapter exists.
- NATIVE_TRACK_B_STATUS: COMPLETE when original-paper evaluation of all required
  sealed runs finishes; otherwise PARTIAL/BLOCKED with explicit reasons.

NATIVE_TRACK_A_STATUS does not determine NATIVE_REPRODUCTION_STATUS.
Paper-score disagreement remains a reported result, not automatic incompleteness.

**Tests**

Same-seed canonical-array/fit-state hash comparison and variation reporting;
evaluator refusal of unsealed inputs; both-track raw-parent equality;
class/geometry/tie behavior; synthetic tests for the independently locked DSS-US
Step I and Step II evaluators; metric aggregation; Track B cannot write raw or
Track A; no oracle imports in the Track A mapping process. Status tests allow
COMPLETE native reproduction with inherent/documented same-seed variation and
reliable verification/provenance. They also allow COMPLETE native reproduction
and COMPLETE native Track B alongside BLOCKED_ADAPTER native Track A; missing
required original data/evaluation still prevents COMPLETE reproduction.

**Artifacts**

Reproducibility report with measured stochastic variation, all raw hashes/seeds/
environment receipts and repeat-run diagnostics; sealed Track A semantic outputs;
native Track A and Track B metrics/overlays; unchanged-raw receipt; separate
NATIVE_REPRODUCTION_STATUS, NATIVE_TRACK_A_STATUS and NATIVE_TRACK_B_STATUS,
including raw lineage and explicit reasons for blocked/partial records.

**Gate**

Native reproduction completion requires faithful original producer execution,
verified sealed raw/provenance, measured repeat-run diagnostics and original paper
Track B evaluation for all required native experiments. Same-seed nondeterminism
is not an automatic PARTIAL/BLOCKED condition when it is inherent/documented.
Use BLOCKED_REPRODUCIBILITY only when nondeterminism prevents reliable
verification/provenance or indicates an implementation/protocol problem; preserve
the evidence and explicit reason.

Native Track A completeness is independent: adapter unavailability alone cannot
make reproduction PARTIAL. Full two-track evaluation requires both tracks
COMPLETE from identical sealed raw; otherwise report the blocked Track A status
separately. Missing original data or unresolved protocol, including unsupported
Step I evaluator details, retain explicit PARTIAL/BLOCKED reasons. Record all
three statuses before ACDC adaptation; GT-assisted results provide no
parameter-selection feedback.

## Phase 4 — Define minimal ACDC protocol after native reproduction

**Files/modules**

- Each package: ACDC_PROTOCOL.md, config/acdc/, src/<method>/acdc_dataset.py,
  acdc_producer.py, provenance.py and raw_output.py.
- Per-method native-to-ACDC deviation table and image-only manifest.
- configs/shared_benchmark/acdc_oracle_raw_id_majority_vote_v1.json.
- Proposed src/shared_benchmark/oracle_mapping.py and
  scripts/evaluate_sealed_acdc_oracle.py, confined to evaluator execution.
- Common frozen ACDC adapter/evaluator bindings and preregistration record.

**Work**

Adopt an official ACDC recipe if one exists. Otherwise create a documented
adaptation only after NATIVE_REPRODUCTION_STATUS is COMPLETE or explicitly
recorded PARTIAL/BLOCKED. NATIVE_TRACK_A_STATUS = BLOCKED_ADAPTER does not block
this transition. Unresolved inherited scientific settings still block execution.

Change only necessary dataset/input/output handling: ED/ES slice enumeration,
file decoding, channel/dtype/range conversion, required benchmark geometry and
output receipts. Preserve architecture, features, losses, optimizer, context
behavior, update order, iterations, stopping, K/clustering, and method
preprocessing. No reduced iterations, extra smoothing or new scientific thresholds.

Select inherited profiles before viewing evaluation outcomes, with a rationale
independent of native scores. Freeze fit sets/split policy, seeds and every
deviation's original behavior, adapted behavior, necessity and test. Keep the
actual 224 benchmark grid and transforms explicit; do not substitute another
method's preprocessing or relabel a native grid as the common grid.

Track A uses the existing common frozen ACDC BG/RV/MYO/LV adapter/spec.

**One ACDC Track B rule for both methods: raw_id_majority_vote_v1.**

1. Read the same sealed raw integer map and its hash-bound GT on the exact
   evaluation grid, inside the separate GT-assisted evaluator only.
2. For each sample and each anonymous raw ID, collect all pixels with that ID
   across the entire sample, including disconnected regions. Do not decompose
   the ID into connected components or write a new raw partition.
3. Count valid GT pixels of BG=0, RV=1, MYO=2 and LV=3 across that whole raw ID.
   Assign every pixel of the ID to the class with the largest count, including BG.
   Break ties by the fixed class order BG, RV, MYO, LV. If an ID has no valid GT
   pixels, assign the common VOID value; keep GT-ignore/VOID handling under the
   frozen common metric contract.
4. Allow many raw IDs to map to the same semantic class. A single raw ID must
   never be split across multiple classes. No connected-component splitting,
   Hungarian matching, max-overlap component selection, Dice optimization or
   class quotas. Raw-ID renumbering cannot affect the result.
5. Save the oracle semantic map in Track B's evaluation directory only. Score
   it with the identical common class/empty-class/VOID/aggregation metric contract
   used in Track A.

Freeze exactly one raw-ID rule/spec/hash, tie policy and class schema for both
methods and all declared profiles/seeds. No method-specific oracle branch.
This is a majority-mapping diagnostic, not a guaranteed Dice optimum or
theoretical upper bound. Native paper evaluators remain unchanged and are not
used for ACDC Track B.

**Tests**

Native-to-ACDC scientific-config diff; synthetic image-only loader and channel
tests; coordinate/range/constant-image checks; approved split/fit membership;
unchanged native regression tests; Track A/Track B process separation;
synthetic common-oracle tests for whole-ID majority/ties, disconnected regions
with opposing GT still receiving one class per ID, many-to-one mapping, raw-ID
permutation invariance, ignored GT/VOID and unchanged raw. Verify identical
oracle rule/spec/hash and metric contract for both methods.

**Artifacts**

Frozen ACDC protocol/configs; complete deviation tables; input/geometry receipts;
fit/seed inventories; Track A adapter hash; one common ACDC oracle spec/hash;
common metric contract hash; preregistration.

**Gate**

All inherited scientific fields known; only necessary I/O deviations approved in
the protocol; official-ACDC search resolved; both evaluation contracts frozen.
Both methods use the same ACDC oracle rule/spec and metric contract.
No native Track B score used to choose an ACDC producer or Track A setting.

## Phase 5 — GT-free ACDC smoke, full generation and raw sealing

**Files/modules**

- Each package: scripts/run_acdc.py, verify_acdc_run.py, seal_raw.py, tests/acdc/.
- Existing shared write_raw_partition / verify_raw_partition APIs with generic
  contract validation; each baseline supplies its own provenance receipt.
- Separate .runtime/<method>_acdc_<profile>_<seed>_<run>/raw/ directories.

**Work**

First run a predeclared small image-only subset through the complete inherited
algorithm, with no reduced iterations or scientific shortcuts. Check same-seed
repeats, access logs, geometry, resource limits and artifact integrity.

Then generate the full approved manifest under the same frozen settings and
fit-set policy. Save integer [224,224] raw maps, source/model-input/grid hashes,
seeds, configs, method/implementation identity, fit/checkpoint state, stopping
metadata, timing, code/environment hashes and access traces. No label/GT/oracle
fields. Seal the exact complete inventory before either evaluation track.
Retain all preregistered runs; no alternate-seed rescue or best-run selection.

**Tests**

GT firewall; exact output shape/dtype; deterministic repeats; resource bounds;
manifest completeness; tamper detection; resume identity enforcement; rejection
of nonempty output roots and incomplete seals; no evaluator in producer imports.

**Artifacts**

Smoke verification report and sealed smoke raw; full sealed ACDC raw runs;
RAW_COMPLETE receipts; configuration/provenance/access logs and raw-only previews.

**Gate**

Smoke must pass before full generation. Every full raw sample verifies before
RAW_COMPLETE. Neither evaluation track may accept an incomplete/unsealed run.

## Phase 6 — Evaluate every sealed ACDC run in both tracks and report separately

**Files/modules**

- Common frozen adapter API and scripts/evaluate_visualize_shared_benchmark.py.
- Proposed scripts/evaluate_frozen_two_tracks.py: method-agnostic seal preflight,
  explicit dataset/grid/class contracts and separate evaluator launches.
- Shared scripts/evaluate_sealed_acdc_oracle.py and frozen
  raw_id_majority_vote_v1 spec; no per-method ACDC oracle entrypoints.
- reports/acdc/<method>/<profile>/<seed>/{track_a,track_b}/.

**Work**

For every sealed run, verify the aggregate raw seal and bind both tracks to the
same raw sample/file hashes. Track A applies the frozen common adapter without
GT, seals its semantic outputs, and then runs the common metric evaluator.
Track B evaluates the same raw with the single frozen raw_id_majority_vote_v1
oracle and the same common metric contract. Both metric processes may read GT;
neither producer nor Track A adapter may do so. Both reports bind to identical
raw manifest, sample and partition hashes; reject any mismatch.

Report Track A per-class/foreground Dice, coverage/VOID and existing common
metrics. Report the same metrics for Track B in separate oracle tables, with
the common rule/spec hash and aggregation stated. Native paper metrics remain
in native reports only. Never present an oracle score as GT-free performance.

Publish all declared profiles/seeds, failures, the three separate native statuses
and deviations. Compare methods only when cohort, geometry and metric contracts
match. Reverify seals after evaluation; no result-triggered producer/adapter
selection, tuning, raw rewriting or promotion of an oracle output.

**Tests**

Raw-parent/sample/hash equality across tracks; raw and semantic seal preflight;
correct class/grid/adapter-spec linkage; identical oracle rule/spec for both
methods; common metric-contract equality; absence of oracle calls in Track A
mapping; evaluator write isolation; complete result inventory; unchanged seals.

**Artifacts**

Sealed Track A semantic outputs; separate Track A and Track B ACDC reports,
metrics and overlays; paired raw-lineage manifest; common oracle/spec and metric
hashes; DSS-US/SGSCN comparison eligibility table; final unchanged-hash receipt.

**Gate**

Every completed frozen run has both tracks reported from exactly the same raw
predictions. ACDC Track B must use the same oracle rule/spec for both methods.
Any blocked track remains explicit. Low scores are valid results, not permission
to tune or substitute Track B for Track A.

## Completion checklist

- Native protocols: CAMUS; PH2 and SYSU-US, with partial availability disclosed.
- All raw predictions anonymous, verified and frozen before evaluation.
- Every native and ACDC run has separate Track A/Track B lineage and reports;
  blocked native Track A records retain the same frozen raw reference.
- NATIVE_REPRODUCTION_STATUS, NATIVE_TRACK_A_STATUS and NATIVE_TRACK_B_STATUS
  are reported separately; adapter absence alone never makes reproduction PARTIAL.
- ACDC protocol inherited unchanged except necessary documented dataset/I/O.
- No GT-assisted feedback into producer or Track A selection.
- Native Track B retains each original evaluator; ACDC Track B uses one shared
  raw-ID-majority oracle rule and common metric contract for both methods.
- DSS-US independent reimplementation; SGSCN isolated GPL-3.0.
- Shared benchmark infrastructure stays method-agnostic.

Only this plan document is revised. No implementation, training, inference,
environment setup or evaluation starts as part of this task.
