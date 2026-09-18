# Cardiac adapter audit and implementation plan

## 1. Scientific purpose and boundary

`cardiac_adapter_v1` is a shared, deterministic, non-learned semantic resolver
for **frozen anonymous partitions**.  It is not part of CUTS or DFC, cannot use
FreeMask candidate/auditor evidence, and must run before the isolated GT
evaluator.

```
FreeMask / Full                -> native semantic output -> semantic freeze -> evaluator
CUTS -> latent -> PHATE/KMeans -> anonymous partition ---+
DFC  -> per-image DFC          -> anonymous partition ---+-> cardiac_adapter_v1
future anonymous baseline      -> anonymous partition ---+       -> semantic freeze -> evaluator
```

FreeMask must not be sent through the adapter merely for diagram symmetry.
Any confidence-neutralized FreeMask analysis is a separate registered analysis,
not adapter logic.

## 2. Allowed and forbidden adapter inputs

The minimal allowed input is a frozen lossless integer partition `[H,W]`, an
image-only central image on the exact same declared grid, and immutable common
manifest/spatial provenance.  The adapter may derive connected components,
area/fraction, border contact, normalized centroid, 4-neighbour adjacency,
holes/enclosure, and deterministic within-image component intensity summaries.
It may use orientation only when the shared geometry contract proves a
short-axis mapping and the required in-plane anatomical direction; current
FreeMask records orientation codes but does not derive that mapping, so v1 must
not require or normally use orientation.

Forbidden: GT/masks/annotations, Dice/HD/validation score, Hungarian or oracle
mapping, FreeMask semantic candidates, candidate bank, audit score, O_fit,
O_select, O_verify, model confidence, learned parameters, method name, patient
specific rules, test-set tuning, or a manually selected label permutation.

The current FreeMask `ontology.py` is evidence for generic geometry notions
(border, connected components, holes/enclosure, adjacency, explicit ambiguity)
only.  Its fitting-view masking, candidate/auditor workflow, predictive score,
study continuity, ontology alternatives, and FreeMask-specific input metadata
are prohibited here.

## 3. Recommended v1 decision: naming/merging only

Choose **Option A: naming + many-to-one merging only**, operating on connected
components of the existing partition.  It may assign several raw components to
one class and may mark components VOID.  It must not split a component by
intensity, watershed, contour, morphology, or any other reconstruction rule.

Connected-component analysis does not violate this constraint: it changes the
unit of *assignment*, not the raw partition's pixel boundaries.  The final
semantic map copies every source component exactly into one semantic label or
VOID.  This permits disconnected same-ID regions to receive different semantic
assignments, which is essential because a raw cluster ID is not an anatomical
object.

Option B (deterministically splitting anonymous regions) could fabricate a
missing boundary and make DFC MinL3 appear capable of four roles.  It is a
different image-segmentation method, requires substantially stronger prior
justification, and should not be hidden in an adapter.  It is rejected for v1.
Thus MinL3 can legitimately yield incomplete semantics/VOID; it cannot be
silently repaired into BG/RV/MYO/LV.

## 4. Input and component-graph contract

Proposed input schema `shared_benchmark.anonymous_partition.v1`:

* identity: `sample_id`, dataset, source/common-manifest ID and SHA-256,
  split, adapter-input schema version;
* lossless partition artifact: 2-D signed/unsigned integer array, exact shape,
  dtype, content hash, file hash; IDs are opaque and may be sparse/negative
  only if the schema permits them explicitly;
* central image artifact: 2-D finite array on the same target grid, image
  content/file hashes, declared normalization representation; no context is
  needed in v1;
* spatial provenance: target grid/transform version, no-crop FOV declaration,
  source geometry/orientation/spacing validity and whether anatomical
  orientation is usable; and
* no method field available to semantic rules.  Producer-specific artifact
  wrappers may differ only before normalization into this schema.

The adapter first labels 4-connected components separately for each raw ID.
For each component it records a deterministic canonical component key based on
geometry (top-left raster seed, then extent), while retaining the opaque source
ID only in provenance.  Features are: area/pixel fraction; touches-border and
border-pixel fraction; normalized centroid; bounding box; finite mean/median
central-image intensity; adjacency boundary length to other components; and
holes/enclosure.  Adjacency is a count of 4-neighbour cross-component pixel
pairs.  A component A encloses B when B pixels inside non-border-connected
holes of A meet a frozen fraction threshold.  Component enumeration and all
ties are sorted by canonical geometry keys, never raw numeric ID.

## 5. Permutation invariance

For every bijection on raw cluster IDs, component masks and all permitted
geometry/intensity features are unchanged.  Implementation must discard raw ID
before rule ranking; canonical component geometry keys, not IDs, order lists
and break non-scientific representation ties.  Provenance can retain a
source-ID-to-component trace, but that trace is excluded from semantic and
validity output hashes.

Required property test: create many random bijections, including sparse and
negative/non-monotone IDs if supported; relabel the same raster partition;
assert exact equality of semantic array, validity array, semantic output hash,
validity output hash, coverage, unresolved-reason histogram, and canonical
geometry trace.  Only raw-ID provenance may differ.

## 6. Conservative anatomy rules

All rules are generic image/topology rules and must be frozen in an external
config whose hash is recorded.  Numeric thresholds below are unresolved until
synthetic topology-only specification is fixed; do not estimate them from GT.

1. **BG candidate.** Components touching the image border are candidates.
   Assign BG only to the unique dominant border-connected external component,
   or to border components that are unambiguously connected through the
   partition's background-like exterior under a declared rule.  Multiple
   fragmented border components may map BG only when the deterministic
   evidence is non-conflicting.  No largest-area fallback may force a non-border
   anatomy component to BG; absent/ambiguous BG is VOID.
2. **LV candidate.** After BG removal, find a unique component B substantially
   enclosed by a distinct candidate A.  A must have a true non-border hole and
   B's enclosure share must clear the frozen threshold.  This pair supports
   `A=MYO`, `B=LV` without using intensity or left/right cues.
3. **MYO candidate.** The encloser A above maps to MYO.  Multiple components
   may map MYO only when each independently participates in the same uniquely
   resolved myocardial enclosure relation; otherwise extra components are VOID.
4. **RV candidate.** A non-BG, non-enclosed component directly adjacent to the
   uniquely assigned MYO may map RV only if unique under the declared rule.
   Never choose RV by raw ID or arbitrary x/y position.
5. **Intensity.** It is only an optional, relative diagnostic/tie disambiguator
   after topology, based on within-image component median/mean ranks.  Absolute
   thresholds are forbidden.  With constant, nonfinite, missing, or tied
   intensity, it resolves nothing and emits VOID.  Cross-site MRI intensity,
   scanner variability, possible inversion, and different method
   normalizations make intensity insufficient as a primary rule.
6. **Orientation.** Disabled by default.  It may be a versioned optional
   tie-breaker only if a future common manifest supplies a verified short-axis
   view and affine-derived pixel direction toward patient-left.  NIfTI axis
   codes alone do not prove this, and neither FreeMask nor the current baseline
   contracts establishes it for all ACDC/M&Ms records.

This deliberately has lower semantic coverage on basal/apical/pathological or
undersegmented images.  That is scientifically preferable to forcing labels.

## 7. Cardinality and many-to-one behavior

Rules act on components, so the count is component count after raw-ID splitting,
not merely K.  Under naming/merging only:

| Raw K | Defensible behavior |
|---:|---|
| 1 | At most one role is supported; normally all VOID unless a sole component is uniquely defensible BG. |
| 2 | BG plus one unresolved/possibly supported role; cannot establish MYO/LV enclosure and RV. |
| 3 | May establish BG/MYO/LV if a unique enclosing pair exists; RV absent/VOID. DFC MinL3 cannot promise four roles. |
| 4 | May establish all roles only if topology resolves them uniquely. Four IDs do not imply four semantics. |
| >4 | Several components can map BG/MYO/RV/LV only under explicit many-to-one evidence; fragments/conflicts remain VOID. CUTS K=10 must not be compacted by ID order. |

The validity map is per pixel, derived from the assigned component.  `VOID`
marks pixels whose component is unsupported, tied, contradictory, or whose
role depends on an inadmissible rule.  Semantic coverage is the fraction of
non-VOID pixels and must be reported with per-role coverage and unresolved
reason counts; valid-only metrics alone are forbidden.

## 8. Deterministic ties and VOID policy

Ties that are scientifically meaningful (two equal LV enclosure candidates,
two RV candidates, competing BG exteriors, equal intensity when intensity
would decide) are not broken by component ID, raster order, method, patient, or
dataset.  All candidates involved become VOID with reason codes such as
`ambiguous_bg`, `ambiguous_enclosure`, `missing_enclosed_cavity`,
`ambiguous_rv`, `unsupported_cardinality`, `orientation_unavailable`, or
`intensity_noninformative`.  Raster ordering is permitted only for stable
serialization of an already non-ambiguous trace, never to turn equivalent
anatomical choices into a semantic choice.

VOID is both pixel-level output and component-level provenance.  The adapter
does not convert VOID to BG.  An evaluator must score a GT anatomy pixel
predicted VOID as a false negative and must report coverage/failure denominators
over the complete frozen inventory.

## 9. Output, provenance, and freeze contract

Proposed output schema `shared_benchmark.cardiac-semantic.v1`:

* `semantic_map [H,W]`, fixed encoding `0=BG, 1=RV, 2=MYO, 3=LV, 4=VOID`;
* `validity_map [H,W]` boolean, false exactly where semantic is VOID;
* adapter version, frozen rule/config hash, code identity, input partition
  content/file hash, central-image hash, common manifest/sample identity and
  target-grid transform hash;
* deterministic component-graph digest; canonical component assignments;
  unresolved component/pixel reasons; per-role pixel counts; coverage; output
  content/file hashes; and
* raw-ID provenance in a separate non-semantic trace, excluded from canonical
  semantic hash comparisons under permutation.

Freeze raw partition plus input image binding before adaptation; then freeze
semantic map, validity map, adapter config/code identity, and all hashes before
the evaluator gains GT access.  Mutation of any bound artifact must fail the
evaluator gate.

## 10. GT firewall and cross-baseline fairness

The adapter API should accept a typed `AnonymousPartitionInput` only.  It must
reject unknown fields and specifically reject names/paths/objects for `gt`,
`mask`, `label`, `annotation`, `dice`, `metric`, `hungarian`, `oracle`,
`reference`, FreeMask evidence, or method-specific semantic correspondence.
The adapter package must not import evaluator/GT modules.  Integration testing
must run it with annotation paths absent/unreadable and demonstrate identical
output if a separately held reference changes.

CUTS raw `int64 [H,W]` and DFC raw little-endian `int32 [H,W]` must be
normalized losslessly into the common integer-array contract with same target
shape, central image representation, source/manfiest identity, and grid
provenance.  Dtype/schema adaptation is allowed before the boundary; semantic
logic may not branch on `CUTS`, `DFC`, profile, K, dataset, or producer.
Current differences needing normalization are CUTS' latent-grid/optional
`target_hw` path versus DFC fixture declared grid, inconsistent manifest
schemas, absolute versus relative paths, and their different central image
normalization representations.

## 11. Test plan before implementation

Place shared tests in `tests/shared_benchmark/`:

| Test | Required assertion |
|---|---|
| Clean BG/RV/MYO/LV topology | Unique border BG, MYO enclosure/LV, and adjacent RV resolve. |
| Random raw-ID permutations | Exact semantic/validity arrays and hashes invariant across many bijections. |
| Split background | Multiple defensibly exterior pieces map BG, or ambiguity is VOID by frozen rule. |
| Many-to-one class | Several supported components can map same role without pixel-boundary change. |
| Three raw clusters | No fabricated fourth boundary; unsupported anatomy is VOID. |
| Missing LV cavity | No LV/MYO forced; correct reasons/coverage. |
| Ambiguous enclosure | All competing affected components VOID. |
| Disconnected same-ID regions | Components receive independent assignments; no ID-level semantic assumption. |
| Border-contact ambiguity | No arbitrary BG selection. |
| Constant image | Intensity cannot decide; topology-only or VOID result. |
| Orientation unavailable | No x/y semantic decision is made. |
| Non-square image | Graph/features/output preserve H,W. |
| Repeat execution | Byte-identical maps/traces/hashes. |
| Input mutation | Partition/image/rule change changes bound hash and invalidates freeze. |
| Firewall negatives | GT/mask/metric/oracle/method fields rejected and no evaluator import. |
| Cross-producer normalization | Equivalent CUTS/DFC normalized artifacts yield same result; no producer branch. |

Synthetic fixtures may use hand-authored partitions and images, but no synthetic
GT may be passed to the adapter.  A distinct post-freeze oracle permutation
diagnostic may later compare frozen semantic maps to GT as an upper-bound
report; it must have a separate evaluator-only API and must never select rules,
K, seed, threshold, or output.

## 12. Proposed architecture and P1 stages

The least disruptive location is a project-level `src/shared_benchmark/`, not
either baseline or `self_audit_maskfree`, because it owns a method-neutral
benchmark contract.  Proposed files:

```
src/shared_benchmark/
  semantic_contract.py     # schemas, encodings, allowed input validation
  region_graph.py          # deterministic components/features/adjacency/holes
  adapter.py               # naming/merging-only v1 rules
  freeze.py                # raw-to-semantic binding and validation
  manifest.py              # common projection/contract validation
tests/shared_benchmark/
  test_manifest_contract.py
  test_adapter_permutation.py
  test_adapter_topology.py
  test_adapter_firewall.py
  test_adapter_determinism.py
```

P1 stages: (1) freeze the shared source-manifest/projection and target grid;
(2) freeze this adapter rule/config specification and synthetic fixtures;
(3) implement contract/graph/adapter with no producer imports; (4) pass the
test matrix and semantic freeze gate; (5) implement isolated evaluation
coverage/denominator policy; (6) only then authorize real raw-to-semantic
generation and later evaluator access.  Acceptance requires all tests above,
no method branch, no GT-capable adapter import/API, permutation invariance,
no source-pixel boundary modification, complete provenance, and coverage
reporting.

## 13. Risks and unresolved items

* Real shared ACDC/M&Ms manifests and source/grid/orientation evidence are
  absent, so orientation cannot be enabled and exact input transform cannot be
  certified.
* Topology-only rules will have high VOID rates on basal/apical, diseased,
  fragmented, or undersegmented partitions.  High VOID is an outcome to report,
  not a reason to add GT-tuned repair.
* A precise enclosure threshold, component connectivity convention, and
  background-fragment rule need freezing from synthetic geometric behavior,
  not calibration to GT.
* An explicit evaluator policy for VOID/empty classes/patient aggregation is
  still required; it must keep complete-cohort denominators.

Adapter implementation is therefore **not ready** until the shared manifest /
grid contract is fixed and the unresolved synthetic-rule constants are frozen.

## 14. Evidence-based readiness verdict

| Item | Verdict | Evidence |
|---|---|---|
| Adapter boundary | **READY** | CUTS and DFC raw cores can end at lossless anonymous `[H,W]` maps; FreeMask remains native semantic. |
| Adapter allowed inputs | **NEEDS FIX** | Conservative allowed/forbidden API is specified, but no common normalized artifact schema exists. |
| Permutation invariance | **NEEDS FIX** | Implementation design and mandatory tests are clear; no implementation/test exists yet. |
| Anatomical rule specification | **NEEDS FIX** | Naming/merging-only topology rules are defined, but enclosure/background constants must be frozen from synthetic behavior. |
| VOID policy | **READY** | Per-component/pixel VOID, coverage, and no-VOID-to-BG policy are specified. |
| Adapter test specification | **READY** | Full synthetic/permutation/firewall/determinism matrix is defined. |
| Adapter implementation ready | **NO** | Shared manifest/grid remains unfrozen and no implementation is authorized by this audit. |

## 15. Data-layer implementation update (2026-09-18)

The prerequisite anonymous-partition input provenance is now locally frozen:
`shared_benchmark_manifest.v1` supplies one canonical sample identity,
patient/split/frame/z/context identity, image-only source hash, native/stored
geometry, and a config-provenanced whole-FOV shared grid.  CUTS and DFC now
consume that same projection and record the manifest/grid/source hashes plus
their normalization provenance.  No semantic map, region graph, component
operation, class selection, VOID assignment, or adapter test was implemented.

The shared schema firewall rejects GT/mask/annotation/oracle/metric-shaped
generation fields and requires a separately supplied root-bounded image-only
source at scientific validation time.  This closes the local adapter-input
schema blocker, while real ACDC/M&Ms materialization and physical server
isolation remain deferred.

### Updated readiness

| Item | Verdict | Evidence and limit |
|---|---|---|
| Adapter boundary | **READY** | Method cores remain raw anonymous-partition producers; no adapter code was added. |
| Adapter allowed inputs | **READY locally** | Common image-only manifest/grid/provenance and firewall contracts now exist. |
| Permutation invariance | **NEEDS FIX** | Still requires adapter implementation and random-permutation tests. |
| Anatomical rule specification | **NEEDS FIX** | Naming/merging-only rule constants remain deliberately unfrozen. |
| VOID policy | **READY (design)** | The audit's per-component/pixel coverage policy is unchanged. |
| Adapter test specification | **READY (design)** | Synthetic topology/permutation/firewall plan remains pending implementation. |
| Adapter implementation ready | **NO** | Anatomical rules, deterministic tie handling, and semantic tests remain blockers; real scientific manifests are also deferred. |

## 16. Pre-implementation specification freeze (2026-09-18)

`benchmark_freezes/cardiac_benchmark_v1/configs/adapter_v1_spec.json` now
freezes the v1 boundary before any semantic code: 4-connected assignment
components, 4-neighbour adjacency, strict hole containment, topology-only
BG/LV/MYO/RV rules, no region splitting, disabled intensity/orientation
resolution, deterministic VOID on every unresolved tie, and fixed
`BG=0,RV=1,MYO=2,LV=3,VOID=4` output encoding.  The spec and its 14 synthetic
expected-output fixtures are hash-bound in proposed freeze
`cardiac-benchmark-v1-6feede910fdcb9c9`.

This closes the rule/tie/fixture design blocker for fixture-only adapter
implementation.  It does not convert the overall benchmark to scientific
READY: real manifests and physical GT-isolation evidence remain deferred, and
no adapter implementation or GT access occurred in this freeze task.

## 17. Regenerated fixture-contract status (2026-09-18)

The invalid pre-implementation freeze `cardiac-benchmark-v1-6feede910fdcb9c9`
was removed and regenerated from explicit pre-freeze sources.  The bad
`high_k_cuts_like_extra_components_void` expectation remains semantic
`BVMLLMB` with validity `1011111`; `VOID` is never marked valid.  Global
validation now enforces the semantic alphabet, rectangular dimensions, binary
validity, exact validity derivation, unique canonical fixture IDs, and
GT-shaped-field rejection before freezing.

Replacement freeze identity:
`cardiac-benchmark-v1-2200633e7f727d37`; payload SHA-256:
`2200633e7f727d379058c11713308a4fd1dace76e8c573cb7ce4d23dba326391`;
adapter specification SHA-256:
`34b1faeb7b40f77c2d4d9789a6e1a342e891fcb8ff8db4edd30957b2ae9d114a`;
synthetic fixture SHA-256:
`cdf03cd9f2d312a5156a8ae444b116a1e2231eff8d05f575804a2b398fe6a823`.
No topology rule, split, inventory, grid, or adapter implementation changed.
