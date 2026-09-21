# Source of truth, identities and replay boundaries

**Authority order:** immutable source/data identities and executable evidence → root reports00–12 → qualified worker investigations → historical narrative. A worker saying succeeded records delivery, not acceptance of its conclusions. Research proposals and engineering targets are not experimental observations.

## Repository and change scope

Repository [QuocKhanhLuong/Self-Audit](https://github.com/QuocKhanhLuong/Self-Audit). Verified main after checkout and fast-forward pull: `d4f503b7107eca4e8ac85f37a5dd01ec50479666`, exactly the requested merge. Audit branch: `QuocKhanhLuong/astra-pseudolabel-v3-audit`; worktree `/Users/alvinluong/orca/workspaces/Self-Audit/astra-pseudolabel-v3-audit`.

Authorized changes are confined to `src/self_audit_pseudolabel/system_v3.py`, its explanatory `docs/pseudolabel_system_v3.md`, new `tests/test_pseudolabel_v3_audit.py`, and `reports/astra_v3/`. Canonical Dynamic Window, AnnotationExpert, data loaders, old C0/C1/event evidence and main remain untouched. The original 236 dirty/untracked files were hashed before work and rechecked without mismatch. [Source/host identity](evidence/source_identity.json), [original hashes](evidence/original_dirty_hashes.json). Final validation/staging receipts are saved alongside this report before the isolated-branch commit; the Git commit itself identifies the final package without a self-referential hash.

## Experiment and software identities

| Identity | Status and evidence |
|---|---|
| Base merge d4f503b | Original three scoped tests pass; UNKNOWN255 and ambiguous-mixture regressions fail |
| astra-v3-software-audit-fix-v1 | Two contract corrections;16 scoped tests pass; no trained-weight quality claim |
| astra-v3-r0-geometric-seeds-20260920-v1 | Aborted on unknown physical units before GT; [blocker](evidence/round0_v1_blocker.json) |
| astra-v3-r0-dimensionless-seeds-20260920-v2 | Seven frozen heuristic arms;16 volumes; independent named-semantics gate FAIL |
| astra-v3-B1-cine-anchored-ssl-v1 | Exact proposed next baseline, NOT IMPLEMENTED/TRAINED; gate-blocked |

Frozen v2 manifest SHA256 `c0be95e4f3e6bf63277e576f80f0d89aacc949b48630bf211883e786d141196e`; config `c8ec87720cbea78ff3b6b287e888e79d5a379e3da8903e7b0ac32329295c9dfc`; generator `acdea9e6127afa1212a6a2816ceb3585187d223d7eb265ef2fd66f6b9414d7f4`; original patient split `bc42574017276a6defc668bf533330fd41a64a5e90a75ea1b8958fc9e469265a`.

Freeze at 2026-09-20T16:17:58.062889Z. H finished verification at 16:21:07.606241Z, then first opened GT at 16:21:07.606312Z. Root evaluator replay again checked all freeze dependencies before and after access. Every evaluated mask's identity is recorded by the independent evaluator; no segmentation masks or raw MRI arrays were copied into this report package.

[Original frozen manifest](evidence/round0_v2_FROZEN.json), [config](evidence/round0_v2_config.json), [immutable generator](evidence/run_round0_seeds_dimensionless.py). Byte-identical generated-label NPZs and patient split are archived under [round0_v2_archive](evidence/round0_v2_archive/ARCHIVE_MAP.json). This archival map relocates copies without rewriting the original freeze. Source-image hashes and original absolute paths remain in the manifest. Licensed original images and evaluator masks are required to rerun the metrics; archived predictions alone do not recreate GT.

## Evidence register

| Question | Accepted evidence | Limits |
|---|---|---|
| Tensor/gradient/profile behavior | [Root tests](../../tests/test_pseudolabel_v3_audit.py), [before](evidence/audit_tests_before_fix.txt), [after](evidence/audit_tests_after_fix.txt) | Synthetic CPU software behavior |
| Interventions/identifiability | [Baseline](evidence/root_interventions_baseline.json), [candidate](evidence/root_interventions_candidate.json), [script](evidence/root_interventions.py) | Random weights; no anatomical utility inference |
| Actual data availability/geometry | [200-image inventory](evidence/root_header_geometry.json) | Header/array facts bounded to inspected roots; native physical truth unavailable |
| Round-0 named quality | [H metrics](workers/H_evidence/evaluation_results.json), [H evaluator](workers/H_evidence/evaluator.py) |4 train+4 dev patients; sparse heuristic labels, not trained teacher |
| Independent metric/tamper check | [Root verifier](evidence/root_verify_evaluator.py), [receipt](evidence/root_evaluator_verification.json) | Separate post-freeze evaluator process; not a malicious-process security proof |
| Inference resources | [Root benchmark](evidence/root_benchmark.json), [script](evidence/root_benchmark.py) | Random FP32 weights;12-core24GB macOS CPU/MPS; not constrained target/CUDA |
| Historical failures | [Copied aggregate reports](evidence/historical_nogt/SOURCE.json) | Different phase aggregation and acquisition provenance; not a matched R0 comparison |
| Worker completion | [Final tasks](orchestration/final_tasks.json), [final workers](orchestration/final_workers.json) | Completion messages do not validate report content |

## Reproduction protocol

Use Python3.11.16 from `/Users/alvinluong/miniforge3/bin/python`; dependency versions are recorded in source_identity.json. The audit did not modify the project's dependency specification or claim a fully portable environment lock. Shell commands used the user-required RTK wrapper.

```bash
rtk proxy env PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /Users/alvinluong/miniforge3/bin/python -m pytest -q tests/test_pseudolabel_system_v3.py tests/test_pseudolabel_v3_audit.py
```

Root scripts are retained verbatim with their experiment-specific paths. Inspect their arguments and output locations before replay; do not overwrite frozen output directories or substitute data beneath a recorded identity. On another machine, construct a declared relocation map and verify byte-identical dependencies before GT evaluation, or allocate a new experiment identity. Root benchmark measurements include fresh-process first-forward and peak RSS,40 warm CPU repeats and30 synchronized MPS repeats; method details are in report 07. Do not replace them with F's non-inference-mode timings.

Primary evaluation is equal patient→ED/ES phase→RV/MYO/LV class mean of native stored-volume Dice, no background, UNKNOWN as false negatives, no Hungarian mapping. Physical-space correctness is a separate unpassed gate. Historical development exposure prevents calling this cohort an untouched confirmatory test. No checkpoint was selected on these GT results; future development stop/go feedback must remain disclosed.

## Primary literature and availability

Verified source URLs and claim-specific limitations are linked in [report 04](04_PSEUDOLABEL_RESEARCH.md). Root corrected Ferreira2210.04979, Bai1907.02757 and Qin1806.04066, distinguished CineMA SSL from supervised weights, and inspected CUTS' GT-assisted binary foreground selection. CM-TAPE primary protocol and the exact official special-issue call remained inaccessible; no numeric/protocol/deadline claim is accepted for them. A primary source that cannot be accessed stays UNKNOWN. Worker D/D2 text is not a substitute for that evidence.

## Required deliverables

[00 Executive summary](00_EXECUTIVE_SUMMARY.md) · [01 Orchestration](01_ORCHESTRATION.md) · [02 Code](02_CODE_AUDIT.md) · [03 Supervision](03_SUPERVISION_AND_IDENTIFIABILITY.md) · [04 Research](04_PSEUDOLABEL_RESEARCH.md) · [05 Dynamic Window](05_DYNAMIC_WINDOW_REVIEW.md) · [06 Data](06_DATA_AND_FULL_CINE.md) · [07 Resources/system](07_RESOURCE_ADAPTATION.md) · [08 Tests](08_TEST_RESULTS.md) · [09 Experiments/baseline](09_EXPERIMENT_PLAN.md) · [10 Special issue](10_SPECIAL_ISSUE_FIT.md) · [11 Red team](11_RED_TEAM.md) · this source register. Workers A–G plus H are under `workers/`; D2 revises D with an original preserved as unverified.

## Audited source hashes

| File | Audited base SHA256 | Candidate SHA256 |
|---|---|---|
| `src/self_audit_pseudolabel/system_v3.py` | `4c07ee074daa060f0a33095162c5a75096e745f97082dcc34d8fa38cfda83226` | `1077187c880f463d38f171eb8d2690cf5f0aa0ca022d797e25d1e34a36bafb06` |
| `src/self_audit_pseudolabel/__init__.py` | `ce4a0df7912b1c0642bed019a378e0023497b4df22b0bef6979bfeb2b98b165d` | `ce4a0df7912b1c0642bed019a378e0023497b4df22b0bef6979bfeb2b98b165d` |
| `src/self_audit/models/dynamic_window.py` | `5cf0a07ef9b50254d5bcffce605aea2849772fb0bed16a95685f17ab6eda9fd4` | `5cf0a07ef9b50254d5bcffce605aea2849772fb0bed16a95685f17ab6eda9fd4` |
| `src/self_audit/models/annotation_expert.py` | `e85a4572982954812b805fe4ced28acb873233d99dc4184f591e65f0458deb07` | `e85a4572982954812b805fe4ced28acb873233d99dc4184f591e65f0458deb07` |
| `src/self_audit_maskfree/data/discovery.py` | `d32b5fbbb0415078b54f08715414ea94f08831a18e155c4bcc150cac145b0ce3` | `d32b5fbbb0415078b54f08715414ea94f08831a18e155c4bcc150cac145b0ce3` |
| `src/self_audit_maskfree/data/dataset.py` | `653a0d74da405405252fe739891cc48f51088adf1ecacb72f77d6cc279d214f4` | `653a0d74da405405252fe739891cc48f51088adf1ecacb72f77d6cc279d214f4` |
| `src/self_audit_maskfree/data/firewall.py` | `92767510e96f4feb816d60e190a6a913cc231d97d0f180c7ffce857ac3ceaca3` | `92767510e96f4feb816d60e190a6a913cc231d97d0f180c7ffce857ac3ceaca3` |
| `src/self_audit_maskfree/losses.py` | `7cf5aaeb40c7a31c75647b24eff3ccdbe74c49717a62655fe1c425260ab4bf9b` | `7cf5aaeb40c7a31c75647b24eff3ccdbe74c49717a62655fe1c425260ab4bf9b` |
| `src/self_audit_maskfree/ontology.py` | `22c14e9984c7e5c4b27d3a7d6a6bd85a1c839744bc0d920656956cfb8eb2825c` | `22c14e9984c7e5c4b27d3a7d6a6bd85a1c839744bc0d920656956cfb8eb2825c` |
| `src/self_audit/data/mnms.py` | `e5bcaae36e1223bbf1ee5000f67a091ea514cc3bc91aaa4d9af191efc5e79575` | `e5bcaae36e1223bbf1ee5000f67a091ea514cc3bc91aaa4d9af191efc5e79575` |
| `scripts/preprocess_acdc.py` | `d35608111c8e7b1bd3f15d7b491d5923d110262872248c2ce1fe4164bbb986b7` | `d35608111c8e7b1bd3f15d7b491d5923d110262872248c2ce1fe4164bbb986b7` |
| `tests/test_pseudolabel_system_v3.py` | `4e4cb600cb478675f3b855a93a422fbfb65ce729a510d687a2f5ceadce3d15e4` | `4e4cb600cb478675f3b855a93a422fbfb65ce729a510d687a2f5ceadce3d15e4` |
| `tests/test_pseudolabel_v3_audit.py` | `None` | `2addf17345f04770787deb656533f64fefe580a6263a2d2aaa8283c359098bf9` |
| `docs/pseudolabel_system_v3.md` | `74e7b31e3fd12d5aac31887373fe98ca6fcc49477c2aacc8948a65d850d9b8ca` | `603c544be73b16fb45a534f7bdb9266045d803217be1d5dc93e4a81d4cb70350` |
