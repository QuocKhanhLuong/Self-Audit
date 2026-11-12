# Q3D expanded pre-review (Claude Code, read-only)

Date: 2026-11-12 · Task `task_631c8563ca7e` · Scope: uncommitted working tree on `main@016706f`
(34 modified files plus untracked `stages.py`, `artifacts.py`, and new Q3D tests).
Sources read: `AGENTS.md`, `ACTIVE_PIPELINE.json`, all of `docs/CANDIDATE_Q3D_SPEC.md` (§1–13 plus the
2026-11-12 amendments), and the diffs or full source of `engine.py`, `stages.py`, `artifacts.py`,
`alignment.py`, `teacher_features.py`, `losses.py`, `export.py`, `config.py`, `train.py`,
`scripts/q3d_campaign.py`, and parts of `scripts/q3d_preflight.py`. I also diffed the arm configs.

The review edited nothing in the repository. It ran no training, no real data, no GPU, no campaign, and
no full smoke. Every check below is CPU or synthetic and counts as **SOFTWARE_ONLY** evidence.
`git diff --stat -- self_audit/candidate_a self_audit/candidate_a3d` is empty, so the frozen baselines are untouched.

## Bounded CPU checks run

| Command | Result |
| --- | --- |
| `pytest tests/test_q3d_artifact_manifests.py` | 6 passed |
| `pytest tests/test_q3d_stage_artifacts.py` | **collection ImportError**: `make_stage_artifact_ref` (and `read_stage_artifact_bundle`, `validate_shared_stage_artifacts`, `validate_stage_artifact_bundle`, `write_stage_artifact_bundle`) are not in `artifacts.py` |
| `pytest test_q3d_engine_phase, _engine_weighting, _kd_weighting, _export, _interface_diagnostics, _config_campaign, test_candidate_q3d_weighting` | 63 passed, **2 failed** (see M5, L1) |
| `pytest tests/test_q3d_engine.py -k v1_joint_resume` | passed. Note: this test calls `_run_joint_stage` directly and bypasses `run_training`, so stage export is never exercised |
| Scratch repro `repro_deps.py` (calls `write_stage_manifest` with the exact ref shapes `stages.py` produces) | `Q export: ValueError T dependency path must be absolute` · `P export: ValueError P stage manifest is missing content hashes: ['cache_sha256']` |

---

## CRITICAL: blocks the T→Q→P→J pipeline through `run_training`

### C1. Q and P stage export passes the wrong dependency-ref shape, so every Q or P export raises
- `self_audit/candidate_q3d/stages.py:729-734` builds `dependencies` from `state["stage_artifacts"][prior]`.
  Those are `_artifact_ref` dicts (`stages.py:336-350`) with the keys `manifest_path`, `manifest_sha256`, `checkpoint_*`, and `identity`.
- `self_audit/candidate_q3d/artifacts.py:121-127` (write) and `:304` (load: `set(ref) == {"path","sha256"}`) require
  the `stage_manifest_file_ref` shape (`artifacts.py:342-346`).
- Repro: any VQ arm (`v1`, `v1u`, `s1`, …) through `run_training`. T exports fine because it has no dependencies. Q completes, its
  completed state is saved (`engine.py:1366-1367` / `1478-1479`), and then `export_stage_artifact(...,"Q")` raises
  `T dependency path must be absolute`. On resume Q is skipped as completed, then P export fails for the same reason
  (and for C2). The output dir is wedged.
- Fix: pass `artifacts.stage_manifest_file_ref(ref["manifest_path"])` for each dependency. Also add a
  `run_training`-level synthetic test that goes through T→Q→P export (the existing engine tests bypass it).
- Classification: probably an integration seam still in progress, but the current code shows real API drift
  between the two new modules, not just a missing symbol.

### C2. The P manifest content-hash key does not match the contract
- `stages.py:799-800` writes `cache_payload_sha256` and `cache_file_sha256`. `artifacts.py:151-155` requires
  `cache_sha256` for P. The import side (`stages.py:560`, `:576`) reads the `cache_payload_*` and `cache_file_*` names.
- Repro: as in C1, every P export raises `P stage manifest is missing content hashes: ['cache_sha256']`.
- Fix: pick one key set and use it on all three sides (writer, `_normalize_content_hashes`, importer). The simplest fix is to add
  `cache_sha256` (canonical payload SHA) to the writer and keep the file SHA as an extra key.

## HIGH

### H1. Two incompatible stage-artifact APIs coexist; the shared-stage test suite cannot import
- `tests/test_q3d_stage_artifacts.py:9-15` expects a *bundle* API: `make_stage_artifact_ref(..., feature_source=,
  feature_stride=, codebook_sha256=)`, `validate_shared_stage_artifacts`, and `read/write_stage_artifact_bundle`.
  None of these exist. `stages.py` and `artifacts.py` implement a *manifest* API built on `source_config` equality.
- The test also names sources `"encoder.x_res_3"`, while config, engine, and `teacher_features` use `"encoder_stride8"`
  (`teacher_features.py:20-28`). That is a naming drift that will reject legitimate sharing once the bundle API lands.
- Classification: transient missing symbol, plus a design decision the root must make. Choose one API and one
  feature-source vocabulary before more callers bind to either.

### H2. The campaign does not wire the shared T/Q/P DAG, so V1, V1-u, and C1 would each train their own teacher, C0, and cache
- `scripts/q3d_campaign.py:41-42,46` imports `load_stage_manifest`, `stage_manifest_file_ref`, and `SHARED_DAG_SCHEMA`, but
  none of them is used anywhere. `_job_command` (`q3d_campaign.py:695-706`) launches `train.py --config …` per
  experiment without `--q3d-stage` or `--q3d-stage-artifact`.
- Spec impact: this violates §9 [AMEND] B4 ("V1, V1-u and C1 use the same teacher, C0 and cache with SHA; J … does not
  retrain T/Q itself"). It confounds the V1-vs-V1-u (physiology) and V1-vs-C1 (VQ) decision rules in §11, because
  each arm would get a different teacher and codebook.
- Classification: likely in-progress. It must be closed before any real campaign launch. A campaign plan test should assert
  that the V1, V1-u, and C1 J jobs reference identical T/Q/P manifest SHAs, and that O1, S1, stride4, and decoder-stride8 share only T.

### H3. Stage completion and artifact export are not atomic, so resume after a crash can wedge permanently
- The stage functions append to `completed_stages` and persist `latest.pt` *before* `export_stage_artifact`
  runs (T: `engine.py:1183-1193`; P: `engine.py:1617-1625`; export at `engine.py:3191-3196`). The stage ref only reaches disk
  on the next save.
- Repro: kill the process after `_run_teacher_stage` returns but before `export_stage_artifact` finishes (or after
  `checkpoint.pt` is written but before `manifest.json`). On resume T is skipped, so it is never exported. Then Q export raises
  `lacks a verified dependency reference`. Retrying hits the same state, and partial artifact files make the immutable writer
  refuse (`stages.py:695-696`).
- This fails closed, which is correct, but nothing recovers it. Fix: on resume, for each completed T/Q/P that lacks a ref, either
  adopt an existing on-disk manifest after full `load_stage_manifest` and teacher/C0 SHA comparison, or export it then.
  Alternatively, export inside the stage before marking it complete.

### H4. C1's declared stages (`T`,`P`,`J`) cannot export its own P
- `config.py:79-80`: C1 has `tokenizer: False` with stages `("T","P","J")`. `stages.py:731` always requires a Q dependency for P,
  and `stages.py:736-744` requires tokenizer state or metadata for P. A standalone C1 run therefore crashes after P.
- `artifacts.py:113-114` explicitly allows P with only a T dependency ("may also depend on Q"), and the intended test
  `test_non_vq_physiology_bundle_can_use_teacher_without_tokenizer` expects that too. The writer contradicts both.
- Spec says C1 should *reuse* V1's P (B4). Either forbid standalone C1 P (require import, with a clear error) or make
  Q and the tokenizer optional for P when the arm has no tokenizer. The current behavior is an accidental crash.

## MEDIUM

### M1. A legitimate resume with the same `--q3d-stage-artifact` arguments is rejected
- `stages.py:478-482` requires `existing_refs == supplied_refs` exactly. Repro: start V1 with
  `--q3d-stage-artifact T=… --q3d-stage-artifact Q=…`. The run trains P itself and exports it, so the refs in state become {T,Q,P}. It is then
  interrupted during J. Re-running the identical command fails with `resumed Q-3D state stage artifact references differ`.
- Fix: require supplied ⊆ existing and exact equality on the intersection.

### M2. J does not cross-check the physiology cache against the live teacher and C0 (non-resume path)
- Import accepts P with T but without Q (`stages.py:472-475`). When Q is absent, the codebook expectation falls back to
  the cache's own value (`stages.py:569`). If a V1 run imports T+P and lets Q train locally (no `--q3d-stage`),
  J consumes a cache whose `codebook_sha256` differs from the live C0 and whose manifest depends on another Q.
  `_validate_resumed_physiology_cache` runs only on resume (`engine.py:3124-3128`). J entry
  (`engine.py:~2286-2345`) checks the tokenizer freeze, but not `cache.codebook_sha256 == tokenizer.codebook_sha256` or
  `cache.teacher_sha256 == _state_sha256(teacher)`.
- Scientific impact is small (w_pat does not depend on C0), but this breaks the B4 identity contract ("J verifies SHA,
  source, cohort"). Fix: for tokenizer arms, importing P requires importing Q, and J entry asserts that the cache's teacher and C0 SHAs match the live modules.

### M3. Synthetic secondary-source arms are dead or misleading code paths
- The engine's synthetic branch maps `stride4*`, `decoder_stride8`, and `bottleneck_stride16` to tiny feature indices
  (`engine.py:~566-570`) and avg-pools for stride16 (`engine.py:_build_models` and `~603`). But
  `teacher_feature_phase` runs first (`engine.py:~557`) and supports only `encoder_stride8` on `TinyEncoder3D`
  (`teacher_features.py:74-76`). So synthetic S1, `v1_stride4`, and `v1_decoder_stride8` raise.
- If that guard were relaxed, synthetic `decoder_stride8` would silently reuse the encoder feature (index 3). Synthetic
  stride16 would also be phase-shifted by the AvgPool(1,2,2) (+4 px in-plane) relative to any declared phase, because the pool is not
  in the phase chain.
- Failing closed is acceptable. Recommendation: remove the unreachable mappings, or include the AvgPool in the
  synthetic phase chain and label those settings `NOT_RUN` in the synthetic campaign.

### M4. The O1 "separate Q/P" requirement is not enforced
- The Q source config (`stages.py:190-218`) excludes `freeze_joint` and `online_joint`, and the P source excludes
  `cache_refresh_joint_epochs` (`stages.py:220-224`). So O1's Q/P `source_config` equals V1's, and O1 can import V1's Q/P.
  Spec §9 B4: "O1, S1, stride 4 and decoder stride 8 use separate Q/P but share T." Scientifically, Q and P are
  identical at stage time. This is spec conformance, not a correctness bug. The root should decide whether to add an
  arm-family tag to the Q/P source or amend the spec.

### M5. The omega scale-invariance test fails by 1 ulp
- `tests/test_q3d_engine_weighting.py:342`: after scaling every U w_pat by 0.1, omega differs
  (`0.18181818181818182` vs `0.1818181818181818`). `w/mean` is not bitwise scale-invariant in float64.
- Spec §7 asks for loss/grad ratios of ≈1.000 rounded to three decimals, and for **V1 vs V1-u equal weights bit for bit**. Bitwise
  invariance under rescaling is stricter than the spec. Fix the test (tolerance on omega, keep bitwise equality for the
  V1/V1-u w_cell maps), or document a canonical-scale rule. No patient-weight cancellation was found (see verified list).

## LOW / informational

- **L1.** `tests/test_q3d_config_campaign.py:271` still expects the secondary settings without `v1_decoder_source`
  (added at `q3d_campaign.py:50`). The test is stale and the new setting is transient.
- **L2.** The Q source config seals config-declared student keys (`stages.py:195-209`), including `epa_support`, so E1
  cannot share V1's Q even though the tokenizer does not depend on EPA. It also seals a declared
  `native_origin_zyx`/`[1,16,16]` instead of the measured `student_grid_phase(student)`. Recommendation: seal the measured
  phase and drop the irrelevant student keys.
- **L3.** P import cache path: validation may use a fallback `cache_state["path"]` (`stages.py:550-557`), but state is
  always rewritten to `p_path.parent/"physiology_cache.json"` (`stages.py:630-631`). With the fallback, resume points at a
  missing file.
- **L4.** `stage_identity` has an unused `runtime` variable (`stages.py:92`). The identity is only dataset, protocol, K, and seed. The cohort is
  bound via `content_hashes.data_identity_sha256`, which is correct.
- **L5.** `scripts/q3d_preflight.py` hard-codes `PINNED_MEDNEXT_SOURCE = /tmp/q3d-mednext-phase-pin`. Elsewhere it
  degrades to `NOT_RUN`. Make it a CLI or config input.
- **L6.** `align_teacher_to_student4` clamps to the *padded* teacher feature extent (`alignment.py:166`). The last student
  cells can interpolate toward teacher centers that lie in the replicate-pad region. This matches the §4 limitation, but the
  docstring's "endpoint replication" wording is slightly inaccurate.

## Verified against spec (no defect found)

- **codeKD / feature-KD reduction (§7 B1 API):** `codekd_loss` and `feature_kd_loss` return per-cell `[B,D,h,w]` maps
  (`losses.py`). `_distillation_or_zero` (`engine.py:1914-1942`) calls `weighted_class_balanced_reduce` per frame
  with confidence-only `w_cell` (`_cell_weights`, `engine.py:1895-1911`, threshold 0.7) and omega outside. The old
  epsilon reducer `_class_balanced_weighted_mean` is deleted. L omega = 1 (`engine.py:1829-1830`). U omega comes from the
  frozen full-U cohort map. A real gate failure raises at J (`engine.py:2314-2318`) and at P import (`stages.py:646-648`).
  I found **no patient-weight cancellation**.
- **MedNeXt source pin and feature:** real phase requires `encoder.source_commit == UPSTREAM_MAIN_SHA`
  (`0b78ed86…`, `teacher_features.py:239-242`; config `UPSTREAM_MEDNEXT_COMMIT` matches), a `MedNeXtL3DAdapter` with
  `_candidate_a_native_v1`, and channels from `adapter.channels`. The main source is the `enc_block_3` output (= `x_res_3`, the
  A3D native sequence at `candidate_a3d/teacher.py:316-324`). When checkpointed, the hook goes on the last child, and a one-fire check and a shape check follow.
  The residual `res_conv` phase is validated against the main path.
- **Phase math (§4):** `grid_phase` implements `o += (d(k−1)/2 − p)·j`, `j *= a`. The student phase is read from live
  modules. Teacher targets are pooled on centered student cells with pad excluded from both numerator and denominator.
- **Synthetic override unreachable for real runs:** Q/P import gates require `runtime.synthetic is True` plus the explicit
  override (`stages.py:389-425`). `run_training` rejects an override without synthetic smoke. `runtime_synthetic` is sealed
  in the stage `source_config`, so synthetic artifacts cannot be imported into real runs.
- **T import:** loads `best_teacher_state` strictly and checks the SHA against the manifest and checkpoint. Q import checks C0 against the metadata, the manifest, and
  the T SHA. P import re-verifies the cache file, canonical SHA, rows, and cohort summary, and recomputes runtime omega for the destination arm.
  Only allow-listed state crosses arms (`stages.py:600-664`).
- **Export cold-load:** reproduces runtime dtype, threads, and determinism in the child process, hashes logits via a uint8 view
  (bf16-safe), and checks that no teacher module is imported. The C1 feature head is rejected.
- **Interface diagnostic (§13 [OPEN]):** logs `vq_gate` and `‖g·W_q(z)‖/‖F4‖`, handles a zero denominator, and applies the
  0.05 claim threshold.

## Suggested routing

1. C1 and C2 (one small patch across `stages.py` and `artifacts.py`) plus a `run_training` T→Q→P export test. Do this first.
2. H1: root decides between the bundle and manifest APIs and the source vocabulary. Then H4 (C1 P policy) and M2 (P requires Q for tokenizer arms; J SHA cross-check).
3. H2: wire the campaign DAG before any real launch. H3: resume adoption or export of completed stages without refs.
4. M1, M3, M5, L1–L6 as cleanup.

Evidence status: all of the above is SOFTWARE_ONLY. P0–P5 acceptance, the network-phase probe on real MedNeXt, GPU, real data, and the
campaign remain NOT_RUN.
