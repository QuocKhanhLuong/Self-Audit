# Claude review — Q3D ResEnc-L config / pretrained provenance (read-only)

Date: 2026-11-12 (snapshot ~16:55 +07)
Worktree: `/Users/alvinluong/orca/workspaces/Self-Audit/q3d-resenc-20261112` (uncommitted working tree on `main@37851c7`)
Scope: `self_audit/candidate_q3d/config.py`, `self_audit/candidate_q3d/pretrained.py`, `configs/candidate_q3d/*.json`, `requirements.txt`, and the pretrained/config tests.
Mode: read-only. I changed no repository files. The only things I wrote were probes in my session scratchpad and this report. `git status --porcelain` was identical before and after the pytest runs.

**Concurrency caveat.** Another worker edited `config.py`, `pretrained.py` and `teacher_resenc.py` between 16:53:33 and 16:53:44, while I was reviewing (it added `class_mapping_scope` and fixed the strides comment). I re-ran every probe below against the files as they stood after those edits. Any finding may already be fixed if edits continued after that.

## Verdict

The core requirements are met:

- Only the nnFoundation ResEnc-L teacher passes as READY.
- The real teacher build pins the SHA, HF revision/URL, license, Table D7 dataset and overlap declaration.
- There is no SCRATCH, TinyEncoder or random fallback on the real path.
- The K8 manifest lock works.
- The stride and feature-stage defaults are correct.
- The ERF-based zero phase offsets are consistent with the final probe.
- Seeded rebuilds are bit-identical.

I found one concrete provenance-correctness bug and three fail-closed gaps in scope. I also found three concrete bugs in files owned by other workers.

## Verified OK (with evidence)

| Requirement | Result | Evidence |
|---|---|---|
| Only `nnfoundation_resenc_l` + READY for real teacher | OK | `pretrained.build_q3d_teacher` raises unless the architecture is `nnfoundation_resenc_l`. `_declared_provenance` requires `provenance_status == READY` and an exact field set. |
| Exact HF pin / SHA / URL / license / Table D7 / overlap note | OK at teacher build | `pretrained.py:32-40,71-106`. The URL must equal `https://huggingface.co/MIC-DKFZ/nnFoundationCNN/resolve/f443c093.../checkpoint_final.pth`. The SHA must equal `ac262d3e...`. The dataset, `overlap_acdc=overlap_mnms=False`, the "not independently verified" note and `CC-BY-SA-4.0` are all pinned. |
| No SCRATCH / random / Tiny fallback | OK | Real path: `load_nnfoundation_encoder` (sha256 + `weights_only=True` + strict 448-key and shape check), then `Q3DResEncTeacher(encoder=...)`. `Q3DResEncTeacher(encoder=None, synthetic=False)` raises. `runtime.allow_unverified_teacher=True` is rejected. The only route into the synthetic teacher is `runtime.synthetic=true`. `train.py` and `engine.train` validate with `real_run=not synthetic`. |
| No `source_repo` / `source_commit` / checkouts in config | OK in scope | These keys are not in `_base_config` or the exact teacher key set (`config.py:666-670`), so a config that carries them is rejected. Stale references remain in other files; see the cross-scope section. |
| Real checkpoint strict load | PASS (re-run) | `build_q3d_teacher(canonical_config('v1', k=8))` against the external `checkpoint_final.pth` gave `loaded_key_count=448`, no missing or unexpected keys, `decoder_loaded=False`, `hf_revision=f443c093...`. |
| DNA 0.4.4 actually installed and enforced | OK | `requirements.txt` pins `dynamic-network-architectures==0.4.4`. `importlib.metadata` reports 0.4.4. `_require_dependency_pins` checks both the declared pins and the installed version. |
| Seeded repeat of new heads | OK | Two real builds with the same seed gave bit-identical `state_dict` (985 entries). The build runs inside `torch.random.fork_rng`, so global RNG is not disturbed. |
| K8 TRAIN `e524c753...` / DEV `ba469...` lock; wrong SHA fails real | OK | `config.py:925-932`. Probe: ACDC/A/K8/seed42 with a wrong TRAIN SHA under `real_run=True` was rejected with "ACDC K=8 TRAIN manifest SHA-256 does not match the locked campaign input". |
| K2/5/8/10 remain valid | OK | `canonical_config('v1', k=…)` was accepted for all four values. |
| S1 = stage 4 / 320, main = stage 3 / 256 | OK | `v1 → encoder_stride8, 256`; `s1 → encoder_stride16, 320`. Stride 16 is rejected for any arm other than `s1`. |
| Default strides `[[1,1,1],[2,2,2],[1,2,2]×4]` | OK | `NNFOUNDATION_DEFAULT_ENCODER_STRIDES`. Cumulative stride is (2,8,8) at stage 3 and (2,16,16) at stage 4, matching the ERF probe's `stride_schedule_zyx` and `cumulative_stride_zyx=[2,8,8]`. The stale "isotropic" comment has since been fixed. |
| Phase offsets vs ERF rule | OK | The final `erf_probe.json` (logical shape `[33,97,97]`, 16 samples, seed_start 42) reports **PASS** with a per-axis absolute mean centroid offset of `[0.572, 1.727, 0.868]`, at most 2.0. The pre-registered rule therefore selects the main-convolution phase, so `phase_offsets = 0` is correct. The superseded `erf_probe_even_unaligned` (2.198 > 2) must not be cited. |
| Config hash covers strides / phase / provenance | OK | `stages.stage_source_config` puts `encoder_strides`, `phase_offsets`, `pretrained` and `dependency_pins` into the T/Q/P `source_config`. `artifacts.write_stage_manifest` hashes the full `source_config` (`source_config_sha256`). |
| Old feature-specific arms removed; `teacher_supervised` rename | OK in configs | `configs/candidate_q3d/` has 13 recipes. `v1_stride4`, `v1_decoder_stride8` and `mednext_supervised` are deleted. `teacher_supervised.json` is new and still untracked. Every JSON round-trips equal to `canonical_config(...)`. |
| Defaults carry actual READY provenance | OK | Every teacher-using recipe ships READY with the real SHA, URL and revision. |
| No personal weight file committed | OK | `git ls-files` has no `.pth`, `.pt` or `checkpoint_final`. The 409 MB checkpoint exists only under `Self-Audit-runs/`. |
| Scoped tests | 53 passed | `tests/test_q3d_pretrained.py`, `test_q3d_config_campaign.py`, `test_q3d_teacher_resenc.py`, `test_q3d_alignment.py`, `test_q3d_preflight.py`. `scripts/check_repo.py` passes (q3d: 13 configs). |

## Findings in scope

### F1 (correctness, provenance): the weight inventory labels all 448 loaded pretrained encoder keys as randomly initialized
`pretrained.py:272-282`:

```python
region_keys = sorted(key for key in teacher.state_dict() if key.startswith("region_decoder."))
...
"reinitialized_keys": sorted(set(boundary_keys + region_keys)),
"region_decoder_random_keys": region_keys,
```

DNA's `UNetDecoder` registers `self.encoder`, so `teacher.state_dict()` also emits `region_decoder.encoder.*`. That is an alias of the pretrained encoder; I confirmed it is the same storage with `data_ptr` equal. Measured on the real checkpoint:

- `reinitialized_keys` has 537 entries, of which 448 are `region_decoder.encoder.*`.
- The truly new decoder and boundary keys number only 89.

The manifest therefore claims that the pretrained encoder was randomly re-initialized, which contradicts `loaded_keys` in the same record. Any audit that reads `reinitialized_keys` or `region_decoder_random_keys` gets the provenance wrong.

Fix: exclude keys whose tensor shares storage with `teacher.encoder`, or drop the `region_decoder.encoder.` prefix. Add a test that `reinitialized_keys ∩ alias(loaded_keys) = ∅` and that the reinitialized count is 89 on the real checkpoint (or the matching synthetic count).

The engine's optimizer groups are unaffected: `_teacher_parameter_groups` dedups by identity. The load path is also fine: `Q3DResEncTeacher.load_state_dict` rejects aliases whose values diverge.

### F2 (fail-closed gap): config-level real validation does not pin the weights SHA
`config.validate_real_teacher_provenance` (`config.py:~833-878`) and the READY branch of `_validate_model` check only that `weights_sha256` is 64-character hex and matches the file. They do not compare it with `NNFOUNDATION_WEIGHTS_SHA256`.

Probe: an arbitrary 1-byte file whose declared SHA is its own SHA, together with the exact URL, was ACCEPTED by `validate_real_teacher_provenance`.

The real run still fails closed later, because `pretrained._declared_provenance` pins `_WEIGHTS_SHA256`. However, `scripts/q3d_campaign.py:~402-410` uses `validate_real_teacher_provenance` to mark plan entries READY. A wrong checkpoint would therefore appear as READY in a PLAN and fail only at T. The pinned URL and a wrong SHA also contradict each other.

Fix: add `digest == NNFOUNDATION_WEIGHTS_SHA256` in both places. Add a test that a wrong SHA is rejected by `validate_config(real_run=True)` before any file I/O.

### F3 (fail-closed gap): locked strides and phase offsets are not locked for real runs
`_validate_encoder_strides` accepts any six positive triples whose first entry is `[1,1,1]`, and `phase_offsets` accepts any finite numbers. Probes:

- `[[1,1,1]]*6` (cumulative stride 1, but still labeled `encoder_stride8`) was accepted.
- The official isotropic schedule was accepted.
- An offset of `[1e6,-1e6,5]` was accepted.

Alignment uses the actual step and offset, so this does not misalign silently. But the spec (§4) says strides and phase are locked before training as part of the Phase-0 decision. A real run can still use an unlocked schedule, and the name `encoder_stride8/16` no longer means an in-plane stride of 8/16.

Fix: under `real_run=True` (or `stage_source_config` for T), require `encoder_strides == NNFOUNDATION_DEFAULT_ENCODER_STRIDES` and `phase_offsets == NNFOUNDATION_DEFAULT_PHASE_OFFSETS`. Ideally also bind the ERF receipt SHA (`erf_probe.json`) into the config or T manifest. Independently, assert that the cumulative in-plane stride is 8 at stage 3 and 16 at stage 4.

### F4 (portability, low): personal absolute weights path is the default and is part of stage identity
`NNFOUNDATION_DEFAULT_WEIGHTS_PATH = "/Users/alvinluong/Self-Audit-runs/..."` is baked into `config.py` and every teacher recipe JSON. `pretrained.weights_path` is inside the T/Q/P `source_config` hash. So the same verified checkpoint at a different path, such as a GPU host, produces a different stage identity, and the canonical recipes fail `real_run` on any other machine.

No weights are committed, so the "no personal weight file" rule holds. This is a reproducibility and sharing hazard, not a leak. Consider keeping the path out of the identity hash (the SHA already identifies the bytes), or resolving it from an env or CLI value.

### F5 (minor)
- `canonical_config(..., source_stride=, feature_source=)` still exposes knobs whose only non-default outcomes are now rejected. This is dead API from the removed stride4/decoder arms.
- `_attach_channel_alias` accepts the synthetic widths `(4,8,12,16,20,20)` on the real path too. It is harmless today because the real path always builds full widths, but it is not strict.
- `_require_dependency_pins` makes synthetic smoke require DNA 0.4.4 installed. That is probably intended; noting it as an assumption.

### Test gaps (scope)
- No test asserts the K8 TRAIN/DEV locked-SHA rejection (`K8_LOCKED_*`). `grep` finds no reference in `tests/`.
- No test covers inventory correctness (F1), config-level wrong-SHA rejection (F2), or real-run stride/phase locking (F3).
- No test checks that different seeds give different head init. Only same-seed identity is tested.
- `test_real_checkpoint_loads_strict_encoder_only_when_phase0_weight_is_present` skips when the external file is absent, so CI without the file never runs the strict-load check. This is acceptable but should be reported as such.

## Cross-scope issues (other workers' files; possibly still in progress)

1. **`self_audit/candidate_q3d/baselines.py:450-463` (concrete bug).** `_validate_teacher_pretraining` still requires `teacher["source_repo"]` and `teacher["source_commit"]` to be non-empty strings. The config schema now forbids those keys, so every real `teacher_supervised` validation without a shared T raises "requires validated source_repo and source_commit". The `:736` comparison uses `source_commit` too.
2. **`scripts/smoke_q3d.py:93-105` (concrete bug).** `_ALL_RECIPES` still lists `v1_stride4` and `v1_decoder_stride8`. `_recipe_config` raises for both ("feature_stride must be 8 or 16" / "feature_source must match the selected encoder stage"), so `smoke_q3d.py --all-arms` crashes. The file was last modified at 16:41, before the config rewrite.
3. **`self_audit/candidate_q3d/stages.py:222,283,329`** still list `source_repo` and `source_commit` in the teacher provenance selectors. This is harmless because `.get` and `_select` skip absent keys, but it is residue against the "no old source_repo/source_commit" rule.
4. **Other residue:** `scripts/q3d_campaign.py:740,749` has MedNeXt baseline wording, and `export.py:168` has a `v1-stride4` alias.
5. **Untracked new files:** `self_audit/candidate_q3d/teacher_resenc.py`, `tests/test_q3d_teacher_resenc.py`, `tests/test_q3d_teacher_schedule.py`, `tests/test_q3d_reduced_campaign.py` and `configs/candidate_q3d/teacher_supervised.json` are untracked. As a result, `tests/test_q3d_repository_clean.py::test_full_pytest_suite_leaves_tracked_working_source_clean` fails: the tracked-only copy hits `ModuleNotFoundError: self_audit.candidate_q3d.teacher_resenc`. The root must `git add` these files when committing. This is not a code defect.

## Assumptions / not verified
- I did not run the full `pytest -q` or `scripts/smoke_q3d.py --out ...`, because other workers were still editing concurrently. Only scoped tests and `check_repo.py` were run.
- The geometry audit's median TRAIN spacing `[1,1,1]` mm, with 28 of 80 entries at z=10 mm, drives the z-stride-2 default. I did not re-derive the planner decision, and it is outside my scope. It is worth a sanity check that manifest spacing for ACDC is not a header artifact.
- ERF phase-offset units are input voxels at the nominal main-convolution phase. I assumed that the teacher's `phase_offsets` use the same convention.
- No patient training, dataset download, checkpoint download or repository edit was performed.
