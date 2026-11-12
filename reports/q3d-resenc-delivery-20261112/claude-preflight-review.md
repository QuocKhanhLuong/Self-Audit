# Claude independent review: Q3D ResEnc Phase-0 preflight

Date: 2026-11-12. Reviewer: Claude (Orca task `task_515da1742418`, dispatch `ctx_c6ba2456750f`).
Mode: READ-ONLY. No repo edits, checkpoint/dataset downloads, patient training or patient-data reads.
Target worktree: `/Users/alvinluong/orca/workspaces/Self-Audit/q3d-resenc-20261112` (base `37851c7`, dirty working tree with concurrent workers).

Scope reviewed: `scripts/q3d_preflight.py` (1177 lines), `tests/test_q3d_preflight.py` (273 lines), and the external
`geometry-audit.json/.md` and `erf_probe.py` (+ `erf_probe.json`, `erf_probe_stage3_padding.*`,
`erf_probe_even_unaligned.*`). I also read `self_audit/candidate_q3d/teacher_resenc.py` (untracked, owned by another
worker) because the preflight depends on it; findings there are listed separately as out of scope.

## What I ran (read-only, temp output only)

| Command | Result |
| --- | --- |
| `python -m pytest -q tests/test_q3d_preflight.py -p no:cacheprovider` | 11 passed |
| `python scripts/q3d_preflight.py --config configs/candidate_q3d/v1.json --run-erf --output <scratchpad>/p0_v1.json` | `NOT_READY`; strict live load PASS; `phase=REVIEW_REQUIRED_RESIDUAL_SKIP_PHASE`; `erf=NOT_RUN ("ResEnc API did not expose this probe")`; datasets NOT_RUN (v1 config has null TRAIN manifest) |
| `_resenc_phase_inventory(build_res_enc_l(config strides).encoder)` | main stage-3 phase step (2,8,8), offset (0,0,0); pure residual-skip phase offset (0.5,3.5,3.5); stage 5 skip offset (0.5,15.5,15.5) |
| Independent reimplementation of nnU-Net `get_pool_and_conv_props` (sensitivity only, not the pinned source) | spacing (1,1,1), shape (9,256,216), min fm 4 → `[(1,1,1),(2,2,2),(1,2,2),(1,2,2),(1,2,2),(1,2,2)]`, pools [1,5,5]; matches the audit. With real ACDC z spacing (10, 1.37–1.56, same): `[(1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2),(1,2,2)]` |
| Statistics over the 16 `erf_probe.json` samples | per-axis mean/sd/SE z 0.572/0.225/0.056, y 1.727/0.856/0.214, x 0.868/0.884/0.221 |

## Verified facts (no issue)

- The live strict loader in the preflight path loads the real checkpoint: SHA `ac262d3e…22bbd` checked before `torch.load(weights_only=True)`. It checks `arch_class_name=ResEncL` and `key_to_stem/key_to_encoder`, compares the exact key set and per-key shapes, uses `strict=True`, and loads no `decoder.*` or head keys. `hf_revision` is attached only when the SHA equals the pin. The receipt `checkpoint_inspection.json` records 448 keys with no missing or unexpected keys. `erf_probe.json` reports the same 448 keys from its own load.
- The repo config `v1.json` declares `encoder_strides=[(1,1,1),(2,2,2),(1,2,2),(1,2,2),(1,2,2),(1,2,2)]`: cumulative (2,32,32), stage 3 (2,8,8). This matches the audit's planner result on median metadata (spacing (1,1,1), shape (9,256,216), min fm 4).
- `_erf_geometry_defaults` gives shape (33,97,97), center (16,48,48), source step (2,8,8) and cell (8,6,6). The odd shape centers the cell exactly, so the even-shape lower-center bias is gone (covered by a test).
- The ERF rule in `erf_probe.py` matches the user's pre-registered rule. It is the per-axis `abs(mean signed centroid offset) ≤ 2`; the maximum single offset is reported, not gated. Inputs are 16 random z-score samples (seeds 42–57) on the pretrained encoder, with the signed mean over all 256 stage-3 channels and absolute-gradient centroids.
- The 32×96×96 receipt is kept and labeled `HISTORICAL_UNALIGNED_DIAGNOSTIC`, excluded from the lock.
- Patient data: the preflight reads only `labeled_frame_keys()` and `load_labeled_frame` (L). A test asserts U is never touched. DEV/TEST manifests are refused by role. The geometry audit read manifest JSON metadata only. `gpu_diagnostics` is `NOT_RUN`; CUDA is absent locally.

## Findings in scope (ranked)

### P1. ERF is not part of preflight readiness, and the in-repo ERF probe never runs
`scripts/q3d_preflight.py` lines 586–610 and 1034–1063. `--run-erf` looks for `erf_diagnostics`, `resenc_erf_diagnostics` or `effective_receptive_field` on `teacher_resenc`, and none of them exist. The live run therefore reports `erf: NOT_RUN`. `build_report` never puts ERF status into `readiness_reasons`. So `SOFTWARE_PREFLIGHT_COMPLETE` can be reached with no ERF at all, even though the spec (§12 P0 and §4) requires "ERF ≥16 mẫu với luật 2 voxel". The only ERF evidence is the external `erf_probe.py`, and the preflight neither ingests nor verifies its receipt.
Fix: either ingest the external receipt, checking SHA, sample count ≥16, shape, cell, checkpoint SHA, stride schedule and status, or implement the probe in the repo. Then gate readiness on `erf.status ∈ {PASS (main phase), MEASURED_OFFSET_LOCKED}` and treat absent or NOT_RUN as not ready.

### P1. Phase readiness is gated on the theoretical skip lattice, not on the ERF rule
Lines 486–495 and 1047–1052. The status `PASS_LOCKED_OFFSET` vs `REVIEW_REQUIRED_RESIDUAL_SKIP_PHASE` comes from `max_abs_residual_skip_offset ≤ 2`. For any ResEnc with stride-2 in-plane at stages 1–3, the pure-AvgPool path offset is 0.5+1+2 = 3.5 voxels (live result: (0.5,3.5,3.5)). This check can therefore never pass, so preflight stays `NOT_READY` for good, whatever the ERF result. The spec asks only that the skip-lattice maximum deviation be *recorded*. The 2-voxel rule applies to the measured ERF centroid, and the erf_probe interpretation explicitly says the measured offset "is not replaced by a theoretical skip-path offset". As written, the gate also mislabels a skip value ≤2 as `PASS_LOCKED_OFFSET` without any ERF.
Fix: report the skip lattice as `RECORDED` (informational). Derive the phase decision only from the ERF receipt: main phase if `abs(mean) ≤ 2` on every axis, otherwise the measured signed offset written into `phase_offsets` and the manifest. Also compare the skip offset *relative to* the main offset, not as an absolute value (equal here because main=0, but not general).

### P2. Fail-open fallback reports a declared phase as PASS
Lines 563–572. If `_resenc_phase_inventory` returns `None` (topology not recognized, or any AttributeError/IndexError/TypeError/ValueError swallowed at 482–483), the code falls back to `teacher.feature_phases`. Those are the *configured* `phase_offsets` (default 0), and the code labels them `status: "PASS_LOCKED_OFFSET"`. That passes the readiness phase check without measuring anything. A DNA attribute rename would silently produce PASS.
Fix: use `status: "DECLARED_NOT_MEASURED"` (or NOT_RUN) and treat it as not ready.

### P2. Stride derivation is not checked by the preflight; median shape is not reported
`_manifest_diagnostics` reports spacing quantiles and depth D only. It does not compute or report median H/W, does not run the planner rule, and does not compare `model.teacher.encoder_strides` with the planner result for the configured TRAIN manifest. Spec P0 lists "median TRAIN spacing/strides" and the planner's min-feature-map/shape conditions. Today the (2,32,32) lock exists only in the external audit.
Fix: compute the median shape (Z,Y,X) and the planner strides from the manifest metadata (pinned rule, min fm 4). Report them and fail readiness when they differ from config. Also record the exact planner inputs.

### P2. The stride lock depends on 52/80 unverified unit-spacing entries
External `geometry-audit.json`. The TRAIN median spacing of (1,1,1) comes entirely from 52/80 entries declaring exactly [1,1,1] mm. The other 28 entries are z = 10 mm with 1.37–1.88 mm in-plane. Unit spacing with 6–18 slices is not physically plausible for ACDC cine (9 mm of heart coverage), so these are probably placeholder headers. Under real ACDC spacing the planner (my reimplementation; not a pinned-source run) moves the single z pool from stage 1 to stage 3: `[(1,1,1),(1,2,2),(1,2,2),(2,2,2),(1,2,2),(1,2,2)]`. The stage-3 lattice step stays (2,8,8) and the total stays (2,32,32), so the phase arithmetic does not change. The z receptive field and the ERF probe operator do change, and so would the ERF z-centroid. The audit labels this `CONDITIONAL` and the coordinator brief notes it is not independently verified. I agree that the lock should be stated as conditional on manifest spacing provenance. This also affects anything in mm (HD95, ASSD, volumes in mL), though not the per-patient ratios r_SV/r_M, which are scale-invariant.
Recommendation: before locking, either verify the 52 headers against the source NIfTI (read-only metadata) or pre-register the choice. Do not pick the schedule after seeing ERF/DEV outcomes.

### P2. The ERF receipt and script are out of sync, and the receipt does not record its script hash
`erf_probe.json` (16:52:34) is byte-identical to `erf_probe_stage3_padding.json` (16:54:46). Current `erf_probe.py` (16:55:26, SHA differs from the `ca5eafaa…` recorded in `geometry-audit.json`) now pads to the full six-stage multiple (2,32,32), giving model input (34,128,128) and pad (1,31,31). It derives the cell from the logical stage-3 shape. The PASS currently cited in `geometry-audit.md` (pad [34,104,104]) is therefore from the stage-3-padding variant. The coordinator says a rerun is in progress (concurrent work, not a defect by itself). The receipt JSON stores no script SHA, DNA version, torch version or command line. The geometry audit holds the only script hash, and it is now stale.
Fix: embed `sha256(__file__)`, `dynamic_network_architectures.__version__` (spec pin 0.4.4; locally 0.4.4), the torch version and argv in the receipt. Regenerate `geometry-audit.json` after the rerun.

### P2. The ERF PASS margin on Y is small, and the canonical run must be fixed before results are seen
The current receipt has y = 1.727 (SE 0.214), so the 95% upper bound is about 2.15 and above the threshold. x = 0.868 (SE 0.221). Under the pre-registered rule the result is PASS and I would not change the rule. However, three variants now exist: even-unaligned (REVIEW 2.198), stage-3 padding (PASS) and full-stride padding (pending). Choosing between them after seeing outcomes would be a forking-paths risk. The full-stride variant matches the teacher's `_pad_high` multiple, so it is defensible on geometric grounds alone.
Recommendation: record in the report *why* the canonical variant is canonical, independently of its result, and apply the rule to that run whatever its outcome. If it fails, use the measured signed offset (spec §4).

### P3. ERF boundary treatment differs from the teacher's
`erf_probe.py` pads with zeros (`mode="constant"`); `Q3DResEncTeacher._pad_high` uses `replicate`. On the high side the network sees features of a zero image, which are non-zero after bias, InstanceNorm and LeakyReLU. On the low side it sees literal zero conv padding. With pad (1,31,31) roughly 43% of the model input is padding, and that also shifts the InstanceNorm statistics. This asymmetry can bias the centroid toward or away from the high side. It is a limitation, not a bug, because a random-noise probe is architecture-only. Cheap control: rerun with the logical shape enlarged so the RF (r95 ≈ 51 voxels) clears the boundary, or mirror the probe (flip input axes and check that the signed offset keeps its architectural sign).

### P3. Smaller fail-closed and serialization gaps
- `--erf-samples` accepts ≥1 in both the preflight and `erf_probe.py`. The spec minimum is 16, so fail or label `BELOW_SPEC` when samples < 16.
- `_erf_geometry_defaults` silently replaces an invalid or even configured `erf_input_shape` with (33,97,97), and invalid strides with `source_cell=None`. It should fail closed or report the override.
- `_teacher_diagnostics` falls back to `RESENC_CHECKPOINT` and `RESENC_CHECKPOINT_SHA256` when the config declares no weights (for example `provenance_status=NOT_RUN`). The teacher is then reported `ACTUAL_RESENC_MODULES_INSPECTED`, which does not reflect the config. Require `provenance_status=="READY"` and a config SHA equal to the pin before loading.
- `_resenc_checkpoint_inventory` compares the receipt with the *config* SHA, not directly with `RESENC_CHECKPOINT_SHA256`. Today the receipt's own pinned values make this fail closed indirectly; a direct equality check would be clearer.
- The receipt path is hardcoded via an env default and the receipt itself is not hashed. The live loader also returns no key count, so the 448 in the receipt is never cross-checked against the live load. Add `encoder_key_count=len(state)` to the loader provenance and compare it.
- Invalid-config branch (line 954): `weights.pretraining_dataset` defaults to `"OpenMind/nnFoundationCNN"`, which invents a value that differs from the spec's Table D7 string. Report `null` instead.
- `_resenc_network_inventory` lets an API `phase_probe`/`geometry_diagnostics` *overwrite* the live-measured inventory (lines 586–601). Live measurement should take precedence, or both should be kept.
- `erf_probe.py`: the first `status` (max-offset rule) is dead code overwritten by the mean rule. That is harmless, but `REVIEW_OFFSET_OVER_THRESHOLD` in the dict literal is misleading on read. The probe also computes the checkpoint SHA without comparing it to the pin, and does not check `arch_class_name`.
- `gpu_diagnostics: {"status": "NOT_RUN"}` has no reason; add "local CUDA unavailable".

### Test gaps (`tests/test_q3d_preflight.py`)
- There is no test that readiness requires ERF ≥16 or treats ERF NOT_RUN as not ready. The existing inventory test only mocks `erf_diagnostics` on a fake API.
- No test covers the fail-open fallback (topology unrecognized → declared phases → PASS).
- No test checks that the skip-lattice value is informational and that the phase decision follows the ERF rule (both ≤2 → main phase, and >2 → measured offset passed into `phase_offsets`).
- No test compares the planner strides with config strides, or covers median shape reporting.
- The phase fixture uses isotropic (2,2,2) strides. Add the anisotropic config schedule (expected skip offset (0.5,3.5,3.5)), as the spec's "anisotropic stride" test item requires.
- No `_ground_truth_physiology` test (ED/ES pick, invalid-patient counts, L-only).
- No `build_report` test with a valid manifest fixture, covering the end-to-end readiness reasons and `SOFTWARE_PREFLIGHT_COMPLETE` reachability.

## Out of scope: concurrent work by other workers (observations only)

- `teacher_resenc.py::_pad_high` uses `target = max(2*multiple, ceil(len/multiple)*multiple)`. Spec §2 [AMEND] says "không đặt kích thước tối thiểu gấp đôi". For D=9 with z multiple 2 this pads to 10, but for a volume with D ≤ 3 or in-plane ≤ 63 the doubling applies. The InstanceNorm concern needs only more than one spatial element in total, not 2 per axis. Flag this to the teacher owner.
- `teacher_resenc.RESENC_L_DEFAULT_STRIDES` is isotropic (32,32,32). It is unused when config strides are present, but a config-less real build would quietly get a schedule that differs from Phase 0.
- `teacher_resenc` exposes no ERF or phase probe API, which is the root of P1. Whoever owns ERF should decide whether it lives in the repo or stays an external receipt.

## Assumptions and limits of this review

- I did not run the planner from the pinned nnU-Net source (nnunetv2 is not installed). The sensitivity row uses my reimplementation. It reproduces the audit's (1,1,1) result exactly, but the z=10 result is indicative, not authoritative.
- I did not open any patient image or mask. L-GT coverage, r_SV/r_M and the MYO < 2% check stay unverified by me (NOT_RUN in the v1 config because its TRAIN manifest is null).
- I did not rerun `erf_probe.py`; the aligned rerun is in progress elsewhere.
- None of the above is patient, GPU or efficacy evidence. All verification here is SOFTWARE_ONLY.
