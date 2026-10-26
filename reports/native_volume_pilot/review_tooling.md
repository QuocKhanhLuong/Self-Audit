# Independent Astra tooling review

Status: **ACCEPTED — tooling/CLI software scope, CPU manufactured evidence only**. Finalized 2026-10-26T06:38:12.739751+00:00 on branch `feat/native-volume-ema-pilot`, HEAD `486fae40c490e382d58896442215476d7d32095e`. This reviewer changed only this report; fixes were routed through Orca to their original owners. No open owned blocker remains. Historical OPEN/failed/raced entries below are retained audit history and are superseded by the final dispositions here.

The coordinator-executed [post-CLI gate](POST_CLI_GATE.json) passes `check_repo` and **422 tests with 1 existing TorchScript warning (29.27 s)**. I inspected its receipt and verified every recorded source/config/test hash against current files. The inspected [stable core gate](verified/FINAL_VALIDATION.json) records 59 successful CLI commands, all 23 registry arms (V00 reference-only; 22 executable), six CPU profiles, and [exact resume/export checks](verified/ROUNDTRIP.json). Independent reviewer execution includes the 32-test CLI/registry replay, 25-test evaluation/geometry replay, actual bounded CLI smokes, negative dispatch/authorization/protocol/deadline/frame probes, and all six profile-rate recomputations. Per coordinator instruction I did not redundantly rerun the final full suite; my earlier 258-test run is explicitly intermediate.

Core source SHA-256: `3668c8cdd9523d2440a7e93ab2ed0564ba08ddd6bf8998ef88210ff6c96632ac`. Final 73-file source/config/test/doc mapping SHA-256: `f8e2b5cc1e5a7cde08c02f588b21b98c95dc4172e938de22d128771d51f4ae04`. Full paths and hashes appear at the end.

| Scope | Final disposition and evidence |
|---|---|
| Historical A0/V00 and original reports/figures | Preserved; exact A0 hash, V00 reference-only/0 steps; only modified pre-existing file is train.py. No parity/efficacy claim. |
| Registry factors and parents | CLOSED: V01 historical confounds, O02 optimizer/LR confounds, O03 matched base weight, T00 spatial-only, V05 matched skip policy, distinct CNN plans and correction budgets. |
| Strict opt-in dispatch and bounds | CLOSED: schema flags, unknown fields, oversized/nonfinite overrides, missing assets, correction protocol mismatch and post-transaction overruns fail closed. Deadlines remain cooperative. |
| Profiling/denominators | CLOSED: actual microbatch/accumulation, excluded warmup, measured L/U and class/support denominators; six stored rates independently recomputed. CUDA/VRAM NOT RUN. |
| Preparation/evaluation/export/resume | CLOSED in software scope: physical metadata input, label-read gates, native inversion/support, pinned frozen model, teacher-free export and exact CPU resumed state. |
| Correction CLI | CLOSED: role gates before forbidden mask access, protocol/budget checks, final authorization regression coverage, and actual invalid/valid time-index subprocesses. Physical/authorization detail is also covered by the separate independent geometry review. |

**Limits:** no server was rented; no target PRO4000/CUDA/AMP/VRAM validation, actual pinned DINO weight load, real-patient TRAIN smoke, patient accuracy, human annotation timing, long training or scientific efficacy/novelty result was produced. The configured DINO commit `6876159a11b4df116f30f667f8c9888617df0751` and weight SHA-256 `21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54` are declared config provenance, not locally loaded/verified weight evidence. Native support and absent classes are explicit; physical HD95 is not computed. This acceptance establishes no broader scientific, clinical or GPU conclusion.

## Review history and retained intermediate evidence

Read the entire `ASTRA_IMPLEMENTATION_PROMPT.md`, `IMPLEMENTATION_CONTRACT.md`, `ACTIVE_PIPELINE.json`, `docs/METHOD.md`, all 23 registry row configs, registry, runbook, all seven `scripts/volume_*.py` commands, CLI/registry tests, and relevant integration/model source. No CodeGraph index exists; Semble was used for code discovery. No server, patient training, external asset, license, or download was used. No non-Orca subagents were used.

## Independently executed software checks

All shell commands were prefixed `rtk proxy`. Python was the local intended `python` environment. Outputs are manufactured CPU software evidence only.

| Command / check | Observed result at intermediate snapshot |
|---|---|
| `python scripts/check_repo.py` | PASS |
| `python -m pytest -q` | 258 passed, 1 TorchScript deprecation FutureWarning, 25.53 s |
| `python -m pytest -q tests/test_volume_registry.py tests/test_volume_cli.py` | 9 passed, 2.36 s |
| `python scripts/volume_preflight.py --config configs/native_volume/V04.json` | PASS, original registry hash `53c18cfb93e0f0a0dfcd0a6a77fa7cc02c1ec66fffb930568621875940f0be5d` |
| `python scripts/volume_smoke.py --config configs/native_volume/V04.json --steps 1 --device cpu --max-seconds 60` | PASS; 1 supervised, 1 pseudo, 1 EMA; output 1x4x3x64x64; ~1.00 s |
| Same bounded command with V06 | PASS; output 1x4x5x64x64; ~1.38 s |
| Same bounded command with R01 | PASS; region loss 0.001059254; nonempty cache hash |
| Same bounded command with T01 | PASS; temporal loss 0.002927852, support 448 voxels |
| Same bounded command with I01 | PASS; 1 fixture click, 1 hard-manual voxel, 1181 propagated; NIfTI/JSON outputs created in temporary fixture |
| Same bounded command with C01 | PASS after owner update; 1 supervised plus 1 extra supervised update, 1 EMA |
| `volume_prepare.py --manifest <manufactured fixture> --manifest-sha256 <hash> --out <temporary path> --synthetic` | PASS; receipt mask_paths_read=0 |
| Compare all tracked canonical self_audit/config/history/report/figure bytes against HEAD | 223 files checked, no differences |

These are representative arms, not an efficacy matrix. Temporary fixture outputs were deleted by their temporary-directory contexts. No trained pretrained-weight hash exists for this review. CLI fake-model tests are distinguished from the actual unmocked model smokes above.

## Actionable findings sent to coordinator

Line references in this section refer to the observed source before concurrent owner fixes. Final dispositions and final hashes will be recorded below.

1. **P1 — Full-run guard bypass and ignored model override.** `train.py:33-42` validated configured steps (8) before forwarding unrestricted `--steps`; `engine.py:527` inferred full-run permission from a larger steps override. `--model-config` was parsed but ignored in the volume branch. Safe exact reproduction (no training): import `self_audit.volume`; replace `train_volume` with a lambda returning its arguments; invoke `runpy.run_path("train.py", run_name="__main__")` with `--config configs/native_volume/V04.json --synthetic --device cpu --out /tmp/not_written --steps 10000 --model-config /missing/model.json`. Observed forwarding of 10000 and unchanged tiny_test model with exit 0. Require explicit opt-in after all overrides and reject unsupported/irrelevant flags before side effects.

2. **P1 — Direct preflight silently skips the strict validator.** `scripts/volume_preflight.py:114-119` swallowed ImportError, while direct execution lacked a repository-root bootstrap. Reproduction: copy V04 JSON to a temporary directory under the same filename, add `training.unknown_flag=true`, then run the script by absolute path with cwd `/tmp`. Actual CLI returned PASS/exit 0; `validate_volume_config` on the same JSON raised `unknown training fields`. Strict validation must be mandatory and tested in a clean subprocess outside the repo.

3. **P1 — Profiling did not execute requested batching/accumulation.** `volume_profile.py:86,110,113,180` and `volume_smoke.py:154,186,189` hardcoded B=1; the profile echoed configured accumulation without applying it. An actual CPU profile of temporary V04 with microbatch=3, accumulation=2, `--iterations 1 --warmup 0` failed `batch B must equal microbatch*accumulation for gradient accumulation`. Require actual logical batches, exact exposure accounting, and CLI tests with unequal microbatch/accumulation. The intermediate cap `microbatch<=8` also accidentally rejected immutable V00 batch24 globally; coordinator notified immediately.

4. **P1 — Registry comparisons were not the declared factors.** T00 enabled identity temporal loss (weight .1), although requested T00 is spatial-only. O03 changed final pseudo weight from 1 to .5 as well as ramp duration. V01 changed backbone, fixed window, crop, batch and prediction model relative to V00 while listing only volume batching/legacy reduction. V05 changed both depth control and skip fusion. O02 must explicitly declare optimizer plus LR changes (and choose an AdamW head rate independently from SGD). I00 and I01 JSON configs were identical; only a filename special case in smoke invoked correction. Require honest changed-path declarations, actual executable correction config, and preserved fixed-window/pinned DINO pilot overrides. V00 itself must remain a historical reference descriptor, not a reinterpreted volume-trainer run.

5. **P1 — Forbidden click-mask path touched before role rejection.** `volume_correct.py:149-162` loaded `--simulate-from-gt` before checking `--dataset-role`. Reproduction using manufactured image and a nonexistent mask path with `--dataset-role U` produced FileNotFoundError in `_load_volume` at line150 rather than rejecting U first. Default role TRAIN is not proof that a patient belongs to TRAIN-L. Validate authorized role/budget before any mask-path access, with read-trap regression tests.

6. **P1 — Real preparation CLI had no physical metadata input for U.** `volume_prepare.py:39` requires U records contain only id/image/image_sha256; NPY images carry no affine; `build_sidecar` has an `affine_overrides` API but CLI lines71-80 exposed no equivalent. Therefore a valid real U record could not receive physical geometry through the CLI. Add explicit hash-pinned external metadata input without adding masks to U records; reject missing geometry.

7. **P1 — Frozen real evaluation/deployment incomplete.** Initial `volume_evaluate.py:40-41` always rejected non-synthetic evaluation with no loader option. `evaluation.py:61-87` pooled dataset confusion on the current grid rather than native inversion then frame/patient means. `load_exported_student` rebuilt through the pretrained constructor before loading exported state, requiring original DINO source/weights. Require an executable frozen held-out image/reference manifest path, pinned prediction-model choice, native probabilities/support inversion before argmax, declared patient/frame reduction, and exported student reload without external teacher/cache/network or original weight files. Missing real assets can prevent measurement but cannot substitute for this executable software.

8. **P2 — Receipt denominators/class counts incorrect.** Actual V04 smoke reported present_foreground=23 and absent_foreground=-20, despite 3 foreground classes; pseudo accepted=969 over valid=969 but coverage=.078857421875 (969/12288 padded grid). Teacher-student disagreement initially unavailable. Require class-count bounds, valid-support denominators, separate measured label exposure, and absent/empty support tests. Reported manufactured Dice is not patient accuracy.

9. **P2 — GPU bound/headroom accounting incomplete.** Profile timer initially began after model allocation, warmup had no deadline check, and headroom used total VRAM minus this process reserved memory rather than current free memory. Smoke also did not check elapsed cap after its last transaction. Require checks before/after setup/warmup/timed transactions, synchronized timing, true free-memory observation and conservative headroom gate; disclose that a single noninterruptible operation may overrun a cooperative deadline. No GPU runtime validation was performed.

10. **P2 — Runbook omitted working resume/frozen-eval/deployment paths.** Initial real V04 example selected tiny_test/window-none config and had no explicit pinned DINO override; document exact executable override/schema and resume, student/EMA evaluation, export/reload commands. Do not claim the ready command will run a real primary arm when it necessarily rejects synthetic initialization. The runbook must distinguish source-only checks from artifact-producing training/preparation and must not suggest an unbounded run.

11. **Accepted/ignored architecture fields require final model-owner review.** Prior static source consumed only widths[0] and ignored CNN skip settings/hybrid kernel plans while echoing provenance. The attempted direct witness raced an owner rewrite and failed at the initial V06 build with `cnn3d architecture.widths must match z_kernels length`; no claimed bit-identical witness is valid. This correction was sent immediately to coordinator. Recheck final config-builder compatibility and reject non-default unsupported fields.

## Historical invariants verified

- A0 JSON SHA-256: `efa1817634320e0aab29d25ccbe9dfec4633a09f9db54b9c735865fbf9938bef`.
- Current-architecture figure SHA-256: `011f09b4a4ea7861eae00e1de01a8b308c09a9fda46c41de185409845272d822`.
- EMA-training figure SHA-256: `5dfab56cac1064633ccec8a46c423050311b145b2946cc6065da099f855788c7`.
- Existing historical reports/configs and canonical modules inspected were byte-identical to HEAD. The new train.py router necessarily changes aggregate source identity; compatibility must remain explicit.

## Evidence boundary

VERIFIED: the explicitly listed local CPU commands, manufactured fixtures, immutable-file checks, and negative reproductions at their observed snapshots. REPORTED: owner implementation reports and coordinator progress messages, not promoted to independent verification. NOT RUN: target PRO4000, CUDA/AMP behavior, real DINO weights, real TRAIN smoke, patient Dice, human annotation time, long training, server costs/throughput. The user has not rented the server. No scientific, patient, GPU, clinical, novelty or production success follows from the software results.

## Historical interim rechecks

Early test totals below were not final acceptance; the final disposition is at the top and the final hash inventory is at the end.

Additional bounded roundtrip attempt: a temporary V04 config with steps=1 and pseudo_delay=0 was passed to actual `train.py --synthetic --device cpu --out <temporary checkpoint>`. It failed before training with `hybrid architecture.widths must contain four native stages` at the then-current models.py:635. Source digest before and after was exactly `98b4291f3bfbf377d16b9143c4802008c8f83a2ae4a8db413ac545261e3b817f`; no evaluation/export was attempted after this failure. The tightened model builder and unchanged registry widths were incompatible at this snapshot. A separate initial V06 builder probe failed `cnn3d architecture.widths must match z_kernels length`. These regressions were routed to the coordinator without editing configs.


Local environment: macOS 26.2 arm64; Python 3.11.16, PyTorch 2.14.0, NumPy 2.4.6; `torch.cuda.is_available()` is false. A second actual one-step train attempt with `default_volume_config()` failed before training with `volume25d does not support z_strides; use neutral (1,1,1) values`; the source digest was unchanged across the attempt at `0c8372e5bfe3db0fa5cc5d9d7fe5fc773fc626430a569630c6a224d2ed472c11`. Neither failed attempt produced a checkpoint or any inference score.

Coordinator explicitly requested this dispatch remain active pending an owner-ready signal; no approval or settlement has been sent. Further runtime checks are paused to avoid testing moving source.

### Initial source snapshot (not acceptance)

Captured 2026-10-26T01:29:29.460605+00:00; source/config files can change after this read. These hashes describe inspection provenance, not the exact earlier 258-test snapshot.

```text
7bd854bb873a2f679d0388780e32c9c446d6406a1133c24491dfd961ab2af155  train.py
5a45657fbbd5b90270dc897dd1f6dc8f7812a0fd8d091540dec166cd2623bc99  docs/native_3d/RUNBOOK.md
b44fe154fdc91bd3b0398cee110f0f9a918b32b6d8252b55c66043a2adb93c76  configs/native_volume/C00.json
c7512aff207d9821d479ab4c8d01c755fea6de11d2a9e340fba5a31ee0c49cf0  configs/native_volume/C01.json
1c87fc797030b79b1f28bf856dd72079d470e5666a667f154eaa8734b470561a  configs/native_volume/I00.json
26ab5abc02b1fc7877655b0fdee806bfb76d98f4f5f7d516fed44ff2652894fb  configs/native_volume/I00_correction.json
10a35cc01f870bde952a5da9e77dff0d1878f5bb54043c502632b4bfbcde0f44  configs/native_volume/I01.json
c0f0162fcfc1535261e6991ee02cf10ae057da6a9ca36f724504c258b1d56a0f  configs/native_volume/I01_correction.json
7cbe153ded11f648434a73649d1fe1963411468783a45818fccc4ad5fe79f3f1  configs/native_volume/O01.json
ae725747e3fa562c20fb31ce5128b896c3a223bc54c85dfa6276908453d7ea43  configs/native_volume/O02.json
85a08dcb8bf14dd3b64a8a561e093340e6e83a6511937f0bc1707af4e8192ced  configs/native_volume/O03.json
dd45b74f7818d9d928299c64171b7e6ac06abe3af32a5894a0b036488e45993a  configs/native_volume/R00.json
2d7ad26779eca62596d700b34ba5e9f7846e016f83299048c2e3c00b330b6df6  configs/native_volume/R01.json
60450753ea67dc6cb8d4d2a8e7bb7c9692aa4c5861e8781fa3da262bf38725f5  configs/native_volume/R02.json
009caa3bdc7dc5f48adffbb6c1041012bf756aa1194ecf2393814aac3704d471  configs/native_volume/REGISTRY.json
e2a0f4e5e50d6ebceaf0356c073eae6e8e4fb3096a312a9eba1d3428a47198ca  configs/native_volume/S01.json
5a221f8877542b115ffa6d1c1ca832fe7b340ffd517526475a8372c75b22301e  configs/native_volume/S02.json
dd45b74f7818d9d928299c64171b7e6ac06abe3af32a5894a0b036488e45993a  configs/native_volume/T00.json
cbc1e74863f9cc781d725d79fecbe4a40d3364b34f1f8b73282872135c03d99b  configs/native_volume/T01.json
add11a23c650f72cb1f64cccd0d783b730cbf580b9a046988829e9947f3fb649  configs/native_volume/V00.json
9c7dd338085d9f03c6005ac51434210b5d02debdbb06ff2eb8c39973d1a751d8  configs/native_volume/V01.json
ca17eb3c2e0d18b2ce542080dd90f5346547510d89f50e2138c3e8bf034b59a6  configs/native_volume/V02.json
b82408e1f97a77d04df733fafba6c136441b0fc74e7adff0d931eb6901eb71e0  configs/native_volume/V03.json
dd45b74f7818d9d928299c64171b7e6ac06abe3af32a5894a0b036488e45993a  configs/native_volume/V04.json
27c4af754d734a2bca99911bd8e47ded01f8c068392cb880564d6956ad663015  configs/native_volume/V05.json
294ae8f026b17a55c4e07298538ad569fe4630c347450b9bf24122a9ba159433  configs/native_volume/V06.json
25088b5f6a984cb54a728c487203f3afd08ffe8985e5b13572f586c7d7733cf3  configs/native_volume/V07.json
3c7ae8f4ae3a3fbb2302a71bb17a878627b3e129dc601574b4164950864728f5  configs/native_volume/V08.json
9b4f467fc94aae7456bbf390927a557ee96ff98cf61443feecc0c47cdc45687b  scripts/volume_correct.py
c29b3f54f2614c936c7d91288980ffde718bd06c43ddbcd54d2d8344104ec645  scripts/volume_evaluate.py
e7c2ee25f8427af9009cfc5f0721221e1389cf59445b05694798332b05d72497  scripts/volume_export.py
0f79816ea59ad4200237f553dec904bfc9593589e9455675974af91bd065697b  scripts/volume_preflight.py
085ca732de291201ee85c152088ddfee689824384f656810c7c5d2423a026114  scripts/volume_prepare.py
636017479d8f36e31532e17b1a59cc8c38197da8d10372297aae077bbbcc50ca  scripts/volume_profile.py
71a40cd5e0e3d5594c5be24ad37e8c9d7e30e069f5ca64e26ea5ac242d94a31e  scripts/volume_smoke.py
6532199f2b4eadc0b7b05cc1d7dee85fdc1c6155a5d9de1378acde375fdfd0fe  self_audit/volume/__init__.py
41376b37ef8d4c12138cf29f620e72fa1f0c1cffa9368ac650395a8bfe5d5504  self_audit/volume/checkpoint.py
4e27b5972f6b2ea0aa7617c4042013527b67e9fd2ec35db8aaf2b8991cc9a127  self_audit/volume/config.py
f8b1397e9e3816c5b5df0c722f2e99f3548ac2ef5b1b9cf0d8cc3c97e5515cc3  self_audit/volume/correction.py
cb9af492ce996d1bbd2b7011e7c6fa6ebb7c9965810c422533379a5b0b5dafa5  self_audit/volume/data.py
ee8cb0583fd8961c93e3798d7cbaace7fafd8976542b3571ad192b0424131c95  self_audit/volume/engine.py
f1b88049b6748f6ce50247a575500abad885b52a072ae2f759a9b0ad88810a90  self_audit/volume/fixtures.py
7ef5b021f147d0c540cdb4db07366fd4cb6a0cd464a1412b7de1ae75ebf50113  self_audit/volume/fusion.py
069353b75a994c1fa1f7f232462b5fba34b62871568db31a67a3ccf8c17c2da4  self_audit/volume/geometry.py
1cbe6f9f472bc4500be7566d6b332f4df54f7a9c6bf6718095d59b3c5ecb9877  self_audit/volume/legacy_compat.py
489552f683a35fa9665dd3aaeee486f47a94c5a7ffdf5802e8ba26073883270d  self_audit/volume/losses.py
b0bad6e3e5fdca8c953ae22f4d6fe442b88b6488e6f22e1ab947204459808e37  self_audit/volume/models.py
502b3ef1d87d7b9add16790a6d7ff93b24642309fc063148672e4e45a8c63fdb  self_audit/volume/optim.py
d88f8b5a049cd9047ab1aa01e0e5d1862248fee14394688c5a2145396bd13b38  self_audit/volume/regions.py
6f0dd2dc6fd47079e90594a1ab7f08b3071ecf66ec89b88a1a23ef5b8dee38cc  self_audit/volume/temporal.py
52acb41e470aa20627df19dd9fc130b169df7a4b14978d939e227da5c51dc071  tests/test_volume_cli.py
f19d2f80aea8d7753e77b45aac36d4352d2717b6edd19bf2f2845deed82cde44  tests/test_volume_registry.py
```

## Owner-ready evaluation/export recheck

Coordinator released evaluation/export for independent verification while tooling/config work remains active. `rtk proxy python -m pytest -q tests/test_volume_evaluation.py`: **6 passed in 1.37 s**.

Actual subprocess CLIs invoked from `/tmp` using a manufactured scratch CNN3D checkpoint, 2 patients, 2 frames each, shape `(2,3,9,11)`, unequal physical fixture geometry, XY tile7/overlap2 and slab2/overlap1. `scripts/volume_evaluate.py` image-only mode returned labels_read=0; `--score` returned labels_read=4 with frame/patient reduction. Both returned native probabilities `[2,4,3,9,11]` and support594 per patient. `scripts/volume_export.py --synthetic` and `load_exported_student` succeeded; original/exported logits were bitwise equal on a CPU fixture and metadata marked teacher/pretrained weights not required. This closes the prior missing executable evaluation path for this manufactured CNN test. No DINO pretrained assets, GPU or real patient was used.

Source digest before and after all subprocesses: `e38b58adf5661a80f17cb213b83db0247a7db990f76ef80fbec09223cd205f6e`. Temporary checkpoint manifest hash: `6a27bbb4aa4bd64ad10f322c1683b5c0f6d904ddad9752d48e87462fa333e828`; temporary export manifest hash: `7a4998a316c891a3dd9da17c975d4cc633dc499bc742dd84d93fb458a04c25ff`. Temporary artifacts were discarded after verification. These hashes identify manufactured software artifacts, not trained scientific weights.

## Scoped tooling recheck (fixes partly verified)

- `rtk proxy python -m pytest -q tests/test_volume_registry.py tests/test_volume_cli.py`: **13 passed in 5.12 s**. Strict absolute-path preflight from `/tmp` now rejects an injected `training.unknown_flag`, exit2; earlier strict-import finding is fixed in the inspected source.
- An actual profile with temporary V04 microbatch2/accumulation2, 1 warmup/1 measured CPU transaction succeeds: logical batch4, effective L+U volumes8, slices24, valid voxels6783, measured supervised/pseudo/EMA updates1/1/1. This verifies actual accumulation/batching instead of the earlier B=1 stub; the varying valid count is actual fixture support, not padded grid size.
- Actual I00 and I01 commands with their respective explicit correction protocol JSONs both still returned one prompt, one manual voxel and 957 propagated voxels. I00 falsely labels the one-click path `automatic_only`; protocol strings do not change `_correction_witness` behavior. **OPEN P1**: automatic arm must consume zero clicks and preserve model-only prior; interactive arm consumes the separate click budget.
- **OPEN P1**: `synthetic_runtime_config(...microbatch=100000)` plus strict validation accepts100000, bypassing original config microbatch<=8 safety bound. No oversized allocation was executed. Bound overrides after normalization, reject bool/nonfinite values, and add subprocess/validation tests; current time-limit comparisons also accept NaN.
- **OPEN P2**: profile pseudo threshold/delay overrides remain unrecorded; measured label-valid exposure and C01 measured-extra counts are absent. Smoke `optimizer_steps` omits the extra supervised transaction despite recording separate extra counter. Runbook still states `--steps` does not exist and lacks callable resume/evaluate/export examples.
- T00 now disables temporal objective, O03 preserves base pseudo weight, and V05 retains legacy skip fusion. Production width correction and explicit V00-to-new exposure/schedule confounds are still assigned to owners; no grouping-only causal inference is accepted.
- **OPEN P1 evaluation geometry**: a repinned manufactured evaluation manifest with each mask item declaring a +1000mm translation affine was scored successfully (2 masks read); mask affine/source/frame metadata are ignored by `_read_evaluation_mask` even when contradictory. Image-only inference correctly tolerated nonexistent mask paths and read zero labels. The mask-geometry issue was routed to the evaluation owner via coordinator.

Additional guard recheck: actual `train.py --config configs/native_volume/V04.json --synthetic --device cpu --out /tmp/not_written --steps 10000` now exits2 with the explicit full-run gate; no training occurs. A missing `--model-config` now raises on reading that exact file, so it is no longer ignored. Actual correction CLI with `--dataset-role U`, nonexistent image and nonexistent mask rejects the role before any image/mask file access.

New **OPEN P1 legacy-schema dispatch**: with historical `self_audit.engine.train` replaced by a no-op (no training), `train.py --config configs/pilot.json --data /missing/data.json --data-sha256 <64-zero hash> --out /tmp/not-written --resume /missing/checkpoint --resume-sha256 <64-zero hash> --steps 10000 --geometry /missing/geometry.json` still dispatched fresh historical training at configured_steps32. Native-only flags must be rejected for the legacy schema; legacy-only flags must be rejected on the volume path.

Actual bounded CPU `volume_profile.py --config configs/native_volume/V04.json --iterations 1 --warmup 0 --device cpu --max-minutes nan` returned PASS, total_seconds0.9993. This confirms the nonfinite timer bypass without executing a long profile.

### Further verified closures

- Physical preparation CLI now accepts `--physical-metadata`. A tiny manufactured raw NIfTI with explicit mm affine and percentile-normalized NPY U cine exercised the non-synthetic format branch successfully, with mask_paths_read=0 and geometry hash `28949b3c61a7a81a52caae251317482f51e8ccddba3bfe77cd21da76ea7cc74e`. The schema flag was false solely to exercise real-format validation; all data were manufactured and this is not real-patient evidence.
- Evaluation affine negative replay now rejects the exact earlier +1000mm translation with `mask affine does not match native image geometry`; image-only inference still reads zero masks even when paths do not exist. Evaluation source SHA-256 was unchanged before/after: `e84ffad14344dd909953fb238b516f87f2b584f0b376e6d6b8892648ba6bb49a`. The prior contradictory-reference-affine finding is CLOSED at this snapshot; configured V08 inference handling remains pending.

### Tooling owner-ready replay (2026-10-26, second repair)

Negative CLI checks are now fixed: `--max-minutes nan` rejects with exit2 and a finite-bound error; `--batch 100000` rejects with exit2 before allocation. Actual V00 smoke returns HISTORICAL_REFERENCE_ONLY, steps0. Profiling microbatch2/accumulation2 with 1 excluded warmup and 1 measured transaction returns logical4, L+U volumes8, slices24, valid voxels6783, L label-valid3230, U geometric-valid3553, measured optimizer4/supervised1/pseudo1/EMA1, and explicit pseudo-threshold/delay plus synthetic decoder-width overrides. These throughput values are local manufactured CPU values and are not hardware guidance for the target GPU.

The first repair replay suite was **1 failed, 16 passed in 4.65s**, specifically automatic `prior_unchanged is True` at test_volume_cli.py:135. Actual I00 used zero clicks/manual voxels but its returned probability hash differed from the prior; I01 used one click. Source scripts changed across the sequential replay, so this is a failed/raced check, not a stable comparison or a final verdict on the next repair. Coordinator notified.

Outstanding inspected registry metadata still called DINO production arms `tiny_test scratch`, omitted V01-vs-V00 batch/prediction/schedule confounds and described O02 only as grouped AdamW despite an encoder-LR change. Runbook gained actual resume/eval/export examples, but its step0008 checkpoint -> --steps8 resume example cannot continue an already-complete step8 checkpoint. These were returned to owner for correction.

Stable correction recheck: **17 tooling tests passed in 4.97s**. Actual I00/I01 CLI invocations shared frozen student probability hash `e7c288f698016902878588f0f7241de3266cfebdce6e47ac74af37b2a3de1bad`. I00 used zero prompts/manual/propagated voxels and preserved the prior within the declared numerical tolerance (normalization changes its exact byte hash); I01 used one prompt/manual voxel and propagated492 voxels. Smoke source was unchanged before/after at `653cc0d5c1b34dcec8502d248d368ce7e7eeba63cae91646d3a8e8df1cf9b685`. The actual automatic-versus-interactive protocol finding is CLOSED at that hash.

### Configured evaluation and remaining config replay

Independent `rtk proxy python -m pytest -q tests/test_volume_evaluation.py`: **13 passed in 1.40s**. Actual unmocked subprocess `volume_evaluate.py`, launched from `/tmp`, loaded a hash-pinned checkpoint produced by one bounded manufactured CPU CNN transaction and honored checkpoint geometry spacing `[2,1.8,8]` mm, FOV `[8,8,8]` mm and flip=true. Declared CLI slab1/tile99 became effective slab2/tile5x5, recorded per patient. Both two-frame fixture patients returned finite normalized probabilities of native shape `[2,4,3,9,11]`, labels_read=0, and explicit native support324 of594 voxels per patient (resampling support is partial, not fabricated full coverage). Source digest remained `b30a710944f312136c357be3f97fc9978687b3ad1802a3012f30b261ddc16544` before/after. A preliminary harness used the wrong trainer import, and a second untrained checkpoint attempt failed the newly required sampler-state validation; neither was a successful CLI inference. The successful check used the actual bounded trainer and valid sampler state.

Latest config/CLI recheck yielded **1 failed,21 passed in4.59s**, while registry tests/configs were being revised by their owner: the traceback displayed the new stride assertion but reported the previous `all(...)` assertion. This is raced evidence, retained without promoting it to a stable defect. Coordinator then confirmed additional V06/V07 XY stride and O02 AdamW LR changes were still active; final ready replay remains pending. Source inspection now shows V01 production DINO initialization with explicit batch/prediction/schedule/horizon/delay/exposure confounds, O02 optimizer+encoder/head LR confounds, and a genuine stop4 checkpoint-directory to target8 runbook. Native-only flags are explicitly rejected on the legacy route and vice versa; final negative subprocess replay remains pending.

### Final config and negative-dispatch replay

The settled tooling/config snapshot independently passed **24 tests in4.79s**. Actual subprocesses reject legacy-schema `--resume`, `--steps10000`, and `--geometry` with exit2 before dispatch, and reject volume-schema `--legacy-checkpoint` with exit2. No paths were opened or outputs produced for these invalid combinations. The tooling/config hash mapping was unchanged during this check (mapping digest `997f226ff69cb09a6e06b302283f12694d41696f3ccb4ffdefe4cb132bdc4bc0`). Actual one-step CPU V06 and V07 CLI smokes with crop32 also PASS, returning `[1,4,5,32,32]`, supervised1/pseudo1/EMA1 and optimizer4; loss2.8235054 and2.4551120 respectively. Production CNN widths16/32/64/128, four XYZ kernel/stride stages, and explicit tiny widths8/16/24/32 were inspected. O02 headLR3e-4 and all registry confounds are now explicit. No source/config changed across the two actual smoke subprocesses.

**OPEN P2: smoke deadline postcheck.** `scripts/volume_smoke.py:213-214` only checks elapsed before a transaction, and lines301-307 measure elapsed before the final teacher witness. Deterministic safe reproduction: `import scripts.volume_smoke as s; from types import SimpleNamespace; samples=iter([0.0,0.0,1000.0]); s.time=SimpleNamespace(perf_counter=lambda:next(samples)); s.run_smoke(Path("configs/native_volume/V04.json"),requested_steps=1,device_name="cpu",crop_size=32,max_seconds=1.0)` returns PASS and elapsed_seconds1000.0. The CPU operation is real and bounded; the timer is manufactured, no actual1000s wait occurred. This is the still-open smoke portion of original finding9, routed to coordinator. Profile deadline repairs do not close it.

Coordinator reports independent geometry reviewer found configured-support leakage and missing raw/prepared binding on real image-only evaluation; evaluation remains under owner repair. The configured evaluation success above verifies execution and receipts only, not closure of those separate defects. Full suite still held by explicit coordinator instruction.

**OPEN P2: missing/contradictory declared correction witness.** With stable smoke source `f76dff0c01a83a1a49d21367e459fa526a19096aa6cc6f07907f91bbe747c849`, actual `volume_smoke.py --config configs/native_volume/I01.json --steps 1 --crop-size 32 --max-seconds 60 --device cpu` returns PASS without any correction witness; adding `--correction-config configs/native_volume/I00_correction.json` returns PASS but executes automatic_only. I01 declares interactive_extra_supervision, so omitted/mismatched companion protocols must fail closed or explicitly report incomplete coverage. `tests/test_volume_cli.py:112-120` currently expects omitted-witness behavior. Both bounded actual CLI repros were sent to coordinator; no source edits were made.

### Final tooling repairs independently closed

Smoke deadline replay now raises `TimeoutError: synthetic smoke exceeded1.0s during after transaction0 (elapsed1000.000s)` for the exact clock-controlled reproduction; hash `f76dff0c01a83a1a49d21367e459fa526a19096aa6cc6f07907f91bbe747c849` was stable. The later protocol repair preserves these postchecks. GPU synchronization is present in source; CUDA execution remains NOT RUN and time bounds are cooperative, unable to preempt a single kernel.

Independent CLI/registry suite after correction guard repair: **32 passed in4.80s**. Actual bounded CPU subprocesses now reject both I01 without a companion and I01 with I00 companion (exit2/FAIL), with explicit required/mismatched protocol errors. Matching I00/I01 subprocesses PASS with frozen prior hash `62ae6615bc3143c68af60399e1cab60e7ce8f410656974df8afb4251028d69a0`: I00 zero prompts/manual/propagated voxels and prior_unchanged=true; I01 one prompt/manual voxel and480 propagated voxels. The factory-spy regression tests also establish rejection before model construction for protocol/mode/role/budget mismatches. Source and I00/I01 configs were unchanged across these tests and four subprocesses. Smoke hash `5cb0a63295b58a8cac7658e93869974812ac6c64143a9192273762a5919b8225`; CLI tests hash `fa5147da073e76ba70a464280c539d36a1028176e35d5ffe059f86c9ee0668a6`. Both tooling P2 items above are CLOSED at this snapshot.

The repaired evaluation/geometry scope independently passes **25 tests in1.98s**. Configured inference now merges current-grid probabilities before one native inversion and enforces native validity; real image-only input validation checks normalized raw/prepared correspondence without labels. These are manufactured CPU checks, not actual patient/GPU deployment evidence. No tooling-specific blocker remains after the scoped replays; final shared-source gate/hashes are still pending coordinator freeze.

Final receipt-denominator replay: actual one-step V04 CPU smoke with crop32 returned present_foreground3, absent_foreground0, pseudo_accepted969, pseudo_valid_voxels969 and pseudo_coverage1.0; teacher/student disagreement0.0619195 is now explicitly reported. L geometry-valid969 and label-valid969, U geometry-valid969 and label-valid0 were independently checked. Relevant smoke/engine/loss source hashes were stable across this check. The original malformed class counts and padded pseudo-coverage denominator finding is CLOSED. These fixture Dice values are not patient efficacy results.

Coordinator has now frozen source, released all code owners, and is running the final shared gate in `reports/native_volume_pilot/verified`. Per explicit instruction this reviewer does not redundantly rerun full pytest; the final report will distinguish the root-executed gate from independent scoped replays.

### Shared gate attempt1 is not final acceptance

Coordinator REPORTED `verified_attempt1`: check_repo PASS,411 pytest passes with1 warning,58 successful CLI commands including23 preflight/smokes and5 profiles, exact full4-versus-stop2/resume4 state, and image-only evaluation with0 labels. The final export equality harness incorrectly generated `randn` outside normalized input range and was corrected to `rand`; this was a harness error, not an observed production failure. A separate independent reviewer found a real TRAIN-L correction authorization/binding P1, which remains assigned for repair. Therefore attempt1 does not certify final acceptance despite its passing suite. The frozen73-file mapping captured before this later repair was `8246d4e292b4b93616f0553a8f89bdaa0b8f356547a11347efed82dd3e340ee2`; it is superseded if correction source changes. HEAD remained486fae40c490e382d58896442215476d7d32095e; only train.py differed among tracked files, and historical A0 hash remained exact.

### Stable core gate: root-executed, receipts independently inspected

Read `verified/FINAL_VALIDATION.json` (SHA256 `7a8dd5417ff3e032d9e2e668fb32a2e7d274f2a63386612b53c169a39bb8a81a`), `verified/ROUNDTRIP.json` (`b4f086710c0be8711dd22c77ea396418405f7ec84a4e8e75a30c52190a8ea9ed`), `COMMANDS.json`, check_repo/pytest logs, selected arm witnesses and all six profile JSONs. These were executed by the coordinator, not rerun by this reviewer. They document check_repo PASS; **418 tests passed,1 existing TorchScript deprecation warning,27.57s**; all59 commands exited0; all23 preflights/smokes covered V00 as provenance-only and22 executable arms. Region R01/R02 have nonzero losses/cache witnesses, T00 has no temporal witness while T01 has nonzero motion loss, I00/I01 use the identical frozen prior with0/1 click budgets, and C01 explicitly executes extra supervised updates with EMA once per outer step.

The coordinator's actual CLI roundtrip used planned horizon4, stopped at2 and resumed to4: complete checkpoint payload comparison has no differences, intermediate save cadence is exercised, frozen image-only inference reads0 labels, and exported student logits are bitwise equal with no external teacher/pretrained weight requirement. Manufactured hashes: fixture manifest `6932474cbab796f47ee5e98088b588158a737ddac1e0c37064d8c2bdefcc6794`; geometry `f7ef503cf11ead739f6596359f81592375d423cc44c4bb874442bac3c25cd205`; full checkpoint manifest `98e1991ba3fb4b05483bddb9a78c83c32f2fca16265cab34847b1e80b2382060`; resumed checkpoint manifest `c580100a56895c5e8e0c68b11a7a0393e758f7d013f2c74d8aa61000084b6b9b`; export manifest `ef4d96061080cf767cdc502a291f6ee057aeb0b45e1ce14dc7a268e0a299f8b0`. These are CPU manufactured artifacts, not scientific trained weights.

Profile arithmetic was independently recomputed from raw exposure/elapsed values, all matching receipts. With10 measured iterations and2 excluded warmups, V01/V04/V06-matchedZ3 each expose20 L+U volumes,60 slices,16473 valid voxels; native V06-Z5 separately exposes20 volumes,100 slices,23902 valid voxels. Microbatch2/accumulation1 and microbatch1/accumulation2 both expose40 volumes,120 slices,33592 valid voxels, with10 supervised/pseudo/EMA updates and40 optimizer updates. CPU rate values must not be extrapolated to CUDA throughput or memory. All VRAM/CUDA observations are null/NOT RUN.

Core source digest before/after and my independent snapshot agree: `3668c8cdd9523d2440a7e93ab2ed0564ba08ddd6bf8998ef88210ff6c96632ac`. The matching73-file mapping at06:25:18Z is `9f49fe02397416bc9bc6ba91a7abcca3abaf307803fe1bbf40645117bd114f81`. A separate C7 correction CLI time-index clamp repair is still pending; final verdict/hashes await its independent closure and post-CLI full-suite gate.

## Final post-CLI closure and source inventory

Read `POST_CLI_GATE.json`, SHA-256 `730821e3a2aa6129c8a0dcadaa9be1bf605413f9276bfde1dc64524b5c154c7e`, and independently compared all 67 recorded code/config/test hashes against current files: zero mismatches. Final gate: check_repo PASS, 422 tests passed, 1 existing warning in 29.27s. Core source remains `3668c8cdd9523d2440a7e93ab2ed0564ba08ddd6bf8998ef88210ff6c96632ac` from the 59-command gate; the final correction CLI-only frame guard is covered by this postgate.

Final independent C7 replay used a manufactured two-frame NPZ cine and 3D NPZ image. Actual `volume_correct.py` subprocesses rejected time-index -1, index 2 for a two-frame cine, and index 1 for 3D before accessing a deliberately missing prompt path (exit 1 with explicit ValueError, no FileNotFoundError). Actual boundary indices 0 and 1 succeeded (exit 0); direct selector outputs matched both distinct frame arrays exactly. No persistent output or real patient data was used.

Inventory captured 2026-10-26T06:38:12.739751+00:00; HEAD `486fae40c490e382d58896442215476d7d32095e`. Existing historical reports/configs/figures remain unchanged; train.py is the only modified pre-existing file. The coordinator has staged additive implementation/report paths during finalization; this reviewer did not stage or commit anything. A preliminary finalization assertion incorrectly treated those staged additions as modifications, aborted before writing the report, and was corrected to compare only pre-existing modified/deleted paths. Source hashes did not change.

The following mapping digest hashes UTF-8 `json.dumps(path_to_sha256, sort_keys=True)`: `f8e2b5cc1e5a7cde08c02f588b21b98c95dc4172e938de22d128771d51f4ae04`.

```text
4a4645611c31f56d3bb2b33da5f3abbb7d54cca2987dbe7d9e339742adc4ae5e  ACTIVE_PIPELINE.json
a88672044cf5c3157a16400a3bb064c446ab8d93cde8c4dcca3bc38a8260d3f8  configs/native_volume/C00.json
57ce8ee4cdc9894a4a125e11a9905d4c8b7149c9449e4059d94d16c4d99728ba  configs/native_volume/C01.json
04aaab7c8b4e12b76400cfd50d3a853c44c68c4485bd38bbf7e18d96d330e0a0  configs/native_volume/I00.json
26ab5abc02b1fc7877655b0fdee806bfb76d98f4f5f7d516fed44ff2652894fb  configs/native_volume/I00_correction.json
ef7b318358178516e8c11790a9fca888888a6b61c484c551a3ecebd702f2cdcf  configs/native_volume/I01.json
c0f0162fcfc1535261e6991ee02cf10ae057da6a9ca36f724504c258b1d56a0f  configs/native_volume/I01_correction.json
8a59a4a225470bc3d51f80463a71f52282a205c451655613cd45aad91d485172  configs/native_volume/O01.json
b5a93abeed6cfcb614041151b503fb1935a7d67ebfa3d4a37b3551e5cfb5a4f8  configs/native_volume/O02.json
6cea48e481e3bf5188158e70c2348fa38dbb3443234eb511d3e50e6e1527f1f6  configs/native_volume/O03.json
673150ea917c7406c38173923f7a320c50247ba2ce4d92f9de6735c61aa0a4c0  configs/native_volume/R00.json
f1dd19d98f311fb4df0148468e518b3326b8b9eb4e19369b380d11c77976f822  configs/native_volume/R01.json
0e3558110db662a206f284f350a55d525e6a08f85310d5fa8f16b42370498003  configs/native_volume/R02.json
1700cede94dab65a28021056cf68bdcba3b7cb69970d7cd25bd712fa1c60f00d  configs/native_volume/REGISTRY.json
556d2185d56e175dcf0593d53d7a717ab385cb6a02594d408a6751db5e504dca  configs/native_volume/S01.json
06a5e1385e6a6370a78ace85dc425b04ab9bb4c0871008387368154085404c6d  configs/native_volume/S02.json
673150ea917c7406c38173923f7a320c50247ba2ce4d92f9de6735c61aa0a4c0  configs/native_volume/T00.json
32c6475762af604b6deeda8fec0b10b6b8a1b547bea47eb24a1abb79e554b720  configs/native_volume/T01.json
add11a23c650f72cb1f64cccd0d783b730cbf580b9a046988829e9947f3fb649  configs/native_volume/V00.json
a3420876b613eb5d5b7b6f6920a406d4e8d1e252a62463b1c4b220c8612cf2d7  configs/native_volume/V01.json
7220206f1bd0e08553933e7aaeb75e86962563fdb5a38875258b2708c66c4e94  configs/native_volume/V02.json
7ab1376789a652fea829c75ce62f2a52209f81c5c72f30ce19ea7ac9fd2ddd42  configs/native_volume/V03.json
673150ea917c7406c38173923f7a320c50247ba2ce4d92f9de6735c61aa0a4c0  configs/native_volume/V04.json
0a26559cbae7ab39c86dbf13c6407f0978b017be003538435fe213f8084bd3f1  configs/native_volume/V05.json
46005090b5628c9ad0954f7611c3bd65df00eeecd84c9441453d2ca8f96f7257  configs/native_volume/V06.json
8e2134d8e04f5599248d54d416d62e3f4971cfd9d78da55a1276d357d30763c7  configs/native_volume/V07.json
829d772f9c16401bc796cc71dbae9fa069d91284043c33943e098b9542848c94  configs/native_volume/V08.json
4f40d50647eeb7c0cd3ae5fe73655cde8e0ce1650f4f3437f3707edfab874ae7  docs/METHOD.md
8a72241d4e25b7c227461e7c7153f52923b7ef45bb1eecad74e3feb249a94244  docs/native_3d/ASTRA_IMPLEMENTATION_PROMPT.md
43f5d26aa3b4876d6dcc3a1a670422569e4604166af6545b70c23b0627cc996a  docs/native_3d/IMPLEMENTATION_CONTRACT.md
5dc9f22d437a5919c2db3c6fccff6760c4d9dcf563e4cdfbaef8cba517b14c7b  docs/native_3d/RUNBOOK.md
515383134071c224db443822e33032940e532accdde334434c1051cbbcdb2f9e  scripts/check_repo.py
85439b39b33c6f0ac8d814ec8b1e11faee6223660434325a0d1f3303fb86e69f  scripts/volume_correct.py
9a73e678ae9f8b8b0829ce7cb087d2c6f6121a4af6713f92fbc29f5fc99e5acf  scripts/volume_evaluate.py
0ed301b677864d50ff949b4cc4a180b8929d1dd6145bd93aab91eb763c84a247  scripts/volume_export.py
9b6098cb78ee4abfde8912caff76997d1a14a5299b375472b46862886a7503d9  scripts/volume_preflight.py
32929bb2e1fec70dbd6ff9ca2381c6312b2e0a3433a700e9854e486eeefed881  scripts/volume_prepare.py
7a78f8352cfef083d09df36481673e31a234959b995b7ad704360e8117c37866  scripts/volume_profile.py
f1db88a88136099fa96e00d9468bed9752154960115c131894e7d521ba1f4b81  scripts/volume_smoke.py
43b22499363362f3a6c57ad21cf6c2cbf35d83678f9d396671b2985570a1335d  self_audit/volume/__init__.py
3245000e3b894167ce86dafd6717560b40cdd4ced5829e89d6caf83cbe4fe80a  self_audit/volume/checkpoint.py
23e9acf9d514d7f36665c9089bc0220a95ab393196e45cd47636f0e3aeea7673  self_audit/volume/config.py
fdb27bad361d67786652f2d8515ba132639fb606738d99b4a9a0cc1134947f08  self_audit/volume/correction.py
a5e7448c112e9cd70cd8b9e222837cfd0aa9b0a8d23414c51ecdc8f2d1a18cc8  self_audit/volume/data.py
13a7b82ee75e12ded3d7e08f292ec26f14e85034f048e72e5ee7b7da2f6fdf27  self_audit/volume/engine.py
90e5b7207af274a841a5944a33775e57e9d656af5730f8a9289e6c4455ae82c6  self_audit/volume/evaluation.py
f1b88049b6748f6ce50247a575500abad885b52a072ae2f759a9b0ad88810a90  self_audit/volume/fixtures.py
6437b518a08f7b3eb91833b8aaa8ce0b361d29dc99e84ee7c17fbf001fa61779  self_audit/volume/fusion.py
3fdb0986e4ff572c6855743414fa1f443e631d57450fe3b7803e5e52f3e1c9bd  self_audit/volume/geometry.py
47a455396033fee4048bb5f2970a443007f43e7147270d11478fbd75367e8e6e  self_audit/volume/legacy_compat.py
49dd32ec8f275333f584cc8da21dcbc3b2d89519a25e06ba282f698f5f844ed0  self_audit/volume/losses.py
4095051e59cc7595f9983e70f67fc4eeb05c82720ed1f6307b6e4919006d1dec  self_audit/volume/models.py
ae52a7866d88c364240cc63db22f657518be77aa2fac4d881013b8b25475fde3  self_audit/volume/optim.py
a192976e0a08247589a6b2e2ee688782cbf28b76cb53c1774ddd0fd6caa3eca6  self_audit/volume/regions.py
854bf74222402446d9d984dfe1eccabec44dd152a7ef52cdd43932e48f26881d  self_audit/volume/temporal.py
724adf4717300bca05299abe10e1239595e70c952cc421e456aab117be8fad8b  tests/test_volume_cli.py
1ab6563a4f414f5f8497203fa61fe14d1ffb7d9418c11aba9d352e3ec632e6f9  tests/test_volume_config.py
0fbd19770dffc99393241f60ac4e68989619447b809917009ef70a7447e77108  tests/test_volume_correction.py
00a0d5438e24c7b50ef229e2e6e84d814b26e4c574e67857169fdaa4d0a7cc41  tests/test_volume_data.py
431e5826622d7f94df2370f4744adfbc5857d13e2f77ce2dafc01e4e16f1affc  tests/test_volume_engine.py
780f7063bdb101306fb7042538280ca17749e16364d8543380a5ea89bf8ca6ef  tests/test_volume_evaluation.py
1bd75b87f72225bcdf682739218047266423c059967baa8b528e3d23c4d7d8fe  tests/test_volume_fusion.py
08c3dda96c2d8abb0896b64774476afc8d737dbae82d7ae408f73e6e95bfcc3f  tests/test_volume_geometry.py
30c4712a25c3e151b8fa642d15877aa179c0e463ee01a40ff2c34949a5ecd6b5  tests/test_volume_integration_followup.py
b042a0905be6870587d2b8f85d88d3ca71a76387b0a5f1c51de1b84a882bdaec  tests/test_volume_legacy.py
6f7c21bd053581f92f96b3d9b6a840d9c57280ad9ecb550d93bb93620848fcc2  tests/test_volume_losses.py
ee375f7d0f135392024a354dcfb60865c6561b3a98b5d3123330a6a7883176ad  tests/test_volume_models.py
bbe06278e43fd4e612740e0e65bc36e39bccbf041851ecdbdd680a269ff40b6c  tests/test_volume_optim.py
379080cd766d2e84ba280bab9aa9cfa76245fef976ca8c18b04c9886b6fbe620  tests/test_volume_regions.py
c23640474a6b0ad1a3d52a94cc0c5bda42aca51aa636af4190c56cec0e620af1  tests/test_volume_registry.py
0d9b77dd3984c01e34bf77548b532dd42747966cdd19268601353369b06631d5  tests/test_volume_resume.py
3af741402f665f522671949070ae2c5f842abddf4eef9e3ca8ea03835abf5c23  tests/test_volume_temporal.py
49a18abff36210d351ad74b09c8c47704ffc290125fbeeeef97f115da737140f  train.py
```
