# Independent Claude review: Q3D nnFoundation ResEnc-L teacher (worker `resenc_teacher`)

- Date: 2026-11-12
- Workspace reviewed: `/Users/alvinluong/orca/workspaces/Self-Audit/q3d-resenc-20261112` (uncommitted working tree on top of `37851c7`)
- Scope: `self_audit/candidate_q3d/teacher_resenc.py` (new), `teacher_features.py` (rewritten), `alignment.py` (unchanged), `tests/test_q3d_teacher_resenc.py` (new), `tests/test_q3d_alignment.py` (unchanged, still being cleaned up by the worker)
- Mode: read-only. No edits, training, downloads, commits or pushes. Probes ran from the scratchpad against synthetic tensors and the already-present external checkpoint.
- Evidence class: software and static evidence only. Nothing here is real-data or GPU evidence.

## Verdict

The architecture matches the pinned spec. Strict encoder loading is correct and fails closed on hash, key and shape errors. I found no blocking correctness bug in the teacher forward path itself. The main problems are in provenance and fail-closed behaviour: the revision label is hard-coded, the expected SHA is not pinned, real construction defaults to random weights and isotropic strides, and parameter aliasing is invisible to strict state loading. Two items are pending geometry rather than bugs: the feature-phase offsets (ERF measurement still running) and the half-cell context shift in the boundary path. Engine and config integration is still incomplete, which matches the worker's in-progress state.

## What I verified

| Check | Result |
|---|---|
| `build_res_enc_l` compared with `get_res_enc_l` (from `architecture_registry.py`, nnssl `3d774e94`) | Identical: n_stages 6, features `[32,64,128,256,320,320]`, kernels 3³, strides `(1,2,2,2,2,2)`, blocks `[1,3,4,6,6,6]`, decoder `[1]*5`, conv_bias True, InstanceNorm3d(eps 1e-5, affine), LeakyReLU(inplace), default block. Only `num_classes=4` differs, and it affects the discarded head only. |
| DNA version | `dynamic_network_architectures 0.4.4` installed |
| Strict load of the real `checkpoint_final.pth` (SHA `ac262d3e…2bbd`) | 448 keys (stem 8 + stages 440), no missing or unexpected keys, every tensor equal to `network_weights["encoder."+k]`. `decoder.*` is never passed in. `decoder.encoder.*` is byte-identical to `encoder.*` in the checkpoint, so ignoring it loses nothing. |
| Strict load with anisotropic strides `[(1,1,1)]+[(1,2,2)]*5` | Loads; weight shapes do not depend on strides. |
| Wrong SHA | `ValueError: nnFoundation checkpoint SHA-256 mismatch`, raised before `torch.load`. `weights_only=True` is used. |
| Channels | S3 (index 3, "stride8") = 256, S4 (index 4, "stride16") = 320, last stage = 320. Synthetic build uses the same class with widths (4,8,12,16,20,20). |
| Z-score | Computed per sample over native voxels **before** padding (`teacher_resenc.py:344-347`). The test shows invariance to affine intensity changes. A constant image stays finite thanks to the std clamp. |
| Padding / crop | Pads only on the high side, to the cumulative stride per axis, with no 2× minimum. Region, boundary and SDF outputs are cropped to the native shape. Anisotropic case `(6,40,40)` gives pads `(0,24,24)`; outputs are native. Iso default with D=6 pads Z to 32. |
| Small D | D=6, 8, 10 run with both stride schedules; feature grids are `ceil(padded/stride)`. |
| Seeded repeat build | Two `torch.manual_seed(3)` builds give identical state_dicts. |
| Tests | `tests/test_q3d_teacher_resenc.py`: 6 passed, including the real-checkpoint test. `tests/test_q3d_alignment.py`: **collection error** (`ImportError: cannot import name 'UPSTREAM_MEDNEXT_COMMIT'`), expected while the worker cleans up old tests. |
| Frozen baselines | No diffs under `self_audit/candidate_a/` or `self_audit/candidate_a3d/`. `_BoundaryPath3D` is imported, not edited. |

## Findings (concrete issues, most severe first)

### F1. Provenance: `hf_revision` is hard-coded and the expected SHA is not pinned (correctness/provenance, medium-high)
`teacher_resenc.py:202` always reports `"hf_revision": "f443c093…"`. `load_nnfoundation_encoder` checks the file only against the **caller-supplied** `sha256` (`:173-177`). Any ResEncL-shaped checkpoint whose SHA the caller passes will load and be labelled with revision `f443c093`, which breaks the "pretrained source and hash" fail-closed contract. The upper-case form of the hash is also accepted (harmless, but it is not canonical).
*Fix:* add a module constant `EXPECTED_SHA256 = "ac262d3e…2bbd"` keyed by revision. Reject a mismatch with the pinned value for real runs, or take `revision` as an argument and check it against a pinned `{revision: sha}` table. Echo the supplied revision instead of a literal.

### F2. Real construction silently falls back to random weights and isotropic strides (fail-closed, medium)
`Q3DResEncTeacher(encoder=None, synthetic=False)` (`:255-264`) builds a randomly initialised full-width ResEnc-L, and `strides=None` means `RESENC_L_DEFAULT_STRIDES` (iso). For ACDC (Z=6..10) the iso schedule pads Z from 6 to 32, so about 80% of the volume is replicated padding. Because every block uses InstanceNorm, that padding also changes the statistics, and therefore the features, of every native voxel. Neither default should be reachable in a real run.
*Fix:* when `synthetic=False`, require an `encoder` produced by `load_nnfoundation_encoder` (for example, check for a provenance attribute set by the loader) and require explicit, Phase-0-locked strides. Keep the iso default for synthetic or explicit opt-in only.

### F3. Parameter aliasing: `region_decoder.encoder` duplicates the encoder in `state_dict` and is invisible to strict loading (maintainability/integrity, medium)
DNA's `UNetDecoder.__init__` stores `self.encoder = encoder` (`unet_decoder.py:46`). As a result:
- With the real encoder, the teacher state_dict has 985 keys, 448 of which are `region_decoder.encoder.*` duplicates. Synthetic: 345 keys, 128 aliases, 133 unique params. `parameters()` deduplicates, so the optimizer is fine.
- On `load_state_dict`, whichever of `encoder.*` or `region_decoder.encoder.*` comes later silently overwrites the other. A diverged or tampered teacher artifact passes `strict=True` without any error.
- Inventories built from `state_dict()` keys (for example the `random_keys = sorted(teacher.state_dict())` pattern in `pretrained.py`) double-count the encoder.
*Fix:* drop `region_decoder.encoder.*` from saved teacher state (override `state_dict`/`_load_from_state_dict`, or hold the decoder's encoder reference outside module registration), or assert that each alias equals its `encoder.*` tensor on load. Add a test that pins the expected key count.

### F4. Phase offsets are hard-coded instead of measured from modules (pending geometry, plus a maintainability note)
`_derive_feature_phases` (`:299-309`) hard-codes offset `(0,0,0)`. I measured with `alignment.grid_phase`:
- Main path (stem conv, then each stage's `blocks[0].conv1.conv`, k3/p1): offset **0** at stride 8 and at stride 16. The hard-coded value matches the main-path phase.
- Residual skip path (`AvgPool3d(stride)` then 1×1): offset **3.5** in-plane at stride 8 and **7.5** at stride 16 (z = 0 with a (1,2,2) schedule).

Per the user's rule (2-voxel threshold applies to the main phase; otherwise use the measured offset), this is **pending the 16-sample Phase-0 ERF**, not an automatic STOP. Recommendations: derive the main phase with `grid_phase` from the live modules rather than a literal, so a stride or kernel change cannot desynchronise it, and record the ERF-measured centroid offset next to it. Note that `"stride8"` and `"stride16"` are labels: under `(1,2,2)` the actual steps are `(1,8,8)` and `(1,16,16)`, and `phase_for_source` already returns the actual steps.

### F5. Boundary path context has a half-cell shift (pending geometry, inherited, low-medium)
`_BoundaryPath3D.forward` (frozen A3D) upsamples `skips[-1]` with `F.interpolate(..., align_corners=False)`. That assumes deep-cell centres at `(i+0.5)·32−0.5`, while the main lattice puts them at `32·i`, a shift of 15.5 voxels in-plane (none on z with a z-stride-1 schedule). The branch is trained fresh in stage T and the context is coarse, so this is a known limitation rather than a bug. It cannot be fixed without editing the baseline. If the ERF/phase work requires it, pre-align `skips[-1]` in Q3D before calling the boundary path.

### F6. Hash-then-load TOCTOU and loose payload type checks (security, low)
`_sha256(path)` and `torch.load(path)` read the file separately (`:175-178`), so the file could change between the two reads. With `weights_only=True` the impact is limited to loading unverified tensors.
*Fix:* read the bytes once, hash them, then `torch.load(io.BytesIO(data), weights_only=True)`; 410 MB is acceptable.
Also, `_strict_encoder_state` calls `payload.get("nnssl_adaptation_plan", {}).get(...)` before the `isinstance(plan, Mapping)` check, so a non-mapping plan raises `AttributeError` instead of `ValueError`. `pretrain_num_input_channels` is not checked against `input_channels`; the stem shape check catches a mismatch anyway.

### F7. Feature "crop" is a no-op; features stay on the padded lattice (maintainability, low)
`forward` (`:361-362`) calls `_crop(skips[3], native_shape)` on a stride-reduced tensor, so the slice never removes anything. `align_teacher_to_student4` accepts this (it requires `>= ceil(native/step)` and samples only native centres), so results are correct. The call is misleading, though, and implies native-sized features.
*Fix:* crop to `ceil(native/step)` or drop the call and document "padded-lattice feature".

### F8. Geometry validation edge cases (low)
`_normalise_geometry` rejects a NumPy `shape` with a misleading "must equal" message, because `np.ndarray` is not a `Sequence`. It raises `TypeError` instead of `ValueError` when `affine_zyx=None`. The engine path goes through `canonical_geometry`, which yields tuples and lists, so this affects direct callers only. `affine` is validated but unused (fine).

### F9. Throwaway decoder construction consumes RNG (low)
`build_res_enc_l` instantiates a full `ResidualEncoderUNet`, including the fresh decoder and head, and `load_nnfoundation_encoder` and the teacher keep only `.encoder`. This wastes memory and time, and it advances the global RNG. Downstream seeded init (decoder, boundary, student) therefore depends on whether that path ran. The behaviour is deterministic, but it couples seeds across the real and synthetic paths. Building `ResidualEncoder(..., return_skips=True)` directly with the same kwargs would avoid it; keep the comparison with the full wrapper in a test.

## Test gaps
- `test_official_checkpoint_strict_encoder_load_and_ignore_decoder` hashes the file it then loads, which is tautological. It should assert against the pinned SHA `ac262d3e…`. It also hard-codes an absolute user path as the default.
- No tests for: anisotropic strides; D=6 shapes; key-mismatch, shape-mismatch, wrong-architecture and bad-plan fail-closed paths; alias key count or divergence; declared phase vs `grid_phase` from the live modules; seeded repeat build; non-synthetic teacher refusing random init (after the F2 fix).
- `tests/test_q3d_alignment.py` still imports `UPSTREAM_MEDNEXT_COMMIT`, `TinyEncoder3D` and MedNeXt helpers, and fails at collection (worker cleanup still pending).

## Integration state (pending, not bugs in the reviewed files)
- `pretrained.py` (`build_q3d_teacher`, `_synthetic_teacher`, `_construct_mednext_teacher`) still builds a MedNeXt `DualBranchTeacher3D` and requires `architecture == "mednext_l_k3"`. `teacher_features` now raises `TypeError` for anything except `Q3DResEncTeacher`, so the engine and smoke feature path will fail until the teacher builder is rewired (found by inspection; not executed).
- Configs `s1.json` (`bottleneck_stride16`), `v1_stride4.json` (`stride4_decoder_block`) and `v1_decoder_stride8.json` (`decoder_stride8`) use sources that `_canonical_source` no longer accepts. They fail closed, but they need migration or retirement.
- `engine._features_to_student_grid` checks `source in {"stride16","bottleneck_stride16"}`. `encoder_stride16` takes the other branch, but both branches call `align_teacher_to_student4`, so the behaviour is equivalent; this is cosmetic.

## Side note (outside scope)
`geometry-audit.md` reports a median TRAIN spacing of `[1,1,1]` mm while 28/80 entries have 10 mm z. This is unusual for native ACDC (typically about 1.25–1.6 mm in-plane and 5–10 mm z). It is worth confirming that the manifest `spacing_zyx` is the native spacing, because the boundary path conditions on it and the stride-schedule choice depends on it.
