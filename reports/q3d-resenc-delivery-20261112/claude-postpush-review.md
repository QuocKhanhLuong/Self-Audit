# Claude post-push review — Q3D nnFoundationCNN ResEnc-L teacher

- Target worktree: `/Users/alvinluong/orca/workspaces/Self-Audit/q3d-resenc-20261112`
- Commit: `6c5065262dddf2ee872b4dcabf4324f78a83df86` (== `origin/main` after fetch; worktree clean, 0 changed files)
- Mode: read-only. No repo edits, no downloads, no patient training. Probe script and outputs were kept in a session scratchpad only.
- Scope read: `AGENTS.md`, `ACTIVE_PIPELINE.json`, all of `docs/CANDIDATE_Q3D_SPEC.md` (420 lines), `self_audit/candidate_q3d/{teacher_resenc,pretrained,teacher_features,alignment}.py`, teacher-related parts of `engine.py` (`_teacher_stage`, `_make_teacher_optimizer`, `_teacher_head_parameters`, `_teacher_parameter_groups`, `_run_teacher_stage`, the teacher feature/alignment call sites), and the test inventory of `tests/test_q3d_teacher_resenc.py`, `tests/test_q3d_pretrained.py`, `tests/test_q3d_alignment.py`, `tests/test_q3d_preflight.py`.

## Verdict

**No blocking findings.** I found no concrete, reproducible correctness bug in the teacher migration at this commit. There are four non-blocking notes below: one test gap and three pieces of dead code or config.

## Independently verified (reproduced, not copied from earlier reports)

I ran a probe against the actual external checkpoint `/Users/alvinluong/Self-Audit-runs/q3d-resenc-20261112/checkpoint_final.pth`, using the locked strides `[(1,1,1),(2,2,2),(1,2,2)x4]` and `dynamic_network_architectures` 0.4.4:

1. **Strict encoder only.** `load_nnfoundation_encoder` loaded 448 keys under strict mode, and `hf_revision` resolved to `f443c093…`. The checkpoint holds 956 tensors, 508 of them `decoder.*`, and none of those were passed to the model. All 448 loaded tensors are bit-identical (`torch.equal`) to the checkpoint's `encoder.*` entries. The filter (`encoder.stem.` / `encoder.stages.` prefixes) excludes `decoder.encoder.*` and keeps the `all_modules.*` aliases, as spec §3 requires. Before the load, the code checks key-set equality and every tensor shape.
2. **Architecture parity.** `build_res_enc_l` uses the same arguments as nnssl `get_res_enc_l` at the pinned commit: widths [32,64,128,256,320,320], blocks [1,3,4,6,6,6], k3, InstanceNorm3d (affine, eps 1e-5), LeakyReLU, conv_bias. Changing the strides does not change any weight shape, which is why the strict load passes with the anisotropic strides.
3. **Phase.** I computed `grid_phase` from the live main-path convs (the stem conv, then `blocks[0].conv1/conv2` of each stage). It gives stage 3 `((2,8,8),(0,0,0))` and stage 4 `((2,16,16),(0,0,0))`. Both equal `Q3DResEncTeacher.phase_for_source(...)`. Under the preregistered rule (mean ERF offset (0.672, 1.947, 1.722) ≤ 2 voxels on every axis), the main phase with zero correction applies. `pretrained._require_phase_measurement` pins the receipt path and SHA (`2f05888b…`), PASS, 16 samples, threshold 2.0, stage-3 offset 0, and stage 4 explicitly `NOT_MEASURED`. It re-hashes the receipt file.
4. **Z-score, pad and crop.** The teacher alone applies a volume-wise z-score before padding. It then replicate-pads on the high side only, to the cumulative stride. Logits and SDF are cropped back to native shape. An input of shape (9,61,53) produced region, boundary and SDF outputs of (1,4,9,61,53), and features stride8 (1,256,5,8,8) and stride16 (1,320,5,4,4). The features stay on the padded encoder lattice, which matches the documented alignment contract. The student input is never normalised by the teacher.
5. **Parameter aliases.** `region_decoder.encoder` holds the same parameter objects as `encoder`: 293 unique parameters, versus 985 when duplicates are counted. The optimizer groups in `_teacher_parameter_groups` and `_teacher_head_parameters` are built by object identity, so the groups are disjoint. In T-a, `requires_grad` is off for the encoder; in T-b the encoder gets 1e-4 and decoder/heads 1e-3. The state dict round-trips with strict mode. A checkpoint whose `region_decoder.encoder.*` diverges from `encoder.*` is rejected with `ValueError`.
6. **Provenance and no fallback.** A real build requires all of the following, or it fails:
   - `architecture=nnfoundation_resenc_l`;
   - an exact provenance key set with `READY`;
   - the pinned SHA and the exact HF resolve URL with its revision;
   - the Table D7 dataset string, `overlap_acdc=overlap_mnms=false` and the not-independently-verified note;
   - CC-BY-SA-4.0;
   - DNA, nnssl and nnU-Net pins, plus a runtime check of the installed DNA version.

   `Q3DResEncTeacher(encoder=None, synthetic=False)` raises an error. Random or reduced weights are reachable only through `synthetic=True`, which uses the same class.
7. **Old teacher removed.** I grepped `self_audit/candidate_q3d`, `configs/candidate_q3d`, `scripts/*q3d*`, `train.py` and `evaluate.py` for `mednext|tinyencoder|decoder_stride8|stride4|x_res_3|bottleneck_stride16|random_init_fallback`. The only hit is the dead alias in note 2 below.
8. **Frozen teacher in Q, P and J.** The Q stage (`engine.py:1501`) and the J stage (`engine.py:1734`) call `_teacher_stage(..., "joint", freeze=True)`, which sets eval mode and turns `requires_grad` off. `forward_teacher_with_features` runs under `no_grad` and detaches every output.
9. **A/A-3D unchanged.** `git diff 37851c7 HEAD -- self_audit/candidate_a self_audit/candidate_a3d` is empty. The last commits touching those paths are old baseline commits (`595a1b3`, `e4e26ee`, `b643e9d`).

## Non-blocking notes (ranked)

1. **Test gap: teacher phase is analytic at runtime.** `Q3DResEncTeacher._derive_feature_phases` (`teacher_resenc.py:380`) computes the step as a product of strides and takes the offset from config. It does not walk the live conv modules. For the pinned DNA blocks (k3, padding 1) the result is correct: my probe matched `grid_phase` on the real convs for the locked strides, and `scripts/q3d_preflight.py:545` derives the phase from the real chain. No unit test asserts `phase_for_source(...) == grid_phase(actual main chain)` for an anisotropic encoder. Adding one would catch a future DNA or kernel change silently diverging. Today it causes no wrong output.
2. **Dead alias.** `engine.py:613` still accepts `"bottleneck_stride16"` and `"stride16"` in `_features_to_student_grid`. It cannot be reached: config validation (`config.py:737`) allows only `encoder_stride8/16`, and `teacher_features._canonical_source` rejects `bottleneck_stride16`. Cleanup only.
3. **Dead config.** `training.teacher_boundary_lr` is validated (`config.py:888`) but never used. The boundary path is in the `teacher_heads` group at `teacher_head_lr`. Both are 1e-3, so behaviour matches spec §3. The field is misleading if someone changes it later.
4. **Minor robustness.** The checkpoint is hashed in `pretrained.py:317`, hashed again in `load_nnfoundation_encoder`, and only then loaded with `torch.load`. This leaves a small time-of-check/time-of-use window on a local file, and `weights_only=True` reduces the risk. Separately, the alias check in `Q3DResEncTeacher.load_state_dict` would raise a `RuntimeError` from `torch.equal` if someone mixed tensors from different devices in one state dict. Neither happens in current code paths.

## Limits and assumptions

- I did not rerun the full pytest suite or the smoke run. I relied on the coordinator's report of 511 passed and 13/13 smoke, and I ran only the targeted CPU probe above.
- No GPU, VRAM or real-data execution. Strict load and the ERF are evidence about software and geometry only. They say nothing about patient efficacy.
- I did not have nnssl source locally. Parity with `get_res_enc_l` was checked against the arguments written in the spec and the code, the live DNA 0.4.4 module structure, and the 448/448 strict key and shape match with the real checkpoint.
- O1 and S1 paths are out of the campaign and only lightly reviewed. S1 stage 4 is analytic phase, explicitly unmeasured, as the spec declares.
- I did not review other workers' unfinished scopes. Nothing from earlier reports was repeated without rechecking it in the final source.
