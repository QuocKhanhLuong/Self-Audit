# Candidate PG v2 — Claude logic review

**Reviewer:** single Claude reviewer (no sub-agents).
**Phase:** §0–§4 are the PRELIMINARY pass (accepted by root). §5 holds the INTEGRATED pass findings, with reviewed file SHA prefixes. §6 is the final disposition table, completed once root reports the integration tests.
**Date:** 2026-11-16. **Tree:** `/Users/alvinluong/Self-Audit` `main` @ `17efb5f` plus uncommitted worker files.
**Scope:** code logic only. No server, GPU, training, or external messaging. This file is the only file I wrote. It makes no efficacy claim and asks for no new experimental quality gates.

Severity: **BLOCKER** (the real flow is wrong or cannot run as specified) · **HIGH** (wrong formula, gradient, leakage, or comparability) · **MEDIUM** (contract or config ambiguity that will cause drift) · **LOW** (hygiene).

---

## 0. Implementation status when this snapshot was taken (about 02:44)

- `self_audit/candidate_pg/` holds only `__init__.py`, `engine.py` (383 lines), `evaluation.py` (408), and `teacher_mednext.py` (252). `configs/candidate_pg/` does not exist. There are no `tests/test_*pg*` files. `views.py`, `reliability.py`, `student_pg.py`, `cinema_init.py`, and `codebook.py` are missing.
- **STUB SIGNAL — `engine.py` is a placeholder, not the spec flow** (details in §3). Workers are still writing, so I will re-check this in the integrated pass.

---

## 1. Spec-level traps found against the existing APIs (they apply to every worker)

### 1.1 BLOCKER — the official MedNeXt factory is isotropic, so the anisotropic MedNeXt-L in the spec cannot come from `create_mednext_v1`
- `candidate_a3d/teacher.py:358-420` `_load_mednext` calls `factory(num_input_channels, num_classes=4, model_id="L", kernel_size=3, deep_supervision=False)`. Upstream MedNeXt-v1 hard-codes stride 2 on all three axes. `MedNeXtDownBlock.conv1` and `res_conv` use stride 2. `MedNeXtUpBlock` uses `ConvTranspose3d(stride=2)` followed by `F.pad(x,(1,0,1,0,1,0))` on **all** axes.
- The spec asks for `down_strides [[1,2,2],[1,2,2],[2,2,2],[1,2,2]]` with matching up strides. The upstream z-pad in the up block breaks shapes whenever z stride is 1.
- **Fix:** build PG-owned anisotropic down/up blocks that reuse the upstream block internals (MedNeXtBlock, GRN, norm), keeping `block_counts`/`exp_ratios`/channels identical. Upsampling must pad only the strided axes, or use `output_padding` per axis. Keep the hash-pinned upstream checkout as the reference for the block internals.
- **Tests:** forward a native shape with D odd (for example D=9) and H, W not multiples of 16. Assert that skip shapes match at every decoder level and that the output is cropped to native. Phase test: derive all five phases from the live `conv1` modules **and** from `res_conv`, and check that both equal `((1,1,1),(1,2,2),(1,4,4),(2,8,8),(2,16,16))` with offsets 0.

### 1.2 HIGH — `align_feature` / `align_teacher_to_student4` always detach their output
- `candidate_fuse/fusion.py:51` returns `aligned.detach()`. `candidate_q3d/alignment.py:203,224` detaches both the input and the output.
- These helpers are correct for **teacher** targets only. Using them on any student-side tensor silently kills the gradient. The places this could happen are mapping student stage3/stage4 onto the CineMA grid in S1, moving `P_S(z_S)` in CBKD, or any student-to-teacher grid move. The loss would still be finite, so a smoke test cannot catch it.
- **Fix:** always move the frozen side (CineMA, teacher) onto the student grid. Never pass student tensors through these helpers.
- **Test:** for S1 and CBKD, compute `loss.backward()` and assert that `student.stage3/stage4` parameter grads are non-None and nonzero.

### 1.3 HIGH — CineMA mapping must not reuse `align_teacher_to_student4`
- That helper assumes the same native voxel lattice (differing only by stride and phase). It **clamps** coordinates to the endpoint (`alignment.py:180-195`), so cells outside the source get replicated values.
- CineMA works on its own resampled 192×192×16 crop. The mapping needs native-voxel → world (`affine_zyx`) → CineMA-crop-voxel coordinates, followed by trilinear sampling. Every coordinate outside the cropped support, and every CineMA z-pad slice when D<16, must have `valid=0`, not a clamped value.
- **Test:** build a synthetic affine with a known shift and scale. Assert that the sampled feature equals an analytic linear field inside the crop and that `valid==0` exactly outside it, including the D>16 crop case and the D<16 pad case.

### 1.4 HIGH — arm comparability depends on RNG streams that are independent of the switches
- `StrongInplaneAugment` (`candidate_fuse/augmentation.py:25`) has **one** private generator that `spatial()` and `intensity()` both consume. Dropout3d (FP) and `nn.Dropout` use the global torch RNG.
- In A1 (pg=false, cbkd=false), the U views, strong intensity, CutMix boxes, and FP dropout are not drawn. Unless the streams are separated, the labeled augmentation sequence, keep-masks, and the U sampler cursor in A1 diverge from A2–A4. That breaks "same augmentation/update count".
- Module construction order has the same problem. If A1 does not build `code_branch`/`P_S`/`H_S_Q`, the RNG state when later random modules (decoder) are initialised differs across arms.
- **Fix:** (a) construct every module in every arm in the same order, or better, materialise one shared full-student init `S0` after S1 and load it into all arms. (b) Use separate seeded generators per stream: labeled aug, U-A aug, U-B aug, CutMix, keep mask, FP dropout (use a Bernoulli mask with a generator), and the U sampler. Alternatively, always draw the U samples and augmentations regardless of the switches.
- **Test:** run A1 and A4 for 3 synthetic updates and assert that the labeled (image, target) tensors, keep masks, and sampler keys are bitwise equal per update. Also assert that the initial `state_dict` of shared keys is equal across all four arms.

### 1.5 HIGH — the valid mask must follow the same spatial draws
- `StrongInplaneAugment.spatial` (`augmentation.py:94-138`) returns no validity. Rotate, scale, and elastic use `padding_mode="zeros"`, so padded regions become image 0 **and label 0 (background)**.
- **Fix:** snapshot `generator.get_state()` before the spatial call. Replay the identical draws on a ones-mask passed as `target` (nearest), then restore the post-call state so the stream advances exactly once. Do not use a "nonzero image" heuristic, because real background is ≈0.
- **Test:** for a constant-1 image, the replayed valid mask equals the support of the warped image, and the generator state after `views` equals the state after one plain `spatial()` call.
- **Note (LOW, decide explicitly):** the labeled CE/Dice in `supervised_loss` currently counts zero-padded regions as background, as FUSE A-aug did. PG can either keep that (and document it) or set invalid labeled voxels to 255. `_sdf_targets` rejects 255 (`candidate_a3d/bank.py:252`), so the SDF target must be computed before masking.

### 1.6 HIGH — SDF channel count conflict: spec says `sdf channels3`, Q3D needs 4
- `balanced_near_far_sdf_loss` requires `[B,4,D,H,W]` (`candidate_q3d/losses.py:53`). `q3d_teacher_loss` requires the boundary/sdf shape to equal the region shape (`:113`). `_sdf_targets` emits `max(classes)+1 = 4` channels with background at channel 0 (`bank.py:424`).
- The spec text says "heads region4/sdf channels3" and also says "Supervision exact q3d_teacher_loss". Both cannot hold. The draft teacher emits 4 SDF channels through `_BoundaryPath3D(...,4)`, which is correct for the unchanged loss.
- **Fix:** record the decision as 4 channels (bg + RV/MYO/LV), with the PG boundary band defined as `min over c∈{1,2,3} |d_c| < 0.5`. Including the background channel only adds the union boundary, which is redundant. State it in config and test it.
- **SDF normalisation:** targets are signed physical voxel-face distance / 5 mm, clipped to [-1,1]. An absent class is +1 and a full volume is −1 (`bank.py:363-381`). The teacher output is `tanh` (`a3d/teacher.py:540`). The band threshold 2.5/5 = 0.5 is therefore exact only if the PG radius is 5 and near is 2.5.
- **Test:** a hand-built 1-D-like volume with known spacing (for example 1.5 mm in-plane) gives an exact band membership at the 2.5 mm edge.

### 1.7 MEDIUM — `_schedule_lr` is not in `candidate_fuse`, and it needs a `warmup_fraction` the PG spec never states
- `_schedule_lr` lives in `candidate_q3d/engine.py:668`. FUSE imports it in `teacher.py:15`. Its signature requires `warmup_fraction`. FUSE J uses its own inline variant (`fuse/engine.py:1047-1051`), and its progress term is off by one relative to `_schedule_lr`.
- The PG spec lists no warmup fraction, and FUSE semi_k8 uses 0.05. "No hidden hyperparameter defaults" means the base config must write `training.warmup_fraction` explicitly, and likewise for the teacher, S3, and S1 schedules. `candidate_fuse/teacher.py:196` has a hidden `.get("warmup_fraction", 0.05)` that must not be inherited.
- Batch sizes are also unspecified for S1 (10000 updates), S2 (8000), and S3 (4000), as are `gradient_clip_norm` and `amp_dtype`. All of these must be explicit in `pg_k8_base.json`.

### 1.8 MEDIUM — λ_u / λ_cb schedule wording is ambiguous; fix one formula and test it
- FUSE `_lambda_u(update,total,fraction) = min(1, update/ceil(fraction*total))` has **no** warmup offset (`fuse/engine.py:114-116`).
- PG says "warmup_updates_labeled_only 800 / ramp_fraction .1". The recommended reading is `λ_u = 0` for `u < 800`, else `min(1,(u-800+1)/ceil(.1*8000))`. `λ_cb` is a step to 0.5 at `u ≥ 800`. Write the chosen formula into the code docstring and REQUEST/README, and add a unit test at u = 799, 800, 1599, and 1600.

### 1.9 MEDIUM — the order of threshold update vs weights must be fixed
- The spec sequence is "pack probabilities, class weights and valid. Update threshold A+B". FreeMatch updates first and then masks. Pick one order (recommended: compute weights with τ **before** the update, then update with the combined A∪B valid voxels), document it, and test it.
- EMA state must be float64 or float32 that is **not** under autocast, so that resume is bit-identical.
- Init τ_g = p̃_c = 1/C gives τ_c = clip(.25·1, .5, .95) = 0.5.
- **Test:** two manual steps with hand-computed values.

### 1.10 MEDIUM — weighted Dice class set
- `supervised_loss` uses foreground-only Dice (`self_audit/losses.py:29-30`). The spec "weighted soft Dice per class … skips classes without weighted target mass" does not say whether background is included. The recommendation is FG {1,2,3}, matching the labeled loss. Weighted CE uses `sum(w·ce)/sum(valid)`, so the denominator is the valid count, **not** the weight sum. The test must catch a `/sum(w)` slip.

### 1.11 MEDIUM — teacher semi stream: CutMix or not?
- The S2 text says "intensity-only strong streams". The shared augmentation block says CutMix is on. If the teacher uses CutMix, the SDF-consistency target and confident mask must be mixed with the same box. Decide which and state it in the config (`teacher.semi.cutmix`).

### 1.12 MEDIUM — two-U-frame sampler; `FuseSampler.next_pair` is one-U
- `FuseSampler.next_pair` (`fuse/data.py:398-407`) yields one labeled key and **one** U key from a patient-epoch permutation. PG needs a new checkpointable sampler that yields 1 L key plus U-A and U-B keys with `patient(A) ≠ patient(B)`, patient-balanced over all 80 TRAIN patients. U keys come from `FuseData.unlabeled_keys()`, which already includes unlabeled-patient ED/ES and excludes only the 16 L frames (`fuse/data.py:305-310`).
- **Tests:** A≠B patient on every update, no L key in U, state round-trip, and a coverage histogram that is uniform per patient-epoch.

### 1.13 MEDIUM — the FP target and partner geometry
- FP (UniMatch) must use the **unmixed** clean-weak-A packet as its target, not the CutMix packet. s1/s2 use the mixed packet with the same boxes.
- The partner B packet `(p_W, y, w, valid)` must be center crop/padded to A's canvas **including depth** (`FuseStudent._shape` requires `D_img == D_geom`, `student.py:23`). Padded cells get `valid=0`. Agreement and band are computed on B's own native canvas **before** crop/pad. The student forward on the mixed image uses A's geometry.

### 1.14 MEDIUM — export reuse silently drops unknown keys
- `export_model` (`fuse/export.py:69-75`) keeps only keys present in a fresh `FuseStudent(include_fusion=False, include_mimic=False)`. Any extra key (`code_branch.*`, `P_S.*`, a mistakenly registered `teacher.*`) is dropped **without error**.
- The spec asks for `strip_prefixes` plus `assert_no_teacher_keys`. Implement an explicit strip, then assert `set(stripped) == set(FuseStudent(...).state_dict())`. Also assert that no key starts with `teacher`/`encoder.`/`fusions.`/`mimic_heads.`.
- `load_export` rejects any schema except FUSE `EXPORT_SCHEMA` (`export.py:170`). If PG writes its own schema, it needs its own loader. `verify_cold_load` requires exactly three volumes (`export.py:230+`).

### 1.15 LOW — metrics columns need ASSD and FG HD95/ASSD
- `fuse/evaluation.py:evaluate_modes` computes only Dice and HD95. `candidate_q3d/evaluation.py:117-190` already returns `assd_mm` and FG means over present classes with absent-class `None`. PG should reuse it. Patient macro means frames → patient → split, as FUSE does at `fuse/evaluation.py:140-170`.

### 1.16 LOW — student phases vs codebook tap
- The student stage3 phase derives as `(1,8,8)` with offset 0 (stem (1,3,3) stride 1, then three (1,2,2) downs; `student.py:80-85`). The teacher level-3 phase is `(2,8,8)`. `align_feature` interpolates z ×2. That matches the spec. Only post-EPA `s3` (128 ch with channels `[16,32,64,128,256]`) is the tap, not `s3_pre`.
- Valid for CBKD/qtask comes from `FuseStudent._valid_centers` (internal pad) **times** the pooled augmentation valid.

---

## 2. Required gradient and leakage invariants (to verify in the integrated pass)

1. Labeled update: the teacher runs under `no_grad` on the weak L view. Fusion and student-only paths use the same strong L. `loss = .5·sup(fusion) + .5·sup(student)`.
2. **Gates and projectors (`fusions.*`) get gradients only from the labeled terms.** Test: `torch.autograd.grad(loss_pg + loss_cbkd, fusions_params, allow_unused=True)` gives all None or zero.
3. s1, s2, FP, and the code branch call `student(..., teacher_features=None)`. Test: patch `teacher.forward` and `GatedFusion.forward` with call counters during the student-stream section.
4. Targets: `p_W` (fusion weak, keep=1, `no_grad`), `pT`, `pS_cleanweak` used inside the agreement, `y`, `w`, and `rho_T` must have `requires_grad=False`. The clean-wa student forward keeps grad only for its stage3 when CBKD is on, and its probabilities are detached for agreement.
5. CBKD: C is a buffer or frozen parameter that is never in an optimizer param group. `e_st = z + (e - z).detach()`. `commit = ||z - sg(e)||²`. `KL(sg ρT ‖ ρS)` direction is `Σ ρT (log ρT − log ρS)`. Code order is preserved by SHA of C before and after the arms.
6. TEST: per-epoch TEST rows exist but are never read by selection. Test: perturb TEST metrics and assert that `best_val` epoch and export are unchanged. Ties go to the earliest epoch.
7. No VAL/TEST image enters S1 (the CineMA holdout must be a fixed TRAIN subset), S2, S3, or the arms. U never loads a mask. `FuseData.load(labeled=False)` already rejects L frames.

---

## 3. First look at worker drafts (snapshot about 02:44, may already have changed)

### 3.1 `teacher_mednext.py`
- **BLOCKER (§1.1):** it wraps `_load_mednext` → isotropic upstream L. With the real factory, `derive_mednext_feature_phases` will raise on the steps check (good fail-closed behaviour), but that means the real anisotropic teacher **cannot be built**.
- **HIGH — wrong phase chain module:** `_first_conv` (`:41-50`) prefers `module.conv3`. In upstream `MedNeXtDownBlock`, `conv3` is the 1×1 stride-1 **compression** conv, and the strided sampler is `conv1` (depthwise k=3, stride 2), with a parallel `res_conv` (1×1, stride 2). The derived steps would be (1,1,1) at every level, and the channel check also reads `conv3.out_channels`. **Fix:** chain `stem` and `down_i.conv1`, and validate `down_i.res_conv` separately.
- **MEDIUM:** `_pad` (`:180-184`) pads D up to a multiple of 16, with a minimum of 16. With the locked strides the z stride product is ×2, not ×4 (correction accepted by root; my preliminary text said 4). Over-padding D (for example 9→16) adds mostly replicate slices to the z-stride-2 levels. This is correct for phase (high-side) but wasteful, so use per-axis multiples from the stride product `(2,16,16)`.
- **MISSING:** FP dropout3d on all five skips needs access to the decoder. `forward_with_features` (`a3d/teacher.py:283-335`) is an opaque fixed sequence, so FP is not implementable through it. A PG decoder-owning forward with an optional `skip_dropout` is needed. Teacher input normalisation (z-score or [0,1]) is not stated, and it must be.
- OK: it keeps `boundary_logits` as the documented compatibility head with the exact Q3D key set (`:218-224`). It crops outputs to native. `freeze()` clears grads and makes `train()` sticky-false.

### 3.2 `engine.py` — STUB relative to the spec
Concrete deviations in `run_training` (`engine.py:263-376`):
- S2 (`:301-309`) is a supervised-only loop over `data["labeled"][update % n]`. It has no UniMatch, no threshold, no SDF consistency, no warmup or ramp, no per-epoch eval, no VAL selection, and no LR schedule. `weight_decay=1e-4` is hard-coded.
- Arms (`:339-361`): the student is called **without** teacher features on L, so there is no fusion path. There are no augmentations, no two-U sampler, no CutMix, and no FP. PG uses hard-argmax teacher CE (`_supervised(..., pseudo)`) instead of the fusion-weak weighted CE+Dice. `lambda_u` is absent. The CBKD call silently becomes a no-op if `cbkd_loss` is missing (`:355-357`), which is a hidden scientific fallback.
- Hidden defaults: `.get("lr",1e-3)`, `.get("lambda_cb",0.5)`, `.get("updates",8000)`, `.get("updates_per_epoch",80)`, and `seed .get(...,42)`. A real run must fail closed on a missing key.
- S1 and S3 return `NOT_RUN` when their module is missing, rather than failing for real configs (`:286-291, 312-317`).
- Logging: only VAL rows (no TRAIN16/TEST50 rows, no fusion path, no fixed column order). TEST goes in a non-schema `test_fg_dice` column. `loss_pg = loss - .5·sup` mislabels CBKD as PG. `best_val = (val == best)` marks a later **tie** as best too, which violates earliest-tie (`:366-369`).
- Resume: the checkpoint has no sampler, augmentation, threshold, or NumPy/CUDA-generator state beyond `_rng_state()`. Resume restores the RNG **after** load but never truncates metrics rows. Checkpoints happen every 50 updates rather than at epoch boundaries, which allows duplicate rows on resume.
- Data: `_load_data` (`:192-203`) eagerly loads **every** TRAIN U frame (about 2k cine frames) into RAM. Memory risk on the server; use lazy `FuseData.load` per key.
- Diag: only loss. Missing tau_c, agreement, gate means, coverage, perplexity, memory.

### 3.3 `evaluation.py`
Not yet reviewed in detail (still being written). I will cover it in the integrated pass against §1.14/§1.15 and §2.6.

---

## 4. Integrated-review checklist (for the next pass)
Required local checks 1–12 from REQUEST.md, plus the items above: §1.1 phases on the live anisotropic modules, §1.2 student grads, §1.3 CineMA affine and valid, §1.4 arm RNG parity, §1.5 valid replay, §1.6 SDF band exact, §1.9 threshold resume bit-identical, §1.14 strict export key set plus subprocess logit parity, §2 gradient/teacher-isolation tests, config files differing only in `pg.enabled`/`cbkd.enabled`, the fixed CSV/JSONL column order with `best_val` last, and one row per (epoch, split, path).

---

## 5. Integrated pass (started 2026-11-16, about 02:50)

All findings below were sent to root by orchestration status, in batches 1–5. SHA = first 12 hex characters of the file's sha256 when it was reviewed. Workers are still rewriting, so every item is **OPEN until the owner reports a fix and I re-review the new SHA.**

Resolved common policies from root, which I am now checking the code against: weighted Dice uses FG {1,2,3}. Voxel weights use the threshold from before the update, and then one combined A/B update follows. λ_u = 0 for u<800, else `min(1,(u−800+1)/ceil(.1·8000))`. λ_cb is a step to 0.5 at u=800. The student uses `_schedule_lr` with warmup .05. S1, S2, and S3 use a constant LR. S1 batch is 1; S2, S3, and the arms use L1+U2. Runtime is deterministic with TF32 off, bf16 on GPU and fp32 on CPU, and grad clip 1. S2 uses CutMix with the same box for SDF targets, confidence, and packet. The FP target is the unmixed A packet. Public SDF has 3 channels, plus an internal 1-channel background head and a 4-channel `sdf_all` used only inside the loss adapter. The teacher z stride product is 2.

### 5.1 Models
| ID | Sev | File@SHA | Finding | Fix / test |
|---|---|---|---|---|
| M1 | BLOCKER | teacher_mednext@574aa1bb | Real build goes through `_load_mednext`, i.e. upstream `create_mednext_v1(model_id="L")`, which is isotropic (stride-2 `conv1`/`res_conv`, `ConvTranspose3d` stride 2 + `F.pad(1,0,1,0,1,0)` on all axes). The locked anisotropic strides cannot be produced. The test fixture `_NativeMedNeXtFixture` is hand-written, so tests pass while the real path cannot. | Build PG-owned per-axis down/up blocks around the upstream block internals. Pad on upsampling only along strided axes. Test shapes on the real build at D=9 with H, W not multiples of 16. |
| M2 | HIGH | teacher_mednext@574aa1bb (rewritten at 20b59fbd) | `_first_conv` preferred `conv3` (1×1 stride-1) over the strided `conv1`, so the parallel `res_conv` phase was never checked. | Re-check on 20b59fbd: it now chooses conv1/conv3 by stride. Still require that the `res_conv` phase equals the main-path phase. |
| M3 | HIGH | teacher_mednext@574aa1bb | The SDF adapter (REQUEST:47) was not implemented: one 4-channel `sdf`. | Public `sdf` 3 FG channels, internal 1-channel BG head, `sdf_all` only inside the loss adapter. Band and U-consistency use FG `sdf` only. |
| M4 | HIGH | teacher_mednext@574aa1bb | No FP skip-dropout path, because the a3d `forward_with_features` is an opaque fixed sequence. | Decoder-owning forward with `skip_dropout`, applied to all five skips. |
| M5 | HIGH | student_pg@d94c1579 | `code_branch` runs on every training forward, including the teacher-fused L pass and the s1/s2/FP passes. | Explicit opt-in, only on the student-only clean-weak-U-A pass and the student-only L pass. |
| M6 | MED | student_pg@d94c1579 | FP `F.dropout3d` uses the global torch RNG, so global RNG use diverges across arms. | Bernoulli channel mask from a checkpointed generator. |

### 5.2 Views, reliability, losses
| ID | Sev | File@SHA | Finding | Fix / test |
|---|---|---|---|---|
| V1 | HIGH | views@09f255f4 | `weak()` warps only image and valid (valid is passed as `target`). A labeled target cannot share the draw. Dead `pass` at :306-307. | `weak(image,target,valid)`: replay the saved generator state for valid, then restore the post-call state. Test alignment and that the generator advanced exactly once. |
| V2 | MED | views@09f255f4 | The CutMix box must lie in the intersection of A-valid and cropped-B-valid. `mix_packet` silently uses itself as partner when none is given. | Engine passes the combined mask. A partner is required. |
| R1 | HIGH | reliability@77fbd438 | `voxel_weights` takes y and confidence from `pT_region`. Spec: y = argmax pW (fusion weak), confidence = max pW, agreement = sqrt(pT[y]·pS[y]). | Three-probability API. Test with argmax pW ≠ argmax pT. |
| L1 | MED→policy | losses@c433b140 | `weighted_dice` includes background. | Root policy is FG {1,2,3}, so a fix is required. |

### 5.3 Codebook / CBKD
| ID | Sev | File@SHA | Finding | Fix / test |
|---|---|---|---|---|
| K1 | BLOCKER | codebook@6287d19f | `train_codebook` was a NOT_RUN stub, and the engine fallback trained on `z²`. | Real S3: aligned teacher level-3 → student stage-3 grid; pooled GT CE+Dice plus confidence-weighted U pseudo; VQ + .25 commit + .01 entropy; restart every 500; freeze; receipts. |
| K2 | HIGH | codebook@6287d19f | qtask labels are resized by nearest interpolation (cell corner, misaligned) instead of pooled. | `pool_native_cells(one_hot, native, student_phase, valid)` soft target. |
| K3 | HIGH | codebook@6287d19f | `cbkd_loss` returns a silent zero when packets are missing. Commit 0.25 is hard-coded. L-assign is skipped when `labeled_teacher` is None, and the engine never passes it. | Raise on missing input. Weights from config. Include L assign and commit. |
| K4 | MED | codebook@6287d19f | `_as_options` and `_option(default=...)`: config keys `size`/`dim` are never read, and the `cbkd` section is shadowed by `codebook`. | Read exact keys with no defaults. |
| K5 | MED | codebook@6287d19f | The `assign_kl` channel-first heuristic misreads channel-last `[B,D,H,W,K]`. | Make channel-first explicit. |
| K6 | LOW | codebook@6287d19f | "Dead" means usage EMA ≤ 0 with a hidden EMA of .99, so a code used once is never dead. | Define dead as zero use in the last 500 updates. |

### 5.4 CineMA S1
| ID | Sev | File@SHA | Finding | Fix / test |
|---|---|---|---|---|
| C1 | HIGH (**confirmed by execution**) | cinema_init@6f426748 | Crop translation sign error. For 16×240×240 at spacing (10,1,1) with crop start (0,24,24), native (0,24,24) maps to processed (0,48,48), where it should map to (0,0,0). | Translation `+s/scale`. Exact affine test. |
| C2 | MED | cinema_init@6f426748 | align_corners=False resample: the half-voxel term is missing from the affine. | `r=(n+.5)·scale−.5`. |
| C3 | HIGH | cinema_init@6f426748 | Back-projection uses block-centred cells on the padded grid, but the student phase is offset 0 with step 8/16. Internal-pad cells are not excluded from valid. | Use `student.phases[3/4]`; valid = in-crop AND centre < native. |
| C4 | HIGH (PLAUSIBLE) | cinema_init@6f426748 | Token reshape order is (D,H,W), but the official input is HWD-patchified. The 1/8 conv tap is chosen by list position. | Verify against official source with a coordinate-coded test. Select the tap by derived stride; if it is absent, log that and use tokens only. |
| C5 | HIGH | cinema_init@6f426748 | Epoch rows and checkpoints are checked only at the end of each data pass, so most epochs have no row. | Move the check into the per-update loop. |
| C6 | HIGH | engine/cinema | The engine calls the non-existent `run_init`, so S1 silently records NOT_RUN. `student_init.pt` is never loaded into the arms. | Integrate and verify key subset and SHA. |
| C7 | MED | cinema_init@6f426748 | Eager materialisation of every TRAIN canvas. No RNG/resume state. The lr/wd/updates values come from hidden defaults. Permissive fallbacks (`_call_factory`, try/except `pass`). | Lazy preparation, explicit keys, single real path. |

### 5.5 Engine / entrypoints (engine@37e2215a → 84dcbd7e: STUB)
| ID | Sev | Finding |
|---|---|---|
| E1 | BLOCKER | S2: the "strong" input is a **different U frame** paired with the L weak target (`_teacher_unimatch_loss(teacher, image, uimage, ...)`). Fixed threshold. No views, CutMix, FP, SDF consistency, or `q3d_teacher_loss`. No per-epoch eval or VAL selection. |
| E2 | BLOCKER | Arms: keep is all ones (no dropout schedule). No views, CutMix, or FP. The PG target is teacher argmax instead of fusion weak. The threshold is updated with teacher probabilities and an all-ones valid. Weights use p_student from the **strong** logits. No FP .5 term, λ_u, or warmup. CBKD is inert. The supervised weight is .25 (0.5 applied twice). |
| E3 | HIGH | Optimizers ignore the config (`teacher.lr`/`training.lr` vs `.optimizer.lr`). Hard-coded wd. No `_schedule_lr`, clip, or AMP. Switches come from an internal table, not the resolved config. `validate_config` is never called. |
| E4 | HIGH | `_evaluate` computes FG/BG voxel accuracy, not patient-macro metrics. TRAIN16 and the fusion/teacher/qT paths are never evaluated. No CSV, patient files, or best snapshot. The earliest tie is broken. TEST sits in a non-schema column. |
| E5 | HIGH | Checkpoints every 50 updates, not at epochs. Missing view/keep/FP generator state. Metrics are not truncated, so rows duplicate on resume. `weights_only=False`. |
| E6 | HIGH | All U/VAL/TEST frames are loaded eagerly into RAM. |
| E7 | HIGH | `evaluate.py` PG path scores prediction records embedded in the sidecar instead of the model, which is a fabrication route. The data SHA check fails open. `train.py` PG uses an identity validator, an ImportError fallback, a cwd-relative base, and runs all arms when `--pg-arm` is omitted. |

### 5.6 Export / evaluation / logging / analysis
| ID | Sev | File@SHA | Finding |
|---|---|---|---|
| X1 | HIGH | export@7a674d02 | The cold-load factory defaults to `PGStudent`, which always builds `fusions`, so a strict load of the stripped state fails. A fusion-free deployment class is required. |
| X2 | HIGH | export@7a674d02 | The `strip_export_keys` assert is vacuous. It must assert the key set equals the student-only architecture. |
| X3 | MED | export@7a674d02 | Permissive multi-signature `_instantiate`/`_call_model`. The TEST transaction's manifest and receipt SHAs are optional. |
| A1 | MED | analysis@3c17dec3 | `compare_logged_export` passes when fields are missing. `paired_comparisons` swallows errors and has no HD95/ASSD. The code shuffle includes pad cells. |
| G1 | MED | logging@03c7a9da, evaluation@2fc5aef1 | Three selection implementations and two epoch-CSV writers. EpochLogger resume keeps rows written after the checkpoint, so resume dies on a duplicate. The key omits stage/arm. |
| OK | — | — | Patient-macro metrics with ASSD and absent-class None, VAL-only earliest-tie selection, fixed columns with `best_val` last, the PGSampler derangement (A≠B, one use per epoch), config arm files limited to the two switches, and the all-4 TEST ledger refusing reruns. |

### 5.7 Missing required tests (REQUEST "Required local checks")
#3 gradients reaching gates and targets, #4 teacher isolation, #5 threshold resume inside the engine, #8 resolved-config diff from the engine, #9 cold load of a real PGStudent, #10 phases on the real anisotropic build, #12 a TEST perturbation leaving selection unchanged, A1/A4 RNG parity, and a full synthetic S1→A4 flow with resume. At the time of review: `pytest tests/test_candidate_pg_*.py` gave 36 passed. I did not run the full suite or `check_repo`.

---

## 6. Disposition table (rolling; last updated during the integrated pass)

FIXED means the change was verified at the SHA shown, by code reading and, where noted, by execution. OPEN means not yet fixed or not yet re-reviewed.

| ID | Disposition | Evidence (SHA prefix) |
|---|---|---|
| M1 anisotropic MedNeXt | FIXED (code reading; the real official build cannot run locally because no MedNeXt checkout exists) | teacher_mednext@9ed83b3d: `_PGAnisotropicMedNeXt` copies the official modules, sets per-axis `conv1`/`res_conv` stride, and pads upsampling only on strided axes. Upstream Block/Down/Up forward (GRN, resample_do_res) traced and equal. |
| M1b `_apply` shadowing `nn.Module._apply` (found by root) | Reported fixed by owner; recheck pending | teacher_mednext@4eca9e03 |
| M2 phase chain | FIXED | strided `conv1` is used and the `res_conv` branch phase is checked |
| M3 SDF adapter | FIXED | public `sdf` 3 FG channels, BG head, `sdf_all` only through `as_q3d_outputs` / `_q3d_teacher_loss` |
| M4 teacher FP skip dropout | FIXED | decoder-owning `skip_dropout` on 5 skips with a generator |
| M5 code branch gating | FIXED | student_pg@df6e085d: `return_code` opt-in, valid from `_valid_centers` |
| M6 FP RNG | PARTIAL | generator path exists. Falls back to the global RNG when the generator is None, so the engine must always pass it (engine recheck) |
| V1 valid/target replay | FIXED | views@fb5c9152: save, valid call, restore, target call; one advance |
| V2 CutMix intersection / partner | FIXED | partner required; `partner_valid` passed |
| R1 voxel weights pW/pT/pS | FIXED | reliability@ac83d0ae |
| L1 FG-only weighted Dice | FIXED | losses@c506ba1e |
| K1 real S3 | FIXED (loop present) | codebook@cdb2346b `run_codebook` |
| K2 pooled GT | FIXED for S3 | `_pooled_gt` via `pool_native_cells` |
| K3 `cbkd_loss` silent zero / hard 0.25 / L-assign | OPEN at b670ab77; recheck | |
| Student L_Qtask softCE-only (spec 4.7) | OPEN; owner told to revert CE+Dice | codebook@b670ab77 `loss_qtask` |
| S3 resume ordering (checkpoint before log/commit) | OPEN | codebook@cdb2346b |
| S3 epoch rows FG-only | OPEN (MED) | |
| C1 crop sign | FIXED (executed: (0,24,24)→(0,0,0)) | cinema_init@cae0a334 |
| C2 half voxel | FIXED (executed: z 5→10 mm gives 0.5n−0.25) | cinema_init@cae0a334 |
| C3 student-phase lattice | FIXED | target stride/offset come from `student.phases` |
| C4 token order | FIXED | reshape H,W,D then permute |
| C5 per-epoch S1 rows | FIXED | check moved inside the update loop |
| C8 (new) source cell-centre half-cell / half-slice shift | Reported fixed by owner (11 tests); recheck pending at 62a78138 | Found by execution at cae0a334: native z=1 sampled CineMA z=0.5 |
| X1/X2 export deployment class / key set | FIXED | export@898c1648 FuseStudent(include_fusion=False) strict load; real-PGStudent cold-load test reported |
| X3 TEST transaction SHAs | FIXED | manifest path+SHA and receipt SHAs required |
| A1 shuffle valid / compare missing | PARTIAL | shuffle restricted to valid cells. `paired_comparisons` still swallows errors |
| G1 logger resume | FIXED in logger (commit/recover API). Callers: S2 does not call `commit_epoch` (OPEN); S3 ordering (OPEN) | logging@fac13475 |
| S2 B crop/pad, FP kwargs, epoch means, SDF weighting, AMP/determinism, near_mm | FIXED | teacher_training@7c8a617f |
| S2 `commit_epoch` | OPEN | |
| Config exact reference | FIXED (verified a recursive subset against `/private/tmp/pg-request-base-exact.json`; stream weights .25/.25/.5; arm files `pg_k8_A*.json`) | config@b9d831e8 |
| Config switch-lock (found by root) | Reported fixed by owner | |
| train.py dispatch on missing `method` key | OPEN (HIGH, new) | train.py:260/293 vs base config |
| Engine E1–E7 | OPEN; rewrite in progress | |
| Missing engine-level required tests (#3, #4, #5, #8, #9, #10, #12, parity, full flow) | OPEN | |

**Correction to my preliminary text:** the stream weights are .25/.25/.5. I quoted the prose shorthand "s1.25" without flagging it as ambiguous. The exact user JSON is authoritative.

### 6.1 Integrated-pass additions (training_step@82d8b81e, engine@5f44c6d9, finalization@b46abbca, teacher_training@7c8a617f, codebook@cdb2346b)

| ID | Sev | Finding | Status |
|---|---|---|---|
| M1 real build | — | Root ran the official MedNeXt (pin 0b78ed86) through `build_teacher` on CPU: forward/backward on native 5×17×19 passed, 61,907,084 parameters, all 5 phases correct, FP generator used (`/private/tmp/pg-official-mednext-cpu.json`). | FIXED, executed (random init, CPU, software-only) |
| C8 | HIGH | CineMA source cell-centre half-slice shift | FIXED, executed at cinema_init@62a78138 (z gives [0,1,2,3,4]) |
| TS1 | HIGH | `training_step` passes `packet_b` (pW/y/w/valid) to `mix_packet` on B's native canvas. Real different-shape pairs raise. | OPEN |
| TS2 | HIGH | The student threshold is updated with teacher probabilities, while the weights compare pW confidence. It must observe pW A∪B. | OPEN |
| TS3 | HIGH | Permissive view defaults (strong→weak, strong2→strong1, valid→ones, weak→raw) can silently collapse the two strong views. | OPEN |
| EN1 | HIGH | Arms resume: checkpoint every 50 updates, commit at each epoch, no `recover_from_checkpoint`, so an epoch row duplicates after a crash. | OPEN |
| EN2 | HIGH | S1 trains CineMA distillation on the 16 L frames only. Spec: all TRAIN images. | OPEN |
| EN3 | MED | Arms: no clip 1.0, no bf16 autocast, threshold ema/clip from defaults, discarded VAL/TEST evals, last-update loss columns. | OPEN |
| EN4 | MED | Arm provenance: exports and identity use the base config, so A1's export says pg/cbkd enabled. The engine-loop export uses the last epoch. | OPEN |
| EN5 | HIGH | `train.py` dispatches on `config.method`, which the base and arm configs lack. | OPEN |
| S2c | HIGH | S2 never calls `logger.commit_epoch`, so an epoch row duplicates after a crash. | OPEN |
| S3r | HIGH | S3 saves the checkpoint before logging and commits after, so an epoch row is lost after a crash. | OPEN |
| CB-res | MED | `cbkd_loss` reducer: L-assign skipped when `labeled_teacher` is None, hard-coded 0.25. Production uses `training_step._cbkd_loss` with config weights, so the reducer is legacy. | OPEN (LOW if the reducer is removed) |
| F1 | — | The TEST transaction evaluator now consumes models loaded from the ledger artifacts. | FIXED (export.py:345) |

---

## 7. Final integrated disposition (current SHAs)

Reviewed SHA prefixes: analysis d0e4056b, cinema_init 75f20fdb, codebook 17b04d7c, config b9d831e8, data 6d33a90c, engine 01715494, evaluation 5f00e9c6, export 2a8ad1e0, finalization 99deaaa9, logging 821ceec4, losses 5dbef4dc, reliability 987eaada, student_pg df6e085d, teacher_mednext 4eca9e03, teacher_training fb7d7d58, training_step 2c30147b, views 72898dd5, train.py 374e0ec4, evaluate.py cc47239a, configs base 1cefe5b5 / A1 781def61 / A2 639ae079 / A3 a622c2ef / A4 6abb6173.

My local evidence: `pytest tests/test_candidate_pg_*.py` gave 91 passed (CPU, synthetic or fixture only). Root owns the full-repo pytest, `check_repo`, and the Q3D smoke. Root's official MedNeXt CPU build receipt is `/private/tmp/pg-official-mednext-cpu.json`. Nothing here is GPU, patient, or efficacy evidence. The real S1/S2/S3/A1–A4 runs remain NOT_RUN.

### 7.1 Required-check coverage (REQUEST "Required local checks")
| # | Check | Code status |
|---|---|---|
| 1 | CutMix 16-px alignment, full depth, same packet, pad valid 0 | Verified in views/training_step. The unaligned fallback was removed (root). Cropping B and its packet to A is fixed. |
| 2 | Strong views keep geometry and range | `strong()` is intensity-only on the weak canvas, clamped to [0,1]. Two independent views are required. |
| 3 | Targets get zero gradient; gates get no PG/CBKD gradient | Verified by reading: targets under no_grad and detached, fusion only in the labeled graph. Root reports engine tests. |
| 4 | No teacher features in learning streams | s1/s2/FP/clean/code paths call the student without teacher features. |
| 5 | Threshold resume is bit-identical | Buffers are checkpointed. Arm and S2 resume use committed-epoch recovery. |
| 6 | Normalized SDF band | FG `sdf` (tanh) band `<2.5/5`. SDF targets come from the Q3D voxel-face convention. |
| 7 | Code ordering, STE, frozen C | Student C is a buffer copy; `rho` keeps order; STE identity. Tests exist. |
| 8 | Configs differ only by switches | Arm files hold only the switches. The base is a recursive superset of the exact user JSON (I verified this). Per-arm identity hash. |
| 9 | Clean export, subprocess load, logit parity | FuseStudent(include_fusion=False) strict load, bitwise in-process and fresh-process checks, real-PGStudent test (reported). |
| 10 | Actual MedNeXt phases | Anisotropic wrapper. Root executed the official build: all 5 phases correct. |
| 11 | Unchanged FUSE/Q3D suites | Root's full-suite evidence is pending at time of writing. |
| 12 | Fixed epoch schema, uniqueness, VAL-only best unaffected by TEST | EpochLogger fixed columns with `best_val` last, `(epoch,split,path)` unique per stage/arm directory, earliest-tie VAL selection. Arms use the student_only/fusion paths with lowercase splits (root policy). |

### 7.2 Finding dispositions
FIXED (verified at the SHA above, or at the earlier SHA noted in §6/§6.1): M1, M1b (root), M2, M3, M4, M5, V1, V2, R1, L1, K1, K2, K3, K4 (exact config keys for the production paths), K5 (explicit `channel_first` in callers), K6, C1, C2, C3, C4, C5, C6, C8, X1, X2, X3, A1 (whole-vector valid-cell shuffle, root), G1, TS1, TS2, TS3, S2 (B crop, FP weak-A (root), hard-argmax CE (root), epoch means, SDF weights, AMP, CUBLAS, commit order, constructor-order resume), S3r, EN1 (constructor promotion and recovery), EN2 (S1 gets all TRAIN images), EN3 (clip/AMP/threshold config), EN4 (per-arm identity and export switches), EN5 (schema dispatch), F1, E1–E7 (engine rewritten; the stub is gone), the CBKD student qtask is softCE only, and evaluate.py runs the model on the hash-pinned manifest.

ACCEPTED, fix assigned by root at time of writing (re-verify in root's final evidence):
- **ST1 (MED)** single-arm CLI with stage `all` would re-run shared S1–S3. Root will default to `arms` and reject `all`.

OPEN at time of writing:
- **D1 (HIGH, real GPU only)** strict `torch.use_deterministic_algorithms(True)` in engine.py:484 and teacher_training.py:558. FUSE uses `warn_only` on CUDA (fuse/engine.py:739). Trilinear-interpolate and grid_sample backward on CUDA have no deterministic kernel, so the first GPU backward would raise. CPU tests cannot detect this. Fix: reuse the FUSE helper and record the flags.

Residual LOW/MED (not blockers; listed for transparency):
- The engine materialises all ~1.9k TRAIN frames for S1 (RAM).
- S1 `cosine_train` evaluates every TRAIN frame every epoch (cost).
- The legacy flat-prefix `u_views` path still defaults strong→weak, though the engine does not use it.
- Fixture adapters (`_SequenceTripleSampler`, `_call_factory`, `_TinyTeacher`) remain in production modules.
- `torch.load(weights_only=False)` for resume checkpoints.
- S2 crops B to A's canvas *before* B's weak draw (an accepted interpretation; record it in the README).
- The S3 U averaging weight (L + .5·(U1+U2)) and the full-depth CutMix interpretation when B has fewer z slices should be documented as ancillary resolutions.

### 7.3 Reviewer misses corrected by root (recorded for honesty)
`_apply` shadowing `nn.Module._apply`; the config switch lock; the S2 FP stream needing to be weak-A, not strong; UniMatch hard-argmax CE; the per-channel code shuffle; the EpochLogger constructor truncation order; the K6 double call; and stream weights .25/.25/.5 (I had repeated the prose ".25" shorthand without flagging it as ambiguous).

### 7.4 Closing update (latest bytes)
Late edits re-checked at: cinema_init b1b473ff, codebook 5cf5ec36, engine 9739b0af, export 3333be87, runtime c06e936a (new), teacher_training 4113da6e, train.py af1d37c3. All other modules are unchanged from §7.

- **D1 FIXED.** `runtime.configure_runtime` sets CUBLAS before CUDA, calls `use_deterministic_algorithms(True, warn_only=cuda)`, sets cuDNN deterministic with benchmark off and TF32 off, and returns flags for the receipts. engine, S1, S2, and S3 all use it. A grep finds no strict CUDA call left.
- **ST1 FIXED.** engine:461 rejects a single arm unless stage is `arms`. The CLI defaults a selected arm to `arms`.
- The export metadata `parameters_total` is now the stripped deploy count, with training parameters recorded separately. No math changed.
- Root evidence (not re-run by me): `check_repo` PASS (`/private/tmp/pg-check-repo-final.json`); Q3D smoke exit 0 with the synthetic gate override (`/private/tmp/pg-q3d-final.log`); full synthetic PG smoke ran S1/S2/S3/A1–A4, bitwise cold load, and one TEST transaction (`/private/tmp/pg-root-final-smoke.log`). The full-repo pytest (`/private/tmp/pg-final-pytest.log`, 04:18) gave 1081 passed, 4 skipped, and 2 failed. Both failures were the stale `test_pg_resume_*` tests that used single-arm stage `all`, which was made invalid by ST1. Root corrected them. Locally, at the latest bytes, `tests/test_candidate_pg_engine.py` gave 4 passed and `tests/test_candidate_pg_*.py` gave **94 passed**. Root still owns a final full-repo rerun.

**Reviewer verdict:** the code is complete; no stub stages remain. I have no open BLOCKER or HIGH findings. Every required local check from §7.1 has a code path and tests (CPU, synthetic, fixture-only). The residual LOW/MED items are listed in §7.2. The real S1/S2/S3/A1–A4 runs are NOT_RUN. This review makes no efficacy, GPU, or patient-level claim.

---

## 8. Narrow-fix revision review (2026-11-16, tree `0798b46` + uncommitted)

**Scope:** only the user's narrow fixes B1, L1, L2, L3a, L3b and R1, plus the root-owned logging helper, arm start/end/summary lines, CLI stdout removal, and the bounded S1 `max_updates` used for the 2-update GPU profile. I did not run a server, GPU, or training job, and I wrote no source or test file. Nothing in this section is evidence of efficacy, GPU behaviour, or patient-level results. The real S1/S2/S3/A1–A4 runs remain **NOT_RUN**.

**Reviewed SHA-256 prefixes (latest bytes):** codebook `633b9b7bf31a`, export `7103394f420d`, engine `39101203238f`, teacher_training `8ea2653deae4`, logging `7b1089bee904`, cinema_init `694a55ffdfd5`, train.py `fa29ce3adff7`, scripts/pg_campaign.py `dc7149a2d110`, scripts/smoke_pg.py `8c55e8e0bc7f`, tests/test_candidate_pg_codebook.py `269cc245189b`. training_step, views, student_pg and the configs are unchanged in this revision.

### 8.1 Findings sent to root during this pass

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| N-B1t | test bug | `test_cold_load_child_receives_parent_runtime_flags` failed: it called `set_float32_matmul_precision("high")` after `cuda.matmul.allow_tf32=False`, and the two are coupled, so TF32 was switched back on. | FIXED (owner). The export tests pass locally. |
| N-L3r | HIGH | A regression in L3a: `run_codebook` raised `FileNotFoundError` whenever `resume=True` and no S3 checkpoint existed. The engine passes `resume` through, so a campaign that crashed in S2 could not be resumed. | FIXED (root). codebook:1469 now loads from the checkpoint only when `resume and checkpoint.is_file()`, and seeds otherwise. Root reports a direct resume-before-checkpoint test and an initializer spy that shows a restored C is never reseeded. |
| N-L3m | MED | The first L3a version ran the teacher over every TRAIN L+U frame and kept every valid vector on the CPU, which would take several GB. | FIXED (root). Frames are visited in a seeded random order, and the loop stops once at least K valid vectors have been collected. K are then sampled without replacement. `frames_visited` is recorded. |
| N-L3n | LOW | The `nullcontext` fallback in `_autocast_disabled` could silently skip the fp32 guard. | FIXED (root). It is now plain `torch.autocast(device_type, enabled=False)`. |
| N-R1t | LOW–MED | The S3 console row timed only the last update (`step_started`). S3 also had no start/end lines, so the smoke CLI stdout test failed. | FIXED (owner). There is an `epoch_started` timer (codebook:1552/1692/1699), plus `[S3] start` (:1551) and `[S3] end` (:1765). |
| N-MU | MED | Bounded `max_updates` receipts say `COMPLETED`, and stage-only or resume imports did not check the update count. A 2-update profile directory could therefore be reused as a full run. | FIXED (root) by locking identity rather than adding new statuses. engine:468-472 refuses an out dir whose `EXPERIMENT_LOCK.requested_max_updates` differs. `requested_max_updates` and `profile_only` go into the identity, the stage receipts and RESULT. Root reports an early-rejection test. |

### 8.2 Disposition table — B1 / L1 / L2 / L3a / L3b / R1

| Fix | Disposition | Evidence |
|---|---|---|
| **B1** cold-load runtime parity + tolerance | **FIXED** (code reading + local tests) | export `_cold_load_runtime` captures threads, deterministic flags, `warn_only`, cuDNN deterministic/benchmark/TF32, matmul TF32 and precision, and `CUBLAS_WORKSPACE_CONFIG`. `_cold_worker` restores them before the model is built, and the child's runtime must equal the parent's or the run fails as a configuration error. The result is PASS only when `max_abs_logit_delta <= 1e-4` **and** every argmax is identical (in-memory vs loaded vs fresh process). Otherwise it returns `status: "WARN"` with per-volume diagnostics and does not raise. Load, subprocess, shape and NaN/Inf failures still raise. finalization:498 stores the WARN receipt and does not raise. Tests cover a WARN above tolerance, a PASS below tolerance, a WARN on an argmax change within tolerance, and runtime propagation. |
| **L1** teacher labeled loss on strong-L | **FIXED** | teacher_training:672 feeds `lstrong` to `_q3d_teacher_loss`. The target and geometry stay the weak-aligned `ly_aug`/`lg_aug`; strong is intensity-only. `lstrong` was already drawn by `joint_views`, so the RNG stream is unchanged. Test: `test_teacher_stage_supervision_uses_labeled_strong_input`. |
| **L2** independent second CutMix box, same packet mix | **FIXED** | When `n_strong_views == 2`, a second `cutmix_box` is drawn from `cutmix_generator` on `av&bv`. `packet_s2 = mix_packet(pa,ya,wa,av,box_s2, …, sdf)` uses the same A/B packet as stream 1. `s2_img` uses `box_s2`, and `loss_s2` and `sdf_s2` use `packet_s2`/`confident_s2`. The threshold still updates from the unmixed A∪B valid voxels (spec). The student arms already drew one box per stream (engine:888, training_step:656). Test: `test_teacher_stage_strong_view2_uses_distinct_box_and_packet_alignment`. |
| **L3a** C seeded from 128 valid projected teacher z at a fresh S3 start, never on resume | **FIXED** after N-L3r and N-L3m | `_initialize_codebook_from_train` uses TRAIN L/U only (never VAL/TEST). It keeps the valid support from `_aligned_level3`, which excludes internal padding, and filters out non-finite vectors. Sampling is seeded and without replacement. A short pool raises. C is copied before AdamW is built. Seeding is skipped only when a real checkpoint is loaded. **Residual note (LOW, not a defect):** the stop-at-K rule means the 128 seeds usually come from a single randomly chosen frame, because one frame has thousands of valid stage-3 cells. That is a literal reading of the spec. If more diversity is wanted later, that is a separate user decision. |
| **L3b** rho / vq / assign_kl, TeacherCodebook and StudentCodeBranch in float32 with autocast disabled; AMP-parity tests | **FIXED** | `rho`, `vq_ste`, `assign_kl`, `TeacherCodebook._output` (P_T, distances, argmax, embedding, STE) and `StudentCodeBranch.forward` (P_S, distances, argmax, embedding, STE) all run inside `torch.autocast(enabled=False)` on `.float()` inputs. Gradients still reach P_T, P_S and the teacher C through the differentiable casts, and the student C stays a frozen buffer. The `H_*_Q` heads stay outside the guard, which the fix did not require. Test: `test_code_assignments_are_identical_under_cpu_autocast_for_teacher_and_branch`. **Residual note (LOW):** that test uses small random features. It would probably also pass on the old bf16 code, so it guards against regressions but is weak at proving the fix. A near-tie fixture would make it decisive. |
| **R1** one epoch summary row per epoch for S2/S3/A1–A4, one start/end per stage, no other stdout | **FIXED** (code reading; root owns the final CLI test receipt) | `logging.epoch_console_line` emits `[X] ep e/E \| loss x \| train FG x \| val FG x (best x@e) \| test FG x \| Ns`. S2 (teacher_training:604/734/781), S3 (codebook:1551/1693/1765) and arms (engine:752/1020/1106) each print one start line, one row per completed epoch, and one end line. The `[PG]` CLI prints in train.py, pg_campaign.py and smoke_pg.py are removed. Synthetic S2/S3 no longer redirect stdout. By root scope decision, S1 keeps its own cosine row and start/end lines, and synthetic S1 output stays suppressed (outside R1). S2's partial-epoch evaluation, added by its owner, was reverted by root, so the 80-update selection boundaries are unchanged. |
| Root: bounded S1 `max_updates` | **FIXED** | cinema_init:794-795 applies `min(updates, max_updates)`. The engine passes the bound to both the real and the synthetic S1, and N-MU stops profile artifacts from being reused. |

### 8.3 Open / residual items carried honestly

- **OPEN-B1f (MED, scoped out by root):** `finalization._compare_all12` (atol 1e-7) still raises if drift changes the metrics of the selected VAL/TEST export. A cold-load WARN that comes with an argmax change will therefore still stop finalization at the separate metric-parity check. Root chose to keep that contract, because B1 was scoped to `verify_cold_load` only.
- **LOW:** `num_interop_threads` is not part of the cold-load runtime. This was not requested.
- The §7.2 residual LOW/MED list is unchanged by this revision. This revision introduced no new BLOCKER or HIGH that is still open.
- **External/preflight (root, not code):** the server MedNeXt pin PASSES. `b1251ee…` is the CineMA HuggingFace **model** revision, not a GitHub source commit, and the official GitHub source has no such commit. The source pin is waiting on the user, and root is correcting CINEMA_SOURCE.md. The config manifests, root and mounted data that the real campaign needs are missing, and the GPU profile has not been run.

### 8.4 Local test receipts (CPU, synthetic or fixture only)

- At export, codebook and logging bytes earlier in this pass: `pytest tests/test_candidate_pg_*.py` gave 102 passed and 1 failed. The failure was the smoke CLI stdout test, while R1/S3 was still in progress, and it is since fixed by the owner.
- At the latest bytes above: `pytest tests/test_candidate_pg_codebook.py tests/test_candidate_pg_export.py tests/test_candidate_pg_logging.py` gave **26 passed**. One earlier run of the same trio failed once in `test_run_codebook_synthetic_loop_checkpoint_resume_and_receipts` at the same codebook SHA while root was still editing tests. Two later runs (the codebook file alone, and the trio again) passed, so I am recording it as transient; root's final suite is authoritative.
- Root owns the full PG suite, the full-repo pytest, `check_repo`, the Q3D smoke, and the Linux matrix with OMP unset and with OMP=2. Those receipts may be added after this review. I did not re-run the full suites, as root asked.

**Reviewer verdict (this revision):** every one of B1, L1, L2, L3a, L3b and R1 is FIXED at the SHAs above, by code reading and focused CPU tests. The only open code item is OPEN-B1f, a MED that root scoped out. No efficacy, GPU, or patient-level claim is made.

### 8.5 Coordinator verification addendum

The final implementation passes **107 PG tests** on macOS (Python 3.11.16,
PyTorch 2.14.0) and the two-update synthetic PG smoke (S1–S3, all four arms,
exports/finalization), with no stderr output. The smoke is software evidence only.
After the review, the coordinator strengthened L3b's fixture to a known near-tie:
`1.002` rounds to `1.0` in bf16, changing the nearest code without the guard.
Both teacher and student must retain code 1 and bit-identical rho under autocast;
all **14 codebook tests pass**. This closes the weak-test LOW note in §8.2.
The seed diversity note and OPEN-B1f remain unchanged.

Linux acceptance with **OMP_NUM_THREADS unset: 17 passed** (173.09 seconds).
Linux acceptance with **OMP_NUM_THREADS=2: 17 passed**. Exact logs, file hashes and the completed matrix
are recorded in `FIXES_RECEIPT.json`. This narrow revision reruns the user-requested
PG suite and smoke; the prior full-repository/Q3D receipts remain historical.
FUSE/Q3D implementation files and all configs have no changes in this revision.

A separate user-assigned server coordinator owns the fresh manifest preparation
and sole GPU launch. The preflight established that `b1251ee` belongs to the HF
model repository; official GitHub source is separately identified as `c10daa1`.
No GPU profile result is claimed here while that coordinated launch is pending.

### 8.6 Real profile follow-up (coordinator verification)

The requested `train.py --pg-stage all --pg-max-updates 2 --device cuda`
was run once on committed integration source `e2fbff9`, with unchanged settings.
It completed S1/S2 and failed during S3 VAL evaluation with
`ValueError: unlabeled_keys are available only for TRAIN`. This is a **FAILED**
profile, not a full-pipeline PASS; its original receipt is preserved in
`GPU_PROFILE_ATTEMPT.json`. Peak allocated/reserved memory was
12,031,922,176 / 16,773,021,696 bytes, with no OOM. Arms were not reached.

**N-S3-EVAL: FIXED in software, GPU recheck pending.** `_qval_metrics` now
requests only labeled keys, so it never calls TRAIN-only U accessors on VAL/TEST.
The same evaluation pass now aggregates flat valid code assignments across
patients, since native patient grids have different shapes. Regression tests
exercise VAL18 and TEST50 accessors that reject U, two different native grids,
and an exact valid-cell count. This is a small profile-blocking bug fix;
FUSE and training settings remain unchanged.

After this follow-up, the full current PG suite passes **158 tests**, the
separate two-update synthetic full smoke completes, and all **16 codebook tests
pass on Linux** (OMP=2, CPU). The original Linux B1
acceptance matrix remains 17/17 with OMP unset and 17/17 with OMP=2.
This addendum is the coordinator's verification, not an additional Claude review.
The sole server owner has started full S1 under the separate user instruction;
GPU rechecking of this follow-up is deferred to its stage boundary, without
interrupting S1 or launching a concurrent GPU job. OPEN-B1f remains open.
