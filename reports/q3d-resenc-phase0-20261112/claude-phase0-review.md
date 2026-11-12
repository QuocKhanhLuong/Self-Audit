# Q3D teacher migration MedNeXt-L to nnFoundationCNN ResEnc-L: Phase 0 read-only audit

Date: 2026-11-12. Worker: Claude (Orca task task_5865471cba44, dispatch ctx_d2a2bed8ce28).
Scope: read-only source and protocol audit. Evidence class: SOFTWARE_ONLY (CPU forward of random input plus strict state-dict load). No patient data, no training, no GPU, no dataset download, no checkpoint download (I reused the root's `checkpoint_final.pth`).

## 0. Inputs and provenance checked

| Item | Value | Status |
| --- | --- | --- |
| Target checkout | `/Users/alvinluong/orca/workspaces/Self-Audit/q3d-resenc-20261112` @ `37851c71355a4c82b52c0de64533bba1615f0436` | Read only. I made no edits. `git status` shows `M self_audit/candidate_q3d/artifacts.py` (+35 lines, mtime 16:34). Another agent made that change, not this audit. |
| Spec | `docs/CANDIDATE_Q3D_SPEC.md` (all 435 lines) and `ACTIVE_PIPELINE.json` | Read in full |
| Code read | `candidate_q3d/{pretrained,teacher_features,alignment}.py`, `engine.py` (`_teacher_parameter_groups`, `_make_teacher_optimizer`, T-stage loop), `configs/candidate_q3d/v1.json` teacher block, frozen `candidate_a3d/teacher.py` (`DualBranchTeacher3D._pad`, `set_teacher_stage`), `prepare_acdc.py` normalization | Read |
| nnssl `architecture_registry.py` @ `3d774e942256bd70e5e264c6b94fa38e25f7ed74` | sha256 `3fbe0fc4…9610` | I re-fetched it from GitHub at the pin. It matches the root's copy byte for byte. |
| nnUNet `pretrainedTrainer.py` @ `c04f7ec8bbffaf3ebc3b26aa2c89042e8a8e368b` | sha256 `615b8ed8…a065` | I re-fetched it at the pin. It matches the root's copy byte for byte. |
| nnUNet `dynamicPretrainedTrainer.py`, `like_nnssl.py`, `documentation/finetuning_from_nnssl_checkpoints.md` @ same pin | dyn sha256 `b5d0f9b0…abea` | Fetched to scratch for context only |
| dynamic_network_architectures | Both pins require `>=0.4.4,<0.5`. Local version is `0.4.4`. | OK, but the real run must record it |
| Checkpoint | HF `MIC-DKFZ/nnFoundationCNN` rev `f443c093…`, sha256 `ac262d3e…2bbd`, 409,579,442 B, license CC-BY-SA-4.0 | The root downloaded it. I did not download it again. |

## 1. Executive verdict

ResEnc-L encoder weights load strictly (stem 8 keys + stages 440 keys, zero missing or unexpected). I verified this with both isotropic and in-plane-only strides. However, the current Q3D teacher stack is hard-wired to MedNeXt and fails closed in at least 7 places. Two of the gaps change the spec contract rather than just the code, and they need a reviewed spec amendment before implementation:

1. **Residual skip phase (BLOCKER, needs an [OPEN] contract like the decoder one).** The ResEnc first block of each downsampling stage is ResNet-D: main path `conv k3 s2 p1` (offset 0), parallel skip `AvgPool k2 s2 p0` followed by a 1x1 conv (offset +0.5 input step). The residual sum mixes lattices. At the stride-8 source, the possible branch offsets span 0 to 3.5 native px per strided axis. That is the same magnitude as the 4 px bug the spec just removed.
2. **The pretrained-region-head assumption breaks.** The nnssl checkpoint is MAE-pretrained. Its decoder ends in 1-channel reconstruction `seg_layers`, and the official trainer re-initializes the decoder from scratch. Spec §3 Init ("encoder and region branch from pretrained"), the T-a schedule (freeze encoder+region and train only the boundary branch), and the `region_head_pretrained: True` / class-mapping-must-equal-BG/RV/MYO/LV loader contract all become wrong.

## 2. Findings

### F1. Residual avgpool skip phase vs main conv (BLOCKER)

Source: DNA 0.4.4 `BasicBlockD`, at the version range both pins require. When `has_stride` is set, `skip = Sequential(AvgPool(stride, stride), ConvDropoutNormReLU(1x1, bias=False))`. The main path is `conv1` (k3, stride s, pad 1) then `conv2` (k3, s1, pad 1). Only `blocks[0]` of stages 1..5 has the pooled skip. Stage 0 and all later blocks use an identity skip.

Measured with the repo's own `alignment.grid_phase` on live DNA modules:

| Stage (cumulative stride) | Main-path offset | Skip-only chain offset |
| --- | --- | --- |
| stages.1 (2) | 0.0 | 0.5 |
| stages.2 (4) | 0.0 | 1.5 |
| stages.3 (8) = Q3D source | 0.0 | 3.5 |

With in-plane-only strides `(1,2,2)`, the z offset stays 0 and y/x show the same 0.5/1.5/3.5 offsets.

Consequences:
- If someone ports `_mednext_phase_chain` and checks `res_conv` phase against the main path, the check raises on purpose. MedNeXt `res_conv` was a k1 s2 conv with offset 0, so it matched. ResEnc's skip does not match, so that check fails closed. This is correct behavior. It must not be "fixed" by dropping the AvgPool from the check or by counting only the main path silently.
- The §4 network-phase probe (box kernels, tolerance 0.5 px) is ill-defined here. The measured centre of the summed output depends on the relative branch gains, anywhere between 0 and 3.5 px. The probe will either fail or pass for the wrong reason.
- Recommendation: add a ResEnc residual-phase [OPEN] item mirroring the decoder [OPEN] item in §4. Stop real-data alignment for ResEnc (`NOT_RUN`) until a reviewed contract picks one of these: (a) main-path convention plus a documented bias of at most 3.5 px; (b) an empirical phase measured from the loaded weights, e.g. the impulse-response centroid per stage, which is frozen after T because the weights change during T fine-tuning; (c) another reviewed choice. Do not self-select a phase.

### F2. Anisotropic stride adaptation (HIGH)

- The official `get_res_enc_l` is fixed isotropic: strides `[1,2,2,2,2,2]` on all axes, kernels 3³, 6 stages, pretraining patch 192³. Official `PretrainedTrainer` builds exactly this (`get_network_from_name("ResEncL")`) and never adapts strides.
- The official anisotropic route is `DynamicPretrainedTrainer` + `nnUNetv2_plan_like_dynamic`. It adapts kernel sizes by averaging and loads with **`strict=False`**. That conflicts with the Q3D strict-load requirement, so do not adopt it.
- Strict-compatible option, verified: change only `strides` (e.g. `[[1,1,1],[1,2,2],[1,2,2],[1,2,2],[2,2,2],[1,2,2]]`) and keep kernels 3³. Parameter shapes do not depend on stride, so stem+stages still load with `strict=True`, and the forward pass runs. Side effects:
  - The skip AvgPool kernel follows the stride and becomes `(1,2,2)`, so z phase is clean and F1 remains in-plane only.
  - Teacher z resolution at stride 8 would equal native D. This removes the §13 risk "z interpolation of 1–2 teacher z cells" but changes the receptive field and the pretrained features' z semantics. Pretraining used `noresample` data around 1 mm, while ACDC slices are about 5–10 mm.
  - This is a NON-OFFICIAL adaptation. Config and provenance must record `strides`, the choice reason, and `adaptation: "stride_only_non_official"`. The phase calculator must read live module strides, not constants.
- If isotropic strides are kept, D≈10 is replicate-padded to ≥32 (see F4), and the stride-8 source has ceil(10/8)=2 real z cells, which is worse than MedNeXt's situation.
- This is a design decision for the coordinator or user. Either way it needs a spec amendment (§2 z note, §4, §13).

### F3. Teacher-only z-score (HIGH)

- nnssl pretraining plan: `ZScoreNormalization`, `use_mask_for_norm=[false]`, `spacing_style: noresample`. The HF README says "Z-score normalised, no resampling". The Q3D pipeline feeds `[0,1]` produced by `prepare_acdc.py:171-172`: per-cine percentile clip at 0.5/99.5 then min-max.
- Required contract: z-score inside the teacher adapter only. Compute it per sample and per frame over **native voxels before high-side replicate padding**, `(x-mean)/max(std,eps)`. The student keeps `[0,1]`, so the deploy contract (§10) is unchanged. Z-scoring the `[0,1]` volume equals z-scoring raw intensity except that the percentile clip removes tails and stats are per-frame vs per-cine. Declare this as a known deviation from the pretraining distribution.
- Ordering: spatial and intensity augmentation run in `[0,1]` (loader), then the teacher z-scores. For U codeKD the teacher input is not intensity-augmented (§9). The same z-score path must be used in T, Q, P (physiology cache) and J. Never compute batch-level stats, and never include pad voxels.
- Current code has no normalization hook. `DualBranchTeacher3D` (frozen A-3D) passes the input straight to `_pad`, so z-score has to live in a Q3D-side adapter or subclass.

### F4. Padding: no minimum×2 (HIGH)

- Frozen `DualBranchTeacher3D._pad` sets `padding_multiple = 2**(len(channels)-1)`. ResEnc has 6 stages, giving 32 instead of MedNeXt's 16. `minimum = 2*multiple` whenever `_external_region_path`, which is true for any adapter exposing `forward_region` other than the tiny encoder. The result for ResEnc is a **minimum of 64 on every axis**: D≈10 becomes 64 (54 replicate slices), H/W at least 64.
- Remove the ×2 minimum for ResEnc and use a per-axis multiple equal to the product of that axis's strides (32 isotropic; z=2 for the anisotropic example above). Constraint to keep: InstanceNorm3d at the deepest stage needs more than one spatial element in training mode, which per-axis multiples satisfy because in-plane is ≥2×2. I observed `[1,320,1,2,2]` for a 32×64×64 input and `[1,320,8,2,2]` for anisotropic 16×64×64.
- `_pad` lives in frozen `candidate_a3d/teacher.py`. Override it in a Q3D subclass, and keep pad high-side-only replicate so the phase origin does not shift. The scalar `padding_multiple` cannot express per-axis multiples, so the subclass must replace `_pad`, not just set the attribute.

### F5. Strict stem/stages load and the `all_modules` aliases (MEDIUM, verified)

- Checkpoint top-level keys are `network_weights`, `citations` and `nnssl_adaptation_plan`. Key prefixes: `encoder.stem` 8, `encoder.stages` 440, `decoder.encoder` 448 (aliases of the encoder), `decoder.stages` 40, `decoder.transpconvs` 10, `decoder.seg_layers` 10 (shape `[1,C,1,1,1]`, MAE reconstruction).
- `all_modules.*` keys are aliases of `conv.*` / `norm.*` because `ConvDropoutNormReLU` registers both. The DNA version must register `all_modules` (0.4.4 does), otherwise strict load fails. The README warns not to deduplicate.
- I verified locally on DNA 0.4.4 with random input: `encoder.stages.load_state_dict(strict=True)` (440 keys) and `encoder.stem.load_state_dict(strict=True)` (8 keys) succeed for both stride variants. This matches the root's `PASS_STRICT_ENCODER` (448). The official trainer does the same (submodule `load_state_dict`, default strict). `stem_in_encoder` is False because `"encoder.stem"` is not itself a key, so the trainer loads stem and stages separately.
- Gaps in the current Q3D loader (`pretrained.py`):
  - `_extract_state` rejects the wrapper. There is no `state_dict`/`model_state_dict`, and the mapping holds non-tensor values, so the error is "must contain state_dict or model_state_dict tensors".
  - `_load_exact_mednext_state` requires full-model strict equality and `class_mapping == BG/RV/MYO/LV`, and forbids classifier reinit. The checkpoint has no classes and no usable seg head.
  - `architecture == "mednext_l_k3"` and `source_commit == UPSTREAM_MAIN_SHA` are hard checks.
- Needed contract: load `network_weights` keys prefixed `encoder.stem.` and `encoder.stages.` only. Load strictly into those two submodules. Assert the alias pairs are value-identical. Record decoder, seg head and boundary path as `reinitialized_keys`. Never feed `decoder.encoder.*` into the full model (shape mismatch at `seg_layers`, and it would double-count). Any SHA or inventory over `teacher.state_dict()` will see `decoder.encoder.*` duplicates, so dedupe by `data_ptr` or hash only named parameters.
- Provenance fields are not embedded in the checkpoint: `pretraining_dataset`, `overlap_acdc`, `overlap_mnms`. The nnFoundation pretraining corpus (arXiv 2609.26924) must be checked for ACDC or M&Ms overlap before real runs. If that cannot be determined, the run must stay fail-closed. Do not set `false` without evidence. CC-BY-SA-4.0 share-alike applies to derived teacher weights; note it in the export and provenance.

### F6. T param groups (MEDIUM, plus a pre-existing mismatch)

- `engine._teacher_parameter_groups` classifies by hard-coded MedNeXt names `{up_*, dec_block_*, out_0}` under `encoder.model.*`. For a ResEnc adapter, no name matches, so the whole randomly initialized decoder (`decoder.stages`, `transpconvs`, `seg_layers`) silently falls into `teacher_trunk` at the encoder LR of 1e-4. The intent is head LR, since the decoder starts from scratch. Use explicit allowlists per architecture and fail if any parameter is unclassified.
- Pre-existing spec-vs-code mismatch, even for MedNeXt: spec §3 [AMEND] says the region head is `teacher.encoder.model.out_0` only and the rest is trunk. The code puts all MedNeXt `up_*`/`dec_block_*` in the head group at 1e-3. Flag this for the main track regardless of the migration.
- T-a (5%: boundary only, encoder+region frozen via `set_teacher_stage("warmup")`): with a random ResEnc decoder, the region logits stay random through T-a. Recommend a spec amendment so T-a trains boundary path + ResEnc decoder, or follow the official recipe. Official `PretrainedTrainer`: whole-net linear warmup for 50 epochs, SGD at 1e-3 (`initial_lr`), momentum 0.99, then poly. AdamW variant: 1e-4/3e-4. Q3D T-b uses AdamW, 1e-4 encoder / 1e-3 heads, 5% warmup then cosine. Record any departure as a deliberate deviation.
- Alias safety: `named_parameters()` dedupes `decoder.encoder.*` (first name `encoder.*` wins), so groups stay disjoint. Any classifier keyed on the `decoder.` prefix over `state_dict()` would double-assign encoder tensors, so add a disjointness and coverage assertion (`id(p)`).

### F7. Other hard-wired MedNeXt points (MEDIUM)

- `teacher_features.py`: `isinstance(encoder, MedNeXtL3DAdapter)`, `_MEDNEXT_SOURCES`, `_candidate_a_native_v1`. The ResEnc stride-8 equivalent of `x_res_3` is encoder skip index 3 (`stages.3` output, 256 ch, the same width as config `feature_channels: 256` by coincidence). Read C_T from the adapter. Bottleneck S1 at stride 16 is `stages.4` (320 ch). ResEnc has an extra stride-32 stage.
- `DualBranchTeacher3D` boundary path reads `features[0]` and `features[-1]`. For ResEnc, `features[-1]` is stride 32 (320 ch, 1–2 cells per axis after padding). This differs from the MedNeXt stride-16 bottleneck, and the receptive-field/semantics change should be documented.
- `alignment.teacher_feature_phase` checks `encoder.source_commit == UPSTREAM_MAIN_SHA` (MedNeXt). It needs a ResEnc branch keyed to the nnssl pin and DNA version, plus F1's contract.
- Official fine-tuning uses 192³ crops and `like_pretrained` spacing. With `spacing: null` the pretraining plan falls back to the dataset's default nnU-Net spacing, which resamples. Q3D forbids resampling, so the only matching official mode is `no_resample`. Native full-volume, no-crop inference is a Q3D-specific usage and has no official evidence behind it.
- Config: `model.teacher.architecture`, `source_repo`/`source_commit` (MedNeXt `0b78ed8…`), `pretrained.provenance_status: NOT_FOUND_OFFICIAL`. A ResEnc config needs new fields: nnssl pin, DNA version, HF repo and revision, weights sha256, strides, normalization, and padding policy. Keep Candidate A/A-3D configs untouched.

## 3. Proposed spec items (for the coordinator; not applied)

1. §3: teacher = nnFoundationCNN ResEnc-L. Encoder (stem+stages) pretrained with strict load. Decoder, 4-class seg head and boundary path random. Revise T-a/T-b and the param groups accordingly.
2. §4: add [OPEN] "ResEnc ResNet-D skip phase" (F1). ResEnc real alignment stays `NOT_RUN` until it is reviewed.
3. §2/§4/§13: stride decision (isotropic official vs stride-only anisotropic non-official). Per-axis padding without the ×2 minimum.
4. §3/§9: teacher-only per-frame z-score before padding. Student unchanged.
5. §3 provenance: nnssl/nnUNet pins, DNA version, HF revision, sha256, license, and an overlap determination from the nnFoundation corpus.
6. P0 JSON additions: ResEnc kernel/stride/padding for stem and stages.1–3 including the skip AvgPool, F1 offsets, C_T=256, strict-load inventory, and the alias check.

## 4. Not done / NOT_RUN

Patient data, training, GPU, the network-phase probe on real weights, the P0 class-support report, and the overlap determination are all NOT_RUN. No repository files changed by this worker. The only file written is this report.
