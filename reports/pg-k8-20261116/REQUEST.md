# Candidate PG v2 implementation contract

Source: explicit user request 2026-11-16. Latest scope amendment: **code and local software checks only; no server testing or training in this delivery**. Real S1/S2/S3/A1/A2/A3/A4 results remain NOT_RUN. Initial design is proposed, untrained. Implement complete runnable flow, no intermediate model-quality gates and no scientific fallbacks. Data identity, correct formulas and finite calculations remain necessary. Existing tracks are immutable.

## Ownership and workflow

Use parallel Orca Codex workers with model gpt-5.6-luna, exactly one Claude logic reviewer. Every worker is in a shared checkout: never revert others. Root owns integration, publication and report. Workers do not commit/push or run SSH/GPU. Before PG implementation read `INSPECTION.md`; it must document required source reads. Tests only write to pytest tmp_path. Old A/A3D/Q3D/FUSE behavior, reports and run history are protected. Cleanup requires a concrete reviewed allowlist. Canonical entrypoints remain train.py/evaluate.py.

## Shared protocol

ACDC TRAIN80/VAL18/TEST50. Exact Track A K8 lock from configs/candidate_fuse/semi_k8.json and reports/fuse-semi-20261113/track-A; 8 patients/16 ED/ES masks. U is all TRAIN cine frames except these masks (including unlabeled-patient ED/ES). Patient-balanced U pairs from different patients. No VAL/TEST training images or masks. One labeled frame + two U frames per update. Native student geometry; permitted augmentation/CineMA preprocessing and partner crop/pad are explicit view operations. No change to source data.

S1 shared student encoder init, S2 shared teacher, S3 shared teacher codebook; then four arms, identical resolved config except pg.enabled/cbkd.enabled. A1 false/false, A2 true/false, A3 false/true, A4 true/true. Same seed42, init, teacher, codebook, update count, optimizer, augmentation and VAL student-only selection. No per-arm tuning.

## Exact configuration

Base `configs/candidate_pg/pg_k8_base.json`, arm files contain ONLY the two switch dictionaries. Schema self_audit.candidate_pg.v2. All numbers below fixed for campaign; explicit runtime/manifests paths may be materialized from existing hash-pinned lock. No hidden hyperparameter defaults.

data: dataset acdc; split_lock configs/candidate_fuse/semi_k8.json; k_labeled8; unlabeled all_train_cine_frames_minus_labeled_masks; forbid_splits_in_training VAL18/TEST50; native_geometry true; per_step labeled1/unlabeled2; u_partner_policy different_patient; u_sampling patient_balanced.

augmentation: policy inplane_strong_v1; weak spatial_only; strong intensity_only_on_weak_canvas; n_strong_views2; clamp[0,1]; cutmix area_fraction[.25,.5], align_px16, full_depth true, pad_weight0.

student: arch unetrpp_lite_fuse; channels[16,32,64,128,256]; num_classes4; init cinema_distill; fusion levels5/gate_bias-2/inside_encoder true; teacher_dropout_on_labeled start0/end.5; fp_skip_dropout.5.

cinema_init: hf_repo mathpluscode/CineMA; files pretrained/cinema.safetensors and pretrained/config.yaml ONLY, never ACDC/M&Ms/M&Ms2 finetuned checkpoints. taps cinema_conv_1_8->student.stage3, cinema_tokens_1_16->student.stage4; projector conv1x1x1_discard_after_init; loss cosine; data TRAIN images only; AdamW lr3e-4/wd1e-4; updates10000.

teacher: arch mednext_l; kernel3; channels[32,64,128,256,512]; block_counts[3,4,8,8,8,8,8,4,3]; exp_ratios[3,4,8,8,8,8,8,4,3]; down_strides[[1,2,2],[1,2,2],[2,2,2],[1,2,2]]; feature_phases[[1,1,1],[1,2,2],[1,4,4],[2,8,8],[2,16,16]]; heads region4/sdf channels3 radius_mm5; init random; optimizer AdamW lr1e-3/wd1e-4; updates8000; select VAL18 FG teacher region.
teacher.semi: method unimatch_v1; target self_weak_stopgrad; stream_weights {"s1":0.25,"s2":0.25,"fp":0.5}; fp dropout3d_0.5_on_encoder_skips; threshold mode class_adaptive/ema.999/clip[.5,.95]; sdf_consistency weight.5/on confident_voxels/loss l1_normalized; lambda_u max1/warmup_updates_labeled_only800/ramp_fraction.1.

codebook: teacher_tap mednext_level3_phase_2_8_8_aligned_to_student_stage3_grid_1_8_8; P_T conv1x1x1_256_to_64; codebook size128/dim64/dead_code_restart_every500/usage_entropy_weight.01; H_T_Q conv1x1x1_64_64_relu_64_4; train_labels L_K pooled GT + detached teacher pseudo-labels on U (confidence-weighted); commit_beta.25; rho_temperature1; updates4000.

pg: enabled true; target_source fusion_weak; threshold mode class_adaptive/ema.999/clip[.5,.95]; agreement paths teacher_only_region/student_only_clean_weak/form sqrt(pT[y]*pS[y]); boundary near_mm2.5/radius_mm5/mode soft_weight; stream_weights {"s1":0.25,"s2":0.25,"fp":0.5}; lambda_u max1/warmup_updates_labeled_only800/ramp_fraction.1; loss weighted_ce_plus_weighted_dice.

cbkd: enabled true; variant aux_code_branch; codebook full_C_frozen; student_tap stage3_post_epa_128ch; P_S conv1x1x1_128_to_64; H_S_Q conv1x1x1_64_64_relu_64_4; rho_temperature1; weights assign1/qtask1/commit.25/qpred0; lambda_cb.5; inputs clean weak U (no CutMix) assign/commit plus labeled student-only assign/commit/qtask; warmup_updates_labeled_only800.

training: AdamW lr1e-3/wd1e-4/schedule reuse candidate_fuse _schedule_lr; updates8000; updates_per_epoch80; eval_every_epochs1; eval_splits train_labeled/VAL18/TEST50; selection splitVAL18/path student_only/metric fg_dice_patient_macro.
export: strip_prefixes fusions./mimic_heads./code_branch./P_S./H_S_Q.; assert_no_teacher_keys true.

## S1 cinema_init.py

Read real pretrained SAX architecture from official config, not hardcoded guesses. Verify source, SHA, config; log parameter count and actual spacing/intensity/crop192x192x16 preprocessing and handling !=16 slices. CineMA own preprocessed crop, student native input. Map features back using physical geometry and trilinear sampling; valid cell mask must include cropped support, not padded coordinates. Distill stem conv about1/8 into stage3 when actually present (otherwise log absent and use only tokens); ViT tokens1/16, z-patch1 into stage4. Loss mean valid 1-cos(projector(student_feat),cinema_feat). Only student stem through stage4 train; decoder remains random. Projectors discarded. Save encoder-only student_init.pt, CineMA/data-list SHA, final cosine receipt. Holdout cosine uses a fixed TRAIN-only image subset, never VAL/TEST images.

## S2 teacher_mednext.py

Actual MedNeXt-L kernel3 with anisotropic down/up strides, full decoder and region/sdf heads. Wrap official modules if needed; old git source is reference. Derive phases from actual modules using grid_phase, compare all five to FuseStudent.teacher_phases. Forward(image,geometry,return_features=True) -> region_logits,sdf,features5; keep boundary output needed by unchanged q3d_teacher_loss (document compatibility head). SDF tanh/radius5. Supervision exact q3d_teacher_loss using L_K only. Single network UniMatch weak stopgrad target, intensity-only strong streams, FP dropout3d on all five skips. Class threshold owns checkpointed state. SDF L1 consistency only confident voxels. 800 labeled-only warmup then fixed ramp. Select best teacher VAL FG, freeze permanently.

Inspection resolution: unchanged `q3d_teacher_loss` requires four SDF channels and a four-class boundary head. Preserve public PG `sdf` as the three foreground channels; add a one-channel background SDF compatibility head and expose `sdf_all = cat(background_sdf, sdf)`. Only the loss adapter passes `sdf_all` under the old loss key `sdf`, with exactly its three expected keys. PG boundary weights and U consistency use foreground `sdf` only. Record the internal boundary/background heads explicitly in the config and report; this reconciles the requested public three-channel interface with exact reuse of the existing teacher objective.

## views.py and reliability.py pure logic

Spatial weak transform per image using existing StrongInplaneAugment, own valid mask transformed with same draws. Strong only intensity on weak canvas, clamp[0,1], geometry unchanged. CutMix full-depth in-plane box coordinates multiples16, chosen within valid intersection; fraction .25-.5 when feasible. Different-patient partner center crop/pad to anchor canvas; all padded cells valid0. Mix packet p_W,y,w,valid using same box. No teacher features on s1/s2/fp/code branch.

Class-adaptive threshold: initialize tau_g=1/C and p_tilde=1/C; EMA(.999) mean valid U maxprob and mean class probs; tau_c=clip(tau_g*p_tilde/max(p_tilde),.5,.95). Update one combined A/B observation; teacher and student separate state, save/resume exactly. Weights from fusion weak pW, y=argmax: agreement sqrt(pT_region[y]*pS_cleanweak[y]); band min_c abs(d_T,c)<2.5/5. Inside w=confidence*agreement, outside w=1[confidence>=tau_y]*agreement; multiply valid; all detached.

Pure functions with hand-computed tests: cutmix_box, mix_packet, class_adaptive_threshold_update, voxel_weights, boundary_band, weighted_ce, weighted_dice, rho, vq_ste, assign_kl, strip_export_keys. Weighted CE=sum(w*ce)/sum(valid), weighted soft Dice per class uses w mask and skips classes without weighted target mass.

## student_pg.py and engine.py

Subclass/wrap FuseStudent without modifying it, include_mimic=False. Add dropout of f0/s1/s2/s3/s4 immediately before decode (FP only), return post-EPA stage3(128ch), train-only code branch. L: frozen teacher on weak L; student fusion on same spatial canvas strong intensity L with sampled teacher keep; student-only same strong L. loss=.5*seg(fusion,L)+.5*seg(student,L). Fusion gates get gradients ONLY from these labeled losses.
For U when switches active after warmup: frozen teacher wa/wb; clean student-only wa (grad iff CBKD). PG fusion weak target A/B under no_grad and keep1. Pack detached probabilities, class weights and valid. Update threshold A+B. Strong1/2 CutMix packets same boxes; student-only strong predictions and clean-wa FP. PG loss=.25*wseg(s1)+.25*wseg(s2)+.5*wseg(fp); add lambda_u. CBKD labeled stage3 plus clean weak A stage3; add lambda_cb after800. U is image-only. A1 remains same init/teacher/codebook/budget; switches are the only config difference.

No quality gates, no extra arms, no per-arm tuning. Explicit OOM adaptation may reduce strong views2->1+FP; redistribute strong .5, FP .5 and apply same decision to all arms/mechanisms, never independently. Decision must be receipted before affected arms; preserve geometry/model/budget. Avoid hidden retries or changes. Cheap tensor shape/range/valid/no-target-grad asserts only inside loop.

Fixed implementation resolutions (common to every arm): zero-based update u uses lambda_u=0 for u<800, otherwise min(1,(u-800+1)/ceil(.1*8000)); lambda_cb becomes .5 at u=800. Compute packet weights with the pre-update threshold, then update the threshold from combined A/B valid voxels. Weighted Dice uses foreground classes1,2,3. S2 strong streams use CutMix, including the same box for SDF targets/confidence/valid; FP uses unmixed A. Labeled spatial padding keeps the existing FUSE background convention. Student LR uses the reused schedule with explicit warmup_fraction .05; S1/S2/S3 use constant LR at their specified values. S3 AdamW is explicitly lr1e-3/wd1e-4. S1 image batch1; other stages L1/U2. Common runtime uses clip norm1, CPU float32/GPU bfloat16, deterministic algorithms and disabled TF32. Separate labeled/optional-U RNG streams preserve common labeled augmentation; reset common full student initialization before every arm.

## S3 codebook.py / CBKD

Frozen teacher level3 256ch phase(2,8,8) aligned to student stage3(1,8,8) using align_feature, P_T256->64, C128x64, H_T_Q64->64ReLU->4. S3 learn P_T/C/H_T_Q: pooled L GT CE+Dice and detached confidence-weighted U pseudo-label task; VQ codebook loss plus .25 commitment and .01 usage-entropy penalty; restart dead codes every500. Freeze all at end. Log perplexity/dead/codes_per_class/qT_VAL pooled FG, never gate.
rho(z,C)=softmax(-squared_euclidean(z,C)/tau), tau1, preserve shared code order. Student P_S128->64, frozen C nearest+STE, H_S_Q64->64ReLU->4 sees only code tensors. L_assign KL(sg rhoT || rhoS) L/U; L_commit ||zS-sg e_nearest||^2 L/U; L_Qtask softCE pooledGT L only; L_Qpred implemented weight0. L_CBKD=assign+qtask+.25commit. Frozen codebook never optimizer-updated. Valid masks exclude internal pad. qS code shuffle diagnostic spatially shuffle codes (not inputs) at VAL after selection for A3/A4.

## Metrics, logging, export, report

Every80updates = epoch. At each epoch all TRAIN16(no augmentation), VAL18, TEST50 evaluation; eval uses RNG/mode preservation and no_grad. S2 path teacher; S3 path qT on pooled grid with correct mm cell geometry; arms student_only and fusion. Selection only teacher VAL/S3 fixed final/student_only VAL respectively, ties earliest. TEST values never affect any selection. Per-epoch TEST is explicitly authorized monitoring and makes TEST descriptive; final TEST_ONCE is one transaction locking all four selected export hashes, not a claim TEST previously unseen. Do not modify old FUSE ledger behavior.
Fixed columns in order: stage,arm,epoch,update,split,path,loss_total,loss_sup,loss_pg,loss_cbkd,fg_dice,rv_dice,myo_dice,lv_dice,fg_hd95,rv_hd95,myo_hd95,lv_hd95,fg_assd,rv_assd,myo_assd,lv_assd,lr,seconds. Append best_val as final column to satisfy explicit mark requirement without changing ordered prefix. Training epoch-mean loss columns only train rows; others empty. One unique row per(epoch,split,path), CSV and JSONL same rows; mark best_val consistently after selection. Patient rows each split/path/epoch in patients_epochXXX.csv, frame metrics aggregated per patient before macro. S1 separate epoch,update,cosine_train,cosine_holdout.
Every50 updates file-only diag.jsonl: loss components, class/band coverage, tau_c, agreement, gate means, teacher-vs-fusion disagreement, code perplexity/dead, memory/time. run.log DEBUG file-only, no warnings repeated each step. Console max one line per epoch, one start/end line per stage. Every checkpoint saves model/optimizer/threshold/sampler/augmentation and Python/NumPy/torch CPU/CUDA RNG, configuration and source/data identities. Reuse FUSE deterministic/TF32 seed policy and record it on each receipt. Resume exact and do not duplicate epoch rows.
Student-only export strips fusion/mimic/code branch/teacher keys, strict independent cold-load in subprocess, logits match in-memory; reuse FUSE export machinery when compatible. For selected snapshot compare VAL and TEST logged epoch metrics to exported recomputation. All four final exports one TEST transaction. FG and RV/MYO/LV Dice, HD95 and ASSD patient-macro with existing absent-class conventions. Paired A2-A1,A3-A1,A4-A1,A4-A2,A4-A3: Wilcoxon +10000 patient-bootstrap seed42 reuse fuse_compare_results.py. Plot learning curves and diag; no fabricated metrics.
Report README.md with measurements/interpretation/deviations separate; files/tests, actual counts/memory when measured, S1/S2/S3 receipts, 2x2 TRAIN/VAL/TEST table at VAL-selected epoch, curves/paired/shuffle. Current code-only delivery: real runs NOT_RUN, describe runnable commands and missing measured evidence. EXPERIMENT_LOCK/source/config/data-list/launch receipts. Commit/push main after local checks, report verified remote SHA; second report commit when real report exists (not in code-only scope).

## Required local checks

1 CutMix box16 alignment/full-depth/same packet/padvalid0. 2 Strong geometry/range. 3 target zero gradients and gates no PG/CBKD grads. 4 no teacher features in learning streams. 5 threshold resume bit-identical. 6 normalized SDF band exact. 7 code ordering/STE/frozen C. 8 configs differ switches only. 9 clean export/subprocess/logit parity. 10 actual MedNeXt phases. 11 unchanged existing FUSE/Q3D suites. 12 fixed epoch schema, uniqueness, VAL-only best unaffected by TEST changes. Additional tiny full-flow synthetic software smoke appropriate, never substitute fixture for real CineMA provenance or patient evidence. Full python scripts/check_repo.py and python -m pytest -q; Q3D smoke per AGENTS.

## Final local implementation resolutions

S1 holdout is the last 10% of the stable TRAIN-only image list (explicit base-config value), loaded lazily. Its cosine columns report similarity, while loss_cosine reports 1-cosine. No VAL/TEST image is used for that holdout.

CutMix always uses 16-aligned coordinates. Valid in-plane support is intersected across acquired slices; entirely padded partner z slices retain valid=0 and weight=0 across the full-depth box. This explicitly resolves different slice counts without an unaligned-box fallback. Code-shuffle moves whole code vectors together across spatial positions, preserving code identity.
