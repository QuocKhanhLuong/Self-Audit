# Q3D ResEnc Phase 0 — inspection evidence

This report records software/checkpoint/manifest-metadata inspection only. No patient training, image/mask target construction, dataset download or active-job change was performed. The external checkpoint is not committed.

## Checkpoint and strict load

- Repository: `MIC-DKFZ/nnFoundationCNN`; revision `f443c09319988e833b3d9589bf95909e4443d7b2`.
- [Pinned checkpoint](https://huggingface.co/MIC-DKFZ/nnFoundationCNN/resolve/f443c09319988e833b3d9589bf95909e4443d7b2/checkpoint_final.pth), 409,579,442 bytes.
- SHA-256: `ac262d3e8c226c79f9567fddc34d38e011284730ddf2294b5c6c319ce2f22bbd`.
- Top-level keys: `network_weights`, `citations`, `nnssl_adaptation_plan`.
- Adaptation: `ResEncL`, `key_to_stem=encoder.stem`, `key_to_encoder=encoder.stages`.
- Strict encoder: **448 keys, missing=[], unexpected=[]**. Decoder/heads are not loaded.
- Prefix counts: encoder.stem=8, encoder.stages=440; ignored decoder.encoder=448, decoder.stages=40, decoder.transpconvs=10, decoder.seg_layers=10.
- All key names and shapes: [checkpoint_keys_shapes.json](checkpoint_keys_shapes.json). Adaptation metadata: [adaptation_plan.json](adaptation_plan.json).

The factory uses nnssl `3d774e942256bd70e5e264c6b94fa38e25f7ed74`, `get_res_enc_l`, with `dynamic_network_architectures==0.4.4`. nnU-Net planner/finetuning reference is pinned to `c04f7ec8bbffaf3ebc3b26aa2c89042e8a8e368b`.

Weights license: **CC-BY-SA-4.0**. Declared pretraining: nnFoundation (~2.1M CT/MR/PET, 125 sources; arXiv 2609.26924 Table D7). ACDC/M&Ms overlap=false is **according to Table D7, not independently audited**. Neither strict load nor this declaration establishes patient efficacy or absence of undocumented overlap.

## Evidence boundaries

The locked K8 TRAIN JSON was freshly hashed; only its patient IDs and geometry metadata were read. Raw NIfTI headers/arrays were not available for independent geometry verification in this delivery. DEV/K5/K10 expected hashes come from the preserved few-shot plan; those manifest bytes were not rehashed locally. No U/DEV/TEST mask was read.

GPU forward/VRAM measurement is **NOT_RUN: CUDA is unavailable in the local verification environment**. This does not assert that no remote GPU exists. Largest declared TRAIN volume: patient057, native ZYX=(8,512,428), 1,753,088 voxels. No estimated VRAM number is substituted for a measurement.

The complete geometry/ERF receipts and final lock are included alongside this report. Random-input ERF is architecture evidence, not cardiac segmentation evidence. Historical probe variants are retained as NONCALIBRATION diagnostics; their mismatch with the actual teacher path is documented rather than hidden.

## Locked geometry and phase

TRAIN median spacing ZYX = **(1,1,1) mm**, median native shape = **(9,256,216)**. The 52/80 unit-spacing records are an explicit unverified-header limitation. The pinned planner topology rule with min feature-map size 4 gives strides **[(1,1,1),(2,2,2),(1,2,2),(1,2,2),(1,2,2),(1,2,2)]**, full padding multiple **(2,32,32)**. This is a native-metadata topology adaptation, not the complete resampling nnU-Net preprocessing recipe. One native volume is processed at a time.

Main stage-3 phase: step **(2,8,8)**, offset **(0,0,0)**. Maximum pure avgpool-skip displacement: **(0.5,3.5,3.5)**, reported independently of the empirical decision.

The canonical probe matches the actual teacher: z-score before high-side **replicate** padding, logical shape (33,97,97), model shape (34,128,128), cell (8,6,6), signed mean across 256 channels; absolute input-gradient mass centroid over 16 samples, seeds 42–57. Mean signed centroid offset ZYX = **(0.67213374376297,1.9470438957214355,1.722203016281128) voxels**. Each absolute mean is ≤2, therefore the predeclared rule locks the **main phase, zero correction**. Maximum individual offset **5.1299591064453125** is reported only. The Y mean is close to the threshold; this is a point-estimate rule, not a confidence guarantee.

The earlier even-center, stage-3-only-padding and full-zero-padding probes do not reproduce the delivered teacher preprocessing and are retained as **NONCALIBRATION**. The canonical choice was corrected for operator/context equivalence, regardless of its measured outcome; no threshold or seed was changed. S1 remains outside the six-run campaign and uses the analytical stage-4 phase, without claiming a separate empirical stage-4 calibration.
