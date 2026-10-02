# DSS-US / SGSCN paper-faithful profiles: runnable with declared conventions (2026-10-02)

This report continues `dss_us_sgscn_paper_faithful_20261001.md` and
`dss_us_sgscn_paper_protocol_evidence_20261001.md`. **No profile below reproduces a paper exactly.**
Every runnable paper-faithful profile is of class `PAPER_FAITHFUL_WITH_DECLARED_CONVENTIONS`, and each
records `paper_equivalence = NOT_EXACT_PAPER_REPRODUCTION (declared conventions/fallbacks)`.

## Profile classes and readiness

The readiness receipt is `dss_us_sgscn_native_readiness_paper_faithful_runnable_20261002.json`, written by
`scripts/native_protocol_readiness.py`. It reports `profile_class` per profile and summarises results by class.

| Class | DSS-US | SGSCN |
|---|---|---|
| PAPER_REPRODUCTION | 11 BLOCKED_PROTOCOL | 2 BLOCKED_PROTOCOL |
| PAPER_FAITHFUL_REIMPLEMENTATION (template; values required) | 11 BLOCKED_PROTOCOL | 2 BLOCKED_PROTOCOL |
| PAPER_FAITHFUL_WITH_DECLARED_CONVENTIONS | **RUNNABLE**: `step2_dss_baseline_dss_paper_faithful_declared_conventions` | **RUNNABLE**: `ph2_paper_faithful_declared_conventions`, `sysu_us_paper_faithful_declared_conventions` |
| OFFICIAL_REFERENCE | n/a | RUNNABLE (unchanged gates; label renamed from REFERENCE_READY) |

Track B:
- DSS-US Step I (per-image remapped Dice) is EVALUATOR_READY.
- DSS-US Step II is BLOCKED_PROTOCOL.
- SGSCN paper metrics are BLOCKED_PROTOCOL. HM/XOR, the tie policy and the overlap measure are not defined in the paper and were not invented.

Data: CAMUS, PH2, SYSU-US and the DINO ViT-S/8 checkpoint are BLOCKED_DATA. Runners report this with exit code 3,
a precise reason per missing item, and no output directory. Nothing is downloaded.

Instantiation: `scripts/instantiate_native_profile.py` fills only null declared fields. Each filled field records a
status (`IMPLEMENTATION_CONVENTION`, `OFFICIAL_CODE_FALLBACK`, `USER_SUPPLIED` or
`BLOCKED_DATA_SUPPLIED_AT_RUNTIME`) and a source. The script writes a new hashed profile and never edits the base
template. The declaration files are `baseline/{SGSCN,DSS_US}/config/declared_conventions/*.json`.

## SGSCN

The paper fixes these values, and the code locks them (`PAPER_CONSTANTS`, `PAPER_LEARNING_RATE`):
- 3 conv layers, 3x3, stride 1, padding 1, 100 filters.
- SGD with momentum 0.9.
- Learning rate 0.1 for PH2 and 0.05 for SYSU-US.
- Unweighted sum L_ce + L_ss + L_cc.

The settings validator rejects any other value.

Declared values (PH2 and SYSU-US identical):

| Field | Value | Status | Reason |
|---|---|---|---|
| layer_order | conv_relu_bn | OFFICIAL_CODE_FALLBACK | official conv→ReLU→BN |
| output_normalization | final_batchnorm | OFFICIAL_CODE_FALLBACK | official final BN gives the zero-mean/unit-variance map |
| context_density | channel_softmax | IMPLEMENTATION_CONVENTION | Eqs. 3–4 need a pdf. The official epsilon shift gives signed weights; on synthetic data it gave unbounded losses and collapsed to one label, so it was rejected. Normalisation is exact, in the log domain |
| input_encoding | bgr_unit_interval | OFFICIAL_CODE_FALLBACK | official cv2 BGR / 255 |
| stopping_rule | official_max_iterations_min_labels | OFFICIAL_CODE_FALLBACK | paper says "until stable" |
| max_iterations | 50 | OFFICIAL_CODE_FALLBACK | official maxIter |
| min_labels | 3 | OFFICIAL_CODE_FALLBACK | official minLabels (pre-update label count) |
| final_forward_mode | train | OFFICIAL_CODE_FALLBACK | official labels from the last training forward |
| initialization / weight_decay | framework default / 0 | IMPLEMENTATION_CONVENTION (fixed in template) | paper silent |
| cohort_inventory (SYSU-US) | supplied_at_runtime | BLOCKED_DATA_SUPPLIED_AT_RUNTIME | the runner binds it to the image-inventory SHA-256 |

Readings of the paper (unchanged from 2026-10-01):
- Eq. 1 is read as standard cross-entropy, because the printed `ln Ŝ` is a typo.
- Eq. 2 is a literal sum.

Both declared profiles were exercised by synthetic end-to-end CLI tests. Each produced a sealed raw run with
provenance that includes `profile_class`, `stop_rule` and the environment contract.

## DSS-US

Runner: `baseline/DSS_US/scripts/run_native.py`. Its flow:
1. canonical environment check
2. BLOCKED_DATA checks: inventory, image-only staging root, DINO source pinned at `7c446df5`, checkpoint and its
   SHA-256
3. framework warm-up before the GT firewall
4. guarded production: DINO ViT-S/8 keys → Step I eigensegments (15) → official-fallback Step II
5. sealed raw output with full provenance (DINO receipt, cohort hash, code hashes, environment contract)

Declared values for `step2_dss_baseline_dss` (upstream `alexaatm/UnsupervisedSegmentor4Ultrasound@d4ac44c6`):

| Field | Value | Status |
|---|---|---|
| feature_l2_normalization | true | OFFICIAL_CODE_FALLBACK |
| dino_input_policy | rgb_totensor_imagenet_crop_to_patch_multiple | OFFICIAL_CODE_FALLBACK |
| eigenvectors / discard_trivial / normalize_rows | 14 / true / false | OFFICIAL_CODE_FALLBACK |
| upscale_method | nearest | OFFICIAL_CODE_FALLBACK |
| camus_input_conversion | prepared_8bit_image_files_loaded_rgb | OFFICIAL_CODE_FALLBACK |
| semantic_clusters | 15 | OFFICIAL_CODE_FALLBACK |
| phi_image | DINO ViT-S/8 output embedding of the ImageNet-normalised bbox crop | OFFICIAL_CODE_FALLBACK |
| dss_step2_definition | official_bbox_pipeline_v1 (border-fraction background; erode 2 / dilate 5; bbox ×8; pad crops < 8 px; L2; MiniBatchKMeans 4096/5000/10, seed 1; background → 0) | OFFICIAL_CODE_FALLBACK |
| step2_crf_applied | false | IMPLEMENTATION_CONVENTION (Table 2 has no CRF mark) |
| kmeans seed / n_init / max_iter / tol / algorithm | 1 / 10 / 300 / 1e-4 / lloyd | OFFICIAL_CODE_FALLBACK (seed from upstream, the rest from the scikit-learn 1.3.0 defaults pinned upstream) |
| camus_cohort_inventory, dino_checkpoint_sha256 | supplied_at_runtime | BLOCKED_DATA_SUPPLIED_AT_RUNTIME |

DSS-US remains an independent reimplementation. No upstream source was copied.

### Rows that stay blocked, and why

- **Step I rows: CRF.** The paper's Table 1 requires CRF, and upstream uses `simplecrf==0.2.1.1`. That package is
  source-only: no wheels and no declared build requirements. Adding it would make the hash-locked canonical
  environment unreliable. Per the milestone rule, it was not added and the gap is reported here; Step I gives
  BLOCKED_PROTOCOL with that reason. No canonical package provides a dense CRF.
- **proc rows (preprocessing):** official preprocessing uses kornia CLAHE, which is not in the canonical
  environment. No compatible fallback exists, so these stay templates.
- **aff/comb rows:** δ_ssd, δ_mi and the KNN k are unspecified in the paper, and the official formulas differ from
  the paper's equations. No value was invented.
- **Ours step2 rows:** C_mask and C_position have no paper or official value, and the mask/position embeddings are
  not wired.

## Environment

No change. Everything runs in `self-audit-canonical` v1 (CPython 3.10.20, torch 2.4.1+cpu, scikit-learn 1.7.2,
scikit-image 0.24.0, opencv-python 4.10.0.84). Nothing under `environments/` was modified.

## Verification (canonical environment `.runtime/envs/self-audit-canonical-cpu`)

| Suite | Result |
|---|---|
| `scripts/check_environment.py --variant cpu` | exit 0 |
| core/environment (`tests/test_self_audit_*.py`, `tests/test_pseudolabel_*.py`) | 134 passed |
| `tests/shared_benchmark` | 107 passed |
| `baseline/CUTS/tests` | 30 passed |
| `tests/native_baselines` | 47 passed (+11 subtests) |
| `baseline/SGSCN/tests` (incl. e2e PH2/SYSU-US declared profiles) | 32 passed |
| `baseline/DSS_US/tests` (incl. synthetic e2e DSS step2 run) | 42 passed |
| `scripts/native_protocol_readiness.py` | exit 0, byte-identical to the receipt |

Historical-224 freezes: v10 and v11 validate against their source snapshots (IMMUTABLE_HISTORICAL_SUPERSEDED), and
v12 is active and validates against the checkout. No freeze artifact was modified. No dataset was downloaded and no
real experiment was started.
