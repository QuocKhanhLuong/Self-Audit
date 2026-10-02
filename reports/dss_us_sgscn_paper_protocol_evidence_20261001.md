# DSS-US / SGSCN paper-protocol evidence audit (2026-10-01)

Sources (all read-only evidence; no upstream code copied, no data used):

- DSS-US paper: arXiv 2408.02043v1 HTML, sha256 `15110a55216bb448c5db0330c4b3a492413350a50f4f118930a274ac61f1c198`.
- DSS-US official code: `alexaatm/UnsupervisedSegmentor4Ultrasound@d4ac44c60df18b921c590796f6994a4c8ac0726c` (455 commits searched).
- DINO: `facebookresearch/dino@7c446df5b9f45747937fb0d72314eb9f7b66930a`.
- SGSCN paper: arXiv 2107.04934 PDF, sha256 `222a28c659e3f507dfed519a3e8b8945fa2afcb0a84a7ab428205e807d831051`.
- SGSCN official code: `osmond332/Spatial_Guided_Self_Supervised_Clustering@592efb6e72ceeef15c8be0630a4673eda5dce6f5`.

## DSS-US

| Blocker | Resolved value / rule | Source (type) | Status |
|---|---|---|---|
| CAMUS row recipes (11 rows) | none: C_ssd, C_mi, C_pos, delta, KNN k, preprocessing choice and Step II embedding weights are symbolic in the paper; no CAMUS/cardiac config, file or commit exists upstream | paper §2.1-2.2, Eqs. 1-6; repo `configs/dataset` holds only README/.gitkeep | **UNRESOLVED** |
| CAMUS cohort / frames | 500 test-set images, 50 patients, "5 for each ES and ED"; views/frames not enumerated, no list | paper §3 (PAPER) | **UNRESOLVED** |
| Oversegmentation count | 15 eigensegments | paper §2.1, Table 1 (PAPER); `configs/defaults.yaml segments_num: 15` (OFFICIAL_CODE) | resolved |
| Step II semantic clusters (CAMUS) | not fixed: official sweeps use 6-15; paper silent | `configs/sweep/num_clusters.yaml` | **UNRESOLVED** |
| DINO backbone | `dino_vits8`, keys of the last attention layer, patch size 8 | paper §2.1 (PAPER); `configs/model/dino_vits8.yaml`, `configs/spectral_clustering/defaults.yaml` (OFFICIAL_CODE) | resolved |
| DINO checkpoint | official ViT-S/8 weights `dino_deitsmall8_pretrain.pth` (torch.hub default) | `extract_utils.get_model`; DINO `hubconf.py::dino_vits8` (OFFICIAL_CODE) | resolved identity; file + SHA-256 **BLOCKED_DATA** |
| CRF usage | applied to all Step I rows | Table 1 "*CRF postprocessing" (PAPER) | resolved |
| CRF parameters | conflicting official sets: `crf/defaults.yaml` (10/80/13/3/3/5) vs sweep configs (15/7/10/5/5/10) | OFFICIAL_CODE conflict | **UNRESOLVED** |
| Step I evaluated stage | `crf_multi_region` | Table 1 (PAPER); sweep `seg_for_eval` (OFFICIAL_CODE) | resolved |
| Step I eval_per_image | True | `configs/eval/defaults.yaml`; `pipeline.py::evaluate` (OFFICIAL_CODE) | resolved |
| Step I pairing | equal counts: Hungarian max-IoU (cost pixel_count-IoU); unequal: exclusive majority vote (ascending GT classes, first max-IoU segment, later class wins only on strictly greater IoU, displaced class not revisited) | `eval_utils.py::match`, `hungarian_match`, `majority_vote_exclusive` (OFFICIAL_CODE) | resolved |
| Step I IoU threshold | 0.0 (exclusive branch ignores it) | `configs/eval/defaults.yaml` (OFFICIAL_CODE) | resolved |
| Step I void / unmatched | unmatched segments -> 0; label 0 excluded; void_label 0 | `remap_labels`, `evaluate_dataset_with_remapping` (OFFICIAL_CODE) | resolved |
| Step I aggregation | per-image mean foreground Dice (absent label = 0); dataset mean and population std | `evaluate_dataset_with_remapping` (OFFICIAL_CODE); Table 1 "DICE, std" (PAPER) | resolved |
| CAMUS n_classes | 4 (background + LV endocardium, myocardium, left atrium) | paper §3 (DERIVED_MECHANICALLY) | resolved |
| Step II matching mechanics | same per-image remapped matching | paper §3 (PAPER); `pipeline.py::evaluate` (OFFICIAL_CODE) | resolved |
| Step II evaluated stage | segmaps vs crf_segmaps undetermined (no CRF mark in Table 2; no Step II `seg_for_eval`) | — | **UNRESOLVED** |
| Label consistency (LC) | only prose; not computed by official code; ties and std aggregation unspecified | paper §3 | **UNRESOLVED** |
| Seeds / repetition / selection | none reported; bbox K-means `seed: 1` is a code default | `configs/bbox/defaults.yaml` | unresolved (not required for Step I evaluator) |

## SGSCN

| Blocker | Resolved value / rule | Source (type) | Status |
|---|---|---|---|
| Architecture | paper: 3 conv layers 3x3/s1/p1/100 filters each with ReLU+BN (order unstated); code: 3x3->ReLU->BN x2 (nConv=2) then 1x1->BN without ReLU | paper §3.2 (PAPER) vs `demo_final.py::CSNet` (OFFICIAL_CODE) | **DISCREPANCY** |
| Output normalisation | zero-mean/unit-variance map before argmax = final BN | paper §2.2 + `CSNet.bn3` | resolved (agree) |
| Optimizer | SGD, momentum 0.9; lr 0.1 PH2, 0.05 SYSU-US | paper §3.2 + `demo_final.py` | resolved (agree) |
| Max iterations / stopping | paper: "until clustering and loss become stable" (no criterion); code: maxIter 50, stop after the step when pre-update label count <= minLabels 3; train-mode final forward | paper §2.6 vs `demo_final.py` | **DISCREPANCY** (code-only rule) |
| Context loss | enabled, weight 1 | paper §2.5-2.6 (PAPER); `--center` (OFFICIAL_CODE, non-default flag) | resolved |
| Loss formula / weights | paper: unweighted sum of pixel-summed CE (Eq. 1) + L1 (Eq. 2) + context; code: mean CE + 5 x mean L1 + context | paper Eqs. 1-2, §2.6 vs `demo_final.py` | **DISCREPANCY** (both reduction and weight) |
| Max-overlap rule | evaluate only the cluster with the largest overlap with the GT segment | paper §3.1 (PAPER) | resolved in principle; overlap measure **UNRESOLVED** |
| Tie behaviour | not specified | — | **UNRESOLVED** |
| DSC | standard Dice of selected cluster vs GT | paper §3.1 | resolved |
| HM | definition not given; reported HM 130.8% / 165.2% exclude the 1-Jaccard Hammoude form | paper Tables 1-2 | **UNRESOLVED** |
| XOR | named only | paper §3.1 | **UNRESOLVED** |
| Aggregation | mean over images ("Mean%") | paper Tables 1-2 | resolved |
| Official evaluator | none in the pinned repository | repo | absent |
| PH2 cohort | all 200 studies | paper §2.1 | resolved |
| PH2 input format | unspecified; official sample input is JPEG (PH2 ships BMP) | paper silent; `input_images/test/IMD017.jpg` | **UNRESOLVED** |
| SYSU-US cohort | 100 images, random 5 per sequence; no seed or list | paper §2.1 | **UNRESOLVED** |
| Seeds / repetition | not specified (single run per image) | paper, code | instrumentation only |

The official-code reference profiles are unchanged and remain explicitly
non-paper (`paper_equivalence: UNRESOLVED`). No paper profile was made executable
by substituting reference settings.

## Statuses

- DSS_US_PROTOCOL_STATUS: **BLOCKED_PROTOCOL** (all 11 producer profiles).
- DSS_US_TRACK_B_STATUS: Step I evaluator **EVALUATOR_READY** (4 Step I profiles); Step II **BLOCKED_PROTOCOL**; overall BLOCKED_PROTOCOL.
- SGSCN_PROTOCOL_STATUS: paper profiles **BLOCKED_PROTOCOL**; official-code reference profiles REFERENCE_READY.
- SGSCN_TRACK_B_STATUS: **BLOCKED_PROTOCOL**.
- Native reproduction (both): BLOCKED_PROTOCOL; Track A: BLOCKED_ADAPTER; data: BLOCKED_DATA.

Safe to run once data/checkpoints exist: SGSCN official-code reference producers
(PH2, SYSU-US) on an image-only staging set, explicitly as non-paper reference runs
without paper Track B. No DSS-US producer profile and no paper profile is runnable.
The DSS-US Step I evaluator becomes usable only once a Step I producer is unblocked.
