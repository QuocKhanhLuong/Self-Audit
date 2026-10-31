# Historical B8 Recipe Audit and Lineage Report

**Audit Target**: Historical B8 evidence and source in `results/compact_acdc_l8_20261029/experiment_20261028/b8_recovery/` and commit `0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca`.
**Scope**: Read-only audit owning exclusively `reports/b8_reproduction_20261031/RECIPE_AUDIT.md`. No server actions, no training, no modifications to trainer or historical files.

---

## 1. Executive Summary

1. **Exact Recipe & Source Closure**: The historical B8 run is an ACDC L8/U72 semi-supervised `ema_full` pipeline executed for 10,000 steps with 1,000 steps of full-model supervised warmup. The canonical source commit is `0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca` with exact source closure SHA256 `3c7fa85f00d54a6e087af3cd6c7045351fb72d532f1069bf09d95f9cd72359b7` spanning all 11 constituent `.py` files.
2. **Evaluator Identity (Raw Student vs. EMA)**: Code-level audit of [`self_audit/engine.py`](https://github.com/QuocKhanhLuong/Self-Audit/blob/0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca/self_audit/engine.py) and [`self_audit/evaluation.py`](https://github.com/QuocKhanhLuong/Self-Audit/blob/0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca/self_audit/evaluation.py) proves conclusively that `load_checkpoint` only loads `payload["student"]`, ignoring `payload["teacher"]`. All published test metrics (TEST50 Class-First Macro4 = 0.880049, Foreground Dice = 0.841210) evaluated the **raw student**, not the EMA teacher.
3. **Missing Payload & Exact Resume Gaps**: The original physical checkpoint payload `weights.pt` (SHA256 `0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6`, manifest `ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa`) was not stored in Git. Mid-run resumption was unsupported in the historical code (`"resume": "NOT_SUPPORTED"` in manifest; absence of CLI `--resume`; unrecorded CUDA RNG state; step-loop reset; exclusive file lock on `steps.jsonl`). As authorized, the user approved exactly ONE clean 10,000-step retrain and verified remote backup.

---

## 2. Reconstructed Commands from Recorded Artifacts

### 2.1 Training Invocations
The following command is reconstructed from the stored CONFIG/REPORT and historical CLI. The original shell invocation was not recovered, so this is not evidence of its original config filename. Training used `train.py` from commit `0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca`:
```bash
python train.py \
  --config results/compact_acdc_l8_20261029/experiment_20261028/b8_recovery/full/CONFIG.json \
  --data /workspace/Self-Audit/data/prepared/ACDC_L8_S42/train.json \
  --data-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --out /workspace/compact-acdc-l8/20261028/b8_recovery/full \
  --device cuda \
  --allow-full-run \
  > /workspace/compact-acdc-l8/20261028/b8_recovery/full.stdout.log \
  2> /workspace/compact-acdc-l8/20261028/b8_recovery/full.stderr.log
```
*Note*: When using a modular backbone configuration, `--model-config` provides the explicit provenance JSON. In `results/compact_acdc_l8_20261029/experiment_20261028/b8_recovery/full/CONFIG.json`, the `model` object was embedded directly.

### 2.2 Frozen Evaluation Invocation
The frozen-evaluation receipt and historical `evaluate.py` permit this equivalent reconstruction (50 patients, 100 frames):
```bash
python evaluate.py \
  --checkpoint /workspace/compact-acdc-l8/20261028/b8_recovery/full/final \
  --checkpoint-sha256 ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa \
  --data /workspace/Self-Audit/data/prepared/ACDC_TEST50/test.json \
  --data-sha256 8f89d631bc67f7eccef94740866513fe6f794872e7cb6e61302b1663e17e8f9e \
  --out /workspace/compact-acdc-l8/20261028/b8_recovery/test50_receipt \
  --device cuda \
  > /workspace/compact-acdc-l8/20261028/b8_recovery/eval.stdout.log \
  2> /workspace/compact-acdc-l8/20261028/b8_recovery/eval.stderr.log
```

### 2.3 Unlabeled Teacher Target Cache Build
The cache identity/manifest supports the following reconstructed command. It is not a recovered shell transcript; `--historical-source` must also be supplied when the pinned identity requires that bridge:
```bash
python scripts/build_compact_teacher_cache.py \
  --data-manifest /workspace/Self-Audit/data/prepared/ACDC_L8_S42/train.json \
  --data-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --checkpoint-dir /workspace/compact-acdc-l8/20261028/b8_recovery/full/final \
  --teacher-identity /workspace/compact-acdc-l8/20261028/b8_recovery/TEACHER_IDENTITY.json \
  --teacher-identity-sha256 1cc82863264331c98d1ffd48fa5e2da61cdc6dcce5bcc1a55aa94762bfe4526b \
  --output-dir /dev/shm/compact_teacher_b8 \
  --device cuda \
  --precision float32 \
  --batch-size 8 \
  > /workspace/compact-acdc-l8/20261028/b8_recovery/cache.stdout.log \
  2> /workspace/compact-acdc-l8/20261028/b8_recovery/cache.stderr.log
```

---

## 3. Effective Settings and Hyperparameters

The exact effective configuration as stored in `results/compact_acdc_l8_20261029/experiment_20261028/b8_recovery/full/CONFIG.json`:

| Parameter | Effective Setting | Evidence & Operational Role |
| :--- | :--- | :--- |
| `schema` | `"self_audit.semi_v2"` | Top-level schema validator; disallows legacy v1/bank configs |
| `method` | `"ema_full"` | Supervised + pseudo full-model AdamW/SGD updates followed by EMA |
| `update_scope` | `"head"` | Legacy pseudo-guard field; does **not** freeze body in `ema_full` |
| `steps` | `10000` | Total outer training iterations (10,000 steps = 2,440.04s elapsed) |
| `warmup_steps` | `1000` | Steps 0..999 run supervised updates only; pseudo updates suppressed |
| `seed` | `42` | Global initialization seed for PyTorch, CUDA, and NumPy Generator |
| `body_lr` | `3e-05` (0.00003) | AdamW learning rate for all non-head trainable parameters |
| `head_lr` | `0.003` (0.003) | SGD learning rate for `model.head` parameters |
| `pseudo_lr` | `0.001` (0.001) | Retained in schema; **inactive** in `ema_full` (uses body/head LRs) |
| `weight_decay` | `0.0001` (1e-4) | Decoupled weight decay applied to body parameters in AdamW |
| `ema_decay` | `0.99` | Exponential moving average parameter decay: $\theta_T \leftarrow 0.99\theta_T + 0.01\theta_S$ |
| `pseudo_threshold`| `0.7` | Minimum teacher softmax probability confidence for pseudo-labeling |
| `ridge` | `1e-06` | Legacy numerical stabilization constant |
| `risk_budget` | `0.0` | Risk budget constraint parameter |
| `guard_tolerance` | `1e-08` | Guard check numerical tolerance |
| `backtracks` | `8` | Maximum backtrack trials |
| `save_every` | `1000` | Checkpoint emission interval (steps 1000, 2000, ..., 9000, final) |
| `threads` | `2` | CPU thread concurrency limit (`torch.set_num_threads(2)`) |
| `batch_size` | `8` | Number of slices per sampled batch (single-patient per role) |
| `crop_size` | `256` | Native spatial crop window: $256 \times 256$ pixels |
| `window.mode` | `"none"` | Adaptive semantic window disabled in baseline arm |
| `meta_window.lr` | `0.001` | Inactive when `window.mode == "none"` |

---

## 4. Architecture, Optimization Groups, and Mathematics

### 4.1 Architecture Specification
- **Model Class**: `NativeStudent` defined in [`self_audit/model.py`](https://github.com/QuocKhanhLuong/Self-Audit/blob/0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca/self_audit/model.py). Total parameters: **32,865,530**.
- **Input Adapter**: `CenterDifference`. Maps 3 adjacent cine context slices $[x_{z-1}, x_z, x_{z+1}]$ into normalized RGB representation via difference channels:
  $$\text{diff} = [x_{z-1} - x_z, x_{z+1} - x_z], \quad x_{\text{norm}} = \frac{x_z - \mu}{\sigma} + \text{Conv2D}_{2 \to 3}(\text{diff})$$
  with $\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$.
- **Backbone Encoder**: DINOv3 ConvNeXt-Tiny (`OfficialStages`), embed dimensions $(96, 192, 384, 768)$.
- **Decoder & Feature Fusion**:
  - Lateral: $1 \times 1$ convs projecting backbone stages to $(96, 128, 192, 256)$.
  - Up: $1 \times 1$ convs + bilinear $2\times$ interpolation.
  - Refine: 2 Residual blocks per resolution stage.
  - Detail: Conv2D $3 \to 32$, GELU, Residual block (operating at native input resolution).
  - Fuse: Conv2D $(96 + 32) \to 32$, GELU, Residual block.
- **Head**: Conv2D $32 \to 4$, kernel size $1 \times 1$, producing 4 segmentation logits (`BG`, `RV`, `MYO`, `LV`).

### 4.2 Optimizer Construction and Gradient Clipping
In [`self_audit/engine.py`](https://github.com/QuocKhanhLuong/Self-Audit/blob/0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca/self_audit/engine.py):
1. **Body Parameters**: All parameters `p.requires_grad` where `not name.startswith("head.")`.
   - Optimizer: `torch.optim.AdamW(body, lr=3e-05, weight_decay=1e-04, foreach=None)`.
2. **Head Parameters**: `model.head.parameters()`.
   - Optimizer: `torch.optim.SGD(model.head.parameters(), lr=0.003, momentum=0.0, weight_decay=0.0)`.
3. **Gradient Norm Clipping**:
   - `torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0, error_if_nonfinite=True)`.
   - Executed inside both `supervised_step` and `full_pseudo_step`.

### 4.3 Training Step Schedule and Pseudo Updates
- **Warmup Phase (Steps 0 to 999)**:
  - Supervised step: Computes CE + present-foreground Dice on labeled batch $x_s, y_s$. Steps both AdamW and SGD.
  - Audit and unlabeled batches $x_a, x_u$ are sampled via NumPy RNG to advance the generator state deterministically, but **no** pseudo forward pass or optimizer update is executed.
  - EMA update executes at the end of each step.
- **Semi-Supervised Phase (Steps 1,000 to 9,999)**:
  - Supervised step executed first on $x_s, y_s$.
  - Teacher forward pass: $q = \text{teacher}(x_u).softmax(1)$.
  - Pseudo filtering: Confidence $conf = \max_c q_c$, predicted class $\hat{y} = \text{argmax}_c q_c$.
  - Validity mask: $valid = (conf \ge 0.70) \land u_{valid}$.
  - Pseudo update: If $valid.any()$, computes:
    $$\mathcal{L}_{\text{pseudo}} = \frac{\sum_{i \in valid} \text{CrossEntropy}(model(x_u^{\text{strong}})_i, \hat{y}_i)}{\sum_{i \in valid} 1}$$
    Gradients clipped to 5.0; both body (AdamW) and head (SGD) are stepped.
  - EMA update executes once at the conclusion of every step.

---

## 5. Data Sampling, Cropping, and Augmentation Protocols

### 5.1 Patient Sampling Protocol
- **Single-Patient per Role**: For each step, all 8 slices in a batch originate from one single patient.
  - Supervised index: $l_i = step \pmod 8$.
  - Unlabeled index: $u_i = step \pmod{72}$.
  - Audit index: $a_i = (l_i + 1 + \text{rng.integers}(7)) \pmod 8$. Guaranteed $a_i \ne l_i$.
- **Slice Slicing**:
  - Context: 3 adjacent Z slices $[z-1, z, z+1]$ clamped at cine boundaries $[0, Z-1]$.
  - Labeled slice choice: 50% probability to sample uniformly among slices with foreground present (`mask > 0 & mask < 4`), 50% uniformly among all slices in the frame.
  - Unlabeled slice choice: Frame $t$ and slice $z$ sampled uniformly at random.

### 5.2 Geometry-Only Padding and Cropping
- Implemented in `crop_native` ([`self_audit/data.py`](https://github.com/QuocKhanhLuong/Self-Audit/blob/0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca/self_audit/data.py)):
  - Target size: $256 \times 256$.
  - If spatial dimensions $H < 256$ or $W < 256$, symmetric replicate padding is applied to images.
  - Validity mask $valid$ is padded with `False`. Labeled mask is padded with ignore index `255`.
  - Crop coordinates $top \in [0, H_{\text{pad}} - 256]$ and $left \in [0, W_{\text{pad}} - 256]$ sampled uniformly at random.
  - Anatomy labels **never** determine crop coordinates.

### 5.3 Augmentations
- **Supervised Batch**:
  - Random horizontal flip: 50% probability via `aligned_flip(x, y, rng, valid)`.
- **Unlabeled Batch**:
  - Teacher receives unaugmented $x_u$.
  - Student receives strong intensity augmented $x_u^{\text{strong}}$:
    $$x_u^{\text{strong}} = \text{clamp}\left(x_u^\gamma + \epsilon, 0, 1\right), \quad \gamma \sim \text{Uniform}(0.8, 1.2), \quad \epsilon \sim \mathcal{N}(0, 0.03^2)$$
    No spatial augmentation on unlabeled slices to preserve teacher-student coordinate registration.

---

## 6. Source Closure, Hashes, and Lineage Evidence

### 6.1 Source Closure Verification
- **Historical Git Commit**: `0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca`
- **Source Closure Algorithm**: SHA256 of `filename:sha256\n` for all `self_audit/*.py` sorted alphabetically, appended by `train.py`.
- **Computed Value**: `3c7fa85f00d54a6e087af3cd6c7045351fb72d532f1069bf09d95f9cd72359b7` (exact match to historical protocol).

```
Constituent File Hashes at commit 0f54455b:
  __init__.py:        1ae542cceedefe1274a0ef07292460fffbb09e678bfc5198755caa9b51a32911
  control.py:         be31bdabbd5389e1989b31c6e7a416e8cf6a3172f72ae67c6d8c0d15a91dbe29
  data.py:            f6e53d46264fc170898a20ba42fe21f3dbd997e1d38e6a52de3e218e584b0260
  decoder_control.py: 41ac096a5894503cbaf4e8ad8019d314dbacd205b1261cabbf0013ed151f5d1f
  engine.py:          37aa4fd472b4dcd9de7e366063b73e3c6f35038636173baa2ea9a674097f8317
  evaluation.py:      abc805c1f94e31be948720882bbeba87fdeaed430bc2cbd219d50293d66c9898
  losses.py:          b3753b89447f3d83ef5cc40d89f288606104ef9e526cc338a1cd2f17cff93a73
  meta_window.py:     2182ac6fbbe64ec48c91541a8d0280ffec47b0df9b95fe5fe876ab8a05c8268d
  model.py:           80e3b11b46c11d4660aa13d99ecbf30640247c987f5425dcfe62aa1c1f90fc03
  window.py:          8b661ce007ef692c4575c418b90e89b59b7217f21f098c7ab19863de2e04fa72
  train.py:           3d0cc4fa64bb1b6482bbe8dd1f82a7d7691ad25137d87f5dc43011aa0585abb5
```

### 6.2 Lineage and Pretrained Hashes
- **DINOv3 Source Repository**: Clean checkout at commit `6876159a11b4df116f30f667f8c9888617df0751`.
- **DINOv3 Weights**: `dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` with SHA256 `21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54`. (Recovered physically from server backup).
- **Training Manifest**: `/workspace/Self-Audit/data/prepared/ACDC_L8_S42/train.json` with SHA256 `e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4`.
  - 8 Labeled patients: `patient007`, `patient010`, `patient011`, `patient024`, `patient026`, `patient041`, `patient077`, `patient099`.
  - 72 Unlabeled patients: `patient001`, `patient002`, ..., `patient100` (excluding `patient055` and `patient076`).
- **Evaluation Manifest**: `8f89d631bc67f7eccef94740866513fe6f794872e7cb6e61302b1663e17e8f9e` (50 patients, 100 frames).

---

## 7. Historical Evaluator: Raw Student vs. EMA Teacher

A critical question is whether the published evaluation numbers represent the Student or the EMA Teacher.

### 7.1 Code Evidence in Checkpoint Loading
In [`self_audit/engine.py`](https://github.com/QuocKhanhLuong/Self-Audit/blob/0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca/self_audit/engine.py):
```python
def load_checkpoint(folder, expected_sha256, device="cpu"):
    folder = Path(folder); mp = folder / "MANIFEST.json"
    if digest(mp) != expected_sha256:
        raise ValueError("checkpoint manifest hash mismatch")
    m = json.loads(mp.read_text())
    ...
    model, _ = build_model(m["config"]["model"], synthetic=m["synthetic"], window=m["config"].get("window"))
    payload = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["student"], strict=True)
    return model.to(device).eval(), m
```
`load_checkpoint` explicitly unpacks `payload["student"]`. The `payload["teacher"]` weights, while preserved inside `weights.pt`, are **never** unpacked into the model.

### 7.2 Evidence Across Manifests and Receipts
1. **Manifest Declaration**: `MANIFEST.json` contains `"prediction_model": "student"`.
2. **Evaluation Receipt**: `results/compact_acdc_l8_20261029/experiment_20261028/b8_recovery/test50_receipt/RECIPE.json` specifies `"prediction_model": "student"`.
3. **Teacher Quality Receipt**: `QUALITY_RECEIPT.json` specifies `"inference_state": "student"`.
4. **Delivery Summary**: `DELIVERY_RECEIPT.json` explicitly denotes `"inference_state": "raw_student"`.
5. **Archival Documentation**: `results/compact_acdc_l8_20261029/README.md` explicitly states:
   > *"The frozen KD teacher was the recovered **raw student B8** (TEST50 Macro4 0.880049, foreground Dice 0.841210)."*

**Conclusion**: All historical validation and TEST50 metrics correspond to the **raw student**, not the EMA teacher. The teacher weights were utilized exclusively as the target generator during semi-supervised training.

---

## 8. Unrecoverable Settings, Missing Payload, and Exact Resume Gaps

### 8.1 Missing Original Checkpoint Payload
The original physical checkpoint `weights.pt` with SHA256 `0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6` (manifest SHA256 `ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa`) is absent from the available Git/server/Drive inventories. The cause of the missing payload is not established.

### 8.2 Exact Resume Gaps in Historical Codebase
Mid-run resumption of the historical B8 run was fundamentally impossible due to several structural limitations in commit `0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca`:
1. **Manifest Contract**: `MANIFEST.json` explicitly pins `"resume": "NOT_SUPPORTED"`.
2. **Missing CLI Interface**: `train.py` at commit `0f54455b` contains no `--resume` or `--step` flags.
3. **Missing Engine Logic**: `engine.train()` unconditionally initializes `step = 0` and iterates over `range(c["steps"])`.
4. **Absence of CUDA RNG State Serialization**: `save_checkpoint` saves only `torch_rng` (CPU state) and `numpy_rng`. It never saves `torch.cuda.get_rng_state()` or `torch.cuda.get_rng_state_all()`. Consequently, bit-exact continuation of CUDA execution cannot be achieved.
5. **Unserialized Loop & Audit State**: `seen_supervised`, `seen_audit`, `seen_unlabeled`, `counts` dictionary, and accumulated timings are not preserved in the checkpoint payload.
6. **Filesystem Collision**: `steps.jsonl` is opened with mode `"x"` (exclusive creation), which immediately raises `FileExistsError` if output directory exists.

### 8.3 Authorized Retrain Protocol
Because exact mid-run resumption was structurally impossible and physical payload `0ae593...` was lost, the user explicitly authorized **exactly ONE clean 10,000-step full retrain** (`ema_full`, seed 42) to produce a new durable teacher checkpoint with verified local and Google Drive backup (`scripts/run_b8_reproduction.py`).

---

## 9. Verification Summary

- Repo sanity test: `python scripts/check_repo.py` -> **PASS** (`{"status": "PASS", "architecture": "dinov3_convnext_tiny_native_decoder_v2", "supervision": "semi_supervised; dense TRAIN subset incl audit; not no-GT"}`)
- Unit test suite: `python -m pytest -q` -> **905 passed, 2 warnings in 142.91s** (`tests/test_b8_backup_runner.py` and full suite passing).
- Source closure computation: Verified bit-exact match against `3c7fa85f00d54a6e087af3cd6c7045351fb72d532f1069bf09d95f9cd72359b7`.

Runtime limitation: the current server uses RTX 5000 Ada, Python 3.12.15 and torch 2.14.1+cu132. Core package versions match the archived B52 environment from the same historical setup; the complete B8 CUDA numeric/runtime fingerprint was not preserved. This is a fixed-recipe reproduction with a new identity, not a bitwise hardware/runtime replication or a guaranteed 88% result. No learning setting is inferred from current defaults.
