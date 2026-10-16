# Handoff hiện tại: train student Astra/DINOv3 trên RTX 4070

Ngày cập nhật: 2026-10-16. Branch: **`codex/astra-student-training-20261016`**.
Mục tiêu của Codex trên 4070: xác minh đúng backbone mới, chạy pilot có giới hạn,
kiểm tra checkpoint rồi mới thực hiện budget train đã chọn. Agent tạo handoff
này chỉ hoàn thiện code và kiểm thử CPU; **chưa load official DINO weights,
chưa chạy patient training hoặc GPU, chưa có kết quả hội tụ/accuracy**.

## 1. Chọn đúng code và entrypoint

Entry mới là **`scripts/train_astra_student.py`**. Nó dựng
`LayerGuidedStudent` + official DINOv3 ConvNeXt-Tiny và chạy AdamW thật.
Các entrypoint historical/stable-guidance vẫn dùng student cũ; không dùng chúng
để tuyên bố đang train kiến trúc này. Đọc [policy và các deviation](../../experiments/layer_guided_student_v1/training_v1/README.md).

Từ repo trên 4070, xem git status/worktree và GPU jobs hiện có. Tạo worktree riêng
ở đường dẫn chưa tồn tại; giữ nguyên main, baseline, dữ liệu và mọi run cũ.
Ví dụ sau dùng shell Bash và `rtk` theo quy ước repo:

```bash
rtk git status --short --branch
rtk git worktree list
rtk git fetch origin codex/astra-student-training-20261016
rtk git worktree add --detach ../Self-Audit-astra-4070-run FETCH_HEAD
cd ../Self-Audit-astra-4070-run
rtk git rev-parse HEAD
rtk git status --short
rtk proxy nvidia-smi
```

Nếu worktree đích đã tồn tại, chọn tên mới; không reset/xóa worktree có sẵn.
Ghi HEAD vào receipt của run, giữ checkout này cố định trong suốt training/reload.
Không merge vào main. Không dừng job của người khác để lấy VRAM.

## 2. Xác định input thật trên máy

Chọn Python hiện có với CUDA torch; kiểm tra import torch/torchvision/numpy,
pytest và dependency của official DINO checkout. Phiên bản CPU đã kiểm thử nằm
trong [verification receipt](../../reports/astra_training_integration_20261016/README.md).
Không cài bản CPU torch đè lên môi trường GPU đang dùng. Ghi `pip freeze`, torch,
CUDA runtime, GPU name, driver và git HEAD vào thư mục receipt ngoài output của
runner. Ví dụ biến bên dưới cần được điền bằng đường dẫn có thật:

```bash
repo="$PWD"
experiment="$repo/experiments/layer_guided_student_v1"
python_bin=/absolute/path/to/cuda/environment/bin/python
bank=/absolute/path/to/original/bank
approval=/absolute/path/to/reviewed-dinov3-approval.json
approval_sha=EXTERNALLY_REVIEWED_RECEIPT_SHA256
bank_sha=f5a7b51328cc877ebdc4e70cbf7f90bebfe6da672597082beb5ae270f3e98665
receipt=/absolute/path/to/new/astra-4070-receipts
rtk proxy mkdir "$receipt"
rtk git rev-parse HEAD > "$receipt/git-head.txt"
rtk proxy "$python_bin" -m pip freeze > "$receipt/pip-freeze.txt"
rtk proxy nvidia-smi > "$receipt/nvidia-smi.txt"
rtk proxy "$python_bin" -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
```

**Bank:** provenance hiện biết trỏ tới
`${REPO_ROOT}/runs/stable_guidance_main513ddda_acdc_4070_20261015_210312/run/bank`.
`${REPO_ROOT}` là repo root của run cũ, không nhất thiết worktree mới. Dùng bank
ORIGINAL có `FROZEN.json` và patient `.npz`, đối chiếu hash cố định ở trên theo
`reports/stable_guidance_main513ddda_b12_20261016/completed_run/BANK_REUSE.json`.
Bản trong `reports/.../run/bank` chỉ là metadata đã sanitize, hash khác và thiếu
arrays; không dùng thay ORIGINAL. Không sửa manifest để vượt kiểm tra.

**DINO:** dùng official source checkout sạch, commit được pin, origin chính thức
và weights **`dinov3_convnext_tiny`** đã được cấp quyền trên host. CineMA bank
không chứa DINO weights. Receipt phải có các trường sau; placeholder không hợp lệ:

```json
{
  "schema": "layer_guided.backbone_approval.v1",
  "approved": true,
  "model": "dinov3_convnext_tiny",
  "lineage": "official_self_supervised_dinov3",
  "repo": "/absolute/path/to/reviewed/dinov3",
  "weights": "/absolute/path/to/reviewed/weights.pth",
  "weights_sha256": "REVIEWED_64_HEX_DIGEST",
  "source_commit": "REVIEWED_40_HEX_COMMIT",
  "review_record": "source and weight provenance review record"
}
```

Weights phải ở ngoài source checkout để checkout sạch. Đối chiếu nguồn/checksum
rồi pin SHA256 của receipt; tính hash một file chưa kiểm tra không chứng minh
nguồn pretrained. Loader gọi `torch.hub.load(source='local', pretrained=True,
weights=...)`; không tải weights hoặc fallback random. Nếu thiếu quyền/weights/
bank thật, ghi rõ artifact thiếu. Không tự tạo receipt giả hay chấp nhận điều
khoản thay người dùng. Nguồn chính thức: [DINOv3](https://github.com/facebookresearch/dinov3).

## 3. Software và official-backbone preflight

Tất cả output dưới đây phải là đường dẫn mới. Verifier chỉ dùng dữ liệu tổng hợp.

```bash
rtk proxy "$python_bin" "$experiment/verify_port.py" \
  --out "$receipt/software-parity"
rtk proxy env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$experiment:$repo/src" \
  "$python_bin" -m integration.cli \
  --recipe "$experiment/recipe.json" \
  --approval "$approval" --expected-approval-sha256 "$approval_sha" \
  --out "$receipt/official-backbone-preflight" --device cuda
```

Preflight thứ hai load weights thật và forward/backward input tổng hợp 65×97;
không có optimizer/patient step. Receipt phải xác nhận official weights hash
và CUDA. Input nhỏ này chưa chứng minh native patient cine sẽ vừa VRAM.

## 4. Pilot 32 update trên TRAIN bank

```bash
pilot_out=/absolute/path/to/new/astra-dinov3-pilot32
set -o pipefail
rtk proxy env PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$python_bin" scripts/train_astra_student.py \
  --bank "$bank" --bank-manifest-sha256 "$bank_sha" \
  --approval "$approval" --approval-sha256 "$approval_sha" \
  --recipe "$experiment/training_v1/pilot.json" \
  --out "$pilot_out" --device cuda 2>&1 | tee "$receipt/pilot-console.log"
```

Runner tạo output mới sau kiểm tra input/model. Không mkdir trước `pilot_out`.
Pilot dừng sau 32 optimizer updates; có thể mới đi qua một phần của một patient,
không phải 32 epochs và không phải kiểm tra toàn bộ phân bố kích thước TRAIN.

Kiểm tra `STARTUP.json` ngay khi chạy:

- `class = LayerGuidedStudent`, `architecture = layer_guided_convnext_tiny_v1`;
- `backbone = dinov3_convnext_tiny`, `pretrained_loaded = true`, đúng weights hash;
- `device = cuda`, đúng bank/config/source hashes; parameter count được ghi thật
  (reference random full-size smoke có 32,890,558 params, không phải CNN cũ ~351k).

Pilot chỉ đạt software/run gate khi exit code 0, có `REPORT.json` với
`receipt_status=SUCCESS`, `status=COMPLETED_MAX_STEPS`, đúng 32 updates và 32 dòng
metrics, `semantic_updates>0`, loss/gradient hữu hạn, checkpoint load lại thành
công. Kiểm tra VRAM, thời gian/step và RAM từ host. Không coi các gate này là
chứng nhận hội tụ hoặc chất lượng anatomy.

`metrics.jsonl` có feature loss, semantic CE, target entropy, semantic KL,
gradient norm trước clip, elapsed time và peak allocated CUDA bytes. CE có sàn
entropy; đọc KL và CE cùng target entropy. Các batch/patient khác nhau không tạo
ra một đường loss của tập cố định. Không đổi Q sang Z hoặc bỏ UNKNOWN trong
training chỉ để làm loss/coverage đẹp hơn.

Nếu OOM, nonfinite, thiếu semantic update hoặc provenance check fail: giữ nguyên
logs/partial output, ghi nguyên nhân, dừng trước full run. Không có `REPORT.json`
thành công nghĩa là incomplete, kể cả đã có checkpoint epoch. Không tự giảm
resolution, đổi target, bật AMP, thay model hay dùng weights random. Đề xuất
thay đổi phải thành recipe/code version mới có review và pilot riêng.

## 5. Reload checkpoint thật, image-only smoke

Đoạn này đọc checkpoint của pilot và chạy một forward tổng hợp; không đọc GT,
không export/evaluate bệnh nhân. Hash lấy từ receipt đã lưu độc lập của pilot.

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$experiment:$repo/src" \
  "$python_bin" - "$pilot_out" "$approval" "$approval_sha" <<'PY'
import json, sys
from pathlib import Path
import torch
from integration.artifacts import sha256
from training_v1.checkpoint import load_checkpoint

run = Path(sys.argv[1])
approval_path = Path(sys.argv[2])
assert sha256(approval_path) == sys.argv[3]
report = json.loads((run / "REPORT.json").read_text())
assert report["receipt_status"] == "SUCCESS"
assert report["optimizer_steps"] == 32 and report["semantic_updates"] > 0
model, metadata = load_checkpoint(
    run / "final", report["final_manifest_sha256"],
    json.loads(approval_path.read_text()), device="cuda",
)
assert metadata["status"] == "PSEUDO_SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED"
with torch.no_grad():
    outputs = model(torch.zeros(1, 3, 65, 97, device="cuda"))
assert tuple(outputs["logits"].shape) == (1, 4, 65, 97)
assert torch.isfinite(outputs["logits"]).all()
print(json.dumps({"status": "PASS_RELOAD_SOFTWARE_ONLY", "checkpoint_sha256": report["final_manifest_sha256"], "architecture": metadata["config"]["architecture"], "semantic_updates": metadata["lineage"]["semantic_updates"]}))
PY
```

Lưu stdout/exit code của bước này vào receipt. Loader dùng schema mới
`layer_guided.training_checkpoint.v1`; checkpoint `last.pt` của student cũ hoặc
schema representation-only không tương thích. Nạp lại chỉ hỗ trợ inference;
optimizer snapshot để audit, **chưa hỗ trợ resume chính xác**.

API `predict_cine` trả `(dense, gated, confidence, entropy, normalization_receipt)`.
Dense dùng argmax 4 lớp; gated giữ 255 khi reject. Với raw NIfTI, phải gọi
`self_audit_guidance.io.read_cine(path)` trước rồi đưa array TZYX và
`metadata['image_sha256']` vào `predict_cine`. Không trực tiếp min/max raw NIfTI
vì sẽ khác preprocessing TRAIN. Không tự chạy VAL/TEST hoặc thêm GT vào loader.

## 6. Full recipe sau pilot

`training_v1/full50.json` có 50 epochs, `max_steps=null`, batch size 1, seed 42,
AdamW lr 3e-4, weight decay 1e-4, feature/semantic weights 1/1, clip norm 5.
Đây là config khởi đầu, chưa được chứng minh hội tụ trên 4070. Ước lượng runtime
và disk từ pilot, xem số context của TRAIN bank trước khi cam kết chạy đầy đủ;
pilot chưa đo các patient lớn nhất. Lưu config/budget được chọn trước update đầu.

Lệnh dưới đây dành cho Codex trên 4070 sau các kiểm tra trên. Full run **khởi tạo
mới từ cùng pretrained weights**, không resume pilot. Output/log mới hoàn toàn:

```bash
full_out=/absolute/path/to/new/astra-dinov3-full50
set -o pipefail
rtk proxy env PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$python_bin" scripts/train_astra_student.py \
  --bank "$bank" --bank-manifest-sha256 "$bank_sha" \
  --approval "$approval" --approval-sha256 "$approval_sha" \
  --recipe "$experiment/training_v1/full50.json" \
  --out "$full_out" --device cuda 2>&1 | tee "$receipt/full-console.log"
```

Không dùng test/val để chọn checkpoint, threshold hoặc epoch. Runner không tự
evaluate/export. Chưa có giao thức đánh giá matched mới trong task này.

## 7. Giới hạn và kết quả cần bàn giao

Nhánh exact split head hiện học **feature + coarse logits**. Native-detail chưa
có objective nên detail output vẫn zero; không gọi kết quả này là đã học fine
boundary của kiến trúc đầy đủ. Guard/refiner tắt. Thay đổi các yếu tố đó là một
thử nghiệm riêng có version. Baseline batch12 cũ và run batch1 mới chưa phải
matched comparison.

Giữ toàn bộ `CONFIG.json`, `STARTUP.json`, `metrics.jsonl`, `REPORT.json`,
checkpoint manifests/payloads và receipts môi trường/preflight/reload. Checkpoint
50 epochs gồm optimizer snapshot từng epoch có thể chiếm hàng chục GB; kiểm tra
disk trước khi chạy. Không tự xóa run cũ để giải phóng chỗ.

Khi báo cáo, phân biệt `SOFTWARE_PASS`, `OFFICIAL_BACKBONE_LOADED`,
`PATIENT_PILOT_COMPLETED`, `FULL_TRAIN_COMPLETED`, và `QUALITY_NOT_EVALUATED`.
Ghi HEAD, GPU, bank/weights/config hashes, actual epochs/steps/semantic updates,
memory/runtime, failure/partial status nếu có. Không push patient arrays, raw
images hoặc model weights vào git. Chỉ công bố receipt đã kiểm tra và được phép.

Prompt ngắn cho Codex trên 4070:

> Fetch `codex/astra-student-training-20261016` vào worktree riêng. Đọc và làm theo
> `docs/prompts/ASTRA_TRAIN_4070_LATEST.md`: dùng entrypoint mới, xác minh ORIGINAL
> bank và official DINO weights, chạy preflight + pilot32 + reload. Ghi kết quả
> thực tế và ước lượng budget trước full50. Giữ baseline/reference/run cũ, không
> merge main, không auto VAL/TEST hoặc biến UNKNOWN thành BG.
