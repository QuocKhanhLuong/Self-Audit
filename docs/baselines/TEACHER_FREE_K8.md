# Chạy teacher-free A-3D trên ACDC k=8

Config: [`configs/candidate_a3d/teacher_free_acdc_k8.json`](../configs/candidate_a3d/teacher_free_acdc_k8.json).
Đây là bản copy nguyên byte từ `s42_teacher_free.json` của campaign đang dùng
code nền `498b56e979dc5e6699edb74e460d7682e1a83a14`; SHA-256 config là
`3d40df16d3be00a0535de7405ac37cd574371d1c1470ebcc0a6c658142d14ab8`.
Gói bàn giao chỉ thêm config/tài liệu, không thay model hoặc trainer.

## Arm này kiểm tra gì?

- Bỏ teacher MedNeXt vùng/biên và query proposals từ teacher: `use_teacher=false`,
  `freeze_teacher=true`, `teacher_topk=0`. Không cần checkpoint hay source MedNeXt;
  block `model` trong config được giữ nguyên để truy xuất nguồn gốc nhưng không
  được khởi tạo khi `use_teacher=false`.
- Vẫn học segmentation, phép đo REGION/BOUNDARY và K1 utility từ nhãn TRAIN L8.
- Vẫn dùng student self-EMA consistency trên ảnh TRAIN U72, không đọc mask U.
- Giữ encoder/student, query bank, code-only decoder, giá trị 8 bit và trần mã
  4096 bit của full. Actual code bytes/query count phải lấy từ báo cáo.
- Seed 42, 100 epochs, native 3D FP32, micro-batch 1, decoder chunk 16384.
  Không có hai giai đoạn teacher pretrain/warm-up cho arm này.

## Dữ liệu bắt buộc

Dùng prepared cache và manifest của split ACDC đã khóa: TRAIN80, DEV18, TEST50.
L8 gồm `patient007`, `patient010`, `patient011`, `patient024`, `patient026`,
`patient041`, `patient077`, `patient099`; 72 TRAIN patients còn lại chỉ có ảnh.
`k=8` là số bệnh nhân có nhãn; file config không tự chọn hay khóa tám IDs này,
vì loader lấy ngân sách từ manifest.

SHA-256 của manifest gốc trên server:

- TRAIN k8: `e524c7531079996232fe71cdfe982a46a5950d49d91a8203eee396b0e67434a3`.
- DEV18: `ba46934afe8ea5c47dd798019bee233c97c98320b358e2d40165e87a651dc7df`.

Dataset/cache/manifest runtime không được đóng trong Git. Khi chuyển cache sang
máy khác, giữ nguyên bytes của arrays, patient IDs, geometry và split. Nếu phải
đổi absolute paths trong manifest, lưu bản gốc cùng receipt relocation và SHA
mới; không dùng SHA cũ cho JSON đã sửa. Không tự chia lại tập dữ liệu hoặc mở
TEST GT. Class mapping là BG=0, RV=1, MYO=2, LV=3.

## Lệnh chạy riêng student

Trong môi trường đã cài dependencies/CUDA phù hợp, từ root repository, thay ba
đường dẫn sau. `RUN_OUT` phải là thư mục mới, tách khỏi prepared cache.

```bash
set -euo pipefail
TRAIN=/absolute/path/to/k8/train.json
DEV=/absolute/path/to/development.json
RUN_OUT=/absolute/path/to/runs/acdc-k8-teacher-free-s42

# Kiểm tra ngân sách và IDs trước khi chạy; không đọc mask.
python - "$TRAIN" "$DEV" <<'PY'
import json, sys
train = json.load(open(sys.argv[1]))
dev = json.load(open(sys.argv[2]))
expected = {"patient007", "patient010", "patient011", "patient024",
            "patient026", "patient041", "patient077", "patient099"}
assert train["dataset"] == "acdc" and train["role"] == "training"
assert train["label_budget"] == 8 and train["label_seed"] == 42
assert {r["id"] for r in train["labeled"]} == expected
assert len(train["labeled"]) == 8 and len(train["unlabeled"]) == 72
assert all(not r.get("masks") for r in train["unlabeled"])
assert dev["role"] == "development" and len(dev["patients"]) == 18
train_ids = {r["id"] for r in train["labeled"] + train["unlabeled"]}
assert not train_ids.intersection(r["id"] for r in dev["patients"])
print("ACDC L8/U72 + disjoint DEV18: PASS")
PY

# Hash các manifest thực sự được loader sử dụng, kể cả bản relocate đã ghi receipt.
TRAIN_SHA=$(sha256sum "$TRAIN" | cut -d' ' -f1)
DEV_SHA=$(sha256sum "$DEV" | cut -d' ' -f1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py \
  --config configs/candidate_a3d/teacher_free_acdc_k8.json \
  --data "$TRAIN" --data-sha256 "$TRAIN_SHA" \
  --validation-data "$DEV" --validation-data-sha256 "$DEV_SHA" \
  --out "$RUN_OUT" --device cuda --allow-full-run
```

Chạy lệnh cuối trong tmux để không phụ thuộc kết nối SSH. Không truyền
`--teacher-checkpoint`; không cần đợi teacher k=8. Checkpoint định kỳ đang tắt
(`checkpoint_every_epochs=0`), final checkpoint được lưu cuối 100 epochs;
không có resume một job bị ngắt giữa chừng.

## Artifacts cần trả lại

Giữ `CONFIG.json`, `STARTUP.json`, `updates.jsonl`, `epochs.jsonl`,
`metrics.csv`, `metrics.jsonl`, `REPORT.json`, thư mục `final/` và student export
nếu được tạo. Ghi Git SHA, manifest/config/checkpoint hashes, GPU/CUDA/PyTorch
và thời gian thực đo. DEV chỉ báo cáo, không dùng chọn checkpoint/early stopping;
TEST chưa chạy. Không xem kết quả smoke là kết quả ACDC.

Đối chiếu với full k=8 cùng protocol. So sánh latency/VRAM cần đo lại các arm
trên cùng GPU và môi trường. Hai ablation còn lại là `fixed_queries` (bỏ policy
chọn query thích nghi) và `global_only_queries` (zero local features, giữ nguyên
query bank); chúng cần cùng checkpoint teacher k=8 sau pretrain/warm-up với full.
