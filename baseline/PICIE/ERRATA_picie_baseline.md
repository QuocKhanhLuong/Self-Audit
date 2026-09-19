# Errata — Báo cáo Phân tích Baseline PiCIE

**Bản gốc đính chính:** `picie_baseline.docx` (bản cuối, có hiệu lực)
**Ngày tạo errata:** 2026-09-19, sau khi hoàn thành Phase 2 implementation.

---

## 1. Tên file không tồn tại — `acdc_picie_wrapper`

### Đính chính chính thức (từ bản cuối docx)

**Mục 4, điểm 4 — nguyên văn:**
> Đã sửa wrapper (acdc_picie_wrapper) và collate_eval để truyền thông tin
> case_id, slice_idx, và spacing. Tại vòng lặp evaluate, code đã được
> nâng cấp để gom nhóm các slice theo bệnh nhân (group-by case_id),
> dọn dẹp bộ nhớ an toàn (chống leak state), và sắp xếp chính xác trục Z
> bằng np.argsort(slice_idx) để khôi phục khối 3D Volume trước khi tính
> Metric.

**Thực tế:** Không có file `acdc_picie_wrapper` (hay bất kỳ biến thể nào
của tên này) trong repo PiCIE — cả trước lẫn sau implementation. Các file
thực đã tạo:

- `data/acdc_train_dataset.py` — class `TrainACDC` (training dataset, 2.5D)
- `data/acdc_eval_dataset.py` — class `EvalACDC` (eval dataset + metadata)
- `data/mnms_eval_dataset.py` — class `EvalMnMs` (M&Ms eval + label remap)

Phần `collate_eval` là có thật (đã sửa trong `utils.py`). Phần metadata
truyền `case_id, slice_idx, spacing` và reconstruct 3D volume cũng đã
được implement — nhưng nằm trong `eval_picie.py` và `data/acdc_eval_dataset.py`,
không phải trong file mà báo cáo đặt tên.

### Ghi chú lịch sử audit

Claim tương tự — cùng tên file giả `acdc_picie_wrapper` — đã từng xuất hiện
ở **bản nháp PDF** (`picie_baseline.pdf`, 2026-09-17), tại hai vị trí khác
với cách diễn đạt khác:

- **Điểm 3:** "Luồng Adapter đã sẵn sàng tại `acdc_picie_wrapper.py` và
  `adapters/` để bọc dữ liệu, tuân thủ ngắt 2.5D."
- **Điểm 5:** "Đã kiểm tra `acdc_picie_wrapper.py`"

Việc tên file này xuất hiện xuyên suốt từ bản nháp sang bản cuối — với cách
diễn đạt thay đổi nhưng tên giữ nguyên — cho thấy đây không phải lỗi đánh
máy nhất thời mà là một chi tiết sai bị mang qua nhiều vòng chỉnh sửa báo
cáo mà không ai kiểm chứng lại với code thật.

---

## 2. Thiếu thông tin M&Ms label remapping

**Mục 3 — nguyên văn:**
> Đồng nhất nhãn về 4 lớp: {0: Background, 1: RV, 2: MYO, 3: LV}

**Thiếu:** Báo cáo không đề cập rằng M&Ms sử dụng **thứ tự nhãn khác**
so với ACDC:

| Label | ACDC        | M&Ms (raw)  |
|-------|-------------|-------------|
| 0     | Background  | Background  |
| 1     | RV          | **LV**      |
| 2     | MYO         | MYO         |
| 3     | LV          | **RV**      |

Đã áp dụng remapping `{0:0, 1:3, 2:2, 3:1}` trong
`data/mnms_eval_dataset.py` (hằng số `MNMS_TO_ACDC`).

Xác nhận từ:
- Technical Specification PDF (Self-Audit)
- Config `configs/self_audit_acdc_to_mnms.yaml` (field `raw_to_acdc`)
