# Teacher-memory pilot — delivery 2026-10-24

Đã triển khai NativeStudent + fixed-window + `ema_full` và các cấu hình A0–A6 trên nhánh `feat/native-teacher-memory-pilot`. Luna viết code, Astra review độc lập và kiểm tra lại các sửa lỗi. Phạm vi hoàn thành là code, kiểm thử, smoke và profile ngắn; **chưa có kết luận patient Dice, lợi ích pairing hay novelty**.

## Git và bảo toàn công việc

- Base đã fetch: `9341cc81fc046d61135b928f5ac51ebed19c19ed`, chứa commit người dùng kiểm tra `0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca`.
- Server window-pilot không có commit chưa push hay sửa đổi chưa commit tại thời điểm kiểm tra. Các commit mới trên nhánh window được giữ nguyên; không dùng kết quả TEST cũ để chọn thiết kế hay batch.
- 236 mục dirty/untracked của checkout gốc được đối chiếu lại SHA256: không thay đổi. Worktree window local vẫn ở `0f54455`; hai checkout cũ trên server vẫn sạch tại SHA ban đầu. Watcher cũ vẫn tồn tại; không dừng job của người dùng. [Local receipt](PRESERVATION.json), [server receipt](server/SERVER_PRESERVATION.json).
- Không reset, clean, force-push hoặc merge main. Nhánh pilot chỉ thay hai module v2 hiện hữu (`model.py`, `engine.py`); các module, configs, tests và tài liệu pilot được thêm riêng. Các recipe baseline/fixed/meta cũ giữ nguyên.

## Code và review

[Runbook](../../docs/TEACHER_MEMORY_PILOT.md) ghi công thức, class map, API, controls và lệnh chạy. Memory lấy h sau `refine[0]`, dùng bank bất biến, detached gate, W và sáu alpha; thêm **3078 tham số học** trên DINO NativeStudent. A2/A3 dùng cùng vectors và số tham số. A2 marginalize độc lập hai đầu; A4 xáo correspondence cùng class-pair; A5 chỉ đọc frozen A0; A6 thay hướng feature bằng random chuẩn hóa.

Teacher offline dùng đúng checkpoint segmentation M&Ms SAX seed 0. Adapter full-FOV image-only lưu/invert physical transforms, không gọi đường crop theo GT LV. A1 thêm external soft pseudo-CE trong transaction U vốn đã đủ điều kiện EMA; không tạo update riêng nếu chỉ external target đủ điều kiện. Đây là comparator A1 có điều kiện, được ghi rõ trong protocol. Student export nhúng bank và bỏ đường dẫn cache/bank ngoài.

Astra phát hiện và yêu cầu sửa các lỗi về padding stride-4, support khi invert geometry, ghép endpoints qua lớp thứ ba, provenance A5, ghi A1 cache theo dòng và CLI export khi chạy ngoài repo. Các receipt giữ cả phát hiện ban đầu và closure; kiểm tra hash cuối xác nhận code được review chính là code chạy:

- [Memory/integration review](REVIEW_MEMORY_INTEGRATION.md): ACCEPT, gồm disabled-path đối chiếu base và export CLI.
- [Adapter/builder review](REVIEW_TEACHER_ADAPTER.md): APPROVED FOR BOUNDED ENGINEERING RUN.
- [Profile review](REVIEW_PROFILE.md): kiểm tra timing, update counters, budget và phạm vi manufactured data.
- [Prior-art audit](../../docs/TEACHER_MEMORY_PRIOR_ART.md): ProC-KD, IC-KD, BRPG và Dual-Teacher SAM+prototype MICCAI 2026; novelty **UNCONFIRMED**.

## Kiểm thử thực thi

| Nơi chạy | check_repo.py | pytest |
| --- | --- | --- |
| Local CPU, Python 3.11.16 / Torch 2.14.0 | PASS | 148 passed, 1 warning, 10.78 s |
| Pro4000 server, Python 3.12.15 / Torch 2.14.1+cu132 | PASS | 148 passed, 1 warning, 22.59 s |

Warning là deprecation `torch.jit.script` ở test v2 hiện hữu. Suite kiểm tra disabled parity, class map, physical geometry, forbidden mask reads, bank immutability, teacher không có gradient, projection/gamma gradients, missing pairs, shuffle control, checkpoint/export và inference không cần external teacher. [Local logs/receipt](FINAL_LOCAL_VALIDATION.json), [server logs/receipt](server/SERVER_VALIDATION.json). Đây là kiểm chứng phần mềm; pytest chạy trên server không đồng nghĩa mọi test dùng CUDA.

## Teacher và bank thật

Strict load checkpoint thật thành công. Hook-on/off và `head(feature)` đều có max absolute logit delta **0.0**. Smoke dùng input manufactured oblique 32×32×3, padded tile 192×192×16: inference **0.661 s**, peak allocated **1.04 GiB**, reserved **1.22 GiB**. Thời gian wall gồm import/load **8.83 s**. Đây không phải tốc độ cho một volume bệnh nhân đầy đủ. [Receipt](server/teacher-smoke.json).

Đã dựng bank teacher từ **3/80 bệnh nhân TRAIN, mỗi người 1 timepoint** trong **26.47 s wall**. Tổng 90 timepoints trong receipt là của ba bệnh nhân được chọn, không phải toàn bộ TRAIN. Theo thứ tự `(BG,RV),(BG,MYO),(BG,LV),(RV,MYO),(RV,LV),(MYO,LV)`, số pairs là **[8,8,0,8,0,8]**: 32 pairs / 64 vectors, D=32. BG–LV và RV–LV không có support; không điền lớp giả. Không tạo A1 cache. [Build receipt](server/bounded-real-bank/BUILD_RECEIPT.json), [bank inspection](server/BANK_INSPECTION.json).

A5 cũng đã chạy builder thật trên cùng ba timepoints, dùng checkpoint A0 mới sau hai bước và không có external-teacher input/environment: **10.29 s wall**, pairs **[0,0,0,0,0,0]**. Bank rỗng được giữ nguyên. Đây là witness cho provenance/luồng xử lý A5, **không phải bank phù hợp cho so sánh chất lượng hoặc vector-budget-matched A3/A5**. [A5 receipt](server/bounded-student-bank/BUILD_RECEIPT.json).

## Profile Pro4000

GPU: NVIDIA RTX PRO 4000 Blackwell, 24,467 MiB. Dùng backbone DINO và weights thật; inputs, labels và bank dày 8 pairs/group trong profile là manufactured. Teacher ngoài không tham gia. Mỗi arm có 1 transaction warmup + 2 đo; mỗi transaction gồm supervised update, pseudo update và một EMA. Batch L=U, không gradient accumulation. Denominator là **L+U processed exposures**, không phải unique patients hay số lần forward teacher.

Batch thăm dò, examples/s:

| Batch mỗi nhánh L/U | A2 | A3 |
| --- | --- | --- |
| 4 | 55.26 | 57.16 |
| 8 | 66.38 | 66.54 |
| 16 | 71.28 | 70.72 |
| 32 | 71.48 | 73.51 |

Chọn **batch 32 chung** trong các batch đã đo để chạy profile cuối và real TRAIN smoke. Khoảng đo rất ngắn; không suy ra tốc độ tối ưu tuyệt đối hay khác biệt có ý nghĩa thống kê.

| Arm | L+U examples/s | Peak allocated GiB | Peak reserved GiB |
| --- | --- | --- | --- |
| A0 | 75.23 | 12.21 | 14.83 |
| A1 | 75.56 | 12.21 | 14.88 |
| A2 | 71.89 | 12.37 | 15.19 |
| A3 | 72.99 | 12.25 | 14.88 |
| A4 | 73.07 | 12.25 | 14.88 |
| A5 | 73.74 | 12.25 | 14.88 |
| A6 | 74.38 | 12.25 | 14.88 |

[All-arm receipt](server/matched-b32-all-arms/PROFILE.json): mỗi arm có 128 exposures đo, cùng init/input/crop 256 và update counts; tổng cả warmup là 3 supervised + 3 pseudo, 6 AdamW/body steps, 6 head steps, 3 EMA updates. Peak gồm construction/warmup; timing CUDA đã synchronize và có overhead gradient diagnostics ở memory arms. A1 chỉ đo external-loss operator với target manufactured; A5 trong profile chỉ đo memory operator với bank manufactured. Chi phí I/O dữ liệu, cache và checkpoint không nằm trong throughput này. Teacher offline có receipt riêng ở trên. Peak A2 khoảng **16.31 GB = 15.19 GiB reserved**, nằm trong VRAM hiện có; process RSS/long-run RAM peak **NOT MEASURED**.

## Real TRAIN smoke và deploy

A0 và A3 mỗi arm chạy `train.py` thật **2 steps**, batch 32, crop 256, warmup 0, pseudo threshold 0 để buộc đi qua U path. Đây là override phục vụ smoke, không phải recipe nghiên cứu. Mỗi arm ghi nhận **2 supervised / 2 pseudo / 2 EMA**, không geometry update và không label read ngoài L budget. Wall lần lượt **14.10 / 14.10 s**, engine elapsed **4.45 / 4.55 s**. Không đánh giá VAL/TEST. [Commands/receipt](server/REAL_TRAIN_SMOKE.json), [A0 report](server/real-train-smoke-A0/REPORT.json), [A3 report](server/real-train-smoke-A3/REPORT.json).

Export A3 chỉ chứa payload `student`, có bank buffers nhúng. Khi chặn mọi Python file-open vào teacher assets, cache và bank gốc, export vẫn load/infer được trên một TRAIN slice 256×216: **logits khớp chính xác (max delta 0.0)**; không có attempt truy cập bị cấm. Single inference đã synchronize **0.01453 s**, không phải benchmark ổn định. [Deployment receipt](server/TEACHER_FREE_DEPLOYMENT.json). Student vẫn cần pinned DINO architecture/source và backbone-construction assets; chưa export ONNX/TorchScript độc lập.

## Hashes và nơi lưu

| Artifact | SHA / revision |
| --- | --- |
| Executed v2 source digest | `7f5856216d59bc787633f8a703fa362cf83582427b830ab1ce09c15aeb55e8a9` |
| CineMA source | `c10daa1d93f0ea28d8b9ad9206b0f673d25805c1` |
| Hugging Face revision | `b1251ee50423bceeca84c080782fc3bc7756dea6` |
| Segmentation weights | `ee11e5392cc89f0a03f7cd2c171e87de0771b839d414835cb59044077679acb8` |
| Teacher config | `aab63af2dc5b1592fd76faa209b37d84b846c7bcf09639a6fb8f7469a31fbea1` |
| Bounded teacher bank file | `0fc4c558e944a9204e2cac7a62247acef16a22e200cc3b3fe42f5229f23278b6` |
| Bounded A5 bank file | `39e6e435762ae72acc1a4fe1ab7c0b27f113d2ebe12e47583086e6391216ab87` |
| TRAIN manifest | `e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4` |
| DINO weights | `21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54` |
| A3 export manifest | `7c24aaa90d8708a8a8add67bca9881134e3da5ef5663afef8564b6a8b7e9b2c4` |

Source digest covers `self_audit/*.py` + `train.py`; per-file hashes and scripts are also pinned in receipts and the delivery inventory. NPZ file SHA and canonical content SHA are distinct and are labeled separately in training reports. Teacher weights, NIfTI images, banks and checkpoint tensors stay on server outside Git. Only reports/configs/logs are copied into this delivery.

Server worktree: `/root/Self-Audit-teacher-memory-pilot`; assets: `/root/self-audit-teacher-memory-assets`; outputs: `/root/self-audit-teacher-memory-execution`. Additional MONAI/einops/timm/safetensors dependencies are isolated under `assets/deps`. Actual NumPy 2.5.4 differs from upstream's pinned environment; this delivery verifies the executed environment, not a full upstream-environment reproduction.

## NOT RUN và bước tiếp theo

**NOT RUN:** complete 80-patient/all-timepoint teacher bank; full real A1 cache and A1 training with that cache; useful full A5 bank; seven-arm patient training; long training; patient Dice/boundary efficacy; seed statistics; superiority A3>A2/A4; subject-overlap resolution; publication-level novelty confirmation.

Teacher public lineage reports UKB pretraining + M&Ms fine-tuning; ACDC subject overlap remains **UNKNOWN**. Hash/strict-load verification cannot establish unseen-subject provenance. Full-FOV image-only normalization deliberately differs from upstream GT-crop statistics. No ACDC teacher was substituted. See [provenance audit](../../docs/TEACHER_PROVENANCE.md).

The [runbook](../../docs/TEACHER_MEMORY_PILOT.md#server-commands-for-subsequent-work) contains full extraction and subsequent training commands. A1–A6 config files are guarded templates until actual complete assets are bound; they are not claimed ready for a full experiment. Use common batch **32**, budgets, schedule and seed for matched runs; log actual accepted-U update counts, not just configured steps. Any mismatch must be reported and resolved before scientific comparison. Freeze the full-bank hashes and A0 snapshot rule before outcomes. Do not reuse the bounded smoke bank as a full experimental bank without declaring that factor.

For an exact **two-step engineering repeat** with the already bound A3 smoke config (not a long train), use a new output path:

```sh
cd /root/Self-Audit-teacher-memory-pilot
/venv/main/bin/python train.py \
  --config /root/self-audit-teacher-memory-execution/smoke-configs/A3.json \
  --model-config /root/self-audit-pilot-artifacts/BACKBONE.json \
  --data /root/self-audit-pilot-artifacts/data/cache/L8_S42/train.json \
  --data-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --device cuda --out /root/self-audit-teacher-memory-execution/A3-repeat-new
```

If later A3 does not improve on class memory A2 and correspondence-shuffled A4 under a prespecified matched comparison, retain the negative result and drop the pairing claim. These comparisons have not yet been run for patient quality.
