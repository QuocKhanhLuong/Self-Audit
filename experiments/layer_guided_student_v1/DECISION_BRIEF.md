# Quyết định triển khai — 2026-10-15

Đã triển khai đúng reference và bổ sung đường tích hợp riêng tại
`experiments/layer_guided_student_v1/`. Chưa có bằng chứng về chất lượng phân đoạn
giải phẫu. Baseline/ablation, source lịch sử, config, split và frozen evidence
được giữ nguyên.

Nguồn là branch `feat/layer-level-upgrade-20261015`, commit
`f4b0fe7d9e49ee398354e20ff0981c818840f5d4`; manifest
`3f442aa35872bea5db00eb8880c002b1049a1e53da2a88accfbf8e6e5cec2576`.
Triển khai nằm trên branch riêng `codex/astra-layer-upgrade-20261015`.

| Nội dung | Trạng thái và giới hạn |
| --- | --- |
| Phase A | Sáu module, 47 test gốc và probe/config giữ nguyên byte; đối chiếu bằng package identity độc lập |
| Phase B | Có recipe, constructor/loader/schema riêng, hash/provenance, normalization từng cine, registration teacher, transport mass/support và masked-view feature loss |
| Control | Head conventional có cùng số tham số; chưa chạy so sánh trên bệnh nhân; overhead riêng đã ghi rõ |
| Guard/refiner | Chỉ là primitive tách biệt; không nhận bảo đảm cho AdamW; refiner trả proposal detached, admission off |
| Kiểm thử CPU | 88/88 test experiment pass; gồm 47 test gốc cùng differential, loader, geometry, rollback và failure cases. Thêm 454 test pipeline hiện có pass, 1 warning có sẵn; lệnh/hash trong `PARITY_REPORT.json` và `receipts/EXISTING_REGRESSION.json` |
| Full model smoke | RANDOM torchvision ConvNeXt-Tiny: 32,890,558 tham số = 27,818,592 encoder + 5,071,966 còn lại; forward/backward manufactured input, không pretrained |
| DINOv3 chính thức | `BLOCKED_PRETRAINED`: chưa cung cấp source/weights local được duyệt; fixture giả chỉ kiểm tra hợp đồng loader |
| Dữ liệu/GPU/Dice | NOT RUN / NOT MEASURED; không suy ra cải thiện giải phẫu hay generalization |

Reviewer độc lập phát hiện source-side registration còn thiếu và môi trường
import bank chưa rõ. Đã bổ sung receipt ràng buộc geometry nguồn/đích, pin
manifest/producer, kiểm tra thay đổi manifest trong lúc đọc, và test import qua
subprocess theo đúng `PYTHONPATH` trong runbook. Kết luận cuối và các giới hạn
được reviewer giữ riêng trong `INDEPENDENT_REVIEW.md`: PASS cho phạm vi
Phase A+B CPU/software scaffolding, không còn blocker triển khai trong phạm vi đó.

Hai giới hạn của reference được giữ nguyên và có test tái hiện: projection
float32 có thể mất tính khả thi sau cast dù receipt ghi `KKT_VERIFIED`; tham số
normalization NaN tùy biến có thể qua constructor gốc. Integration chỉ dùng kết
quả kiểm tra finite-step thực tế, khóa defaults và kiểm tra buffer checkpoint.
Giảm proxy energy vẫn có thể dẫn tới lớp sai. Xem `REFERENCE_DEVIATIONS.md`.

Cleanup tập trung vào code tích hợp rõ trách nhiệm, tài liệu điều hướng, hợp đồng
input/output và bằng chứng có thể chạy lại. Không format operator reference hay
xóa lịch sử/baseline. Mọi thay đổi của nhánh triển khai nằm trong thư mục experiment.

Đề xuất tiếp theo có điều kiện: cung cấp DINO local được duyệt và bank TRAIN đã
đăng ký geometry, chạy preflight chính thức, rồi mới duyệt matched pilot giữa
hai head. Guard và refiner là các yếu tố sau đó. Không launch training, push hay
merge nhánh này trong phiên triển khai theo ranh giới của prompt.
