# Nhiệm vụ Codex: implement hai diagnostic cho Self-Audit

Làm việc trong checkout `Self-Audit` đang mở (Mac thường là `~/Self-Audit`). Viết **code chạy được trên server**, tích hợp loader, KD loss, export loader và evaluator lịch sử. Không dừng ở kế hoạch hoặc notebook minh họa. Đây là task **implement + test**: mặc định chỉ chạy tests và smoke nhỏ; không truy cập nhãn VAL, không chạy patient pilot, không train R, không mở 28 cấu hình, không tác động job đang chạy. Bàn giao lệnh để tôi chủ động chạy trên server.

## 0. Đọc source, khóa protocol

- Kiểm tra `AGENTS.md`, Git SHA, working tree. Tôn trọng thay đổi local. Đọc `GRADIENT_BRIDGE_PILOT_V2.md` và `alternate-proposal-review/ASSESSMENT_VI.md` dưới `artifacts/fewshot-design-20261028/adaptive-codistillation-20261101/`; v2 thay phần controls của spec cũ. Đọc `self_audit/{model,compact_model,compact_control}.py`, loader/evaluator thực và `DELIVERY_RECEIPT.json`. Nếu tài liệu thiếu, vẫn hoàn thành phần code/test dựa trên contract dưới đây; báo phần chưa đối chiếu.
- Chuẩn bị `protocol.lock.json` ghi source, SHA256 payloads/manifests, raw/EMA state, student–teacher lineage, split, ontology, preprocessing, evaluator, feature tap/bridge, sampling, dtype, seeds, tolerances và quy tắc quyết định. Xác minh **bytes của payload**, không coi receipt là checkpoint. Profile lịch sử: B8 **raw student** SHA256 `0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6`; wide plain-KD export `32a35c1fc3b83d57d083de4ed17ab80bda6ba38a8ebc50133e5aede17ab00380`. Không tự thay bằng B52, EMA state hoặc B8 tái lập 86,82%. Cặp exploratory chỉ hợp lệ khi có profile riêng và đúng lineage.
- Tách `predict` (chỉ ảnh) khỏi `score` (đọc GT sau khi predictions/protocol đã hash và khóa). Dùng scorer VAL18 đúng quy tắc patient/class, absent class, ignore và native geometry. Không đọc TEST50. Khóa **cả hai** protocol trước lần mở nhãn VAL đầu tiên nếu dự định so sánh trong cùng vòng.

## 1. Diagnostic A — context ba lát, forward-only

Với ba lát **cùng frame t, khác vị trí Z**, tạo `x_lambda=[x0+lambda*(x_minus-x0), x0, x0+lambda*(x_plus-x0)]`, `lambda ∈ {0, 0.5, 1}`. Dùng cùng tensor đã chuẩn bị, cùng geometry, không normalize riêng từng view; giữ quy tắc lặp lát biên. Teacher và student chạy eval độc lập. `lambda=1` phải khớp đường infer gốc cho từng model (identity fast path để tránh làm tròn không cần thiết). Wide lịch sử có mốc đối chiếu VAL18 Macro4 `0.777413392930114`, foreground `0.7060258837373868`; chỉ đối chiếu sau khi đã khóa predictions và đúng profile. Không lấy mốc TEST làm mốc VAL.

Tách ba kết quả: độ nhạy prediction; utility thật (sai→đúng, đúng→sai, FP); transfer gap. Cho mỗi patient và mỗi lớp GT RV/MYO/LV, trên các pixel **cả hai đều sai ở lambda=0**, đếm ở lambda=1: chỉ teacher sửa / chỉ student sửa / cả hai sửa / cả hai vẫn sai. Lưu counts, denominator tập cả hai sai và tổng pixel GT lớp; không biến denominator rỗng thành 0% thành công. Báo từng lớp, Dice/recall và harm trên toàn frame, lát biên so với lát trong, phân tích `lambda=0.5`. Kết quả A âm **không chặn B**. Chưa implement/fit response loss hoặc adaptive encoder trong lượt này.

## 2. Diagnostic B — gradient thật, student cố định

- Chọn đúng một tensor feature student `h` trước adapter. Khóa `h` tap, support/subspace hợp lệ, native decoder suffix, vị trí nối adapter `P` với teacher decoder, skips/raw-detail, mapping logits và interpolation. Giữ các điều kiện phụ cố định giữa arms. Đường không sửa qua feature tap phải khớp exported student (D0). Nếu thiếu adapter đã align, implement lệnh **fit-P riêng, chỉ TRAIN**, budget và loss khai báo trước, khóa hash P; không tự fit trong `predict`, không dùng P ngẫu nhiên cho kết luận bệnh nhân. Có thể chạy toàn bộ tests bằng fixture nhỏ.
- Teacher và student ở `eval()`, parameters/buffers không thay đổi; `h` là biến cần đạo hàm. Teacher target `q_T` tạo từ ảnh, detach. Không bọc đường `h→P→D_T` trong `no_grad`. Quy về **cùng lưới logits, class order và đơn vị logits** trước khi trao đổi cotangent. Tính `uS=detach(∂KD_S/∂zS)`, `uB=detach(∂KD_B/∂zB)` bằng **đúng loss đang chạy**, cùng target, temperature, confidence/valid mask cố định, weights và denominator. Không hard-code `p-q`, không áp temperature hoặc normalization lần hai. Dùng VJP/autograd, không dựng ma trận Jacobian.

| Arm | Hướng trước khi lấy dấu âm | Vai trò |
|---|---|---|
| D0 | zero | Native student không sửa |
| D1 | `JSᵀ uS` | Self-gradient, cùng teacher target |
| D2 | `JBᵀ uB` | Bridge-gradient đầy đủ |
| D3 | hướng shuffle ngẫu nhiên đã khóa seed | Negative control cùng norm; chỉ shuffle trong vùng hợp lệ, không dùng GT |
| D4 | `JBᵀ uS` | Thay đường truyền, giữ nguyên sai số đầu ra của student |
| D5 | `JSᵀ uB` | Giữ đường truyền student, thay sai số đầu ra bằng của bridge |

D4/D5 là tùy chọn **phân tích cơ chế không cần retrain**, không phải điều kiện buộc phải thắng mới cho phép nghiên cứu R. `JB` bao gồm P, alignment và teacher suffix được khai báo; không quy riêng cho pretrained teacher decoder. Assert D1/D2 tính qua VJP bằng direct autograd của đúng objective tương ứng.

- Trên từng mẫu: cùng `r = rho * ||h_valid||₂`, cùng feature mask/subspace, `delta = -r*g_valid/||g_valid||₂`. Log actual displacement norm. Khóa một `rho` chính, lưới sensitivity phụ và ngưỡng norm bằng TRAIN trước VAL. Near-zero/empty support → fallback D0 cho arm đó; không chuẩn hóa noise. NaN/Inf phải có status riêng và chặn kết luận nếu ảnh hưởng độ tin cậy. Giữ tất cả bệnh nhân/ca fallback trong cohort; cosine chỉ tính trên both-active. D3 định nghĩa rõ gradient nguồn và ghép norm/fallback với arm nguồn.
- Chấm **mọi correction qua decoder student** tại `h+delta`; không sửa weights, optimizer hoặc feature của arm kế tiếp. Báo native KD, bridge KD, native GT loss, patient/class Dice, foreground Dice, Macro4, RV recall/precision, gradient norm, nhóm both-active/self-only/bridge-only/neither và chi phí forward/backward. Native GT loss chỉ được tính ở scoring (lưu shard logits hoặc đủ thống kê cần thiết, bounded memory), tuyệt đối không dùng GT để sinh hướng hay chọn `rho` cho từng ca.
- Kiểm tra bridge có phụ thuộc `h`: so output/loss/norm gradient khi giữ `h`, zero `h`, shuffle `h` giữa bệnh nhân phù hợp shape; thêm ablation raw-detail nếu nhánh thực sự dùng nó. Log tất cả conditioning. Nếu adapter chưa align/D0 lệch/bridge bị bypass, báo `INVALID_DIAGNOSTIC` hoặc `BLOCKED`, không đánh đồng với giả thuyết thất bại.

## 3. Quy tắc quyết định, không chỉnh sau khi nhìn VAL

Primary metric: foreground Dice của **native student**, metric phụ như trên. Protocol phải khai báo bằng **điểm Dice tuyệt đối**: mức cải thiện tối thiểu thực dụng, primary rho, quy tắc bất định và độ nhạy leave-one-patient-out. Đọc từ protocol đã khóa; nếu chưa có, tạo template và buộc điền trước khi khóa protocol/chạy trên VAL, không tự chọn bằng VAL. Bootstrap paired theo **bệnh nhân**, cùng resample cho mọi arm, tính lại class-first metric đúng chính sách lớp vắng. Leave-one-patient-out = tính lại metric khi bỏ bệnh nhân, **không retrain**. Dùng cùng 18 bệnh nhân cho hai contrast chính `D2−D0`, `D2−D1`.

`GO_TO_R_PILOT` chỉ khi D2 vượt **cả D0 và D1** theo tiêu chí primary đã khóa. Chỉ thắng KD hoặc GT-CE không đủ. Tách `NO_GO`, `INCONCLUSIVE`, `BLOCKED` khỏi nhau. D4/D5 chỉ giới hạn phạm vi diễn giải; bridge đầy đủ vẫn có thể hữu ích dù controls thành phần không thắng. Gate chỉ ghi quyết định, **không tự khởi chạy training**. Ghi rõ follow-up R0 = supervised + KD; R1 = cùng R + detached self-gradient target; R2 = cùng R + detached bridge-gradient target; ba arm cùng graph/init/budget, R2 phải hơn cả R0/R1. Không implement toàn bộ trainer R hoặc đổi online EMA ở task này.

## 4. Tests và bàn giao

Viết automated tests kiểm tra: lambda=1 và D0 parity với đường chuẩn; lát biên/từng lớp, rescue/harm với counts biết trước; không thể đọc VAL masks ở predict hoặc U masks ở fit; reject hash/lineage sai; VJP đúng cả khi `T≠1`, mask/weights không đồng đều; counterexample hai Jacobian bằng nhau nhưng `uS` và `uB` trái dấu; zero/empty/nonfinite và matched norm; bridge phải còn grad đối với `h`, không accumulate parameter grads; frozen state không mutate, RNG deterministic; bootstrap/LOO paired và không bỏ bệnh nhân fallback; các nhánh logic GO/NO_GO/INCONCLUSIVE. Chạy `check_repo.py`, pytest liên quan và CLI smoke end-to-end trên fixture (CPU; CUDA nếu có). Nếu thiếu dependency/weights/GPU, báo phần thật sự chưa chạy; thiếu patient assets không được ngăn việc viết code, test và runbook.

Bàn giao: file code/config đã sửa; CLI thật kèm `--help`; lệnh theo thứ tự `preflight → lock-A → context-predict → fit/load-P trên TRAIN → lock-B → bridge-predict → context-score → bridge-score → report`. Như vậy context chạy trước và cả hai bộ predictions đều đóng băng trước khi đọc GT. Nếu B bị chặn, A vẫn có thể score riêng; ghi A là phân tích đã mở VAL và không giả nhận protocol B đã khóa từ trước; config mẫu không hard-code đường dẫn máy cá nhân; report Markdown/JSON, bảng CSV theo bệnh nhân/lớp, hash predictions/inputs, seed và trạng thái từng bước. Liệt kê `IMPLEMENTED / VERIFIED_ON_FIXTURE / PATIENT_NOT_RUN / BLOCKED` đúng thực tế. Không đưa kết quả synthetic vào bảng Dice bệnh nhân. Không push main, không restart job hoặc thay đổi baseline lịch sử.
