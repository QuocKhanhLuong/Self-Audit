# Research Report: Astra #4 — tối ưu, năng lực student và phép thử phân biệt

## 1. Research Question

**Round 1 độc lập, 2026-10-16.** CE gần phẳng có nghĩa student ngừng học, hay mục tiêu mềm/gate che khuất tiến bộ? Evidence publication: `92a60744efef85558d7c0226c84b7c49d47217ec`; source thực thi: `513ddda8cec3a166f13cfe68fe6379cab014a1a7`. Astra mới: nhánh riêng `2c5796084e4412ccab70830afca0adbb6a3e1d23`, không phải model của run này. Không đọc patient arrays/checkpoint, chạy GPU, huấn luyện bệnh nhân hoặc dùng VAL/TEST để chọn phương án.

## 2. Key Findings

**VERIFIED_LOCAL_TEXT—saved losses ghi nhận tiến bộ.** KL online ước tính giảm `0.077121→0.038853` (49.62%); trung bình epoch 31–40→41–50 giảm `0.044009→0.039950` (9.22%). Feature loss giảm 41.94%. Do đó “không học” và “đã hội tụ hoàn toàn” đều chưa được hỗ trợ. CE cuối `1.085173` đi cùng entropy floor `1.046319`. **REPORTED:** audit đã tái hiện bank/RNG/batch gate/denominator; phiên này không replay arrays. Đây không phải KL của checkpoint cố định ([summary.json:70–115](../../stable_guidance_main513ddda_b12_20261016/summary.json); [audit.py:58–106](../../stable_guidance_main513ddda_b12_20261016/completed_run/analysis/semantic_convergence/audit.py)).

**VERIFIED—gradient không bị cắt giữa semantic head và encoder.** `enc0/1/2` nhận cả cosine feature loss và soft CE; `project` chỉ nhận feature gradient; `decode/head` chỉ nhận semantic gradient. Teacher, Q và reliability được detach; tổng loss là `fl+sl`. AdamW có LR cố định `3e-4`, weight decay `1e-4`, global clipping `5`; không scheduler. Loss có số lớn hơn không đồng nghĩa gradient lớn hơn ([student.py:23–35,38–59,75–105](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/student.py)).

**EXECUTED—probe CPU xác nhận wiring, không xác nhận nguyên nhân trên dữ liệu thật.** Student có **351,268 tham số**. Một feature-only update giữ nguyên chính xác decode/head nhưng đổi encoder và logits. Một phản ví dụ khác giữ cùng ba samples, đổi grouping thì semantic gradient đổi do foreground gate. Cuối cùng, hai tensor logits có cùng nội suy xuống coarse grid nhưng native coverage khác `0→50%`: loss coarse không xác định đầy đủ output native. Các kết quả dùng tensor ngẫu nhiên/chế tạo, một optimizer step; teacher không được khởi tạo ([optimization.py](../probes/optimization.py); [optimization.json](../checks/optimization.json)).

## 3. Related Work

Đối chiếu implementation nội bộ; không gán benchmark bên ngoài.

| Phương án / trạng thái | Ý tưởng và điểm mạnh | Hạn chế / liên quan |
|---|---|---|
| GuidanceStudent, executed | 2.5D ba lát; encoder ba mức; skip native, hai head; nhỏ và tái hiện được | Cùng encoder phải fit feature lẫn Q; coarse loss/native gate khác miền |
| Feature-only, control đề xuất | Giữ target feature, đo logit drift khi semantic head đứng yên | Không tạo named supervision; chưa chạy patient control |
| Semantic-only, control đề xuất | Cô lập khả năng fit Q và ảnh hưởng feature gradient | Có thể làm representation xấu; không chứng minh anatomy |
| Astra layer-guided, code branch | Multiscale ConvNeXt, coarse/detail factorization, detail head khởi tạo zero | Smoke dùng random encoder, 32,890,558 tham số; chưa pretrained/patient training |

Nguồn: [executed student.py:19–35](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/student.py); `2c57960:experiments/layer_guided_student_v1/layer_upgrade/model.py:53–114`; `2c57960:experiments/layer_guided_student_v1/receipts/FULL_MODEL_CHECK.json:1–6,60–67`. Code Astra có sẵn; receipt là software smoke, không phải lợi ích segmentation.

## 4. Technical Comparison

**Capacity mismatch là giả thuyết hợp lý nhưng chưa phân xử.** Teacher có 125,670,400 tham số frozen, xem tile `192×192×16`, rồi random-project feature về 64 chiều và nội suy grid48; student chỉ dùng ba lát cùng thời điểm. Feature teacher có thể chứa thông tin vượt input student, nên residual feature loss không nhất thiết do optimizer. Parameter ratio cũng không chứng minh student thiếu năng lực: nhiệm vụ của student là fit compressed feature/Q, không tái tạo toàn bộ teacher ([teacher.py:34–38,61–84,93–131](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/teacher.py); [VRAM REPORT:168–169](../../stable_guidance_main513ddda_b12_20261016/vram_probe/REPORT.json)).

**Sampling/step budget là confound thực.** Theo metadata TRAIN, mỗi epoch có 20,215 contexts. B8 cần 2,553 updates; B12 cần 1,714: cùng 50 epoch giảm 32.86% số updates. `127,650` B8 là phép tính phản thực tế, run B8 đã bị ngắt. Patient được shuffle, rồi xử lý toàn bộ lát của từng patient; shape native thay đổi, tail batches vẫn được một optimizer step và một trọng số trong epoch mean. B12 có 54 tail batches/epoch. Vì semantic gate xét cả batch và CE chuẩn hóa riêng từng batch, regrouping còn đổi support và trọng số hiệu dụng; không chỉ đổi gradient noise ([student.py:82–112](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/student.py); [metadata arithmetic](../checks/optimization.json)).

VRAM probe chỉ năm steps/case, median ba steps cuối; mọi student case đều semantic-inactive. `49.13→50.43 samples/s` không đo full-run speedup hoặc accuracy. Không thể kết luận B12 làm model tệ hơn, cũng không thể kết luận tối ưu VRAM vô hại với objective ([executed_probe.py:60–85](../../stable_guidance_main513ddda_b12_20261016/vram_probe/executed_probe.py); [REPORT:60–162](../../stable_guidance_main513ddda_b12_20261016/vram_probe/REPORT.json)).

## 5. Novelty Gap

Thiếu bằng chứng cho ba cơ chế: gradient interference; capacity/input-information ceiling; coarse-fit không chuyển thành native confidence. Log hiện tại chỉ có mean losses, steps, time, memory; không có per-class KL curves, fixed-checkpoint KL, gradient norm/cosine, clipping frequency hoặc actual AdamW displacement. Checkpoint cuối không lưu optimizer state, nên continuation không thể được gọi là resume chính xác ([student.py:108–125](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/student.py)).

Việc thêm depth, pretraining, thay Q→Z và đổi optimizer đồng thời sẽ không phân xử được cơ chế. Astra mới giải quyết một phần cấu trúc coarse/detail bằng thiết kế; chưa có evidence rằng nó chữa semantic source sai hoặc chất lượng RV.

## 6. Dataset and Metric Suggestions

Chỉ dùng **TRAIN80 ACDC cine MRI tim**, ảnh native TZYX, contexts ba lát; bank feature/Q/reliability frozen, không GT. Khóa patient, slice, source hash, common support và grouping trước khi đo. Dùng full-distribution weighted KL trong strata do **Q argmax/source** cố định, cùng entropy, feature cosine, weighted target mass, gate pass rate; không gọi các strata đó là anatomy truth. Component `q_c log(q_c/p_c)` riêng một lớp có thể âm, không được trình bày như KL độc lập.

## 7. Recommended Direction

**PROPOSED / NOT RUN trên bệnh nhân**, theo thứ tự; chưa cho phép triển khai tự động:

1. **P0a—frozen checkpoint, zero optimizer-step.** Trên tối đa 128 batches TRAIN khóa sẵn và đủ source/class strata khả dụng, đo coarse KL, native/coarse gate disagreement, norm/cosine từng encoder block/loss. Coarse phải là `softmax(bilinear(native_logits))` đúng source. **P0b riêng:** candidate update trên clone là optimizer-step, chỉ thực hiện khi được cho phép. Không sửa threshold. Coarse fit tốt nhưng target/native gate thiếu support làm yếu giả thuyết optimizer chính; cosine âm riêng lẻ chưa định vị nguyên nhân.
2. **P1—interference, 32 steps tối đa.** Ba arms `fl+sl`, `sl`, `fl`, cùng weights/RNG/batches/Q/w/support và fresh AdamW giống hệt nhau; không gọi là resume. Đánh giá mỗi tám steps trên TRAIN witness cố định, không chỉ training batch. Chỉ giữ giả thuyết feature cản semantic nếu semantic-only cải thiện KL ít nhất 5% so với joint, lặp lại ở hai cửa sổ TRAIN độc lập, cùng dấu với actual displacement interference; dừng nếu hiệu ứng không lặp hoặc chỉ đổi support. Ngưỡng 5% là proposal cần preregister, không suy từ VAL.
3. **P2—LR, chỉ khi P0/P1 còn residual có thể giảm.** So sánh `3e-4` và `1e-4`, 32 equal updates, cùng state và fresh moments. Dừng đề xuất schedule nếu KL không cải thiện trên witness hoặc class/source stratum đã khóa xấu đi. Không ngoại suy thành yêu cầu train lâu hơn.
4. **P3—batch/capacity sau cùng.** Tách regrouping khỏi update budget bằng effective-batch24: tích lũy microbatch8×3 và 12×2, chung gate và global denominator, một update giống nhau. Đây là control mới, không replay legacy loss. Chỉ thử capacity sau khi nhỏ-model one/few-batch fitting failure lặp lại; giữ target/grid/pretraining cố định khi đổi width, chưa nhảy thẳng sang Astra đầy đủ.

## 8. Open Questions

KL thấp có tập trung ở BG/source dễ? Semantic-inactive encoder drift mạnh đến đâu tại checkpoint cuối? Thông tin teacher vượt ba-lát input tạo residual bao nhiêu? Native gate failure đến từ target uncertainty hay interpolation? Hiện không có phép đo cho các câu hỏi này. Quyết định hiện tại: **ưu tiên P0; chưa có căn cứ quy lỗi chính cho optimizer hoặc mở rộng training/model**.

## Phản biện chéo — round 2

**Astra #1:** chất vấn căn cứ nâng conflict/capacity thành nguyên nhân. Tôi xác nhận **UNKNOWN**, và sửa nhãn replay thành REPORTED, tách P0a zero-step khỏi P0b cloned update. Tôi phản biện ngược: accepted RV `TP14/FP21435` định vị output failure, chưa quy lỗi source-Q hay native mapping. #1 chấp nhận; cần so Q/student cùng support/grid ([01_run_evidence.md](01_run_evidence.md)).

**Astra #3:** yêu cầu coarse operator đúng source; tôi chấp nhận. Qua pilot 5% chỉ hỗ trợ cải thiện fit-Q, chưa là named anatomy; bounded null result chỉ dừng remedy ở budget này, không bác bỏ mọi khả năng dài hạn ([03_geometry_sources.md](03_geometry_sources.md)).

**Root:** Astra D/U contract không tự cung cấp native named supervision; model lớn/detail head chưa có bằng chứng chữa semantics. Tôi đồng ý. Với simplex, `pmax≥.7` đã kéo theo margin≥.4, nên đổi riêng margin .2 không cứu coverage. **Revised stance:** giữ P0a trước; synthetic drift/nullspace chỉ là đường lỗi khả dĩ.

**Astra #2:** chấp nhận phản biện của tôi: Q→Z đổi gradient `Δg_logits=w(Q−Z)/denominator`, nên cùng coefficients không giữ feature/semantic balance. Paired gain chỉ nhận diện tổng tác động đổi target. Hai bên thống nhất cần norm/cosine và fixed-Q/common/native diagnostics; sharpening entropy-matched là control bổ sung nếu claim đặc hiệu Z, nhưng entropy bằng nhau không bảo đảm gradient bằng nhau ([02_target_theory.md](02_target_theory.md)).
