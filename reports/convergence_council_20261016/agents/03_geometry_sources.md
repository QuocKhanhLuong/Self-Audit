# Research Report: Geometry và nguồn định danh teacher — Astra #3

## 1. Research Question

Vì sao student học được loss nhưng named segmentation yếu? Audit độc lập tại `92a60744efef85558d7c0226c84b7c49d47217ec`, đối chiếu snapshot thực thi `513ddda8cec3a166f13cfe68fe6379cab014a1a7`. `S/` dưới đây là `reports/stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/`; `R/` là `completed_run/run/`. VERIFIED chỉ mã/metadata đã đọc; DERIVED là phép tính; REPORTED là kết quả arrays đã lưu; chưa có arrays thì không kết luận nguyên nhân thực nghiệm.

## 2. Key Findings

**Ưu tiên kiểm tra nguồn định danh, không tăng năng lực trước.** RV được phép học: 26/80 TRAIN có RV hard support; VAL vẫn xuất RV ở 22/36 frame, tổng TP=14, FP=21.435, FN=230.267. Vì vậy “RV bị tắt hoàn toàn” sai; hạ threshold đơn thuần không giải quyết bằng chứng định danh sai trên phần đã chấp nhận. Đây là reduction từ saved counts, chưa định vị từng false positive. [Bank REPORT:2–7](../../stable_guidance_main513ddda_b12_20261016/completed_run/run/bank/REPORT.json#L2), [checks/geometry.json](../checks/geometry.json).

TRAIN/VAL/TEST đều được gán `+x` (80/18/50); `no_orientation_slices=0`. Không có bằng chứng axis swap trong các phép transpose. Chuỗi native XYZT→TZYX→teacher XYZ→TZYXD và export TZYX→XYZT nhất quán về chỉ số. Tuy vậy, đúng chỉ số chưa chứng minh hướng trái/phải giải phẫu. [S/io.py:69–89,128–140](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/io.py#L69), [S/teacher.py:90–132](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/teacher.py#L90).

## 3. Related Work

So sánh implementation nội bộ, không tuyên bố khảo sát literature:

| Phương án | Nguồn semantic | Điểm mạnh | Giới hạn và code |
|---|---|---|---|
| `cardinal_v1`, hiện tại | Cụm feature + enclosure + tương phản + một trục trái/phải | Deterministic, no target GT | Không kiểm tra coded affine; snapshot `S/core.py`, `S/io.py` |
| `world_left_v2`, lịch sử | Cùng prior, covector từ affine có code | Từ chối header không được mã hóa | Khác run; không phải đối chứng training hiện tại; [geometry attribution](../../guidance_research_continuation_20261014/GEOMETRY_ATTRIBUTION.md) |
| Frozen CineMA MAE | Feature tái dựng, không có named segmentation head | Checkpoint hash được ghim | Adapter tự khai lineage/preprocessing chưa audit đầy đủ; [S/teacher.py:41–71](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/teacher.py#L41) |

## 4. Technical Comparison — bốn giả thuyết xếp hạng

**H1 — Prior tạo tên không đủ đặc hiệu cho tim (mạnh nhất về cơ chế; causal attribution chưa xác lập).** Feature chỉ tạo cluster. Code gán MYO cho “wall”, LV cho “cavity”, RV cho component liền wall ở phía phải và giống cường độ cavity. Không có nguồn độc lập xác nhận cấu trúc đó là tim; chính docstring nói prior không định danh tim duy nhất. Votes được hòa uniform rồi truyền trên graph; solver hội tụ không sửa tên sai. [S/core.py:193–270,274–290](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/core.py#L193).

Ủng hộ: 4.135 RV hypotheses nhưng chỉ 5.199 hard RV pixels trên 46.575.360 nodes; RV precision trên VAL cực thấp. Phản chứng còn thiếu: có thể bank đúng nhưng student đặt prediction sai. Falsifier: freeze stagewise `R0→Q→student common→native` trước scoring; nếu R0/Q có named quality cao và student làm hỏng thì H1 hạ ưu tiên. Kiểm tra no-GT trước bằng cấu trúc vòng không phải tim, mirrored input và null features/intensity, giữ nguyên seed/graph/threshold. Fix ghép cặp: chỉ thay một nguồn prior có chứng cứ độc lập, có abstention; không xem consistency là bằng chứng tên đúng.

**H2 — Cardinal dùng hình học thiếu provenance (rủi ro thật; không phải lỗi sign đã chứng minh).** `read_cine` dùng `nii.affine`, không lưu/kiểm sform/qform code, lấy projection `-A[0,:2]/spacing` và chấp nhận magnitude ≥0,9. Với transpose hiện tại, `+x` tương ứng cột tăng; dấu âm phù hợp quy ước world-left đã chọn. [S/io.py:75–88](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/io.py#L75), [S/core.py:245–258](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/core.py#L245).

Join metadata lịch sử với current raw-image SHA, affine và shape khớp **80/80**: 28 header codes `(0,0)` cung cấp 4.322/5.199 accepted RV pixels; 52 coded header cung cấp 877. Đây là nguồn metadata cùng file đã ghi hash, chưa tái đọc NIfTI. Fallback geometry không phải xác nhận laterality. Phản đối: tất cả current affines đều axis-aligned; giả thuyết “cardinal bỏ RV vì oblique” không giải thích run. Probe 20° chỉ chứng minh side predicate có thể sai ở dữ liệu khác, không chứng minh lỗi hiện tại. Falsifier: audit acquisition/orientation provenance độc lập, cùng image hashes, rồi đối chiếu prior trên cohort đó. Fix: lưu selected form/codes/units và abstain nguồn hướng thiếu chứng cứ; tuyệt đối không chọn flip bằng GT. `world_left_v2` chỉ là candidate control; lịch sử không chứng minh cải thiện run hiện tại. [checks](../checks/geometry.json).

**H3 — Semantic loss và native export dùng hai toán tử khác nhau (VERIFIED cơ chế, mức tác động BLOCKED).** Training bilinear-resize native logits xuống `48×48`, sau đó log-softmax/CE; inference softmax/gate trực tiếp native logits. Bilinear sampling không ràng buộc độc lập tất cả native logits; với shapes TRAIN, chỉ 4,21–26,72% vị trí native có hệ số sampling trực tiếp khác 0 (mean patient 16,83%). Convolution chia sẻ weights vẫn ràng buộc gián tiếp, nên không được gọi phần còn lại “hoàn toàn không học”. [S/student.py:47–58,138–151](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/student.py#L47).

Probe tạo hai native logits có cùng trung bình coarse nhưng gate khác nhau: xác nhận low coarse loss không bảo đảm native coverage. Teacher cũng chỉ có token grid `12×12×16` mỗi tile; interpolate lên48 không sinh chi tiết thật. Phản đối: tọa độ center của teacher và image downsample đồng nhất; chưa thấy lệch tọa độ xác định. Falsifier: cùng frozen checkpoint, report paired ungated/gated common/native logits, entropy và disagreement trước scoring. Fix có điều kiện: đăng ký trước một operator-consistent decoder/export hoặc native loss từ Q upsample, cùng checkpoint/bank đối với diagnostic và cùng budget đối với training tương lai; không đổi gate theo VAL. [S/teacher.py:64–65,113–125](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/teacher.py#L113).

**H4 — Teacher/student khác normalization và trường nhìn (ưu tiên thấp hơn).** Reader clip percentile và z-score toàn cine; teacher thêm min–max mỗi frame, zero-padding tile `192×192×16`; student nhận ba lát z liên tiếp, nguyên z-score. MAE được gọi riêng từng thời điểm, không phải teacher temporal transformer trên chuỗi t. [S/io.py:80–89](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/io.py#L80), [S/teacher.py:93–106](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/teacher.py#L93), [S/student.py:62–63](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/student.py#L62).

Đây là distillation target có context rộng hơn input student; một phần khó khớp là plausible. Phản đối: feature loss giảm; input khác scale không tự nó là bug và không giải thích tên RV sai. Falsifier: TRAIN-only invariance tests và residual theo context/shape/padding, chưa đổi semantic pipeline. Fix factorial đơn yếu tố normalization/context, giữ Q, budget và seed; không tăng model trước khi H1–H3 được phân biệt.

## 5. Novelty Gap

Hash đúng checkpoint không tạo semantic authority. Đóng góp có thể kiểm chứng là nguồn prior có độ đặc hiệu và transfer qua cùng operator; thêm backbone mạnh hơn hiện chưa giải quyết khoảng trống này. Readout vẫn là experimental, không phải pretrained segmentation head.

## 6. Dataset and Metric Suggestions

ACDC cardiac cine MRI, native NIfTI XYZT; no-GT TRAIN, named RV/MYO/LV chỉ đánh giá frozen VAL. Recovery 14 frames xác nhận exact cine/frame voxel identity, không resample, không đổi checkpoint/predictions/threshold. Vì thế đây là giải quyết điều kiện scoring, không giải thích loss hay sửa anatomy. [S/evaluate.py:133–164,193–207](../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/evaluate.py#L133), [RECOVERY_STATUS:2–9](../../stable_guidance_main513ddda_b12_20261016/completed_run/evaluation_recovery/RECOVERY_STATUS.json#L2). Giữ TP/FP/FN, undefined precision, frame→patient aggregation và UNKNOWN là FN; không gộp chỉ số micro với macro.

## 7. Recommended Direction

Trước training mới: giữ nguyên baseline; kiểm provenance hướng và export operator; freeze matched stagewise predictions bằng thresholds hiện tại rồi mới independent scoring nếu được cho phép. Probe đã chạy: `rtk proxy python3 -B reports/convergence_council_20261016/probes/geometry.py`; PASS metadata và toán CPU tổng hợp. Không load patient arrays, không training/GPU/download/GT tuning/merge/push.

## 8. Open Questions

Chưa có vị trí false positive, pixelwise R0/Q, actual CineMA-token alignment qua upstream patch embedding, hay native logits của checkpoint; không kết luận lỗi flatten XYZ chỉ từ shape hoặc mock. Historical source/weights chỉ có recorded identity, chưa chứng thực đầy đủ preprocessing pretraining. Cần các dữ kiện này để phân biệt seed sai, source geometry sai và student/native transfer sai.

## Phản biện chéo và lập trường sau thảo luận

- **Astra #2 → #3:** 4.322/5.199 RV pixels từ fallback header chỉ là association về tạo support; không phải tỷ lệ lỗi và có thể confound với giải phẫu. Coded52 không trở thành trusted anatomy. **#3 chấp nhận:** H2 vẫn là rủi ro nguồn, không được nâng thành nguyên nhân chính đã xác nhận. Đổi sform hoặc chọn flip không tự chữa RV.
- **Astra #3 → #2:** no-GT equivariance, annulus và null-source controls có thể bác bỏ, không xác nhận tên đúng. Admission mismatch tồn tại độc lập; Q→Z giữ argmax nên không giải thích/sửa RV precision rất thấp. **#2 chấp nhận ưu tiên source/operator audit trước Q→Z**, giữ bound entropy 43,49% trên effective weighted TRAIN measure, không gọi đó là tỷ lệ h<0,6 hay native voxels.
- **Astra #4 → #3:** probe output logits có cùng coarse interpolation nhưng native coverage khác chỉ chứng minh non-identifiability của toán tử; chưa chứng minh student sinh đúng perturbation ấy. **#3 đồng ý:** fraction sampling trong H3 không phải phần trăm ảnh thiếu mọi gradient; shared convolution/GroupNorm vẫn có coupling. Cả hai ưu tiên frozen diagnostic trước capacity/input intervention.
- **Astra #3 → #4:** P0 phải dùng `softmax(bilinear(native logits))` cho common grid đúng loss; interpolate probabilities/argmax phải báo riêng. Dù semantic-only cải thiện KL, kết quả chỉ phân xử fit-Q. **#4 đã sửa P0a theo objection**, đồng ý pilot âm trong budget nhỏ không bác bỏ capacity toàn cục.

**Lập trường cuối:** giữ H1→H2→H3→H4 như thứ tự audit, không gọi là ước lượng phần trăm đóng góp nguyên nhân. Có đồng thuận về diagnostic/source-first; chưa có đồng thuận thực nghiệm về nguồn gây RV sai, tác động orientation hay benefit của thay architecture.
