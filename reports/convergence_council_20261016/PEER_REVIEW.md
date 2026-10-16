# Phản biện chéo có quy trách nhiệm

Bốn agent có execution riêng, được giao chuyên môn khác nhau và lưu round 1
trước khi thảo luận. Root cung cấp evidence path chung và phân công peer review;
đây không phải đánh giá blind. Bảng dưới tóm tắt trao đổi đã thực hiện qua message
trực tiếp; không phải transcript nguyên văn. Chi tiết/phản hồi nằm trong appendix
của từng báo cáo agent.

| Người phản biện → người nhận | Phản biện | Thay đổi hoặc bất đồng còn lại |
| --- | --- | --- |
| Astra #1 → #2 | Mean entropy không trực tiếp cho tỷ lệ voxel dưới threshold. | #2 chứng minh extremal-entropy bound; #1 chấp nhận ≥43,493% **effective weighted TRAIN mass**, từ uniform-CE-normalized measure. Cả hai loại ngoại suy sang h<0,6 prevalence, native/VAL voxel coverage hoặc anatomy. |
| Astra #2 ↔ #3 | Q→Z giữ argmax; geometry/source validity phải được tách khỏi confidence. Header fallback support không phải error attribution. | #3 giữ orientation là rủi ro provenance, không gọi là nguyên nhân RV đã chứng minh. #2 đặt source audit trước Q/Z và giữ counterexample wrong-name amplification. |
| Astra #1 ↔ #4 | Gradient conflict/capacity chưa đo trên checkpoint thật; chiều ngược, RV output sai không tự chứng minh Q sai. | Cả hai giữ teacher-vs-student attribution UNKNOWN; ưu tiên paired common/native audit. Chỉ scalar loss hoặc negative cosine synthetic không đủ để chọn optimizer mới. |
| Astra #3 → #4 | Common probabilities phải là `softmax(bilinear(native_logits))`, không phải resize probabilities. | #4 sửa protocol theo operator thực thi; các operator khác phải báo riêng. Native discrepancy không bị tự động gán cho geometry hoặc teacher. |
| Astra #1 → #4 | Tách host-reported bank replay khỏi local text verification; zero-step audit khác cloned update. | #4 tách P0a frozen measurement/P0b counterfactual clone; các replay chưa làm tại đây được ghi REPORTED. |
| Astra #4 → #2 | Q/Z cũng đổi gradient balance so với feature loss; matched coefficients không cô lập cơ chế conditional evidence. | Protocol tổng hợp yêu cầu norm/cosine, giữ entropy-floor khác biệt, và control entropy-matched sharpening nếu cần phân xử. Gain Q/Z không tự là semantic correctness. |

Các điểm cùng được giữ sau phản biện: run hiện tại vẫn cải thiện objective;
named performance thất bại; probability/target/operator mismatch có cơ chế rõ;
source correctness, real gradient conflict và capacity attribution còn chưa phân xử.
Không có bỏ phiếu biến một hypothesis thành fact; confidence của từng hypothesis
được giữ trong báo cáo riêng. Chưa có agent nào chứng minh một giải pháp nâng Dice.

Astra #1 đọc lại tổng hợp của root và phát hiện ký hiệu `H=0` trong protocol bị
nhầm với source mass `h=0`: entropy bằng zero có thể là target one-hot hợp lệ.
Root đã sửa thành `h=0`; reviewer không tìm thấy blocker khác trong số liệu,
phân loại bằng chứng, matched controls và các giới hạn kết luận.

Agent #1 bổ sung tính dư thừa của margin gate: pmax≥0,7 trên simplex kéo theo gap≥0,4,
nên margin0,2 không là nút chặn hiện tại. Agent #2 xác nhận và bổ sung vào lý thuyết.
Root chọn thứ tự **đo cùng nguồn/grid → source/operator audit → micro-pilot tối ưu →
candidate Q/Z/capacity**, dựa trên giá trị phân biệt nguyên nhân, không dựa trên
độ hấp dẫn của model mới.
