# Đánh giá nghiên cứu của GPT khác: context-response distillation

Ngày 2026-11-01. Phạm vi: đọc source, receipts, Git refs đang có và tài liệu gốc; không chạy inference/training bệnh nhân, không thay job. Đây là đánh giá thiết kế, không phải kết quả của phương pháp context mới.

## Kết luận

Ưu tiên phép thử context của bản này **trước** gradient-predictor được đề xuất ở vòng trước. Nó có câu hỏi gần dữ liệu hơn và có thể bác bỏ bằng inference với hai checkpoint đúng lineage. Tuy nhiên, chưa nên lấy “teacher dạy khi nào context hữu ích” làm claim: loss được viết chỉ truyền **phản ứng của teacher**, chưa biết phản ứng đó đúng trên U.

Giữ CrossKD-style bridge làm baseline transfer mạnh. Chưa ghép response KD, adaptive gate, gradient-predictor và online EMA vào cùng phương pháp. Chỉ mở nhánh phức tạp sau khi diagnostic đã chỉ ra tín hiệu còn thiếu.

## 1. Những điểm đúng và được kiểm tra lại

- `self_audit/model.py:23-34`: teacher dùng center image chuẩn hóa cộng convolution trên hai chênh lệch lát bên. `model.py:70-72,113-137`: có raw-input detail branch đọc ba lát.
- `self_audit/compact_model.py:66-131,184-205,238-246`: compact có đầu vào ba kênh, GroupNorm/GELU, Kaiming; decoder concatenate skip rồi chiếu 1x1.
- `self_audit/data.py:302-327` và `compact_inference.py:265-274`: ba lát là ba vị trí Z cùng frame t; biên được clamp. Không phải temporal ED/ES triplet.
- Receipt `results/compact_acdc_l8_20261029/experiment_20261028/DELIVERY_RECEIPT.json` khớp các số compact/teacher lịch sử đã dẫn, và xác nhận raw student state của recipe EMA.

Từ exact class means của compact, giới hạn giả định khi chỉ RV lên 1 còn BG/MYO/LV cố định là 0.8537146898. Để Macro4 đạt 0.87 với BG hiện tại, foreground mean cần 0.8296005919. Đây là phép tính về phạm vi cải thiện cần có, không chứng minh lỗi do thiếu context; giả định giữ các class khác nguyên có thể không thực hiện được bằng một thay đổi prediction thật.

Một giới hạn trong suy luận từ kiến trúc: convolution đầu của compact cũng có thể học phép trừ giữa các kênh. Việc không viết riêng `CenterDifference` không chứng minh compact không thể dùng context. Giả thuyết cần nói là **chưa học được cách dùng context hữu ích dưới recipe hiện tại**, không phải bất khả biểu diễn.

## 2. Snapshot trạng thái trong bản được gửi đã cũ

Git ref `origin/QuocKhanhLuong/cptu-launch-20261031` có commit `e561729`, mới hơn phần NOT_RUN được trích. Snapshot ghi plain KD đã xong 2,000 bước; scalar đang chạy, larger-tie và utility đang chờ. Không có kết luận so sánh đủ bốn arm. Nguồn: [CPTU_STATUS_e561729.md](evidence/CPTU_STATUS_e561729.md), [PLAIN_KD_FINAL_SUMMARY_e561729.json](evidence/PLAIN_KD_FINAL_SUMMARY_e561729.json).

Snapshot cũng dùng teacher B8 **tái lập** với TEST50 Macro4 0.8682269381, khác teacher lịch sử 0.8800492398. Hai nguồn weight không được trộn. Không so trực tiếp score compact của pilot 2,000 bước với run compact 10,000 bước để quy kết cho selector; cả budget và teacher provenance khác nhau. Snapshot Git không phải trạng thái live của server tại thời điểm đọc.

Commit `a51924a` xác nhận focal+adaptive run hoàn tất training, nhưng tại snapshot đó evaluation chưa chạy. Nguồn: [FOCAL_COMPLETION_a51924a.md](evidence/FOCAL_COMPLETION_a51924a.md). Không suy ra score tốt từ TRAIN loss. Một receipt nói có backup local không bảo đảm chính checkpoint B8 lịch sử và compact wide cần cho diagnostic đã hiện diện và runnable trong checkout này.

## 3. Sửa diagnostic context cho đúng câu hỏi

### Phân biệt sensitivity và utility

$\Delta_T=m_T(x_{full})-m_T(x_{center})$ đo sensitivity theo đầu vào, không đo usefulness. Teacher có thể đổi mạnh nhưng đổi sai. Diagnostic có nhãn phải đo cả:

- patient/class Dice full trừ center;
- sai -> đúng và đúng -> sai;
- các pixel/class mà teacher được context giúp nhưng student không được giúp;
- magnitude response, kể cả khi argmax không đổi.

Không chỉ tính global pixel accuracy vì BG chiếm ưu thế. Báo class-specific TP/FP/FN, patient-level distributions và net benefit; rescue/harm counts không thay thế Dice vì Dice là phi tuyến.

Giữ mask ngoài predictor và ngoài construction của intervention. TRAIN-L có thể dùng cho thiết kế/khóa quy tắc. VAL18 dùng diagnostic development sau khi protocol đã khóa, không tạo train targets, weights hoặc per-image gates từ VAL mask. TEST50 đã exposed không là confirmation độc lập.

### Thêm đường can thiệp liên tục

$$
x(a_-,a_+)=[x_z+a_-(x_{z-1}-x_z),x_z,x_z+a_+(x_{z+1}-x_z)].
$$

Các view khóa trước: full (1,1), center (0,0), half (0.5,0.5); thêm (1,0) và (0,1) nếu cần tách hai phía. Cùng normalization/crop/validity cho các view, không tính lại normalization theo từng intervention.

Center-only làm zero nhánh difference teacher nhưng thay cả raw detail input. Với compact, nó đưa cùng ảnh vào ba convolution channel khác trọng số. Vì vậy không gọi đây là xóa duy nhất một module hay xác định nguyên nhân riêng CenterDifference. Nếu muốn quy kết branch, phải có can thiệp riêng trên branch, là diagnostic khác.

Lát biên, ED/ES, khoảng cách vật lý Z và slice position phải được theo dõi riêng. Input lặp lát có thể lệch phân phối ở nội thất volume; half-context giúp kiểm tra độ nhạy với mức độ can thiệp nhưng vẫn không chứng minh tất cả view nằm trên phân phối tự nhiên.

Nếu teacher không có lợi ích đáng tin từ context, dừng **transfer từ teacher hiện tại theo tín hiệu này**; không kết luận context MRI nói chung vô ích. Nếu average gain gần zero, kiểm tra rescue/harm có triệt tiêu nhau theo bệnh nhân/lớp hay không trước khi bỏ.

## 4. Loss mới thực sự thêm điều gì?

Đặt $e_f=m_S(x_f)-m_T(x_f)$ và $e_c=m_S(x_c)-m_T(x_c)$. Khi đó:

$$\Delta_S-\Delta_T=e_f-e_c.$$

Với squared loss:

$$\|e_f-e_c\|^2=\|e_f\|^2+\|e_c\|^2-2e_f^\top e_c.$$

Đây là ràng buộc tương quan giữa sai số ở hai view. Nó không có thêm GT. Nếu $e_f=e_c\ne0$, response loss bằng zero dù student còn sai. Huber có cùng zero-set, dù không có đúng khai triển squared ở mọi miền.

Vì vậy luôn giữ supervised/full-view KD anchor. Với student khớp teacher hoàn hảo ở cả hai view, response loss tự bằng zero. Gain nếu có đến từ ưu tiên optimization/regularization trong regime hữu hạn, không phải một nguồn annotation độc lập.

Đối chứng bắt buộc:

1. Full-view KD.
2. Full+center KD với cùng hai teacher/student forwards.
3. Cùng hai view và cùng backbone + response loss.

So thêm equal-wall-clock khi khác backward/loss cost. Không đổi normalization hoặc thêm class weighting chỉ ở arm 3 rồi quy gain hoàn toàn cho response objective.

Probability margins $p_c-p_{BG}$ bị saturation. Tuy nhiên vector đủ ba foreground margins cùng điều kiện tổng xác suất bằng 1 xác định được phân phối bốn lớp; vấn đề ở đây là sensitivity/saturation và common error, không phải thiếu định danh đại số. Logit margins có thể là ablation nhưng nhạy temperature/scale và không mặc nhiên tốt hơn. Báo response-target histogram, class-wise loss mass và tỷ lệ gần zero để tránh mục tiêu bị BG/unchanged pixels chi phối.

## 5. Novelty: phần nào đã bị prior art bao phủ?

- [CSAM](https://www.alphaxiv.org/pdf/2311.04942): original full text có cross-slice attention và ACDC cardiac experiments. “Dùng lát bên + attention nhẹ” không mới.
- [Knowledge Transfer with Jacobian Matching](https://proceedings.mlr.press/v80/srinivas18a.html): original paper liên hệ output matching dưới input noise với derivative matching. Response finite difference thuộc vùng ý tưởng rất gần. Không nói phép bỏ hoàn toàn lát bên tương đương chính xác với định lý local/small-noise đó.
- [CrossKD](https://www.alphaxiv.org/pdf/2306.11369): gradient qua frozen teacher head về student features đã được dùng. Giữ làm baseline.
- [CPU-based fast cardiac cine MRI segmentation using lightweight CNNs and knowledge distillation](https://doi.org/10.1016/j.jocmr.2026.102795): xác nhận được DOI và abstract từ scholarly metadata, gồm MobileUNetV2/V3/EfficientLiteUNet, ACDC+M&Ms và CPU deployment. Publisher/PDF trả 403 nên chưa kiểm tra full-text protocol. Không so Dice trực tiếp với L8. Xem [metadata đã lưu](evidence/cardiac_KD_metadata_abstract_only.txt).
- ACSFormer: tìm được metadata `Selective intra- and inter-slice interaction for efficient anisotropic medical image segmentation`, DOI 10.1016/j.patcog.2025.112895; chưa truy cập full text. Không coi chi tiết cơ chế trong bản GPT là đã xác minh ở lượt này.

Claim ứng viên phù hợp hơn:

> We study whether transferring a teacher's response to adjacent-slice context improves compact few-patient segmentation beyond matched two-view distillation.

Chỉ khi diagnostic và experiments hỗ trợ mới nâng lên “task-relevant context response” hoặc “adaptive context utilization”. Chưa claim causal reasoning, independent truth, proven generalization hay novel cross-slice attention.

## 6. Kiến trúc adaptive và hệ thống inference

Giữ nhánh center mạnh cộng hai residual context branches ở độ phân giải thấp là thiết kế hợp lý để thử. Gate chỉ đọc student-derived inputs khi deploy. First-layer convolution thường cũng xử lý được difference nên phải so explicit center/difference frontend và plain stacked-input model trước khi quy gain cho attention.

Constant-gate control cô lập phần nào việc thay đổi gate theo ảnh nhưng thường có ít tham số học được hơn. Muốn claim capacity-matched cần một static residual block với tham số/MACs hoạt động tương đương, không cộng dummy unused parameters.

Phép tách concat+1x1 là đúng với code hiện tại có bias=False:

$$W[\operatorname{up}(d);s]=\operatorname{up}(W_d d)+W_s s.$$

Giữ GN/GELU sau tổng, cùng interpolation/padding. Đây là tối ưu implementation, không đổi hàm toán học trong exact arithmetic. Cần parity odd/even shape, output/full model và precision thực tế. Script `/mnt/data/verify_compact_decoder_fusion.py` của GPT khác không có trong phiên này nên các số CPU error của họ chưa được tái lập. Không suy ra GPU speedup từ tính tương đương hoặc từ peak VRAM reserved.

## 7. Thứ tự đã điều chỉnh

1. Xác định đúng checkpoint teacher lịch sử hoặc reproduction và compact; kiểm tra payload/hash/data availability, không dùng receipt thay payload.
2. Khóa diagnostic context trước khi xem mask VAL; chạy khi payload/data sẵn trên host được phép.
3. Nếu teacher-helped/student-missed có tín hiệu lặp lại, thử response KD trên **cùng compact** trước để tách tín hiệu loss.
4. Sau đó thử adaptive context encoder với fixed/static capacity controls. Không đồng thời đổi backbone và loss.
5. CrossKD bridge/feature KD là đối chứng transfer mạnh; pretrained compact/crop parity vẫn là baseline representation cần giữ.
6. Chỉ sau đó mở online mentor/EMA hoặc Self-Audit. Gradient-predictor ở vòng trước hạ xuống nhánh nghiên cứu sau.

Nếu diagnostic không ủng hộ context, ưu tiên pretrained compact, representation transfer và decoder systems optimization. Không mở hàng chục model/loss chỉ để tìm một kết quả dương.

## Kiểm chứng và giới hạn

Đã đối chiếu current source, receipt lịch sử, newer local Git ref, CSAM full text và Jacobian Matching original PDF. Chưa xác nhận tất cả citations của bản GPT, chưa truy cập full text bài JCMR/ACSFormer, chưa chạy diagnostic bệnh nhân, chưa tái lập script CPU của họ, chưa kiểm tra live server. Source tree không sửa. Các checks và hashes ở [PROVENANCE.json](PROVENANCE.json).
