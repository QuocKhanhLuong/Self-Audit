# Pilot v2: native KD gradient versus teacher-decoder bridge gradient

Date: 2026-11-01. Status: **DESIGN ONLY — no patient inference/training launched**. This document supersedes the experimental-control sections of `GRADIENT_BRIDGE_SPEC.md`; the original file is preserved. The 28-row matrix is a research queue, not an authorized sweep.

## Quyết định

Chấp nhận đối chứng mới: cùng teacher target, cùng feature, nhưng gradient lấy qua decoder student hoặc decoder teacher. Đây là thiếu sót quan trọng trong v1: random/shuffled-gradient và ordinary residual controls chưa cô lập được đóng góp của decoder teacher so với gradient KD sẵn có của student.

Cần sửa tiêu chí thắng: **bridge không phải thắng self-gradient về chính native KD surrogate ở bước rất nhỏ**. Native self-gradient đã tối ưu first-order descent của surrogate đó theo chuẩn L2. Câu hỏi có ích là hướng bridge có cải thiện prediction của native decoder theo GT tốt hơn không, và liệu student-only residual có học lại được lợi ích đó không.

## 1. Hai loss, ba gradient cần phân biệt

Giữ một feature tensor trước adapter $h$, teacher target $q_T$ detached, và mọi feature/skip còn lại cố định:

$$L_K(h)=\mathrm{CE}_{soft}(q_T,p_S(h)),\qquad
L_B(h)=\mathrm{CE}_{soft}(q_T,p_B(h)).$$

Ở cả hai đường, dùng cùng target, temperature, valid mask, confidence policy, class normalization và reduction. Đặt $g_K=\nabla_h L_K$, $g_B=\nabla_h L_B$. GT không được dùng để tạo hai hướng trong primary diagnostic, kể cả khi ảnh nằm trong TRAIN-L hay VAL.

Khi có nhãn phục vụ scoring, $g_Y=\nabla_h L_Y(h;y)$ là gradient của native student loss theo GT. Nếu tính $g_Y$ trên VAL thì chỉ dùng cho phân tích sau khi direction/step/predictions đã khóa; không tham gia selector, target hoặc training.

### Định lý cục bộ đơn giản, không phải contribution mới

Cho cùng norm correction $r$, với hai gradient khác zero:

$$\delta_K=-r\frac{g_K}{\|g_K\|_2},\qquad
\delta_B=-r\frac{g_B}{\|g_B\|_2}.$$

First-order Taylor cho native KD:

$$\Delta L_K(\delta_K)=-r\|g_K\|_2+O(r^2),$$
$$\Delta L_K(\delta_B)=-r\|g_K\|_2\cos(g_K,g_B)+O(r^2).$$

Vì cosine không vượt 1, self-gradient là hướng giảm native KD mạnh nhất ở first order. Đây là hệ quả Cauchy–Schwarz cho ràng buộc L2, không áp dụng nguyên xi nếu đổi norm, support, clipping hoặc preconditioning giữa hai arm. Với cùng projection/mask lên feature subspace, áp dụng lập luận trên projected gradients.

Không được biến kết luận first-order thành định lý ordering tuyệt đối tại mọi finite step. Hai hướng gần thẳng hàng có thể có chênh lệch bậc một rất nhỏ; curvature chi phối tại bước hữu hạn. Nếu bridge chỉ thắng native KD ở bước lớn, gọi là finite-step/curvature effect, không phải bằng chứng local descent vượt steepest descent.

Với GT loss, đại lượng liên quan là $g_Y^\top\delta$. Bridge có thể kém về native KD nhưng tốt hơn về GT. Điều đó là tín hiệu đáng nghiên cứu, không phải bảo đảm generalization trên U hoặc bệnh nhân chưa thấy.

## 2. Phản ví dụ trong bản phản biện là đúng

Binary example: bridge logit $h$, student logit $-h$, target teacher $q=0.9$, nhãn thật foreground. Tại $h=0$, hai prediction đều 0.5. Sửa $h$ lên 0.2:

- bridge soft CE: 0.693147 -> 0.618139;
- native student GT CE: 0.693147 -> 0.798139.

Các số được kiểm tra lại bằng script `verify_gradient_review.py`. Đây là đại số trên dữ liệu tạo sẵn, không phải patient result. Alignment output tại một điểm không bảo đảm alignment Jacobian quanh điểm đó.

## 3. Gate 0 — đóng băng điều kiện so sánh

Manifest phải khóa teacher payload hash, inference state (raw/EMA), student checkpoint hash, source commit, student teacher-lineage, preprocessing, feature location, bridge graph và dataset manifests. Không trộn historical B8 88% với B8 reproduction 86.82%, hoặc 2,000-step VAL với 10,000-step TEST.

Fit projectors trên TRAIN-only, giữ label budget, sau đó khóa P. Ghi riêng chi phí fit P và supervision đã dùng. Tạo gradient qua frozen teacher decoder với autograd trên input; không dùng `no_grad` cho bridge. Teacher target forward phải detach.

Chọn một feature level và một bridge interface cụ thể:

- Nếu full multi-scale decoder, khai báo mọi projected student feature và mọi teacher/raw-detail input.
- Nếu suffix decoder sau refinement block, khai báo chính xác cut point và conditioning còn lại; không gọi một auxiliary one-input head là toàn bộ native teacher decoder.
- Giữ tất cả conditioning cố định khi perturb h.
- Eval mode, dropout off, không đổi buffers/RNG giữa các arm. GroupNorm vẫn couple channel/spatial positions trong một sample; không diễn giải pixel gradients như các thử nghiệm độc lập. BatchNorm train mode sẽ thêm coupling giữa samples và không thuộc pilot này.

Bridge dependence checks: h bình thường, h=0, h shuffle giữa samples; thêm raw-detail ablation nếu bridge có nhánh đó. Log output change và gradient norms. Low bridge loss + không phụ thuộc h là shortcut, chưa phải transfer.

## 4. Gate 1 — direct feature-correction diagnostic

Các arm dùng cùng frozen student:

| Arm | Correction | Vai trò |
|---|---|---|
| D0 | 0 | Native prediction reference |
| D1 | normalized native KD gradient | Essential teacher-target control |
| D2 | normalized bridge KD gradient | Teacher-decoder direction |
| D3 | shuffled/random direction, matched norm | Negative control, không thay thế D1 |

Chọn $r=\rho\|h\|_2$ trên cùng feature support. Một primary rho và một lưới sensitivity nhỏ phải được khóa bằng TRAIN-L trước khi mở VAL labels. Báo toàn bộ lưới đã khóa; không chọn rho tốt nhất riêng từng ca trên VAL. Dùng cùng crop/validity/feature coordinates và cùng displacement norm thực tế.

**Gradient gần zero:** không biến nó thành correction norm lớn. Khóa threshold/dtype trước; log separately both-active, self-only, bridge-only, neither-active. Conditional angle/direction comparisons chỉ trên both-active. End-to-end outcome vẫn báo trên toàn cohort với fallback no correction, tránh loại ca làm đẹp kết quả. Giữ raw-gradient norm distribution vì equal-norm diagnostic chỉ kiểm tra direction, không chứng minh magnitude target tốt.

### Scoring

1. Primary scientific utility: native exported-path patient/class Dice, foreground mean, RV/MYO/LV; paired patient deltas và harm/rescue counts.
2. Native GT smooth loss là diagnostic hỗ trợ. Hard-Dice có plateau nên cả loss và Dice đều cần.
3. Native KD change và bridge loss change được báo riêng, không thay thế utility.
4. Cosine $\cos(g_K,g_B)$ chỉ mô tả tương quan hai surrogate. Alignment với GT cần $g_Y$, chỉ scoring/analysis sau khi khóa predictions.
5. GT không đi vào construction của v_self/v_bridge hay vào step choice trên VAL. VAL là development, không independent confirmation.

Nếu D2 chỉ giảm bridge loss thì dừng mở predictor lớn. Nếu D2 không hơn D1 về GT utility, dừng nhánh hiện tại theo ngân sách nghiên cứu. Đây là **practical screening gate**, không chứng minh mọi co-adaptation giữa encoder/decoder/bridge đều bất khả thi. Ghi rõ fixed-checkpoint limitation.

Một kết quả dương cũng chưa cô lập teacher pretrained prior khỏi phần learning thêm trong P. Nếu muốn claim riêng geometry của teacher decoder, cần follow-up control auxiliary non-teacher head/bridge với training budget phù hợp; không tự quy toàn bộ gain cho teacher.

## 5. Gate 2 — residual predictor với đúng ba đối chứng cùng kiến trúc

Giữ inference graph, initialization, R capacity, context inputs, native losses, optimizer và schedule giống nhau cho ba arm sau:

| Arm | Training của R |
|---|---|
| R0 | Native supervised + ordinary KD |
| R1 | R0 + detached self-gradient target |
| R2 | R0 + detached bridge-gradient target |

Student không có R và CrossKD-style bridge không có R là **baseline khác graph**, vẫn cần báo, nhưng không được nói chúng có cùng inference architecture với ba arm R0/R1/R2.

Tạo targets tại pre-adapter feature, không dùng prediction đã sửa làm target reference một cách ngầm định. `create_graph=False`, detached targets; giữ P/teacher fixed. Khóa policy gradient của auxiliary loss có vào encoder hay không; đó là factor cần khai báo.

Vòng đầu dùng cùng q_T cho R1/R2 để cô lập decoder route. Nếu thêm target GT trên TRAIN-L cho gradient supervision thì thêm đối xứng cả hai arm và ghi đó là factor khác; không để R2 nhận thêm nhãn mà R1 không nhận.

Student context $c_S$ phải có tại inference. Không đưa teacher features, q_T hoặc nhãn vào R khi train rồi bỏ chúng lúc export. R vẫn có forward overhead; teacher/bridge training-only mới được loại bỏ hoàn toàn.

Direction fidelity không đủ: cần report magnitude error, zero-target frequency, native segmentation outcome và export parity. R2 phải có lợi ích thực tế so với R0/R1 và strong KD baselines dưới budget hợp lý. So cả matched steps và wall-clock vì target construction có extra teacher backward.

## 6. Prior art bổ sung đã đọc

### Nguyen & Ray, MIDL 2020

[Original paper](https://proceedings.mlr.press/v121/nguyen20a.html), section 2 / original PDF page 4. Dùng approximating network để tạo differentiable bypass cho dynamic programming không khả vi trong LV segmentation. Không phải một compact student học correction feature để deploy một forward. Tuy nhiên, “synthetic gradients đầu tiên cho cardiac segmentation” không đứng được.

### Perceptual Gradient Networks, 2021

[Original paper](https://www.alphaxiv.org/pdf/2105.01957), section 3. PGN nhận **cặp ảnh generated/reference** và xấp xỉ gradient perceptual loss từ VGG để tối ưu generator. Paper còn chỉ ra raw predicted gradient có thể cho vector field không ổn định, rồi dùng constrained proxy targets.

Khác biệt cần ghi: PGN vẫn có reference image khi dự đoán gradient; R của ta chỉ thấy student-derived information tại inference. Vì vậy prior art này vừa giới hạn novelty “forward nhỏ thay backward lớn”, vừa nêu ra bottleneck về khả năng dự đoán khi bỏ privileged information. Không chuyển nguyên bảo đảm ổn định của proxy trong pixel space sang feature residual của mình.

### CrossKD và Synthetic Gradients

[CrossKD](https://www.alphaxiv.org/pdf/2306.11369) đã truyền KD gradient qua frozen teacher head. [Synthetic Gradients](https://proceedings.mlr.press/v70/jaderberg17a.html) đã học dự đoán gradient. Tổ hợp các thành phần quen thuộc chưa tự thành novelty. Candidate claim phải dựa vào cả direct-gradient GT utility và student-only learned benefit, với R1 làm đối chứng chính.

## 7. Crop baseline và online EMA

Crop 160 vs 256 nên khóa cùng patient/center sampling và evaluator. Input 256 + loss chỉ vùng giữa 160 hữu ích để tách bớt số pixel supervised, nhưng **không cô lập hoàn toàn receptive-field context**: GroupNorm spatial statistics, padding, feature geometry và KD target crop policy cũng có thể đổi. Giữ outer input pixels trong graph là chủ đích; chỉ mask loss, không xóa context khỏi encoder. Định nghĩa denominator trên cùng inner valid pixels và dùng teacher targets theo một policy cố định. Coi đây là context-under-current-architecture diagnostic, không pure anatomy-only causal proof.

Giữ teacher fixed ở pilot đầu. Online mentor/EMA chỉ mở sau khi cơ chế có tín hiệu. EMA theo committed mentor optimizer steps, không theo student trial/acceptance count; tất cả student candidates của transaction dùng cùng teacher version. Không đổi active jobs để thực hiện plan này.

## 8. Candidate research question

> Does a frozen teacher decoder induce feature corrections that improve ground-truth segmentation beyond the native student KD direction, and can the useful component be learned by a student-only residual module?

Đây là hypothesis, không phải kết quả. Context-response diagnostic của vòng trước vẫn là hướng riêng; không ghép vào gradient pilot trước khi từng cơ chế có bằng chứng.

## Execution status

V2 cập nhật thiết kế và prior-art audit. Không tạo experiment/run, không fit bridge, không train R, không tải weights, không chạy patient evaluation. Chỉ kiểm tra đại số của phản ví dụ và các artifact links. Xem `GRADIENT_REVIEW_CHECKS.json` và `GRADIENT_V2_PROVENANCE.json`.
