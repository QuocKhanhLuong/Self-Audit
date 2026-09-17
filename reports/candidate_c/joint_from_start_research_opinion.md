# Nhận định nghiên cứu sau khi chuyển sang joint từ epoch 1

Ngày: 2026-09-12. Code đối chiếu: `bb639ec118af72dac0d5e24339a5e325e2d6d163`.
Đây là phản biện nhiều góc độ, tổng hợp nghiên cứu đã làm và đối chiếu code mới.
Không mở lại tìm kiếm literature, không chạy huấn luyện khoa học, không thay đổi
model, loss hay acceptance policy trong bản nhận định này.

**Quyết định:** giữ Candidate C làm giả thuyết cần kiểm chứng. Joint từ epoch 1
đúng với lựa chọn của người dùng và bỏ sự chờ đợi do curriculum cũ, nhưng chưa
chứng minh vòng feedback học được, càng chưa chứng minh cần historical geometry.
Không đổi sang Candidate B.

## 1. Bằng chứng đã có và chưa có

| Loại | Đã có | Giới hạn |
|---|---|---|
| Nghiên cứu đã làm | Phân tích cơ chế, prior-art có giới hạn, bảy deduction và protocol bác bỏ trong [01](../research_deepening/2026-09-12/01_problem_formulation.md), [03](../research_deepening/2026-09-12/03_prior_art_challenges.md), [04](../research_deepening/2026-09-12/04_falsification_protocol.md) | Không phải kết quả trained model; không đóng hết prior art |
| Kiểm tra toán học đã chạy | Năm phản ví dụ số học, [JSON](../research_deepening/2026-09-12/probes/restitution_counterexamples.json) | Chỉ kiểm tra arithmetic của ví dụ |
| Phần mềm joint đã kiểm tra ở lượt trước | 107 test; native CLI synthetic ACDC và M&Ms, mỗi bên hai optimizer steps; [review](joint_from_start_review.md) | CPU, cấu hình tiny; không đo RTX 4070 hoặc hiệu quả segmentation |
| Kết quả solver trong hai smoke đó | Mỗi cohort: 4 attempts, 0 feasible, 4 fallback | Không được diễn giải thành C sửa được lỗi; số replay failure bằng 0 không tự chứng minh replay đã được thử đầy đủ |
| Kết quả huấn luyện thực của người dùng | Chưa có artifact/log thời gian/checkpoint được kiểm tra trong nhận định này | Không biết Dice, conditional evidence quality, correction gain, latency hoặc hiệu quả M&Ms thực |

Các đoạn nói về bootstrap 100 epoch trong báo cáo nghiên cứu trước thuộc HEAD
`9726e6d`. Chúng không mô tả profile joint mới. Profile cũ vẫn là baseline riêng,
không được xóa dấu vết lịch sử hay nhầm rằng mọi entrypoint đã đổi default.

## 2. Góc học máy: đồng thời cập nhật chưa đủ tạo vòng học sửa lỗi

Ở profile mới, tất cả epoch dùng
\[
L=L_{ann}(S_{retained},Y)+L_{audit}(\operatorname{stopgrad}(A,B,H),Y).
\]
Hai mạng trainable; Auditor được học cả attempted transition bị reject. GT chỉ
sinh loss/target sau inference. Evidence và gate runtime đến từ dự đoán.
Xem [compute_joint_losses](../../src/self_audit/training/finetune_joint.py:215)
và [gate của infer](../../src/self_audit/models/self_audit_net.py:1058).

**Deduction từ code:** với một mẫu bị reject ở lượt đầu, state cuối là A0;
annotation-loss gradient tới riêng AnnotationExpert bằng 0 trên mẫu đó.
A0/encoder và Auditor vẫn có thể học. Vì gate rời rạc và detached, annotation
loss không dạy Auditor rằng quyết định reject đã làm mất cơ hội học refinement.

Đây là một vòng phụ thuộc có thể kẹt: annotator chưa sửa tốt → Auditor học reject
→ ít accepted history và ít gradient cho refinement → annotator khó cải thiện.
Đó là *khả năng*, không phải bằng chứng collapse đã xảy ra. Random feedback ban
đầu cũng không đồng nghĩa accept/reject phân bố 50/50.

Thay đổi còn bỏ population synthetic positive/negative/hard-neutral của interval
Auditor-only khỏi profile mới: joint loss dùng các transition thực sự attempted.
Lợi ích tiềm năng là đúng trạng thái policy đang thăm; rủi ro là thiếu các loại
transition hiếm hoặc hữu ích khi policy sớm HALT. Vì vậy cần đo phân bố
positive/negative/neutral, không chỉ loss Auditor giảm.

## 3. Góc temporal: Candidate C có thể giảm số cơ hội học conditioning

Trên đường đi ba lượt thông thường có replay hợp lệ:
`ordinary(t0) → restitution(t1) → ordinary(t2)`.

- t0: generator nhận zero previous evidence.
- t1: solver đọc predicted evidence nhưng replay operator/parameters đóng băng;
  override realized coordinates không huấn luyện generator dự đoán từ evidence đó.
- t2: nếu t1 được Auditor accept, ordinary generator mới nhận evidence về
  transition t1. Nếu t1 reject, HALT.
- Replay/state failure đi ordinary fallback, nên là ngoại lệ cần đếm riêng.

Do đó cần tách:
`real_evidence_attempts` và
`ordinary_annotation_real_evidence_attempts`.
Code đã có cả hai trong
[finetune_joint.py](../../src/self_audit/training/finetune_joint.py:349).

**Deduction, với nhánh ba lượt nêu trên:** cơ hội ordinary có feedback ở t2 đòi
hỏi đồng thời accept(t0), replay-valid(t1), accept(t1). Cơ hội loss đi qua chính
ordinary output t2 còn đòi hỏi accept(t2). Đây là giao các sự kiện, không giả định
chúng độc lập; đạt các điều kiện cũng chưa bảo đảm gradient khác 0 hoặc hữu ích.
Hai lượt trong tiny smoke không thể chứng minh ordinary generator học feedback
trên nhánh này. Xem [partition/transfer](../../src/self_audit/models/self_audit_net.py:1218).

Không có gradient xuyên solver không có nghĩa C không ảnh hưởng training:
innovation detached vẫn đổi điểm tính supervised loss, nên gradient qua retained
state có thể khác. Điều chưa có là mục tiêu trực tiếp dạy generator chọn support
theo evidence của correction ấy, hoặc dạy representation dễ sửa bằng một bước.

**Ý kiến:** bật joint sớm giải quyết lịch đóng băng, nhưng cần kiểm chứng riêng
exposure, gradient và usefulness. Không coi tổng evidence count là chứng nhận
mismatch đã được sửa. Không âm thầm thêm forced accept hoặc candidate loss để
né vấn đề: các thay đổi đó sẽ là một quyết định nghiên cứu khác.

## 4. Góc cơ chế: tại sao không rollback thẳng?

Theo định nghĩa true REGRESS, nhãn trước transition chính là nhãn đúng. Với
rollback cứng trên mask M:
\[
\Delta N_{correct}=|M\cap R_{true}|-|M\cap F_{true}|.
\]
Do đó oracle rollback trên đúng R sửa hết R và không phá F. Nó dùng GT, chỉ là
offline ceiling. Runtime baseline dùng cùng predicted evidence mới là so sánh
công bằng. Nếu evidence là conditional probabilities đã calibrated, action
rollback cứng có lợi ích pixel kỳ vọng `p_R-p_F`; kết luận này không tự áp dụng
cho Dice, rollback logits có gate hoặc một Auditor chưa calibrated.
Deduction đầy đủ tại [01, D1/D6](../research_deepening/2026-09-12/01_problem_formulation.md).

**Giả thuyết đáng bảo vệ:** giới hạn action vào những thay đổi có thể tạo ra bằng
sampling trên feature ảnh giúp chịu được evidence sai tốt hơn rollback output.
Đây là giả thuyết về không gian correction, không phải thêm thông tin quan sát:
Auditor và solver vẫn dùng cùng image/features. Sự liên kết không gian cũng có
thể lan sửa sai sang vùng chưa được bảo vệ. Cần đo hai chiều.

Phép bác bỏ trực tiếp: cùng frozen history, cùng predicted masks, cùng FIX
protection và cùng final Auditor; protected rollback rẻ hơn đạt utility/harm
tương đương với khoảng bất định đủ hẹp. Nếu xảy ra, geometry không xứng làm
headline, dù code replay hoàn hảo.

## 5. Góc tối ưu và chuyển kết quả

Code xác minh C1 rồi lấy một gradient, thử tối đa hai proposal. Nó không cung cấp
nghiệm tối ưu hoặc giấy chứng nhận infeasible.
[Solver](../../src/self_audit/models/self_audit_net.py:294).

Hai deduction đã được kiểm tra toán học vẫn còn đúng:

1. Tại Q=P, gradient của `lambda * ||Q-P||²` bằng 0. Lambda sàng lọc proposal,
   không đổi hướng gradient duy nhất. Không gọi đây là tìm minimum-cost action.
2. Khi S=B, C3 trở thành `N=(1-g)B+g B_cf`. Counterfactual đã sửa class có thể
   bị gate làm mất repair. Ví dụ pairwise margin -0.2 → +0.3 với g=0.2 cho
   margin sau C3 là -0.1. Với bốn class phải xét mọi đối thủ.

Cần ba kết quả riêng: **B_cf → N sau C3 → state retained sau Auditor**. Nếu C
thua ở bước chuyển kết quả, tăng capacity window hoặc giảm solver loss không
trả lời nguyên nhân. Nếu thất bại do hai điểm thử bỏ lỡ hướng feasible, cũng
không được kết luận support geometry không có headroom.

## 6. Góc novelty: đâu là phần có thể bảo vệ?

[Đối chiếu prior-art đã hoàn tất](../research_deepening/2026-09-12/03_prior_art_challenges.md)
đặt f-BRS/G-BRS là đối thủ trực tiếp của frozen-network auxiliary-variable
refinement; counterfactual attention thách thức phép trừ factual; recourse
thách thức target-plus-proximity; historical state editing thách thức claim
rộng về sửa computation trước đó. Không tìm lại hoặc tái chứng nhận toàn bộ
nguồn trong lượt này.

Phần có thể kiểm chứng riêng:
\[
N=S+g\odot[U_{frozen}(r;\widehat Q)-U_{frozen}(r;P)],
\quad U_{frozen}(r;P)\simeq B,
\]
với r là correction ordinary vừa được accept, Q là realized support, restoration
và preservation dùng signed predicted transition evidence, rồi official re-audit.
Tính riêng biệt của tổ hợp này chưa chứng minh giá trị khoa học. Current-state
coordinate optimization là đối chứng bắt buộc để kiểm tra vai trò *history*;
frozen feature/bias refinement kiểm tra vai trò *coordinates* nếu C vượt qua
rollback rẻ hơn.

Đánh giá chủ quan giữ nguyên: Dynamic Window novelty 2/10; Candidate C 4/10.
Joint từ epoch 1 là thay đổi curriculum/experimental condition, không làm tăng
điểm novelty của cơ chế.

## 7. Góc protocol M&Ms và chi phí

Phải báo cáo riêng ba experiment:
(1) ACDC native; (2) frozen ACDC-selected checkpoint → external M&Ms;
(3) native supervised M&Ms. Native M&Ms không chứng minh OOD generalization của
ACDC, và external M&Ms không được tune tau/solver/checkpoint cho experiment (2).

Hai audit script hiện tại được lưu tại [native](mnms_native_followup_review.md)
và [external](mnms_external_followup_review.md). Software verdict không xác nhận
cohort thật có đủ paired files hoặc RTX 4070 có đủ bộ nhớ batch 8.

Profile joint không có interval Auditor-only ở epoch 101–120. Một process đã
chạy với staged config không tự chuyển curriculum sau git pull. Generic launcher
và YAML `self_audit_full*.yaml` vẫn chọn staged nếu không chỉ rõ profile mới.
Không có log timing từ máy người dùng nên chưa xác định tỉ lệ thời gian của
data loading, proposal generation, solver, validation hay checkpoint.

Một C action có thể gồm factual replay, coordinate backward, tối đa hai candidate
checks và final audit, với original batch composition để giữ replay identity.
Hãy đo wall-time/memory theo action và theo epoch; không suy tốc độ GPU từ CPU
smoke. Khả năng chi phí C đáng giá là một giả thuyết riêng.

## 8. Thí nghiệm nhỏ tiếp theo nên quyết định điều gì?

Không chạy full ablation suite trong lượt này. Khi có checkpoint thực:

1. **Vòng học có hoạt động?** Đo first-turn acceptance, attempted/accepted history,
   ordinary evidence exposure, annotation-refinement gradient/updates, t1/t2
   reach rate, local REGRESS precision và true-FIX coverage trên accepted
   transitions. Các metric GT chỉ offline/training diagnostics.
2. **C có cơ hội thực sự?** Đếm patient có accepted mixed FIX/REGRESS, eligible
   solver, C1 attempted/pass, no-regress, no-improvement, infeasible, no-op HALT.
   Không thay A0 để tạo headroom giả.
3. **Action nào hữu ích với cùng thông tin?** Một frozen-history paired pilot:
   identity, protected rollback, current-state support optimization và C.
   Count compute và đo repair, damage, FIX destroyed ở ba state kể trên.
   Đối chứng chưa triển khai phải được ghi pending, không coi switch rollback
   hiện có mặc nhiên đã đủ constraint parity.
4. **Lợi ích do cơ chế hay curriculum?** Trước hết so current-window với C cùng
   profile joint mới và cùng training/data/seed budget. Nếu muốn kết luận về
   joint-vs-staged, đó là factor thứ hai; staged-vs-joint-C đơn lẻ không cô lập C.
5. **Có đáng giữ trong bài?** Đánh giá toàn bộ policy trên tất cả patient, rồi
   mới conditional-eligible subset; uncertainty ở patient level. Mốc useful gain,
   tolerable harm và compute budget cần chốt từ development trước locked test,
   không chọn sau khi thấy external M&Ms.

Nếu không có exposure/gradient thì sửa giả thuyết về vòng học trước. Nếu có
evidence nhưng rollback tương đương với đủ độ chính xác và rẻ hơn, bỏ C khỏi
headline. Nếu solver sửa nhưng C3/audit xóa lợi ích, xác định điểm nghẽn ấy.
Chỉ khi vượt các phép thử này mới có cơ sở viết C thành paper-level contribution.
