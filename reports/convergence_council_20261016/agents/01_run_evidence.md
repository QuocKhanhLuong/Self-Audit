# Research Report: Kiểm toán evidence run main513ddda batch 12

**Round 1 độc lập, 2026-10-16; vai trò Astra #1.** Bản này được chốt trước phản biện chéo. Audit HEAD: `92a60744efef85558d7c0226c84b7c49d47217ec`; source thực thi: `513ddda8cec3a166f13cfe68fe6379cab014a1a7`. Không sửa baseline/ablation; không training, GPU, ảnh bệnh nhân hoặc mask.

## 1. Research Question

Run không tối ưu được objective, hay đang học objective không dẫn tới segmentation hữu ích? Evidence hiện có nghiêng về **learning–deployment mismatch**. Chưa đủ căn cứ tuyên bố hội tụ đầy đủ, hoặc quy lỗi chính cho optimizer/capacity.

## 2. Key Findings

**VERIFIED_LOCAL_TEXT:** [probe](../probes/published_evidence.py) chạy thành công; [receipt](../checks/published_evidence.json) xác nhận 118/118 file đúng SHA/bytes, 24/24 source archive trùng Git thực thi, các phép tổng hợp metric và CE−H nhất quán. Điều này không xác thực binary bị lược khỏi publication. Replay 80 bank hashes, RNG và 14 frame/cine identities vẫn là **REPORTED**, chưa tái chạy tại đây. [README:95–103][readme]

| Đại lượng | Epoch 1 | Epoch 50 |
|---|---:|---:|
| Feature loss | 0.361056 | 0.209630 |
| Semantic CE | 1.123137 | 1.085173 |
| Weighted entropy H | 1.046016 | 1.046319 |
| Estimated online KL | 0.077121 | 0.038853 |
| Active-step KL, suy từ all-step mean | 0.083293 | 0.042042 |

KL giảm 49.62%, feature loss giảm 41.94%; KL trung bình epoch 31–40 → 41–50 giảm thêm 9.22%. Đây là trung bình online qua các model thay đổi, không phải KL của checkpoint cố định. [Entropy report:14–61][entropy]

VAL Dice 3.1896%, recall 2.5163%, coverage macro 0.496020%; UNKNOWN macro 99.503980%. Pooled coverage từ toàn bộ voxel là 0.473392%: khác mẫu số. **RV có 21,449 accepted voxels nhưng TP=14, FP=21,435.** Vì vậy failure không chỉ là abstention. BG chỉ có 294 accepted voxels; BG precision 100% trên 5 bệnh nhân có mẫu số xác định không chứng minh BG collapse. [Saved metrics:3–2990][metrics], [summary:26–64][summary]

## 3. Related Work / Các implementation liên quan

Không khảo sát literature ngoài: câu hỏi này cần provenance của code đã chạy.

| Implementation | Trạng thái | Liên quan / giới hạn |
|---|---|---|
| GuidanceStudent, `513ddda…` | Run thực thi | CNN 2.5D, feature projection + semantic decoder; teacher/bank cố định |
| Historical `world_left_v2` | Không được merge vào run | Không gán kết quả hiện tại cho thay đổi orientation của nhánh này |
| Q→Z / thiết kế Astra | Chưa được chứng minh bởi run này | Phải đánh giá như giả thuyết mới; không xem đây là thất bại của Z |

Source khai báo `cardinal_v1`, không sửa scientific code; snapshot xác minh trùng Git. [Launch:5–46][launch], [student.py:19–35][student]

## 4. Technical Comparison

Với batch b, đặt `a_b` là foreground eligibility, `w` reliability đã mask permission, `D_b=max(sum(w),1e-8)`. Source dùng:

`CE_epoch = mean_b[a_b * sum_i w_i CE(q_i,p_i) / D_b]`.

Entropy audit thay CE bằng H(q), giữ nguyên gate, nhóm batch và mẫu số; do đó CE−H ước lượng đúng weighted online KL theo **batch measure** này. Không được trừ entropy voxel-pooled hay chỉ active-batch entropy khỏi all-step CE. [student.py:38–59, 86–112][student], [audit.py:64–85][audit]

79,221/85,700 = 92.4399% batch eligible. Permission bật cho cả bốn lớp vì class-patient counts `[62,26,55,47]`; count này không đo correctness, cũng không đảm bảo semantic gradient khác zero. Inactive batch vẫn cập nhật shared encoder bằng feature loss. [student REPORT:3–11][student_report], [student.py:77–105][student]

Entropy chiếm 96.42% CE epoch 50. Điều này giải thích CE floor; không đủ suy ra phần trăm voxel có `max(q)<0.7`. Export kiểm tra native logits, còn loss kiểm tra logits sau bilinear projection về grid 48. Teacher feature loss cũng không chứng minh tên RV/MYO/LV đúng. [config:2–27][config], [student.py:47–58, 138–151][student]

## 5. Novelty Gap / Giả thuyết có thể bác bỏ

**H1 — Mục tiêu mềm và gate không tương thích.** Confidence: cao về khả năng cơ chế, trung bình về mức đóng góp. Q được trộn uniform rồi propagation; student được thưởng vì khớp Q, không vì vượt 0.7. Thuận: H cao, KL giảm, UNKNOWN cao. Nghịch: thiếu phân phối `max(q)`, margins và coverage của perfect imitator. Falsifier tối thiểu: trên artifact TRAIN cố định, tính acceptance của Q và p trên cùng grid, cùng measure. Nếu Q đủ confident nhưng p không, H1 không giải thích chính. Solution: chẩn đoán trước khi thử Z/sharpening. [core.py:265–286][core]

**H2 — Semantic naming/localization sai.** Confidence: cao rằng accepted RV sai, trung bình về nguồn lỗi. Thuận: 14 TP/21,435 FP dù đã vượt gate; support permission không là accuracy. Nghịch: lỗi có thể phát sinh từ student/native projection thay vì anchor. Falsifier: stagewise frozen comparison R0→Q→student, tách correctness có nhãn khỏi confidence không nhãn. Solution ưu tiên kiểm tra named-source/geometry, không mặc định nhiều epoch sẽ sửa. Diagnostic có ảnh/GT chưa được chạy hoặc cho phép trong audit này.

**H3 — Loss grid thấp để lọt native prediction.** Confidence: trung bình. Thuận: native logits bị resize xuống 48×48 trước CE nhưng export trực tiếp native logits; native variation ngoài sampling constraints có thể ảnh hưởng gate. Nghịch: chưa có bằng chứng artifact chứng minh đây là lỗi thực tế. Falsifier: so sánh frozen native prediction với projection về loss-grid rồi ánh xạ categorical cố định; không tuning ngưỡng. Solution chỉ sau khi thấy hiệu ứng: thiết kế loss nhất quán grid ở nhánh thử nghiệm. [student.py:53,147–150][student], [pipeline.py:43–49,81–91][pipeline]

**H4 — Chưa tối ưu đủ hoặc gradient conflict.** Confidence: thấp–trung bình. Thuận: KL vẫn giảm cuối run; shared encoder nhận hai loss. Nghịch: thiếu gradient norms/cosine, fixed-checkpoint KL, capacity control; scalar loss scale không chứng minh conflict. Falsifier: fixed-model objective và gradient decomposition trên cùng frozen minibatches khi có quyền; tiếp đó mới bounded equal-update control. Solution hiện tại là instrumentation, chưa có căn cứ đổi backbone hoặc chạy thêm 50 epoch.

## 6. Dataset and Metric Suggestions

Giữ ACDC 80 TRAIN / 18 VAL, 36 labeled frames; TEST chỉ export. Segmentation cine MRI tim với BG/RV/MYO/LV và UNKNOWN=255. UNKNOWN gây FN trên toàn volume, không bị loại khỏi Dice. Precision thiếu prediction là undefined và bị loại khỏi mean; RV/MYO/LV có lần lượt 11/16/14 defined patients. Giữ frame→patient→class aggregation, không thay bằng pooled counts. Các count pooled ở trên là diagnostic riêng. [evaluator:18–63,71–88][evaluator], [publisher:24–59][publisher]

Giữ nguyên original strict-evaluation failure và recovery; không quy failure cuối pipeline thành training không hoàn tất. Recovery khai báo checkpoint/prediction/threshold bất biến; binary identity vẫn chưa xác minh tại audit. [Recovery:2–9][recovery]

## 7. Recommended Direction

Ưu tiên: (1) chấp nhận diagnostic text đã tái kiểm tra; (2) định nghĩa stagewise perfect-target/gate và native-grid checks; (3) đo gradient/fixed-model KL chỉ nếu phần target/export chưa giải thích; (4) mới cân nhắc matched experimental arm. Mọi thay đổi ở nhánh riêng, giữ baseline bất biến.

Một falsifier rẻ: với probability simplex, `p1>=0.7` suy ra `p1-p2>=0.4`. Margin gate 0.2 đang dư thừa; chỉ hạ margin không thể cứu coverage. Hạ probability sẽ tăng coverage nhưng chưa chứng minh precision/recall hữu ích, đặc biệt RV đang sai. Không chọn threshold trên VAL đã xem.

## 8. Open Questions

- Perfect-Q acceptance và classwise weighted KL hiện chưa có.
- Target error hay student/native error đóng góp bao nhiêu?
- Loss plateau theo checkpoint cố định, gradient conflict và capacity bottleneck chưa đo.
- Batch 8 bị ngắt và batch 12 fresh-init không tạo ablation ngang update; không suy causal batch effect.

Reproduce text-only receipt: `rtk proxy python3 reports/convergence_council_20261016/probes/published_evidence.py`.

Memory chỉ nhắc ranh giới provenance; mọi số liệu trên lấy từ publication hiện tại. Không dùng metric cũ của `world_left_v2` làm evidence cho run này.

<!-- Round 2 peer review will be appended below without rewriting independent conclusions. -->

## Round 2 — Phản biện chéo và stance sau phản biện

**Provenance:** trao đổi trực tiếp với `/root/astra_target_theory` và `/root/astra_optimization` sau khi Round 1 cùng probe đã lưu. Những kết luận Round 1 phía trên được giữ nguyên để thấy thay đổi lập trường.

### Target theory: chấp nhận cận dưới, giữ ranh giới measure

Tôi phản biện rằng mean H không cho exact acceptance fraction; Q→Z có thể làm sai nhãn trở nên tự tin hơn; coarse weighted KL không giới hạn native-grid risk. Peer đồng ý hai caveat cuối, nhưng sửa thiếu sót trong lập luận thứ nhất: mean H vẫn cho **lower bound** bằng extremal entropy.

Theo [receipt target_theory.json:34–40](../checks/target_theory.json#L34), từ moment đã công bố: `m=uniformCE/log(4)=0.9223874133`, `Hbar=H/m=1.1343600625`. Trên simplex bốn lớp, `qmax>=0.7` cho `H<=H(0.7,0.1,0.1,0.1)=0.9404479887`; phần còn lại có `H<=log(4)`. Do đó phần khối lượng không qua probability gate thỏa:

`f >= (Hbar-0.9404479887)/(log(4)-0.9404479887) = 0.4349302491`.

**Stance sửa:** H1 có hỗ trợ định lượng mạnh hơn Round 1: ít nhất khoảng 43.493% effective weighted TRAIN target mass không thể đạt ngưỡng 0.7 khi perfect-fit Q, dưới approximate normalization của float32 targets. Đây không phải tỉ lệ voxel, VAL, native output, `h<0.6`, hoặc chứng minh nguồn nhãn đúng. Dùng `m` giữ đúng ảnh hưởng denominator clamp; không thay bằng tỷ lệ active steps. Exact fraction vẫn cần histogram. Peer cũng tiếp nhận chứng minh margin gate 0.2 dư thừa tại probability gate 0.7. Không có bất đồng còn lại về các giới hạn này.

### Optimization: chấp nhận phản biện về causal localization

Tôi yêu cầu evidence real-data trước khi nâng gradient conflict/capacity thành nguyên nhân chính. Peer xác nhận **không có evidence đó**: source chỉ chứng minh shared gradient paths; probe random/manufactured chỉ xác nhận wiring, logit drift và tồn tại coarse/native ambiguity. [04_optimization.md:9–13,36–40](04_optimization.md#L9)

Peer phản biện ngược đúng rằng accepted RV TP/FP không xác định lỗi nằm ở Q/anchor hay student/native mapping, và UNKNOWN không đủ chỉ định cơ chế chính. **Stance giữ:** output failure của RV được xác nhận bằng saved counts; causal localization vẫn UNKNOWN. Bound TRAIN của target theory củng cố một cơ chế abstention, không giải thích định lượng failure VAL.

Tôi yêu cầu hai chỉnh diễn đạt ở report peer: tách saved-arithmetic verification khỏi REPORTED original-bank RNG replay; tách P0 forward/gradient không optimizer-step khỏi counterfactual optimizer update trên clone. Đó là chẩn đoán đề xuất cho giai đoạn được cho phép sau này, không phải công việc đã chạy trong council.

**Kết luận sau đối thoại:** evidence bác bỏ cách đọc “CE gần phẳng nên model không học”. Chưa có cơ sở chọn optimizer, capacity hoặc naming làm nguyên nhân duy nhất. Ưu tiên target/gate và coarse/native diagnosis vẫn giữ; gradient/capacity controls đứng sau. Không đề nghị sửa baseline, tuning VAL hoặc mở training ở phiên này.

[readme]: ../../stable_guidance_main513ddda_b12_20261016/README.md#L95
[summary]: ../../stable_guidance_main513ddda_b12_20261016/summary.json#L26
[metrics]: ../../stable_guidance_main513ddda_b12_20261016/completed_run/run/evaluation_val_verified_grid/METRICS.json#L3
[launch]: ../../stable_guidance_main513ddda_b12_20261016/completed_run/LAUNCH_MANIFEST.json#L5
[entropy]: ../../stable_guidance_main513ddda_b12_20261016/completed_run/analysis/semantic_convergence/REPORT.json#L14
[audit]: ../../stable_guidance_main513ddda_b12_20261016/completed_run/analysis/semantic_convergence/audit.py#L64
[student]: ../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/student.py#L19
[student_report]: ../../stable_guidance_main513ddda_b12_20261016/completed_run/run/student/REPORT.json#L3
[config]: ../../stable_guidance_main513ddda_b12_20261016/completed_run/source/configs/stable_guidance_experimental.json#L2
[core]: ../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/core.py#L265
[pipeline]: ../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/pipeline.py#L43
[evaluator]: ../../stable_guidance_main513ddda_b12_20261016/completed_run/source/src/self_audit_guidance/evaluate.py#L18
[publisher]: ../../stable_guidance_main513ddda_b12_20261016/completed_run/source/scripts/publish_guidance_metrics.py#L24
[recovery]: ../../stable_guidance_main513ddda_b12_20261016/completed_run/evaluation_recovery/RECOVERY_STATUS.json#L2
