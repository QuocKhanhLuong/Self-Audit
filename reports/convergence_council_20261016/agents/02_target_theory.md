# Astra #2 — Target/objective theory, independent round 1

**Kết luận:** có bằng chứng student đang tối ưu objective đã giao; chưa có bằng chứng objective đó xác định đúng RV/MYO/LV. Hai vấn đề riêng là target mềm khó vượt admission và nguồn đặt tên chưa được chứng nhận. Q→Z có cơ sở đại số nhưng không tự sửa nhãn sai.

## Phạm vi và provenance

Audit worktree `main@92a6074`, executed source commit `513ddda8cec3a166f13cfe68fe6379cab014a1a7`, **cardinal_v1**. Đặt `E=reports/stable_guidance_main513ddda_b12_20261016/completed_run`; mọi `core.py`/`student.py` dưới đây là `E/source/src/self_audit_guidance/`, không phải historical world_v2. Probe kiểm tra SHA256 khớp `E/run/bank/FROZEN.json`: core `1b44186c…66ed0`, student `87c33df2…4888`. Chỉ đọc source/JSON; không đọc arrays, checkpoint, không training/GPU, không sửa defaults.

**VERIFIED_LOCAL:** hash source, cộng saved counts, đại số synthetic. **REPORTED:** replay 80 bank hashes/79.221 semantic updates trong audit đã công bố; chưa replay arrays hiện tại. **DERIVED:** các mệnh đề dưới đây với giả thiết nêu rõ. **NOT RUN:** bất kỳ thí nghiệm student mới hoặc kiểm định anatomy nào. Memory cũ chỉ gợi câu hỏi; đã xác minh operator trên snapshot hiện tại.

## Objective thực thi và điều nó không bảo đảm

Ở `core.py:265–280`, đặt `s=1-exp(-mass)`, `V` là phân phối votes đã chuẩn hóa, `U=(1/4,…,1/4)`. Khi đó `R0=sV+(1-s)U`. Với `P` stochastic và `α=.35`, solver có kernel `K=α[I-(1-α)P]⁻¹`; suy ra `M=K(sV)`, `h=Ks`, `Q=M+(1-h)U=hZ+(1-h)U`, `Z=M/h` khi `h>0`. Đây là restart propagation, không phải clamped Dirichlet. Chứng chỉ residual/α (`core.py:147–168`) chỉ chứng nhận nghiệm numerical; cả 80 solver đã báo converged.

`student.py:55–58,95–106` dùng full-Q CE, weights detached, gate theo batch, rồi cộng feature cosine loss. Với `a_b` là gate và `d_b=max(Σw,10⁻⁸)`, loss semantic là `a_b Σw CE(Q,p)/d_b`; do đó CE−H là weighted KL theo đúng measure đó. Không được thay nó bằng unweighted pixel KL. Epoch 50: CE=1,085173, H=1,046319, residual=0,038853; epoch 1 residual=0,077121 (`E/analysis/semantic_convergence/REPORT.json:18–36`). Giảm 49,62% phù hợp với tiến bộ fit, chưa chứng minh hội tụ tối ưu hay tổng quát hóa. Online KL ghép nhiều model trong epoch, không phải KL của checkpoint cuối.

## H1 — Target entropy và admission không cùng mục tiêu

**Evidence / confidence:** cao cho cơ chế, vừa cho phần nguyên nhân hiệu năng. Vì `max Q≤.25+.75h`, `Qmax≥.7` đòi `h≥.6`. Đồng thời `margin(Q)=h·margin(Z)`. Full-Q CE tối ưu tại `p=Q` ở mỗi điểm có weight dương; ví dụ Q=(.15,.55,.15,.15) có KL=0 nhưng export abstain. Ngưỡng export nằm ở `student.py:147–150`; không có reliability gate lúc inference.

Có bound dựa riêng receipt hiện tại: uniformCE/log4=0,922387 là tổng effective measure, đã giữ batch gate/clamp (`audit.py:49–54,71–83`). Entropy chuẩn hóa=1,134360. Nếu Qmax≥.7 thì H(Q)≤H(.7,.1,.1,.1)=0,940448. Do H≤log4, ít nhất khoảng **43,49% effective weighted TRAIN target mass** có Qmax<.7, modulo sai số simplex float32. Đây không phải tỷ lệ voxel, không chứng minh tỷ lệ h<.6, không dự đoán VAL coverage.

**Counterevidence:** target có hard support thật: tổng 868.877/46.575.360 bank nodes, tức 1,8655%; foreground 73.688, tức 0,1582%. Không thể nói toàn bộ target neutral. TRAIN bank grid và VAL native grid/cohort khác nhau; không so hai coverage thành teacher→student causal drop.

**Falsifier / remedy:** histogram TRAIN-only của h, Qmax, Zmax và weighted mass tại từng gate sẽ định lượng nguyên nhân. Q→Z chỉ đáng thử sau source audit, trên common support với cùng weights/gates/denominator. Hạ threshold đơn thuần chưa được biện minh. Điều kiện đủ hữu ích: tên y phải đúng, được phép, `Z_y≥.7+ε`, `Z_y−max_other Z≥.2+2ε` và `||p_Z−Z||∞≤ε`; khi đó export Z đúng, trong khi `Q_y+ε<.7` khiến bản fit Q abstain. Điều kiện về y chưa được xác minh ở run này. Q→Z giữ nguyên thứ tự class khi h>0, chỉ tăng chênh lệch/probability; không sửa argmax sai. Tiny-h cần loại hoặc giữ abstention vì phép chia làm sai số lớn.

## H2 — Nguồn đặt tên sai, solver và distillation tái tạo sai ổn định

**Evidence / confidence:** cao rằng chưa có đảm bảo identifiability; vừa rằng đây là nguyên nhân chính, vì chưa có paired source truth. `core.py:193–199` tự ghi priors không xác định duy nhất tim; wall/cavity được gán MYO/LV tại 242, RV từ adjacency, phía phải và appearance tại 245–258. Feature teacher không phải segmentation head. `E/run/student/REPORT.json:3–11` có class_patients=[62,26,55,47], nên cả bốn lớp allowed: **RV không bị permission tắt**. Permission chỉ là ≥2 patients có accepted hard pixels (`student.py:77–80`), không xác nhận nhãn đúng.

**Counterevidence:** accepted RV support có 5.199 bank nodes; có tên được tạo, nhưng chưa biết độ đúng. Kết quả RV precision rất thấp đã báo trên VAL chứng minh output sai nhiều, không tự định vị lỗi ở anchors thay vì geometry/generalization.

**Counterexample:** probe dùng P gồm hai hàng (.5,.5), α=.35, s=(0,.9), duy nhất nguồn thứ hai vote RV sai. Node neutral nhận h=.2925, Q_RV=.469375, reliability=.023647; Q abstain nhưng Z_RV=1. Weights gốc vẫn dương, batch có foreground nên full-Q và Z đều được học. Tại p=Q, chuyển target sang Z tạo gradient RV=−.530625 trước nhân weight/denominator, đẩy tên sai mạnh hơn. Đây là synthetic stochastic graph, không khẳng định anatomical-anchor realizability. Nguồn sai vẫn sai dù solver residual≈5,55×10⁻¹⁷.

**Falsifier / remedy:** phải chứng minh source semantic validity bằng tiêu chí độc lập/đối chứng đã khóa, không lấy confidence hay source agreement làm accuracy. Giữ unknown ở nơi chưa xác định; không thêm model capacity để chữa tên sai.

## H3 — Batch gate và weighting chọn measure khác task segmentation

**Evidence / confidence:** cao cho code path, vừa cho tác động. Chỉ cần một pixel foreground argmax có w>0 là toàn batch được CE (`student.py:55–58`). Pixel hard-unknown vẫn học nếu reliability>0; class mask lọc argmax chứ không xóa các target channels. Trên run này mọi class allowed nên không có lỗi thiếu quyền; về tổng quát, class bị cấm vẫn có neutral target mass trên pixel được phép. Probe chứng minh gradient có thể tăng channel đó, nhưng đây **không phải nguyên nhân hiện run**.

Chuẩn hóa từng batch khiến batch bằng chứng ít nhưng tổng w>clamp vẫn có semantic scale gần batch dày; không thể hiểu reliability là giảm tuyệt đối ảnh hưởng cả batch. 25/80 patients không có hard foreground không có nghĩa mọi batch của họ inactive. Semantic active 79.221/85.700 bác bỏ nhận định gate chặn gần hết optimization.

**Counterevidence:** weighting/gate ngăn BG-only updates có mục đích; bỏ chúng sẽ đổi estimand và có thể khuếch đại nhiễu. **Falsifier / remedy:** audit effective class mass, batch weight sum, active fraction theo patient và cosine của feature/semantic gradients trên TRAIN; chưa đổi loss hay sampler. KL nhỏ vẫn không chặn lỗi ở low-weight/zero-weight region và không nối TRAIN online measure với native VAL.

## Protocol phân biệt tối thiểu, đề xuất chưa chạy

1. Khóa source/SHA/config/splits; tái hiện R0/Q/reliability/labels trước mọi candidate. Cached R0 không xác định riêng s,V; cần tái dựng raw votes đúng precision. Không chuẩn hóa M như simplex; solve `[sV,1-s]` thành `[M,1-h]` nếu dùng simplex solver.
2. TRAIN-only, không GT: báo histogram/cross-tab source→Q→Z→admission, h, contradictions, gate crossings và weighted class mass. Contradiction/null controls chỉ có thể bác bỏ; không chứng nhận anatomy correctness.
3. Nếu đủ provenance và source grounding, paired Q/Z với cùng initialization, fresh optimizer, seed, minibatch order, feature branch, Q-derived weights/permissions/activation, denominator và common valid-h mask. Báo excluded weight; reject nếu không negligible. Không bật thêm logits classes hoặc tăng weight bằng Z entropy. Theo dõi full-Q KL và entropy cùng conditional-target diagnostics; không dùng CE_Q so trực tiếp CE_Z như cùng floor.
4. Dừng nếu Z chỉ tăng confidence/coverage mà source identity không được xác lập. Không chỉnh bằng VAL/TEST đã xem; thử nghiệm mới cần protocol đánh giá chưa bị thích nghi. Nghiên cứu không-GT vẫn cần thừa nhận rằng internal consistency không thể chứng minh named accuracy.

Receipt: `checks/target_theory.json`; tái lập bằng `rtk proxy python3 reports/convergence_council_20261016/probes/target_theory.py`. Kết quả chỉ là **PASS_TEXT_RECEIPTS_AND_SYNTHETIC_ALGEBRA_ONLY**.

## Phản biện chéo và lập trường sau trao đổi

**Astra #3 / geometry** (`03_geometry_sources.md`, trao đổi trực tiếp): metadata fallback28 đóng góp 4.322/5.199 bank RV pixels nhưng chưa đủ quy lỗi RV cho orientation. Tôi phản biện rằng tỷ lệ support này không phải tỷ lệ lỗi; coded52 cũng chưa là trusted anatomy. Hai bên thống nhất ưu tiên source/operator audit trước Q→Z; reflection/rotation kèm affine, nonheart-annulus và null-source controls chỉ bác bỏ consistency/đặc hiệu, không chứng nhận named correctness. Revised stance: H1 admission là cơ chế có chứng minh, còn H2 source-vs-geometry causal attribution chưa giải quyết.

**Astra #1 / evidence** phản đối suy fraction từ mean entropy. Tôi làm rõ đây là *lower bound*, không exact prevalence: nếu f là weighted mass dưới .7 thì `Hbar≤(1−f)H(.7,.1,.1,.1)+f·log4`; thay receipt cho f≥.434930. Astra #1 chấp nhận correction này. Cần histogram để biết tỷ lệ thực. Tôi chấp nhận hai bổ sung: coarse weighted KL không kiểm soát native risk; ngưỡng probability .7 đã kéo theo runner≤.3, margin≥.4, nên margin gate .2 **dư thừa** với simplex4 và thresholds hiện tại. Chỉnh riêng margin không đổi output. Saved counts RV=21.449 accepted, TP14/FP21.435 (Astra #1) càng bác bỏ đề xuất mở admission đơn thuần. Kết luận giữ nguyên: lượng hóa admission loss trước, nhưng chỉ cho phép sharpening khi nguồn tên và operator được kiểm tra độc lập.

**Astra #4 / optimization** (`04_optimization.md`) chất vấn target-specific attribution; tôi đồng ý. Giữ coefficients/AdamW không giữ gradient balance: `∇logits L_Z−∇logits L_Q=w(Q−Z)/d`, rồi shared-encoder Jacobian đổi norm/hướng so với feature gradient. Revised caveat: paired Q/Z chỉ nhận diện tổng tác động đổi target. Ghi per-block norms/cosines và fixed-Q/common/native diagnostics; CE_Z thấp hoặc coverage cao chưa chứng nhận anatomy. Nếu muốn claim riêng conditional evidence, cần thêm sharpening control entropy-matched được khóa bằng TRAIN; entropy bằng nhau vẫn không bảo đảm gradient vectors bằng nhau. Không có probe/training bổ sung cho trao đổi này.
