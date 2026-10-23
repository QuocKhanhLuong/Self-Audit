# Vòng nghiên cứu 2 — từ nghiệm cục bộ đến cập nhật thực

**Kết luận: chưa có cơ sở nâng cấp solver hoặc công bố joint guard tốt hơn scalar.** Vòng này làm rõ ba giới hạn độc lập bằng CPU probe và phép tính trên kết quả đã lưu: nghiệm local tốt hơn không bảo đảm endpoint sau guard tốt hơn; guard Brier không bảo đảm Dice; kết quả một transaction không xác định hiệu ứng toàn bộ training policy. Đây là failure characterization và numerical qualification, không phải phương pháp mới hay kết quả bệnh nhân mới.

Đọc cùng [vòng 1](../20261023_astra_ultra/DECISION_BRIEF.md). Source semi-supervised v2: `637f38ced217fe1db2347a87924695533056d8f0`; bảng kết quả cũ: `8960b6baac4c6600553d900c303f192c2fcfa4f0`; parent report: `ccfde9bb99b6050e00ad94fbd6d57fcf9afc8b3f`. Sáu worker Astra ultra tiếp tục sáu phần độc lập, sau đó phản biện chéo. Không chạy bệnh nhân/GPU, đổi production, baseline, config hoặc báo cáo vòng 1.

## 1. Normalization sửa được false stall nhưng chưa phải bản thay solver

[A — numerical qualification](workers/A_numerics.md) giữ cùng objective, ridge, box, budgets và class weights trong **114 fixture**. Trong 39 ca scalar có reference độ chính xác cao, số lỗi coefficient lớn hơn `1e-3` giảm từ **10 xuống 0**. Nhưng một ca gần đồng tuyến ở response lớn khiến normalized SLSQP báo success rồi bị kiểm tra ở đơn vị risk gốc từ chối, trả zero; một ray khả thi đã có improvement khoảng **0.0043569**. Do đó không thể kết luận normalization luôn tốt hơn.

Các điểm được trả về đều qua tolerance `1e-10` đã khai báo. Tolerance này khác exact budget: 17 raw normalized points qua tolerance nhưng vượt exact budget trên hệ số float đã lưu; nhiều mức vượt rất nhỏ. Phản biện F yêu cầu chuyển feasibility/ranking từ Decimal70 sang rational arithmetic, và chỉ rõ Gram matrix sau làm tròn không tự giữ exact PSD. Kiểm tra dấu trên hệ số đã lưu không chứng minh tính chính xác trước làm tròn, global optimum hoặc patient risk.

Giữ scalar/coordinate ray làm feasible incumbent có thể bảo đảm local objective không kém **những điểm thật sự được giữ trong cùng tập**. Scalar phải dùng ridge `m*lambda`. Điều này không bảo đảm kết quả sau finite guard.

## 2. Exact joint local optimum vẫn có thể thua scalar sau guard

[B — exact geometry](workers/B_geometry.md) dựng cùng finite softmax head và hai direction cố định, có chứng minh nghiệm joint `J=(1,0)` tối ưu trên box. Cả regularized objective và raw local mean Brier của J đều thấp hơn `S=(1,1)`.

Tuy nhiên J làm actual true-class risk tăng ở mọi scale dương; **cả 8 trial mặc định đều bị từ chối**, trong khi S được nhận ở scale 1 và giảm mọi observed foreground class risk. Đây là phản ví dụ không cần solver lỗi. Với positive tolerance và grid dài hơn, bước rất nhỏ có thể được nhận; mệnh đề thực thi giữ đúng grid/tolerance đã ghi. B cũng giữ trường hợp equality và joint có lợi thật, nên kết quả không khẳng định scalar luôn thắng. Các direction của B được chế tạo thủ công, không phải output của pseudo-CE generator.

## 3. Họ direction sinh bởi code thật vẫn có thể làm Dice giảm

[F — generated-direction witness](workers/F_realizable.md) đóng giới hạn về generator bằng `make_candidates` và `controlled_head_step` thật, teacher là bản sao finite affine head, U/audit features khác nhau, không có U mask. Cả RV/MYO/LV đều có positive audit support.

Trong fixture đã làm tròn, joint được nhận ở scale 0.5 và giảm actual half-Brier từng group từ `[0.1305066, 0.1459237, 0.1674263]` xuống `[0.1230072, 0.1364954, 0.1665799]`, nhưng **RV Dice giảm từ 2/3 xuống 1/2** vì thêm một false positive. Scalar matched comparator cũng có cùng Dice giảm. Đây là giới hạn của surrogate, không phải lỗi riêng joint hoặc chứng cứ plain pseudo tốt hơn tổng thể.

Search có seed cố định, dừng ở success đầu tiên tại trial 292; giữ đủ 292 attempts, gồm 64 guard evaluations. Không dùng `1/292` hoặc `1/64` làm tần suất bệnh nhân. Fixture gốc dùng LR lớn khoảng 9.603; phép đổi biểu diễn về LR 0.001 giữ nguyên effective step và tạo feature lớn, **không chứng minh default LR với native backbone thường gặp lỗi này**. Năm replay variants có cả float32; trạng thái teacher/student là đầu vào tức thời được stipulate, chưa phải lịch sử EMA đã chứng minh reachable.

## 4. Endpoint hiện có không xác định guard headroom

[C — retrospective endpoint arithmetic](workers/C_headroom.md) chọn một complete final model cho toàn bộ mỗi bệnh nhân bằng exposed VAL labels, chỉ để tính oracle sau sự kiện:

| Thư viện final models | Best constant foreground Dice | Whole-patient oracle | Gain |
|---|---:|---:|---:|
| EMA-block / supervised | 75.4727% | 76.8233% | +1.3506 pp |
| EMA-block / EMA-full | 81.9047% | 81.9047% | 0 pp |
| Cả bốn model đã hoàn tất | 81.9047% | 82.1007% | +0.1960 pp |

Full thắng block trên saved patient foreground score ở **18/18** bệnh nhân. Các con số chỉ giới hạn việc chọn giữa final outputs đã có; không chặn headroom của model train lại, ensemble hoặc subpatient routing. Chọn endpoint block/supervised vẫn thấp hơn full 5.0814 pp trên cohort này, nhưng không thể quy khoảng cách đó cho các pseudo transaction gây hại.

C còn tính sharp coupling bounds từ confusion marginals và dựng các probability trajectories có cùng endpoint summaries nhưng guard history ngược nhau. Đây là phép chứng minh không xác định được lịch sử từ aggregates, không phải hai lịch sử native training đã được dựng lại. Không có checkpoint/transaction payload đủ trong Git để chạy lại bệnh nhân trong phiên này.

## 5. Frozen replay và toàn bộ policy trả lời hai câu hỏi khác nhau

[E — sequential identification](workers/E_identification.md) dựng toy affine-probability hai bước: J tốt hơn S trên audit A hiện tại nhưng xấu hơn trên support B; sau EMA, candidate histories phân kỳ. Final B RV half-Brier là **0.405 với J và 0.10125 với S**. Các scalar contrast này là RV-group half-Brier, không phải mean ba foreground groups.

B labels/scores không được dùng trong cập nhật, nhưng B features đi vào teacher proposal ở bước hai. Vì vậy đây là **manufactured transductive support**, không phải untouched test. Proposal map được chế tạo; không giả là pseudo-CE thật. Shadow isolation của toy qua **12/12** kiểm tra thứ tự/lặp; chưa chứng nhận native CUDA replay. Frozen P replay không ước lượng được các state chỉ xuất hiện khi J tự thay đổi teacher và acceptance history.

[D — claim map](workers/D_claims.md) tách rõ local regularized optimum, numerical output, nonlinear accepted endpoint và unseen outcome. Policy chọn actual mean risk nhỏ nhất giữa zero/S/J chỉ có mean dominance trên các endpoint được đánh giá ở đúng audit đó; không có componentwise dominance giữa S/J, novelty hoặc population guarantee. Nó là policy khác với chi phí thêm. Reused TRAIN audit không trở thành độc lập chỉ vì cuối cùng còn hai hoặc ba lựa chọn.

## Quyết định nghiên cứu tiếp theo

1. **Giữ numerical qualification trước:** exact/high-precision scalar reference, original-unit feasibility, finite incumbent, effective movement và retained failure cases. Chưa thay production solver.
2. **So sánh actual S/J endpoints trên cùng frozen state**, với ridge `m*lambda`, cùng candidate/denominator/guard; ghi cả local regularized objective, raw local risk, actual class risk, Dice/FP/FN và chi phí. Nếu optimizer accuracy chưa tương đương, chỉ gọi là so sánh hai implementation.
3. **Chỉ nghiên cứu toàn bộ policy sau khi có transaction provenance đủ:** mỗi policy tự giữ student/teacher/optimizer timeline với exogenous schedules matched. Same-state replay là mechanism screen, không thay final patient comparison. Không mở thêm kiến trúc hoặc ma trận thí nghiệm lớn từ các witness này.

Các bước trên là `PROPOSED/NOT_RUN` trên bệnh nhân. Đóng vòng nghiên cứu này bằng báo cáo và executable fixtures; chưa có bằng chứng mới về hiệu quả patient/GPU hoặc lý do để nâng claim khoa học.

## Kiểm chứng và phản biện

Sáu dossier được kiểm tra chéo theo vòng A→F→A, B→E→C→D→B; [review summary](REVIEW_SUMMARY.md) ghi các challenge và sửa đổi. Coordinator đã chạy lại sáu primary probes trong bản copy cô lập, rồi chạy các probe thay đổi sau review và review probes mới; xem [replay ban đầu](evidence/coordinator_replay/RECEIPT.json) và [replay sau review](evidence/coordinator_replay_final/RECEIPT.json). [Refresh cuối](evidence/coordinator_replay_refresh/RECEIPT.json) chỉ chạy lại F reviewer sau sửa tên trường metadata; cả 11 primary/review executables có receipt khớp hash giao cuối. Fixed F fixture được replay; không lặp discovery search.

Canonical checks trong phiên này: `scripts/check_repo.py` PASS, `pytest -q` **43 passed**, một `torch.jit` FutureWarning; [receipt](evidence/canonical_checks.json). Đây là kiểm chứng software/CPU, không phải clinical evidence. [Preservation receipt](evidence/PRESERVATION.json) và [inventory](INVENTORY.json) ghi phạm vi giữ nguyên và hash artifacts được giao.
