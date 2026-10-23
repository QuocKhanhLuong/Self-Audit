# Quyết định nghiên cứu từ Git hiện tại — Astra ultra, 23/10/2026

Status: completed report/probe study with six reciprocal reviews. Research evidence, not authorization for a new patient run.

## Kết luận chính

Nên tiếp tục bằng một nghiên cứu có đối chứng về **lợi ích thực của phân bổ pseudo-update theo lớp**, sau khi kiểm định độ chính xác số của solver. Chưa có bằng chứng để tăng độ phức tạp kiến trúc hoặc tuyên bố joint guard cải thiện phân đoạn bệnh nhân. Hai nguồn Git quyết định là canonical main `637f38ced217fe1db2347a87924695533056d8f0` và nhánh thí nghiệm `8960b6baac4c6600553d900c303f192c2fcfa4f0`.

Hướng hiện tại là **semi-supervised**: nhãn dùng cho supervised step và audit đều thuộc ngân sách bệnh nhân TRAIN. Báo cáo no-GT/Astra ngày 18/10 là lịch sử; không mô tả đúng pipeline hiện tại. Nguồn khởi tạo DINO, EMA, kiến trúc decoder và local quadratic solve không tự tạo thành đóng góp mới.

## Bằng chứng đã có

Các số dưới đây là tính lại từ bảng bệnh nhân được commit trên nhánh thí nghiệm, không phải chạy lại checkpoint. Đây là VAL phát triển đã được xem, một cấu hình L8/S42, crop256/batch52, final10000; không phải một nghiên cứu nhiều seed hay kết quả xác nhận độc lập.

| Arm | Mean foreground Dice | So với supervised | Giới hạn diễn giải |
|---|---:|---:|---|
| supervised | 75.4727% | — | Mốc so sánh đã hoàn tất |
| supervised_extra | 76.2768% | +0.8041 điểm % | Thêm cập nhật có nhãn; không phải đối chứng cùng thời gian chính xác |
| ema_full | 81.9047% | +6.4320 điểm % | Khác block tham số, optimizer/state, learning rate và clipping |
| ema_block | 73.4592% | −2.0135 điểm % | Thấp hơn supervised trong cấu hình đã chạy; chưa biết có tái lập qua seed/subset |
| joint_guard và 4 arm allocation/guard còn lại | NOT_RUN ở full10000 | Không xác định | Đã hủy theo yêu cầu dừng trước đó; pilot/profile không thay thế quality endpoint |

Foreground macro ở bảng là tổng hợp mô tả mới từ ba lớp, không phải endpoint đã preregister. Dice từng lớp kế thừa quy tắc evaluator hiện hành: trung bình các frame có GT của lớp đó trong mỗi bệnh nhân/lớp, rồi trung bình bệnh nhân và lớp foreground. Mỗi frame được tính trên volume Z,H,W; endpoint này không gộp confusion của các frame ED/ES trước khi tính Dice bệnh nhân. Evaluator đặt Dice=null cho mọi frame không có GT của lớp, kể cả khi có FP, rồi bỏ khỏi trung bình. Đây là quy tắc của endpoint hiện hành; khi có FP thì tỷ lệ Dice thông thường vẫn xác định và bằng 0. Phải đọc thêm confusion/FP và mẫu số. Chưa có đủ frame payload để định lượng ảnh hưởng của quy tắc này lên thứ hạng đã báo cáo. Xem `workers/C_empirical.md` và `workers/F_adversarial.md`.

## Phát hiện mới và phạm vi

1. **Solver có thể dừng quá sớm ở đáp ứng nhỏ.** Trên moments hợp lệ, nghiệm SLSQP có thể gần 0 trong khi nghiệm scalar giải tích gần 1. Một fixture dùng xác suất và Jacobian affine-head thực cũng tái hiện. Kiểm tra feasibility vẫn đạt, nên `NUMERICAL_FEASIBLE` không chứng minh đã tìm được một cập nhật hữu ích. Đây là giới hạn numerical/liveness đã tái hiện trên CPU; chưa chứng minh các run bệnh nhân bị ảnh hưởng. Cần moments hoặc trạng thái tối ưu theo run để kết luận điều đó.
2. **Brier giảm không kéo theo Dice tăng, ngay trên cùng audit.** Hai probe độc lập đưa displacement chế tạo vào guard thật: cả ba class-Brier giảm, nhưng hard RV Dice giảm. Đây là phản ví dụ cho một suy luận mạnh hơn docs hiện tại, không phải lỗi của guarantee đã giới hạn trong docs. Hướng chế tạo không chứng minh EMA thường tạo ra tình huống này.
3. **Backtracking có thể từ chối mọi bước dương hợp lệ cục bộ.** Một softmax-head fixture nằm trên biên budget có đạo hàm lớp bằng 0 và độ cong bất lợi: local alpha gần 1, mọi trial hữu hạn đều bị từ chối. Không có strict descent margin thì không bảo đảm liveness; fallback về supervised reference vẫn đúng hợp đồng.
4. **Lịch bệnh nhân có phase-lock.** Với `li=t mod L`, `ui=t mod (80−L)`, mỗi bệnh nhân U luôn ghép với một supervised patient cố định khi L là 4, 8 hoặc 16. Vì audit loại supervised patient hiện hành, một labeled patient không bao giờ đủ điều kiện audit cho U patient đó. B52 vẫn lấy nhiều crop từ cùng patient, không phá liên hệ này. Đây là giới hạn bao phủ/đại diện đã kiểm tra từ source; không phải rò rỉ nhãn hay bằng chứng gây hại thực nghiệm.
5. **Thiếu đối chứng phân biệt joint allocation với giảm bước đơn giản.** Cần scalar aggregate pseudo-direction với cùng finite class guard, alongside same-block plain EMA và extra labeled optimization. `joint_no_class` chỉ bỏ constraint ở local solve; finite class guard vẫn còn. `ema_full` là baseline mạnh hơn với nhiều khác biệt, không cô lập tác dụng của scope.
6. **Chi phí hiện tại gồm instrumentation.** Decoder plain `ema_block` vẫn tính per-source gradients/JVP/moments và có encoder forward lặp. Timing hiện tại không phải tối ưu latency của plain SSL. Manufactured parity checks có thể chỉ ra phần dư thừa, nhưng không chứng minh GPU speedup.

Các điểm 1–4 là kết quả source/toy cụ thể; chúng không chứng minh joint guard thất bại trên bệnh nhân. Ngược lại, việc 43 tests hiện hành đều qua cũng không chứng minh solver hữu ích ở mọi thang đo, Dice được bảo vệ, hay hiệu quả y khoa.

## Nghiên cứu nên theo đuổi

**RQ1:** Sau khi solver được kiểm định về scale và độ tối ưu, joint class-response allocation có giữ lại được lợi ích hơn scalar shrinkage dùng cùng finite guard trên cùng candidate directions không? Đầu tiên kiểm tra trên fixtures có nghiệm tham chiếu và trên replay moments đủ provenance nếu chúng tồn tại. Không dùng tên method hay acceptance rate để thay cho một lợi ích đo được.

**RQ2:** Nếu RQ1 có bằng chứng thuận lợi, lợi ích đó có còn trên dự đoán bệnh nhân ngoài TRAIN khi so với same-block EMA và extra labeled optimization, dưới cùng label budget, init, input, patient schedule và cách chọn checkpoint không? Báo cáo quality, stall, class collateral và chi phí; không biến một tradeoff compute thành attribution thuần thuật toán.

Giả thuyết đóng góp hợp lý nhất hiện tại là **đặc tả và kiểm chứng điều kiện mà tương tác các hướng pseudo-update đem lại giá trị thêm**. Chưa có cơ sở cho tuyên bố first-ever về reweighting, output sensitivity, constrained updates, curvature, hay line search. Đánh giá prior art mới và HCO full algorithm nằm trong `workers/D_prior_art.md`.

Dừng hướng method nếu lợi ích biến mất trước scalar+finite-guard hoặc extra labeled optimization; nếu chỉ có rejection/stall mà không có U utility; nếu phụ thuộc một labeled subset; hoặc chỉ xuất hiện sau thay đổi đồng thời schedule/scope/optimizer. Giữ các kết quả âm và không thêm module để bảo vệ câu chuyện ban đầu.

## Kiểm tra và sản phẩm

Canonical source: `scripts/check_repo.py` PASS; `python -m pytest -q`: **43 passed, 1 FutureWarning**; mechanisms runner PASS trên CPU. Những kiểm tra này được chạy trong phiên hiện tại. Môi trường và log đầy đủ nằm trong `evidence/LOCAL_VALIDATION.json`. Coordinator đã chạy lại 5 script probe trong bản sao source tách biệt, đều exit 0; xem `evidence/coordinator_replay/RECEIPT.json`. Các probe riêng và kiểm tra chéo có receipts riêng; không cộng số scenario thành số bệnh nhân hay số thử nghiệm độc lập.

Báo cáo chỉ bổ sung tài liệu/probe. Source/config/entrypoint sản xuất, dữ liệu và checkpoint được giữ nguyên. 236 tệp sửa dở/untracked của checkout gốc giữ đúng hash, HEAD gốc giữ nguyên; remote main và experiment tip cũng được kiểm tra lại không đổi. Study protocol tiếp theo là đề xuất có thể review, không phải job đang chạy.
