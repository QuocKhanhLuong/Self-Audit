# Self-Audit / Candidate C: nghiên cứu từ giả thuyết đến phép bác bỏ

Ngày nghiên cứu: **12/09/2026**. Code được đối chiếu với latest `main` tại thời điểm bắt đầu: **`9726e6d519c33035489ca7a5e4681a046493d89e`**. Phạm vi là nghiên cứu; không đổi model, curriculum, cấu hình hay acceptance policy.

**Trạng thái cuối:** đã dừng search theo yêu cầu và hoàn tất review/tổng hợp từ nguồn đã có. Bản nháp 03 bị thay thế; bản 04 được viết lại và kiểm tra. Đây là bộ nghiên cứu có giới hạn nguồn được ghi rõ, không phải chứng nhận exhaustive novelty hoặc kết quả thực nghiệm.

## Kết luận chính

**Candidate C đáng kiểm chứng, nhưng hiện chưa đủ bằng chứng để coi nó là đóng góp phương pháp chính của bài.** Dynamic Window hiện tại nên được xem là thành phần triển khai/baseline. Câu hỏi mạnh hơn không phải “attention này khác các attention khác ở đâu?”, mà là:

> Khi evidence về FIX/REGRESS có sai số, can thiệp vào computation của correction đã được chấp nhận có giúp giữ phần sửa đúng và khôi phục phần làm sai tốt hơn rollback trực tiếp hoặc sửa trạng thái hiện tại, với cùng giới hạn chi phí hay không?

Đây là câu hỏi kiểm chứng được. Nó buộc **geometry** và **historical replay** phải chứng minh vai trò của mình, thay vì dùng hai yếu tố này làm tiền đề.

## Những phát hiện làm thay đổi hướng lập luận

**1. Có một đối chứng lý thuyết rất mạnh: oracle local rollback.** Theo định nghĩa true REGRESS, nhãn trước correction là nhãn đúng. Nếu biết chính xác vùng này và trả riêng nó về nhãn cũ, ta sửa được toàn bộ REGRESS mà không đụng true FIX. Đây là oracle dùng GT chỉ để phân tích, không dùng khi inference. Vì vậy geometry không thể thắng oracle ấy trên riêng chỉ số “khôi phục REGRESS”; giá trị của C phải xuất hiện khi evidence có sai số, trên các sửa đổi hữu ích khác hoặc trên đánh đổi lợi ích–tác hại–chi phí.

**2. Với rollback cứng, có công thức giá trị rất đơn giản.** Nếu `M` là vùng được chọn:

\[
\Delta N_{\rm correct}=|M\cap R_{\rm true}|-|M\cap F_{\rm true}|.
\]

Nếu evidence là xác suất được hiệu chỉnh đúng, lợi ích kỳ vọng trên một pixel của action này là `p_REGRESS − p_FIX`. Đây là một baseline sát bài toán Self-Audit mà C phải vượt qua. Không cần đổi tên một cơ chế attention để có phép so sánh đó.

**3. Solver hiện tại không giải bài toán tối ưu như ký hiệu argmin dễ khiến người đọc hiểu.** Nó lấy một gradient tại factual support, thử tối đa hai proposal rồi fallback. Hạng `lambda * ||Q-P||²` có gradient bằng 0 tại điểm khởi đầu; λ chỉ tham gia sàng lọc kết quả. Một proposal bị loại không chứng minh không tồn tại hướng sửa khả thi. Báo cáo có phản ví dụ cụ thể cho điều này.

**4. Counterfactual sửa được chưa chắc trạng thái tiếp theo sửa được.** Khi retained state bằng factual state, C3 là nội suy logits giữa factual và counterfactual. Gate có thể quá nhỏ để nhãn đúng mới vượt nhãn sai cũ. Vì vậy phải đo riêng `B_counterfactual`, candidate sau C3 và state cuối cùng sau Auditor. Chỉ báo cáo loss giảm bên trong solver sẽ bỏ qua lỗi chuyển kết quả này.

**5. Exact replay là điều kiện của can thiệp tính toán, không phải bằng chứng về nguyên nhân giải phẫu.** C1 giúp so sánh hai nhánh của cùng một computation. Nó không chứng minh tọa độ ban đầu là nguyên nhân thực sự của lỗi, và không biến tọa độ mới thành lời giải thích y khoa. Đồng thời, symmetry theo thứ tự K điểm không làm vô nghĩa các đại lượng geometry bất biến như mật độ, spread và khoảng cách tới boundary.

**6. Có hai nhiễu thực nghiệm lớn: exposure và số lượt được sửa.** Evidence ở auxiliary bootstrap đến từ Auditor chưa được huấn luyện. Nonzero evidence không đồng nghĩa với evidence hữu ích. Với ba lượt, đường đi thông thường chỉ có một action restitution; no-op vẫn có thể bị Auditor reject rồi HALT. Một chênh lệch kết quả có thể đến từ exposure hoặc trajectory, không nhất thiết đến từ geometry.

## Nghiên cứu rộng hơn ở đâu?

Phạm vi được mở từ attention sang **backpropagating refinement, preservation/consistency, algorithmic recourse, spatial perturbation, historical state editing, sequential state-distribution shift, và risk calibration**. Với phương pháp gần, báo cáo dùng phép thay biến và đối chứng thực nghiệm để tìm phần trùng lặp, không chỉ so sánh tên bài.

Ba tài liệu 2026 đã kiểm tra trực tiếp minh họa vì sao cần thận trọng:

* [ActionSplice, §3](https://arxiv.org/html/2609.08230v1) dùng matched rollback để học chuyển trạng thái khi action thay đổi; deployment tránh replay đã hoàn tất. Nó thách thức claim rộng về sửa lịch sử computation, nhưng không đồng nhất với C.
* [LeCor, §3.3](https://arxiv.org/html/2609.09477v1) học phản ứng sau bước cập nhật từ click trên case adapters. Nó thách thức hướng “huấn luyện để correction tốt hơn”; khác C ở nguồn feedback và biến được tối ưu.
* [Conformal Risk Control for Non-Monotonic Losses](https://arxiv.org/html/2602.20151v1) cho thấy không được kết luận calibration bất khả thi chỉ vì loss không đơn điệu. Những giả định cần thiết vẫn chưa được chứng minh cho policy Self-Audit.

Các tài liệu này được ghi là preprint khi chưa xác minh venue. Không có tuyên bố rằng toàn bộ prior art đến năm 2026 đã được đóng hết.

## Cách đọc bộ báo cáo

| Báo cáo | Câu hỏi được giải quyết |
|---|---|
| [01 — Problem formulation](01_problem_formulation.md) | Bài toán thật là gì? Có những deduction, giới hạn và phản ví dụ nào? |
| [02 — Mechanism analysis](02_mechanism_analysis.md) | Code C1/C2/C3 thực sự tính gì? Replay, gate, Jacobian, history và gradient có giới hạn nào? |
| [03 — Prior-art challenges](03_prior_art_challenges.md) | Những cơ chế nào có thể giải thích hoặc thay thế ý tưởng này? |
| [04 — Falsification protocol](04_falsification_protocol.md) | Đo gì, so với ai, kiểm soát nhiễu ra sao, khi nào phải bác bỏ claim? |
| [05 — Evidence ledger](05_evidence_ledger.md) | Nguồn nào đã đọc đến phương pháp; điều gì chỉ là suy luận hoặc chưa đo? |
| [06 — Research decision](06_research_decision.md) | C nên giữ vai trò gì trong bài và cần vượt qua các bước kiểm chứng nào? |
| [07 — Review record](07_review_record.md) | Worker nào được dùng, bản nháp nào bị loại, kiểm tra nào thực sự được chạy? |

Đã chạy [năm kiểm tra số học độc lập](probes/restitution_counterexamples.json) cho các deduction. Đây không phải kết quả model được train, không phải test GPU, và không chứng minh novelty hay hiệu quả.

## Quyết định nghiên cứu tiếp theo

Làm theo thứ tự **opportunity → evidence → controllability → transfer/preservation → necessity → generalization**. Đầu tiên dùng cùng một factual history cho các action để cô lập cơ chế; sau đó mới so toàn bộ policy có accept/reject/HALT. Baseline phải có cùng evidence, cùng bảo vệ FIX và đối chiếu chi phí thực tế. Tách ACDC, frozen ACDC→M&Ms và native M&Ms.

Nếu một đối chứng rẻ hơn đạt cùng lợi ích trong một phép so sánh đủ chính xác, C không nên là headline contribution. Nếu C có lợi ích riêng bền vững trên các kiểm chứng này, phần có thể bảo vệ là **can thiệp có tham chiếu transition để giữ correction hữu ích dưới evidence không hoàn hảo**. Đánh giá chủ quan hiện tại về novelty: **Dynamic Window 2/10; Candidate C 4/10**. Chưa có căn cứ nâng điểm bằng các software test.
