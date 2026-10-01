# Audit adapter benchmark — 2026-09-25

## Kết luận

Adapter hiện tại phù hợp với việc đặt tên cho một partition đã có cấu trúc gần lý tưởng, nhưng chưa đủ khả năng chuyển partition phân cụm thực tế thành segmentation tim hữu ích. Vấn đề nằm ở cả biểu diễn (mỗi connected component chỉ nhận một nhãn), điều kiện nhận dạng quá cứng, phụ thuộc giữa các lớp và thiếu chọn giả thuyết cạnh tranh. Chỉ đổi thứ tự gán hoặc hạ một ngưỡng không giải quyết được đầy đủ.

Đề xuất ưu tiên một phiên bản mới: sinh ứng viên theo nhóm component, chấm điểm mềm, chọn cấu hình BG/MYO/LV/RV đồng thời và giữ VOID khi bằng chứng yếu. Nếu MYO và BG thực sự dính trong một component, cần thêm bước tách vùng dựa trên ảnh hoặc cải thiện partition đầu vào; bộ đổi tên component thuần túy không thể khôi phục đường biên đã mất.

## Phạm vi và bằng chứng

- Đọc implementation tại commit `5985e4f`: `adapter.py`, `region_graph.py`, `semantic_contract.py`, spec v3, integration artifact, script đánh giá và metric contract.
- Workspace ban đầu báo `region_graph.py` có thay đổi; `git diff` không cho thấy nội dung khác biệt. Không sửa file đó hoặc implementation.
- Chạy 7 ca tổng hợp qua chính `adapt_partition()`, bằng helper tạo record hợp lệ trong test hiện có. Mã tái hiện: `reproduce.py`; kết quả: `synthetic_results.json` trong thư mục báo cáo này. GT tổng hợp chỉ được dùng sau adapter để tính Dice.
- Chạy lại bốn module test adapter (topology, permutation, frozen fixtures, firewall): 29 pass, 1 fail ở hash byte của frozen spec. Đây là lỗi LF/CRLF đã xác minh ở lượt trước: file checkout trên Windows có CRLF, hash sau chuẩn hóa LF khớp hash frozen. Không sửa fixture để che lỗi kiểm tra này.
- Đọc report STEGO/PiCIE ở `reports/shared_benchmark_sa224_dev2`, xem montage PiCIE patient004 ED. Mỗi report có 20 slice, 2 volume của patient004; cả hai báo final foreground Dice = 0. Coverage trung bình STEGO = 0.172513, PiCIE = 0.695231.
- Không tìm thấy run cụ thể có Dice khoảng 0.00018 mà người dùng đề cập. Thư mục local này có report/ảnh, không có các thư mục raw/semantic artifact tương ứng. Vì vậy chưa đo được tần suất từng nguyên nhân trên run đó, chưa thể quy toàn bộ Dice thấp cho một nhánh logic.

## Kiểm tra bốn nhận định

### 1. BG có thể nuốt MYO — đúng với component dính liền

`adapter.py:62` chọn component chạm biên có diện tích lớn nhất, sau đó gán BG cho nó và các component chạm biên cùng raw ID. Không kiểm tra mức độ thuần nền, hình dạng hoặc bằng chứng tim bên trong. Diện tích lớn/chạm biên chỉ là prior, chưa đủ chứng minh toàn bộ vùng là BG.

Cần phân biệt **cluster ID** với **connected component**. `build_region_graph()` tách một ID thành các thành phần liên thông 4 hướng. Một MYO tách rời khỏi BG nhưng cùng ID vẫn được xét riêng. Probe `bg_myo_same_id_disconnected` giữ được MYO/LV, Dice mỗi lớp = 1; RV vẫn thất bại do thêm ứng viên kề.

Ngược lại, khi BG và MYO liên thông cùng ID, toàn component bị gán BG. Probe `bg_myo_same_connected_component` cho MYO/LV/RV đều không có pixel dự đoán, Dice = 0, dù coverage = 94.81%.

Đây còn là giới hạn của contract `region_splitting=false`: ngay cả một bộ chọn nhãn hoàn hảo cũng không thể gán hai nhãn cho hai phần của component đó. Muốn sửa triệt để phải cho phép tách có kiểm soát hoặc thay partition upstream.

### 2. Quyết định tuần tự và không xét lại — đúng, nhưng không phải mọi lỗi đều do thứ tự

`adapt_partition()` chạy BG → MYO/LV → RV. `_resolve_enclosure()` loại component đã gán; `_resolve_rv()` cũng bỏ qua chúng. Không có rollback hoặc bước tối ưu lại toàn bộ cấu hình.

Điểm quan trọng: `RegionGraph.component_encloses()` vốn đã loại mọi outer/inner chạm biên. Tất cả component được gán BG đều chạm biên. Vì vậy riêng việc bỏ exclusion BG hoặc đổi thứ tự BG/MYO **không đủ** để cứu MYO dính nền: còn điều kiện chạm biên và giới hạn không tách vùng.

Phụ thuộc gây hại rõ nhất là RV chỉ được xét khi có đúng một MYO đã resolve. MYO thất bại kéo theo RV thất bại, kể cả khi hình dạng RV trong partition chưa thay đổi.

### 3. Logic quá chặt — xác nhận bằng phản ví dụ

`adapter.py:88` yêu cầu đúng một cặp bao kín trên toàn ảnh. `region_graph.py` yêu cầu bao kín tuyệt đối, chạm nhau trực tiếp, cả hai không chạm biên. `adapter.py:119` yêu cầu đúng một ứng viên RV kề MYO.

| Ca tổng hợp | Kết quả thực tế | Dice RV / MYO / LV |
|---|---|---|
| Partition lý tưởng | Cả 4 nhãn resolve | 1 / 1 / 1 |
| BG và MYO cùng component | MYO/LV missing; RV unsupported | 0 / 0 / 0 |
| MYO hở đúng 1 pixel | Mất quan hệ bao kín | 0 / 0 / 0 |
| LV chia thành 2 cluster | Hai cặp bao kín → ambiguous | 0 / 0 / 0 |
| Thêm một vòng kín nhỏ ở xa | Toàn bộ MYO/LV ambiguous | 0 / 0 / 0 |
| Thêm một pixel nhiễu kề MYO | RV ambiguous | 0 / 1 / 1 |
| BG/MYO cùng ID nhưng tách rời | MYO/LV vẫn resolve | 0 / 1 / 1 |

Các Dice này là bằng chứng cơ chế trên ảnh tổng hợp, không phải ước lượng chất lượng ACDC. Probe LV chia đôi đặc biệt quan trọng: biên giải phẫu vẫn còn nguyên nhưng overclustering làm adapter bỏ hết foreground. Đây là lỗi năng lực của adapter, không cần viện đến thiếu đường biên upstream.

### 4. Chưa giải quyết xung đột — đúng

`_assign()` có guard ném lỗi nếu component đã mang nhãn khác. Đây là phát hiện xung đột lập trình, không phải giải quyết cạnh tranh giữa giả thuyết giải phẫu. Nhiều ứng viên dẫn đến VOID; một ứng viên thì được chọn mà không có score chất lượng tối thiểu.

Do đó hệ thống vừa quá chặt khi có nhiều ứng viên tốt, vừa có thể quá dễ chấp nhận một ứng viên duy nhất nhưng sai. Tính duy nhất không chứng minh đó là tim. Không nên thay `len == 1` bằng chọn vùng lớn nhất hoặc ứng viên đầu tiên một cách vô điều kiện.

## Những vấn đề bổ sung

1. **Thiếu ghép nhiều component thành một cấu trúc.** LV/MYO/RV thực tế có thể bị chia thành nhiều cluster. Implementation chỉ resolve một outer/inner pair và một RV; BG chỉ ghép các vùng chạm biên cùng ID. Nền phân thành nhiều ID có thể sót thành VOID hoặc cạnh tranh làm RV.
2. **Xung đột toàn ảnh thay vì cục bộ.** Vòng kín nhỏ ở vị trí khác phủ quyết một cấu hình tim tốt. Không có định vị vùng ứng viên tim hoặc so sánh chất lượng giữa các cụm giải phẫu.
3. **Không dùng độ mạnh của bằng chứng.** Graph đã có area, centroid, số pixel biên và chiều dài tiếp giáp; resolver chủ yếu dùng điều kiện nhị phân. Tiếp giáp một pixel cũng đủ vào tập RV và có sức phủ quyết như ứng viên lớn.
4. **Nhạy với topology ở độ phân giải pixel.** Một khe nhỏ làm mất hole; liên thông 4 hướng nhạy với tiếp xúc chéo. Đây không phải lý do đổi sang 8 hướng tùy tiện: cần ablation connectivity và tolerance theo kích thước ảnh để tránh tạo quan hệ bao sai.
5. **Không có mô hình cấu trúc vắng mặt hoặc hỗ trợ liên slice.** Các lát cắt không có đủ RV/MYO/LV vẫn bị đưa qua cùng chuỗi điều kiện. Không nên ép đủ bốn lớp. Hiện API xử lý từng slice, không dùng tính nhất quán 3D/thời gian.
6. **Ảnh có sẵn nhưng không giúp quyết định.** `central_image` được kiểm tra/hash; intensity và orientation đều disabled. Khi topology mơ hồ, hệ thống không còn bằng chứng bổ sung. Dùng tương phản tương đối hoặc thông tin hình học hợp lệ là hướng thử nghiệm, không phải bảo đảm đúng. Không giả định tim luôn ở giữa hoặc RV luôn cùng một phía ảnh.
7. **Coverage dễ gây hiểu sai.** Coverage đếm cả BG. Probe đạt 94.81% coverage nhưng mọi foreground Dice bằng 0. Cần thêm diện tích dự đoán từng lớp, tỷ lệ slice resolve từng lớp, tỷ lệ foreground/VOID và độ chính xác trên phần được gán.
8. **Test hợp đồng không chứng minh hiệu quả adapter.** Test hiện tại chủ yếu kiểm tra deterministic, invariant raw ID, firewall và đúng hành vi VOID đã quy định. Các điều này vẫn pass khi adapter không tìm được tim.
9. **Không thể nới bằng sửa JSON vài ngưỡng.** `load_and_validate_spec()` kiểm tra canonical hash cố định; quyết định được hard-code. Cần version mới đồng bộ implementation, spec, fixtures và provenance, giữ kết quả phiên bản cũ có thể tái hiện.

## Dice thấp có phải do cách tính?

Trong `scripts/evaluate_visualize_shared_benchmark.py:234`, thống kê lấy từ semantic và GT trên toàn grid; validity không được dùng để xóa GT khỏi mẫu số. Nhãn VOID = 4 nằm ngoài RV/MYO/LV, nên VOID trên foreground thật trở thành false negative. Điều này phù hợp mục tiêu segmentation đầy đủ: không dự đoán được tim phải bị trừ điểm.

`foreground_dice_exclude_v1` loại lớp rỗng ở cả pred lẫn GT, không loại pixel VOID khỏi đánh giá. Nếu GT có lớp mà pred không có, Dice lớp đó = 0. Vì vậy không nên sửa metric để bỏ vùng VOID chỉ nhằm nâng Dice. Có thể báo selective Dice như số phụ, luôn kèm coverage và full-image Dice.

Qua code và report hiện có, chưa thấy dấu hiệu phép tính này tự tạo ra mức thấp một cách sai toán học. Tuy nhiên cần artifact run 0.00018 để kiểm tra label mapping, source mask, hướng ảnh, resize và aggregation của chính run đó. Báo cáo này chưa xác nhận các bước dữ liệu của run chưa có.

## Thiết kế đề xuất để adapter thực sự hoạt động

### Mức A: thích nghi tốt hơn, vẫn giữ nguyên biên component

1. Sinh ứng viên BG/MYO/LV/RV trước khi chốt nhãn. BG là một giả thuyết có score, không phải khóa cứng ngay đầu.
2. Cho phép các nhóm component liền kề tạo thành LV hoặc MYO. Kiểm tra mức độ bao quanh của nhóm, có tolerance với khe nhỏ; không bắt buộc một raw cluster tạo nguyên vòng kín.
3. Chấm điểm mềm bằng tỷ lệ tiếp giáp, mức bao quanh, area tương đối, compactness và bằng chứng nền. Giới hạn số nhóm bằng top-k/beam search để tránh tổ hợp bùng nổ. Tách confidence tuyệt đối với margin giữa hai cấu hình đứng đầu.
4. Chọn cấu hình chung bằng objective minh bạch: unary score của vùng + consistency của MYO/LV/RV − penalty mâu thuẫn. Mỗi component tối đa một nhãn, cho phép nhiều component cùng nhãn, cho phép lớp vắng mặt. Chỉ chốt sau khi so sánh các cấu hình cạnh tranh.
5. VOID dành cho phần không đủ bằng chứng hoặc margin thấp, không tự động phủ quyết toàn bộ cấu hình vì một component nhiễu. Ghi top alternatives, score, margin và lý do chọn/bỏ để audit.

Mức A cứu được các trường hợp chia nhỏ LV, MYO nhiều mảnh, nhiễu RV và nhiều ứng viên cạnh tranh nếu còn đủ bằng chứng. Nó **không thể** cứu hoàn toàn MYO/BG dính cùng component.

### Mức B: xử lý component hỗn hợp

Thử nhánh riêng cho phép tách component bằng superpixel/biên ảnh hoặc watershed có seed từ giả thuyết giải phẫu. Sinh nhiều phương án rồi dùng cùng bộ giải quyết xung đột; không cắt tùy ý chỉ để tạo đủ lớp. Các kỹ thuật này là đề xuất cần thử nghiệm, chưa được kiểm chứng trên dữ liệu repo.

Bước này tạo biên mới, nên kết quả là baseline + refinement từ ảnh. Phải báo riêng mức A và B, áp dụng cùng chính sách cho các baseline và đo thời gian bổ sung. Không tiếp tục gọi đây là chỉ đổi tên cluster. Tốt nhất sửa adapter mức A trước, đo giới hạn biểu diễn rồi mới quyết định cần mức B đến đâu.

## Kế hoạch xác minh và tiêu chí quyết định

1. Thu thập raw partition, semantic, adapter metadata, GT hậu kiểm và provenance đúng run 0.00018. Thống kê theo method/patient/slice: số component, số enclosure, lý do VOID, diện tích từng lớp, BG trên GT foreground và foreground bị VOID.
2. Đo hai diagnostic dùng GT chỉ ở evaluator: remapping theo cluster và remapping theo component. Nếu dùng majority-label, gọi đó là diagnostic pixel accuracy, không gọi là trần Dice. Muốn kết luận trần Dice phải giải bài toán tối ưu gán tương ứng hoặc nêu rõ bound dùng gì. So với mức A để tách lỗi đặt tên khỏi thiếu biên upstream.
3. Chạy cùng raw partitions qua bản cũ, grouping, soft enclosure, joint conflict resolution; sau đó mới thêm intensity/splitting. Đánh giá full-image Dice theo lớp/volume, coverage, precision/recall, tần suất lớp bị bỏ và runtime. Không chạy lại training giữa các ablation adapter nếu không cần.
4. Dùng tập phát triển tách patient để chọn ngưỡng; khóa cấu hình trước test. Nếu dùng GT phát triển để chọn ngưỡng phải ghi rõ calibration có supervision, dù runtime không đọc GT. Không tune trực tiếp theo Dice test.
5. Bổ sung regression: các probe trong báo cáo, MYO nhiều mảnh, vùng tiếp xúc chéo, BG nhiều ID, lớp vắng mặt, component hỗn hợp, hai giả thuyết ngang điểm, hoán vị raw ID và lặp chạy. Giữ test firewall và provenance.
6. Chỉ nhận phiên bản mới khi tăng Dice trên các patient phát triển đã định trước, không chỉ tăng coverage, không tạo nhiều foreground giả trên slice không có tim, và test giữ invariance. Báo cả trường hợp giảm điểm; chưa có bằng chứng để hứa một mức Dice cụ thể.

## Artifact và tái hiện

Từ repository root: `python reports/adapter_audit_2026-09-25/reproduce.py`.

Báo cáo và probe này không thay đổi adapter, frozen spec hoặc metric. Không sử dụng GT để đưa ra nhãn trong adapter.
