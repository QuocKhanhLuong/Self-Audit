# Protocol phân biệt nguyên nhân — đề xuất, chưa chạy

## Ranh giới và đầu vào

Mục tiêu đầu tiên là định vị thất bại, không tối đa hóa Dice trên VAL đã thấy.
Giữ baseline/history và reference Astra nguyên vẹn. Mỗi phép đo phải gắn SHA
source, bank, teacher, checkpoint, split, view transform và sampling plan.
Binary artifacts không có trong publication; các lệnh audit text của phiên này
không thay cho những phép đo dưới đây. Không tự tải dữ liệu/weights hoặc chạy GPU.

Recipe hiện tại: GuidanceStudent width32, cardinal_v1, Q cố định, batch12,
LR3e-4, seed42. Checkpoint cuối không chứa optimizer moments; pilot từ đó dùng
fresh optimizer, không gọi là resume chính xác. Mọi arm có cùng nguồn dữ liệu,
state ban đầu và split, trừ yếu tố chủ định thay đổi. Training vẫn no-GT.

## P0 — Frozen TRAIN audit, ưu tiên đầu tiên

**Không optimizer step; không sửa artifacts.** Khóa tối đa 128 minibatches TRAIN,
bao phủ 80 patients và các Q/source strata khả dụng; ghi cả stratum không có
support. Sampling không dựa trên correctness/VAL. Hạn mức này là đề xuất engineering,
không phải tính toán statistical power. Giữ eval mode và deterministic state.

Trên từng batch lưu:

- Native logits `L`; common logits `A(L)=bilinear(L,48x48,align_corners=False)`;
  `p_common=softmax(A(L))` đúng operator training, và `p_native=softmax(L)`.
  Không thay bằng `resize(softmax(L))`. Nếu kiểm phép nội suy probabilities,
  ghi đó là một operator khác.
- Q, reliability, allowed mask, active gate và denominator `max(sum(w),1e-8)`.
  KL/entropy dùng đúng cùng measure; tách all-batch mean, active-batch mean,
  pooled weighted mean. Q-argmax strata không phải ground-truth classes.
- Perfect-Q acceptance; confidence/margin/entropy histogram; native/common
  acceptance và disagreement trên cùng support. TRAIN và VAL/coarse và native
  không được gộp thành một chỉ số teacher→student drop.
- Source `s,V,M,h,Z` chỉ khi có provenance tái dựng hợp lệ. R0/Q/reliability
  cache không đủ xác định riêng raw vote mass; trước tiên tái hiện saved Q và
  labels với tolerance đã khóa. h=0 phải unsupported; tiny-h mask và excluded
  weight được báo đầy đủ.
- Gradient norms/cosines của feature và semantic objectives theo encoder block;
  norm trước clip, clip activation. Backward/autograd là phép đo, không update.

**Decision rules:**

| Kết quả P0 | Điều bị bác bỏ / bước tiếp |
| --- | --- |
| Q đã hiếm vượt gate và p common fit sát Q | “Chỉ cần train lâu hơn để confidence lên” không được objective hiện tại hỗ trợ; điều tra target/source/admission |
| Q rõ, p common còn fit kém | Xem residual theo class/source và input/context; mới làm P2 optimization |
| Common fit tốt nhưng native discrepancy lớn | Kiểm operator/decoder; đừng gán ngay cho LR hoặc anchor |
| Named nguồn chưa xác lập, hoặc nguồn fail negative control | Không dùng confidence/coverage gain làm lý do mở rộng semantic admission |

P0 không chứng nhận anatomy. Source correctness cần đánh giá độc lập; nếu chỉ có
TRAIN no-GT, kết luận đó phải giữ UNKNOWN. VAL đã xem chỉ là bằng chứng mô tả hiện
tại, không được dùng làm vòng chọn giải pháp.

## P1 — Cô lập nguồn và hình học

Khóa cùng ảnh/hash và source pipeline. Kiểm các phép equivariance/invariance có
kỳ vọng rõ: mirror kèm cập nhật đúng affine, permutation không gian được theo dõi,
nguồn hình học không-cardinal, nguồn mất coded affine, và cấu trúc vòng không phải
tim trên manufactured input. Nếu đổi data transform mà không đổi metadata thì đó
là negative control, không coi output invariance là yêu cầu.

Giữ raw counts theo source: số hypotheses, accepted support, contradictions,
patient coverage. Header fallback `(0,0)` là provenance cần kiểm, không đồng nghĩa
chắc chắn laterality sai; coded header cũng không chứng nhận anatomy. `world_v2`
chỉ được so như control trên cùng image bytes, config và operator đã khóa.

**Điều kiện dừng:** source vẫn phát nhãn tên một cách tự tin trên negative/null
controls, hoặc đúng-sai của tên chưa được định nghĩa độc lập. Không cứu bằng thay
threshold theo observed VAL. Chỉ thay một prior/source sau khi giả thiết của nó
được nêu rõ; chưa có cam kết rằng các kiểm tra no-GT này chứng minh source đúng.

## P2 — Micro-pilot tối ưu sau khi P0 còn lý do để làm

Tối đa 32 updates/arm, đo mỗi 8 updates trên hai TRAIN witness panels có patient
disjoint, khóa trước khi chạy. Cùng selected samples, order, geometry, Q/w/gates,
initial state và fresh AdamW moments; giữ số updates và số samples/effective batch
để so sánh. Ghi per-step actual parameter displacement, objective changes,
gradient cosine/norm, class/source-stratified full-distribution KL và feature loss.

1. **Loss conflict:** joint `fl+sl`, semantic-only `sl`, feature-only `fl`.
   Đo drift logits ở feature-only dù decode/head không đổi. Cần hiệu ứng lặp trên
   cả hai witness panels, không chỉ cosine âm ở một minibatch.
2. **LR:** chỉ sau loss-conflict control; so `3e-4` và `1e-4` với các yếu tố khác
   giữ nguyên. Đây là candidate đã chỉ định, không search VAL.
3. **Grouping:** effective-batch24 với microbatch8×3 và12×2, chung aggregate gate
   và global denominator. Đây là objective-control mới; không gọi nó là exact
   replay của legacy per-microbatch normalization/gating.

Mức cải thiện KL tương đối 5% trên cả hai TRAIN panels là **ngưỡng proposal cần
chốt trước**, không được suy ra từ VAL hoặc xem là significance/anatomy benefit.
Dừng nếu hiệu ứng không lặp, chỉ đổi support, hoặc hại một stratum cần giữ. Giữ
negative/mixed results. Không chạy lại 50 epoch chỉ từ một micro-pilot.

## P3 — Candidate objective và capacity, từng yếu tố

**Q/Z:** chỉ khi đã tái hiện mass/support và làm source audit. Cùng initialization,
fresh optimizer, Q-derived weights/permissions/gates, full denominator, minibatch
order và common valid-h mask cho cả hai arms; ghi excluded weight. Không lấy
entropy của Z để tăng weights, không tự mở classes. Đánh giá cả full-Q và
conditional-target diagnostics; CE_Q và CE_Z có entropy floors khác nhau.

Q→Z giữ argmax khi h>0, nên không sửa wrong naming. Nó còn đổi gradient magnitude
và balance với feature loss. Nếu có gain, chưa thể kết luận conditional evidence
là cơ chế đặc hiệu; cần báo norm/cosine và, khi cần phân xử, thêm một control
sharpening được khóa theo cùng entropy trên TRAIN. Việc match entropy không chứng
nhận source đúng. Dừng nếu chỉ confidence/coverage tăng trong khi named source
vẫn chưa xác lập.

**Architecture:** chỉ khi capacity/input-information hypothesis còn đứng vững
sau kiểm one/few-batch fit trên TRAIN. So matched width/context hoặc pretrained
encoder từng yếu tố. Nhánh Astra `2c57960` có conventional-vs-split head control;
chỉ dùng weights được duyệt, cùng bank/schedule/source và compute accounting.
Exact D/U không sinh supervision native hoặc xác nhận RV/MYO/LV. Không bật guard
và image refiner cùng lúc để giải thích gain của backbone.

## Quyết định chất lượng và điều chưa được phép suy ra

Sau khi design/thresholds/checkpoint selection đã khóa, mới lập protocol đánh
giá anatomy độc lập được cho phép. Không dùng current VAL để tune rồi báo như
unseen confirmation; không dùng TEST để chọn thiết kế. Report quality cần đồng
thời coverage, precision/recall, confusion/unknown counts, defined denominators,
patient dispersion và comparison matched. Giảm loss, energy, drift hoặc tăng
coverage riêng lẻ chưa đáp ứng tiêu chí đó.
