# Frozen Transition Bank + B0–B6 — PLAN ONLY

2026-09-08. Không training, không tạo real bank, không triển khai Proposal 2/3 trong handoff này.

## Prerequisites

Chọn trained checkpoint có producer/state/source identity; chạy calibration mới theo entropy
2.0.0 và named evaluation contract; khóa threshold trước independent evaluation. Xác nhận
effective patient splits, preprocessing và reuse history. Development ACDC đã dùng chọn thiết kế
không được gọi lại là fresh test; M&Ms giữ evaluation ngoài miền riêng.
Chuẩn bị immutable soft-state/image-feature sidecars và state IDs cho B3/B4/B5; schema hiện tại
không tự cung cấp đủ tensors để train mọi comparator. Review storage/compute budget trước chạy.

## Tầng A — frozen same-candidate bank

Proposal generation không nhận reference GT. Evaluator mới gắn Q_previous/Q_candidate sau đó.
Tách on-policy, always-accept-prefix, synthetic; không trộn synthetic vào primary deployment score.
Mọi comparator nhận cùng candidates/splits và class/empty/neutral contract.

| Comparator | Vai trò / fairness |
|---|---|
| B0 initial/reject-all | Baseline realized gain = 0; không thay independent fully trained single-stage comparator |
| B1 always-accept | Đo proposer headroom và harm, không dùng làm oracle |
| B2 confidence/entropy/edit-size | Fit threshold chỉ trên calibration; đánh giá fingerprint shortcuts |
| B3 candidate-only S(H,A_cand) | Quality baseline được train đầy đủ, không cố dùng head yếu |
| B4 S(H,A_cand)-S(H,A_prev) | Shared S, cùng GT quality targets/data; báo cả capacity-matched và measured compute-matched |
| B5 current transition auditor | Cùng bank/supervision budget, image/state ablations để kiểm tra relational use |
| B6 no-counterfactual/budget-matched | Thay synthetic bằng thêm on-policy transitions; match optimizer steps và examples, báo unique states |

B3/B4/B5 phải báo parameter count, FLOPs/latency thực đo và training exposure. Không gọi so sánh
fair chỉ vì cùng epoch khi số transitions khác nhau. Giữ seeds/cohort/checkpoint lineage.

Metrics: beneficial/harmful/neutral proposal fractions, valid-Q denominator, realized gain,
harmful acceptance frequency **và magnitude**, beneficial capture, coverage/no-op và one-step
regret trên cùng candidate. Regret = max(0,Q_candidate-Q_previous) minus realized one-step gain;
không gọi greedy oracle là upper bound toàn trajectory. Ghi cả undefined cases, không loại A0-empty
có foreground hallucination. Slice-proxy chỉ diagnostic; chính patient/volume, ghi case aggregation
rồi patient aggregation và bootstrap resample theo patient (không coi ED/ES/slices độc lập).

## Tầng B — closed-loop

Sau khi khóa bank comparator/threshold, chạy từng gate trên full trajectory vì reject/HALT và
feedback thay đổi distribution. So real/zero/shuffled local evidence trên **cùng previous state**
để đo quality effect, sau đó trajectory effect. Báo latency/compute tăng thêm và paired patient CI;
nhiều seeds khi thực sự chạy. Native surface metrics chỉ khi geometry đã được xác minh riêng.

## Decision gates

1. Nếu best available proposals không có meaningful gain trong CI: ưu tiên proposer/headroom
   investigation; không thêm relational auditor chỉ để tạo complexity.
2. Nếu B5 không hơn B4 về realized gain/harm tradeoff với budget tương đương: hạ transition novelty
   claim, cân nhắc quality-difference đơn giản hơn; Proposal 2 chỉ là hypothesis mới cần approval.
3. Nếu B6 matched-data/compute xóa lợi ích counterfactual: không claim cơ chế counterfactual hữu ích
   chỉ vì tăng số mẫu; báo sampling effect.
4. Local map chỉ giữ claim feedback contribution khi cải thiện quality, không chỉ làm output đổi.
   Không có positive result nào được giả định trước; không bịa mức Dice kỳ vọng.
