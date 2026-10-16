# Tái lập audit local

Từ root worktree `codex/convergence-audit-20261016`, dùng interpreter đã có sẵn:
Python 3.11.16, torch 2.14.0 trên CPU. Không cài dependencies hoặc tải weights.
Ba probe đầu chỉ dùng Python standard library; probe optimization dùng torch và
các import của source snapshot. Các script chỉ ghi receipt của council này;
không sửa historical run hoặc source. Chạy lại sẽ cập nhật các receipt dẫn xuất
trong `checks/`, nên dùng working copy riêng nếu cần giữ nguyên delivery snapshot.

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 \
  /Users/alvinluong/miniforge3/bin/python \
  reports/convergence_council_20261016/probes/published_evidence.py

rtk proxy env PYTHONDONTWRITEBYTECODE=1 \
  /Users/alvinluong/miniforge3/bin/python \
  reports/convergence_council_20261016/probes/target_theory.py

rtk proxy env PYTHONDONTWRITEBYTECODE=1 \
  /Users/alvinluong/miniforge3/bin/python \
  reports/convergence_council_20261016/probes/geometry.py

rtk proxy env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/alvinluong/miniforge3/bin/python \
  reports/convergence_council_20261016/probes/optimization.py
```

`published_evidence.py` pin manifest tại `92a6074` và đối chiếu source với
`513ddda`; không đòi HEAD luôn bằng parent trước khi thêm báo cáo. Ba probe còn
lại ghi source/input hashes; chạy probe publication trước để kiểm input package.
Đối chiếu `VERIFICATION.json` để thấy hash của các script đã thực thi.

| Probe | Phạm vi PASS |
| --- | --- |
| published_evidence | 118 published hash/size, 24 exact archived-source identities, saved-count metric arithmetic, epoch CE−H bookkeeping |
| target_theory | Entropy bound, perfect-fit abstention, wrong-name Q/Z counterexample, bank support arithmetic |
| geometry | Header-metadata join, support strata, sampling geometry và synthetic operator counterexample |
| optimization | Manufactured gradient wiring, một feature-only optimizer step, batch-regrouping và coarse/native counterexamples, step-budget arithmetic |

Một optimizer step synthetic **đã chạy** trong probe optimization; không có
patient training. Nó không nạp patient checkpoint hoặc khởi tạo CineMA teacher.
Không có prototype solution được kiểm chứng trên dữ liệu thật ở đây.

Tạo lại hình vào thư mục mới:

```bash
rtk proxy env MPLCONFIGDIR=/private/tmp/self-audit-plot-config \
  XDG_CACHE_HOME=/private/tmp/self-audit-plot-cache \
  /Users/alvinluong/miniforge3/bin/python \
  reports/convergence_council_20261016/probes/plot_convergence.py \
  --repo . --out /private/tmp/self-audit-convergence-plot-NEW
```

Thay `NEW` bằng tên chưa tồn tại. Plot script kiểm CSV với Git pin, vẽ đủ 50 epoch,
ghi PNG/SVG và hashes. Hình chỉ biểu diễn published online losses, không phải
fixed-checkpoint fidelity hoặc quality trajectory. File PNG trong delivery đã
được xem kiểm tra bố cục; không chỉnh sửa số liệu hoặc làm trơn curves.

Chưa chạy: bank-array replay, pixelwise checkpoint analysis, official Astra
pretrained runtime, patient pilot, source correctness evaluation mới hoặc GPU.
Không chạy lại test toàn repo vì source/runtime/config/baseline không đổi; kiểm
tra tập trung vào các phép tính và probes mới, source integrity và bảo toàn file.
