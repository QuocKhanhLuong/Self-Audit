# Proposal 1 — execution plan

Ngày: 2026-09-08, Asia/Ho_Chi_Minh. Trạng thái: PLAN COMPLETE; implementation NOT STARTED.
Update sau plan: người dùng yêu cầu “ko quan tâm ci, làm luôn”. Implementation đã được
khởi động qua Orca Run `run_688c52658c21`; remote CI không còn là blocking gate.
Local full suite, compile và review correctness vẫn bắt buộc. Không mô tả CI chưa chạy
là PASS; kết luận completion phải ghi rõ ngoại lệ Gate F do người dùng chấp thuận.
Planner/reviewer: Astra; implementer duy nhất: AGY qua Orca.
Đây là kế hoạch thực thi, không thay thế docs.md và không phải tuyên bố Proposal 1 COMPLETE.

## 1. Kết luận preflight

Đã pull `--ff-only origin main`: Already up to date. HEAD local và origin/main:
`1305692914c8f50f11135d00d9035b4059f297a5`.
Working tree không có tracked modification; có sẵn `?? reports/astra_review/` và
`?? self_audit_architecture.html`. Bảo toàn cả hai, không tự stage chúng.
File kế hoạch này là file mới duy nhất của lượt lập kế hoạch.

Đã đọc README.md, docs.md và ba báo cáo `reports/astra_review/2026-09-08/{review,evidence,proposals}.md`.
Đọc lại các đường targets → cache → sweep/calibration → runner/evaluation/decomposition;
dataset discovery → selection → validator/loader; entropy → expert/auditor;
preprocessing/resize/volume evaluation; state machine và workflow CI.
Nhận định dưới đây dựa trên source hiện tại và probe chạy lại, không chỉ lấy review cũ làm đúng.

Môi trường: macOS, Python 3.11.16; CPU synthetic tests, không kiểm chứng checkpoint y khoa thật.
Orca 1.4.192 reachable; chưa có Run bind cho phiên này, chưa dispatch AGY.
Đã đọc skill orchestration và hướng dẫn runtime; sẽ dùng executable `orca` nhất quán.
Không dùng terminal AGY của repository khác hoặc thay AGY bằng worker Codex.

## 2. Findings tái kiểm chứng

Mọi file:line sau thuộc SHA `1305692914c8f50f11135d00d9035b4059f297a5`.

| Finding | Bằng chứng hiện tại | Kết luận / xử lý |
|---|---|---|
| Metric mixing | `audit/targets.py:73`, `training/finetune_joint.py:607`, `evaluation/threshold.py:204` dưới `src/self_audit/` | Tái hiện: legacy delta +0.133333 nhưng exclude delta -0.1; replay 0.633333 khác direct 0.4. W1 |
| A0-empty filtering | `training/finetune_joint.py:610` lọc toàn trajectory theo finite A0 | Xác nhận code; thêm regression blank→hallucination. W1 |
| Checkpoint best/last | `scripts/train_self_audit.py:706`, `:721`, `:729` | Cache từ live model, hash chọn best sau đó; chưa có test best≠last. W3 |
| Calibration binding yếu | `evaluation/threshold.py:330`, `:410`; `scripts/audit_checkpoint.py:232` | Schema/margin checking có, chưa ràng buộc đầy đủ actual state/config/protocol. W3 |
| Split validator≠loader | `data/acdc.py:167`, `training/_utils.py:352`, `data/common.py:268` | Probe validator PASS nhưng loader chọn patient001_ED train và patient001_ES val. Dedup sớm còn che duplicate. W2 |
| Entropy heuristic | `models/annotation_expert.py:15`, `:110` | Tái hiện normalized entropy 0.729574 thay vì 0.995511; shift error 0.265937. W2 |
| CI install / collection | `.github/workflows/self-audit-ci.yml:20`, `:25` | PyYAML từ torch index thất bại; glob không bao phủ tests/. W4 |
| Geometry | `data/common.py:374`, `scripts/preprocess_acdc.py:63`, `evaluation/volume_inference.py:581` | Resize không cập nhật sample spacing; raw orientation/preprocessed chain chưa đủ để xác nhận native metrics. W2 minimal hardening; real-native verification deferred |
| Baseline invariants | `models/self_audit_net.py:96`, `:175`, `:201`, `:264` | Gate strict >, reject→HALT và detached auditor còn đúng. Probe audit loss không tạo annotation gradients. Giữ nguyên |

Các vấn đề feedback training, rejected-candidate auxiliary loss, relational auditor,
dead FPN hoặc counterfactual efficacy KHÔNG được sửa trong phase này.

## 3. Quyết định kỹ thuật giới hạn phase

### Metric contract

- Tách tên/version `audit_target_legacy_one_v1` khỏi evaluation contract `foreground_dice_exclude_v1`.
  Giữ nguyên phép tính training targets, loss, boundary của objective hiện tại; golden fixture bảo vệ.
- Mỗi contract khai báo metric space, aggregation, class set, class mapping, empty policy,
  neutral margin và quy tắc biên, schema/contract version. `slice_proxy`, `volume_resized`,
  `volume_native` không được tráo đổi dù cùng tên Dice.
- Cache lưu state scores Q_previous/Q_candidate và đủ class-wise TP/FP/FN để recompute
  selected final state/volume. Không cộng initial của contract A với delta của B.
- Both-empty toàn foreground vẫn là undefined trong exclude: explicit validity/status,
  không đổi thành 0 hay 1 để pass test. Không loại trajectory vì A0 undefined.
  Candidate hallucinating foreground phải có score 0, FP được giữ trong thống kê/evaluation.
  Undefined delta không gán nhãn neutral; báo số undefined riêng.
- Replay chọn state rồi tính cùng scorer/aggregation với direct; so sánh cả finite values
  và undefined masks. Volume Dice tính từ sufficient statistics đã gộp volume, không mean slice Dice.
- Slice calibration tiếp tục mang nhãn diagnostic-only. Cohort/eligibility và số blank/hallucination
  phải hiển thị; cohort không có usable outcome hoặc không có tau thỏa constraint phải fail,
  không âm thầm fallback sang tau vi phạm. Patient/volume protocol phải được khóa trước bank thật.
- Legacy oracle/decomposition nếu giữ score cũ phải ghi contract cũ rõ ràng;
  evaluation mới sử dụng contract yêu cầu. Không đổi deployable gate hoặc gọi greedy oracle
  là global trajectory upper bound.

### Provenance

- Chọn checkpoint → load exact state → kiểm tra state identity → collect/cache/calibrate/diagnose.
  Không chỉ hash một file bên cạnh live model khác.
- Lineage gồm file SHA256, actual model-state digest, resolved configuration/backend,
  checkpoint-producing git SHA, calibration-code git SHA, metric contract,
  preprocessing/geometry signature, effective split/cohort/protocol signatures, t_max và class mapping.
  Producing SHA không được suy từ ngày upload hoặc thay bằng SHA hiện tại khi không biết.
- Cohort calibration và evaluation có vai trò riêng trong frozen protocol: không bắt chúng
  bằng nhau, vì như vậy sẽ ép test reuse. Validate đúng cohort được khai báo cho từng vai trò;
  thay một cohort tùy ý hoặc sai lineage phải hard-fail.
- Legacy thiếu provenance chỉ được inspect/import dưới nhãn unverified, không dùng như
  calibration hợp lệ của strict evaluation. CLI tau không được bypass artifact mismatch.

### Geometry

- Bảo toàn original affine/orientation/shape/spacing nếu có, ghi transform chain và network grid.
  Cập nhật spacing mỗi resize theo kích thước input/output thực, trục tương ứng rõ ràng.
- Không thay interpolation, normalization hoặc canonicalization recipe để làm metadata đẹp.
  Không dựng affine/orientation giả cho NPY cũ. Unknown physical geometry phải chặn native-mm
  HD95/ASSD; pixel metrics nếu dùng phải được gắn nhãn riêng.
- Synthetic anisotropic/axis fixtures kiểm chứng bookkeeping, KHÔNG chứng minh full raw-ACDC
  native correctness. Reorientation/inverse-resampling refactor lớn và raw-dataset verification
  DEFERRED nếu cần; phải giữ safety guard, không sửa bằng một lần đổi tên metric space.

## 4. Wave-by-wave tasks

Đổi thứ tự so với gợi ý: split/preprocessing precede provenance, vì hash của cohort/geometry
không có giá trị nếu resolver và resize semantics chưa thống nhất. AGY làm tuần tự các task
nhỏ; không dispatch nhiều worker cùng sửa `_utils.py`/runner.

### Wave 1 — metric contract và measurement parity

**W1.1 — contract và scoring primitives.**

- Objective: explicit contracts, scorer/sufficient statistics, preserve historical targets.
- Files dự kiến: `src/self_audit/audit/semantics.py`, module contract mới dưới `evaluation/`,
  `evaluation/metrics.py`, targeted annotations trong `audit/targets.py`, regression tests mới.
- Invariants: training target values/loss unchanged; class IDs và exclude semantics không đổi.
- Tests: sign reversal; both-empty/one-sided empty; TP/FP/FN aggregation; legacy golden outputs;
  incompatible space/class/policy/version rejection.
- PASS: số cũ của training giữ nguyên; contract mismatch không thể đi qua API strict;
  regression fail trên code cũ, pass trên patch. Bất kỳ objective drift nào → REVISE.

**W1.2 — cache/replay/calibration/decomposition wiring.**

- Objective: direct selected-state score bằng replay score, không bỏ blank A0 trajectories.
- Files: `training/finetune_joint.py`, `evaluation/threshold.py`,
  `evaluation/audit_decomposition.py`, `evaluation/volume_inference.py`,
  `scripts/cache_validation_transitions.py`, `scripts/calibrate_threshold.py`, focused tests.
- Invariants: proposal generation/evidence trajectory không đổi; GT scoring evaluator-only;
  legacy training diagnostics nếu giữ phải có namespace/contract riêng.
- Tests: direct==replay cho reject-all, accept-prefix, mixed batch, t_max=0,
  blank→foreground hallucination, undefined masks, volume aggregation;
  cache incompatible/legacy rejection và no-feasible-threshold error.
- PASS: Gate B đạt trên synthetic actual collector→sweep path, không chỉ scorer toy;
  final mean không được reconstruct bằng mixed-contract deltas.

### Wave 2 — effective records, entropy, safe geometry

**W2.1 — authoritative split resolver.**

- Objective: validator và DataLoader dùng cùng effective records và cùng signature.
- Files: `data/common.py`, `data/acdc.py`, `training/_utils.py`, dataset regression tests;
  adapter dataset khác chỉ nếu active call graph thực sự dùng cùng resolver.
- Invariants: không sửa tracked manifests; ACDC/M&Ms không gộp; không sinh test split mới.
  Explicit manifest quyết định membership; tag xung đột phải fail, không âm thầm override.
- Tests: patient001_ED train/patient001_ES val fail trước training; duplicate case/path,
  missing/unknown manifest entry, tag conflict, absent optional test, reordered discovery;
  validator_effective_records == loader_effective_records cho tất cả configured splits.
- PASS: fixture leakage bị chặn; phát hiện duplicate trước dedup; signature deterministic.
  Content-hash duplicate audit optional nhưng phải báo có/chưa chạy, không claim bao phủ nếu thiếu.

**W2.2 — typed entropy APIs.**

- Objective: explicit entropy_from_logits / entropy_from_probabilities, không đoán min/max.
- Files: `models/annotation_expert.py`, `models/__init__.py`, helper mới nếu cần;
  entropy callers trong auditor/net/training chỉ khi cần wiring API, tests mới.
- Invariants: preserve đúng probability callers và normalization conventions:
  expert normalized entropy; auditor đang dùng entropy không normalized không được đổi scale.
- Tests: constant-shift invariance, batch-composition invariance, uniform logits≈1,
  probability simplex invalid/NaN rejection, valid probability parity và gradient finite.
- PASS: Gate E; không sửa các probability/target helper khác ngoài nhu cầu đã trace.
  Đây là bug fix có thể đổi predictions của checkpoint cũ, nên preprocessing/model semantic
  identity phải phân biệt version; không tuyên bố old tau vẫn tương thích.

**W2.3 — minimal geometry guard và descriptor.**

- Objective: spacing đúng trên declared grid, provenance giữ original geometry, fail closed khi thiếu.
- Files: `data/common.py`, `data/acdc.py` metadata adapter, `scripts/preprocess_acdc.py`,
  `evaluation/volume_inference.py`, `evaluation/metrics.py`, geometry tests.
- Invariants: không thay pixel processing recipe hoặc tự sửa data cũ; preserve original metadata.
- Tests: anisotropic two-resize chain, axis permutation descriptor, missing spacing/affine,
  image-mask geometry mismatch; native surface request thiếu geometry phải fail.
- PASS: scope nhỏ chứng minh bookkeeping và safety. Nếu cần thay canonicalization/inverse
  transform diện rộng → BLOCK subtask đó và ghi DEFERRED; không chặn phần slice/resized readiness
  đã chứng minh, nhưng không cho phép native claim hoặc native bank.

### Wave 3 — checkpoint/calibration binding

**W3.1 — selected state load và provenance capture.**

- Objective: exact selected best/last state được dùng bởi cache, calibration và diagnostics.
- Files: `scripts/train_self_audit.py`, `training/_utils.py` checkpoint helpers,
  `scripts/cache_validation_transitions.py`, provenance module mới, tests.
- Invariants: không đổi A→B/B→C training recipe, best-selection objective hoặc optimizer behavior.
- Tests: tiny best!=last checkpoint; capture observed parameter/state digest tại cả ba consumers;
  missing checkpoint, incompatible state dict, configuration/backend mismatch hard-fail.
- PASS: Gate C producer-side; hash alone không đủ nếu actual model vẫn là last.

**W3.2 — strict artifact consumer và end-to-end mismatch tests.**

- Objective: versioned lineage artifact validated trước scoring/inference được gắn calibration.
- Files: `evaluation/threshold.py`, `scripts/calibrate_threshold.py`,
  `scripts/audit_checkpoint.py`, `scripts/train_self_audit.py`, tests.
- Invariants: strict > gate unchanged; không silent upgrade historical artifacts;
  calibration cohort và independent evaluation cohort phân vai theo protocol.
- Tests: lần lượt thay weights/config/code semantics/metric/preprocessing/cohort/t_max/class mapping;
  mọi mismatch fail; positive matched artifact round-trip; permitted disjoint evaluation cohort;
  CLI override không bypass provenance; best!=last full synthetic runner calibration branch.
- PASS: Gate C toàn đường, Gate B vẫn pass. Unknown old lineage không được giả lập như verified.

### Wave 4 — CI và frozen-bank readiness

**W4.1 — complete CI.**

- Objective: torch dùng CPU index riêng; general dependencies từ PyPI; collect/execute toàn tests/;
  source/scripts/tests compile. Không kéo external teacher submodules không cần cho active suite.
- Files: `.github/workflows/self-audit-ci.yml`, minimal test dependency specification nếu cần;
  không sửa external gitlinks chỉ để dập warning nếu chúng không gây test failure.
- Invariants: không skip/xfail regression để xanh, không thu hẹp suite.
- Tests: clean CPU dependency install, `pytest --collect-only tests`, full pytest tests,
  compile source/scripts/tests; đối chiếu collected/executed/skipped counts.
- PASS local chưa phải Gate F. Gate F cần GitHub Actions trên exact publication commit PASS.

**W4.2 — transition export schema và evaluator attachment.**

- Objective: path/schema sẵn sàng, không tạo bank nghiên cứu thật.
- Files: `evaluation/` exporter/schema mới, `scripts/audit_checkpoint.py`,
  cache/diagnostic adapter cần thiết, tests; không sửa recurrent operator/loss.
- Fields: patient_id, case_id, slice_idx, stage, previous_state_id, candidate_state_id,
  delta_q, observed accepted/rejected, Q_previous/Q_candidate và validity,
  source on_policy/synthetic/always_accept_prefix, local input/output evidence summary,
  metric contract version, checkpoint hash, split/protocol identity.
- Invariants: snapshots detached; evaluator gắn GT sau deployable proposal generation;
  chỉ export active attempted rows, không fake transitions sau HALT.
  Synthetic GT-derived generation phải tag rõ `gt_used_for_generation`, không quảng cáo GT-free.
  Always-accept policy outcome khác hypothetical learned gate outcome; không trộn hai cột.
- Tests: schema round-trip, stable IDs/chain continuity, duplicate/mismatched state detection,
  sources separated, undefined quality validity, GT-firewall qua actual wrapper và powered
  oracle positive control; missing lineage rejected.
- PASS: tiny synthetic integration export→validate→replay; không gọi đó là real frozen bank.

## 5. Orca dispatch và review contract

Sau khi kế hoạch này được trả, tạo Run mới đúng repository và xác nhận supported AGY agent
launch identity. Nếu Orca không launch/attest AGY được, BLOCK dispatch, không thay worker.
Mỗi task specification phải copy objective/files/invariants/tests/PASS ở trên và ghi base HEAD.
Một AGY worker trong current worktree là mặc định; không đụng terminal khác đang làm việc.

AGY không làm việc một mình trong workspace: preserve edits/reports có sẵn, không revert
thay đổi người khác, không đổi ngoài ownership; phát sinh file ngoài dự kiến phải escalate.
Không auto commit/push; không dài training. Mỗi task return files, rationale, exact commands,
full test output/exit codes, unresolved concerns và Orca task/dispatch identifiers.

Astra dùng task-list/dispatch-show và worker_done receipt; đọc actual diff, không chỉ summary.
Mỗi wave: focused tests → full tests của AGY → Astra chạy lại independently → PASS/REVISE/BLOCK.
REVISE quay lại cùng worker với regression/root cause cụ thể. Runtime timeout không phải failure;
không spawn replacement vì chờ lâu. Reuse hoặc release settled worker đúng Orca lifecycle.

Gate A được kiểm lại sau MỌI wave: supervised model, ConvNeXt-Tiny, A0, spatial residual gate,
global strict threshold, per-sample HALT, detached audit boundary và objectives không drift.
Gates B–E tích lũy nhưng chạy lại trên integrated final diff; Gate F bắt buộc remote evidence.

## 6. Publication và điểm cần quyết định trước push

Chưa commit/push trong lượt này. Có dependency thực tế: GitHub Actions không thể test một
commit chỉ tồn tại local. Do đó không được tuyên bố A–F PASS trước khi có remote CI.

Sau local review, có hai đường hợp lệ cần chốt trước publication:

1. Cho phép nhánh CI tạm: logical commits → push nhánh CI sau Astra local approval → remote
   full-suite PASS → Astra final approval → fast-forward main cùng commits (không squash).
2. Chỉ main: Astra phê duyệt publication sau local gates, push main, chờ Actions; trạng thái vẫn
   PARTIAL cho đến CI PASS. Đây là conditional publication approval, không giả gọi final gate PASS.

Không tự tạo/push nhánh ngoài main khi chưa được cho phép. Nếu yêu cầu tuyệt đối là A–F PASS
trước mọi push, publication BLOCKED bởi thứ tự này; cần người dùng chọn một trong hai đường.
Không force push/rewrite history. Nếu push thất bại, giữ logical commits và báo nguyên lỗi.

Final `implementation_review.md` chỉ được viết với trạng thái thực, gồm HEAD before/after,
finding statuses, invariant/test matrix, CI URL+SHA+counts, contracts, lineage, effective splits,
geometry deferrals và blockers. Report-only follow-up commit ghi CI/tested code SHA rõ ràng,
không dùng report tự chứng thực producing SHA của chính nó.
Chỉ A–F đều PASS mới ghi PROPOSAL 1 COMPLETE; còn lại PARTIAL hoặc BLOCKED.

## 7. Bằng chứng đã chạy lại trong lượt lập kế hoạch

Shell commands dùng `rtk proxy sh -c` theo RTK.md. Không chạy training.

```text
git pull --ff-only origin main
Already up to date.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests
103 passed, 1 warning in 4.15s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 python /tmp/self-audit-astra-MhQ1nC/probe.py
metric legacy previous=.6666666865 candidate=.8000000119 delta=+.1333333254
metric exclude previous=.5 candidate=.4 delta=-.1
actual cache final=.6333333254; direct final=.4
entropy actual=.7295740247; correct=.9955106378; shift_error=.2659366131
audit-only gradient: encoder=0, FPN=0, initial_head=0, expert=0; auditor nonzero
powered firewall: deployable prediction/logit/decision differences=0
oracle positive control: changed pixels=2048, logit_max_diff=8, changed decisions=2

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python /tmp/self-audit-astra-MhQ1nC/split_probe.py
validator validated=True
actual train=[patient001_ED]; val=[patient001_ES, patient002_ED]
Patient leakage: patient001 appears in both train and val

gh run list --workflow self-audit-ci.yml --limit 3 --json databaseId,headSha,status,conclusion,url,createdAt
latest run 34140248580, SHA 1305692914c8f50f11135d00d9035b4059f297a5, completed/failure
gh run view 34140248580 --log-failed
ERROR: Could not find a version that satisfies the requirement pyyaml
ERROR: No matching distribution found for pyyaml
```

CI: https://github.com/QuocKhanhLuong/Self-Audit/actions/runs/34140248580
Run created 2026-09-07T15:49:24Z, không phải run mới của các sửa chữa chưa thực hiện.
Full probe source nằm trong evidence.md đã đọc; temp paths không phải durable deliverable.
103 tests PASS không phủ các lỗi tái hiện; phải chuyển fixtures thành regression tests tracked.
Compile gate chưa chạy lại trong lượt planning này, sẽ chạy trong implementation gates.

## 8. Next-phase handoff — chỉ PLAN sau khi Proposal 1 PASS

Thiết kế frozen transition bank cùng candidates cho B0 initial/reject-all, B1 always-accept,
B2 confidence/entropy/edit-size, B3 candidate-only quality, B4 shared quality difference,
B5 current transition auditor, B6 no-counterfactual/budget-matched control.
Khóa checkpoint, cohort, metric, candidate source và calibration trước đánh giá độc lập;
capacity/data/compute matched, patient bootstrap, latency, realized gain, harmful acceptance
cả frequency/severity, beneficial capture, no-op/coverage và fixed-bank one-step regret.
Closed-loop đánh giá riêng vì feedback/gate thay trajectory. Không train B3/B4/B6 trong phase này.
Headroom, B5 vs B4 và counterfactual value beyond sample-count là ba decision gates tiếp theo.

Không full retraining; không claim Dice improvement; novelty chưa được validated.
