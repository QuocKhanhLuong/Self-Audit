# Proposal 1 — implementation review

## FINAL VERDICT — 2026-09-08 12:46 UTC

**PROPOSAL 1 COMPLETE theo gate local đã được người dùng điều chỉnh.** Astra đã
review diff và tự chạy full suite: **507 passed, 1 warning trong 108.14s**, compile
**51 source files PASS**, `git diff --check` sạch. Remote CI **WAIVED / NOT VERIFIED**,
không có CI run/link chứng minh PASS và không tuyên bố đáp ứng Gate F nguyên bản.
Các phần ghi “ongoing”, “PARTIAL”, “chưa commit” bên dưới là lịch sử, không phải trạng thái cuối.
Đây là REVIEW/IMPLEMENTATION RECORD; các proposal không thay source of truth kiến trúc.

### 1. Revision, môi trường và quyền thực hiện

- HEAD before: `1305692914c8f50f11135d00d9035b4059f297a5`, branch `main`; remote main
  kiểm tra lại trước release vẫn cùng SHA. HEAD after và logical commits ghi ở mục publication bên dưới.
- macOS 26.2 arm64; Python 3.11.16 conda-forge; torch 2.13.0; CUDA unavailable.
  Chỉ synthetic CPU fixtures/subprocess CLI, không real-trained-checkpoint validation.
- Giữ nguyên untracked có trước: `reports/astra_review/`, `self_audit_architecture.html`;
  không stage bản worker report lạc chỗ `wave1_revision2_worker.md` ở root.
- AGY/Claude thực hiện qua Orca, ownership riêng; Astra yêu cầu REVISE khi probe còn bypass,
  kiểm tra actual diff/test output, không tự viết lại production thay worker.

### 2. Findings và quyết định cuối

| Finding | Trạng thái | Bằng chứng / giới hạn |
|---|---|---|
| P0 metric mismatch/cache sign reversal | fixed | Named contracts, Q/stats riêng, direct/replay parity, hallucinated foreground không bị bỏ vì A0 empty |
| P0 best-versus-last checkpoint | fixed | Load/bind exact selected state trước cache/calibration/diagnostics; tiny best != last runner tests |
| P0 calibration lineage/header bypass | fixed | Strict header/nested lineage; null/deleted/type/tau/source mutations hard-fail; 185 tests |
| P1 split authority | fixed | Một resolver cho validator/loader; patient001 ED/ES cross-split fail; manifest không đổi |
| P1 entropy heuristic | fixed | Logits/probability APIs rõ ràng; shift/batch invariance; low-precision stability |
| P1 CI installation/collection | deferred, waived | Local chạy toàn tests/; không sửa workflow, không có remote CI PASS |
| P1/P2 geometry | partially fixed | Resize spacing/original metadata và unsafe-grid guards; native orientation/inverse/raw verification DEFERRED |
| Frozen-bank schema/path | fixed trong phạm vi readiness | 71 tests, real inference trace + evaluator attachment + semantic validation; chưa sinh bank thật |

Các lo ngại trên được tái xác nhận bằng code/probe trước sửa, không mặc định review cũ đúng.
Không có cơ sở kết luận proposer headroom, auditor superiority hoặc Dice gain từ các fixture này.

### 3. Source-of-truth invariants — Gate A PASS

Giữ supervised BG/RV/MYO/LV, 2.5D center slice, ConvNeXt-Tiny/shared encoder/FPN/A0,
shared recurrent expert, dynamic window, audit stop-gradient và loss objective.
Core inference state machine không đổi: strict `delta_q > tau`; reject giữ state rồi HALT;
T_max chỉ hard cap. Không đổi cấu hình/split/docs.md, không triển khai Proposal 2/3.
Entropy bug fix có thể đổi output của cùng weights: **không claim prediction parity**;
entropy contract version mới được đưa vào lineage và cần calibration mới.

### 4. Final metric/calibration contract — Gate B PASS

- Training target: `audit_target_legacy_one_v1`, foreground {1,2,3}, both-empty = 1;
  giữ nguyên giá trị target/objective lịch sử.
- Evaluation: `foreground_dice_exclude_v1`, foreground {1,2,3}, both-empty excluded;
  undefined được giữ explicit, không biến thành neutral hoặc bỏ candidate hallucination.
- Metadata chứa metric space, class set, empty policy, neutral margin, version.
  Cache schema 1 lưu Q_previous/Q_candidate, validity và sufficient statistics; không cộng
  initial exclude với delta legacy. Replay chọn state/stats rồi tính trong cùng contract.
- Slice proxy, volume_resized và volume_native không đồng nghĩa. Case-grouped replay là
  case-macro volume, **không phải patient-balanced ED/ES**. Native physical metrics chưa chứng minh.
- Calibration artifact schema 2; header phải khớp nested lineage trước coercion/consumption.
  Audit CLI chỉ chấp nhận exclude contract nó thực sự tính, không relabel bằng metadata.

### 5. Checkpoint lineage — Gate C PASS

Lineage schema 1 gồm checkpoint SHA256, loaded/live state digests, resolved model identity,
producer git SHA (legacy unknown không được backfill), preprocessing signature, cohort/split
membership, class mapping, t_max, metric contract/space/margin, entropy version 2.0.0 và source
content signature v2. Signature v2 bao phủ src/self_audit, scripts và configs, kể cả active data
code và shell/PowerShell wrappers; lỗi đọc required source fail closed, không ghi partial-known.
Reports/tests không làm đổi source identity. Membership signature không phải pixel-content hash.

Calibration phải dùng complete cohort đã khai báo; artifact stale source không được restamp.
Consumer xây expectation từ runtime thật, không từ artifact. Mismatch hard-fail ngay cả khi
CLI override tau. Cohort độc lập phải được authorize explicit và patient-disjoint; same-cohort
diagnostics không được gọi là independent test. 4 public CLI subprocess fixtures kiểm tra
positive chain, replaced checkpoint, override bypass và authorized disjoint cohort.

### 6. Effective split resolution — Gate D PASS

Explicit manifest là authority; validator và DataLoader dùng cùng effective records.
Ambiguous precedence, duplicate cases, missing/unexpected entries và patient cross-split
(bao gồm ED/ES) bị từ chối. Không sửa tracked manifest. Chưa audit duplicate pixel content
trên raw datasets; không suy software fixture thành provenance sạch của toàn nghiên cứu.

### 7. Test matrix / Gate E và local Gate F

| Root independent check | Kết quả |
|---|---|
| Baseline invariants focused | 67 PASS |
| Checkpoint/source producer focused | 55 PASS |
| Calibration final focused | 185 PASS |
| Bank final focused | 71 PASS |
| Header missing/null/type independent mutations + positive control | 17/17 invalid cases rejected; valid binding accepted |
| Actual CLI legacy-contract preflight | Rejected trước missing config/checkpoint access |
| **Final integrated tests/** | **507 PASS, 1 warning, 108.14s** |
| In-memory compile src/ + scripts/ | **51 PASS** |
| Git diff whitespace check | PASS |
| GitHub Actions | WAIVED / NOT VERIFIED; không có link run |

Exact root full-suite command:
```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests
```
Root wrapper dùng `compile(path.read_bytes(), str(path), 'exec')` cho 51 Python source files;
SHA256 fingerprint của 77 source/test/config/split files trước và sau full suite bằng nhau:
`93cbec21573f9a2e024d97e5b5d48a39531654396618a2fd8b43cfe48325c028`.
Actual output: `507 passed, 1 warning in 108.14s (0:01:48)`, `UNCHANGED True`,
`SOURCE_COMPILE 51 PASS`, exit 0. Warning có trước ở test_self_audit_core.py:32 chuyển
requires-grad tensor sang scalar. Các focused counts không cộng vào full count.
Worker reports là receipt, không thay independent root evidence ở bảng này.

### 8. Remaining blockers trước Frozen Transition Bank thật

Cần trained checkpoint có lineage phù hợp, dữ liệu/cohort hợp lệ và calibration mới;
chưa tạo bank thật. Native HD95/ASSD vẫn blocked khi thiếu geometry chứng minh được.
Bank schema 1 hiện kiểm tra transition IDs, source, state identifiers, scores/stats, local
summary và protocol identities; cần thiết kế state/feature sidecars cho matched B3/B4 training.
GT-firewall kiểm tra generation/evaluator API với positive oracle control, **không bao phủ
toàn preprocessing/discovery provenance**; foreground-only selection cần protocol audit riêng.
Không có claim “clinical certification” hoặc “không còn bất kỳ lỗi nào” từ worker được Astra chấp nhận.

### 9. Next-phase handoff — PLAN ONLY

Xem `next_phase_plan.md`: frozen same-candidate B0–B6 trước, closed-loop sau. Không thực hiện
Proposal 2/3; giữ baseline mạnh, không cố làm yếu A0/pretraining để tạo gain.

### 10. Research / publication statements

**No full retraining was performed. No Dice improvement is claimed. Novelty has not yet
been validated.** Không gộp M&Ms vào development ACDC. Software correctness không chứng minh
novelty, headroom, external-domain performance hay tiết kiệm công annotation.
Astra approve logical commits và normal push main sau gate local này; không force-push.

### Publication ledger

HEAD after code: `d543986cf1e2a9aacff03a666421e1097c76b475`.

| Commit | Nội dung |
|---|---|
| `05d2b9fa606eca22a1aaaee3de341cf8c1211a7b` | Typed entropy |
| `a6ec2f4e3fe9fa9e4a82475507df1b507e93c846` | Authoritative splits / safe geometry |
| `92e2a062d7abd9d8291c24d1a17dfbd8a059ff89` | Metric/cache/calibration/provenance và regression tests |
| `d543986cf1e2a9aacff03a666421e1097c76b475` | Bank diagnostics schema / exporter / tests |

Measurement/provenance giữ một atomic logical commit vì contracts, producer và consumer
phụ thuộc chặt, không tách thành các commit giả vờ independently runnable. Chỉ integrated tree
được tuyên bố full-suite PASS; các commit trung gian không chạy full suite độc lập.
Report commit là descendant kế tiếp; SHA của chính report commit tra bằng git log (không tự
nhúng SHA gây self-reference). Normal push chỉ sau approval 12:46 UTC; kết quả remote được
báo trong final handoff, không suy local commit thành đã publish.

---

## Historical execution record (không thay verdict cuối ở trên)

## Current verdict — 2026-09-08 12:17 UTC

**Update 12:35 UTC:** root full `tests/` run: **471 passed, 1 warning in 90.83s**.
Command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests`.
Warning remains the existing requires-grad scalar conversion in `test_self_audit_core.py:32`.
Producer and bank gates now PASS (55 and 71 root focused tests respectively). Final
calibration null/type/preflight closure task `task_66376ca66853` / `ctx_1fb00b5e568e`
is active; this 471-test snapshot is not the final release approval. Scope/report updates
below retain earlier review history. No commit/push yet.

**PARTIAL.** Baseline Gate A, metric Gate B, split Gate D, entropy Gate E and minimum
geometry safety have passed their bounded local checks. Exact provenance Gate C and
bank validator closure remain REVISE; final integrated full suite is pending.
Remote CI is explicitly waived as a blocker by the user, not claimed successful.

Latest root checks: producer/runner **43 PASS**, invariants **67 PASS**, bank **54 PASS**;
calibration/metric/legacy/public-CLI combined **177 PASS / 3 FAIL**. The latter failures
are two legacy-fixture migrations plus an error-message assertion, assigned for correction
without weakening fail-closed behavior. Separately reproduced P0 header-versus-lineage
bypass is also assigned and remains a release blocker. All four actual public-CLI
subprocess integration tests passed in that combined run; these use tiny synthetic
volumes/checkpoints and are not real-trained-model validation.

Claude session quota exhausted; existing edits preserved. Three AGY workers now own
non-overlapping source-signature, calibration, bank closure tasks through Orca. See
`parallel_execution_update.md` for current dispatches and lifecycle evidence.
`git ls-remote origin refs/heads/main` rechecked 12:17:24 UTC still returns
`1305692914c8f50f11135d00d9035b4059f297a5`, matching local main HEAD.
No commits/push, no full retraining, no Dice improvement claim, novelty not validated.

The dated sections below are historical checkpoints of this implementation review;
their earlier counts and pending statuses do not supersede this current verdict.

## Latest execution update — parallel Claude workers

User requested multiple workers/gates on 2026-09-08. Three Claude workers now implement
separate ownership sets concurrently; see `parallel_execution_update.md`. Producer receipt
`msg_17a74fad82fe` was accepted operationally (earlier malformed receipt `msg_dc9b4a706b64`
was rejected for missing taskId). Independent full tests on that handoff:
**222 passed, 1 warning in 11.45s** using the standard full-suite command below.
`git diff --check` clean. This is an interim combined snapshot, not final integration PASS.

Producer gate remains **REVISE**: cache CLI lineage still omitted metric-space/margin,
contract definition and supplied margin could disagree, and observed/truncated cohort
claims needed tightening. Same Claude terminal reused for task `task_65e1fdd77649`, dispatch
`ctx_1729fcee7642`, including bounded runner integration with calibration worker.
W3.2 dispatch `ctx_b0cee3308413` and W4 dispatch `ctx_8bb7caf52c29` input accepted.
Four explicit Orca review gates are pending; no commit/push or research-result claim.

Status: PARTIAL — Wave 1 metric/cache, W2.1 split và W2.2 entropy đã được Astra
approve trong phạm vi fixture local; W2.3 minimum geometry safety đã PASS, native-mm
DEFERRED. W3.1 provenance đang được Claude Code tiếp tục theo chỉ thị mới của user và
frozen-bank readiness chưa hoàn tất. Chi tiết theo thời điểm nằm bên dưới.

HEAD before: `1305692914c8f50f11135d00d9035b4059f297a5` trên main.
HEAD after: chưa commit. Không push, không full retraining, không claim Dice improvement;
novelty chưa được validated. Các thay đổi đang dở không phải release-ready.

## Authority update

Sau execution_plan.md, người dùng chỉ thị “ko quan tâm ci, làm luôn”. Remote CI không còn
là blocking release gate; vẫn phải báo NOT VERIFIED nếu chưa có run đúng commit.
Local full tests, source compile và scientific correctness gates không được miễn.

## Orchestration evidence

- Run `run_688c52658c21`; Wave 1 task `task_ff3aa716046b`.
- AGY được launch bằng Orca `--agent antigravity`, không dùng worker ngoài Orca.
- Dispatch `ctx_aa8717f3c53f` được runtime xác nhận input accepted và có heartbeat.
- Orca restart làm terminal incarnation thay đổi, dispatch bị revoked/terminal_missing.
- Resume attempt `ctx_c162b6898fc7` báo agent_prompt_stalled dù terminal AGY thực sự
  tiếp tục đọc/sửa source. Không coi attempt này là dispatch PASS.
- AGY tiếp tục tại terminal được runtime list lại sau recovery; không launch editor thứ hai
  khi worker cũ đang sửa. Cần fresh supervised revalidation receipt trước khi approve Wave 1.
- Receipt recovery check từng trả runtime_unavailable rồi request_mismatch sau restart;
  không suy các lỗi runtime này thành lỗi model hoặc test failure.

## Findings / gates hiện tại

| Mục | Trạng thái |
|---|---|
| Metric/cache correctness | PASS fixture local (Wave 1); không phải real-checkpoint validation |
| Exact checkpoint/calibration binding | Claude Code đang tiếp tục W3.1; Gate C chưa PASS |
| Split authority | Gate D PASS: resolver/validator/loader parity và patient leakage fixtures |
| Typed entropy | Gate E PASS: typed APIs, invariance, fp16 stability, invalid-input guards |
| Geometry safety | Minimum safety PASS; strict native-mm blocked/DEFERRED, raw correctness chưa kiểm chứng |
| Frozen-bank schema/path | Chưa thực hiện |
| CI workflow repair | Chưa thực hiện; remote CI waived as blocker |
| Gate A invariants | Preflight verified; integrated patch verification pending |
| Gates B–E | B/D/E PASS phạm vi đã review; C pending |
| Gate F local tests/compile | Snapshot W2.3: 192 tests PASS, compile 47 PASS; final integrated pending; remote NOT VERIFIED |

Preflight full suite: 103 passed, 1 warning (không chứng minh patch hiện tại PASS).
Chi tiết baseline probes/CI cũ nằm trong execution_plan.md. Final contracts, lineage schema,
effective resolver, commit SHAs và next-phase handoff chỉ được chốt sau actual diff review.

## Wave 1 independent review checkpoints (patch còn đang thay đổi)

- Recovery task `task_0cbe9cbfc7c7`, dispatch `ctx_0379ee16be78` đã được Orca xác nhận
  ready/input_accepted trên đúng AGY terminal sau restart.
- Focused tests lần đầu: `1 failed, 6 passed in 0.70s`; fixture volume truyền kwargs không
  thuộc API. Đã trả worker sửa fixture đúng public path, không thêm ignored kwargs.
- Full suite sau đó: `110 passed, 1 warning in 3.99s` bằng lệnh
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests`.
  Đây là interim test run, chưa phải approval cuối wave.
- Draft review phát hiện pooling toàn cohort thành fake volume Dice; đã yêu cầu group
  theo case trước aggregation, không gộp nhiều bệnh nhân thành một volume.
- Probe strict sweep thiếu contract đã bị chặn. Tuy nhiên metadata-only cache với
  `initial=.5, delta=.133333333, metric_contract=foreground_dice_exclude_v1` vẫn replay
  `.633333333` khi chưa yêu cầu state scores. Yêu cầu worker chặn cả đường này.
- Probe forged contract dict tên exclude/policy legacy_one được chấp nhận ở bản nháp;
  đã yêu cầu full required metadata và validate name/version semantics.
- Khi thêm Q_previous=.5/Q_candidate=.4 với delta dương, bản nháp mới đã ném
  ContractMismatchError. Chỉ xác nhận kiểm tra inconsistency đó, không suy toàn cache sạch.
- CLI calibration cần subprocess integration: selected row chứa ndarray có thể gây lỗi
  `json.dumps(selected)`; metric-space flag phải được so với cache, không chỉ stamp metadata.
- Temp probe `/tmp/self-audit-astra-MhQ1nC/probe.py` không còn sau restart; rerun path đó
  trả file-not-found, KHÔNG phải test application failure. Durable source còn trong evidence.md.

Các review request trên được gửi qua Orca; chưa kết luận đã sửa trước khi kiểm diff/tests cuối.

### Gate review sau worker_done — REVISE

Worker_done `msg_6f891b9d872f` (task `task_0cbe9cbfc7c7`, dispatch `ctx_0379ee16be78`)
đã nhận và acknowledge. Astra chạy full suite: **115 passed, 1 warning in 5.42s**.
Tuy vậy Gate B chưa PASS. Probe độc lập trên cùng cache base
`initial=.5, delta_q=1, actual_delta=.133333333, contract=foreground_dice_exclude_v1`:

| Biến thể | Actual behavior trước revision | Expected |
|---|---|---|
| `stats_candidate=[]` | ACCEPTED; final=.633333333 | Reject incomplete statistics |
| `q_candidate=.4`, thiếu previous | ACCEPTED; harmful_total=0 | Reject missing previous |
| `q_previous` shape [1,2], candidate [1,1] | ACCEPTED; harmful_total=0 | Reject shape mismatch |
| prev=.5, cand=.4, actual_delta=NaN | ACCEPTED; harmful_total=0 | Reject inconsistent validity mask |

Quyết định **REVISE**, không mở Wave 2. Đã reuse đúng AGY cho task `task_e06e6582ab7f`,
dispatch `ctx_5a9fc41bd585`, input_accepted. Scope: centralized strict cache validation,
complete schema/contracts, state-score/shape/validity/continuity checks và negative regressions.
Không cho phép chỉ có một statistics key để bypass strict mode. Đây là review correction,
không thay baseline hoặc training objective.

Revision 1 worker_done `msg_f253c0eff989`: Astra rerun **126 passed, 1 warning in 5.18s**,
nhưng vẫn REVISE: `validate_and_normalize_transition_cache({'_normalized': True})` được
accept và cache có `cache_schema_version=999` cũng được accept. Hai bypass này đã được
gửi khi worker còn làm nhưng chưa được sửa trước bàn giao. Revision 2 chỉ sửa hai blocker:
task `task_ffae34f6055e`, dispatch `ctx_b6adba5f4cb2` input_accepted; delivery trước đã ack.
Worker vẫn là cùng AGY, không có editor song song. Wave 2 chưa mở.

### Recovery revision 2 — 2026-09-08

AGY provider của dispatch `ctx_b6adba5f4cb2` kết thúc với lỗi
`bdbfebd2-d660-4478-a361-61b2ad931d12-821`, trở về prompt, không có worker_done.
Orca worker-stop trả `stop_unknown` vì terminal là external/restored; worker-abandon
đã đóng lifecycle dispatch, không giết tài nguyên external. Edits được giữ nguyên.
Fresh AGY terminal `term_4991158f-02ba-4966-80ad-dfb1b2b608f5` gặp startup race
ở `ctx_6ee987564ec9`; retry đúng terminal thành công ở `ctx_2fb49876baaf`,
task `task_ffae34f6055e`, heartbeat 2026-09-08T08:12:19Z. Chỉ một worker đang sửa.
Đây là recovery vận hành, không phải bằng chứng application PASS. Wave 1 còn REVISE.

Independent interim probes trên revision 2 đang làm: hai payload `_normalized=True`
và `cache_schema_version=999` hiện đều bị `ContractMismatchError`.
Focused test run giữa lúc cập nhật fixtures: **16 failed, 6 passed in 1.72s**;
nhiều failures vì fixture cũ chưa có schema version. Đây không phải final gate run.
Remote main recheck vẫn `1305692914c8f50f11135d00d9035b4059f297a5`.

Finding bổ sung cần task riêng trong W1: cache hai hàng với Dice 1 và 0,
TP/FP/FN foreground lần lượt (1,0,0) và (0,9,9), scores/deltas nhất quán:
không `case_ids` trả `volume_macro_dice=0.1, volume_case_count=1`, trong khi
`case_ids=['case1','case2']` trả `0.5, count=2`. Không có identity không đủ
bằng chứng đây là một volume. Phải fail closed hoặc không phát hành volume metric;
không silently assume single volume. Chưa sửa, đã gửi worker note `msg_ea62b0950467`.

### Revision 2 bounded gate — PASS (not overall Wave 1 PASS)

Worker_done `msg_91ee03fbef27`, dispatch `ctx_2fb49876baaf`, report
`wave1_revision2_worker.md`; worker reports 131 passed in 5.15s.
Astra independently obtained **131 passed, 1 warning in 5.79s** using the full-suite
command above. Syntax-only compile of all `src/**/*.py` and `scripts/**/*.py`:
**47 files PASS**, no imports executed. Exact two attack payloads rejected;
18 combinations (six malformed versions × validate/evaluate/sweep) rejected;
mutating delta after normalization and setting `_normalized=True` also rejected.
Read public boundary/private core and CLI diff: serialized trust marker removed,
collector emits schema version 1, CLI uses strict validation. PASS for these two
specific blockers only. No diff in models/, losses/, configs/, splits/, docs.md;
finetune_joint.py changes confined to imports and measurement collector.

Delivery also contained two stale heartbeats from abandoned `ctx_b6adba5f4cb2`;
they are historical, not evidence of current concurrent work. Worker claim
"workspace clean" means task ready for review, NOT git-clean: tracked edits remain.
New bounded W1.3 task `task_2a825730a0e0` follows `wave1_volume_task.md`.
Overall Proposal 1 remains PARTIAL; no commits/push or full retraining.

### W1.3 / Wave 1 measurement gate — PASS, synthetic scope

Worker_done `msg_08e2a40d2927`, task `task_2a825730a0e0`, dispatch
`ctx_7b3851306882`, report `wave1_volume_worker.md`. Astra final full-suite rerun:
**136 passed, 1 warning in 5.70s**; source syntax compile **47 files PASS**;
`git diff --check` clean. Worker compile count 61 includes tests; root 47 covers
src/ and scripts/, so counts describe different scopes, not conflicting outcomes.

Independent two-case fixture: no IDs -> volume metrics unavailable, no numeric pooled
score; IDs -> volume Dice .5/count2. NaN/Inf/negative/fractional counts rejected;
volume_resized/native contracts rejected by slice cache boundary. Read actual Q/stat
consistency/bundle checks, separate slice/derived-volume metadata, collector guards,
normalized case IDs and NumPy-safe selection. Existing regression paths preserve blank
to hallucination and t_max=0. No model/loss/objective/state-machine edits in this wave.

PASS applies to measurement/cache correctness covered by these code paths and synthetic
tests, not real checkpoint evaluation, native geometry or complete Proposal 1.
Worker report calls case-macro average "patient-level" in places: this is **case/volume
macro**, not patient-balanced ED/ES aggregation. Patient protocol remains next-phase work.

W2.1 opened with task `task_3945e271726f`, dispatch `ctx_3b7cbf0792c7`, same AGY terminal,
input_accepted. Scope follows `wave2_split_task.md`; no entropy/geometry/provenance work
authorized within that task. W1 delivery `delivery_6ae1079a1e6f` acknowledged after reuse.

### W2.1 interim review (not yet PASS)

Resolver-level ED/ES fixture now raises `Patient leakage: 'patient001' appears in both
'train' and 'val'` for two synthetic VolumeRecords; no real files/data needed for this
resolver-only check. Tracked manifest parse regressed in draft: `n_patients` was treated
as an unknown split key. Also identified first-key-wins ambiguity and same-stem NPY/NPZ
collision hiding. Pending corrections are recorded in `wave2_split_review_notes.md`.

Operational correction: worker tried searching provider/runtime directories for pending
Orca messages. Coordinator instructed it to stop that out-of-repo investigation and read
the complete notes file in this repo instead (public terminal send accepted). No account
or credential access is needed/authorized for this remediation. Later bounded read shows
worker back in data/acdc.py. This event is not an application test or completion receipt.

### W2.1 split gate — PASS, controlled fixtures

Worker_done `msg_eb5d935ea39d` / `ctx_3b7cbf0792c7`, report `wave2_split_worker.md`.
Astra independent focused suite: **25 passed in 0.53s**. Final full suite:
**161 passed, 1 warning in 5.43s**. Source compile: **47 files PASS**;
diff-check clean. Tracked manifest reads **160 train / 40 val cases**, no manifest edits.
Actual diff resolves summary metadata regression, conflicting representations and
same-stem file collisions, and returns JSON-safe effective record descriptors.

Additional independent integration probe reused the existing synthetic manifest fixture,
then called actual `build_patient_dataset` + `build_data_loader` for every configured split:
train 2 cases, val 1, test 1. Validator descriptors (case ID/image path/mask path) matched
dataset records, and all batch case IDs matched split membership. TemporaryDirectory
removed only its own fixture afterward. This is not real ACDC-data validation.
Task W2.2 will persist this stronger all-splits assertion in the existing regression test.

Effective split authority: manifest -> validated tags if no manifest -> deterministic
untagged patient fallback; ambiguous mixed sources fail. Signature is sorted membership
identity only, NOT a data-content hash; content-hash duplicate detection remains deferred.
No pixel recipe or objective changes in W2.1. Gate D PASS in tested scope.

W2.2 dispatched as `task_36bce70a17f9` / `ctx_515c0d62b202`, input_accepted,
same AGY worker; scope `wave2_entropy_task.md` plus the test-only parity strengthening.

Orca later delivered a duplicate W2.1 worker_done (`msg_03839edeb1d8`) with revoked old
dispatch capability, claiming 26/162 tests after queued follow-up guidance. It was rejected
by runtime and is NOT a new accepted completion. Current W2.2 heartbeat is valid for
`ctx_515c0d62b202`; keep that worker active. Any post-handoff changes are subject to the
next actual diff/full-suite review; earlier 161-pass receipt remains the earlier snapshot.

### W2.2 entropy — REVISE (numerical safety)

Accepted worker receipt `msg_4e0d0ef908bc` / `ctx_515c0d62b202` claimed
13 entropy tests and 175 full tests; these are worker results, not final approval.
Astra independently reproduced NaN entropy and nonfinite gradients for valid fp16
one-hot probabilities, and NaN entropy for finite fp16 logits
`[65504, -65504, 0, 0]`. Original-dtype epsilon underflow / log-softmax overflow
requires correction; typed API existence alone does not pass Gate E.

Narrow revision dispatched as `task_487bf40fd0e5` / `ctx_965b2b5684b0` to the same
AGY terminal, input accepted. Scope: stable entropy precision and focused regressions
only; see `wave2_entropy_review_notes.md`. Prior delivery acknowledged after reuse.

Revision receipt `msg_cc5a247cac2e`: fp16 corrections verified; Astra full suite
**179 passed, 1 warning in 6.04s**. Still **REVISE**, because a new guard workaround
maps NaN/Inf logits to zero entropy with nonfinite gradients. Reject invalid inputs
explicitly rather than hiding them. Final narrow task `task_8c152dfedfeb` /
`ctx_8ee228c3f2ce` accepted on same terminal; receipt acknowledged after reuse.

Read-only remote recheck during this continuation: local HEAD and origin/main remain
`1305692914c8f50f11135d00d9035b4059f297a5`; no commit/push. Newly prepared
`wave4_bank_task.md` is a task specification only, not implemented bank readiness.

### W2.2 entropy Gate E — PASS, controlled numerical/integration tests

Final worker receipt `msg_459ed1784ab6`, report `wave2_entropy_final_worker.md`,
task `task_8c152dfedfeb` / dispatch `ctx_8ee228c3f2ce`. Actual diff read: finite-input
guards precede one-channel shortcuts; half precision uses fp32 computation, output dtype
and autograd preserved; expert explicitly consumes logits entropy. Literal all-callers
search found no remaining range-dispatch caller. No learned parameters/state keys added.

Astra full suite: **181 passed, 1 warning in 5.94s**. Additional independent one-hot and
extreme-logit probes passed output/gradient finiteness for fp16/bfloat16/fp32/fp64;
NaN/+Inf/-Inf rejected. Source compile **47 PASS** and diff-check clean. These are CPU
synthetic checks, not full mixed-precision training or real-checkpoint validation. Earlier
worker phrasing "fully protected against NaNs" exceeds this bounded evidence.

Entropy semantics version 2.0.0 changes erroneous conditioning of historical checkpoints;
same tensor keys do not guarantee identical old predictions. Recalibration/lineage must
encode this version in W3. Baseline architecture, loss objectives and reject/HALT unchanged.

W2.3 dispatched `task_14bdb20de9d2` / `ctx_bec9932454c3`, same AGY terminal,
input_accepted. Scope `wave2_geometry_task.md`, including skipped-array stale-provenance
regression. W2.2 delivery acknowledged after reuse. W3/W4 not yet implemented.

### W2.3 initial geometry receipt — REVISE

Worker receipt `msg_935c1b514530` claims 188 tests; Astra independently reproduced
**188 passed, 1 warning in 5.93s**, but `git diff --check` reports five added trailing
whitespace lines. More importantly, actual probe using a 2x4x4 label array with foreground:
`evaluate_volume_native(p,p,spacing=(5,2,2),affine=[[1]],strict_physical=True)` accepts
malformed single-affine input and emits `hd95_mm=0.0`. annotation_metrics also accepts
physical spacing `(0,2,2)`, `(NaN,2,2)`, `(-5,2,2)` with hd95=0. This is not fail-closed.

Additional code/test findings: skip path can overwrite partial existing image/mask pairs;
subset preprocessing can drop historical metadata; second-resize test uses manual spacing
arithmetic rather than actual preprocessing -> loader chain. Requirements consolidated
in `wave2_geometry_review_notes.md` (five points); full-suite success does not waive them.

Revision `task_ef15692f46cb` / `ctx_bf85f8a0daf5` accepted on same AGY worker. Prior
receipt acknowledged after reuse; current worker heartbeat accepted. Native physical
verification may remain explicitly DEFERRED, but unsupported strict evaluation must fail
and unverified native-mm output must not masquerade as a verified measurement.

W2.3 revision interim: Astra **192 tests passed, 1 warning in 5.93s**, but before
completion reproduced incompatible retained/new-grid merge in a TemporaryDirectory.
Existing arrays 8x8x2 + metadata target_size [8,8], only NEW patient input requested
size4; actual CLI main with synthetic preprocess callback accepted and wrote global
target_size [4,4] while retained array stayed 8x8x2. This is a CLI merge-path probe,
not raw-NIfTI integration. Worker subsequently read updated five-point notes and is
still revising; no new PASS receipt yet. Diff-check also still had added whitespace.

Latest W2.3 revision local review: focused **11 passed in 0.93s**, full **192 passed,
1 warning in 6.14s**, compile47 PASS, diff-check clean. Actual synthetic-NIfTI -> saved
metadata -> ACDCDataset second-resize test replaces manual-arithmetic fixture. Independent
native malformed guard / suppressed unverified-mm / invalid physical spacing probes PASS;
mixed-grid CLI now rejects before writing new arrays and preserves metadata bytes.

Before final worker receipt, Orca wait lost connection (`runtime_unavailable`, request
`d8e50d3d-7329-4bb8-b00a-d2cc59017ad8`); exact retry returned replayed cancelled /
connectionLost. Same runtime returned. Worker tail explicitly showed execution error
`75ee7f37-446d-4bd1-a674-ae9e3401d4e0-1247`, so coordinator sent bounded recovery input
to the same terminal/task/dispatch via public terminal API. Input accepted, no duplicate
worker spawned. This is runtime recovery, not model/test failure. Receipt still pending.

### W2.3 final review — minimum safety PASS; native physical correctness DEFERRED

Provider recovery first failed with network error suffix1254, second bounded retry resumed
same task. Final accepted receipt `msg_333395a1f752` / `ctx_bf85f8a0daf5` confirms all
five notes, report updated. Astra evidence above is independent:192 full tests6.14s,
11 focused0.93s, compile47, diff-check, powered previously-failing probes. Native-mm is
suppressed in unverified mode and strict physical mode raises DEFERRED; this is a safety
guard, not successful inverse-geometry validation. Native Dice remains caller-supplied
reference/grid analysis explicitly marked reconstruction DEFERRED, not a verified clinical
native result. Current affine checks are conservative, not a general orientation solver.

No pixel/interpolation/normalization changes were needed. New preprocessing records preserve
real affine/orientation and resize descriptors, legacy records are not retroactively assigned
new geometry. Changed-grid reprocessing should use a fresh output directory under this safe
guard; retained mixed-grid mutation is rejected. Raw dataset/native validation remains debt.

W3.1 dispatched `task_15d1ae019ddc` / `ctx_ba4c6e694b84`, same AGY terminal, accepted.
Scope follows `wave3_checkpoint_task.md`; W3.2 strict consumers and W4 exporter not yet open.
Prior geometry delivery acknowledged after reuse. No commits/push or long retraining.

## Current handoff — PARTIAL, worker quota blocked (2026-09-08 11:11 UTC)

W3.1 worker terminal explicitly reports **Individual quota reached**, estimated reset
34m26s, error `75ee7f37-446d-4bd1-a674-ae9e3401d4e0-1298`. This is provider availability,
not application correctness. No W3.1 producer implementation or accepted completion exists.
Current runner remains unchanged from HEAD; _utils/cache script edits are preceding-wave
changes, not checkpoint binding. No account/provider settings or model were changed.

Coordinator fenced dispatch `ctx_ba4c6e694b84` using public `worker-abandon`; response:
state abandoned, processAction none, resources retained (no process/data deletion).
Task `task_15d1ae019ddc` is now **blocked**, independently checked via task-list.
Retained terminal `term_4991158f-02ba-4966-80ad-dfb1b2b608f5` must be inspected before reuse;
do not send stale worker_done capability. No pending check/wait process remains.

### Resume instructions

After provider quota returns, inspect current HEAD/diff, current Orca run and terminal.
Read this report plus wave3_checkpoint_task.md and wave3_checkpoint_review_notes.md.
Set the existing blocked task ready via `orca orchestration task-update`, then use
worker-start with that task and retained terminal (if healthy), `--retry-of ctx_ba4c6e694b84`;
accept freshly injected dispatch IDs. If terminal genuinely unavailable use normal Orca
worker recovery, not another non-Orca implementer. Continue W3.1 -> independently review ->
W3.2 strict consumer tests -> W4 readiness. No root production replacement implementation.

### Snapshot deliverable / explicit limitations

- HEAD before/after still `1305692914c8f50f11135d00d9035b4059f297a5`, branch main; working
  patch uncommitted. No push attempted because final correctness gate is incomplete.
- Metric/cache mismatch fixed in tested strict paths: named legacy training contract
  `audit_target_legacy_one_v1` stays distinct from `foreground_dice_exclude_v1` evaluation;
  state Q_previous/Q_candidate and validated sufficient statistics avoid mixed arithmetic.
  Explicit cache schema1; derived volume_resized metrics require case identity. Calibration
  selection is still slice diagnostic, not patient-balanced independent volume evidence.
- Checkpoint lineage schema is **not final/not implemented**; hash/config/producer/
  preprocessing/split binding remains W3. No artifact can be called trustworthy from W1 alone.
- Effective split resolver and typed entropy are fixed in controlled fixtures. Entropy
  semantics2.0.0 requires explicit future lineage/recalibration of historical checkpoints.
- Geometry minimum safety fixed; native-mm verification/reconstruction **DEFERRED**.
  Patient-balanced reporting and content-duplicate audits remain subsequent protocol debt.
- Frozen-bank exporter/schema integration remains unimplemented; wave4_bank_task.md is
  specification only. No real frozen bank generated; no trained checkpoint/data evaluation.
- Local latest validated snapshot: **192 passed, 1 pre-existing warning in 6.14s**;
  source compile **47 PASS**, diff-check clean. See exact commands/evidence above.
- GitHub Actions: **NOT VERIFIED / waived as blocker by user**; no new CI result/link exists.
  Local success is not CI success. CI workflow repair was not performed in this continuation.
- Source-of-truth: supervised baseline/backbone/loss objectives/reject-HALT/audit stop-gradient
  were not redesigned. Final whole-phase integrated Gate A remains to be closed with W3/W4.
- **No full retraining was performed. No Dice improvement is claimed. Novelty has not yet
  been validated.** Proposal 2 and Proposal 3 remain unopened. Next-phase B0–B6 plan is not
  a measured result and should be finalized only after Proposal 1 correctness completion.

Preserved unrelated original reports/astra_review/ and self_audit_architecture.html.
Root-level wave1_revision2_worker.md is an AGY-attributed duplicate report; retained and
not staged/deleted. Root added only review/task Markdown; AGY made production/test edits.

## User-authorized Claude Code continuation

User: "gọi claude code nốt đi". Claude Code via Orca replaces AGY for remaining tasks;
Astra retains independent review and all Proposal 1 constraints. Latest local and remote
main rechecked: still1305692914c8f50f11135d00d9035b4059f297a5, existing dirty work preserved.
Task task_15d1ae019ddc restored to ready. Retry linking abandoned AGY dispatch was rejected
task_not_startable with no worker created; ordinary start of the ready task succeeded.
Accepted Claude dispatch **ctx_8d1708d5bcfb**, terminal
**term_174c7616-9fa5-46f0-924e-e0abb09eacd9**, same checkout. No model/effort override
requested; runtime launch reports agent claude. Prior AGY capability remains fenced.
Remaining waves remain W3.1 producer -> W3.2 strict consumers -> W4 bank readiness.
No final approval, commit/push, retraining, or Dice/novelty claim.
