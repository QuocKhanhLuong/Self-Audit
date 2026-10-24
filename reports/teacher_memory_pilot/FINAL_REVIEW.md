# Final delivery consistency review

Date: 2026-10-24. Reviewer: `astra_novelty_review`, with an independent profile and
runbook consistency check by `dual_teacher_reference_check`.

**Verdict: ACCEPT for branch publication as an engineering pilot. No release
blocker remains in the reviewed report/runbook claims.** This final pass inspected
saved receipts, logs, commands and hashes; it did not rerun tests or review code.

## Claims checked against saved evidence

| Claim | Evidence checked | Result |
| --- | --- | --- |
| Local and server validation | `FINAL_LOCAL_VALIDATION.json`, `pytest-final.log`, `server/SERVER_VALIDATION.json`, `server/server-pytest.log` | Both logs say **148 passed, 1 warning**; pytest times are 10.78 s and 22.59 s. Command wall times are separately recorded. The warning is the existing `torch.jit.script` deprecation. Server execution is not presented as all tests using CUDA. |
| Exact matched B32 profile | `server/matched-b32-all-arms/PROFILE.json` | A0–A6, B=32, crop 256, 1 warmup + 2 measured transactions, 128 measured L+U processed exposures each. All reported throughput and allocated/reserved GiB values agree after rounding. Total counts include warmup: 3 L transactions, 3 U transactions, 6 steps per optimizer, 3 EMA updates. |
| Manufactured profile scope | Same profile protocol/data-mode fields | Real pinned DINO with manufactured inputs, labels and dense banks. A1 is a manufactured soft-target operator measurement. Profile A5 is not student-derived bank evidence. Offline teacher, I/O and checkpoint costs are excluded. |
| Real A0/A3 smoke | `server/REAL_TRAIN_SMOKE.json`, both arm `REPORT.json` files | Two steps each, B32, threshold 0, warmup 0; 2 L / 2 U / 2 EMA, no geometry updates or label reads outside L. Both have `synthetic=false` and `quality=NOT_EVALUATED`. Reported wall and engine times agree. |
| Real teacher bank | `server/bounded-real-bank/BUILD_RECEIPT.json`, `server/BANK_INSPECTION.json`, `server/bounded-real-bank-execution.json` | Three of 80 TRAIN patients, one timepoint each, incomplete coverage; pair counts **[8,8,0,8,0,8]**, 64 valid vectors, D32, no A1 cache. Wall 26.472 s. The 90 total timepoints concern the selected three patients. |
| Real A5 plumbing witness | `server/bounded-student-bank/BUILD_RECEIPT.json`, `server/BANK_INSPECTION.json`, `server/EXPORT_A5_EXECUTION.json` | Same three-timepoint manifest, frozen two-step A0, no external teacher arguments/environment; counts **[0,0,0,0,0,0]**, zero valid vectors. Wall 10.292 s. This is explicitly unsuitable as a useful or vector-budget-matched A3/A5 quality comparison. |
| Teacher-free exported inference | `server/TEACHER_FREE_DEPLOYMENT.json`, copied export manifest | Payload keys contain only `student`; embedded bank; blocked original teacher/cache/bank asset roots; no forbidden access attempts; max logit difference **0.0** on the reported TRAIN slice. Pinned DINO architecture/source and construction weights remain required. |
| Real teacher smoke | `server/teacher-smoke.json`, `server/teacher-smoke-execution.json` | Strict real checkpoint smoke uses manufactured oblique 32x32x3 input; hook/head parity deltas are 0.0. Timings and memory values agree with the delivery text. No full-patient throughput or quality inference is made. |
| Preservation | `PRESERVATION.json`, `server/SERVER_PRESERVATION.json` | Receipts support 236 unchanged pre-existing dirty/untracked entries, unchanged original checkout heads/status, clean older server checkouts and the retained watcher. |

The earlier `REVIEW_PROFILE.md` evaluated the implementation with a then-planned
B4 run and a tiny CPU orchestration probe. The final B32 numerical evidence comes
from the separately identified completed server receipt above; those scopes are
not interchangeable.

## Hash consistency

Recomputed current v2 source digest:
`7f5856216d59bc787633f8a703fa362cf83582427b830ab1ce09c15aeb55e8a9`.
It matches the server validation, profile, deployment and checkpoint records.

Recomputed copied manifest/file hashes:

| Artifact | SHA-256 |
| --- | --- |
| A0 final manifest | `83964dafec627a70bbc83ebcabd74aa60c8f45f3f3c00e1922106610a0d1592f` |
| A3 final manifest | `9e9178ff0b7b6f826be30c803ddc7313b91024fbb6576e372b094fd7113cadd0` |
| A3 exported manifest | `7c24aaa90d8708a8a8add67bca9881134e3da5ef5663afef8564b6a8b7e9b2c4` |
| Matched B32 profile JSON | `dd7fa03761a008a252fdbbd40072d3cb6c9b223004354f729795c33f232e43d7` |

The report's teacher config/weight/source/HF pins agree with the teacher-smoke
lineage. The DINO weight hash agrees with the profile and export manifests. The
teacher/A5 bank file hashes agree across build, inspection and consuming-run
receipts. Bank and checkpoint tensor payloads remain on the server: this final
pass checked their saved identities, not a new independent payload download.

## Wording closure and retained limits

The initial draft claimed an explicit CineMA import block during deployment, but
the saved deployment receipt only recorded asset-open blocking. The coordinator
removed the extra claim. The final text retains the receipt-backed asset-access
and exact-parity statements.

Full 80-patient/all-timepoint extraction, a real complete A1 cache/A1 training,
a useful full A5 bank, seven-arm patient training, long training, patient Dice or
boundary efficacy, seed statistics and pairing superiority remain **NOT RUN**.
Subject overlap remains **UNKNOWN** and novelty remains **UNCONFIRMED**.
The two measured profile transactions do not establish statistically meaningful
speed differences or an absolute optimal batch size. Long-run process RSS is not
measured. The empty bounded A5 bank is retained without a positive-result claim.

Runbook commands are labeled subsequent work and require new bound assets/output
paths. The concrete A3 repeat command is explicitly a two-step engineering
repeat. The guarded A1–A6 templates are not represented as ready full experiments.

## Final prose snapshot SHA-256

- `reports/teacher_memory_pilot/REPORT.md`: `491c85d516bb61a6caaf54477cf9b8317ee835065540593e200af09357c96fa7`
- `docs/TEACHER_MEMORY_PILOT.md`: `af5d3b2a2bc0b128da01aa6760a71a6afe28dada4d483d5392bb5f40a3d2ec1c`

## Publication whitespace exception: exact executed bytes retained

Independent staged checks found exactly these three cosmetic warnings:

```text
rtk proxy git diff --cached --check
self_audit/teacher_memory.py:409: new blank line at EOF.
tests/test_cinema_teacher.py:227: new blank line at EOF.
tests/test_teacher_memory_core.py:227: new blank line at EOF.
exit code: 2
```

A second check disables only the `blank-at-eof` diagnostic:

```text
rtk proxy git -c core.whitespace=-blank-at-eof diff --cached --check
no diagnostics; exit code: 0
```

Each listed file ends in exactly two newline bytes, and its staged bytes match
its working-tree bytes. There are no other staged whitespace findings under the
remaining default whitespace rules. The raw check is **not clean** and must not
be reported as passing.

Retaining these three EOF blank lines is accepted as a narrow publication
exception. The executed/reviewed source bytes are preserved, including the core
module digest `d491ec83f2b3e466eeca8e937bbaf3de09f4c9fcedbc2e30fd2c348d1257a5a8`
and aggregate source digest
`7f5856216d59bc787633f8a703fa362cf83582427b830ab1ce09c15aeb55e8a9`.
Changing the core module merely to remove its blank line would change the strict
source identity required by the saved checkpoints. No source edit or test rerun
was performed for this cosmetic exception.
