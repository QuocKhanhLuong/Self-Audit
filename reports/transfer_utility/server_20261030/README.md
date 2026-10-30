# Server verification — BLOCKED, patient pilot NOT_RUN

Tested source `c7a7a09565da549631d876dbbcb64a8931837d7c` (production source closure `e3d70d291dcbe190c24d6898b260698ef92e15027a54fe8c6f034374d968ff27`), branch
`QuocKhanhLuong/cptu-server-verification-20261030`. The original checkout and unrelated
jobs were preserved. AGY workers implemented the fixes through Orca orchestration;
the coordinator reviewed them and executed the server checks.

| Check | Actual result |
|---|---|
| Repository check | PASS |
| Final repository suite | 900 passed, 2 warnings; 301.79 s pytest time |
| Executable original-selector parity | 80/80 PASS |
| Four-arm CUDA smoke | PASS, manufactured data, four steps per arm |
| Export/reload without teacher, support or cache | PASS, manufactured data |
| CUDA sampler RNG, L/U schedule and selector decisions | PASS in the four-step replay witness |
| CUDA weights/optimizer bitwise replay | FAIL |
| Target-architecture 32-step profile | NOT_RUN; stopped at replay gate |
| Four-arm ACDC patient pilot and VAL comparison | NOT_RUN |

The maximum weight difference was `6.109476089477539e-7`. Differences already appeared
between independent fresh CUDA runs before the resume boundary. All actual random
generator states matched, including the probe bit generator; the only differences
inside the probe state container were measured sampling/cache time counters. Decisions
and L/U schedules matched. Deterministic algorithms were disabled in this runtime.
This does not establish a sampler-state restoration bug or isolate the numerical cause.
It also does not satisfy bitwise replay; no tolerance was relaxed to mark it PASS.

## Changes

- `tests/test_automatic_engine.py`: CLI subprocesses use the running interpreter.
- `tests/test_compact_probe_resume.py`: explicitly remove CUDA RNG from a negative
  checkpoint fixture and update its hashes, so the test is valid on CPU and GPU hosts.
- `self_audit/compact_training.py`: initialize CUDA before resetting allocator peak
  statistics; model construction/loading remains inside the measured allocation scope.
- `tests/test_compact_training.py`: cover fresh/resume startup ordering and the original
  uninitialized-allocator failure. No optimizer, loss, architecture or pilot config changed.

The first server attempt recorded 897 PASS/2 FAIL. The portable test fix produced
899 PASS, then fresh CUDA smoke exposed the allocator startup bug. The corrected
source passed CUDA smoke and the final 900-test suite. Failed receipts are retained;
`verification_v3/SERVER_VERIFICATION.json` intentionally remains FAIL at bitwise replay.

## Real assets

ACDC TRAIN L8/U72 is restored at
`/root/self-audit-assets/data/prepared/ACDC_L8_S42/train.json`, with exact hash
`e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4`. Labels remain patient007/010/011/024/026/041/077/099.

Canonical preparation with `--evaluation` produced an identical TRAIN manifest and
restored VAL18. All 72 image/reference/geometry hashes match the pinned historical VAL
manifest `cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50`. Compatibility symlinks restore its original paths;
no existing dataset was overwritten. Preparation opened no U masks and no TEST data.
VAL references were prepared and hashed only; no model was scored or selected with them.
The evaluator still freezes predictions before scoring.

The exact raw-student B8 payload remains missing:
`0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6`.
The pinned teacher probability cache is also missing. The available selected compact
export is a different artifact and is not substituted. The final real preflight exits
2 at the missing physical teacher directory; the pilot output directory was not created.

## Commands and receipts

Runtime: `/workspace/cptu-runtime-20261030/bin/python`.
Server checkout: `/workspace/Self-Audit-cptu-20261030`.
All exact executed argv are recorded in `PREFLIGHT_FINAL_COMMAND.json`,
`PREPARE_VAL_COMMAND.json`, `verification_v3/SERVER_VERIFICATION.json` and the smoke
`COMMANDS.json`. The full final suite was executed separately after the failed replay
check, as recorded by `run_final_suite.py` and `verification_v3/FINAL_SUITE.json`.

```bash
ssh vast-gpu 'cat /workspace/cptu-server-setup-20261030/verification_v3/FINAL_SUITE.json'
ssh vast-gpu 'cat /workspace/cptu-server-setup-20261030/verification_v3/CUDA_RESUME_DIAGNOSTIC.json'
ssh vast-gpu 'cat /workspace/cptu-server-setup-20261030/PREFLIGHT_FINAL.stdout'
```

The existing `docs/TRANSFER_UTILITY_RUNBOOK.md` remains the run contract. Restore the
exact teacher/cache and resolve the replay gate before changing the tested preflight
command to `--mode all`. The budget remains 2000 total steps, 1000 warmup, L/U batch8/8,
evaluation at 1000/1500/2000 and the same eight scales. No sweep, joint run, extra labels,
full training, teacher retraining or conditional fifth arm was started.

Server logs/checkpoints: `/workspace/cptu-server-setup-20261030/verification_v3/`.
They are manufactured verification artifacts, not trained patient models.
Full offline archive: `/Users/alvinluong/.local/share/openresearch/files/self-audit/server-verification-20261030/cptu-server-verification-20261030.tar.gz`
(SHA256 `26b454054ffe3c913704c290ba920504a91e3c48ddf20d12b9675e19019df4f6`). It preserves the smoke checkpoints and all failed/pass
receipts without patient image/mask arrays or the missing B8 payload.

**Decision: BLOCKED / no patient GO or NO_GO.** There are no VAL metrics, patient
learning curves, meaningful equal-time comparisons or B8 inference-speed comparisons
from this run. Neither Dice87 nor a research contribution is inferred from these checks.
