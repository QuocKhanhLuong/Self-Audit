# Execution receipt — 2026-11-01

| Workstream | Actual status |
|---|---|
| DATA_PREP | BLOCKED_DRIVE_QUERY_QUOTA; 0/202 payloads downloaded |
| S0 | NOT_RUN; real-data CPU smoke and A100 gate remain closed |
| S1 / S2 | BLOCKED; exact historical B8 raw payload and verified cache missing |
| HISTORICAL_DIAGNOSTICS | BLOCKED; historical teacher/runtime prerequisites unchanged |

The user chose to wait for Drive quota and avoid further spend. The owned CPU
copy app was stopped after its BLOCKED receipt was committed. A subsequent
control-plane listing returned zero active containers. No A100 was allocated,
no patient training/evaluation ran, and no background retry was scheduled.

Actual CPU preparation attempts:

```bash
conda run --no-capture-output -n selfaudit-modal python -m modal run --detach \
  scripts/modal_fresh_s0.py --namespace /assets/fresh_s0/acdc_l8_s42_20261101_r1 --action prepare
conda run --no-capture-output -n selfaudit-modal python -m modal run --detach \
  scripts/modal_fresh_s0.py --namespace /assets/fresh_s0/acdc_l8_s42_20261101_r2 --action prepare
conda run --no-capture-output -n selfaudit-modal python -m modal run --detach \
  scripts/modal_fresh_s0.py --namespace /assets/fresh_s0/acdc_l8_s42_20261101_r3 --action prepare --observed-inventory
```

R1/R2 failed during scoped Drive listing. R3 used the already observed training
folder ID and a 202-file allowlist totaling 4,450,177,600 bytes; its copy and an
independent direct-folder query were blocked by `rateLimitExceeded` on quota
`Queries`. Raw stderr/config/token were withheld. File count inside the mounted
container remained zero. The unchanged historical receipt SHA is
`c4ffe94c921820f0bc3c67586f3ad64f5c6ec02fe2359146c1f2fa8c7b22029d`.

The exact TRAIN manifest was recovered from embedded checkpoint metadata and
validated against `e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4`.
That metadata recovery is not physical data preparation or teacher recovery.

Local verification in conda `selfaudit-modal`:

```bash
python scripts/check_repo.py                  # PASS
python -m pytest -q                          # 1083 passed, 2 warnings, 161.31 seconds
python -m modal run scripts/modal_fresh_s0.py --help  # PASS; no remote function call
git diff --check                            # PASS
```

`PYTEST.txt` preserves both warnings: an intentional duplicate-optimizer-group
negative test and PyTorch JIT deprecation. No tests were removed or skipped.
The 19 added tests include the full wide architecture/crop160/batch8 two-step
fixture smoke; all encoder/decoder/head gradient and weight-change checks pass.
Atomic checkpoint/export/reload and CPU resume equality to uninterrupted
execution pass. Associated fixture hashes/config are in the three `FIXTURE_*`
receipts. Those weights are synthetic software witnesses, never patient models.

The immutable copied `DATA_PREP.json` pins the staging source actually executed.
`STATUS.json` separately pins the final locally tested code/config closure;
later code fixes do not retroactively relabel the staging attempt.

The billing API snapshot is workspace month-to-date, not exact task attribution:
meteredCost `$0.05039856`, billedCost `$0.00000000`. The user reported $30 credit;
the implemented future S0 ceiling remains at most $15 including already spent
credit and reserves. No future paid calls are authorized by this receipt alone
when data/smoke/credit gates fail. See [the runbook](../../docs/MODAL_FRESH_S0.md)
for commands to continue in fresh namespace R4 once Drive access recovers.
