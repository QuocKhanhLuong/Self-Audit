# Q3D exploratory continuation and overnight monitoring

The user explicitly authorized continuing the full pipeline after its quality
gates failed, then requested a check every 5–10 minutes while asleep. This is
a post-observation protocol amendment, not a passing strict-design experiment.
The original plan, recipes, manifests, splits, geometry and update budgets remain
unchanged. Candidate A/A-3D sources remain unchanged.

## Observed stop

On 2026-11-12 at 16:22 UTC, `q3d-campaign` on `vast-gpu` had exited. K8 T
completed 8,000 updates. K8 Q completed 1,000 updates and failed its measured
gate: perplexity 20.87472152709961, dead-code fraction 0.6328125, and foreground
class-code coverage RV/MYO/LV = 3/0/5. The gate remains `NO_GO`, `passed=false`.
The selected teacher validation FG Dice is 0.8035429289034229 at update 2400.
These are real execution observations, not synthetic efficacy claims.

The failed Q checkpoint SHA is
`cf16cacac89ebbdb39732d9f80ff4af219b30753d446f3342bbe3dc2dc18bf94`.
The original plan SHA is
`be8cd9689584389e655007b6a6fec8f3b647dc6e61f51713f3dfe41d26bb9ec4`.
Recovery archives the original state and checkpoint before resuming Q at its
completed cursor. It must not repeat T or Q optimizer updates.

## Explicit continuation policy

Every invocation supplies the exact reason:

```text
--q3d-exploratory-quality-waiver "User-authorized full-pipeline exploratory continuation 2026-11-12"
```

The waiver covers tokenizer quality, U physiology support/ESS, and global KD
support. A positive mean retains actual relative normalized physiology weights;
zero mean uses a recorded uniform omega=1 fallback. Zero KD support contributes
graph-preserving zero loss. Failed measurements remain failed; checkpoints,
artifacts, exports and reports carry the exploratory policy. The same explicit
reason is required on resume/import. Data provenance, SHA, no-GT, shape and
finite-value validation are unchanged. Without this flag, the strict behavior
remains unchanged.

## Queue and monitoring

Server base: `/root/mount_data/runs/q3d-resenc-pixdim-20261112`.

Order: V1 K8 → V1 K5 → V1 K10 → K8 V1-u → K8 C1 → K8 C0.
`scripts/q3d_resume_queue.py` owns a lifetime lock and explicitly recovers
failed stages from verified checkpoints. Generic campaign execution retains
its prohibition on implicit resume. Recovery receipts and superseded logs are
preserved under the server base.

`scripts/q3d_watchdog.py` is the server-resident 420-second monitor. It records
`MONITOR_STATUS.json` and `monitor.jsonl`, restarts only after positively
observing process exit and absence of a live writer, and bounds restart attempts.
An alive but stale process is flagged for inspection rather than killed.

An additional Astra agent alarm checks every 10 minutes through Orca automation
`bf11b8cd-f129-4d11-9e5c-fc36fe30c460` (`Q3D overnight 10-minute repair`).
Its bounded, read-only SSH precheck skips healthy runs. The agent alarm depends
on the local Orca host remaining available; the server watchdog does not.
The user explicitly requested Astra instead of Claude. Claude workers were
stopped and remaining implementation/review was reassigned to Astra.

## Evidence status

Software checks are `SOFTWARE_ONLY`: `check_repo` PASS; full pytest **778 passed, 2 skipped in 222.44s** (including the isolated clean-copy suite); all 13 synthetic recipes completed. See `pytest.txt`, `check-repo.json`, `smoke-summary.json` and `review.md`. Deployment, resumed P/J progress and
monitor activation are `NOT_RUN` until a separate timestamped launch receipt
is added to this directory. Completion of the six J runs and held-out test
evaluation are `NOT_RUN` at the time of this source amendment.
