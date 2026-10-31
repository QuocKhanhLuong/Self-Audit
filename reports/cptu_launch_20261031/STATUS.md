# RUNNING: four-arm exploratory CPTU pilot

Launched from source `786511b47ce0a9aa6da8e166bafdfe52126b8838` on RTX 5000 Ada.
Plain KD completed 2,000/2,000 steps and its final exported-student VAL18
evaluation. Original scalar guard subsequently passed 1,000 steps and was
training with nonzero accepted U updates. Larger-tie and cross-patient utility
remain queued. This is a running-state snapshot, not a completed four-arm result.

| Arm / final step 2,000 | VAL18 Macro4 | Foreground | BG | RV | MYO | LV |
|---|---:|---:|---:|---:|---:|---:|
| Plain KD | 70.6674% | 61.2466% | 98.9298% | 49.9931% | 62.0642% | 71.6825% |

Plain KD has not met the 87% deployment quality gate. Pooled RV recall/precision
are 44.061% / 69.980%; these are explicitly pooled diagnostic counts, not
patient-macro recall/precision. `PLAIN_KD_FINAL_SUMMARY.json` binds these values
to its export and metric hashes. No comparative utility verdict is available.

Teacher remains **NO_GO** (TEST50 Macro4 0.8682269380656147), and raw-export
Drive verification remains **UPLOAD_PENDING**. The user explicitly authorized
this exploratory pilot. Deployment acceptance remains Macro4 >= 0.87 plus
matched measured inference resource improvements; none is claimed here.

Server checks passed: `check_repo.py`, **949 tests / 2 warnings**, four-arm CUDA
smoke, exact checkpoint/resume weights/optimizer/RNG/selector replay, and a
32-step real-data bounded profile. Profile accepted 13 nonzero U updates and
rejected 11 after 8 warm-up steps; this is a mechanism/resource check, not a
patient quality result. Encoder, decoder and head weight changes were measured.

- Run root: `/workspace/cptu-launch-20261031`.
- tmux: `cptu_exploratory_20261031` (independent of SSH connection).
- Exact executed commands: `server_launch/LAUNCH_COMMAND.json` and `COMMANDS.json`.
- Training source tree: `/workspace/Self-Audit-cptu-launch-20261031`, detached at the source commit above.
- Arm logs/checkpoints: `pilot/runs/<arm>/segments/step_001000`, `step_001500`, `step_002000`.
- Runner log: `runner.log`; process-tree RSS and GPU samples: `RESOURCE_SAMPLES.jsonl`.
- CUDA verification: `server_launch/cuda_verification/VERIFICATION.json`.
- Measured profile: `server_launch/pilot/profile/PROFILE_RECEIPT.json`.
- Final comparison (when all runs finish): `pilot/TRANSFER_UTILITY_PILOT_REPORT.json`.

Budget remains 2,000 steps including 1,000 warm-up per arm, seed 42, L8/U72,
L/U batch 8, crop 160, original optimizer/loss/threshold/guard settings and eight
scales. All four arms share compact initialization and L/U schedule. Teacher
cache FP32 was generated once: 72 U volumes, 16,507,115,008 bytes. Creation
wall time was 102.06 seconds (model/cache writer measured 83.75 seconds); peak
GPU allocated/reserved 2,400,991,744 / 3,709,861,888 bytes. Cache and images are
streamed; no batch or precision change was made. Profile elapsed 76.68 seconds
includes process setup; it is not an inference benchmark or a precise full-run ETA.

The unchanged screening rule requires utility final VAL18 Macro4 to exceed all
three controls by >0.002. A same-batch control runs only if utility exceeds
larger-tie by >0.002. No joint run, sweep, or teacher retraining is queued.
VAL18 checkpoints are evaluated at 1,000/1,500/2,000; the final comparison uses
step 2,000. TEST50 is not used by compact training or selector decisions.

Exact launch argv (already running; **do not rerun it**):

```bash
/workspace/cptu-runtime-20261030/bin/python /workspace/Self-Audit-cptu-launch-20261031/scripts/continue_b8_transfer_pilot.py --plan /workspace/b8-repro-setup-20261031/LOCKED_CPTU_EXPLORATORY_786511b.json --plan-sha256 970a4abfa7d18b9a6f97793ee6ba29b14ab94c40d9caac7d671ab506da116ea2 --allow-backup-deferred --allow-exploratory-teacher --exploratory-authorization /workspace/cptu-launch-20261031/EXPLORATORY_TEACHER_AUTHORIZATION.json --exploratory-authorization-sha256 dbcea6c02557c39ef7e411b8b6bd54aff6185f0808e03753c407b5a7237a53bd
```

Monitor without changing the job:

```bash
ssh vast-gpu 'cat /workspace/cptu-launch-20261031/STATUS.json; tail -n 10 /workspace/cptu-launch-20261031/runner.log'
ssh vast-gpu 'tail -f /workspace/cptu-launch-20261031/RESOURCE_SAMPLES.jsonl'
```

`RUNNER_EXIT.json` is written when the runner ends. A failed process must be
inspected and resumed from its existing state; do not launch a replacement seed
or delete the lock/output. HANDOVER includes the exploratory authorization,
source/config/cache hashes, run paths, and observed state.

## Earlier teacher measurement and initial stop

# CPTU launch: explicitly authorized exploratory pilot

**Latest instruction:** after seeing the actual below-threshold teacher result,
the user explicitly chose “Chạy bốn arm thăm dò với teacher hiện tại”.
`EXPLORATORY_TEACHER_AUTHORIZATION.json` pins that decision to the exact teacher,
evaluation hashes, and unchanged budget. The four arms are being prepared; no
training is claimed before a live launch receipt. Teacher quality remains
**NO_GO**, deployment Macro4 remains **>= 0.87**, and backup remains
**UPLOAD_PENDING**. This explicit exploratory authorization supersedes the
earlier instruction to stop at the teacher quality gate for this pilot only.

The section below records the completed gate measurement and initial stop
before the user authorized exploratory continuation.

Observed 2026-10-31 UTC. The user's request to start training after the Drive
quota block was recorded as an explicit backup deferral. No metric, label,
training budget, teacher state, or scientific gate was changed.

The completed original B8 reproduction was evaluated before starting the four
compact arms. Its final raw student does **not** pass the previously locked
TEST50 Macro4 >= 0.87 gate. All four arms are **NOT_RUN**, and this is **not** an
empirical NO_GO finding about cross-patient utility. No new teacher run, larger
budget, replacement seed, focal-teacher substitution, or joint run was launched.

| Final raw B8 evaluation | Patients / frames | BG | RV | MYO | LV | Macro4 | Foreground |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAL18 | 18 / 36 | 99.5943% | 76.2355% | 82.9313% | 90.6001% | **87.3403%** | 83.2556% |
| TEST50 | 50 / 100 | 99.6051% | 76.4968% | 81.7547% | 89.4341% | **86.8227%** | 82.5619% |

Macro4 is the mean of four class-level patient means. Each class-level patient
score averages its valid reference frames. This is not pooled-voxel Dice and
foreground is reported separately. The raw saved frame records independently
reproduce both aggregates; prediction coverage is complete for both cohorts.
All class metrics have all patients present. Predictions were frozen and hashed
before opening reference masks, using the unchanged historical evaluator.

RV patient-macro recall/precision are 74.8997% / 86.9744% on VAL18 and
77.1741% / 82.6912% on TEST50. Every evaluated patient has nonzero foreground
predictions on its reference frames. `TEACHER_DIAGNOSTICS.json` preserves class
prediction/reference counts and the five lowest RV cases per cohort. These
diagnostics describe this frozen result; they were not used to tune training.

TEST50 remains an exposed development reference, not a fresh confirmatory
holdout. The final checkpoint was fixed before evaluation; no checkpoint search
or tuning was performed. The historical B8 reference was Macro4
0.8800492397526454 and foreground 0.8412102397257094. The reproduction is lower
by 1.18223 percentage points Macro4 and misses the 87% gate by 0.17731 points.
This is a measured quality-reproduction shortfall. Its cause has not been
isolated; it is not evidence of global foreground collapse, an established
implementation bug, or a general failure of few-patient learning.

## Bound execution and evidence

- Teacher training source: `c06c4c9ae08610e047f1e960712f551690de4067`.
- Source closure: `07809c252ae67d3ef17b148511c5973296c2367f18284a41766e8248fa3fdc2d`.
- Final checkpoint manifest: `a743b23eaf07a1a0ad957399547199c7ddd877654f0faa090b3faa61801520f1`.
- Full payload: `ae89c3378fe02f30bf5e64c8977bfc260e126a4c7b9218510eb6dcdfc99cb8e1`.
- Raw student: `100e811c1cd1b02db99d8d0e78b4c26ab5a1ba96454ff1a039dd2c25c5279ef2`.
- Historical evaluator: `0dc928c24dfa6ba860a8805088b29391dba03051e78555a9d5243c87938756ff`.
- VAL metrics: `01b4bd9c7e2b3d6ee426a7846a57b97e7b30834fd7e98e05499cc5901c833600`.
- TEST metrics: `887e05762b9963e79bb36de9c84e4bf1aebbf279f9a3698e0c1868a609ca8158`.

The teacher completed its single 10,000-step run, with 1,000 warm-up steps,
10,000 supervised updates, 9,000 pseudo updates and 10,000 EMA updates. Trainer
elapsed time was 2,151.44656 seconds. L8/U72 coverage and label lineage remain
unchanged. `load_checkpoint` strictly loads `payload['student']`; EMA was not
scored. Runtime was RTX 5000 Ada, FP32, native uncropped cines, batch 1, two CPU
threads, no TTA or postprocessing. These evaluation times are not a matched
inference benchmark.

Server teacher root:
`/workspace/ema_full_B8_repro_20261031T012656Z_c06c4c9`.
Evaluation logs, exact executed argv, hashes and authorization:
`/workspace/cptu-launch-20261031`.
The evaluation tmux job exited normally. Four-arm configuration remains 2,000
steps each, 1,000 warm-up, seed 42, L/U batch 8, crop 160 and the original eight
trial scales. No cache/profile/arm training was started after the failed gate.

`EVAL_COMMANDS.json` records the exact commands; `evaluate_teacher.py` is the
executed operator wrapper. `INDEPENDENT_METRIC_CHECK.json` records the second
aggregation and coverage check. `val18/` and `test50/` preserve the original
metrics, recipe and frozen-prediction receipts, including patient/frame results.

Backup remains a separate **UPLOAD_PENDING** task. Existing full-checkpoint
readback evidence is preserved in the earlier B8 reports; this evaluation does
not certify completion of the missing raw-student/provenance Drive uploads.
The already-trained focal variant has not been silently substituted as teacher.

The evaluation evidence archive was uploaded using the connected Drive tool,
then downloaded through an authenticated Drive raw-file read and independently
hashed. Rclone readback still failed; that failed attempt is preserved on the
server. [Drive evaluation archive](https://drive.google.com/file/d/1dS2VyGdDK_s3Z9KHfh5fwHHTI1Z1NYJT/view?usp=drivesdk):
63,408 bytes, SHA256 `27b981f200827f4ac9b971a8dc72580c370267c7e4836f488d11f9ac93c62eea`.
`EVALUATION_DRIVE_PROOF.json` is the verification receipt. This archive contains
reports, not the outstanding raw-student weight payload.
