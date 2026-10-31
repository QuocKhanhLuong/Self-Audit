# B8 focal + Dice / adaptive LR — launch and completion status

**Update 2026-10-31 06:19 UTC:** training completed all 10,000 steps.
Full checkpoint Drive SHA256 is verified; raw-student/provenance backup remains
blocked by quota403. Evaluation and compact arms have not run. See
[the completion and backup receipt](COMPLETION_20261031.md). The launch record
below is preserved as historical evidence.

The one additional authorized full-model run started **2026-10-31 03:14:30 UTC
(10:14:30 Vietnam)**. The observed prefix reached 1,144/10,000 iterations, passed
warm-up, saved `step_001000`, and recorded the first scheduler observation at
iteration 1,100. No training failure was present. This is a launch receipt, not
a final experiment result.

- **IMPLEMENTED:** opt-in Focal+Dice, TRAIN-only plateau LR, complete scheduler
  checkpoint/resume, runner, raw export, rclone backup/readback and evaluation.
- **VERIFIED:** AGY worker implementation reviewed by coordinator; server
  `check_repo.py` PASS, full suite **127 passed, 1 warning**; CPU exact resume
  before/after LR decay; CUDA resume smoke; actual L8 patient optimizer scope.
- **RUNNING:** one patient run, fixed 10,000 iterations including 1,000 warm-up.
- **NOT_RUN:** final VAL18/TEST50 evaluation and final-weight Drive verification.
- **GO/NO_GO:** not decided. No Dice, inference-speed or research gain is claimed.

## Immutable execution identity

| Item | Value |
|---|---|
| Branch | `QuocKhanhLuong/b8-focal-adaptive-20261031` |
| Training Git SHA | `dbcf83ccd84a5b774a1e33faead7ed6cbd1f973a` |
| Runner Git SHA | `780a632d7228b730725b945d96119d1f380935ae` |
| Training source SHA256 | `e3b5b9469d5378b50677b7225eddd16421d4de856ee6d883b4dfc719f98905d2` |
| Config SHA256 | `281670278fd0f2af08ac514f481a1c42368e997560f8acc91aa16ed942eb5f8d` |
| Locked plan SHA256 | `e6724c68a2844ef93683d919dc82f316f23beaa72ca5d897379220e00cec159c` |
| TRAIN manifest SHA256 | `e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4` |
| DINO pretrained SHA256 | `21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54` |
| GPU | NVIDIA RTX 5000 Ada Generation, 32 GB |
| Runtime | Python 3.12.15, torch 2.14.1+cu132; full receipt in `RUNTIME.json` |

ACDC labeled IDs remain patient007, patient010, patient011, patient024,
patient026, patient041, patient077, patient099; U72 has image paths only.
Seed42, batch8, crop256, initial body/head LRs 3e-5/3e-3, DINO initialization,
decoder initialization, augmentation, sampling, pseudo CE, confidence0.7 and
EMA0.99 remain fixed. This run changes two factors together: supervised Focal
gamma2 plus original foreground Dice (1:1, no alpha reweight), and dual
ReduceLROnPlateau on complete 100-iteration TRAIN-loss windows after warm-up.
It starts fresh and does not resume the completed CE+Dice baseline.

`PARITY_FINAL.json` records a six-iteration manufactured CPU comparison against
historical source c06c4c9: identical default config, full student/EMA/optimizer/RNG
payloads and non-timing log rows. `MATCHED_SAMPLER_PREFIX.json` records the actual
patient metadata comparison against the completed B8 baseline for the observed
1,144-step prefix. These are bounded checks, not universal parity guarantees.

Actual-patient one-step diagnostic verified optimizer ownership, nonzero
gradients and weight changes in encoder, input adapter, lateral/up/refine,
detail, fuse and head. It took 9.03 seconds including setup/checkpoint diagnostic
work; peak CUDA allocated/reserved were 3,314,556,928 / 3,726,639,104 bytes.
Diagnostic weights are retained separately and are not this experiment's
initialization. See `patient_scope/GPU_SCOPE_SMOKE.json`.

## Live paths and commands

Run root:
`/workspace/ema_full_B8_focal_adaptive_20261031T030334Z_dbcf83c`

Tmux: `b8_focal_20261031`; observational resource monitor:
`b8_focal_resources_20261031`. Existing user session `ssh_tmux` was preserved.

The actual runner command (already launched; **do not launch a second run**) is:

```bash
/workspace/cptu-runtime-20261030/bin/python \
  /workspace/Self-Audit-b8-focal-runner-20261031/scripts/run_b8_focal_adaptive.py \
  --plan /workspace/b8-focal-adaptive-setup-20261031/LOCKED_FOCAL_PLAN.json \
  --plan-sha256 e6724c68a2844ef93683d919dc82f316f23beaa72ca5d897379220e00cec159c
```

Monitor from the local machine:

```bash
ssh vast-gpu 'cat /workspace/ema_full_B8_focal_adaptive_20261031T030334Z_dbcf83c/STATUS.json'
ssh vast-gpu 'tail -f /workspace/ema_full_B8_focal_adaptive_20261031T030334Z_dbcf83c/training/steps.jsonl'
```

The canonical training command is preserved exactly in `TRAIN_STARTED.json`.
Logs/checkpoints live under `training/`; final paths will be
`training/final/weights.pt` and `raw_student/raw_student.pt`.
Resource samples are in `RESOURCE_SAMPLES.jsonl`; host RAM is sampled, not an
exact whole-run peak. Only an interrupted same run may use runner `--resume`.

## Backup and evaluation

Destination: [new Drive folder](https://drive.google.com/drive/folders/14CO6QFWtKBHFB9i0FinbdOvG607Lk7AA).
No final weights exist there yet. A probe copied into this folder was downloaded
by its exact ID through rclone and SHA256-verified before training; its creation
method and the original rclone upload/readback witness are recorded in
`DRIVE_WRITE_VERIFIED.json`.

The runner will finalize atomically, export/reload raw student, attempt both real
weight payloads first, verify remote bytes, then evaluate final raw student on
VAL18 and exposed-development TEST50 with the historical evaluator. Backup uses
flat names such as `full_checkpoint__weights.pt`. Google Drive's shared OAuth
project has intermittent quota403 failures; `UPLOAD_PENDING` preserves local
weights and upload intents, and never retrains. An uncertain write is not
blindly repeated, avoiding duplicate creates. Remote file IDs and checksum
verification remain required before a completed-backup claim.

The original CE+Dice B8 run/checkpoints are preserved. The focal variant does
not automatically replace its teacher or start compact/CPTU arms.
