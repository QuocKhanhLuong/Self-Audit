# One additional B8 focal + Dice / adaptive-LR run

This is a separately authorized experiment based on the working full-model B8
recipe at `c06c4c9ae08610e047f1e960712f551690de4067`. The completed CE+Dice
baseline and its checkpoint are preserved. This variant starts from the same
DINOv3 initialization and seed; it does not continue the trained B8 checkpoint.
It changes two factors together and cannot isolate their individual effects.

`configs/b8_focal_adaptive.json` is the locked server recipe. A launch plan pins
its SHA256, code, DINO source/pretrained bytes, TRAIN/VAL/TEST manifests and the
actual server paths. Only the new supervised loss and scheduler blocks, plus
required local pretrained paths, may differ from the historical B8 config.

- Full-model `ema_full`, DINOv3 ConvNeXt-Tiny + unchanged native decoder/head.
- ACDC L8/U72; labeled IDs patient007, patient010, patient011, patient024,
  patient026, patient041, patient077, patient099. No M&Ms or extra labels.
- Seed 42, batch 8, native crop 256, window none, FP32, two CPU threads.
- 10,000 total iterations, including 1,000 full-model supervised warm-up.
- Initial body AdamW LR 0.00003, head SGD LR 0.003, original weight decay,
  clipping, foreground-aware TRAIN sampling and image augmentation.
- EMA decay 0.99, U confidence 0.7; unchanged hard pseudo CE and optimizer scope.
  One EMA update after the committed student state per iteration.

The supervised loss is focal all-class loss with gamma=2 and no alpha/class
reweighting, plus the original present-foreground soft Dice, coefficients 1:1.
Focal is `-(1-p_target)^2 log(p_target)`, averaged over valid pixels. Label 255
and padded pixels contribute neither loss nor denominator. Foreground Dice
continues to use RV/MYO/LV with the historical present-class policy. No extra CE
term is added. This explicitly changes supervision, not the pseudo-target loss.

Adaptive LR means `ReduceLROnPlateau` for both existing optimizers using only the
mean supervised TRAIN loss over complete 100-iteration windows after warm-up.
Both schedulers receive the same scalar: factor 0.5, patience 5 monitoring
events, relative improvement threshold 0.0001, cooldown 0, eps 1e-8, minimum LR
0.01 times each initial LR. The first observation is after iteration 1,100;
losses from warm-up are excluded. Scheduling occurs after the iteration and
sets the next iteration's L/U LRs. There is no VAL/TEST-driven scheduler,
early stopping, step extension or post-result threshold adjustment.

Checkpoints must include both scheduler state dictionaries and the partial
monitor-window sum/count; missing or mismatched state cannot silently restart
the schedule. Logs expose used/next LRs, monitor loss and scheduler events.
Default recipes without either new block retain CE+Dice and constant LR.

## Runner and backup

The only training entry point remains `train.py`. The outer
`scripts/run_b8_focal_adaptive.py` validates a hash-pinned plan, uses the existing
canonical trainer, exports final raw student weights, uploads full checkpoint
and raw payloads through rclone, reads remote bytes back for SHA256/size, then
evaluates the final raw student. It records `TRAIN_STARTED.json` and refuses a
second fresh trajectory in the same run. An interrupted run uses `--resume`
with optimizer/scheduler/RNG state; upload errors never authorize retraining.

```bash
python scripts/run_b8_focal_adaptive.py --plan /absolute/LOCKED_PLAN.json \
  --plan-sha256 ACTUAL_SHA256 --preflight-only
python scripts/run_b8_focal_adaptive.py --plan /absolute/LOCKED_PLAN.json \
  --plan-sha256 ACTUAL_SHA256
```

Use a dedicated tmux session and the actual plan/commands recorded in the launch
receipt, not placeholders. Check GPU occupancy first. The new Drive directory
uses `ema_full_B8_focal_adaptive_<UTCtimestamp>_<gitsha>` under the user's
`KhanhDrive:Self-Audit-v2-artifacts/`. Existing backups are preserved. A bounded
upload failure records `UPLOAD_PENDING` and keeps all local weights. Successful
byte readback is persisted in `UPLOAD_PROGRESS.json` before querying the Drive
file ID, so an API metadata quota failure cannot erase checksum evidence.

Evaluation uses the same historical native frame → patient/class → class mean
aggregation. Report Macro4, foreground and BG/RV/MYO/LV separately on VAL18 and
the already-exposed development TEST50. Final checkpoint only; no tuning on
TEST50. The historical 0.8800492397526454 Macro4 / 0.8412102397257094 foreground
reference is not this run's result. No Dice or speed improvement is promised.
The variant identity schema is separate and does not automatically replace the
original B8 teacher for the pending compact/CPTU study.
