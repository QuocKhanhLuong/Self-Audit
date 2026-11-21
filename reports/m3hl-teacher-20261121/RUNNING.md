# Running (vast RTX 5000 Ada, 32 GB) - 2026-11-21

tmux session `w2s`, one window with two panes (top = flow A, bottom = flow B); `ctrl+b o` switches pane, mouse is on.

| Flow | What | Command / log |
|---|---|---|
| A (MedNeXt W) | W done on Modal (val10 0.836, test20 0.860, tau_v 0.99) -> CB (resumed from Modal at update 2050) -> A4SA | `/root/flowA.sh`; log `runs/w2s-ssl4mis-20261120/pipeline_vast_flowA.log`; next: `--steps "CB A4SA A4SAFINAL"` |
| B (M3HL) | T-M3HL official training (30k iterations, seed 1337) -> `scripts/run_pipeline_m3hl_teacher.sh` | `/root/flowB.sh`; log `runs/m3hl-teacher-20261121/T-M3HL/train_stdout.log`; after it ends: `M3WAIT ... S2M3FINAL` |

Workarounds for the unmodified M3HL repo (pin 19151b3): `/root/shims/asyncore.py` (removed in Python 3.12), `/root/shims/utils/ramps.py`
(missing from the repo, taken from SSL4MIS 06df6047), data path must contain `ACDC` (`m3hl/data/ACDC` -> `m3hl/h5_ssl4mis`), `./code` symlink in the run dir.
Per-epoch Dice/Jaccard/HD95/ASD for val and test (student arms): `EPOCH_FULL_METRICS=val,test` -> `<arm>/epoch_full_metrics.{jsonl,csv}`.

## Queue after the 150-epoch rule (2026-11-21 evening, vast RTX 5000 Ada)

Every student from now on runs 150 epochs (12,000 updates = `training.epochs` x `updates_per_epoch` 80). Finished arms keep their 100-epoch budget and are **not comparable** to 150-epoch arms:
S0 (100), S1kd (100), S2-m3hl-arb (100). A4-SA 100-epoch partial run is kept untouched in `A4SA_e100_partial` (stopped at epoch about 27, never resumed).
Schedules: fraction-defined (cosine LR over the total updates, `warmup_fraction` 0.05, alpha ramp 20% -> 50%, lambda_u ramp_fraction) scale with the total; absolute onsets (labeled-only warm-up 800
updates, CBKD onset 1200 with a 400-update ramp, PG onset) stay at their absolute update counts. Teacher/codebook stages (W, T-M3HL, CB, M3CB, S1) keep their budgets.

Order (`/root/queueE.sh`, one student at a time, `flock /root/gpu_heavy.lock`): `A4SA` -> `A4SAFINAL`; `S1KDM3` -> `S1KDM3FINAL`; `M3CB`; `S2M3` -> `S2M3FINAL` (only if CB-out passes the fallback rule).
Projected ends, from the measured 85-90 s/epoch of a student running alone (CBKD stages may be slower): A4SA about 3.7 h from 20:54 UTC, S1KDM3 about 3.8 h after it, M3CB about 1 h, S2M3 about 4 h.
Not queued (blocked, see the message to the user): S1-ANISO and the two-arm screening (S0-unetrpp-150, S0-aniso).

## Queue after the 150-epoch rule (2026-11-21 evening, vast RTX 5000 Ada)

Every student from now on runs 150 epochs (12,000 updates = `training.epochs` x `updates_per_epoch` 80). Finished arms keep their 100-epoch budget and are **not comparable** to 150-epoch arms:
S0 (100), S1kd (100), S2-m3hl-arb (100). The 100-epoch A4-SA partial run is kept untouched in `A4SA_e100_partial` (stopped at epoch about 27, never resumed).
Schedules: fraction-defined (cosine LR over the total updates, `warmup_fraction` 0.05, alpha ramp 20% -> 50%, lambda_u ramp_fraction) scale with the total; absolute onsets (labeled-only warm-up 800
updates, CBKD onset 1200 with a 400-update ramp, PG onset) stay at their absolute update counts. Teacher/codebook stages (W, T-M3HL, CB, M3CB, S1) keep their budgets.

Order (`/root/queueE.sh`, one student at a time, `flock /root/gpu_heavy.lock`): `A4SA` -> `A4SAFINAL`; `S1KDM3` -> `S1KDM3FINAL`; `M3CB`; `S2M3` -> `S2M3FINAL` (only if CB-out passes the fallback rule).
Projected ends, from the measured 85-90 s/epoch of a student running alone (CBKD stages and any concurrent job are slower): A4SA about 3.7 h from 20:54 UTC, S1KDM3 about 3.8 h after it, M3CB about 1 h, S2M3 about 4 h.
Not queued (blocked): S1-ANISO and the two-arm screening (S0-unetrpp-150, S0-aniso), because the S1 distillation taps only exist at CineMA 1/8 and 1/16 (config `taps`), so distilling into E0-E2 needs new taps that are not defined.
