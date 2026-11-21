# Running (vast RTX 5000 Ada, 32 GB) - 2026-11-21

tmux session `w2s`, one window with two panes (top = flow A, bottom = flow B); `ctrl+b o` switches pane, mouse is on.

| Flow | What | Command / log |
|---|---|---|
| A (MedNeXt W) | W done on Modal (val10 0.836, test20 0.860, tau_v 0.99) -> CB (resumed from Modal at update 2050) -> A4SA | `/root/flowA.sh`; log `runs/w2s-ssl4mis-20261120/pipeline_vast_flowA.log`; next: `--steps "CB A4SA A4SAFINAL"` |
| B (M3HL) | T-M3HL official training (30k iterations, seed 1337) -> `scripts/run_pipeline_m3hl_teacher.sh` | `/root/flowB.sh`; log `runs/m3hl-teacher-20261121/T-M3HL/train_stdout.log`; after it ends: `M3WAIT ... S2M3FINAL` |

Workarounds for the unmodified M3HL repo (pin 19151b3): `/root/shims/asyncore.py` (removed in Python 3.12), `/root/shims/utils/ramps.py`
(missing from the repo, taken from SSL4MIS 06df6047), data path must contain `ACDC` (`m3hl/data/ACDC` -> `m3hl/h5_ssl4mis`), `./code` symlink in the run dir.
Per-epoch Dice/Jaccard/HD95/ASD for val and test (student arms): `EPOCH_FULL_METRICS=val,test` -> `<arm>/epoch_full_metrics.{jsonl,csv}`.
