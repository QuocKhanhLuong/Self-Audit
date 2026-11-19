# Server 4070 memory probe, Phase S1 (2026-11-19)

Command: `scripts/cbkd_sa_probe.py --updates 20` on GPU 1 (RTX 4070, 11.6 GiB visible), config `A4_SA_CONFIG.json`, code `1d8242a`. Raw: `selfaudit_root/logs/probe.log`, `selfaudit_root/runs/probe-a4sa/PROBE.json`. A first attempt (S1 only) stopped at S2 for the missing MedNeXt checkout (`probe_attempt1_*` kept). The W2S-Match student step is not probed here (Part B, later).

## Measured
| Stage | Peak allocated | Wall per update (20 updates incl. start-up, one eval) | In-loop step (logged diag row) |
|---|---|---|---|
| S1 CineMA distillation | 3.67 GiB | 25.5 s | n/a (no diag) |
| S2 MedNeXt-L teacher | 4.77 GiB | 14.0 s | 1.8 s |
| S3-SA codebook | 5.64 GiB | 16.1 s | 1.1 s |
| A4-SA student (PG + CBKD-SA) | **10.13 GiB** | 15.8 s | 25.5 s (one row, see below) |

- No stage OOMed. No memory measure (§2 a, b, c) was applied; batch, crop, resampling, precision and update counts are unchanged.
- Headroom is small on A4-SA: 10.13 GiB allocated against 11.6 GiB visible; reserved/fragmented memory was not recorded. Interpretation: an OOM later (for example when CBKD-SA starts at update 1200) is possible; `expandable_segments:True` is the first allowed measure if it happens.

## Projection (low confidence, interpretation)
Update counts from the config: S1 10,000, S2 8,000, S3 4,000, A4-SA 8,000.
The 20-update wall time is dominated by start-up, checkpoint and evaluation, so updates x wall x 1.15 is a gross over-estimate (for A4-SA it would give about 40 h for 8,000 updates). The logged in-loop step time is one row per run, so it is not a rate either (the A4-SA row may include diagnostics). With the in-loop rows: S2 about 4.0 h x 1.15, S3 about 1.2 h x 1.15; S1 and A4-SA cannot be projected from this probe.
The real per-epoch time (80 updates per epoch) comes from the first epoch lines of the actual run; the totals for Part A and Part B will be reported from those. No budget caps (`CAP_HOURS_*`, `MAX_HOURS`) were set for the launch, so the 1.3x guard is inactive until a projection is available.

## First real-run rate (added at launch, measured)
S1 epoch 1 of 125 (80 updates per epoch, incl. one holdout evaluation and checkpoint) took about 9.9 min (data list written 21:55:31, epoch 1 checkpoint 22:05:24). If that holds, S1 alone is about 20 h on the 4070; S2, S3-SA and A4-SA come on top. GPU 1: 4.8 GB used, 100 % utilization. The first epoch includes warm-up, so epoch 2 is the better rate; totals for Part A and Part B will be re-projected from later epoch lines.
