# A-aug — ACDC semi-supervised K8

Status: RUNNING. A-aug started stage T on GPU at 2026-11-14 01:17:54 UTC
(08:17:54 Asia/Ho_Chi_Minh). Final A-aug VAL/TEST comparison: NOT_RUN.
Training source is pinned to `e4836f7af084f19f7a47cfe47c7fc138df50a469`.

Validation: 991 passed, 4 skipped, 2 warnings, 3 subtests passed; check_repo PASS;
CPU smoke PASS with three-mode bitwise cold-load. GPU augment tests: 11 passed.
One independent Claude review: PASS, no blocking findings. See raw outputs and review.
Phase0 and all 12 real L8 previews completed; visual QA PASS.

The sole recipe change from old Track A is `augmentation.policy=inplane_strong_v1`.
The L8/U manifests, native geometry, seed42, architecture, physical teacher batch2,
T500 epochs (8,000 frame exposures / 4,000 optimizer steps), J100 epochs (8,000 pairs),
VAL selection and deployment margin remain unchanged. See `EXPERIMENT_LOCK.json`.

TEST is opening 2: **TEST lần 2, sau khi TEST đã mở**. TEST results will be descriptive;
the primary paired comparison is student A-aug minus student A on VAL18.
B/B0 and all historical run directories are preserved and are not launched.

Implementation is coordinated through Orca `run_d1950a7d7e29`, with parallel
Luna workers and one independent Claude reviewer. CPU checks are SOFTWARE_ONLY.
Real L8 diagnosis/previews and GPU profiles are reported separately when completed.

New run: `/root/mount_data/runs/fuse-acdc-k8-aug-20261114`.
New checkout: `/root/Self-Audit-fuse-aug-20261114`.
The campaign has zero automatic retries. On completion, the server runner
materializes the comparison report and a publish-ready receipt; the coordinator
will verify and push the report to main.

GPU profile on NVIDIA RTX PRO 4000 Blackwell, bfloat16, 20 updates per stage:

| Stage | Matched legacy recipe (s/update) | A-aug (s/update) | Change | A-aug peak GiB |
|---|---:|---:|---:|---:|
| T | 1.0292 | 1.0612 | +3.11% | 6.153 |
| J | 0.7042 | 0.7123 | +1.15% | 12.933 |

These are measured GPU timings, including load/preparation/backward/step; they
exclude validation, checkpoint I/O, setup, export and TEST. The legacy timing
is a new matched control, not a historical timing from old A. Both profiles
include the largest relevant native volumes and passed without OOM. Neither
stage exceeded the 50% slowdown threshold. See `PROFILE_COMPARISON.json`.

Live log: tmux `fuse-k8-aug-training`, with epoch/VAL per-class metrics and raw
training output. The dedicated Supervisor config is `SUPERVISORD.conf` inside
the new run. A read-only watchdog checks every 420 seconds; the completion
automation checks every 5 minutes and publishes verified results after TEST.
Its identity and the live launch verification are in `LAUNCH_RECEIPT.json`.
