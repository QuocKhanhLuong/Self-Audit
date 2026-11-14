# A-aug — ACDC semi-supervised K8

Status: SOFTWARE_READY. A-aug full patient training/VAL/TEST: NOT_RUN; GPU profile follows.

Validation: 991 passed, 4 skipped, 2 warnings, 3 subtests passed; check_repo PASS;
CPU smoke PASS with three-mode bitwise cold-load. GPU augment tests: 11 passed.
One independent Claude review: PASS, no blocking findings. See raw outputs and review.
Phase0 and all12 real L8 previews completed; visual QA PASS.

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
