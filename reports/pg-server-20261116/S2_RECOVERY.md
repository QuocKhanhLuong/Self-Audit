# PG K8 S2 recovery — 2026-11-16

Scope: `/root/mount_data/runs/pg-k8-20261116` on `vast-gpu`, host
`ae8a6d2632f0`. Recovery was coordinated with setup Run
`run_f5b7f82c0eb4` and code Run `run_260e742283ea`; both confirmed this
monitor's sole recovery ownership and no concurrent publication or GPU work.

At **11:29:56 UTC**, S2 was running at **update 50/8,000** with finite
loss `2.0415289402008057`, one training process and one matching CUDA process.
The snapshot showed 99% GPU utilization and 18,860/24,467 MiB device memory;
this is a warmup snapshot, not a post-warmup peak-memory measurement. There
was still exactly one `pg-k8` tmux window. The five-minute monitor remained
enabled. No finalization or final TEST transaction existed at this check.

S1 completed 10,000 updates at 11:06 UTC. Its encoder artifact SHA-256 is
`f596a68a692799326aac736fdff085543a495ad13f0d8c82270b4724782b1b8f`.
The exact run config remains
`3ab6655b3cd71ffdef26735713feb958c0541d478d2c460326082b2f2e950fe4`.

Two distinct setup failures were diagnosed and fixed before their respective
recovery invocations:

1. The S2 boundary updated source to `5478f74`, then pytest reported 158
   passed test bodies and two teardown errors. PyTorch's dynamic cuBLAS
   `allow_tf32` attribute could not be deleted by monkeypatch teardown.
   Commit `2ade94c4de7ed7cb71c1a606a881e1b795a71c54` restores that backend
   setting through its public setter in a fixture `finally` block. Both
   runtime cases passed locally and in the server environment. The full
   server boundary then passed repository checks and 158 PG tests.
2. The queue always supplied `--pg-resume`, including to a never-started S2.
   Its first dispatch failed before any optimizer update because
   `S2/teacher_latest.pt` did not exist. The S2 directory was empty, and the
   traceback preceded logger creation and the training loop. Commit
   `7ac281f1465288df3bd47f33fe579d94aca699b3` starts an empty/new S2 fresh;
   any retained S2 artifact still requires strict resume. It also deep-copies
   failed stage records so subsequent progress cannot mutate attempt history.
   Twenty focused tests passed locally. The new full server boundary passed
   repository checks and **163 PG tests in 171.36 seconds**.

The recovery regressions cover empty S2, retained progress without a latest
checkpoint, resumed S2/S3/A4 dispatch, completed shared-stage preservation,
failure-history preservation, and reuse of an already committed TEST
transaction. They are software evidence, not patient efficacy evidence.

Original receipts, queue snapshots and logs remain on the server in:

- `code_updates/failed-S2-20261116T110918Z/`
- `code_updates/recovery-S2-20261116T111907Z/`

Both archives contain hash-bound preservation records. The second retains
the original failed S2 log and the earlier successful boundary validation.
Each restart used the existing `pg-campaign` service with a concrete
`--resume-failed` reason, `autorestart=false`, and `startretries=0`. The queue
skips completed S1. No model recipe, data, split, update budget, historical
track, M&Ms training, or tmux window was changed. The S3 fix `fd12e91` remains
an ancestor of the deployed source; no separate GPU profile was rerun.

Live outcome and verified preservation hashes are recorded in
`S2_RECOVERY_RECEIPT.json`. This is a recovery receipt, not campaign
completion: S3, four arm exports, finalization and committed final TEST
receipts remain required. The PG monitor stays enabled.
