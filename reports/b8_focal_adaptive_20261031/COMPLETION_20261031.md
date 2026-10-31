# Training complete; backup partially verified

Observed 2026-10-31 06:19 UTC (13:19 Vietnam).

| Stage | Actual status |
|---|---|
| Original B8 CE+Dice training | COMPLETED, 10,000 iterations; evaluation pending backup |
| Additional B8 Focal+Dice/adaptive training | COMPLETED, 10,000 iterations, no FAILURE.json |
| Focal full checkpoint on Drive | VERIFIED by exact-ID download, size and SHA256 |
| Focal raw student on Drive | UPLOAD_PENDING: quota403 on bounded retries |
| Focal full/raw off-server local backup | VERIFIED, both SHA256 values match |
| VAL18 and TEST50 metrics | NOT_RUN; backup-before-evaluation gate remains |
| Four compact/CPTU arms | NOT_RUN; original continuation is BLOCKED on backup |
| Quality GO/NO_GO | NOT_DECIDED; no final Dice evidence |

The focal trainer reports 2,161.4038 seconds (36m01s), 10,000 supervised updates,
9,000 accepted pseudo updates, 10,000 EMA updates and 90 scheduler observations.
Final body/head learning rates are 3.75e-6 and 3.75e-4, respectively. Label reads
outside the declared L8 pool are empty. Low TRAIN losses are not evidence of
VAL/TEST quality. Source, recipe, label budget and training budget were not
changed; no additional training was launched during this recovery.

- Full checkpoint: 526,262,024 bytes, SHA256
  `50d47171a2dca5c8de06c6c679e6324595e1a02efc0bb678a7cbf0df4000b05e`.
  [Drive payload](https://drive.google.com/file/d/1P4-k6gkTJwvkpPtja0EP8x5nmWoRF79k/view),
  file ID `1P4-k6gkTJwvkpPtja0EP8x5nmWoRF79k`.
- Raw student: 131,543,874 bytes, SHA256
  `86be1885559e7ffff0ad2c30079cc16ee03de5fe8df1f12fb1f65f4068a1f8b2`.
  Strict state and fresh-model forward reload passed during export. No verified
  Drive raw-student payload exists yet.
- Full manifest SHA256:
  `8364ef55a2b62c3c0aeba31f12e2138dba9da7f1a64049d2716cd190bd2b2e16`.

Server run root remains
`/workspace/ema_full_B8_focal_adaptive_20261031T030334Z_dbcf83c`.
The bounded recovery logs and full readback are in
`backup_retry_20261031T0614Z/`. The uploader's original intent journal was
preserved before adding the observed, SHA256-verified full-checkpoint file ID.
The raw retry still failed with Google API quota403 on project202264815644.
The runner is stopped at `UPLOAD_PENDING`; it is not training or evaluating.

Additional verified local backup:
`/Users/alvinluong/.local/share/openresearch/files/self-audit/ema_full_B8_focal_adaptive_20261031T030334Z_dbcf83c/`.
This protects both payloads outside the ephemeral server; it does not replace
the requested Drive verification.

Evidence: `FINAL_TRAIN_REPORT.json`, `REMOTE_PARTIAL_RECOVERY.json`,
`LOCAL_BACKUP_VERIFIED.json`. Next: complete raw-student and provenance backup,
verify remote payloads, resume postprocessing of the same completed run, then
evaluate final raw student. No retraining, checkpoint selection on TEST, or
automatic replacement of the original B8 teacher is authorized by this status.
