# Q3D shared-stage plan (2026-11-12)

Status: `PLAN_ONLY_NOT_RUN`. This is a plan artifact only; no campaign jobs, patient data, or GPU training were executed. Real-data metrics and GPU validation remain `NOT_RUN`.

## Selection

Datasets: ACDC and M&Ms. Protocols: A and B. K values: 2, 5, and 10. Seed: 42. Settings: 4 primary, 8 secondary (including `v1_decoder_stride8`), and 3 baselines; 180 experiment records total.

## Shared stage DAG

| Stage | Planned jobs |
|---|---:|
| T | 12 |
| Q | 60 |
| P | 60 |
| J | 168 |

There are 12 distinct T producers, 60 Q producers, and 60 P producers. V1 and V1-u have 12 planned shared-reference records, each pinning the same T/Q/P producer job IDs with manifest SHA-256 values unset until production. C1, V1, V1-u, and E1 also share the same T/Q/P refs per dataset/protocol/K/seed; C1 records Q as optional source metadata. O1, S1, stride4, and decoder_stride8 use separate Q/P source artifacts while sharing T.

The MedNeXt supervised comparison records reuse of the shared Q3D T region-head for possible export/evaluation, while `independent_baseline=NOT_RUN`; this is not an independent MedNeXt result. Under Protocol B, shared T uses the all-TRAIN teacher view and does not establish a separate K-labeled supervised baseline.

The real stride4 and decoder_stride8 Q/P/J jobs are marked `NOT_RUN` with the open feature-grid phase contract reason. Synthetic variants remain marked `SMOKE_READY` in synthetic-only plans; this report itself includes no synthetic run evidence.

## Evidence boundary

The plan records expected jobs and artifact paths. All stage manifests and checkpoint hashes remain unresolved; patient training, GPU validation, real-data metrics, and novelty claims remain `NOT_RUN` or `NOT_CLAIMED`.
