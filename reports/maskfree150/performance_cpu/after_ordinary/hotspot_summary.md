# Maskfree150 local profiling hotspot summary

- status: **completed_bounded**
- benchmark kind: `optimized`
- measured batches: `20`
- evidence class: bounded local software timing; not RTX 5070 Ti, ACDC, clinical, or production evidence

## Stage timing (inclusive p95 / exclusive mean)

| stage | inclusive p95 (s) | exclusive mean (s) |
| --- | ---: | ---: |

## Batch accounting

- full `_train_batch` p50: `2.2758077290200163` s
- full `_train_batch` p95: `2.3936878777545645` s
- process CPU p50: `6.228997499999998` s
- unattributed batch p95 (Python/interval work): `2.3936878777545645` s
- checkpoint calls/time: `{'calls': 0, 'wall_seconds': 0.0, 'process_cpu_seconds': 0.0}`
- connected-component calls/time: `{'calls': 0, 'wall_seconds': 0.0, 'process_cpu_seconds': 0.0}`

No optimization is implied by this ranking; preserve bank, rounds, max-fit, regional, student and RNG contracts when comparing a later run.
