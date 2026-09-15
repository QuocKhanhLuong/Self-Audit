# Maskfree150 local profiling hotspot summary

- status: **completed_bounded**
- benchmark kind: `optimized`
- measured batches: `20`
- evidence class: bounded local software timing; not RTX 5070 Ti, ACDC, clinical, or production evidence

## Stage timing (inclusive p95 / exclusive mean)

| stage | inclusive p95 (s) | exclusive mean (s) |
| --- | ---: | ---: |
| `audit` | 0.675708772946382 | 0.6066393896500812 |
| `producer_step` | 0.598032510120538 | 0.583231566703762 |
| `bank_generation` | 0.4814806662849151 | 0.4520310557389166 |
| `student_no_audit_step` | 0.20335979556548409 | 0.19795513755525462 |
| `student_audited_step` | 0.20187768894829788 | 0.19728970825672149 |
| `feature_forward` | 0.06302926214702893 | 0.06029914790124167 |
| `data_load` | 0.05853109163581394 | 0.050952977244742216 |
| `optimizers.step` | 0.014876064233249055 | 0.011739908298477531 |
| `batch.to_device` | 0.0008690333314007148 | 0.00048008104786276815 |
| `features.to_cpu` | 1.0062515502795578e-05 | 8.364600944332778e-06 |

## Batch accounting

- full `_train_batch` p50: `2.2109471040021162` s
- full `_train_batch` p95: `2.2524264273873995` s
- process CPU p50: `6.030232500000004` s
- unattributed batch p95 (Python/interval work): `0.043605876879883` s
- checkpoint calls/time: `{'calls': 2, 'wall_seconds': 0.04704870900604874, 'process_cpu_seconds': 0.04592099999999277}`
- connected-component calls/time: `{'calls': 5897, 'wall_seconds': 3.7897662156028673, 'process_cpu_seconds': 3.768257999999701}`

No optimization is implied by this ranking; preserve bank, rounds, max-fit, regional, student and RNG contracts when comparing a later run.
