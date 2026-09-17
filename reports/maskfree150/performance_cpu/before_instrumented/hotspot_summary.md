# Maskfree150 local profiling hotspot summary

- status: **completed_bounded**
- benchmark kind: `baseline`
- measured batches: `20`
- evidence class: bounded local software timing; not RTX 5070 Ti, ACDC, clinical, or production evidence

## Stage timing (inclusive p95 / exclusive mean)

| stage | inclusive p95 (s) | exclusive mean (s) |
| --- | ---: | ---: |
| `bank_generation` | 4.502003563917242 | 3.813725754461484 |
| `audit` | 2.0944672102370534 | 1.838913204841083 |
| `producer_step` | 0.6144652538641822 | 0.5858466854406288 |
| `student_audited_step` | 0.2029129940434359 | 0.19773997279407923 |
| `student_no_audit_step` | 0.19954203366360163 | 0.1934020042535849 |
| `data_load` | 0.09231077290023677 | 0.08408064800023567 |
| `feature_forward` | 0.060524028824875134 | 0.05772587079845835 |
| `optimizers.step` | 0.012038613136974163 | 0.010407293742173352 |
| `batch.to_device` | 0.000781410385388881 | 0.0004774103523232043 |

## Batch accounting

- full `_train_batch` p50: `6.893134333018679` s
- full `_train_batch` p95: `7.4557229187194025` s
- process CPU p50: `10.811662999999996` s
- unattributed batch p95 (Python/interval work): `0.048805889603681867` s
- checkpoint calls/time: `{'calls': 2, 'wall_seconds': 0.046149375033564866, 'process_cpu_seconds': 0.045050000000003365}`
- connected-component calls/time: `{'calls': 5897, 'wall_seconds': 96.19659716781462, 'process_cpu_seconds': 96.16948399999931}`

No optimization is implied by this ranking; preserve bank, rounds, max-fit, regional, student and RNG contracts when comparing a later run.
