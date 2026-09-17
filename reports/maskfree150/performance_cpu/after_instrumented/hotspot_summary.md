# Maskfree150 local profiling hotspot summary

- status: **completed_bounded**
- benchmark kind: `optimized`
- measured batches: `20`
- evidence class: bounded local software timing; not RTX 5070 Ti, ACDC, clinical, or production evidence

## Stage timing (inclusive p95 / exclusive mean)

| stage | inclusive p95 (s) | exclusive mean (s) |
| --- | ---: | ---: |
| `audit` | 0.6810043583827792 | 0.6310686791926855 |
| `producer_step` | 0.5944191496790154 | 0.5798174208961427 |
| `bank_generation` | 0.4809708421729738 | 0.4599901563546155 |
| `student_audited_step` | 0.20617880176287146 | 0.19861930629704147 |
| `student_no_audit_step` | 0.2038367253378965 | 0.1985205583536299 |
| `feature_forward` | 0.06332735859614332 | 0.06072416660026647 |
| `data_load` | 0.06331137264205608 | 0.051350866653956474 |
| `optimizers.step` | 0.014448110392550007 | 0.011376087402459234 |
| `batch.to_device` | 0.0008335642283782365 | 0.000494685402372852 |
| `features.to_cpu` | 9.35057760216296e-06 | 8.331250865012407e-06 |

## Batch accounting

- full `_train_batch` p50: `2.2270801665144973` s
- full `_train_batch` p95: `2.297709006222431` s
- process CPU p50: `6.226600000000005` s
- unattributed batch p95 (Python/interval work): `0.0431048790604109` s
- checkpoint calls/time: `{'calls': 2, 'wall_seconds': 0.04545954102650285, 'process_cpu_seconds': 0.044795999999990954}`
- connected-component calls/time: `{'calls': 5897, 'wall_seconds': 3.7181927891797386, 'process_cpu_seconds': 3.686754999999554}`

No optimization is implied by this ranking; preserve bank, rounds, max-fit, regional, student and RNG contracts when comparing a later run.
