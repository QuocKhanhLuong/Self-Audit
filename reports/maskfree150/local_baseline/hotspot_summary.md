# Maskfree150 local profiling hotspot summary

- status: **completed_bounded**
- benchmark kind: `baseline`
- measured batches: `20`
- evidence class: bounded local software timing; not RTX 5070 Ti, ACDC, clinical, or production evidence

## Stage timing (inclusive p95 / exclusive mean)

| stage | inclusive p95 (s) | exclusive mean (s) |
| --- | ---: | ---: |
| `bank_generation` | 4.228892084496328 | 3.4759232708456693 |
| `audit` | 1.3886277369165327 | 1.2406580353563186 |
| `producer_step` | 0.623958588973619 | 0.6046499895019224 |
| `student_no_audit_step` | 0.21632191280659754 | 0.2040539916983107 |
| `student_audited_step` | 0.21322692054964137 | 0.20477405854908284 |
| `data_load` | 0.07220893543562852 | 0.05829505419824273 |
| `feature_forward` | 0.06775286458432675 | 0.06187505210400559 |
| `optimizers.step` | 0.01256932286778465 | 0.010218085494125262 |
| `batch.to_device` | 0.0006410377944121138 | 0.00044918544590473176 |

## Batch accounting

- full `_train_batch` p50: `5.875419104471803` s
- full `_train_batch` p95: `6.74719943991804` s
- process CPU p50: `9.720155000000005` s
- unattributed batch p95 (Python/interval work): `0.02750835941405967` s
- checkpoint calls/time: `{'calls': 26, 'wall_seconds': 0.2777517081121914, 'process_cpu_seconds': 0.26751300000000455}`
- connected-component calls/time: `{'calls': 6305, 'wall_seconds': 87.69144190906081, 'process_cpu_seconds': 87.48337099999904}`

No optimization is implied by this ranking; preserve bank, rounds, max-fit, regional, student and RNG contracts when comparing a later run.
