# Strict correction frame validation

Date: 2026-10-26
Scope: `scripts/volume_correct.py`, `tests/test_volume_correction.py`

## Outcome

The correction CLI now fails closed on temporal frame selection. Negative
indices are rejected before input I/O; 4-D images require
`0 <= --time-index < T`; and 3-D images accept only `--time-index 0`. The
image frame is validated before prompt loading, ground-truth loading, or the
authoritative real TRAIN-L manifest/mask binding. The existing cheap
pre-I/O forbidden-role gate remains in place, while full authorization runs
only after image shape/frame validation.

Ground-truth 4-D selection uses the same strict non-clamping helper, while an
already 3-D mask remains the single native frame expected by the correction
engine. No changes were made under `self_audit/volume/` or to `train.py`.

## Verification receipts

```text
python -m pytest -q tests/test_volume_correction.py tests/test_volume_cli.py
77 passed in 10.43s

python scripts/check_repo.py
{"status": "PASS", "architecture": "dinov3_convnext_tiny_native_decoder_v2", "supervision": "semi_supervised; dense TRAIN subset incl audit; not no-GT"}
```

The new focused tests cover negative and out-of-range 4-D indices, nonzero
3-D indices, prompt/GT read ordering, and exact frame-0/final-frame boundary
selection. Full-suite execution, patient/GPU training, server work, and
commits were intentionally not run.

## SHA-256

| File | SHA-256 |
|---|---|
| `scripts/volume_correct.py` | `85439b39b33c6f0ac8d814ec8b1e11faee6223660434325a0d1f3303fb86e69f` |
| `tests/test_volume_correction.py` | `0fbd19770dffc99393241f60ac4e68989619447b809917009ef70a7447e77108` |
| `self_audit/volume/correction.py` (unchanged C5/C6 source) | `fdb27bad361d67786652f2d8515ba132639fb606738d99b4a9a0cc1134947f08` |
| `self_audit/volume/models.py` (unchanged model source) | `4095051e59cc7595f9983e70f67fc4eeb05c82720ed1f6307b6e4919006d1dec` |
| `train.py` (pre-existing dirty work, untouched) | `49a18abff36210d351ad74b09c8c47704ffc290125fbeeeef97f115da737140f` |
