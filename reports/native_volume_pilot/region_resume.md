# Region batched-resume/capacity repair

Date: 2026-10-26
Owner: Luna (task `task_f54233631c10`)
Scope: `self_audit/volume/regions.py`, `tests/test_volume_regions.py`

## Defect and repair

The R3 witness used `RegionManager(mode="ema", refresh_every=5,
max_cache_size=2)` with step 0 batch `[A=zeros, B=ones]` and teacher class 1.
After deep-copying the state into a default manager, step 1 batch
`[A=zeros, C=twos]` used teacher class 3.  The prior loader restored the last
combined batch object into the per-sample LRU, creating a third entry and
evicting A; uninterrupted A stayed class 1 while resumed A became class 3
(maximum target-probability difference `1.0`).

`RegionManager.load_state_dict` now restores `cached_targets` only as the
separate last-combined-return object.  It never inserts that object into the
per-sample identity cache.  Serialized cache entries are restored in their
original LRU order and trimmed only to the restored `max_cache_size`; no
combined-target legacy fallback is used.

## Executed proof

The new regression `test_batched_resume_capacity_boundary_preserves_sample_lru_and_partitioning`
checks all of the following:

- state contains exactly two per-sample entries and the combined hash is not a
  sample-cache key;
- loading into a default manager restores capacity `2`, exact LRU order, and
  exact per-sample labels/weights/teacher targets;
- the separate last combined target is restored exactly without consuming
  capacity;
- uninterrupted and resumed `[A, C]` targets agree bitwise, with A class 1 and
  maximum absolute difference `0.0`;
- resumed processing split into `[A]` then `[C]` agrees with the logical batch,
  covering the direct microbatch-partition path.

An independent direct witness produced:

```text
state_cache_entries 2
state_cache_hash_in_entries False
resumed_capacity 2 resumed_cache_entries 2
combined_hash_in_resumed_cache False
A_class_uninterrupted 1
A_class_resumed 1
max_abs_diff 0.0
```

## Verification receipts

```text
rtk python -m pytest -q tests/test_volume_regions.py tests/test_volume_resume.py tests/test_volume_engine.py
44 passed in 2.11s
```

This is CPU/manufactured software evidence only; no patient, GPU, or full
training evidence was run.  Owned Python files end with exactly one newline.

Source SHA-256:

```text
self_audit/volume/regions.py       a192976e0a08247589a6b2e2ee688782cbf28b76cb53c1774ddd0fd6caa3eca6
tests/test_volume_regions.py       379080cd766d2e84ba280bab9aa9cfa76245fef976ca8c18b04c9886b6fbe620
```
