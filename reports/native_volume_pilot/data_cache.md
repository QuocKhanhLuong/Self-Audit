# Native volume bounded data cache

Status: **IMPLEMENTED / CPU-manufactured software verified only**.

`VolumeTrainingData` now accepts `max_cached_arrays` (default `8`) and rejects
booleans, non-integers, and values below one.  Payloads remain hash-checked on
every access before a cache hit is promoted.  The cache is a deterministic
least-recently-used `OrderedDict`; inserting a new mmap handle evicts only the
least-recent handle until the configured capacity is met.

Eviction deliberately does **not** close a NumPy mmap.  A caller can retain a
returned array/view after its cache entry is evicted, and Python's normal
reference lifetime keeps that mapping readable.  Therefore `len(loader.cache)`
is a bounded count of retained cache handles, not a bound on all live mmap
views held by callers; repeated misses can temporarily leave more live views
than the cache capacity.  Existing label-read accounting, image/mask hash
validation, geometry provenance checks, and image-only U role gates are
unchanged.

## Verification

```text
rtk pytest -q tests/test_volume_data.py
20 passed

rtk pytest -q tests/test_volume_data.py tests/test_volume_geometry.py
28 passed

rtk python scripts/check_repo.py
PASS (semi-supervised native v2 contract)
```

The tests use only deterministic manufactured fixtures.  They cover strict
capacity validation, loading more arrays than capacity, LRU hit promotion and
eviction, readability of an evicted mmap view, byte-identical labeled samples
with capacities 1 and 8, and the existing image-only U mask/read guard.

Source hashes for this delivery:

| File | SHA-256 |
|---|---|
| `self_audit/volume/data.py` | `a5e7448c112e9cd70cd8b9e222837cfd0aa9b0a8d23414c51ecdc8f2d1a18cc8` |
| `tests/test_volume_data.py` | `00a0d5438e24c7b50ef229e2e6e84d814b26e4c574e67857169fdaa4d0a7cc41` |
| `self_audit/volume/geometry.py` (unchanged accepted source) | `3fdb0986e4ff572c6855743414fa1f443e631d57450fe3b7803e5e52f3e1c9bd` |

No patient data, GPU execution, full training, network asset, or full-suite
pytest run was performed for this narrow cache task.  These receipts do not
establish patient efficacy, GPU throughput, or production mmap residency.
