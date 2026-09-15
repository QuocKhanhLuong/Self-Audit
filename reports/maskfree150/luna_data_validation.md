# Luna data and epoch-validation handoff

## Scope

This handoff covers the bounded image-source cache and the existing image-only
development validation loop. It is a local CPU/software correctness artifact;
it makes no GPU, throughput, real-cohort, clinical, or production-performance
claim.

## Implementation

- `src/self_audit_maskfree/data/cache.py` adds a process-local bounded LRU of
  immutable decoded native rank-3 source frames. Keys include the canonical
  image path, full manifest digest and manifest identity, source/frame
  fingerprints, frame/depth selectors, and `st_dev`, `st_ino`, `st_size`,
  `st_mtime_ns`, and `st_ctime_ns`.
- Entries are stored contiguous and read-only; public frame reads return a
  writable clone, while stack extraction copies only the requested three
  planes. Decode and oversized three-plane fallback paths verify source stat
  identity before and after reading and reject a concurrent mutation.
- A frame larger than the configured byte budget uses the original canonical
  three-plane reader and is never retained. `stats()`/`snapshot()` expose
  hits, misses, resident bytes, evictions, invalidations, loads,
  loaded-bytes, uncacheable reads, entry count, and the configured bound.
  No mask/GT data, normalization, resizing, role partition, or RNG state is
  cached or changed. The singleton is process-local; no multiprocess worker
  or sampler/resume behavior was introduced.
- `ImageOnlyDataset`, `build_training_unit`, `load_verification_unit`, and
  `load_full_input` preserve their existing positional contracts and accept an
  optional cache. Arbitrary public manifest calls recompute their complete
  digest; the dataset and canonical frozen validation cohort may pass a
  private precomputed digest to avoid repeated hashing without memoising a
  mutable mapping.
- `observe_epoch` still sorts the development IDs, preserves physical batch
  tails, predicts both students, exports native records, freezes before the
  isolated reference process, and restores RNG/module modes. It records a
  source-cache snapshot/delta in receipts but does not use cache counters for
  scientific decisions.

## Correctness evidence

Focused local checks (CPU software only):

```text
python -m py_compile \
  src/self_audit_maskfree/data/cache.py \
  src/self_audit_maskfree/data/dataset.py \
  src/self_audit_maskfree/data/geometry.py \
  src/self_audit_maskfree/epoch_validation.py
env PYTHONPATH=.:src python -m pytest \
  tests/test_maskfree_data_cache.py \
  tests/test_maskfree_epoch_validation_performance.py \
  tests/test_maskfree_data.py \
  tests/test_maskfree_firewall.py -q
```

```text
32 passed in 5.79s
```

The cache tests cover cold/warm equality, no-alias returns, real NIfTI frame
and stack parity, repeated same-volume load counting, stat/manifest
invalidation, LRU byte bounds and oversized skip, rank-3 selector compatibility,
non-trailing rank-4 frame axes, source mutation rejection, exact RNG/order,
and a proxy assertion that oversized fallback requests exactly three planes.
Validation tests cover sorted IDs, the tail batch, both student/native export
records, cohort overlap rejection before loading, receipt counters (including
uncacheable reads), RNG/mode restoration, immutable snapshot identity, and the
rule that a custom callback accepting only ``**kwargs`` is not given the
private frozen-digest argument (only a loader with an explicit compatible
keyword parameter receives it).

`git diff --check` and the four-module compile check pass. No commit, push,
train benchmark, or external data/GPU run has been performed; the bounded
quiet-window timing below is synthetic CPU/NIfTI evidence only.

## Measurement status

The paired timing window was held until the coordinator's explicit quiet-window
grant. The bounded deterministic NIfTI source comparison and canonical
development validation comparison are recorded below; they remain diagnostic
CPU/software evidence rather than a production-throughput claim.

## Authorized quiet-window measurement

The coordinator granted `DATA_QUIET_START`; the bounded run ended with
`DATA_QUIET_END` at UTC `2026-09-15T15:42:09.649158+00:00` (start
`2026-09-15T15:42:09.568305+00:00`). This is a synthetic CPU/NIfTI fixture
measurement, not a training benchmark. The fixture was a deterministic
float32 NIfTI-1 volume of shape `[48, 48, 16]`; discovery produced 16 units
from one source volume, and the measurement assigned those units to `dev`
only for this bounded validation comparison. Cache budget was `294912` bytes;
validation used batch size `5`, image size `24`, and student initialization
seed `20260915`.

Source reads were exactly equivalent. The uncached baseline took
`0.015658500022254884` seconds and called `nib.load` 16 times; cache cold took
`0.0018168750102631748` seconds and called it once; a warm pass took
`0.001032707979902625` seconds. The post-pass cache counters were
`hits=31, misses=1, loads=1, loaded_bytes=147456, bytes=147456,
evictions=0, invalidations=0, uncacheable=0, entries=1, max_bytes=294912`.

The canonical `observe_epoch` comparison used the same manifest, model
initialization, sorted development IDs, physical tails, and loader values. A
local export stub captured native records/labels and a reference stub returned
`UNAVAILABLE` without opening reference data. Both modes emitted 32 native
exports and had exactly equal loaded image tensors and native export outputs.
The uncached observer took `0.04375220800284296` seconds; the cached observer
took `0.010436707991175354` seconds. The cached receipt delta was
`hits=15, misses=1, loads=1, loaded_bytes=147456, evictions=0,
invalidations=0, uncacheable=0`, with resident `bytes=147456` and
`entries=1` after the run; the uncached delta was zero for every counter and
resident bytes stayed at zero.

Raw measurement record:

```json
{
  "schema": "maskfree150.luna_data_validation_measurement.v1",
  "utc_start": "2026-09-15T15:42:09.568305+00:00",
  "utc_end": "2026-09-15T15:42:09.649158+00:00",
  "source": {
    "baseline_uncached_seconds": 0.015658500022254884,
    "baseline_nib_load_calls": 16,
    "cache_cold_seconds": 0.0018168750102631748,
    "cache_warm_seconds": 0.001032707979902625,
    "cache_nib_load_calls": 1,
    "outputs_equal": true,
    "warm_outputs_equal": true,
    "cache_stats_after_cold_warm": {
      "bytes": 147456,
      "entries": 1,
      "evictions": 0,
      "hits": 31,
      "invalidations": 0,
      "loaded_bytes": 147456,
      "loads": 1,
      "max_bytes": 294912,
      "misses": 1,
      "uncacheable": 0
    }
  },
  "validation": {
    "uncached_seconds": 0.04375220800284296,
    "cached_seconds": 0.010436707991175354,
    "uncached_status": "UNAVAILABLE",
    "cached_status": "UNAVAILABLE",
    "loaded_image_values_equal": true,
    "native_export_outputs_equal": true,
    "uncached_export_count": 32,
    "cached_export_count": 32,
    "cached_cache_delta": {
      "hits": 15,
      "misses": 1,
      "loads": 1,
      "loaded_bytes": 147456,
      "evictions": 0,
      "invalidations": 0,
      "uncacheable": 0
    },
    "cached_cache_after": {
      "bytes": 147456,
      "entries": 1,
      "evictions": 0,
      "hits": 15,
      "invalidations": 0,
      "loaded_bytes": 147456,
      "loads": 1,
      "max_bytes": 67108864,
      "misses": 1,
      "uncacheable": 0
    },
    "uncached_cache_delta": {
      "hits": 0,
      "misses": 0,
      "loads": 0,
      "loaded_bytes": 0,
      "evictions": 0,
      "invalidations": 0,
      "uncacheable": 0
    },
    "uncached_cache_after": {
      "bytes": 0,
      "entries": 0,
      "evictions": 0,
      "hits": 0,
      "invalidations": 0,
      "loaded_bytes": 0,
      "loads": 0,
      "max_bytes": 67108864,
      "misses": 0,
      "uncacheable": 0
    }
  },
  "scientific_boundary": "synthetic CPU/NIfTI and stubbed reference/export only; not a real-data, GPU, clinical, or production-throughput claim"
}
```

The complete reproducible probe is checked in beside this report at
`reports/maskfree150/luna_data_validation_probe.py`. It creates and cleans a
temporary fixture/run directory, prints one raw JSON record, and writes no
repository files. Invoke it with:

```text
env PYTHONPATH=.:src python reports/maskfree150/luna_data_validation_probe.py
```

The source timing is a small synthetic diagnostic only. The validation timing
uses student/export/reference stubs, so full per-epoch validation throughput
and native filesystem-export timing are **UNMEASURED**; the observed values are
not a production speedup. All work remains non-representative of real data,
CUDA, or the 150-epoch production run.
