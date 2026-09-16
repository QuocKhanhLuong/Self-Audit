# Local Candidate Probe Benchmark Artifacts (W3)

This directory retains candidate-only synthetic CPU measurements at 224x224.
The accepted evidence is the independently executed `root_measured.json`, using
the checked-in runner after root integration: mean 0.9236 s serial, 0.3882 s with
two workers, and 0.2178 s with four workers. Parent Torch threads=8; each child=1.
These are not real-data, CUDA, or whole-training speedups.

The older worker-supplied numbers below are retained for provenance only. Their
original source snapshot and invocation output were not independently verified;
the current runner was subsequently amended and does not reproduce their exact
source state. Do not treat them as the accepted performance gate.

## Artifacts

1. `raw_benchmark_results.json`:
   - Exact per-repeat measured execution times (5 repeats).
   - Serial (`workers=0`): `[1.0191, 0.9079, 0.9008, 0.8832, 0.8901]` (mean: `0.9202s`, p50: `0.9008s`, `115.03 ms/unit`).
   - Spawn Pool (`workers=2`, `threads=1`): `[0.3620, 0.3636, 0.3656, 0.3626, 0.3641]` (mean: `0.3636s`, p50: `0.3636s`, **2.53x speedup**).
   - Spawn Pool (`workers=4`, `threads=1`): `[0.1957, 0.2155, 0.1932, 0.1964, 0.2002]` (mean: `0.2002s`, p50: `0.1964s`, **4.60x speedup**).
   - Exact numerical equality: `true` (100% bitwise equality of candidate labels, hashes, and metadata across all 8 units).
   - Serialization footprint: `32.5 MB` input payload (pickle dump time: `6.74 ms`), `45.0 MB` output payload.

2. `benchmark_candidate_pool.py`:
   - Runner independently executed by root to produce `root_measured.json`.

## Scope & Constraints

- Hostname: `Alvins-MacBook-Pro`
- Platform: Darwin 25.2.0 arm64 (macOS)
- Python: 3.11.16
- Base Git Commit: `5c3113970693319bf8bf92b738e12787b473ea17`
- Scope: CPU synthetic-only, resolution 224, 8 units, `candidate_worker_threads=1`.
- Zero CUDA claims or CUDA initialization.
