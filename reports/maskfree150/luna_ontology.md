# maskfree150 connected-components CPU path

## Scope

This report covers the bounded `ontology.connected_components` optimization in
`src/self_audit_maskfree/ontology.py` and its independent equivalence tests in
`tests/test_maskfree_ontology_fast.py`. It is CPU software evidence only; no
training, real MRI, clinical, or GPU performance claim is made here.

## Implementation

The CPU branch keeps the original input validation, raster ID initialization,
masking, synchronous recurrence, stability-check placement, and finite budget
`B = 2 * (height + width)`. Before entering that recurrence it uses SciPy's
documented [`ndimage.label`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html)
with an explicit self-plus-four cross to enumerate the same 4-connected
components. For each component, the maximum raster ID is selected as its sole
seed; [`ndimage.binary_dilation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_dilation.html)
with the same cross, `mask=mask`, `iterations=B-1`, and `brute_force=False`
checks whether every component seed reaches all of its pixels by the last state
that the original stability check could inspect. A full coverage certificate
returns the component labels remapped to those original maximum IDs with
`converged=True`; a non-full certificate runs the existing exact NumPy
self-plus-four recurrence, preserving partial labels and the original cap
boundary. CUDA/MPS/other non-CPU devices continue through the original torch
implementation unchanged, including deterministic algorithm behavior and device
locality.

### Exactness argument

Consider the graph induced by the `True` pixels, with raster ID `r(v)` at each
vertex. After `t` synchronous max-propagation updates, a vertex contains the
maximum `r(u)` among vertices at graph distance at most `t`; this follows by
induction from the self-plus-four recurrence. Therefore a component with
maximum-ID seed `s` is stable after `e+1` checks, where `e` is the largest
distance from `s` to that component, because the implementation checks for
stability before assigning the next state. With cap `B`, it returns a complete
labeling and `True` exactly when `e <= B-1`. The masked cross dilation computes
the same graph balls from every `s` for `B-1` iterations, so full coverage is a
certificate for the complete fast return; otherwise the unchanged recurrence is
the exact fallback.

## Validation status

The independent test module freezes the original recurrence with an explicit
Python reference and covers exhaustive small masks, seeded random masks,
seeded 128-pixel dense/sparse/non-contiguous masks, empty/all-one/one-pixel/1xN/Nx1
inputs, non-contiguous masks, finite-budget serpentine non-convergence and
cap-1/cap/cap+1 boundaries, disconnected components with different
eccentricities, validation errors, an explicit maximum-raster-ID assertion, and
an optional deterministic CUDA fallback check. The focused command
`rtk env PYTHONPATH=src pytest -q tests/test_maskfree_ontology_fast.py` is rerun
after this change and passes with **22 passed, 1 skipped** (CUDA unavailable;
0.88 s). The hash-verified immutable source snapshot
(`88b1a3c51c0f38d6a875a65703ab9cad59e816e57ed1e7695a220351f701c375`) was also
loaded independently and matched labels, flags, and input immutability over
**716 cases**, including all 3x3 masks, seeded 128-pixel layouts, cap-boundary
snakes, disconnected components, and edge cases.
Root also independently imported the hash-verified immutable baseline source
and matched labels plus convergence flags over 541 cases, including serpentine
non-convergence and a 1x128 case.

The existing maskfree integration checks
`rtk env PYTHONPATH=src pytest -q tests/test_maskfree_hypotheses.py
tests/test_maskfree_auditor_fast.py` also pass: **10 passed** (0.98 s).

## Performance evidence

No worker CPUmicro benchmark was run: root retained the quiet timing gate while
the trainer/data/profiler harness stabilized and will own the accepted matched
timing later. Therefore this report makes no speedup claim; any future result
must use an identical representative 128x128 mask set (including hard masks),
be labelled `CPUmicro`, and remain strictly CPU software evidence with no GPU
claim.
