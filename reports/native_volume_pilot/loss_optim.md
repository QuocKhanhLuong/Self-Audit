# Loss / optimizer owner report

## Scope and implementation

Owned paths are `self_audit/volume/losses.py`, `self_audit/volume/optim.py`,
`tests/test_volume_losses.py`, and `tests/test_volume_optim.py`. Existing
`self_audit/losses.py` and all historical training modules were not edited.

## Astra review correction

The initial implementation had a real volume Dice defect: after slicing one
volume, the tensor still had `B,C,Z,H,W` layout, but `_volume_mean` reduced
only `(0,2,3)`, leaving the W axis. This produced the seed-11 float64
`B=2,C=4,Z=3,H=4,W=5` mixed-support value `2.461081459` (and per-column
presence `29/-23`) instead of the independent flattened-volume oracle
`2.454750545`. The reduction now uses `(0,2,3,4)` and counts foreground
presence/absence per supported volume; the legacy pooled path is unchanged.

The regression suite includes an independent flattened-voxel complete-logical-
batch oracle for both scalar and gradients, mixed geometric/label support,
unknown labels, all-empty rejection, an empty-volume plus supported-volume
accumulation case, and exact scheduler warmup/decay/pseudo-delay boundaries.

An additional FP16 guard promotes only low-precision Dice/CE computation to
FP32. This prevents half-precision reductions from overflowing at 65,536
supported voxels while leaving FP32/FP64 legacy behavior unchanged; a bounded
CPU finite-loss/finite-gradient regression covers the case.

The loss API is:

```python
supervised_volume_loss(
    logits, labels, geometric_valid, label_valid,
    mode="legacy_pooled" | "volume_mean", return_metrics=False,
)
pseudo_volume_ce(logits, targets, geometric_valid, pseudo_valid,
                 weight=1.0, return_metrics=False)
combine_accumulated_losses(losses, volume_counts, mode="volume_mean")
pseudo_weight_ramp(step, base_weight, ramp_steps=0, pseudo_delay=0)
```

Logits are `B,4,Z,H,W`; labels/support masks are `B,Z,H,W`; labels use
`UNKNOWN=255`. Geometric support and label support are intersected, and
invalid/padded/unknown positions never enter CE or Dice. `legacy_pooled`
flattens `B*Z` and reproduces the historical CE + present-foreground pooled
Dice exactly when all support is valid. `volume_mean` is explicitly a changed
objective: each supported volume contributes one CE plus present-foreground
Dice value, then volume values are averaged. Absent foreground classes have no
Dice term while CE still includes background/negative false-positive signal.
Pooled Dice cannot be reconstructed from scalar microbatch losses, so
multi-microbatch `legacy_pooled` accumulation fails closed; the volume-mean
accumulator weights microbatch losses by supported volume count and has a
reference-gradient test.

The optimizer API is:

```python
optimizers, metadata = build_optimizers(model, optimization_config)
schedulers, scheduler_metadata = build_schedulers(optimizers, optimization_config)
receipt = safe_optimizer_step(optimizers, clip_norm=..., has_support=...)
```

`grouped_adamw` creates explicit encoder/decoder/head rates and disjoint
parameter ownership. `legacy_adamw_sgd` keeps body AdamW and momentum-free head
SGD separate. `OuterIterationScheduler` indexes the outer transaction, applies
one warmup/decay rate to supervised and pseudo updates in that iteration, and
reports pseudo-delay readiness without mutating optimizer state. Scheduler
state is serializable. `safe_optimizer_step` skips `step()` for empty support
or absent gradients, preventing AdamW decay-only mutations, and returns actual
gradient/update norms and optimizer update counts.

## Verification

Exact bounded command:

```text
rtk python -m pytest -q tests/test_volume_losses.py tests/test_volume_optim.py
```

Result: **16 passed** (CPU, Python 3.11.16, torch 2.14.0; 2026-10-26). The
tests cover historical loss value/gradient equivalence, mixed valid support,
absent foreground CE, UNKNOWN/padding exclusion, volume-mean accumulation
gradient equality against an independent flattened-volume oracle, empty-volume
weighting, pooled accumulation fail-closed behavior, empty pseudo acceptance,
detached-target semantics, pseudo ramps, grouped optimizer rates, legacy
body/head optimizer separation, scheduler warmup/decay/delay boundaries,
scheduler resume, and empty-support no-decay/update receipts. `rtk python -m py_compile`
and `rtk git diff --check` also pass.

The repository gate `rtk python scripts/check_repo.py` returned `PASS`. A
whole-repository run was not repeated in this bounded worker dispatch; the
coordinator owns the final full-suite gate. No loss/optimizer test failed.

This is CPU/software evidence only: no patient masks, U/VAL/TEST masks,
weights, GPU, or long training were accessed. The integration owner still must
export these APIs from the new volume package, route the explicit
`self_audit.volume.v1` schema, call the scheduler once per outer iteration,
and checkpoint each optimizer/scheduler state. Pseudo-weight scaling is a
loss contribution control; AdamW parameter updates are not assumed
proportional to that scalar.

## SHA-256 receipt (working tree, 2026-10-26)

```text
77dbd1cd86e4a74702c63b2c56955405e9c8d4a48c7f3f37630589b54bdb77cd  self_audit/volume/losses.py
502b3ef1d87d7b9add16790a6d7ff93b24642309fc063148672e4e45a8c63fdb  self_audit/volume/optim.py
3d19fc6a6b61186a658ab422732dcc88a900f2e233780cebf6c719d20f7ca30a  tests/test_volume_losses.py
fcc8932fd676d48e668f9a72fc82332e6207c4f7512cd68f0232fe7a0046d36d  tests/test_volume_optim.py
```

The exact receipt command is:

```text
rtk shasum -a 256 self_audit/volume/losses.py self_audit/volume/optim.py tests/test_volume_losses.py tests/test_volume_optim.py
```
