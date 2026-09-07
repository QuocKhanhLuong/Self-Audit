# AGY-4 — Volume-level / native-geometry evaluation

**File owned and edited:** `src/self_audit/evaluation/volume_inference.py` (only file touched)
**Report:** `reports/workers/AGY4_volume_geometry.md`
**Baseline:** `a901eeae56cf6835bb9a1d9c79a9d5d3e99374c6`
**Scope:** measurement only. No training path, no model, no config, no checkpoint semantics touched.

---

## 1. What was wrong

- `evaluate_comparison_modes` reconstructed a Z-volume and then **nearest-resized the ground truth *down* to the 256×256 network grid** before scoring it (the `F.interpolate(..., mode="nearest")` on `target_tensor`). Nothing in the returned dict said so, so any Dice taken from it would have been quoted as if it were a volume-level result at acquisition geometry.
- Zero callers anywhere in the repo, so the mislabelling had not yet propagated into a reported number.
- ED/ES pooled; no per-class RV/MYO/LV breakout; no empty-class policy; `per_class_dice`'s legacy `empty_score=1.0` inflated the macro whenever a class was absent from both prediction and target.
- Latent bug: `oracle_predictions: list[Tensor]` / `oracle_initial: list[Tensor]` referenced an undefined name `Tensor`. Harmless at runtime (local annotations are never evaluated, and `from __future__ import annotations` is in effect) but wrong. Fixed to `list[torch.Tensor]`.

## 2. What changed and why

### 2.1 Module docstring — the three metric spaces are now stated in the file

A new docstring section names `slice_proxy` / `volume_resized` / `volume_native`, says which functions produce which, and states explicitly that **`volume_native` is not derivable from `preprocessed_data/`**: `scripts/preprocess_acdc.py:80` nearest-resized the stored mask to 256×256 destructively, so inverse-resizing a prediction and comparing it to an inverse-resized copy of that same downsampled mask measures the resampler, not the model. Native geometry *metadata* exists (`orig_shape`, `orig_spacing`, `effective_spacing`, `num_slices`); native *ground truth* does not.

### 2.2 Everything is stamped, nothing is inferred

Constants are imported from `self_audit.audit.semantics` (`METRIC_SPACE_SLICE_PROXY`, `METRIC_SPACE_VOLUME_RESIZED`, `METRIC_SPACE_VOLUME_NATIVE`) — not redefined here. Every metric block carries `metric_space` as a field. A caller never has to guess.

### 2.3 Empty-class policy, exactly per the shared contract

`per_class_dice_with_policy` applies `empty_class_score(policy)` **only when the denominator is zero**, i.e. only when the class is empty in *both* prediction and target (`nan` under the default `"exclude"`). A class present in exactly one of them has a non-zero denominator and therefore scores `0.0` by the ordinary Dice formula and is never excluded — total misses stay visible. Macro aggregation goes through `macro_mean` (nan-aware; `nan` when every class is excluded). Verified in both directions in §4.B.

### 2.4 HD95 is opt-in on real spacing only

`_resolve_physical_spacing` has **no default**. Without a caller-supplied `(z, y, x)` mm tuple the `hd95_mm` / `assd_mm` / `per_class_hd95_mm` keys are **absent from the dict entirely** — not `None`, not 1.0 mm pixel units. HD95/ASSD use `scipy.ndimage.distance_transform_edt` (spacing-aware, exact, O(N)) rather than the `metrics.surface_metrics` brute-force pairwise fallback, which is O(N²) in surface voxels and would not survive a native-resolution cardiac volume.

## 3. New and changed public signatures

Changed (backwards compatible — all additions are keyword-only with defaults; positional signature untouched):

```python
def evaluate_comparison_modes(
    model, volume, ground_truth, *,
    image_size=256, depth_axis=None, tau_accept=0.0, t_max=3,
    device=None, batch_size=8, metrics_fn=None,
    empty_policy=None,          # NEW  -> semantics.resolve_empty_policy, default "exclude"
    neutral_margin=None,        # NEW  -> semantics.resolve_neutral_margin, default 0.005
    num_classes=4,              # NEW
    class_names=None,           # NEW  default {1:"RV", 2:"MYO", 3:"LV"}
    geometry=None,              # NEW  declared metadata ONLY, nothing computed from it
) -> dict[str, Any]
```

New:

```python
DEFAULT_CLASS_NAMES: dict[int, str] = {1: "RV", 2: "MYO", 3: "LV"}
ACDC_PHASES = ("ED", "ES");  UNKNOWN_PHASE = "unknown"

def resolve_class_names(class_names=None, num_classes=4) -> dict[int, str]
def per_class_dice_with_policy(prediction, target, *, num_classes=4, empty_policy=None) -> dict[int, float]
def dice_block(prediction, target, *, metric_space, num_classes=4, class_names=None,
               empty_policy=None, geometry=None) -> dict[str, Any]
def to_native_geometry(prediction_zhw, orig_shape, *, order=0, shape_order="hwz") -> np.ndarray
def evaluate_volume_native(prediction_zhw, native_ground_truth_zhw=None, *,
                           orig_shape=None, spacing=None, empty_policy=None,
                           num_classes=4, class_names=None, shape_order="hwz") -> dict[str, Any]
def split_cases_by_phase(case_ids) -> dict[str, list[str]]   # keys "ED", "ES", "unknown" always present
```

A module `__all__` was added listing the public surface.

### 3.1 Return shape of `evaluate_comparison_modes`

Per-mode, for each of the four modes:

| key | status |
|---|---|
| `inference` | unchanged (`VolumeInferenceResult`) |
| `metrics` | unchanged (whatever `metrics_fn` / `annotation_metrics` returns) |
| `metric_space` | **new** — always `"volume_resized"` |
| `volume_dice` | **new** — the self-describing block below |
| `macro_dice_delta_vs_initial` | **new** — macro delta against `initial_only` |
| `transition_class` | **new** — `classify_delta(delta, neutral_margin)` as `+1/0/-1`, or `None` if the delta is not finite |

`volume_dice` block: `metric_space`, `num_classes`, `empty_policy`, `per_class_dice` (int-keyed), `per_class_dice_named` (`RV`/`MYO`/`LV`), `excluded_classes`, `excluded_classes_named`, `macro_dice`, `class_names`, `geometry`.

One **new non-mode top-level key `"evaluation_meta"`** was added (`metric_space`, `grid`, `image_size`, `empty_policy`, `neutral_margin`, `num_classes`, `class_names`, `geometry`, `modes`, `ground_truth_source="preprocessed_resized"`, `native_dice_available=False`, `native_dice_note`).
⚠️ **Consumers must iterate `COMPARISON_MODES`, not `results.keys()`.** There were zero callers at the time of writing, so nothing breaks today; `"modes"` inside `evaluation_meta` gives the list explicitly.

### 3.2 `orig_shape` axis order — deliberately refuses to guess

`preprocess_acdc.py` writes `orig_shape` as `img_data.shape` = `[H, W, Z]`. `to_native_geometry` therefore defaults to `shape_order="hwz"`, accepts a 2-tuple `(H, W)`, and accepts `shape_order="zhw"` for a depth-first triple. When a 3-entry `orig_shape` declares a depth that does not match the prediction's `Z`, it **raises** naming both conventions rather than silently transposing.

### 3.3 Loud behaviour-change notice

- **`empty_policy` now defaults to `"exclude"` (`nan`), not the legacy `1.0`.** Any macro Dice computed through `dice_block` / `per_class_dice_with_policy` will be **lower** than the pre-remediation number whenever a foreground class was absent from both prediction and target — on ACDC this is routine on apical and basal slices. That drop is the correction, not a regression. `empty_policy="legacy_one"` reproduces the old value for explicit comparison and is verified to still work (§4.B).
- **A one-sided-empty class makes the macro HD95/ASSD `inf`** (that class scores `inf`, and `macro_mean` propagates it). This is intentional: a total miss must not be averaged away into a flattering millimetre number. Per-class values are always exposed alongside the macro.

## 4. Verification (real output, synthetic tensors only — no `preprocessed_data/`, no `weights/`)

Scripts: `…/scratchpad/agy4_verify.py`, `…/scratchpad/agy4_ecm.py` (outside the repo).

### 4.A — hand-built 3-slice volume, per-class Dice computed by hand

```
class1: |P|=12 |T|=8 |P&T|=8
class2: |P|=2 |T|=2 |P&T|=1
class3: |P|=0 |T|=0
hand-computed: RV=2*8/(12+8)=0.8000000000  MYO=2*1/(2+2)=0.5000000000  LV=empty-empty
dice_block -> {'metric_space': 'volume_resized', 'per_class_dice_named': {'RV': 0.8, 'MYO': 0.5, 'LV': nan},
               'excluded_classes_named': ['LV'], 'macro_dice': 0.65, 'empty_policy': 'exclude'}
PASS RV Dice matches hand value 0.8000000000
PASS MYO Dice matches hand value 0.5000000000
PASS LV (empty in BOTH) -> nan == excluded
PASS excluded_classes == [3]
PASS macro = mean of the 2 SURVIVING classes = 0.6500000000 (not /3)
PASS metric_space stamped volume_resized
```

### 4.B — empty-class semantics in both directions

```
per_class: {1: 0.0, 2: 1.0, 3: nan}
PASS RV present in GT / absent in pred -> 0.0 (total miss NOT hidden)
PASS RV is NOT excluded
PASS MYO perfect -> 1.0
PASS LV empty-empty -> excluded
pred-only RV: {1: 0.0, 2: nan, 3: nan}
PASS RV predicted but absent in GT -> 0.0, not excluded
legacy_one: {1: 0.0, 2: 1.0, 3: 1.0}
PASS legacy_one policy still gives 1.0 for empty-empty (comparison only)
```

### 4.C — slice order preserved through `reconstruct_volume` → `to_native_geometry`

Each slice carries a distinct constant label (1, 2, 3), fed in **shuffled** with `slice_indices=[2,0,1]`:

```
reconstruct signature per slice: [1, 2, 3]
PASS reconstruct restored order 1,2,3
to_native_geometry shape: (3, 13, 7) signature per slice: [1, 2, 3]
PASS shape [Z, orig_H, orig_W] == (3,13,7); Z untouched
PASS slice order preserved through inverse resize
```

### 4.D — no new label values, no interpolation

```
input label set: [0, 1, 3]
  -> (37, 41) output labels: [0, 1, 3] shape (3, 37, 41)
  -> (5, 5)   output labels: [0, 1, 3] shape (3, 5, 5)
  -> (256, 3) output labels: [0, 1, 3] shape (3, 256, 3)
PASS no new labels at (37, 41) / (5, 5) / (256, 3)
  order=1 -> to_native_geometry only supports order=0 (nearest neighbour); got order=1. Interpolating a…
PASS order=1 refused (no bilinear on labels, ever)
  bad depth -> orig_shape [3, 37, 41] under shape_order='hwz' declares depth 41, but the prediction has 3 slices. preprocess_…
PASS depth mismatch raises instead of guessing the axis order
PASS shape_order='zhw' accepts a depth-first triple
```

Label 2 was deliberately removed from the input so a fabricated intermediate label would be visible. The subset assertion is also enforced inside the function itself, not only in the test.

### 4.E — `evaluate_volume_native`

```
evaluate_volume_native requires real native ground truth (raw ACDC NIfTI labels at the original
acquisition geometry). The 256x256 mask in preprocessed_data/ is NOT a substitute: it was
destructively nearest-resized by scripts/preprocess_acdc.py, so inverse-resizing it measures the
resampler, not the model. Refusing to return a metric_space='volume_native' number without it.
PASS raises and names the substitute trap
PASS omitting native_ground_truth entirely also raises

no-spacing block keys: ['class_names', 'empty_policy', 'excluded_classes', 'excluded_classes_named',
 'geometry', 'ground_truth_source', 'macro_dice', 'metric_space', 'native_shape', 'num_classes',
 'per_class_dice', 'per_class_dice_named', 'spacing_known']
PASS metric_space == volume_native
PASS RV exactly 1.0 after inverse resize (=1.0)
PASS LV in GT only -> 0.0
PASS MYO empty-empty -> excluded
PASS NO hd95 key at all without spacing (nothing invented)
PASS spacing_known False
PASS GT source recorded

with spacing: hd95_mm=inf assd_mm=inf per_class_hd95={1: 0.0, 2: nan, 3: inf}
PASS hd95 present ONLY when real spacing given
PASS perfect RV -> HD95 0.0 mm
PASS one-sided-empty LV -> inf, not a flattering number
  bad spacing -> spacing must have 3 values in (z, y, x) order, got (1.0, 1.0)
PASS spacing length validated
```

### 4.F — `split_cases_by_phase`

```
{'ED': ['patient001_ED'], 'ES': ['patient001_ES'], 'unknown': ['weird_case']}
PASS ED bucket / ES bucket / unrecognised suffix -> "unknown", NOT guessed
PASS all three keys always present
{'ED': ['patient099_ed'], 'ES': ['patient002_ES'], 'unknown': ['nosuffix']}
```

### 4.G — `evaluate_comparison_modes` end-to-end (synthetic deterministic stub model)

```
top-level keys: ['always_accept_refinement', 'evaluation_meta', 'initial_only', 'oracle_accept', 'self_audit']

initial_only               metric_space=volume_resized
   per_class_dice_named: {'RV': 1.0, 'MYO': nan, 'LV': 0.0}
   macro_dice=0.5 excluded=['MYO'] delta_vs_initial=0.0 transition_class=0
always_accept_refinement   metric_space=volume_resized
   per_class_dice_named: {'RV': 1.0, 'MYO': 0.0, 'LV': 0.0}
   macro_dice=0.3333333333333333 excluded=[] delta_vs_initial=-0.1666… transition_class=-1
oracle_accept              metric_space=volume_resized
   per_class_dice_named: {'RV': 1.0, 'MYO': nan, 'LV': 1.0}
   macro_dice=1.0 excluded=['MYO'] delta_vs_initial=0.5 transition_class=1

evaluation_meta:
   metric_space             'volume_resized'
   grid                     [3, 16, 16]
   ground_truth_source      'preprocessed_resized'
   native_dice_available    False
   geometry                 {'orig_shape': [24, 20, 3], 'orig_spacing': [1.5, 1.5, 10.0],
                             'effective_spacing': [10.0, 2.25, 1.875], 'num_slices': 3,
                             'source': 'declared_metadata'}
ALL evaluate_comparison_modes ASSERTIONS PASSED
grid actually scored on: [3, 16, 16] <- NOT the native [24, 20, 3]
```

Asserted in that run: legacy `inference`/`metrics` keys still present on all four modes; a class predicted but absent from GT scores `0.0` and is *not* excluded, while the same class absent from both is; the oracle mode (which copies GT) scores exactly 1.0 on the resized grid; `initial_only` has delta 0.0 and `transition_class == 0`.

**Incidental empirical finding worth recording.** A 1-voxel-thick native structure is annihilated by the nearest downsample to the network grid:

```
thin 1-row LV in NATIVE gt -> resized-grid per-class: {'RV': 1.0, 'MYO': nan, 'LV': nan}
   excluded: ['MYO', 'LV']
```

A class that genuinely exists in the native ground truth is scored as *absent-in-both* — hence excluded — on the resized grid. That is a direct, reproducible demonstration of why `volume_resized` and `volume_native` must never be conflated, and it is a second, independent reason (beyond the resampler-measuring-itself argument) that native Dice cannot be recovered from `preprocessed_data/`.

### 4.H — test suite (honest result)

```
$ python3 -m pytest tests/ -q
ERROR tests/test_audit_decomposition.py
E   ModuleNotFoundError: No module named 'self_audit'
!!!! Interrupted: 1 error during collection !!!!
```

```
$ python3 -m pytest tests/ -q --ignore=tests/test_audit_decomposition.py
FAILED tests/test_self_audit_hardening.py::test_metrics_use_consistent_both_empty_dice_convention
1 failed, 69 passed, 1 warning in 3.17s
```

**Neither is mine.** Both come from files another worker is editing right now (`git status` shows `evaluation/audit_decomposition.py`, `evaluation/metrics.py`, `evaluation/threshold.py` all modified):

1. `audit_decomposition.py:27` does `from self_audit.audit.semantics import …` (absolute), but `tests/` import the package as `src.self_audit.…`, so the absolute name is not importable at collection time. **`volume_inference.py` uses a relative import (`from ..audit.semantics import …`) and imports cleanly under both roots** — verified: `src.* import OK volume_native` / `self_audit.* import OK volume_native`.
2. `test_metrics_use_consistent_both_empty_dice_convention` asserts `annotation_metrics` gives `{1: 1.0, 2: 1.0, 3: 1.0}` for an all-background case; `metrics.py` now returns `nan` under the new `"exclude"` default. That is the intended contract change; the old test still encodes the legacy convention. It touches `metrics.py` + `tests/`, neither of which I own.

With those two excluded, **69 passed, 0 failures**.

## 5. What I could NOT do

- **Re-export the new functions from `src/self_audit/evaluation/__init__.py`.** I do not own that file. `to_native_geometry`, `evaluate_volume_native`, `split_cases_by_phase`, `dice_block`, `per_class_dice_with_policy` and `resolve_class_names` are currently reachable only as `from self_audit.evaluation.volume_inference import …`. **Requested integration edit** — add to the existing `from .volume_inference import (...)` block and to `__all__`:
  `DEFAULT_CLASS_NAMES`, `dice_block`, `evaluate_volume_native`, `per_class_dice_with_policy`, `resolve_class_names`, `split_cases_by_phase`, `to_native_geometry`.
- **No test file added.** `tests/` is not in my ownership list and the spec did not authorise a new file there. The acceptance checks live in the two scratchpad scripts named above and their output is pasted verbatim in §4; they are self-contained and can be promoted to `tests/test_volume_geometry.py` verbatim by whoever owns `tests/`.
- **No end-to-end run on real data.** There is no `preprocessed_data/` and no `weights/` in this checkout, and by construction no raw ACDC NIfTI either, so `evaluate_volume_native` has never been executed against a real native ground truth. Every number in §4 is from synthetic tensors and is labelled as such. No real-data number is claimed.
- **`evaluate_comparison_modes` still has zero callers.** Wiring it (and the ED/ES split) into `scripts/audit_checkpoint.py` is AGY-6's task.

## 6. Observations in files I do NOT own (described, not edited)

1. **`src/self_audit/evaluation/metrics.py` — `metric_space` key collision (important).** `annotation_metrics` already returns a key named `metric_space` whose values are `"physical"` / `"pixel"`. That is a *different* axis from the `slice_proxy` / `volume_resized` / `volume_native` metric space in `audit.semantics`, and the two now coexist inside the same result tree: `results[mode]["metric_space"] == "volume_resized"` while `results[mode]["metrics"]["metric_space"] == "pixel"`. This is exactly the kind of conflation the workstream exists to remove. Suggest renaming the metrics.py one to `distance_units` (or `surface_metric_units`) with the old key kept as a deprecated alias. I did not touch it. My blocks never reuse the name for the physical/pixel meaning.
2. **`metrics.py::surface_metrics` does not scale to native geometry.** `_pairwise_min_distances` materialises a full `len(A) × len(B)` float64 matrix. At native ACDC resolution a single class surface can be tens of thousands of voxels, i.e. a multi-gigabyte allocation per class per volume. I sidestepped it with `scipy.ndimage.distance_transform_edt` inside my own file. Suggest metrics.py adopt the same EDT path when SciPy is importable (SciPy 1.17.1 is present in this environment) and keep the pairwise version as a fallback.
3. **`src/self_audit/evaluation/audit_decomposition.py:27`** uses an absolute `from self_audit.audit.semantics import …`. Under the repo's own test import convention (`src.self_audit.…`) this breaks collection of the whole suite. A relative import (`from ..audit.semantics import …`) fixes it. Owner of that file should apply it.
4. **`scripts/preprocess_acdc.py` writes `orig_shape` as `[H, W, Z]`** while `effective_spacing` is `[Z, Y, X]` — the two are in different axis orders in the same JSON record. `data/acdc.py:84-95` already comments on this. It is not a bug, but it is a live trap for anyone reading `metadata.json`; `to_native_geometry` guards against it with `shape_order` plus a depth-consistency check rather than trusting the caller.
5. **`tests/test_self_audit_hardening.py:89-93`** still asserts the legacy `empty_score=1.0` convention and must be updated to the `"exclude"` contract by whoever owns `tests/`.

## 7. Guardrails

No auditor/expert architecture change, no decoder added, no stage embeddings touched, no gate input change, no Phase-A objective or `stage_weights` change, no A0 weakening, no `CounterfactualGenerator` change, no `epsilon_neutral` change, no retraining, no split change, nothing called "test", no SOTA claim, no checkpoint-loading change. Training behaviour is bit-identical: nothing in `infer_patient_volume` or any training path was modified. The only non-additive edit outside the evaluation surface is the `list[Tensor]` → `list[torch.Tensor]` annotation repair, which has no runtime effect.
