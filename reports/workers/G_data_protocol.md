# Worker G — Data / Protocol Audit

**Repo:** `/Users/alvinluong/Self-Audit` @ `a901eea` (branch `main`)
**Scope:** ACDC split integrity, 2.5-D construction, preprocessing, empty-slice policy, augmentation, volume reconstruction, pretraining, and whether the reported Dice is comparable to published ACDC numbers.
**Mode:** read-only. No file under `src/`, `scripts/`, `configs/`, `tests/` was modified. One file written: this report.

## Environment / evidence caveats

- `torch 2.13.0` and `numpy` import fine under `/Users/alvinluong/miniforge3/bin/python3`, so all executable checks below actually ran. `pytest` is **not** installed in that interpreter (`ModuleNotFoundError: No module named 'pytest'`), so `tests/test_self_audit_data.py` could not be executed as a suite; its assertions were re-derived by hand instead.
- **`preprocessed_data/ACDC` does not exist in this working tree** (`ls: preprocessed_data/: No such file or directory`), and neither does `data/`. Therefore every claim that requires the actual `.npy` volumes — per-split *slice* counts, the real fraction of empty slices, native ACDC in-plane matrix sizes — is **not measurable here** and is flagged as such. Claims about *code behavior* were verified by running the real repo code against synthetic ACDC-shaped volumes.
- No training was run.

---

## 1. ACDC split: patient-level or slice-level?

**Patient-level. Verified. No leakage found in the split artifact or in the data-loading path.**

Only one split file exists: `splits/acdc_patient_split_seed42.json`.

Generation: `scripts/acdc_split.py:62-116` (`build_split_manifest`) groups every `patientXXX_{ED,ES}` volume by the `patientXXX` prefix (`patient_id_from_volume`, `acdc_split.py:29-38`), shuffles the **patient** list (`acdc_split.py:75-77`), then slices patients — not volumes — at `int(len(patients) * 0.8)` (`acdc_split.py:78-83`). `validate_manifest` (`acdc_split.py:119-143`) rejects both direct patient overlap and volume-derived patient overlap.

Consumption: `ACDCDataset.__init__` (`src/self_audit/data/acdc.py:203-243`) → `resolve_acdc_records` (`acdc.py:157-194`) → `read_split_manifest` (`src/self_audit/data/common.py:268-291`), which itself calls `validate_patient_split` (`common.py:216-228`) and raises on any patient present in two splits.

### Executable proof (ran against the checked-in split file)

```
train_patients 80 val_patients 20 overlap []
train_volumes 160 val_volumes 40
volume-derived patient overlap: []
dup volumes across splits: []
patients whose phases straddle splits: {}
patients missing ED or ES: {}
total patients 100 total volumes 200
read_split_manifest keys: {'train': 160, 'val': 40}
sample train case ids: ['patient001_ED', 'patient001_ES', 'patient002_ED'] val: ['patient004_ED', 'patient004_ES', 'patient005_ED']
validate_patient_split: PASS (no patient in two splits)
```

**ED/ES straddle check specifically:** for every one of the 100 patients, both `_ED` and `_ES` land in the same split (`patients whose phases straddle splits: {}`, and every patient has exactly `{ED, ES}`). So the ED/ES pair of a patient can never be split. This is the single most common ACDC leakage mode and it is **not** present.

Loader-level confirmation: running the real `ACDCDataset` on synthetic ACDC-shaped data with a 3-patient / 1-patient manifest yields `train: volumes=6`, `val: volumes=2` — the manifest sets are honored exactly, with no cross-contamination.

### 1b. FINDING (medium) — the preflight split gate validates a *different* split than the one actually trained on

`validate_dataset_splits` (`src/self_audit/training/_utils.py:319-...`) is the gate called at `scripts/train_self_audit.py:227-229` and printed as `phase_a_split_stats=...`. At `_utils.py:369-379` it looks for a `test` split in the manifest; the checked-in manifest has **only** `train` and `val`, so the `else` branch fires and **the whole manifest is discarded** in favour of `patient_level_split(sorted(discovered), seed=42)` — an 80/10/10 partition built with a *different RNG* (`np.random.default_rng`, `common.py:250`) than the manifest's (`random.Random`, `acdc_split.py:76`).

Executable proof:

```
manifest splits present: ['train', 'val']
discovered cases: 200
FALLBACK is used. counts: {'train': 160, 'val': 20, 'test': 20}
fallback val patients  : 10   manifest val patients  : 20
patients the GATE calls train but the LOADER calls val:
  ['patient004','patient005','patient018','patient028','patient029','patient030','patient032',
   'patient055','patient070','patient076','patient082','patient087','patient089','patient095','patient098']
```

15 of the 20 real validation patients are labelled `train` by the gate.

**This is not train/val leakage.** The DataLoaders are built by `build_patient_dataset` (`_utils.py:270-288`), which passes `split_manifest` straight to `ACDCDataset`; those get the correct 80/20 manifest split. The defect is that the printed patient/slice counts and the "splits validated" assurance describe a partition that is never used. Any paper table sourced from `phase_*_split_stats` would be wrong.

## 2. Counts per split

From `splits/acdc_patient_split_seed42.json` (measured, above):

| | patients | volumes (ED+ES) | slices |
|---|---|---|---|
| train | 80 | 160 | **not measurable here** |
| val | 20 | 40 | **not measurable here** |
| test | **0 — no test split exists** | 0 | 0 |
| total | 100 | 200 | — |

Slice counts require the `.npy` volumes, which are absent from this tree (`metadata.json` with `num_slices` is also absent). ACDC's public training set is 100 patients, so this manifest covers all of them.

**FINDING (high, for reporting): there is no held-out test set.** The manifest has no `test` key, and both `finetune_joint.py:816` and `scripts/train_self_audit.py:329,535` select the *best checkpoint* by the validation metric (`metric = float(validation["final_foreground_macro_dice"])`, then `best_metric`). The 20-patient val set is therefore a **development set used for model selection and then reported as the result**. That is an optimistic bias on top of everything in §9.

## 3. 2.5-D construction

**Clean. No cross-patient or cross-phase contamination is possible.**

- Depth axis: configs pin `depth_axis: 2` (`configs/self_audit_annotation.yaml:10`, `self_audit_auditor.yaml:10`, `self_audit_joint.yaml:10`, `self_audit_acdc_to_mnms.yaml:22`), matching the `[H,W,Z]` layout written by `scripts/preprocess_acdc.py:75-80`. `to_depth_first` (`common.py:112-124`) then moves it to `[Z,H,W]` with no through-plane interpolation.
  - When `depth_axis` is *not* pinned, `infer_depth_axis` (`common.py:91-109`) guesses. Measured: `infer_depth_axis((224,224,10)) = 2` ✓, `(224,224,70) = 2` ✓ (falls through to `argmin(shape)`), `(10,224,224) = 0` ✓. The heuristic happens to be right for ACDC, but it is a heuristic; the configs correctly override it.
- Neighbor picking at volume boundaries: **clamp / replicate**, `build_25d_triplet` (`common.py:187-200`), line 199: `neighbors = (max(index-1, 0), index, min(index+1, n_slices-1))`. Identical logic in the eval path, `volume_inference.py:66-81` line 75-77.
- **Center slice is channel 1.** Measured on a volume whose slice *k* is filled with the value *k*:

```
center=0 -> channels [0, 0, 1]   (center value in channel 1 = 0)
center=1 -> channels [0, 1, 2]
center=2 -> channels [1, 2, 3]
center=4 -> channels [3, 4, 4]
build_25d_batch ch1 per row = [0, 1, 2, 3, 4]
```

- **Can a neighbor come from a different patient or a different phase? No.** `build_25d_triplet` receives a single `[Z,H,W]` array loaded from one `VolumeRecord` (`common.py:373-378`), and one record is exactly one `patientXXX_{ED|ES}.npy` file (`acdc.py:130-144`). There is no concatenation of volumes anywhere on the indexing path; `index_map` stores `(record_index, slice_index)` pairs (`common.py:324-337`) and never indexes across records. The supervision target is the center slice only (`common.py:377`).

## 4. Preprocessing

`scripts/preprocess_acdc.py`:

- **Resampling: none.** There is no resampling to a common physical spacing (no `zoom`, no `resample`, no target mm grid — repo-wide grep for `resample` returns only `preprocess_myops2020.py`). Slices are resized **in-plane to a fixed pixel grid** `(224, 224)` by default (`preprocess_acdc.py:99,103,75-80`), independent of each patient's actual mm/pixel. Through-plane (Z) is untouched (`preprocess_acdc.py:73` keeps `orig_spacing[2]`).
  - Consequence: each patient ends up at a *different* effective mm/pixel. `preprocess_acdc.py:71-73` computes and records this (`effective_spacing = [z, sy*H/224, sx*W/224]`) into `metadata.json` (`:82-86,141-148`), and `acdc.py:74-100` reads it back — but **nothing in the Dice path ever uses spacing** (see §7).
  - Also: because the resize forces a square `(224,224)` from whatever the native matrix is, a non-square native matrix is **aspect-distorted**. ACDC natively is typically non-square in-plane (e.g. 216×256), which would mean anisotropic distortion — I could not verify the native shapes because the raw data is absent, so treat this specific point as *likely but unverified*.
- **Resize/crop target:** resize only, `skimage.transform.resize(..., order=1, anti_aliasing=True)` for images and `order=0, anti_aliasing=False` for masks (`preprocess_acdc.py:79-80`). The dataset then resizes again at load time from 224 → the configured `image_size: 256` (`common.py:203-213` `resize_sample`; bilinear for image, **nearest** for mask). Measured: `resize_sample 224 -> (3,256,256) / (256,256), dtype int64, labels [0,1,2,3]` — label set preserved.
- **Intensity normalization: per-volume, twice.**
  1. `preprocess_acdc.py:19-25` `normalize_zscore` — 0.5/99.5 percentile clip then z-score over the **whole 3-D volume** (`img_nii.get_fdata()`), before resize.
  2. `common.py:144-160` `percentile_clip_and_zscore` — the *same* 0.5/99.5 clip + z-score applied **again** over the whole `[Z,H,W]` volume at load time (`common.py:361`), and again in the eval path (`volume_inference.py:53-63,183`).
  Double-normalizing is idempotent-ish for a z-score and is not leakage, but it is redundant and means the documented "locked" normalization is applied twice with the second one operating on already-normalized data.
  Neither is per-slice; both are per-volume. Neither uses the GT.
- **Does any crop use the GT mask to centre or bound the FOV? No — checked hard, and it clears.** There is no cropping of any kind in the ACDC path. Repo-wide grep for `crop|bbox|roi|centre` across `src/self_audit` and `scripts` returns only matplotlib `bbox_inches`/`bbox_to_anchor` in visualizers and unrelated `mask.nonzero(...)` row-selection helpers in `finetune_joint.py:113,170`, `audit_decomposition.py:61`, `counterfactual.py:122` — those select *batch rows*, not spatial regions. Full-FOV evaluation is retained. If anything this makes Dice *harder*, not easier.

## 5. Foreground-only behavior — what happens to empty slices at validation

**Empty slices are NOT dropped, from train or val.** But the metric then awards them a free 1.0, which is *worse* than dropping them.

- The filter exists — `common.py:334-337`: `if self.foreground_only and not np.any(mask[slice_index] > 0): continue`.
- It defaults to `False` (`common.py:304`, `acdc.py:215`, `mnms.py:161`) and **no config and no training script ever sets it**. Repo-wide grep for `foreground_only` returns only those definitions plus the pass-through at `acdc.py:235` / `mnms.py:182`. `build_patient_dataset` (`_utils.py:270-288`) never passes the key.
- Measured on synthetic ACDC-shaped data (Z=10, 3 slices with no foreground per volume):

```
train: volumes=6 slices=60 (=6*10) fully-empty-GT slices kept=18
val:   volumes=2 slices=20 (=2*10) fully-empty-GT slices kept=6
val with foreground_only=True: slices = 14 (dropped 6 empty slices)
```

**So: for val, every slice of every volume is evaluated, including slices with zero foreground. That part is honest.** The problem is what the metric does with them — see §9.1.

## 6. Augmentation

Transforms are `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotate90` (p=0.5 each), composed as `RandomGeometricTransform` (`src/self_audit/data/transforms.py:36-80`). All are applied jointly to the 3-channel image and the center mask via `_transform_pair` (`transforms.py:11-23`). No intensity augmentation, no elastic/affine.

**Nothing is active at validation.** `common.py:393-398` only augments when `self.augment` is set; `build_patient_dataset` sets `"augment": bool(train and config.get("augment", True))` (`_utils.py:283`) and the val dataset is always constructed with `train=False` (`scripts/train_self_audit.py:147-151`). Measured on the real dataset object: `train: augment_flag=True`, `val: augment_flag=False, transform=None`. ✓

(Note `RandomRotate90` on cardiac short-axis is an unusual choice — it produces orientations that never occur in the modality — but it is train-only and does not affect the reported metric.)

## 7. Volume reconstruction — **this is the core comparability break**

**Nothing in the training or evaluation pipeline computes a volume-level Dice at original spacing/shape. Every reported Dice is a 2-D slice-grid Dice at 256×256, against a *degraded* ground truth.**

Three separate facts:

1. **No path resamples predictions back to the native ACDC grid.** Grep for `nib.save|affine|orig_shape|resample` across `src/self_audit` and `scripts` matches only `preprocess_myops2020.py`. The original ACDC shape is recorded at `preprocess_acdc.py:83` and never read again in the ACDC pipeline.
2. **The one function that does 3-D stitching resizes the GT down to the network grid, not the other way round.** `evaluate_comparison_modes` (`volume_inference.py:236-320`) at lines 274-281 pushes the GT through `F.interpolate(..., size=(image_size, image_size), mode="nearest")` and then scores every mode against `resized_ground_truth` at lines 316-319. So even the "volume" path is a stack of 256×256 slices compared to a nearest-resampled 256×256 GT.
3. **That function is dead code for reporting anyway.** Grep for `evaluate_comparison_modes` outside its own module returns only the `__init__.py` re-export; `infer_patient_volume` is called only by `scripts/visualize_predictions.py:96`. No training or evaluation entry point calls either.

What the headline numbers actually are:

| Reported metric | Definition site | Granularity | Grid |
|---|---|---|---|
| `val_macro_foreground_dice` (Phase A) | `train_annotation.py:242,253` via `per_class_dice` (`metrics.py:40-59`) | pooled over the 4 slices in one **batch**, averaged over batches | 256×256 |
| `final_foreground_macro_dice` (Phase C) | `finetune_joint.py:286-287,464-466` via `multiclass_dice` (`audit/targets.py:73-87`) | strictly **per 2-D slice**, then meaned | 256×256 |
| `modes/{initial,always_accept,self_audit,oracle}_dice` | `audit_decomposition.py:253-254,296-314` via the same `multiclass_dice` | per 2-D slice | 256×256 |
| `phase_a/a0_dice`, `phase_a/final_dice` | `audit_decomposition.py:472-476` via the same `multiclass_dice` | per 2-D slice | 256×256 |

Published ACDC numbers (~0.90–0.92 mean over RV/MYO/LV) are per-**volume**, per-class Dice at native resolution, averaged over patients. **The two are not the same quantity**, and the gap is not a wash — it is systematically in the project's favour (§9).

The GT the model is scored against has been through 216×256(native) → 224×224 nearest → 256×256 nearest. Any error the model makes at a scale finer than the 224 grid is *invisible* to this metric.

## 8. Pretrained ConvNeXt

- `pretrained_encoder: true` in `configs/self_audit_annotation.yaml:34` and `configs/self_audit_acdc_to_mnms.yaml:26`. It is `false` in `self_audit_auditor.yaml:34` and `self_audit_joint.yaml:38`.
- **Those two `false`s are dead in the real run.** `scripts/train_self_audit.py:242` builds a **single** model from `config_a` and reuses that same instance for Phases B and C (`train_self_audit.py:243` prints "Full Pipeline: Single Shared Model"). So ImageNet ConvNeXt-Tiny weights are live throughout.
- The weights are genuinely ImageNet: `src/self_audit/models/encoder.py:90-102` calls `timm.create_model("convnext_tiny", features_only=True, pretrained=True, in_chans=3)`, and `encoder.py:103-108` **raises** rather than silently falling back when `pretrained=True` and timm/weights are unavailable. So a run that reports pretrained=True really did load them.
- ImageNet pretraining is legitimate and widely used on ACDC. It just has to be **stated** in any results table, together with the fact that the 2.5-D triplet is fed into 3 ImageNet RGB channels.

## 9. Ranked protocol reasons the reported Dice can exceed published ACDC SOTA

Ranked by expected magnitude of inflation. All directions are **upward** (inflating).

### 9.1 — `empty_score = 1.0` on absent classes, applied per slice. Direction: ↑. Magnitude: **large, plausibly +0.04 to +0.10 absolute.** *(dominant cause)*

Both Dice implementations return **1.0** for a class whose union of prediction and GT is empty:
- `metrics.py:45,58`: `empty_score: float = 1.0`; `scores[cls] = empty_score if denom == 0 else ...`
- `audit/targets.py:81-85`: `torch.where(denominator == 0, torch.ones_like(...), ...)` — and this one is computed **per sample** (`flatten(1)`, i.e. per 2-D slice).

Measured:

```
per_class_dice(all-background pred, all-background gt) = {1: 1.0, 2: 1.0, 3: 1.0}
dice_score(all-background, all-background)             = 1.0
multiclass_dice(all-background, all-background)        = 1.0
```

And, decisively, on a slice where the model is **completely wrong** about the only structure present:

```
slice with only RV present, RV predicted with ZERO overlap -> {1: 0.0, 2: 1.0, 3: 1.0}, mean = 0.667
```

A total failure on that slice scores 0.667 because MYO and LV were absent from both prediction and GT.

This compounds with §5: empty slices are *kept* in val (good intent) and then *scored 1.0* (bad outcome). Keeping them is strictly worse than dropping them under this metric.

Simulated magnitude, ACDC-like Z=10 volumes, true per-structure Dice fixed at 0.88:

```
10 of 10 slices contain foreground ->  0.8800   (no inflation)
 8 of 10                           ->  0.9040   (+0.024)
 6 of 10                           ->  0.9280   (+0.048)
 4 of 10                           ->  0.9520   (+0.072)
```

This models only *fully* empty slices. In real ACDC the RV also disappears in apical slices while LV/MYO remain, so the per-class freebie fires more often than the fully-empty count suggests, and the real inflation is above these numbers. **A slice-level mean of ~0.97 under this metric is entirely consistent with a volume-level Dice in the mid-0.80s.** I cannot pin the exact number without the data.

### 9.2 — Slice-level (and batch-level) averaging instead of volume-level. Direction: ↑. Magnitude: moderate, +0.01 to +0.03.

Phase C averages per-slice Dice (`audit/targets.py:87` `.mean(dim=1)` per sample, then `finetune_joint.py:464-466` means over samples). Phase A pools over a batch of 4 consecutive slices (`train_annotation.py:242`, `batch_size: 4`, val loader `shuffle=False`). Published ACDC Dice pools over the whole volume before dividing. Slice-wise means over-weight easy mid-ventricular slices and let a single catastrophic apical slice be diluted rather than dragging the volume's TP/FP/FN ledger.

### 9.3 — Dice measured on a resized 256×256 grid against a twice-resampled GT. Direction: ↑. Magnitude: small–moderate, +0.005 to +0.02.

See §7. The GT went native → 224 nearest → 256 nearest before being used as truth. Sub-224-grid boundary error is unmeasurable, and boundary voxels are exactly where Dice is lost on cardiac structures. Also, no resampling to a common mm spacing means each patient contributes pixels of different physical size, so the "mean Dice" is not a physically-weighted quantity.

### 9.4 — No held-out test set; the reported val set is the model-selection set. Direction: ↑. Magnitude: +0.005 to +0.02.

`finetune_joint.py:816` / `train_self_audit.py:329,535` pick the best checkpoint by the same val metric that is then reported, over 100 (Phase A) + 20 + 10 epochs. With only 20 val patients, best-epoch selection on that set is a real optimistic bias.

### 9.5 — Small validation set (20 patients / 40 volumes). Direction: variance, not bias. Magnitude: ±0.02 on the mean.

Not inflation per se, but 20 patients makes the reported mean noisy enough that a favourable seed is worth ~0.02.

### 9.6 — Full-FOV evaluation with no ROI crop. Direction: ↓ (deflating).

Listed for fairness: no GT-guided or predicted-ROI cropping exists (§4), so background false positives *do* count against the model. This is one place where the protocol is stricter than some published pipelines. It does not offset 9.1.

### 9.7 — ImageNet ConvNeXt-Tiny pretraining. Direction: ↑, but legitimate. Magnitude: real but fair.

§8. Must be reported, not discounted.

**Summary judgment:** items 9.1 + 9.2 + 9.3 together can plausibly account for the entire gap between a ~0.97 reported figure and a ~0.88–0.90 volume-level equivalent. The reported number should not be read as evidence of a modelling win until it is recomputed volume-wise.

## 10. FINAL CLASSIFICATION

### **Internal-validation-only.**

Not "officially comparable", and not even "partially comparable".

Justification, in order of weight:

1. **The metric is not the ACDC metric.** Published ACDC Dice is per-volume, per-class, at native geometry, with absent structures handled by exclusion or by counting them as a miss — not by awarding 1.0. This repo's Dice awards 1.0 to any class absent from both prediction and GT (`metrics.py:45,58`; `audit/targets.py:81-85`), and does so at the granularity of a single 2-D slice. A slice on which the model is totally wrong can still score 0.667 (measured). Any number produced this way is on a different scale from the literature.
2. **No volume-level number is produced at all.** The only 3-D stitching function is uncalled by any training or evaluation entry point, and even it scores against a downsampled GT (`volume_inference.py:274-281,316-319`).
3. **No native-geometry evaluation.** Predictions are never resampled back; the GT is resampled twice before being used as truth; spacing is recorded but never used.
4. **No held-out test set.** The reported set is the checkpoint-selection set (§2).
5. Points *in favour* of the protocol, which is why this is a metric problem and not an integrity problem: the patient split is genuinely patient-level with ED/ES kept together and zero overlap (proved in §1); no GT-guided cropping exists anywhere (§4); empty slices are not silently dropped from val (§5); augmentation is train-only (§6); the 2.5-D triplet cannot cross patient or phase boundaries (§3).

**No SOTA claim of any kind is supportable from the current numbers.** The internal comparisons the project actually cares about — `initial_only` vs `always_accept` vs `self_audit` vs `oracle_accept` — are all computed with the *same* biased metric, so their *deltas* remain meaningful as a relative signal, even though their absolute levels do not. That is the honest framing available today.

### Minimum changes required to reach "officially comparable" (not implemented — audit is read-only)

1. Score per **volume**, not per slice/batch.
2. Resample the predicted label volume back to each case's native shape and spacing before scoring; score against the **original** `_gt.nii.gz`, never a resampled copy.
3. Stop awarding 1.0 for absent classes at scoring time — on a whole ACDC volume, RV/MYO/LV are always present, so the degenerate case simply should not arise; if it does, it must be excluded from the mean, not counted as a success.
4. Add a real held-out test split and report on it, keeping val for checkpoint selection.
5. Fix `_utils.py:369-379` so the preflight gate validates the manifest actually in use.
6. Report ImageNet pretraining explicitly.

---

## Uncertainty register

| Claim | Status |
|---|---|
| Patient-level split, no ED/ES straddle, no overlap | **Verified executably** against the checked-in manifest |
| Center slice in channel 1, clamp at boundaries, no cross-patient/phase neighbors | **Verified executably** |
| Empty slices kept in train and val; `foreground_only` never enabled | **Verified executably** + grep |
| Augmentation off at val | **Verified executably** |
| `empty_score = 1.0` behavior, including the 0.667-on-total-failure case | **Verified executably** |
| No GT-guided crop anywhere | **Verified** by repo-wide grep; nothing found |
| No volume-level / native-geometry Dice anywhere | **Verified** by grep on all call sites |
| Preflight gate falls back and diverges from the real split | **Verified executably** by replaying `_utils.py:369-379` on the real manifest |
| Slice counts per split | **NOT measurable** — `preprocessed_data/ACDC` absent |
| Actual fraction of empty / RV-absent slices in ACDC val | **NOT measurable** — data absent; §9.1 magnitudes are simulated, not measured |
| Aspect-ratio distortion from the square 224×224 resize | **Likely but unverified** — depends on native ACDC matrix shapes, which are not in this tree |
| Whether the ~0.97 figure specifically came from Phase A or Phase C | **Unknown** — no results file exists in the repo (`reports/` contains only `test_pipeline/calibration_result.json`); both validators share the same bias |
| `tests/test_self_audit_data.py` passes | **Not run** — `pytest` not installed in the available interpreter |
