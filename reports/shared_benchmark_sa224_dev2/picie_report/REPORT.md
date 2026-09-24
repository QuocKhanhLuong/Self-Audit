# PICIE Shared-Benchmark Semantic Audit

## Run Scope
- Artifact root: `reports/shared_benchmark_sa224_dev2/picie_artifacts`
- Samples: `20`
- Volumes: `2`

## Summary
- Slice metric contract: `foreground_dice_exclude_v1`
- Slice metric space: `slice_proxy`
- Volume metric contract: `foreground_dice_volume_resized_v1`
- Volume metric space: `volume_resized`
- Mean adapter coverage: `0.695230787627551`
- `final_foreground_macro_dice`: `0.0`

## Volume Highlights
- Best `foreground_dice_volume_resized_v1`: `patient004_ED` -> `0.0`
- Worst `foreground_dice_volume_resized_v1`: `patient004_ED` -> `0.0`

## Visual Outputs
- `summary_metrics.png`
- `slice_extremes.png`
- one `*_montage.png` per volume

## Notes
- Headline metrics follow the Self-Audit canonical foreground Dice contracts on the resized grid.
- Anonymous raw clusters are retained only as debug metadata; headline visuals focus on semantic cardiac output.
- The scientific runner itself remains GT-free; GT is used only in this downstream audit report.
- VOID pixels stay outside {RV, MYO, LV}; under `foreground_dice_exclude_v1`, both-empty classes are excluded and one-sided empties score 0.0.
