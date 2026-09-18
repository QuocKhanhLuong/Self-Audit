# Preflight corrections, before training or GT evaluation

Initial unit run: 9 passed, 2 failed. Firewall incorrectly rejected Torch source
`_mask_buffer.py`; corrected to distinguish executable source from data. Geometry
test compared NIfTI float32-stored affine to an unstored float64 literal; now it
checks exact equality to the actual input NIfTI affine. No scientific setting changed.

First preparation attempted `.nii.gz` only but local images are `.nii`, so it
rejected all missing patients before opening any image. Add both NIfTI encodings;
failed cache retained at /tmp/astra_nogt_cache224, new attempt uses cache224_v2.

PROVENANCE LIMITATION discovered by reading prior `evidence/copy_acdc.py`: the older
copy script enumerated paired images through `_gt` filenames. Thus this is a fixed
provided ED/ES image cohort, not a freshly certified image-only acquisition process.
Current preparation reads only images and never uses masks/crops. Independently
cross-check all available images against Info.cfg ED/ES phase metadata, but do not
erase the historical mask-presence-dependent copy rule. Any strict whole-pipeline
no-GT/cohort claim remains PROVISIONAL pending independent original-release image-only
re-ingestion. The paper must disclose expert ED/ES phase availability and this history.

After all method runs froze, the added evaluator-boundary test exposed missing optional
`sklearn`. The first evaluation process failed during import, before opening any GT.
Replaced that dependency with the standard contingency-table ARI formula, checked on
perfect, permuted and crossed partitions. No model/config/prediction/selection changed.
