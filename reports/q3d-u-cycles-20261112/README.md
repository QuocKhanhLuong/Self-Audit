# Q3D U cycles and clean test runs — 2026-11-12

Evidence: **SOFTWARE_ONLY**. Base: `aa4b09a28e6d5776ee3b548c371f452af3ec6a21`.

## Delivered

1. The A-3D adapter test writes its receipt to `tmp_path / baseline_test_adapter.json`. The committed receipt is historical and byte-for-byte unchanged.
2. `test_q3d_repository_clean.py` snapshots current tracked working source into a fresh local Git repository, commits that snapshot locally, runs the complete pytest suite, and asserts exact empty `git status --porcelain`. The child skips only its own launcher to prevent recursion; all remaining tests run. All three new tests were in the Git index before acceptance, so both copied suites include them. No user untracked outputs are copied or deleted.
3. U sampling visits each patient exactly once per shuffled cycle (`seed + 1`), with a separate random frame stream (`seed + 2`). The L patient/frame random stream is unchanged, including C0. Tests cover shuffled order, V1/V1-u patient order, L compatibility, cursor/permutation validation, boundary and mid-cycle resume.
4. Sampler and research checkpoint schemas are v3. Engine v2 resume fails explicitly; schema consumers and fixtures agree. Runtime consumers share `CHECKPOINT_SCHEMA` where practical.
5. Engine cycle logs contain `realized_mean_omega`, max and zero fraction computed from consumed weights. Observations persist across checkpoint reload. Complete V1/C1 cycles with valid gates enforce mean 1 to absolute tolerance 1e-12. The engine-level tests run actual J updates and checkpoint round-trips for V1, V1-u and C1.
6. Removed the old with-replacement U stream, set-coverage cycle tracker, and partial-cycle reset on O1 cache refresh. Mixed O1 snapshots retain their observations and report their actual mean and snapshot identities.

## Verification

Both complete-suite invocations unset `OMP_NUM_THREADS`; one unsets `Q3D_OFFICIAL_MEDNEXT_SOURCE`, the other sets it to `/tmp/q3d-mednext-phase-pin` at official source commit `0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba`. This is source-only verification; no weights or patient data were downloaded. Raw stdout/stderr is retained without rewriting in `pytest-default.txt` and `pytest-mednext.txt`. Each parent suite runs a second full suite in its clean Git copy.

`check-repo.json` contains the PASS output. The separate synthetic smoke completed T/Q/P/J using the explicit synthetic tokenizer-gate override: the measured gate remains NOT_READY, not a real-data gate pass. `smoke-u-cycle-events.json` records two complete U cycles, each with realized mean 1.0. `source-inventory.json` pins changed source bytes and confirms frozen baseline paths and the historical receipt are unchanged.

The original workspace already contained unrelated untracked research artifacts and .DS_Store files. They were preserved. Both suite receipts compare the workspace porcelain before/after; the clean-copy receipts independently record empty porcelain. This does not claim the original workspace's pre-existing untracked files disappeared, or that ignored cache files are prohibited.

## Optional P/Q decoupling

**NOT_RUN (optional change not implemented).** Physiology scores use teacher region outputs, not codebook tensor values. The existing P artifact/cache contract nevertheless reads and verifies the Q/C0 SHA for provenance (`stages.py` import/export and `physiology.py` load/refresh). This delivery preserves that dependency and historical PLAN; removing it would require a separate cache/artifact identity migration. No claim is made that physiology mathematically depends on C0.

## Independent review and disposition

Luna workers implemented the sampler and test isolation through Orca run `run_5e2fb09f0ab3`; the coordinator integrated engine logging, schema consumers and acceptance. Claude independently reviewed the diff read-only and ran 14 targeted tests: no blocking findings. The unchanged review is `claude-review.md`.

- M1: clean-copy acceptance intentionally uses the requested porcelain contract. It does not assert absence of ignored build/cache files.
- M2: resolved before acceptance: new tests were added to the index and both copy receipts confirm all are present.
- L1: O1 is excluded from the frozen-cohort runtime mean assertion; the requested V1/C1 assertion is enforced. O1 means and snapshots remain observable. A dedicated engine-level O1 refresh/resume regression is NOT_RUN; the existing O1 engine checks and new mixed-snapshot accumulator test passed.
- L2: synthetic override and gate status are explicit in separate run gate/weighting events; full-cycle rows refer to the same snapshots. They do not duplicate those flags.
- L3: the production J path requires the cohort metadata before any U step; a missing map cannot reach its cycle assertion. The direct `_new_patient_omega` probe fallback is not a full engine execution.
- I1: the old receipt is historical evidence only and was not used as proof of current source.

## NOT_RUN

Real-data P0–P5 acceptance, patient training/evaluation, GPU, clinical efficacy, campaign execution, optional P/Q identity migration, and SOTA/novelty evidence. Synthetic smoke and CPU tests establish software behavior only.

## Exact pytest output

### Default environment

```text
........................................................................ [ 15%]
........................................................................ [ 31%]
........................................................................ [ 46%]
.........s.............................................................. [ 62%]
........................................................................ [ 77%]
..........s............................................................. [ 93%]
...............................                                          [100%]
=============================== warnings summary ===============================
tests/test_q3d_tokenizer.py::test_decoder_stride8_ablation_hooks_final_child_once
  /Users/alvinluong/miniforge3/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py:1548: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. Starting in PyTorch 2.9, calling checkpoint without use_reentrant will raise an exception. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
    return fn(*args, **kwargs)

tests/test_q3d_tokenizer.py::test_decoder_stride8_ablation_hooks_final_child_once
  /Users/alvinluong/miniforge3/lib/python3.11/site-packages/torch/utils/checkpoint.py:242: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
    check_backward_validity(args)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
461 passed, 2 skipped, 2 warnings in 261.46s (0:04:21)
```

### Official MedNeXt source configured

```text
........................................................................ [ 15%]
........................................................................ [ 31%]
........................................................................ [ 46%]
........................................................................ [ 62%]
........................................................................ [ 77%]
........................................................................ [ 93%]
...............................                                          [100%]
=============================== warnings summary ===============================
tests/test_q3d_alignment.py::test_actual_pinned_mednext_feature_phases_and_source_modules
tests/test_q3d_official_channels.py::test_c1_feature_head_and_loss_match_bounded_official_source_forward
tests/test_q3d_tokenizer.py::test_decoder_stride8_ablation_hooks_final_child_once
  /Users/alvinluong/miniforge3/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py:1548: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. Starting in PyTorch 2.9, calling checkpoint without use_reentrant will raise an exception. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
    return fn(*args, **kwargs)

tests/test_q3d_alignment.py::test_actual_pinned_mednext_feature_phases_and_source_modules
tests/test_q3d_official_channels.py::test_c1_feature_head_and_loss_match_bounded_official_source_forward
tests/test_q3d_tokenizer.py::test_decoder_stride8_ablation_hooks_final_child_once
  /Users/alvinluong/miniforge3/lib/python3.11/site-packages/torch/utils/checkpoint.py:242: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
    check_backward_validity(args)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
463 passed, 6 warnings in 273.21s (0:04:33)
```
