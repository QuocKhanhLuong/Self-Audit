# Initial CI findings and targeted repairs

Initial cleaned commit 70a490cbcb39ca9a1a715bee66e49980224e85e1 ran the entire
suite in GitHub Actions run 39027206647, job 117380440440: 150 passed, 3 failed.
The layout/model-identity checks passed. The failures were:

1. A regression test reads a published original bank FROZEN.json. That single
   metadata file is retained byte-for-byte at its original path; the rest of
   the old run bundle stays archived. No production code depends on old trainers.
2. The compatibility launcher's help lost the text '32-update pilot'. Restore
   that help text without restoring a separate parser or trainer.
3. A cross-batch-size continuous-confidence assertion was narrower than observed
   CPU roundoff (maximum absolute mismatch 7.1525574e-7; relative 1.77e-6).
   Set only that cross-batch continuous-value comparison to rtol=5e-6, atol=1e-6.
   Same-input checkpoint logits and discrete dense/gated labels remain exactly
   equal; all original differential-reference tolerances remain exactly zero.
   Model code, gates, optimization and checkpoint hashes are unchanged.

These are disclosed fixture/help/numerical-test changes, not new segmentation
results. The new CI run must be inspected before claiming all tests pass.
