# Verification receipts

All runs here are local CPU software/manufactured-input checks. No patient
arrays, pretrained backbone weights, GPU or evaluation references were used.

| Receipt | What actually ran |
| --- | --- |
| `../PARITY_REPORT.json` | Final-tree experimental verifier: 88 tests, zero failures/errors/skips; source/reference hashes and exact argv |
| `TEST_RESULTS.xml`, `tests.txt` | Final experimental pytest result, including the unchanged 47 reference tests |
| `FULL_MODEL_CHECK.json`, `full_model.txt` | Random full-size torchvision ConvNeXt-Tiny forward/backward, 32,890,558 parameters |
| `PROBE_RESULTS.json`, `probes.txt` | Gradient, decomposition, guard and image-energy probes |
| `REFERENCE_RECEIPT.json`, `REFERENCE_TEST_RESULTS.xml` | Original archive materialization/hash verification and 47-test run |
| `EXISTING_REGRESSION.json`, `EXISTING_REGRESSION.xml` | 454 existing core/pseudolabel/guidance tests; one existing requires-grad tensor-to-scalar warning |
| `PRESERVATION.json` | 236 pre-existing dirty/untracked files still byte-identical; protected source/baseline/config/split/checkpoint paths unchanged |

Commands use fresh temporary output directories. The JSON and XML are copied
without rewriting their original timestamps, argv, paths or runtime values;
log suffixes alone are changed from `.log` to `.txt` for tracked delivery. The
checkout SHA in the parity receipt is the reference parent before the delivery
commit, while the `implementation_sha256` inventory identifies the exact tested
new code. The commit containing these receipts is the delivery snapshot; the
receipts do not claim that the experiment was present in the parent commit.

The earlier main cleanup is a separate delivery at
`513ddda8cec3a166f13cfe68fe6379cab014a1a7`, with 466 tests passing on its own
tree. Its 12 added tracker tests are absent from this reference-based worktree,
which explains the 454-test existing-pipeline count here. No history is replaced.
