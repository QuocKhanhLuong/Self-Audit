Completed world_left_v2 ACDC run, 14 October 2026

Start with summary.json, metrics.csv and comparison.csv.
run_evidence/ contains every unique text log, runtime receipt, configuration,
stage report, epoch history, frozen manifest, evaluation count and W&B text export
from the completed run. Symlink aliases are listed in PUBLICATION.json.
baseline_forensics/ contains the mathematical probes, 1-patient and 80-patient
TRAIN checkpoint audits, VAL count recomputation and the 120-test verification.
executed_source/ preserves the exact compact scientific source, entrypoints and
122-test suite from the executed snapshot; it is evidence, not a main-code merge.
The base source commit is 7d1c6a04064cda473d1eccfbed818392c37b1a37.
run_evidence/launcher_integration.patch records the local launcher/logging wiring.

Logs and reports have machine paths, host/email and GPU UUIDs replaced with
documented placeholders. Numeric measurements and original artifact hashes are
preserved. MANIFEST.json records original and published hashes for every copied
file. Sanitized FROZEN.json files are report copies, not runnable frozen artifacts;
their embedded hashes describe the original artifacts retained on the machine.

Dataset images, reference masks, predicted arrays, model weights, source tarball,
W&B binary event stores and runtime caches are not included. The complete text
training log and W&B text exports are included. No credentials are included.
Official TEST is unlabeled and was exported only, never scored.

Critical metric caveat: new RV precision is undefined because no RV was predicted.
The reported new macro precision averages MYO/LV only; the old macro includes RV.
The increase in this macro cannot be claimed as a like-for-like precision gain.
Foreground Dice remains very low (5.069%), with 99.633% UNKNOWN coverage.

Test-source copies use .py.txt to avoid automatic collection by repository pytest.
Remove only the trailing .txt when reconstructing the isolated executed test tree.

CSV line endings are normalized to LF without changing values. The W&B core
debug log is included by dereferencing its runtime-cache symlink.
