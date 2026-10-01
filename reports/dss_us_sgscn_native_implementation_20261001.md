# DSS-US / SGSCN native implementation milestone

Scope: frozen Revision 4, native protocol locks and verification groundwork only.
No ACDC adaptation, native GT evaluation, dataset download, or clinical run.

## Implemented

- Exact reference pins: DSS-US d4ac44c60df18b921c590796f6994a4c8ac0726c;
  SGSCN 592efb6e72ceeef15c8be0630a4673eda5dce6f5;
  licensed DINO 7c446df5b9f45747937fb0d72314eb9f7b66930a.
- Protocol matrix: 11 DSS-US CAMUS rows and distinct SGSCN PH2/SYSU-US profiles.
  Canonical indices prevent silent field/gate edits; configs bind the canonical
  native evaluator spec hash before any GT-assisted execution.
- Isolated GPL official-derived SGSCN producer: original architecture, context
  arithmetic, optimizer/update/stop/final-forward preserved; device handling,
  seed/provenance instrumentation and integer raw persistence added.
- Independent DSS-US equation kernels and licensed DINO feature interface.
- Method-agnostic image allowlists, access logs, seal/inventory/tamper checks,
  repeat diagnostics and independent native statuses.
- Native evaluator gates separate DSS-US Step I/II and retain SGSCN max-overlap.
  No guessed complete paper evaluator is substituted.

## Local validation

Separate private environments, Python 3.12 / Torch 2.11.0+cpu / NumPy 1.26.4.
Local venvs inherit existing Torch; receipts capture actual versions.
SGSCN architecture/loss/gradient parity is checked against extracted official
functions entirely inside the isolated GPL tests.
Full reference settings (50-iteration cap, enabled context) are used for synthetic
CLI smoke. Raw predictions seal; same-seed fresh-process repeats are identical.
Per-run seals differ because timing/access metadata are run-specific, while raw
array hashes and scientific inputs match.

Initial infrastructure failures (pytest global temp permissions, lazy Torch
cache probing and caller-source stat access) were corrected by dedicated test
temp roots, pre-guard framework initialization and explicit source allowlists.
No scientific settings were changed to pass a test. Failed smoke roots remain
in .runtime and are not sealed/committed as successful runs.

Final local checks:

- `tests/native_baselines` + `baseline/SGSCN/tests`: **29 passed**.
- `baseline/DSS_US/tests`: **8 passed**. The licensed DINO architecture is loaded
  with a strict random-weight checkpoint fixture; this does not validate a real
  pretrained checkpoint or CAMUS inference.
- Fresh-process SGSCN CLI repeats use full official-reference settings, seal both
  runs and match every raw integer pixel. Receipts:
  `dss_us_sgscn_synthetic_repeat_20261001.json` (synthetic input only).
- Existing `tests/shared_benchmark`: **84 passed, 2 failed**. Both failures were
  repeated unchanged on detached base commit `a107b734` without this implementation:
  Windows checkout line endings change an existing frozen-spec file SHA; an
  existing adapter-v3 execution fixture violates the current 224 grid contract.
  No old adapter, fixture, freeze hash or scientific configuration was modified.
- The isolated GPL reference snapshot is preserved as exact upstream Git blob
  bytes using scoped `.gitattributes`; receipt hashes are checked in tests.
  Its original trailing whitespace is preserved. Whitespace checks cover new
  implementation files separately from that immutable reference snapshot.

The new implementation checks pass. The two pre-existing regression failures
remain explicitly recorded, rather than being represented as a fully green suite.

Commands used (Python executables inside the separate local venvs):

~~~
python -m pytest tests/native_baselines baseline/SGSCN/tests -q -p no:cacheprovider --basetemp .runtime/native_sgscn_qa_lockbinding1 --tb=short
python -m pytest baseline/DSS_US/tests -q -p no:cacheprovider --basetemp .runtime/native_dss_qa_lockbinding1 --tb=short
python scripts/verify_native_raw.py .runtime/sgscn_cli_smoke/raw_final_a --repeat .runtime/sgscn_cli_smoke/raw_final_b
~~~

## Explicit dependencies / remaining native work

DSS-US: full row recipes/inventory/CRF and embedding correspondence unresolved;
real DINO checkpoint unavailable locally; end-to-end native orchestration not
yet enabled. Step I evaluator pairing/aggregation unresolved; unequal-count
Step II helper absent in pinned source. Full native execution BLOCKED_PROTOCOL.

SGSCN: paper architecture/stopping differs from demo; full paper evaluator
HM/XOR/tie definitions and original SYSU sampled inventory unresolved. Reference
code is executable, but full paper reproduction remains BLOCKED_PROTOCOL.

CAMUS, full PH2 and SYSU-US inventories are not supplied locally: BLOCKED_DATA.
Native Track A compatibility remains BLOCKED_ADAPTER independently.
No NATIVE_REPRODUCTION_STATUS or NATIVE_TRACK_B_STATUS is claimed COMPLETE.

Resolving these dependencies is required before paper-profile reproduction.
This is a verified implementation milestone, not completed native reproduction.
The frozen ACDC raw_id_majority_vote_v1 rule is unchanged and not implemented
early; no ACDC adaptation starts at this milestone.

## Publication / deployment boundary

Publish this as native protocol groundwork with a verified SGSCN reference
producer and independent DSS-US primitives, not a completed paper reproduction.
Server checkout/environment preparation follows a verified successful push.
No server training or native dataset execution is authorized by a synthetic
smoke receipt; unresolved profiles retain their execution refusals.
