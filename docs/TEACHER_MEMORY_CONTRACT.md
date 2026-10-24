# Teacher-memory pilot delivery contract

This task authorizes an isolated implementation on `feat/native-teacher-memory-pilot`, based on `9341cc81` (including the checked `0f54455`). Existing baseline, fixed-window, meta-window recipes, results and dirty work must remain intact. Only code, tests and short smoke/profile runs are authorized; long training and main-branch merging are outside this delivery.

The final implementation, formulas, controls and command reference are in [TEACHER_MEMORY_PILOT.md](TEACHER_MEMORY_PILOT.md). That document supersedes the provisional API sketch used to coordinate workers. The execution evidence is in the [delivery report](../reports/teacher_memory_pilot/REPORT.md).

## Invariants

- Canonical v2 `NativeStudent`, fixed window, full EMA, one existing optimizer timeline and one EMA update. No archived trainer, matching loss or meta loss.
- Exact pinned M&Ms SAX seed-0 segmentation teacher; strict loading, image-only physical tiling and inversion; no GT-driven crop or U/VAL/TEST mask read.
- Immutable TRAIN bank, at most 8 pairs per class pair, endpoint confidence >= 0.8, shared boundary within 3 mm, patient balance and explicit missing groups.
- A2/A3 share vector budget and trainable capacity. A4 breaks correspondence while preserving class marginals; A5 uses a frozen matched A0 exclusively; A6 is an explicitly random feature control.
- Student inference embeds the bank and requires no external teacher, per-image teacher cache or original bank file. Student backbone dependencies remain explicit.
- Report source, weight and bank hashes, actual tests, measured timing/VRAM and missing evidence. Synthetic checks do not establish patient Dice. Cross-dataset subject overlap and novelty remain unconfirmed.
- Luna implements; Astra independently reviews. Findings and failed intermediate checks remain in review receipts. Publication is only to the pilot branch after checks, without reset, clean or force-push.
