# W2.1 — authoritative effective split records (TASK SPEC, not implemented)

Execute only after Astra opens this task via Orca. AGY implements; Astra reviews.
Read execution_plan.md W2.1 and docs.md. No training, commits/push, manifest edits,
architecture/objective changes, or external-domain pooling. Preserve all other edits.

## Evidence and objective

Current `resolve_acdc_records` returns directory-tagged records before inspecting an
explicit manifest. `validate_dataset_splits` instead reads manifest or random patient
split, without using actual selected records. `read_split_manifest` deduplicates lists,
and discovery deduplicates by case ID before collision checks. Preflight patient001_ED
train / patient001_ES val fixture passed validator but leaked in actual loader.

## Ownership / implementation

Expected files: data/acdc.py, data/common.py, training/_utils.py, focused new tests.
Only necessary callers/exports elsewhere; no tracked split/config changes.

- One authoritative resolver produces all effective split record lists and a deterministic
  signature. Dataset and validator call it; no separate reimplementation in validator.
- Explicit manifest governs membership. Conflicting directory tags fail clearly rather
  than override the manifest. Without manifest, use complete explicit tags when available;
  mixed/ambiguous tags fail, fallback deterministic patient split only for untagged input.
- Requested train/validation splits must exist; absent optional test is allowed and reported.
  Respect configured aliases consistently; unsupported/ambiguous aliases fail.
- Validate all effective splits jointly: patient cross-split leakage including ED/ES,
  duplicate cases, manifest duplicates, unknown entries and discovered cases absent from
  manifest. Do not silently set-deduplicate these before validation.
- Discovery must distinguish a second traversal of the *same resolved image/mask pair*
  from two distinct files with the same case ID. Raw NIfTI root.rglob plus split roots
  currently traverse identical paths twice: dedup traversal identity safely, preserve
  most-specific consistent tag, but fail real collisions/conflicting ownership.
- Expose validator effective identities/signature so parity with dataset.records can be
  asserted for every configured split. Signature cannot depend on discovery ordering.
- Do not claim content-hash duplicate detection unless actually implemented/tested;
  it is optional and can be deferred explicitly.

## Tests and pass criteria

1. Controlled ED-train/ES-validation same patient fails before model training.
2. Validator effective records == actual ACDCDataset/DataLoader records in every split,
   with explicit manifest, tag-only and deterministic untagged fallback.
3. Missing/extra manifest entries, duplicate list entries and real duplicate files fail.
4. Manifest-vs-tag conflict and mixed tagged/untagged ambiguity fail clearly.
5. Optional test absent accepted; configured train/val missing rejected.
6. Discovery order does not change record identities or signature. Same raw-file recursive
   traversal is not falsely reported as a duplicate dataset case.
7. Focused tests -> full tests/ -> source compile; report exact commands/counts/output.

Keep diff small; no changes to pixel values, labels, augmentation or geometry in this task.
New concise worker report: wave2_split_worker.md in this directory. No claim of actual
raw-dataset or clinical checkpoint validation from controlled fixtures.
