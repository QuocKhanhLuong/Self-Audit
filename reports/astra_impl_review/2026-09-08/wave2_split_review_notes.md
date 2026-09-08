# Current W2.1 review notes — coordinator guidance

Task task_3945e271726f, dispatch ctx_3b7cbf0792c7. Continue same task.

Do NOT search ~/.orca, ~/.gemini or other provider/account/session directories for
orchestration messages. Do not read credentials/session internals. Repo work only.
If public Orca inbox cannot retrieve a message, use this file as the complete guidance;
do not reverse engineer runtime storage. No need to locate the original message IDs.

1. Independent probe read_split_manifest('splits/acdc_patient_split_seed42.json')
   failed: ValueError Manifest contains unrecognized split key: 'n_patients'.
   n_patients, num_patients, n_volumes, num_volumes are legitimate scalar summary metadata.
   Fix parser, DO NOT rewrite tracked manifest. Add tracked-file parse regression.
2. Validate every membership representation (train_cases/train_volumes/train/training/
   nested splits). Accept redundant identical sets if required; reject conflicting
   memberships, rather than first-key-wins. Preserve duplicate-entry checks within lists.
3. _paired_files dict by p.stem silently overwrites same-stem .npy/.npz. Detect collisions
   before creating dictionaries. No content hashing required.
4. validate_dataset_splits now returns raw VolumeRecord dataclasses in effective_records;
   public CLI/report values should be JSON-safe descriptors/identities, or all real callers
   must support serialization. Prefer descriptors; parity test compares exact paths/IDs.

Finish task tests + concise report + worker_done with current injected lifecycle.
No entropy/geometry/provenance edits, training, commit/push, or outside-repo investigation.
