# W3 latest Astra gate: REVISE

Read this once as consolidated feedback; supersedes repetitive inbox review reminders. Current metadata snapshot guard + independent summary/finish tries are improvements. Complete remaining fixes, then run only test_trainer_lifecycle.py and deliver. Do not broaden suites.

1. Failure cleanup belongs in a finally independent of payload preparation. Current outer try catches metadata errors but can still skip finish if e.g. str(exc) raises. Primary error must propagate; minimal fallback reason is fine.
2. Preserve strict config_signature property during normal execution; do not weaken provenance by catching everything and returning None globally. Catch its failures only while preparing failure metadata.
3. Do not overwrite unrelated failure.json on init/output ownership refusal; preserve existing bytes and write a uniquely named attempt artifact when canonical name belongs elsewhere. Apply same policy in CLI pre-init helper.
4. Add requested aliases global_epoch, git_sha, config_identity to failure artifacts (document zero-based current global_epoch vs completed_epochs); all are known on post-config init failure except unavailable values explicitly null.
5. Per-epoch report contains explicit last_checkpoint_committed_epoch, completed_epochs and status. A validation row can exist after checkpoint failure only with explicit validation_complete and checkpoint_committed=false. Once report+checkpoint commits succeed, increment completed_epochs BEFORE W&B adapter call.
6. Telemetry failure status must be captured AFTER summary/finish, in the returned and durable report. Current final report precedes them and can claim active when finish fails. Preserve nonfatal telemetry policy; do not hide it behind stderr only.
7. Requested diagnostics when calibration disabled must not silently appear successful. Preserve recipe; either execute independent diagnostics with configured fixed tau or explicitly report requested/executed/skipped reason as current skip behavior. Canonical calibration-enabled path must execute diagnostics before completed true.
8. Subprocess fixture: image_size32, explicit depth_axis and matching (32,32,2) axis2 input, inherit env and set timeout. No canonical config/model changes. Existing logging fixture size32 edits accepted for ConvNeXt minimum input.

Required focused evidence: existing W3 failure tests plus metadata preparation failure still finishes, summary exception still finishes, ownership refusal preserves old failure JSON, logger failure after durable report has correct committed counters, final telemetry failure report truthful. Root will review actual diff and use final integrated suite; no repeat full focused suite in multiple environments.
