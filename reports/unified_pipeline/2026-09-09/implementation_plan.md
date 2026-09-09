# Unified training continuation plan

## Verified starting point

Fetched origin/main on 2026-09-09: local HEAD and origin/main both
`f1dcae326dffad1cd7e2cf668425e83df4bae936` (M&Ms/OOM PR merged).
The working tree contains an unfinished unified implementation from the earlier
run; preserve it and all unrelated reports. No long training or publication.

The public shell/PowerShell runners and README still use three stages/configs.
The draft Python runner has one schedule, but still auto-dispatches legacy flags.
The intended curriculum remains epochs [0,100), [100,120), [120,130), carrying
live weights, with optimizer resets at 100/120. Strong pretrained ConvNeXt,
stop-gradient, reject-to-HALT and Proposal-1 provenance are fixed constraints.

## Existing local edits to reconcile through AGY

Before the explicit Astra/AGY assignment, the coordinator repaired dataclass
constants leaking into config roundtrips; forwarded preprocessing to factories,
mapping/augmentation through the unified adapter; rejected nonidentity ACDC
mapping; handled legacy equals-form flags; restricted external M&Ms evaluation
to test/testing; rejected fractional/nonfinite labels in mapping.apply.
These are unapproved draft edits, not an AGY delivery. AGY must inspect and test
them, including the fact that the common loader casts masks before mapping.
Copies run_full_pipeline_legacy.sh/.ps1 exist; canonical wrappers are unchanged.

## Small implementation waves and review gates

1. **Unified execution closure:** strict config roundtrip and runtime consumers,
   canonical one-config CLI and shell/PowerShell wrappers, explicit legacy-only
   entrypoints, actual LR telemetry, safe resume at ordinary and boundary epochs,
   preserved output artifacts, complete-run selection/calibration firewalls.
   Test differential loss/update semantics and resume RNG/cardinality/cohort.
2. **Dataset identities:** audit the merged M&Ms/OOM diff. Verify real factories,
   mapping/unknown labels before casts, shape/depth and patient identities.
   Keep external testing frozen. Support M&Ms-native train/val only with genuine
   patient-disjoint split validation and add a separate full config if verified.
   Do not merge datasets or let native settings leak into the external protocol.
3. **Proposal-1 downstream integration:** selected best.pt binding, complete
   validation calibration, diagnostics and frozen bank with one resolved config.
   Test best != last, provenance roundtrip and tamper failures. Update canonical
   docs/commands and preserve historical reproduction via explicit legacy paths.
4. **Final gate:** independently review combined diff, full suite, real subprocess
   synthetic ACDC unified smoke and frozen external M&Ms smoke. Record exact
   schedule, commands, test counts, unsupported hardware/data and recipe impact.
   Only after review approval commit relevant files and push main normally.

AGY owns production changes through Orca; reviewer owns this plan, read-only
review and independent tests. Do not substitute another implementation provider
if AGY is unavailable. One worker/wave in this dirty checkout avoids conflicts.

## Baseline evidence

Initial full run: 576 passed, 7 failed, 1 warning, 101.38 seconds. Six failures
were draft unified config/telemetry; one subprocess hit sandbox shared memory.
After local config fix, focused tests outside sandbox: 60 passed, 1 failed,
24.80 seconds. Remaining test_canonical_learning_rate_keys expects base LR
while the scheduler has already applied warmup; preserve real optimizer logging
and correct the expectation only after checking actual scheduler semantics.
This is not final integrated approval or scientific evidence.

## Orchestration status

Orca skills loaded. Initial status stale_bootstrap / app not running; first open
timed out. An escalated open is being attempted. No AGY dispatch exists yet.
