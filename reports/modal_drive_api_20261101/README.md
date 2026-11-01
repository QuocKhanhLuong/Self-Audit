# Direct Drive API attempt and delivery

Actual CPU Modal app: `ap-cprm3FlylRcbp5iDRNTXaT` (stopped, no GPU).
Volume receipt: `/assets/fresh_s0/acdc_api_20261101_r1/API_SMOKE.json`.
The unchanged downloaded copy is `API_SMOKE_401.json` with its hash in
`STATUS.json`. The granted access-only token returned HTTP401 `authError`;
there was no matching refresh client in the Secret. No rclone process ran.
All17 historical files and STAGING receipt hashes matched. No MRI was copied
through the Mac. The smoke predates the final code changes; its exact dirty
source snapshot was not captured and is not relabeled with delivery hashes.

The user subsequently enabled Edge access, but computer-use tool calls still
returned `Computer Use was not approved to use Microsoft Edge`. No browser
cookies or alternate UI control mechanism were used to bypass that denial.
An OAuth Desktop client and read-only consent remain required. The local
authorization helper and CPU API pipeline are implemented; the user's Google
account authorization was not completed and no new Secret is claimed.

`CHECK_REPO.txt` and `PYTEST.txt` record checks against the delivered source,
whose individual hashes are in `STATUS.json`. The API tests cover atomic
downloads, source checksums, crash recovery, verified-file reuse, bounded
pagination, sanitized HTTP errors, credential/client pairing,17-file
preservation, mask allowlists, new-container checks, duplicate launch refusal,
GPU secret/resource boundaries, config locks and100-step telemetry. One fixture
executes202 fake-file transfers, canonical preparation of98 manufactured cines,
native geometry/hash validation and a new-container TRAIN batch; a second pass
reuses all202 verified payloads. Fixture historical hashes are explicitly
isolated test witnesses, never replacements for the production pins.

DATA_PREP and S0 remain BLOCKED on OAuth/data. S1/S2 additionally require the
exact historical raw B8 payload and TRAIN-U cache. No patient training,
evaluation, GPU profile or Dice claim has been made. Source-specific software
checks do not resolve these physical asset blockers.

The executable commands, credential contract, monitoring and explicit checkpoint
resume are in [the API runbook](../../docs/MODAL_DRIVE_API_RUNBOOK.md). S0 uses the
unchanged canonical trainer. No compact KD arm or historical diagnostic is
automatically substituted when its required payload is missing.
