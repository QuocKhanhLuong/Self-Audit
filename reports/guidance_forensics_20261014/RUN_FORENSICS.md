# Completed-run forensics, 14 October 2026

**Result: software defects and diagnostic gaps reproduced; attribution of the actual failed run is BLOCKED by inaccessible original artifacts. Full training: NO-GO.** This is a separate continuation, not a replacement for the failed baseline or prior theory reports.

## Identity and preservation

- Reviewed remote `main`: `e4d3269ce244d3d4ca0afdb1c92615f049116911`, committed `2026-10-14T01:18:05+07:00`, “feat: add guidance metrics tracking and verified ACDC evaluation”. Remote refs were refreshed before work.
- Run: `stable_guidance_acdc_4070_20261013_222918`; reported training source `d5a45f31481f0a702a1f9d5d418889fe704e9105`; W&B `jeti9jc9`.
- Expected student SHA256: `516e569c9141e2d8edb00b598df5cf12c8a9a4741cf1ae0cca8cbf23d6d03353`. The checkpoint itself has NOT been verified.
- Isolated branch: `QuocKhanhLuong/guidance-forensics-20261014`; base SHA is the reviewed `main` above.
- Worktree: `/Users/alvinluong/orca/workspaces/Self-Audit/guidance-forensics-20261014`.
- Outputs: `reports/guidance_forensics_20261014/`. Original `reports/stable_guidance_20261014/`, the original audit worktree, and the dirty main checkout were preserved. At analysis completion no run directory was overwritten and no commit, push or merge had been performed; the files were local/uncommitted at that point. The user subsequently authorized experimental-branch publication. See the [runbook](../../docs/guidance_forensics_runbook.md) for delivery and execution instructions; no main merge is included.

The existing `io.py`, `core.py` and `student.py` were byte-identical between the reported training SHA and reviewed main before our changes. Thus their operators are relevant to that reported source. This does not prove that the run used checked-in defaults, or that its working source was clean.

## Artifact access and exact limitations

The user identified machine alias `4070`. Its configured SSH destination responds but rejects the available key: `ssh -o BatchMode=yes -o ConnectTimeout=10 4070 'pwd'` returned **255**, `Permission denied (publickey,password)`. A verbose read-only check confirmed that the default Ed25519 key was offered and rejected. No password was requested in chat; no credential was changed. The user was asked to enable key access or identify an exported artifact bundle.

The newest relevant Downloads ZIP is `SelfAudit_ACDC_Run_e4d3269_Review_and_Prompt.zip` (13,782 bytes). Its four entries are `ASTRA_RECOVERY_PROMPT.md`, `PROBE_RESULTS.json`, `REVIEW_VI.md`, and `math_and_interface_probes.py`. It contains review material and manufactured mathematical probes, not run artifacts. ZIP and source hashes are in `DIAGNOSTICS.json`. We reproduced the relevant probes independently rather than treating this review as patient evidence.

Bounded searches covered the local Self-Audit checkout, its Orca worktrees and Downloads. Ninety-nine `last.pt` files were hashed; none matched the expected checkpoint. No filename matched the exact run name or W&B ID. Orca searches for both identifiers and the checkpoint hash found this ongoing review session, not the original launcher. Orca had no configured remote environment. This search does not establish absence on other disks or on 4070.

The W&B web page could not be retrieved by the web tool. A corrected GraphQL metadata request returned HTTP 200 with `project: null`; no W&B API key was present in the environment. This is an access limitation, not evidence that the run does not exist. No artifact was retrieved from W&B. An initial GraphQL query used an unsupported `File.size` field (HTTP 400), then was corrected.

| Required evidence | Current status | Consequence |
|---|---|---|
| Published summary and four-row CSV | READ; consistency checked | Reported results can be reconciled, not independently rescored |
| Run-local RECIPE/config/source snapshot | BLOCKED | Effective alpha, graph, thresholds, batching and source cleanliness unknown |
| Effective split, original split, preflight | BLOCKED | Cohort identities beyond the published list and duplicate-byte receipts unverified |
| Bank manifests, REPORT and patient NPZs | BLOCKED | R0/Q/hard support, permissions and soft loss mass unmeasured |
| Student checkpoint, complete history and REPORT | BLOCKED | Checkpoint identity, actual semantic updates, historical activity, CE/H/KL and gradients unmeasured |
| Frozen VAL/TEST exports and prediction pair lock | BLOCKED | Frozen prediction integrity cannot be independently verified |
| Frame-level METRICS.json and geometry receipts | BLOCKED | TP/FP/FN recomputation and aligned/recovery strata unavailable |
| Raw cine/frame headers and arrays belonging to this run | BLOCKED | Patient orientation and exact frame/cine equality not independently checked |

The public repository contains no per-frame confusion counts for this run. Aggregate Dice/precision/recall cannot uniquely reconstruct them. `DIAGNOSTICS.csv` therefore contains checks and explicit BLOCKED rows, **not invented patient counts**. Resolved run configuration and unavailable measurements are null in JSON.

## What was actually verified about the published evaluation

The published foreground metrics equal the mean of the RV/MYO/LV class means. All 16 percentages in the four-row CSV agree with the JSON within six-decimal rounding. Coverage plus UNKNOWN is one. These are 21 summary-consistency checks, not recomputation from saved TP/FP/FN.

Reported values remain: Dice 4.684410%, precision 19.623489%, recall 3.273159%, coverage 0.614758%, UNKNOWN 99.385242%. Foreground macro precision is neither pooled pixel accuracy nor a posterior probability of correctness. Undefined precision is excluded: RV precision has 11 defined patients, MYO/LV 18, BG 6. No undefined denominator was replaced with zero or one.

The current evaluator's SHA256 equals the published evaluator digest `44cfff418bc1510d61c4f3b9b6296d1ece9421b14739398d3900d2d7028315d6`. Source inspection verifies the intended contract:

1. Classes are BG=0, RV=1, MYO=2, LV=3; UNKNOWN=255 contributes foreground false negatives.
2. Raw cine XYZT becomes internal TZYX; evaluation transposes the selected prediction frame back to XYZ.
3. ACDC filename frame numbers become zero-based indices via subtraction of one; exactly two reference frames per prediction patient are required.
4. Shape must match. Affine mismatch is refused by default. Opt-in recovery checks raw-cine hash, matching frame/mask shape and affine, and exact array equality of the original unlabeled frame to the indexed cine frame. No resampling or class remapping occurs.
5. Class scores average frames within patients and then patients within classes. Foreground macro is the mean of the three class means. HD95 nulls retain their reasons.

The **reported**, not reexecuted, recovery covers 14/36 frames, leaving 22 aligned frames. Their separate scores are BLOCKED. All 18 patients must remain in the overall analysis; strata are additional views, not filters for a preferred cohort.

The reported exclusions are patient076, duplicate of TRAIN patient074 (SHA256 `69487c77380cf665520ffbe6ce3abc0eba5a121dc2270e564079be765c296a52`), and patient055, duplicate of official TEST patient118 (`ae245957fee700499974cbd7341c9397cc34d333915c10ed7d7285f29a1b8a47`). We preserved this policy and did not change splits. The underlying bytes and exclusion timing remain unverified.

## Verification and evidence boundary

The regression suite covers the existing stable-guidance, full-pipeline, final-metrics and tracker contracts plus new forensics tests. The new tests exercise real nibabel on manufactured images, no-reference discovery, world geometry, readout integration, CE decomposition, immutable gradient inspection, native/coarse ambiguity, metric nulls/aggregation, blocked artifact handling and opt-in logging parity. They also run a tiny manufactured CPU optimizer comparison with identical seeds: enabling diagnostics leaves model tensors bit-identical. This is **not a real TRAIN pilot** or a CineMA quality experiment.

See `COMMANDS.json`, `VERIFICATION.json` and `tests.xml` for executed commands, exit codes and final counts. `math_checks/DIAGNOSTICS.json` contains independently executed mathematical checks. Initial testing passed 35 tests with one tensor-to-scalar warning; the test was corrected to detach, and subsequent full checks passed without that warning. No warning or failed access was interpreted as a successful real check.

No dataset, reference mask, model weight or credential is included in this evidence directory. No dataset download, threshold sweep, teacher inference on real patients, GPU training or official TEST scoring was performed. VAL has already been exposed: this development is explicitly reference-informed in motivation, although the repairs and unit fixtures do not read its masks.

## Read inventory

Directly inspected: the two published run files; `src/self_audit_guidance/{io,core,student,pipeline,full,evaluate,preflight}.py`; `scripts/publish_guidance_metrics.py`; relevant tracker snippets; `docs/stable_guidance_full.md` sections on tracking, preflight, evaluation and recovery; the stable-guidance and metrics tests; workflow test commands; local Orca identity/worktree metadata and SSH connection diagnostics. The new diagnostics and tests were executed locally. Some broad metadata outputs were truncated; no conclusion relies on claiming to have read those in full. No inaccessible original run file is claimed as read.
