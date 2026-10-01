# DSS-US / SGSCN native milestone: server-audit fixes

Base: `e91df3d954771d3dff1809bff4fd60a7549dfb92` (`origin/codex/dss-us-sgscn-native`),
developed in a separate worktree on branch `codex/dss-us-sgscn-native-fixes`.
Scope: native code/protocol groundwork only. No clinical dataset, checkpoint, GPU
run, native GT evaluation or ACDC work. No scientific value was changed.

## Server `main` is independently broken

Server `main` (`fd995fb`) contains committed merge-conflict markers introduced by
`fabe030` ("add cuts original protocol") in `src/shared_benchmark/{__init__,manifest,
semantic_contract,spatial}.py`, `scripts/run_{cuts,dfc}_scientific.py` and
`tests/shared_benchmark/test_cardiac_freeze_validator.py`. `import shared_benchmark`
raises `SyntaxError`, so every native test/script fails at collection on `main`.
The native implementation commits (`4cf909c`, `e91df3d`) contain no markers. This
patch does not touch or resolve the CUTS/DFC conflict; that needs its own fix.

## Fixes

1. **Reference pinning** (`scripts/pin_native_references.py`). A `--no-checkout`
   clone has an empty index, so `git status` reported every file deleted and the
   script refused its own fresh clone. A never-populated clone (empty index, only
   `.git`) is now distinguished from an edited checkout; the pinned commit is
   checked out and verified clean afterwards. Edited or untracked content is still
   refused. Real fresh pin of all three references succeeded without manual steps.
2. **GT firewall** (`src/shared_benchmark/native_artifacts.py`, `run_native.py`).
   - `ImageInventory(..., image_root=...)` is mandatory: every image must lie in an
     image-only staging root that contains exactly the listed images (plus,
     optionally, the inventory). Unlisted entries/links fail closed before the
     producer starts, without echoing names. `run_native.py` requires `--image-root`
     and refuses an output root that overlaps it.
   - `ProducerAccessGuard` now: listing/scandir/glob only in output and trusted code
     trees, never in or above the staging root; open/stat only the exact listed
     images and readable files; `os.access` guarded; traversal (`..`) normalized;
     overlapping output/trusted/staging roots refused; a denial swallowed by a
     caller (`os.path.exists`, pathlib glob) still fails the guarded block on exit.
   - Residual: a Python guard is not an OS sandbox (C-level file access is not
     traced). If the inventory JSON is stored inside a dataset directory, that
     directory's metadata (not its listing) is stat-able. Keep the inventory in or
     beside the staging root and do not mount GT.
3. **DSS-US Step II evidence.** `majority_vote_exclusive` IS defined in pinned
   `evaluation/eval_utils.py:202` (called by `match` at line 172). SOURCE_LEDGER,
   PROTOCOL_LOCK, `evaluation/track_b/spec.json`, `step2.py` and the seven Step II
   configs are corrected, with an erratum. The gate stays BLOCKED_PROTOCOL and now
   names the actual unresolved items: which `segm_eval.py` branch/config
   (`eval_per_image`, `iou_thresh`, `void_label`; pinned defaults True/0.0/0 are not
   proven per row) produced each Table 2 row, remapping equivalence,
   conflict/unmatched handling and aggregation. Nothing was reimplemented.
4. **SGSCN context loss.** Paper (arXiv 2107.04934, PDF sha256
   `222a28c659e3f507dfed519a3e8b8945fa2afcb0a84a7ab428205e807d831051`) section 2.6:
   the overall loss is the *sum* of cross-entropy, sparse spatial and context losses
   -> context loss VERIFIED_PAPER, weight 1. The official demo enables it only via
   the non-default `--center` flag. Reference profiles keep their name but record
   `reference_invocation` (official arithmetic + `--center`, paper learning rate;
   not the demo default, not paper reproduction). New discrepancy recorded: code
   spatial weight 5 vs the paper's unweighted sum, added as a paper-profile gate.
   Upstream mathematics and snapshot are untouched.
5. **Seal hardening.** `RawRunWriter` refuses to reseal or write after sealing
   (exclusive `raw_seal.json` creation). `write_seal_receipt` writes a write-once,
   read-only, self-hashed receipt outside the raw root
   (`scripts/verify_native_raw.py RAW --write-receipt PATH`). Both native evaluators
   now require `seal_receipt=` and call `verify_seal_receipt` (receipt + raw seal)
   before protocol or GT access; a tampered-and-resealed raw run is rejected.

## Regenerated hashes

| Item | Old | New |
|---|---|---|
| DSS-US evaluator spec (canonical) | `c8b38333…805e88` | `c8cf2c04c19230fc707e29a10db4af91e257904dfb53aa96557dd084e17a010f` |
| SGSCN evaluator spec (canonical) | `811d5a6a…0a9721` | unchanged |
| SGSCN upstream receipt | — | unchanged |

DSS-US index (`baseline/DSS_US/config/native/index.json`), all 11 rebound:
step1_dss_baseline `e476b6ba…`, step1_ours_aff `95a533b1…`, step1_ours_comb `2233b00e…`,
step1_ours_proc `5446d4e2…`, step2_dss_baseline `8642b6be…`, step2_ours_aff_dss `976c592b…`,
step2_ours_aff_ours `49b9a851…`, step2_ours_comb_dss `dc476654…`, step2_ours_comb_ours `178f2c25…`,
step2_ours_proc_dss `644c3c05…`, step2_ours_proc_ours `f618d5e5…`.
SGSCN index: ph2_official_reference `f8e24385…`, ph2_paper `a32967f7…`,
sysu_us_official_reference `ed13f250…`, sysu_us_paper `ce907d85…`.
Full values are in the two `index.json` files. All `scientific` blocks are
byte-for-byte equal to `e91df3d` in canonical JSON.

## Verification (Linux, Python 3.12.13, torch 2.11.0+cpu, numpy 1.26.4)

Isolated venvs `.runtime/native_sgscn` and `.runtime/native_dss_us`; references
pinned by the fixed script into git-ignored `.scratch/`.

- `tests/native_baselines` + `baseline/SGSCN/tests`: **50 passed**, 0 skipped.
- `baseline/DSS_US/tests`: **10 passed**, 0 skipped (licensed DINO architecture with
  a random-weight fixture; pinned-source evidence check ran).
- Subset: GT firewall + reference pinning + seal receipts + tamper tests: 24 passed.
- `scripts/native_protocol_readiness.py`: exit 0; receipt
  `dss_us_sgscn_native_readiness_fixes_20261001.json`. Compared with `4cf909c`, no
  gate was weakened; 9 gates gained a reason.

## Remaining gates

- DSS-US, all 11 CAMUS profiles: producer BLOCKED_PROTOCOL (row recipe, CAMUS
  cohort); Track B BLOCKED_PROTOCOL (Step I pairing/aggregation; Step II row branch/
  config, remapping, aggregation). DINO checkpoint not supplied.
- SGSCN paper profiles: BLOCKED_PROTOCOL (architecture/stopping, spatial weight);
  Track B BLOCKED_PROTOCOL (tie policy, HM/XOR). Reference profiles: REFERENCE_READY.
- CAMUS, PH2, SYSU-US: BLOCKED_DATA. NATIVE_TRACK_A_STATUS: BLOCKED_ADAPTER.
- NATIVE_REPRODUCTION_STATUS / NATIVE_TRACK_B_STATUS: BLOCKED_PROTOCOL for both.
