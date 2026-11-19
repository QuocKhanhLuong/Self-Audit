# Agent prompt: set up Modal and train A4-SA (PG + CBKD-SA) in place of A4 (2026-11-19, rev 1)

> Self-contained run prompt for Claude Code on the user's Mac. Read it all before acting.
> Method spec: `CBKD_V2_AGENT_PROMPT_20261118.md` (CBKD-SA rev 4), in the same folder `reports/agent-prompts-20261119/`. This prompt decides scope, compute and order; rev 4 decides the method. If they conflict, this prompt wins.
> User constraints: one seed (42), about USD 30 of Modal credit (A100-40GB), no time for extra arms or tuning.

---

## 0. Context

- Repo `~/Self-Audit` (`QuocKhanhLuong/Self-Audit`, branch `main` at `dc102c9` (contains the analysed `fd12e91` plus 12 PG memory / recovery fixes)). Packages `self_audit/candidate_pg`, `self_audit/candidate_fuse`. Base config `configs/candidate_pg/pg_k8_base.json`.
- The old vast server is gone. Everything of campaign `pg-k8-20261116` that survives is the Drive bundle `pg_k8_A1-A4_bundle.zip` (204 MB). Raw ACDC (`training/`, `testing/`) is on Drive.
- Compute: Modal. Launcher `scripts/modal_app.py` provides `pull_drive(folder_id, dest)`, `cpu(cmd)`, `gpu(cmd)` (A100-40GB, 24 h timeout). Volume `self-audit-mount` is mounted at `/root/mount_data`, the old server path, so configs keep their paths.
- Already done by the user (verify, do not redo): `modal setup`, secret `rclone-gdrive`.
- Drive folder IDs: raw ACDC `1EelzBVjIoDQ4uzt0_2JzmF_PuUHsD93e`; bundle folder `11ZkoQv6XZBV5igXNsF6zsGwbVxpgWVgW`.
- New campaign `cbkd-sa-a4-20261119`. Runs: `/root/mount_data/runs/cbkd-sa-a4-20261119/<stage>`. Report: `reports/cbkd-sa-a4-20261119/`.

### Goal

Train **A4-SA = A4 with CBKD-v1 replaced by CBKD-SA** (`pg.enabled = true`, `cbkd.enabled = true`, `cbkd.variant = "cbkd_sa"`), everything else as campaign `pg-k8` (split lock, K8, U = remaining TRAIN cine frames, student, teacher config, PG config, optimizer, 8000 updates, augmentation, VAL18 student-only FG selection). Report the deployed student on VAL18 and TEST50.

### Pipeline (stage dependencies)

```
raw ACDC ──prepare──► prepared/ (same split lock as pg-k8)
CineMA ──S1──► student encoder init        (skip if a valid S1 encoder is in the bundle)
prepared ──S2──► MedNeXt-L teacher         (old teacher_best.pt is not in the bundle: retrain, same config)
S2 ──S3-SA──► P_T, C, D_A, tau*            (rev 4 §4)
S1 + S2 + S3-SA ──A4-SA──► student         (rev 4 §5, PG on)
optional: A4-rerun (CBKD-v1, PG on, new S2) only if the budget allows (§4)
```

## 1. Phase 0: setup and inspection, CPU only (`reports/cbkd-sa-a4-20261119/INSPECTION.md`)

1. `modal --version`, `modal secret list` (expect `rclone-gdrive`), `modal volume list`.
2. If missing, pull data: `modal run scripts/modal_app.py::pull_drive --folder-id 1EelzBVjIoDQ4uzt0_2JzmF_PuUHsD93e --dest raw/ACDC` and the bundle to `bundles/pg-k8-20261118`; unzip the bundle on the `cpu` function.
3. Environment check on the `gpu` function (one short call): `import torch, nnunet_mednext, cinema`, device name. If the CineMA pip install changed torch or fails, pin versions in `scripts/modal_app.py` and report.
4. Bundle inventory: list every checkpoint and receipt. Record whether it contains (a) a usable S1 CineMA encoder artifact (check hash against the S1 receipt), (b) `best_student` checkpoints for A1–A4, (c) the S3 codebook, (d) any teacher checkpoint.
5. Re-prepare data from `raw/ACDC` with the repo's own preparation code and the `pg-k8` split lock (`configs/candidate_fuse/semi_k8.json`). Compare the resulting manifest hashes with the locks in `pg_k8_base.json` (`train.json` sha256 `ef7562a2…`, `source_k8_sha256` `60dd1174…`). Matching hashes mean the data are identical to `pg-k8`; mismatches are reported, new locks written into the new config, and the run continues.
6. Rev 4 Phase 0 items that need no teacher, on CPU: O1 (pooled-code oracle) and O2 (shape-atom oracle) on VAL18; O4 if the old S3 codebook is in the bundle; O6 config diff.

Stop and report if raw ACDC is incomplete, the split lock cannot be reproduced (K8 or VAL18/TEST50 patients differ), or the environment check fails after one pinning attempt.

## 2. Phase 1: implementation and tests (CPU)

- Implement CBKD-SA exactly per rev 4 §2, §4, §5 and the guards in §1b: shape-atom codebook (`candidate_pg/shape_atoms.py`), `cbkd.variant = "cbkd_sa"` with `L_pm` (clean weak U and L_K) and `L_atom`, margin `m_T`, `τ*`; CBKD-v1 untouched.
- Rev 4 §8.8 optimization protocol: per-term grad-norm logging every 50 updates, CBKD-SA gradients clipped as their own group with the clip factor logged, CBKD-SA onset staggered from the PG onset (PG at 800 as before; CBKD-SA starts at 1200 with its own ramp).
- Pipeline script `scripts/run_pipeline_cbkd_sa_a4.sh` (root path from env `SELF_AUDIT_ROOT`, default `/root/mount_data`, so it also runs on the 4070 server):
  - runs prepare check → S1 (only if needed) → S2 → S3-SA → A4-SA → final evaluation, in one container;
  - each stage writes a `DONE.json` marker (config hash, artifact hashes); on restart, finished stages are skipped and an unfinished stage resumes from its last checkpoint (checkpoint every 500 updates);
  - budget guard: reads per-stage hour caps from `BUDGET.md`-derived env vars; aborts cleanly (checkpoint + marker `ABORTED_BUDGET`) if a stage exceeds 1.3× its projected time or the total wall clock exceeds `MAX_HOURS`.
- Evaluation cost: VAL18 every epoch for selection; **TEST50 only once, on the VAL18-selected checkpoint** (pg-k8 monitored TEST50 each epoch; it was never used for decisions).
- Tests: rev 4 §9 tests 1–10, plus: pipeline skip/resume on a toy stage, budget guard abort on a toy stage, export clean. Run them with `modal run scripts/modal_app.py::cpu --cmd "python -m pytest -q ..."`. Existing suites must still pass.
- Commit and push; record the SHA.

## 3. Phase 2: timing probe and budget (GPU, short)

- One `gpu` call that runs 100 updates of each needed stage type (S1 if needed, S2, S3-SA, A4-SA) without saving final artifacts, and reports seconds per update and peak memory.
- Write `BUDGET.md`: cost rate (about USD 2.5/h for the `gpu` function), projected hours per stage (updates × s/update × 1.15 for evaluation and I/O), projected total, credit remaining as reported by the user or the Modal dashboard.
- Plan: required stages are S1 (if needed), S2, S3-SA, A4-SA. Add A4-rerun only if the remaining credit after the required stages is ≥ 1.3× its projected cost.
- If the required stages exceed the credit, **stop and report** the numbers and options (for example reusing a bundle artifact, or dropping S1 by reusing a bundle encoder). Do not shorten updates or change configs to fit.

## 4. Phase 3: launch and stop

- Launch the pipeline once: `modal run --detach scripts/modal_app.py::gpu --cmd "MAX_HOURS=<from BUDGET.md> bash scripts/run_pipeline_cbkd_sa_a4.sh"`.
- Record the app ID and the monitoring commands in `reports/cbkd-sa-a4-20261119/RUNNING.md`: `modal app logs <app-id>`, `modal volume ls self-audit-mount runs/cbkd-sa-a4-20261119`, `modal volume get ...`.
- End the session after the launch is confirmed running (first epoch line visible). Do not poll for hours.

## 5. Phase 4 (later session, when the user asks): results and report

- Pull logs and artifacts; verify markers; if aborted, report why and what resumes.
- Evaluate with the same evaluation code and the same prepared data: new S2 teacher (rev 4 O3), S3-SA reconstruction vs raw teacher vs O2, A4-SA; and the **old A2 and A4 `best_student` from the bundle** (rev 4 O5) so the comparison uses identical evaluation.
- Report (`README.md`), measurements / interpretation / deviations separated:
  - Phase 0 table (O1, O2, O4 if available), data-hash match result, bundle inventory.
  - S2 and S3-SA metrics; student code perplexity curve; `m_T` boundary vs interior.
  - A4-SA vs old A4 and old A2 (and A4-rerun if run): VAL18 and TEST50 FG, RV, MYO, LV Dice, HD95; per-patient Wilcoxon and 10,000-bootstrap 95 % CI on TEST50; "supported" only if the CI excludes 0 on TEST50 and the sign agrees on VAL18; otherwise "inconclusive".
  - **Confounds to state explicitly:** A4-SA uses a retrained teacher (old A4's teacher is lost); CBKD-SA has its own gradient clip group and staggered onset (rev 4 §8.8), which A4 did not; if data hashes mismatched, data differ too. Only A4-rerun removes the teacher confound.
  - Spend vs budget.

## 6. Hard constraints

1. Inspect first; write `INSPECTION.md` before editing code.
2. Supervised terms use K8 masks only; S3-SA fitting and `τ*` use TRAIN only; VAL GT is used for evaluation and selection only; TEST50 is never used for any decision.
3. Seed 42 only. No tuning of λ, τ, learning rates, clip norms, or update counts.
4. Resolved configs of A4-SA and A4 may differ only in `cbkd.variant`, the CBKD-SA fields and the CBKD clip group; show the diff.
5. Export the student only; cold-load test; no teacher / codebook / `D_A` / `P_S` keys.
6. Never write to the bundle or raw data folders; never delete volume data without asking the user.
7. Commit and push after Phase 1 and after the report; report SHAs.
8. Any GPU job longer than 15 minutes runs detached through the pipeline script, never as a blocking call in the session.

## 7. Non-goals

W2S-Match / arbiter, A3-SA and A3-raw arms, extra seeds, fixing the PG class-adaptive threshold (kept as in A4 for comparability), M&Ms, calibration studies.
