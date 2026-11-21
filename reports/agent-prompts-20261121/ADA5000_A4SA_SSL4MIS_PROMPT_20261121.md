# Agent prompt: A4-SA on the SSL4MIS split, on a rented RTX 5000 Ada 32 GB (2026-11-21, rev 1)

> Self-contained. Read it all before acting.
> User decision (2026-11-21): run **A4-SA** (PG + CBKD-SA, MedNeXt-L teacher) **in parallel** with the M3HL-teacher campaign, on a newly rented **RTX 5000 Ada (32 GB)**, on the **SSL4MIS split** (70/10/20, 7 labeled, ED/ES-only U). One seed (42).
> References (this prompt wins on conflict): `reports/agent-prompts-20261120/W2S_SSL4MIS_SPLIT_AGENT_PROMPT_20261120.md` (split, metrics, budget, constraints); `reports/agent-prompts-20261119/CBKD_SA_A4_MODAL_PROMPT_20261119.md` and `reports/agent-prompts-20261119/CBKD_V2_AGENT_PROMPT_20261118.md` (A4-SA and CBKD-SA rev 4 definition); `reports/w2s-ssl4mis-20261120/INSPECTION.md` (facts about the split and the data).

---

## 0. Context and division of labour

| Machine | Campaign | Owns |
|---|---|---|
| RTX 4070 server (GPU 1, tmux `LQK`) | `m3hl-teacher-20261121` (prompt `reports/agent-prompts-20261121/M3HL_TEACHER_AGENT_PROMPT_20261121.md`) | T-M3HL, M3HL-teacher student arms, S0 |
| **RTX 5000 Ada 32 GB (this machine)** | `w2s-ssl4mis-20261120`, MedNeXt branch | **W (MedNeXt-L teacher, "W-weak") → SAT → CB (S3-SA codebook) → A4-SA**; optionally S2-main |

The 4070 does **not** train W-weak (the M3HL prompt defers it). This machine is its only owner. Do not run M3HL or S0 here unless the user asks.

- Repo `QuocKhanhLuong/Self-Audit`, branch `claude/w2s-ssl4mis-20261120` (pull; record the SHA). Already on the branch: SSL4MIS lock + lists (`configs/candidate_pg/ssl4mis_*`), `scripts/ssl4mis_prepare_acdc.py`, `self_audit/candidate_pg/ssl4mis_data.py`, `evaluation_ssl4mis_compat.py`, the resumable pipeline `scripts/run_pipeline_w2s_ssl4mis.sh` → `scripts/w2s_ssl4mis_pipeline.py`, with steps `CONFIG LINK W WFINAL SAT CONFIG2 LINKW CB S2MAIN …`. The A4-SA pipeline is `scripts/cbkd_sa_pipeline.py` (80/20/50 only); CBKD-SA is in `shape_atoms.py`.
- Root: `SELF_AUDIT_ROOT` (choose a fast local disk on this machine, e.g. `/workspace/mount_data`). Runs go to `$SELF_AUDIT_ROOT/runs/w2s-ssl4mis-20261120/` (same campaign name as the split prompt, so artifacts are interchangeable). The report goes to `reports/w2s-ssl4mis-20261120/ADA5000.md`.
- Reference cost on A100-40GB (`reports/cbkd-sa-a4-20261119/BUDGET.md`): S2 teacher 2.36 s/update (8000 upd, about 6 h), S3-SA 1.6 s/update (4000 upd, about 2 h), A4-SA 3.0 s/update (8000 upd, about 8.7 h), peak memory 18 GB. Expect this GPU to be slower; measure it.

## 1. Arms on this machine

| Step | What | Required |
|---|---|---|
| W | MedNeXt-L 3D teacher, repo teacher stage (`unimatch_v1`, SDF head kept), **fixed threshold 0.95** (`teacher.semi.threshold.clip = [0.95, 0.95]`), L7 + U63 (ED/ES), 8000 updates, selection on val10 | yes |
| SAT | Saturation check on U (decides τ_v for any later arbiter arm; cheap) | yes |
| CB | S3-SA shape-atom codebook on frozen W (rev 4 §4, feature-based `P_T`), TRAIN = L7 ∪ U63 only | yes |
| **A4-SA** | Student with `pg.enabled = true` (PG fusion target as in campaign pg-k8), `cbkd.enabled = true`, `cbkd.variant = "cbkd_sa"` (rev 4 §5), everything else as in the A4-SA run prompt, **only the split changed** | yes |
| S2-main-W | Student, W2S arbiter + gated CBKD-SA with W as teacher (the existing `S2MAIN` step) | optional; only if the budget allows after A4-SA |

Two deviations from the old A4-SA definition, both forced by the new protocol. Record them in the report.
1. Split, labels and U follow the SSL4MIS lock (L7, U63 ED/ES, val10 selection, test20 once).
2. The teacher threshold is fixed at 0.95 (the W2S decision), instead of the class-adaptive rule of the old S2. The PG student threshold stays as in A4 (class-adaptive) for comparability with old A4; state this.

## 2. Phase 0: machine setup and inspection (`reports/w2s-ssl4mis-20261120/ADA5000.md`, section Setup)

1. `nvidia-smi` (record the GPU, driver, CUDA), disk free space, RAM, CPU count. Create a Python env matching the repo's `requirements.txt`, plus `nnunet_mednext` from the pinned `external/MedNeXt` commit and `cinema` (same versions as the Modal image in `scripts/modal_app.py`). Record the torch version and check that bf16 autocast works on this GPU.
2. **Data.** Raw ACDC (`training/`, `testing/`): copy it from the 4070 server (`rsync` over ssh, if the user gives the host) or from Google Drive (rclone, folder id `1EelzBVjIoDQ4uzt0_2JzmF_PuUHsD93e`, read-only). Verify against the hashes recorded in the lock or in the server's preparation receipt. Then run `scripts/ssl4mis_prepare_acdc.py` and check that the prepared manifests match the lock (L7 / U63 / val10 / test20).
3. **S1 artifact** (CineMA → student encoder; reused, not retrained). Copy `student_init.pt` and its receipt/config from the 4070 server run (`runs/cbkd-sa-a4-20261119/...`) or from the Drive backup made by `scripts/cbkd_sa_upload.py`. Verify that sha256 equals `c8195fb4a79ee665b169da36a492c99192eb33a1962c42079404107c1d2a01bf` (the default `S1_SHA256` in `w2s_ssl4mis_pipeline.py`). Stop and ask if it does not match. Never retrain S1 silently.
4. Run the full test suite on this machine (`python -m pytest -q`), and do a dry run of the pipeline steps `CONFIG LINK` only.
5. Ask the user for: the ssh host of the 4070 server (if copying from it), the hourly price of this machine, and the maximum spend. Write them in `BUDGET` (§4).

Stop and report if data hashes, the split or the S1 hash do not match, or if MedNeXt / CineMA do not import.

## 3. Phase 1: add the A4-SA step for the SSL4MIS split (CPU)

- Add an arm `A4SA` to the SSL4MIS config builder (`scripts/w2s_make_configs_ssl4mis.py`) and the pipeline (`scripts/w2s_ssl4mis_pipeline.py`): steps `A4SA` and `A4SAFINAL`, using W from step `W` and the codebook from step `CB`. Build its config from the A4-SA config of `cbkd-sa-a4-20261119` (`cbkd_sa_make_config.py`), changing **only** the data/split fields and the selection split (val10), plus the teacher threshold of §1. Prove it with a config diff against the old A4-SA config: only split/data/selection/teacher-threshold fields may differ.
- Keep the rev 4 §8.8 optimisation protocol of A4-SA: CBKD-SA in its own gradient-clip group with the factor logged, PG onset at 800, CBKD-SA onset at 1200 with its own ramp, per-term gradient norms every 50 updates.
- Tests: the config diff test above; `A4SA` resumes from a checkpoint on a toy stage; the export has no teacher / codebook / `D_A` / `P_S` keys; existing suites unchanged.
- Commit and push to `claude/w2s-ssl4mis-20261120`. Pull first; the 4070 agent pushes to the same branch, so rebase on conflict and never force-push.

## 4. Phase 2: timing probe and budget

- 200 updates each of W, CB and A4SA on this GPU; seconds per update, peak memory. Note whether 32 GB allows the default batch without changes. Do not change batch size or update counts.
- `reports/w2s-ssl4mis-20261120/ADA5000_BUDGET.md`: hourly price, projected hours per step (×1.15 for evaluation and I/O), total, maximum spend. Required steps: W, SAT, CB, A4SA. Add S2-main-W only if the remaining budget is at least 1.3× its projected cost.
- If the required steps exceed the maximum spend, stop and report options. Do not shorten.

## 5. Phase 3: launch

- Launch detached in tmux on this machine, for example: `SELF_AUDIT_ROOT=... MAX_HOURS=... CAP_HOURS_W=... scripts/run_pipeline_w2s_ssl4mis.sh --steps "CONFIG LINK W WFINAL SAT CONFIG2 LINKW CB A4SA A4SAFINAL"`. Add `S2MAIN S2MAINFINAL` only if the budget allows.
- **Back up artifacts as each step finishes:** W `teacher_best.pt`, the S3-SA codebook, and the A4-SA student export, with their receipts. Back them up to Google Drive (rclone) and/or copy them to the 4070 server, so a terminated rental does not lose them. The 4070 needs W and CB later for the weak-teacher regime arms.
- Write `RUNNING.md` (tmux session/window, log paths, monitoring commands) and end the session after the first epoch line of W is visible.

## 6. Phase 4 (later, when the user asks): report

Add to `reports/w2s-ssl4mis-20261120/ADA5000.md`, with measurements / interpretation / deviations kept separate:

- W on val10 and test20 (native 3D metric in mm, and the SSL4MIS-compatible metric in voxels); the saturation result and τ_v.
- CB reconstruction report: q_T vs raw W on val10, perplexity, dead codes, m_T at boundary vs interior cells.
- A4-SA at its val10-selected checkpoint: val10 and test20, both metric sets, per class; student code perplexity curve; per-term gradient norms and clip factors.
- If available, S2-main-W, and a pointer to how A4-SA compares with the 4070 arms (S0, M3HL-teacher arms). Use paired per-patient Wilcoxon and bootstrap CI on test20, with the same "supported / inconclusive" rule as the split prompt.
- Spend vs budget.

## 7. Hard constraints

1. Inspect first; write the Setup section before editing code.
2. Supervised terms use L7 masks only. U63 gives images and detached teacher outputs only. CB and τ* use L7 ∪ U63 only. val10 is the only selection signal; test20 is opened once per arm.
3. Seed 42. No tuning of λ, τ, learning rate, clip norms, batch size or update counts.
4. The student export holds the student only; run the cold-load test.
5. Never write to raw data or to other campaigns' runs; never stop jobs on the 4070 server; never delete data without the user's OK; never commit weights or data.
6. Commit and push after Phase 1 and after the report (pull/rebase first, no force-push); report the SHAs.

## 8. Non-goals

M3HL, S0, S1-kd, S2-arb on this machine; the old 80/20/50 split; retraining S1; extra seeds; M&Ms; TEST50; full cine frames as U.
