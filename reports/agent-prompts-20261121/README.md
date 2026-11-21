# Agent prompts, 2026-11-21

| File | Role |
|---|---|
| `ADA5000_A4SA_SSL4MIS_PROMPT_20261121.md` | **Rented RTX 5000 Ada 32 GB, in parallel.** MedNeXt-L teacher W ("W-weak") → SAT → CB (S3-SA) → A4-SA (PG + CBKD-SA) on the SSL4MIS split; optional S2-main with W. This machine owns W-weak. |
| `M3HL_TEACHER_AGENT_PROMPT_20261121.md` | **Run this one now.** M3HL (MICCAI 2025) as the teacher: reproduce it on the SSL4MIS split (7 labeled), then train the 3D student with W2S-Match and with output-space CBKD-SA (`cbkd_sa_out`). Extends `reports/agent-prompts-20261120/W2S_SSL4MIS_SPLIT_AGENT_PROMPT_20261120.md`. |

Kick-off for Claude Code on the RTX 4070 server (tmux `LQK`, branch `claude/w2s-ssl4mis-20261120`):

```
git pull
Read reports/agent-prompts-20261121/M3HL_TEACHER_AGENT_PROMPT_20261121.md fully, plus
reports/agent-prompts-20261120/W2S_SSL4MIS_SPLIT_AGENT_PROMPT_20261120.md and
reports/w2s-ssl4mis-20261120/INSPECTION.md.
Execute Phase 0 (inspection, incl. building the SSL4MIS h5 data and checking L7 = first 136 slices),
then Phase 1 (M3HL wrapper, cbkd_sa_out, tests), then Phase 2 (train T-M3HL with official defaults,
reproduction check vs 90.47, timing probe, BUDGET.md), then Phase 3 (launch the student pipeline
detached) and stop. Ask me before stopping any running job.
Report INSPECTION.md, test results, T-M3HL val/test numbers, BUDGET.md, commit SHA, tmux window / log path.
```

Kick-off for Claude Code on the rented RTX 5000 Ada (repo cloned, branch `claude/w2s-ssl4mis-20261120`):

```
Read reports/agent-prompts-20261121/ADA5000_A4SA_SSL4MIS_PROMPT_20261121.md fully, plus the prompts and
reports it references. Execute Phase 0 (machine setup, data + S1 artifact with hash checks; ask me for
the 4070 ssh host, hourly price and max spend), Phase 1 (A4SA step for the SSL4MIS split + tests),
Phase 2 (timing probe + ADA5000_BUDGET.md), then Phase 3 (launch W -> SAT -> CB -> A4SA detached in tmux,
with artifact backups) and stop. Pull/rebase before every push; never force-push.
Report the Setup section, test results, budget, commit SHA, tmux window / log path.
```
