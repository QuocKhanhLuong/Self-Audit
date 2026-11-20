# Agent prompts, 2026-11-20

| File | Role |
|---|---|
| `W2S_SSL4MIS_SPLIT_AGENT_PROMPT_20261120.md` | **Run this one now.** Move W2S-Match to the SSL4MIS ACDC split (70/10/20, 7 labeled, ED/ES-only U), keep S1, rerun from S2 (teacher) onward. |

Method specs it references: `reports/agent-prompts-20261119/W2S_MATCH_AGENT_PROMPT_20261119.md`, `reports/agent-prompts-20261119/CBKD_V2_AGENT_PROMPT_20261118.md`.

Kick-off for Claude Code on the RTX 4070 server (tmux `LQK`, repo checked out on branch `claude/w2s-ssl4mis-20261120`):

```
Read reports/agent-prompts-20261120/W2S_SSL4MIS_SPLIT_AGENT_PROMPT_20261120.md fully, plus the two
prompts in reports/agent-prompts-20261119/ that it references.
Execute Phase 0 (inspection) first. Before launching anything, list the jobs still running on the old
80/20/50 split and ask me whether to stop them. Then do Phase 1 (implementation + tests), Phase 2
(timing probe + BUDGET.md) and Phase 3 (launch the S2 -> S3 -> student pipeline, detached), then stop.
Report INSPECTION.md, BUDGET.md, test results, commit SHA and the tmux window / log path.
```
