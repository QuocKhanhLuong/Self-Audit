# Agent prompts, 2026-11-19

| File | Role |
|---|---|
| `CBKD_SA_A4_MODAL_PROMPT_20261119.md` | **Run this one now.** Set up Modal and train A4-SA (PG + CBKD-SA) in place of A4. |
| `CBKD_V2_AGENT_PROMPT_20261118.md` | Method spec of CBKD-SA (rev 4), referenced by the run prompt. |
| `W2S_MATCH_AGENT_PROMPT_20261119.md` | W2S-Match campaign. Deferred: the current Modal credit is used for A4-SA. |

Launcher: `scripts/modal_app.py` (volume `self-audit-mount` mounted at `/root/mount_data`).

Kick-off for Claude Code (repo root, this branch checked out):

```
Read reports/agent-prompts-20261119/CBKD_SA_A4_MODAL_PROMPT_20261119.md and
reports/agent-prompts-20261119/CBKD_V2_AGENT_PROMPT_20261118.md fully.
Execute Phases 0–3: set up, implement, test, probe, then launch the detached pipeline and stop.
Report INSPECTION.md, BUDGET.md, test results, commit SHA and the Modal app ID.
```

Manual prerequisites (browser login, done once by the user): `modal setup`; `rclone authorize "drive"` then
`modal secret create rclone-gdrive RCLONE_CONFIG_GDRIVE_TYPE=drive RCLONE_CONFIG_GDRIVE_SCOPE=drive.readonly RCLONE_CONFIG_GDRIVE_TOKEN='<json>'`.
