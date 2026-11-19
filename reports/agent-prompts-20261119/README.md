# Agent prompts, 2026-11-19

| File | Role |
|---|---|
| `CBKD_SA_A4_MODAL_PROMPT_20261119.md` | **Run this one now.** Set up Modal and train A4-SA (PG + CBKD-SA) in place of A4. |
| `CBKD_V2_AGENT_PROMPT_20261118.md` | Method spec of CBKD-SA (rev 4), referenced by the run prompt. |
| `W2S_MATCH_AGENT_PROMPT_20261119.md` | W2S-Match campaign, run after A4-SA. |
| `SERVER_4070_PROMPT_20261119.md` | **Primary compute.** RTX 4070 server (GPU 1, tmux `LQK`): A4-SA first, then W2S-Match. Modal is backup. |

Launcher: `scripts/modal_app.py` (volume `self-audit-mount` mounted at `/root/mount_data`).

Kick-off on the 4070 server (Claude Code inside tmux `LQK`):

```
Read reports/agent-prompts-20261119/SERVER_4070_PROMPT_20261119.md and every prompt it references, fully.
Execute Phase S0, Phase S1 and Part A up to the launch of the A4-SA pipeline, then stop.
Report SETUP.md, PROBE.md, test results, commit SHA and the tmux window / log path.
```

Kick-off for Claude Code on Modal (backup; repo root, this branch checked out):

```
Read reports/agent-prompts-20261119/CBKD_SA_A4_MODAL_PROMPT_20261119.md and
reports/agent-prompts-20261119/CBKD_V2_AGENT_PROMPT_20261118.md fully.
Execute Phases 0–3: set up, implement, test, probe, then launch the detached pipeline and stop.
Report INSPECTION.md, BUDGET.md, test results, commit SHA and the Modal app ID.
```

Manual prerequisites (browser login, done once by the user): `modal setup`; `rclone authorize "drive"` then
`modal secret create rclone-gdrive RCLONE_CONFIG_GDRIVE_TYPE=drive RCLONE_CONFIG_GDRIVE_SCOPE=drive.readonly RCLONE_CONFIG_GDRIVE_TOKEN='<json>'`.
