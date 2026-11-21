# Agent prompts, 2026-11-22

| File | Role |
|---|---|
| `QUEUE_TPC_ANISO_PROMPT_20261122.md` | **Run now on the vast RTX 5000 Ada.** Fixed 150 epochs for every new student arm, restart A4-SA at 150, re-order the GPU queue (S1-kd-m3hl, CB-out/S2-m3hl, S1-ANISO + screening); CPU-side: metric parity + bootstrap S2-m3hl-arb vs T-M3HL, offline TPC-Gate analysis E0-E3 with pre-registered kill criteria, AnisoStudent implementation + tests. |

Kick-off:

```
git pull
Read reports/agent-prompts-20261122/QUEUE_TPC_ANISO_PROMPT_20261122.md fully, plus the reports it lists.
Do Part Q (150-epoch rule, restart A4-SA at 150, queue), then Parts C1, C2, C3 on CPU in that order,
committing after each part. Report the queue/ETA, the parity verdict, the E0-E3 GO/NO-GO table and AnisoStudent params/FLOPs.
