# Budget (Phase 2 timing probe, A100-40GB, 100 updates per stage, seed 42)

Probe: `scripts/cbkd_sa_probe.py` on Modal app `ap-hWUR8zaMVE5fKtUKbpW9sb`. All warmups are set to 0 in a probe copy so the
U / PG / CBKD-SA paths are timed; the real config is untouched. Rate used: USD 2.5 / h (from the run prompt).

| Stage | Updates | s/update (probe) | Peak mem | Notes | Projected h (x1.15) |
|---|---|---|---|---|---|
| S1 CineMA init | 10000 | 0.33 | 3.6 GB | the 100-update probe took 557 s wall because of CineMA load/startup | 1.1 |
| S2 teacher (MedNeXt-L) | 8000 | 2.36 | 7.9 GB | plus per-epoch VAL/TEST evaluation | 6.0 |
| S3-SA | 4000 | 1.6 | 6.0 GB | plus tau* calibration and VAL18 reconstruction report | 2.1 |
| A4-SA | 8000 | 3.0 | 18.0 GB | plus about 33 s per epoch evaluation x 100 epochs (0.9 h) | 8.7 |
| **Total** | | | | | **about 17.9 h, about USD 45** |

Not included: the probe itself (about 0.5 h GPU) and the optional A4-rerun (about 13 h more).
A4-SA costs about 3.0 s/update because the CBKD group is back-propagated separately (grouped clipping) and the atom
decoder runs at full resolution.

## Decision

The required stages exceed the stated credit (about USD 30). Per the run prompt I did not shorten updates or change any
config, and I did **not launch** the pipeline.

## Options

1. Add credit (about USD 45-50 for S1 + S2 + S3-SA + A4-SA in one pass) and launch as is.
2. Run the pipeline on the RTX 4070 server instead (`SERVER_4070_PROMPT_20261119.md` names it primary compute). The same
   `scripts/run_pipeline_cbkd_sa_a4.sh` resumes there (root from `SELF_AUDIT_ROOT`).
3. Split: S1 + S2 + S3-SA on Modal (about USD 25), A4-SA elsewhere. Stages are independent artifacts with DONE markers.
4. Cut cost without changing the method (needs your approval because it changes the protocol): drop the per-epoch
   `fusion`-path evaluation (about 15 % of A4 time) and TEST monitoring in S2.
