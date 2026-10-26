# Native-volume implementation status

All listed software paths are IMPLEMENTED and VERIFIED on CPU/manufactured inputs. This does not establish patient efficacy or target-GPU behavior. The final repository gate passed 422 tests; see [REPORT](../../reports/native_volume_pilot/REPORT.md) and its exact receipts.

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| A0 / historical v2 | train.py; volume/legacy_compat.py | test_volume_legacy.py; V2_FINAL_PARITY.json | IMPLEMENTED / VERIFIED_SOFTWARE |
| G0 physical geometry/native inversion | volume/geometry.py; data.py; scripts/volume_prepare.py | test_volume_geometry.py; test_volume_data.py; test_volume_evaluation.py | IMPLEMENTED / VERIFIED_SOFTWARE |
| G1 volume-batched 2.5D | volume/models.py | test_volume_models.py | IMPLEMENTED / VERIFIED_SOFTWARE |
| G2 hybrid / depth / capacity controls | volume/models.py | test_volume_models.py; test_volume_registry.py | IMPLEMENTED / VERIFIED_SOFTWARE |
| G3/G8 anisotropic CNN scratch | volume/models.py; V06/V07 configs | test_volume_models.py; test_volume_registry.py | IMPLEMENTED / VERIFIED_SOFTWARE |
| G4 CE/Dice estimators and accumulation | volume/losses.py | test_volume_losses.py; test_volume_engine.py | IMPLEMENTED / VERIFIED_SOFTWARE |
| Optimizer/scheduler/pseudo ramp | volume/optim.py; engine.py | test_volume_optim.py; test_volume_engine.py | IMPLEMENTED / VERIFIED_SOFTWARE |
| G5 volume EMA transaction | volume/engine.py | test_volume_engine.py; test_volume_integration_followup.py | IMPLEMENTED / VERIFIED_SOFTWARE |
| Resume / strict state/data identity | volume/checkpoint.py; engine.py | test_volume_resume.py; verified/ROUNDTRIP.json | IMPLEMENTED / VERIFIED_SOFTWARE |
| G6 skip fusion / interventions | volume/fusion.py; models.py | test_volume_fusion.py; test_volume_models.py | IMPLEMENTED / VERIFIED_SOFTWARE |
| G7 image/EMA regions and cache | volume/regions.py | test_volume_regions.py; test_volume_resume.py | IMPLEMENTED / VERIFIED_SOFTWARE |
| Temporal T correspondence | volume/temporal.py; data.py | test_volume_temporal.py | IMPLEMENTED / VERIFIED_SOFTWARE |
| G9 correction / TRAIN-L simulation | volume/correction.py; scripts/volume_correct.py | test_volume_correction.py; test_volume_cli.py | IMPLEMENTED / VERIFIED_SOFTWARE |
| 23-arm registry and opt-in schema | configs/native_volume; volume/config.py | test_volume_registry.py; test_volume_config.py | IMPLEMENTED / VERIFIED_SOFTWARE |
| Preflight / bounded smoke / profile | scripts/volume_preflight.py; volume_smoke.py; volume_profile.py | test_volume_cli.py; verified/COMMANDS.json | IMPLEMENTED / VERIFIED_SOFTWARE |
| Frozen native eval / student deployment | volume/evaluation.py; scripts/volume_evaluate.py; volume_export.py | test_volume_evaluation.py; verified/ROUNDTRIP.json | IMPLEMENTED / VERIFIED_SOFTWARE |

Paths beginning volume/ are under self_audit/. Tests are under tests/ and receipts under reports/native_volume_pilot/. All historical baseline/fixed/meta paths remain available. Each later module is opt-in with ablations and tests; no proposal-only/stub module is claimed complete.

NOT RUN: actual PRO4000/CUDA/VRAM/AMP, real DINO weights, real-patient geometry and TRAIN smoke, patient Dice, human annotation utility/time, long training and novelty/efficacy. Missing assets block those executions, not the implemented software paths. See [RUNBOOK](RUNBOOK.md) for future bounded commands and declared limitations.
