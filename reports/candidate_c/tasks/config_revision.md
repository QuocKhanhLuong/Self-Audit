# Config worker review: REVISE

Coordinator inspected your actual three-file diff. Retain ownership from previous dispatch. No commits/push/reverts/spawn. You are not alone. Complete these small revisions:

1. Remove _builder_accepts_execution_mode and _apply_execution_mode_kwargs and its builder call. Core constructor is now implemented; no production compatibility scaffolding for an intermediate working tree. Update focused tests/report accordingly.
2. Remove _bind_execution_mode and its binding call: runtime worker owns authoritative provenance.py resolve_model_identity/verify_model_config binding, including mode AND full solver settings in signature. Appending a field after hashing leaves a mode-blind signature. Coordinate with runtime delivery, do not edit provenance.py. Existing binding should rely on the centralized verifier. Revise tests so they cover actual final mechanism mismatch through it when available; don't add fallback that bypasses provenance.
3. Fix report CLI: canonical entrypoint is `python scripts/train_self_audit.py --config ...`, not `python -m self_audit.training.unified_trainer`. No new uncreated config path presented as already existing; explain copying/editing explicitly. Correct reported test arithmetic (69+24+55 is 148, not 93).
4. In report, document old saved last.pt configs/source/calibration as fail-closed incompatible with changed schema/source; NO silent resume migration. Ordinary old config files still load with defaults; these are distinct claims.

Focused test budget unchanged, no repeat full suite. Deliver amended baseline_interfaces.md and exact tests; report remaining dependency if centralized provenance hasn't landed. Worker_done then idle.
