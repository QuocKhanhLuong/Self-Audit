from __future__ import annotations

import torch
import json
from pathlib import Path

from self_audit.candidate_pg.config import default_config
from self_audit.candidate_pg.export import (
    export_student,
    load_student_export,
    strip_export_keys,
    verify_cold_load,
)
from self_audit.candidate_pg.student_pg import PGStudent


def test_strip_export_keys_removes_all_train_only_branches() -> None:
    state = {
        "conv.weight": torch.ones(4, 1, 1, 1, 1), "conv.bias": torch.zeros(4),
        "fusions.0.weight": torch.ones(1), "mimic_heads.s2.weight": torch.ones(1),
        "code_branch.assign.weight": torch.ones(1), "teacher.body.weight": torch.ones(1),
    }
    result = strip_export_keys(state)
    assert set(result) == {"conv.weight", "conv.bias"}


def test_real_pgstudent_plain_deployment_cold_load_three_native_volumes(tmp_path) -> None:
    torch.set_num_threads(1)
    config = default_config(synthetic=True)
    model = PGStudent(config)
    path = tmp_path / "pgstudent.pt"
    export_receipt = export_student(model, config, path, verify=True)
    geometry = {"shape": (2, 17, 19), "spacing_zyx": [1.0, 1.0, 1.0]}
    volumes = [(torch.rand(1, 1, 2, 17, 19), geometry) for _ in range(3)]
    receipt = verify_cold_load(model, path, volumes)
    assert receipt["status"] == "PASS" and receipt["volumes"] == 3
    deployed = load_student_export(path)
    assert export_receipt["metadata"]["parameters_total"] == sum(p.numel() for p in deployed.parameters()) == 3074852
    assert export_receipt["metadata"]["training_parameters_total"] > export_receipt["metadata"]["parameters_total"]
    assert not any(key.startswith("fusions.") or key.startswith("mimic_heads.")
                   for key in deployed.state_dict())


def test_four_export_transaction_loads_sha_locked_models_before_evaluator(tmp_path) -> None:
    from self_audit.candidate_pg.export import ARM_NAMES, evaluate_all4_transaction, sha256_file

    torch.set_num_threads(1)
    config = default_config(synthetic=True)
    model = PGStudent(config)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"role": "test", "synthetic": True}))
    manifest_sha = sha256_file(manifest)
    bundle = {}
    for arm in ARM_NAMES:
        artifact = tmp_path / f"{arm}.pt"
        receipt = export_student(model, config, artifact)
        bundle[arm] = receipt
    seen = {}

    def evaluator(models, *, manifest_path, device):
        seen["models"] = models
        seen["manifest"] = manifest_path
        assert set(models) == set(ARM_NAMES)
        assert all(isinstance(value, torch.nn.Module) for value in models.values())
        # The evaluator is the first code allowed to open labels/manifest data.
        json.loads(manifest_path.read_text())
        return {"status": "SOFTWARE_ONLY"}

    ledger = evaluate_all4_transaction(bundle, tmp_path / "TEST_ONCE", evaluator,
                                       expected_manifest_sha256=manifest_sha,
                                       manifest_path=manifest)
    assert ledger["status"] == "COMMITTED"
    assert set(seen["models"]) == set(ARM_NAMES)
    assert Path(ledger["manifest"]) == manifest.resolve()
