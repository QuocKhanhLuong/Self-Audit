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


def _cold_load_fixture(tmp_path: Path) -> tuple[PGStudent, Path, list[tuple[torch.Tensor, dict]]]:
    config = default_config(synthetic=True)
    model = PGStudent(config)
    path = tmp_path / "pgstudent.pt"
    export_student(model, config, path, verify=True)
    geometry = {"shape": (2, 17, 19), "spacing_zyx": [1.0, 1.0, 1.0]}
    volumes = [(torch.rand(1, 1, 2, 17, 19), geometry) for _ in range(3)]
    return model, path, volumes


def test_cold_load_numerical_mismatch_is_warn_not_exception(tmp_path) -> None:
    torch.set_num_threads(1)
    model, path, volumes = _cold_load_fixture(tmp_path)
    with torch.no_grad():
        # A common shift preserves every class argmax while making the
        # exported and in-memory logits visibly differ.
        model.logits_head.bias.add_(0.01)
    receipt = verify_cold_load(model, path, volumes)
    assert receipt["status"] == "WARN"
    assert receipt["bitwise_equal"] is False
    assert receipt["argmax_equal"] is True
    assert receipt["max_abs_logit_delta"] > receipt["tolerance"]


def test_cold_load_sub_tolerance_mismatch_passes_with_diagnostics(tmp_path) -> None:
    torch.set_num_threads(1)
    model, path, volumes = _cold_load_fixture(tmp_path)
    with torch.no_grad():
        model.logits_head.bias.add_(5e-5)
    receipt = verify_cold_load(model, path, volumes)
    assert receipt["status"] == "PASS"
    assert receipt["bitwise_equal"] is False
    assert receipt["argmax_equal"] is True
    assert receipt["max_abs_logit_delta"] <= receipt["tolerance"]


def test_cold_load_changed_argmax_warns_even_within_logit_tolerance(tmp_path) -> None:
    torch.set_num_threads(1)
    model, path, volumes = _cold_load_fixture(tmp_path)
    with torch.no_grad():
        model.logits_head.weight.zero_()
        model.logits_head.bias.zero_()
    export_student(model, default_config(synthetic=True), path, verify=True)
    with torch.no_grad():
        model.logits_head.bias[1] = 5e-5
    receipt = verify_cold_load(model, path, volumes)
    assert receipt["max_abs_logit_delta"] <= receipt["tolerance"]
    assert receipt["argmax_equal"] is False
    assert receipt["status"] == "WARN"


def test_cold_load_child_receives_parent_runtime_flags(tmp_path) -> None:
    previous = {
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True, warn_only=True)
        # ``highest`` is the no-TF32 precision mode and remains compatible
        # with explicitly setting the legacy TF32 flags below.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        model, path, volumes = _cold_load_fixture(tmp_path)
        receipt = verify_cold_load(model, path, volumes)
        assert receipt["runtime"]["child"] == receipt["runtime"]["parent"]
        assert receipt["runtime"]["child"]["num_threads"] == 2
        assert receipt["runtime"]["child"]["deterministic_algorithms"] is True
        assert receipt["runtime"]["child"]["deterministic_warn_only"] is True
        assert receipt["runtime"]["child"]["cudnn_deterministic"] is True
        assert receipt["runtime"]["child"]["cudnn_benchmark"] is False
        assert receipt["runtime"]["child"]["cudnn_allow_tf32"] is False
        assert receipt["runtime"]["child"]["cuda_matmul_allow_tf32"] == receipt["runtime"]["parent"]["cuda_matmul_allow_tf32"]
    finally:
        torch.use_deterministic_algorithms(previous["deterministic"], warn_only=previous["warn_only"])
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = previous["cuda_matmul_allow_tf32"]
        torch.set_float32_matmul_precision(previous["matmul_precision"])
        if torch.get_num_threads() != previous["threads"]:
            torch.set_num_threads(previous["threads"])


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
