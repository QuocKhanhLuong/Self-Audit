"""Focused W5 bulk-dispatch guard tests.

The profiler's paired throughput window owns timing claims.  These checks only
assert dispatch eligibility and compatibility boundaries on CPU objects.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from self_audit_maskfree import auditor, hypotheses
from self_audit_maskfree.contracts import AuditResult
from self_audit_maskfree.export import export_prediction
from self_audit_maskfree.observation import ObservationModel
from self_audit_maskfree.trainer import MaskfreeTrainer


def _trainer(*, components, observation):
    trainer = object.__new__(MaskfreeTrainer)
    trainer.components = components
    trainer.observation = observation
    return trainer


def _canonical_components():
    return SimpleNamespace(
        audit_banks=auditor.audit_banks,
        generate_bank=hypotheses.generate_bank,
        audit_bank=auditor.audit_bank,
    )


def _clone_value(value):
    """Detach tensor leaves while retaining metric/container structure."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return value


def _without_physical(value):
    """Drop intentional scalar/bulk physical-work differences from a trace."""
    if isinstance(value, dict):
        return {
            key: _without_physical(item)
            for key, item in value.items()
            if key != "physical_work"
        }
    if isinstance(value, list):
        return [_without_physical(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_physical(item) for item in value)
    return value


def _assert_nested_close(left, right, *, path=""):
    """Compare audit/search traces exactly except for float round-off."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        torch.testing.assert_close(left, right, rtol=3e-10, atol=3e-10, msg=path)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        assert set(left) == set(right), path
        for key in left:
            _assert_nested_close(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert isinstance(left, type(right)) and len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_close(left_item, right_item, path=f"{path}[{index}]")
        return
    if isinstance(left, float) or isinstance(right, float):
        assert isinstance(left, (int, float)) and isinstance(right, (int, float)), path
        assert math.isclose(float(left), float(right), rel_tol=3e-10, abs_tol=3e-10), path
        return
    assert left == right, path


def _assert_nested_exact(left, right, *, path=""):
    """Require byte-identical nested checkpoint/state payloads."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert np.array_equal(left, right), path
        return
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        assert set(left) == set(right), path
        for key in left:
            _assert_nested_exact(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert isinstance(left, type(right)) and len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_exact(left_item, right_item, path=f"{path}[{index}]")
        return
    assert left == right, path


def _audit_capture(audit: AuditResult) -> dict:
    """Keep candidate/decision/semantic tensors for scalar-v-bulk checks."""
    return {
        "bank_ids": [candidate.candidate_id for candidate in audit.bank],
        "bank_labels": [candidate.labels.detach().cpu().clone() for candidate in audit.bank],
        "bank_probabilities": [candidate.probabilities.detach().cpu().clone()
                               for candidate in audit.bank],
        "bank_validity": [candidate.validity.detach().cpu().clone() for candidate in audit.bank],
        "initial": audit.initial.candidate_id,
        "selected": audit.selected.candidate_id,
        "accepted": bool(audit.trace.get("accepted")),
        "semantic_unresolved": bool(audit.selected.semantic_unresolved),
        "regional_margin": audit.regional_margin.detach().cpu().clone(),
        "validity": audit.validity.detach().cpu().clone(),
        "trace": _without_physical(_clone_value(audit.trace)),
    }


def _flat_state(trainer: MaskfreeTrainer) -> dict[str, torch.Tensor]:
    return {
        f"{model_name}.{parameter_name}": parameter.detach().cpu().clone()
        for model_name, model in trainer.models.items()
        for parameter_name, parameter in model.state_dict().items()
    }


def _instrument_trainer(trainer: MaskfreeTrainer, captures: dict) -> None:
    """Capture targets, losses, pre-step gradients and audit results in-process."""
    original_optimizer_step = trainer._optimizer_step

    def capture_optimizer_step(accumulator):
        captures["gradients"].append({
            f"{model_name}.{parameter_name}": parameter.grad.detach().cpu().clone()
            for model_name, model in trainer.models.items()
            for parameter_name, parameter in model.named_parameters()
            if parameter.grad is not None
        })
        return original_optimizer_step(accumulator)

    trainer._optimizer_step = capture_optimizer_step
    original_record_audit = trainer._record_audit

    def capture_record_audit(**kwargs):
        captures["audits"].append(_audit_capture(kwargs["audit"]))
        return original_record_audit(**kwargs)

    trainer._record_audit = capture_record_audit


def _instrumented_components(base, captures: dict, *, bulk: bool):
    """Wrap canonical losses without changing bulk eligibility or RNG calls."""
    def capture_producer_loss(*args, **kwargs):
        loss, metrics = base.producer_loss(*args, **kwargs)
        captures["producer_loss"].append(_clone_value(loss))
        captures["producer_metrics"].append(_clone_value(metrics or {}))
        return loss, metrics

    def capture_student_loss(*args, **kwargs):
        probabilities, validity = args[1], args[2]
        captures["student_targets"].append(
            (_clone_value(probabilities), _clone_value(validity))
        )
        loss, metrics = base.student_loss(*args, **kwargs)
        captures["student_loss"].append(_clone_value(loss))
        captures["student_metrics"].append(_clone_value(metrics or {}))
        return loss, metrics

    components = replace(
        base,
        producer_loss=capture_producer_loss,
        student_loss=capture_student_loss,
    )
    if bulk:
        components = replace(components, audit_banks=auditor.audit_banks)
    return components


def test_bulk_dispatch_requires_canonical_components_and_methods() -> None:
    components = _canonical_components()
    model = ObservationModel()
    assert _trainer(components=components, observation=model)._bulk_audit_enabled()

    # An injected callback with the same signature must not reorder custom
    # side-effects or RNG; callback identity is part of the opt-in contract.
    custom = _canonical_components()
    custom.generate_bank = lambda *args, **kwargs: []
    assert not _trainer(components=custom, observation=ObservationModel())._bulk_audit_enabled()


def test_bulk_dispatch_rejects_subclasses_and_instance_fit_score_overrides() -> None:
    components = _canonical_components()

    class ChildObservation(ObservationModel):
        pass

    assert not _trainer(components=components, observation=ChildObservation())._bulk_audit_enabled()

    overridden = ObservationModel()
    overridden.fit = lambda *args, **kwargs: None  # type: ignore[method-assign]
    assert not _trainer(components=components, observation=overridden)._bulk_audit_enabled()

    overridden = ObservationModel()
    overridden.score = lambda *args, **kwargs: None  # type: ignore[method-assign]
    assert not _trainer(components=components, observation=overridden)._bulk_audit_enabled()


def test_optional_bulk_component_keeps_legacy_components_scalar() -> None:
    # Existing frozen test doubles construct Components without the additive
    # ``audit_banks`` field.  ``getattr``-based opt-in must remain false.
    legacy = SimpleNamespace(
        generate_bank=hypotheses.generate_bank,
        audit_bank=auditor.audit_bank,
    )
    assert not _trainer(components=legacy, observation=ObservationModel())._bulk_audit_enabled()


def test_tiny_trainer_batch_records_shared_bulk_api_calls_once(tmp_path, monkeypatch) -> None:
    """Canonical fixture exercises BxK flattening without a timing claim."""
    # Reuse the existing W5 synthetic fixture, but explicitly opt it into the
    # canonical bulk auditor.  Importing the fixture keeps all producer/student
    # loss and checkpoint semantics identical to the established trainer tests.
    monkeypatch.syspath_prepend(str(__file__).replace("/test_maskfree_trainer_performance.py", ""))
    from test_maskfree_trainer import build_components, build_config

    base = build_components()
    components = replace(base, audit_banks=auditor.audit_banks)
    trainer = MaskfreeTrainer(
        build_config(tmp_path / "bulk", max_steps=1), components=components
    )
    trainer.run()
    audit = trainer.history[0]["audit"]
    assert audit["physical_fit_many_calls"] == 1
    assert audit["physical_score_many_calls"] == 1
    assert audit["physical_fit_many_items"] == audit["candidates"]
    assert audit["physical_score_many_items"] == audit["candidates"]


def test_real_trainer_scalar_bulk_targets_losses_gradients_and_state_match(
    tmp_path, monkeypatch
) -> None:
    """Run one real neural batch through both paths and compare all learning inputs."""
    monkeypatch.syspath_prepend(str(__file__).replace("/test_maskfree_trainer_performance.py", ""))
    from test_maskfree_trainer import build_components, build_config

    def run_variant(root: Path, *, bulk: bool):
        captures = {
            "producer_loss": [],
            "producer_metrics": [],
            "student_targets": [],
            "student_loss": [],
            "student_metrics": [],
            "gradients": [],
            "audits": [],
            "model_outputs": [],
        }
        components = _instrumented_components(
            build_components(), captures, bulk=bulk
        )
        trainer = MaskfreeTrainer(
            build_config(root, max_steps=1), components=components
        )
        _instrument_trainer(trainer, captures)
        trainer.setup()
        handles = []
        for model_name, model in trainer.models.items():
            def capture_output(_module, _inputs, output, *, _name=model_name):
                captures["model_outputs"].append((_name, _clone_value(output)))

            handles.append(model.register_forward_hook(capture_output))
        trainer.run()
        for handle in handles:
            handle.remove()
        return trainer, captures

    scalar_trainer, scalar = run_variant(tmp_path / "scalar", bulk=False)
    bulk_trainer, bulk = run_variant(tmp_path / "bulk", bulk=True)

    # Both students see the same detached probabilities/validity tensors in
    # the same arm order, and both producer/student objective returns match.
    assert len(scalar["student_targets"]) == len(bulk["student_targets"]) == 2
    for (scalar_prob, scalar_valid), (bulk_prob, bulk_valid) in zip(
        scalar["student_targets"], bulk["student_targets"]
    ):
        assert torch.equal(scalar_prob, bulk_prob)
        assert torch.equal(scalar_valid, bulk_valid)
    _assert_nested_exact(scalar["producer_metrics"], bulk["producer_metrics"], path="producer_metrics")
    _assert_nested_exact(scalar["student_metrics"], bulk["student_metrics"], path="student_metrics")
    _assert_nested_exact(scalar["producer_loss"], bulk["producer_loss"], path="producer_loss")
    _assert_nested_exact(scalar["student_loss"], bulk["student_loss"], path="student_loss")
    _assert_nested_exact(scalar["model_outputs"], bulk["model_outputs"], path="model_outputs")

    # The independent scalar reference and flattened bulk decisions agree on
    # every discrete candidate/semantic/acceptance field and on score-search
    # traces; physical-work accounting is intentionally excluded above.
    assert len(scalar["audits"]) == len(bulk["audits"]) == 2
    for scalar_audit, bulk_audit in zip(scalar["audits"], bulk["audits"]):
        assert scalar_audit["bank_ids"] == bulk_audit["bank_ids"]
        assert scalar_audit["initial"] == bulk_audit["initial"]
        assert scalar_audit["selected"] == bulk_audit["selected"]
        assert scalar_audit["accepted"] == bulk_audit["accepted"]
        assert scalar_audit["semantic_unresolved"] == bulk_audit["semantic_unresolved"]
        for scalar_items, bulk_items in zip(
            scalar_audit["bank_labels"], bulk_audit["bank_labels"]
        ):
            assert torch.equal(scalar_items, bulk_items)
        for scalar_items, bulk_items in zip(
            scalar_audit["bank_probabilities"], bulk_audit["bank_probabilities"]
        ):
            assert torch.equal(scalar_items, bulk_items)
        for scalar_items, bulk_items in zip(
            scalar_audit["bank_validity"], bulk_audit["bank_validity"]
        ):
            assert torch.equal(scalar_items, bulk_items)
        _assert_nested_close(scalar_audit["trace"], bulk_audit["trace"], path="audit.trace")
        torch.testing.assert_close(
            scalar_audit["regional_margin"], bulk_audit["regional_margin"],
            rtol=3e-10, atol=3e-10,
        )
        torch.testing.assert_close(
            scalar_audit["validity"], bulk_audit["validity"],
            rtol=3e-10, atol=3e-10,
        )

    # Gradients are captured immediately before the optimizer mutates state;
    # post-step model tensors must remain numerically equivalent as well.
    assert len(scalar["gradients"]) == len(bulk["gradients"]) == 1
    assert set(scalar["gradients"][0]) == set(bulk["gradients"][0])
    for key in scalar["gradients"][0]:
        assert torch.equal(scalar["gradients"][0][key], bulk["gradients"][0][key])
    scalar_state, bulk_state = _flat_state(scalar_trainer), _flat_state(bulk_trainer)
    assert set(scalar_state) == set(bulk_state)
    for key in scalar_state:
        assert torch.equal(scalar_state[key], bulk_state[key]), key

    # Logical counters remain unchanged; only physical bulk fields differ.
    assert scalar_trainer.history[0]["audit"]["candidates"] == bulk_trainer.history[0]["audit"]["candidates"]
    assert bulk_trainer.history[0]["audit"]["physical_fit_many_calls"] == 1
    assert bulk_trainer.history[0]["audit"]["physical_score_many_calls"] == 1


def test_canonical_bulk_interrupted_resume_matches_uninterrupted(tmp_path, monkeypatch) -> None:
    """The flattened path restores RNG, cursor and optimizer state exactly."""
    monkeypatch.syspath_prepend(str(__file__).replace("/test_maskfree_trainer_performance.py", ""))
    from test_maskfree_trainer import build_components, build_config

    def bulk_components():
        return replace(build_components(), audit_banks=auditor.audit_banks)

    uninterrupted = MaskfreeTrainer(
        build_config(tmp_path / "full", max_epochs=2), components=bulk_components()
    )
    uninterrupted_report = uninterrupted.run()

    interrupted = MaskfreeTrainer(
        build_config(tmp_path / "split", max_steps=1), components=bulk_components()
    )
    interrupted_report = interrupted.run()
    checkpoint = str(interrupted.paths.last_checkpoint)
    assert interrupted_report["global_optimizer_steps"] == 1

    resumed = MaskfreeTrainer(
        build_config(tmp_path / "split", max_epochs=2, resume=checkpoint),
        components=bulk_components(),
    )
    resumed_report = resumed.run()

    assert resumed_report["global_optimizer_steps"] == uninterrupted_report["global_optimizer_steps"]
    assert resumed_report["last_completed_epoch"] == uninterrupted_report["last_completed_epoch"]
    resumed_state, uninterrupted_state = _flat_state(resumed), _flat_state(uninterrupted)
    assert set(resumed_state) == set(uninterrupted_state)
    for key in resumed_state:
        assert torch.equal(resumed_state[key], uninterrupted_state[key]), key

    # Compare serialized model/optimizer/RNG/sampler state, not just the
    # in-memory model tensors.  Workspaces differ, so identity/config/timing
    # metadata is intentionally excluded from this scientific continuation
    # check; those hashes are already guarded by _load_checkpoint itself.
    full_payload = torch.load(
        uninterrupted.paths.last_checkpoint, map_location="cpu", weights_only=False
    )
    resumed_payload = torch.load(
        resumed.paths.last_checkpoint, map_location="cpu", weights_only=False
    )
    for key in (
        "models", "optimizers", "scaler", "rng", "sampling_generator", "epoch_permutation",
        "global_step", "component_steps", "micro_batches_seen", "last_completed_epoch",
        "epochs_completed", "epoch", "batch_cursor", "accumulation_boundary",
        "pending_accumulation_microbatches",
    ):
        _assert_nested_exact(full_payload[key], resumed_payload[key], path=f"checkpoint.{key}")
    scientific_history_fields = (
        "global_epoch", "dataset", "protocol", "contract_version", "pseudo_label_version",
        "label_ramp", "lr", "global_optimizer_steps", "component_steps", "permutation_hash",
        "epoch_complete", "batch_cursor", "batches", "units_visited",
        "units_available", "producer", "students", "audit", "coverage", "detail",
    )
    assert len(full_payload["history"]) == len(resumed_payload["history"]) == 2
    for epoch, (full_row, resumed_row) in enumerate(
        zip(full_payload["history"], resumed_payload["history"])
    ):
        for field in scientific_history_fields:
            _assert_nested_exact(
                full_row[field], resumed_row[field], path=f"checkpoint.history[{epoch}].{field}"
            )
    assert full_payload["history"][0]["resumed_from_batch"] == 0
    assert resumed_payload["history"][0]["resumed_from_batch"] == 1

    # The resumed epoch replaces the interrupted partial record and remains on
    # the canonical bulk path, with one shared API call per physical batch.
    assert [row["global_epoch"] for row in resumed.history] == [0, 1]
    for resumed_row, full_row in zip(resumed.history, uninterrupted.history):
        _assert_nested_close(resumed_row["audit"], full_row["audit"], path="resume.audit")
        _assert_nested_close(resumed_row["producer"], full_row["producer"], path="resume.producer")
        _assert_nested_close(resumed_row["students"], full_row["students"], path="resume.students")
        _assert_nested_close(resumed_row["coverage"], full_row["coverage"], path="resume.coverage")
        assert resumed_row["audit"]["physical_fit_many_calls"] == full_row["audit"]["physical_fit_many_calls"]
        assert resumed_row["audit"]["physical_score_many_calls"] == full_row["audit"]["physical_score_many_calls"]


def test_canonical_bulk_preflight_records_physical8_image128_receipt(
    tmp_path, monkeypatch
) -> None:
    """A file-backed canonical preflight preserves logical and physical counters."""
    monkeypatch.syspath_prepend(str(__file__).replace("/test_maskfree_trainer_performance.py", ""))
    import test_maskfree_trainer as fixture

    # The fixture's role masks and image generator read these module globals,
    # so this is an actual 128x128, eight-unit physical batch rather than a
    # metadata-only configuration check.
    monkeypatch.setattr(fixture, "IMAGE_SIZE", 128)
    monkeypatch.setattr(fixture, "UNITS", 8)

    def discover_all_train(*args, **kwargs):
        manifest = fixture.discover_dataset(*args, **kwargs)
        for record in manifest["records"]:
            record["split"] = "train"
        return manifest

    components = replace(
        fixture.build_components(),
        discover_dataset=discover_all_train,
        audit_banks=auditor.audit_banks,
        export_prediction=export_prediction,
    )
    trainer = MaskfreeTrainer(
        fixture.build_config(
            tmp_path / "preflight",
            image_size=128,
            batch_size=8,
            total_epochs=4,
        ),
        components=components,
    )
    receipt = trainer.preflight()

    assert receipt["status"] == "pass"
    assert receipt["physical_batch"] == receipt["actual_physical_batch"] == 8
    assert receipt["audited_units"] == 8
    assert receipt["audit_attempts"] == 24
    assert receipt["audit_counters"]["candidates"] == 32
    assert receipt["physical_audit"] == {
        "physical_fit_many_calls": 1,
        "physical_fit_many_items": 32,
        "physical_score_many_calls": 1,
        "physical_score_many_items": 32,
        "physical_prepare_score_calls": 32,
        "physical_score_region_calls": receipt["audit_counters"]["physical_score_region_calls"],
        "physical_scalar_region_score_calls": 0,
    }
    assert receipt["physical_audit"]["physical_score_region_calls"] > 0

    # Both the gate and the preflight report are persisted files, so a later
    # consumer cannot mistake in-memory counters for a durable gate artifact.
    parent_gate = trainer.paths.gate_receipt
    probe_root = trainer.paths.root / "preflight_probe"
    probe_report = probe_root / "reports" / "preflight_report.json"
    assert parent_gate.is_file() and probe_report.is_file()
    assert json.loads(parent_gate.read_text())["physical_audit"] == receipt["physical_audit"]
    assert json.loads(probe_report.read_text())["audit_counters"] == receipt["audit_counters"]
