"""Audit execution-device dispatch and identity checks.

These are CPU-only control-plane tests.  They verify that the observation
owner receives the trainer-resolved device, that explicit CUDA requests fail
closed when CUDA is unavailable, and that an exact resume cannot cross a
recorded audit backend.  They do not claim GPU execution or numerical parity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit_maskfree import runtime
from self_audit_maskfree.config import ConfigError, MaskfreeConfig, load_config
from self_audit_maskfree.trainer import (
    Components,
    MaskfreeTrainer,
    ResumeIdentityError,
    TrainerContractError,
)


class _SpyObservation:
    """Injected W1 double that records the additive execution-device kwarg."""

    def __init__(self, *, max_iterations=5, variance_floor=0.05, beta=0.01,
                 execution_device=None):
        self.max_iterations = max_iterations
        self.variance_floor = variance_floor
        self.beta = beta
        self.execution_device = execution_device


def _unavailable(name):
    def _raise(*_args, **_kwargs):
        raise AssertionError(f"{name} should not be called in this control-plane test")

    return _raise


def _components(observation_factory=_SpyObservation) -> Components:
    return Components(
        discover_dataset=_unavailable("discover_dataset"),
        dataset_factory=_unavailable("dataset_factory"),
        load_verification_unit=_unavailable("load_verification_unit"),
        load_full_input=_unavailable("load_full_input"),
        make_models=_unavailable("make_models"),
        producer_loss=_unavailable("producer_loss"),
        student_loss=_unavailable("student_loss"),
        generate_bank=_unavailable("generate_bank"),
        audit_bank=_unavailable("audit_bank"),
        observation_model=observation_factory,
        run_bank_experiments=_unavailable("run_bank_experiments"),
        matched_coverage=_unavailable("matched_coverage"),
        verify_frozen_bank=_unavailable("verify_frozen_bank"),
        export_prediction=_unavailable("export_prediction"),
        freeze_predictions=_unavailable("freeze_predictions"),
        validate_freeze=_unavailable("validate_freeze"),
    )


def _config(tmp_path: Path, **overrides) -> MaskfreeConfig:
    values = {
        "dataset": "acdc",
        "data_root": str(tmp_path / "images"),
        "output_dir": str(tmp_path / "runs"),
        "device": "cpu",
        "allow_cpu": True,
        "amp": False,
        "run_id": "audit-device-test",
    }
    values.update(overrides)
    return MaskfreeConfig(**values)


def _resume_payload(trainer: MaskfreeTrainer) -> dict:
    """Build a complete minimal checkpoint so ``_load_checkpoint`` runs fully."""
    return {
        "config": trainer.config.to_dict(),
        "identity": trainer.identity(),
        "completed": False,
        "models": {},
        "optimizers": {},
        "scaler": None,
        "rng": runtime.capture_rng_state(),
        "sampling_generator": trainer.sampling_generator.get_state(),
        "global_step": 0,
        "component_steps": dict(trainer.component_steps),
        "micro_batches_seen": 0,
        "last_completed_epoch": -1,
        "history": [],
        "epoch": 0,
        "batch_cursor": 0,
        "partial_epoch": None,
        "epoch_permutation": [],
        "timing": {"seconds": {"checkpoint.load": 1.25}, "calls": {"checkpoint.load": 1}},
        "log_offsets": {},
    }


def test_config_accepts_audit_device_and_production_templates_pin_auto() -> None:
    root = Path(__file__).resolve().parents[1]
    for dataset in ("acdc", "mnms"):
        config = load_config(root / f"configs/maskfree_{dataset}_150.yaml")
        assert config.audit_device == "auto"
        assert config.to_dict()["audit_device"] == "auto"

    assert _config(Path("/tmp"), audit_device="cpu").audit_device == "cpu"
    with pytest.raises(ConfigError, match="audit_device"):
        _config(Path("/tmp"), audit_device="mps")


def test_auto_dispatches_observation_to_resolved_model_device(tmp_path: Path) -> None:
    trainer = MaskfreeTrainer(_config(tmp_path, audit_device="auto"), components=_components())
    assert trainer.device == torch.device("cpu")
    assert trainer.audit_device == trainer.device
    assert trainer.observation.execution_device == torch.device("cpu")
    identity = trainer.identity()
    assert identity["audit_device_requested"] == "auto"
    assert identity["audit_device_identity"]["type"] == "cpu"
    assert identity["audit_numerical_backend"] == "torch.float64"
    assert identity["audit_execution"]["execution_device_source"] == "trainer_model_device"


def test_explicit_unavailable_cuda_audit_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(TrainerContractError, match="audit_device='cuda'.*unavailable"):
        MaskfreeTrainer(
            _config(tmp_path, audit_device="cuda"),
            components=_components(),
        )


def test_resolved_cuda_requires_execution_device_capable_observation_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _LegacyObservation:
        def __init__(self, *, max_iterations=5, variance_floor=0.05, beta=0.01):
            self.max_iterations = max_iterations
            self.variance_floor = variance_floor
            self.beta = beta

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: type("Props", (), {"name": "test", "total_memory": 1, "major": 0, "minor": 0})(),
    )
    with pytest.raises(TrainerContractError, match="requires an ObservationModel factory"):
        MaskfreeTrainer(
            _config(tmp_path, audit_device="cuda"),
            components=_components(_LegacyObservation),
        )


def test_exact_resume_rejects_audit_backend_identity_change(tmp_path: Path) -> None:
    trainer = MaskfreeTrainer(_config(tmp_path), components=_components())
    stored_identity = trainer.identity()
    stored_identity["audit_numerical_backend"] = "torch.float32"
    checkpoint = tmp_path / "mismatched-audit-backend.pt"
    torch.save(
        {
            "config": trainer.config.to_dict(),
            "identity": stored_identity,
            "completed": False,
        },
        checkpoint,
    )
    with pytest.raises(ResumeIdentityError, match="audit_numerical_backend mismatch"):
        trainer._load_checkpoint(checkpoint)


def test_exact_resume_rejects_cpu_to_cuda_audit_identity_change(tmp_path: Path) -> None:
    trainer = MaskfreeTrainer(_config(tmp_path), components=_components())
    stored_identity = trainer.identity()
    stored_identity["audit_device_identity"] = {
        **stored_identity["audit_device_identity"],
        "type": "cuda",
    }
    checkpoint = tmp_path / "cpu-to-cuda-audit.pt"
    torch.save(
        {"config": trainer.config.to_dict(), "identity": stored_identity, "completed": False},
        checkpoint,
    )
    with pytest.raises(ResumeIdentityError, match="audit_device_identity mismatch"):
        trainer._load_checkpoint(checkpoint)


def test_exact_resume_rejects_missing_audit_backend_identity(tmp_path: Path) -> None:
    trainer = MaskfreeTrainer(_config(tmp_path), components=_components())
    stored_identity = trainer.identity()
    stored_identity.pop("audit_numerical_backend")
    checkpoint = tmp_path / "missing-audit-backend.pt"
    torch.save(
        {"config": trainer.config.to_dict(), "identity": stored_identity, "completed": False},
        checkpoint,
    )
    with pytest.raises(ResumeIdentityError, match="audit_numerical_backend mismatch"):
        trainer._load_checkpoint(checkpoint)


def test_load_checkpoint_restores_audit_cuda_timing_device_without_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restore path keeps audit CUDA timing placement (simulated on CPU)."""
    trainer = MaskfreeTrainer(_config(tmp_path), components=_components())
    # No CUDA runtime is required: TimingAccumulator stores placement metadata
    # and does not synchronize until a later CUDA stage.  This isolates the
    # regression where checkpoint restore replaced audit placement with the
    # neural model's CPU device.
    monkeypatch.setattr(trainer, "audit_device", torch.device("cuda"))
    checkpoint = tmp_path / "audit-cuda-timing.pt"
    torch.save(_resume_payload(trainer), checkpoint)

    trainer._load_checkpoint(checkpoint)

    assert trainer.device == torch.device("cpu")
    assert trainer.timing.device == torch.device("cuda")
    assert trainer.timing.seconds["checkpoint.load"] == pytest.approx(1.25)
    assert trainer.timing.calls["checkpoint.load"] == 1


def test_profiler_exposes_audit_device_override() -> None:
    import scripts.profile_maskfree as profiler

    args = profiler.build_parser().parse_args(
        ["--synthetic", "--audit-device", "cpu"]
    )
    assert args.audit_device == "cpu"
