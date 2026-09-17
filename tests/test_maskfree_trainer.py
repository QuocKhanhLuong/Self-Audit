"""Focused W5 checks: resume parity, resume identity policy, completion honesty.

These use synthetic CPU fixtures. They are software evidence about the trainer's
control flow and state handling only; no real cardiac data, GPU or cine metadata
exists in this checkout, so nothing here says anything about label quality.

The observation model under test is W1's real ``ObservationModel``; the data,
model, bank and export layers are deliberately small local doubles that honour
the frozen interfaces from ``reports/maskfree150/architecture_contract.md``, so
the trainer is exercised against the contract rather than against a stub of
itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from self_audit_maskfree.config import MaskfreeConfig
from self_audit_maskfree.contracts import FittingView, ScoringView, TrainingUnit
from self_audit_maskfree.auditor import audit_bank
from self_audit_maskfree.hypotheses import generate_bank
from self_audit_maskfree.losses import producer_loss, student_loss
from self_audit_maskfree.models import make_models
from self_audit_maskfree.observation import ObservationModel
from self_audit_maskfree.trainer import (
    STATUS_PARTIAL,
    Components,
    MaskfreeTrainer,
    ResumeIdentityError,
    TrainerContractError,
)

IMAGE_SIZE = 32
BLOCK = 8
UNITS = 4


# ---------------------------------------------------------------------------
# synthetic data honouring the W3 interface
# ---------------------------------------------------------------------------
def _role_masks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed 8px block roles: two fit blocks, one select block, one verify block."""
    fit = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    select = torch.zeros_like(fit)
    verify = torch.zeros_like(fit)
    half = IMAGE_SIZE // 2
    fit[:half, :] = True
    select[half:, :half] = True
    verify[half:, half:] = True
    return fit, select, verify


def _synthetic_image(index: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1000 + index)
    base = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, IMAGE_SIZE), torch.linspace(-1, 1, IMAGE_SIZE), indexing="ij"
    )
    radius = (grid_x**2 + grid_y**2).sqrt()
    base += (radius < 0.35).float() * 1.5          # inner pool
    base += ((radius >= 0.35) & (radius < 0.6)).float() * 0.7  # wall
    base += torch.randn(IMAGE_SIZE, IMAGE_SIZE, generator=generator) * 0.05
    return base


def discover_dataset(root, dataset, *, seed=42, protocol="auto", depth_axis=2):
    records = []
    for index in range(UNITS):
        records.append(
            {
                "unit_id": f"unit-{index:03d}",
                "study_id": f"study-{index:03d}",
                "patient_id": f"patient-{index // 2:03d}",
                "split": "train" if index < 3 else "test",
                "path": f"{root}/synthetic/{index}.npy",
                "native_geometry": False,
                "shape": [IMAGE_SIZE, IMAGE_SIZE],
            }
        )
    return {
        "schema_version": "maskfree150.data.test",
        "dataset": dataset,
        "manifest_id": f"synthetic-{dataset}-{seed}",
        "records": records,
        "resolved_protocol": "spatial_predictive",
        "resolved_protocol_reason": "synthetic fixture: no cine frames",
        "readiness": {"masks_read": False},
        "limitations": ["synthetic fixture, not real data"],
    }


class FakeImageOnlyDataset:
    def __init__(self, manifest, split="train", image_size=IMAGE_SIZE, seed=42):
        self.records = [r for r in manifest["records"] if r["split"] == split]
        self.image_size = image_size
        self.seed = seed
        self.partition_id = f"{manifest['manifest_id']}:{split}"

    def __len__(self) -> int:
        return len(self.records)

    @property
    def unit_ids(self) -> list[str]:
        return [record["unit_id"] for record in self.records]

    def __getitem__(self, index: int) -> TrainingUnit:
        record = self.records[index]
        image = _synthetic_image(int(record["unit_id"].split("-")[1]))
        fit, select, _verify = _role_masks()
        fit_values = image[fit]
        mean = float(fit_values.mean())
        std = float(fit_values.std().clamp_min(1e-6))
        normalized = (image - mean) / std

        fit_image = torch.where(fit, normalized, torch.zeros_like(normalized)).unsqueeze(0)
        context = fit_image.repeat(3, 1, 1)
        select_image = torch.where(select, normalized, torch.zeros_like(normalized)).unsqueeze(0)
        fitting = FittingView(
            image=fit_image.contiguous(),
            support=fit.clone(),
            context=context.contiguous(),
            study_id=record["study_id"],
            unit_id=record["unit_id"],
            protocol="spatial_predictive",
            metadata={"normalization": {"mean": mean, "std": std}},
            partition_id=self.partition_id,
        )
        selection = ScoringView(
            image=select_image.contiguous(),
            support=select.clone(),
            study_id=record["study_id"],
            unit_id=record["unit_id"],
            role="select",
            partition_id=self.partition_id,
        )
        return TrainingUnit(fitting=fitting, selection=selection, record=dict(record))


def load_verification_unit(manifest, unit_id, freeze_manifest):
    raise RuntimeError("verification fixture not exercised by these checks")


def load_full_input(manifest, unit_id):
    raise RuntimeError("deployment fixture not exercised by these checks")


# ---------------------------------------------------------------------------
# inert W6/W7 doubles (finalization is not reached by bounded checks)
# ---------------------------------------------------------------------------
def _not_exercised(name):
    def _raise(*args, **kwargs):
        raise RuntimeError(f"{name} is not exercised by these focused checks")

    return _raise


def build_components() -> Components:
    return Components(
        discover_dataset=discover_dataset,
        dataset_factory=FakeImageOnlyDataset,
        load_verification_unit=load_verification_unit,
        load_full_input=load_full_input,
        make_models=make_models,
        producer_loss=producer_loss,
        student_loss=student_loss,
        generate_bank=generate_bank,
        audit_bank=audit_bank,
        observation_model=ObservationModel,
        run_bank_experiments=_not_exercised("run_bank_experiments"),
        matched_coverage=_not_exercised("matched_coverage"),
        verify_frozen_bank=_not_exercised("verify_frozen_bank"),
        export_prediction=_not_exercised("export_prediction"),
        freeze_predictions=_not_exercised("freeze_predictions"),
        validate_freeze=_not_exercised("validate_freeze"),
    )


def build_config(tmp_path: Path, **overrides) -> MaskfreeConfig:
    payload = {
        "dataset": "acdc",
        "data_root": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "out"),
        "total_epochs": 4,
        "seed": 42,
        "batch_size": 2,
        "accumulation_steps": 1,
        "image_size": IMAGE_SIZE,
        "lr": 0.01,
        "warmup_epochs": 1,
        "width": 16,
        "feature_dim": 16,
        "device": "cpu",
        "allow_cpu": True,
        "amp": False,
        "wandb_mode": "disabled",
        "run_id": "focused-check",
        "max_epochs": 2,
    }
    payload.update(overrides)
    return MaskfreeConfig(**payload)


def _flat_state(trainer: MaskfreeTrainer) -> dict[str, torch.Tensor]:
    flat = {}
    for name, model in trainer.models.items():
        for key, value in model.state_dict().items():
            flat[f"{name}.{key}"] = value.detach().clone()
    return flat


# ---------------------------------------------------------------------------
# check 1: interruption + resume reproduces the uninterrupted state exactly
# ---------------------------------------------------------------------------
def test_interrupted_resume_matches_uninterrupted_state(tmp_path):
    uninterrupted = MaskfreeTrainer(
        build_config(tmp_path / "full"), components=build_components()
    )
    full_report = uninterrupted.run()
    assert full_report["status"] == STATUS_PARTIAL  # max_epochs is a bounded run
    assert full_report["last_completed_epoch"] == 1
    reference_state = _flat_state(uninterrupted)
    reference_steps = full_report["global_optimizer_steps"]
    assert reference_steps == 4  # 3 train units -> 2 batches/epoch, 2 epochs

    interrupted = MaskfreeTrainer(
        build_config(tmp_path / "split", max_steps=3), components=build_components()
    )
    partial_report = interrupted.run()
    assert partial_report["status"] == STATUS_PARTIAL
    assert partial_report["global_optimizer_steps"] == 3
    assert partial_report["last_completed_epoch"] == 0  # stopped inside epoch 1

    resumed = MaskfreeTrainer(
        build_config(
            tmp_path / "split",
            resume=str(interrupted.paths.last_checkpoint),
            num_workers=1,  # operational difference is allowed on resume
        ),
        components=build_components(),
    )
    resumed_report = resumed.run()

    assert resumed_report["global_optimizer_steps"] == reference_steps
    assert resumed_report["last_completed_epoch"] == 1
    assert resumed_report["component_steps"] == full_report["component_steps"]
    resumed_state = _flat_state(resumed)
    assert set(resumed_state) == set(reference_state)
    for key, value in reference_state.items():
        assert torch.equal(value, resumed_state[key]), f"state divergence at {key}"

    # The continued epoch must not replay units it already consumed, and the
    # truncated record of the interrupted epoch is replaced, not double counted.
    assert [record["global_epoch"] for record in resumed.history] == [0, 1]
    assert resumed.history[-1]["resumed_from_batch"] == 1
    assert resumed.history[-1]["epoch_complete"] is True
    assert resumed.history[-1]["units_visited"] == uninterrupted.history[-1]["units_visited"]
    assert resumed.history[-1]["audit"] == uninterrupted.history[-1]["audit"]
    assert resumed.history[-1]["coverage"] == uninterrupted.history[-1]["coverage"]


# ---------------------------------------------------------------------------
# check 2: scientific mismatch fails closed, operational difference resumes
# ---------------------------------------------------------------------------
def test_resume_separates_scientific_from_operational_differences(tmp_path):
    seed_run = MaskfreeTrainer(
        build_config(tmp_path / "base", max_steps=2), components=build_components()
    )
    seed_run.run()
    checkpoint = str(seed_run.paths.last_checkpoint)
    checkpoint_bytes_before = Path(checkpoint).read_bytes()

    # Layer 1: the run directory itself is bound to one scientific identity.
    with pytest.raises(ResumeIdentityError, match="already belongs to another run"):
        MaskfreeTrainer(
            build_config(tmp_path / "base", lr=0.02, resume=checkpoint),
            components=build_components(),
        ).setup()

    # Layer 2: even in a clean directory, the checkpoint refuses a different
    # scientific configuration.
    mismatched = MaskfreeTrainer(
        build_config(tmp_path / "other", lr=0.02, resume=checkpoint),
        components=build_components(),
    )
    mismatched.setup()
    with pytest.raises(ResumeIdentityError, match="scientific configuration mismatch"):
        mismatched._load_checkpoint(checkpoint)
    assert Path(checkpoint).read_bytes() == checkpoint_bytes_before  # no overwrite on refusal

    # A different dataset root is a different experiment, not an operational knob.
    other_data = MaskfreeTrainer(
        build_config(tmp_path / "elsewhere", data_root=str(tmp_path / "other-data"),
                     resume=checkpoint),
        components=build_components(),
    )
    other_data.setup()
    with pytest.raises(ResumeIdentityError):
        other_data._load_checkpoint(checkpoint)

    # A checkpoint written by different package sources is refused too: the
    # resume must restore the same experiment, not migrate state across builds.
    tampered = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tampered["identity"] = dict(tampered["identity"], source_hash="0" * 64)
    tampered_path = Path(checkpoint).with_name("other_source.pt")
    torch.save(tampered, tampered_path)
    other_source = MaskfreeTrainer(
        build_config(tmp_path / "base", resume=str(tampered_path)),
        components=build_components(),
    )
    other_source.setup()
    with pytest.raises(ResumeIdentityError, match="source_hash mismatch"):
        other_source._load_checkpoint(str(tampered_path))

    # Operational differences resume, and restore the exact position.
    operational = MaskfreeTrainer(
        build_config(tmp_path / "base", resume=checkpoint, num_workers=2, max_steps=None),
        components=build_components(),
    )
    operational.setup()
    operational._load_checkpoint(checkpoint)
    assert operational.global_step == 2
    assert operational.start_epoch == 1
    assert operational.start_batch == 0
    assert Path(checkpoint).read_bytes() == checkpoint_bytes_before


# ---------------------------------------------------------------------------
# check 3: a bounded smoke can never present itself as a finished run
# ---------------------------------------------------------------------------
def test_bounded_smoke_cannot_masquerade_as_completion(tmp_path):
    trainer = MaskfreeTrainer(
        build_config(tmp_path / "smoke", max_steps=1), components=build_components()
    )
    report = trainer.run()

    assert report["completed"] is False
    assert report["status"] == STATUS_PARTIAL
    assert report["bounded_run"] is True
    assert report["epochs_completed"] < report["total_epochs"]
    assert report["finalization"]["available"] is False

    payload = torch.load(trainer.paths.last_checkpoint, map_location="cpu", weights_only=False)
    assert payload["completed"] is False
    assert payload["bounded_run"] is True
    assert not list(trainer.paths.checkpoints.glob("*_final.pt"))

    capped_resume = MaskfreeTrainer(
        build_config(tmp_path / "smoke", max_steps=1,
                     resume=str(trainer.paths.last_checkpoint)),
        components=build_components(),
    )
    capped_report = capped_resume.run()
    assert capped_report["global_optimizer_steps"] == 1
    for key, value in _flat_state(trainer).items():
        assert torch.equal(value, _flat_state(capped_resume)[key])

    # The completion flag itself is guarded, not merely unset by accident.
    with pytest.raises(TrainerContractError, match="never be marked completed"):
        trainer._save_checkpoint(
            epoch=trainer.config.total_epochs,
            batch_cursor=0,
            permutation=[],
            completed=True,
            status="completed",
        )

    # And a completed checkpoint cannot be resumed into more training.
    payload["completed"] = True
    torch.save(payload, trainer.paths.checkpoints / "fake_completed.pt")
    replay = MaskfreeTrainer(
        build_config(tmp_path / "smoke", resume=str(trainer.paths.checkpoints / "fake_completed.pt")),
        components=build_components(),
    )
    replay.setup()
    with pytest.raises(ResumeIdentityError, match="already marked completed"):
        replay._load_checkpoint(str(trainer.paths.checkpoints / "fake_completed.pt"))
