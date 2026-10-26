import copy

import pytest
import torch
from torch import nn

from self_audit.volume.optim import (
    OuterIterationScheduler,
    build_optimizers,
    build_schedulers,
    safe_optimizer_step,
)


class ToyVolumeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv3d(3, 4, 1)
        self.decoder = nn.Conv3d(4, 4, 1)
        self.head = nn.Conv3d(4, 4, 1)

    def parameter_groups(self):
        return {
            "encoder": list(self.encoder.parameters()),
            "body": list(self.decoder.parameters()),
            "head": list(self.head.parameters()),
        }


def test_grouped_adamw_exposes_nonduplicated_rates_and_metadata():
    model = ToyVolumeModel()
    optimizers, metadata = build_optimizers(model, {
        "recipe": "grouped_adamw", "encoder_lr": 1e-4, "decoder_lr": 2e-4,
        "head_lr": 3e-4, "body_lr": 1e-4, "weight_decay": 1e-2,
    })
    assert len(optimizers) == 1
    assert [group["lr"] for group in optimizers[0].param_groups] == [1e-4, 2e-4, 3e-4]
    assert [record["name"] for record in metadata["groups"]] == ["encoder", "decoder", "head"]
    names = [name for record in metadata["groups"] for name in record["parameter_names"]]
    assert len(names) == len(set(names)) == len(list(model.named_parameters()))
    assert metadata["scheduler_index"] == "outer_iteration"
    with pytest.raises(ValueError, match="unknown optimization fields"):
        build_optimizers(model, {"bogus": 1})


def test_legacy_recipe_keeps_head_sgd_separate_and_scheduler_warmup_delay():
    model = ToyVolumeModel()
    optimizers, metadata = build_optimizers(model, {"recipe": "legacy_adamw_sgd", "body_lr": 1e-4, "head_lr": 1e-2})
    assert len(optimizers) == 2
    assert isinstance(optimizers[0], torch.optim.AdamW)
    assert isinstance(optimizers[1], torch.optim.SGD)
    assert optimizers[1].defaults["momentum"] == 0.0
    schedulers, scheduler_meta = build_schedulers(optimizers, {
        "recipe": "legacy_adamw_sgd", "lr_warmup_steps": 2, "pseudo_delay": 3,
        "scheduler": "constant", "body_lr": 1e-4, "head_lr": 1e-2,
    })
    scheduler = schedulers[0]
    assert scheduler.step(0) == pytest.approx([5e-5, 5e-3])
    assert not scheduler.pseudo_active(2)
    scheduler.step(3)
    assert scheduler.pseudo_active(3)
    assert scheduler.state_dict()["last_outer_iteration"] == 3


def test_safe_optimizer_step_does_not_decay_on_empty_support_and_records_update():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.5)
    before = copy.deepcopy(model.state_dict())
    receipt = safe_optimizer_step([optimizer], has_support=False)
    assert receipt["updated"] is False and receipt["optimizer_updates"] == 0
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())

    loss = model(torch.ones(3, 2)).square().mean()
    loss.backward()
    receipt = safe_optimizer_step([optimizer], clip_norm=1.0)
    assert receipt["updated"] is True
    assert receipt["optimizer_updates"] == 1
    assert receipt["gradient_norm"] > 0 and receipt["parameter_update_norm"] > 0


def test_scheduler_resume_restores_learning_rates():
    model = ToyVolumeModel()
    optimizers, _ = build_optimizers(model, {"scheduler": "cosine", "total_steps": 10, "body_lr": 1e-3, "head_lr": 1e-2})
    original = OuterIterationScheduler(optimizers, {"scheduler": "cosine", "total_steps": 10, "body_lr": 1e-3, "head_lr": 1e-2})
    original.step(3)
    state = original.state_dict()
    resumed = OuterIterationScheduler(optimizers, {"scheduler": "cosine", "total_steps": 10, "body_lr": 1e-3, "head_lr": 1e-2})
    resumed.load_state_dict(state)
    assert resumed.get_last_lr() == pytest.approx(original.get_last_lr())


def test_scheduler_warmup_decay_and_delay_boundaries_are_outer_iteration_exact():
    model = ToyVolumeModel()
    optimizers, _ = build_optimizers(
        model,
        {
            "scheduler": "cosine",
            "total_steps": 4,
            "lr_warmup_steps": 2,
            "min_lr_ratio": 0.1,
            "body_lr": 1e-2,
            "head_lr": 1e-1,
            "pseudo_delay": 2,
        },
    )
    scheduler = OuterIterationScheduler(
        optimizers,
        {
            "scheduler": "cosine",
            "total_steps": 4,
            "lr_warmup_steps": 2,
            "min_lr_ratio": 0.1,
            "body_lr": 1e-2,
            "head_lr": 1e-1,
            "pseudo_delay": 2,
        },
    )
    # Indexing is by outer transaction: the first update is half-rate, the
    # second completes warmup, and the final declared step reaches the floor.
    assert scheduler.step(0) == pytest.approx([5e-3, 5e-3, 5e-2])
    assert not scheduler.pseudo_active(1)
    assert scheduler.step(1) == pytest.approx([1e-2, 1e-2, 1e-1])
    assert not scheduler.pseudo_active(1)
    assert scheduler.step(2) == pytest.approx([5.5e-3, 5.5e-3, 5.5e-2])
    assert scheduler.pseudo_active(2)
    assert scheduler.step(3) == pytest.approx([1e-3, 1e-3, 1e-2])
    assert scheduler.step(4) == pytest.approx([1e-3, 1e-3, 1e-2])
    with pytest.raises(ValueError, match="cannot move backwards"):
        scheduler.step(3)
