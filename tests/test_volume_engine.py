import copy

import pytest
import torch
from torch import nn

from self_audit.volume.config import default_volume_config
from self_audit.volume.engine import VolumeTrainer


class ToyVolume(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 4, 1)

    def forward(self, image, valid_mask=None):
        return self.conv(image)


def cfg(**training):
    c = default_volume_config()
    c["training"].update(training)
    return c


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    image = torch.rand(1, 3, 3, 5, 7, generator=g)
    labels = torch.zeros(1, 3, 5, 7, dtype=torch.long)
    valid = torch.ones_like(labels, dtype=torch.bool)
    return {"image": image, "labels": labels, "geometric_valid": valid, "label_valid": valid}


def test_transaction_updates_in_declared_order_and_ema_once():
    torch.manual_seed(2)
    model = ToyVolume()
    initial_teacher = copy.deepcopy(model.state_dict())
    trainer = VolumeTrainer(model, cfg(pseudo_threshold=0.0))
    record = trainer.transaction(batch(), {"image": batch(1)["image"], "geometric_valid": torch.ones(1, 3, 5, 7, dtype=torch.bool)})
    assert trainer.supervised_updates == 1
    assert trainer.pseudo_updates == 1
    assert trainer.ema_updates == 1
    assert record["supervised_update_norm"] > 0
    assert record["pseudo_update_norm"] > 0
    assert record["pseudo_accepted"] == 105
    assert any(not torch.equal(initial_teacher[k], v) for k, v in trainer.teacher.state_dict().items())
    assert all(not p.requires_grad for p in trainer.teacher.parameters())


def test_empty_pseudo_support_does_not_optimizer_or_ema_skip():
    trainer = VolumeTrainer(ToyVolume(), cfg(pseudo_threshold=1.0))
    record = trainer.transaction(batch(), {"image": batch(1)["image"], "geometric_valid": torch.ones(1, 3, 5, 7, dtype=torch.bool)})
    assert trainer.supervised_updates == 1
    assert trainer.pseudo_updates == 0
    assert trainer.skipped_pseudo_updates == 1
    assert trainer.ema_updates == 1
    assert record["pseudo_accepted"] <= 105


def test_supervised_method_has_no_pseudo_transaction_but_updates_ema():
    trainer = VolumeTrainer(ToyVolume(), cfg(method="supervised"))
    trainer.transaction(batch(), {"image": batch(1)["image"], "geometric_valid": torch.ones(1, 3, 5, 7, dtype=torch.bool)})
    assert trainer.supervised_updates == 1
    assert trainer.pseudo_updates == 0
    assert trainer.ema_updates == 1


def test_zero_pseudo_weight_does_not_take_adam_step_or_create_pseudo_state():
    trainer = VolumeTrainer(ToyVolume(), cfg(method="ema_full", pseudo_threshold=0.0, pseudo_weight=0.0))
    b = batch()
    trainer.transaction(b, {"image": b["image"], "geometric_valid": b["geometric_valid"]})
    assert trainer.pseudo_updates == 0
    assert trainer.optimizer_updates == len(trainer.optimizers)
    assert trainer.skipped_pseudo_updates == 1
    assert all(len(optimizer.state) > 0 for optimizer in trainer.optimizers)


def test_scheduler_warmup_is_applied_before_first_optimizer_update():
    c = cfg(method="supervised")
    c["optimization"].update({"scheduler": "cosine", "lr_warmup_steps": 2, "min_lr_ratio": 0.1})
    trainer = VolumeTrainer(ToyVolume(), c)
    base_lr = float(trainer.optimizers[0].param_groups[0]["lr"])
    trainer.transaction(batch())
    assert trainer.schedulers[0].last_outer_iteration == 0
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(base_lr / 2.0)


def test_pseudo_coverage_uses_valid_voxel_denominator():
    trainer = VolumeTrainer(ToyVolume(), cfg(method="ema_full", pseudo_threshold=0.0))
    b = batch()
    valid = b["geometric_valid"].clone()
    valid[..., :2, :] = False
    record = trainer.transaction(b, {"image": b["image"], "geometric_valid": valid})
    assert record["pseudo_valid_voxels"] == int(valid.sum())
    assert record["pseudo_coverage"] == pytest.approx(1.0)


def test_amp_fails_closed_on_cpu_instead_of_silently_running_fp32():
    c = cfg(method="supervised", precision="amp")
    with pytest.raises(RuntimeError, match="AMP requires CUDA"):
        VolumeTrainer(ToyVolume(), c)


def test_microbatch_configuration_cannot_be_silently_ignored():
    trainer = VolumeTrainer(ToyVolume(), cfg(method="supervised", microbatch=2, accumulation=1))
    with pytest.raises(ValueError, match="microbatch cannot be ignored"):
        trainer.transaction(batch())
