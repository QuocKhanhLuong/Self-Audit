import copy
import json

import torch
from torch import nn

from self_audit.volume.config import default_volume_config
from self_audit.volume.engine import VolumeTrainer
from self_audit.volume.engine import train_volume
from self_audit.volume.fixtures import make_volume_fixture
from self_audit.volume.evaluation import evaluate_frozen


class ToyVolume(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 4, 1)

    def forward(self, image, valid_mask=None):
        return self.conv(image)


def make_batch(batch_size=1, seed=0, *, unknown=False):
    g = torch.Generator().manual_seed(seed)
    image = torch.rand(batch_size, 3, 3, 5, 5, generator=g)
    labels = torch.zeros(batch_size, 3, 5, 5, dtype=torch.long)
    valid = torch.ones_like(labels, dtype=torch.bool)
    if unknown:
        labels.fill_(255)
    return {"image": image, "labels": labels, "geometric_valid": valid, "label_valid": valid}


def cfg(**updates):
    c = default_volume_config()
    c["training"].update(updates)
    return c


def test_volume_mean_accumulation_executes_and_matches_single_logical_batch():
    torch.manual_seed(7)
    base = ToyVolume()
    a = ToyVolume(); a.load_state_dict(copy.deepcopy(base.state_dict()))
    b = ToyVolume(); b.load_state_dict(copy.deepcopy(base.state_dict()))
    full_cfg = cfg(method="supervised", accumulation=1, microbatch=2)
    accum_cfg = cfg(method="supervised", accumulation=2, microbatch=1)
    full_cfg["loss"]["mode"] = accum_cfg["loss"]["mode"] = "volume_mean"
    full = VolumeTrainer(a, full_cfg)
    accum = VolumeTrainer(b, accum_cfg)
    batch = make_batch(batch_size=2)
    full.transaction(batch)
    accum.transaction(batch)
    for left, right in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-5)


def test_empty_supervision_skips_optimizer_weight_decay_and_ema():
    model = ToyVolume()
    before = copy.deepcopy(model.state_dict())
    trainer = VolumeTrainer(model, cfg(method="supervised"))
    record = trainer.transaction(make_batch(unknown=True))
    assert record["skip_reason"] == "empty_supervision"
    assert trainer.supervised_updates == trainer.ema_updates == 0
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert all(not opt.state for opt in trainer.optimizers)


def test_c01_supervised_extra_commits_second_labeled_update_before_one_ema():
    trainer = VolumeTrainer(ToyVolume(), cfg(method="supervised_extra"))
    batch = make_batch()
    record = trainer.transaction(batch, extra_labeled_batch=make_batch(seed=3))
    assert record["supervised_extra_updates"] == 1
    assert trainer.supervised_updates == trainer.supervised_extra_updates == 1
    assert trainer.optimizer_updates == 2
    assert trainer.ema_updates == 1


def test_c01_empty_extra_support_skips_second_update_with_complete_receipt():
    trainer = VolumeTrainer(ToyVolume(), cfg(method="supervised_extra"))
    batch = make_batch()
    empty = make_batch(seed=3, unknown=True)
    record = trainer.transaction(batch, extra_labeled_batch=empty)
    assert record["supervised_extra_updates"] == 0
    assert record["supervised_extra_update_norm"] == 0.0
    assert record["supervised_extra_grad_norm"] == 0.0
    assert record["extra_label_voxels"] == 0
    assert trainer.supervised_updates == 1
    assert trainer.optimizer_updates == 1
    assert trainer.ema_updates == 1


def test_region_and_temporal_auxiliaries_update_with_zero_pseudo_support():
    c = cfg(method="ema_full", pseudo_threshold=1.0)
    c["regions"].update({"mode": "image", "weight": 1.0, "target_num_regions": 4, "min_region_size": 2})
    c["temporal"].update({"mode": "identity", "weight": 1.0})
    trainer = VolumeTrainer(ToyVolume(), c)
    batch = make_batch()
    valid = batch["geometric_valid"]
    unlabeled = {"image": batch["image"], "geometric_valid": valid,
                 "adjacent_image": batch["image"].flip(-1),
                 "adjacent_geometric_valid": valid,
                 "metadata": [{"spacing_xyz_mm": [1.0, 1.0, 1.0]}]}
    record = trainer.transaction(batch, unlabeled)
    assert record["pseudo_accepted"] == 0
    assert record["pseudo_accepted_per_class"] == [0, 0, 0, 0]
    assert record["teacher_student_disagreement"] == 0.0
    assert record["pseudo_updates"] == 1
    assert record["aux_grad_witness"] > 0
    assert record["region_loss"] > 0
    assert record["temporal_loss"] > 0


def test_aux_support_counts_are_additive_across_microbatch_layouts():
    base = ToyVolume()
    left = ToyVolume(); left.load_state_dict(copy.deepcopy(base.state_dict()))
    right = ToyVolume(); right.load_state_dict(copy.deepcopy(base.state_dict()))
    one = cfg(method="ema_full", pseudo_threshold=1.0, microbatch=2, accumulation=1)
    two = cfg(method="ema_full", pseudo_threshold=1.0, microbatch=1, accumulation=2)
    one["temporal"].update({"mode": "identity", "weight": 1.0})
    two["temporal"].update({"mode": "identity", "weight": 1.0})
    one["loss"]["mode"] = two["loss"]["mode"] = "volume_mean"
    first = VolumeTrainer(left, one)
    second = VolumeTrainer(right, two)
    batch = make_batch(batch_size=2)
    valid = batch["geometric_valid"].clone()
    valid[0].view(-1)[:51] = False  # 24 supported voxels
    valid[1].view(-1)[:73] = False  # 2 supported voxels
    unlabeled = {"image": batch["image"], "geometric_valid": valid,
                 "adjacent_image": batch["image"], "adjacent_geometric_valid": valid,
                 "metadata": [{"spacing_xyz_mm": [1.0, 1.0, 1.0]}, {"spacing_xyz_mm": [1.0, 1.0, 1.0]}]}
    rec_one = first.transaction(batch, unlabeled)
    rec_two = second.transaction(batch, unlabeled)
    assert rec_one["temporal_valid_voxels"] == rec_two["temporal_valid_voxels"] == 26
    assert rec_one["temporal_valid_fraction"] == rec_two["temporal_valid_fraction"]
    assert torch.allclose(left.conv.weight, right.conv.weight, atol=1e-6, rtol=1e-5)


def test_frozen_evaluation_does_not_mutate_source_model():
    model = ToyVolume()
    model.train()
    before = copy.deepcopy(model.state_dict())
    batch = make_batch()
    result = evaluate_frozen(model, [batch], prediction_model="student", device="cpu")
    assert result["prediction_model"] == "student"
    assert model.training is True
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_train_volume_stop_target_preserves_scheduler_horizon_and_save_cadence(tmp_path):
    manifest, manifest_sha, geometry, geometry_sha = make_volume_fixture(tmp_path / "fixture", shape_tzyx=(3, 3, 8, 8))
    c = cfg(method="supervised", steps=4, save_every=1, threads=1)
    c["geometry"]["crop_size"] = 8
    receipt = train_volume(
        c, synthetic=True, steps=2, device="cpu", out=str(tmp_path / "run"),
        manifest=str(manifest), manifest_sha256=manifest_sha,
        geometry_path=str(geometry), geometry_sha256=geometry_sha,
    )
    assert receipt["steps"] == 2
    assert receipt["schedule_steps"] == 4
    assert (tmp_path / "run.step000001").is_dir()
    manifest_out = json.loads((tmp_path / "run" / "MANIFEST.json").read_text())
    assert manifest_out["config"]["training"]["steps"] == 4
    assert manifest_out["scheduler_total_steps"] == [4]
