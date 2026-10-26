import random
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from self_audit.volume.checkpoint import load_volume_checkpoint, save_volume_checkpoint
from self_audit.volume.config import default_volume_config
from self_audit.volume.engine import VolumeTrainer
from self_audit.volume.engine import train_volume
from self_audit.volume.fixtures import make_volume_fixture


MANIFEST_PIN = "a" * 64
GEOMETRY_PIN = "b" * 64


class ToyVolume(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 4, 1)

    def forward(self, image, valid_mask=None):
        return self.conv(image)


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    image = torch.rand(1, 3, 2, 4, 4, generator=g)
    labels = torch.zeros(1, 2, 4, 4, dtype=torch.long)
    valid = torch.ones_like(labels, dtype=torch.bool)
    return {"image": image, "labels": labels, "geometric_valid": valid, "label_valid": valid}


def make_cfg():
    c = default_volume_config()
    c["training"].update({"steps": 2, "method": "supervised", "threads": 1})
    return c


def test_checkpoint_resume_restores_state_and_rng(tmp_path):
    torch.manual_seed(4)
    np.random.seed(4)
    random.seed(4)
    cfg = make_cfg()
    cfg["training"]["method"] = "ema_full"
    trainer = VolumeTrainer(ToyVolume(), cfg)
    trainer.transaction(make_batch(1))
    trainer.sampler_state = {"numpy_rng": np.random.default_rng(4).bit_generator.state, "iteration": trainer.outer_step}
    folder = tmp_path / "ckpt"
    provenance = {"synthetic": True, "manifest_sha256": MANIFEST_PIN, "geometry_sha256": GEOMETRY_PIN}
    sha = save_volume_checkpoint(folder, trainer, cfg, data_provenance=provenance)
    before = trainer.state_dict()
    resumed, manifest = load_volume_checkpoint(folder, sha, model=ToyVolume(), device="cpu")
    assert manifest["schema"] == "self_audit.volume.checkpoint.v1"
    assert resumed.outer_step == before["outer_step"]
    assert resumed.supervised_updates == before["supervised_updates"]
    assert resumed.ema_updates == before["ema_updates"]
    for left, right in zip(trainer.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(left, right)
    assert resumed.teacher.training is False


def test_checkpoint_payload_requires_complete_rng_and_trainer_state(tmp_path):
    cfg = make_cfg()
    trainer = VolumeTrainer(ToyVolume(), cfg)
    trainer.transaction(make_batch(1))
    trainer.sampler_state = {"numpy_rng": np.random.default_rng(4).bit_generator.state, "iteration": trainer.outer_step}
    folder = tmp_path / "ckpt"
    provenance = {"synthetic": True, "manifest_sha256": MANIFEST_PIN, "geometry_sha256": GEOMETRY_PIN}
    sha = save_volume_checkpoint(folder, trainer, cfg, data_provenance=provenance)
    payload_path = folder / "weights.pt"
    payload = torch.load(payload_path, weights_only=False)
    del payload["rng"]["torch"]
    torch.save(payload, payload_path)
    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_volume_checkpoint(folder, sha, model=ToyVolume(), device="cpu")


def test_checkpoint_rejects_invalid_state_before_mutating_supplied_model(tmp_path):
    cfg = make_cfg()
    trainer = VolumeTrainer(ToyVolume(), cfg)
    trainer.transaction(make_batch(1))
    trainer.sampler_state = {"numpy_rng": np.random.default_rng(8).bit_generator.state, "iteration": trainer.outer_step}
    folder = tmp_path / "ckpt"
    save_volume_checkpoint(folder, trainer, cfg, data_provenance={
        "synthetic": True, "manifest_sha256": MANIFEST_PIN, "geometry_sha256": GEOMETRY_PIN,
    })
    payload_path = folder / "weights.pt"
    payload = torch.load(payload_path, weights_only=False)
    payload["trainer"]["sampler_state"] = None
    torch.save(payload, payload_path)
    manifest_path = folder / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["payload_sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    supplied = ToyVolume()
    before = {key: value.clone() for key, value in supplied.state_dict().items()}
    with pytest.raises(ValueError, match="sampler_state"):
        load_volume_checkpoint(folder, model=supplied, device="cpu")
    assert all(torch.equal(before[key], value) for key, value in supplied.state_dict().items())


def test_region_cache_state_is_required_before_model_mutation(tmp_path):
    cfg = make_cfg()
    cfg["training"]["method"] = "ema_full"
    cfg["regions"].update({"mode": "image", "weight": 1.0})
    trainer = VolumeTrainer(ToyVolume(), cfg)
    trainer.transaction(make_batch(1))
    trainer.sampler_state = {"numpy_rng": np.random.default_rng(9).bit_generator.state, "iteration": trainer.outer_step}
    folder = tmp_path / "region_ckpt"
    save_volume_checkpoint(folder, trainer, cfg, data_provenance={
        "synthetic": True, "manifest_sha256": MANIFEST_PIN, "geometry_sha256": GEOMETRY_PIN,
    })
    payload_path = folder / "weights.pt"
    payload = torch.load(payload_path, weights_only=False)
    payload["trainer"]["region_state"] = None
    torch.save(payload, payload_path)
    manifest_path = folder / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["payload_sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    supplied = ToyVolume()
    before = {key: value.clone() for key, value in supplied.state_dict().items()}
    with pytest.raises(ValueError, match="region cache"):
        load_volume_checkpoint(folder, model=supplied, device="cpu")
    assert all(torch.equal(before[key], value) for key, value in supplied.state_dict().items())


def _reseal_payload(folder, payload):
    payload_path = folder / "weights.pt"
    torch.save(payload, payload_path)
    manifest_path = folder / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["payload_sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("python", ()),
        ("numpy", ("MT19937", np.zeros(1, dtype=np.uint32), 0, 0, 0.0)),
        ("torch", torch.empty(0, dtype=torch.uint8)),
    ],
)
def test_malformed_rng_state_fails_before_caller_or_global_mutation(tmp_path, field, bad_value):
    cfg = make_cfg()
    trainer = VolumeTrainer(ToyVolume(), cfg)
    trainer.transaction(make_batch(2))
    trainer.sampler_state = {"numpy_rng": np.random.default_rng(4).bit_generator.state, "iteration": trainer.outer_step}
    folder = tmp_path / f"bad_{field}"
    save_volume_checkpoint(folder, trainer, cfg, data_provenance={
        "synthetic": True, "manifest_sha256": MANIFEST_PIN, "geometry_sha256": GEOMETRY_PIN,
    })
    payload = torch.load(folder / "weights.pt", weights_only=False)
    payload["rng"][field] = bad_value
    _reseal_payload(folder, payload)

    supplied = ToyVolume()
    before = {key: value.clone() for key, value in supplied.state_dict().items()}
    random.seed(100 + len(field))
    np.random.seed(200 + len(field))
    torch.manual_seed(300 + len(field))
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state()
    with pytest.raises(ValueError, match="RNG"):
        load_volume_checkpoint(folder, model=supplied, device="cpu")
    assert all(torch.equal(before[key], value) for key, value in supplied.state_dict().items())
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0] and np.array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize(
    "provenance",
    [
        {"synthetic": 1, "manifest_sha256": MANIFEST_PIN, "geometry_sha256": GEOMETRY_PIN},
        {"synthetic": True, "manifest_sha256": None, "geometry_sha256": GEOMETRY_PIN},
        {"synthetic": True, "manifest_sha256": "not-a-sha", "geometry_sha256": GEOMETRY_PIN},
        {"synthetic": True, "manifest_sha256": MANIFEST_PIN, "geometry_sha256": None},
    ],
)
def test_checkpoint_save_requires_exact_manufactured_provenance_pins(tmp_path, provenance):
    cfg = make_cfg()
    trainer = VolumeTrainer(ToyVolume(), cfg)
    with pytest.raises(ValueError, match="provenance"):
        save_volume_checkpoint(tmp_path / "invalid", trainer, cfg, data_provenance=provenance)


def test_checkpoint_load_rejects_null_provenance_pin_after_reseal(tmp_path):
    cfg = make_cfg()
    trainer = VolumeTrainer(ToyVolume(), cfg)
    folder = tmp_path / "ckpt"
    save_volume_checkpoint(folder, trainer, cfg, data_provenance={
        "synthetic": True, "manifest_sha256": MANIFEST_PIN, "geometry_sha256": GEOMETRY_PIN,
    })
    manifest_path = folder / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["data_provenance"]["geometry_sha256"] = None
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="provenance"):
        load_volume_checkpoint(folder, model=ToyVolume(), device="cpu")


def test_train_volume_identity_mismatch_fails_before_global_rng_restore(tmp_path):
    manifest, manifest_sha, geometry, geometry_sha = make_volume_fixture(
        tmp_path / "fixture", shape_tzyx=(2, 2, 8, 8)
    )
    cfg = make_cfg()
    cfg["training"].update({"steps": 2, "save_every": 1})
    cfg["geometry"]["crop_size"] = 8
    first = train_volume(
        cfg,
        synthetic=True,
        steps=1,
        device="cpu",
        out=str(tmp_path / "run"),
        manifest=str(manifest),
        manifest_sha256=manifest_sha,
        geometry_path=str(geometry),
        geometry_sha256=geometry_sha,
    )
    random.seed(801)
    np.random.seed(802)
    torch.manual_seed(803)
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state()
    with pytest.raises(ValueError, match="geometry_sha256"):
        train_volume(
            cfg,
            synthetic=True,
            steps=2,
            device="cpu",
            resume=str(tmp_path / "run"),
            resume_sha256=first["checkpoint_manifest_sha256"],
            manifest=str(manifest),
            manifest_sha256=manifest_sha,
            geometry_path=str(geometry),
            geometry_sha256="0" * 64,
        )
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0] and np.array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
