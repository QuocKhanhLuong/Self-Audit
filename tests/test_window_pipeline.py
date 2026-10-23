"""Manufactured integration checks; no patient or Dice-improvement evidence."""
import copy
import json

import numpy as np
import pytest
import torch

from self_audit.data import TrainingData, synthetic_fixture
from self_audit.engine import (DEFAULT, full_pseudo_step, load_checkpoint,
                               supervised_step, train, validate_config)
from self_audit.losses import aligned_flip
from self_audit.meta_window import geometry_named_parameters
from self_audit.model import build_model


def pilot_recipe(arm):
    c = copy.deepcopy(DEFAULT)
    c.update(method="ema_full_meta_window" if arm == "meta" else "ema_full",
             steps=3, warmup_steps=1, batch_size=1, crop_size=40,
             pseudo_threshold=.25, body_lr=.001, head_lr=.03, save_every=1)
    c["model"]["backbone"] = "tiny_test"
    c["window"]["mode"] = {"baseline": "none", "fixed": "fixed", "meta": "adaptive"}[arm]
    c["window"]["dim"] = 4
    return c


def test_old_recipes_keep_window_disabled_and_invalid_combinations_fail():
    c = copy.deepcopy(DEFAULT)
    del c["window"], c["meta_window"]
    assert validate_config(c)["window"]["mode"] == "none"
    for mode in ("none", "fixed"):
        bad = pilot_recipe("meta"); bad["window"]["mode"] = mode
        with pytest.raises(ValueError, match="requires adaptive"):
            validate_config(bad)
    bad = pilot_recipe("fixed"); bad["method"] = "joint_guard"
    with pytest.raises(ValueError, match="not guarded"):
        validate_config(bad)
    for value in (True, 0, -1, float("nan")):
        bad = pilot_recipe("meta"); bad["meta_window"]["lr"] = value
        with pytest.raises(ValueError, match="meta_window"):
            validate_config(bad)


def test_geometric_validity_is_independent_of_labeled_class_and_flip(tmp_path):
    manifest, sha = synthetic_fixture(tmp_path / "data")
    data = TrainingData(manifest, sha, synthetic=True)
    old_rng, new_rng = np.random.default_rng(4), np.random.default_rng(4)
    old = data.sample_batch(0, old_rng, 2, labeled=True, crop_size=40)
    x, y, meta, valid = data.sample_batch(0, new_rng, 2, labeled=True, crop_size=40, return_valid=True)
    assert torch.equal(x, old[0]) and torch.equal(y, old[1]) and meta == old[2]
    assert new_rng.bit_generator.state == old_rng.bit_generator.state
    assert int(valid.sum()) == 2 * 32 * 32
    assert (y[~valid] == 255).all()
    xf, yf, vf = aligned_flip(x, y, np.random.default_rng(2), valid)
    assert torch.equal(xf, x.flip(-1)) and torch.equal(yf, y.flip(-1))
    assert torch.equal(vf, valid.flip(-1))


def test_real_steps_freeze_geometry_but_update_native_body():
    torch.manual_seed(9); torch.set_num_threads(2)
    c = pilot_recipe("meta")
    model, _ = build_model(c["model"], synthetic=True, window=c["window"])
    geometry = geometry_named_parameters(model)
    body = [p for n, p in model.named_parameters() if not n.startswith("head.") and n not in geometry]
    opts = [torch.optim.AdamW(body, lr=.001), torch.optim.SGD(model.head.parameters(), lr=.03)]
    before = {n: p.detach().clone() for n, p in geometry.items()}
    detail_before = model.detail[0].weight.detach().clone()
    x = torch.rand(1, 3, 32, 32); y = torch.randint(0, 4, (1, 32, 32))
    mask = torch.ones_like(y, dtype=torch.bool)
    for p in geometry.values(): p.grad = torch.ones_like(p)  # stale meta grads must be cleared
    supervised_step(model, opts, x, y, valid_mask=mask, freeze_geometry=True)
    full_pseudo_step(model, opts, x, y, mask, input_valid=mask, freeze_geometry=True)
    for name, p in geometry.items():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
        assert p.grad is None
    assert not torch.equal(model.detail[0].weight, detail_before)


def test_three_pilot_arms_sampling_checkpoint_and_exact_ema_timeline(tmp_path):
    manifest, sha = synthetic_fixture(tmp_path / "data")
    reports, histories, payloads = {}, {}, {}
    for arm in ("baseline", "fixed", "meta"):
        out = tmp_path / arm
        c = pilot_recipe(arm)
        reports[arm] = train(c, manifest, sha, out, device="cpu", synthetic=True)
        assert reports[arm]["ema_updates"] == c["steps"]
        assert reports[arm]["supervised_updates"] == c["steps"]
        assert reports[arm]["pseudo_accepted"] == c["steps"] - c["warmup_steps"]
        assert reports[arm]["label_reads_outside_budget"] == []
        assert reports[arm]["quality"] == "NOT_EVALUATED"
        histories[arm] = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
        payloads[arm] = torch.load(out / "final" / "weights.pt", weights_only=True)
        previous = torch.load(out / "step_000002" / "weights.pt", weights_only=True)
        restored, metadata = load_checkpoint(out / "final", reports[arm]["final_manifest_sha256"])
        parameter_names = set(dict(restored.named_parameters()))
        # All state is floating point in this native fixture. Teacher is mixed
        # once using the final student, including newly committed geometry.
        for name, student in payloads[arm]["student"].items():
            if name in parameter_names:
                expected = previous["teacher"][name] * c["ema_decay"] + student * (1-c["ema_decay"])
                torch.testing.assert_close(payloads[arm]["teacher"][name], expected)
        assert metadata["prediction_model"] == "student"
        assert (restored.window is None) == (arm == "baseline")
        prediction = restored(torch.rand(1, 3, 31, 35))
        assert prediction.shape == (1, 4, 31, 35) and torch.isfinite(prediction).all()
    assert reports["meta"]["geometry_updates"] == 2
    assert reports["baseline"]["geometry_updates"] == reports["fixed"]["geometry_updates"] == 0
    assert len(payloads["meta"]["optimizers"]) == 3
    assert all(row.get("meta_window", {}).get("geometry_grad_norm", 0) > 0
               for row in histories["meta"][1:])
    for fixed, meta, base in zip(histories["fixed"], histories["meta"], histories["baseline"], strict=True):
        for role in ("supervised", "audit", "unlabeled"):
            assert fixed[role] == meta[role] == base[role]
        assert meta["supervised"]["patient"] != meta["audit"]["patient"]


def test_meta_skips_empty_pseudo_support_and_keeps_geometry_frozen(tmp_path):
    manifest, sha = synthetic_fixture(tmp_path / "data")
    c = pilot_recipe("meta"); c.update(steps=2, warmup_steps=0, pseudo_threshold=1.)
    torch.manual_seed(c["seed"])
    initial, _ = build_model(c["model"], synthetic=True, window=c["window"])
    before = {n: p.detach().clone() for n,p in geometry_named_parameters(initial).items()}
    out = tmp_path / "run"
    result = train(c, manifest, sha, out, device="cpu", synthetic=True)
    assert result["geometry_updates"] == result["pseudo_accepted"] == 0
    assert result["ema_updates"] == 2
    checkpoint = torch.load(out / "final" / "weights.pt", weights_only=True)
    for n, value in before.items():
        torch.testing.assert_close(checkpoint["student"][n], value, rtol=0, atol=0)
