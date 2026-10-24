"""Synthetic integration contracts; these are not patient or efficacy evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from self_audit.data import TrainingData, synthetic_fixture
from self_audit.engine import DEFAULT, load_checkpoint, load_exported_student, train, validate_config
from self_audit.external_pseudo import ExternalPseudoCache
from self_audit.model import build_model, digest
from self_audit.teacher_memory import MemoryBank, save_bank
from self_audit.window import WINDOW_DEFAULT


def test_student_export_cli_help_runs_outside_repo_without_pythonpath(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_teacher_memory_student.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--checkpoint-sha256" in result.stdout


def bank_fixture(path):
    rng = np.random.default_rng(13)
    endpoints = rng.normal(size=(6, 3, 2, 32)).astype(np.float32)
    endpoints /= np.linalg.norm(endpoints, axis=-1, keepdims=True)
    confidences = np.full((6, 3, 2), .9, dtype=np.float32)
    valid = np.ones((6, 3), dtype=bool)
    return save_bank(MemoryBank(endpoints, confidences, valid,
                                {"provenance": {"source": "synthetic_fixture"}}), path)


def memory_recipe(bank_path, bank_sha):
    c = copy.deepcopy(DEFAULT)
    c.update(method="ema_full", steps=2, warmup_steps=1, batch_size=1, crop_size=40,
             pseudo_threshold=.2, body_lr=.001, head_lr=.02, save_every=1)
    c["model"]["backbone"] = "tiny_test"
    c["window"] = copy.deepcopy(WINDOW_DEFAULT)
    c["window"].update(mode="fixed", dim=4)
    c["teacher_memory"].update(arm="A3", bank_path=str(bank_path), bank_sha256=bank_sha,
                               tau=.2)
    return c


def test_disabled_memory_keeps_initialization_rng_outputs_and_gradients():
    base = copy.deepcopy(DEFAULT); base["model"]["backbone"] = "tiny_test"
    explicit = copy.deepcopy(base); explicit["teacher_memory"] = copy.deepcopy(DEFAULT["teacher_memory"])
    torch.manual_seed(91)
    first, _ = build_model(base["model"], synthetic=True)
    state_first, rng_first = copy.deepcopy(first.state_dict()), torch.get_rng_state().clone()
    torch.manual_seed(91)
    second, _ = build_model(explicit["model"], synthetic=True)
    assert state_first.keys() == second.state_dict().keys()
    for key, value in state_first.items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    assert torch.equal(rng_first, torch.get_rng_state())
    x1 = torch.rand(1, 3, 31, 35); x2 = x1.clone()
    first(x1).square().mean().backward(); second(x2).square().mean().backward()
    torch.testing.assert_close(first(x1).detach(), second(x2).detach(), rtol=0, atol=0)
    for (_, p1), (_, p2) in zip(first.named_parameters(), second.named_parameters(), strict=True):
        torch.testing.assert_close(p1.grad, p2.grad, rtol=0, atol=0)


def test_disabled_memory_training_preserves_student_teacher_optimizer_and_rng(tmp_path):
    manifest, data_sha = synthetic_fixture(tmp_path / "data", size=32)
    old_recipe = copy.deepcopy(DEFAULT)
    old_recipe.update(method="ema_full", steps=2, warmup_steps=1, batch_size=1,
                      pseudo_threshold=.2, body_lr=.001, head_lr=.02, save_every=2)
    old_recipe["model"]["backbone"] = "tiny_test"
    old_recipe["window"].update(mode="fixed", dim=4)
    del old_recipe["teacher_memory"]
    explicit_recipe = copy.deepcopy(old_recipe)
    explicit_recipe["teacher_memory"] = copy.deepcopy(DEFAULT["teacher_memory"])
    first = train(old_recipe, manifest, data_sha, tmp_path / "old", device="cpu", synthetic=True)
    second = train(explicit_recipe, manifest, data_sha, tmp_path / "explicit", device="cpu", synthetic=True)
    left = torch.load(tmp_path / "old" / "final" / "weights.pt", weights_only=True)
    right = torch.load(tmp_path / "explicit" / "final" / "weights.pt", weights_only=True)
    assert first["ema_updates"] == second["ema_updates"] == 2
    def assert_tree(a, b):
        if torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a: assert_tree(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True): assert_tree(x, y)
        else:
            assert a == b
    for key in ("student", "teacher", "optimizers", "torch_rng"):
        assert_tree(left[key], right[key])
    old_rows = [json.loads(s) for s in (tmp_path / "old" / "steps.jsonl").read_text().splitlines()]
    new_rows = [json.loads(s) for s in (tmp_path / "explicit" / "steps.jsonl").read_text().splitlines()]
    for a, b in zip(old_rows, new_rows, strict=True):
        for key in a.keys() - {"elapsed_seconds", "cuda_peak_allocated"}:
            assert a[key] == b[key]


def test_memory_projection_and_alpha_receive_task_loss_gradients(tmp_path):
    bank_path = tmp_path / "bank.npz"; bank_sha = bank_fixture(bank_path)
    from self_audit.teacher_memory import load_bank
    bank = load_bank(bank_path, bank_sha)
    config = memory_recipe(bank_path, bank_sha)
    torch.manual_seed(5)
    model, _ = build_model(config["model"], synthetic=True, window=config["window"], memory_bank=bank)
    logits = model(torch.rand(2, 3, 33, 35))
    assert logits.shape == (2, 4, 33, 35) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    memory = model.teacher_memory
    assert memory.projection.grad is not None and memory.projection.grad.norm() > 0
    assert memory.alpha.grad is not None and memory.alpha.grad.norm() > 0


def test_memory_checkpoint_load_is_teacher_free_and_bank_path_independent(tmp_path):
    data_manifest, data_sha = synthetic_fixture(tmp_path / "data", size=32)
    bank_path = tmp_path / "bank.npz"; bank_sha = bank_fixture(bank_path)
    config = memory_recipe(bank_path, bank_sha)
    report = train(config, data_manifest, data_sha, tmp_path / "run", device="cpu", synthetic=True)
    assert report["ema_updates"] == 2
    assert report["pseudo_accepted"] == 1
    # Checkpoint student state embeds the complete fixed bank for model creation.
    checkpoint = torch.load(tmp_path / "run" / "final" / "weights.pt", weights_only=True)
    assert "teacher_memory.bank.endpoints" in checkpoint["student"]
    bank_path.unlink()
    restored, metadata = load_checkpoint(tmp_path / "run" / "final", report["final_manifest_sha256"], "cpu")
    assert metadata["prediction_model"] == "student"
    prediction = restored(torch.rand(1, 3, 31, 35))
    assert prediction.shape == (1, 4, 31, 35) and torch.isfinite(prediction).all()
    from scripts.export_teacher_memory_student import export
    exported_sha = export(tmp_path / "run" / "final", report["final_manifest_sha256"], tmp_path / "export")
    exported, exported_meta = load_exported_student(tmp_path / "export", exported_sha)
    assert exported_meta["teacher_included"] is False
    export_input = torch.rand(1, 3, 31, 35)
    torch.testing.assert_close(exported(export_input), restored(export_input), rtol=0, atol=0)


def test_external_pseudo_cache_uses_sampled_native_crop_and_valid_mask(tmp_path):
    manifest, data_sha = synthetic_fixture(tmp_path / "data", size=5)
    data = TrainingData(manifest, data_sha, synthetic=True)
    record = data.unlabeled[0]
    # Distinct native probability at each pixel makes crop displacement visible.
    yy, xx = np.mgrid[:5, :5]
    logits = np.stack((xx, yy, xx + yy, np.full_like(xx, 3)), axis=0).astype(np.float32)
    probs = np.exp(logits - logits.max(0, keepdims=True)); probs /= probs.sum(0, keepdims=True)
    support = np.ones((1, 1, 5, 5), dtype=bool); support[0, 0, 2, 2] = False
    probs[:, 2, 2] = 0.0  # abstain outside geometric teacher support
    cache_data = tmp_path / "patient_probs.npy"
    np.save(cache_data, probs[None, None].astype(np.float32))
    support_data = tmp_path / "patient_support.npy"
    np.save(support_data, support)
    cache_json = tmp_path / "cache.json"
    geometry = {r["id"]: {"shape_tzyx": [1, 1, 5, 5],
        "affine_xyz_to_world_mm": np.eye(4).tolist(), "probability_shape_tzcyx": [1, 1, 4, 5, 5],
        "axis_mapping": "native_NIfTI_XYZT_to_prepared_TZYX=transpose(3,2,1,0); probability_TZCYX_plane=class,Y,X from the same native affine XYZ grid",
        "valid_support": {"path": support_data.name, "sha256": digest(support_data), "shape_tzyx": [1, 1, 5, 5]}}
        for r in data.unlabeled}
    cache_json.write_text(json.dumps({
        "schema": "self_audit.external_pseudo.v1", "role": "training", "synthetic": True,
        "class_map": ["BG", "RV", "MYO", "LV"], "probability_layout": "TZCYX", "evaluation_ids": [],
        "provenance": {"native_geometry_by_patient": geometry},
        "records": [{"id": r["id"], "image_sha256": r["image_sha256"],
                     "probabilities": cache_data.name, "probabilities_sha256": digest(cache_data),
                     "shape": list(probs[None, None].shape)} for r in data.unlabeled],
    }))
    cache = ExternalPseudoCache(cache_json, digest(cache_json), data)
    _, metadata, valid = data.sample_batch(0, np.random.default_rng(4), 1, labeled=False,
                                          crop_size=8, return_valid=True)
    aligned, aligned_support = cache.sample_batch(metadata, device="cpu")
    crop = metadata["crop"]
    left, right, top, bottom = crop["pad_lrtb"]
    expected = torch.from_numpy(probs).clone()
    expected = torch.nn.functional.pad(expected[None], (left, right, top, bottom), value=0)[0]
    expected_support = torch.from_numpy(support[0, 0]).float()[None, None]
    expected_support = torch.nn.functional.pad(expected_support, (left, right, top, bottom), value=0)[0, 0].bool()
    y0, x0, size = crop["top"], crop["left"], crop["size"]
    expected = expected[:, y0:y0 + size, x0:x0 + size][None]
    expected_support = expected_support[y0:y0 + size, x0:x0 + size][None]
    torch.testing.assert_close(aligned, expected, rtol=0, atol=0)
    assert torch.equal(aligned_support, expected_support)
    assert aligned.shape == (1, 4, 8, 8)
    torch.testing.assert_close(aligned.sum(1)[aligned_support], torch.ones_like(aligned.sum(1)[aligned_support]))
    assert torch.all(aligned.sum(1)[~aligned_support] == 0)
    assert torch.all(aligned_support <= valid)
    invalid_map = json.loads(cache_json.read_text())
    invalid_map["class_map"] = ["BG", "LV", "MYO", "RV"]
    cache_json.write_text(json.dumps(invalid_map))
    with pytest.raises(ValueError, match="identity/layout"):
        ExternalPseudoCache(cache_json, digest(cache_json), data)
    invalid_map["class_map"] = ["BG", "RV", "MYO", "LV"]
    invalid_probs = np.load(cache_data, allow_pickle=False).copy()
    invalid_probs[0, 0, :, 2, 2] = .25
    np.save(cache_data, invalid_probs)
    invalid_map["records"] = [dict(r, probabilities_sha256=digest(cache_data)) for r in invalid_map["records"]]
    cache_json.write_text(json.dumps(invalid_map))
    with pytest.raises(ValueError, match="zero outside support"):
        ExternalPseudoCache(cache_json, digest(cache_json), data)


def test_a1_external_loss_shares_one_pseudo_optimizer_transaction_and_ema(tmp_path, monkeypatch):
    import self_audit.engine as engine

    manifest, data_sha = synthetic_fixture(tmp_path / "data", size=32)
    data = TrainingData(manifest, data_sha, synthetic=True)
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    records = []
    support_array = np.ones((1, 1, 32, 32), dtype=bool); support_array[0, 0, 0, 0] = False
    for record in data.unlabeled:
        source = data.array(record["image"], record["image_sha256"])
        shape = (source.shape[0], source.shape[1], 4, source.shape[2], source.shape[3])
        probability_path = cache_root / f"{record['id']}.npy"
        probabilities = np.full(shape, .25, dtype=np.float32); probabilities[0, 0, :, 0, 0] = 0.0
        np.save(probability_path, probabilities)
        support_path = cache_root / f"{record['id']}_support.npy"
        np.save(support_path, support_array)
        record["_support_path"] = support_path.name
        record["_support_sha256"] = digest(support_path)
        records.append({"id": record["id"], "image_sha256": record["image_sha256"],
                        "probabilities": probability_path.name, "probabilities_sha256": digest(probability_path),
                        "shape": list(shape)})
    cache_manifest = cache_root / "cache.json"
    geometry = {r["id"]: {"shape_tzyx": [1, 1, 32, 32],
        "affine_xyz_to_world_mm": np.eye(4).tolist(), "probability_shape_tzcyx": [1, 1, 4, 32, 32],
        "axis_mapping": "native_NIfTI_XYZT_to_prepared_TZYX=transpose(3,2,1,0); probability_TZCYX_plane=class,Y,X from the same native affine XYZ grid",
        "valid_support": {"path": r["_support_path"], "sha256": r["_support_sha256"], "shape_tzyx": [1, 1, 32, 32]}}
        for r in data.unlabeled}
    records = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    cache_manifest.write_text(json.dumps({"schema": "self_audit.external_pseudo.v1", "role": "training",
        "synthetic": True, "class_map": ["BG", "RV", "MYO", "LV"],
        "probability_layout": "TZCYX", "evaluation_ids": [],
        "provenance": {"native_geometry_by_patient": geometry}, "records": records}))
    config = copy.deepcopy(DEFAULT)
    config.update(method="ema_full", steps=2, warmup_steps=1, batch_size=1, crop_size=40,
                  pseudo_threshold=.2, body_lr=.001, head_lr=.02, save_every=1)
    config["model"]["backbone"] = "tiny_test"
    config["window"].update(mode="fixed", dim=4)
    config["teacher_memory"].update(arm="A1", cache_path=str(cache_manifest),
        cache_sha256=digest(cache_manifest), external_weight=.3)
    calls = []
    original = engine.full_pseudo_step

    def counted(*args, **kwargs):
        calls.append(kwargs.get("external_probabilities") is not None)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "full_pseudo_step", counted)
    report = engine.train(config, manifest, data_sha, tmp_path / "run", device="cpu", synthetic=True)
    assert calls == [True]
    assert report["ema_updates"] == 2 and report["pseudo_accepted"] == 1
    rows = [json.loads(line) for line in (tmp_path / "run" / "steps.jsonl").read_text().splitlines()]
    assert rows[1]["external_accepted_pixels"] == rows[1]["unlabeled_valid_pixels"] - 1
    previous = torch.load(tmp_path / "run" / "step_000001" / "weights.pt", weights_only=True)
    final = torch.load(tmp_path / "run" / "final" / "weights.pt", weights_only=True)
    restored, _ = load_checkpoint(tmp_path / "run" / "final", report["final_manifest_sha256"])
    parameter_names = dict(restored.named_parameters())
    for name, student in final["student"].items():
        if name in parameter_names:
            expected = previous["teacher"][name] * config["ema_decay"] + student * (1 - config["ema_decay"])
            torch.testing.assert_close(final["teacher"][name], expected)

    # Opposite support: external targets are accepted but EMA targets are not.
    # Root's matched-step contract keeps external CE inside an EMA U update;
    # this must not create an external-only optimizer transaction.
    one_hot_records = []
    for record in records:
        path = cache_root / record["probabilities"]
        source = np.load(path, allow_pickle=False)
        one_hot = np.zeros_like(source); one_hot[:, :, 0] = 1.0; one_hot[0, 0, :, 0, 0] = 0.0
        np.save(path, one_hot)
        changed = dict(record, probabilities_sha256=digest(path))
        one_hot_records.append(changed)
    cache_manifest.write_text(json.dumps({"schema": "self_audit.external_pseudo.v1", "role": "training",
        "synthetic": True, "class_map": ["BG", "RV", "MYO", "LV"],
        "probability_layout": "TZCYX", "evaluation_ids": [],
        "provenance": {"native_geometry_by_patient": geometry}, "records": one_hot_records}))
    no_ema_recipe = copy.deepcopy(config)
    no_ema_recipe["pseudo_threshold"] = 1.0
    no_ema_recipe["teacher_memory"]["cache_sha256"] = digest(cache_manifest)
    calls.clear()
    no_ema = engine.train(no_ema_recipe, manifest, data_sha, tmp_path / "run_no_ema_support",
                          device="cpu", synthetic=True)
    assert calls == [] and no_ema["pseudo_accepted"] == 0 and no_ema["ema_updates"] == 2
    no_ema_rows = [json.loads(line) for line in
                   (tmp_path / "run_no_ema_support" / "steps.jsonl").read_text().splitlines()]
    assert no_ema_rows[1]["accepted_pixels"] == 0
    assert no_ema_rows[1]["external_accepted_pixels"] == no_ema_rows[1]["unlabeled_valid_pixels"] - 1
    assert no_ema_rows[1]["external_loss_pixels"] == 0


def test_memory_config_requires_explicit_verified_assets_and_arm_constraints(tmp_path):
    c = copy.deepcopy(DEFAULT); c.update(method="ema_full")
    c["window"].update(mode="fixed")
    c["teacher_memory"].update(arm="A3", bank_path=str(tmp_path / "missing.npz"), bank_sha256="0" * 64)
    with pytest.raises(ValueError, match="file/hash"):
        validate_config(c)
    c["window"].update(mode="adaptive")
    with pytest.raises(ValueError, match="fixed window"):
        validate_config(c)
