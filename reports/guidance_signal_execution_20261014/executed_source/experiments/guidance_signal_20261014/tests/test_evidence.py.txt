"""CPU synthetic software contracts; no real-data or anatomical accuracy claims."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from scipy import sparse

from experiments.guidance_signal_20261014 import runtime
from experiments.guidance_signal_20261014.evidence import (
    MAX_Z_ERROR, Q_ATOL, ReproductionError, anatomical_evidence, decompose,
    linear_restart, reconstruct)
from experiments.guidance_signal_20261014.prepare import (
    ORIGINAL_RECIPE_BY_BANK, geometry_arguments, guard_fresh_output, load_signal_bank,
    load_signal_patient, prepare, scalar_audit, verify_original)


def scene():
    labels = np.zeros((1, 1, 16, 16), dtype=int)
    labels[..., 4:12, 4:12] = 2
    labels[..., 6:10, 6:10] = 3
    labels[..., 6:10, 2:4] = 1
    image = np.asarray([-1., 2., 0., 2.], np.float32)[labels]
    features = np.eye(4, dtype=np.float32)[labels]
    cfg = runtime.load_runtime().core.GuidanceConfig(
        grid=16, feature_dim=4, clusters=(4, 4), kmeans_steps=3, min_region_pixels=2)
    return image, features, cfg


@pytest.fixture
def original(tmp_path, monkeypatch):
    """Explicit local fixture writer, never a runtime synthetic fallback."""
    r = runtime.load_runtime()
    tmp_path = tmp_path/"original_run"
    tmp_path.mkdir()
    image, features, cfg = scene()
    arrays, bank_report = r.core.make_bank(image, features, "+x", cfg)
    bank = tmp_path/"original"
    bank.mkdir()
    np.savez_compressed(bank/"p1.npz", image=image, features=features.astype(np.float16), **arrays)
    config = json.loads(json.dumps(cfg.dictionary()))
    splits = dict(train=["p1"], val=["v1"], test=["t1"])
    split_path = tmp_path/"original_split.json"
    r.io.write_json(split_path, splits)
    r.io.write_json(bank/"split.json", splits)
    r.io.write_json(bank/"config.json", config)
    teacher = dict(frozen=True, segmentation_finetune=False,
                   checkpoint_sha256=runtime.TEACHER_SHA256,
                   config_sha256=runtime.TEACHER_CONFIG_SHA256,
                   model_revision=runtime.TEACHER_REVISION,
                   adapter_source_sha256=runtime.EXPECTED_SOURCE_SHA256["teacher.py"])
    meta = dict(shape_xyzt=[16, 16, 1, 1], image_sha256="a"*64,
                affine=np.eye(4).tolist(), patient_left_axis="+x", raw_image="/local/synthetic/p1.nii")
    manifest = r.io.freeze(bank, dict(role="guidance_bank", split="train", patients=["p1"],
        locked_splits=splits, source_split_sha256=r.io.digest(split_path), config=config,
        teacher=teacher, seed=42, source_sha256=dict(runtime.EXPECTED_SOURCE_SHA256),
        image_records={"p1": meta}, patient_reports={"p1": bank_report}))
    recipe = tmp_path/"RECIPE.json"
    r.io.write_json(recipe, dict(schema="self_audit.full_guidance_recipe.v1", config=config,
        config_sha256=r.io.digest(bank/"config.json"), splits=splits,
        split_manifest=str(split_path), split_sha256=r.io.digest(split_path), seed=42,
        expected_teacher_sha256=runtime.TEACHER_SHA256,
        teacher_config_sha256=runtime.TEACHER_CONFIG_SHA256,
        expected_model_revision=runtime.TEACHER_REVISION,
        source_sha256=dict(runtime.EXPECTED_SOURCE_SHA256)))
    bank_sha = r.io.digest(bank/"FROZEN.json")
    # Explicit fixture-only trust injection; the CLI accepts only the published
    # original bank/recipe pair hardcoded in prepare.py.
    monkeypatch.setitem(ORIGINAL_RECIPE_BY_BANK, bank_sha, r.io.digest(recipe))
    return bank, recipe, bank_sha, manifest


def test_runtime_private_snapshot_all_modules():
    r = runtime.load_runtime()
    assert len(r.source_sha256) == 11
    for name in ("core", "io", "student", "teacher", "pipeline", "evaluate"):
        module = getattr(r, name)
        assert module.__name__.startswith(runtime.PRIVATE_PACKAGE+".")
        assert Path(module.__file__).parent == runtime.SOURCE_ROOT
    assert r.geometry is r.experimental_geometry
    assert r.diagnostics.__name__.endswith("experimental_diagnostics")


def test_runtime_tamper_rejected_before_import(tmp_path, monkeypatch):
    source = tmp_path/"snapshot"
    shutil.copytree(runtime.SOURCE_ROOT, source)
    (source/"preflight.py").write_text("raise AssertionError('MUST NOT IMPORT')")
    monkeypatch.setattr(runtime, "SOURCE_ROOT", source)
    with pytest.raises(ValueError, match="source hash mismatch"):
        runtime.load_runtime()


def test_linear_rhs_mass_not_simplex_and_residual():
    p = sparse.csr_matrix([[.6, .4], [.2, .8]])
    b = np.asarray([[.01, 0.], [0., .06]])
    actual, receipt = linear_restart(p, b, .35)
    exact = np.linalg.solve(np.eye(2)-.65*p.toarray(), .35*b)
    assert np.max(np.abs(actual-exact)) <= receipt["error_bound"]+1e-15
    assert not np.allclose(actual.sum(-1), 1)
    with pytest.raises(ValueError, match="simplex"):
        runtime.load_runtime().core.anchored_solve(p, b)
    with pytest.raises(ValueError, match="stochastic"):
        linear_restart(p*2, b)
    with pytest.raises(ValueError, match="finite"):
        linear_restart(p, np.full_like(b, np.nan))
    with pytest.raises(RuntimeError, match="NOT_CONVERGED"):
        linear_restart(p, b, max_iterations=1)


@pytest.mark.parametrize("mass", [0., 1e-12, 1e-6])
def test_zero_tiny_and_exact_floor_invalid(mass):
    result, _ = decompose(sparse.eye(2), np.full(2, mass), np.tile([0., 1., 0., 0.], (2, 1)))
    assert not result["valid"].any()
    assert np.isnan(result["Z"]).all()
    np.testing.assert_allclose(result["M"].sum(-1), result["h"], atol=1e-15)
    if mass == 0:
        np.testing.assert_array_equal(result["Q"], .25)


def test_mixed_evidence_stays_mixed_and_wrong_name_is_not_repaired():
    p = sparse.csr_matrix([[.5, .5], [.5, .5]])
    s = np.asarray([.8, .8])
    v = np.asarray([[0., 1., 0., 0.], [0., 0., 0., 1.]])
    result, receipt = decompose(p, s, v)
    assert result["valid"].all()
    assert np.all(result["Z"][:, [1, 3]] > 0)
    np.testing.assert_allclose(result["M"].sum(-1), result["h"], atol=1e-10)
    assert receipt["class_conflict_pixels"] == 2
    assert np.max(result["z_error_bound"]) <= MAX_Z_ERROR
    wrong, _ = decompose(sparse.eye(1), np.asarray([.9]), np.asarray([[0., 1., 0., 0.]]))
    assert wrong["Z"].argmax(-1).item() == 1  # No oracle changes a wrong class name.


def test_quotient_certificate_rejects_tiny_but_positive_h():
    p = sparse.csr_matrix([[.9, .1], [.2, .8]])
    result, receipt = decompose(p, np.asarray([1e-8, 0]),
        np.asarray([[0., 1., 0., 0.], [0., 0., 0., 0.]]), min_h=1e-12)
    assert np.all(result["h"] > 1e-12)
    assert receipt["linear_solver"]["error_bound"] <= 1e-10
    assert not result["valid"].any()
    assert np.isnan(result["Z"]).all()


@pytest.mark.parametrize("left,vector", [("+x", None), ("", None), ("", [0., 1.]), ("+x", [0., 0.])])
def test_anatomical_replay_exact_and_source_ledger(left, vector):
    r = runtime.load_runtime()
    image, features, cfg = scene()
    expected = r.core.anatomical_anchors(image, features, left, cfg, left_vector_yx=vector)
    replay = anatomical_evidence(image, features, left, cfg, left_vector_yx=vector, patient_id="p1")
    np.testing.assert_array_equal(replay["r0"], expected[0])
    np.testing.assert_array_equal(replay["strength"], expected[1])
    assert replay["anatomy"] == expected[2]
    assert replay["sources"]
    assert all(x["independent_evidence"] is False for x in replay["sources"])
    assert len({x["source_id"] for x in replay["sources"]}) == len(replay["sources"])
    assert len({x["repeated_mask_group"] for x in replay["sources"]}) < len(replay["sources"])
    support = sparse.csr_matrix((replay["source_support_data"], replay["source_support_indices"],
                                replay["source_support_indptr"]), shape=tuple(replay["source_support_shape"]))
    np.testing.assert_allclose(np.asarray(support.sum(1)).reshape(image.shape), replay["strength"], atol=2e-7)
    for cls in range(4):
        columns = [i for i, entry in enumerate(replay["sources"]) if entry["class_id"] == cls]
        target = replay["strength"]*replay["V"][..., cls]
        np.testing.assert_allclose(np.asarray(support[:, columns].sum(1)).reshape(image.shape), target, atol=2e-7)


def test_original_graph_q_reproduction(monkeypatch):
    r = runtime.load_runtime()
    image, features, cfg = scene()
    expected, expected_report = r.core.make_bank(image, features, "+x", cfg)
    actual, receipt, _ = reconstruct(image, features, "+x", cfg,
        cached_q=expected["q"], cached_r0=expected["r0"],
        cached_reliability=expected["reliability"], cached_labels=expected["labels"])
    assert receipt["graph"] == expected_report["graph"]
    assert receipt["reproduction"]["hard_label_mismatches"] == 0
    np.testing.assert_allclose(actual["Q"], expected["q"], rtol=0, atol=Q_ATOL)
    p, _ = r.core.build_graph(image, features, cfg)
    np.testing.assert_allclose(p.sum(1), 1, atol=2e-6)
    with pytest.raises(ReproductionError, match="FAIL_REPRODUCTION"):
        reconstruct(image, features, "+x", cfg, cached_q=np.roll(expected["q"], 1, -1), cached_r0=expected["r0"])
    perturbed = expected["q"].copy()
    perturbed[..., 0] += 1e-5
    perturbed[..., 1] -= 1e-5
    with pytest.raises(ReproductionError, match="FAIL_REPRODUCTION"):
        reconstruct(image, features, "+x", cfg, cached_q=perturbed, cached_r0=expected["r0"])


def test_prepare_roundtrip_private_and_overwrite(original, tmp_path):
    bank, recipe, expected, _ = original
    output = tmp_path/"signals"
    frozen = prepare(bank, recipe, output, expected_bank_sha=expected)
    assert frozen["status"] == "PASS_REPRODUCTION"
    assert frozen["role"] == "guidance_signal_bank_v1"
    assert frozen["gt_read"] is False and frozen["inference_run"] is False
    assert frozen["original_permission_patient_counts"] == [1, 1, 1, 1]
    assert frozen["original_allowed_classes"] == [False, False, False, False]
    assert load_signal_bank(output, bank) == frozen
    arrays = load_signal_patient(output, "p1")
    assert arrays["valid"].any()
    assert (output/"p1.npz").stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        prepare(bank, recipe, output, expected_bank_sha=expected)
    with pytest.raises(ValueError, match="patient"):
        load_signal_patient(output, "t1")
    with (output/"p1.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        load_signal_patient(output, "p1")


def test_sanitized_missing_payload_and_identity_fail(original, tmp_path):
    bank, recipe, expected, _ = original
    with pytest.raises(ValueError, match="SHA256"):
        verify_original(bank, recipe, "0"*64)
    data = json.loads(recipe.read_text())
    data["root"] = "${REPO_ROOT}/data/ACDC"
    recipe.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="sanitized"):
        prepare(bank, recipe, tmp_path/"fail", expected_bank_sha=expected)
    assert not (tmp_path/"fail").exists()
    (bank/"p1.npz").unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify_original(bank, recipe, expected)


def test_nontrain_rejected(original, tmp_path, monkeypatch):
    bank, recipe, _, _ = original
    path = bank/"FROZEN.json"
    manifest = json.loads(path.read_text())
    manifest["split"] = "val"
    path.write_text(json.dumps(manifest))
    monkeypatch.setitem(ORIGINAL_RECIPE_BY_BANK, runtime.sha256(path), runtime.sha256(recipe))
    with pytest.raises(ValueError, match="TRAIN"):
        prepare(bank, recipe, tmp_path/"fail", expected_bank_sha=runtime.sha256(path))


def test_world_geometry_reproduces_zero_abstention_and_detects_tamper(original):
    _, _, _, manifest = original
    r = runtime.load_runtime()
    image, _, _ = scene()
    meta = manifest["image_records"]["p1"]
    geo = r.geometry.world_left_geometry(meta["affine"], (16, 16), (16, 16), sform_code=0, qform_code=0)
    geo["image_sha256"] = meta["image_sha256"]
    manifest["orientation_policy"] = "world_left_v2"
    meta["world_left_geometry"] = geo
    args, _ = geometry_arguments(r, manifest, "p1", image, image)
    assert args["left_vector_yx"] == [0., 0.]
    geo["left_vector_yx"] = [0., 1.]
    with pytest.raises(ValueError, match="geometry mismatch"):
        geometry_arguments(r, manifest, "p1", image, image)


def test_reproduction_failure_unfrozen_receipt(original, tmp_path, monkeypatch):
    bank, recipe, _, manifest = original
    path = bank/"p1.npz"
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    data["features"] = np.ones_like(data["features"])
    np.savez_compressed(path, **data)
    frozen_path = bank/"FROZEN.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["files"]["p1.npz"] = runtime.sha256(path)
    frozen_path.write_text(json.dumps(frozen))
    monkeypatch.setitem(ORIGINAL_RECIPE_BY_BANK, runtime.sha256(frozen_path), runtime.sha256(recipe))
    output = tmp_path/"failed_replay"
    with pytest.raises(ReproductionError):
        prepare(bank, recipe, output, expected_bank_sha=runtime.sha256(frozen_path))
    assert not (output/"FROZEN.json").exists()
    assert json.loads((output/"FAILED.json").read_text())["status"] == "FAIL_REPRODUCTION"


def test_conflicting_raw_source_replay_exact(monkeypatch):
    r = runtime.load_runtime()
    first = np.zeros((1, 1, 16, 16), int)
    first[..., 3:13, 3:13] = 2
    first[..., 5:11, 5:11] = 3
    second = np.zeros_like(first)
    second[..., 5:11, 5:11] = 2
    second[..., 6:10, 6:10] = 3
    image = np.zeros(first.shape, np.float32)
    image[..., 5:11, 5:11] = 1
    image[..., 6:10, 6:10] = 3
    features = np.ones((*image.shape, 4), np.float32)
    cfg = r.core.GuidanceConfig(grid=16, feature_dim=4, clusters=(3, 4), min_region_pixels=2)
    monkeypatch.setattr(r.core, "_cluster", lambda f, k, steps, max_points: first if k == 3 else second)
    expected = r.core.anatomical_anchors(image, features, "", cfg, left_vector_yx=[0., 0.])
    replay = anatomical_evidence(image, features, "", cfg, left_vector_yx=[0., 0.])
    np.testing.assert_array_equal(replay["r0"], expected[0])
    np.testing.assert_array_equal(replay["strength"], expected[1])
    assert replay["anatomy"] == expected[2]
    assert replay["anatomy"]["conflicting_vote_pixels"] > 0
    assert ((replay["V"] > 0).sum(-1) > 1).any()


def test_output_guards_preserve_original_roots(original, tmp_path):
    bank, recipe, expected, _ = original
    for output in (bank/"new", recipe.parent/"new"):
        with pytest.raises(ValueError, match="overlap"):
            prepare(bank, recipe, output, expected_bank_sha=expected)
        assert not output.exists()
    locked = tmp_path/"locked"
    locked.mkdir()
    (locked/"PREDICTIONS_LOCKED.json").write_text("{}")
    with pytest.raises(ValueError, match="locked"):
        guard_fresh_output(locked/"new")
    alias = tmp_path/"link"
    alias.symlink_to(locked, target_is_directory=True)
    with pytest.raises(ValueError, match="locked"):
        guard_fresh_output(alias/"new")


def test_resealed_false_pass_report_rejected(original, tmp_path):
    bank, recipe, expected, _ = original
    output = tmp_path/"signals"
    prepare(bank, recipe, output, expected_bank_sha=expected)
    path = output/"FROZEN.json"
    manifest = json.loads(path.read_text())
    manifest["patient_reports"]["p1"]["reproduction"]["r0_max_error"] = .01
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reproduction error"):
        load_signal_bank(output, bank)


def test_original_recipe_optimizer_mutation_fails(original, tmp_path):
    bank, recipe, expected, _ = original
    payload = json.loads(recipe.read_text())
    payload["lr"] = .1
    recipe.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="recipe identity"):
        prepare(bank, recipe, tmp_path/"signals", expected_bank_sha=expected)
    assert not (tmp_path/"signals").exists()


def test_caller_cannot_override_original_bank_pin(original, tmp_path):
    bank, recipe, _, _ = original
    path = bank/"FROZEN.json"
    payload = json.loads(path.read_text())
    payload["seed"] = 43
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="no identity override"):
        prepare(bank, recipe, tmp_path/"signals", expected_bank_sha=runtime.sha256(path))


def test_scalar_audit_separates_argmax_ties_uniform_and_winning_contrast():
    q = np.full((1, 1, 2, 2, 4), .25, np.float32)
    shape = q.shape[:-1]
    original = dict(q=q, r0=q.copy(), labels=np.full(shape, 255, np.uint8))
    arrays = dict(raw_votes=np.zeros_like(q), h=np.zeros(shape), valid=np.zeros(shape, bool))
    audit = scalar_audit(arrays, original, [])
    assert audit["classes"]["BG"]["Q_argmax_pixels"] == 4
    assert audit["classes"]["BG"]["Q_positive_winning_contrast_pixels"] == 0
    assert audit["classes"]["BG"]["Q_above_uniform_pixels"] == 0
    assert audit["original_UNKNOWN_pixels"] == 4
    assert audit["h_on_valid_quantiles"] is None
