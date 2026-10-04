"""DSS-US PAPER_FAITHFUL_REIMPLEMENTATION: equations, flow and profile separation (synthetic only)."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src")]
from dss_us import paper_faithful as pf  # noqa: E402
from dss_us.affinity import mi_affinity, ssd_affinity  # noqa: E402
from shared_benchmark.native_protocol import ProtocolBlocked, load_lock  # noqa: E402

CONFIGS = BASE / "config/native"
RNG = np.random.default_rng(0)
GRID = (4, 4)
KEYS = RNG.normal(size=(16, 6))
GRAY = RNG.integers(0, 256, size=(32, 32)).astype(float)
SETTINGS = {"ultrasound_affinities": True, "feature_l2_normalization": False, "patch_intensity_scale": 1 / 255,
            "c_ssd": 0.5, "c_mi": 0.25, "c_pos": 0.75, "delta_ssd": 0.1, "delta_mi": 1.0, "mi_bins": 8,
            "mi_intensity_range": [0.0, 1.0], "positional_knn_k": 5, "positional_symmetrization": "max",
            "eigenvectors": 6, "discard_trivial_eigenvector": True, "normalize_embedding_rows": False,
            "upscale_method": "nearest", "crf_parameters": {"stub": True}, "kmeans_seed": 0, "kmeans_n_init": 3,
            "kmeans_max_iter": 100, "kmeans_tolerance": 1e-4, "kmeans_algorithm": "lloyd"}


def identity_crf(image, partition, parameters):  # test stub only; production has no CRF default
    return partition.astype(np.int64)


def test_eq1_feature_affinity_is_the_positive_part_of_the_gram_matrix():
    weights = pf.step1_affinity(KEYS, GRAY, GRID, {**SETTINGS, "ultrasound_affinities": False})
    gram = KEYS @ KEYS.T
    np.testing.assert_allclose(weights, gram * (gram > 0))


def test_patches_follow_the_8_pixel_transformer_grid():
    patches = pf.image_patches(GRAY, GRID)
    assert patches.shape == (16, 8, 8)
    np.testing.assert_array_equal(patches[5], GRAY[8:16, 8:16])  # row 1, column 1
    with pytest.raises(ValueError, match="patch grid"):
        pf.image_patches(GRAY[:30], GRID)


def test_eq5_combination_with_symmetrised_positional_knn():
    weights = pf.step1_affinity(KEYS, GRAY, GRID, SETTINGS)
    patches = pf.image_patches(GRAY, GRID) / 255
    gram = KEYS @ KEYS.T
    ys, xs = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4), indexing="ij")
    psi = np.stack([xs.ravel(), ys.ravel()], axis=1)
    distance = np.linalg.norm(psi[:, None] - psi[None], axis=2)
    knn = np.zeros_like(distance)
    for j in range(16):
        for i in np.argsort(distance[j], kind="stable")[:5]:
            knn[j, i] = 1 - distance[j, i]
    expected = (gram * (gram > 0) + 0.5 * ssd_affinity(patches, delta=0.1)
                + 0.25 * mi_affinity(patches, delta=1.0, bins=8, intensity_range=(0.0, 1.0))
                + 0.75 * np.maximum(knn, knn.T))
    np.testing.assert_allclose(weights, expected)


def test_step1_produces_15_segments_upscaled_and_requires_crf():
    with pytest.raises(pf.ProtocolSettingError, match="CRF"):
        pf.step1_partition(KEYS, GRAY, GRID, (32, 32), SETTINGS)
    partition, fit = pf.step1_partition(KEYS, GRAY, GRID, (32, 32), SETTINGS, crf_backend=identity_crf)
    assert partition.shape == (32, 32) and partition.dtype == np.int32
    assert len(np.unique(partition)) == pf.STEP1_SEGMENTS == 15
    assert np.all(partition[:8, :8] == partition[0, 0])  # nearest upscale of one patch


@pytest.mark.parametrize("missing", ["feature_l2_normalization", "c_mi", "delta_ssd", "positional_knn_k",
                                     "positional_symmetrization", "eigenvectors", "upscale_method", "kmeans_seed"])
def test_unspecified_values_are_required(missing):
    with pytest.raises(pf.ProtocolSettingError, match=missing):
        pf.step1_partition(KEYS, GRAY, GRID, (32, 32), {**SETTINGS, missing: None}, crf_backend=identity_crf)


def test_step2_variants_follow_the_paper():
    partition = np.repeat(np.arange(4).reshape(2, 2), 4, axis=0).repeat(4, axis=1)
    images = [(RNG.random((8, 8)), partition), (RNG.random((8, 8)), partition)]
    phi = lambda crop: np.array([crop.mean(), crop.std(), crop.shape[0], crop.shape[1]])
    settings = {**SETTINGS, "step2_variant": "dss_step2", "semantic_clusters": 2}
    outputs, fit = pf.step2_semantic(images, settings, phi_image=phi)
    assert len(outputs) == 2 and outputs[0].shape == (8, 8)
    with pytest.raises(pf.ProtocolSettingError, match="mask and position"):
        pf.step2_semantic(images, {**settings, "step2_variant": "ours_step2", "c_mask": 1.0, "c_position_embedding": 1.0},
                          phi_image=phi)
    ours, _ = pf.step2_semantic(images, {**settings, "step2_variant": "ours_step2", "c_mask": 1.0, "c_position_embedding": 1.0},
                                phi_image=phi, phi_mask=lambda m: np.array([m.mean(), m.sum(), 0.0, 0.0]),
                                phi_position=lambda box, shape: np.array([box[0], box[2], 0.0, 0.0], dtype=float))
    assert len(ours) == 2


ROWS = {"step1_dss_baseline": (False, False), "step1_ours_proc": (True, False),
        "step1_ours_aff": (False, True), "step1_ours_comb": (True, True)}


def test_paper_faithful_profiles_encode_the_paper_rows_and_stay_blocked():
    faithful = sorted(CONFIGS.glob("*_paper_faithful.yaml"))
    assert len(faithful) == 11
    for path in faithful:
        config = json.loads(path.read_text())
        assert config["profile_class"] == "PAPER_FAITHFUL_REIMPLEMENTATION"
        assert config["scientific"]["step1_segments"] == 15 and config["scientific"]["patch_size"] == 8
        assert config["scientific"]["step1_crf"] is True
        row = "_".join(config["profile"].split("_")[1:3])
        proc, aff = ROWS[f"step1_{row}"]
        assert (config["scientific"]["preprocessing"], config["scientific"]["ultrasound_affinities"]) == (proc, aff)
        assert ("c_ssd" in config["paper_unspecified"]) == aff and ("preprocessing_method" in config["paper_unspecified"]) == proc
        assert all(entry["value"] is None for entry in config["paper_unspecified"].values())
        with pytest.raises(ProtocolBlocked) as error:
            load_lock(path)
        reasons = " ".join(error.value.reasons)
        if config["stage"] == "I":  # paper Table 1 CRF; official dependency is source-only
            assert "simplecrf==0.2.1.1" in reasons and "crf_parameters" in reasons
        else:  # Step II CRF is conditional (Table 2 carries no CRF mark)
            assert "crf_parameters" not in config["paper_unspecified"] and "crf_parameters" in config["conditional_values"]
            assert "semantic_clusters" in reasons
        if config["stage"] == "I":
            assert load_lock(path, purpose="native_track_b")["native_track_b"]["n_classes"] == 4
        else:
            with pytest.raises(ProtocolBlocked):
                load_lock(path, purpose="native_track_b")
    original = json.loads((CONFIGS / "step1_ours_comb.yaml").read_text())
    assert original["implementation"] == "independent_reimplementation" and "profile_class" not in original
