# SPDX-License-Identifier: GPL-3.0
"""SGSCN PAPER_FAITHFUL_REIMPLEMENTATION: equations, architecture and profile separation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src")]
from sgscn import paper_faithful as pf  # noqa: E402
from sgscn.losses import loss_terms  # noqa: E402
from shared_benchmark.native_protocol import ProtocolBlocked, load_lock  # noqa: E402
from shared_benchmark.native_artifacts import verify_raw_run  # noqa: E402

torch.set_num_threads(1)
CONFIGS = BASE / "config/native"
FILLED = {"layer_order": "conv_relu_bn", "output_normalization": "per_channel_standardization",
          "standardization_epsilon": 1e-5, "context_density": "channel_softmax", "input_encoding": "rgb_unit_interval",
          "stopping_rule": "stable_label_count_and_relative_loss", "max_iterations": 4, "stability_patience": 2,
          "relative_loss_tolerance": 1e-3, "final_forward_mode": "train", "weight_decay": 0.0}


def settings(**overrides):
    base = json.loads((CONFIGS / "ph2_paper_faithful.yaml").read_text())["scientific"]
    return {**base, **FILLED, **overrides}


def test_architecture_is_three_3x3_relu_bn_layers_with_100_filters():
    model = pf.PaperFaithfulSGSCN(3, settings())
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 3 and len([m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]) == 3
    for conv in convs:
        assert conv.kernel_size == (3, 3) and conv.stride == (1, 1) and conv.padding == (1, 1) and conv.out_channels == 100


@pytest.mark.parametrize("order", ["conv_relu_bn", "conv_bn_relu"])
def test_layer_order_follows_the_declared_option(order):
    model = pf.PaperFaithfulSGSCN(3, settings(layer_order=order, output_normalization="global_standardization"))
    x = torch.randn(1, 3, 6, 5)
    expected = x
    for conv, norm in zip(model.convs, model.norms):
        y = conv(expected)
        expected = norm(F.relu(y)) if order == "conv_relu_bn" else F.relu(norm(y))
    expected = (expected - expected.mean(dim=(1, 2, 3), keepdim=True)) / torch.sqrt(
        expected.var(dim=(1, 2, 3), unbiased=False, keepdim=True) + 1e-5)
    torch.testing.assert_close(model(x), expected)


def test_output_normalisations_give_zero_mean_unit_variance():
    x = torch.randn(1, 3, 7, 6)
    per_channel = pf.PaperFaithfulSGSCN(3, settings())(x)[0]
    torch.testing.assert_close(per_channel.mean(dim=(1, 2)), torch.zeros(100), atol=1e-5, rtol=0)
    globally = pf.PaperFaithfulSGSCN(3, settings(output_normalization="global_standardization"))(x)[0]
    assert abs(float(globally.mean())) < 1e-5 and abs(float(globally.var(unbiased=False)) - 1) < 1e-3
    with pytest.raises(pf.ProtocolSettingError, match="final_batchnorm requires"):
        pf.PaperFaithfulSGSCN(3, settings(layer_order="conv_bn_relu", output_normalization="final_batchnorm"))


def test_eq1_standard_cross_entropy_is_summed_over_pixels():
    s_hat = torch.randn(5, 4, 3)
    labels = s_hat.argmax(0).reshape(-1)
    mean = F.cross_entropy(s_hat.permute(1, 2, 0).reshape(-1, 5), labels)
    torch.testing.assert_close(pf.cross_entropy_sum(s_hat, labels), mean * 12)


def test_eq2_matches_the_printed_double_sum():
    s_hat = torch.randn(4, 5, 6)  # [C, H, W]
    total = 0.0
    width, height = 6, 5
    for k in range(1, width):          # k = 1..W-1 (1-based), width axis
        for l in range(1, height):     # l = 1..H-1, height axis
            here = s_hat[:, l - 1, k - 1]
            total += float((s_hat[:, l - 1, k] - here).abs().sum() + (s_hat[:, l, k - 1] - here).abs().sum())
    assert float(pf.sparse_spatial_sum(s_hat)) == pytest.approx(total, rel=1e-6)


def test_eq3_eq4_match_an_explicit_loop():
    s_hat = torch.randn(3, 4, 5)
    density = torch.softmax(s_hat, dim=0)
    expected = 0.0
    for n in range(3):
        mass = float(density[n].sum())
        ck = sum(k * float(density[n, l, k]) for l in range(4) for k in range(5)) / mass
        cl = sum(l * float(density[n, l, k]) for l in range(4) for k in range(5)) / mass
        expected += sum(((k - ck) ** 2 + (l - cl) ** 2) * float(density[n, l, k]) / mass for l in range(4) for k in range(5))
    value = pf.context_consistency_sum(s_hat, settings(context_density="channel_softmax"))
    assert float(value) == pytest.approx(expected, rel=1e-5)


def test_literal_density_on_a_zero_mean_map_is_refused_not_repaired():
    # Exactly representable zero-sum channels: Eq. 3 divides by sum(S_hat) = 0.
    s_hat = torch.tensor([[[1.0, -1.0], [2.0, -2.0]], [[0.5, 0.5], [-0.5, -0.5]]])
    assert torch.all(s_hat.sum(dim=(1, 2)) == 0)
    with pytest.raises(FloatingPointError, match="zero mass"):
        pf.context_consistency_sum(s_hat, settings(context_density="literal_normalized_map"))


def test_paper_loss_is_the_unweighted_sum_and_differs_from_the_official_loss():
    s_hat = torch.randn(100, 6, 5)
    total, labels, terms = pf.paper_loss(s_hat, settings())
    torch.testing.assert_close(total, terms["ce"] + terms["ss"] + terms["cc"])
    official = json.loads((CONFIGS / "ph2_official_reference.yaml").read_text())["scientific"]
    official_total, _, official_terms = loss_terms(s_hat, official)
    pixels = 6 * 5
    torch.testing.assert_close(terms["ce"], official_terms["ce"] * pixels)  # sum vs mean reduction
    assert official["spatial_weight"] == 5 and "spatial_weight" not in json.loads((CONFIGS / "ph2_paper_faithful.yaml").read_text())["scientific"]
    assert not torch.isclose(total, official_total)


def test_stopping_is_the_declared_stability_convention_not_the_official_rule():
    image = np.random.default_rng(1).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    partition, receipt = pf.predict_paper_faithful(image, settings(max_iterations=3, stability_patience=50), seed=1)
    assert receipt["iterations"] == 3 and receipt["stop_reason"] == "max_iterations"
    assert partition.shape == (8, 8) and partition.dtype == np.int32
    again, receipt_again = pf.predict_paper_faithful(image, settings(max_iterations=3, stability_patience=50), seed=1)
    assert np.array_equal(partition, again) and receipt["trajectory"] == receipt_again["trajectory"]


@pytest.mark.parametrize("missing", ["layer_order", "output_normalization", "context_density", "input_encoding",
                                     "max_iterations", "stability_patience", "relative_loss_tolerance",
                                     "final_forward_mode", "standardization_epsilon"])
def test_every_required_value_is_enforced(missing):
    with pytest.raises(pf.ProtocolSettingError, match=missing):
        pf.validate_settings(settings(**{missing: None}))
    with pytest.raises(pf.ProtocolSettingError, match="context_epsilon"):
        pf.validate_settings(settings(context_density="epsilon_shift"))


def test_profiles_are_separate_and_differ_where_the_evidence_says():
    for name in ("ph2", "sysu_us"):
        faithful = json.loads((CONFIGS / f"{name}_paper_faithful.yaml").read_text())
        reference = json.loads((CONFIGS / f"{name}_official_reference.yaml").read_text())
        assert faithful["profile_class"] == "PAPER_FAITHFUL_REIMPLEMENTATION"
        assert reference["implementation"] == "isolated_GPL_official_code"
        assert faithful["scientific"]["learning_rate"] == reference["scientific"]["learning_rate"]
        assert faithful["scientific"]["momentum"] == reference["scientific"]["momentum"] == 0.9
        assert faithful["scientific"]["conv_layers"] == 3 and reference["scientific"]["n_conv"] == 2
        assert reference["scientific"]["max_iterations"] == 50 and reference["scientific"]["min_labels"] == 3
        assert faithful["implementation_conventions"]["max_iterations"]["value"] is None
        assert "min_labels" not in faithful["scientific"]
        with pytest.raises(ProtocolBlocked) as error:
            load_lock(CONFIGS / f"{name}_paper_faithful.yaml")
        assert any("layer_order" in reason for reason in error.value.reasons)
        assert load_lock(CONFIGS / f"{name}_official_reference.yaml")["profile"] == reference["profile"]


def _instantiated_profile(tmp_path):
    config_dir = tmp_path / "config" / "native"
    shutil.copytree(CONFIGS, config_dir)
    values = {name: {"value": value, "source": "unit-test user choice"} for name, value in FILLED.items()
              if name not in {"stopping_rule", "weight_decay"}}
    spec = __import__("importlib").util.spec_from_file_location("instantiate", ROOT / "scripts/instantiate_native_profile.py")
    tool = __import__("importlib").util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool.instantiate(config_dir / "ph2_paper_faithful.yaml", values, "ph2_paper_faithful_unit_test"), tool


def test_user_instantiation_creates_a_new_profile_and_never_edits_the_base(tmp_path):
    target, tool = _instantiated_profile(tmp_path)
    base_before = (CONFIGS / "ph2_paper_faithful.yaml").read_bytes()
    config = load_lock(target)
    assert config["derived_from"]["profile"] == "ph2_paper_faithful"
    assert config["paper_unspecified"]["layer_order"]["status"].startswith("USER_SUPPLIED")
    assert (tmp_path / "config/native/ph2_paper_faithful.yaml").read_bytes() == base_before
    with pytest.raises(ValueError, match="already has a value"):
        tool.instantiate(target, {"layer_order": {"value": "conv_bn_relu", "source": "x"}}, "again")
    with pytest.raises(ValueError, match="not a declared required field"):
        tool.instantiate(tmp_path / "config/native/ph2_paper_faithful.yaml",
                         {"learning_rate": {"value": 1.0, "source": "x"}}, "bad")


def test_instantiated_profile_runs_end_to_end_with_paper_faithful_provenance(tmp_path):
    import cv2
    target, _ = _instantiated_profile(tmp_path)
    staging = tmp_path / "images"
    staging.mkdir()
    image = np.random.default_rng(3).integers(0, 256, (12, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(staging / "synthetic.png"), image)
    manifest = {"schema": "medical-native.image-only.v1", "dataset": "PH2", "records": [
        {"sample_id": "synthetic", "role": "image", "image_path": str(staging / "synthetic.png"),
         "image_sha256": hashlib.sha256((staging / "synthetic.png").read_bytes()).hexdigest()}]}
    (tmp_path / "inventory.json").write_text(json.dumps(manifest))
    output = tmp_path / "raw"
    result = subprocess.run([sys.executable, str(BASE / "scripts/run_native.py"), "--config", str(target),
                             "--images-manifest", str(tmp_path / "inventory.json"), "--image-root", str(staging),
                             "--output", str(output), "--seed", "1", "--threads", "1"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    raw = verify_raw_run(output)
    assert raw["run"]["provenance"]["implementation_kind"] == "paper_faithful_reimplementation"
    assert raw["run"]["provenance"]["profile_class"] == "PAPER_FAITHFUL_REIMPLEMENTATION"
    assert raw["run"]["config"]["profile"] == "ph2_paper_faithful_unit_test"
    blocked = subprocess.run([sys.executable, str(BASE / "scripts/run_native.py"), "--config",
                              str(CONFIGS / "ph2_paper_faithful.yaml"), "--check-protocol"],
                             capture_output=True, text=True, timeout=60)
    assert blocked.returncode == 2 and "layer_order" in blocked.stdout
