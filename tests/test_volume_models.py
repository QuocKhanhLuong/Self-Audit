import torch
import pytest
import subprocess
from torch import nn

from self_audit import model as legacy_model
import self_audit.volume.models as volume_models
from self_audit.volume.models import _executed_mac_proxy, build_volume_model


def _volume(seed=7):
    torch.manual_seed(seed)
    return torch.rand(1, 3, 3, 9, 11)


def _fake_dino_repo(tmp_path):
    repo = tmp_path / "fake_dino"
    repo.mkdir()
    marker = repo / "loader.py"
    marker.write_text("# pinned local fake DINO source\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Native Volume Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "loader.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fake-dino"], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return repo, marker, head


class _FakeDino(nn.Module):
    embed_dims = (96, 192, 384, 768)

    def __init__(self):
        super().__init__()
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(3, 96, 4, stride=4),
            nn.Conv2d(96, 192, 2, stride=2),
            nn.Conv2d(192, 384, 2, stride=2),
            nn.Conv2d(384, 768, 2, stride=2),
        ])
        self.stages = nn.ModuleList([nn.Identity() for _ in range(4)])


def test_volume25d_shape_and_disjoint_parameter_groups():
    model, provenance = build_volume_model({"architecture": {"name": "volume25d"}}, synthetic=True)
    out = model(_volume())
    assert out.shape == (1, 4, 3, 9, 11)
    assert provenance["external_teacher"] is False
    assert provenance["encoder_widths"] == [8, 16, 32, 64]
    assert provenance["decoder_widths"] == [8, 16, 24, 32]
    assert provenance["initialization"] == "native_tiny_synthetic"
    groups = model.parameter_groups()
    ids = [id(p) for values in groups.values() for p in values]
    assert len(ids) == len(set(ids)) == sum(p.requires_grad for p in model.parameters())


def test_hybrid_depth_mix_neighbor_influence_and_off_control():
    x = _volume()
    changed = x.clone()
    changed[:, :, 0] = 1.0 - changed[:, :, 0]
    mix, _ = build_volume_model(
        {"architecture": {"name": "hybrid", "width": 8, "depth_mode": "mix"}}, synthetic=True
    )
    # Make the influence deterministic and nonzero rather than relying on a
    # random draw to expose the cross-Z dependency.
    with torch.no_grad():
        layer = mix.body["depth_mixer"].mixer
        layer.weight.zero_()
        layer.bias.zero_()
        layer.weight[:, :, 0, 0, 0] = 0.25
    assert not torch.allclose(mix(x)[:, :, 1], mix(changed)[:, :, 1])
    off, _ = build_volume_model(
        {"architecture": {"name": "hybrid", "width": 8, "depth_mode": "off"}}, synthetic=True
    )
    assert torch.allclose(off(x)[:, :, 1], off(changed)[:, :, 1])


def test_hybrid_gradients_reach_encoder_and_depth_mixer():
    model, provenance = build_volume_model(
        {"architecture": {"name": "hybrid", "width": 8, "depth_mode": "mix", "skip_fusion": "delayed"}},
        synthetic=True,
    )
    logits = model(_volume()).mean()
    logits.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.body["depth_mixer"].parameters())
    assert provenance["skip_fusion"] == "delayed"


def test_hybrid_zero_residual_is_exact_native_student_function():
    torch.manual_seed(11)
    model, _ = build_volume_model(
        {
            "model": {"backbone": "tiny_test"},
            "architecture": {"name": "hybrid", "depth_mode": "mix", "residual_init": "zero"},
            "window": {"mode": "fixed"},
        },
        synthetic=True,
    )
    x = _volume(seed=12)
    with torch.no_grad():
        actual = model(x)
        flat = x.permute(0, 2, 1, 3, 4).reshape(3, 3, 9, 11)
        expected = model.slice_model(flat).reshape(1, 3, 4, 9, 11).permute(0, 2, 1, 3, 4)
    assert torch.equal(actual, expected)

    # Every G6 skip mode uses the same zero-final-projection contract.
    for skip_mode in ("delayed", "matched_additive"):
        candidate, _ = build_volume_model(
            {
                "architecture": {
                    "name": "hybrid",
                    "depth_mode": "mix",
                    "residual_init": "zero",
                    "skip_fusion": skip_mode,
                }
            },
            synthetic=True,
        )
        assert torch.equal(candidate(x), candidate.slice_model(flat).reshape(1, 3, 4, 9, 11).permute(0, 2, 1, 3, 4))

    # The depth-off comparator uses the same zero-final-projection contract;
    # it must construct and preserve parity without a dormant mixer.
    off, _ = build_volume_model(
        {"architecture": {"name": "hybrid", "depth_mode": "off", "residual_init": "zero"}},
        synthetic=True,
    )
    assert torch.equal(off(x), off.slice_model(flat).reshape(1, 3, 4, 9, 11).permute(0, 2, 1, 3, 4))


def test_hybrid_fork_rng_preserves_common_post_init_stream():
    torch.manual_seed(123)
    build_volume_model({"architecture": {"name": "volume25d"}}, synthetic=True)
    reference = torch.rand(7)
    torch.manual_seed(123)
    build_volume_model(
        {"architecture": {"name": "hybrid", "depth_mode": "mix", "residual_init": "zero"}},
        synthetic=True,
    )
    after = torch.rand(7)
    assert torch.equal(after, reference)


def test_hybrid_respects_explicit_disabled_window_control():
    disabled, _ = build_volume_model(
        {"architecture": {"name": "hybrid", "depth_mode": "off"}, "window": {"mode": "none"}},
        synthetic=True,
    )
    assert disabled.slice_model.window is None


def test_planar_control_has_active_per_slice_parameters():
    model, _ = build_volume_model(
        {"architecture": {"name": "hybrid", "width": 8, "depth_mode": "planar_control"}}, synthetic=True
    )
    loss = model(_volume()).square().mean()
    loss.backward()
    residuals = [model.body["depth_mixer"], *model.body["depth_residuals"]]
    assert all(any(p.grad is not None and torch.isfinite(p.grad).all() for p in r.parameters()) for r in residuals)
    x = _volume()
    y = x.clone()
    y[:, :, 0] = 1.0 - y[:, :, 0]
    assert torch.allclose(model(x)[:, :, 1], model(y)[:, :, 1])


@pytest.mark.parametrize("depth_mode", ["mix", "planar_control"])
def test_zero_residual_only_terminal_projection_and_learns_input_dependence(depth_mode):
    torch.manual_seed(19)
    model, _ = build_volume_model(
        {
            "architecture": {"name": "hybrid", "depth_mode": depth_mode, "residual_init": "zero"},
            "window": {"mode": "fixed"},
        },
        synthetic=True,
    )
    residuals = [model.body["depth_mixer"], *model.body["depth_residuals"]]
    for residual in residuals:
        convs = [module for module in residual.mixer.modules() if isinstance(module, torch.nn.Conv3d)]
        assert convs and all(torch.count_nonzero(conv.weight).item() > 0 for conv in convs[:-1])
        assert torch.count_nonzero(convs[-1].weight).item() == 0
    skip_convs = [module for module in model.skip.modules() if isinstance(module, torch.nn.Conv3d)]
    assert all(torch.count_nonzero(conv.weight).item() > 0 for conv in skip_convs[:-1])
    assert torch.count_nonzero(skip_convs[-1].weight).item() == 0

    x = _volume(seed=20)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        optimizer.step()
    changed = x.clone()
    changed[:, :, 1] = 1.0 - changed[:, :, 1]
    def residual_center(value):
        b, _, z, h, w = value.shape
        flat = value.permute(0, 2, 1, 3, 4).reshape(b * z, 3, h, w)
        native = model.slice_model(flat).reshape(b, z, 4, h, w).permute(0, 2, 1, 3, 4)
        return (model(value) - native)[:, :, 1]

    center = residual_center(x)
    center_changed = residual_center(changed)
    # The learned residual must respond to image content, not only add a
    # constant bias after the terminal projection update.
    assert not torch.allclose(center, center_changed)
    assert center.std() > 0


def test_non_synthetic_requires_hash_pinned_dino_fields():
    with pytest.raises(ValueError, match="requires source_repo"):
        build_volume_model(
            {"model": {"backbone": "dinov3_convnext_tiny"}, "architecture": {"name": "volume25d"}},
            synthetic=False,
        )


def test_non_synthetic_loader_seam_accepts_verified_mock(monkeypatch):
    calls = []

    def fake_loader(config, *, synthetic, window):
        calls.append((config, synthetic, window))
        return legacy_model.NativeStudent(legacy_model.TinyEncoder(), widths=(8, 16, 24, 32), window=window), {
            "pretrained": True,
            "weights_sha256": config["weights_sha256"],
            "source_commit": config["source_commit"],
        }

    monkeypatch.setattr(volume_models, "build_model", fake_loader)
    model, provenance = build_volume_model(
        {
            "model": {
                "backbone": "dinov3_convnext_tiny",
                "source_repo": "/verified/source",
                "source_commit": "a" * 40,
                "weights": "/verified/weights.pt",
                "weights_sha256": "b" * 64,
            },
            "architecture": {"name": "volume25d"},
            "window": {"mode": "none"},
        },
        synthetic=False,
    )
    assert calls and calls[0][1] is False
    assert provenance["native_provenance"]["pretrained"] is True
    assert model(_volume()).shape == (1, 4, 3, 9, 11)


def test_architecture_only_dino_uses_clean_local_source_without_weights(monkeypatch, tmp_path):
    repo, marker, head = _fake_dino_repo(tmp_path)
    missing_weights = tmp_path / "missing-pretrained.pth"
    calls = []

    def fake_hub_load(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeDino()

    monkeypatch.setattr(volume_models.torch.hub, "load", fake_hub_load)
    config = {
        "model": {
            "backbone": "dinov3_convnext_tiny",
            "source_repo": str(repo),
            "source_commit": head,
            "weights": str(missing_weights),
            "weights_sha256": "a" * 64,
        },
        "architecture": {
            "name": "volume25d",
            "widths": [96, 128, 192, 256],
            "z_kernels": [[1, 3, 3]],
            "z_strides": [[1, 1, 1]],
        },
        "window": {"mode": "none"},
    }
    model, provenance = build_volume_model(config, synthetic=False, architecture_only=True)
    assert provenance["architecture_only"] is True
    assert provenance["initialization"] == "architecture_only_uninitialized"
    assert provenance["native_provenance"]["pretrained"] is False
    assert provenance["encoder_widths"] == [96, 192, 384, 768]
    assert provenance["decoder_widths"] == [96, 128, 192, 256]
    assert tuple(model.slice_model.encoder.channels) == (96, 192, 384, 768)
    assert tuple(layer.out_channels for layer in model.slice_model.lateral) == (96, 128, 192, 256)
    assert calls and calls[0][0][0] == str(repo.resolve())
    assert calls[0][1].get("source") == "local"
    assert calls[0][1].get("pretrained") is False
    assert not missing_weights.exists()

    # A complete exported state replays strictly into a fresh architecture-only
    # construction; no original weight file is consulted on the second load.
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    restored, _ = build_volume_model(config, synthetic=False, architecture_only=True)
    restored.load_state_dict(state, strict=True)
    assert all(torch.equal(state[name], value) for name, value in restored.state_dict().items())

    # The normal training route retains strict pretrained-file validation.
    with pytest.raises((FileNotFoundError, ValueError)):
        build_volume_model(config, synthetic=False, architecture_only=False)

    # Source pinning remains mandatory even when the weight path is ignored.
    mismatch = dict(config)
    mismatch["model"] = dict(config["model"])
    mismatch["model"]["source_commit"] = "0" * 40
    with pytest.raises(ValueError, match="clean recorded commit"):
        build_volume_model(mismatch, synthetic=False, architecture_only=True)

    marker.write_text("# tracked dirty source\n")
    with pytest.raises(ValueError, match="clean recorded commit"):
        build_volume_model(config, synthetic=False, architecture_only=True)


def test_cnn3d_scratch_anisotropic_shape_and_groups():
    model, provenance = build_volume_model(
        {
            "architecture": {
                "name": "cnn3d",
                "width": 8,
                "z_kernels": [[1, 3, 3], [3, 3, 3]],
                "z_strides": [[1, 1, 1], [1, 2, 2]],
            }
        },
        synthetic=True,
    )
    x = _volume()
    out = model(x)
    assert out.shape == (1, 4, 3, 9, 11)
    side_changed = x.clone()
    side_changed[:, 0] = 1.0 - side_changed[:, 0]
    side_changed[:, 2] = 1.0 - side_changed[:, 2]
    assert torch.allclose(out, model(side_changed))
    assert provenance["initialization"] == "scratch"
    assert hasattr(model, "decoder") and len(model.decoder) == 1
    groups = model.parameter_groups()
    assert {id(p) for values in groups.values() for p in values}.__len__() == sum(len(v) for v in groups.values())
    loss = out.square().mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.decoder.parameters())
    for depth, height, width in ((1, 7, 9), (2, 5, 13), (4, 11, 15)):
        assert model(torch.rand(1, 3, depth, height, width)).shape == (1, 4, depth, height, width)


def test_cnn3d_stage_width_tail_is_honored():
    model, provenance = build_volume_model(
        {
            "architecture": {
                "name": "cnn3d",
                "widths": [8, 24],
                "z_kernels": [[1, 3, 3], [3, 3, 3]],
                "z_strides": [[1, 1, 1], [1, 1, 1]],
            }
        },
        synthetic=True,
    )
    assert model.stage_widths == (8, 24)
    assert provenance["parameter_count"] == sum(parameter.numel() for parameter in model.parameters())


def test_architecture_controls_are_not_silently_ignored():
    with pytest.raises(ValueError, match="positive integer"):
        build_volume_model({"architecture": {"name": "cnn3d", "width": "not-a-width"}}, synthetic=True)
    with pytest.raises(ValueError, match="four native stages"):
        build_volume_model(
            {"architecture": {"name": "hybrid", "widths": [8, 16], "depth_mode": "off"}}, synthetic=True
        )
    with pytest.raises(ValueError, match="z_strides"):
        build_volume_model(
            {
                "architecture": {
                    "name": "hybrid",
                    "depth_mode": "mix",
                    "z_kernels": [[3, 1, 1]],
                    "z_strides": [[1, 2, 2]],
                }
            },
            synthetic=True,
        )
    with pytest.raises(ValueError, match="depthwise"):
        build_volume_model(
            {
                "architecture": {
                    "name": "hybrid",
                    "depth_mode": "mix",
                    "z_kernels": [[3, 3, 3]],
                }
            },
            synthetic=True,
        )
    with pytest.raises(ValueError, match="off/planar_control"):
        build_volume_model(
            {
                "architecture": {
                    "name": "hybrid",
                    "depth_mode": "off",
                    "z_kernels": [[3, 1, 1]],
                }
            },
            synthetic=True,
        )
    with pytest.raises(ValueError, match="neutral values"):
        build_volume_model(
            {"architecture": {"name": "cnn3d", "depth_mode": "mix"}}, synthetic=True
        )
    with pytest.raises(ValueError, match="does not support z_strides"):
        build_volume_model(
            {"architecture": {"name": "volume25d", "z_strides": [[1, 2, 2]]}}, synthetic=True
        )


def test_width_lists_and_hybrid_depth_kernels_change_capacity():
    narrow, _ = build_volume_model(
        {
            "architecture": {
                "name": "hybrid",
                "widths": [8, 16, 24, 32],
                "depth_mode": "mix",
                "z_kernels": [[3, 1, 1]],
            }
        },
        synthetic=True,
    )
    wider, _ = build_volume_model(
        {
            "architecture": {
                "name": "hybrid",
                "widths": [8, 16, 24, 32],
                "width": 16,
                "depth_mode": "mix",
                "z_kernels": [[5, 1, 1]],
            }
        },
        synthetic=True,
    )
    assert narrow.capacity_report()["parameters"] != wider.capacity_report()["parameters"]
    assert narrow.capacity_report()["macs_per_output_voxel"] != wider.capacity_report()["macs_per_output_voxel"]


def test_capacity_report_is_exact_for_whole_model_convolutions():
    model, provenance = build_volume_model(
        {"architecture": {"name": "hybrid", "depth_mode": "mix", "residual_init": "zero"}},
        synthetic=True,
    )
    report = provenance["capacity"]
    assert report["parameters"] == sum(parameter.numel() for parameter in model.parameters())
    assert report["macs_per_output_voxel"] == (
        report["native"]["macs_per_output_voxel"]
        + report["new_residual"]["macs_per_output_voxel"]
        + report["skip"]["macs_per_voxel"]
    )
    proxy = report["executed_mac_proxy"]
    assert proxy["input_shape"] == [1, 3, 3, 32, 32]
    assert proxy["macs"] > 0 and proxy["layers"]
    assert "normalization" in proxy["omitted_ops"]


def test_capacity_proxy_matches_same_shape_across_g6_controls():
    reports = {}
    for label, skip_fusion in (("S01", "delayed"), ("S02", "matched_additive")):
        model, provenance = build_volume_model(
            {
                "architecture": {
                    "name": "hybrid",
                    "widths": [8, 16, 24, 32],
                    "depth_mode": "mix",
                    "skip_fusion": skip_fusion,
                    "residual_init": "zero",
                    "z_kernels": [[3, 1, 1]],
                }
            },
            synthetic=True,
        )
        reports[label] = provenance["capacity"]
        assert reports[label]["executed_mac_proxy"]["input_shape"] == [1, 3, 3, 32, 32]
    assert reports["S01"]["parameters"] != reports["S02"]["parameters"]
    assert reports["S01"]["executed_mac_proxy"]["macs"] != reports["S02"]["executed_mac_proxy"]["macs"]


def test_executed_mac_proxy_accounts_for_group_divisor():
    module = nn.Sequential(
        nn.Conv3d(3, 4, kernel_size=1),
        nn.Conv3d(4, 4, kernel_size=3, padding=1, groups=2),
    )
    proxy = _executed_mac_proxy(module, (1, 3, 2, 4, 5))
    grouped = [record for record in proxy["layers"] if record["kind"] == "conv3d"][1]
    assert grouped["output_elements"] == 1 * 4 * 2 * 4 * 5
    assert grouped["macs"] == grouped["output_elements"] * (4 // 2) * (3 * 3 * 3)


def test_invalid_volume_shapes_fail_closed():
    model, _ = build_volume_model({"architecture": {"name": "hybrid"}}, synthetic=True)
    try:
        model(torch.zeros(1, 1, 2, 4, 4))
    except ValueError as exc:
        assert "Bx3xZxHxW" in str(exc)
    else:
        raise AssertionError("invalid channel count must fail closed")
