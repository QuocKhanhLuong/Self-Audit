import torch
from torch.nn import functional as F

from self_audit.model import build_model
from self_audit.window import SemanticWindow, bilinear_sample, validate_window_config


def tiny_config():
    return {"backbone": "tiny_test", "source_repo": "", "source_commit": "",
            "weights": "", "weights_sha256": ""}


def test_bilinear_sample_matches_grid_sample_interior():
    torch.manual_seed(3)
    feature = torch.randn(1, 2, 5, 6, dtype=torch.double)
    coordinates = torch.rand(1, 3, 4, 2, 2, dtype=torch.double)
    coordinates[..., 0] = coordinates[..., 0] * 2.5 + 1.1
    coordinates[..., 1] = coordinates[..., 1] * 2.0 + 1.1
    values, valid = bilinear_sample(feature, coordinates)
    grid = coordinates.clone()
    grid[..., 0] = 2 * (grid[..., 0] + .5) / feature.shape[-1] - 1
    grid[..., 1] = 2 * (grid[..., 1] + .5) / feature.shape[-2] - 1
    reference = F.grid_sample(feature, grid.reshape(1, -1, 1, 2), align_corners=False)
    reference = reference.reshape(1, 2, 3, 4, 2)
    torch.testing.assert_close(values, reference, rtol=1e-10, atol=1e-10)
    assert valid.all()


def test_bilinear_sample_has_second_derivatives_away_from_boundaries():
    feature = torch.tensor([[[[0.2, 1.0, -0.3], [0.5, 0.7, 1.4],
                             [1.1, -0.8, 0.6]]]], dtype=torch.double, requires_grad=True)
    coordinates = torch.tensor([[[[[1.2, 1.35]]]]], dtype=torch.double, requires_grad=True)

    def fn(feat, coords):
        return bilinear_sample(feat, coords)[0]

    assert torch.autograd.gradcheck(fn, (feature, coordinates), eps=1e-6, atol=1e-5)
    assert torch.autograd.gradgradcheck(fn, (feature, coordinates), eps=1e-6, atol=1e-5)


def test_invalid_footprint_and_all_invalid_attention_are_finite():
    feature = torch.arange(9., dtype=torch.float32).reshape(1, 1, 3, 3)
    coordinates = torch.tensor([[[[[1.0, 1.0], [4.0, 1.0]]]]])
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    mask[:, 0, 0] = False
    values, valid = bilinear_sample(feature, coordinates, mask)
    assert valid.shape == (1, 1, 1, 2)
    assert valid[0, 0, 0, 0] and not valid[0, 0, 0, 1]
    assert values[0, 0, 0, 0, 0] == 4
    bad_coordinates = coordinates.clone()
    bad_coordinates[0, 0, 0, 0, 0] = float("nan")
    bad_coordinates[0, 0, 0, 1, 1] = float("inf")
    bad_values, bad_valid = bilinear_sample(feature, bad_coordinates)
    assert torch.isfinite(bad_values).all()
    assert not bad_valid.any() and torch.equal(bad_values, torch.zeros_like(bad_values))

    window = SemanticWindow(4, 6, {"mode": "adaptive", "dim": 3})
    query = torch.randn(1, 4, 2, 2)
    coarse = torch.randn(1, 6, 1, 1)
    none = torch.zeros(1, 2, 2, dtype=torch.bool)
    output = window(query, coarse, none)
    assert torch.isfinite(output).all() and torch.equal(output, torch.zeros_like(output))
    # The same zero fallback applies when coordinates are invalid by bounds,
    # even without an explicit geometric mask.
    window._coordinates = lambda q, c: q.new_full((1, 2, 2, window.K, 2), 99)
    assert torch.equal(window(query, coarse), torch.zeros_like(output))


def test_window_mask_rejects_partial_padding_footprints():
    window = SemanticWindow(4, 6, {"mode": "fixed", "dim": 3, "radius_min": .25, "initial_radius": .5})
    query = torch.randn(1, 4, 4, 4)
    coarse = torch.randn(1, 6, 2, 2)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    valid[:, :, -1] = False
    output = window(query, coarse, valid)
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, :, :, -1], torch.zeros_like(output[:, :, :, -1]))


def test_fixed_and_adaptive_start_equal_and_baseline_has_no_window_state():
    torch.manual_seed(19)
    baseline, _ = build_model(tiny_config(), synthetic=True)
    baseline_next = torch.rand(5)
    torch.manual_seed(19)
    fixed, _ = build_model(tiny_config(), synthetic=True, window={"mode": "fixed"})
    fixed_next = torch.rand(5)
    assert torch.equal(baseline_next, fixed_next)

    torch.manual_seed(19)
    adaptive, _ = build_model(tiny_config(), synthetic=True, window={"mode": "adaptive"})
    torch.manual_seed(27)
    x = torch.rand(1, 3, 31, 35)
    torch.manual_seed(19)
    disabled, _ = build_model(tiny_config(), synthetic=True, window={"mode": "none"})
    torch.testing.assert_close(baseline(x), disabled(x), rtol=0, atol=0)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, fixed.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(fixed(x), adaptive(x), rtol=0, atol=0)
    for key, value in fixed.state_dict().items():
        if not key.startswith("window.geometry"):
            torch.testing.assert_close(value, adaptive.state_dict()[key], rtol=0, atol=0)
    assert not any(p.requires_grad for p in fixed.window.geometry.parameters())
    assert all(p.requires_grad for p in adaptive.window.geometry.parameters())
    assert not any(key.startswith("window.") for key in baseline.state_dict())


def test_implicit_image_mask_equals_explicit_all_true_on_odd_shape():
    torch.manual_seed(31)
    model, _ = build_model(tiny_config(), synthetic=True, window={"mode": "adaptive"})
    x = torch.rand(1, 3, 31, 35)
    all_true = torch.ones(1, 31, 35, dtype=torch.bool)
    torch.testing.assert_close(model(x), model(x, valid_mask=all_true), rtol=0, atol=0)


def test_window_config_validation_is_closed():
    assert validate_window_config({"mode": "none"})["dim"] == 32
    for bad in ({"mode": "learned"}, {"mode": None}, {"dim": 0}, {"radius_min": 0},
                {"radius_min": 1, "radius_max": 1}, {"offset_limit": -1},
                {"mode": "adaptive", "gamma_init": 0}, {"unknown": 1}):
        try:
            validate_window_config(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid window config: {bad}")
