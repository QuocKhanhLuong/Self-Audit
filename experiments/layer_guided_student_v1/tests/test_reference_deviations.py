"""Preserved source limits; these tests document rather than repair the oracle."""

import torch

from layer_upgrade import CenterDifferenceInput, project_direction


def test_float32_projection_receipt_is_not_post_cast_certificate(oracle):
    torch.manual_seed(0)
    for _ in range(4):
        scale = torch.randint(-4, 8, (1,)).item()
        direction = (torch.randn(20) * 10**scale).float()
        guards = torch.randn(4, 20).float()
    result, receipt = project_direction(direction, guards)
    expected, expected_receipt = oracle.project_direction(direction, guards)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert receipt == expected_receipt and receipt["status"] == "KKT_VERIFIED"
    # Actual finite-step acceptance is necessary even when the QP status passes.
    assert float((guards.double() @ result.double()).min()) < -0.01


def test_reference_nonfinite_normalization_customization(oracle):
    for module in (CenterDifferenceInput, oracle.CenterDifferenceInput):
        malformed = module(mean=(float("nan"), 0.456, 0.406))
        assert not torch.isfinite(malformed(torch.ones(1, 3, 4, 4))).all()
