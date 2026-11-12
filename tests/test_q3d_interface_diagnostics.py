"""SOFTWARE_ONLY checks of the spec section 13 code-interface diagnostic."""

import torch
from torch import nn

from self_audit.candidate_q3d.engine import _code_interface_diagnostics


def test_gate_and_norm_ratio_report_the_actual_contribution():
    model = nn.Module()
    model.vq_gate = nn.Parameter(torch.tensor(-0.2))
    model.quantized_projector = nn.Identity()
    outputs = {"f4": torch.ones(1, 4, 2, 3, 3), "quantized": torch.full((1, 4, 2, 3, 3), 2.0)}
    result = _code_interface_diagnostics(model, outputs)
    assert abs(result["vq_gate"] + 0.2) < 1e-7
    assert abs(result["code_to_f4_norm_ratio"] - 0.4) < 1e-7
    assert result["interface_claim_supported"]
    model.vq_gate.data.zero_()
    result = _code_interface_diagnostics(model, outputs)
    assert result["code_to_f4_norm_ratio"] == 0
    assert not result["interface_claim_supported"]


def test_zero_dense_norm_is_explicit_and_does_not_invent_a_ratio():
    model = nn.Module()
    model.vq_gate = nn.Parameter(torch.tensor(0.2))
    model.quantized_projector = nn.Identity()
    result = _code_interface_diagnostics(model, {
        "f4": torch.zeros(1, 4, 2, 3, 3), "quantized": torch.ones(1, 4, 2, 3, 3),
    })
    assert result["status"] == "ZERO_F4_NORM"
    assert result["code_to_f4_norm_ratio"] is None
    assert not result["interface_claim_supported"]
