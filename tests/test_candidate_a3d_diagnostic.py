"""Contracts for the privileged A-3D TRAIN-GT capacity diagnostic."""

import json

import pytest

from scripts.diagnose_candidate_a3d import _ids_for_budget
from scripts.diagnose_candidate_a3d import diagnose
from self_audit.candidate_a3d.bank import BankConfig3D, QueryBank3D
from self_audit.candidate_a3d.codec import MeasurementCodec3D
from self_audit.candidate_a3d.data import synthetic_fixture
from self_audit.model import digest


def _codec():
    bank = QueryBank3D(BankConfig3D(
        classes=(1, 2, 3), region_grid=(2, 2, 2), boundary_grid=(2, 2, 2),
        region_scales=(0.5,), radius_mm=5.0,
    ))
    return bank, MeasurementCodec3D(bank, value_bits=8)


def test_gt_diagnostic_uses_fixed_r0_and_actual_budget():
    bank, codec = _codec()
    geometry = {"shape": (4, 6, 8), "spacing_zyx": (2.5, 0.7, 1.3)}
    baseline = (0, 8, 16)
    budget = codec.cost(baseline, shape=geometry["shape"], spacing=geometry["spacing_zyx"]) + codec.incremental_cost(
        baseline, 1, shape=geometry["shape"], spacing=geometry["spacing_zyx"])
    ids = _ids_for_budget(codec, bank, geometry, budget, baseline=baseline)
    assert ids[:3] == baseline
    assert len(ids) >= len(baseline)
    assert codec.cost(ids, shape=geometry["shape"], spacing=geometry["spacing_zyx"]) <= budget
    with pytest.raises(ValueError, match="fixed R0"):
        _ids_for_budget(codec, bank, geometry, codec.cost(baseline, shape=geometry["shape"], spacing=geometry["spacing_zyx"]) - 1,
                        baseline=baseline)


def test_gt_diagnostic_rejects_budget_smaller_than_packet_header():
    bank, codec = _codec()
    geometry = {"shape": (4, 6, 8), "spacing_zyx": (2.5, 0.7, 1.3)}
    header = codec.cost((), shape=(1, 1, 1), spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="fixed R0"):
        _ids_for_budget(codec, bank, geometry, header - 1, baseline=(0, 8, 16))


def test_gt_diagnostic_rejects_held_out_or_relabelled_train_manifest(tmp_path, monkeypatch):
    fixture = synthetic_fixture(tmp_path / "fixture", n_labeled=1, n_unlabeled=0,
                                n_evaluation=1, shape_tzyx=(1, 2, 3, 4))
    held_out = json.loads(fixture["evaluation"].read_text())
    held_out["role"] = "training"
    relabelled = tmp_path / "relabelled.json"
    relabelled.write_text(json.dumps(held_out))

    # The structural role guard runs before loading any checkpoint/data.  The
    # dummy arguments therefore cannot accidentally turn this into evidence.
    with pytest.raises(ValueError, match="canonical TRAIN|relabelled"):
        diagnose("not-a-checkpoint", "not-a-checkpoint-sha", relabelled,
                 digest(relabelled), tmp_path / "out", budgets=[2048])
