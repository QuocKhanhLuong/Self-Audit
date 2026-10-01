"""CUTS training records and enforces the canonical environment contract."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
for path in (ROOT / "baseline" / "CUTS" / "src", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import environment_contract  # noqa: E402
from cardiac_benchmark import provenance  # noqa: E402


def test_environment_identity_records_the_contract_in_the_canonical_environment():
    identity = provenance.environment_identity()
    contract = identity["environment_contract"]
    assert contract["environment_id"] == "self-audit-canonical"
    assert contract["official"] is True and contract["problems"] == []
    assert contract["lock_sha256"]


def test_environment_identity_refuses_a_non_canonical_environment(monkeypatch):
    def mismatch(*args, **kwargs):
        raise environment_contract.EnvironmentMismatch({
            "environment_id": "self-audit-canonical", "environment_version": "1",
            "problems": ["numpy 2.2.6 != 1.26.4"], "official": False,
        })

    monkeypatch.setattr(environment_contract, "require_official_environment", mismatch)
    with pytest.raises(environment_contract.EnvironmentMismatch, match="numpy 2.2.6"):
        provenance.environment_identity()
