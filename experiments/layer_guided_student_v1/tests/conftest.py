"""Load the independently hash-verified oracle under a distinct package name."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "src"))


@pytest.fixture(scope="session")
def oracle(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "reference_materializer",
        REPO / "scripts/materialize_layer_upgrade_reference.py",
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    payloads = verifier.verified_payloads()
    package = tmp_path_factory.mktemp("verified_oracle") / "oracle_layer_upgrade"
    package.mkdir()
    for name, data in payloads.items():
        if name.startswith("layer_upgrade/"):
            (package / Path(name).name).write_bytes(data)
    spec = importlib.util.spec_from_file_location(
        "oracle_layer_upgrade",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import layer_upgrade

    assert module is not layer_upgrade
    assert module.LayerGuidedStudent is not layer_upgrade.LayerGuidedStudent
    assert Path(module.__file__).resolve() != Path(layer_upgrade.__file__).resolve()
    return module


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261015)
    yield
    torch.use_deterministic_algorithms(previous)
