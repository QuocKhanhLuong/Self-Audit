"""DSS-US on the frozen ACDC contract: synthetic end-to-end producer runs and CLI gates.

Uses the pinned licensed DINO ViT-S/8 architecture with a random-weight checkpoint generated
in the test (no pretrained weights) and synthetic NIfTI fixtures (no real ACDC data): wiring,
sealing and provenance only.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src"), str(BASE / "scripts"), str(ROOT / "tests/native_baselines"),
                str(ROOT / "scripts")]
import acdc_synthetic  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("dss_us_run_acdc", BASE / "scripts/run_acdc.py")
run_acdc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_acdc)
from shared_benchmark.acdc_native import verify_acdc_seal_receipt  # noqa: E402
from shared_benchmark.native_protocol import ProtocolBlocked  # noqa: E402

ADAPTATION = BASE / "config/acdc/acdc_step2_dss_baseline_dss_paper_faithful_declared_conventions.json"
DINO = ROOT / ".scratch/dino"


@pytest.fixture(scope="module")
def random_dino(tmp_path_factory):
    if not (DINO / "hubconf.py").exists():
        pytest.skip("run scripts/pin_native_references.py for the licensed DINO source")
    import torch
    torch.manual_seed(0)
    model = torch.hub.load(str(DINO), "dino_vits8", source="local", pretrained=False)
    checkpoint = tmp_path_factory.mktemp("weights") / "random_vits8.pth"
    torch.save(model.state_dict(), checkpoint)
    return checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def _run(tmp_path, dino, name):
    contract, records, image_root, _ = acdc_synthetic.build(tmp_path / f"data_{name}", patients=("patient001", "patient002"))
    result = run_acdc.run(contract=contract, records=records, image_root=image_root, output=tmp_path / name,
                          receipt=tmp_path / "receipts" / f"{name}.json", adaptation_path=ADAPTATION, dino_repo=DINO,
                          dino_checkpoint=dino[0], dino_checkpoint_sha256=dino[1], threads=2, extra_trusted_roots=[HERE])
    return result, records


def test_adaptation_is_protocol_ready():
    result = subprocess.run([sys.executable, str(BASE / "scripts/run_acdc.py"), "--adaptation", str(ADAPTATION),
                             "--check-protocol"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "ACDC_PRODUCER_READY"
    assert report["profile_class"] == "PAPER_FAITHFUL_WITH_DECLARED_CONVENTIONS"


def test_missing_data_and_checkpoint_are_blocked_data_without_output(tmp_path):
    output = tmp_path / "raw"
    result = subprocess.run([sys.executable, str(BASE / "scripts/run_acdc.py"), "--adaptation", str(ADAPTATION),
                             "--image-root", str(tmp_path / "absent"), "--output", str(output),
                             "--receipt", str(tmp_path / "receipt.json")], capture_output=True, text=True, timeout=120)
    assert result.returncode == 3, result.stderr
    reasons = json.loads(result.stdout)["reasons"]
    assert any("ACDC image root" in reason for reason in reasons)
    assert any("checkpoint" in reason for reason in reasons)
    assert not output.exists()


def test_blocked_rows_cannot_be_adapted():
    import declare_native_acdc_adaptations as declare
    for profile in ("step1_dss_baseline_paper_faithful", "step1_ours_comb_paper_faithful",
                    "step2_ours_comb_ours_paper_faithful"):
        with pytest.raises(ProtocolBlocked):
            declare.adaptation("DSS_US", profile)
    assert sorted(p.name for p in (BASE / "config/acdc").iterdir()) == [ADAPTATION.name]


def test_synthetic_end_to_end_run_is_sealed_and_deterministic(tmp_path, random_dino):
    first, records = _run(tmp_path, random_dino, "first")
    second, _ = _run(tmp_path, random_dino, "second")
    verified = verify_acdc_seal_receipt(tmp_path / "first", tmp_path / "receipts" / "first.json")
    run = verified["run"]
    assert run["provenance"]["dino"]["checkpoint_sha256"] == random_dino[1]
    assert run["provenance"]["step2_fit_cohort"] == sorted(r["sample_id"] for r in records)
    assert run["provenance"]["paper_equivalence"].startswith("NOT_EXACT_PAPER_REPRODUCTION")
    for record in records:
        artifact = verified["artifacts"][record["sample_id"]]
        provenance = artifact.metadata["baseline_provenance"]
        assert artifact.partition.shape == (224, 224) and artifact.partition.dtype == np.int32
        assert set(np.unique(artifact.partition)) <= set(range(15))  # anonymous official Step II IDs
        assert provenance["patch_grid"] == [28, 28] and provenance["step1_segments"] <= 15
        assert artifact.metadata["seed"] == 1  # kmeans_seed of the declared profile
        assert np.array_equal(artifact.partition, second["verified"]["artifacts"][record["sample_id"]].partition)
    assert all(entry["allowed"] for entry in json.loads((tmp_path / "first" / "access_log.json").read_text()))
