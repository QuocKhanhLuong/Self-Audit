import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shared_benchmark.native_artifacts import verify_raw_run


def test_paper_gates_before_any_input_or_output(tmp_path):
    for method, profile in (("DSS_US", "step1_ours_comb"), ("SGSCN", "ph2_paper")):
        output = tmp_path / method
        result = subprocess.run([
            sys.executable, str(ROOT / "baseline" / method / "scripts/run_native.py"),
            "--config", str(ROOT / "baseline" / method / "config/native" / (profile + ".yaml")),
            "--images-manifest", str(tmp_path / "do_not_open_gt.json"), "--output", str(output)],
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 2, result.stderr
        assert json.loads(result.stdout)["status"] == "BLOCKED_PROTOCOL"
        assert not output.exists()


@pytest.mark.parametrize("dataset,profile", [("PH2", "ph2"), ("SYSU-US", "sysu_us")])
def test_sgscn_end_to_end_image_only_seal(tmp_path, dataset, profile):
    import hashlib
    import cv2
    import numpy as np
    image = np.random.default_rng(42).integers(0, 256, (16, 20, 3), dtype=np.uint8)
    staging = tmp_path / "images"
    staging.mkdir()
    path = staging / "synthetic_image.png"
    cv2.imwrite(str(path), image)
    manifest = {"schema": "medical-native.image-only.v1", "dataset": dataset,
                "records": [{"sample_id": "synthetic_io_sample", "role": "image",
                             "image_path": str(path), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
    location = tmp_path / "image_inventory.json"
    location.write_text(json.dumps(manifest))
    output = tmp_path / "raw"
    result = subprocess.run([
        sys.executable, str(ROOT / "baseline/SGSCN/scripts/run_native.py"),
        "--config", str(ROOT / f"baseline/SGSCN/config/native/{profile}_official_reference.yaml"),
        "--images-manifest", str(location), "--image-root", str(staging), "--output", str(output),
        "--seed", "1", "--threads", "1"],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    raw = verify_raw_run(output)
    assert raw["seal"]["status"] == "RAW_COMPLETE"
    assert raw["run"]["config"]["paper_equivalence"] == "UNRESOLVED"
    assert not any(not row["allowed"] for row in json.loads((output / "access_log.json").read_text()))
