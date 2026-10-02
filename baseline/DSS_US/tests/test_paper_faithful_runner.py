"""DSS-US paper-faithful runner: official-fallback mechanics and synthetic end-to-end runs.

The end-to-end tests use the pinned licensed DINO ViT-S/8 architecture with a random-weight
checkpoint generated in the test (no pretrained weights, no CAMUS data): they validate
wiring, sealing and provenance only.
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
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src")]
from dss_us import paper_faithful as pf  # noqa: E402
from shared_benchmark.native_artifacts import verify_raw_run  # noqa: E402
from shared_benchmark.native_protocol import load_lock  # noqa: E402

PROFILE = BASE / "config/native/step2_dss_baseline_dss_paper_faithful_declared_conventions.yaml"
DINO = ROOT / ".scratch/dino"


def test_background_is_the_segment_with_most_border_pixels():
    segmap = np.array([[3, 3, 3, 3], [3, 1, 2, 3], [3, 1, 2, 3], [5, 5, 3, 3]])
    relabelled = pf.infer_background_relabel(segmap)
    assert np.array_equal(relabelled == 0, segmap == 3)
    assert set(np.unique(relabelled)) == {0, 1, 2, 5}


def test_boxes_skip_background_and_keep_masks_that_would_vanish():
    segmap = np.zeros((12, 12), int)
    segmap[2:9, 3:10] = 4      # large segment: eroded twice then dilated five times
    segmap[11, 11] = 7         # one pixel: erosion would empty it, so it is kept
    boxes = dict(pf.official_segment_boxes(segmap))
    assert 0 not in boxes
    assert boxes[7] == (6, 12, 6, 12)            # dilated single pixel, clipped by the image
    top, bottom, left, right = boxes[4]
    assert top <= 2 and bottom >= 9 and left <= 3 and right >= 10


def test_small_crops_follow_the_official_padding():
    import torch
    padded = pf._official_pad(torch.ones(1, 3, 5, 9))
    assert tuple(padded.shape) == (1, 3, 13, 17)   # pad = max(8 - size, 8) on bottom/right
    assert float(padded[0, 0, 12, 16]) == 0.0


def test_declared_profile_is_runnable_and_fully_labelled():
    config = load_lock(PROFILE)
    assert config["profile_class"] == "PAPER_FAITHFUL_WITH_DECLARED_CONVENTIONS"
    assert config["scientific"]["step1_segments"] == 15 and config["scientific"]["step2_variant"] == "dss_step2"
    for group in ("paper_unspecified", "implementation_conventions", "required_data"):
        for name, entry in config[group].items():
            assert entry["status"].split(" ")[0] in {"OFFICIAL_CODE_FALLBACK", "IMPLEMENTATION_CONVENTION",
                                                     "BLOCKED_DATA_SUPPLIED_AT_RUNTIME"}, name
    assert pf.validate_dss_step2_settings(pf.dss_settings(config))


def test_missing_data_is_reported_as_blocked_data_without_output(tmp_path):
    output = tmp_path / "raw"
    result = subprocess.run([sys.executable, str(BASE / "scripts/run_native.py"), "--config", str(PROFILE),
                             "--images-manifest", str(tmp_path / "missing.json"), "--image-root", str(tmp_path / "none"),
                             "--output", str(output)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 3, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "BLOCKED_DATA"
    assert any("inventory" in reason for reason in report["reasons"])
    assert any("checkpoint" in reason for reason in report["reasons"])
    assert not output.exists()


def test_step1_and_ours_step2_rows_stay_blocked_with_precise_reasons():
    for name, needle in (("step1_dss_baseline_paper_faithful", "simplecrf==0.2.1.1"),
                         ("step2_ours_comb_ours_paper_faithful", "Ours step2 embeddings")):
        result = subprocess.run([sys.executable, str(BASE / "scripts/run_native.py"), "--config",
                                 str(BASE / "config/native" / f"{name}.yaml"), "--check-protocol"],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 2 and needle in result.stdout


@pytest.fixture()
def synthetic_camus(tmp_path):
    import cv2
    if not (DINO / "hubconf.py").exists():
        pytest.skip("run scripts/pin_native_references.py for the licensed DINO source")
    import torch
    torch.manual_seed(0)
    model = torch.hub.load(str(DINO), "dino_vits8", source="local", pretrained=False)
    checkpoint = tmp_path / "random_vits8.pth"
    torch.save(model.state_dict(), checkpoint)
    staging = tmp_path / "images"
    staging.mkdir()
    rng = np.random.default_rng(0)
    records = []
    for index in range(3):
        yy, xx = np.mgrid[:64, :72]
        image = np.where(((yy - 30) ** 2 + (xx - 36 - 4 * index) ** 2 < 300)[..., None], 70, 180)
        image = np.clip(image + rng.normal(0, 15, (64, 72, 1)), 0, 255).astype(np.uint8).repeat(3, axis=2)
        path = staging / f"patient{index:02d}_frame.png"
        cv2.imwrite(str(path), image)
        records.append({"sample_id": f"p{index}", "role": "image", "image_path": str(path),
                        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps({"schema": "medical-native.image-only.v1", "dataset": "CAMUS", "records": records}))
    return {"checkpoint": checkpoint, "staging": staging, "manifest": manifest,
            "sha": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def test_synthetic_end_to_end_cli_produces_a_sealed_run(tmp_path, synthetic_camus):
    output = tmp_path / "raw"
    result = subprocess.run([sys.executable, str(BASE / "scripts/run_native.py"), "--config", str(PROFILE),
                             "--images-manifest", str(synthetic_camus["manifest"]), "--image-root", str(synthetic_camus["staging"]),
                             "--output", str(output), "--dino-repo", str(DINO),
                             "--dino-checkpoint", str(synthetic_camus["checkpoint"]),
                             "--dino-checkpoint-sha256", synthetic_camus["sha"], "--threads", "1"],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    raw = verify_raw_run(output)
    provenance = raw["run"]["provenance"]
    assert provenance["profile_class"] == "PAPER_FAITHFUL_WITH_DECLARED_CONVENTIONS"
    assert provenance["dino"]["checkpoint_sha256"] == synthetic_camus["sha"]
    assert provenance["cohort_inventory_sha256"] == raw["run"]["image_manifest_sha256"]
    assert not any(not row["allowed"] for row in json.loads((output / "access_log.json").read_text()))
    for sample in ("p0", "p1", "p2"):
        with np.load(output / sample / "raw.npz") as data:
            assert data["partition"].shape == (64, 72)
        metadata = json.loads((output / sample / "metadata.json").read_text())["metadata"]
        assert metadata["step1_segments"] <= 15 and metadata["patch_grid"] == [8, 9]
