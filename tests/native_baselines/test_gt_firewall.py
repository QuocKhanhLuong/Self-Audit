"""Synthetic PH2-style sibling-GT leak regressions for the producer GT firewall."""
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shared_benchmark.native_artifacts import ImageInventory, NativeContractError, ProducerAccessGuard


def write_image(path, value=0):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((6, 7, 3), value, np.uint8))
    return path


@pytest.fixture()
def ph2_layout(tmp_path):
    """Native PH2-like tree: the image and its lesion GT are sibling directories."""
    dataset = tmp_path / "PH2_Dataset"
    case = dataset / "IMD001"
    image = write_image(case / "IMD001_Dermoscopic_Image" / "IMD001.bmp", 7)
    lesion = write_image(case / "IMD001_lesion" / "IMD001_lesion.bmp", 255)
    return {"dataset": dataset, "case": case, "image": image, "lesion": lesion}


def write_manifest(path, images):
    records = [{"sample_id": image.stem, "role": "image", "image_path": str(image),
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()} for image in images]
    path.write_text(json.dumps({"schema": "medical-native.image-only.v1", "dataset": "PH2", "records": records}))
    return path


def staged_copy(tmp_path, ph2_layout):
    staging = tmp_path / "staging" / "ph2_images"
    staging.mkdir(parents=True)
    image = Path(shutil.copy2(ph2_layout["image"], staging / "IMD001.bmp"))
    manifest = write_manifest(tmp_path / "staging" / "inventory.json", [image])
    return ImageInventory(manifest, image_root=staging)


def test_native_layout_root_is_rejected_without_leaking_names(tmp_path, ph2_layout):
    manifest = write_manifest(tmp_path / "inventory.json", [ph2_layout["image"]])
    with pytest.raises(NativeContractError, match="unlisted entries") as error:
        ImageInventory(manifest, image_root=ph2_layout["dataset"])
    assert "lesion" not in str(error.value) and "IMD001" not in str(error.value)


def test_unlisted_file_or_link_inside_staging_root_is_rejected(tmp_path, ph2_layout):
    inventory = staged_copy(tmp_path, ph2_layout)
    copied_mask = inventory.image_root / "IMD001_extra.bmp"
    shutil.copy2(ph2_layout["lesion"], copied_mask)
    with pytest.raises(NativeContractError, match="unlisted entries"):
        ImageInventory(inventory.path, image_root=inventory.image_root)
    copied_mask.unlink()
    (inventory.image_root / "link.bmp").symlink_to(ph2_layout["lesion"])
    with pytest.raises(NativeContractError, match="unlisted entries"):
        ImageInventory(inventory.path, image_root=inventory.image_root)


def test_image_outside_staging_root_is_rejected(tmp_path, ph2_layout):
    staging = tmp_path / "empty_staging"
    staging.mkdir()
    manifest = write_manifest(tmp_path / "inventory.json", [ph2_layout["image"]])
    with pytest.raises(NativeContractError, match="outside the image-only staging root"):
        ImageInventory(manifest, image_root=staging)


def guarded(inventory, tmp_path):
    return ProducerAccessGuard(images=[r["resolved_image_path"] for r in inventory.records],
                               output_root=tmp_path / "output", image_root=inventory.image_root,
                               readable_files=[inventory.path])


def test_sibling_lesion_discovery_is_denied(tmp_path, ph2_layout):
    """Reproduces the audited leak: listing the image's parent revealed IMD001_lesion."""
    case, lesion_dir = ph2_layout["case"], ph2_layout["lesion"].parent
    # The image directory itself is the (image-only) staging root here.
    manifest = write_manifest(tmp_path / "inventory.json", [ph2_layout["image"]])
    inventory = ImageInventory(manifest, image_root=ph2_layout["image"].parent)
    probes = {
        "listdir parent": lambda: os.listdir(case),
        "scandir parent": lambda: list(os.scandir(case)),
        "iterdir parent": lambda: list(case.iterdir()),
        "glob parent": lambda: glob.glob(str(case / "*")),
        "glob recursive": lambda: glob.glob(str(ph2_layout["dataset"] / "**" / "*lesion*"), recursive=True),
        "pathlib glob": lambda: list(case.glob("*")),
        "walk dataset": lambda: list(os.walk(ph2_layout["dataset"])),
        "listdir staging root": lambda: os.listdir(inventory.image_root),
        "exists sibling dir": lambda: os.path.exists(lesion_dir),
        "isdir sibling dir": lambda: os.path.isdir(lesion_dir),
        "Path.exists sibling file": lambda: ph2_layout["lesion"].exists(),
        "lstat sibling": lambda: os.lstat(lesion_dir),
        "access sibling": lambda: os.access(lesion_dir, os.R_OK),
        "stat parent of staging": lambda: os.stat(case),
        "open via traversal": lambda: open(os.path.join(inventory.image_root, "..", "IMD001_lesion",
                                                        "IMD001_lesion.bmp"), "rb").read(),
        "read sibling bytes": lambda: ph2_layout["lesion"].read_bytes(),
    }
    guard = guarded(inventory, tmp_path)
    returned = {}
    with pytest.raises(NativeContractError, match="suppressed"):
        with guard:
            assert inventory.read_bgr(inventory.records[0]).shape == (6, 7, 3)
            os.stat(ph2_layout["image"])
            for name, probe in probes.items():
                before = len(guard.log)
                try:
                    returned[name] = probe()
                except NativeContractError:
                    pass
                assert any(not row["allowed"] for row in guard.log[before:]), f"probe not denied: {name}"
    # Helpers that swallow the denial only ever see a non-revealing negative answer.
    assert all(value in (False, []) for value in returned.values()), returned
    assert str(ph2_layout["image"]) not in {row["path"] for row in guard.log if not row["allowed"]}


def test_clean_guarded_run_exits_normally(tmp_path, ph2_layout):
    inventory = staged_copy(tmp_path, ph2_layout)
    with guarded(inventory, tmp_path) as guard:
        inventory.read_bgr(inventory.records[0])
    assert guard.log and all(row["allowed"] for row in guard.log)


def test_staged_copy_hides_native_tree(tmp_path, ph2_layout):
    inventory = staged_copy(tmp_path, ph2_layout)
    with pytest.raises(NativeContractError, match="suppressed"), guarded(inventory, tmp_path):
        assert inventory.read_bgr(inventory.records[0]).shape == (6, 7, 3)
        for target in (ph2_layout["dataset"], ph2_layout["case"], tmp_path, inventory.image_root.parent):
            with pytest.raises(NativeContractError):
                os.listdir(target)
        with pytest.raises(NativeContractError):
            ph2_layout["lesion"].stat()


def test_guard_refuses_overlapping_roots(tmp_path, ph2_layout):
    inventory = staged_copy(tmp_path, ph2_layout)
    images = [r["resolved_image_path"] for r in inventory.records]
    with pytest.raises(NativeContractError, match="overlap"):
        ProducerAccessGuard(images=images, output_root=inventory.image_root / "out", image_root=inventory.image_root)
    with pytest.raises(NativeContractError, match="overlap"):
        ProducerAccessGuard(images=images, output_root=tmp_path / "out", image_root=inventory.image_root,
                            trusted_roots=[tmp_path])
    with pytest.raises(NativeContractError, match="outside the image-only staging root"):
        ProducerAccessGuard(images=[ph2_layout["lesion"]], output_root=tmp_path / "out",
                            image_root=inventory.image_root)


def test_cli_rejects_native_layout_before_output(tmp_path, ph2_layout):
    manifest = write_manifest(tmp_path / "inventory.json", [ph2_layout["image"]])
    output = tmp_path / "raw"
    result = subprocess.run([
        sys.executable, str(ROOT / "baseline/SGSCN/scripts/run_native.py"),
        "--config", str(ROOT / "baseline/SGSCN/config/native/ph2_official_reference.yaml"),
        "--images-manifest", str(manifest), "--image-root", str(ph2_layout["dataset"]),
        "--output", str(output), "--threads", "1"], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "unlisted entries" in result.stderr
    assert not output.exists()
