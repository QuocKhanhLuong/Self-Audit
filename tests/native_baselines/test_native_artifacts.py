import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shared_benchmark.native_artifacts import (
    ImageInventory, NativeContractError, ProducerAccessGuard, RawRunWriter, array_hash,
    verify_raw_run, compare_repeat_runs,
)


def make_inventory(root):
    import cv2
    image = np.random.default_rng(2).integers(0, 255, (8, 10, 3), dtype=np.uint8)
    path = root / "image.png"
    cv2.imwrite(str(path), image)
    manifest = {"schema": "medical-native.image-only.v1", "dataset": "SYNTHETIC",
                "records": [{"sample_id": "sample1", "role": "image", "image_path": "image.png",
                             "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
    location = root / "inventory.json"
    location.write_text(json.dumps(manifest))
    return ImageInventory(location), image


def make_raw(root, inventory, image, partition=None):
    config = {"method": "synthetic-method", "dataset": "SYNTHETIC", "profile": "test"}
    writer = RawRunWriter(root, inventory=inventory, config=config, provenance={"seed": 1})
    if partition is None:
        partition = np.zeros((8, 10), np.int32)
    writer.write(inventory.records[0], partition, {"input_native_hw": [8, 10],
                                                "decoded_image_sha256": array_hash(image)})
    writer.seal([])
    return root


def test_seal_verify_repeat_and_no_overwrite(tmp_path):
    inventory, image = make_inventory(tmp_path)
    first = make_raw(tmp_path / "first", inventory, image)
    second = make_raw(tmp_path / "second", inventory, image)
    assert verify_raw_run(first)["seal"]["status"] == "RAW_COMPLETE"
    assert compare_repeat_runs(first, second)["all_equal"]
    with pytest.raises(NativeContractError, match="nonempty"):
        make_raw(first, inventory, image)


@pytest.mark.parametrize("name", ["run.json", "access_log.json", "sample1/metadata.json", "sample1/raw.npz"])
def test_tampering_rejected(tmp_path, name):
    inventory, image = make_inventory(tmp_path)
    output = make_raw(tmp_path / "raw", inventory, image)
    with (output / name).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(NativeContractError):
        verify_raw_run(output)


def test_incomplete_inventory_cannot_seal(tmp_path):
    inventory, _ = make_inventory(tmp_path)
    writer = RawRunWriter(tmp_path / "raw", inventory=inventory,
                          config={"method": "test", "dataset": "test", "profile": "test"}, provenance={})
    with pytest.raises(NativeContractError, match="incomplete"):
        writer.seal([])


def test_unknown_metadata_and_sensitive_paths_rejected(tmp_path):
    inventory, _ = make_inventory(tmp_path)
    document = inventory.document
    document["records"][0]["gt_path"] = "never_exists.png"
    inventory.path.write_text(json.dumps(document))
    with pytest.raises(NativeContractError, match="image-only"):
        ImageInventory(inventory.path)
    document["records"][0].pop("gt_path")
    document["records"][0]["image_path"] = "patient_gt.png"
    inventory.path.write_text(json.dumps(document))
    with pytest.raises(NativeContractError, match="before probing"):
        ImageInventory(inventory.path)


def test_changed_input_rejected(tmp_path):
    inventory, _ = make_inventory(tmp_path)
    (tmp_path / "image.png").write_bytes(b"changed")
    with pytest.raises(NativeContractError, match="input image hash"):
        inventory.read_bgr(inventory.records[0])


def test_access_guard_blocks_open_stat_and_listing(tmp_path):
    inventory, _ = make_inventory(tmp_path)
    forbidden = tmp_path / "patient_gt.png"
    forbidden.write_bytes(b"do not read")
    guard = ProducerAccessGuard(images=[tmp_path / "image.png"], output_root=tmp_path / "out")
    with guard:
        assert (tmp_path / "image.png").read_bytes()
        with pytest.raises(NativeContractError):
            forbidden.read_bytes()
        with pytest.raises(NativeContractError):
            forbidden.stat()
        with pytest.raises(NativeContractError):
            list((tmp_path / "unapproved").iterdir())
    assert len([row for row in guard.log if not row["allowed"]]) == 3


def test_repeat_variation_is_reported_not_overwritten(tmp_path):
    inventory, image = make_inventory(tmp_path)
    a = np.tile(np.array([0, 1] * 5), (8, 1)).astype(np.int32)
    b = a + 5
    first = make_raw(tmp_path / "first", inventory, image, a)
    second = make_raw(tmp_path / "second", inventory, image, b)
    report = compare_repeat_runs(first, second)
    assert not report["all_equal"]
    assert report["samples"][0]["partition_equivalent_up_to_id_permutation"]


def test_native_evaluators_require_seal_before_gt(tmp_path):
    for method in ("DSS_US", "SGSCN"):
        path = ROOT / "baseline" / method / "evaluation/track_b/run.py"
        spec = importlib.util.spec_from_file_location("gate_" + method, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pytest.raises(FileNotFoundError, match="raw_seal"):
            module.evaluate_native(tmp_path, protocol_path=tmp_path / "missing_protocol.yaml",
                                   gt_manifest=tmp_path / "not_opened_gt.json")
    assert not (tmp_path / "not_opened_gt.json").exists()


def test_method_agnostic_infrastructure_import_boundary():
    for path in (ROOT / "src/shared_benchmark").glob("native_*.py"):
        tree = ast.parse(path.read_text())
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(name.startswith(("sgscn", "dss_us")) for name in imports)


def test_trusted_code_root_does_not_allow_annotation_data(tmp_path):
    trusted = tmp_path / "package"
    trusted.mkdir()
    (trusted / "labels.py").write_text("# source code only")
    (trusted / "patient_gt.png").write_bytes(b"must not be read")
    guard = ProducerAccessGuard(images=[], output_root=tmp_path / "output", trusted_roots=[trusted])
    with guard:
        assert (trusted / "labels.py").read_text()
        with pytest.raises(NativeContractError):
            (trusted / "patient_gt.png").read_bytes()


def test_producer_metadata_rejects_evaluation_fields(tmp_path):
    inventory, image = make_inventory(tmp_path)
    writer = RawRunWriter(tmp_path / "raw", inventory=inventory,
                          config={"method": "test", "dataset": "test", "profile": "test"}, provenance={})
    with pytest.raises(NativeContractError, match="GT/oracle fields"):
        writer.write(inventory.records[0], np.zeros((8, 10), np.int32),
                     {"input_native_hw": [8, 10], "decoded_image_sha256": array_hash(image), "gt": [1, 2]})
