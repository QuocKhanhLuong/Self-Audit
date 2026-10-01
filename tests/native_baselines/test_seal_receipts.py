"""External write-once seal receipts and no-reseal regressions."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shared_benchmark.native_artifacts import (
    NativeContractError, RawRunWriter, array_hash, file_hash, verify_raw_run, verify_seal_receipt,
    write_seal_receipt,
)
from shared_benchmark.native_protocol import ProtocolBlocked, value_hash
from test_native_artifacts import load_evaluator, make_inventory, make_raw


@pytest.fixture()
def sealed(tmp_path):
    inventory, image = make_inventory(tmp_path)
    raw = make_raw(tmp_path / "raw", inventory, image)
    receipt = tmp_path / "receipts" / "raw_receipt.json"
    write_seal_receipt(raw, receipt)
    return raw, receipt, inventory, image


def reseal_after_tamper(raw):
    """What an attacker/careless rewrite would do: change raw, recompute a valid internal seal."""
    np.savez_compressed(raw / "sample1/raw.npz", partition=np.ones((8, 10), "<i4"))
    metadata = json.loads((raw / "sample1/metadata.json").read_text())
    metadata["partition_sha256"] = array_hash(np.ones((8, 10), "<i4"))
    (raw / "sample1/metadata.json").write_text(json.dumps(metadata))
    seal = json.loads((raw / "raw_seal.json").read_text())
    seal.pop("seal_sha256")
    seal["samples"][0].update(raw_file_sha256=file_hash(raw / "sample1/raw.npz"),
                              metadata_sha256=file_hash(raw / "sample1/metadata.json"),
                              partition_sha256=metadata["partition_sha256"])
    seal["seal_sha256"] = value_hash(seal)
    (raw / "raw_seal.json").write_text(json.dumps(seal))
    verify_raw_run(raw)  # the internal seal alone is fooled


def test_receipt_verifies_and_is_write_once(sealed):
    raw, receipt, _, _ = sealed
    assert verify_seal_receipt(raw, receipt)["receipt"]["seal_sha256"] == verify_raw_run(raw)["seal"]["seal_sha256"]
    assert not os.stat(receipt).st_mode & 0o222
    with pytest.raises(FileExistsError):
        write_seal_receipt(raw, receipt)


def test_receipt_cannot_live_inside_raw_root(tmp_path):
    inventory, image = make_inventory(tmp_path)
    raw = make_raw(tmp_path / "raw", inventory, image)
    with pytest.raises(NativeContractError, match="outside the raw root"):
        write_seal_receipt(raw, raw / "receipt.json")
    verify_raw_run(raw)  # nothing was added to the sealed root


def test_tampered_and_resealed_raw_is_rejected(sealed):
    raw, receipt, _, _ = sealed
    reseal_after_tamper(raw)
    with pytest.raises(NativeContractError, match="finalized seal receipt"):
        verify_seal_receipt(raw, receipt)


def test_edited_receipt_is_rejected(sealed):
    raw, receipt, _, _ = sealed
    document = json.loads(receipt.read_text())
    document["seal_sha256"] = "0" * 64
    os.chmod(receipt, 0o644)
    receipt.write_text(json.dumps(document))
    with pytest.raises(NativeContractError, match="receipt hash"):
        verify_seal_receipt(raw, receipt)


def test_writer_cannot_reseal_or_append(tmp_path):
    inventory, image = make_inventory(tmp_path)
    writer = RawRunWriter(tmp_path / "raw", inventory=inventory,
                          config={"method": "test", "dataset": "test", "profile": "test"}, provenance={})
    writer.write(inventory.records[0], np.zeros((8, 10), np.int32),
                 {"input_native_hw": [8, 10], "decoded_image_sha256": array_hash(image)})
    writer.seal([])
    with pytest.raises(NativeContractError, match="already sealed"):
        writer.seal([])
    with pytest.raises(NativeContractError, match="already sealed"):
        writer.write(inventory.records[0], np.zeros((8, 10), np.int32),
                     {"input_native_hw": [8, 10], "decoded_image_sha256": array_hash(image)})


@pytest.mark.parametrize("method,profile", [("DSS_US", "step2_ours_comb_ours"), ("SGSCN", "ph2_official_reference")])
def test_evaluators_verify_receipt_before_protocol_and_gt(sealed, method, profile):
    raw, receipt, _, _ = sealed
    gt = raw.parent / "never_opened_gt.json"
    config = ROOT / "baseline" / method / "config/native" / f"{profile}.yaml"
    evaluator = load_evaluator(method)
    seal_before = file_hash(raw / "raw_seal.json")
    # Valid receipt: reaches the (still blocked) original-evaluator gate, never GT.
    with pytest.raises(ProtocolBlocked):
        evaluator.evaluate_native(raw, seal_receipt=receipt, protocol_path=config, gt_manifest=gt)
    assert file_hash(raw / "raw_seal.json") == seal_before
    reseal_after_tamper(raw)
    with pytest.raises(NativeContractError, match="finalized seal receipt"):
        evaluator.evaluate_native(raw, seal_receipt=receipt, protocol_path=config, gt_manifest=gt)
    assert not gt.exists()
