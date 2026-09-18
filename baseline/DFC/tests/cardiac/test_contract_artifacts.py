import json
from pathlib import Path
import numpy as np
import pytest
from cardiac_benchmark.dataset import DataContractError, load_primary_2d, normalize_central_slice
from cardiac_benchmark.freeze import FreezeError, create_freeze, validate_freeze
from cardiac_benchmark.manifest import ManifestError, make_fixture_manifest, validate_manifest, validate_scientific_run
from cardiac_benchmark.output import AttemptLedger, write_raw_artifact
from cardiac_benchmark.provenance import canonical_json, file_sha256, sha256_json

def fixture(tmp_path):
    a = np.arange(3 * 9 * 13, dtype=np.float32).reshape(3,9,13); np.save(tmp_path / "image_0.npy", a)
    return make_fixture_manifest(tmp_path), a

def test_schema_fixture_and_scientific_fail_closed(tmp_path):
    manifest, _ = fixture(tmp_path); validate_manifest(manifest)
    with pytest.raises(ManifestError): validate_scientific_run(manifest, tmp_path)
    bad = dict(manifest); bad["records"] = [dict(manifest["records"][0], mask="forbidden")]
    with pytest.raises(ManifestError): validate_manifest(bad)

def test_loader_is_central_only_geometry_and_nonfinite_rejection(tmp_path):
    manifest, original = fixture(tmp_path); record = next(row for row in manifest["records"] if row["slice_index"] == 1)
    tensor, meta = load_primary_2d(record, tmp_path)
    assert tensor.shape == (1,1,9,13) and tensor.dtype == __import__("torch").float32
    assert meta["central_slice"] == 1 and abs(float(tensor.mean())) < 1e-5
    altered = original.copy(); altered[0] += 1e6; altered[2] -= 1e6; np.save(tmp_path / "image_0.npy", altered)
    # Neighbors cannot alter central-plane preprocessing; bypass manifest source hash here deliberately.
    next_tensor, _ = load_primary_2d(record, tmp_path)
    assert np.array_equal(tensor.numpy(), next_tensor.numpy())
    assert np.all(normalize_central_slice(np.ones((5,7), np.float32)) == 0)
    with pytest.raises(DataContractError): normalize_central_slice(np.array([[np.nan, 1],[1,1]], np.float32))

def test_canonical_hash_and_freeze_reject_mutations(tmp_path):
    assert canonical_json({"b":"é", "a":1}) == b'{"a":1,"b":"\\u00e9"}'
    assert sha256_json({"a":1,"b":2}) == sha256_json({"b":2,"a":1})
    artifact = tmp_path / "artifact"; written = write_raw_artifact(artifact, np.array([[4,2],[1,4]], dtype=np.int64), {"sample_id":"x"})
    config = tmp_path / "profile.yaml"; config.write_text("frozen-config")
    entry = {"sample_id":"x", "status":"success", "artifact_dir":"artifact", "partition":{"path":"raw_cluster_map.npy","file_hash":written["file_hash"]}, "metadata":{"path":"metadata.json","file_hash":written["metadata_hash"]}}
    freeze = create_freeze(tmp_path, scope="fixture", expected_sample_ids=["x"], entries=[entry], manifest_hash="m", config_hash="c", code_identity="d", environment_id="e", config_path=str(config))
    validate_freeze(freeze, tmp_path)
    with pytest.raises(FreezeError): validate_freeze(freeze, tmp_path, scientific=True)
    config.write_text("mutated-config")
    with pytest.raises(FreezeError): validate_freeze(freeze, tmp_path)
    config.write_text("frozen-config")
    raw = artifact / "raw_cluster_map.npy"; data=bytearray(raw.read_bytes()); data[-1] ^= 1; raw.write_bytes(data)
    with pytest.raises(FreezeError): validate_freeze(freeze, tmp_path)
    with pytest.raises(Exception): create_freeze(tmp_path / "missing", scope="fixture", expected_sample_ids=["x","y"], entries=[entry], manifest_hash="m", config_hash="c", code_identity="d", environment_id="e")

def test_failure_accounting():
    ledger = AttemptLedger(["a","b"]); ledger.terminal("a","failure", error="unreadable"); ledger.terminal("b","success")
    ledger.assert_complete(); assert ledger.current()["a"]["status"] == "failure"
