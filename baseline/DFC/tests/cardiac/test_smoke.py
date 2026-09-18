import numpy as np
from cardiac_benchmark.dfc_runner import DFCConfig, run_dfc
from cardiac_benchmark.output import write_raw_artifact
from cardiac_benchmark.freeze import create_freeze, validate_freeze
from cardiac_benchmark.provenance import derive_sample_seed, sha256_json

def test_bounded_non_scientific_end_to_end_smoke(tmp_path):
    image=np.arange(35,dtype=np.float32).reshape(1,1,5,7)/35
    result=run_dfc(__import__('torch').from_numpy(image), DFCConfig(profile_id="TEST-ONLY-bounded",maxIter=1,minLabels=0), derive_sample_seed("fixture:smoke")["sample_seed"])
    written=write_raw_artifact(tmp_path/"sample",result.raw_cluster_map,{"scope":"fixture","sample_id":"fixture:smoke","optimization":{"updates":result.update_count,"forwards":result.forward_count}})
    entry={"sample_id":"fixture:smoke","status":"success","artifact_dir":"sample","partition":{"path":"raw_cluster_map.npy","file_hash":written["file_hash"]}}
    frozen=create_freeze(tmp_path,scope="fixture",expected_sample_ids=["fixture:smoke"],entries=[entry],manifest_hash="fixture",config_hash=sha256_json({"test_only":True}),code_identity="local",environment_id="local")
    validate_freeze(frozen,tmp_path); assert result.raw_cluster_map.dtype == np.dtype("<i4")
