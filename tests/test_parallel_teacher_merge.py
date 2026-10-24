"""Parallel prefix-merge equivalence on manufactured candidate pools."""
import numpy as np
import torch

from self_audit.memory_bank import pack_bank,patient_balanced_select,PAIRS,_rank_candidate
from self_audit.teacher_memory import MemoryBank
from scripts.build_teacher_memory_parallel import merge_selected_banks


def make_bank(pools):
    payload=pack_bank(patient_balanced_select(pools),{'source':'manufactured merge test'})
    return MemoryBank(payload['endpoints'],payload['confidences'],payload['valid'],payload['metadata'])


def test_shard_prefix_merge_is_bitwise_equal_to_global_selection():
    # Cover sparse groups, few patients requiring repeat turns, and >8 patients.
    for seed in range(12):
        rng=np.random.default_rng(seed);pools={}
        for p in range(1+seed):
            pid=f'patient{p:03d}';pools[pid]={}
            for pair in PAIRS:
                candidates=[]
                for i in range(int(rng.integers(0,13))):
                    vec=rng.normal(size=(2,32)).astype(np.float32)
                    vec/=np.linalg.norm(vec,axis=1,keepdims=True)
                    item=dict(endpoints=vec,confidence=[.9,.95],patient=pid,timepoint=i,class_pair=list(pair),
                        coord_c_xyz=[i,0,0],coord_d_xyz=[i,1,0],world_c_mm=[i,0,0],world_d_mm=[i,1,0],
                        distance_mm=1.,image_sha256=str(p)*64,source_sha256='a'*64,feature_source='manufactured')
                    item['_rank']=_rank_candidate(item);candidates.append(item)
                pools[pid][pair]=candidates
        expected=make_bank(pools);ids=list(pools);parts=[ids[i::3] for i in range(min(3,len(ids)))]
        shards=[make_bank({pid:pools[pid] for pid in part}) for part in parts]
        merged=merge_selected_banks(shards,{'source':'manufactured merge test'})
        assert torch.equal(merged.endpoints,expected.endpoints)
        assert torch.equal(merged.confidences,expected.confidences)
        assert torch.equal(merged.valid,expected.valid)
        assert merged.metadata['entry_metadata']==expected.metadata['entry_metadata']
