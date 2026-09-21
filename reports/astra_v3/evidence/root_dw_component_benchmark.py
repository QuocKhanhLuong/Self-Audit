"""Instrumented CPU attribution, separate from ordinary deployment timings."""
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from self_audit_pseudolabel.system_v3 import AdaptiveAnnotationStudent


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(1729)
    model=AdaptiveAnnotationStudent(width=32,window_k=8).eval()
    x=torch.randn(1,3,224,224)
    reference=json.loads(Path('reports/astra_v3/evidence/root_benchmark.json').read_text())
    assert state_hash(model)==reference['model_state_sha256']
    result={'experiment_id':'astra-v3-dw-cpu-attribution-224-v1','model_state_sha256':state_hash(model),
            'input_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            'mode':'CPU FP32 eval+inference_mode batch1 224x224;4intra/1inter threads',
            'platform':platform.platform(),'profiles':{},
            'limitations':['instrumented;hook overhead included','DW timings include its generator/QKV/sampling/attention/output',
                           'nested DW interval is inside refiner;do not add twice','no CUDA attribution','shared host not4core8GBcap']}
    current={};starts={};handles=[]
    def pre(name):
        def f(module,args):starts[name]=time.perf_counter()
        return f
    def post(name):
        def f(module,args,output):current[name].append(1000*(time.perf_counter()-starts[name]))
        return f
    for name,module in [('encoder',model.encoder),('refiner',model.refiner),('dw',model.refiner.refinement_block)]:
        handles += [module.register_forward_pre_hook(pre(name)),module.register_forward_hook(post(name))]
    with torch.inference_mode():
        for profile in ['compact','balanced','accurate']:
            samples=[]
            for repeat in range(50):
                current={n:[] for n in ['encoder','refiner','dw']}
                start=time.perf_counter();model(x,profile=profile);total=1000*(time.perf_counter()-start)
                row={n+'_ms':sum(v) for n,v in current.items()}
                row.update({'total_ms':total,'refiner_without_dw_ms':row['refiner_ms']-row['dw_ms'],
                            'other_ms':total-row['encoder_ms']-row['refiner_ms'],
                            'counts':{n:len(v) for n,v in current.items()}})
                assert row['counts']=={'encoder':1,'refiner':{'compact':0,'balanced':1,'accurate':2}[profile],
                                       'dw':{'compact':0,'balanced':1,'accurate':3}[profile]}
                if repeat>=10:samples.append(row)
            result['profiles'][profile]={'warmups':10,'repeats':40,'samples':samples,
                'quantiles':{key:dict(zip(['p50','p90','p95'],map(float,np.percentile([s[key] for s in samples],[50,90,95]))))
                             for key in ['total_ms','encoder_ms','refiner_ms','dw_ms','refiner_without_dw_ms','other_ms']}}
    for handle in handles:handle.remove()
    out=Path('reports/astra_v3/evidence/root_dw_component_benchmark.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({p:x['quantiles'] for p,x in result['profiles'].items()},indent=2))


if __name__=='__main__':main()
