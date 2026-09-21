"""Random-weight deployment diagnostics; no accuracy or target-hardware claim."""
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.nn import functional as F
from self_audit_pseudolabel.system_v3 import AdaptiveAnnotationStudent, CinePseudoTeacher


def quantiles(values):
    return dict(zip(['p50_ms', 'p90_ms', 'p95_ms'], map(float, np.percentile(values, [50, 90, 95]))))


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1729)
    model = AdaptiveAnnotationStudent(width=32, window_k=8).eval()
    x = torch.randn(1, 3, 224, 224)
    if len(sys.argv) > 1 and sys.argv[1] == 'cold-child':
        start = time.perf_counter()
        with torch.inference_mode():
            model(x, profile=sys.argv[2])
        print(json.dumps({'first_forward_ms': 1000*(time.perf_counter()-start),
                          'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}))
        return
    result = {'experiment_id': 'astra-v3-runtime-randomweights-224-v1', 'seed': 1729,
              'model_state_sha256': state_hash(model), 'mode': 'eval+inference_mode, FP32, batch1',
              'system': {'platform': platform.platform(), 'python': platform.python_version(),
                         'torch': torch.__version__, 'physical_cores': psutil.cpu_count(logical=False),
                         'ram_gib': psutil.virtual_memory().total/2**30, 'threads': 4, 'interop_threads': 1,
                         'cpu_affinity_enforced': False, 'memory_cap_enforced': False,
                         'cuda_available': torch.cuda.is_available(), 'mps_available': torch.backends.mps.is_available()},
              'params': {'student': sum(p.numel() for p in model.parameters()),
                         'teacher': sum(p.numel() for p in CinePseudoTeacher().parameters()),
                         'encoder': sum(p.numel() for p in model.encoder.parameters()),
                         'a0': sum(p.numel() for p in model.a0_head.parameters()),
                         'refiner': sum(p.numel() for p in model.refiner.parameters())},
              'profiles': {}, 'limitations': ['random weights', 'shared macOS host', '4 threads are not 4 pinned cores',
                    'not 8GB memory-capped', 'no trained checkpoint load', 'UI and clinical geometry export NOT RUN']}
    with torch.inference_mode():
        # Calls and MAC counters are outside timing. Only Conv2d MACs are counted.
        for profile in ('compact', 'balanced', 'accurate'):
            counts = {'encoder': 0, 'refiner': 0, 'dw': 0, 'conv2d_macs': 0}
            handles = []
            for name, module in [('encoder', model.encoder), ('refiner', model.refiner), ('dw', model.refiner.refinement_block)]:
                def count(m, a, o, name=name): counts[name] += 1
                handles.append(module.register_forward_hook(count))
            def mac(m, args, out):
                counts['conv2d_macs'] += out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
            handles.extend(m.register_forward_hook(mac) for m in model.modules() if isinstance(m, torch.nn.Conv2d))
            model(x, profile=profile)
            for h in handles: h.remove()
            for _ in range(10): model(x, profile=profile)
            elapsed = []
            for _ in range(40):
                start=time.perf_counter(); model(x, profile=profile); elapsed.append(1000*(time.perf_counter()-start))
            result['profiles'][profile] = {'calls_and_conv_macs': counts, 'warm_forward': quantiles(elapsed),
                                          'warm_samples_ms': elapsed, 'warmups': 10,
                                          'peak_process_rss_mib_cumulative': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        # Real image-only phase volume from the frozen image mirror. No GT access.
        import nibabel as nib
        source = Path('/private/tmp/astra_v3_r0_dimensionless_20260920_v2/images/patient093_frame01.nii')
        def volume_pass(profile):
            t0=time.perf_counter(); nii=nib.load(source); volume=np.asarray(nii.dataobj,dtype=np.float32)
            lo,hi=np.percentile(volume,[1,99]); volume=np.clip((volume-lo)/max(hi-lo,1e-6),0,1).astype(np.float32)
            t1=time.perf_counter(); masks=[]
            for z in range(volume.shape[2]):
                a=np.stack([volume[:,:,max(z-1,0)],volume[:,:,z],volume[:,:,min(z+1,volume.shape[2]-1)]])
                h,w=a.shape[-2:]; scale=224/max(h,w); sh,sw=round(h*scale),round(w*scale)
                resized=F.interpolate(torch.from_numpy(a)[None],(sh,sw),mode='bilinear',align_corners=False)
                top=(224-sh)//2; left=(224-sw)//2
                net=F.pad(resized,(left,224-sw-left,top,224-sh-top))
                logits=model(net,profile=profile)['final_logits'][:,:,top:top+sh,left:left+sw]
                native=F.interpolate(logits,(h,w),mode='bilinear',align_corners=False).argmax(1)[0].to(torch.uint8).numpy()
                masks.append(native)
            np.stack(masks,axis=2)
            t2=time.perf_counter()
            return {'total_ms':1000*(t2-t0),'read_normalize_ms':1000*(t1-t0),'slice_pipeline_ms':1000*(t2-t1)}
        result['stored_grid_volume']={'source_image':str(source),'image_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                                     'shape':[180,224,10], 'geometry':'array grid only; units/affine unverified', 'profiles':{}}
        for profile in result['profiles']:
            volume_pass(profile)
            repeats=[volume_pass(profile) for _ in range(5)]
            result['stored_grid_volume']['profiles'][profile]={'samples':repeats,**quantiles([r['total_ms'] for r in repeats])}
        if torch.backends.mps.is_available():
            model.to('mps'); xm=x.to('mps')
            for profile in result['profiles']:
                for _ in range(10): model(xm,profile=profile)
                torch.mps.synchronize(); elapsed=[]
                for _ in range(30):
                    torch.mps.synchronize(); t=time.perf_counter(); model(xm,profile=profile); torch.mps.synchronize()
                    elapsed.append(1000*(time.perf_counter()-t))
                result['profiles'][profile]['mps_warm_forward']={**quantiles(elapsed),'samples_ms':elapsed,
                    'live_allocated_bytes':torch.mps.current_allocated_memory(), 'peak_memory':'NOT MEASURED'}
            model.cpu()
    # Fresh process startup includes interpreter/import/model initialization/one forward, not checkpoint load.
    for profile in result['profiles']:
        t=time.perf_counter()
        p=subprocess.run([sys.executable,__file__,'cold-child',profile],capture_output=True,text=True,check=True)
        result['profiles'][profile]['fresh_process']={**json.loads(p.stdout),'process_wall_ms':1000*(time.perf_counter()-t)}
    Path('reports/astra_v3/evidence/root_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({p:d['warm_forward'] for p,d in result['profiles'].items()},indent=2))


if __name__ == '__main__': main()
