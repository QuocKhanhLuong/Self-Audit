#!/usr/bin/env python3
"""Cost-only canonical train.py probe; checkpoint saves replaced by state hashes."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def state_digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous()
            h.update(str((str(a.dtype), tuple(a.shape))).encode())
            h.update(a.numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(repr(k).encode()); visit(x[k])
        elif isinstance(x, (list, tuple)):
            for item in x: visit(item)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def main():
    args = sys.argv[1:]
    prefetch = '--prefetch-inputs' in args
    if prefetch: args.remove('--prefetch-inputs')
    cached = '--cached-inputs' in args
    if cached: args.remove('--cached-inputs')
    out = Path(args[args.index('--out')+1])
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total-int(2.5*2**30))/total)
    from self_audit import engine
    close = lambda: None
    if prefetch:
        from scripts.prefetch_runtime import install as install_prefetch
        config = json.loads(Path(args[args.index('--config')+1]).read_text())
        close = install_prefetch(config['steps'])
    elif cached:
        from scripts.runtime_performance import install
        install()
    captured = {}
    def no_checkpoint(folder, model, teacher, optimizers, config, data, step, provenance, rng):
        captured.update(student_sha256=state_digest(model.state_dict()),
                        teacher_sha256=state_digest(teacher.state_dict()),
                        optimizers_sha256=state_digest([o.state_dict() for o in optimizers]),
                        final_rng=rng.bit_generator.state)
        return captured['student_sha256']
    engine.save_checkpoint = no_checkpoint
    original = engine.time.monotonic
    def synchronized():
        torch.cuda.synchronize()
        return original()
    engine.time.monotonic = synchronized
    try:
        sys.argv = [str(ROOT/'train.py'), *args]
        runpy.run_path(str(ROOT/'train.py'), run_name='__main__')
    finally:
        close()
        if out.is_dir():
            captured.update(cost_only=True, checkpoints_saved=False, cached_inputs=cached or prefetch,
                            prefetch_inputs=prefetch,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved())
            (out/'RUNTIME_PROBE.json').write_text(json.dumps(captured, indent=2)+'\n')


if __name__ == '__main__': main()
