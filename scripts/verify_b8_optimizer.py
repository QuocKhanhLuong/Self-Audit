#!/usr/bin/env python3
"""Bounded, discarded-weight supervised diagnostic on declared labeled TRAIN."""
import argparse
import json
from pathlib import Path
import sys
import time


def verify(source, config, data_path, data_sha, out):
    sys.path.insert(0, str(Path(source).resolve()))
    import torch
    import numpy as np
    from self_audit.engine import supervised_step, source_digest
    from self_audit.model import build_model, digest
    from self_audit.data import TrainingData
    c = json.loads(Path(config).read_text())
    torch.set_num_threads(c['threads'])
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    data = TrainingData(data_path, data_sha, synthetic=False)
    rng = np.random.default_rng(c['seed'])
    model, provenance = build_model(c['model'], window=c['window'])
    model = model.cuda()
    opts = [torch.optim.AdamW([p for n, p in model.named_parameters()
                             if p.requires_grad and not n.startswith('head.')],
                            lr=c['body_lr'], weight_decay=c['weight_decay']),
            torch.optim.SGD(model.head.parameters(), lr=c['head_lr'], momentum=0, weight_decay=0)]
    x, y, meta, _ = data.sample_batch(0, rng, c['batch_size'], labeled=True,
                                     crop_size=c['crop_size'], return_valid=True)
    before = {n: v.detach().clone() for n, v in model.named_parameters()}
    start = time.monotonic()
    loss, _ = supervised_step(model, opts, x.cuda(), y.cuda())
    torch.cuda.synchronize()
    scopes = {}
    for scope in ['encoder', 'input_adapter', 'lateral', 'up', 'refine', 'detail', 'fuse', 'head']:
        params = [(n, v) for n, v in model.named_parameters() if n.startswith(scope + '.')]
        scopes[scope] = {
            'grad_norm': float(sum(v.grad.detach().square().sum() for _, v in params if v.grad is not None).sqrt()),
            'weight_change_norm': float(sum((v.detach()-before[n]).square().sum() for n, v in params).sqrt()),
            'optimizer_owns_all': all(any(v is q for o in opts for g in o.param_groups for q in g['params']) for _, v in params)}
        if not (scopes[scope]['grad_norm'] > 0 and scopes[scope]['weight_change_norm'] > 0
                and scopes[scope]['optimizer_owns_all']):
            raise RuntimeError(f'optimizer scope failed: {scope}: {scopes[scope]}')
    receipt = {'status': 'PASS', 'scope': 'one supervised diagnostic; discarded weights; no quality claim',
               'source_sha256': source_digest(), 'config_sha256': digest(config), 'backbone': provenance,
               'scopes': scopes, 'loss': loss, 'seconds': time.monotonic()-start,
               'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved(),
               'labeled_patient': meta, 'label_reads': sorted(data.label_reads),
               'torch': torch.__version__, 'numpy': np.__version__}
    Path(out).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'config', 'data', 'data-sha256', 'out'):
        parser.add_argument('--' + name, required=True)
    a = parser.parse_args()
    verify(a.source, a.config, a.data, a.data_sha256, a.out)
