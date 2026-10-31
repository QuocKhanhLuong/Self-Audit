#!/usr/bin/env python3
"""Bounded manufactured CUDA/CPU execution smoke, never cardiac quality evidence."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def smoke(out, device):
    import torch
    from self_audit.engine import DEFAULT, train, load_checkpoint, source_digest
    from self_audit.data import synthetic_fixture
    from self_audit.model import digest
    from self_audit.losses import FOCAL_DICE_DEFAULT
    from self_audit.lr_schedule import DEFAULT_PLATEAU

    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest, data_sha = synthetic_fixture(root / 'data')
    config = copy.deepcopy(DEFAULT)
    config.update(method='ema_full', steps=8, warmup_steps=2, save_every=2,
                  pseudo_threshold=.25, body_lr=.001, head_lr=.03)
    config['model']['backbone'] = 'tiny_test'
    config['supervised_loss'] = copy.deepcopy(FOCAL_DICE_DEFAULT)
    config['lr_scheduler'] = copy.deepcopy(DEFAULT_PLATEAU)
    # Accelerated scheduler events are specific to this manufactured smoke.
    config['lr_scheduler'].update(interval_steps=2, patience=0, threshold=1., threshold_mode='abs')
    (root / 'CONFIG.json').write_text(json.dumps(config, indent=2)+'\n')
    output = root / 'trajectory'
    train(config, manifest, data_sha, output, device=device, synthetic=True, stop_after_step=5)
    checkpoint = output / 'step_000005'
    partial = torch.load(checkpoint / 'weights.pt', map_location='cpu', weights_only=True)
    assert partial['lr_scheduler']['metric_count'] == 1
    assert partial['lr_scheduler']['num_updates'] == 1
    report = train(config, manifest, data_sha, output, device=device, synthetic=True,
                   resume=str(checkpoint), resume_sha256=digest(checkpoint / 'MANIFEST.json'))
    final = output / 'final'
    payload = torch.load(final / 'weights.pt', map_location='cpu', weights_only=True)
    assert report['status'] == 'COMPLETED' and report['ema_updates'] == 8
    assert report['supervised_updates'] == 8 and not report['label_reads_outside_budget']
    assert payload['lr_scheduler']['num_updates'] == 3
    assert payload['lr_scheduler']['metric_count'] == 0
    assert payload['optimizers'][0]['param_groups'][0]['lr'] < config['body_lr']
    assert payload['optimizers'][1]['param_groups'][0]['lr'] < config['head_lr']
    model, metadata = load_checkpoint(final, digest(final / 'MANIFEST.json'), device)
    with torch.inference_mode():
        logits = model(torch.zeros((1, 3, 32, 32), device=device))
    assert logits.shape == (1, 4, 32, 32) and torch.isfinite(logits).all()
    receipt = {'status': 'PASS', 'scope': 'manufactured execution, checkpoint/resume and LR decay; no quality or bitwise CUDA parity claim',
               'source_sha256': source_digest(), 'config_sha256': digest(root / 'CONFIG.json'),
               'manifest_sha256': data_sha, 'payload_sha256': metadata['payload_sha256'],
               'checkpoint_manifest_sha256': digest(final / 'MANIFEST.json'), 'device': device,
               'report': report, 'lr_scheduler': {
                   'num_updates': payload['lr_scheduler']['num_updates'],
                   'metric_count': payload['lr_scheduler']['metric_count'],
                   'body_lr': payload['optimizers'][0]['param_groups'][0]['lr'],
                   'head_lr': payload['optimizers'][1]['param_groups'][0]['lr']},
               'restored_partial_window_count': partial['lr_scheduler']['metric_count'],
               'restored_cuda_rng_states': len(partial['cuda_rng']),
               'peak_allocated': torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
               'peak_reserved': torch.cuda.max_memory_reserved() if device.startswith('cuda') else None}
    (root / 'RESUME_SMOKE.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    smoke(args.out, args.device)
