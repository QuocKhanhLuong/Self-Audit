#!/usr/bin/env python3
"""Real-image memory diagnostic, not training or an accuracy experiment.

Repeats one native slice from a budgeted TRAIN patient to measure batch memory.
No optimizer/EMA step; optional decoder JVP uses the native supervised gradient.
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import torch
from torch.func import functional_call, jvp
from self_audit.data import TrainingData, label_subset
from self_audit.engine import source_digest
from self_audit.losses import supervised_loss
from self_audit.model import build_model, digest


def profile(a):
    ids = label_subset(json.loads(a.split.read_text())['train_patients'], 8, 42)
    if a.patient not in ids:
        raise ValueError('profile patient must be in declared L8 seed42 pool')
    folder = a.root / a.patient
    cine_path = next(folder.glob(f'{a.patient}_4d.nii*'))
    mask_path = sorted(folder.glob(f'{a.patient}_frame*_gt.nii*'))[0]
    frame_path = Path(str(mask_path).replace('_gt.nii', '.nii'))
    cine = np.asarray(nib.load(cine_path).dataobj)
    mask_img, frame_img = nib.load(mask_path), nib.load(frame_path)
    mask = np.asarray(mask_img.dataobj)
    t = int(mask_path.name.split('_frame')[1].split('_gt')[0]) - 1
    if not np.array_equal(cine[..., t], np.asarray(frame_img.dataobj)) or mask.shape != frame_img.shape or not np.allclose(mask_img.affine, frame_img.affine):
        raise ValueError('selected TRAIN label geometry mismatch')
    lo, hi = np.percentile(cine, [.5, 99.5])
    native = np.clip((cine-lo)/max(float(hi-lo), 1e-6), 0, 1).transpose(3, 2, 1, 0).astype(np.float32)
    z = native.shape[1] // 2
    x = TrainingData.context(native, t, z).repeat(a.batch, 1, 1, 1).cuda()
    y = torch.from_numpy(mask.transpose(2, 1, 0)[z].astype(np.int64).copy())[None].repeat(a.batch, 1, 1).cuda()
    torch.manual_seed(42); torch.set_num_threads(2)
    model, provenance = build_model(json.loads(a.backbone.read_text()))
    model = model.cuda().eval()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.monotonic()
    result = {'status': 'STARTED', 'diagnostic_only': True, 'optimizer_steps': 0, 'ema_updates': 0,
              'patient': a.patient, 'declared_L8_S42_ids': ids, 'mask_read_patients': [a.patient],
              'batch_size': a.batch, 'input_shape': list(x.shape), 'protocol': 'repeated_same_real_slice_memory_probe',
              'backbone': provenance, 'training_source_sha256': source_digest(), 'cine_sha256': digest(cine_path),
              'mask_sha256': digest(mask_path), 'decoder_jvp_requested': a.decoder_jvp}
    try:
        loss = supervised_loss(model(x), y); loss.backward()
        result.update(loss=float(loss.detach()), encoder_gradient_norm=float(model.encoder.stages[0][0].dwconv.weight.grad.norm()),
                      detail_gradient_norm=float(model.detail[0].weight.grad.norm()))
        torch.cuda.synchronize()
        result['supervised_seconds'] = time.monotonic()-start
        result['supervised_peak_bytes'] = torch.cuda.max_memory_allocated()
        if a.decoder_jvp:
            names, values, directions = [], [], []
            for name, p in model.named_parameters():
                if not name.startswith(('encoder.', 'input_adapter.')):
                    names.append(name); values.append(p.detach()); directions.append(-.001*p.grad.detach())
            model.zero_grad(set_to_none=True)
            def predict(v):
                return functional_call(model, dict(zip(names, v)), (x,), strict=False).softmax(1)
            with torch.no_grad():
                p, dp = jvp(predict, (tuple(values),), (tuple(directions),))
            torch.cuda.synchronize()
            result.update(decoder_jvp_finite=bool(torch.isfinite(dp).all()), decoder_jvp_norm=float(dp.norm()))
        result['status'] = 'PASS'
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status='OOM', error=str(exc))
    finally:
        result['elapsed_seconds'] = time.monotonic()-start
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(result, indent=2))
    return result['status'] == 'PASS'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--split', type=Path, required=True)
    p.add_argument('--backbone', type=Path, required=True); p.add_argument('--patient', default='patient007')
    p.add_argument('--batch', type=int, required=True); p.add_argument('--decoder-jvp', action='store_true')
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    if a.batch < 1 or a.out.exists(): p.error('positive batch and new output path required')
    raise SystemExit(0 if profile(a) else 2)
