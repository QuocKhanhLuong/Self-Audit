#!/usr/bin/env python3
"""Freeze native uint8 argmax volumes before scoring; equivalent Dice storage.

No probabilities are quantized before argmax. The original checkpoint and
reference hashes remain required. No threshold, crop or postprocessing exists.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from self_audit.data import TrainingData, write_json
from self_audit.engine import load_checkpoint
from self_audit.evaluation import confusion_matrix, class_metrics
from self_audit.model import digest


def evaluate_labels(checkpoint, checkpoint_sha, data_manifest, data_sha, out,
                    device='cuda', batch_size=8, threads=2, compress_predictions=False):
    manifest, out = Path(data_manifest).resolve(), Path(out).resolve()
    if digest(manifest) != data_sha:
        raise ValueError('evaluation manifest hash mismatch')
    m = json.loads(manifest.read_text())
    if m.get('schema') != 'self_audit.data.v2' or m.get('role') not in ('development', 'external_frozen'):
        raise ValueError('separate frozen evaluation manifest required')
    if batch_size < 1 or threads < 1:
        raise ValueError('positive inference batch and thread count required')
    torch.set_num_threads(threads)
    model, meta = load_checkpoint(checkpoint, checkpoint_sha, device)
    ids = [p['id'] for p in m['patients']]
    if len(ids) != len(set(ids)) or set(ids) & set(meta['labeled_ids'] + meta['unlabeled_ids']):
        raise ValueError('evaluation patient overlap with TRAIN')
    if bool(m.get('synthetic', False)) != meta['synthetic']:
        raise ValueError('synthetic/patient mismatch')
    if out.exists() or out.is_relative_to(manifest.parent) or manifest.parent.is_relative_to(out):
        raise ValueError('new output outside evaluation data required')
    resolve = lambda p: Path(p) if Path(p).is_absolute() else manifest.parent / p
    out.mkdir(parents=True); pred_dir = out / 'predictions'; pred_dir.mkdir()
    recipe = dict(checkpoint_sha256=checkpoint_sha, data_sha256=data_sha,
        scope=m.get('scope', m['role']), prediction_model='student', synthetic=meta['synthetic'],
        input='full native cines; no crop/resize', probability_dtype='float32; no AMP',
        output='softmax(logits).argmax; uint8 native TZYX; no threshold/postprocessing',
        prediction_storage=('gzip lossless uint8 labels' if compress_predictions else 'uint8 labels') + '; no probability calibration analyses',
        batch_size=batch_size, threads=threads, metric_classes=['BG','RV','MYO','LV'])
    write_json(out / 'RECIPE.json', recipe)
    inventory = []; start = time.monotonic()
    with torch.inference_mode():
        for record in m['patients']:
            ip = resolve(record['image'])
            if digest(ip) != record['image_sha256']:
                raise ValueError('evaluation image hash mismatch')
            x = np.load(ip, mmap_mode='r', allow_pickle=False)
            if x.ndim != 4 or not np.isfinite(x).all():
                raise ValueError('invalid native image')
            op = pred_dir / (record['id'] + ('.npy.gz' if compress_predictions else '.npy'))
            labels = (np.empty(x.shape,dtype='uint8') if compress_predictions else
                      np.lib.format.open_memmap(op, mode='w+', dtype='uint8', shape=x.shape))
            indices = [(t,z) for t in range(len(x)) for z in range(x.shape[1])]
            for offset in range(0, len(indices), batch_size):
                chunk = indices[offset:offset+batch_size]
                inputs = torch.cat([TrainingData.context(x,t,z) for t,z in chunk]).to(device)
                logits = model(inputs)
                if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                    raise FloatingPointError('nonfinite or non-FP32 evaluation logits')
                predicted = logits.softmax(1).argmax(1).cpu().numpy().astype(np.uint8)
                for (t,z), plane in zip(chunk,predicted):labels[t,z] = plane
            if compress_predictions:
                with op.open('wb') as raw_file:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=raw_file,compresslevel=1,mtime=0) as stream:
                        np.save(stream,labels,allow_pickle=False)
            else:labels.flush()
            del labels
            inventory.append(dict(id=record['id'], file=op.name, sha256=digest(op)))
            print(json.dumps(dict(patient=record['id'], frozen=len(inventory), total=len(ids))), flush=True)
    # No reference array was opened during this process before this receipt.
    write_json(out / 'PREDICTIONS_FROZEN.json', dict(recipe_sha256=digest(out/'RECIPE.json'), files=inventory))
    frames = []; total = np.zeros((4,5), dtype=np.int64)
    for record, frozen in zip(m['patients'], inventory):
        op = pred_dir / frozen['file']
        if digest(op) != frozen['sha256']:
            raise ValueError('frozen prediction changed')
        if op.name.endswith('.gz'):
            with gzip.open(op,'rb') as stream:pred=np.load(stream,allow_pickle=False)
        else:pred = np.load(op, mmap_mode='r', allow_pickle=False)
        for frame, mask in record['masks'].items():
            mp = resolve(mask['path'])
            if digest(mp) != mask['sha256']:
                raise ValueError('reference hash mismatch')
            y = np.load(mp, allow_pickle=False)
            cm = confusion_matrix(y, np.asarray(pred[int(frame)])); total += cm
            frames.append(dict(patient=record['id'], frame=int(frame), raw=class_metrics(cm), raw_confusion=cm.tolist()))
    patients = {}
    for pid in ids:
        rows = [f for f in frames if f['patient'] == pid]; patients[pid] = {}
        for j,name in enumerate(['BG','RV','MYO','LV']):
            values = [f['raw'][j]['dice'] for f in rows if f['raw'][j]['dice'] is not None]
            patients[pid][name] = float(np.mean(values)) if values else None
    macro = {}
    for name in ['BG','RV','MYO','LV']:
        values = [p[name] for p in patients.values() if p[name] is not None]
        macro[name] = dict(dice=float(np.mean(values)) if values else None, present_patients=len(values))
    def aggregate(classes):
        values = [[p[k] for k in classes if p[k] is not None] for p in patients.values()]
        values = [float(np.mean(v)) for v in values if v]
        return float(np.mean(values)) if values else None
    result = dict(scope=recipe['scope'], synthetic=meta['synthetic'], macro_patient=macro,
        patients=patients, frames=frames, pooled_confusion=total.tolist(),
        foreground_patient_macro=aggregate(['RV','MYO','LV']),
        all_class_patient_macro=aggregate(['BG','RV','MYO','LV']),
        elapsed_seconds=time.monotonic()-start,
        physical_boundary_metrics='NOT_COMPUTED; no spacing/HD95 certification',
        warning='VAL18 and previously reported Drive TEST50 are exposed; this is not independent confirmation')
    write_json(out / 'METRICS.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','checkpoint-sha256','data','data-sha256','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--compress-predictions',action='store_true');a=p.parse_args()
    result=evaluate_labels(a.checkpoint,a.checkpoint_sha256,a.data,a.data_sha256,a.out,a.device,a.batch_size,a.threads,a.compress_predictions)
    print(json.dumps(result['macro_patient'],indent=2))
