"""Read-only TRAIN-bank entropy audit; replay the actual epoch batch reductions."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    output = Path(__file__).resolve().parent
    job = output.parents[1]
    bank = job / 'run/bank'
    student = job / 'run/student'
    bm = json.loads((bank / 'FROZEN.json').read_text())
    sm = json.loads((student / 'FROZEN.json').read_text())
    report = json.loads((student / 'REPORT.json').read_text())
    history = json.loads((student / 'training.json').read_text())
    bank_sha = sha(bank / 'FROZEN.json')
    assert bank_sha == sm['bank_manifest_sha256']
    assert sha(student / 'training.json') == sm['files']['training.json']
    assert sha(student / 'REPORT.json') == sm['files']['REPORT.json']
    assert sha(job / 'source/src/self_audit_guidance/student.py') == sm['source_sha256']['student.py']
    assert bm['split'] == 'train' and set(bm['patients']) <= set(bm['locked_splits']['train'])
    allowed = np.asarray([n >= 2 for n in report['class_patients']])
    enabled = bool(allowed[1:].any())
    cache = {}
    for index, pid in enumerate(bm['patients'], 1):
        path = bank / f'{pid}.npz'
        assert sha(path) == bm['files'][path.name], pid
        with np.load(path, allow_pickle=False) as f:
            q = f['q'].astype(np.float32)
            w = f['reliability'].astype(np.float32)
        hard = q.argmax(-1)
        w = (w * allowed[hard]).astype(np.float64)
        q = q.astype(np.float64)
        assert np.isfinite(q).all() and (q >= 0).all()
        assert np.allclose(q.sum(-1), 1, atol=2e-4, rtol=0)
        logq = np.zeros_like(q)
        np.log(q, out=logq, where=q > 0)
        entropy = -(q * logq).sum(-1)
        cache[pid] = {
            'shape': q.shape[:2],
            'mass': w.sum((-2, -1)),
            'entropy_numerator': (entropy * w).sum((-2, -1)),
            'uniform_numerator': (q.sum(-1) * w * np.log(4)).sum((-2, -1)),
            'active': ((hard != 0) & (w > 0)).any((-2, -1)),
        }
        if index % 10 == 0:
            print(f'Verified and summarized {index}/{len(bm["patients"])} TRAIN patients', flush=True)
    cfg = sm['training']
    rng = np.random.default_rng(cfg['seed'])
    rows = []
    for logged in history:
        floor_sum = uniform_sum = 0.
        steps = active_steps = 0
        for pid in rng.permutation(bm['patients']):
            data = cache[str(pid)]
            ids = np.array(list(np.ndindex(data['shape'])))
            rng.shuffle(ids)
            for start in range(0, len(ids), cfg['batch_size']):
                t, z = ids[start:start + cfg['batch_size']].T
                steps += 1
                if enabled and data['active'][t, z].any():
                    active_steps += 1
                    denominator = max(data['mass'][t, z].sum(), 1e-8)
                    floor_sum += data['entropy_numerator'][t, z].sum() / denominator
                    uniform_sum += data['uniform_numerator'][t, z].sum() / denominator
        assert steps == logged['steps']
        floor = floor_sum / steps
        rows.append({
            'epoch': logged['epoch'], 'steps': steps, 'active_steps': active_steps,
            'feature_loss': logged['feature_loss'], 'semantic_ce': logged['semantic_loss'],
            'target_entropy_floor': floor,
            'online_kl_estimate': logged['semantic_loss'] - floor,
            'uniform_prediction_ce': uniform_sum / steps,
        })
    assert sum(r['active_steps'] for r in rows) == report['semantic_updates']
    assert sha(bank / 'FROZEN.json') == bank_sha
    with (output / 'epochs.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {}
    for start, stop in [(1, 10), (31, 40), (41, 50)]:
        segment = rows[start - 1:stop]
        means[f'epochs_{start}_{stop}'] = {
            k: float(np.mean([row[k] for row in segment]))
            for k in ['semantic_ce', 'target_entropy_floor', 'online_kl_estimate', 'feature_loss']
        }
    result = {
        'job': str(job), 'bank_manifest_sha256': bank_sha,
        'training_sha256': sha(student / 'training.json'),
        'source_student_sha256': sm['source_sha256']['student.py'],
        'bank_patients_hash_verified': len(cache), 'training': cfg,
        'optimizer_steps': sum(r['steps'] for r in rows),
        'active_semantic_steps': sum(r['active_steps'] for r in rows),
        'method': 'Replay numpy patient and slice permutations, reliability weights, class mask, foreground batch gate, denominator clamp, and all-step epoch mean. Entropy uses float64; logs use float32 CUDA reductions.',
        'interpretation': 'CE-H is an estimated online training KL averaged across changing within-epoch models, not a fixed-checkpoint validation KL. No GT read; no model, bank or export modified.',
        'first_epoch': rows[0], 'last_epoch': rows[-1], 'means': means,
        'relative_reductions_percent': {
            k: 100 * (1 - rows[-1][k] / rows[0][k])
            for k in ['semantic_ce', 'online_kl_estimate', 'feature_loss']
        },
    }
    (output / 'REPORT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
