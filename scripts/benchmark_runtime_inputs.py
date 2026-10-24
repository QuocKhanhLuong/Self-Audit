#!/usr/bin/env python3
"""CPU-only real TRAIN input parity and cost; never an accuracy measurement."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.runtime_performance import CachedTrainingData
from scripts.prefetch_runtime import PrefetchTrainingData
from self_audit.data import TrainingData
from self_audit.losses import aligned_flip, strong_intensity


def benchmark(manifest, sha, out, steps=32, batch=24, threads=1):
    torch.set_num_threads(threads)
    rows = []
    for cls in (TrainingData, CachedTrainingData, PrefetchTrainingData):
        data = cls(manifest, sha, **({'max_steps':steps} if cls is PrefetchTrainingData else {}))
        rng = np.random.default_rng(42)
        h = hashlib.sha256()
        times = []
        augmentation_seconds = 0.
        for step in range(steps):
            li, ui = step % len(data.lids), step % len(data.uids)
            ai = (li + 1 + int(rng.integers(len(data.lids)-1))) % len(data.lids)
            start = time.monotonic()
            results = [data.sample_batch(index, rng, batch, labeled=labeled, crop_size=256, return_valid=True)
                       for index, labeled in ((li, True), (ai, True), (ui, False))]
            times.append(time.monotonic()-start)
            for result in results:
                for value in result:
                    h.update(value.numpy().tobytes() if torch.is_tensor(value)
                             else json.dumps(value, sort_keys=True).encode())
            start = time.monotonic()
            flip = data.flip if isinstance(data, PrefetchTrainingData) else aligned_flip
            intensity = data.strong if isinstance(data, PrefetchTrainingData) else strong_intensity
            flip(results[0][0], results[0][1], rng, results[0][-1])
            strong = intensity(results[2][0], rng)
            augmentation_seconds += time.monotonic()-start
            h.update(strong.numpy().tobytes())
        rows.append(dict(implementation=cls.__name__, steps=steps,
                         measured_steps=steps-8, seconds_per_transaction=sum(times[8:])/(steps-8),
                         cpu_augmentation_seconds=augmentation_seconds,
                         input_sha256=h.hexdigest(), rng_final=rng.bit_generator.state,
                         label_reads=sorted(data.label_reads)))
        if isinstance(data, PrefetchTrainingData): data.close()
    assert all(r['input_sha256']==rows[0]['input_sha256'] for r in rows), 'real input tensor mismatch'
    assert all(r['rng_final']==rows[0]['rng_final'] for r in rows), 'RNG mismatch'
    assert all(r['label_reads']==rows[0]['label_reads'] for r in rows), 'label access mismatch'
    result = dict(status='PASS', measured_on='real TRAIN L8, CPU only; not GPU/accuracy evidence',
                  batch=batch, crop=256, threads=threads, rows=rows,
                  sampling_speedup=rows[0]['seconds_per_transaction']/rows[1]['seconds_per_transaction'],
                  manifest_sha256=sha)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    benchmark(args.manifest, args.sha256, args.out)
