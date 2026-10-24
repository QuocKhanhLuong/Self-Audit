"""Opt-in input cache for the canonical trainer, with separate runtime provenance.

Only immutable, hash-checked budgeted TRAIN masks are cached. No sampling RNG,
crop, augmentation, model operation, precision or optimizer is changed.
"""
from pathlib import Path
import hashlib
import json
import runpy
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from self_audit.data import TrainingData


class CachedTrainingData(TrainingData):
    """Validate each labeled mask and find its foreground slices once per run."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.foreground_cache = {}

    def sample_labeled(self, index, rng, *, foreground_probability=.5):
        r = self.labeled[index]
        cine = self.array(r["image"], r["image_sha256"])
        frame = sorted(r["masks"], key=int)[int(rng.integers(len(r["masks"])))]
        rec = r["masks"][frame]
        mask = self.array(rec["path"], rec["sha256"])
        key = (str(self.resolve(rec["path"])), rec["sha256"], tuple(cine.shape[1:]))
        if key not in self.foreground_cache:
            if mask.shape != cine.shape[1:] or not np.isin(mask, [0, 1, 2, 3, 255]).all():
                raise ValueError("mask shape/ontology mismatch")
            self.foreground_cache[key] = np.flatnonzero(
                ((mask > 0) & (mask < 4)).reshape(len(mask), -1).any(1))
        self.label_reads.add(r["id"])
        foreground = self.foreground_cache[key]
        z = (int(rng.choice(foreground)) if len(foreground) and rng.random() < foreground_probability
             else int(rng.integers(len(mask))))
        x = self.context(cine, int(frame), z)
        y = torch.from_numpy(np.array(mask[z], dtype=np.int64, copy=True))[None]
        return x, y, {"patient": r["id"], "frame": int(frame), "slice": z}


def install():
    from self_audit import engine
    engine.TrainingData = CachedTrainingData


def main():
    from self_audit import engine
    args = sys.argv[1:]
    prefetch = '--prefetch-inputs' in args
    if prefetch: args.remove('--prefetch-inputs')
    out = Path(args[args.index("--out")+1]) if "--out" in args else None
    close = lambda: None
    if prefetch:
        from scripts.prefetch_runtime import install as install_prefetch
        config = json.loads(Path(args[args.index('--config')+1]).read_text())
        close = install_prefetch(config['steps'])
    else: install()
    receipt = dict(schema="self_audit.runtime_performance.v1",
                   canonical_source_sha256=engine.source_digest(),
                   launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   prefetch_inputs=prefetch,
                   prefetch_sha256=(hashlib.sha256((ROOT/'scripts/prefetch_runtime.py').read_bytes()).hexdigest()
                                    if prefetch else None),
                   cache="budgeted immutable TRAIN mask ontology and foreground slices",
                   numerical_operations_changed=False, canonical_cli=args)
    try:
        sys.argv = [str(ROOT / "train.py"), *args]
        runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
    finally:
        close()
        if out is not None and out.is_dir():
            (out / "RUNTIME_PERFORMANCE.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
