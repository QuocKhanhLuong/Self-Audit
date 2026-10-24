import json
import numpy as np
import pytest
import torch

from scripts.runtime_performance import CachedTrainingData
from self_audit.data import TrainingData, synthetic_fixture
from self_audit.model import digest
from self_audit.losses import aligned_flip, strong_intensity
from scripts.prefetch_runtime import PrefetchTrainingData


def test_cached_sampling_preserves_every_tensor_metadata_and_rng(tmp_path):
    manifest, sha = synthetic_fixture(tmp_path / "data", size=36)
    original, cached = TrainingData(manifest, sha, synthetic=True), CachedTrainingData(manifest, sha, synthetic=True)
    generators = [np.random.default_rng(42), np.random.default_rng(42)]
    for step in range(20):
        for labeled in (True, False, True):
            results = [data.sample_batch(step % 3, rng, 4, labeled=labeled, crop_size=40 if step % 2 else 24,
                                         return_valid=True)
                       for data, rng in zip((original, cached), generators)]
            for a, b in zip(*results):
                if torch.is_tensor(a):
                    assert torch.equal(a, b)
                else:
                    assert a == b
            assert generators[0].bit_generator.state == generators[1].bit_generator.state
    assert original.label_reads == cached.label_reads
    assert len(cached.foreground_cache) == 3


def test_cache_still_rejects_invalid_train_mask(tmp_path):
    manifest, _ = synthetic_fixture(tmp_path / "data")
    m = json.loads(manifest.read_text())
    rec = m["labeled"][0]["masks"]["0"]
    mask = manifest.parent / rec["path"]
    a = np.load(mask)
    a[0, 0, 0] = 7
    np.save(mask, a)
    rec["sha256"] = digest(mask)
    manifest.write_text(json.dumps(m))
    data = CachedTrainingData(manifest, digest(manifest), synthetic=True)
    with pytest.raises(ValueError, match="ontology"):
        data.sample_labeled(0, np.random.default_rng(42))


def test_prefetch_preserves_rng_at_every_boundary_and_never_advances_live_label_reads(tmp_path):
    manifest, sha = synthetic_fixture(tmp_path / 'data', size=36)
    original = TrainingData(manifest, sha, synthetic=True)
    prefetched = PrefetchTrainingData(manifest, sha, synthetic=True, max_steps=12)
    generators = [np.random.default_rng(42), np.random.default_rng(42)]
    def same(a,b):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            if torch.is_tensor(x): assert torch.equal(x,y)
            else: assert x==y
        assert generators[0].bit_generator.state == generators[1].bit_generator.state
    try:
        for step in range(12):
            li,ui=step%3,step%3
            audit=[(li+1+int(r.integers(2)))%3 for r in generators]
            assert audit[0]==audit[1]
            roles=[]
            for index,labeled in ((li,True),(audit[0],True),(ui,False)):
                results=[d.sample_batch(index,r,4,labeled=labeled,crop_size=40,return_valid=True)
                         for d,r in zip((original,prefetched),generators)]
                same(*results);roles.append(results)
            flip0=aligned_flip(roles[0][0][0],roles[0][0][1],generators[0],roles[0][0][-1])
            flip1=prefetched.flip(roles[0][1][0],roles[0][1][1],generators[1],roles[0][1][-1])
            same(flip0,flip1)
            a=strong_intensity(roles[2][0][0],generators[0])
            b=prefetched.strong(roles[2][1][0],generators[1])
            same((a,),(b,))
            if prefetched.future: prefetched.future.result()
            assert prefetched.label_reads == original.label_reads
    finally: prefetched.close()
