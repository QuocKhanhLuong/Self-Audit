"""One-step CPU lookahead with explicit canonical RNG-state handoff.

The worker owns a cloned RNG and a separate label-read set. It never evaluates
a model, touches CUDA, changes the live RNG early, or accesses non-TRAIN masks.
Serving each role advances the live RNG to exactly its canonical boundary.
"""
from concurrent.futures import ThreadPoolExecutor
import copy

import numpy as np
import torch

from scripts.runtime_performance import CachedTrainingData
from self_audit.losses import aligned_flip, strong_intensity


class PrefetchTrainingData(CachedTrainingData):
    def __init__(self, *args, max_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='train-input')
        self.max_steps = max_steps
        self.future = self.active = None
        self.step = self.role = 0
        self.spec = None
        # Immutable payload caches can be shared; consuming labels is tracked
        # only by the training thread, after it requests each prepared role.
        self.worker_data = copy.copy(self)
        self.worker_data.label_reads = set()

    def _prepare(self, step, state):
        rng = np.random.default_rng(0)
        rng.bit_generator.state = copy.deepcopy(state)
        li, ui = step % len(self.lids), step % len(self.uids)
        ai = (li+1+int(rng.integers(len(self.lids)-1))) % len(self.lids)
        after_ai = copy.deepcopy(rng.bit_generator.state)
        batch, crop = self.spec
        roles, states = [], []
        for index, labeled in ((li, True), (ai, True), (ui, False)):
            roles.append(CachedTrainingData.sample_batch(self.worker_data, index, rng, batch,
                         labeled=labeled, crop_size=crop, return_valid=True))
            states.append(copy.deepcopy(rng.bit_generator.state))
        flipped = aligned_flip(roles[0][0], roles[0][1], rng, roles[0][-1])
        after_flip = copy.deepcopy(rng.bit_generator.state)
        gamma = float(rng.uniform(.8, 1.2))
        noise = torch.from_numpy(rng.normal(0, .03, tuple(roles[2][0].shape)))
        return dict(step=step, indices=(li, ai, ui), roles=roles, states=states,
                    after_ai=after_ai, flipped=flipped, after_flip=after_flip,
                    gamma=gamma, noise=noise, final_rng=copy.deepcopy(rng.bit_generator.state))

    def sample_batch(self, index, rng, batch_size, *, labeled, crop_size=0, return_valid=False):
        assert return_valid, 'prefetch supports the canonical validity-aware training transaction'
        if self.spec is None: self.spec = (batch_size, crop_size)
        assert self.spec == (batch_size, crop_size)
        if self.future is not None:
            self.active = self.future.result()
            self.future = None
            assert self.active['step'] == self.step
            assert rng.bit_generator.state == self.active['after_ai'], 'prefetch RNG mismatch before supervised role'
        if self.active is None:
            result = super().sample_batch(index, rng, batch_size, labeled=labeled,
                                         crop_size=crop_size, return_valid=True)
        else:
            assert self.role < 3 and self.active['indices'][self.role] == index
            assert labeled == (self.role < 2)
            if self.role:
                assert rng.bit_generator.state == self.active['states'][self.role-1]
            result = self.active['roles'][self.role]
            rng.bit_generator.state = copy.deepcopy(self.active['states'][self.role])
            if labeled: self.label_reads.add(result[2]['patient'])
        self.role += 1
        return result

    def flip(self, x, y, rng, valid_mask=None):
        assert self.role == 3
        if self.active is None: return aligned_flip(x, y, rng, valid_mask)
        assert x is self.active['roles'][0][0] and y is self.active['roles'][0][1]
        assert valid_mask is self.active['roles'][0][-1]
        assert rng.bit_generator.state == self.active['states'][2]
        rng.bit_generator.state = copy.deepcopy(self.active['after_flip'])
        return self.active['flipped']

    def strong(self, x, rng):
        assert self.role == 3
        if self.active is None:
            result = strong_intensity(x, rng)
        else:
            assert rng.bit_generator.state == self.active['after_flip']
            noise = self.active['noise'].to(device=x.device, dtype=x.dtype)
            result = (x.pow(self.active['gamma']) + noise).clamp(0, 1)
            rng.bit_generator.state = copy.deepcopy(self.active['final_rng'])
        self.active = None
        self.step += 1
        self.role = 0
        if self.max_steps is None or self.step < self.max_steps:
            self.future = self.executor.submit(self._prepare, self.step, copy.deepcopy(rng.bit_generator.state))
        return result

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def install(max_steps):
    from self_audit import engine
    holder = {}
    class InstalledData(PrefetchTrainingData):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, max_steps=max_steps, **kwargs)
            holder['data'] = self
    engine.TrainingData = InstalledData
    engine.aligned_flip = lambda *args, **kwargs: holder['data'].flip(*args, **kwargs)
    engine.strong_intensity = lambda *args, **kwargs: holder['data'].strong(*args, **kwargs)
    def close():
        if 'data' in holder: holder['data'].close()
    return close
