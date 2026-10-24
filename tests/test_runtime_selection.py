from copy import deepcopy

from scripts.optimize_training_runtime import choose_runtime


def row(batch=24, threads=8, cached=False, seconds=1., digest='same'):
    return dict(arm='A2', batch=batch, threads=threads, cached_inputs=cached, status='PASS',
                seconds_per_step=seconds, state=dict(student_sha256=digest, teacher_sha256=digest,
                    optimizers_sha256=digest, final_rng={'state':42}))


def test_wall_time_selection_preserves_exposure_and_requires_cache_parity():
    original = row()
    cached = row(cached=True, seconds=.8)
    mismatch = deepcopy(cached); mismatch['state']['optimizers_sha256']='different'
    # A faster, larger batch still changes exposure at 10k steps and is excluded.
    large = row(batch=48, seconds=.1)
    chosen = choose_runtime([original, mismatch, large])
    assert chosen['batch']==24 and not chosen['cached_inputs']
    assert not chosen['cache_state_bitwise_parity']
    faster_threads = row(threads=4, cached=True, seconds=.7)
    chosen = choose_runtime([original, cached, large, faster_threads])
    assert chosen['batch']==24 and chosen['threads']==4 and chosen['cached_inputs']


def test_noisy_small_gain_does_not_change_live_suite():
    chosen=choose_runtime([row(),row(cached=True,seconds=.99)])
    assert chosen['threads']==8 and not chosen['cached_inputs']


def test_prefetch_cannot_inherit_cache_parity():
    prefetch=row(cached=True,seconds=.5,digest='wrong')
    prefetch['prefetch_inputs']=True
    chosen=choose_runtime([row(),row(cached=True,seconds=.8),prefetch])
    assert chosen['cached_inputs'] and not chosen['prefetch_inputs']
    assert not chosen['prefetch_state_bitwise_parity']
    prefetch['state']=row()['state']
    chosen=choose_runtime([row(),row(cached=True,seconds=.8),prefetch])
    assert chosen['prefetch_inputs'] and chosen['prefetch_state_bitwise_parity']
