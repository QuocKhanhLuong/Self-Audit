"""Large neural batch / bounded IPC regression tests. CPU fixtures, not GPU timing."""
from dataclasses import asdict, replace
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.check_maskfree_audit_device import build_synthetic_units
from self_audit_maskfree import auditor
from self_audit_maskfree.candidate_execution import (
    CandidatePoolExecutor, CandidateExecutionError, MAX_PAYLOAD_BYTES_PER_UNIT,
    MAX_RESULT_TENSOR_BYTES_PER_UNIT, generate_candidate_banks,
)
from self_audit_maskfree.config import ConfigError, MaskfreeConfig, RUNTIME_FIELDS
from self_audit_maskfree.export import export_prediction
from self_audit_maskfree.hypotheses import generate_bank
from self_audit_maskfree.trainer import MaskfreeTrainer, ResumeIdentityError


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def candidate_state(bank):
    """Compare every scientific field; measured runtime cannot be equal."""
    values = []
    for hypothesis in bank:
        state = asdict(hypothesis)
        latency = state["metadata"].pop("generation_latency")
        assert set(latency) == {"bank_wall_seconds"}
        assert math.isfinite(latency["bank_wall_seconds"]) and latency["bank_wall_seconds"] >= 0
        values.append(state)
    return values


@pytest.fixture(scope='module')
def batch32():
    units = build_synthetic_units(size=224, seed=42, count=32)
    rng = torch.Generator().manual_seed(918)
    return [u.fitting for u in units], [torch.randn(16, 224, 224, generator=rng) for _ in units], list(range(400, 432))


@pytest.mark.parametrize('batch,workers', [(8, 2), (16, 4), (32, 8), (32, 2)])
def test_chunked_real_candidates_exact(batch32, batch, workers):
    views, features, seeds = [x[:batch] for x in batch32]
    reference = [generate_bank(v, features=f, seed=s) for v, f, s in zip(views, features, seeds)]
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with CandidatePoolExecutor(workers=workers, chunk_size=8) as pool:
        actual = pool.generate_banks(views, features, seeds)
        stats = pool.stats()
        assert stats['units'] == batch and stats['chunks'] == (batch + 7) // 8
        assert stats['max_inflight_units'] <= 8
        assert stats['max_input_tensor_bytes'] <= 8 * MAX_PAYLOAD_BYTES_PER_UNIT
        assert stats['max_result_window_tensor_bytes'] <= 8 * MAX_RESULT_TENSOR_BYTES_PER_UNIT
    for ref, got in zip(reference, actual):
        exact(candidate_state(ref), candidate_state(got))
    exact(before, (random.getstate(), np.random.get_state(), torch.get_rng_state()))


def test_window_order_default_seeds_and_short_tail(batch32, monkeypatch):
    views, features, _ = [x[:17] for x in batch32]
    reference = generate_candidate_banks(views, features, workers=0, chunk_size=8)
    windows = []

    class ReversedPool:
        def map(self, fn, payloads, chunksize):
            items = list(payloads)
            windows.append((len(items), [p[-1] for p in items]))
            return iter(reversed([fn(p) for p in items]))
        def shutdown(self, **kwargs):
            pass

    with CandidatePoolExecutor(workers=2, chunk_size=3) as pool:
        monkeypatch.setattr(pool, '_get_pool', lambda: ReversedPool())
        actual = pool.generate_banks(views, features)
    assert [n for n, _ in windows] == [3, 3, 3, 3, 3, 2]
    assert [s for _, seeds in windows for s in seeds] == list(range(42, 59))
    for ref, got in zip(reference, actual):
        exact(candidate_state(ref), candidate_state(got))


@pytest.mark.parametrize('chunk', [0, -1, 9, True, '8'])
def test_invalid_chunk_config(chunk):
    with pytest.raises(CandidateExecutionError):
        CandidatePoolExecutor(workers=8, chunk_size=chunk)
    with pytest.raises(ConfigError):
        MaskfreeConfig(dataset='acdc', data_root='x', output_dir='y', candidate_chunk_size=chunk)


def test_invalid_late_chunk_and_length_fail_closed(batch32):
    views, features, seeds = batch32
    with CandidatePoolExecutor(workers=0) as pool:
        with pytest.raises(CandidateExecutionError, match='Seeds count'):
            pool.generate_banks(views, features, seeds[:8])
        assert pool.stats()['chunks'] == 0
    with CandidatePoolExecutor(workers=0) as pool:
        with pytest.raises(CandidateExecutionError, match='not a FittingView'):
            pool.generate_banks(views[:8] + ['bad'], features[:9], seeds[:9])
        with pytest.raises(CandidateExecutionError, match='closed'):
            pool.generate_banks([])


def test_large_batch_config_and_profile_cli(tmp_path):
    from scripts import profile_maskfree as p
    from scripts import benchmark_maskfree_rental as rental
    cfg = MaskfreeConfig(dataset='acdc', data_root=str(tmp_path), output_dir=str(tmp_path / 'out'),
                         batch_size=32, image_size=224, amp=False, device='cpu', allow_cpu=True,
                         candidate_workers=8, candidate_chunk_size=8)
    assert 'candidate_chunk_size' in RUNTIME_FIELDS
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg.to_dict()))
    args = p.build_parser().parse_args(['--config', str(path), '--image-size', '224', '--batch-size', '32'])
    resolved, info = p._prepare_config(args, tmp_path / 'profile')
    assert resolved.batch_size == 32 and resolved.candidate_workers == 8
    assert info['scientific_requirements']['batch_size'] == 32
    # A B8 comparator is not silently repurposed into B32 by the config.
    args.batch_size = 8
    with pytest.raises(ValueError, match='mismatches'):
        p._prepare_config(args, tmp_path / 'bad')
    args = rental.parser().parse_args(['--output', str(tmp_path / 'gate'), '--batch-size', '32',
                                     '--candidate-workers', '8', '--candidate-chunk-size', '4', '--print-plan'])
    configs, commands = rental.configs_and_commands(args)
    assert all(c.batch_size == 32 and c.candidate_workers == 8 and c.candidate_chunk_size == 4 for c in configs.values())
    command = commands[-1]
    assert command[command.index('--batch-size') + 1] == '32'


def large_components(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import test_maskfree_trainer as fixture
    def discover(root, dataset, **kwargs):
        manifest = fixture.discover_dataset(root, dataset, **kwargs)
        template = manifest['records'][0]
        manifest['records'] = [{**template, 'unit_id': f'unit-{i:03d}', 'study_id': f'study-{i:03d}',
                                'patient_id': f'patient-{i:03d}', 'path': f'{root}/{i}.npy', 'split': 'train'}
                               for i in range(96)]
        return manifest
    components = replace(fixture.build_components(), discover_dataset=discover,
                         audit_banks=auditor.audit_banks, export_prediction=export_prediction)
    # Explicit fixture boundary, not a claim about CI/rental CPU quota.
    monkeypatch.setattr('self_audit_maskfree.resources.resource_snapshot',
                        lambda **kwargs: {'cgroup': {'visible_cpu_upper_bound_cores': 8}})
    return fixture, components


@pytest.mark.parametrize('batch', [16, 32])
def test_large_preflight_and_gate_identity(tmp_path, monkeypatch, batch):
    fixture, components = large_components(monkeypatch)
    config = fixture.build_config(tmp_path, total_epochs=150, max_epochs=None, max_steps=None,
                                 batch_size=batch, candidate_workers=2, candidate_chunk_size=8)
    probe = MaskfreeTrainer(config, components=components)
    receipt = probe.preflight()
    assert receipt['status'] == 'pass', receipt
    assert receipt['actual_physical_batch'] == batch
    assert receipt['audited_units'] == batch
    assert receipt['runtime_options']['candidate_chunk_size'] == 8
    runner = MaskfreeTrainer(config, components=components)
    try:
        runner.setup()
        original = runner.paths.gate_receipt.read_bytes()
        runner._validate_preflight_gate()
        assert runner.paths.gate_receipt.read_bytes() == original
    finally:
        runner._close_candidate_executor()


def test_b16_serial_parallel_checkpoint_and_resume(tmp_path, monkeypatch):
    fixture, components = large_components(monkeypatch)
    def run(root, workers, steps, resume=None, chunk=8):
        cfg = fixture.build_config(root, total_epochs=150, max_epochs=None, max_steps=steps,
                                   batch_size=16, candidate_workers=workers, candidate_chunk_size=chunk,
                                   epoch_validation=False, resume=resume)
        trainer = MaskfreeTrainer(cfg, components=components)
        trainer.run()
        return trainer, torch.load(trainer.paths.last_checkpoint, weights_only=False)
    serial, ref = run(tmp_path / 'serial', 0, 2)
    fast, got = run(tmp_path / 'fast', 2, 2)
    partial, _ = run(tmp_path / 'resume', 2, 1)
    resumed, resumed_state = run(tmp_path / 'resume', 2, 2, str(partial.paths.last_checkpoint))
    for key in ('models', 'optimizers', 'rng', 'sampling_generator', 'component_steps'):
        exact(ref[key], got[key])
        exact(got[key], resumed_state[key])
    changed = resumed.config.replace(resume=str(resumed.paths.last_checkpoint), candidate_chunk_size=4)
    with pytest.raises(ResumeIdentityError, match='runtime_options'):
        MaskfreeTrainer(changed, components=components).setup()
