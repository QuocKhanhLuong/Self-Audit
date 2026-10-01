"""Synthetic CPU checks only: no training, diffusion backend, or real artifacts."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from utils.artifact_contract import validate_hierarchy
from utils.output_saver import OutputSaver


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, SRC / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_from_source(relative, name):
    # Load pure functions without importing optional diffusion/metric backends.
    tree = ast.parse((SRC / relative).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), scope)
    return scope[name]


def test_guided_relabel_224_does_not_overflow():
    relabel = function_from_source('utils/metrics.py', 'guided_relabel')
    truth = np.zeros((224, 224), dtype=np.int64)
    truth[200:] = 1
    predicted = np.where(truth == 0, 40000, 70000)
    np.testing.assert_array_equal(relabel(predicted, truth), truth)


def test_persistent_large_ids_match_safe_exporter():
    persistent = function_from_source('utils/diffusion_condensation.py', 'get_persistent_structures')
    exporter = load_file('cuts_persistent_test', 'scripts_analysis/export_diffusion_persistent.py')
    levels = np.full((3, 4, 4), 40000, dtype=np.int64)
    levels[-1] = 90000  # ID only in the excluded final frame.
    np.testing.assert_array_equal(persistent(levels), np.full((4, 4), 40000))
    np.testing.assert_array_equal(persistent(levels), exporter._upstream_persistent_structures_safe(levels))


def test_image_only_loader_never_probes_gt(tmp_path, monkeypatch):
    loader = load_file('cuts_acdc_test', 'datasets/acdc_original_style.py')
    patient = tmp_path / 'patient001'
    patient.mkdir()
    (patient / 'Info.cfg').write_text('ED: 1\nES: 1\n')
    nib.save(nib.Nifti1Image(np.arange(32., dtype=float).reshape(4, 4, 2), np.eye(4)),
             patient / 'patient001_frame01.nii')
    original_load, original_is_file = nib.load, Path.is_file

    def guarded_load(path, *args, **kwargs):
        assert '_gt' not in str(path)
        return original_load(path, *args, **kwargs)

    def guarded_is_file(path):
        assert '_gt' not in str(path)
        return original_is_file(path)

    monkeypatch.setattr(nib, 'load', guarded_load)
    monkeypatch.setattr(Path, 'is_file', guarded_is_file)
    dataset = loader.ACDCOriginalStyle(str(tmp_path), image_only=True)
    image, dummy = dataset[0]
    assert image.shape == (1, 224, 224)
    assert image.dtype == np.float32 and image.min() >= -1 and image.max() <= 1
    assert not dummy.any()
    saver = OutputSaver(str(tmp_path / 'export'), image_only=True,
                        sample_metadata=dataset.sample_metadata,
                        provenance={'checkpoint_sha256': 'fixture'})
    saver.save(torch.from_numpy(image[None]), torch.from_numpy(image[None]), None,
               torch.zeros((1, 2, 224, 224)))
    with np.load(tmp_path / 'export/numpy_files/sample_00000.npz') as payload:
        assert 'label' not in payload
        assert json.loads(str(payload['sample_metadata']))['patient_id'] == 'patient001'
        assert json.loads(str(payload['provenance']))['checkpoint_sha256'] == 'fixture'
    with pytest.raises(FileExistsError):
        OutputSaver(str(tmp_path / 'export'))


def test_multiclass_diffusion_without_pixel_artifacts():
    tree = ast.parse((SRC / 'scripts_analysis/run_metrics.py').read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not)
                 and isinstance(n.test.operand, ast.Attribute)
                 and n.test.operand.attr == 'is_binary')
    class Config:
        is_binary = False
    scope = dict(hparams=Config(), has_diffusion=True, has_pixel_diffusion=False,
                 hashmap={'labels_diffusion': np.zeros((2, 3, 3)),
                          'label_true': np.zeros((3, 3))}, entity_tuples=[],
                 guided_relabel=lambda label_pred, label_true: label_pred)
    exec(compile(ast.Module(body=[block], type_ignores=[]), 'metrics_branch', 'exec'), scope)


@pytest.mark.parametrize('levels,granularities', [(None, None),
    (np.zeros((2, 4), dtype=float), [1, 2]),
    (np.zeros((2, 4), dtype=int), [1]),
    (np.full((2, 4), -1), [1, 2])])
def test_reject_invalid_hierarchy(levels, granularities):
    with pytest.raises(ValueError):
        validate_hierarchy(levels, granularities, 4)


def test_persistent_export_preserves_identity_without_reading_gt(tmp_path):
    exporter = load_file('cuts_export_test', 'scripts_analysis/export_diffusion_persistent.py')
    source, target = tmp_path / 'input', tmp_path / 'output'
    source.mkdir()
    np.savez(source / 'sample_00000.npz', image=np.zeros((2, 2)),
             labels_diffusion=np.ones((3, 4), dtype=np.int64),
             granularities_diffusion=np.array([1, 2, 3]),
             label=np.array([object()], dtype=object),  # Access would fail allow_pickle=False.
             sample_metadata=np.asarray(json.dumps({'patient_id': 'patient001'})))
    records = exporter.export(source, target)
    assert records[0]['sample_metadata']['patient_id'] == 'patient001'
    assert np.load(target / 'sample_00000.npy').dtype == np.int64
    with pytest.raises(FileExistsError):
        exporter.export(source, target)
    with pytest.raises(ValueError, match='provenance'):
        exporter.export(source, tmp_path / 'strict_output', require_image_only=True)


def test_original_loader_still_returns_gt(tmp_path):
    loader = load_file('cuts_acdc_original_test', 'datasets/acdc_original_style.py')
    patient = tmp_path / 'patient001'
    patient.mkdir()
    (patient / 'Info.cfg').write_text('ED: 1\nES: 1\n')
    image = np.arange(32., dtype=float).reshape(4, 4, 2)
    label = np.ones((4, 4, 2), dtype=np.int16) * 3
    nib.save(nib.Nifti1Image(image, np.eye(4)), patient / 'patient001_frame01.nii')
    nib.save(nib.Nifti1Image(label, np.eye(4)), patient / 'patient001_frame01_gt.nii')
    dataset = loader.ACDCOriginalStyle(str(tmp_path))
    assert np.all(dataset[0][1] == 3)
    assert dataset.sample_metadata(0)['image_only'] is False


def test_prepare_dataset_passes_image_only_and_canvas(tmp_path):
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, Dataset
    tree = ast.parse((SRC / 'data_utils/prepare_dataset.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'prepare_dataset')
    calls = []

    class FakeDataset(Dataset):
        def __init__(self, **kw):
            calls.append(kw)
        def num_image_channel(self):
            return 1
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return index

    scope = dict(AttributeHashmap=SimpleNamespace, ACDCOriginalStyle=FakeDataset,
                 DataLoader=DataLoader)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'prepare_dataset', 'exec'), scope)
    config = SimpleNamespace(dataset_name='acdc_original_style', dataset_path=str(tmp_path),
                             image_size=[224, 224], image_only=True, batch_size=1, num_workers=0)
    loader, channels = scope['prepare_dataset'](config, mode='test')
    assert calls == [dict(base_path=str(tmp_path), out_shape=(224, 224), image_only=True)]
    assert len(loader.dataset) == 2 and channels == 1


@pytest.mark.parametrize('export_after_train,expected', [(True, ['train', 'test']),
                                                        (False, ['train'])])
def test_entrypoint_dispatch_without_executing_training(export_after_train, expected):
    from types import SimpleNamespace
    tree = ast.parse((SRC / 'main.py').read_text())
    dispatch = tree.body[-1].body[-1]
    calls = []
    scope = dict(args=SimpleNamespace(mode='train'),
                 config=SimpleNamespace(export_after_train=export_after_train),
                 train=lambda **kw: calls.append('train'),
                 test=lambda **kw: calls.append('test'))
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), 'entrypoint_dispatch', 'exec'), scope)
    assert calls == expected
