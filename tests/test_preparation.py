import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from prepare_acdc import prepare, read_patient_split


def test_existing_split_variants_agree(tmp_path):
    p=tmp_path/'split.json';p.write_text(json.dumps({'train_patients':['patient001','patient002','patient003'],
          'val_patients':['patient004'],'splits':{'train':{'patients':['patient003','patient001','patient002']}}}))
    assert read_patient_split(p)['test']==[]
    x=json.loads(p.read_text());x['splits']['train']['patients']=['patient008'];p.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='conflicting'):read_patient_split(p)


def test_preparation_opens_only_budgeted_masks_with_mock_reader(tmp_path,monkeypatch):
    root=tmp_path/'raw';root.mkdir();registry={};reads=[]
    class Image:
        def __init__(self,a):
            self.dataobj=a;self.affine=np.eye(4);self.shape=a.shape
            self.header=types.SimpleNamespace(get_xyzt_units=lambda: ('mm', None))
    rng=np.random.default_rng(2)
    for i in range(1,5):
        pid=f'patient{i:03d}';d=root/pid;d.mkdir();a=rng.integers(0,100,(5,6,2,2),dtype=np.int16)
        for name,value in [(f'{pid}_4d.nii',a),(f'{pid}_frame01.nii',a[...,0]),
                           (f'{pid}_frame01_gt.nii',rng.integers(0,4,(5,6,2),dtype=np.int16))]:
            p=d/name;p.write_bytes(value.tobytes());registry[str(p)]=Image(value)
    def load(path):reads.append(str(path));return registry[str(path)]
    monkeypatch.setitem(sys.modules,'nibabel',types.SimpleNamespace(load=load))
    split=tmp_path/'split.json';split.write_text(json.dumps({'train':['patient001','patient002','patient003'],
                                                          'val':['patient004'],'test':[]}))
    out=tmp_path/'prepared';prepare(root,split,out,2,42,False)
    m=json.loads((out/'train.json').read_text());lids={r['id'] for r in m['labeled']}
    mask_ids={Path(p).parent.name for p in reads if '_gt' in p}
    assert mask_ids==lids and len(lids)==2
    assert not any('patient004' in p for p in reads)
    assert all(set(r)=={'id','image','image_sha256'} for r in m['unlabeled'])


def _bridge_fixture(tmp_path, *, wrong_frame=False, wrong_mask_affine=False,
                    matching_frame_affine=False):
    root = tmp_path / 'raw'; root.mkdir()
    registry = {}

    class Header:
        def __init__(self, units, zooms):
            self._units, self._zooms = units, tuple(zooms)
            self.binaryblock = f'{units}:{self._zooms}'.encode()

        def get_xyzt_units(self):
            return self._units, None

        def get_zooms(self):
            return self._zooms

    class Image:
        def __init__(self, array, affine, units, zooms):
            self.dataobj, self.affine, self.shape = array, affine, array.shape
            self.header = Header(units, zooms)

    rng = np.random.default_rng(8)
    for i in range(1, 4):
        pid = f'patient{i:03d}'; folder = root / pid; folder.mkdir()
        cine = rng.integers(0, 100, (4, 5, 2, 2), dtype=np.int16)
        cine_affine = np.diag([2., 3., 4., 1.])
        cine_image = Image(cine, cine_affine, 'mm', (2., 3., 4., 1.))
        frame_data = cine[..., 0].copy()
        if wrong_frame and i == 1:
            frame_data[0, 0, 0] += 1
        frame_affine = cine_affine.copy() if matching_frame_affine else np.eye(4)
        mask_data = rng.integers(0, 4, frame_data.shape, dtype=np.int16)
        mask_affine = frame_affine.copy()
        if wrong_mask_affine and i == 1:
            mask_affine[0, 3] = 1.
        entries = [
            (f'{pid}_4d.nii', cine_image),
            (f'{pid}_frame01.nii', Image(frame_data, frame_affine, 'unknown', (2., 3., 4.))),
            (f'{pid}_frame01_gt.nii', Image(mask_data, mask_affine, 'unknown', (2., 3., 4.))),
        ]
        for name, image in entries:
            path = folder / name; path.write_bytes(np.asarray(image.dataobj).tobytes()); registry[str(path)] = image
    return root, registry


def _bridge_split(tmp_path):
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'train': ['patient001', 'patient002', 'patient003'],
                                 'val': [], 'test': []}))
    return split


def test_verified_cine_bridge_accepts_exact_voxel_bridge_and_records_headers(tmp_path, monkeypatch):
    root, registry = _bridge_fixture(tmp_path)
    monkeypatch.setitem(sys.modules, 'nibabel', types.SimpleNamespace(load=lambda path: registry[str(path)]))
    out = tmp_path / 'prepared'
    prepare(root, _bridge_split(tmp_path), out, 2, 42,
            spatial_unit_override='mm', frame_geometry_policy='verified-cine-bridge')
    preparation = json.loads((out / 'PREPARATION.json').read_text())
    assert preparation['frame_geometry_policy'] == 'verified-cine-bridge'
    assert preparation['geometry_policy_summary']['geometry_source'] == 'cine_affine'
    assert preparation['geometry_policy_summary']['accepted_header_affine_mismatches'] == 2
    assert all(item['exact_frame_cine_voxel_bridge'] for item in preparation['geometry_audits'])
    assert all(item['accepted_header_affine_mismatch'] for item in preparation['geometry_audits'])


def test_verified_bridge_can_inherit_known_cine_unit_after_exact_bridge(tmp_path, monkeypatch):
    root, registry = _bridge_fixture(tmp_path)
    monkeypatch.setitem(sys.modules, 'nibabel', types.SimpleNamespace(load=lambda path: registry[str(path)]))
    out = tmp_path / 'inherited-unit'
    prepare(root, _bridge_split(tmp_path), out, 2, 42,
            frame_geometry_policy='verified-cine-bridge')
    audits = json.loads((out / 'PREPARATION.json').read_text())['geometry_audits']
    assert {item['unit_resolution']['frame'] for item in audits} == {
        'inherited_from_cine_after_exact_bridge',
    }
    assert {item['resolved_units']['frame'] for item in audits} == {'mm'}


def test_strict_policy_rejects_header_mismatch_even_with_exact_voxels(tmp_path, monkeypatch):
    root, registry = _bridge_fixture(tmp_path)
    monkeypatch.setitem(sys.modules, 'nibabel', types.SimpleNamespace(load=lambda path: registry[str(path)]))
    with pytest.raises(ValueError, match='frame/cine geometry mismatch'):
        prepare(root, _bridge_split(tmp_path), tmp_path / 'strict', 2, 42,
                 spatial_unit_override='mm')


def test_verified_bridge_rejects_wrong_frame_voxels(tmp_path, monkeypatch):
    root, registry = _bridge_fixture(tmp_path, wrong_frame=True)
    monkeypatch.setitem(sys.modules, 'nibabel', types.SimpleNamespace(load=lambda path: registry[str(path)]))
    with pytest.raises(ValueError, match='exact voxel bridge'):
        prepare(root, _bridge_split(tmp_path), tmp_path / 'wrong-frame', 2, 42,
                 spatial_unit_override='mm', frame_geometry_policy='verified-cine-bridge')


def test_strict_policy_rejects_wrong_frame_voxels_with_matching_affine(tmp_path, monkeypatch):
    root, registry = _bridge_fixture(tmp_path, wrong_frame=True, matching_frame_affine=True)
    monkeypatch.setitem(sys.modules, 'nibabel', types.SimpleNamespace(load=lambda path: registry[str(path)]))
    with pytest.raises(ValueError, match='exact voxel bridge'):
        prepare(root, _bridge_split(tmp_path), tmp_path / 'strict-wrong-frame', 2, 42,
                 spatial_unit_override='mm', frame_geometry_policy='strict')


def test_verified_bridge_rejects_frame_mask_affine_mismatch(tmp_path, monkeypatch):
    root, registry = _bridge_fixture(tmp_path, wrong_mask_affine=True)
    monkeypatch.setitem(sys.modules, 'nibabel', types.SimpleNamespace(load=lambda path: registry[str(path)]))
    with pytest.raises(ValueError, match='reference/frame geometry mismatch'):
        prepare(root, _bridge_split(tmp_path), tmp_path / 'wrong-mask-affine', 2, 42,
                 spatial_unit_override='mm', frame_geometry_policy='verified-cine-bridge')
