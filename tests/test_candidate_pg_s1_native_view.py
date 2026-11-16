from types import SimpleNamespace

import torch

from self_audit.candidate_pg.engine import _LazyTrainImageSequence
from self_audit.candidate_pg.cinema_init import prepare_cinema_sax


def test_s1_explicit_source_crop_keeps_native_student_and_uses_no_u_target():
    image = torch.arange(20 * 210 * 220, dtype=torch.float32).reshape(1, 1, 20, 210, 220)
    image /= image.max()
    geometry = {'shape': [20, 210, 220], 'spacing_zyx': [10, 1, 1],
                'affine_zyx': [[10, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                'context': 'full_volume'}
    class Source:
        def load(self, key, *, labeled):
            assert labeled is False
            return SimpleNamespace(image=image, geometry=geometry, patient_id=key[0], frame_index=key[1])
    item = _LazyTrainImageSequence(Source(), [], [('u-patient', 7)])[0]
    assert item['image'] is image
    assert 'cinema_crop' not in geometry
    assert item['geometry']['cinema_crop']['start_zyx'] == [2, 9, 14]
    prepared = prepare_cinema_sax(item['image'], item['geometry'], synthetic=False)
    assert prepared.model_input_hwd.shape == (1, 1, 192, 192, 16)
    assert prepared.valid_hwd.bool().all()
    assert item['geometry']['shape'] == geometry['shape']
    assert 'target' not in item
