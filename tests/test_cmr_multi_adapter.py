from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

try:
    from self_audit.data.cmr_multi import (
        CMR_MULTI_RAW_TO_COMMON,
        CMRMultiAdapter,
        infer_zt,
        reconstruct_zt,
    )
except ImportError:
    from src.self_audit.data.cmr_multi import (
        CMR_MULTI_RAW_TO_COMMON,
        CMRMultiAdapter,
        infer_zt,
        reconstruct_zt,
    )


pytestmark = pytest.mark.skipif(nib is None, reason="nibabel is required")


def _make_flattened_case(root: Path, *, case_id: str = "CINE_SAX_001") -> tuple[Path, Path]:
    image_dir = root / "CINE_MULTI" / "SAX_TR" / "image"
    mask_dir = root / "CINE_MULTI" / "SAX_TR" / "anno"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    z_count, t_count = 6, 20
    mask_zt = np.zeros((32, 32, z_count, t_count), dtype=np.int16)
    for z in range(z_count):
        x0 = 2 + (z % 3) * 9
        y0 = 2 + (z // 3) * 12
        mask_zt[x0 : x0 + 3, y0 : y0 + 3, z, :] = 1
    flat_mask = mask_zt.reshape(32, 32, z_count * t_count, order="C")
    image = flat_mask.astype(np.float32) + 0.1
    image_path = image_dir / f"{case_id}.nii.gz"
    mask_path = mask_dir / f"{case_id}.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), str(image_path))
    nib.save(nib.Nifti1Image(flat_mask, np.eye(4)), str(mask_path))
    return image_path, mask_path


def test_infer_zt_accepts_clear_time_fastest_layout() -> None:
    mask = np.zeros((20, 20, 120), dtype=np.int16)
    for z in range(6):
        x0 = 1 + (z % 3) * 6
        y0 = 1 + (z // 3) * 8
        mask[x0 : x0 + 3, y0 : y0 + 3, z * 20 : (z + 1) * 20] = 1
    inference = infer_zt(mask)
    assert inference.accepted is True
    assert (inference.z, inference.t) == (6, 20)
    assert inference.temporal_dice >= 0.9
    assert inference.wrap_dice >= 0.8
    reconstructed = reconstruct_zt(mask, inference)
    assert reconstructed.shape == (20, 20, 6, 20)
    np.testing.assert_array_equal(reconstructed[:, :, 2, 7], mask[:, :, 47])


def test_infer_zt_marks_ambiguous_candidate_for_manual_review() -> None:
    inference = infer_zt(np.zeros((8, 8, 360), dtype=np.int16))
    assert inference.accepted is False
    assert inference.status in {"manual_review", "exclude"}
    with pytest.raises(ValueError, match="not accepted"):
        reconstruct_zt(np.zeros((8, 8, 360), dtype=np.int16), inference, require_accepted=True)


def test_cmr_multi_adapter_discovers_sax_time_units_and_maps_labels(tmp_path: Path) -> None:
    _make_flattened_case(tmp_path)
    adapter = CMRMultiAdapter(tmp_path)
    units = adapter.discover_units()
    assert len(units) == 20
    assert units[0].subject_id == "CINE_SAX_001"
    assert units[0].time_index == 0
    loaded = adapter.load_unit(units[0])
    assert loaded.image_zhw.shape == (6, 32, 32)
    assert loaded.mask_zhw.shape == loaded.image_zhw.shape
    mapped = adapter.remap_labels(np.asarray([[0, 1, 2, 3]], dtype=np.int16))
    np.testing.assert_array_equal(mapped, [[0, 2, 3, 1]])
    assert CMR_MULTI_RAW_TO_COMMON == {0: 0, 1: 2, 2: 3, 3: 1}


def test_cmr_multi_pairing_is_strict(tmp_path: Path) -> None:
    image_path, _ = _make_flattened_case(tmp_path)
    (image_path.parent / "orphan.nii.gz").write_bytes(image_path.read_bytes())
    with pytest.raises(ValueError, match="pairing"):
        CMRMultiAdapter(tmp_path).discover_units()
