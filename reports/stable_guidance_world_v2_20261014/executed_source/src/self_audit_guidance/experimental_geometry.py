"""Opt-in world_left_v2 geometry. No reference access or assumed default side."""
from pathlib import Path
import numpy as np
from .io import digest, safe_image


def world_left_geometry(affine, native_hw, bank_hw, *, sform_code, qform_code):
    """Bank centroids -> native pixel centers -> NIfTI RAS world coordinates.

    Left is world -X. For two regions in the SAME z plane the affine translation,
    z term and half-pixel offset cancel. The returned covector acts on (y,x).
    Header authority follows NIfTI's selected sform, then qform; an uncoded
    fallback has no trustworthy patient orientation and explicitly abstains.
    """
    result = dict(policy='world_left_v2', valid=False, left_vector_yx=[0., 0.],
                  sform_code=int(sform_code), qform_code=int(qform_code),
                  selected_form='sform' if sform_code else 'qform' if qform_code else 'fallback')
    a = np.asarray(affine, dtype=float)
    sizes = np.asarray([*native_hw, *bank_hw], dtype=float)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or np.linalg.matrix_rank(a[:3, :3]) < 3
            or sizes.shape != (4,) or not np.isfinite(sizes).all()
            or (sizes < 1).any() or (sizes != np.floor(sizes)).any()):
        return {**result, 'reason': 'INVALID_AFFINE_OR_GRID'}
    if sform_code <= 0 and qform_code <= 0:
        return {**result, 'reason': 'NO_CODED_WORLD_GEOMETRY'}
    basis, _ = np.linalg.qr(a[:3, :2])
    projected_left_norm = float(np.linalg.norm(basis.T @ [-1., 0., 0.]))
    if projected_left_norm <= 1e-8:
        return {**result, 'reason': 'NO_IN_PLANE_LEFT_INFORMATION'}
    sy, sx = sizes[:2] / sizes[2:]
    transform = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
    return {**result, 'valid': True, 'reason': 'CODED_WORLD_LEFT',
            'native_hw': list(map(int, native_hw)), 'bank_hw': list(map(int, bank_hw)),
            'bank_xy_to_native_xy_pixel_centers': transform.tolist(),
            'in_plane_left_projection_norm': projected_left_norm,
            'left_vector_yx': [-float(a[0, 1]*sy), -float(a[0, 0]*sx)]}


def header_geometry(meta, native_hw, bank_hw):
    """Read a hash-bound raw IMAGE header only, never a mask or patient array."""
    import nibabel as nib
    path = safe_image(Path(meta['raw_image']).resolve())
    before = digest(path)
    if before != meta['image_sha256']:
        raise ValueError('raw cine changed since bank creation')
    nii = nib.load(path)
    if (list(nii.shape) != meta['shape_xyzt']
            or not np.allclose(nii.affine, meta['affine'], rtol=0, atol=1e-5)):
        raise ValueError('header differs from frozen native geometry')
    sform, sc = nii.get_sform(coded=True)
    qform, qc = nii.get_qform(coded=True)
    result = world_left_geometry(nii.affine, native_hw, bank_hw,
                                 sform_code=sc, qform_code=qc)
    result.update(sform=None if sform is None else sform.tolist(),
                  qform=None if qform is None else qform.tolist(),
                  sform_qform_agree=None if sform is None or qform is None else
                  bool(np.allclose(sform, qform, rtol=0, atol=1e-4)),
                  image_sha256=before, spatial_unit=nii.header.get_xyzt_units()[0])
    if digest(path) != before:
        raise ValueError('raw cine changed during geometry audit')
    return result
