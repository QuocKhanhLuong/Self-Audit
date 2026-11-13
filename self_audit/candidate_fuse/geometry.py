"""Policy-bound adjudication of the public M&Ms open-release NIfTI headers.

This module reads only a NIfTI header.  The returned runtime geometry remains
native and orthogonal; no image data is loaded, cropped, or resampled.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from ..candidate_a3d.geometry import canonical_geometry


_POLICY_KEYS = {
    "schema", "name", "dataset_release", "units_evidence_url",
    "units_evidence_sha256", "source_identities_sha256", "reason",
}
UNITS_EVIDENCE_URL = "https://diposit.ub.edu/server/api/core/bitstreams/f2f7dc43-becc-4554-910d-360ed7fd339c/content"
UNITS_EVIDENCE_SHA256 = "c3584856b3933cdeb440d3d2215f1bcdc6d90a7bbe5803bee615e7a823b4c7fd"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_GRAM_ERROR = 1e-3
_MAX_CORNER_DISPLACEMENT_MM = 0.05
_NORM_RTOL = 1e-5
_NORM_ATOL = 1e-5


def _validate_policy(policy: Mapping[str, Any]) -> None:
    if not isinstance(policy, Mapping) or set(policy) != _POLICY_KEYS:
        raise ValueError("policy must contain exactly the locked M&Ms geometry policy fields")
    if policy["schema"] != "self_audit.mnms_geometry_policy.v1":
        raise ValueError("unsupported M&Ms geometry policy schema")
    if policy["name"] != "mnms_open345_mm_polar_v1":
        raise ValueError("unsupported M&Ms geometry policy name")
    if policy["dataset_release"] != "original_2020_open345":
        raise ValueError("M&Ms geometry policy applies only to original_2020_open345")
    url = policy["units_evidence_url"]
    if not isinstance(url, str) or not url.startswith("https://") or not url[8:].strip():
        raise ValueError("units_evidence_url must be a nonempty https URL")
    if url != UNITS_EVIDENCE_URL or policy["units_evidence_sha256"] != UNITS_EVIDENCE_SHA256:
        raise ValueError("units evidence URL/SHA256 do not match the locked M&Ms source")
    for key in ("units_evidence_sha256", "source_identities_sha256"):
        if not isinstance(policy[key], str) or not _SHA256.fullmatch(policy[key]):
            raise ValueError(f"{key} must be a lowercase 64-character SHA256")
    if not isinstance(policy["reason"], str) or not policy["reason"].strip():
        raise ValueError("policy reason must be nonempty")


def _affine(affine: Any, name: str) -> np.ndarray:
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 affine")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} must have a homogeneous final row")
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-12:
        raise ValueError(f"{name} spatial transform must be nonsingular")
    return matrix


def _corners(shape_xyz: tuple[int, int, int]) -> np.ndarray:
    return np.asarray([
        (x, y, z, 1.0)
        for x in (0, shape_xyz[0] - 1)
        for y in (0, shape_xyz[1] - 1)
        for z in (0, shape_xyz[2] - 1)
    ], dtype=np.float64).T


def adjudicate_geometry(header: Any, *, policy: dict) -> dict[str, Any]:
    """Validate and canonicalize one NIfTI header under the locked M&Ms policy.

    ``header`` must be a ``nibabel.Nifti1Header``.  Input geometry is taken
    from a coded sform (preferred) or coded qform; no fallback is inferred.
    """
    _validate_policy(policy)
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ValueError("nibabel is required for NIfTI header adjudication") from exc
    if not isinstance(header, nib.Nifti1Header):
        raise ValueError("header must be a nibabel.Nifti1Header")

    shape = tuple(int(value) for value in header.get_data_shape())
    if len(shape) not in (3, 4) or any(value < 2 for value in shape[:3]):
        raise ValueError("NIfTI shape must be 3D or 4D with each spatial dimension >= 2")
    shape_xyz = shape[:3]

    pixdim = np.asarray(header["pixdim"], dtype=np.float64)[1:4]
    if pixdim.shape != (3,) or not np.isfinite(pixdim).all() or np.any(pixdim <= 0):
        raise ValueError("NIfTI pixdim must contain finite positive spatial values")

    sform, s_code = header.get_sform(coded=True)
    qform, q_code = header.get_qform(coded=True)
    candidates = []
    if int(s_code) > 0:
        candidates.append(("sform", _affine(sform, "coded sform")))
    if int(q_code) > 0:
        candidates.append(("qform", _affine(qform, "coded qform")))
    if not candidates:
        raise ValueError("NIfTI requires a coded sform or qform; identity fallback is forbidden")

    # If both forms are coded, validate both rather than silently ignoring a
    # malformed secondary declaration.  The NIfTI convention prefers sform.
    raw_unit = header.get_xyzt_units()[0]
    scales = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if raw_unit in scales:
        unit_scale = scales[raw_unit]
        unit_interpretation = f"declared_{raw_unit}_converted_to_mm"
    elif raw_unit in (None, "unknown", ""):
        unit_scale = 1.0
        unit_interpretation = "unknown_raw_unit_interpreted_as_mm_by_explicit_policy"
    else:
        raise ValueError(f"unsupported NIfTI spatial unit {raw_unit!r}")

    pixdim_mm = pixdim * unit_scale
    for name, candidate in candidates:
        candidate_mm = candidate[:3, :3] * unit_scale
        candidate_norms = np.linalg.norm(candidate_mm, axis=0)
        if not np.allclose(candidate_norms, pixdim_mm, rtol=_NORM_RTOL, atol=_NORM_ATOL):
            raise ValueError(f"coded {name} column norms disagree with converted pixdim")
        candidate_directions = candidate_mm / candidate_norms
        candidate_gram_error = float(np.max(np.abs(candidate_directions.T @ candidate_directions - np.eye(3))))
        if not np.isfinite(candidate_gram_error) or candidate_gram_error > _MAX_GRAM_ERROR:
            raise ValueError(f"coded {name} normalized affine Gram error exceeds 1e-3")

    chosen_name, original_native = next((item for item in candidates if item[0] == "sform"), candidates[0])
    original_affine_xyz = original_native.copy()
    effective = original_native.copy()
    effective[:3, :] *= unit_scale
    norms = np.linalg.norm(effective[:3, :3], axis=0)
    if not np.allclose(norms, pixdim_mm, rtol=_NORM_RTOL, atol=_NORM_ATOL):
        raise ValueError("coded affine column norms disagree with converted pixdim")
    directions = effective[:3, :3] / norms
    gram_error = float(np.max(np.abs(directions.T @ directions - np.eye(3))))
    if not np.isfinite(gram_error) or gram_error > _MAX_GRAM_ERROR:
        raise ValueError(f"normalized affine Gram error {gram_error:g} exceeds 1e-3")

    left, _singular, right = np.linalg.svd(directions)
    polar = left @ right
    # U@Vt retains the source handedness, including reflections.
    adjusted = effective.copy()
    adjusted[:3, :3] = polar * pixdim_mm[np.newaxis, :]
    corner_world_delta = (effective - adjusted) @ _corners(shape_xyz)
    max_corner_displacement = float(np.max(np.linalg.norm(corner_world_delta[:3], axis=0)))
    if not np.isfinite(max_corner_displacement) or max_corner_displacement > _MAX_CORNER_DISPLACEMENT_MM:
        raise ValueError(
            f"polar correction moves a native corner {max_corner_displacement:g} mm; limit is 0.05 mm"
        )

    # Convert xyz indexing to the runtime D,H,W / z,y,x column convention.
    permutation = (2, 1, 0)
    affine_zyx = np.eye(4, dtype=np.float64)
    affine_zyx[:3, :3] = adjusted[:3, permutation]
    affine_zyx[:3, 3] = adjusted[:3, 3]
    shape_zyx = tuple(shape_xyz[index] for index in permutation)
    spacing_zyx = tuple(float(pixdim_mm[index]) for index in permutation)
    geometry = canonical_geometry({
        "shape": shape_zyx,
        "spacing_zyx": spacing_zyx,
        "affine_zyx": affine_zyx.tolist(),
        "context": "full_volume",
    })

    return {
        "policy_name": policy["name"],
        "geometry": geometry,
        "raw_unit": raw_unit,
        "effective_unit": "mm",
        "original_affine_xyz": original_affine_xyz.tolist(),
        "effective_affine_xyz": adjusted.tolist(),
        "gram_error": gram_error,
        "max_corner_displacement_mm": max_corner_displacement,
        "unit_interpretation": unit_interpretation,
        "affine_source": chosen_name,
        "canonical_valid": True,
    }


__all__ = ["adjudicate_geometry", "UNITS_EVIDENCE_URL", "UNITS_EVIDENCE_SHA256"]
