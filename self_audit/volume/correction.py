"""Executable native-3D point and sparse correction workflow.

This module implements G9: deterministic prompt-conditioned image and probability
refinement with strict hard manual preservation, provenance tracking, and iterative persistence.

Contracts and boundaries:
- Pure deterministic algorithms: image-guided geodesic and bilateral propagation.
- Hard manual preservation: manual edits are immutable and never overwritten by
  propagation, smoothing, or prior probabilities.
- Iterative correction persistence: prior manual edits (from prior_provenance == PROVENANCE_MANUAL)
  are strictly preserved unless explicitly superseded by a later edit at the same coordinate.
- Separation of provenance: ORIGINAL (1), MANUAL (2), PROPAGATED (3), PSEUDO (4),
  and UNKNOWN (255) are strictly tracked per voxel.
- Simulated clicks strictly respect the TRAIN-L budget; reading, opening, or simulating
  clicks from U/VAL/TEST masks is forbidden and fails closed before any mask access.
- Click simulation authorization uses an allowlist for TRAIN-L or explicit synthetic fixture;
  TRAIN-U, TESTING, TYPO, and plain unproven TRAIN are rejected.
- Physical geometry: spatial coordinates and spacing are in millimeters. Declared NIfTI meter
  units are converted to mm; missing physical geometry fails closed for real data.
- Interactive correction is an additional-supervision protocol (I01), not automatic
  inference (I00), and is explicitly callable with a declared click budget.
- Real ISO 8601 timestamps are recorded; human interaction timing is never fabricated.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from .geometry import (
    UNKNOWN_LABEL,
    sha256_file,
    spacing_from_affine,
    voxel_to_world,
    world_to_voxel,
)

SCHEMA = "self_audit.volume.correction.v1"

# Provenance identifiers
PROVENANCE_ORIGINAL = 1
PROVENANCE_MANUAL = 2
PROVENANCE_PROPAGATED = 3
PROVENANCE_PSEUDO = 4
PROVENANCE_UNKNOWN = 255

PROVENANCE_NAMES = {
    PROVENANCE_ORIGINAL: "original",
    PROVENANCE_MANUAL: "manual",
    PROVENANCE_PROPAGATED: "propagated",
    PROVENANCE_PSEUDO: "pseudo",
    PROVENANCE_UNKNOWN: "unknown",
}


def validate_geometry(
    affine: np.ndarray | None,
    spacing_xyz_mm: tuple[float, float, float] | Sequence[float] | None,
    *,
    synthetic_fixture: bool = False,
) -> tuple[np.ndarray | None, tuple[float, float, float]]:
    """Validate physical affine matrix and voxel spacing.

    Fails closed on missing, singular, or non-finite geometry for real data.
    Synthetic fixtures can opt-in to default identity / 1mm spacing via synthetic_fixture=True.
    """
    if affine is None and spacing_xyz_mm is None:
        if not synthetic_fixture:
            raise ValueError(
                "Missing physical geometry (affine or spacing_xyz_mm) for real data. "
                "Guessing identity affine or 1mm spacing is forbidden; "
                "declare explicit fixture geometry or set synthetic_fixture=True for synthetic test fixtures."
            )
        return np.eye(4, dtype=float), (1.0, 1.0, 1.0)

    val_affine: np.ndarray | None = None
    if affine is not None:
        val_affine = np.asarray(affine, dtype=float)
        if val_affine.shape != (4, 4):
            raise ValueError(f"Affine matrix must be 4x4, got {val_affine.shape}")
        if not np.all(np.isfinite(val_affine)):
            raise ValueError("Affine matrix must contain finite numbers (no NaN or Inf)")
        det = float(np.linalg.det(val_affine[:3, :3]))
        if not np.isfinite(det) or abs(det) < 1e-8:
            raise ValueError(f"Affine matrix must be nonsingular (got determinant {det})")

    if val_affine is not None:
        calc_spacing = tuple(float(x) for x in spacing_from_affine(val_affine))
        if spacing_xyz_mm is None:
            val_spacing = calc_spacing
        else:
            val_spacing = tuple(float(x) for x in spacing_xyz_mm)
            if len(val_spacing) != 3:
                raise ValueError(f"spacing_xyz_mm must be 3-tuple, got {val_spacing}")
            if not all(np.isfinite(s) and s > 0 for s in val_spacing):
                raise ValueError(f"spacing_xyz_mm must be strictly positive and finite, got {val_spacing}")
            if not np.allclose(calc_spacing, val_spacing, atol=1e-3, rtol=1e-3):
                raise ValueError(
                    f"Inconsistent geometry: affine spacing {calc_spacing} != spacing_xyz_mm {val_spacing}"
                )
    else:
        val_spacing = tuple(float(x) for x in spacing_xyz_mm)  # type: ignore[arg-type]
        if len(val_spacing) != 3:
            raise ValueError(f"spacing_xyz_mm must be 3-tuple, got {val_spacing}")
        if not all(np.isfinite(s) and s > 0 for s in val_spacing):
            raise ValueError(f"spacing_xyz_mm must be strictly positive and finite, got {val_spacing}")
        if synthetic_fixture:
            val_affine = np.diag([val_spacing[0], val_spacing[1], val_spacing[2], 1.0])

    return val_affine, val_spacing


@dataclass
class PointPrompt:
    """A sparse point edit prompt in native volume or physical coordinates."""

    coord: tuple[float, float, float]
    coord_system: str = "native_voxel_xyz"  # "world_mm", "native_voxel_xyz", "native_voxel_zyx"
    target_class: int = 1
    prompt_type: str = "positive"  # "positive", "negative", "edit"
    radius_voxels: float = 0.0
    weight: float = 1.0
    slice_index: int | None = None
    prompt_id: str | None = None

    def __post_init__(self) -> None:
        self.coord = (float(self.coord[0]), float(self.coord[1]), float(self.coord[2]))
        self.target_class = int(self.target_class)
        self.radius_voxels = float(self.radius_voxels)
        self.weight = float(self.weight)
        if self.coord_system not in ("world_mm", "native_voxel_xyz", "native_voxel_zyx"):
            raise ValueError(f"Unsupported coord_system: {self.coord_system}")
        if self.target_class < 0:
            raise ValueError(f"target_class must be non-negative, got {self.target_class}")
        if self.radius_voxels < 0:
            raise ValueError(f"radius_voxels must be >= 0, got {self.radius_voxels}")
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")
        if self.prompt_type not in ("positive", "negative", "edit"):
            raise ValueError(
                f"Unsupported prompt_type: {self.prompt_type!r}. Must be 'positive', 'negative', or 'edit'."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PointPrompt:
        coord = data.get("coord")
        if coord is None or len(coord) != 3:
            raise ValueError(f"PointPrompt requires 3-element 'coord', got {coord}")
        return cls(
            coord=(float(coord[0]), float(coord[1]), float(coord[2])),
            coord_system=str(data.get("coord_system", "native_voxel_xyz")),
            target_class=int(data.get("target_class", 1)),
            prompt_type=str(data.get("prompt_type", "positive")),
            radius_voxels=float(data.get("radius_voxels", 0.0)),
            weight=float(data.get("weight", 1.0)),
            slice_index=int(data["slice_index"]) if data.get("slice_index") is not None else None,
            prompt_id=str(data["prompt_id"]) if data.get("prompt_id") is not None else None,
        )


def resolve_prompt_to_voxel_zyx(
    prompt: PointPrompt,
    shape_zyx: tuple[int, int, int],
    affine: np.ndarray | None = None,
) -> tuple[int, int, int]:
    """Resolve a PointPrompt into discrete native integer (z, y, x) indices."""
    zdim, h, w = shape_zyx
    c = prompt.coord
    if prompt.coord_system == "world_mm":
        if affine is None:
            raise ValueError("affine matrix required to resolve world_mm coordinates")
        vox_xyz = world_to_voxel(affine, c)
        vx, vy, vz = vox_xyz[0], vox_xyz[1], vox_xyz[2]
        ix, iy, iz = int(np.round(vx)), int(np.round(vy)), int(np.round(vz))
    elif prompt.coord_system == "native_voxel_xyz":
        ix, iy, iz = int(np.round(c[0])), int(np.round(c[1])), int(np.round(c[2]))
    elif prompt.coord_system == "native_voxel_zyx":
        iz, iy, ix = int(np.round(c[0])), int(np.round(c[1])), int(np.round(c[2]))
    else:
        raise ValueError(f"Unknown coord_system: {prompt.coord_system}")

    if not (0 <= iz < zdim and 0 <= iy < h and 0 <= ix < w):
        raise ValueError(
            f"Prompt coordinates {c} in {prompt.coord_system} resolved to voxel "
            f"(z={iz}, y={iy}, x={ix}) which is out of bounds for shape {shape_zyx}"
        )
    return iz, iy, ix


def save_prompts(prompts: Sequence[PointPrompt], path: str | Path) -> None:
    """Save point prompts to a reproducible JSON fixture."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "count": len(prompts),
        "prompts": [pr.to_dict() for pr in prompts],
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_prompts(path: str | Path) -> list[PointPrompt]:
    """Load point prompts from a JSON fixture."""
    p = Path(path)
    data = json.loads(p.read_text())
    prompts_data = data.get("prompts", [])
    return [PointPrompt.from_dict(d) for d in prompts_data]


def compute_geodesic_distance_map_3d(
    image_zyx: np.ndarray,
    seeds_zyx: Sequence[tuple[int, int, int]],
    spacing_xyz_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    gamma_intensity: float = 10.0,
    max_distance_mm: float | None = None,
    seed_weights: Sequence[float] | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute exact 3D Dijkstra geodesic distance map from seeds with intensity guidance."""
    import heapq

    zdim, h, w = image_zyx.shape
    dist = np.full((zdim, h, w), np.inf, dtype=np.float32)
    valid = np.ones((zdim, h, w), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != (zdim, h, w):
        raise ValueError(f"valid_mask shape {valid.shape} != image shape {(zdim, h, w)}")
    if not seeds_zyx:
        return dist

    sx, sy, sz = spacing_xyz_mm
    pq: list[tuple[float, int, int, int]] = []

    # Initialize seeds
    for idx, (z, y, x) in enumerate(seeds_zyx):
        if 0 <= z < zdim and 0 <= y < h and 0 <= x < w and valid[z, y, x]:
            w_val = float(seed_weights[idx]) if seed_weights is not None else 1.0
            init_d = 0.0 if w_val >= 1.0 else (1.0 - w_val) * 5.0
            dist[z, y, x] = min(dist[z, y, x], init_d)
            heapq.heappush(pq, (dist[z, y, x], z, y, x))

    neighbors = [
        (-1, 0, 0, sz * sz),
        (1, 0, 0, sz * sz),
        (0, -1, 0, sy * sy),
        (0, 1, 0, sy * sy),
        (0, 0, -1, sx * sx),
        (0, 0, 1, sx * sx),
    ]

    gamma_sq = gamma_intensity * gamma_intensity

    while pq:
        d, z, y, x = heapq.heappop(pq)
        if d > dist[z, y, x]:
            continue
        if max_distance_mm is not None and d > max_distance_mm:
            break

        val_cur = float(image_zyx[z, y, x])

        for dz, dy, dx, spatial_sq in neighbors:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < zdim and 0 <= ny < h and 0 <= nx < w and valid[nz, ny, nx]:
                val_next = float(image_zyx[nz, ny, nx])
                diff_val = val_cur - val_next
                step_cost = float(np.sqrt(spatial_sq + gamma_sq * (diff_val * diff_val)))
                cand_d = d + step_cost
                if cand_d < dist[nz, ny, nx]:
                    dist[nz, ny, nx] = cand_d
                    heapq.heappush(pq, (cand_d, nz, ny, nx))

    return dist


def compute_geodesic_distance_map_2d(
    image_zyx: np.ndarray,
    seeds_zyx: Sequence[tuple[int, int, int]],
    spacing_xyz_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    gamma_intensity: float = 10.0,
    inter_slice_penalty_mm: float = 20.0,
    seed_weights: Sequence[float] | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute slice-wise 2D geodesic distances with anisotropic inter-slice coupling."""
    import heapq

    zdim, h, w = image_zyx.shape
    dist = np.full((zdim, h, w), np.inf, dtype=np.float32)
    valid = np.ones((zdim, h, w), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != (zdim, h, w):
        raise ValueError(f"valid_mask shape {valid.shape} != image shape {(zdim, h, w)}")
    if not seeds_zyx:
        return dist

    sx, sy, sz = spacing_xyz_mm
    gamma_sq = gamma_intensity * gamma_intensity
    neighbors_2d = [
        (-1, 0, sy * sy),
        (1, 0, sy * sy),
        (0, -1, sx * sx),
        (0, 1, sx * sx),
    ]

    seeds_by_z: dict[int, list[tuple[int, int, float]]] = {}
    for idx, (z, y, x) in enumerate(seeds_zyx):
        w_val = float(seed_weights[idx]) if seed_weights is not None else 1.0
        seeds_by_z.setdefault(z, []).append((y, x, w_val))

    for z_seed, yx_seeds in seeds_by_z.items():
        if not (0 <= z_seed < zdim):
            continue
        slice_dist = np.full((h, w), np.inf, dtype=np.float32)
        pq: list[tuple[float, int, int]] = []
        for y, x, w_val in yx_seeds:
            if 0 <= y < h and 0 <= x < w and valid[z_seed, y, x]:
                init_d = 0.0 if w_val >= 1.0 else (1.0 - w_val) * 5.0
                slice_dist[y, x] = min(slice_dist[y, x], init_d)
                heapq.heappush(pq, (slice_dist[y, x], y, x))

        slice_img = image_zyx[z_seed]
        while pq:
            d, y, x = heapq.heappop(pq)
            if d > slice_dist[y, x]:
                continue
            val_cur = float(slice_img[y, x])
            for dy, dx, spatial_sq in neighbors_2d:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and valid[z_seed, ny, nx]:
                    val_next = float(slice_img[ny, nx])
                    diff_val = val_cur - val_next
                    step_cost = float(np.sqrt(spatial_sq + gamma_sq * (diff_val * diff_val)))
                    cand_d = d + step_cost
                    if cand_d < slice_dist[ny, nx]:
                        slice_dist[ny, nx] = cand_d
                        heapq.heappush(pq, (cand_d, ny, nx))

        for z in range(zdim):
            z_offset_mm = abs(z - z_seed) * sz + (abs(z - z_seed) * inter_slice_penalty_mm if z != z_seed else 0.0)
            lo, hi = sorted((z_seed, z))
            path_valid = np.all(valid[lo : hi + 1], axis=0)
            allowed = valid[z] & path_valid
            dist[z] = np.minimum(dist[z], np.where(allowed, slice_dist + z_offset_mm, np.inf))

    return dist


def compute_image_guided_affinity(
    image_zyx: np.ndarray,
    seeds_zyx: Sequence[tuple[int, int, int]],
    spacing_xyz_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigma_spatial_mm: float = 8.0,
    gamma_intensity: float = 10.0,
    seed_weights: Sequence[float] | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute vectorized image-guided bilateral affinity from seed points."""
    zdim, h, w = image_zyx.shape
    affinity = np.zeros((zdim, h, w), dtype=np.float32)
    valid = np.ones((zdim, h, w), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != (zdim, h, w):
        raise ValueError(f"valid_mask shape {valid.shape} != image shape {(zdim, h, w)}")
    if not seeds_zyx:
        return affinity

    sx, sy, sz = spacing_xyz_mm
    zz, yy, xx = np.ogrid[:zdim, :h, :w]
    spatial_scale = 2.0 * (sigma_spatial_mm * sigma_spatial_mm)
    # Image-guided affinity is otherwise a direct Gaussian from each seed and
    # would jump across an invalid slab.  Connected-component gating makes the
    # support boundary an actual barrier while retaining local propagation
    # inside each valid component.
    component_labels, _ = ndimage.label(
        valid,
        structure=ndimage.generate_binary_structure(3, 1),
    )

    for idx, (sz_pt, sy_pt, sx_pt) in enumerate(seeds_zyx):
        if not (0 <= sz_pt < zdim and 0 <= sy_pt < h and 0 <= sx_pt < w and valid[sz_pt, sy_pt, sx_pt]):
            continue
        w_val = float(seed_weights[idx]) if seed_weights is not None else 1.0
        seed_val = float(image_zyx[sz_pt, sy_pt, sx_pt])
        d_spatial_sq = (
            ((zz - sz_pt) * sz) ** 2
            + ((yy - sy_pt) * sy) ** 2
            + ((xx - sx_pt) * sx) ** 2
        )
        d_intensity_sq = (image_zyx - seed_val) ** 2
        cur_aff = np.exp(-(d_spatial_sq / spatial_scale) - (gamma_intensity * d_intensity_sq)) * w_val
        cur_aff[component_labels != component_labels[sz_pt, sy_pt, sx_pt]] = 0.0
        affinity = np.maximum(affinity, cur_aff.astype(np.float32))

    return affinity


@dataclass
class CorrectionResult:
    """Outcome of volume correction with separated probabilities and provenance."""

    probabilities: np.ndarray  # (C, Z, Y, X) float32
    labels: np.ndarray  # (Z, Y, X) uint8
    provenance: np.ndarray  # (Z, Y, X) uint8
    summary: dict[str, Any]
    prompts: list[PointPrompt] = field(default_factory=list)
    affine: np.ndarray | None = None
    patient_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    synthetic_fixture: bool = False

    def export_nifti(
        self,
        out_labels_path: str | Path | None = None,
        out_provenance_path: str | Path | None = None,
        out_probs_path: str | Path | None = None,
        synthetic_fixture: bool = False,
    ) -> None:
        """Export native NIfTI segmentation and provenance volumes with declared mm units."""
        import nibabel as nib

        is_synthetic = synthetic_fixture or self.synthetic_fixture
        if self.affine is None:
            if not is_synthetic:
                raise ValueError(
                    "export_nifti requires known physical affine. "
                    "Inventing identity affine is forbidden for real data; set synthetic_fixture=True for synthetic test fixtures."
                )
            aff = np.eye(4, dtype=float)
        else:
            aff = np.asarray(self.affine, dtype=float)
            if aff.shape != (4, 4) or not np.all(np.isfinite(aff)):
                raise ValueError("export_nifti requires finite 4x4 affine matrix")
            det = float(np.linalg.det(aff[:3, :3]))
            if not np.isfinite(det) or abs(det) < 1e-8:
                raise ValueError(f"export_nifti requires nonsingular affine matrix (got determinant {det})")

        if out_labels_path is not None:
            labels_xyz = np.transpose(self.labels, (2, 1, 0))
            nii_labels = nib.Nifti1Image(labels_xyz.astype(np.uint8), aff)
            nii_labels.header.set_xyzt_units("mm", "sec")
            p_lab = Path(out_labels_path)
            p_lab.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nii_labels, str(p_lab))

        if out_provenance_path is not None:
            prov_xyz = np.transpose(self.provenance, (2, 1, 0))
            nii_prov = nib.Nifti1Image(prov_xyz.astype(np.uint8), aff)
            nii_prov.header.set_xyzt_units("mm", "sec")
            p_prov = Path(out_provenance_path)
            p_prov.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nii_prov, str(p_prov))

        if out_probs_path is not None:
            probs_xyz_c = np.transpose(self.probabilities, (3, 2, 1, 0))
            nii_probs = nib.Nifti1Image(probs_xyz_c.astype(np.float32), aff)
            nii_probs.header.set_xyzt_units("mm", "sec")
            p_probs = Path(out_probs_path)
            p_probs.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nii_probs, str(p_probs))

    def export_json(self, out_json_path: str | Path) -> None:
        """Export comprehensive audit and provenance metadata."""
        p = Path(out_json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "created_at": self.created_at,
            "patient_id": self.patient_id,
            "synthetic_fixture": self.synthetic_fixture,
            "prompt_count": len(self.prompts),
            "prompts": [pr.to_dict() for pr in self.prompts],
            "summary": self.summary,
        }
        p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class VolumeCorrector:
    """Engine for prompt-conditioned 3D volume refinement with hard manual preservation."""

    def __init__(
        self,
        *,
        num_classes: int = 4,
        propagation_mode: str = "geodesic_3d",
        sigma_mm: float = 6.0,
        gamma_intensity: float = 8.0,
        prompt_weight: float = 0.90,
        max_distance_mm: float | None = 60.0,
        min_affinity: float = 0.05,
    ) -> None:
        if propagation_mode not in ("geodesic_3d", "geodesic_2d", "image_guided", "manual_only"):
            raise ValueError(f"Unknown propagation_mode: {propagation_mode}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = num_classes
        self.propagation_mode = propagation_mode
        self.sigma_mm = float(sigma_mm)
        self.gamma_intensity = float(gamma_intensity)
        self.prompt_weight = float(prompt_weight)
        self.max_distance_mm = float(max_distance_mm) if max_distance_mm is not None else None
        self.min_affinity = float(min_affinity)

    def correct(
        self,
        image_zyx: np.ndarray,
        prompts: Sequence[PointPrompt],
        *,
        prior_probabilities: np.ndarray | None = None,
        prior_labels: np.ndarray | None = None,
        prior_provenance: np.ndarray | None = None,
        affine: np.ndarray | None = None,
        spacing_xyz_mm: tuple[float, float, float] | None = None,
        geometric_valid: np.ndarray | None = None,
        patient_id: str | None = None,
        prior_is_pseudo: bool = False,
        synthetic_fixture: bool = False,
    ) -> CorrectionResult:
        """Execute deterministic correction with strict hard manual preservation and iterative persistence."""
        img = np.asarray(image_zyx, dtype=np.float32)
        if img.ndim != 3:
            raise ValueError(f"image_zyx must be 3D (Z, Y, X), got shape {img.shape}")
        zdim, h, w = img.shape
        shape_zyx = (zdim, h, w)

        affine, spacing_xyz_mm = validate_geometry(
            affine, spacing_xyz_mm, synthetic_fixture=synthetic_fixture
        )

        # Check geometric valid mask
        if geometric_valid is not None:
            geo_invalid = ~np.asarray(geometric_valid, dtype=bool)
            if geo_invalid.shape != shape_zyx:
                raise ValueError(f"geometric_valid shape {geo_invalid.shape} != image shape {shape_zyx}")
        else:
            geo_invalid = np.zeros(shape_zyx, dtype=bool)

        # Normalize intensities to robust [0, 1] range if needed for stable
        # metric gradients. Invalid geometric support is excluded from the
        # statistics so out-of-support values cannot perturb valid voxels.
        valid_support = ~geo_invalid
        if np.any(valid_support):
            valid_values = img[valid_support]
            p_low, p_high = np.percentile(valid_values, [1.0, 99.0])
            scale = max(float(p_high - p_low), 1e-4)
            norm_img = np.clip((img - p_low) / scale, 0.0, 1.0).astype(np.float32)
        else:
            # Keep a finite no-op array; outputs are forced UNKNOWN below.
            norm_img = np.zeros_like(img, dtype=np.float32)

        # Build initial base probabilities and track prior unknown voxels
        has_prior = (prior_labels is not None) or (prior_probabilities is not None)
        p_lab = None
        if prior_labels is not None:
            p_lab = np.asarray(prior_labels)
            if p_lab.shape != shape_zyx:
                raise ValueError(f"prior_labels shape {p_lab.shape} != image shape {shape_zyx}")
            prior_unknown = (p_lab == UNKNOWN_LABEL)
            prior_known = (p_lab != UNKNOWN_LABEL)
            p_prior = np.full((self.num_classes, zdim, h, w), 1.0 / self.num_classes, dtype=np.float32)
            if np.any(prior_known):
                for c in range(self.num_classes):
                    p_prior[c, p_lab == c] = 0.95
                    p_prior[c, prior_known & (p_lab != c)] = 0.05 / (self.num_classes - 1)
            p_prior = p_prior / np.sum(p_prior, axis=0, keepdims=True)
            prior_argmax = p_lab.copy()
        elif prior_probabilities is not None:
            p_prior = np.asarray(prior_probabilities, dtype=np.float32)
            if p_prior.shape != (self.num_classes, zdim, h, w):
                raise ValueError(
                    f"prior_probabilities shape {p_prior.shape} != expected ({self.num_classes}, {zdim}, {h}, {w})"
                )
            p_prior = np.clip(p_prior, 1e-6, 1.0)
            p_prior = p_prior / np.sum(p_prior, axis=0, keepdims=True)
            prior_argmax = np.argmax(p_prior, axis=0).astype(np.uint8)
            prior_unknown = np.zeros(shape_zyx, dtype=bool)
        else:
            p_prior = np.full((self.num_classes, zdim, h, w), 1.0 / self.num_classes, dtype=np.float32)
            prior_argmax = np.full(shape_zyx, UNKNOWN_LABEL, dtype=np.uint8)
            prior_unknown = np.ones(shape_zyx, dtype=bool)

        # Check prior provenance if provided (for iterative correction persistence)
        p_prov = None
        prior_manual_mask = np.zeros(shape_zyx, dtype=bool)
        if prior_provenance is not None:
            p_prov = np.asarray(prior_provenance)
            if p_prov.shape != shape_zyx:
                raise ValueError(f"prior_provenance shape {p_prov.shape} != image shape {shape_zyx}")
            prior_manual_mask = (p_prov == PROVENANCE_MANUAL)

        # IF NO PROMPTS AND NO PRIOR MANUAL EDITS: return baseline unchanged
        if not prompts and not np.any(prior_manual_mask):
            if not has_prior and p_prov is None:
                out_labels = np.full(shape_zyx, UNKNOWN_LABEL, dtype=np.uint8)
                out_prov = np.full(shape_zyx, PROVENANCE_UNKNOWN, dtype=np.uint8)
            else:
                out_labels = prior_argmax.copy()
                if p_prov is not None:
                    out_prov = p_prov.copy()
                else:
                    out_prov = np.full(
                        shape_zyx,
                        PROVENANCE_PSEUDO if prior_is_pseudo else PROVENANCE_ORIGINAL,
                        dtype=np.uint8,
                    )
                # Prior unknown voxels stay unknown
                out_prov[prior_unknown] = PROVENANCE_UNKNOWN
                out_labels[prior_unknown] = UNKNOWN_LABEL

            # Geometric invalidity strictly enforces UNKNOWN
            out_prov[geo_invalid] = PROVENANCE_UNKNOWN
            out_labels[geo_invalid] = UNKNOWN_LABEL

            summary = {
                "total_voxels": int(zdim * h * w),
                "manual_voxels": int(np.sum(out_prov == PROVENANCE_MANUAL)),
                "propagated_voxels": int(np.sum(out_prov == PROVENANCE_PROPAGATED)),
                "original_voxels": int(np.sum(out_prov == PROVENANCE_ORIGINAL)),
                "pseudo_voxels": int(np.sum(out_prov == PROVENANCE_PSEUDO)),
                "unknown_voxels": int(np.sum(out_prov == PROVENANCE_UNKNOWN)),
                "affected_slices": [],
                "class_distribution": {
                    c: int(np.sum(out_labels == c)) for c in range(self.num_classes)
                },
                "unknown_label_count": int(np.sum(out_labels == UNKNOWN_LABEL)),
                "protocol": "automatic_only",
                "supervision_type": "automatic_inference",
                "simulated": False,
                "human_annotation_claimed": False,
                "human_annotation_time_sec": None,
            }
            return CorrectionResult(
                probabilities=p_prior,
                labels=out_labels,
                provenance=out_prov,
                summary=summary,
                prompts=[],
                affine=affine,
                patient_id=patient_id,
                synthetic_fixture=synthetic_fixture,
            )

        # Resolve prompts to voxel coordinates
        resolved_seeds_by_class: dict[int, list[tuple[int, int, int]]] = {}
        seed_weights_by_class: dict[int, list[float]] = {}
        manual_mask = np.zeros(shape_zyx, dtype=bool)
        manual_labels = np.zeros(shape_zyx, dtype=np.uint8)
        negative_mask = np.zeros(shape_zyx, dtype=bool)
        affected_slices: set[int] = set()

        # Incorporate prior manual edits into manual_mask (iterative persistence)
        if np.any(prior_manual_mask):
            manual_mask[prior_manual_mask] = True
            if p_lab is not None:
                manual_labels[prior_manual_mask] = p_lab[prior_manual_mask]
            elif p_prov is not None:
                manual_labels[prior_manual_mask] = prior_argmax[prior_manual_mask]

            for z, y, x in np.argwhere(prior_manual_mask):
                c = int(manual_labels[z, y, x])
                if c < self.num_classes:
                    resolved_seeds_by_class.setdefault(c, []).append((int(z), int(y), int(x)))
                    seed_weights_by_class.setdefault(c, []).append(1.0)
                    affected_slices.add(int(z))

        for pr in prompts:
            iz, iy, ix = resolve_prompt_to_voxel_zyx(pr, shape_zyx, affine)
            if geo_invalid[iz, iy, ix]:
                raise ValueError(
                    f"Prompt coordinate {(iz, iy, ix)} falls on invalid geometric support."
                )
            affected_slices.add(iz)
            c = pr.target_class
            if c >= self.num_classes:
                raise ValueError(f"Prompt target_class {c} >= num_classes {self.num_classes}")

            target_label = 0 if pr.prompt_type == "negative" else c

            if pr.radius_voxels <= 0.5:
                vox_mask = np.zeros(shape_zyx, dtype=bool)
                vox_mask[iz, iy, ix] = True
            else:
                rad = pr.radius_voxels
                z_min, z_max = max(0, int(iz - np.ceil(rad))), min(zdim, int(iz + np.ceil(rad) + 1))
                y_min, y_max = max(0, int(iy - np.ceil(rad))), min(h, int(iy + np.ceil(rad) + 1))
                x_min, x_max = max(0, int(ix - np.ceil(rad))), min(w, int(ix + np.ceil(rad) + 1))
                zz_sub, yy_sub, xx_sub = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
                sphere = (
                    ((zz_sub - iz) ** 2) + ((yy_sub - iy) ** 2) + ((xx_sub - ix) ** 2)
                ) <= (rad * rad)
                vox_mask = np.zeros(shape_zyx, dtype=bool)
                vox_mask[z_min:z_max, y_min:y_max, x_min:x_max] = sphere
                vox_mask &= valid_support

            manual_mask[vox_mask] = True
            manual_labels[vox_mask] = target_label

            resolved_seeds_by_class.setdefault(target_label, []).append((iz, iy, ix))
            seed_weights_by_class.setdefault(target_label, []).append(pr.weight)

            if pr.prompt_type == "negative":
                negative_mask[vox_mask] = True

        # Propagation
        refined_p = p_prior.copy()
        active_prop = np.zeros(shape_zyx, dtype=bool)

        if self.propagation_mode == "manual_only":
            pass  # only manual edits, no propagation
        elif self.propagation_mode in ("geodesic_3d", "geodesic_2d"):
            dist_maps: dict[int, np.ndarray] = {}
            for c, seeds in resolved_seeds_by_class.items():
                s_weights = seed_weights_by_class.get(c)
                if self.propagation_mode == "geodesic_3d":
                    dist = compute_geodesic_distance_map_3d(
                        norm_img,
                        seeds,
                        spacing_xyz_mm=spacing_xyz_mm,
                        gamma_intensity=self.gamma_intensity,
                        max_distance_mm=self.max_distance_mm,
                        seed_weights=s_weights,
                        valid_mask=valid_support,
                    )
                else:
                    dist = compute_geodesic_distance_map_2d(
                        norm_img,
                        seeds,
                        spacing_xyz_mm=spacing_xyz_mm,
                        gamma_intensity=self.gamma_intensity,
                        seed_weights=s_weights,
                        valid_mask=valid_support,
                    )
                dist_maps[c] = dist

            # Convert geodesic distance to soft affinities
            affinities = np.zeros((self.num_classes, zdim, h, w), dtype=np.float32)
            for c, dist in dist_maps.items():
                affinities[c] = np.exp(-dist / self.sigma_mm)

            sum_aff = np.sum(affinities, axis=0, keepdims=True)
            max_aff = np.max(affinities, axis=0)

            valid_aff = sum_aff > 1e-5
            aff_probs = np.zeros_like(affinities)
            np.divide(affinities, sum_aff, out=aff_probs, where=valid_aff)

            alpha = np.where(max_aff >= self.min_affinity, np.clip(max_aff * self.prompt_weight, 0.0, 1.0), 0.0)[None, ...]
            blended = (1.0 - alpha) * p_prior + alpha * aff_probs
            refined_p = blended / np.sum(blended, axis=0, keepdims=True)
            active_prop = (max_aff >= self.min_affinity) & (~manual_mask) & valid_support

        elif self.propagation_mode == "image_guided":
            affinities = np.zeros((self.num_classes, zdim, h, w), dtype=np.float32)
            for c, seeds in resolved_seeds_by_class.items():
                s_weights = seed_weights_by_class.get(c)
                affinities[c] = compute_image_guided_affinity(
                    norm_img,
                    seeds,
                    spacing_xyz_mm=spacing_xyz_mm,
                    sigma_spatial_mm=self.sigma_mm,
                    gamma_intensity=self.gamma_intensity,
                    seed_weights=s_weights,
                    valid_mask=valid_support,
                )
            sum_aff = np.sum(affinities, axis=0, keepdims=True)
            max_aff = np.max(affinities, axis=0)
            valid_aff = sum_aff > 1e-5
            aff_probs = np.zeros_like(affinities)
            np.divide(affinities, sum_aff, out=aff_probs, where=valid_aff)
            alpha = np.where(max_aff >= self.min_affinity, np.clip(max_aff * self.prompt_weight, 0.0, 1.0), 0.0)[None, ...]
            blended = (1.0 - alpha) * p_prior + alpha * aff_probs
            refined_p = blended / np.sum(blended, axis=0, keepdims=True)
            active_prop = (max_aff >= self.min_affinity) & (~manual_mask) & valid_support

        # Negative prompt effect: suppress target_class probabilities
        if np.any(negative_mask):
            refined_p[1:, negative_mask] = 0.0
            refined_p[0, negative_mask] = 1.0

        # HARD MANUAL PRESERVATION:
        # At manual edit voxels, enforce exact label and probability 1.0
        for c in range(self.num_classes):
            c_mask = manual_mask & (manual_labels == c)
            refined_p[c, c_mask] = 1.0
            other_c = [k for k in range(self.num_classes) if k != c]
            for oc in other_c:
                refined_p[oc, c_mask] = 0.0

        # Derived final discrete labels
        final_labels = np.argmax(refined_p, axis=0).astype(np.uint8)
        # Verify manual preservation
        final_labels[manual_mask] = manual_labels[manual_mask]

        # Determine provenance
        if p_prov is not None:
            provenance = p_prov.copy()
        elif has_prior:
            provenance = np.full(
                shape_zyx,
                PROVENANCE_PSEUDO if prior_is_pseudo else PROVENANCE_ORIGINAL,
                dtype=np.uint8,
            )
        else:
            provenance = np.full(shape_zyx, PROVENANCE_UNKNOWN, dtype=np.uint8)

        # Untouched voxels (neither manual nor active propagation)
        untouched = (~manual_mask) & (~active_prop)
        if has_prior:
            if p_lab is not None:
                final_labels[untouched] = p_lab[untouched]
            untouched_unknown = untouched & prior_unknown
            final_labels[untouched_unknown] = UNKNOWN_LABEL
            provenance[untouched_unknown] = PROVENANCE_UNKNOWN
        else:
            final_labels[untouched] = UNKNOWN_LABEL
            provenance[untouched] = PROVENANCE_UNKNOWN

        # Propagated voxels
        if np.any(active_prop):
            if p_lab is not None:
                changed = active_prop & (prior_unknown | (final_labels != p_lab))
            elif prior_probabilities is not None:
                changed = active_prop & (final_labels != prior_argmax)
            else:
                changed = active_prop
            provenance[changed] = PROVENANCE_PROPAGATED

        # Manual voxels (includes both new and preserved prior manual edits)
        provenance[manual_mask] = PROVENANCE_MANUAL

        # Unknown / invalid support strictly enforces UNKNOWN
        if np.any(geo_invalid):
            provenance[geo_invalid] = PROVENANCE_UNKNOWN
            final_labels[geo_invalid] = UNKNOWN_LABEL

        # Provenance summary statistics
        counts = {
            "total_voxels": int(zdim * h * w),
            "manual_voxels": int(np.sum(provenance == PROVENANCE_MANUAL)),
            "propagated_voxels": int(np.sum(provenance == PROVENANCE_PROPAGATED)),
            "original_voxels": int(np.sum(provenance == PROVENANCE_ORIGINAL)),
            "pseudo_voxels": int(np.sum(provenance == PROVENANCE_PSEUDO)),
            "unknown_voxels": int(np.sum(provenance == PROVENANCE_UNKNOWN)),
            "affected_slices": sorted(affected_slices),
            "class_distribution": {
                c: int(np.sum(final_labels == c)) for c in range(self.num_classes)
            },
            "unknown_label_count": int(np.sum(final_labels == UNKNOWN_LABEL)),
            "protocol": "interactive_extra_supervision" if prompts else "automatic_only",
            "supervision_type": "additional_supervision" if prompts else "automatic_inference",
            "simulated": False,
            "human_annotation_claimed": False,
            "human_annotation_time_sec": None,
        }

        return CorrectionResult(
            probabilities=refined_p,
            labels=final_labels,
            provenance=provenance,
            summary=counts,
            prompts=list(prompts),
            affine=affine,
            patient_id=patient_id,
            synthetic_fixture=synthetic_fixture,
        )


def authorize_simulation(
    dataset_role: str | None,
    *,
    synthetic_fixture: bool = False,
    label_budget: int | None = None,
    patient_id: str | None = None,
    train_manifest: str | Path | None = None,
    train_manifest_sha256: str | None = None,
    frame_index: int | None = None,
    mask_path: str | Path | None = None,
    image_path: str | Path | None = None,
) -> dict[str, Any]:
    """Authorize click simulation strictly under TRAIN-L budget or synthetic fixture.

    CRITICAL SEMI-SUPERVISED BOUNDARY:
    - Never read, hash, or simulate clicks from U, VAL, or TEST masks.
    - TRAIN-U, TESTING, TYPO must reject via an allowlist, NOT a blacklist.
    - Default 'TRAIN' alone is not proof of L (semi-supervised TRAIN has both L and U).
    - Requires explicit role allowlist and, for real data, explicit label budget, patient ID,
      and authoritative membership in --train-manifest BEFORE touching the GT path.
    - Known synthetic test fixtures must declare synthetic_fixture=True.
    """
    if not dataset_role:
        raise ValueError(
            "dataset_role must be specified for click simulation. "
            "Must be 'TRAIN-L' (with explicit label_budget, patient_id, and train_manifest) "
            "or 'FIXTURE' with synthetic_fixture=True."
        )

    role = str(dataset_role).strip().upper()

    # Immediate rejection of unauthorized/forbidden roles
    if role in ("TRAIN-U", "U", "VAL", "TEST", "TESTING", "TYPO"):
        raise ValueError(
            f"Click simulation forbidden on dataset_role='{dataset_role}'! "
            "Semi-supervised contract strictly prohibits U/VAL/TEST mask access."
        )

    if role in ("TRAIN",):
        raise ValueError(
            f"Click simulation forbidden on unproven dataset_role='{dataset_role}'. "
            "Default 'TRAIN' alone is not proof of TRAIN-L budget. "
            "Must specify dataset_role='TRAIN-L' with explicit label_budget, patient_id, and train_manifest, "
            "or synthetic_fixture=True for test fixtures."
        )

    if synthetic_fixture:
        allowed = {"FIXTURE", "SYNTHETIC", "TRAIN-L", "L"}
        if role not in allowed:
            raise ValueError(
                f"Click simulation forbidden for role '{dataset_role}'. "
                f"Synthetic fixture requires role in {sorted(allowed)}."
            )
        return {"authorized": True, "mode": "synthetic_fixture", "role": role}

    # Real data simulation: STRICT ALLOWLIST
    allowed = {"TRAIN-L", "L"}
    if role not in allowed:
        raise ValueError(
            f"Click simulation forbidden on dataset_role='{dataset_role}'! "
            "Semi-supervised contract strictly prohibits U/VAL/TEST mask access. "
            f"Role must be in {sorted(allowed)} with explicit label_budget, or synthetic_fixture=True."
        )

    if label_budget is None or int(label_budget) <= 0:
        raise ValueError(
            "Explicit positive label_budget (e.g. 8) is required for TRAIN-L click simulation."
        )

    if patient_id is None or str(patient_id).strip() == "":
        raise ValueError(
            "Explicit patient_id is required for TRAIN-L click simulation."
        )

    if train_manifest is None:
        raise ValueError(
            "Real TRAIN-L simulation requires authoritative --train-manifest binding. "
            "User role declaration alone is not permitted."
        )

    # Validate the hash-pinned manifest before looking at any GT path.  The
    # real click path is deliberately fail-closed: a caller cannot provide a
    # role/budget/patient declaration and then cause a foreign mask to be
    # stat'ed, hashed, or opened while that declaration is being checked.
    p_manifest = Path(train_manifest).expanduser().absolute()
    if not p_manifest.is_file():
        raise FileNotFoundError(f"Train manifest not found: {p_manifest}")

    if not isinstance(train_manifest_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", train_manifest_sha256.strip()
    ):
        raise ValueError(
            "Real TRAIN-L simulation requires a well-formed 64-character train_manifest_sha256."
        )
    actual_sha = sha256_file(p_manifest)
    if actual_sha != train_manifest_sha256.strip().lower():
        raise ValueError(
            f"Train manifest SHA-256 mismatch: expected {train_manifest_sha256}, got {actual_sha}"
        )

    try:
        manifest_data = json.loads(p_manifest.read_text())
    except Exception as e:
        raise ValueError(f"Failed to parse train manifest {p_manifest}: {e}") from e

    # Only the native v2 TRAIN manifest is authoritative.  Generic train /
    # val / test split manifests are intentionally unsupported: they cannot
    # prove the semi-supervised L/U boundary or charge labels to the budget.
    if not isinstance(manifest_data, dict):
        raise ValueError(f"Malformed train manifest {p_manifest.name}: expected a JSON object.")
    if manifest_data.get("schema") != "self_audit.data.v2":
        raise ValueError(
            f"Unsupported train manifest schema {manifest_data.get('schema')!r}; "
            "real click simulation requires self_audit.data.v2."
        )
    if manifest_data.get("role") != "training":
        raise ValueError(
            f"Unsupported train manifest role {manifest_data.get('role')!r}; role must be 'training'."
        )
    if type(manifest_data.get("synthetic")) is not bool or manifest_data.get("synthetic") is not False:
        raise ValueError("Real TRAIN-L simulation requires manifest synthetic=false.")

    labeled_records = manifest_data.get("labeled")
    unlabeled_records = manifest_data.get("unlabeled")
    evaluation_ids = manifest_data.get("evaluation_ids", [])
    if not isinstance(labeled_records, list) or not isinstance(unlabeled_records, list):
        raise ValueError(
            "Generic train splits are unsupported; self_audit.data.v2 requires labeled and unlabeled lists."
        )
    if not isinstance(evaluation_ids, list):
        raise ValueError("evaluation_ids must be a list in self_audit.data.v2 training manifests.")

    def _ids(records: Sequence[Any], label: str) -> list[str]:
        result: list[str] = []
        for item in records:
            if not isinstance(item, Mapping):
                raise ValueError(f"{label} records must be JSON objects.")
            ident = item.get("id")
            if not isinstance(ident, str) or not ident.strip():
                raise ValueError(f"{label} patient IDs must be non-empty strings.")
            result.append(ident)
        if len(set(result)) != len(result):
            raise ValueError(f"duplicate patient IDs in {label} records are unsupported.")
        return result

    labeled_ids = _ids(labeled_records, "labeled")
    unlabeled_ids = _ids(unlabeled_records, "unlabeled")
    if any(not isinstance(ident, str) or not ident.strip() for ident in evaluation_ids):
        raise ValueError("evaluation_ids must contain non-empty strings.")
    eval_ids = [str(ident) for ident in evaluation_ids]
    if len(set(eval_ids)) != len(eval_ids):
        raise ValueError("duplicate patient IDs in evaluation_ids are unsupported.")
    if set(labeled_ids) & set(unlabeled_ids):
        raise ValueError("labeled and unlabeled patient IDs must be disjoint.")
    if set(labeled_ids) & set(eval_ids) or set(unlabeled_ids) & set(eval_ids):
        raise ValueError("TRAIN patient IDs and evaluation_ids must be disjoint.")

    if not isinstance(patient_id, str) or not patient_id:
        raise ValueError("Explicit patient_id is required for TRAIN-L click simulation.")
    if patient_id not in set(labeled_ids):
        if patient_id in set(unlabeled_ids):
            raise ValueError(
                f"Patient '{patient_id}' is in UNLABELED records of train manifest {p_manifest.name}! "
                "Simulating clicks on unlabeled patients is strictly forbidden."
            )
        if patient_id in set(eval_ids):
            raise ValueError(
                f"Patient '{patient_id}' is in EVALUATION records of train manifest {p_manifest.name}! "
                "Simulating clicks on evaluation patients is strictly forbidden."
            )
        raise ValueError(
            f"Patient '{patient_id}' is not in authoritative TRAIN-L labeled records of {p_manifest.name}!"
        )

    manifest_budget = manifest_data.get("label_budget")
    if type(manifest_budget) is not int or manifest_budget <= 0:
        raise ValueError("training manifest label_budget must be an exact positive integer.")
    if type(label_budget) is not int or label_budget <= 0:
        raise ValueError("Explicit positive label_budget is required for TRAIN-L click simulation.")
    if label_budget != manifest_budget or manifest_budget != len(labeled_ids):
        raise ValueError(
            "label_budget must exactly equal both manifest label_budget and the labeled TRAIN patient count."
        )
    rec = next(item for item in labeled_records if item.get("id") == patient_id)

    # A real invocation must identify both the image source and the declared
    # frame before it can touch the GT payload.  These checks use path strings
    # only; a foreign path is rejected without stat/hash/open on that path.
    if frame_index is None or type(frame_index) is not int or frame_index < 0:
        raise ValueError("Real TRAIN-L simulation requires an explicit non-negative integer frame_index.")
    if mask_path is None:
        raise ValueError("Real TRAIN-L simulation requires the declared frame mask_path.")
    if image_path is None:
        raise ValueError("Real TRAIN-L simulation requires the declared patient image_path.")

    def _resolve_declared(value: Any, *, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty path string.")
        path = Path(value)
        return (p_manifest.parent / path).absolute() if not path.is_absolute() else path.absolute()

    declared_image_path = rec.get("image")
    declared_image_sha = rec.get("image_sha256")
    if not isinstance(declared_image_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", declared_image_sha):
        raise ValueError("labeled patient image_sha256 must be a well-formed 64-character SHA-256.")
    image_expected = _resolve_declared(declared_image_path, field="labeled image path")
    image_actual = Path(image_path).absolute()
    if image_actual != image_expected:
        raise ValueError(
            f"Image path mismatch for patient '{patient_id}': {image_actual} != manifest {image_expected}"
        )
    image_sha = sha256_file(image_actual)
    if image_sha != declared_image_sha.lower():
        raise ValueError(
            f"Image SHA-256 mismatch for patient '{patient_id}': expected {declared_image_sha}, got {image_sha}"
        )

    masks = rec.get("masks")
    if not isinstance(masks, Mapping) or not masks:
        raise ValueError(f"labeled patient '{patient_id}' requires a non-empty masks mapping.")
    frame_key = str(frame_index)
    if frame_key not in masks:
        raise ValueError(
            f"Patient '{patient_id}' has no declared mask for frame {frame_index}; GT path was not inspected."
        )
    expected_mask = masks[frame_key]
    if not isinstance(expected_mask, Mapping):
        raise ValueError("declared frame mask must contain path and sha256 fields.")
    declared_mask_path = expected_mask.get("path")
    declared_mask_sha = expected_mask.get("sha256")
    if not isinstance(declared_mask_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", declared_mask_sha):
        raise ValueError("declared frame mask sha256 must be a well-formed 64-character SHA-256.")
    mask_expected = _resolve_declared(declared_mask_path, field="declared frame mask path")
    mask_actual = Path(mask_path).absolute()
    if mask_actual != mask_expected:
        raise ValueError(
            f"Mask path mismatch for patient '{patient_id}' frame {frame_index}: "
            f"{mask_actual} != manifest {mask_expected}"
        )
    mask_sha = sha256_file(mask_actual)
    if mask_sha != declared_mask_sha.lower():
        raise ValueError(
            f"Mask SHA-256 mismatch for patient '{patient_id}' frame {frame_index}: "
            f"expected {declared_mask_sha}, got {mask_sha}"
        )

    return {
        "authorized": True,
        "mode": "real_train_l",
        "role": role,
        "patient_id": patient_id,
        "manifest": str(p_manifest),
        "manifest_sha256": actual_sha,
        "frame_index": frame_index,
        "image_path": str(image_expected),
        "image_sha256": image_sha,
        "mask_path": str(mask_expected),
        "mask_sha256": mask_sha,
        "label_budget": manifest_budget,
    }


def simulate_clicks_from_ground_truth(
    ground_truth_labels: np.ndarray,
    predicted_labels_or_probs: np.ndarray,
    *,
    dataset_role: str | None = None,
    label_budget: int | None = None,
    patient_id: str | None = None,
    synthetic_fixture: bool = False,
    train_manifest: str | Path | None = None,
    train_manifest_sha256: str | None = None,
    frame_index: int | None = None,
    mask_path: str | Path | None = None,
    image_path: str | Path | None = None,
    max_clicks: int = 5,
    strategy: str = "error_component_center",
    valid_mask: np.ndarray | None = None,
    affine: np.ndarray | None = None,
    seed: int = 42,
) -> list[PointPrompt]:
    """Simulate user point prompts from ground truth errors strictly under TRAIN-L budget.

    CRITICAL SEMI-SUPERVISED BOUNDARY:
    Simulating clicks from U, VAL, or TEST sets is an explicit protocol violation.
    Real calls require the hash-pinned TRAIN-L manifest, image/mask source paths,
    frame, and payload hashes; this function verifies that binding before it
    consumes the caller-provided GT array and immediately fails closed.
    """
    authorize_simulation(
        dataset_role,
        synthetic_fixture=synthetic_fixture,
        label_budget=label_budget,
        patient_id=patient_id,
        train_manifest=train_manifest,
        train_manifest_sha256=train_manifest_sha256,
        frame_index=frame_index,
        mask_path=mask_path,
        image_path=image_path,
    )

    gt = np.asarray(ground_truth_labels)
    if gt.ndim != 3:
        raise ValueError(f"ground_truth_labels must be 3D (Z, Y, X), got {gt.shape}")
    zdim, h, w = gt.shape

    if predicted_labels_or_probs.ndim == 4:
        pred = np.argmax(predicted_labels_or_probs, axis=0).astype(np.uint8)
    elif predicted_labels_or_probs.ndim == 3:
        pred = np.asarray(predicted_labels_or_probs, dtype=np.uint8)
    else:
        raise ValueError("predicted_labels_or_probs must be 3D labels or 4D probabilities")

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    rng = np.random.default_rng(seed)

    # Eligible error voxels: where prediction disagrees with GT and GT is not UNKNOWN (255)
    error_mask = (gt != pred) & (gt != UNKNOWN_LABEL)
    if valid_mask is not None:
        error_mask &= valid_mask

    prompts: list[PointPrompt] = []
    current_pred = pred.copy()

    for click_idx in range(max_clicks):
        cur_errors = (gt != current_pred) & (gt != UNKNOWN_LABEL)
        if valid_mask is not None:
            cur_errors &= valid_mask

        if not np.any(cur_errors):
            break

        if strategy == "error_component_center":
            # Find largest connected error component across slices
            labeled_comp, num_features = ndimage.label(cur_errors)
            if num_features == 0:
                break
            sizes = ndimage.sum(cur_errors, labeled_comp, range(1, num_features + 1))
            largest_comp_id = int(np.argmax(sizes)) + 1
            comp_mask = (labeled_comp == largest_comp_id)

            # Find the slice with the maximum error voxels in this component
            slice_counts = np.sum(comp_mask, axis=(1, 2))
            best_z = int(np.argmax(slice_counts))
            slice_mask = comp_mask[best_z]

            # Find center in 2D on that slice via distance transform
            dist_2d = ndimage.distance_transform_edt(slice_mask)
            iy, ix = np.unravel_index(np.argmax(dist_2d), dist_2d.shape)
            iz = best_z
        elif strategy == "random_error":
            err_indices = np.argwhere(cur_errors)
            pick = rng.choice(len(err_indices))
            iz, iy, ix = err_indices[pick]
        else:
            raise ValueError(f"Unknown click simulation strategy: {strategy}")

        iz, iy, ix = int(iz), int(iy), int(ix)
        target_c = int(gt[iz, iy, ix])

        # If affine is available, also produce world_mm representation
        if affine is not None:
            world_xyz = voxel_to_world(affine, [ix, iy, iz])
            prompt = PointPrompt(
                coord=(float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])),
                coord_system="world_mm",
                target_class=target_c,
                prompt_type="positive" if target_c > 0 else "negative",
                slice_index=iz,
                prompt_id=f"sim_click_{click_idx:03d}",
            )
        else:
            prompt = PointPrompt(
                coord=(float(ix), float(iy), float(iz)),
                coord_system="native_voxel_xyz",
                target_class=target_c,
                prompt_type="positive" if target_c > 0 else "negative",
                slice_index=iz,
                prompt_id=f"sim_click_{click_idx:03d}",
            )

        prompts.append(prompt)
        # Update current_pred at that location to simulate sequential click selection
        current_pred[iz, iy, ix] = target_c

    return prompts


def refine_volume_probabilities(
    image_zyx: np.ndarray,
    prompts: Sequence[PointPrompt],
    *,
    num_classes: int = 4,
    propagation_mode: str = "geodesic_3d",
    sigma_mm: float = 6.0,
    gamma_intensity: float = 8.0,
    prompt_weight: float = 0.90,
    max_distance_mm: float | None = 60.0,
    min_affinity: float = 0.05,
    prior_probabilities: np.ndarray | None = None,
    prior_labels: np.ndarray | None = None,
    prior_provenance: np.ndarray | None = None,
    affine: np.ndarray | None = None,
    spacing_xyz_mm: tuple[float, float, float] | None = None,
    geometric_valid: np.ndarray | None = None,
    patient_id: str | None = None,
    prior_is_pseudo: bool = False,
    synthetic_fixture: bool = False,
) -> CorrectionResult:
    """Convenience functional wrapper around VolumeCorrector.correct."""
    corrector = VolumeCorrector(
        num_classes=num_classes,
        propagation_mode=propagation_mode,
        sigma_mm=sigma_mm,
        gamma_intensity=gamma_intensity,
        prompt_weight=prompt_weight,
        max_distance_mm=max_distance_mm,
        min_affinity=min_affinity,
    )
    return corrector.correct(
        image_zyx,
        prompts,
        prior_probabilities=prior_probabilities,
        prior_labels=prior_labels,
        prior_provenance=prior_provenance,
        affine=affine,
        spacing_xyz_mm=spacing_xyz_mm,
        geometric_valid=geometric_valid,
        patient_id=patient_id,
        prior_is_pseudo=prior_is_pseudo,
        synthetic_fixture=synthetic_fixture,
    )


@dataclass
class InteractiveProtocolResult:
    """Receipt for interactive correction protocol under explicit supervision budget."""

    correction_result: CorrectionResult
    click_budget: int
    clicks_used: int
    clicks: list[PointPrompt]
    patient_id: str | None
    dataset_role: str
    synthetic_fixture: bool
    rounds_executed: int
    metadata: dict[str, Any]


def interactive_correction_protocol(
    image_zyx: np.ndarray,
    *,
    click_budget: int = 5,
    dataset_role: str = "TRAIN-L",
    label_budget: int | None = 8,
    patient_id: str | None = None,
    synthetic_fixture: bool = False,
    train_manifest: str | Path | None = None,
    train_manifest_sha256: str | None = None,
    ground_truth_labels: np.ndarray | None = None,
    provided_prompts: Sequence[PointPrompt] | None = None,
    prior_probabilities: np.ndarray | None = None,
    prior_labels: np.ndarray | None = None,
    prior_provenance: np.ndarray | None = None,
    prior_is_pseudo: bool = False,
    num_classes: int = 4,
    propagation_mode: str = "image_guided",
    sigma_mm: float = 6.0,
    gamma_intensity: float = 8.0,
    prompt_weight: float = 0.90,
    sim_strategy: str = "error_component_center",
    affine: np.ndarray | None = None,
    spacing_xyz_mm: tuple[float, float, float] | None = None,
    geometric_valid: np.ndarray | None = None,
    seed: int = 42,
) -> InteractiveProtocolResult:
    """Execute an interactive correction protocol under an explicit declared supervision budget.

    CRITICAL CONTRACT (I01):
    - Additional supervision protocol, NOT automatic inference.
    - Clicks are strictly bounded by click_budget.
    - Simulated clicks require authorization under TRAIN-L budget or synthetic fixture.
    - No U/VAL/TEST mask access allowed.
    - Manual edits are strictly preserved; provenance is tracked.
    """
    if click_budget < 0:
        raise ValueError(f"click_budget must be >= 0, got {click_budget}")

    prompts: list[PointPrompt] = []
    if provided_prompts:
        prompts.extend(provided_prompts[:click_budget])

    remaining_budget = click_budget - len(prompts)
    if ground_truth_labels is not None and remaining_budget > 0:
        authorize_simulation(
            dataset_role,
            synthetic_fixture=synthetic_fixture,
            label_budget=label_budget,
            patient_id=patient_id,
            train_manifest=train_manifest,
            train_manifest_sha256=train_manifest_sha256,
        )
        if prior_probabilities is not None:
            base_pred = prior_probabilities
        elif prior_labels is not None:
            base_pred = prior_labels
        else:
            base_pred = np.zeros_like(ground_truth_labels, dtype=np.uint8)

        sim_clicks = simulate_clicks_from_ground_truth(
            ground_truth_labels,
            base_pred,
            dataset_role=dataset_role,
            label_budget=label_budget,
            patient_id=patient_id,
            synthetic_fixture=synthetic_fixture,
            train_manifest=train_manifest,
            train_manifest_sha256=train_manifest_sha256,
            max_clicks=remaining_budget,
            strategy=sim_strategy,
            valid_mask=geometric_valid,
            affine=affine,
            seed=seed,
        )
        prompts.extend(sim_clicks)

    corrector = VolumeCorrector(
        num_classes=num_classes,
        propagation_mode=propagation_mode,
        sigma_mm=sigma_mm,
        gamma_intensity=gamma_intensity,
        prompt_weight=prompt_weight,
    )

    res = corrector.correct(
        image_zyx,
        prompts,
        prior_probabilities=prior_probabilities,
        prior_labels=prior_labels,
        prior_provenance=prior_provenance,
        affine=affine,
        spacing_xyz_mm=spacing_xyz_mm,
        geometric_valid=geometric_valid,
        patient_id=patient_id,
        prior_is_pseudo=prior_is_pseudo,
        synthetic_fixture=synthetic_fixture,
    )

    meta = {
        "protocol": "I01_interactive",
        "supervision_type": "additional_supervision",
        "simulated": ground_truth_labels is not None,
        "human_annotation_claimed": False,
        "human_annotation_time_sec": None,
        "click_budget": click_budget,
        "clicks_used": len(prompts),
        "dataset_role": dataset_role,
        "synthetic_fixture": synthetic_fixture,
        "patient_id": patient_id,
        "label_budget": label_budget,
        "summary": res.summary,
    }

    return InteractiveProtocolResult(
        correction_result=res,
        click_budget=click_budget,
        clicks_used=len(prompts),
        clicks=prompts,
        patient_id=patient_id,
        dataset_role=dataset_role,
        synthetic_fixture=synthetic_fixture,
        rounds_executed=1 if prompts else 0,
        metadata=meta,
    )


class InteractiveCorrectionProtocol:
    """Callable protocol object for interactive correction under explicit budget."""

    def __init__(
        self,
        *,
        click_budget: int = 5,
        dataset_role: str = "TRAIN-L",
        label_budget: int | None = 8,
        synthetic_fixture: bool = False,
        train_manifest: str | Path | None = None,
        train_manifest_sha256: str | None = None,
        num_classes: int = 4,
        propagation_mode: str = "image_guided",
        sigma_mm: float = 6.0,
        gamma_intensity: float = 8.0,
        prompt_weight: float = 0.90,
        sim_strategy: str = "error_component_center",
    ) -> None:
        self.click_budget = click_budget
        self.dataset_role = dataset_role
        self.label_budget = label_budget
        self.synthetic_fixture = synthetic_fixture
        self.train_manifest = train_manifest
        self.train_manifest_sha256 = train_manifest_sha256
        self.num_classes = num_classes
        self.propagation_mode = propagation_mode
        self.sigma_mm = sigma_mm
        self.gamma_intensity = gamma_intensity
        self.prompt_weight = prompt_weight
        self.sim_strategy = sim_strategy

    def __call__(
        self,
        image_zyx: np.ndarray,
        *,
        patient_id: str | None = None,
        ground_truth_labels: np.ndarray | None = None,
        provided_prompts: Sequence[PointPrompt] | None = None,
        prior_probabilities: np.ndarray | None = None,
        prior_labels: np.ndarray | None = None,
        prior_provenance: np.ndarray | None = None,
        prior_is_pseudo: bool = False,
        affine: np.ndarray | None = None,
        spacing_xyz_mm: tuple[float, float, float] | None = None,
        geometric_valid: np.ndarray | None = None,
        seed: int = 42,
    ) -> InteractiveProtocolResult:
        return interactive_correction_protocol(
            image_zyx,
            click_budget=self.click_budget,
            dataset_role=self.dataset_role,
            label_budget=self.label_budget,
            patient_id=patient_id,
            synthetic_fixture=self.synthetic_fixture,
            train_manifest=self.train_manifest,
            train_manifest_sha256=self.train_manifest_sha256,
            ground_truth_labels=ground_truth_labels,
            provided_prompts=provided_prompts,
            prior_probabilities=prior_probabilities,
            prior_labels=prior_labels,
            prior_provenance=prior_provenance,
            prior_is_pseudo=prior_is_pseudo,
            num_classes=self.num_classes,
            propagation_mode=self.propagation_mode,
            sigma_mm=self.sigma_mm,
            gamma_intensity=self.gamma_intensity,
            prompt_weight=self.prompt_weight,
            sim_strategy=self.sim_strategy,
            affine=affine,
            spacing_xyz_mm=spacing_xyz_mm,
            geometric_valid=geometric_valid,
            seed=seed,
        )


__all__ = [
    "SCHEMA",
    "PROVENANCE_ORIGINAL",
    "PROVENANCE_MANUAL",
    "PROVENANCE_PROPAGATED",
    "PROVENANCE_PSEUDO",
    "PROVENANCE_UNKNOWN",
    "PROVENANCE_NAMES",
    "PointPrompt",
    "CorrectionResult",
    "VolumeCorrector",
    "refine_volume_probabilities",
    "resolve_prompt_to_voxel_zyx",
    "save_prompts",
    "load_prompts",
    "compute_geodesic_distance_map_3d",
    "compute_geodesic_distance_map_2d",
    "compute_image_guided_affinity",
    "authorize_simulation",
    "simulate_clicks_from_ground_truth",
    "InteractiveProtocolResult",
    "interactive_correction_protocol",
    "InteractiveCorrectionProtocol",
    "validate_geometry",
]
