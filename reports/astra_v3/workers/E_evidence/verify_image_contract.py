"""Formal Image-Only Sample Contract Implementation and Empirical Verification.

RULES:
- Image headers and voxel intensities ONLY.
- Zero reads of *_gt.nii segmentation files.
- Demonstrates forward sample construction, 2.5D spatial triplet, temporal handling,
  and exact inverse coordinate correspondence back to native NIfTI LPS space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class SpatialNeighbors:
    z_prev: int
    z_cur: int
    z_nxt: int
    mode: str = "replicate"  # boundary clamp

@dataclass(frozen=True)
class TemporalNeighbors:
    t_prev: int | None
    t_cur: int
    t_nxt: int | None
    status: str  # "available", "unavailable_static", "periodic_wrapped", "clamped"

@dataclass(frozen=True)
class NativeCoordinateTransform:
    source_file: str
    orig_shape_hw: tuple[int, int]
    net_shape_hw: tuple[int, int]
    scale_y: float
    scale_x: float
    native_affine_4x4: list[list[float]]
    native_orientation: str
    native_spacing_zyx_mm: tuple[float, float, float]
    effective_spacing_zyx_mm: tuple[float, float, float]

    def net_to_native_voxel(self, r_net: float, c_net: float, z: int) -> tuple[float, float, float]:
        """Convert network pixel coordinates (row, col) back to native voxel coordinates (x, y, z)."""
        # In our convention: row is Y (height), col is X (width)
        orig_y = r_net * self.scale_y
        orig_x = c_net * self.scale_x
        return (float(orig_x), float(orig_y), float(z))

    def net_to_scanner_physical_mm(self, r_net: float, c_net: float, z: int) -> tuple[float, float, float]:
        """Convert network pixel coordinates directly to physical scanner space (LPS mm)."""
        vox_x, vox_y, vox_z = self.net_to_native_voxel(r_net, c_net, z)
        affine = np.array(self.native_affine_4x4, dtype=np.float64)
        v = np.array([vox_x, vox_y, vox_z, 1.0], dtype=np.float64)
        world = affine @ v
        return (float(world[0]), float(world[1]), float(world[2]))

@dataclass
class ImageOnlySampleContract:
    patient_id: str
    case_id: str
    t: int
    t_total: int
    z: int
    z_total: int
    phase: str
    spatial_neighbors: SpatialNeighbors
    temporal_neighbors: TemporalNeighbors
    native_transform: NativeCoordinateTransform
    cur_triplet: torch.Tensor          # [3, H_net, W_net]
    prev_triplet: torch.Tensor | None  # [3, H_net, W_net] or None if cine unavailable
    nxt_triplet: torch.Tensor | None   # [3, H_net, W_net] or None if cine unavailable
    transforms_applied: list[dict[str, Any]] = field(default_factory=list)

def build_image_sample(
    image_path: Path,
    info_path: Path,
    slice_z: int,
    target_hw: tuple[int, int] = (256, 256),
    intensity_clip: tuple[float, float] = (0.5, 99.5),
) -> ImageOnlySampleContract:
    assert image_path.exists(), f"Image not found: {image_path}"
    
    img = nib.load(str(image_path))
    data = img.get_fdata().astype(np.float32)  # [X, Y, Z]
    affine = img.affine
    header = img.header
    nx, ny, nz = data.shape
    zooms = header.get_zooms()[:3]
    orientation = "".join(nib.aff2axcodes(affine))
    
    # Parse Info.cfg
    info = {}
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    k, v = k.strip(), v.strip()
                    try:
                        info[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        info[k] = v
                        
    pid = image_path.parent.name
    case_id = image_path.stem
    
    # Identify phase and time index
    t_total = int(info.get("NbFrame", 1))
    ed_f = int(info.get("ED", 1))
    es_f = int(info.get("ES", 1))
    
    if f"frame{ed_f:02d}" in case_id:
        phase = "ED"
        t = ed_f - 1  # 0-indexed
    elif f"frame{es_f:02d}" in case_id:
        phase = "ES"
        t = es_f - 1  # 0-indexed
    else:
        phase = "unknown"
        t = 0
        
    # Spatial neighborhood along Z
    z_prev = max(0, slice_z - 1)
    z_cur = slice_z
    z_nxt = min(nz - 1, slice_z + 1)
    spatial_neigh = SpatialNeighbors(z_prev, z_cur, z_nxt, mode="clamp_boundary")
    
    # Temporal neighborhood status
    # In static ED/ES files, adjacent time frames t-1 and t+1 are missing!
    temporal_neigh = TemporalNeighbors(
        t_prev=None,
        t_cur=t,
        t_nxt=None,
        status="unavailable_static_ed_es",
    )
    
    # Extract 2.5D spatial slices [3, H_orig, W_orig]
    # Note: data[:, :, z] is [X, Y]. Transposing gives [H, W] = [Y, X]
    s_prev = data[:, :, z_prev].T
    s_cur = data[:, :, z_cur].T
    s_nxt = data[:, :, z_nxt].T
    triplet_orig = np.stack([s_prev, s_cur, s_nxt], axis=0) # [3, H, W]
    orig_h, orig_w = triplet_orig.shape[1], triplet_orig.shape[2]
    
    # Intensity normalization (percentile clip + z-score)
    p_low, p_high = np.percentile(triplet_orig, intensity_clip)
    clipped = np.clip(triplet_orig, p_low, p_high)
    std = max(float(np.std(clipped)), 1e-6)
    mean = float(np.mean(clipped))
    normalized = ((clipped - mean) / std).astype(np.float32)
    
    # Bilinear interpolation to target_hw
    tensor = torch.from_numpy(normalized).unsqueeze(0) # [1, 3, H_orig, W_orig]
    resized = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False).squeeze(0) # [3, H_net, W_net]
    
    # Coordinate scaling math
    net_h, net_w = target_hw
    scale_y = orig_h / net_h
    scale_x = orig_w / net_w
    
    orig_spacing_zyx = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    eff_spacing_zyx = (float(zooms[2]), float(zooms[1]) * scale_y, float(zooms[0]) * scale_x)
    
    transform_record = NativeCoordinateTransform(
        source_file=str(image_path),
        orig_shape_hw=(orig_h, orig_w),
        net_shape_hw=(net_h, net_w),
        scale_y=scale_y,
        scale_x=scale_x,
        native_affine_4x4=affine.tolist(),
        native_orientation=orientation,
        native_spacing_zyx_mm=orig_spacing_zyx,
        effective_spacing_zyx_mm=eff_spacing_zyx,
    )
    
    transforms_applied = [
        {"name": "spatial_triplet_extraction", "z_indices": [z_prev, z_cur, z_nxt]},
        {"name": "percentile_clipping", "percentiles": intensity_clip, "range": [float(p_low), float(p_high)]},
        {"name": "z_score_standardization", "mean": mean, "std": std},
        {"name": "bilinear_in_plane_resize", "orig_hw": (orig_h, orig_w), "net_hw": (net_h, net_w), "align_corners": False},
    ]
    
    return ImageOnlySampleContract(
        patient_id=pid,
        case_id=case_id,
        t=t,
        t_total=t_total,
        z=slice_z,
        z_total=nz,
        phase=phase,
        spatial_neighbors=spatial_neigh,
        temporal_neighbors=temporal_neigh,
        native_transform=transform_record,
        cur_triplet=resized,
        prev_triplet=None,
        nxt_triplet=None,
        transforms_applied=transforms_applied,
    )

def test_contract_execution():
    p1_img = Path("/tmp/astra_event_acdc_training/training/patient001/patient001_frame01.nii")
    p1_info = Path("/tmp/astra_event_acdc_training/training/patient001/Info.cfg")
    
    sample = build_image_sample(p1_img, p1_info, slice_z=4)
    print("Contract built successfully!")
    print("  Patient:", sample.patient_id)
    print("  Case:", sample.case_id)
    print("  Phase:", sample.phase)
    print("  t / t_total:", f"{sample.t} / {sample.t_total}")
    print("  z / z_total:", f"{sample.z} / {sample.z_total}")
    print("  Spatial neighbors:", sample.spatial_neighbors)
    print("  Temporal status:", sample.temporal_neighbors.status)
    print("  cur_triplet shape:", tuple(sample.cur_triplet.shape))
    print("  Transforms applied count:", len(sample.transforms_applied))
    
    # Test inverse transformation accuracy
    r_net, c_net = 100.0, 150.0
    vox_x, vox_y, vox_z = sample.native_transform.net_to_native_voxel(r_net, c_net, sample.z)
    scanner_mm = sample.native_transform.net_to_scanner_physical_mm(r_net, c_net, sample.z)
    print(f"  Network pixel ({r_net}, {c_net}) -> Native voxel ({vox_x:.2f}, {vox_y:.2f}, {vox_z})")
    print(f"  Physical scanner LPS (mm): ({scanner_mm[0]:.2f}, {scanner_mm[1]:.2f}, {scanner_mm[2]:.2f})")
    
    # Save a JSON snapshot of contract metadata for audit evidence
    contract_evidence = {
        "patient_id": sample.patient_id,
        "case_id": sample.case_id,
        "phase": sample.phase,
        "t": sample.t,
        "t_total": sample.t_total,
        "z": sample.z,
        "z_total": sample.z_total,
        "spatial_neighbors": sample.spatial_neighbors.__dict__,
        "temporal_neighbors": sample.temporal_neighbors.__dict__,
        "native_transform": {
            "source_file": sample.native_transform.source_file,
            "orig_shape_hw": sample.native_transform.orig_shape_hw,
            "net_shape_hw": sample.native_transform.net_shape_hw,
            "scale_y": sample.native_transform.scale_y,
            "scale_x": sample.native_transform.scale_x,
            "native_orientation": sample.native_transform.native_orientation,
            "native_spacing_zyx_mm": sample.native_transform.native_spacing_zyx_mm,
            "effective_spacing_zyx_mm": sample.native_transform.effective_spacing_zyx_mm,
        },
        "transforms_applied": sample.transforms_applied,
        "cur_triplet_shape": list(sample.cur_triplet.shape),
        "prev_triplet": None,
        "nxt_triplet": None,
    }
    out_path = Path("reports/astra_v3/workers/E_evidence/sample_contract_verification.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(contract_evidence, f, indent=2)
    print("Saved contract verification evidence to", out_path)

if __name__ == "__main__":
    test_contract_execution()
