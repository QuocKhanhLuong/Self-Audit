"""Verification of image-only loading and 2.5-D spatial triplet extraction.

RULES:
- Loads raw IMAGE volume ONLY (patient001_frame01.nii).
- NEVER opens *_gt.nii files.
- Verifies intensity statistics, coordinate affine, and slice extraction.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

IMAGE_PATH = Path("/tmp/astra_event_acdc_training/training/patient001/patient001_frame01.nii")

def test_image_loading():
    assert IMAGE_PATH.exists(), f"Missing {IMAGE_PATH}"
    
    img = nib.load(str(IMAGE_PATH))
    affine = img.affine
    header = img.header
    shape = img.shape
    zooms = header.get_zooms()
    
    print(f"Loaded {IMAGE_PATH.name}")
    print(f"Shape: {shape} (X, Y, Z)")
    print(f"Zooms: {zooms} mm")
    print(f"Orientation: {nib.aff2axcodes(affine)}")
    
    # Load voxel data (IMAGE ONLY)
    data = img.get_fdata().astype(np.float32)
    print(f"Intensity Range: [{data.min():.1f}, {data.max():.1f}], Mean: {data.mean():.1f}, Std: {data.std():.1f}")
    
    # Verify Short-Axis Slices
    # Shape is (X, Y, Z) = (216, 256, 10)
    # Depth axis is 2 (Z)
    nx, ny, nz = shape
    assert nz == 10, f"Expected 10 slices, got {nz}"
    
    # Extract 2.5D spatial triplet for slice z=5
    z = 5
    z_indices = [max(0, z - 1), z, min(nz - 1, z + 1)]
    # In nibabel, data[:, :, z] is [X, Y]. Transpose to [H, W] = [Y, X] or keep [X, Y].
    # Standard convention: [H, W] = [256, 216]
    triplet = np.stack([data[:, :, idx].T for idx in z_indices], axis=0) # [3, H, W]
    print(f"Extracted 2.5-D Spatial Triplet [3, H, W]: {triplet.shape}")
    
    # Z-score normalization with percentile clipping (0.5, 99.5)
    p_low, p_high = np.percentile(triplet, [0.5, 99.5])
    clipped = np.clip(triplet, p_low, p_high)
    norm = (clipped - clipped.mean()) / max(clipped.std(), 1e-6)
    print(f"Normalized Triplet Range: [{norm.min():.2f}, {norm.max():.2f}], Mean: {norm.mean():.4f}, Std: {norm.std():.4f}")
    
    # Resizing to network size 256x256
    tensor = torch.from_numpy(norm).unsqueeze(0) # [1, 3, H, W]
    resized = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
    print(f"Resized Tensor: {resized.shape}")
    
    # Test inverse transform / correspondence mapping
    # A pixel at (r_net, c_net) in resized space corresponds to:
    orig_h, orig_w = triplet.shape[1], triplet.shape[2]
    net_h, net_w = 256, 256
    scale_y = orig_h / net_h
    scale_x = orig_w / net_w
    
    test_r_net, test_c_net = 128, 128
    orig_r = test_r_net * scale_y
    orig_c = test_c_net * scale_x
    print(f"Pixel ({test_r_net}, {test_c_net}) maps back to native plane coords ({orig_r:.2f}, {orig_c:.2f})")
    
    # Native voxel index: (orig_c, orig_r, z) = (x_vox, y_vox, z_vox)
    vox_coord = np.array([orig_c, orig_r, z, 1.0])
    world_coord = affine @ vox_coord
    print(f"Native Voxel Coord: {vox_coord[:3]} -> Scanner World Coord (LPS mm): {world_coord[:3]}")

if __name__ == "__main__":
    test_image_loading()
