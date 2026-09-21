"""Deep statistical and geometric analysis of ACDC raw image headers and metadata.

RULES:
- Reads IMAGE headers and Info.cfg ONLY.
- NEVER reads or opens any *_gt.nii manual segmentation files.
"""

from pathlib import Path
import json
import nibabel as nib
import numpy as np

DATA_DIR = Path("/tmp/astra_event_acdc_training/training")
OUTPUT_JSON = Path("reports/astra_v3/workers/E_evidence/acdc_deep_stats.json")

def analyze():
    patients = sorted([p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("patient")])
    
    nb_frames = []
    ed_indices = []
    es_indices = []
    gaps = []
    
    shapes_xyz = []
    spacings_xyz = []
    orientations = set()
    qform_codes = set()
    sform_codes = set()
    z_spacings = []
    xy_spacings = []
    total_slices_ed = 0
    total_slices_es = 0
    
    patient_details = {}

    for p_dir in patients:
        pid = p_dir.name
        info_path = p_dir / "Info.cfg"
        info = {}
        with open(info_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    k, v = k.strip(), v.strip()
                    try:
                        info[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        info[k] = v
        
        nf = int(info["NbFrame"])
        ed = int(info["ED"])
        es = int(info["ES"])
        nb_frames.append(nf)
        ed_indices.append(ed)
        es_indices.append(es)
        gaps.append(es - ed)
        
        # Load ED and ES image headers
        ed_file = p_dir / f"{pid}_frame{ed:02d}.nii"
        es_file = p_dir / f"{pid}_frame{es:02d}.nii"
        
        assert ed_file.exists(), f"Missing {ed_file}"
        assert es_file.exists(), f"Missing {es_file}"
        
        ed_img = nib.load(str(ed_file))
        es_img = nib.load(str(es_file))
        
        ed_hdr = ed_img.header
        es_hdr = es_img.header
        
        ed_shape = list(ed_img.shape)
        es_shape = list(es_img.shape)
        assert ed_shape == es_shape, f"ED/ES shape mismatch in {pid}: {ed_shape} vs {es_shape}"
        
        shapes_xyz.append(ed_shape)
        total_slices_ed += ed_shape[2]
        total_slices_es += es_shape[2]
        
        ed_zooms = [float(z) for z in ed_hdr.get_zooms()[:3]]
        es_zooms = [float(z) for z in es_hdr.get_zooms()[:3]]
        spacings_xyz.append(ed_zooms)
        xy_spacings.extend([ed_zooms[0], ed_zooms[1]])
        z_spacings.append(ed_zooms[2])
        
        ed_orient = "".join(nib.aff2axcodes(ed_img.affine))
        orientations.add(ed_orient)
        
        qform_codes.add(int(ed_hdr["qform_code"]))
        sform_codes.add(int(ed_hdr["sform_code"]))
        
        patient_details[pid] = {
            "group": info.get("Group"),
            "nb_frames": nf,
            "ed_frame": ed,
            "es_frame": es,
            "gap_es_ed": es - ed,
            "shape_xyz": ed_shape,
            "spacing_xyz": ed_zooms,
            "num_z_slices": ed_shape[2],
            "affine": ed_img.affine.tolist(),
        }

    stats = {
        "n_patients": len(patients),
        "total_ed_slices": total_slices_ed,
        "total_es_slices": total_slices_es,
        "total_slices": total_slices_ed + total_slices_es,
        "temporal": {
            "nb_frames_min": int(np.min(nb_frames)),
            "nb_frames_max": int(np.max(nb_frames)),
            "nb_frames_mean": float(np.mean(nb_frames)),
            "nb_frames_median": float(np.median(nb_frames)),
            "ed_min": int(np.min(ed_indices)),
            "ed_max": int(np.max(ed_indices)),
            "es_min": int(np.min(es_indices)),
            "es_max": int(np.max(es_indices)),
            "gap_es_ed_min": int(np.min(gaps)),
            "gap_es_ed_max": int(np.max(gaps)),
            "gap_es_ed_mean": float(np.mean(gaps)),
            "gap_es_ed_median": float(np.median(gaps)),
        },
        "spatial": {
            "x_dim_min": int(min(s[0] for s in shapes_xyz)),
            "x_dim_max": int(max(s[0] for s in shapes_xyz)),
            "y_dim_min": int(min(s[1] for s in shapes_xyz)),
            "y_dim_max": int(max(s[1] for s in shapes_xyz)),
            "z_dim_min": int(min(s[2] for s in shapes_xyz)),
            "z_dim_max": int(max(s[2] for s in shapes_xyz)),
            "z_dim_mean": float(np.mean([s[2] for s in shapes_xyz])),
            "xy_spacing_min_mm": float(np.min(xy_spacings)),
            "xy_spacing_max_mm": float(np.max(xy_spacings)),
            "xy_spacing_mean_mm": float(np.mean(xy_spacings)),
            "z_spacing_min_mm": float(np.min(z_spacings)),
            "z_spacing_max_mm": float(np.max(z_spacings)),
            "z_spacing_mean_mm": float(np.mean(z_spacings)),
            "orientations": list(orientations),
            "qform_codes": list(qform_codes),
            "sform_codes": list(sform_codes),
        }
    }
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    print("Stats written to", OUTPUT_JSON)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    analyze()
