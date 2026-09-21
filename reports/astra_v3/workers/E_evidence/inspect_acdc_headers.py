"""Header-only inspection of ACDC training images.

RULES:
- Reads IMAGE headers and Info.cfg ONLY.
- NEVER reads or opens any *_gt.nii manual segmentation files.
"""

from pathlib import Path
import json
import nibabel as nib
import numpy as np

DATA_DIR = Path("/tmp/astra_event_acdc_training/training")
OUTPUT_JSON = Path("reports/astra_v3/workers/E_evidence/acdc_header_inventory.json")

def parse_info_cfg(path: Path) -> dict:
    info = {}
    if not path.exists():
        return info
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                try:
                    if "." in v:
                        info[k] = float(v)
                    else:
                        info[k] = int(v)
                except ValueError:
                    info[k] = v
    return info

def inspect_acdc():
    patients = sorted([p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("patient")])
    print(f"Discovered {len(patients)} patient directories in {DATA_DIR}")

    inventory = {
        "dataset": "ACDC",
        "total_patients": len(patients),
        "total_image_volumes": 0,
        "patients": {},
        "summary": {
            "groups": {},
            "shapes": {},
            "spacings": {},
            "orientations": {},
            "nb_frames_distribution": {},
            "has_4d_cine_file": 0,
            "has_ed_frame": 0,
            "has_es_frame": 0,
            "has_other_frames": 0,
        }
    }

    for p_dir in patients:
        pid = p_dir.name
        info = parse_info_cfg(p_dir / "Info.cfg")
        group = info.get("Group", "UNKNOWN")
        nb_frames = info.get("NbFrame", None)
        ed_frame = info.get("ED", None)
        es_frame = info.get("ES", None)

        if group not in inventory["summary"]["groups"]:
            inventory["summary"]["groups"][group] = 0
        inventory["summary"]["groups"][group] += 1

        nb_key = str(nb_frames)
        inventory["summary"]["nb_frames_distribution"][nb_key] = (
            inventory["summary"]["nb_frames_distribution"].get(nb_key, 0) + 1
        )

        # Look for image files (EXCLUDING *_gt.nii)
        img_files = sorted([
            f for f in p_dir.glob("*.nii*")
            if not f.name.endswith("_gt.nii") and not f.name.endswith("_gt.nii.gz")
        ])

        p_record = {
            "info": info,
            "files": {},
        }

        for img_p in img_files:
            inventory["total_image_volumes"] += 1
            if "_4d" in img_p.name:
                inventory["summary"]["has_4d_cine_file"] += 1

            # Check if this matches ED, ES, or other
            is_ed = False
            is_es = False
            if ed_frame is not None and f"frame{ed_frame:02d}" in img_p.name:
                is_ed = True
                inventory["summary"]["has_ed_frame"] += 1
            elif es_frame is not None and f"frame{es_frame:02d}" in img_p.name:
                is_es = True
                inventory["summary"]["has_es_frame"] += 1
            else:
                inventory["summary"]["has_other_frames"] += 1

            # Header inspection ONLY
            img = nib.load(str(img_p))
            header = img.header
            affine = img.affine
            shape = list(img.shape)
            zooms = [float(z) for z in header.get_zooms()]
            orientation = "".join(nib.aff2axcodes(affine))
            xyzt_units = header.get_xyzt_units()

            shape_key = str(shape)
            inventory["summary"]["shapes"][shape_key] = inventory["summary"]["shapes"].get(shape_key, 0) + 1
            inventory["summary"]["orientations"][orientation] = inventory["summary"]["orientations"].get(orientation, 0) + 1

            # Spacing rounding for summary binning
            spacing_rounded = str([round(z, 2) for z in zooms[:3]])
            inventory["summary"]["spacings"][spacing_rounded] = inventory["summary"]["spacings"].get(spacing_rounded, 0) + 1

            p_record["files"][img_p.name] = {
                "shape": shape,
                "zooms": zooms,
                "orientation": orientation,
                "xyzt_units": list(xyzt_units),
                "is_ed": is_ed,
                "is_es": is_es,
                "data_dtype": str(header.get_data_dtype()),
                "affine": affine.tolist(),
            }

        inventory["patients"][pid] = p_record

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    print(f"Inventory written to {OUTPUT_JSON}")
    print("Summary:")
    print("  Total Patients:", inventory["total_patients"])
    print("  Total Image Volumes:", inventory["total_image_volumes"])
    print("  Groups:", inventory["summary"]["groups"])
    print("  4D Cine Files Found:", inventory["summary"]["has_4d_cine_file"])
    print("  ED Frames Found:", inventory["summary"]["has_ed_frame"])
    print("  ES Frames Found:", inventory["summary"]["has_es_frame"])
    print("  Other Frames Found:", inventory["summary"]["has_other_frames"])
    print("  Orientations:", inventory["summary"]["orientations"])
    print("  Distinct Shapes Count:", len(inventory["summary"]["shapes"]))

if __name__ == "__main__":
    inspect_acdc()
