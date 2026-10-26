#!/usr/bin/env python3
"""CLI for native-3D volume point correction and refinement.

Provides deterministic prompt-conditioned image and probability refinement
with strict hard manual preservation and separated provenance tracking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from self_audit.volume.correction import (
    PointPrompt,
    VolumeCorrector,
    authorize_simulation,
    load_prompts,
    save_prompts,
    simulate_clicks_from_ground_truth,
    validate_geometry,
)
from self_audit.volume.geometry import UNKNOWN_LABEL, load_sidecar, sha256_file, spacing_from_affine


def _load_volume(
    path: str | Path,
    key: str | None = None,
    synthetic_fixture: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an image or mask array from .npy, .npz, or .nii/.nii.gz with declared mm units."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    affine: np.ndarray | None = None
    if p.suffix in (".npy",):
        arr = np.load(p)
    elif p.suffix in (".npz",):
        data = np.load(p)
        k = key or list(data.keys())[0]
        arr = data[k]
        if "affine" in data:
            affine = data["affine"]
    elif ".nii" in p.name:
        import nibabel as nib

        nii = nib.load(str(p))
        raw_affine = np.asarray(nii.affine, dtype=float)
        # Check declared spatial units in header
        units = nii.header.get_xyzt_units()
        spatial_unit = units[0] if len(units) > 0 else "unknown"
        affine = raw_affine.copy()
        # NIfTI unit conversion to mm
        if spatial_unit in ("meter", "m", 1):
            affine[:3, :] = affine[:3, :] * 1000.0
        elif spatial_unit in ("micron", "um", 3):
            affine[:3, :] = affine[:3, :] * 0.001
        elif spatial_unit in ("mm", 2):
            pass
        elif spatial_unit in ("unknown", 0, "") or spatial_unit is None:
            if synthetic_fixture:
                pass
            else:
                raise ValueError(
                    f"NIfTI spatial units are unknown ({spatial_unit!r}) in {p.name}. "
                    "Real physical volume operations require declared spatial units (mm, meter, or micron) and reject unknown units."
                )
        else:
            raise ValueError(f"Unsupported NIfTI spatial units: {spatial_unit}")

        data = np.asarray(nii.get_fdata())
        # NIfTI is (X, Y, Z); transpose to (Z, Y, X)
        if data.ndim == 3:
            arr = np.transpose(data, (2, 1, 0))
        elif data.ndim == 4:
            # (X, Y, Z, C) -> (C, Z, Y, X)
            arr = np.transpose(data, (3, 2, 1, 0))
        else:
            arr = data
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    return arr, affine


def _precheck_simulation_role(
    dataset_role: str | None,
    *,
    synthetic_fixture: bool,
    label_budget: int | None,
    patient_id: str | None,
    train_manifest: str | Path | None,
) -> None:
    """Reject unauthorized simulation roles before touching any input paths.

    This deliberately performs only the cheap role/argument checks needed to
    preserve the pre-I/O forbidden-role gate.  Hash-pinned manifest binding is
    still delegated to :func:`authorize_simulation` after the image frame has
    been loaded and validated, so an out-of-range frame cannot cause prompt or
    ground-truth access.
    """
    if not dataset_role:
        raise ValueError(
            "dataset_role must be specified for click simulation. "
            "Must be 'TRAIN-L' (with explicit label_budget, patient_id, and train_manifest) "
            "or 'FIXTURE' with synthetic_fixture=True."
        )

    role = str(dataset_role).strip().upper()
    if role in {"TRAIN-U", "U", "VAL", "TEST", "TESTING", "TYPO"}:
        raise ValueError(
            f"Click simulation forbidden on dataset_role='{dataset_role}'! "
            "Semi-supervised contract strictly prohibits U/VAL/TEST mask access."
        )
    if role == "TRAIN":
        raise ValueError(
            f"Click simulation forbidden on unproven dataset_role='{dataset_role}'. "
            "Default 'TRAIN' alone is not proof of TRAIN-L budget. "
            "Must specify dataset_role='TRAIN-L' with explicit label_budget, patient_id, and train_manifest, "
            "or synthetic_fixture=True for test fixtures."
        )

    if synthetic_fixture:
        allowed_roles = {"FIXTURE", "SYNTHETIC", "TRAIN-L", "L"}
        if role not in allowed_roles:
            raise ValueError(
                f"Click simulation forbidden for role '{dataset_role}'. "
                f"Synthetic fixture requires role in {sorted(allowed_roles)}."
            )
        return

    allowed_roles = {"TRAIN-L", "L"}
    if role not in allowed_roles:
        raise ValueError(
            f"Click simulation forbidden on dataset_role='{dataset_role}'! "
            "Semi-supervised contract strictly prohibits U/VAL/TEST mask access. "
            f"Role must be in {sorted(allowed_roles)} with explicit label_budget, or synthetic_fixture=True."
        )
    if label_budget is None or int(label_budget) <= 0:
        raise ValueError(
            "Explicit positive label_budget (e.g. 8) is required for TRAIN-L click simulation."
        )
    if patient_id is None or str(patient_id).strip() == "":
        raise ValueError("Explicit patient_id is required for TRAIN-L click simulation.")
    if train_manifest is None:
        raise ValueError(
            "Real TRAIN-L simulation requires authoritative --train-manifest binding. "
            "User role declaration alone is not permitted."
        )


def _select_time_frame(
    volume: np.ndarray,
    time_index: int,
    *,
    source: str,
) -> tuple[np.ndarray, int]:
    """Select a declared frame without clamping or silently dropping indices."""
    if time_index < 0:
        raise ValueError(
            f"{source} --time-index must be non-negative (got {time_index})."
        )
    if volume.ndim == 4:
        frame_count = int(volume.shape[0])
        if time_index >= frame_count:
            raise ValueError(
                f"{source} --time-index {time_index} is out of range for "
                f"4D input with {frame_count} frames (valid range 0..{frame_count - 1})."
            )
        return volume[time_index], time_index
    if volume.ndim == 3:
        if time_index != 0:
            raise ValueError(
                f"{source} is 3D; --time-index must be 0 (got {time_index})."
            )
        return volume, 0
    raise ValueError(f"Expected 3D or 4D {source.lower()} array, got ndim={volume.ndim}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Native-3D volume point correction with hard manual preservation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to input 3D or 4D image volume (.npy, .npz, .nii.gz)")
    parser.add_argument("--geometry", default=None, help="Path to geometry sidecar JSON")
    parser.add_argument("--patient-id", default=None, help="Patient ID in geometry sidecar")
    parser.add_argument("--time-index", type=int, default=0, help="Temporal frame index if image is 4D (T, Z, Y, X)")
    parser.add_argument("--prior-probabilities", default=None, help="Path to prior probabilities (C, Z, Y, X)")
    parser.add_argument("--prior-labels", default=None, help="Path to prior discrete labels (Z, Y, X)")
    parser.add_argument("--prior-provenance", default=None, help="Path to prior provenance mask for iterative persistence (Z, Y, X)")
    parser.add_argument("--prompts", default=None, help="Path to JSON file with point prompts")
    parser.add_argument("--out-labels", default=None, help="Path to output corrected segmentation (.nii.gz, .npy, .npz)")
    parser.add_argument("--out-provenance", default=None, help="Path to output provenance mask (.nii.gz, .npy, .npz)")
    parser.add_argument("--out-probabilities", default=None, help="Path to output refined probabilities (.nii.gz, .npy, .npz)")
    parser.add_argument("--out-json", default=None, help="Path to output metadata JSON summary")
    parser.add_argument(
        "--mode",
        choices=["geodesic_3d", "geodesic_2d", "image_guided", "manual_only"],
        default="geodesic_3d",
        help="Prompt propagation algorithm",
    )
    parser.add_argument("--num-classes", type=int, default=4, help="Total number of semantic classes")
    parser.add_argument("--sigma-mm", type=float, default=6.0, help="Spatial propagation bandwidth in mm")
    parser.add_argument("--gamma-intensity", type=float, default=8.0, help="Intensity gradient sensitivity weight")
    parser.add_argument("--prompt-weight", type=float, default=0.90, help="Prompt blending strength (0-1)")
    parser.add_argument("--simulate-from-gt", default=None, help="Path to ground truth labels for click simulation (TRAIN-L ONLY)")
    parser.add_argument(
        "--dataset-role",
        default=None,
        help="Dataset role of the patient (Must be 'TRAIN-L' with explicit --label-budget, or 'FIXTURE' with --synthetic-fixture)",
    )
    parser.add_argument("--label-budget", type=int, default=None, help="Declared dense TRAIN-L patient budget (e.g. 8)")
    parser.add_argument(
        "--synthetic-fixture",
        "--allow-synthetic-fixture",
        action="store_true",
        default=False,
        help="Explicitly mark simulation as using a known synthetic test fixture",
    )
    parser.add_argument("--train-manifest", default=None, help="Path to authoritative TRAIN-L manifest JSON")
    parser.add_argument("--train-manifest-sha256", default=None, help="Expected SHA-256 hash of train manifest")
    parser.add_argument("--max-clicks", type=int, default=5, help="Maximum simulated clicks")
    parser.add_argument(
        "--sim-strategy",
        choices=["error_component_center", "random_error"],
        default="error_component_center",
        help="Click simulation strategy",
    )
    parser.add_argument("--out-prompts", default=None, help="Save generated/simulated prompts to JSON")

    args = parser.parse_args(argv)

    # Negative indices cannot be meaningful frame selectors.  Reject before
    # any role/manifest/image/prompt/ground-truth I/O.
    if args.time_index < 0:
        raise ValueError(
            f"--time-index must be non-negative (got {args.time_index})."
        )

    # 1. CRITICAL SEMI-SUPERVISED SECURITY GATE:
    # Preserve the existing cheap pre-I/O rejection for forbidden roles.  Full
    # manifest/path binding is performed below after the image frame shape and
    # requested frame have been validated.
    if args.simulate_from_gt is not None:
        _precheck_simulation_role(
            dataset_role=args.dataset_role,
            synthetic_fixture=args.synthetic_fixture,
            label_budget=args.label_budget,
            patient_id=args.patient_id,
            train_manifest=args.train_manifest,
        )

    # Load image volume and calculate image hash
    image_sha = sha256_file(args.image)
    img_data, img_affine = _load_volume(args.image, synthetic_fixture=args.synthetic_fixture)
    image_zyx, frame_index = _select_time_frame(
        img_data, args.time_index, source="Image"
    )

    # The image is now the authority for the temporal bound.  Do not touch a
    # prompt, GT, or authoritative mask binding until this check has passed.
    if args.simulate_from_gt is not None:
        authorize_simulation(
            dataset_role=args.dataset_role,
            synthetic_fixture=args.synthetic_fixture,
            label_budget=args.label_budget,
            patient_id=args.patient_id,
            train_manifest=args.train_manifest,
            train_manifest_sha256=args.train_manifest_sha256,
            frame_index=frame_index,
            mask_path=args.simulate_from_gt,
            image_path=args.image,
        )

    affine = img_affine
    spacing_xyz_mm = tuple(float(x) for x in spacing_from_affine(affine)) if affine is not None else None

    # Load and bind geometry sidecar if available
    if args.geometry is not None:
        sidecar = load_sidecar(args.geometry)
        patients_map = sidecar.get("patients", {})
        if args.patient_id is not None:
            pid = args.patient_id
        elif len(patients_map) == 1:
            pid = list(patients_map.keys())[0]
        else:
            raise ValueError(
                "Multiple patients found in geometry sidecar; explicit --patient-id is required."
            )

        if pid not in patients_map:
            raise ValueError(f"Patient {pid} not found in geometry sidecar {args.geometry}")

        rec = patients_map[pid]
        sc_affine = np.asarray(rec["affine_xyz_to_world_mm"], dtype=float)
        sc_spacing = tuple(float(x) for x in rec["spacing_xyz_mm"])
        sc_shape = tuple(int(x) for x in rec["native_shape_zyx"])

        # Validate correspondence
        if sc_shape != image_zyx.shape:
            raise ValueError(
                f"Sidecar native_shape_zyx {sc_shape} != image shape {image_zyx.shape}"
            )
        if rec.get("image_sha256") and rec["image_sha256"] != image_sha:
            raise ValueError(
                f"Sidecar image_sha256 mismatch for patient {pid}: {rec['image_sha256']} vs {image_sha}"
            )
        if img_affine is not None and not np.allclose(img_affine, sc_affine, atol=1e-3, rtol=1e-3):
            raise ValueError(
                f"Image affine does not match sidecar affine for patient {pid}: {img_affine} vs {sc_affine}"
            )

        affine = sc_affine
        spacing_xyz_mm = sc_spacing

    # Physical geometry requirement for real data: fail closed if missing, singular, or inconsistent
    affine, spacing_xyz_mm = validate_geometry(
        affine, spacing_xyz_mm, synthetic_fixture=args.synthetic_fixture
    )

    # Load prior probabilities or labels, and verify grid correspondence
    prior_probs = None
    if args.prior_probabilities is not None:
        p_arr, p_aff = _load_volume(args.prior_probabilities, synthetic_fixture=args.synthetic_fixture)
        if p_aff is not None and affine is not None and not np.allclose(p_aff, affine, atol=1e-3, rtol=1e-3):
            raise ValueError(f"Prior probabilities affine {p_aff} != image affine {affine}")
        prior_probs = p_arr

    prior_labels = None
    if args.prior_labels is not None:
        l_arr, l_aff = _load_volume(args.prior_labels, synthetic_fixture=args.synthetic_fixture)
        if l_aff is not None and affine is not None and not np.allclose(l_aff, affine, atol=1e-3, rtol=1e-3):
            raise ValueError(f"Prior labels affine {l_aff} != image affine {affine}")
        prior_labels = l_arr

    prior_prov = None
    if args.prior_provenance is not None:
        prov_arr, prov_aff = _load_volume(args.prior_provenance, synthetic_fixture=args.synthetic_fixture)
        if prov_aff is not None and affine is not None and not np.allclose(prov_aff, affine, atol=1e-3, rtol=1e-3):
            raise ValueError(f"Prior provenance affine {prov_aff} != image affine {affine}")
        prior_prov = prov_arr

    # Resolve prompts: from file or simulated from GT
    prompts: list[PointPrompt] = []
    if args.prompts is not None:
        prompts.extend(load_prompts(args.prompts))

    gt_sha: str | None = None
    classes_present: list[int] = []

    if args.simulate_from_gt is not None:
        # Now authorized to access mask
        gt_sha = sha256_file(args.simulate_from_gt)
        gt_data, gt_aff = _load_volume(args.simulate_from_gt, synthetic_fixture=args.synthetic_fixture)
        if gt_aff is not None and affine is not None and not np.allclose(gt_aff, affine, atol=1e-3, rtol=1e-3):
            raise ValueError(f"Ground truth affine {gt_aff} != image affine {affine}")

        # Ground-truth masks are consumed by the correction engine as one
        # native 3-D frame.  A 4-D mask follows the image's validated frame;
        # an already 3-D mask is that frame and must not be reinterpreted as a
        # temporal volume just because the image has another frame index.
        if gt_data.ndim == 4:
            gt_data, _ = _select_time_frame(
                gt_data, frame_index, source="Ground-truth"
            )
        elif gt_data.ndim != 3:
            raise ValueError(
                f"Expected 3D or 4D ground-truth array, got ndim={gt_data.ndim}"
            )

        if gt_data.shape != image_zyx.shape:
            raise ValueError(
                f"Shape mismatch: ground truth shape {gt_data.shape} != image shape {image_zyx.shape}"
            )

        unique_classes = np.unique(gt_data)
        for c_val in unique_classes:
            if c_val != UNKNOWN_LABEL:
                if int(c_val) >= args.num_classes or int(c_val) < 0:
                    raise ValueError(
                        f"Ground truth contains class {c_val} out of range [0, {args.num_classes - 1}]"
                    )
                classes_present.append(int(c_val))

        # Base prediction for simulation is prior_probs or prior_labels, or zeros
        base_pred = prior_probs if prior_probs is not None else prior_labels
        if base_pred is None:
            base_pred = np.zeros_like(gt_data, dtype=np.uint8)

        sim_prompts = simulate_clicks_from_ground_truth(
            gt_data,
            base_pred,
            dataset_role=str(args.dataset_role),
            label_budget=args.label_budget,
            patient_id=args.patient_id,
            synthetic_fixture=args.synthetic_fixture,
            train_manifest=args.train_manifest,
            train_manifest_sha256=args.train_manifest_sha256,
            frame_index=frame_index,
            mask_path=args.simulate_from_gt,
            image_path=args.image,
            max_clicks=args.max_clicks,
            strategy=args.sim_strategy,
            affine=affine,
        )
        prompts.extend(sim_prompts)

    if args.out_prompts is not None and prompts:
        save_prompts(prompts, args.out_prompts)

    # Execute correction
    corrector = VolumeCorrector(
        num_classes=args.num_classes,
        propagation_mode=args.mode,
        sigma_mm=args.sigma_mm,
        gamma_intensity=args.gamma_intensity,
        prompt_weight=args.prompt_weight,
    )

    result = corrector.correct(
        image_zyx,
        prompts,
        prior_probabilities=prior_probs,
        prior_labels=prior_labels,
        prior_provenance=prior_prov,
        affine=affine,
        spacing_xyz_mm=spacing_xyz_mm,
        patient_id=args.patient_id,
        synthetic_fixture=args.synthetic_fixture,
    )

    # Update summary with resolved hashes, manifest, and explicit non-human annotation accounting
    result.summary["image_sha256"] = image_sha
    result.summary["gt_sha256"] = gt_sha
    result.summary["train_manifest"] = str(args.train_manifest) if args.train_manifest else None
    result.summary["native_shape_zyx"] = list(image_zyx.shape)
    result.summary["affine"] = affine.tolist() if affine is not None else None
    result.summary["classes"] = sorted(set(classes_present))
    result.summary["patient_id"] = args.patient_id
    result.summary["label_budget"] = args.label_budget
    result.summary["dataset_role"] = args.dataset_role
    result.summary["synthetic_fixture"] = bool(args.synthetic_fixture)
    result.summary["protocol"] = (
        "simulated_additional_supervision"
        if args.simulate_from_gt
        else "interactive_extra_supervision"
        if prompts
        else "automatic_only"
    )
    result.summary["supervision_type"] = (
        "additional_supervision"
        if (args.simulate_from_gt or prompts)
        else "automatic_inference"
    )
    result.summary["simulated"] = bool(args.simulate_from_gt)
    result.summary["human_annotation_claimed"] = False
    result.summary["human_annotation_time_sec"] = None

    # Independent output exports honoring each requested path
    if args.out_labels is not None:
        p_lab = Path(args.out_labels)
        p_lab.parent.mkdir(parents=True, exist_ok=True)
        if p_lab.name.endswith(".npz"):
            np.savez_compressed(p_lab, labels=result.labels, affine=affine)
        elif p_lab.name.endswith(".npy"):
            np.save(p_lab, result.labels)
        else:
            result.export_nifti(out_labels_path=p_lab, synthetic_fixture=args.synthetic_fixture)

    if args.out_provenance is not None:
        p_prov = Path(args.out_provenance)
        p_prov.parent.mkdir(parents=True, exist_ok=True)
        if p_prov.name.endswith(".npz"):
            np.savez_compressed(p_prov, provenance=result.provenance, affine=affine)
        elif p_prov.name.endswith(".npy"):
            np.save(p_prov, result.provenance)
        else:
            result.export_nifti(out_provenance_path=p_prov, synthetic_fixture=args.synthetic_fixture)

    if args.out_probabilities is not None:
        p_prob = Path(args.out_probabilities)
        p_prob.parent.mkdir(parents=True, exist_ok=True)
        if p_prob.name.endswith(".npz"):
            np.savez_compressed(p_prob, probabilities=result.probabilities, affine=affine)
        elif p_prob.name.endswith(".npy"):
            np.save(p_prob, result.probabilities)
        else:
            result.export_nifti(out_probs_path=p_prob, synthetic_fixture=args.synthetic_fixture)

    if args.out_json is not None:
        result.export_json(args.out_json)

    # Print summary
    print(json.dumps({
        "status": "PASS",
        "protocol": result.summary["protocol"],
        "supervision_type": result.summary["supervision_type"],
        "simulated": result.summary["simulated"],
        "human_annotation_claimed": False,
        "human_annotation_time_sec": None,
        "patient_id": args.patient_id,
        "dataset_role": args.dataset_role,
        "synthetic_fixture": bool(args.synthetic_fixture),
        "train_manifest": str(args.train_manifest) if args.train_manifest else None,
        "label_budget": args.label_budget,
        "image_sha256": image_sha,
        "gt_sha256": gt_sha,
        "native_shape_zyx": list(image_zyx.shape),
        "prompts_applied": len(prompts),
        "manual_voxels": result.summary["manual_voxels"],
        "propagated_voxels": result.summary["propagated_voxels"],
        "original_voxels": result.summary["original_voxels"],
        "pseudo_voxels": result.summary["pseudo_voxels"],
        "unknown_voxels": result.summary["unknown_voxels"],
        "affected_slices": result.summary["affected_slices"],
        "class_distribution": result.summary["class_distribution"],
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
