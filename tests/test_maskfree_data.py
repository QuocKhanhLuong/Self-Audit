"""Focused checks for the mask-free image-only data layer (W3).

Three checks, one per risk that would silently invalidate the whole pipeline:

1. the loader works with masks absent and with the mask tree physically
   unreadable, and never touches ACDC's annotation sidecar;
2. mutating selection or verification intensities cannot move anything the
   fitting view sees -- including its normalization statistics;
3. native geometry, the inverse-transform record, patient-disjoint splits and
   the freeze gate on sealed verification data all behave as declared.

Fixtures are synthetic CPU NIfTI/NPY trees. They are software evidence about
the data layer's behaviour, not evidence about real ACDC or M&Ms content.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from self_audit_maskfree.data import (  # noqa: E402
    FreezeReceiptError,
    ImageOnlyDataset,
    MaskAccessError,
    MixedStudyGeometryError,
    assert_image_only,
    build_training_unit,
    discover_dataset,
    load_full_input,
    load_verification_unit,
)

nib = pytest.importorskip("nibabel")

IMAGE_SIZE = 64
NATIVE_HW = (72, 68)
DEPTH = 5
FRAMES = 10


def _affine() -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = 1.25
    affine[1, 1] = 1.25
    affine[2, 2] = 8.0
    return affine


def _volume(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(120.0, 25.0, size=shape)).astype(np.float32)


def _write_nifti(path: Path, array: np.ndarray, *, frame_duration: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(array, _affine())
    image.header.set_zooms(
        (1.25, 1.25, 8.0, frame_duration) if array.ndim == 4 else (1.25, 1.25, 8.0)
    )
    if array.ndim == 4:
        image.header.set_xyzt_units("mm", "sec")
    nib.save(image, str(path))


def _acdc_tree(root: Path, n_patients: int = 6) -> Path:
    """ACDC-shaped native tree: raw 4-D cine, ED/ES re-exports, masks, Info.cfg."""
    training = root / "training"
    for index in range(1, n_patients + 1):
        name = f"patient{index:03d}"
        folder = training / name
        raw = _volume(index, (*NATIVE_HW, DEPTH, FRAMES))
        _write_nifti(
            folder / f"{name}_4d.nii.gz",
            raw,
            frame_duration=0.035,
        )
        for frame in (1, 9):
            _write_nifti(
                folder / f"{name}_frame{frame:02d}.nii.gz",
                # These are exact 3-D re-exports of two 4-D frames. Discovery
                # may collapse only this geometry-and-image-identical case.
                raw[..., frame],
                frame_duration=None,
            )
            _write_nifti(
                folder / f"{name}_frame{frame:02d}_gt.nii.gz",
                np.zeros((*NATIVE_HW, DEPTH), dtype=np.float32),
                frame_duration=None,
            )
        # A same-patient, same-grid series with different pixels must survive
        # deduplication; patient-wide collapsing would erase this acquisition.
        _write_nifti(
            folder / f"{name}_sa.nii.gz",
            _volume(index * 1000, (*NATIVE_HW, DEPTH)),
            frame_duration=None,
        )
        (folder / "Info.cfg").write_text("ED: 1\nES: 9\nGroup: DCM\n", encoding="utf-8")
    return root


def _preprocessed_tree(root: Path, n_patients: int = 4) -> Path:
    """ED/ES-only preprocessed npy tree, as this repository already produces."""
    base = root / "preprocessed_data" / "mnm" / "train"
    volumes = base / "volumes"
    masks = base / "masks"
    volumes.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for index in range(1, n_patients + 1):
        case = f"case{index:03d}"
        for phase in ("ED", "ES"):
            phase_seed = index if phase == "ED" else index + 100
            np.save(
                volumes / f"{case}_{phase}.npy",
                _volume(phase_seed, (*NATIVE_HW, DEPTH)),
            )
            np.save(
                masks / f"{case}_{phase}.npy",
                np.zeros((*NATIVE_HW, DEPTH), dtype=np.uint8),
            )
    return root


def _freeze_receipt(tmp_path: Path, manifest: dict, partition_ids: list[str]) -> dict:
    """Create a complete receipt through W7's authoritative freeze API."""
    from self_audit_maskfree.export import REQUIRED_COMPARISONS, freeze_predictions

    tag = "valid" if partition_ids == [manifest["records"][0]["partition_id"]] else "wrong"
    output = tmp_path / f"freeze_{tag}"
    output.mkdir(parents=True, exist_ok=True)
    unit = manifest["records"][0]
    entries: list[dict] = []
    checkpoints: dict[str, Path] = {}
    for method in REQUIRED_COMPARISONS:
        method_dir = output / "predictions" / method
        method_dir.mkdir(parents=True, exist_ok=True)
        labels = method_dir / "labels.npy"
        validity = method_dir / "validity.npy"
        np.save(labels, np.zeros((DEPTH, *NATIVE_HW), dtype=np.int64))
        np.save(validity, np.ones((DEPTH, *NATIVE_HW), dtype=np.float32))
        entries.append(
            {
                "kind": "volume",
                "prediction_name": method,
                # The strict W7 unit-completeness map is keyed by this exact
                # unit, while the volume id remains the assembly identity.
                "unit_id": unit["unit_id"],
                "volume_id": unit["volume_id"],
                "study_id": unit["study_id"],
                "patient_id": unit["patient_id"],
                "split": unit["split"],
                "dataset": manifest["dataset"],
                "protocol": manifest["resolved_protocol"],
                "manifest_id": manifest["manifest_id"],
                "partition_id": partition_ids[0],
                "checkpoint_id": method,
                "version": "maskfree150.test.v1",
                "grid": "stored",
                "native_export_available": False,
                "native_export_reason": "synthetic test artifact",
                "volume_shape": [DEPTH, *NATIVE_HW],
                "complete_volume": True,
                "files": [
                    str(labels.relative_to(output)),
                    str(validity.relative_to(output)),
                ],
            }
        )
        checkpoint = tmp_path / f"{method}.pt"
        checkpoint.write_bytes(method.encode("utf-8"))
        checkpoints[method] = checkpoint
    nuisance = tmp_path / "nuisance_provenance.json"
    nuisance.write_text(json.dumps({"source": "synthetic-test"}), encoding="utf-8")
    return freeze_predictions(
        output,
        entries,
        checkpoints,
        dataset=manifest["dataset"],
        protocol=manifest["resolved_protocol"],
        required_methods=REQUIRED_COMPARISONS,
        required_unit_ids=[unit["unit_id"]],
        required_checkpoints=list(checkpoints),
        required_nuisance_files=[nuisance],
    )


def test_loads_without_masks_and_refuses_annotation_paths(tmp_path):
    """Masks absent, mask tree unreadable, annotation sidecar never opened."""
    root = _acdc_tree(tmp_path / "acdc")
    manifest = discover_dataset(root, "acdc", seed=42)

    # No annotation path is anywhere in the manifest.
    paths = [r["path"] for r in manifest["records"]] + [
        d["source_path"] for d in manifest["duplicates"]
    ]
    blob = json.dumps(manifest)
    assert "_gt." not in blob
    assert "Info.cfg" not in blob
    for path in paths:
        assert_image_only(path)

    # Every acquired frame and every Z slice is represented. The two exact
    # 3-D re-exports collapse against their matching 4-D frames, while the
    # unrelated same-patient SA series remains a counted volume.
    assert manifest["readiness"]["n_source_files"] == 24
    assert manifest["readiness"]["n_frames_enumerated"] == 6 * (FRAMES + 3)
    assert manifest["readiness"]["n_volumes"] == 6 * (FRAMES + 1)
    assert manifest["readiness"]["n_records"] == 6 * (FRAMES + 1) * DEPTH
    assert manifest["readiness"]["n_duplicates_collapsed"] == 12
    assert sum(r["path"].endswith("_4d.nii.gz") for r in manifest["records"]) == 6 * FRAMES * DEPTH
    assert any(r["path"].endswith("_sa.nii.gz") for r in manifest["records"])

    # Cine is detected honestly, and just as honestly not claimed as the protocol.
    assert manifest["readiness"]["cine_eligible_records"] == 6 * FRAMES * DEPTH
    assert manifest["resolved_protocol"] == "spatial_predictive"
    codes = {item["code"] for item in manifest["limitations"]}
    assert "cine_detected_temporal_transport_unimplemented" in codes

    # Physically forbid the annotation files, then load a full unit anyway.
    mask_files = sorted(root.rglob("*_gt.nii.gz")) + sorted(root.rglob("Info.cfg"))
    assert mask_files
    if os.geteuid() != 0:
        for path in mask_files:
            os.chmod(path, 0)
        try:
            dataset = ImageOnlyDataset(manifest, split="train", image_size=IMAGE_SIZE)
            assert len(dataset) > 0
            unit = dataset[0]
            unit.fitting.validate()
            unit.selection.validate()
            assert unit.record["mask_inputs_used"] is False
            assert unit.record["capabilities"]["verification"] is False
        finally:
            for path in mask_files:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    # A mask path presented directly is refused, not silently read.
    with pytest.raises(MaskAccessError):
        assert_image_only(mask_files[0])

    # A preprocessed ED/ES root loads too, and discloses its cohort bias.
    pre_root = _preprocessed_tree(tmp_path / "mnms")
    pre_manifest = discover_dataset(pre_root, "mnms", seed=42)
    assert pre_manifest["readiness"]["n_volumes"] == 8
    assert pre_manifest["readiness"]["n_records"] == 8 * DEPTH
    pre_codes = {item["code"] for item in pre_manifest["limitations"]}
    assert "annotation_selected_cohort" in pre_codes
    assert "stored_grid_export_only" in pre_codes
    assert all(r["native_grid_export"] is False for r in pre_manifest["records"])


def test_discovery_fails_closed_on_mixed_native_study_grid(tmp_path):
    """One study cannot silently acquire two incompatible role-coordinate maps."""
    root = tmp_path / "acdc" / "training" / "patient001"
    _write_nifti(root / "patient001_sa.nii.gz", _volume(1, (*NATIVE_HW, DEPTH)), frame_duration=None)
    affine = _affine()
    affine[0, 0] = 1.5
    image = nib.Nifti1Image(_volume(2, (*NATIVE_HW, DEPTH)), affine)
    image.header.set_zooms((1.5, 1.25, 8.0))
    nib.save(image, str(root / "patient001_la.nii.gz"))

    with pytest.raises(MixedStudyGeometryError, match="mixed native grid/affine/orientation"):
        discover_dataset(tmp_path / "acdc", "acdc", seed=42)


def test_mnms_time_suffixes_share_one_patient_split_identity(tmp_path):
    volumes = tmp_path / "preprocessed_data" / "mnm" / "train" / "volumes"
    volumes.mkdir(parents=True)
    for time_index in (1, 2):
        np.save(
            volumes / f"case001_t{time_index:02d}.npy",
            _volume(time_index, (*NATIVE_HW, DEPTH)),
        )

    manifest = discover_dataset(tmp_path, "mnms", seed=42)
    assert {record["patient_id"] for record in manifest["records"]} == {"case001"}
    assert {record["split"] for record in manifest["records"]} == {"train"}
    assert manifest["split_provenance"]["split_identity"] == "patient_id"


def test_nifti_spacing_requires_declared_units_but_native_geometry_stands_alone(tmp_path):
    root = tmp_path / "acdc" / "training"
    known = nib.Nifti1Image(_volume(1, (*NATIVE_HW, DEPTH)), _affine())
    known.header.set_zooms((0.001, 0.002, 0.008))
    known.header.set_xyzt_units("meter", None)
    (root / "patient001").mkdir(parents=True)
    nib.save(known, str(root / "patient001" / "patient001_sa.nii.gz"))

    unknown = nib.Nifti1Image(_volume(2, (*NATIVE_HW, DEPTH)), _affine())
    unknown.header.set_zooms((1.25, 1.25, 8.0))
    (root / "patient002").mkdir(parents=True)
    nib.save(unknown, str(root / "patient002" / "patient002_sa.nii.gz"))

    manifest = discover_dataset(tmp_path / "acdc", "acdc", seed=42)
    by_patient = {r["patient_id"]: r for r in manifest["records"] if r["slice_index"] == 0}
    assert by_patient["patient001"]["native_geometry"] == "available"
    assert by_patient["patient001"]["spacing_valid"] is True
    assert by_patient["patient001"]["spacing_mm"] == pytest.approx([1.0, 2.0, 8.0])
    assert by_patient["patient002"]["native_geometry"] == "available"
    assert by_patient["patient002"]["spacing_valid"] is False
    assert by_patient["patient002"]["spacing_mm"] is None


def test_selection_and_verify_mutation_cannot_change_the_fitting_view(tmp_path):
    """Every fitting view of a study is bitwise invariant to every withheld voxel.

    The partition is study-level, so the check is study-level too: all withheld
    voxels of the study are corrupted across every slice, every cine time and
    every unit file, and then *every* TrainingUnit of that study is compared.
    A per-unit role map would pass a single-unit check and still leak, because a
    voxel sealed in one unit would be fitting context in the next.
    """
    root = _acdc_tree(tmp_path / "acdc", n_patients=3)
    manifest = discover_dataset(root, "acdc", seed=42)
    record = manifest["records"][0]
    unit_id = record["unit_id"]

    from self_audit_maskfree.data import build_partition

    partition = build_partition(
        int(record["native_hw"][0]),
        int(record["native_hw"][1]),
        study_id=record["study_id"],
        seed=42,
    )
    assert partition.partition_id == record["partition_id"]
    # Unit identity must not enter the role map at all.
    assert partition.spec["role_mask_scope"] == "study_grid"
    counts = partition.counts()
    assert counts["fit"] > 0 and counts["select"] > 0 and counts["verify"] > 0
    assert counts["guard_dropped"] > 0

    before = build_training_unit(manifest, unit_id, image_size=IMAGE_SIZE, seed=42)

    # Overwrite every withheld native voxel (selection AND verification, plus the
    # guard band) in every acquisition belonging to this patient, across all
    # slices and all cine frames. The 4-D source uses a trailing frame axis;
    # re-exported 3-D frames and the unrelated SA series are included too.
    withheld = (partition.select | partition.verify | partition.guard).numpy()
    study_paths = {
        r["path"] for r in manifest["records"] if r["study_id"] == record["study_id"]
    }
    for raw_path in sorted(study_paths):
        path = Path(raw_path)
        image = nib.load(str(path))
        array = np.asarray(image.dataobj).copy()
        if array.ndim == 4:
            array[withheld, :, :] = 9.9e4
        else:
            array[withheld, :] = 9.9e4
        nib.save(nib.Nifti1Image(array, image.affine, image.header), str(path))

    after = build_training_unit(manifest, unit_id, image_size=IMAGE_SIZE, seed=42)

    assert torch.equal(before.fitting.image, after.fitting.image)
    assert torch.equal(before.fitting.context, after.fitting.context)
    assert torch.equal(before.fitting.support, after.fitting.support)
    assert before.record["normalization"] == after.record["normalization"]
    assert before.record["support_counts"] == after.record["support_counts"]

    # Positive control: the selection view is supposed to notice.
    assert not torch.equal(before.selection.image, after.selection.image)
    assert torch.equal(before.selection.support, after.selection.support)

    # The fitting view never contains a withheld intensity.
    assert torch.all(after.fitting.image[:, ~after.fitting.support] == 0)
    assert torch.all(after.fitting.context[:, ~after.fitting.support] == 0)
    assert not bool((after.fitting.support & after.selection.support).any())

    # Cross-unit: a study with several units must share one role map, and every
    # one of its fitting views must survive corruption of the whole study.
    pre_root = _preprocessed_tree(tmp_path / "mnms", n_patients=2)
    pre_manifest = discover_dataset(pre_root, "mnms", seed=42)
    study_id = pre_manifest["records"][0]["study_id"]
    study_records = [r for r in pre_manifest["records"] if r["study_id"] == study_id]
    assert len(study_records) >= 2, "fixture must give one study more than one unit"
    assert len({r["partition_id"] for r in study_records}) == 1

    study_partition = build_partition(
        int(study_records[0]["native_hw"][0]),
        int(study_records[0]["native_hw"][1]),
        study_id=study_id,
        seed=42,
    )
    assert study_partition.partition_id == study_records[0]["partition_id"]
    study_withheld = (
        study_partition.select | study_partition.verify | study_partition.guard
    ).numpy()

    pre_before = {
        r["unit_id"]: build_training_unit(
            pre_manifest, r["unit_id"], image_size=IMAGE_SIZE, seed=42
        )
        for r in study_records
    }
    for raw_path in sorted({r["path"] for r in study_records}):
        r = next(r for r in study_records if r["path"] == raw_path)
        volume = np.load(r["path"])
        volume[study_withheld, :] = -8.8e4
        np.save(r["path"], volume)

    for r in study_records:
        rebuilt = build_training_unit(
            pre_manifest, r["unit_id"], image_size=IMAGE_SIZE, seed=42
        )
        original = pre_before[r["unit_id"]]
        assert torch.equal(original.fitting.image, rebuilt.fitting.image)
        assert torch.equal(original.fitting.context, rebuilt.fitting.context)
        assert torch.equal(original.fitting.support, rebuilt.fitting.support)
        assert original.record["normalization"] == rebuilt.record["normalization"]
        assert not torch.equal(original.selection.image, rebuilt.selection.image)

    # The deployment view is explicitly a different object with no such claim.
    full, deployment_record = load_full_input(manifest, unit_id, image_size=IMAGE_SIZE)
    assert full.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert deployment_record["deployment_only"] is True
    assert deployment_record["withheld_intensity_prediction_claim"] is False
    assert deployment_record["training_use_permitted"] is False


def test_geometry_splits_and_the_freeze_gate_on_sealed_observations(tmp_path):
    """Native transform records, patient disjointness, and sealed verification."""
    root = _acdc_tree(tmp_path / "acdc", n_patients=8)
    manifest = discover_dataset(root, "acdc", seed=42)

    # Patient-disjoint splits, all non-empty, never described as official.
    by_split: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    for record in manifest["records"]:
        by_split[record["split"]].add(record["patient_id"])
    assert by_split["train"] and by_split["dev"] and by_split["test"]
    assert not (by_split["train"] & by_split["dev"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["dev"] & by_split["test"])
    assert manifest["split_provenance"]["official_test_membership"] is False
    assert "synthetic_split_not_a_fresh_test_set" in {
        item["code"] for item in manifest["limitations"]
    }
    assert discover_dataset(root, "acdc", seed=42)["manifest_id"] == manifest["manifest_id"]

    # Native NIfTI keeps its affine and an invertible transform record.
    unit = build_training_unit(
        manifest, manifest["records"][0]["unit_id"], image_size=IMAGE_SIZE, seed=42
    )
    transform = unit.record["inverse_transform"]
    assert unit.record["native_geometry"] == "available"
    assert unit.record["native_grid_export"] is True
    assert np.allclose(np.asarray(unit.record["native_affine"]), _affine())
    assert transform["forward_resize"]["input_hw"] == list(NATIVE_HW)
    assert transform["forward_resize"]["output_hw"] == [IMAGE_SIZE, IMAGE_SIZE]
    assert transform["inverse_resize"]["output_hw"] == list(NATIVE_HW)
    assert transform["inverse_resize"]["mode_labels"] == "nearest_exact"
    assert transform["crop"] is None
    assert transform["slice_index"] == manifest["records"][0]["slice_index"]
    assert transform["frame_index"] == 0

    # Sealed verification stays sealed without a valid freeze receipt.
    unit_id = unit.record["unit_id"]
    partition_id = unit.record["partition_id"]
    with pytest.raises(FreezeReceiptError):
        load_verification_unit(manifest, unit_id, {}, image_size=IMAGE_SIZE)
    receipt = _freeze_receipt(tmp_path, manifest, [partition_id])

    wrong_manifest = dict(receipt, manifest_id="0" * 32)
    with pytest.raises(FreezeReceiptError):
        load_verification_unit(manifest, unit_id, wrong_manifest, image_size=IMAGE_SIZE)

    wrong_partition = dict(receipt, partition_ids=["not-this-partition"])
    with pytest.raises(FreezeReceiptError):
        load_verification_unit(manifest, unit_id, wrong_partition, image_size=IMAGE_SIZE)

    tampered = dict(receipt, files={k: "0" * 64 for k in receipt["files"]})
    with pytest.raises(FreezeReceiptError):
        load_verification_unit(manifest, unit_id, tampered, image_size=IMAGE_SIZE)

    verify_view = load_verification_unit(manifest, unit_id, receipt, image_size=IMAGE_SIZE)
    verify_view.validate()
    assert verify_view.role == "verify"
    assert verify_view.partition_id == partition_id
    assert bool(verify_view.support.any())
    assert not bool((verify_view.support & unit.fitting.support).any())
    assert not bool((verify_view.support & unit.selection.support).any())

    # The training-side dataset has no route to it at all.
    dataset = ImageOnlyDataset(manifest, split="train", image_size=IMAGE_SIZE)
    with pytest.raises(PermissionError):
        dataset.load_verification_unit(manifest, unit_id, receipt)
    assert dataset.fingerprint() == ImageOnlyDataset(
        manifest, split="train", image_size=IMAGE_SIZE
    ).fingerprint()
