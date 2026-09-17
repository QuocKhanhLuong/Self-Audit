"""Native synthetic ACDC/M&Ms reference matching and freeze-boundary checks."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from self_audit_maskfree.config import MaskfreeConfig
from self_audit_maskfree.epoch_validation import observe_epoch
from self_audit_maskfree.evaluation import epoch_reference
from self_audit_maskfree.trainer import MaskfreeTrainer


def _nifti(path, values, *, shift=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1.5, 1.5, 8., 1.])
    affine[0, 3] = shift
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, path)


def _labels():
    labels = np.zeros((24, 24, 2), np.uint8)
    labels[5:19, 5:19] = 2
    labels[8:16, 8:16] = 1
    labels[2:6, 8:14] = 3
    return labels


def _trainer(tmp_path, dataset, data):
    torch.set_num_threads(1)
    config = MaskfreeConfig(dataset=dataset, data_root=str(data), output_dir=str(tmp_path),
                            total_epochs=3, warmup_epochs=0, batch_size=8, image_size=24,
                            width=4, feature_dim=4, device="cpu", allow_cpu=True, amp=False,
                            wandb_mode="disabled", run_id="native-epoch-ref", max_epochs=1)
    trainer = MaskfreeTrainer(config)
    trainer.setup()
    return trainer


def test_mnms_native_4d_sparse_ed_es_and_fixed_role_mapping(tmp_path):
    data = tmp_path / "OpenDataset"
    rng = np.random.default_rng(510)
    for split, patient in (("Training", "T11111"), ("Validation", "V11111"), ("Testing", "X11111")):
        image = rng.normal(size=(24, 24, 2, 3)).astype(np.float32)
        _nifti(data / split / patient / f"{patient}_sa.nii.gz", image)
    labels = np.stack([_labels(), np.zeros_like(_labels()), _labels()], axis=3)
    _nifti(data / "Validation/V11111/V11111_sa_gt.nii.gz", labels)
    metadata = data / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
    metadata.write_text("External code,ED,ES\nV11111,0,2\n")
    trainer = _trainer(tmp_path, "mnms", data)
    receipt = observe_epoch(trainer, 0)
    assert receipt["status"] == "COMPLETED", receipt
    assert receipt["reference_label_map"] == {"0": 0, "1": 3, "2": 2, "3": 1}
    assert receipt["students"]["student_audited"]["volumes"] == 2
    assert receipt["students"]["student_no_audit"]["volumes"] == 2
    assert len(receipt["matching"]["skipped_volumes"]) == 1
    epoch_root = trainer.paths.root / "validation/epoch_0001"
    # Removing phase metadata cannot turn the empty middle cine frame into a
    # reference-background observation or infer annotated frames from predictions.
    metadata.rename(metadata.with_suffix(".notcsv"))
    result = epoch_reference.evaluate_epoch(
        receipt["freeze_manifest"], epoch_root / "image_manifest.json", epoch_root / "missing_metadata")
    assert result["status"] == "UNAVAILABLE"
    assert "metadata missing" in result["reason"]
    assert all(student["dice"] is None for student in result["students"].values())


def test_acdc_cine_reexport_uses_exact_image_index_proof(tmp_path, monkeypatch):
    data = tmp_path / "ACDC"
    rng = np.random.default_rng(711)
    for patient in range(4):
        folder = data / "training" / f"patient{patient:03d}"
        cine = rng.normal(size=(24, 24, 2, 3)).astype(np.float32)
        _nifti(folder / f"patient{patient:03d}_4d.nii.gz", cine)
        for frame in (0, 2):
            stem = f"patient{patient:03d}_frame{frame + 1:02d}"
            # Same indexed image, different source header: this reproduces the
            # ACDC re-export geometry pattern without resampling either image.
            _nifti(folder / f"{stem}.nii.gz", cine[..., frame], shift=10.)
            _nifti(folder / f"{stem}_gt.nii.gz", _labels(), shift=10.)
    trainer = _trainer(tmp_path, "acdc", data)
    receipt = observe_epoch(trainer, 0)
    assert receipt["status"] == "COMPLETED", receipt
    assert len(receipt["matching"]["index_transfers"]) == 2
    assert receipt["students"]["student_audited"]["volumes"] == 2
    assert all(proof["no_resampling"] for proof in receipt["matching"]["index_transfers"])
    epoch_root = trainer.paths.root / "validation/epoch_0001"
    manifest = json.loads((epoch_root / "image_manifest.json").read_text())
    valid_freeze = json.loads(Path(receipt["freeze_manifest"]).read_text())
    bad_freeze = {**valid_freeze, "epoch": 99}
    invalid = epoch_root / "tampered_freeze.json"
    invalid.write_text(json.dumps(bad_freeze))
    reads = []
    monkeypatch.setattr(epoch_reference, "_auto_config", lambda *args: reads.append("mask matching"))
    with pytest.raises(Exception, match="(?i)hash|freeze"):
        epoch_reference.evaluate_epoch(invalid, epoch_root / "image_manifest.json", epoch_root / "rejected")
    assert not reads

    # The transfer helper also refuses to attach a new affine when the original
    # 3-D re-export is no longer the canonical image frame.
    from self_audit_maskfree.evaluation.native_reference_geometry import transfer_reference_indices
    proof = receipt["matching"]["index_transfers"][0]
    source = Path(proof["original_image_path"])
    image = nib.load(str(source))
    _nifti(source, np.asanyarray(image.dataobj) + 1, shift=10.)
    record = next(row for row in manifest["records"] if row["volume_id"] == proof["canonical_volume_id"])
    with pytest.raises(ValueError, match="does not exactly equal"):
        transfer_reference_indices(proof["original_mask_path"], source, record, epoch_root / "bad_transfer")
