import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from self_audit.model import digest
from self_audit.volume.config import default_volume_config
from self_audit.volume.evaluation import (
    evaluate_frozen,
    evaluate_manifest,
    export_student,
    load_exported_student,
    predict_native_volume,
)
from self_audit.volume.fixtures import make_volume_fixture
from self_audit.volume.models import build_volume_model


class CenterClassModel(nn.Module):
    def __init__(self, cls: int = 0):
        super().__init__()
        self.cls = int(cls)
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, image, valid_mask=None):
        self.calls += 1
        shape = (image.shape[0], 4, *image.shape[2:])
        logits = torch.full(shape, -2.0, dtype=image.dtype, device=image.device)
        logits[:, self.cls] = 2.0 + self.bias
        return logits


class ResamplingProbeModel(CenterClassModel):
    seen_shapes: list[tuple[int, ...]] = []

    def forward(self, image, valid_mask=None):
        type(self).seen_shapes.append(tuple(int(x) for x in image.shape[2:]))
        return super().forward(image, valid_mask=valid_mask)


class NeighborReceptiveFieldModel(nn.Module):
    """A deliberately spatially non-pointwise invalid-value oracle.

    The center input channel's immediate left neighbour drives class-0 logits.
    A correct evaluator must therefore sanitize unsupported native voxels before
    this actual receptive field sees them; checking only a pointwise model would
    miss convolution/context leakage.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 4, kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            # Input channel 1 is the center z context; width offset 0 is x-1.
            self.conv.weight[0, 1, 0, 0, 0] = 10.0

    def forward(self, image, valid_mask=None):
        return self.conv(image)


def _evaluation_manifest(tmp_path: Path, *, n_patients: int = 2):
    source, _source_sha, geometry, _geometry_sha = make_volume_fixture(
        tmp_path / "fixture", n_labeled=n_patients, n_unlabeled=1, shape_tzyx=(2, 3, 5, 7)
    )
    train = json.loads(source.read_text())
    records = train["labeled"][:n_patients]
    payload = {
        "schema": "self_audit.volume.evaluation.v1", "role": "evaluation", "split": "development",
        "synthetic": True, "records": records,
    }
    manifest = source.parent / "evaluation.json"
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    sidecar = json.loads(geometry.read_text())
    sidecar["source_manifest"] = manifest.name
    sidecar["source_manifest_sha256"] = digest(manifest)
    eval_geometry = source.parent / "evaluation_geometry.json"
    eval_geometry.write_text(json.dumps(sidecar, sort_keys=True, indent=2) + "\n")
    return manifest, eval_geometry, digest(manifest), digest(eval_geometry), records


def _real_image_only_manifest(tmp_path: Path, *, mismatch: bool):
    """Build a hash-pinned manufactured real-format NIfTI image-only fixture."""
    nib = pytest.importorskip("nibabel")
    source, _source_sha, geometry, _geometry_sha = make_volume_fixture(
        tmp_path / "fixture", n_labeled=1, n_unlabeled=1, shape_tzyx=(2, 3, 5, 7),
        affine_xyz_to_world_mm=np.eye(4),
    )
    train = json.loads(source.read_text())
    record = train["labeled"][0]
    root = source.parent
    raw = (np.zeros((7, 5, 3, 2), dtype=np.float32) if mismatch
           else np.random.default_rng(91).normal(size=(7, 5, 3, 2)).astype(np.float32))
    raw_path = root / "raw_image.nii.gz"
    raw_image = nib.Nifti1Image(raw, np.eye(4, dtype=float))
    raw_image.header.set_xyzt_units("mm", "sec")
    nib.save(raw_image, raw_path)
    raw_sha = digest(raw_path)
    if not mismatch:
        lo, hi = np.percentile(raw, [0.5, 99.5])
        prepared = np.clip((raw - lo) / max(float(hi - lo), 1e-6), 0, 1)
        prepared = prepared.transpose(3, 2, 1, 0).astype(np.float32)
        image_path = root / record["image"]
        np.save(image_path, prepared)
        record["image_sha256"] = digest(image_path)
    for item in record.get("masks", {}).values():
        # Image-only evaluation must not open labels even when their paths are
        # unusable; keeping these paths missing makes that contract explicit.
        item["path"] = "missing-image-only-mask.npy"
    payload = {
        "schema": "self_audit.volume.evaluation.v1", "role": "evaluation",
        "split": "development", "synthetic": False, "records": [record],
    }
    manifest = root / "evaluation.json"
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    sidecar = json.loads(geometry.read_text())
    sidecar["synthetic"] = False
    sidecar["source_manifest"] = manifest.name
    sidecar["source_manifest_sha256"] = digest(manifest)
    for patient in sidecar["patients"].values():
        patient["orientation"] = "RAS"
        patient["source_image"] = {"path": raw_path.name, "sha256": raw_sha}
    sidecar["patients"][record["id"]]["image_sha256"] = record["image_sha256"]
    eval_geometry = root / "evaluation_geometry.json"
    eval_geometry.write_text(json.dumps(sidecar, sort_keys=True, indent=2) + "\n")
    return manifest, eval_geometry, digest(manifest), digest(eval_geometry)


def test_native_tiled_inference_merges_odd_shape_affine_and_partial_support():
    model = CenterClassModel(2)
    image = np.random.default_rng(5).random((2, 3, 5, 7), dtype=np.float32)
    affine = np.array([[1.3, 0.0, 0.0, 12.0], [0.0, 0.9, 0.0, -4.0], [0.0, 0.0, 4.2, 7.0], [0, 0, 0, 1]], dtype=float)
    valid = np.ones(image.shape, dtype=bool)
    valid[0, 1, 2:, 3:] = False
    before = copy.deepcopy(model.state_dict())
    probabilities, support = predict_native_volume(
        model, image, affine, slab_depth=2, slab_overlap=1, tile_size=(3, 4), tile_overlap=1,
        native_valid=valid,
    )
    assert probabilities.shape == (2, 4, 3, 5, 7)
    assert support.shape == image.shape and np.array_equal(support, valid)
    assert np.allclose(probabilities.sum(axis=1)[support], 1.0, atol=1e-6)
    assert np.all(probabilities[0, :, 1, 2:, 3:] == 0)
    assert model.training is True
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    with pytest.raises(ValueError, match="affine"):
        predict_native_volume(model, image, np.zeros((4, 4)), slab_depth=2)


def test_native_inference_honors_configured_spacing_fov_and_inverts_to_native(tmp_path):
    ResamplingProbeModel.seen_shapes = []
    image = np.random.default_rng(12).random((1, 6, 5, 7), dtype=np.float32)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    model = ResamplingProbeModel(2)
    probabilities, support = predict_native_volume(
        model, image, affine, slab_depth=1, tile_size=99,
        spacing_xyz_mm=(2.0, 2.0, 4.0), fov_xyz_mm=(8.0, 8.0, 8.0),
        resampling="configured", flip=True,
    )
    assert probabilities.shape == (1, 4, 6, 5, 7)
    assert support.shape == image.shape
    # Native spacing would expose (Z,Y,X)=(6,5,7); configured spacing produces
    # (4,3,4), while the physical FOV requests a fixed (Z,Y,X) model window
    # of (3,5,5), including invalid edge padding where needed.
    assert (3, 5, 5) in ResamplingProbeModel.seen_shapes
    assert (1, 5, 7) not in ResamplingProbeModel.seen_shapes
    assert np.allclose(probabilities.sum(axis=1)[support], 1.0, atol=1e-6)
    with pytest.raises(ValueError, match="spacing_xyz_mm"):
        predict_native_volume(model, image, affine, resampling="configured")


def test_configured_invalid_native_values_are_invariant_through_neighbor_receptive_field():
    """Invalid input cannot leak through convolution/context or inverse merge."""
    model = NeighborReceptiveFieldModel()
    image = np.zeros((1, 1, 3, 7), dtype=np.float32)
    native_valid = np.ones_like(image, dtype=bool)
    native_valid[..., 1] = False
    perturbed = image.copy()
    perturbed[..., 1] = 1.0

    # Prove the oracle is genuinely neighbour-sensitive before testing the
    # evaluator: a pointwise-only model would not expose this failure mode.
    raw = torch.zeros((1, 3, 1, 1, 7))
    changed = raw.clone(); changed[..., 1] = 1.0
    assert model(changed)[0, 0, 0, 0, 2] != model(raw)[0, 0, 0, 0, 2]

    base, support = predict_native_volume(
        model, image, np.eye(4), slab_depth=1, tile_size=10,
        native_valid=native_valid, spacing_xyz_mm=(1.5, 1.0, 1.0),
        resampling="configured",
    )
    altered, altered_support = predict_native_volume(
        model, perturbed, np.eye(4), slab_depth=1, tile_size=10,
        native_valid=native_valid, spacing_xyz_mm=(1.5, 1.0, 1.0),
        resampling="configured",
    )
    assert np.array_equal(support, altered_support)
    assert not support[..., 1].any()
    assert np.all(base * (~support[:, None]) == 0)
    assert np.allclose(np.where(support[:, None], base, 0.0),
                       np.where(altered_support[:, None], altered, 0.0), atol=1e-7)
    assert np.all(altered * (~altered_support[:, None]) == 0)


def test_configured_inverse_merge_is_tile_and_shifted_slab_invariant():
    """Current-grid merge prevents inverse linear stencils from splitting at tiles/slabs."""
    image = np.zeros((1, 1, 3, 9), dtype=np.float32)
    model = CenterClassModel(1)
    common = {"spacing_xyz_mm": (1.5, 1.0, 1.0), "resampling": "configured"}
    whole, whole_support = predict_native_volume(
        model, image, np.eye(4), slab_depth=1, slab_overlap=0, tile_size=99, **common
    )
    tiled, tiled_support = predict_native_volume(
        model, image, np.eye(4), slab_depth=1, slab_overlap=0, tile_size=2, **common
    )
    shifted, shifted_support = predict_native_volume(
        model, image, np.eye(4), slab_depth=2, slab_overlap=1, tile_size=2, **common
    )
    assert np.array_equal(whole_support, tiled_support)
    assert np.array_equal(whole_support, shifted_support)
    assert np.allclose(whole, tiled, atol=1e-7)
    assert np.allclose(whole, shifted, atol=1e-7)


def test_manifest_evaluation_reads_labels_only_when_explicit_and_aggregates_frames_patients(tmp_path):
    manifest, geometry, manifest_sha, geometry_sha, records = _evaluation_manifest(tmp_path)
    model = CenterClassModel(0)
    result = evaluate_manifest(
        model, manifest, manifest_sha, geometry, geometry_sha, prediction_model="student",
        split="development", synthetic=True, allow_labels=False, slab_depth=2, tile_size=(3, 4),
    )
    assert result["quality"] == "FROZEN_IMAGE_ONLY"
    assert result["labels_read"] == 0 and result["frames"] == []
    assert result["native_probability_shapes"][records[0]["id"]] == [2, 4, 3, 5, 7]

    scored = evaluate_manifest(
        model, manifest, manifest_sha, geometry, geometry_sha, prediction_model="student",
        split="development", synthetic=True, allow_labels=True, slab_depth=2, tile_size=(3, 4),
        predictions_dir=tmp_path / "merged_predictions",
    )
    assert scored["quality"] == "FROZEN_DEVELOPMENT_ONLY"
    assert scored["labels_read"] == 2 * len(records)
    assert len(scored["frames"]) == 2 * len(records)
    assert all((tmp_path / "merged_predictions" / item["probabilities"]).exists()
               for item in scored["prediction_files"].values())
    # Every patient/frame is scored independently; macro_patient is not a
    # pooled confusion-derived surrogate.
    for patient in records:
        assert patient["id"] in scored["patients"]
    assert scored["pooled_confusion"]


def test_student_and_ema_are_explicit_choices_and_frozen_adapter_does_not_mutate_model():
    image = torch.rand(1, 3, 2, 4, 5)
    labels = torch.zeros(1, 2, 4, 5, dtype=torch.long)
    valid = torch.ones_like(labels, dtype=torch.bool)
    student, teacher = CenterClassModel(0), CenterClassModel(1)
    student.train(); teacher.eval()
    result_student = evaluate_frozen(student, [{"image": image, "labels": labels, "geometric_valid": valid}], prediction_model="student")
    result_ema = evaluate_frozen(student, [{"image": image, "labels": labels, "geometric_valid": valid}], teacher=teacher, prediction_model="ema")
    assert result_student["prediction_model"] == "student"
    assert result_ema["prediction_model"] == "ema"
    assert result_student["pooled_confusion"] != result_ema["pooled_confusion"]
    assert student.training is True and teacher.training is False


def test_frozen_masked_target_support_does_not_create_false_negative():
    image = torch.rand(1, 3, 1, 1, 2)
    labels = torch.tensor([[[[1, 1]]]], dtype=torch.long)
    valid = torch.ones_like(labels, dtype=torch.bool)
    label_valid = torch.tensor([[[[True, False]]]], dtype=torch.bool)
    result = evaluate_frozen(
        CenterClassModel(1),
        [{"image": image, "labels": labels, "geometric_valid": valid, "label_valid": label_valid}],
    )
    assert result["valid_voxels"] == 1
    assert result["frames"][0]["raw"][1]["dice"] == pytest.approx(1.0)


def test_export_roundtrip_is_strict_and_teacher_free(tmp_path):
    config = default_volume_config()
    config["data"] = {"synthetic": True}
    model, _ = build_volume_model(config, synthetic=True)
    out = tmp_path / "student_export"
    manifest_sha = export_student(model, out, config=config, synthetic=True)
    restored, metadata = load_exported_student(out, manifest_sha)
    assert metadata["schema"] == "self_audit.volume.student_export.v2"
    assert metadata["teacher_included"] is False
    assert metadata["pretrained_weights_required_at_load"] is False
    image = torch.rand(1, 3, 1, 9, 11)
    assert restored(image).shape == (1, 4, 1, 9, 11)

    payload = torch.load(out / "weights.pt", map_location="cpu", weights_only=True)
    payload["teacher"] = {}
    torch.save(payload, out / "weights.pt")
    with pytest.raises(ValueError, match="payload hash"):
        load_exported_student(out, manifest_sha)


def test_evaluation_rejects_tampered_manifest_geometry_and_split(tmp_path):
    manifest, geometry, manifest_sha, geometry_sha, _ = _evaluation_manifest(tmp_path)
    model = CenterClassModel(0)
    with pytest.raises(ValueError, match="manifest hash"):
        evaluate_manifest(model, manifest, "0" * 64, geometry, geometry_sha, synthetic=True)
    with pytest.raises(ValueError, match="geometry hash"):
        evaluate_manifest(model, manifest, manifest_sha, geometry, "0" * 64, synthetic=True)
    with pytest.raises(ValueError, match="split"):
        evaluate_manifest(model, manifest, manifest_sha, geometry, geometry_sha, synthetic=True, split="TEST")


def _repin_evaluation(manifest: Path, geometry: Path) -> tuple[str, str]:
    """Repin the sidecar after a deliberate evaluation-manifest mutation."""
    manifest_sha = digest(manifest)
    sidecar = json.loads(geometry.read_text())
    sidecar["source_manifest_sha256"] = manifest_sha
    geometry.write_text(json.dumps(sidecar, sort_keys=True, indent=2) + "\n")
    return manifest_sha, digest(geometry)


def test_scoring_rejects_same_shape_mask_with_translated_affine(tmp_path):
    manifest, geometry, _manifest_sha, _geometry_sha, _records = _evaluation_manifest(tmp_path, n_patients=1)
    payload = json.loads(manifest.read_text())
    for mask in payload["records"][0]["masks"].values():
        mask["affine_xyz_to_world_mm"] = [[1, 0, 0, 1000], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    manifest_sha, geometry_sha = _repin_evaluation(manifest, geometry)
    with pytest.raises(ValueError, match="mask affine"):
        evaluate_manifest(
            CenterClassModel(0), manifest, manifest_sha, geometry, geometry_sha,
            synthetic=True, allow_labels=True, split="development", slab_depth=2, tile_size=(3, 4),
        )


def test_scoring_accepts_nonidentity_permuted_native_affine_and_frame_identity(tmp_path):
    affine = [[0.0, 1.3, 0.0, 12.0], [0.9, 0.0, 0.0, -4.0], [0.0, 0.0, -4.2, 7.0], [0, 0, 0, 1.0]]
    source, _msha, geometry, _gsha = make_volume_fixture(
        tmp_path / "fixture", n_labeled=1, n_unlabeled=1, shape_tzyx=(2, 3, 5, 7),
        affine_xyz_to_world_mm=affine,
    )
    train = json.loads(source.read_text())
    records = train["labeled"][:1]
    for frame, mask in records[0]["masks"].items():
        mask.update({"affine_xyz_to_world_mm": affine, "units": "mm", "frame_index": int(frame),
                     "native_shape_zyx": [3, 5, 7]})
    payload = {"schema": "self_audit.volume.evaluation.v1", "role": "evaluation", "split": "development",
               "synthetic": True, "records": records}
    manifest = source.parent / "evaluation.json"
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    sidecar = json.loads(geometry.read_text())
    sidecar["source_manifest"] = manifest.name
    sidecar["source_manifest_sha256"] = digest(manifest)
    eval_geometry = source.parent / "evaluation_geometry.json"
    eval_geometry.write_text(json.dumps(sidecar, sort_keys=True, indent=2) + "\n")
    result = evaluate_manifest(
        CenterClassModel(0), manifest, digest(manifest), eval_geometry, digest(eval_geometry),
        synthetic=True, allow_labels=True, split="development", slab_depth=2, tile_size=(3, 4),
    )
    assert result["labels_read"] == 2


def test_scoring_rejects_wrong_frame_and_source_hash_without_opening_prediction_paths(tmp_path):
    manifest, geometry, _manifest_sha, _geometry_sha, _records = _evaluation_manifest(tmp_path, n_patients=1)
    payload = json.loads(manifest.read_text())
    first = payload["records"][0]["masks"]["0"]
    first["frame_index"] = 1
    first["source_sha256"] = "0" * 64
    first["source_path"] = "missing-raw-mask.nii.gz"
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    manifest_sha, geometry_sha = _repin_evaluation(manifest, geometry)
    with pytest.raises(ValueError, match="frame"):
        evaluate_manifest(
            CenterClassModel(0), manifest, manifest_sha, geometry, geometry_sha,
            synthetic=True, allow_labels=True, split="development", slab_depth=2, tile_size=(3, 4),
        )


def test_image_only_never_opens_mask_paths_even_when_missing(tmp_path):
    manifest, geometry, _manifest_sha, _geometry_sha, _records = _evaluation_manifest(tmp_path, n_patients=1)
    payload = json.loads(manifest.read_text())
    for mask in payload["records"][0]["masks"].values():
        mask["path"] = "does-not-exist.npy"
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    manifest_sha, geometry_sha = _repin_evaluation(manifest, geometry)
    result = evaluate_manifest(
        CenterClassModel(0), manifest, manifest_sha, geometry, geometry_sha,
        synthetic=True, allow_labels=False, split="development", slab_depth=2, tile_size=(3, 4),
    )
    assert result["labels_read"] == 0 and result["quality"] == "FROZEN_IMAGE_ONLY"


def test_real_image_only_rejects_raw_prepared_voxel_mismatch_before_inference(tmp_path):
    manifest, geometry, manifest_sha, geometry_sha = _real_image_only_manifest(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="prepared image/raw cine voxel correspondence"):
        evaluate_manifest(
            CenterClassModel(0), manifest, manifest_sha, geometry, geometry_sha,
            synthetic=False, allow_labels=False, split="development", slab_depth=1, tile_size=10,
        )


def test_real_image_only_accepts_canonical_prepared_nifti_bridge_without_labels(tmp_path):
    manifest, geometry, manifest_sha, geometry_sha = _real_image_only_manifest(tmp_path, mismatch=False)
    result = evaluate_manifest(
        CenterClassModel(0), manifest, manifest_sha, geometry, geometry_sha,
        synthetic=False, allow_labels=False, split="development", slab_depth=1, tile_size=10,
    )
    assert result["quality"] == "FROZEN_IMAGE_ONLY"
    assert result["labels_read"] == 0 and result["frames"] == []


def test_volume_evaluate_cli_real_path_runs_from_arbitrary_cwd(tmp_path, monkeypatch):
    manifest, geometry, manifest_sha, geometry_sha, _ = _evaluation_manifest(tmp_path, n_patients=1)
    import scripts.volume_evaluate as cli

    class Trainer:
        model = CenterClassModel(0)
        teacher = CenterClassModel(1)

    monkeypatch.setattr(cli, "load_volume_checkpoint", lambda *args, **kwargs: (Trainer(), {"config": {}}))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "nested" / "result.json"
    assert cli.main([
        "--checkpoint", str(tmp_path / "checkpoint"), "--checkpoint-sha256", "a" * 64,
        "--manifest", str(manifest), "--manifest-sha256", manifest_sha,
        "--geometry", str(geometry), "--geometry-sha256", geometry_sha,
        "--synthetic", "--output", str(output), "--tile-size", "3", "--slab-depth", "2",
    ]) == 0
    receipt = json.loads(output.read_text())
    assert receipt["quality"] == "FROZEN_IMAGE_ONLY" and receipt["labels_read"] == 0


def test_volume_evaluate_cli_uses_checkpoint_configured_geometry(tmp_path, monkeypatch):
    manifest, geometry, manifest_sha, geometry_sha, _ = _evaluation_manifest(tmp_path, n_patients=1)
    import scripts.volume_evaluate as cli

    ResamplingProbeModel.seen_shapes = []

    class Trainer:
        model = ResamplingProbeModel(0)
        teacher = None

    checkpoint_config = {"geometry": {
        "resampling": "configured", "spacing_xyz_mm": [2.0, 1.8, 8.0],
        "fov_xyz_mm": [8.0, 8.0, 8.0], "flip": True,
    }}
    monkeypatch.setattr(cli, "load_volume_checkpoint",
                        lambda *args, **kwargs: (Trainer(), {"config": checkpoint_config}))
    output = tmp_path / "configured" / "result.json"
    assert cli.main([
        "--checkpoint", str(tmp_path / "checkpoint"), "--checkpoint-sha256", "a" * 64,
        "--manifest", str(manifest), "--manifest-sha256", manifest_sha,
        "--geometry", str(geometry), "--geometry-sha256", geometry_sha,
        "--synthetic", "--output", str(output), "--tile-size", "99", "--slab-depth", "1",
    ]) == 0
    receipt = json.loads(output.read_text())
    assert receipt["quality"] == "FROZEN_IMAGE_ONLY"
    assert receipt["labels_read"] == 0
    assert receipt["evaluation_source_sha256"] and len(receipt["evaluation_source_sha256"]) == 64
    assert receipt["cli_flags"]["slab_depth"] == 1
    declared = receipt["inference_protocol"]["declared"]
    effective = receipt["inference_protocol"]["effective_by_record"]["synthetic_volume_000"]
    assert declared["slab_depth"] == 1 and declared["tile_size"] == [99, 99]
    assert effective["slab_depth"] == 2 and effective["tile_size"] == [5, 5]
    assert effective["spacing_xyz_mm"] == [2.0, 1.8, 8.0]
    assert effective["fov_xyz_mm"] == [8.0, 8.0, 8.0]
    assert effective["flip"] is True and effective["resampling"] == "configured"
    assert (2, 5, 5) in ResamplingProbeModel.seen_shapes
    assert (2, 3, 5) not in ResamplingProbeModel.seen_shapes
