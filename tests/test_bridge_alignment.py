import json
import hashlib

import pytest
import torch
from torch.nn import functional as F

from self_audit.bridge_alignment import (
    BridgeAlignmentArtifact,
    BridgeAlignmentConfig,
    BridgeFitSample,
    fit_bridge_alignment,
    load_bridge_alignment,
)


IDENTITIES = {
    "student_sha256": "a" * 64,
    "teacher_sha256": "b" * 64,
    "train_manifest_sha256": "c" * 64,
    "source_commit": "fixture-source",
}


def _samples(count=2, split="TRAIN"):
    weight = torch.tensor([[2.0, 0.5], [-1.0, 3.0], [0.25, 0.75]]).reshape(3, 2, 1, 1)
    bias = torch.tensor([0.1, -0.2, 0.3])
    result = []
    for index in range(count):
        base = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        feature = torch.cat((torch.sin(base * 0.31 + index * 0.17),
                             torch.cos(base * 0.53 + index * 0.29)), dim=1)
        target = F.conv2d(feature, weight, bias)
        result.append(BridgeFitSample(
            student_h=feature,
            teacher_pre_refine0=target,
            valid_mask=torch.ones((1, 4, 4), dtype=torch.bool),
            patient_id=f"patient-{index}",
            source_id=f"patient-{index}/frame0/slice0",
            split=split,
        ))
    return result, weight, bias


def test_fit_is_train_only_budgeted_hash_bound_and_round_trips(tmp_path):
    samples, expected_weight, expected_bias = _samples()
    config = BridgeAlignmentConfig(max_source_ids=2, max_train_patients=2, max_pixels=32, ridge=1e-10)
    rng_before = torch.get_rng_state().clone()
    artifact_path = tmp_path / "P.alignment.json"
    artifact = fit_bridge_alignment(samples, config, IDENTITIES, output_path=artifact_path)
    assert torch.equal(rng_before, torch.get_rng_state())
    artifact.verify()
    assert artifact.metadata["loss"] == "patient_balanced_pre_refine0_feature_mse"
    assert artifact.metadata["alignment"] == "bilinear_align_corners_false_then_1x1_affine"
    assert artifact.metadata["raw_detail_conditioning"] is False
    assert artifact.metadata["budget"] == {
        "max_source_ids": 2,
        "max_train_patients": 2,
        "max_pixels": 32,
        "seed": 0,
        "used_source_ids": 2,
        "used_train_patients": 2,
        "used_pixels": 32,
    }
    assert artifact.metadata["train_patient_ids"] == ["patient-0", "patient-1"]
    assert artifact.metadata["train_source_ids"] == ["patient-0/frame0/slice0", "patient-1/frame0/slice0"]
    assert artifact.metadata["train_source_order"] == ["patient-0/frame0/slice0", "patient-1/frame0/slice0"]
    assert artifact.metadata["fit_observations"] == [
        {"source_id": "patient-0/frame0/slice0", "patient_id": "patient-0",
         "student_feature_hw": [4, 4], "teacher_feature_hw": [4, 4], "valid_pixels": 16, "used_pixels": 16},
        {"source_id": "patient-1/frame0/slice0", "patient_id": "patient-1",
         "student_feature_hw": [4, 4], "teacher_feature_hw": [4, 4], "valid_pixels": 16, "used_pixels": 16},
    ]
    assert artifact.metadata["fit_metrics"]["fit_pixels"] == 32
    assert artifact.metadata["fit_metrics"]["patient_balanced_feature_mse"] >= 0
    assert set(artifact.metadata["covariance_sha256"]) == {"input", "input_target", "target", "residual"}
    assert torch.allclose(artifact.projector_weight, expected_weight, atol=1e-5, rtol=1e-5)
    assert torch.allclose(artifact.projector_bias, expected_bias, atol=1e-5, rtol=1e-5)

    loaded = load_bridge_alignment(artifact_path, expected_identities=IDENTITIES)
    assert loaded.artifact_sha256 == artifact.artifact_sha256
    assert torch.equal(loaded.projector_weight, artifact.projector_weight)
    assert torch.equal(loaded.projector_bias, artifact.projector_bias)
    with pytest.raises(FileExistsError):
        from self_audit.bridge_alignment import save_bridge_alignment
        save_bridge_alignment(artifact, artifact_path)


def test_pixel_budget_is_exact_and_later_sources_are_not_consumed():
    samples, _, _ = _samples(count=3)
    consumed = []

    def stream():
        for sample in samples:
            consumed.append(sample.source_id)
            yield sample

    artifact = fit_bridge_alignment(
        stream(),
        BridgeAlignmentConfig(max_source_ids=3, max_train_patients=3, max_pixels=21, ridge=1e-4),
        IDENTITIES,
    )
    assert artifact.metadata["budget"]["used_source_ids"] == 2
    assert artifact.metadata["budget"]["used_train_patients"] == 2
    assert artifact.metadata["budget"]["used_pixels"] == 21
    assert consumed == ["patient-0/frame0/slice0", "patient-1/frame0/slice0"]


def test_non_train_duplicate_sources_and_identity_mismatch_fail_closed(tmp_path):
    val_samples, _, _ = _samples(split="VAL")
    config = BridgeAlignmentConfig(max_source_ids=2, max_train_patients=2, max_pixels=32)
    with pytest.raises(ValueError, match="TRAIN samples only"):
        fit_bridge_alignment(val_samples, config, IDENTITIES)

    duplicates, _, _ = _samples()
    duplicates[1] = BridgeFitSample(
        student_h=duplicates[1].student_h,
        teacher_pre_refine0=duplicates[1].teacher_pre_refine0,
        valid_mask=duplicates[1].valid_mask,
        patient_id=duplicates[1].patient_id,
        source_id=duplicates[0].source_id,
        split="TRAIN",
    )
    with pytest.raises(ValueError, match="duplicate bridge fit source_id"):
        fit_bridge_alignment(duplicates, config, IDENTITIES)

    samples, _, _ = _samples()
    artifact_path = tmp_path / "alignment.json"
    fit_bridge_alignment(samples, config, IDENTITIES, output_path=artifact_path)
    wrong_identity = {**IDENTITIES, "student_sha256": "d" * 64}
    with pytest.raises(ValueError, match="identity mismatch: student_sha256"):
        load_bridge_alignment(artifact_path, expected_identities=wrong_identity)

    payload = json.loads(artifact_path.read_text())
    payload["projector_bias"][0] += 1.0
    artifact_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_bridge_alignment(artifact_path)


def test_fit_config_rejects_unlocked_loss_or_bad_budget():
    with pytest.raises(ValueError, match="loss is locked"):
        BridgeAlignmentConfig(max_source_ids=4, max_train_patients=2, max_pixels=50,
                              fit_loss="arbitrary").validate()
    with pytest.raises(ValueError, match="at least two TRAIN patients"):
        BridgeAlignmentConfig(max_source_ids=1, max_train_patients=1, max_pixels=50).validate()


def test_verify_rejects_semantically_invalid_artifact_even_with_recomputed_self_hash():
    import self_audit.bridge_alignment as module

    samples, _, _ = _samples()
    artifact = fit_bridge_alignment(
        samples,
        BridgeAlignmentConfig(max_source_ids=2, max_train_patients=2, max_pixels=32),
        IDENTITIES,
    )
    forged_metadata = dict(artifact.metadata)
    forged_metadata["fit_split"] = "VAL"
    forged_payload = module._artifact_payload(
        forged_metadata, artifact.projector_weight, artifact.projector_bias,
    )
    forged_sha = hashlib.sha256(module._canonical(forged_payload)).hexdigest()
    forged = BridgeAlignmentArtifact(
        metadata=forged_metadata,
        projector_weight=artifact.projector_weight,
        projector_bias=artifact.projector_bias,
        artifact_sha256=forged_sha,
    )
    with pytest.raises(ValueError, match="locked TRAIN-only fit graph"):
        forged.verify()
