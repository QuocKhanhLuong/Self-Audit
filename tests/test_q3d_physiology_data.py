from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
from self_audit.candidate_q3d.augmentation import (
    apply_spatial_transform,
    augment_unlabeled,
    sample_spatial_transform,
    strong_intensity,
)
from self_audit.candidate_q3d.data import Q3DData, Q3DJointSampler
from self_audit.candidate_q3d.physiology import (
    load_physiology_cache,
    patient_physiology_from_probabilities,
    physiology_cell_weights,
    physiology_cache_needs_refresh,
    physiology_dev_diagnostic,
    physiology_weight_for_patient,
    save_physiology_cache,
    summarize_physiology_cohort,
)
from scripts.q3d_physio_cache import build_cache


torch.set_num_threads(1)


class _CacheTestTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.tensor(1.0))

    def forward(self, image, *, geometry):
        output = torch.zeros((1, 4, *image.shape[-3:]), dtype=image.dtype, device=image.device)
        output[:, 0] = self.bias
        output[:, 3] = -self.bias
        return {"region_logits": output, "boundary_logits": output, "sdf": output}


def _cache_test_factory(*, config, checkpoint_path, expected_sha256, device):
    assert config["runtime"]["synthetic"] is True
    assert checkpoint_path is None and expected_sha256 is None
    return _CacheTestTeacher()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _student_data(folder: Path, *, n_labeled=2, n_unlabeled=2, shape=(2, 4, 8, 9)):
    fixture = synthetic_fixture(
        folder, n_labeled=n_labeled, n_unlabeled=n_unlabeled,
        n_evaluation=2, shape_tzyx=shape,
    )
    return fixture, CandidateA3DData(fixture["train"], fixture["train_sha256"], synthetic=True)


def _all_labeled_teacher_view(folder: Path, student_data: CandidateA3DData) -> CandidateA3DData:
    records = [dict(record) for record in student_data.labeled]
    for source in student_data.unlabeled:
        record = dict(source)
        cine = student_data.base.array(source["image"], source["image_sha256"])
        masks = {}
        for frame in range(cine.shape[0]):
            mask_path = folder / f"{source['id']}_teacher_t{frame:03d}_mask.npy"
            np.save(mask_path, np.zeros(cine.shape[1:], dtype=np.uint8))
            masks[str(frame)] = {"path": mask_path.name, "sha256": _sha(mask_path)}
        record["masks"] = masks
        records.append(record)
    manifest = dict(student_data.manifest)
    manifest["labeled"] = records
    manifest["unlabeled"] = []
    manifest["label_budget"] = len(records)
    teacher_manifest = folder / "teacher_all_labeled.json"
    teacher_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return CandidateA3DData(teacher_manifest, _sha(teacher_manifest), synthetic=True)


def _runtime_geometry(shape=(24, 48, 64), spacing=(2.0, 1.0, 1.0)):
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag(spacing)
    return {
        "shape": tuple(shape),
        "spacing_zyx": tuple(spacing),
        "affine_zyx": affine.tolist(),
        "context": "full_volume",
    }


def _ellipsoid(shape, center, radii):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return (((zz - center[0]) / radii[0]) ** 2
            + ((yy - center[1]) / radii[1]) ** 2
            + ((xx - center[2]) / radii[2]) ** 2) <= 1.0


def _analytic_cine():
    geometry = _runtime_geometry()
    shape = geometry["shape"]
    chamber_radii = ((9, 12, 12), (5, 7, 7), (7, 10, 10))
    probabilities = np.zeros((3, 4, *shape), dtype=np.float32)
    probabilities[:, 0] = 1.0
    myo = _ellipsoid(shape, (12, 5, 32), (2, 3, 3))
    for frame, radii in enumerate(chamber_radii):
        for class_id, center in ((1, (12, 24, 48)), (3, (12, 24, 16))):
            region = _ellipsoid(shape, center, radii)
            probabilities[frame, 0, region] = 0.0
            probabilities[frame, class_id, region] = 1.0
        probabilities[frame, 0, myo] = 0.0
        probabilities[frame, 2, myo] = 1.0
    return probabilities, geometry, chamber_radii, myo


def _minimal_cache_rows(weights):
    return [
        {
            "id": patient_id,
            "protocol": "A",
            "teacher_sha256": "a" * 64,
            "codebook_sha256": None,
            "w_pat": weight,
        }
        for patient_id, weight in weights
    ]


def _minimal_cache_source(rows):
    return {
        "protocol": "A",
        "train_manifest_sha256": "c" * 64,
        "unlabeled_patients": [
            {"id": row["id"], "image_sha256": f"{index + 1:064x}"}
            for index, row in enumerate(rows)
        ],
    }


def _reseal_cache(document):
    payload = document["payload"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_analytic_ellipsoid_volumes_select_predicted_ed_es_and_weight_one():
    probabilities, geometry, radii, myo = _analytic_cine()
    row = patient_physiology_from_probabilities(
        probabilities, geometry, patient_id="U-ELLIPSOID",
        teacher_sha256="a" * 64, codebook_sha256="b" * 64,
        protocol="A", config={"physiology": {
            "stroke_balance": True, "tau_stroke_volume": 0.15, "tau_myocardium": 0.10,
        }},
    )
    voxel_ml = 2.0 / 1000.0
    expected_ed = float(_ellipsoid(geometry["shape"], (12, 24, 16), radii[0]).sum()) * voxel_ml
    expected_es = float(_ellipsoid(geometry["shape"], (12, 24, 16), radii[1]).sum()) * voxel_ml
    assert row["ed_index"] == 0
    assert row["es_index"] == 1
    assert row["volumes_ml_by_frame"]["lv"][0] == pytest.approx(expected_ed)
    assert row["volumes_ml_by_frame"]["lv"][1] == pytest.approx(expected_es)
    assert row["volumes_ml_by_frame"]["myo"] == pytest.approx([float(myo.sum()) * voxel_ml] * 3)
    assert row["r_sv"] == pytest.approx(0.0)
    assert row["r_m"] == pytest.approx(0.0)
    assert row["w_pat"] == pytest.approx(1.0)
    assert row["guard_reasons"] == []


def test_physiology_guards_and_stroke_balance_dataset_switch():
    probabilities, geometry, _, _ = _analytic_cine()
    one_frame = probabilities[:1]
    guarded = patient_physiology_from_probabilities(
        one_frame, geometry, patient_id="U-GUARD", teacher_sha256="a" * 64,
        protocol="A", config={"physio": {"stroke_balance": False, "has_rv": False}},
    )
    assert guarded["lv_stroke_volume_ml"] == 0.0
    assert guarded["r_sv"] is None
    assert guarded["w_pat"] == 0.0
    assert "lv_stroke_volume_nonpositive" in guarded["guard_reasons"]

    disabled = patient_physiology_from_probabilities(
        probabilities, geometry, patient_id="U-NO-RV", teacher_sha256="a" * 64,
        protocol="B", config={"physio": {"stroke_balance": False, "has_rv": False}},
    )
    assert disabled["r_sv"] is None
    assert disabled["w_pat"] == pytest.approx(1.0)
    inferred_no_rv = patient_physiology_from_probabilities(
        probabilities, geometry, patient_id="U-MISSING-RV", teacher_sha256="a" * 64,
        protocol="B", config={
            "dataset": {"class_mapping": ["BG", "MYO", "LV"]},
            "physiology": {"stroke_balance": False},
        },
    )
    assert inferred_no_rv["r_sv"] is None
    assert inferred_no_rv["w_pat"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="requires an RV"):
        patient_physiology_from_probabilities(
            probabilities, geometry, patient_id="U-BAD", teacher_sha256="a" * 64,
            protocol="A", config={"physio": {"stroke_balance": True, "has_rv": False}},
        )


def test_cell_confidence_averages_native_voxels_then_applies_threshold_and_patient_weight():
    maxima = torch.tensor([0.8, 0.9, 0.6, 0.8], dtype=torch.float32)
    favored = torch.log(3.0 * maxima / (1.0 - maxima))
    logits = torch.zeros((1, 4, 1, 1, 4), dtype=torch.float32)
    logits[:, 0, 0, 0, :] = favored
    logits.requires_grad_()
    weights, metrics = physiology_cell_weights(
        logits, [0.5], cell_shape_zyx=(1, 1, 2), threshold=0.71,
    )
    assert tuple(weights.shape) == (1, 1, 1, 1, 2)
    assert weights[0, 0, 0, 0, 0].item() == pytest.approx(0.425, abs=1e-5)
    assert weights[0, 0, 0, 0, 1].item() == 0.0
    assert not weights.requires_grad
    assert metrics["fraction_cells_above_threshold"] == pytest.approx(0.5)


def test_dev_spearman_is_report_only_and_keeps_patient_pairing():
    report = physiology_dev_diagnostic([
        {"id": "p1", "w_pat": 0.0, "rv_dice": 0.4, "rv_hd95": 20.0},
        {"id": "p2", "w_pat": 0.5, "rv_dice": 0.5, "rv_hd95": 15.0},
        {"id": "p3", "w_pat": 1.0, "rv_dice": 0.7, "rv_hd95": 5.0},
    ])
    assert report["status"] == "REPORT_ONLY"
    assert report["patient_count"] == 3
    assert report["spearman_w_pat_rv_dice"] == pytest.approx(1.0)
    assert report["spearman_w_pat_rv_hd95"] == pytest.approx(-1.0)
    assert report["used_for_tau_selection"] is False


def test_cache_hash_binds_protocol_teacher_codebook_tau_and_source(tmp_path):
    probabilities, geometry, _, _ = _analytic_cine()
    config = {"physio": {"stroke_balance": True, "tau_sv": 0.15, "tau_m": 0.1}}
    source = {"protocol": "A", "train_manifest_sha256": "c" * 64,
              "unlabeled_patients": [{"id": "U-ELLIPSOID", "image_sha256": "d" * 64}]}
    row = patient_physiology_from_probabilities(
        probabilities, geometry, patient_id="U-ELLIPSOID", teacher_sha256="a" * 64,
        codebook_sha256=None, protocol="A", config=config,
    )
    target = tmp_path / "A" / "physiology.json"
    checksum = save_physiology_cache(
        target, [row], teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", config=config, source_identity=source,
    )
    loaded = load_physiology_cache(
        target, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", config=config, source_identity=source,
    )
    assert loaded["sha256"] == checksum
    assert loaded["payload"]["patients"][0]["codebook_sha256"] is None
    assert not physiology_cache_needs_refresh(
        loaded, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", epoch=3, config=config, source_identity=source,
    )
    assert physiology_cache_needs_refresh(
        loaded, teacher_sha256="e" * 64, codebook_sha256=None,
        protocol="A", epoch=3, config=config, source_identity=source,
    )
    assert physiology_cache_needs_refresh(
        loaded, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", epoch=10, online_teacher=True, refresh_interval=10,
        config=config, source_identity=source,
    )
    with pytest.raises(ValueError, match="source identity protocol"):
        load_physiology_cache(
            target, teacher_sha256="a" * 64, codebook_sha256=None,
            protocol="B", config=config, source_identity=source,
        )
    with pytest.raises(ValueError, match="tau/gate"):
        load_physiology_cache(
            target, teacher_sha256="a" * 64, codebook_sha256=None,
            protocol="A", config={"physio": {"tau_sv": 0.2}}, source_identity=source,
        )
    with pytest.raises(ValueError, match="source manifest/cohort"):
        load_physiology_cache(
            target, teacher_sha256="a" * 64, codebook_sha256=None,
            protocol="A", config=config, source_identity={"protocol": "A", "unlabeled_patients": [
                {"id": "OTHER", "image_sha256": "d" * 64},
            ]},
        )
    doc = json.loads(target.read_text())
    doc["payload"]["protocol"] = "B"
    target.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_physiology_cache(
            target, teacher_sha256="a" * 64, codebook_sha256=None,
            protocol="A", config=config, source_identity=source,
        )


def test_cache_persists_unequal_u_weight_mean_ess_and_gate(tmp_path):
    rows = _minimal_cache_rows([("U-1", 1.0), ("U-2", 0.5), ("U-3", 0.25)])
    source = _minimal_cache_source(rows)
    config = {"physio": {"stroke_balance": True, "tau_sv": 0.15, "tau_m": 0.1}}
    target = tmp_path / "unequal" / "physiology.json"
    save_physiology_cache(
        target, rows, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", config=config, source_identity=source,
    )

    document = json.loads(target.read_text())
    metadata = document["payload"]["cohort_weighting"]
    expected = summarize_physiology_cohort(rows)
    assert metadata == expected
    assert metadata["patient_ids"] == ["U-1", "U-2", "U-3"]
    assert metadata["count"] == 3
    assert metadata["mean_w_pat"] == pytest.approx(1.75 / 3.0)
    expected_ess = 1.75 ** 2 / (1.0 ** 2 + 0.5 ** 2 + 0.25 ** 2)
    assert metadata["ess"] == pytest.approx(expected_ess)
    assert metadata["ess_fraction"] == pytest.approx(expected_ess / 3.0)
    assert metadata["gate_passed"] is True
    loaded = load_physiology_cache(
        target, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", config=config, source_identity=source,
    )
    assert loaded["payload"]["cohort_weighting"] == metadata


def test_cache_writes_and_loads_all_zero_u_cohort_for_audit(tmp_path):
    rows = _minimal_cache_rows([("U-1", 0.0), ("U-2", 0.0)])
    source = _minimal_cache_source(rows)
    config = {"physio": {"stroke_balance": True, "tau_sv": 0.15, "tau_m": 0.1}}
    target = tmp_path / "all-zero" / "physiology.json"
    checksum = save_physiology_cache(
        target, rows, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", config=config, source_identity=source,
    )
    document = json.loads(target.read_text())
    metadata = document["payload"]["cohort_weighting"]
    assert checksum == document["sha256"]
    assert metadata["mean_w_pat"] == 0.0
    assert metadata["ess"] == 0.0
    assert metadata["ess_fraction"] == 0.0
    assert metadata["gate_passed"] is False
    loaded = load_physiology_cache(
        target, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", config=config, source_identity=source,
    )
    assert loaded["payload"]["cohort_weighting"]["gate_passed"] is False
    assert not physiology_cache_needs_refresh(
        loaded, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", epoch=0, config=config, source_identity=source,
    )


def test_cache_rejects_resealed_forged_cohort_metadata(tmp_path):
    rows = _minimal_cache_rows([("U-1", 1.0), ("U-2", 0.5)])
    source = _minimal_cache_source(rows)
    config = {"physio": {"stroke_balance": True, "tau_sv": 0.15, "tau_m": 0.1}}
    target = tmp_path / "forged" / "physiology.json"
    save_physiology_cache(
        target, rows, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", config=config, source_identity=source,
    )
    document = json.loads(target.read_text())
    document["payload"]["cohort_weighting"]["mean_w_pat"] += 0.125
    _reseal_cache(document)
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="mean_w_pat.*cached rows"):
        load_physiology_cache(
            target, teacher_sha256="a" * 64, codebook_sha256=None,
            protocol="A", config=config, source_identity=source,
        )


def test_cache_rejects_resealed_stale_format_without_cohort_metadata(tmp_path):
    rows = _minimal_cache_rows([("U-1", 1.0), ("U-2", 0.5)])
    source = _minimal_cache_source(rows)
    config = {"physio": {"stroke_balance": True, "tau_sv": 0.15, "tau_m": 0.1}}
    target = tmp_path / "stale" / "physiology.json"
    save_physiology_cache(
        target, rows, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", config=config, source_identity=source,
    )
    document = json.loads(target.read_text())
    del document["payload"]["cohort_weighting"]
    _reseal_cache(document)
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="cohort_weighting metadata is missing.*regenerate"):
        load_physiology_cache(
            target, teacher_sha256="a" * 64, codebook_sha256=None,
            protocol="A", config=config, source_identity=source,
        )
    assert physiology_cache_needs_refresh(
        document, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", epoch=0, config=config, source_identity=source,
    )


def test_cache_binds_existing_image_only_u_cohort_exactly(tmp_path):
    fixture, data = _student_data(tmp_path / "existing-u", n_labeled=2, n_unlabeled=3)
    rows = _minimal_cache_rows([(patient_id, 0.4 + index * 0.1)
                                for index, patient_id in enumerate(data.uids)])
    source = Q3DData(data, protocol="A").cache_source_identity()
    config = {"physio": {"stroke_balance": True, "tau_sv": 0.15, "tau_m": 0.1}}
    target = tmp_path / "existing-u" / "physiology.json"
    save_physiology_cache(
        target, rows, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", config=config, source_identity=source,
    )
    loaded = load_physiology_cache(
        target, teacher_sha256="a" * 64, codebook_sha256=None,
        protocol="A", config=config, source_identity=source,
    )
    assert loaded["payload"]["cohort_weighting"]["patient_ids"] == sorted(data.uids)
    assert loaded["payload"]["cohort_weighting"]["count"] == len(data.uids)
    assert {row["id"] for row in loaded["payload"]["patients"]} == set(data.uids)
    assert set(fixture["train_ids"]) >= set(data.uids)


def test_cache_cli_builds_a_marked_synthetic_snapshot_without_patient_claims(tmp_path):
    fixture, _ = _student_data(tmp_path / "fixture", shape=(2, 4, 8, 9))
    config = {
        "dataset": {"manifests": {"train": {
            "path": str(fixture["train"]), "sha256": fixture["train_sha256"],
        }}},
        "protocol": {"variant": "A"},
        "arm": "c1",
        "runtime": {"synthetic": True},
        "physiology": {"enabled": True, "stroke_balance": False, "has_rv": False},
        "tokenizer": {"enabled": False},
    }
    config_path = tmp_path / "q3d.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "cache" / "physiology.json"
    summary = build_cache(
        config, config_path=config_path,
        teacher_factory="tests.test_q3d_physiology_data:_cache_test_factory",
        output_path=output, device="cpu",
    )
    assert summary["status"] == "SOFTWARE_CACHE_WRITTEN"
    assert summary["teacher_snapshot_kind"] == "synthetic_initialized_state"
    assert len(summary["teacher_sha256"]) == 64
    document = json.loads(output.read_text())
    assert document["payload"]["protocol"] == "A"
    assert len(document["payload"]["patients"]) == len(fixture["train_ids"]) - 2


def test_cache_cli_default_synthetic_teacher_path_is_explicit_and_runs_cpu(tmp_path):
    fixture, _ = _student_data(tmp_path / "fixture", shape=(2, 4, 8, 9))
    config = {
        "dataset": {"manifests": {"train": {
            "path": str(fixture["train"]), "sha256": fixture["train_sha256"],
        }}},
        "protocol": {"variant": "B"},
        "arm": "c1",
        "runtime": {"synthetic": True},
        "physiology": {"enabled": True, "stroke_balance": False, "has_rv": False},
        "tokenizer": {"enabled": False},
        "model": {"teacher": {
            "architecture": "nnfoundation_resenc_l",
            "heads": ["region", "boundary_sdf"],
            "feature_stride": 8,
            "feature_source": "encoder_stride8",
            "feature_channels": 256,
            "encoder_strides": [
                [1, 1, 1], [2, 2, 2], [1, 2, 2],
                [1, 2, 2], [1, 2, 2], [1, 2, 2],
            ],
            "phase_offsets": {
                "encoder_stride8": [0.0, 0.0, 0.0],
                "encoder_stride16": [0.0, 0.0, 0.0],
            },
            "encoder_channels": [32, 64, 128, 256, 320, 320],
            "encoder_blocks": [1, 3, 4, 6, 6, 6],
            "dependency_pins": {
                "dynamic_network_architectures": "0.4.4",
                "nnssl_commit": "3d774e942256bd70e5e264c6b94fa38e25f7ed74",
                "nnunet_commit": "c04f7ec8bbffaf3ebc3b26aa2c89042e8a8e368b",
            },
            "freeze_joint": True,
            "online_joint": False,
            "shared_trunk": True,
            "pretrained": {},
        }},
    }
    config_path = tmp_path / "q3d_synthetic.json"
    config_path.write_text(json.dumps(config))
    summary = build_cache(
        config, config_path=config_path, teacher_factory=None,
        output_path=tmp_path / "cache_b" / "physiology.json", device="cpu",
    )
    assert summary["status"] == "SOFTWARE_CACHE_WRITTEN"
    assert summary["protocol"] == "B"
    assert summary["device"] == "cpu"
    assert summary["synthetic_fixture"] is True


def test_image_only_teacher_cache_visits_cine_frames_without_label_reads(tmp_path):
    _, source = _student_data(tmp_path / "fixture", shape=(3, 4, 8, 9))
    patient_id = source.uids[0]

    class ConstantTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, image, *, geometry):
            self.calls.append((tuple(image.shape), geometry["context"]))
            output = torch.zeros((1, 4, *image.shape[-3:]), dtype=image.dtype, device=image.device)
            output[:, 0] = 2.0
            return {"region_logits": output, "boundary_logits": output, "sdf": output,
                    "feature": torch.ones((1, 2, 1, 1, 1), device=image.device)}

    teacher = ConstantTeacher()
    before = set(source.label_reads)
    row = physiology_weight_for_patient(
        teacher, source, patient_id, teacher_sha256="a" * 64,
        codebook_sha256=None, protocol="A", config={"physio": {"stroke_balance": False}},
    )
    assert len(teacher.calls) == 3
    assert all(call == ((1, 1, 4, 8, 9), "full_volume") for call in teacher.calls)
    assert row["frame_indices"] == [0, 1, 2]
    assert source.label_reads == before
    assert source.unlabeled_image_reads == {patient_id}
    assert set(source.unlabeled[0]) == {"id", "image", "image_sha256"}


def test_protocol_a_b_label_sets_and_resumable_uniform_patient_sampler(tmp_path):
    fixture, student_data = _student_data(tmp_path / "fixture", shape=(2, 4, 8, 9))
    protocol_a = Q3DData(student_data, protocol="A")
    assert set(protocol_a.teacher_patients) == set(protocol_a.labeled_patients)
    assert set(protocol_a.unlabeled_patients).isdisjoint(protocol_a.teacher_patients)
    with pytest.raises(ValueError, match="requires a separate all-TRAIN"):
        Q3DData(student_data, protocol="B")

    teacher_data = _all_labeled_teacher_view(tmp_path / "fixture", student_data)
    protocol_b = Q3DData(student_data, protocol="B", teacher_data=teacher_data)
    assert set(protocol_b.teacher_patients) == set(protocol_b.train_patients)
    assert set(protocol_b.labeled_patients) < set(protocol_b.teacher_patients)
    assert protocol_b.teacher_frame_keys()
    assert protocol_b.unlabeled_frame_keys()

    reads_before = set(student_data.label_reads)
    image, geometry, metadata = protocol_b.load_unlabeled_frame(protocol_b.unlabeled_frame_keys()[0])
    assert tuple(image.shape) == (1, 1, 4, 8, 9)
    assert geometry["context"] == "full_volume"
    assert set(metadata) == {"patient", "frame"}
    assert student_data.label_reads == reads_before

    sampler = Q3DJointSampler(protocol_a, seed=42)
    for _ in range(20):
        sampler.sample_update()
    state = sampler.state_dict()
    expected = [sampler.sample_update() for _ in range(20)]
    resumed = Q3DJointSampler(protocol_a, seed=42)
    resumed.load_state_dict(state)
    actual = [resumed.sample_update() for _ in range(20)]
    assert actual == expected
    counts = {patient: 0 for patient in protocol_a.labeled_patients}
    for _ in range(1000):
        counts[sampler.sample_update()["labeled_patient"]] += 1
    assert all(abs(count / 1000 - 0.5) < 0.08 for count in counts.values())
    incompatible = Q3DJointSampler(protocol_b, seed=42)
    with pytest.raises(ValueError, match="another protocol or label subset"):
        incompatible.load_state_dict(state)


def test_spatial_flips_rotation_and_geometry_are_world_coordinate_exact():
    geometry = _runtime_geometry(shape=(3, 6, 8), spacing=(2.0, 1.0, 1.0))
    coordinate_ids = torch.arange(3 * 6 * 8, dtype=torch.float32).reshape(1, 1, 3, 6, 8)
    original = coordinate_ids / float(coordinate_ids.max())
    mask = (coordinate_ids[:, 0].long() % 4).to(torch.uint8)
    for flip_y, flip_x, rot90_k in ((True, False, 0), (False, True, 0), (True, True, 0),
                                    (False, False, 1), (False, False, 2), (False, False, 3)):
        transformed, transformed_mask, transformed_geometry = apply_spatial_transform(
            original, mask, geometry, flip_y=flip_y, flip_x=flip_x, rot90_k=rot90_k,
        )
        assert transformed_mask is not None
        assert tuple(transformed.shape[-3:]) == tuple(transformed_geometry["shape"])
        assert tuple(transformed_mask.shape[-3:]) == tuple(transformed_geometry["shape"])
        for q in ((0, 0, 0), (1, transformed_geometry["shape"][1] - 1,
                               transformed_geometry["shape"][2] - 1)):
            value = transformed[0, 0, *q].item()
            old_flat = round(value * float(coordinate_ids.max()))
            old_z = old_flat // (6 * 8)
            old_y = (old_flat % (6 * 8)) // 8
            old_x = old_flat % 8
            old_world = np.asarray(geometry["affine_zyx"]) @ np.asarray([old_z, old_y, old_x, 1.0])
            new_world = np.asarray(transformed_geometry["affine_zyx"]) @ np.asarray([*q, 1.0])
            assert new_world == pytest.approx(old_world)

    anisotropic = _runtime_geometry(shape=(3, 6, 8), spacing=(2.0, 0.8, 1.3))
    with pytest.raises(ValueError, match="spacing_y and spacing_x"):
        apply_spatial_transform(original, mask, anisotropic, rot90_k=1)
    transform = sample_spatial_transform(
        anisotropic, rng=__import__("random").Random(19), rotation_probability=1.0,
    )
    assert transform["rot90_k"] == 0


def test_unlabeled_views_share_spatial_geometry_and_intensity_is_reproducible():
    geometry = _runtime_geometry(shape=(2, 6, 8), spacing=(2.0, 1.0, 1.0))
    image = torch.full((1, 1, 2, 6, 8), 0.5)
    weak1, strong1, geom1 = augment_unlabeled(
        image, geometry, rng=__import__("random").Random(7),
        intensity_generator=torch.Generator().manual_seed(13),
    )
    weak2, strong2, geom2 = augment_unlabeled(
        image, geometry, rng=__import__("random").Random(7),
        intensity_generator=torch.Generator().manual_seed(13),
    )
    assert torch.equal(weak1, weak2)
    assert torch.equal(strong1, strong2)
    assert geom1 == geom2
    assert not torch.equal(weak1, strong1)
    assert strong1.min() >= 0 and strong1.max() <= 1
    intensity1 = strong_intensity(image, generator=torch.Generator().manual_seed(5))
    intensity2 = strong_intensity(image, generator=torch.Generator().manual_seed(5))
    assert torch.equal(intensity1, intensity2)
    assert torch.isfinite(intensity1).all()
