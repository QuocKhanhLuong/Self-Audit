from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts.q3d_export import export as export_and_verify
from scripts.q3d_infer import infer
from scripts.q3d_profile import profile_artifact
from self_audit.candidate_q3d.evaluation import (
    codebook_diagnostics,
    compare_primary_arms,
    evaluate_manifest,
    physiology_spearman,
    surface_distances_3d,
    validate_native_image,
)
from self_audit.candidate_q3d.export import (
    _read_verification_volume,
    export_checkpoint,
    load_export,
    verify_cold_load,
)
from self_audit.candidate_q3d.student import CandidateQ3DStudent
from self_audit.data import CLASS_MAPPING, GEOMETRY_MAP_SCHEMA, GEOMETRY_SCHEMA, digest


def _runtime_geometry(shape: tuple[int, int, int]) -> dict:
    spacing = (8.0, 1.25, 1.5)
    affine = np.diag([*spacing, 1.0]).tolist()
    return {"shape": shape, "spacing_zyx": spacing, "affine_zyx": affine,
            "context": "full_volume"}


def _cache_geometry(shape: tuple[int, int, int]) -> dict:
    spacing = (8.0, 1.25, 1.5)
    return {
        "schema": GEOMETRY_SCHEMA,
        "array_axes": ["T", "Z", "Y", "X"],
        "spatial_axes": ["Z", "Y", "X"],
        "shape_zyx": list(shape),
        "spacing_zyx_mm": list(spacing),
        "spacing_yx_mm": list(spacing[1:]),
        "affine_zyx": np.diag([*spacing, 1.0]).tolist(),
        "class_mapping": dict(CLASS_MAPPING),
    }


class _EchoLabelStudent(nn.Module):
    codebook_size = 4

    def forward(self, image: torch.Tensor, geometry=None) -> dict:
        if tuple(image.shape[-3:]) != tuple(geometry["shape"]):
            raise ValueError("test model received mismatched geometry")
        label = torch.round(image * 3.0).long().clamp(0, 3)
        logits = F.one_hot(label[:, 0], num_classes=4).permute(0, 4, 1, 2, 3).float() * 20.0
        depth, height, width = image.shape[-3:]
        codes = torch.zeros((1, depth, (height + 15) // 16, (width + 15) // 16), dtype=torch.long)
        return {"logits": logits, "codes": codes}

    def infer_logits(self, image: torch.Tensor, geometry=None) -> torch.Tensor:
        return self.forward(image, geometry)["logits"]


class _FailingStudent(_EchoLabelStudent):
    def forward(self, image: torch.Tensor, geometry=None) -> dict:
        raise RuntimeError("synthetic inference failure")


def _write_manifest(tmp_path: Path, *, role: str = "development") -> tuple[Path, str, dict]:
    shape = (6, 17, 19)
    frames = np.zeros((2, *shape), dtype=np.uint8)
    frames[0, 1:4, 3:8, 3:8] = 1
    frames[0, 1:4, 9:12, 9:13] = 2
    frames[0, 2:5, 13:16, 4:7] = 3
    frames[1] = frames[0]
    image_path = tmp_path / "cine.npy"
    np.save(image_path, frames.astype(np.float32) / 3.0, allow_pickle=False)
    mask_records = {}
    for frame in (0, 1):
        mask_path = tmp_path / f"mask_{frame}.npy"
        np.save(mask_path, frames[frame], allow_pickle=False)
        mask_records[str(frame)] = {"path": mask_path.name, "sha256": digest(mask_path)}
    patient = {"id": "P001", "image": image_path.name, "image_sha256": digest(image_path),
               "masks": mask_records, "phases": {"ED": 0, "ES": 1}}
    manifest = {
        "schema": "self_audit.data.v2", "role": role, "synthetic": True,
        "dataset": "q3d-synthetic-fixture", "class_mapping": dict(CLASS_MAPPING),
        "geometry": {"schema": GEOMETRY_MAP_SCHEMA, "by_patient": {"P001": _cache_geometry(shape)}},
        "patients": [patient],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, digest(manifest_path), manifest


def _small_q3d_config() -> dict:
    return {
        "schema": "q3d.config.v1",
        "arm": "V1",
        "model": {"student": {
            "channels": [4, 4, 4, 4, 4],
            "in_channels": 1,
            "num_classes": 4,
            "attention": {"support": "uniform"},
            "vq": {"size": 4, "dim": 4, "tau": 0.1},
            "residual_vq": {"enabled": False},
        }},
        "tokenizer": {"codebook_size": 4, "codebook_dim": 4, "tau": 0.1},
    }


def _frozen_codebook(codes: torch.Tensor, version: str = "q3d_teacher_codebook.v1") -> dict:
    value = codes.detach().to(dtype=torch.float32).contiguous().cpu()
    return {
        "codes": value,
        "version": version,
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "size": int(value.shape[0]),
        "dimension": int(value.shape[1]),
        "initialized": True,
        "frozen": True,
    }


def _write_input_npz(path: Path, shape: tuple[int, int, int], seed: int) -> Path:
    rng = np.random.default_rng(seed)
    np.savez_compressed(path, image=rng.random(shape, dtype=np.float32),
                        shape_zyx=np.asarray(shape, dtype=np.int32),
                        spacing_zyx=np.asarray((8.0, 1.25, 1.5)),
                        affine_zyx=np.diag([8.0, 1.25, 1.5, 1.0]),
                        context=np.asarray("full_volume"))
    return path


def test_native_surface_metrics_use_zyx_mm_and_report_absence() -> None:
    reference = np.zeros((8, 8, 8), dtype=bool)
    prediction = reference.copy()
    reference[2:5, 2:5, 2:5] = True
    prediction[3:6, 2:5, 2:5] = True
    result = surface_distances_3d(reference, prediction, (4.0, 1.0, 1.0))
    assert result["status"] == "ok"
    assert result["hd95_mm"] == pytest.approx(4.0)
    assert result["assd_mm"] > 0
    absent = surface_distances_3d(np.zeros_like(reference), np.zeros_like(reference), (4, 1, 1))
    assert absent["status"] == "both_absent"
    assert absent["hd95_mm"] is None


def test_evaluation_manifest_freezes_predictions_then_scores_clinical_and_code_metrics(tmp_path: Path) -> None:
    manifest, manifest_sha, _ = _write_manifest(tmp_path)
    code_map = {("P001", 0): np.zeros((6, 2, 2), dtype=np.int64),
                ("P001", 1): np.zeros((6, 2, 2), dtype=np.int64)}
    eval_dir = tmp_path.parent / f"{tmp_path.name}-eval"
    student = _EchoLabelStudent().train()
    result = evaluate_manifest(
        student, manifest, manifest_sha, eval_dir,
        device="cpu", split="development", train_patient_ids=["TRAIN_01"],
        expected_synthetic=True, teacher_codes_by_frame=code_map,
        physiology_rows=[{"w_pat": 0.2, "rv_dice": 0.5},
                         {"w_pat": 0.6, "rv_dice": 0.8},
                         {"w_pat": 0.9, "rv_dice": 1.0}],
    )
    assert student.training is True
    assert result["evidence_status"] == "SOFTWARE_ONLY_SYNTHETIC"
    assert result["patient_macro"]["fg_dice"] == pytest.approx(1.0)
    assert result["patient_macro"]["hd95_mm"] == pytest.approx(0.0)
    assert result["patient_macro"]["clinical_abs_error"]["LV_EDV_abs_error_ml"]["patient_macro_mae"] == 0.0
    assert result["codebook"]["teacher_student_agreement"] == 1.0
    assert result["physiology_w_pat_spearman"]["rho"] == pytest.approx(1.0)
    frozen = json.loads((eval_dir / "PREDICTIONS_FROZEN.json").read_text())
    assert len(frozen["predictions"]) == 2
    assert (eval_dir / "Q3D_METRICS.json").is_file()


def test_manifest_evaluation_fails_closed_on_train_overlap_and_geometry(tmp_path: Path) -> None:
    manifest_path, manifest_sha, _ = _write_manifest(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_manifest(_EchoLabelStudent(), manifest_path, manifest_sha, tmp_path / "overlap",
                          train_patient_ids=["P001"])
    bad = _runtime_geometry((6, 17, 18))
    with pytest.raises(ValueError, match="full-volume"):
        validate_native_image(torch.zeros((1, 1, 6, 17, 19)), bad)
    bad = _runtime_geometry((6, 17, 19))
    bad["spacing_zyx"] = (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="axis lengths"):
        validate_native_image(torch.zeros((1, 1, 6, 17, 19)), bad)


def test_manifest_evaluation_restores_training_mode_after_inference_error(tmp_path: Path) -> None:
    manifest_path, manifest_sha, _ = _write_manifest(tmp_path)
    student = _FailingStudent().train()
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        evaluate_manifest(
            student, manifest_path, manifest_sha, tmp_path.parent / f"{tmp_path.name}-failure",
            expected_synthetic=True,
        )
    assert student.training is True


def test_codebook_and_dev_physiology_diagnostics() -> None:
    diag = codebook_diagnostics(np.asarray([0, 0, 1, 1]), codebook_size=4,
                                teacher_codes=np.asarray([0, 1, 1, 1]))
    assert diag["perplexity"] == pytest.approx(2.0)
    assert diag["dead_code_count_below_0_1_percent"] == 2
    assert diag["teacher_student_agreement"] == pytest.approx(0.75)
    corr = physiology_spearman(
        [{"w_pat": 0.2, "rv_dice": 0.4}, {"w_pat": 0.5, "rv_dice": 0.7},
         {"w_pat": 0.9, "rv_dice": 0.95}], split="development"
    )
    assert corr["rho"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="DEV-only"):
        physiology_spearman([], split="test")


def test_primary_arm_gates_use_paired_patient_thresholds_and_no_go() -> None:
    def rows(values, hd95, one_absent=None):
        counts = one_absent if one_absent is not None else [0] * len(values)
        return {f"P{i}": {"fg_dice": d, "hd95_mm": h, "one_absent_count": count}
                for i, (d, h, count) in enumerate(zip(values, hd95, counts))}
    arms = {
        "C0": rows([0.70, 0.75, 0.80], [10.0, 9.0, 8.0]),
        "C1": rows([0.72, 0.76, 0.81], [8.0, 8.0, 7.0]),
        "V1": rows([0.74, 0.78, 0.83], [7.0, 7.0, 6.0]),
        "V1-U": rows([0.73, 0.77, 0.82], [7.5, 7.5, 6.5]),
    }
    result = compare_primary_arms(arms, split="test", seed=19, bootstrap_samples=500)
    assert result["decisions"] == {"vq_contribution": "GO", "teacher_contribution": "GO",
                                    "physiology_contribution": "GO"}
    assert result["comparisons"]["vq_vs_continuous_kd"]["paired_patient_count"] == 3
    arms["V1"] = rows([0.72, 0.75, 0.80], [8.0, 9.0, 8.0])
    no_go = compare_primary_arms(arms, split="external_test", bootstrap_samples=500)
    assert no_go["decisions"]["vq_contribution"] == "NO_GO"
    with pytest.raises(ValueError, match="TEST split"):
        compare_primary_arms(arms, split="development")


def test_missing_structure_count_blocks_hd95_no_worse_gate() -> None:
    def rows(values, hd95, counts=None):
        counts = counts if counts is not None else [0] * len(values)
        return {f"P{i}": {"fg_dice": d, "hd95_mm": h, "one_absent_count": count}
                for i, (d, h, count) in enumerate(zip(values, hd95, counts))}
    arms = {
        "C0": rows([0.70, 0.75, 0.80], [10.0, 9.0, 8.0]),
        "C1": rows([0.70, 0.75, 0.80], [10.0, 9.0, 8.0]),
        "V1": rows([0.72, 0.77, 0.82], [8.0, 8.0, 7.0], [1, 0, 0]),
        "V1-U": rows([0.71, 0.76, 0.81], [9.0, 8.5, 7.5]),
    }
    result = compare_primary_arms(arms, split="test", bootstrap_samples=500)
    comparison = result["comparisons"]["vq_vs_continuous_kd"]
    assert comparison["hd95_delta_mm_candidate_minus_reference"] < 0
    assert comparison["one_absent"]["candidate_minus_reference_total"] == 1
    assert comparison["one_absent"]["patients_worse"] == ["P0"]
    assert result["decisions"]["vq_contribution"] == "NO_GO"

    missing_counts = {arm: {pid: dict(row) for pid, row in patients.items()}
                      for arm, patients in arms.items()}
    del missing_counts["V1"]["P0"]["one_absent_count"]
    fail_closed = compare_primary_arms(missing_counts, split="test", bootstrap_samples=500)
    missing_comparison = fail_closed["comparisons"]["vq_vs_continuous_kd"]
    assert missing_comparison["one_absent"]["status"] == "NOT_REPORTED_FAIL_CLOSED"
    assert missing_comparison["one_absent"]["missing_patient_ids"] == ["P0"]
    assert fail_closed["decisions"]["vq_contribution"] == "NO_GO"


def test_physiology_harm_gate_requires_positive_paired_bootstrap_lower_bound() -> None:
    count = 20
    patient_ids = [f"P{i}" for i in range(count)]
    def rows(values):
        return {pid: {"fg_dice": value, "hd95_mm": 8.0, "one_absent_count": 0}
                for pid, value in zip(patient_ids, values)}
    baseline = [0.80] * count
    uniform = [0.779] * 10 + [0.90] * 10
    weighted = [0.781] * 10 + [0.87] * 10
    arms = {"C0": rows(baseline), "C1": rows(baseline),
            "V1-U": rows(uniform), "V1": rows(weighted)}
    result = compare_primary_arms(arms, split="test", seed=7, bootstrap_samples=2000)
    decision = result["decisions"]["physiology_contribution"]
    comparison = result["comparisons"]["physiology_weight_vs_uniform"]
    assert comparison["fg_dice_delta_candidate_minus_reference"] < 0
    assert comparison["harm_rate_reduction_v1_u_minus_v1"] > 0
    assert comparison["harm_rate_reduction_v1_u_minus_v1_bootstrap_ci95"][0] > 0
    assert decision == "GO"

    one_patient_uniform = [0.779] + [0.90] * 19
    one_patient_weighted = [0.781] + [0.87] * 19
    arms["V1-U"] = rows(one_patient_uniform)
    arms["V1"] = rows(one_patient_weighted)
    inconclusive = compare_primary_arms(arms, split="test", seed=7, bootstrap_samples=2000)
    assert inconclusive["comparisons"]["physiology_weight_vs_uniform"][
        "harm_rate_reduction_v1_u_minus_v1_bootstrap_ci95"][0] == 0
    assert inconclusive["decisions"]["physiology_contribution"] == "INCONCLUSIVE"


def test_student_only_export_cold_load_inference_and_profile(tmp_path: Path) -> None:
    config = _small_q3d_config()
    codes = torch.eye(4, dtype=torch.float32)
    model = CandidateQ3DStudent(config, codebook=_frozen_codebook(codes)).eval()
    checkpoint = tmp_path / "checkpoint.pt"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    torch.save({
                "student_state": model.state_dict(),
                "best_student_state": model.state_dict(),
                "tokenizer_artifact": model.codebook_metadata(),
                "teacher_state": {"discard_me": torch.ones(1)},
                "physiology_cache": {"path": "discard"}}, checkpoint)
    artifact_path, manifest, deploy_model = export_checkpoint(
        checkpoint, tmp_path / "q3d.student.pt", expected_checkpoint_sha256=digest(checkpoint),
        config_path=config_path, code_commit="synthetic-test-commit"
    )
    assert manifest["C0"]["version"] == "q3d_teacher_codebook.v1"
    assert manifest["source_config_sha256"] == digest(config_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert artifact["schema"] == "q3d.student_export.v1"
    assert all("sdf" not in key.lower() and "teacher" not in key.lower()
               for key in artifact["state_dict"])
    volumes = [
        _write_input_npz(tmp_path / f"volume_{index}.npz", shape, 200 + index)
        for index, shape in enumerate(((6, 17, 19), (9, 19, 20), (10, 16, 17)))
    ]
    cold = verify_cold_load(deploy_model, artifact_path, volumes)
    assert cold["status"] == "PASSED_BITWISE_THREE_NATIVE_VOLUMES"
    inferred = infer(artifact_path, volumes[0], tmp_path / "prediction.npz")
    assert inferred["evidence_status"] == "INFERENCE_ONLY_NO_LABELS_READ"
    with np.load(tmp_path / "prediction.npz", allow_pickle=False) as output:
        assert output["logits"].shape == (4, 6, 17, 19)
        assert output["mask"].shape == (6, 17, 19)
    cropped_input = _write_input_npz(tmp_path / "cropped-input.npz", (6, 17, 19), 321)
    with np.load(cropped_input, allow_pickle=False) as data:
        cropped_arrays = {key: np.array(data[key], copy=True) for key in data.files}
    cropped_arrays["context"] = np.asarray("crop")
    np.savez_compressed(cropped_input, **cropped_arrays)
    with pytest.raises(ValueError, match="context=full_volume"):
        infer(artifact_path, cropped_input, tmp_path / "cropped-prediction.npz")
    mismatched_input = tmp_path / "mismatched-geometry.npz"
    np.savez_compressed(
        mismatched_input, image=np.zeros((6, 17, 18), dtype=np.float32),
        shape_zyx=np.asarray((6, 17, 19), dtype=np.int32),
        spacing_zyx=np.asarray((8.0, 1.25, 1.5)),
        affine_zyx=np.diag([8.0, 1.25, 1.5, 1.0]),
        context=np.asarray("full_volume"),
    )
    with pytest.raises(ValueError, match="geometry|shape"):
        infer(artifact_path, mismatched_input, tmp_path / "mismatch-prediction.npz")
    profile = profile_artifact(artifact_path, tmp_path / "profile.json", input_path=volumes[0],
                               request_gpu=False, warmup=0, repeats=1)
    assert profile["cpu_latency_ms_per_volume"]["status"] == "COMPUTED"
    assert profile["gpu_latency_ms_per_volume"]["status"] == "NOT_RUN_NOT_REQUESTED"
    assert profile["flops"]["status"] == "PARTIAL_LOWER_BOUND"
    assert profile["flops"]["functional_operator_flops"] > 0
    assert profile["flops"]["operator_calls_counted"]["torch.matmul"] > 0
    assert profile["flops"]["operator_calls_counted"]["scaled_dot_product_attention"] > 0
    assert profile["flops"]["mac_convention"] == "2 floating point operations per multiply-accumulate"
    assert (tmp_path / "profile.json").is_file()
    manifest = export_and_verify(
        checkpoint, artifact_path, checkpoint_sha256=digest(checkpoint), config_path=config_path,
        code_commit="synthetic-test-commit", cost_report_path=tmp_path / "profile.json",
        verify_volumes=[str(path) for path in volumes],
    )
    assert manifest["cold_load"]["status"] == "PASSED_BITWISE_THREE_NATIVE_VOLUMES"
    assert manifest["cost"]["cpu_latency_ms_per_volume"]["status"] == "COMPUTED"
    assert manifest["cost"]["gpu_latency_ms_per_volume"]["status"] == "NOT_RUN_NOT_REQUESTED"
    _, metadata = load_export(artifact_path, expected_sha256=manifest["artifact_sha256"])
    assert metadata["codebook_c0"]["sha256"] == manifest["C0"]["sha256"]
    assert (tmp_path / "q3d.student.pt.manifest.json").is_file()


@pytest.mark.parametrize("recipe_name,student_arm", [
    ("c0", "C0"), ("c1", "C1"), ("v1", "V1"), ("v1u", "V1-U"),
    ("v0", "V0"), ("b1", "B1"), ("e1", "E1"), ("o1", "O1"),
    ("s1", "S1"), ("v1_stride4", "V1"), ("v1_residual2", "V1"),
    ("mean_teacher", "C0"),
])
def test_export_round_trips_all_student_recipes_without_train_only_heads(
    tmp_path: Path, recipe_name: str, student_arm: str,
) -> None:
    torch.set_num_threads(1)
    repo = Path(__file__).resolve().parents[1]
    recipe_path = repo / "configs" / "candidate_q3d" / f"{recipe_name}.json"
    config = json.loads(recipe_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    student_config = config["model"]["student"]
    student_config["arm"] = student_arm
    student_config["channels"] = [4, 4, 4, 4, 4]
    student_config["sdf_head_enabled"] = True
    config["tokenizer"] = {"codebook_size": 4, "codebook_dim": 4, "tau": 0.1}
    if student_arm == "C1":
        config["model"]["teacher"]["feature_channels"] = 4

    source_config_path = tmp_path / f"{recipe_name}.json"
    source_config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    frozen_codebook = _frozen_codebook(torch.eye(4, dtype=torch.float32))
    source_model = CandidateQ3DStudent(
        config, codebook=frozen_codebook if student_arm in {"V1", "V1-U", "V0", "B1", "E1", "O1", "S1"} else None,
    ).eval()
    student_state = source_model.state_dict()
    payload: dict = {
        "config": config,
        "best_student_state": student_state,
        "student_state": student_state,
        "teacher_state": {"training_only": torch.ones(1)},
        "physiology_cache": {"training_only": "not exported"},
    }
    if source_model.use_vq:
        payload["tokenizer_artifact"] = source_model.codebook_metadata()
    checkpoint = tmp_path / f"{recipe_name}.engine-resume.pt"
    torch.save(payload, checkpoint)

    artifact_path, manifest, deploy_model = export_checkpoint(
        checkpoint, tmp_path / f"{recipe_name}.student.pt",
        expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        config_path=source_config_path, code_commit="synthetic-test-commit",
    )
    assert manifest["arm"] == student_arm
    assert manifest["deploy_student_arm"] == ("C0" if student_arm == "C1" else student_arm)
    assert manifest["source_config_name"] == recipe_name
    assert manifest["source_config_sha256"] == hashlib.sha256(source_config_path.read_bytes()).hexdigest()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert "teacher" not in artifact["config"]
    assert "teacher" not in artifact["config"]["model"]
    state_keys = artifact["state_dict"]
    assert all("sdf" not in key.lower() and "teacher" not in key.lower()
               and "continuous_feature_head" not in key.lower() for key in state_keys)
    if student_arm in {"V1", "V1-U", "V0", "B1", "E1", "O1", "S1"}:
        assert artifact["state_dict"]["_extra_state"]["frozen"] is True
        assert manifest["C0"]["sha256"] == frozen_codebook["sha256"]
    else:
        assert artifact["state_dict"]["_extra_state"]["frozen"] is False
        assert manifest["C0"]["status"] == "NOT_APPLICABLE_CONTINUOUS_ARM"

    volumes = [
        _write_input_npz(tmp_path / f"{recipe_name}_volume_{index}.npz", shape, 700 + index)
        for index, shape in enumerate(((6, 17, 19), (9, 19, 20), (10, 16, 17)))
    ]
    source_hashes: list[str] = []
    for volume in volumes:
        image, geometry = _read_verification_volume(volume)
        with torch.inference_mode():
            logits = source_model.infer_logits(image, geometry=geometry)
        source_hashes.append(hashlib.sha256(logits.contiguous().cpu().numpy().tobytes()).hexdigest())
    cold = verify_cold_load(deploy_model, artifact_path, volumes)
    assert cold["status"] == "PASSED_BITWISE_THREE_NATIVE_VOLUMES"
    assert cold["logit_sha256"] == source_hashes


def test_export_consumes_engine_written_best_student_checkpoint(tmp_path: Path) -> None:
    from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
    from self_audit.candidate_q3d.config import canonical_config
    from self_audit.candidate_q3d.data import Q3DData
    from self_audit.candidate_q3d.engine import train as train_q3d
    from self_audit.candidate_q3d.student import CandidateQ3DStudent

    torch.set_num_threads(1)
    fixture = synthetic_fixture(
        tmp_path / "engine-fixture", seed=91, n_labeled=2, n_unlabeled=1,
        n_evaluation=1, shape_tzyx=(2, 4, 16, 16),
    )
    training_data = CandidateA3DData(
        fixture["train"], fixture["train_sha256"], synthetic=True,
    )
    q3d_data = Q3DData(training_data, protocol="A")
    config = canonical_config("c0", dataset="acdc", protocol="A", k=2, seed=91)
    config["runtime"].update({"synthetic": True, "device": "cpu"})
    config["runtime"]["smoke_update_overrides"] = {
        "T": None, "Q": None, "P": None, "J": 1,
    }
    config["dataset"]["manifests"]["train"] = {
        "path": str(Path(fixture["train"]).resolve()),
        "sha256": fixture["train_sha256"], "status": "READY",
    }
    config["dataset"]["manifests"]["development"] = {
        "path": str(Path(fixture["evaluation"]).resolve()),
        "sha256": fixture["evaluation_sha256"], "status": "READY",
    }
    config_path = tmp_path / "engine-c0.json"
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    source_student = CandidateQ3DStudent(config)
    run = train_q3d(config, tmp_path / "engine-output", data=q3d_data,
                    student=source_student,
                    device="cpu", smoke=True)
    checkpoint = Path(run["checkpoint"])
    assert checkpoint.is_file()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["schema"] == "self_audit.candidate_q3d.resume.v3"
    assert isinstance(payload["best_student_state"], dict)
    assert payload["best_student_state"]["_extra_state"]["schema"] == "self_audit.q3d.codebook.v1"
    with pytest.raises(ValueError, match="source checkpoint SHA-256 mismatch"):
        export_checkpoint(
            checkpoint, tmp_path / "wrong-checkpoint.sha.pt",
            expected_checkpoint_sha256="0" * 64, config_path=config_path,
        )
    wrong_config = copy.deepcopy(config)
    wrong_config["runtime"]["device"] = "cuda"
    wrong_config_path = tmp_path / "wrong-engine-config.json"
    wrong_config_path.write_text(json.dumps(wrong_config, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical SHA-256 differs"):
        export_checkpoint(
            checkpoint, tmp_path / "wrong-config.sha.pt",
            expected_checkpoint_sha256=run["checkpoint_sha256"], config_path=wrong_config_path,
        )

    artifact_path, manifest, deploy_model = export_checkpoint(
        checkpoint, tmp_path / "engine-c0.student.pt",
        expected_checkpoint_sha256=run["checkpoint_sha256"], config_path=config_path,
        code_commit="synthetic-engine-test-commit",
    )
    assert manifest["engine_config_sha256"] == payload["config_sha256"]
    assert manifest["source_config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    volumes = [
        _write_input_npz(tmp_path / f"engine-volume-{index}.npz", shape, 901 + index)
        for index, shape in enumerate(((6, 17, 19), (9, 19, 20), (10, 16, 17)))
    ]
    source_student.eval()
    source_hashes: list[str] = []
    for volume in volumes:
        image, geometry = _read_verification_volume(volume)
        with torch.inference_mode():
            logits = source_student.infer_logits(image, geometry=geometry)
        source_hashes.append(hashlib.sha256(logits.contiguous().cpu().numpy().tobytes()).hexdigest())
    cold = verify_cold_load(deploy_model, artifact_path, volumes)
    assert cold["status"] == "PASSED_BITWISE_THREE_NATIVE_VOLUMES"
    assert cold["logit_sha256"] == source_hashes


def test_export_rejects_teacher_or_cache_state_inside_student(tmp_path: Path) -> None:
    config = _small_q3d_config()
    state = {"teacher.weight": torch.ones(1)}
    checkpoint = tmp_path / "bad_checkpoint.pt"
    torch.save({"config": config, "student": state}, checkpoint)
    with pytest.raises(ValueError, match="forbidden non-deploy state key"):
        export_checkpoint(checkpoint, tmp_path / "bad_export.pt")


def test_export_rejects_unsafe_pickle_payload_without_fallback(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unsafe_checkpoint.pt"
    torch.save({"config": _small_q3d_config(), "student_state": {"weight": torch.ones(1)},
                "rng": np.random.get_state()}, checkpoint)
    with pytest.raises(ValueError, match="safe for weights_only loading"):
        export_checkpoint(
            checkpoint, tmp_path / "unsafe_export.pt",
            expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )


def test_export_rejects_retired_resume_v1_schema(tmp_path: Path) -> None:
    config = _small_q3d_config()
    state = CandidateQ3DStudent(
        config, codebook=_frozen_codebook(torch.eye(4, dtype=torch.float32)),
    ).state_dict()
    checkpoint = tmp_path / "resume-v1.pt"
    torch.save({"schema": "self_audit.candidate_q3d.resume.v1", "config": config,
                "best_student_state": state}, checkpoint)
    with pytest.raises(ValueError, match="resume-v1 checkpoints are retired"):
        export_checkpoint(
            checkpoint, tmp_path / "retired-export.pt",
            expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )
