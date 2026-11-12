from __future__ import annotations

import json
import random
import os
import inspect
import copy
from pathlib import Path
import subprocess
import sys

import pytest

from self_audit.candidate_q3d import profiling
from self_audit.candidate_q3d import geometry_repair
from self_audit.data import CLASS_MAPPING, GEOMETRY_MAP_SCHEMA, GEOMETRY_SCHEMA


class FakeClock:
    def __init__(self, start: float = 0.0, tick: float = 0.125) -> None:
        self.value = start
        self.tick = tick

    def __call__(self) -> float:
        self.value += self.tick
        return self.value


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.sync_count = 0
        self.reset_count = 0

    def is_available(self) -> bool:
        return self.available

    def synchronize(self, _device) -> None:
        self.sync_count += 1

    def reset_peak_memory_stats(self, _device) -> None:
        self.reset_count += 1

    def max_memory_allocated(self, _device) -> int:
        return 1024

    def max_memory_reserved(self, _device) -> int:
        return 2048

    def get_device_properties(self, _device):
        return type("Properties", (), {"name": "mock cuda", "major": 9, "minor": 0})()


class FakeTorch:
    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda

    @staticmethod
    def device(value):
        return value


class FakeOptimizer:
    def step(self):
        return None


def _geometry(shape: tuple[int, int, int]) -> dict:
    return {
        "schema": GEOMETRY_SCHEMA,
        "array_axes": ["T", "Z", "Y", "X"],
        "spatial_axes": ["Z", "Y", "X"],
        "shape_zyx": list(shape),
        "spacing_zyx_mm": [2.0, 0.8, 0.7],
        "spacing_yx_mm": [0.8, 0.7],
        "affine_zyx": [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "class_mapping": dict(CLASS_MAPPING),
    }


def _train_manifest() -> dict:
    labeled = [
        {"id": "L-small", "image": "l-small.npy", "image_sha256": "a" * 64, "masks": {"0": {"path": "l-small-mask.npy", "sha256": "b" * 64}}},
        {"id": "L-largest", "image": "l-large.npy", "image_sha256": "c" * 64, "masks": {"0": {"path": "l-large-mask.npy", "sha256": "d" * 64}}},
    ]
    unlabeled = [
        {"id": "U-small", "image": "u-small.npy", "image_sha256": "e" * 64},
        {"id": "U-largest", "image": "u-large.npy", "image_sha256": "f" * 64},
    ]
    geometry = {
        "L-small": _geometry((8, 32, 32)),
        "L-largest": _geometry((12, 64, 64)),
        "U-small": _geometry((8, 32, 32)),
        "U-largest": _geometry((14, 80, 80)),
    }
    # K8 identity is validated by profile preflight; this small fixture tests
    # selection only and therefore uses its own explicit test budget.
    return {
        "schema": "self_audit.data.v2",
        "role": "training",
        "synthetic": False,
        "dataset": "acdc",
        "label_budget": 8,
        "labeled": labeled + [
            {"id": f"L-extra-{index}", "image": f"le-{index}.npy", "image_sha256": str(index) * 64,
             "masks": {"0": {"path": f"le-{index}-mask.npy", "sha256": str(index + 1) * 64}}}
            for index in range(6)
        ],
        "unlabeled": unlabeled + [
            {"id": f"U-extra-{index}", "image": f"ue-{index}.npy", "image_sha256": str(index + 2) * 64}
            for index in range(2)
        ],
        "evaluation_ids": ["DEV1"],
        "geometry": {"schema": GEOMETRY_MAP_SCHEMA, "by_patient": {
            **geometry,
            **{f"L-extra-{index}": _geometry((9, 20, 20)) for index in range(6)},
            **{f"U-extra-{index}": _geometry((9, 20, 20)) for index in range(2)},
        }},
    }


class FakeData:
    def __init__(self) -> None:
        self.teacher_rows = {
            (0, 0): "L-small",
            (1, 0): "L-largest",
            (2, 0): "U-largest",  # A deliberately overbroad fake API result.
        }
        self.labeled_rows = {(0, 0): "L-small", (1, 0): "L-largest"}
        self.unlabeled_patients = ("U-small", "U-largest", "U-middle")

    def teacher_frame_keys(self):
        return list(self.teacher_rows)

    def teacher_patient_for_key(self, key):
        return self.teacher_rows[key]

    def labeled_frame_keys(self):
        return list(self.labeled_rows)

    def labeled_patient_for_key(self, key):
        return self.labeled_rows[key]


class FakeSampler:
    def __init__(self) -> None:
        self.updates = 0
        self._unlabeled_cursor = 0
        self._unlabeled_permutation = ["U-small", "U-middle", "U-largest"]

    def sample_update(self):
        patient = self._unlabeled_permutation[self._unlabeled_cursor]
        self._unlabeled_cursor += 1
        if self._unlabeled_cursor == len(self._unlabeled_permutation):
            self._unlabeled_cursor = 0
        self.updates += 1
        return {
            "labeled_key": (0, 0),
            "labeled_patient": "L-small",
            "unlabeled_key": (0, self.updates),
            "unlabeled_patient": patient,
        }


def _inputs() -> profiling.ProfileInputs:
    return profiling.ProfileInputs(
        config={
            "training": {
                "stage_updates": dict(profiling.PRODUCTION_STAGE_BUDGETS),
                "teacher_warmup_fraction": 0.05,
                "warmup_fraction": 0.05,
            },
            "losses": {"codekd_ramp_fraction": 0.1},
        },
        config_path=Path("/tmp/recipe.json"),
        config_sha256="1" * 64,
        train_manifest_path=Path("/tmp/train.json"),
        train_manifest_sha256="a" * 64,
        development_manifest_path=Path("/tmp/dev.json"),
        development_manifest_sha256="b" * 64,
        largest_l=profiling.PatientVolume("L-largest", (12, 64, 64), 12 * 64 * 64, "L"),
        largest_u=profiling.PatientVolume("U-largest", (14, 80, 80), 14 * 80 * 80, "U"),
        largest_train=profiling.PatientVolume("U-largest", (14, 80, 80), 14 * 80 * 80, "TRAIN"),
        stage_artifacts={},
        labeled_patient_ids=("L-small", "L-largest"),
        unlabeled_patient_ids=("U-largest", "U-middle", "U-small"),
    )


def test_largest_stream_selection_uses_native_volume_metadata_and_keeps_u_image_only():
    largest_l, largest_u, largest_train = profiling.largest_train_patients(_train_manifest())

    assert largest_l.patient_id == "L-largest"
    assert largest_u.patient_id == "U-largest"
    assert largest_train.patient_id == "U-largest"
    assert largest_u.pool == "U"

    manifest = _train_manifest()
    manifest["unlabeled"][0]["masks"] = {"0": {"path": "forbidden.npy", "sha256": "0" * 64}}
    with pytest.raises(profiling.ProfilePrerequisiteError, match="image-only"):
        profiling.largest_train_patients(manifest)


def test_t_selector_is_l_only_and_j_sampler_includes_largest_u_first():
    class FakeEngine:
        @staticmethod
        def _teacher_frame_choice(data, rng):
            return rng.choice(data.teacher_frame_keys())

        @staticmethod
        def _make_q3d_sampler(_data, _seed, _state, *, require_unlabeled=True):
            assert require_unlabeled
            return FakeSampler()

    cuda = FakeCuda()
    clock = FakeClock()
    t_measurement = profiling.UpdateMeasurement("cuda:0", cuda_api=cuda, clock=clock)
    with profiling._engine_profile_hooks(
        FakeEngine, "T", t_measurement, "L-largest", "U-largest", FakeOptimizer,
    ):
        key = FakeEngine._teacher_frame_choice(FakeData(), random.Random(0))
        assert key == (1, 0)
        FakeOptimizer().step()
    assert t_measurement.selection_trace == [{"labeled_patient": "L-largest", "unlabeled_patient": None}]

    j_measurement = profiling.UpdateMeasurement("cuda:0", cuda_api=cuda, clock=clock)
    with profiling._engine_profile_hooks(
        FakeEngine, "J", j_measurement, "L-largest", "U-largest", FakeOptimizer,
    ):
        sampler = FakeEngine._make_q3d_sampler(FakeData(), 42, {}, require_unlabeled=True)
        records = []
        for _ in range(20):
            record = sampler.sample_update()
            records.append(record)
            FakeOptimizer().step()

    assert len(records) == 20
    assert records[0]["unlabeled_patient"] == "U-largest"
    assert records[0]["labeled_patient"] == "L-largest"
    assert all(record["labeled_patient"] == "L-small" for record in records[1:])
    assert [row["labeled_patient"] for row in j_measurement.selection_trace] == [
        "L-largest", *("L-small" for _ in range(19)),
    ]
    assert j_measurement.selection_trace[0]["unlabeled_patient"] == "U-largest"


def test_engine_stage_uses_20_update_limit_without_rewriting_recipe(tmp_path):
    calls = []
    data = FakeData()

    class FakeEngine:
        @staticmethod
        def _teacher_frame_choice(_data, _rng):
            return (0, 0)

        @staticmethod
        def _make_q3d_sampler(_data, _seed, _state, *, require_unlabeled=True):
            assert require_unlabeled
            return FakeSampler()

        @staticmethod
        def train(config, output_dir, **kwargs):
            calls.append((output_dir, kwargs))
            if kwargs["stage"] == "T":
                for _ in range(kwargs["update_limit"]):
                    FakeEngine._teacher_frame_choice(data, random.Random(0))
                    FakeOptimizer().step()
            else:
                sampler = FakeEngine._make_q3d_sampler(data, 42, {}, require_unlabeled=True)
                for _ in range(kwargs["update_limit"]):
                    sampler.sample_update()
                    FakeOptimizer().step()
            return {"runtime_synthetic": False, "status": "PAUSED"}

    inputs = _inputs()
    for stage in ("T", "Q", "P"):
        path = tmp_path / f"{stage}.json"
        path.write_text(json.dumps({"stage": stage}))
        inputs.stage_artifacts[stage] = profiling.StageArtifact(
            stage, path, profiling.sha256_file(path),
        )
    original_budgets = dict(inputs.config["training"]["stage_updates"])
    measurements = {}
    for stage in ("T", "J"):
        measurement = profiling.UpdateMeasurement("cuda:0", cuda_api=FakeCuda(), clock=FakeClock())
        profiling.run_engine_stage(
            inputs, stage, tmp_path / f"profile-{stage.lower()}", measurement,
            device="cuda:0", engine_module=FakeEngine, optimizer_type=FakeOptimizer,
        )
        measurements[stage] = measurement

    assert all(len(measurements[stage].records) == 20 for stage in ("T", "J"))
    assert [call[1]["update_limit"] for call in calls] == [20, 20]
    assert [call[1]["stage"] for call in calls] == ["T", "J"]
    assert calls[1][1]["stage_artifacts"] is not None
    assert calls[1][1]["smoke"] is False
    assert calls[1][1]["synthetic_gate_override"] is False
    assert inputs.config["training"]["stage_updates"] == original_budgets == profiling.PRODUCTION_STAGE_BUDGETS


def test_j_fails_before_engine_when_verified_t_q_p_inputs_are_missing():
    class FakeEngine:
        called = False

        @staticmethod
        def _teacher_frame_choice(_data, _rng):
            return (0, 0)

        @staticmethod
        def _make_q3d_sampler(_data, _seed, _state, *, require_unlabeled=True):
            return FakeSampler()

        @classmethod
        def train(cls, *_args, **_kwargs):
            cls.called = True

    with pytest.raises(profiling.ProfilePrerequisiteError, match="T/Q/P"):
        profiling.run_engine_stage(
            _inputs(), "J", Path("/tmp/profile-j"),
            profiling.UpdateMeasurement("cuda:0", cuda_api=FakeCuda()),
            device="cuda:0", engine_module=FakeEngine, optimizer_type=FakeOptimizer,
        )
    assert FakeEngine.called is False


def test_profile_oom_records_partial_update_and_cleans_only_its_temp_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    report_path = tmp_path / "phase-d.json"
    cuda = FakeCuda()
    torch = FakeTorch(cuda)
    calls = []

    def fake_stage_runner(_inputs, stage, output_dir, measurement, *, device):
        assert output_dir.parent.parent.resolve() == scratch.resolve()
        assert not output_dir.is_relative_to(repo)
        calls.append(stage)
        output_dir.mkdir(parents=True)
        (output_dir / "profile-partial.txt").write_text("owned temp output")
        steps = 20 if stage == "T" else 2
        metrics = []
        for index in range(steps):
            measurement.begin_update()
            measurement.selection_trace.append({
                "labeled_patient": "L-largest" if (stage == "T" and index == 0) or (stage == "J" and index == 0) else "L-small",
                "unlabeled_patient": None if stage == "T" else ("U-largest" if index == 0 else "U-small"),
            })
            measurement.finish_update()
            if stage == "T":
                metrics.append({"stage": "T", "update": index + 1, "phase": "T-a"})
        if stage == "J":
            measurement.begin_update()
            measurement.selection_trace.append({"labeled_patient": "L-small", "unlabeled_patient": "U-middle"})
            raise RuntimeError("CUDA out of memory while executing J update")
        return {"runtime_synthetic": False, "status": "PAUSED", "profile_metrics": metrics}

    report = profiling.run_profile_software_test(
        inputs=_inputs(),
        stage_runner=fake_stage_runner,
        cuda_api=cuda,
        torch_module=torch,
        report_path=report_path,
        scratch_root=scratch,
        device="cuda:0",
        repo_root=repo,
        clock=FakeClock(),
    )

    assert report["status"] == "SOFTWARE_ONLY_FAILURE"
    assert report["evidence_status"] == "SOFTWARE_ONLY"
    assert "gpu" not in report
    assert report["stage_results"]["T"]["measured_updates"] == 20
    assert report["stage_results"]["T"]["status"] == "SOFTWARE_ONLY_COMPLETED"
    assert report["stage_results"]["J"]["status"] == "SOFTWARE_ONLY_OOM"
    assert report["stage_results"]["J"]["measured_updates"] == 2
    assert report["stage_results"]["J"]["failed_update"]["update"] == 3
    assert report["failure"]["retry_attempted"] is False
    assert calls == ["T", "J"]
    assert list(scratch.iterdir()) == []
    persisted = json.loads(report_path.read_text())
    assert persisted["campaign_estimate"]["coverage"]["Q"] == "NOT_RUN"
    assert persisted["campaign_estimate"]["coverage"]["P"] == "NOT_RUN"
    assert persisted["campaign_estimate"]["coverage"]["evaluation"] == "NOT_RUN"
    assert persisted["campaign_estimate"]["coverage"]["T_b_main_teacher"] == "NOT_RUN"
    assert persisted["campaign_estimate"]["full_campaign_seconds"] is None
    assert persisted["campaign_estimate"]["status"] == "PARTIAL_PROJECTION"
    assert str(scratch) not in report_path.read_text()


def test_profile_requires_explicit_real_run_flag_without_writing_anything(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    report = tmp_path / "report.json"

    with pytest.raises(profiling.ProfilePrerequisiteError, match="--allow-real-profile"):
        profiling.run_profile(
            config_path="/tmp/unused.json",
            artifact_paths=[],
            artifact_hashes=[],
            source_train_manifest_path="/tmp/source-train.json",
            source_train_manifest_sha256="0" * 64,
            train_repair_report_path="/tmp/train-repair.json",
            train_repair_report_sha256="0" * 64,
            source_development_manifest_path="/tmp/source-dev.json",
            source_development_manifest_sha256="0" * 64,
            development_repair_report_path="/tmp/dev-repair.json",
            development_repair_report_sha256="0" * 64,
            phase_receipt_path="/tmp/phase.json",
            phase_receipt_sha256="0" * 64,
            report_path=report,
            scratch_root=scratch,
            allow_real_profile=False,
        )

    assert not report.exists()
    assert list(scratch.iterdir()) == []


def test_old_phase_receipt_requires_new_locked_geometry_phase_contract():
    error = profiling._phase_lock_error(
        ValueError("model.teacher.phase_measurement receipt SHA-256 is not the locked ERF receipt")
    )

    assert isinstance(error, profiling.ProfilePrerequisiteError)
    assert "corrected-geometry phase lock" in str(error)
    assert "Phase A/B must pass" in str(error)


def test_phase_receipt_without_corrected_geometry_lineage_is_rejected(tmp_path):
    path = tmp_path / "old-phase.json"
    path.write_text(json.dumps({
        "schema": "self_audit.q3d_resenc.erf_probe.v1",
        "status": "PASS",
        "evidence_class": "SOFTWARE_ONLY_SYNTHETIC_ARCHITECTURE_PROBE",
    }))

    with pytest.raises(profiling.ProfilePrerequisiteError, match="old Phase-0 receipt is insufficient"):
        profiling._validate_geometry_reprobe(
            path,
            profiling.sha256_file(path),
            {"manifest_lineage": {}, "geometry": {}},
            _inputs().config,
        )


def test_geometry_phase_receipt_uses_canonical_probe_architecture_name(tmp_path):
    from scripts.q3d_reprobe_geometry import canonical_probe_parameters, phase_proposal_from_probe_result

    proposal = {
        "manifest_lineage": {"repaired_manifest_sha256": "1" * 64},
        "geometry": {"planner_output": {"stage_strides_zyx": [
            [1, 1, 1], [2, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2],
        ]}},
    }
    checkpoint_sha, registry_sha = "a" * 64, "b" * 64

    def create_receipt(directory: Path, architecture: str):
        directory.mkdir()
        path = directory / "geometry-reprobe.json"
        payload = {
            "schema": "self_audit.q3d_resenc.erf_probe.v1",
            "status": "PASS",
            "evidence_class": "SOFTWARE_ONLY_SYNTHETIC_ARCHITECTURE_PROBE",
            "decision_rule": {"threshold_voxels": 2.0},
            "summary": {
                "mean_centroid_offset_voxels": [0.0, 0.0, 0.0],
                "absolute_mean_centroid_offset_voxels": [0.0, 0.0, 0.0],
                "max_abs_centroid_offset_voxels": 0.0,
            },
            "model": {
                "architecture": architecture,
                "checkpoint_sha256": checkpoint_sha,
                "registry_sha256": registry_sha,
            },
        }
        payload["geometry_reprobe"] = {
            "status": "PROBE_COMPLETED_NOT_A_GEOMETRY_LOCK",
            "manifest_lineage": proposal["manifest_lineage"],
            "proposal_geometry": proposal["geometry"],
            "probe_contract": canonical_probe_parameters(
                proposal["geometry"]["planner_output"]["stage_strides_zyx"]
            ),
            "phase_proposal": phase_proposal_from_probe_result(payload),
            "pinned_probe_assets": {
                "checkpoint_sha256": checkpoint_sha,
                "registry_sha256": registry_sha,
            },
        }
        path.write_text(json.dumps(payload))
        digest = profiling.sha256_file(path)
        config = {
            "model": {"teacher": {
                "architecture": "nnfoundation_resenc_l",
                "pretrained": {"weights_sha256": checkpoint_sha},
                "feature_source": "encoder_stride8",
                "phase_offsets": {"encoder_stride8": [0.0, 0.0, 0.0]},
                "phase_measurement": {"receipt_path": str(path), "receipt_sha256": digest},
            }},
        }
        return path, digest, config

    path, digest, config = create_receipt(tmp_path / "valid", "ResEncL")
    _, _, summary = profiling._validate_geometry_reprobe(path, digest, proposal, config)
    assert summary["phase_proposal"]["status"] == "MAIN_PHASE_ZERO_CORRECTION_PROPOSED"

    wrong_path, wrong_digest, wrong_config = create_receipt(tmp_path / "wrong", "nnfoundation_resenc_l")
    with pytest.raises(profiling.ProfilePrerequisiteError, match="model/checkpoint identity"):
        profiling._validate_geometry_reprobe(wrong_path, wrong_digest, proposal, wrong_config)


def test_dev_repair_audit_uses_full_affine_and_allows_distinct_source_hashes(tmp_path):
    source_path = tmp_path / "source-dev.json"
    shape = [12, 64, 64]
    affine_xyz = [
        [0.7, 0.0, 0.0, 0.0],
        [0.0, 0.8, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    geometry = geometry_repair.native_geometry_zyx(
        affine_xyz, shape, source_axis_order="XYZ",
    )
    source = {
        "schema": "self_audit.data.v2",
        "role": "development",
        "synthetic": False,
        "dataset": "acdc",
        "patients": [{"id": "DEV-1", "image": "dev.npy", "image_sha256": "f" * 64}],
        "geometry": {"schema": GEOMETRY_MAP_SCHEMA, "by_patient": {"DEV-1": geometry}},
    }
    repaired = copy.deepcopy(source)
    source_sha, repaired_sha = "1" * 64, "2" * 64
    receipt = {
        "schema": geometry_repair.RECEIPT_SCHEMA,
        "status": "success",
        "geometry_only": True,
        "source_manifest_sha256": source_sha,
        "repaired_manifest_sha256": repaired_sha,
        "patient_ids": ["DEV-1"],
        "axis_map": geometry_repair._AXIS_MAP,
        "cine_mapping_source": "unit-test-map",
        "cine_mapping_sha256": "3" * 64,
        "cine_mapping_provenance": "synthetic test fixture",
        "voxel_bridge_status": "NOT_VERIFIED",
    }
    row = {
        "patient_id": "DEV-1",
        "errors": [],
        "source_file_sha256": "4" * 64,
        "source_header_sha256": "5" * 64,
        "voxel_bridge_status": "NOT_VERIFIED",
        "cine_mapping_source": receipt["cine_mapping_source"],
        "cine_mapping_sha256": receipt["cine_mapping_sha256"],
        "cine_mapping_provenance": receipt["cine_mapping_provenance"],
        "resolved_spatial_unit": "mm",
        "unit_resolution": "nifti_header",
        "unit_provenance": None,
        "manifest_shape_zyx": shape,
        "manifest_spacing_zyx_mm": geometry["spacing_zyx_mm"],
        "manifest_affine_zyx": geometry["affine_zyx"],
        "repaired_spacing_zyx_mm": geometry["spacing_zyx_mm"],
        "repaired_affine_zyx": geometry["affine_zyx"],
        "source_shape_xyzt": [64, 64, 12, 30],
        "header_zooms_xyz_native": [0.7, 0.8, 2.0],
        "header_zooms_xyz_mm": [0.7, 0.8, 2.0],
        "selected_affine_xyz": affine_xyz,
        "sform_code": 1,
        "qform_code": 0,
        "sform_affine_xyz": affine_xyz,
    }
    repair_report = {
        "schema": geometry_repair.REPAIR_SCHEMA,
        "status": "success",
        "audit_only": False,
        "errors": [],
        "source_manifest_path": str(source_path.resolve()),
        "source_manifest_sha256_expected": source_sha,
        "source_manifest_sha256_actual": source_sha,
        "audit_rows": [row],
        "receipt": receipt,
    }

    summary = profiling._validate_repair_audit_rows(
        source,
        repaired,
        repair_report,
        label="ACDC DEV18 fixture",
        source_manifest_path=source_path,
        source_manifest_sha256=source_sha,
        repaired_manifest_sha256=repaired_sha,
    )
    assert summary["status"] == "PASS"
    assert row["source_file_sha256"] != source["patients"][0]["image_sha256"]

    corrupted = copy.deepcopy(repair_report)
    corrupted["audit_rows"][0]["selected_affine_xyz"][0][3] = 9.0
    with pytest.raises(profiling.ProfilePrerequisiteError, match="affine"):
        profiling._validate_repair_audit_rows(
            source,
            repaired,
            corrupted,
            label="ACDC DEV18 fixture",
            source_manifest_path=source_path,
            source_manifest_sha256=source_sha,
            repaired_manifest_sha256=repaired_sha,
        )


def test_campaign_estimate_is_only_partial_and_marks_unmeasured_stages_not_run():
    stage_results = {
        "T": {"updates": [{"wall_seconds": 2.0} for _ in range(20)]},
        "J": {"updates": [{"wall_seconds": 3.0} for _ in range(20)]},
    }
    estimate = profiling.measured_campaign_estimate(stage_results, _inputs().config)

    assert estimate["status"] == "PARTIAL_PROJECTION"
    assert estimate["coverage"]["T_a_measured_updates"] == 20
    assert estimate["coverage"]["T_b_main_teacher"] == "NOT_RUN"
    assert estimate["coverage"]["Q"] == "NOT_RUN"
    assert estimate["coverage"]["P"] == "NOT_RUN"
    assert estimate["coverage"]["evaluation"] == "NOT_RUN"
    assert estimate["full_campaign_seconds"] is None
    assert estimate["measured_portion_projections_seconds"]["T_a_warmup_400_updates_from_measured_T_a_rate"] == 800.0
    assert estimate["measured_portion_projections_seconds"]["J_v1_8000_updates_from_first_20_J_rate"] == 24000.0
    assert estimate["coverage"]["J_measured_phase"] == "learning_rate_warmup_and_codekd_ramp"


def test_software_harness_success_cannot_emit_gpu_profile_evidence(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    cuda = FakeCuda()
    torch = FakeTorch(cuda)

    def fake_stage_runner(_inputs, stage, _output_dir, measurement, *, device):
        metrics = []
        for index in range(20):
            measurement.begin_update()
            measurement.selection_trace.append({
                "labeled_patient": "L-largest" if index == 0 else "L-small",
                "unlabeled_patient": None if stage == "T" else (
                    "U-largest" if index == 0 else ("U-small" if index % 2 else "U-middle")
                ),
            })
            measurement.finish_update()
            if stage == "T":
                metrics.append({"stage": "T", "update": index + 1, "phase": "T-a"})
        return {"runtime_synthetic": False, "profile_metrics": metrics}

    report = profiling.run_profile_software_test(
        inputs=_inputs(),
        stage_runner=fake_stage_runner,
        report_path=tmp_path / "software-only.json",
        scratch_root=scratch,
        cuda_api=cuda,
        torch_module=torch,
        repo_root=repo,
        clock=FakeClock(),
    )

    assert report["status"] == "SOFTWARE_ONLY"
    assert report["evidence_status"] == "SOFTWARE_ONLY"
    assert "gpu" not in report
    assert all(result["status"] == "SOFTWARE_ONLY_COMPLETED" for result in report["stage_results"].values())
    assert report["stage_results"]["T"]["measured_phase"] == "T-a"
    assert report["stage_results"]["T"]["measured_updates"] == 20
    assert report["stage_results"]["J"]["measured_updates"] == 20


def test_cli_help_bootstraps_from_unrelated_cwd_without_pythonpath(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "q3d_profile_gpu.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=unrelated_cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--allow-real-profile" in completed.stdout
    assert "--stage-artifact-sha" in completed.stdout
    assert "--dev-repair-report-sha256" in completed.stdout
    assert "--phase-receipt-sha256" in completed.stdout


def test_report_publish_failure_still_cleans_only_owned_temp_tree(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    report_path = tmp_path / "unpublished.json"
    cuda = FakeCuda()
    torch = FakeTorch(cuda)

    def fake_stage_runner(_inputs, stage, output_dir, measurement, *, device):
        assert output_dir.parent.parent.resolve() == scratch.resolve()
        output_dir.mkdir(parents=True)
        (output_dir / "owned.txt").write_text(stage)
        measurement.begin_update()
        measurement.selection_trace.append({
            "labeled_patient": "L-largest",
            "unlabeled_patient": None,
        })
        measurement.finish_update()
        raise RuntimeError("stop after one step")

    def fail_write(_path, _payload):
        raise OSError("mock report disk failure")

    monkeypatch.setattr(profiling, "_atomic_write_report", fail_write)
    with pytest.raises(OSError, match="mock report disk failure"):
        profiling.run_profile_software_test(
            inputs=_inputs(),
            stage_runner=fake_stage_runner,
            cuda_api=cuda,
            torch_module=torch,
            report_path=report_path,
            scratch_root=scratch,
            device="cuda:0",
            repo_root=repo,
            clock=FakeClock(),
        )

    assert list(scratch.iterdir()) == []
    assert not report_path.exists()


def test_only_real_profile_api_can_emit_gpu_evidence_and_has_no_test_injection_hooks():
    parameters = inspect.signature(profiling.run_profile).parameters

    assert "inputs" not in parameters
    assert "stage_runner" not in parameters
    assert "cuda_api" not in parameters
    assert "torch_module" not in parameters
    assert "clock" not in parameters
    assert "repo_root" not in parameters
    assert "run_profile_software_test" not in profiling.__all__
