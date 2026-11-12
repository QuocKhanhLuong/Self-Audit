"""Focused synthetic contracts for the Q-3D losses and trainer resume path."""

import copy
import math

import pytest
import torch

from self_audit.candidate_q3d.losses import (
    balanced_near_far_sdf_loss,
    codekd_loss,
    commitment_loss,
    feature_kd_loss,
    q3d_teacher_loss,
)


class _ResumeProbeStudent(torch.nn.Module):
    """Small stochastic model for deterministic interrupted-resume coverage."""

    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout3d(p=0.25)
        self.logit_head = torch.nn.Conv3d(1, 4, kernel_size=1)
        self.sdf_head = torch.nn.Conv3d(1, 4, kernel_size=1)

    def forward(self, image, *, geometry):
        value = self.dropout(image)
        return {
            "logits": self.logit_head(value),
            "sdf": torch.tanh(self.sdf_head(value)),
        }


def test_near_far_sdf_uses_equal_nonempty_band_means_and_renormalizes():
    target = torch.zeros((1, 4, 1, 1, 4))
    target[..., 2:] = 1.0  # far band at the clipped +/-5 mm boundary
    prediction = torch.zeros_like(target)
    prediction[..., :2] = 1.0
    prediction[..., 2:] = 0.5
    loss = balanced_near_far_sdf_loss(prediction, target)
    assert torch.isclose(loss, torch.tensor(0.75))

    # Empty far band gets no residual 0.5 weight; the only populated band is
    # renormalized to one instead of being silently halved.
    near_only = torch.zeros((1, 4, 1, 1, 2))
    assert torch.isclose(balanced_near_far_sdf_loss(torch.ones_like(near_only), near_only), torch.tensor(1.0))


def test_q3d_teacher_loss_keeps_two_segmentation_heads_and_near_far_wrapper():
    labels = torch.tensor([[[[0, 1, 1, 0]]]])
    logits = torch.zeros((1, 4, 1, 1, 4), requires_grad=True)
    target = torch.zeros_like(logits)
    prediction = torch.ones_like(logits, requires_grad=True)
    outputs = {"region_logits": logits, "boundary_logits": logits, "sdf": prediction}
    loss, stats = q3d_teacher_loss(outputs, labels, target)
    assert set(stats) == {"region", "boundary", "sdf_near_far"}
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert prediction.grad is not None


def test_codekd_returns_per_cell_kl_map_without_reducing_classes_or_weights():
    teacher = torch.tensor(
        [[[[[1.0, 1.0, 1.0, 1.0]]], [[[0.0, 0.0, 0.0, 0.0]]]]],
        requires_grad=True,
    )
    student = torch.tensor(
        [[[[[1.0, 1.0, 0.5, 0.5]]], [[[0.0, 0.0, 0.5, 0.5]]]]],
        requires_grad=True,
    )
    per_cell = codekd_loss(teacher, student)
    expected = torch.tensor([[[[0.0, 0.0, math.log(2.0), math.log(2.0)]]]])
    assert per_cell.shape == (1, 1, 1, 4)
    assert torch.allclose(per_cell, expected, atol=1e-6)
    per_cell.sum().backward()
    assert teacher.grad is None
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_codekd_equal_targets_return_graph_connected_per_cell_zeros():
    student = torch.full((1, 2, 1, 1, 2), 0.5, requires_grad=True)
    per_cell = codekd_loss(student.detach(), student)
    assert per_cell.shape == (1, 1, 1, 2)
    assert torch.equal(per_cell, torch.zeros_like(per_cell))
    assert per_cell.grad_fn is not None
    per_cell.sum().backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_codekd_rejects_invalid_probability_mass():
    teacher = torch.full((1, 2, 1, 1, 1), 0.5)
    with pytest.raises(ValueError, match="positive mass"):
        codekd_loss(torch.zeros_like(teacher), teacher)


def test_feature_kd_returns_per_cell_cosine_map_and_detaches_teacher():
    teacher = torch.tensor([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]], requires_grad=True)
    student = torch.tensor([[[[[1.0, 1.0]]], [[[0.0, 0.0]]]]], requires_grad=True)
    per_cell = feature_kd_loss(teacher, student)
    assert per_cell.shape == (1, 1, 1, 2)
    assert torch.allclose(per_cell, torch.tensor([[[[0.0, 1.0]]]]), atol=1e-6)
    per_cell.sum().backward()
    assert teacher.grad is None
    assert student.grad is not None


def test_commitment_updates_adapter_only():
    u = torch.tensor([1.0, 0.0], requires_grad=True)
    code = torch.tensor([0.0, 1.0], requires_grad=True)
    commitment_loss(u, code).backward()
    assert u.grad is not None
    assert code.grad is None


def test_synthetic_c0_training_resume_matches_uninterrupted_state(tmp_path):
    from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
    from self_audit.candidate_q3d.config import canonical_config
    from self_audit.candidate_q3d.data import Q3DData
    from self_audit.candidate_q3d.engine import train

    fixture = synthetic_fixture(
        tmp_path / "fixture", seed=19, n_labeled=2, n_unlabeled=2, n_evaluation=1,
        shape_tzyx=(2, 4, 16, 17),
    )
    student_data = CandidateA3DData(fixture["train"], fixture["train_sha256"], synthetic=True)
    data = Q3DData(student_data, protocol="A")
    config = canonical_config("c0", dataset="acdc", protocol="A", k=2, seed=42)
    config["runtime"].update({"synthetic": True, "device": "cpu"})
    config["runtime"]["smoke_update_overrides"] = {
        "T": None, "Q": None, "P": None, "J": 2,
    }
    config["dataset"]["manifests"]["train"] = {
        "path": str(fixture["train"].resolve()), "sha256": fixture["train_sha256"], "status": "READY",
    }
    config["dataset"]["manifests"]["development"] = {
        "path": str(fixture["evaluation"].resolve()),
        "sha256": fixture["evaluation_sha256"], "status": "READY",
    }

    def evaluate(_model, _manifest, _kind, _device):
        return {"score": 0.5, "patient_count": 1}

    torch.manual_seed(810)
    initial = _ResumeProbeStudent()
    uninterrupted_student = copy.deepcopy(initial)
    resumed_student = copy.deepcopy(initial)
    uninterrupted = train(
        config, tmp_path / "uninterrupted", device="cpu", data=data,
        student=uninterrupted_student, smoke=True, dev_evaluator=evaluate,
    )
    first_leg = train(
        config, tmp_path / "resumed", device="cpu", data=data,
        student=resumed_student, smoke=True, update_limit=1, dev_evaluator=evaluate,
    )
    assert first_leg["status"] == "PAUSED"
    second_leg = train(
        config, tmp_path / "resumed", device="cpu", data=data,
        student=_ResumeProbeStudent(), smoke=True, update_limit=1, dev_evaluator=evaluate,
    )
    assert uninterrupted["status"] == second_leg["status"] == "COMPLETED"
    uninterrupted_state = torch.load(uninterrupted["checkpoint"], map_location="cpu", weights_only=False)
    resumed_state = torch.load(second_leg["checkpoint"], map_location="cpu", weights_only=False)
    assert uninterrupted_state["j_optimizer_updates"] == resumed_state["j_optimizer_updates"] == 2
    assert uninterrupted_state["sampler_state"] == resumed_state["sampler_state"]
    assert uninterrupted_state["student_state"].keys() == resumed_state["student_state"].keys()
    for name, value in uninterrupted_state["student_state"].items():
        assert torch.equal(value, resumed_state["student_state"][name]), name


def test_q_stage_image_identity_and_l_frame_loader_never_open_masks(tmp_path):
    from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
    from self_audit.candidate_q3d.data import Q3DData
    from self_audit.candidate_q3d.engine import (
        _data_identity_without_labels,
        _labeled_image_frame_keys,
        _load_labeled_image_only,
    )

    fixture = synthetic_fixture(
        tmp_path / "fixture", seed=9, n_labeled=2, n_unlabeled=1, n_evaluation=1,
        shape_tzyx=(3, 3, 12, 13),
    )
    student_data = CandidateA3DData(fixture["train"], fixture["train_sha256"], synthetic=True)
    data = Q3DData(student_data, protocol="A")
    mask_paths = {
        mask["path"]
        for patient in student_data.labeled
        for mask in patient["masks"].values()
    }
    original_array = student_data.base.array
    touched_masks: list[str] = []

    def image_only_array(path, digest):
        if path in mask_paths:
            touched_masks.append(path)
            raise AssertionError("Q-stage image-only helper opened a TRAIN mask")
        return original_array(path, digest)

    student_data.base.array = image_only_array
    identity = _data_identity_without_labels(data)
    keys = _labeled_image_frame_keys(data)
    image, geometry, metadata = _load_labeled_image_only(data, keys[-1])
    assert identity["protocol"] == "A"
    assert len(keys) == 6  # every labeled cine frame, including unannotated frames
    assert tuple(image.shape[-3:]) == tuple(geometry["shape"])
    assert metadata["patient"] in data.labeled_patients
    assert touched_masks == []


class _NestedExtraStateProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scalar", torch.tensor(3.0))
        self.register_buffer(
            "channels_last", torch.ones((1, 2, 2, 2, 3)).to(memory_format=torch.channels_last_3d),
        )
        self.extra_tensor = torch.tensor([1.0, 2.0])

    def get_extra_state(self):
        return {"metadata": {"version": "v1", "weights": self.extra_tensor}}

    def set_extra_state(self, state):
        self.extra_tensor = state["metadata"]["weights"].clone()


def test_state_sha256_handles_scalar_channels_last_and_nested_extra_state():
    from self_audit.candidate_q3d.engine import _state_sha256

    first = _NestedExtraStateProbe()
    second = _NestedExtraStateProbe()
    second.load_state_dict(first.state_dict())
    assert _state_sha256(first) == _state_sha256(second)
    second.extra_tensor[0] = 9.0
    assert _state_sha256(first) != _state_sha256(second)

    class ShapeProbe(torch.nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.register_buffer("values", torch.arange(6, dtype=torch.float32).reshape(shape))

    assert _state_sha256(ShapeProbe((2, 3))) != _state_sha256(ShapeProbe((3, 2)))


def test_j_unit_counts_and_loss_scales_keep_l_supervision_at_one():
    from self_audit.candidate_q3d.config import canonical_config, resolve_stage_accounting
    from self_audit.candidate_q3d.engine import _joint_objective_scales

    paired = resolve_stage_accounting(canonical_config("v1"))
    assert paired["stage_updates"]["J"] == 8000
    assert paired["optimizer_updates"]["J"] == 8000
    assert paired["stream_microbatches"]["J"] == 16000
    l_only = resolve_stage_accounting(canonical_config("c0"))
    assert l_only["optimizer_updates"]["J"] == 8000
    assert l_only["stream_microbatches"]["J"] == 8000

    scales = _joint_objective_scales(
        has_unlabeled_stream=True, has_distillation=True, has_commitment=True,
    )
    assert scales == {
        "logical_update_unit": "L_U_pair",
        "accumulation_unit": "stream_microbatch",
        "accumulation_microbatches": 2,
        "optimizer_step_per_logical_update": 1,
        "supervised_l_coefficient": 1.0,
        "kd_stream_divisor": 2.0,
        "commitment_stream_divisor": 2.0,
    }


def test_vq_ema_copies_frozen_c0_and_extra_state_exactly():
    import copy
    import hashlib

    from self_audit.candidate_q3d.engine import _mean_teacher_update
    from self_audit.candidate_q3d.student import CandidateQ3DStudent

    student_config = {
        "student": {
            "arm": "V1", "channels": [4, 4, 4, 4, 4], "num_classes": 4,
            "in_channels": 1, "vq": {"size": 256, "dim": 64},
            "attention": {"support": "uniform", "support_count": 64, "geometric_fraction": 0.5},
            "residual_vq": {"enabled": False, "size": 256},
        },
    }
    student = CandidateQ3DStudent(student_config)
    ema = copy.deepcopy(student).requires_grad_(False)
    codes = torch.nn.functional.normalize(torch.randn(256, 64), dim=1)
    sha = hashlib.sha256(codes.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    artifact = {
        "codes": codes, "version": "q3d_teacher_codebook.v1", "sha256": sha,
        "size": 256, "dimension": 64, "initialized": True, "frozen": True,
    }
    student.install_codebook(artifact)
    ema.install_codebook(artifact)
    with torch.no_grad():
        source_parameter = next(student.parameters())
        target_parameter = next(ema.parameters())
        source_parameter.fill_(2.0)
        target_parameter.zero_()
        student.C0.copy_(torch.nn.functional.normalize(torch.randn_like(student.C0), dim=1))

    _mean_teacher_update(ema, student, decay=0.5)
    assert torch.equal(ema.C0, student.C0)
    assert ema.codebook_metadata() == student.codebook_metadata()
    assert torch.allclose(next(ema.parameters()), torch.ones_like(next(ema.parameters())))


def test_teacher_optimizer_assigns_region_decoder_head_lr_and_o1_scale():
    from self_audit.candidate_q3d.config import canonical_config
    from self_audit.candidate_q3d.engine import _make_o1_teacher_optimizer, _make_teacher_optimizer

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.stem = torch.nn.Conv3d(1, 4, 1)
            self.model.up_3 = torch.nn.Conv3d(4, 4, 1)
            self.model.dec_block_3 = torch.nn.Conv3d(4, 4, 1)
            self.model.out_0 = torch.nn.Conv3d(4, 4, 1)

    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.region_path = torch.nn.Identity()
            self.boundary_path = torch.nn.Conv3d(4, 4, 1)

    teacher = Teacher()
    config = canonical_config("v1")
    pretrain = _make_teacher_optimizer(teacher, config, warmup=False)
    group_lrs = {group["name"]: group["lr"] for group in pretrain.param_groups}
    assert group_lrs == {"teacher_trunk": 1e-4, "teacher_heads": 1e-3}

    config["training"]["teacher_joint_lr_multiplier"] = 0.1
    online = _make_o1_teacher_optimizer(teacher, config)
    group_lrs = {group["name"]: group["lr"] for group in online.param_groups}
    assert group_lrs == {"teacher_trunk": 1e-5, "teacher_heads": 1e-4}


@pytest.mark.parametrize("arm", ["c0", "mean_teacher"])
def test_no_teacher_student_models_build_when_feature_channels_are_absent(arm):
    from self_audit.candidate_q3d.config import canonical_config
    from self_audit.candidate_q3d.engine import _build_models

    config = canonical_config(arm)
    config["runtime"]["synthetic"] = True
    config["model"]["student"]["stage_channels"] = [8, 8, 8, 8, 8]
    assert config["model"]["teacher"]["feature_channels"] is None
    teacher, student, tokenizer, _provenance = _build_models(
        config, device=torch.device("cpu"), teacher=None, student=None, tokenizer=None,
    )
    assert teacher is None
    assert isinstance(student, torch.nn.Module)
    assert tokenizer is None


def _joint_stage_case(tmp_path, arm, *, n_unlabeled=2):
    import copy

    from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
    from self_audit.candidate_q3d.config import canonical_config
    from self_audit.candidate_q3d.data import Q3DData
    from self_audit.candidate_q3d.engine import _build_models

    fixture = synthetic_fixture(
        tmp_path / f"fixture-{arm}", seed=34, n_labeled=2, n_unlabeled=n_unlabeled,
        n_evaluation=1, shape_tzyx=(2, 8, 32, 32),
    )
    raw_data = CandidateA3DData(fixture["train"], fixture["train_sha256"], synthetic=True)
    data = Q3DData(raw_data, protocol="A")
    config = canonical_config(arm, dataset="acdc", protocol="A", k=2, seed=42)
    config["runtime"].update({"synthetic": True, "device": "cpu"})
    config["model"]["student"]["stage_channels"] = [8, 8, 8, 8, 8]
    config["model"]["teacher"]["feature_channels"] = 32
    config["dataset"]["manifests"]["train"] = {
        "path": str(fixture["train"].resolve()), "sha256": fixture["train_sha256"], "status": "READY",
    }
    config["dataset"]["manifests"]["development"] = {
        "path": str(fixture["evaluation"].resolve()),
        "sha256": fixture["evaluation_sha256"], "status": "READY",
    }
    config["losses"]["teacher_confidence_weighting"] = False
    config["losses"]["w_mode"] = "none"
    torch.manual_seed(435)
    teacher, student, tokenizer, _provenance = _build_models(
        config, device=torch.device("cpu"), teacher=None, student=None, tokenizer=None,
    )
    if tokenizer is not None:
        with torch.no_grad():
            tokenizer.C0.copy_(torch.nn.functional.normalize(torch.randn_like(tokenizer.C0), dim=1))
            tokenizer.ema_counts.fill_(1.0)
            tokenizer.ema_sums.copy_(tokenizer.C0)
            tokenizer.is_initialized.fill_(True)
            tokenizer.is_frozen.fill_(True)
        tokenizer.eval().requires_grad_(False)
    return {
        "fixture": fixture, "data": data, "config": config,
        "teacher": teacher, "student": student, "tokenizer": tokenizer,
        "ema": copy.deepcopy(student).eval().requires_grad_(False),
    }


def _execute_joint_stage(case, output_dir, modules, *, update_limit=None, state=None, dev_evaluator=None):
    import random

    from self_audit.candidate_q3d.data import Q3DJointSampler
    from self_audit.candidate_q3d.engine import (
        _data_identity_without_labels, _initial_run_state, _object_sha256,
        _restore_run_state, _run_joint_stage, _seed_all,
    )

    config, data = case["config"], case["data"]
    spatial_rng, intensity_generator = _seed_all(42)
    stage_rng = random.Random(89)
    sampler = Q3DJointSampler(data, seed=42, require_unlabeled=(config["arm"] != "c0"))
    if state is None:
        state = _initial_run_state(
            _object_sha256(config), _data_identity_without_labels(data),
        )
    else:
        if state.get("student_state") is not None:
            _restore_run_state(
                state, teacher=modules["teacher"], student=modules["student"],
                tokenizer=modules["tokenizer"], ema_model=modules["ema"],
                spatial_rng=spatial_rng, intensity_generator=intensity_generator,
                stage_rng=stage_rng,
            )
    if update_limit is not None:
        state["requested_update_limit"] = int(state["completed_updates_total"]) + update_limit

    def evaluate(_model, _manifest, _kind, _device):
        return {"score": 0.5, "patient_count": 1}

    _run_joint_stage(
        config, data=data, teacher=modules["teacher"], student=modules["student"],
        tokenizer=modules["tokenizer"], ema_model=modules["ema"], state=state,
        output_dir=output_dir, device=torch.device("cpu"), spatial_rng=spatial_rng,
        intensity_generator=intensity_generator, stage_rng=stage_rng, sampler=sampler,
        smoke=True, update_limit_active=update_limit is not None,
        dev_evaluator=dev_evaluator or evaluate, update_budget=int(config["training"]["stage_updates"]["J"]),
    )
    return state


def test_v1_joint_resume_after_complete_pair_matches_uninterrupted_state(tmp_path):
    import copy

    from self_audit.candidate_q3d.engine import (
        _load_checkpoint, _same_state_value,
    )

    case = _joint_stage_case(tmp_path, "v1")
    case["config"]["training"]["stage_updates"]["J"] = 2
    base = {name: copy.deepcopy(case[name]) for name in ("teacher", "student", "tokenizer", "ema")}

    uninterrupted_modules = {name: copy.deepcopy(value) for name, value in base.items()}
    uninterrupted = _execute_joint_stage(
        case, tmp_path / "v1-uninterrupted", uninterrupted_modules,
    )

    first_modules = {name: copy.deepcopy(value) for name, value in base.items()}
    output = tmp_path / "v1-resumed"
    first_leg = _execute_joint_stage(case, output, first_modules, update_limit=1)
    assert first_leg["j_optimizer_updates"] == 1
    assert first_leg["j_logical_pairs"] == 1
    assert first_leg["j_stream_microbatches"] == 2

    data_identity = first_leg["data_identity"]
    resume_state = _load_checkpoint(
        output / "latest.pt", config_sha256=first_leg["config_sha256"], data_identity=data_identity,
    )
    resumed_modules = {name: copy.deepcopy(value) for name, value in base.items()}
    resumed = _execute_joint_stage(
        case, output, resumed_modules, update_limit=1, state=resume_state,
    )
    assert uninterrupted["j_optimizer_updates"] == resumed["j_optimizer_updates"] == 2
    assert uninterrupted["j_logical_pairs"] == resumed["j_logical_pairs"] == 2
    assert uninterrupted["j_stream_microbatches"] == resumed["j_stream_microbatches"] == 4
    assert uninterrupted["sampler_state"] == resumed["sampler_state"]
    assert uninterrupted["j_student_code_usage"] == resumed["j_student_code_usage"]
    assert uninterrupted["j_student_code_usage_last"]["status"] == "NOT_EVALUATED"
    for name in ("student_state", "ema_state", "tokenizer_state"):
        left, right = uninterrupted[name], resumed[name]
        assert left.keys() == right.keys()
        assert all(_same_state_value(left[key], right[key]) for key in left), name
    assert torch.equal(resumed_modules["student"].C0, resumed_modules["ema"].C0)
    assert resumed_modules["student"].codebook_metadata() == resumed_modules["ema"].codebook_metadata()


def test_o1_two_pair_run_reenables_teacher_gradients_each_update(tmp_path):
    from self_audit.candidate_q3d.engine import _same_state_value

    case = _joint_stage_case(tmp_path, "o1")
    case["config"]["training"]["stage_updates"]["J"] = 2
    modules = {name: case[name] for name in ("teacher", "student", "tokenizer", "ema")}
    initial_teacher = {key: value.detach().clone() for key, value in modules["teacher"].state_dict().items()
                       if isinstance(value, torch.Tensor)}
    state = _execute_joint_stage(case, tmp_path / "o1-two-pairs", modules)
    updates = [record for record in state["metrics"]
               if record.get("stage") == "J" and record.get("losses", {}).get("teacher_online") is not None]
    assert len(updates) == 2
    assert state["j_optimizer_updates"] == 2
    assert any(not _same_state_value(value, state["teacher_state"][name])
               for name, value in initial_teacher.items())


def test_c1_keeps_zero_cell_confidence_u_backward_graph_with_positive_patient_support(tmp_path, monkeypatch):
    case = _joint_stage_case(tmp_path, "c1")
    case["config"]["training"]["stage_updates"]["J"] = 1
    case["config"]["losses"]["w_mode"] = "physiology"
    case["config"]["losses"]["continuous_feature_kd"] = 1.0
    case["config"]["losses"]["teacher_confidence_weighting"] = False
    modules = {name: case[name] for name in ("teacher", "student", "tokenizer", "ema")}
    assert modules["tokenizer"] is None
    from self_audit.candidate_q3d.engine import _data_identity_without_labels, _initial_run_state, _object_sha256

    state = _initial_run_state(
        _object_sha256(case["config"]), _data_identity_without_labels(case["data"]),
    )
    state["physiology_rows"] = {
        patient: {"w_pat": 0.5} for patient in case["data"].unlabeled_patients
    }
    cell_weights = __import__("self_audit.candidate_q3d.engine", fromlist=["_cell_weights"])._cell_weights

    def zero_only_unlabeled(config, confidence, patient_id, run_state, *, labeled):
        if not labeled:
            return torch.zeros_like(confidence)
        return cell_weights(config, confidence, patient_id, run_state, labeled=True)

    monkeypatch.setattr("self_audit.candidate_q3d.engine._cell_weights", zero_only_unlabeled)
    state = _execute_joint_stage(case, tmp_path / "c1-zero-u", modules, state=state)
    update = next(record for record in state["metrics"]
                  if record.get("stage") == "J" and record.get("update") == 1)
    assert update["losses"]["unlabeled_distillation"] == pytest.approx(0.0)
    assert update["kd_zero_weight_branches"] == ["U"]
    assert update["kd_supported_branches"] == 1


def test_c1_allzero_u_patient_support_fails_before_any_j_update(tmp_path):
    import self_audit.candidate_q3d.engine as engine

    case = _joint_stage_case(tmp_path, "c1")
    case["config"]["training"]["stage_updates"]["J"] = 1
    case["config"]["losses"]["w_mode"] = "physiology"
    modules = {name: case[name] for name in ("teacher", "student", "tokenizer", "ema")}
    state = engine._initial_run_state(
        engine._object_sha256(case["config"]), engine._data_identity_without_labels(case["data"]),
    )
    state["physiology_rows"] = {
        patient: {"w_pat": 0.0} for patient in case["data"].unlabeled_patients
    }
    initial_student = copy.deepcopy(case["student"].state_dict())
    with pytest.raises(RuntimeError, match="zero support for every TRAIN U patient"):
        _execute_joint_stage(case, tmp_path / "c1-all-zero", modules, state=state)
    assert state.get("j_optimizer_updates", 0) == 0
    assert state["physiology_support_gate"]["status"] == "NO_GO_ALL_U_WEIGHTS_ZERO"
    assert not any(item.get("stage") == "J" and item.get("event") not in {
        "physiology_support_gate", "physiology_weighting",
    }
                   for item in state["metrics"])
    for name, value in case["student"].state_dict().items():
        if torch.is_tensor(value):
            assert torch.equal(value, initial_student[name]), name
        else:
            assert value == initial_student[name], name


def test_development_and_report_share_frame_then_patient_dice_with_unequal_support():
    import numpy as np

    from self_audit.candidate_q3d.evaluation import (
        _frame_metrics, _patient_aggregate, aggregate_patient_class_dice, frame_class_dice,
    )

    # The large second frame has 20 reference voxels and Dice 0.5; the small
    # first frame has one reference voxel and Dice 1.0. Pooling counts yields
    # 0.5238..., while the locked frame-macro patient score is exactly 0.75.
    ref_small = np.array([1, 0], dtype=np.uint8).reshape(1, 1, 2)
    pred_small = np.array([1, 0], dtype=np.uint8).reshape(1, 1, 2)
    ref_large = np.zeros(100, dtype=np.uint8)
    pred_large = np.zeros(100, dtype=np.uint8)
    ref_large[:20] = 1
    pred_large[:10] = 1
    pred_large[20:30] = 1
    ref_large = ref_large.reshape(1, 1, 100)
    pred_large = pred_large.reshape(1, 1, 100)
    rows = [frame_class_dice(ref_small, pred_small), frame_class_dice(ref_large, pred_large)]
    shared = aggregate_patient_class_dice(rows)
    report = _patient_aggregate([
        {"metrics": _frame_metrics(ref_small, pred_small, (1, 1, 1))},
        {"metrics": _frame_metrics(ref_large, pred_large, (1, 1, 1))},
    ])
    assert shared["dice_by_class"]["RV"]["dice"] == pytest.approx(0.75)
    assert shared["fg_dice"] == pytest.approx(0.75)
    assert report["dice_by_class"]["RV"]["dice"] == pytest.approx(shared["dice_by_class"]["RV"]["dice"])
    assert report["fg_dice"] == pytest.approx(shared["fg_dice"])


def test_pooled_teacher_targets_native_centers_on_odd_height_width():
    from self_audit.candidate_q3d.engine import _pooled_teacher_targets

    logits = torch.randn(1, 4, 6, 17, 19)
    probs, classes, confidence = _pooled_teacher_targets(
        logits, target_hw=(2, 2), student_phase=((1, 16, 16), (0, 0, 0)),
    )
    assert probs.shape == (1, 4, 6, 2, 2)
    assert classes.shape == (1, 6, 2, 2)
    assert confidence.shape == (1, 6, 2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(confidence), atol=1e-6)
    assert not probs.requires_grad and not confidence.requires_grad


def test_c0_joint_stage_uses_labeled_only_sampler_without_u_access(tmp_path, monkeypatch):
    from self_audit.candidate_q3d.data import Q3DJointSampler

    case = _joint_stage_case(tmp_path, "c0", n_unlabeled=0)
    case["config"]["training"]["stage_updates"]["J"] = 1
    data = case["data"]

    def forbidden_u_access(*_args, **_kwargs):
        raise AssertionError("C0 must not enumerate or read image-only U")

    monkeypatch.setattr(data, "unlabeled_frame_keys", forbidden_u_access)
    sampler = Q3DJointSampler(data, seed=42, require_unlabeled=False)
    record = sampler.sample_update()
    assert "unlabeled_key" not in record
    modules = {name: case[name] for name in ("teacher", "student", "tokenizer", "ema")}
    state = _execute_joint_stage(case, tmp_path / "c0-no-u", modules)
    update = next(item for item in state["metrics"] if item.get("stage") == "J")
    assert state["j_optimizer_updates"] == state["j_logical_updates"] == 1
    assert state["j_stream_microbatches"] == 1
    assert update["logical_update_unit"] == "L_only"
    assert update["sampler"].get("unlabeled_key") is None


def test_student_vq_collapse_guard_requires_full_post_ramp_window():
    from self_audit.candidate_q3d.engine import _update_student_code_usage

    state = {}
    status = None
    for _ in range(199):
        status = _update_student_code_usage(
            state, [[32, 0, 0, 0]], codebook_size=4, ramp_complete=True,
        )
    assert status["status"] == "NOT_EVALUATED"
    assert status["window_updates"] == 199

    status = _update_student_code_usage(
        state, [[32, 0, 0, 0]], codebook_size=4, ramp_complete=True,
    )
    assert status["status"] == "COLLAPSE_FAIL"
    assert status["evaluated"] is True
    assert status["active_codes"] == 1
    assert status["perplexity"] == pytest.approx(1.0)

    diverse = {}
    for index in range(200):
        status = _update_student_code_usage(
            diverse, [[32, 0, 0, 0] if index % 2 == 0 else [0, 32, 0, 0]],
            codebook_size=4, ramp_complete=True,
        )
    assert status["status"] == "NOT_COLLAPSED"
    assert status["active_codes"] == 2
    assert status["perplexity"] == pytest.approx(2.0)
