"""Integration contracts for Q-3D confidence cells and cohort omega.

These tests deliberately enter the real ``_run_joint_stage`` path.  The KD
spies call the production public losses and retain their gradients so this
file checks routing and effective gradients, rather than substituting a
constant loss for the reducer.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from tests.test_q3d_engine import _execute_joint_stage, _joint_stage_case

from self_audit.candidate_q3d import engine
from self_audit.candidate_q3d.data import Q3DJointSampler
from self_audit.candidate_q3d.engine import (
    _data_identity_without_labels,
    _initial_run_state,
    _object_sha256,
)
from self_audit.candidate_q3d.physiology import (
    save_physiology_cache,
    summarize_physiology_cohort,
)


def _case_with_frozen_rows(tmp_path: Path, arm: str, *, n_unlabeled: int = 2):
    case = _joint_stage_case(tmp_path, arm, n_unlabeled=n_unlabeled)
    case["config"]["training"]["stage_updates"]["J"] = 2
    case["config"]["losses"]["w_mode"] = "uniform" if arm == "v1u" else "physiology"
    case["config"]["losses"]["teacher_confidence_weighting"] = True
    modules = {name: case[name] for name in ("teacher", "student", "tokenizer", "ema")}
    state = _initial_run_state(
        _object_sha256(case["config"]), _data_identity_without_labels(case["data"]),
    )
    weights = [1.0, 0.1] + [0.1] * max(0, n_unlabeled - 2)
    state["physiology_rows"] = {
        patient: {"w_pat": weight}
        for patient, weight in zip(case["data"].unlabeled_patients, weights)
    }
    return case, modules, state


def _cycle_joint_sampler(monkeypatch: pytest.MonkeyPatch, data) -> None:
    """Make two updates visit the two U patients in a known order."""

    labeled_keys = {
        patient: next(
            key for key in data.labeled_frame_keys()
            if data.labeled_patient_for_key(key) == patient
        )
        for patient in data.labeled_patients
    }
    unlabeled_keys = {
        patient: next(
            key for key in data.unlabeled_frame_keys()
            if data.unlabeled_patient_for_key(key) == patient
        )
        for patient in data.unlabeled_patients
    }

    def sample_update(sampler):
        index = sampler.updates
        sampler.updates += 1
        labeled_patient = data.labeled_patients[index % len(data.labeled_patients)]
        unlabeled_patient = data.unlabeled_patients[index % len(data.unlabeled_patients)]
        return {
            "protocol": data.protocol,
            "logical_update": index,
            "labeled_patient": labeled_patient,
            "labeled_key": labeled_keys[labeled_patient],
            "unlabeled_patient": unlabeled_patient,
            "unlabeled_key": unlabeled_keys[unlabeled_patient],
        }

    monkeypatch.setattr(Q3DJointSampler, "sample_update", sample_update)


def _controlled_confidence_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real teacher classes/probabilities but provide valid varied confidence.

    The synthetic teacher is intentionally untrained, so its natural confidence
    is below the 0.7 gate.  This wrapper preserves the real target construction
    and only controls the declared confidence fixture used by the software test.
    """

    real_pool = engine._pooled_teacher_targets

    def controlled(region_logits, *, target_hw):
        probabilities, classes, _confidence = real_pool(region_logits, target_hw=target_hw)
        confidence = torch.full_like(_confidence, 0.85)
        confidence[..., 0, 0] = 0.0
        return probabilities, classes, confidence

    monkeypatch.setattr(engine, "_pooled_teacher_targets", controlled)


@pytest.mark.parametrize(
    ("arm", "loss_name"),
    (("v1", "codekd_loss"), ("v1u", "codekd_loss"), ("c1", "feature_kd_loss")),
)
def test_joint_stage_routes_confidence_cells_and_external_omega(
    tmp_path, monkeypatch, arm, loss_name,
):
    """Both production KD branches receive cell weights and cohort omega.

    The two U patients have raw physiology weights 1 and 0.1.  The full-cache
    normalization must therefore emit 1.81818 and 0.181818, while L stays at
    one.  The second update also proves the nonzero-ramp branch is exercised.
    """

    case, modules, state = _case_with_frozen_rows(tmp_path, arm)
    _cycle_joint_sampler(monkeypatch, case["data"])
    _controlled_confidence_targets(monkeypatch)

    real_loss = getattr(engine, loss_name)
    calls = []

    def spy(teacher_target, student_target, classes, weights, *, omega=1.0):
        actual, stats = real_loss(
            teacher_target, student_target, classes, weights, omega=omega,
        )
        reference, _reference_stats = real_loss(
            teacher_target, student_target, classes, weights, omega=1.0,
        )
        actual_gradient = torch.autograd.grad(
            actual, student_target, retain_graph=True,
        )[0]
        reference_gradient = torch.autograd.grad(
            reference, student_target, retain_graph=True,
        )[0]
        calls.append({
            "omega": float(omega),
            "loss": actual.detach().clone(),
            "reference_loss": reference.detach().clone(),
            "gradient": actual_gradient.detach().clone(),
            "reference_gradient": reference_gradient.detach().clone(),
            "weights": weights.detach().clone(),
            "stats": stats,
        })
        return actual, stats

    monkeypatch.setattr(engine, loss_name, spy)
    final_state = _execute_joint_stage(
        case, tmp_path / f"{arm}-joint", modules, state=state,
    )

    assert len(calls) == 4  # L/U for two optimizer updates
    expected_omegas = (
        [1.0, 1.0, 1.0, 1.0]
        if arm == "v1u" else
        [1.0, 1.0 / 0.55, 1.0, 0.1 / 0.55]
    )
    assert [call["omega"] for call in calls] == pytest.approx(expected_omegas)

    # The reference is the same real wrapper with only the external patient
    # multiplier removed.  Equality of both loss and VJP catches omega being
    # omitted, put into the cell denominator, or applied twice.
    for call in calls:
        assert call["reference_loss"].item() > 0.0
        expected_loss = call["reference_loss"] * call["omega"]
        expected_gradient = call["reference_gradient"] * call["omega"]
        assert torch.linalg.vector_norm(expected_gradient).item() > 0.0
        loss_relative_error = float(
            (call["loss"] - expected_loss).abs()
            / expected_loss.abs().clamp_min(torch.finfo(expected_loss.dtype).tiny)
        )
        gradient_relative_error = float(
            torch.linalg.vector_norm(call["gradient"] - expected_gradient)
            / torch.linalg.vector_norm(expected_gradient).clamp_min(
                torch.finfo(expected_gradient.dtype).tiny
            )
        )
        assert loss_relative_error <= 1e-6
        assert gradient_relative_error <= 1e-6
        assert call["stats"]["omega"] == pytest.approx(call["omega"])

    # The confidence-only cell map is unchanged between U patients; .1 must
    # appear only in omega, not as an extra factor in these cells.
    first_weights = calls[0]["weights"]
    assert float(first_weights.min()) == pytest.approx(0.0)
    assert float(first_weights.max()) == pytest.approx(0.85)
    # The controlled confidence zero is replicated over the native-z cells.
    assert int(torch.count_nonzero(first_weights == 0.0)) == first_weights.shape[-3]
    for call in calls[1:]:
        torch.testing.assert_close(call["weights"], first_weights)

    records = [
        record for record in final_state["metrics"]
        if record.get("stage") == "J" and "event" not in record
    ]
    assert [record["u_patient_omega"] for record in records] == pytest.approx(
        [1.0, 1.0] if arm == "v1u" else [1.0 / 0.55, 0.1 / 0.55],
    )
    assert [record["distillation_ramp"] for record in records] == pytest.approx([0.0, 1.0])
    assert final_state["j_optimizer_updates"] == 2
    assert final_state["j_kd_supported_branches"] == 4


@pytest.mark.parametrize(
    ("arm", "expected_omega"),
    (
        ("v1", {"train_unlabeled_000": 1.0 / 0.55, "train_unlabeled_001": 0.1 / 0.55}),
        ("c1", {"train_unlabeled_000": 1.0 / 0.55, "train_unlabeled_001": 0.1 / 0.55}),
        ("v1u", {"train_unlabeled_000": 1.0, "train_unlabeled_001": 1.0}),
    ),
)
def test_v1_c1_v1u_share_full_u_gate_and_l_is_unit(tmp_path, arm, expected_omega):
    case, _modules, state = _case_with_frozen_rows(tmp_path, arm)
    metadata = engine._cohort_weighting_metadata(
        case["config"], case["data"], state,
        context="weighting integration test", synthetic_override=False,
    )
    assert metadata is not None
    assert metadata["patient_ids"] == sorted(case["data"].unlabeled_patients)
    assert metadata["mean_w_pat"] == pytest.approx(0.55)
    assert metadata["ess_fraction"] == pytest.approx(0.599009900990099)
    assert metadata["gate_passed"] is True
    assert metadata["mean_omega"] == pytest.approx(1.0)
    assert metadata["omega"] == pytest.approx(expected_omega)

    labeled_patient = case["data"].labeled_patients[0]
    assert engine._new_patient_omega(
        case["config"], state, labeled_patient, labeled=True,
    ) == 1.0
    for patient, expected in expected_omega.items():
        assert engine._new_patient_omega(
            case["config"], state, patient, labeled=False,
        ) == pytest.approx(expected)

    # Changing confidence cells cannot mutate the frozen patient multiplier.
    confidence = torch.tensor([[[[0.85, 0.0], [0.85, 0.85]]]])
    before = engine._new_patient_omega(
        case["config"], state, case["data"].unlabeled_patients[1], labeled=False,
    )
    cell_weights = engine._cell_weights(
        case["config"], confidence, case["data"].unlabeled_patients[1], state,
        labeled=False,
    )
    after = engine._new_patient_omega(
        case["config"], state, case["data"].unlabeled_patients[1], labeled=False,
    )
    torch.testing.assert_close(
        cell_weights,
        torch.tensor([[[[0.85, 0.0], [0.85, 0.85]]]]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert after == before == pytest.approx(expected_omega[case["data"].unlabeled_patients[1]])


@pytest.mark.parametrize(
    ("weights", "expected_status", "n_unlabeled"),
    (
        ([0.0, 0.0], "NO_GO_ALL_U_WEIGHTS_ZERO", 2),
        ([1.0, 0.001, 0.001, 0.001, 0.001], "NO_GO_U_ESS_GATE", 5),
    ),
)
def test_real_joint_rejects_invalid_u_gate_even_when_override_flag_is_set(
    tmp_path, weights, expected_status, n_unlabeled,
):
    case = _joint_stage_case(tmp_path, "v1", n_unlabeled=n_unlabeled)
    case["config"]["training"]["stage_updates"]["J"] = 1
    case["config"]["losses"]["w_mode"] = "physiology"
    case["config"]["runtime"]["synthetic"] = False
    modules = {name: case[name] for name in ("teacher", "student", "tokenizer", "ema")}
    state = _initial_run_state(
        _object_sha256(case["config"]), _data_identity_without_labels(case["data"]),
    )
    state["synthetic_gate_override_requested"] = True
    state["physiology_rows"] = {
        patient: {"w_pat": value}
        for patient, value in zip(case["data"].unlabeled_patients, weights)
    }

    with pytest.raises(RuntimeError, match="physiology cache reload"):
        _execute_joint_stage(case, tmp_path / "invalid-gate", modules, state=state)
    assert state["physiology_support_gate"]["status"] == expected_status
    assert state["physiology_support_gate"]["passed"] is False
    assert state["physiology_support_gate"]["override_used"] is False


def _resume_cache_state(tmp_path: Path, case, rows):
    teacher_sha = str(rows[0]["teacher_sha256"])
    cache_path = tmp_path / "physiology_cache.json"
    cache_sha = save_physiology_cache(
        cache_path, rows, teacher_sha256=teacher_sha, codebook_sha256=None,
        protocol="A", source_identity=case["data"].cache_source_identity(),
    )
    summary = summarize_physiology_cohort(rows)
    state = _initial_run_state(
        _object_sha256(case["config"]), _data_identity_without_labels(case["data"]),
    )
    state["completed_stages"] = ["P"]
    state["physiology_rows"] = {row["id"]: dict(row) for row in rows}
    state["physiology_cache"] = {
        "path": str(cache_path),
        # save_physiology_cache returns the SHA of the canonical payload, which
        # is the cache identity persisted by the production P stage.
        "sha256": cache_sha,
        "teacher_sha256": teacher_sha,
        "codebook_sha256": None,
        "source_identity": case["data"].cache_source_identity(),
        "cohort_weighting": summary,
    }
    # A current checkpoint seals the normalized omega map alongside the cache
    # summary; older checkpoints intentionally omit this field in the test
    # branches below.
    engine._cohort_weighting_metadata(
        case["config"], case["data"], state,
        context="resume fixture", synthetic_override=False,
    )
    return cache_path, state, summary


def test_resume_requires_sealed_cohort_metadata_and_rejects_tampering(tmp_path):
    case, _modules, _unused_state = _case_with_frozen_rows(tmp_path, "c1")
    teacher_sha = engine._state_sha256(case["teacher"])
    rows = [
        {
            "id": patient,
            "w_pat": weight,
            "protocol": "A",
            "teacher_sha256": teacher_sha,
            "codebook_sha256": None,
        }
        for patient, weight in zip(case["data"].unlabeled_patients, (1.0, 0.1))
    ]
    cache_path, state, summary = _resume_cache_state(tmp_path, case, rows)

    # A valid cache and checkpoint resume together are accepted first.  This
    # binds the test to the payload SHA returned by save_physiology_cache.
    engine._validate_resumed_physiology_cache(
        case["config"], case["data"], state,
        output_dir=tmp_path, teacher=case["teacher"], tokenizer=None,
    )
    assert state["physiology_cohort_weighting"]["mean_omega"] == pytest.approx(1.0)
    assert state["physiology_cache"]["cohort_weighting"] == summary

    # An old cache with no sealed summary is rejected, even if its payload
    # digest is updated to the newly written bytes.
    old_cache_state = copy.deepcopy(state)
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    document["payload"].pop("cohort_weighting")
    document["sha256"] = hashlib.sha256(
        json.dumps(
            document["payload"], sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    cache_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    old_cache_state["physiology_cache"]["sha256"] = document["sha256"]
    with pytest.raises(RuntimeError, match="old physiology cache without cohort_weighting"):
        engine._validate_resumed_physiology_cache(
            case["config"], case["data"], old_cache_state,
            output_dir=tmp_path, teacher=case["teacher"], tokenizer=None,
        )

    # A checkpoint created before the omega metadata was added is rejected even
    # when the sealed cache itself is valid.
    cache_sha = save_physiology_cache(
        cache_path, rows, teacher_sha256=teacher_sha, codebook_sha256=None,
        protocol="A", source_identity=case["data"].cache_source_identity(),
    )
    state["physiology_cache"]["sha256"] = cache_sha
    missing_checkpoint_metadata = copy.deepcopy(state)
    missing_checkpoint_metadata.pop("physiology_cohort_weighting", None)
    with pytest.raises(RuntimeError, match="missing frozen physiology cohort_weighting metadata"):
        engine._validate_resumed_physiology_cache(
            case["config"], case["data"], missing_checkpoint_metadata,
            output_dir=tmp_path, teacher=case["teacher"], tokenizer=None,
        )

    changed_teacher = torch.nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="teacher snapshot differs"):
        engine._validate_resumed_physiology_cache(
            case["config"], case["data"], state,
            output_dir=tmp_path, teacher=changed_teacher, tokenizer=None,
        )

    # Swapping the two patient weights leaves mean and ESS unchanged. Even a
    # matching rederived runtime omega map must not override the sealed rows.
    swapped = copy.deepcopy(state)
    first, second = (row["id"] for row in rows)
    swapped["physiology_rows"][first]["w_pat"] = rows[1]["w_pat"]
    swapped["physiology_rows"][second]["w_pat"] = rows[0]["w_pat"]
    engine._cohort_weighting_metadata(
        case["config"], case["data"], swapped,
        context="swapped checkpoint", synthetic_override=False,
    )
    with pytest.raises(RuntimeError, match="patient rows differ"):
        engine._validate_resumed_physiology_cache(
            case["config"], case["data"], swapped,
            output_dir=tmp_path, teacher=case["teacher"], tokenizer=None,
        )

    # O1 intentionally refreshes the teacher while retaining the cache's
    # recorded snapshot identity.  A changed current teacher is therefore
    # accepted for O1, but the stored payload/checkpoint SHA is untouched.
    o1_case, _o1_modules, _o1_unused_state = _case_with_frozen_rows(tmp_path / "o1", "c1")
    o1_case["config"]["arm"] = "o1"
    o1_teacher_sha = engine._state_sha256(o1_case["teacher"])
    o1_rows = [
        {
            "id": patient,
            "w_pat": weight,
            "protocol": "A",
            "teacher_sha256": o1_teacher_sha,
            "codebook_sha256": None,
        }
        for patient, weight in zip(o1_case["data"].unlabeled_patients, (1.0, 0.1))
    ]
    o1_cache_path, o1_state, _o1_summary = _resume_cache_state(
        tmp_path / "o1", o1_case, o1_rows,
    )
    o1_changed_teacher = torch.nn.Linear(1, 1)
    engine._validate_resumed_physiology_cache(
        o1_case["config"], o1_case["data"], o1_state,
        output_dir=tmp_path / "o1", teacher=o1_changed_teacher, tokenizer=None,
    )
    assert o1_state["physiology_cache"]["path"] == str(o1_cache_path)
    assert o1_state["physiology_cache"]["teacher_sha256"] == o1_teacher_sha

    # Finally reject a checkpoint whose omega was changed independently of the
    # sealed cache.
    tampered = copy.deepcopy(state)
    tampered["physiology_cohort_weighting"]["omega"][rows[0]["id"]] = 9.0
    with pytest.raises(RuntimeError, match="stale or inconsistent"):
        engine._validate_resumed_physiology_cache(
            case["config"], case["data"], tampered,
            output_dir=tmp_path, teacher=case["teacher"], tokenizer=None,
        )
