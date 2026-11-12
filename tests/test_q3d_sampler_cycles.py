from __future__ import annotations

import copy
import random

import pytest

from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
from self_audit.candidate_q3d.data import Q3DData, Q3DJointSampler


def _q3d_data(tmp_path, *, n_labeled=2, n_unlabeled=3, n_frames=3):
    fixture = synthetic_fixture(
        tmp_path,
        n_labeled=n_labeled,
        n_unlabeled=n_unlabeled,
        n_evaluation=1,
        shape_tzyx=(n_frames, 4, 8, 9),
    )
    source = CandidateA3DData(
        fixture["train"], fixture["train_sha256"], synthetic=True,
    )
    return Q3DData(source, protocol="A")


def test_u_patients_follow_seeded_full_permutations_and_report_cycle_progress(tmp_path):
    data = _q3d_data(tmp_path / "fixture")
    seed = 73
    sampler = Q3DJointSampler(data, seed=seed)
    permutation_rng = random.Random(seed + 1)
    frame_rng = random.Random(seed + 2)
    keys_by_patient = {patient: [] for patient in data.unlabeled_patients}
    for key in data.unlabeled_frame_keys():
        keys_by_patient[data.unlabeled_patient_for_key(key)].append(key)

    for cycle in range(1, 4):
        expected_patients = list(data.unlabeled_patients)
        permutation_rng.shuffle(expected_patients)
        records = [sampler.sample_update() for _ in expected_patients]
        assert [record["unlabeled_patient"] for record in records] == expected_patients
        assert len(set(record["unlabeled_patient"] for record in records)) == len(expected_patients)
        assert [record["unlabeled_cycle"] for record in records] == [cycle] * len(records)
        assert [record["unlabeled_cycle_position"] for record in records] == list(
            range(1, len(records) + 1)
        )
        assert [record["unlabeled_cycle_complete"] for record in records] == [
            False
        ] * (len(records) - 1) + [True]
        assert [record["unlabeled_key"] for record in records] == [
            frame_rng.choice(keys_by_patient[patient]) for patient in expected_patients
        ]
        assert sampler.completed_unlabeled_cycles == cycle

    state = sampler.state_dict()
    next_permutation = list(data.unlabeled_patients)
    permutation_rng.shuffle(next_permutation)
    assert state["schema"] == "self_audit.candidate_q3d.joint_sampler.v3"
    assert state["unlabeled_permutation"] == next_permutation
    assert state["unlabeled_cursor"] == 0
    assert state["unlabeled_cycle_count"] == 3
    assert "labeled_rng_state" in state
    assert "unlabeled_permutation_rng_state" in state
    assert "unlabeled_frame_rng_state" in state


def test_l_stream_matches_legacy_random_choice_algorithm_exactly(tmp_path):
    data = _q3d_data(tmp_path / "fixture", n_unlabeled=2)
    seed = 101
    sampler = Q3DJointSampler(data, seed=seed)
    legacy_rng = random.Random(seed)
    labeled_keys_by_patient = {patient: [] for patient in data.labeled_patients}
    for key in data.labeled_frame_keys():
        labeled_keys_by_patient[data.labeled_patient_for_key(key)].append(key)

    expected = []
    for _ in range(50):
        patient = legacy_rng.choice(data.labeled_patients)
        expected.append((patient, legacy_rng.choice(labeled_keys_by_patient[patient])))
    actual = [sampler.sample_update() for _ in range(len(expected))]
    assert [(record["labeled_patient"], record["labeled_key"]) for record in actual] == expected


def test_c0_and_paired_sampler_keep_identical_l_examples(tmp_path):
    data = _q3d_data(tmp_path / "fixture", n_unlabeled=4)
    paired = Q3DJointSampler(data, seed=42, require_unlabeled=True)
    c0 = Q3DJointSampler(data, seed=42, require_unlabeled=False)
    paired_records = [paired.sample_update() for _ in range(40)]
    c0_records = [c0.sample_update() for _ in range(40)]
    assert [
        (record["labeled_patient"], record["labeled_key"]) for record in paired_records
    ] == [
        (record["labeled_patient"], record["labeled_key"]) for record in c0_records
    ]
    assert "unlabeled_patient" not in c0_records[0]
    assert c0.completed_unlabeled_cycles == 0


def test_v1_and_v1u_patient_order_is_independent_of_frame_choice_counts(tmp_path):
    data = _q3d_data(tmp_path / "fixture", n_unlabeled=4)
    v1 = Q3DJointSampler(data, seed=97)
    v1u = Q3DJointSampler(data, seed=97)

    # Mimic different per-patient cine/frame-choice counts while retaining only
    # valid frame references. Those choices must not perturb the permutation RNG.
    for index, patient in enumerate(data.unlabeled_patients):
        first_key = v1u._unlabeled_by_patient[patient][0]
        v1u._unlabeled_by_patient[patient] = [first_key] * (index + 1)

    n_updates = 5 * len(data.unlabeled_patients) + 2
    v1_records = [v1.sample_update() for _ in range(n_updates)]
    v1u_records = [v1u.sample_update() for _ in range(n_updates)]
    assert [record["unlabeled_patient"] for record in v1_records] == [
        record["unlabeled_patient"] for record in v1u_records
    ]


@pytest.mark.parametrize("saved_after_cycles", [0, 1])
def test_midcycle_and_boundary_resume_are_exact(tmp_path, saved_after_cycles):
    data = _q3d_data(tmp_path / f"fixture-{saved_after_cycles}", n_unlabeled=3)
    seed = 121
    sampler = Q3DJointSampler(data, seed=seed)
    saved_after = 1 if saved_after_cycles == 0 else len(data.unlabeled_patients)
    for _ in range(saved_after):
        sampler.sample_update()
    state = sampler.state_dict()
    expected = [sampler.sample_update() for _ in range(20)]

    resumed = Q3DJointSampler(data, seed=seed)
    resumed.load_state_dict(state)
    actual = [resumed.sample_update() for _ in range(20)]
    assert actual == expected
    if saved_after_cycles == 0:
        assert state["unlabeled_cycle_count"] == 0
        assert state["unlabeled_cursor"] == 1
    else:
        assert state["unlabeled_cycle_count"] == 1
        assert state["unlabeled_cursor"] == 0


def test_sampler_rejects_tampered_permutation_cursor_and_v2_state(tmp_path):
    data = _q3d_data(tmp_path / "fixture", n_unlabeled=3)
    sampler = Q3DJointSampler(data, seed=55)
    sampler.sample_update()
    good_state = sampler.state_dict()
    target = Q3DJointSampler(data, seed=55)

    duplicate_patient = copy.deepcopy(good_state)
    duplicate_patient["unlabeled_permutation"][-1] = duplicate_patient["unlabeled_permutation"][0]
    with pytest.raises(ValueError, match="full patient permutation"):
        target.load_state_dict(duplicate_patient)

    outside_cursor = copy.deepcopy(good_state)
    outside_cursor["unlabeled_cursor"] = len(outside_cursor["unlabeled_permutation"])
    with pytest.raises(ValueError, match="cursor is outside"):
        target.load_state_dict(outside_cursor)

    inconsistent_cursor = copy.deepcopy(good_state)
    inconsistent_cursor["unlabeled_cursor"] += 1
    with pytest.raises(ValueError, match="disagree with update count"):
        target.load_state_dict(inconsistent_cursor)

    legacy_state = copy.deepcopy(good_state)
    legacy_state["schema"] = "self_audit.candidate_q3d.joint_sampler.v2"
    with pytest.raises(ValueError, match="v2 state cannot resume"):
        target.load_state_dict(legacy_state)


def test_sampler_rng_state_validation_is_fail_closed(tmp_path):
    data = _q3d_data(tmp_path / "fixture", n_unlabeled=2)
    sampler = Q3DJointSampler(data, seed=8)
    state = sampler.state_dict()
    state["unlabeled_frame_rng_state"] = ("invalid",)
    target = Q3DJointSampler(data, seed=8)
    before = target.state_dict()
    with pytest.raises(ValueError, match="RNG state is invalid"):
        target.load_state_dict(state)
    assert target.state_dict() == before
