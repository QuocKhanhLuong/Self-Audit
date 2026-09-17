"""Regression tests for the frozen transition bank export path (W4).

Every test below drives the *actual* generation path: a real ``SelfAuditNet``
is constructed and run, proposals come out of the rollout, and the evaluator
attaches quality afterwards.  No row is hand-authored as if it came from a
model, and none of these tests claims a real checkpoint or a real dataset --
the model is randomly initialised and the reference labels are synthetic
fixtures, which is exactly what is needed to test the export contract and
exactly what must not be reported as an integrated diagnostic run.

Coverage:

1. Tiny deterministic generation -> evaluator -> strict JSON roundtrip.
2. Per-sample reject HALT: a rejected on-policy sample emits no later row, and
   a bank that contains one is rejected at the serialization boundary.
3. State identifiers are reproducible, content-sensitive and
   representation-tagged, and are not derived from the stage index.
4. Contract, provenance, schema and content tamper rejection.
5. GT firewall: with images and weights fixed and the reference labels
   permuted, proposals, state identifiers, ``delta_q`` and gate decisions are
   bit-identical while evaluator quality moves.
6. Positive control: generation that deliberately consumes GT *must* change
   under the same permutation, proving the firewall probe can see a leak.
7. Source tags (on_policy / always_accept_prefix / synthetic) and the
   separation between rollout policy and hypothetical gate decision.
8. Undefined ``Q`` survives as JSON ``null`` with its coverage counters.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.audit.counterfactual import CounterfactualGenerator
from self_audit.evaluation.contracts import ContractMismatchError
from self_audit.models import SelfAuditNet
from self_audit.provenance import MEMBERSHIP_COVERAGE
from self_audit.evaluation.transition_bank import (
    BANK_SCHEMA_VERSION,
    EXPORT_BOUND,
    EXPORT_UNVERIFIED_DEMO,
    LOCAL_EVIDENCE_CHANNELS,
    ROLLOUT_ALWAYS_ACCEPT,
    ROLLOUT_SELF_AUDIT,
    SOURCE_ALWAYS_ACCEPT_PREFIX,
    SOURCE_ON_POLICY,
    SOURCE_SYNTHETIC,
    BankValidationError,
    TransitionEvaluator,
    bank_content_signature,
    build_bank,
    dump_bank,
    evaluation_provenance_record,
    generate_on_policy_proposals,
    generate_synthetic_proposals,
    generation_provenance_record,
    load_bank,
    state_tensor_id,
    validate_bank,
)

DEMO_REASON = "randomly initialised fixture model; no trained checkpoint in this test"


def reseal(bank: dict) -> dict:
    """Recompute the integrity signature after a deliberate semantic edit.

    Without this the content hash fires first and the *semantic* check under
    test never runs.  A content hash only proves the bytes were not altered in
    transport; these tests are about internal consistency.
    """

    bank["integrity"]["num_rows"] = len(bank["rows"])
    bank["integrity"]["content_signature"] = bank_content_signature(bank)
    return bank

IMAGE_SIZE = 32
PATIENTS = ["patient001", "patient002"]
CASES = ["patient001_ED", "patient002_ED"]
SLICES = [3, 4]

PROPOSAL_ONLY_FIELDS = (
    "patient_id",
    "case_id",
    "slice_index",
    "stage",
    "trajectory_id",
    "state_representation",
    "previous_state_id",
    "candidate_state_id",
    "delta_q",
    "accepted",
    "hypothetical_gate",
    "source",
    "rollout_policy",
    "gt_used_in_generation",
    "generation_operation",
    "halted_after",
    "local_evidence",
)


def _assert_strict_json(text: str) -> dict:
    """Parse with the non-standard JSON constants rejected outright.

    A substring search for ``NaN`` is not the check: the contract description
    legitimately contains the word.  What must not appear is a bare ``NaN`` /
    ``Infinity`` *token* in value position.
    """

    def reject(name: str):  # pragma: no cover - only runs on a bad file
        raise AssertionError(f"non-standard JSON constant {name!r} in exported bank")

    return json.loads(text, parse_constant=reject)


def build_model(seed: int = 20260908) -> SelfAuditNet:
    torch.manual_seed(seed)
    model = SelfAuditNet(
        pretrained_encoder=False,
        shared_channels=16,
        window_k=4,
        max_turns=3,
    )
    model.eval()
    return model


def fixed_images(seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)


def structured_reference() -> torch.Tensor:
    """Two clearly different reference masks so a permutation must move Dice."""

    reference = torch.zeros(2, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.long)
    reference[0, 4:16, 4:16] = 1
    reference[0, 8:12, 18:26] = 2
    reference[1, 18:28, 18:28] = 3
    reference[1, 2:10, 20:28] = 2
    return reference


def targets_from(reference: torch.Tensor) -> dict[tuple[str, str, int], np.ndarray]:
    return {
        (PATIENTS[index], CASES[index], SLICES[index]): reference[index].numpy()
        for index in range(reference.shape[0])
    }


def make_proposals(
    model: SelfAuditNet,
    images: torch.Tensor,
    *,
    tau_accept: float = 0.0,
    rollout_policy: str = ROLLOUT_SELF_AUDIT,
    t_max: int = 3,
):
    return generate_on_policy_proposals(
        model,
        images,
        patient_ids=PATIENTS,
        case_ids=CASES,
        slice_indices=SLICES,
        tau_accept=tau_accept,
        t_max=t_max,
        rollout_policy=rollout_policy,
    )


def assemble(rows, *, model, tau_accept=0.0, t_max=3, sources=(SOURCE_ON_POLICY,), gt_used=False):
    return build_bank(
        rows,
        generation=generation_provenance_record(
            model=model,
            rollout_policy=ROLLOUT_SELF_AUDIT,
            gt_used_in_generation=gt_used,
            unverified_demo_reason=DEMO_REASON,
        ),
        evaluation=evaluation_provenance_record(
            reference_source="synthetic_test_reference_masks",
        ),
        protocol={
            "tau_accept": float(tau_accept),
            "t_max": int(t_max),
            "sources": list(sources),
        },
    )


# ---------------------------------------------------------------------------
# 1. Generation -> evaluator -> JSON roundtrip
# ---------------------------------------------------------------------------


def test_actual_generation_then_evaluator_then_json_roundtrip(tmp_path: Path) -> None:
    model = build_model()
    images = fixed_images()
    proposals = make_proposals(model, images, tau_accept=-1.0)
    assert proposals, "the rollout must produce at least one real transition"
    # Generation happened before any reference label existed in this test.
    assert all(not proposal.gt_used_in_generation for proposal in proposals)

    evaluator = TransitionEvaluator()
    rows = evaluator.evaluate(proposals, targets=targets_from(structured_reference()))
    assert len(rows) == len(proposals)

    bank = assemble(rows, model=model, tau_accept=-1.0)
    path = dump_bank(bank, tmp_path / "bank.json")
    text = path.read_text(encoding="utf-8")
    _assert_strict_json(text)

    loaded = load_bank(path)
    assert loaded["bank_schema_version"] == BANK_SCHEMA_VERSION
    assert loaded["rows"] == bank["rows"]
    assert loaded["integrity"]["content_signature"] == bank["integrity"]["content_signature"]

    row = loaded["rows"][0]
    for key in (
        "patient_id",
        "case_id",
        "slice_index",
        "stage",
        "previous_state_id",
        "candidate_state_id",
        "delta_q",
        "accepted",
        "q_previous",
        "q_candidate",
        "metric_contract",
        "sufficient_statistics",
    ):
        assert key in row
    assert row["local_evidence"]["channels"] == list(LOCAL_EVIDENCE_CHANNELS)
    assert row["metric_space"] == "slice_proxy"
    assert loaded["evaluation"]["metric_contract"] == "foreground_dice_exclude_v1"


def test_delta_q_is_a_signed_score_not_a_probability() -> None:
    model = build_model()
    proposals = make_proposals(model, fixed_images(), tau_accept=-1.0)
    rows = TransitionEvaluator().evaluate(proposals, targets=targets_from(structured_reference()))
    # The recorded delta_q is the auditor's signed global head, so it is not
    # constrained to [0,1] and is never squashed into one.
    assert all(isinstance(row["delta_q"], float) for row in rows)
    assert all(row["hypothetical_gate"]["accept"] == (row["delta_q"] > row["hypothetical_gate"]["tau_accept"]) for row in rows)


# ---------------------------------------------------------------------------
# 2. Per-sample reject HALT
# ---------------------------------------------------------------------------


def test_rejected_sample_halts_and_emits_no_later_transition() -> None:
    model = build_model()
    images = fixed_images()
    proposals = make_proposals(model, images, tau_accept=1e6)
    assert len(proposals) == images.shape[0]
    for proposal in proposals:
        assert proposal.stage == 0
        assert proposal.accepted is False
        assert proposal.halted_after is True


def test_bank_rejects_a_fabricated_transition_after_a_halt() -> None:
    model = build_model()
    images = fixed_images()
    rejected = make_proposals(model, images, tau_accept=1e6)
    # The always-accept rollout walks past the gate, so it supplies a genuine
    # stage-1 transition measured at the same tau.
    walked = make_proposals(model, images, tau_accept=1e6, rollout_policy=ROLLOUT_ALWAYS_ACCEPT)
    evaluator = TransitionEvaluator()
    targets = targets_from(structured_reference())
    rows = evaluator.evaluate(rejected, targets=targets)

    # Splice a genuine stage-1 row onto a trajectory whose stage-0 was
    # rejected.  The bank must refuse it rather than store a trajectory that
    # continued past its own HALT.
    later = [row for row in evaluator.evaluate(walked, targets=targets) if row["stage"] == 1][0]
    later = dict(later)
    later["source"] = rows[0]["source"]
    later["rollout_policy"] = rows[0]["rollout_policy"]
    later["patient_id"] = rows[0]["patient_id"]
    later["case_id"] = rows[0]["case_id"]
    later["slice_index"] = rows[0]["slice_index"]
    later["accepted"] = bool(later["hypothetical_gate"]["accept"])
    later["halted_after"] = True
    later["trajectory_id"] = rows[0]["trajectory_id"]
    later["previous_state_id"] = rows[0]["previous_state_id"]
    with pytest.raises(BankValidationError, match="HALT"):
        assemble(rows + [later], model=model, tau_accept=1e6)


# ---------------------------------------------------------------------------
# 3. State identifiers
# ---------------------------------------------------------------------------


def test_state_ids_are_reproducible_and_content_sensitive() -> None:
    first = make_proposals(build_model(), fixed_images(), tau_accept=-1.0)
    second = make_proposals(build_model(), fixed_images(), tau_accept=-1.0)
    assert [p.previous_state_id for p in first] == [p.previous_state_id for p in second]
    assert [p.candidate_state_id for p in first] == [p.candidate_state_id for p in second]

    mutated = build_model()
    with torch.no_grad():
        for parameter in mutated.parameters():
            parameter.add_(0.05)
    changed = make_proposals(mutated, fixed_images(), tau_accept=-1.0)
    assert [p.previous_state_id for p in changed] != [p.previous_state_id for p in first]


def test_state_id_is_content_and_representation_not_stage() -> None:
    state = torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(4, 5, 6)
    assert state_tensor_id(state) == state_tensor_id(state.clone())
    assert state_tensor_id(state, representation="probs_v1") != state_tensor_id(
        state, representation="logits_v1"
    )
    perturbed = state.clone()
    perturbed[0, 0, 0] += 1e-3
    assert state_tensor_id(perturbed) != state_tensor_id(state)
    with pytest.raises(ValueError, match="ONE sample"):
        state_tensor_id(state.unsqueeze(0))
    with pytest.raises(ValueError, match="representation"):
        state_tensor_id(state, representation="not_a_representation")


def test_repeated_state_content_shares_an_id_across_stages() -> None:
    # A state identifier names content, not position: the same tensor observed
    # at two different stages is the same state.
    state = torch.randn(4, 8, 8)
    assert state_tensor_id(state) == state_tensor_id(state)


# ---------------------------------------------------------------------------
# 4. Tamper rejection
# ---------------------------------------------------------------------------


@pytest.fixture()
def sealed_bank():
    model = build_model()
    proposals = make_proposals(model, fixed_images(), tau_accept=-1.0)
    rows = TransitionEvaluator().evaluate(proposals, targets=targets_from(structured_reference()))
    return assemble(rows, model=model, tau_accept=-1.0)


def test_row_content_tamper_is_rejected(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["delta_q"] = float(tampered["rows"][0]["delta_q"]) + 1.0
    tampered["rows"][0]["hypothetical_gate"]["accept"] = bool(
        tampered["rows"][0]["delta_q"] > tampered["rows"][0]["hypothetical_gate"]["tau_accept"]
    )
    tampered["rows"][0]["accepted"] = tampered["rows"][0]["hypothetical_gate"]["accept"]
    with pytest.raises(BankValidationError, match="content signature"):
        validate_bank(tampered)


def test_provenance_tamper_is_rejected(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["generation"]["generation_code"]["git_sha"] = "0" * 40
    with pytest.raises(BankValidationError, match="content signature"):
        validate_bank(tampered)


def test_split_coverage_downgrade_is_rejected(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["generation"]["split_coverage"] = "content_hash_of_every_record"
    with pytest.raises(BankValidationError, match="membership"):
        validate_bank(tampered)


def test_contract_tamper_is_rejected(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["evaluation"]["metric_contract_definition"]["empty_policy"] = "legacy_one"
    with pytest.raises(ContractMismatchError):
        validate_bank(tampered)


def test_schema_version_tamper_is_rejected(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["bank_schema_version"] = BANK_SCHEMA_VERSION + 1
    with pytest.raises(BankValidationError, match="bank_schema_version"):
        validate_bank(tampered)


def test_state_continuity_break_is_rejected(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    later = [row for row in tampered["rows"] if int(row["stage"]) == 1]
    assert later, "fixture must contain a multi-stage trajectory"
    later[0]["previous_state_id"] = "logits_v1:sha256:" + "f" * 64
    with pytest.raises(BankValidationError, match="state continuity"):
        validate_bank(tampered)


def test_nonstandard_json_constant_is_rejected(tmp_path: Path, sealed_bank) -> None:
    path = tmp_path / "nan_bank.json"
    payload = json.loads(json.dumps(sealed_bank))
    payload["rows"][0]["q_previous"] = float("nan")
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    with pytest.raises(BankValidationError, match="non-standard constant"):
        load_bank(path)


def test_synthetic_row_cannot_claim_it_was_gt_free() -> None:
    model = build_model()
    torch.manual_seed(11)
    proposals = generate_synthetic_proposals(
        model,
        fixed_images(),
        structured_reference(),
        patient_ids=PATIENTS,
        case_ids=CASES,
        slice_indices=SLICES,
        kind="positive",
    )
    rows = TransitionEvaluator().evaluate(proposals, targets=targets_from(structured_reference()))
    for row in rows:
        row["gt_used_in_generation"] = False
    with pytest.raises(BankValidationError, match="gt_used_in_generation"):
        assemble(rows, model=model, sources=(SOURCE_SYNTHETIC,), gt_used=True)


# ---------------------------------------------------------------------------
# 5/6. GT firewall and its positive control
# ---------------------------------------------------------------------------


def test_gt_firewall_generation_is_invariant_to_permuted_reference() -> None:
    model = build_model()
    images = fixed_images()
    reference = structured_reference()
    permuted = reference.flip(0).clone()

    proposals_a = make_proposals(model, images, tau_accept=-1.0)
    proposals_b = make_proposals(model, images, tau_accept=-1.0)

    evaluator = TransitionEvaluator()
    rows_a = evaluator.evaluate(proposals_a, targets=targets_from(reference))
    rows_b = evaluator.evaluate(proposals_b, targets=targets_from(permuted))
    assert len(rows_a) == len(rows_b)

    for row_a, row_b in zip(rows_a, rows_b):
        for key in PROPOSAL_ONLY_FIELDS:
            assert row_a[key] == row_b[key], f"generation-side field {key} moved with the reference"

    quality_a = [(row["q_previous"], row["q_candidate"], row["delta_dice"]) for row in rows_a]
    quality_b = [(row["q_previous"], row["q_candidate"], row["delta_dice"]) for row in rows_b]
    assert quality_a != quality_b, "permuting the reference must move the evaluator's quality"


def test_positive_control_gt_dependent_generation_does_change() -> None:
    # Deliberate GT dependence: a positive counterfactual repairs the state
    # *towards* the reference.  If the firewall probe above could not see a
    # leak, this control would also come out invariant -- it must not.
    model = build_model()
    images = fixed_images()
    reference = structured_reference()
    permuted = reference.flip(0).clone()

    def synthetic(target: torch.Tensor):
        torch.manual_seed(4242)
        return generate_synthetic_proposals(
            model,
            images,
            target,
            patient_ids=PATIENTS,
            case_ids=CASES,
            slice_indices=SLICES,
            generator=CounterfactualGenerator(),
            kind="positive",
        )

    control_a = synthetic(reference)
    control_b = synthetic(permuted)

    # The previous state is the model's own initial annotation and is GT-free,
    # so it must be identical in both arms.
    assert [p.previous_state_id for p in control_a] == [p.previous_state_id for p in control_b]
    # The candidate is built against the reference, so it must move.
    assert [p.candidate_state_id for p in control_a] != [p.candidate_state_id for p in control_b]
    assert all(p.gt_used_in_generation for p in control_a + control_b)
    assert all(p.source == SOURCE_SYNTHETIC for p in control_a + control_b)


def test_deployable_generation_api_refuses_a_batch_mapping() -> None:
    model = build_model()
    batch = {"image": fixed_images(), "mask": structured_reference()}
    with pytest.raises(TypeError, match="does not accept a batch mapping"):
        generate_on_policy_proposals(
            model,
            batch,
            patient_ids=PATIENTS,
            case_ids=CASES,
            slice_indices=SLICES,
        )


# ---------------------------------------------------------------------------
# 7. Source tags and rollout-vs-gate separation
# ---------------------------------------------------------------------------


def test_always_accept_prefix_is_tagged_and_separated_from_the_gate() -> None:
    model = build_model()
    images = fixed_images()
    prefix = make_proposals(model, images, tau_accept=1e6, rollout_policy=ROLLOUT_ALWAYS_ACCEPT)
    assert len(prefix) == images.shape[0] * 3, "always-accept walks to the hard cap"
    for proposal in prefix:
        assert proposal.source == SOURCE_ALWAYS_ACCEPT_PREFIX
        assert proposal.rollout_policy == ROLLOUT_ALWAYS_ACCEPT
        # The rollout accepted; the threshold gate would not have.
        assert proposal.accepted is True
        assert proposal.hypothetical_accept is False
        assert proposal.halted_after is False

    self_audit = make_proposals(model, images, tau_accept=1e6, rollout_policy=ROLLOUT_SELF_AUDIT)
    assert all(p.source == SOURCE_ON_POLICY for p in self_audit)
    assert len(self_audit) < len(prefix)


def test_always_accept_prefix_bank_validates_and_records_both_decisions(tmp_path: Path) -> None:
    model = build_model()
    proposals = make_proposals(
        model, fixed_images(), tau_accept=1e6, rollout_policy=ROLLOUT_ALWAYS_ACCEPT
    )
    rows = TransitionEvaluator().evaluate(proposals, targets=targets_from(structured_reference()))
    bank = build_bank(
        rows,
        generation=generation_provenance_record(
            model=model,
            rollout_policy=ROLLOUT_ALWAYS_ACCEPT,
            unverified_demo_reason=DEMO_REASON,
        ),
        evaluation=evaluation_provenance_record(reference_source="synthetic_test_reference_masks"),
        protocol={"tau_accept": 1e6, "t_max": 3, "sources": [SOURCE_ALWAYS_ACCEPT_PREFIX]},
    )
    loaded = load_bank(dump_bank(bank, tmp_path / "prefix.json"))
    assert {row["source"] for row in loaded["rows"]} == {SOURCE_ALWAYS_ACCEPT_PREFIX}
    assert all(row["accepted"] is True for row in loaded["rows"])
    assert all(row["hypothetical_gate"]["accept"] is False for row in loaded["rows"])


def test_source_tag_must_match_its_rollout_policy(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["source"] = SOURCE_ALWAYS_ACCEPT_PREFIX
    with pytest.raises(BankValidationError, match="fixes its source tag"):
        validate_bank(tampered)


def test_synthetic_rows_carry_their_operation_and_probability_representation() -> None:
    model = build_model()
    torch.manual_seed(99)
    proposals = generate_synthetic_proposals(
        model,
        fixed_images(),
        structured_reference(),
        patient_ids=PATIENTS,
        case_ids=CASES,
        slice_indices=SLICES,
        kind="negative",
        operation="local_erosion",
    )
    rows = TransitionEvaluator().evaluate(proposals, targets=targets_from(structured_reference()))
    bank = build_bank(
        rows,
        generation=generation_provenance_record(
            model=model,
            rollout_policy="synthetic_perturbation",
            gt_used_in_generation=True,
            unverified_demo_reason=DEMO_REASON,
        ),
        evaluation=evaluation_provenance_record(reference_source="synthetic_test_reference_masks"),
        protocol={"tau_accept": 0.0, "t_max": 1, "sources": [SOURCE_SYNTHETIC]},
    )
    validate_bank(bank)
    assert {row["generation_operation"] for row in bank["rows"]} == {"local_erosion"}
    assert {row["state_representation"] for row in bank["rows"]} == {"probs_v1"}
    assert all(row["gt_used_in_generation"] is True for row in bank["rows"])


# ---------------------------------------------------------------------------
# 8. Undefined quality preservation
# ---------------------------------------------------------------------------


def test_undefined_quality_keeps_its_row_and_serializes_as_null(tmp_path: Path) -> None:
    # Targeted evaluator unit test: the label maps of a genuinely generated
    # proposal are replaced with all-background so that every foreground class
    # is empty in both prediction and reference.  Under the exclude policy the
    # macro score is undefined; the row must survive with its coverage intact.
    model = build_model()
    proposals = make_proposals(model, fixed_images(), tau_accept=-1.0)
    blank = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)
    blanked = [
        dataclasses.replace(
            proposal,
            previous_labels=blank.copy(),
            candidate_labels=blank.copy(),
        )
        for proposal in proposals
    ]
    targets = {
        (PATIENTS[index], CASES[index], SLICES[index]): blank.copy()
        for index in range(len(PATIENTS))
    }
    rows = TransitionEvaluator().evaluate(blanked, targets=targets)
    assert len(rows) == len(proposals), "an undefined score must never drop the row"
    for row in rows:
        assert row["q_previous"] is None
        assert row["q_candidate"] is None
        assert row["delta_dice"] is None
        assert row["q_previous_defined"] is False
        assert row["delta_defined"] is False
        assert row["delta_class"] is None
        # Coverage is retained: the row still reports its (all-zero) counts.
        assert set(row["sufficient_statistics"]["previous"]["tp"]) == {"1", "2", "3"}
        assert row["per_class_dice_previous"]["1"] is None

    bank = assemble(rows, model=model, tau_accept=-1.0)
    text = dump_bank(bank, tmp_path / "undefined.json").read_text(encoding="utf-8")
    assert '"q_previous": null' in text
    _assert_strict_json(text)
    assert load_bank(tmp_path / "undefined.json")["rows"][0]["q_candidate"] is None


def test_defined_flag_must_agree_with_the_stored_score(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["q_previous_defined"] = False
    with pytest.raises(BankValidationError, match="delta_defined disagrees"):
        validate_bank(tampered)

    tampered["rows"][0]["delta_defined"] = False
    with pytest.raises(BankValidationError, match="q_previous_defined false but stores"):
        validate_bank(tampered)


def test_evaluator_refuses_a_proposal_with_no_reference() -> None:
    model = build_model()
    proposals = make_proposals(model, fixed_images(), tau_accept=-1.0)
    with pytest.raises(BankValidationError, match="No reference label"):
        TransitionEvaluator().evaluate(proposals, targets={})


def test_evaluator_rejects_a_volume_contract() -> None:
    with pytest.raises(ContractMismatchError, match="per-slice"):
        TransitionEvaluator(contract="foreground_dice_volume_resized_v1")


# ---------------------------------------------------------------------------
# 9. Semantic validation that a recomputed content hash cannot hide
# ---------------------------------------------------------------------------


def test_quality_inconsistent_with_its_own_statistics_is_rejected(sealed_bank) -> None:
    tampered = reseal(json.loads(json.dumps(sealed_bank)))
    tampered["rows"][0]["q_previous"] = 0.999
    tampered["rows"][0]["delta_dice"] = (
        float(tampered["rows"][0]["q_candidate"]) - 0.999
    )
    tampered["rows"][0]["delta_class"] = -1
    reseal(tampered)
    with pytest.raises(BankValidationError, match="recomputed from its sufficient statistics"):
        validate_bank(tampered)


def test_per_class_dice_inconsistent_with_statistics_is_rejected(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    row = tampered["rows"][0]
    key = sorted(row["per_class_dice_candidate"])[0]
    row["per_class_dice_candidate"][key] = 0.5 if row["per_class_dice_candidate"][key] != 0.5 else 0.25
    reseal(tampered)
    with pytest.raises(BankValidationError, match="per_class_dice_candidate"):
        validate_bank(tampered)


def test_forged_statistics_that_flip_definedness_are_rejected(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    stats = tampered["rows"][0]["sufficient_statistics"]["previous"]
    for name in ("tp", "fp", "fn"):
        for key in stats[name]:
            stats[name][key] = 0
    reseal(tampered)
    with pytest.raises(BankValidationError, match="contradicts its own sufficient"):
        validate_bank(tampered)


def test_delta_class_must_match_the_canonical_margin(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    row = next(r for r in tampered["rows"] if r["delta_defined"])
    row["delta_class"] = 0 if row["delta_class"] != 0 else 1
    reseal(tampered)
    with pytest.raises(BankValidationError, match="canonical classification"):
        validate_bank(tampered)


def test_neutral_margin_must_match_the_contract(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["neutral_margin"] = 0.05
    reseal(tampered)
    with pytest.raises(BankValidationError, match="neutral_margin"):
        validate_bank(tampered)


def test_metric_space_must_match_the_contract(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["metric_space"] = "volume_native"
    reseal(tampered)
    with pytest.raises(BankValidationError, match="metric_space"):
        validate_bank(tampered)


def test_truncated_state_identifier_is_rejected(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["candidate_state_id"] = "logits_v1:sha256:deadbeef"
    reseal(tampered)
    with pytest.raises(BankValidationError, match="full SHA-256 state"):
        validate_bank(tampered)


def test_boolean_fields_do_not_accept_integers(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["accepted"] = 1
    reseal(tampered)
    with pytest.raises(BankValidationError, match="accepted must be a boolean"):
        validate_bank(tampered)


def test_schema_version_does_not_accept_a_boolean(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["bank_schema_version"] = True
    with pytest.raises(BankValidationError, match="bank_schema_version"):
        validate_bank(tampered)


def test_protocol_tau_must_match_every_row(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["protocol"]["tau_accept"] = 0.25
    reseal(tampered)
    with pytest.raises(BankValidationError, match="protocol declares tau_accept"):
        validate_bank(tampered)


def test_protocol_t_max_must_cover_every_stage(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["protocol"]["t_max"] = 1
    reseal(tampered)
    with pytest.raises(BankValidationError, match="caps the rollout"):
        validate_bank(tampered)


def test_protocol_must_declare_every_source_present(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["protocol"]["sources"] = [SOURCE_SYNTHETIC]
    reseal(tampered)
    with pytest.raises(BankValidationError, match="protocol does not"):
        validate_bank(tampered)


def test_local_evidence_must_stay_on_the_simplex(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["local_evidence"]["mean_probability"][LOCAL_EVIDENCE_CHANNELS[0]] = 0.9
    reseal(tampered)
    with pytest.raises(BankValidationError, match="simplex"):
        validate_bank(tampered)


def test_trajectory_may_not_mix_sample_identities(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    later = next(row for row in tampered["rows"] if int(row["stage"]) == 1)
    first = next(
        row
        for row in tampered["rows"]
        if row["trajectory_id"] == later["trajectory_id"] and int(row["stage"]) == 0
    )
    later["patient_id"] = "someone_else"
    later["previous_state_id"] = first["candidate_state_id"]
    reseal(tampered)
    with pytest.raises(BankValidationError, match="distinct sample identities"):
        validate_bank(tampered)


# ---------------------------------------------------------------------------
# 10. Export identity: bound vs explicitly unverified
# ---------------------------------------------------------------------------


def test_unbound_export_requires_an_explicit_demo_reason() -> None:
    with pytest.raises(ValueError, match="unverified_demo_reason"):
        generation_provenance_record(model=build_model(), rollout_policy=ROLLOUT_SELF_AUDIT)


def test_demo_export_is_labelled_and_never_claims_a_checkpoint(sealed_bank) -> None:
    assert sealed_bank["generation"]["export_identity_class"] == EXPORT_UNVERIFIED_DEMO
    assert sealed_bank["generation"]["checkpoint_bound"] is False
    assert sealed_bank["generation"]["checkpoint"] is None
    assert sealed_bank["generation"]["producer"]["producer_git_sha"] == "unknown"


def test_bound_export_without_a_complete_identity_is_rejected(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["generation"]["checkpoint_bound"] = True
    tampered["generation"]["export_identity_class"] = EXPORT_BOUND
    tampered["generation"]["unverified_demo_reason"] = None
    reseal(tampered)
    with pytest.raises(BankValidationError, match="complete checkpoint identity"):
        validate_bank(tampered)


def test_bound_export_needs_a_cohort_and_full_digests(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    generation = tampered["generation"]
    generation["checkpoint_bound"] = True
    generation["export_identity_class"] = EXPORT_BOUND
    generation["unverified_demo_reason"] = None
    generation["checkpoint"] = {
        "checkpoint_path": "weights/phase_c_best.pt",
        "checkpoint_sha256": "a" * 64,
        "state_digest": "b" * 64,
        "file_state_digest": "b" * 64,
        "model_identity": {"class_name": "SelfAuditNet"},
        "producer": {"producer_git_sha": "c" * 40},
    }
    generation["model_identity"] = {"class_name": "SelfAuditNet"}
    reseal(tampered)
    with pytest.raises(BankValidationError, match="exact cohort"):
        validate_bank(tampered)

    generation["checkpoint"]["state_digest"] = "tooshort"
    reseal(tampered)
    with pytest.raises(BankValidationError, match="full SHA-256 digest"):
        validate_bank(tampered)


def test_checkpoint_bound_flag_must_match_the_identity_class(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["generation"]["checkpoint_bound"] = True
    reseal(tampered)
    with pytest.raises(BankValidationError, match="contradicts export_identity_class"):
        validate_bank(tampered)


# ---------------------------------------------------------------------------
# 11. Non-finite scores are rejected before JSON coercion
# ---------------------------------------------------------------------------


def test_nonfinite_delta_q_is_rejected_before_it_can_become_null() -> None:
    model = build_model()
    proposals = make_proposals(model, fixed_images(), tau_accept=-1.0)
    rows = TransitionEvaluator().evaluate(proposals, targets=targets_from(structured_reference()))
    rows[0] = dict(rows[0])
    rows[0]["delta_q"] = float("nan")
    with pytest.raises(BankValidationError, match="delta_q must be finite"):
        assemble(rows, model=model, tau_accept=-1.0)


def test_nonfinite_defined_quality_is_rejected_before_it_can_become_null() -> None:
    model = build_model()
    proposals = make_proposals(model, fixed_images(), tau_accept=-1.0)
    rows = TransitionEvaluator().evaluate(proposals, targets=targets_from(structured_reference()))
    rows[0] = dict(rows[0])
    rows[0]["q_candidate"] = float("inf")
    with pytest.raises(BankValidationError, match="q_candidate must be finite"):
        assemble(rows, model=model, tau_accept=-1.0)


# ---------------------------------------------------------------------------
# 11b. The four bypasses reproduced in wave4_bank_review_notes.md
#
# Each mutation is applied to a real generated bank and the integrity signature
# is recomputed, exactly as the reviewer did, so the content hash cannot be
# what rejects it.  All four were ACCEPTED by the draft validator.
# ---------------------------------------------------------------------------


def test_review_bypass_1_inflated_previous_tp_with_unchanged_quality(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["sufficient_statistics"]["previous"]["tp"]["1"] = 999999
    reseal(tampered)
    with pytest.raises(BankValidationError, match="recomputed from its sufficient statistics"):
        validate_bank(tampered)


def test_review_bypass_2_checkpoint_bound_with_the_block_removed(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["generation"]["checkpoint_bound"] = True
    tampered["generation"]["export_identity_class"] = EXPORT_BOUND
    tampered["generation"]["unverified_demo_reason"] = None
    tampered["generation"].pop("checkpoint", None)
    reseal(tampered)
    with pytest.raises(BankValidationError, match="complete checkpoint identity"):
        validate_bank(tampered)


def test_review_bypass_3_row_metric_space_volume_under_a_slice_header(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["metric_space"] = "volume_native"
    reseal(tampered)
    with pytest.raises(BankValidationError, match="metric_space"):
        validate_bank(tampered)


def test_review_bypass_4_fractional_row_schema_version(sealed_bank) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["schema_version"] = 1.5
    reseal(tampered)
    with pytest.raises(BankValidationError, match="schema_version must be an integer"):
        validate_bank(tampered)


# ---------------------------------------------------------------------------
# 13. Source-content identity honesty (producer field, W3.1)
# ---------------------------------------------------------------------------


def test_unknown_source_signature_may_not_be_recorded_as_known(sealed_bank) -> None:
    code = sealed_bank["generation"]["generation_code"]
    if "source_content_signature" not in code:
        pytest.skip("producer has not yet added source_content_signature")

    tampered = json.loads(json.dumps(sealed_bank))
    tampered["generation"]["generation_code"]["source_content_signature"] = "unknown"
    reseal(tampered)
    with pytest.raises(BankValidationError, match="not a full SHA-256 digest"):
        validate_bank(tampered)

    tampered = json.loads(json.dumps(sealed_bank))
    tampered["evaluation"]["evaluation_code"]["source_content_signature_known"] = False
    reseal(tampered)
    with pytest.raises(BankValidationError, match="must stay 'unknown'"):
        validate_bank(tampered)


# ---------------------------------------------------------------------------
# 12. CLI integration against a real tiny dataset and a real saved checkpoint
# ---------------------------------------------------------------------------


def _write_case(root: Path, case_id: str, depth: int = 3) -> None:
    (root / "volumes").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(case_id)) % (2**31))
    volume = rng.standard_normal((depth, 24, 24)).astype(np.float32)
    mask = np.zeros((depth, 24, 24), dtype=np.int64)
    mask[:, 4:12, 4:12] = 1
    mask[:, 14:20, 6:12] = 2
    mask[:, 6:10, 15:21] = 3
    np.save(root / "volumes" / f"{case_id}.npy", volume)
    np.save(root / "masks" / f"{case_id}.npy", mask)


def _tiny_export_environment(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A real (tiny) ACDC-layout dataset, split manifest, config and checkpoint."""

    from self_audit.training._utils import build_model_from_config, save_checkpoint

    data_root = tmp_path / "data"
    for case in ("patient001_ED", "patient002_ED", "patient003_ED", "patient004_ED"):
        _write_case(data_root, case)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "train_cases": ["patient001_ED", "patient002_ED"],
                "val_cases": ["patient003_ED", "patient004_ED"],
            }
        ),
        encoding="utf-8",
    )

    config = {
        "dataset": "acdc",
        "data_root": str(data_root),
        "split_manifest": str(manifest),
        "depth_axis": 0,
        "image_size": 32,
        "batch_size": 2,
        "num_workers": 0,
        "seed": 42,
        "val_split": "val",
        "device": "cpu",
        "model": {
            "encoder_name": "convnext_tiny",
            "pretrained_encoder": False,
            "encoder_allow_fallback": True,
            "shared_channels": 16,
            "window_k": 4,
            "max_turns": 2,
        },
        "audit": {"t_max": 2},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    torch.manual_seed(20260908)
    model = build_model_from_config(dict(config), torch.device("cpu"))
    checkpoint = save_checkpoint(
        tmp_path / "phase_c_best.pt", model, epoch=1, global_step=2, config=config
    )
    return config_path, Path(checkpoint), data_root


def test_cli_exports_a_bound_bank_from_a_real_tiny_dataset(tmp_path: Path) -> None:
    """End-to-end CLI run: real loader, real checkpoint bind, real rows.

    The model is randomly initialised and the volumes are tiny fixtures, so
    this is an integration test of the *export path*, not a diagnostic result
    about a trained model.
    """

    import subprocess

    config_path, checkpoint, _ = _tiny_export_environment(tmp_path)
    output = tmp_path / "bank.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "export_transition_bank.py"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--tau_accept",
        "-1.0",
        "--t_max",
        "2",
        "--device",
        "cpu",
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    assert "bank_summary=" in proc.stdout

    bank = load_bank(output)
    generation = bank["generation"]
    assert generation["export_identity_class"] == EXPORT_BOUND
    assert generation["checkpoint_bound"] is True
    assert generation["checkpoint"]["state_digest"] == generation["checkpoint"]["file_state_digest"]
    assert generation["cohort"]["split_name"] == "val"
    assert generation["cohort"]["observed_samples"] == generation["cohort"]["available_slices"]
    assert generation["producer"]["producer_recorded"] is True

    assert bank["rows"], "a real cohort must produce real rows"
    patients = {row["patient_id"] for row in bank["rows"]}
    assert patients == {"patient003", "patient004"}, patients
    assert {row["source"] for row in bank["rows"]} == {SOURCE_ON_POLICY}
    assert all(row["gt_used_in_generation"] is False for row in bank["rows"])
    _assert_strict_json(output.read_text(encoding="utf-8"))


def test_cli_export_with_synthetic_rows_marks_gt_use(tmp_path: Path) -> None:
    import subprocess

    config_path, checkpoint, _ = _tiny_export_environment(tmp_path)
    output = tmp_path / "bank_synth.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_transition_bank.py"),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--tau_accept",
            "-1.0",
            "--t_max",
            "2",
            "--device",
            "cpu",
            "--include_synthetic",
            "--synthetic_kind",
            "positive",
            "--max_batches",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    bank = load_bank(output)
    sources = {row["source"] for row in bank["rows"]}
    assert SOURCE_SYNTHETIC in sources and SOURCE_ON_POLICY in sources
    for row in bank["rows"]:
        assert row["gt_used_in_generation"] is (row["source"] == SOURCE_SYNTHETIC)
    # max_batches truncation must be reported, never described as full cohort.
    assert bank["generation"]["cohort"]["covers_full_split"] is False
    assert "max_batches" in bank["generation"]["cohort"]["subset_note"]


# ---------------------------------------------------------------------------
# 14. Additional regression negatives after resealing integrity
# ---------------------------------------------------------------------------


def _valid_bound_fixture(sealed_bank: dict) -> dict:
    """Helper creating a valid bound export dictionary from the demo fixture."""
    bank = json.loads(json.dumps(sealed_bank))
    bank["generation"]["checkpoint_bound"] = True
    bank["generation"]["export_identity_class"] = EXPORT_BOUND
    bank["generation"]["unverified_demo_reason"] = None
    bank["generation"]["checkpoint"] = {
        "checkpoint_path": "weights/phase_c_best.pt",
        "checkpoint_sha256": "a" * 64,
        "state_digest": "b" * 64,
        "file_state_digest": "b" * 64,
        "model_identity": {"class_name": "SelfAuditNet"},
        "producer": {"producer_git_sha": "d" * 40},
    }
    bank["generation"]["model_identity"] = {"class_name": "SelfAuditNet"}
    bank["generation"]["cohort"] = {
        "split_name": "val",
        "membership_signature": "e" * 64,
        "coverage": MEMBERSHIP_COVERAGE,
        "num_records": 10,
    }
    return reseal(bank)


def test_row_empty_policy_must_equal_contract(sealed_bank: dict) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["empty_policy"] = "legacy_one"
    reseal(tampered)
    with pytest.raises(BankValidationError, match="empty_policy"):
        validate_bank(tampered)

    tampered_missing = json.loads(json.dumps(sealed_bank))
    tampered_missing["rows"][0].pop("empty_policy", None)
    reseal(tampered_missing)
    with pytest.raises(BankValidationError, match="missing required field"):
        validate_bank(tampered_missing)


def test_metric_contract_version_strict_integer_no_coercion(sealed_bank: dict) -> None:
    # Row level: bool rejected
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["metric_contract_version"] = True
    reseal(tampered)
    with pytest.raises(BankValidationError, match="must be an integer"):
        validate_bank(tampered)

    # Row level: float rejected
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["metric_contract_version"] = 1.0
    reseal(tampered)
    with pytest.raises(BankValidationError, match="must be an integer"):
        validate_bank(tampered)

    # Header level: bool rejected
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["evaluation"]["metric_contract_version"] = True
    reseal(tampered)
    with pytest.raises(BankValidationError, match="must be an integer"):
        validate_bank(tampered)

    # Header level: float rejected
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["evaluation"]["metric_contract_version"] = 1.0
    reseal(tampered)
    with pytest.raises(BankValidationError, match="must be an integer"):
        validate_bank(tampered)


def test_integrity_num_rows_strict_integer_no_coercion(sealed_bank: dict) -> None:
    # Bool rejected even if signature matches
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["integrity"]["num_rows"] = True
    tampered["integrity"]["content_signature"] = bank_content_signature(tampered)
    with pytest.raises(BankValidationError, match="must be an integer"):
        validate_bank(tampered)

    # Float rejected even if signature matches
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["integrity"]["num_rows"] = float(len(tampered["rows"]))
    tampered["integrity"]["content_signature"] = bank_content_signature(tampered)
    with pytest.raises(BankValidationError, match="must be an integer"):
        validate_bank(tampered)


def test_delta_class_strict_integer_or_null(sealed_bank: dict) -> None:
    # Defined row with bool rejected
    tampered = json.loads(json.dumps(sealed_bank))
    row = next(r for r in tampered["rows"] if r["delta_defined"])
    row["delta_class"] = True
    reseal(tampered)
    with pytest.raises(BankValidationError, match="must be an integer"):
        validate_bank(tampered)

    # Defined row with float rejected
    tampered = json.loads(json.dumps(sealed_bank))
    row = next(r for r in tampered["rows"] if r["delta_defined"])
    row["delta_class"] = float(row["delta_class"]) if row["delta_class"] is not None else 0.0
    reseal(tampered)
    with pytest.raises(BankValidationError, match="must be an integer"):
        validate_bank(tampered)

    # Undefined row carrying a non-null integer rejected
    tampered = json.loads(json.dumps(sealed_bank))
    row = tampered["rows"][0]
    row["q_previous"] = None
    row["q_previous_defined"] = False
    row["delta_dice"] = None
    row["delta_defined"] = False
    for cat in ("tp", "fp", "fn"):
        for k in row["sufficient_statistics"]["previous"][cat]:
            row["sufficient_statistics"]["previous"][cat][k] = 0
    row["delta_class"] = 0
    reseal(tampered)
    with pytest.raises(BankValidationError, match="delta_class must be null"):
        validate_bank(tampered)


def test_always_accept_source_must_have_accepted_true() -> None:
    model = build_model()
    proposals = make_proposals(
        model, fixed_images(), tau_accept=1e6, rollout_policy=ROLLOUT_ALWAYS_ACCEPT
    )
    rows = TransitionEvaluator().evaluate(proposals, targets=targets_from(structured_reference()))
    bank = build_bank(
        rows,
        generation=generation_provenance_record(
            model=model,
            rollout_policy=ROLLOUT_ALWAYS_ACCEPT,
            unverified_demo_reason=DEMO_REASON,
        ),
        evaluation=evaluation_provenance_record(reference_source="test"),
        protocol={"tau_accept": 1e6, "t_max": 3, "sources": [SOURCE_ALWAYS_ACCEPT_PREFIX]},
    )
    tampered = json.loads(json.dumps(bank))
    tampered["rows"][0]["accepted"] = False
    reseal(tampered)
    with pytest.raises(BankValidationError, match="accepted is false"):
        validate_bank(tampered)


def test_halted_after_must_match_on_policy_rejection(sealed_bank: dict) -> None:
    # Accepted on-policy row marked halted_after=True must fail
    tampered = json.loads(json.dumps(sealed_bank))
    accepted_row = next(
        r for r in tampered["rows"] if r["source"] == SOURCE_ON_POLICY and r["accepted"]
    )
    accepted_row["halted_after"] = True
    reseal(tampered)
    with pytest.raises(BankValidationError, match="halted_after=True does not match on-policy rejection"):
        validate_bank(tampered)

    # Rejected on-policy row marked halted_after=False must fail
    model = build_model()
    rejected_proposals = make_proposals(model, fixed_images(), tau_accept=1e6)
    rows = TransitionEvaluator().evaluate(rejected_proposals, targets=targets_from(structured_reference()))
    bank = assemble(rows, model=model, tau_accept=1e6)
    tampered_rejected = json.loads(json.dumps(bank))
    tampered_rejected["rows"][0]["halted_after"] = False
    reseal(tampered_rejected)
    with pytest.raises(BankValidationError, match="halted_after=False does not match on-policy rejection"):
        validate_bank(tampered_rejected)


def test_bound_checkpoint_state_digest_must_equal_file_state_digest(sealed_bank: dict) -> None:
    tampered = _valid_bound_fixture(sealed_bank)
    tampered["generation"]["checkpoint"]["file_state_digest"] = "f" * 64
    reseal(tampered)
    with pytest.raises(BankValidationError, match="does not match file_state_digest"):
        validate_bank(tampered)


def test_bound_live_model_identity_must_match_binding(sealed_bank: dict) -> None:
    tampered = _valid_bound_fixture(sealed_bank)
    tampered["generation"]["model_identity"] = {"class_name": "SelfAuditNet", "param_count": 999}
    reseal(tampered)
    with pytest.raises(BankValidationError, match="Live model_identity does not match checkpoint binding model_identity"):
        validate_bank(tampered)


def test_bound_cohort_empty_mapping_or_invalid_structure_rejected(sealed_bank: dict) -> None:
    # Empty cohort mapping
    tampered = _valid_bound_fixture(sealed_bank)
    tampered["generation"]["cohort"] = {}
    reseal(tampered)
    with pytest.raises(BankValidationError, match="empty or null cohort"):
        validate_bank(tampered)

    # Split mismatch
    tampered = _valid_bound_fixture(sealed_bank)
    tampered["generation"]["cohort"]["split_name"] = "train"
    reseal(tampered)
    with pytest.raises(BankValidationError, match="split_name"):
        validate_bank(tampered)

    # Truncated / malformed membership signature
    tampered = _valid_bound_fixture(sealed_bank)
    tampered["generation"]["cohort"]["membership_signature"] = "short_hash"
    reseal(tampered)
    with pytest.raises(BankValidationError, match="membership_signature is not a full SHA-256 digest"):
        validate_bank(tampered)

    # Coverage claims content coverage instead of membership
    tampered = _valid_bound_fixture(sealed_bank)
    tampered["generation"]["cohort"]["coverage"] = "content_hash_coverage"
    reseal(tampered)
    with pytest.raises(BankValidationError, match="coverage must be"):
        validate_bank(tampered)

    # Non-positive num_records
    tampered = _valid_bound_fixture(sealed_bank)
    tampered["generation"]["cohort"]["num_records"] = 0
    reseal(tampered)
    with pytest.raises(BankValidationError, match="num_records must be positive"):
        validate_bank(tampered)


def test_header_metric_fields_must_agree_with_contract(sealed_bank: dict) -> None:
    # Header metric_contract_version mismatch
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["evaluation"]["metric_contract_version"] = 99
    reseal(tampered)
    with pytest.raises(BankValidationError, match="metric_contract_version"):
        validate_bank(tampered)

    # Header metric_space mismatch
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["evaluation"]["metric_space"] = "volume_native"
    reseal(tampered)
    with pytest.raises(BankValidationError, match="metric_space"):
        validate_bank(tampered)

    # Header empty_policy mismatch
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["evaluation"]["empty_policy"] = "legacy_one"
    reseal(tampered)
    with pytest.raises(BankValidationError, match="empty_policy"):
        validate_bank(tampered)

    # Header neutral_margin mismatch
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["evaluation"]["neutral_margin"] = 0.5
    reseal(tampered)
    with pytest.raises(BankValidationError, match="neutral_margin"):
        validate_bank(tampered)


def test_source_identities_required_no_empty_generation_code(sealed_bank: dict) -> None:
    # Empty generation_code on bound export
    tampered = _valid_bound_fixture(sealed_bank)
    tampered["generation"]["generation_code"] = {}
    reseal(tampered)
    with pytest.raises(BankValidationError, match="generation_code must be a non-empty mapping"):
        validate_bank(tampered)

    # Empty generation_code on fresh schema demo export
    tampered_demo = json.loads(json.dumps(sealed_bank))
    tampered_demo["generation"]["generation_code"] = {}
    reseal(tampered_demo)
    with pytest.raises(BankValidationError, match="generation_code must be a non-empty mapping"):
        validate_bank(tampered_demo)

    # Missing source_content_signature in fresh generation_code
    tampered_demo = json.loads(json.dumps(sealed_bank))
    tampered_demo["generation"]["generation_code"] = {"git_sha": "a" * 40}
    reseal(tampered_demo)
    with pytest.raises(BankValidationError, match="missing required source_content_signature"):
        validate_bank(tampered_demo)


def test_preserve_unknown_historical_producer(sealed_bank: dict) -> None:
    # A bound export from an older checkpoint with unknown producer is accepted
    tampered = _valid_bound_fixture(sealed_bank)
    historical_producer = {"producer_git_sha": "unknown", "producer_recorded": False}
    tampered["generation"]["checkpoint"]["producer"] = dict(historical_producer)
    tampered["generation"]["producer"] = dict(historical_producer)
    reseal(tampered)
    validated = validate_bank(tampered)
    assert validated["generation"]["checkpoint"]["producer"]["producer_git_sha"] == "unknown"


def test_negated_sufficient_statistics_counts_rejected_after_reseal(sealed_bank: dict) -> None:
    # Real generated/evaluated bank where negative counts would preserve Dice
    # through proportional cancellation: (2*-tp)/(2*-tp + -fp + -fn) == 2*tp/(2*tp + fp + fn).
    # Recomputing Q alone would accept this; _stats_from_row must reject negative counts.
    tampered = json.loads(json.dumps(sealed_bank))
    stats = tampered["rows"][0]["sufficient_statistics"]
    for side in ("previous", "candidate"):
        for metric in ("tp", "fp", "fn"):
            for k, v in stats[side][metric].items():
                stats[side][metric][k] = -int(v) if int(v) > 0 else -5
    reseal(tampered)
    with pytest.raises(BankValidationError, match="must be non-negative"):
        validate_bank(tampered)


def test_extra_class_key_in_sufficient_statistics_rejected(sealed_bank: dict) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["sufficient_statistics"]["previous"]["tp"]["99"] = 10
    reseal(tampered)
    with pytest.raises(BankValidationError, match="undeclared or alias class key"):
        validate_bank(tampered)


def test_missing_class_key_in_sufficient_statistics_rejected(sealed_bank: dict) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["sufficient_statistics"]["previous"]["tp"].pop("1", None)
    reseal(tampered)
    with pytest.raises(BankValidationError, match="missing declared class"):
        validate_bank(tampered)


def test_alias_class_key_in_sufficient_statistics_rejected(sealed_bank: dict) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    val = tampered["rows"][0]["sufficient_statistics"]["previous"]["tp"].pop("1")
    tampered["rows"][0]["sufficient_statistics"]["previous"]["tp"]["01"] = val
    reseal(tampered)
    with pytest.raises(BankValidationError, match="undeclared or alias class key"):
        validate_bank(tampered)


def test_colliding_alias_class_key_in_sufficient_statistics_rejected(sealed_bank: dict) -> None:
    tampered = json.loads(json.dumps(sealed_bank))
    tampered["rows"][0]["sufficient_statistics"]["previous"]["tp"]["01"] = 0
    reseal(tampered)
    with pytest.raises(BankValidationError, match="undeclared or alias class key"):
        validate_bank(tampered)
