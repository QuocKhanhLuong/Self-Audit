"""Candidate C: coordinate override, exact replay, restitution solver, gating.

These tests exercise the C1/C2/C3 contract of
``reports/candidate_c/architecture_contract.md`` on tiny synthetic tensors.
They demonstrate software behaviour only; no effectiveness, Dice or novelty
conclusion follows from any assertion here.
"""

from __future__ import annotations

import math

import pytest
import torch

from self_audit.models import (
    AnnotationExpert,
    CandidateCConfig,
    DynamicWindowAttention,
    SelfAuditNet,
    StaleReplayRecordError,
    normalized_pixel_step,
    resolve_candidate_c_config,
    validate_coordinate_override,
)
from self_audit.models.annotation_expert import (
    RECORD_KIND_ORDINARY,
    AnnotationExpertOutput,
    ExpertReplayRecord,
)
from self_audit.models.self_audit_net import MAX_CANDIDATE_CHECKS
from self_audit.models.self_audit_net import (
    LOCAL_FIX,
    LOCAL_REGRESS,
    LOCAL_UNCHANGED,
    _AcceptedOrdinaryRecord,
    _halve_towards_factual,
    _project_supports,
    _run_direct_rollback,
    _run_restitution_solver,
    _winning_margin,
)


_NET_CACHE: dict[str, SelfAuditNet] = {}


def _build_net(mode: str, **kwargs) -> SelfAuditNet:
    torch.manual_seed(7)
    return SelfAuditNet(
        encoder_name="convnext_tiny",
        pretrained_encoder=False,
        shared_channels=32,
        num_classes=4,
        window_k=4,
        max_turns=3,
        window_mode=mode,
        **kwargs,
    ).eval()


def _net(mode: str = "candidate_c", *, fresh: bool = False, **kwargs) -> SelfAuditNet:
    """A tiny net for this mode.

    Read-only tests share one cached instance (building ConvNeXt dominates the
    runtime); any test that mutates parameters, gradients or the replay
    generation must ask for ``fresh=True``.
    """

    if fresh or kwargs:
        return _build_net(mode, **kwargs)
    if mode not in _NET_CACHE:
        _NET_CACHE[mode] = _build_net(mode)
    return _NET_CACHE[mode]


def _images(batch: int = 2) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(batch, 3, 32, 32)


def _evidence(shape: tuple[int, int, int], *, regress: float, fix: float) -> torch.Tensor:
    """Synthetic local-audit probabilities [B,3,H,W] on the FIX/UNCHANGED/REGRESS simplex."""

    batch, height, width = shape
    evidence = torch.zeros(batch, 3, height, width)
    half = max(width // 2, 1)
    evidence[:, LOCAL_REGRESS, :, :half] = regress
    evidence[:, LOCAL_FIX, :, half:] = fix
    evidence[:, LOCAL_UNCHANGED] = 1.0 - evidence[:, LOCAL_FIX] - evidence[:, LOCAL_REGRESS]
    return evidence


def _record(net: SelfAuditNet, images: torch.Tensor, *, turn: int, regress: float, fix: float):
    """Build one accepted-ordinary record exactly as ``infer`` would."""

    expert = net.annotation_expert
    with torch.no_grad():
        shared = net.encode(images)["shared"]
        initial = net.initial_head(shared, output_size=images.shape[-2:])
        output = expert(
            shared,
            initial,
            previous_audit_evidence=None,
            turn_index=turn,
            iteration_index=turn,
            capture_geometry=True,
        )
    expert_record = expert.build_replay_record(
        shared_features=shared,
        annotation_logits=initial,
        previous_audit_evidence=None,
        turn_index=turn,
        iteration_index=turn,
        output=output,
    )
    batch, _, height, width = output.candidate_logits.shape
    record = _AcceptedOrdinaryRecord(
        indices=torch.arange(batch),
        expert_record=expert_record,
        audit_evidence=_evidence((batch, height, width), regress=regress, fix=fix),
        turn=turn,
        accepted=torch.ones(batch, dtype=torch.bool),
    )
    # An accepted transition means the retained state *is* the factual output.
    return record, expert_record.factual_candidate_logits.clone(), output


# ---------------------------------------------------------------------------
# Coordinate override (dynamic_window)
# ---------------------------------------------------------------------------


def test_coordinate_override_reproduces_generated_support_exactly() -> None:
    torch.manual_seed(3)
    block = DynamicWindowAttention(16, heads=4, k=4)
    features = torch.randn(2, 16, 6, 5)
    reference, metadata = block(features, return_metadata=True, turn_index=1, iteration_index=2)
    replayed, replay_metadata = block(
        features,
        return_metadata=True,
        turn_index=1,
        iteration_index=2,
        coordinate_override=metadata["coordinates"],
    )
    # Q/K/V/softmax/output are untouched, so supplying the realized support
    # reproduces the original output bit-for-bit.
    assert torch.equal(reference, replayed)
    assert bool(replay_metadata["override_used"])
    assert not bool(metadata["override_used"])
    assert torch.equal(replay_metadata["coordinates"], metadata["coordinates"])
    assert metadata["coordinates_preclamp"].shape == metadata["coordinates"].shape
    assert int(metadata["turn_index"].numel()) == 2


def test_coordinate_override_is_validated_and_never_silently_clamped() -> None:
    block = DynamicWindowAttention(16, heads=4, k=4)
    features = torch.randn(1, 16, 4, 4)
    good = torch.zeros(1, 4, 4, 4, 2)
    block(features, coordinate_override=good)
    with pytest.raises(ValueError, match="sampling domain"):
        block(features, coordinate_override=good + 1.5)
    with pytest.raises(ValueError, match="shape"):
        block(features, coordinate_override=torch.zeros(1, 4, 4, 3, 2))
    with pytest.raises(ValueError, match="non-finite"):
        broken = good.clone()
        broken[0, 0, 0, 0, 0] = float("nan")
        block(features, coordinate_override=broken)
    with pytest.raises(ValueError, match="floating"):
        block(features, coordinate_override=torch.zeros(1, 4, 4, 4, 2, dtype=torch.long))


def test_free_offset_mode_drops_the_ring_contribution() -> None:
    torch.manual_seed(5)
    structured = DynamicWindowAttention(16, heads=4, k=4, offset_mode="structured")
    free = DynamicWindowAttention(16, heads=4, k=4, offset_mode="free")
    free.load_state_dict(structured.state_dict())
    features = torch.randn(1, 16, 5, 5)
    _, structured_metadata = structured(features, return_metadata=True)
    _, free_metadata = free(features, return_metadata=True)
    assert not torch.allclose(structured_metadata["coordinates"], free_metadata["coordinates"])
    assert float(free_metadata["coordinates"].abs().max()) <= 1.0
    # ``free`` uses only the pre-existing residual channels: no new parameters.
    assert sum(p.numel() for p in free.parameters()) == sum(p.numel() for p in structured.parameters())


def test_normalized_pixel_step_freezes_degenerate_axes() -> None:
    step = normalized_pixel_step(1, 8, device=torch.device("cpu"), dtype=torch.float32)
    assert float(step[0]) == pytest.approx(2.0 / 7.0)
    assert float(step[1]) == 0.0


# ---------------------------------------------------------------------------
# Geometry capture and C1 exact replay
# ---------------------------------------------------------------------------


def test_geometry_is_captured_for_every_internal_iteration() -> None:
    torch.manual_seed(13)
    expert = AnnotationExpert(feature_channels=16, num_classes=4, window_k=4, max_turns=3)
    shared = torch.randn(2, 16, 6, 6)
    annotation = torch.randn(2, 4, 24, 24)
    output = expert(shared, annotation, turn_index=2, iteration_index=2, capture_geometry=True)
    assert output.depth == 3
    assert len(output.geometry) == 3
    for index, item in enumerate(output.geometry):
        assert item["internal_iteration"] == index
        assert item["coordinates"].shape == (2, 6, 6, 4, 2)
        assert item["coordinates_preclamp"].shape == item["coordinates"].shape
        assert item["attention"].shape[-1] == 4
        assert item["feature_hw"] == (6, 6)
        assert item["state_identity"] == output.state_identity
        assert int(item["iteration_index"][0]) == min(2 + index, 3)
        assert not item["coordinates"].requires_grad


def test_c1_factual_replay_is_exact_and_multi_iteration() -> None:
    net = _net()
    images = _images()
    record, state, output = _record(net, images, turn=2, regress=0.0, fix=0.0)
    assert record.expert_record.depth == 3
    assert len(record.expert_record.coordinates) == 3
    replayed = net.annotation_expert.replay(record.expert_record)
    assert torch.equal(replayed.candidate_logits, record.expert_record.factual_candidate_logits)
    assert float((replayed.candidate_logits - state).abs().max()) == 0.0


def test_alternative_supports_change_only_the_realized_coordinates() -> None:
    net = _net()
    images = _images()
    record, _, _ = _record(net, images, turn=1, regress=0.0, fix=0.0)
    expert_record = record.expert_record
    moved = tuple((item + 0.05).clamp(-1.0, 1.0) for item in expert_record.coordinates)
    alternative = net.annotation_expert.replay(expert_record, moved)
    assert not torch.allclose(
        alternative.candidate_logits, expert_record.factual_candidate_logits, atol=0.0
    )
    # The recorded state is untouched by a replay at alternative supports.
    for original, current in zip(record.expert_record.coordinates, expert_record.coordinates):
        assert torch.equal(original, current)
    again = net.annotation_expert.replay(expert_record)
    assert torch.equal(again.candidate_logits, expert_record.factual_candidate_logits)


def test_replay_fails_closed_after_a_weight_change_or_invalidation() -> None:
    net = _net(fresh=True)
    images = _images()
    record, _, _ = _record(net, images, turn=0, regress=0.0, fix=0.0)
    net.annotation_expert.replay(record.expert_record)
    net.invalidate_candidate_c_records()
    with pytest.raises(StaleReplayRecordError):
        net.annotation_expert.replay(record.expert_record)

    net2 = _net(fresh=True)
    record2, _, _ = _record(net2, images, turn=0, regress=0.0, fix=0.0)
    with torch.no_grad():
        next(iter(net2.annotation_expert.parameters())).add_(0.01)
    with pytest.raises(StaleReplayRecordError):
        net2.annotation_expert.replay(record2.expert_record)


def test_non_ordinary_records_are_never_replayable() -> None:
    from dataclasses import replace

    net = _net()
    record, _, _ = _record(net, _images(), turn=0, regress=0.0, fix=0.0)
    relabelled = replace(record.expert_record, record_kind="candidate_c")
    with pytest.raises(ValueError, match="not replay eligible"):
        net.annotation_expert.replay(relabelled)


# ---------------------------------------------------------------------------
# C2 solver
# ---------------------------------------------------------------------------


def test_solver_finds_a_feasible_improving_support_and_bounds_its_budget() -> None:
    net = _net()
    images = _images()
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.95)
    config = CandidateCConfig(rho_feature_pixels=0.01, lam=1.0)
    result = _run_restitution_solver(
        net.annotation_expert, record, state, config=config, enforce_fix=True
    )
    assert bool(result["c1_passed"].all())
    assert float(result["c1_abs_error"].max()) == 0.0
    assert bool(result["eligible"].all())
    assert bool(result["settled"].all()), result["reason"]
    # A genuine solve, not the identity fallback.
    assert float(result["innovation"].abs().max()) > 0.0
    for row in range(int(result["settled"].numel())):
        assert result["displacement_after"][row] > 0.0
        assert result["objective_after"][row] < float(result["objective_before"][row])
    evals = result["evals"]
    assert evals["factual_replay"] == 1
    assert evals["coordinate_backward"] == 1
    assert 1 <= evals["candidate_checks"] <= 2
    assert evals["total_forward"] == 1 + evals["candidate_checks"]


def test_solver_preserves_every_protected_fix_pixel() -> None:
    net = _net()
    images = _images()
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.95)
    config = CandidateCConfig(rho_feature_pixels=0.01, margin_fraction=0.5)
    result = _run_restitution_solver(
        net.annotation_expert, record, state, config=config, enforce_fix=True
    )
    assert bool(result["settled"].any())
    factual = record.expert_record.factual_candidate_logits
    winner = factual.argmax(dim=1)
    factual_margin = _winning_margin(factual, winner)
    # Protection covers EVERY thresholded predicted FIX pixel; no pixel is
    # dropped because its winning margin is zero or small.
    protected = record.audit_evidence[:, LOCAL_FIX] >= config.fix_threshold
    assert int(protected.sum()) > 0
    assert int(result["num_protected"].sum()) == int(protected.sum())
    assert int(result["num_ties_excluded"].sum()) == 0
    candidate = state + result["gate"] * result["innovation"]
    # C3 is a convex mixture at every pixel, so the protected winning class
    # survives the transfer, not merely the replayed counterfactual.
    assert bool((candidate.argmax(dim=1) == winner)[protected].all())
    candidate_margin = _winning_margin(factual + result["innovation"], winner)
    slack = (config.margin_fraction * factual_margin - candidate_margin)[protected]
    assert float(slack.max()) <= 1e-5


def test_backtracking_rescues_a_step_that_violates_the_fix_margin() -> None:
    """One projected proposal, then one genuine half step, then factual support.

    The half step halves the *realized* displacement, so it is a distinct point
    from the full proposal and can succeed where the full proposal fails.
    """

    net = _net()
    images = _images()
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.95)
    rescued = _run_restitution_solver(
        net.annotation_expert,
        record,
        state,
        config=CandidateCConfig(rho_feature_pixels=0.5),
        enforce_fix=False,
    )
    assert rescued["evals"]["candidate_checks"] == 2
    assert bool(rescued["settled"].any())
    # The row that settled needed the second, genuinely smaller, point.
    assert not bool(rescued["settled"].all())

    hopeless = _run_restitution_solver(
        net.annotation_expert,
        record,
        state,
        config=CandidateCConfig(rho_feature_pixels=0.2),
        enforce_fix=True,
    )
    assert not bool(hopeless["settled"].any())
    assert all(reason == "infeasible" for reason in hopeless["reason"])
    # An infeasible proposal returns the factual support: exactly zero transfer.
    assert int(torch.count_nonzero(hopeless["innovation"])) == 0
    assert hopeless["evals"]["candidate_checks"] == 2
    assert hopeless["evals"]["coordinate_backward"] == 1
    # A failed feasibility check exposes its measured violation, not a null.
    assert all(value is not None and value > 0.0 for value in hopeless["violation_after"])
    assert all(value is False for value in hopeless["checked_feasible"])
    # ... and the rows were rejected for infeasibility, not for non-improvement.
    assert all(value is True for value in hopeless["checked_improved"])


def test_diagnostics_separate_feasible_from_improved() -> None:
    """A feasible-but-non-improving proposal must not be called infeasible."""

    net = _net()
    images = _images()
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.95)
    result = _run_restitution_solver(
        net.annotation_expert,
        record,
        state,
        config=CandidateCConfig(rho_feature_pixels=1.0),
        enforce_fix=False,
    )
    assert not bool(result["settled"].any())
    assert all(value is True for value in result["checked_feasible"])
    assert all(value is False for value in result["checked_improved"])
    assert all(reason == "no_improvement" for reason in result["reason"])
    # Both quantities are measured for the checked proposal.
    assert all(value is not None for value in result["objective_after"])
    assert all(value is not None for value in result["violation_after"])
    assert all(value is not None for value in result["displacement_after"])
    # The SELECTED support is the factual one, so its displacement is zero.
    assert all(value == 0.0 for value in result["selected_displacement"])
    assert int(torch.count_nonzero(result["innovation"])) == 0


def test_proposal_is_bounded_in_feature_pixels_on_a_non_square_grid() -> None:
    """The step is normalized in feature-pixel units, not normalized units.

    On a non-square grid the two axes have different normalized pixel sizes, so
    a step normalized in raw normalized coordinates would saturate the box and
    make the half step a repeat of the full step.
    """

    net = _net()
    torch.manual_seed(11)
    images = torch.randn(2, 3, 32, 48)
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.95)
    height, width = record.expert_record.feature_hw
    assert (height, width) == (8, 12)
    step = normalized_pixel_step(height, width, device=torch.device("cpu"), dtype=torch.float32)
    assert float(step[0]) != float(step[1]), "grid must be non-square for this test"

    rho = 0.5
    result = _run_restitution_solver(
        net.annotation_expert,
        record,
        state,
        config=CandidateCConfig(rho_feature_pixels=rho),
        enforce_fix=False,
        capture_geometry=True,
    )
    geometry = result["geometry"]
    assert geometry is not None
    for factual, chosen in zip(geometry["factual"], geometry["chosen"]):
        displacement = (chosen["coordinates"] - factual["coordinates"]) / step
        assert float(displacement.abs().max()) <= rho + 1e-6
        assert float(displacement[..., 0].abs().max()) <= rho + 1e-6
        assert float(displacement[..., 1].abs().max()) <= rho + 1e-6


def test_half_check_really_halves_the_realized_displacement() -> None:
    """The second candidate is the midpoint, including at the domain boundary."""

    factual = (torch.tensor([[[[[0.9, -0.2], [0.0, 0.5]]]]]),)
    step = torch.tensor([0.25, 0.5])
    trust_region = 1.0 * step
    # A step large enough that one component saturates the box and another
    # would leave the [-1, 1] domain.
    direction = (torch.tensor([[[[[5.0, -5.0], [0.1, 0.05]]]]]),)
    projected = _project_supports(factual, direction, trust_region)
    half = _halve_towards_factual(factual, projected)

    full_delta = projected[0] - factual[0]
    half_delta = half[0] - factual[0]
    assert torch.allclose(half_delta, 0.5 * full_delta)
    assert float(half_delta.abs().max()) < float(full_delta.abs().max())
    # Box and domain hold for both points.
    for point in (projected[0], half[0]):
        assert float((point - factual[0]).abs().div(trust_region).max()) <= 1.0 + 1e-6
        assert float(point.max()) <= 1.0 and float(point.min()) >= -1.0
    # Saturation actually occurred, so this is not a vacuous check.
    assert float((full_delta.abs() / trust_region).max()) == pytest.approx(1.0)
    assert float(projected[0][0, 0, 0, 0, 0]) == pytest.approx(1.0)


def test_dropping_the_fix_constraint_admits_the_step(
) -> None:
    net = _net()
    images = _images()
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.95)
    config = CandidateCConfig(rho_feature_pixels=0.2)
    constrained = _run_restitution_solver(
        net.annotation_expert, record, state, config=config, enforce_fix=True
    )
    unconstrained = _run_restitution_solver(
        net.annotation_expert, record, state, config=config, enforce_fix=False
    )
    assert not bool(constrained["settled"].any())
    assert all(reason == "infeasible" for reason in constrained["reason"])
    # Without the constraint the only remaining gate is the objective itself.
    assert bool(unconstrained["settled"].all())
    assert float(unconstrained["innovation"].abs().max()) > 0.0
    assert all(reason is None for reason in unconstrained["reason"])


def test_no_fix_variant_removes_only_the_constraint() -> None:
    net = _net()
    images = _images()
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.95)
    config = CandidateCConfig(rho_feature_pixels=0.01)
    unconstrained = _run_restitution_solver(
        net.annotation_expert, record, state, config=config, enforce_fix=False
    )
    assert bool(unconstrained["c1_passed"].all())
    assert unconstrained["evals"]["candidate_checks"] <= 2
    assert int(unconstrained["num_protected"].sum()) > 0


def test_negligible_regress_evidence_returns_exactly_the_factual_support() -> None:
    net = _net()
    images = _images()
    record, state, _ = _record(net, images, turn=1, regress=0.0, fix=0.9)
    result = _run_restitution_solver(
        net.annotation_expert, record, state, config=CandidateCConfig(), enforce_fix=True
    )
    assert bool(result["c1_passed"].all())
    assert not bool(result["eligible"].any())
    assert all(reason == "no_regress_evidence" for reason in result["reason"])
    # Exact zero, not a numerical replay residue.
    assert int(torch.count_nonzero(result["innovation"])) == 0
    assert result["evals"]["candidate_checks"] == 0
    assert result["evals"]["coordinate_backward"] == 0


def test_a_stale_retained_state_disables_the_transfer() -> None:
    net = _net()
    images = _images()
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.0)
    result = _run_restitution_solver(
        net.annotation_expert,
        record,
        state + 1.0,
        config=CandidateCConfig(rho_feature_pixels=0.05),
        enforce_fix=True,
    )
    assert all(reason == "stale_record" for reason in result["reason"])
    assert int(torch.count_nonzero(result["innovation"])) == 0


def test_c1_failure_disables_candidate_c_without_touching_the_baseline() -> None:
    net = _net()
    images = _images()
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.0)
    from dataclasses import replace

    corrupted = _AcceptedOrdinaryRecord(
        indices=record.indices,
        expert_record=replace(
            record.expert_record,
            factual_candidate_logits=record.expert_record.factual_candidate_logits + 3.0,
        ),
        audit_evidence=record.audit_evidence,
        turn=record.turn,
        accepted=record.accepted,
    )
    result = _run_restitution_solver(
        net.annotation_expert,
        corrupted,
        corrupted.expert_record.factual_candidate_logits,
        config=CandidateCConfig(rho_feature_pixels=0.05),
        enforce_fix=True,
    )
    assert not bool(result["c1_passed"].any())
    assert all(reason == "replay_failure" for reason in result["reason"])
    assert int(torch.count_nonzero(result["innovation"])) == 0


def test_direct_rollback_only_moves_thresholded_regress_pixels() -> None:
    net = _net("direct_rollback")
    images = _images()
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.0)
    result = _run_direct_rollback(record, state, config=CandidateCConfig())
    innovation = result["innovation"]
    selected = record.audit_evidence[:, LOCAL_REGRESS] >= 0.5
    assert int(selected.sum()) > 0
    assert float(innovation[:, :, ~selected[0]].abs().max()) == 0.0
    assert float(innovation[:, :, selected[0]].abs().max()) > 0.0
    assert result["evals"]["total_forward"] == 0
    expected = record.expert_record.annotation_logits - record.expert_record.factual_candidate_logits
    assert torch.allclose(innovation[:, :, selected[0]], expected[:, :, selected[0]])


# ---------------------------------------------------------------------------
# Gradient policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wrapper", ["no_grad", "inference_mode"])
def test_solver_works_under_no_grad_and_inference_mode_without_leaking(wrapper: str) -> None:
    net = _net(fresh=True)
    images = _images(1)
    # Pre-existing accumulated gradients must survive the solver untouched.
    seeded = next(iter(net.annotation_expert.parameters()))
    seeded.grad = torch.full_like(seeded, 0.25)
    flags = {name: p.requires_grad for name, p in net.named_parameters()}

    context = torch.no_grad() if wrapper == "no_grad" else torch.inference_mode()
    with context:
        record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.0)
        result = _run_restitution_solver(
            net.annotation_expert,
            record,
            state,
            config=CandidateCConfig(rho_feature_pixels=0.05),
            enforce_fix=True,
        )

    assert bool(result["c1_passed"].all())
    assert {name: p.requires_grad for name, p in net.named_parameters()} == flags
    assert torch.equal(seeded.grad, torch.full_like(seeded, 0.25))
    for name, parameter in net.named_parameters():
        if parameter is seeded:
            continue
        assert parameter.grad is None, f"solver leaked a gradient into {name}"
    assert not result["innovation"].requires_grad
    assert not result["gate"].requires_grad


def test_solver_gradient_never_reaches_the_image_encoder_or_auditor() -> None:
    net = _net(fresh=True)
    images = _images(1).requires_grad_(True)
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.0)
    _run_restitution_solver(
        net.annotation_expert,
        record,
        state,
        config=CandidateCConfig(rho_feature_pixels=0.05),
        enforce_fix=True,
    )
    assert images.grad is None
    for parameter in net.auditor.parameters():
        assert parameter.grad is None
    for parameter in net.encoder.parameters():
        assert parameter.grad is None


def test_candidate_c_inference_still_trains_through_the_retained_state() -> None:
    net = _net(fresh=True)
    images = _images(1)
    output = net.infer(images, mode="always_accept_refinement", t_max=2)
    output["logits"].square().mean().backward()
    assert images.grad is None
    trained = [p for p in net.annotation_expert.parameters() if p.grad is not None]
    assert trained, "the ordinary annotation path must keep its gradients"
    assert all(torch.isfinite(p.grad).all() for p in trained)


# ---------------------------------------------------------------------------
# Temporal contract and Auditor authority
# ---------------------------------------------------------------------------


def test_accepted_restitution_is_never_replayed_recursively() -> None:
    net = _net()
    images = _images(1)
    with torch.no_grad():
        output = net.infer(images, mode="always_accept_refinement", t_max=3)
    rows = output["candidate_c_diagnostics"]
    assert len(rows) == 3
    assert rows[0]["eligible"] is False and rows[0]["fallback_reason"] == "no_accepted_history"
    assert rows[1]["record_kind"] == "ordinary"
    assert rows[1]["c1_passed"] is True
    # The restitution action is accepted, so the ordinary history is cleared
    # and the next active turn is forced back onto the ordinary annotator.
    assert rows[2]["eligible"] is False
    assert rows[2]["fallback_reason"] == "previous_action_not_replayable"
    assert rows[2]["record_kind"] in {"candidate_c", "rollback"}
    assert all(row["accepted"] is not None and row["delta_q"] is not None for row in rows)


def test_official_auditor_still_halts_a_rejected_candidate() -> None:
    net = _net()
    images = _images(2)
    with torch.no_grad():
        output = net.infer(images, mode="self_audit", tau_accept=1e9, t_max=3)
    assert int(output["accepted_count"].max()) == 0
    assert bool((output["halt_turn"] == 0).all())
    assert torch.equal(output["logits"], output["initial_logits"])
    assert len(output["candidate_c_diagnostics"]) == 2
    assert all(row["accepted"] is False for row in output["candidate_c_diagnostics"])


def test_diagnostics_rows_are_json_compatible_and_never_fake_zero() -> None:
    import json

    net = _net()
    with torch.no_grad():
        output = net.infer(_images(1), mode="always_accept_refinement", t_max=3)
    rows = output["candidate_c_diagnostics"]
    json.dumps(rows)
    required = {
        "turn", "sample_index", "eligible", "record_kind", "c1_passed", "c1_max_abs_err",
        "regress_mass", "fix_mass", "num_protected", "num_ties_excluded", "feasible",
        "improved", "objective_before", "objective_after", "constraint_violation",
        "coordinate_displacement", "innovation_magnitude", "fallback_reason",
        "accepted_path", "evals", "delta_q", "accepted",
    }
    for row in rows:
        assert required <= set(row)
    # Unmeasured quantities are null, never a stand-in zero.
    assert rows[0]["c1_max_abs_err"] is None
    assert rows[0]["objective_before"] is None


# ---------------------------------------------------------------------------
# Baseline preservation and configuration
# ---------------------------------------------------------------------------


def test_current_mode_is_the_untouched_baseline() -> None:
    images = _images(2)
    baseline = _net("current")
    default = SelfAuditNet(
        encoder_name="convnext_tiny",
        pretrained_encoder=False,
        shared_channels=32,
        num_classes=4,
        window_k=4,
        max_turns=3,
    ).eval()
    assert default.window_mode == "current"
    assert baseline.annotation_expert.audit_conditioning == "full"
    assert baseline.annotation_expert.refinement_block.offset_mode == "structured"
    with torch.no_grad():
        output = baseline.infer(images, mode="always_accept_refinement", t_max=3)
        annotation = baseline.forward_annotation(images, turns=2)
    assert output["candidate_c_diagnostics"] == []
    assert output["window_mode"] == "current"
    assert torch.equal(output["initial_logits"], annotation["a0_logits"])
    assert not baseline.uses_transition_records


def test_feature_only_removes_both_audit_entry_paths() -> None:
    torch.manual_seed(17)
    expert = AnnotationExpert(
        feature_channels=16, num_classes=4, window_k=4, max_turns=3, audit_conditioning="feature_only"
    )
    shared = torch.randn(1, 16, 6, 6)
    annotation = torch.randn(1, 4, 24, 24)
    evidence = torch.rand(1, 3, 24, 24)
    with torch.no_grad():
        without = expert(shared, annotation, previous_audit_evidence=None, turn_index=1)
        with_evidence = expert(shared, annotation, previous_audit_evidence=evidence, turn_index=1)
    assert torch.equal(without.candidate_logits, with_evidence.candidate_logits)

    full = AnnotationExpert(feature_channels=16, num_classes=4, window_k=4, max_turns=3)
    full.load_state_dict(expert.state_dict())
    with torch.no_grad():
        baseline = full(shared, annotation, previous_audit_evidence=evidence, turn_index=1)
    assert not torch.allclose(baseline.candidate_logits, with_evidence.candidate_logits)


def test_candidate_c_config_is_validated_strictly() -> None:
    assert resolve_candidate_c_config(None) == CandidateCConfig()
    assert resolve_candidate_c_config({"lam": 2.0}).lam == 2.0
    with pytest.raises(ValueError, match="unknown candidate_c keys"):
        resolve_candidate_c_config({"enabled": True})
    with pytest.raises(ValueError, match="non-negative"):
        CandidateCConfig(rho_feature_pixels=-1.0)
    with pytest.raises(ValueError, match="finite"):
        CandidateCConfig(lam=float("inf"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CandidateCConfig(margin_fraction=2.0)
    with pytest.raises(TypeError):
        CandidateCConfig(max_backtracks=1.5)
    with pytest.raises(ValueError, match="window_mode"):
        SelfAuditNet(window_mode="candidate_b")


def test_every_window_mode_runs_and_keeps_the_annotation_contract() -> None:
    images = _images(2)
    for mode in ("current", "feature_only", "free_offsets", "candidate_c", "candidate_c_no_fix", "direct_rollback"):
        net = _net(mode)
        with torch.no_grad():
            output = net.infer(images, mode="always_accept_refinement", t_max=2)
        assert output["logits"].shape == (2, 4, 32, 32)
        assert torch.isfinite(output["logits"]).all()
        assert output["window_mode"] == mode
        assert (len(output["candidate_c_diagnostics"]) > 0) == (mode in {
            "candidate_c", "candidate_c_no_fix", "direct_rollback"
        })


@pytest.mark.parametrize("mode", ["candidate_c", "candidate_c_no_fix", "direct_rollback"])
def test_infer_matches_under_no_grad_and_inference_mode(mode: str) -> None:
    """The deployment path may wrap inference; the solver must still run."""

    net = _net(mode, fresh=True)
    images = _images(2)
    with torch.inference_mode():
        inferred = net.infer(images, mode="always_accept_refinement", t_max=3)
    with torch.no_grad():
        no_grad = net.infer(images, mode="always_accept_refinement", t_max=3)
    assert torch.equal(inferred["logits"], no_grad["logits"])
    assert [row["c1_passed"] for row in inferred["candidate_c_diagnostics"]] == [
        row["c1_passed"] for row in no_grad["candidate_c_diagnostics"]
    ]
    assert [name for name, p in net.named_parameters() if p.grad is not None] == []


def test_replay_stays_exact_inside_the_collection_autocast_context() -> None:
    net = _net()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = net.infer(_images(1), mode="always_accept_refinement", t_max=3)
    checked = [row for row in output["candidate_c_diagnostics"] if row["c1_passed"] is not None]
    assert checked
    # The replay re-executes inside the same autocast context it was collected
    # in, so the factual output is reproduced rather than merely approximated.
    assert all(row["c1_passed"] for row in checked)
    assert all(row["c1_max_abs_err"] == 0.0 for row in checked)


def test_capture_geometry_emits_factual_and_chosen_supports_per_row() -> None:
    net = _net()
    images = _images(1)
    with torch.no_grad():
        plain = net.infer(images, mode="always_accept_refinement", t_max=3)
        detailed = net.infer(images, mode="always_accept_refinement", t_max=3, capture_geometry=True)
    # The ordinary rollout output stays free of raw tensors.
    assert "candidate_c_geometry" not in plain
    geometry = detailed["candidate_c_geometry"]
    rows = detailed["candidate_c_diagnostics"]
    assert len(geometry) == len(rows)
    assert torch.equal(plain["logits"], detailed["logits"])

    available = [item for item in geometry if "unavailable_reason" not in item]
    assert available, "at least one turn must replay an accepted ordinary record"
    for element, row in zip(geometry, rows):
        assert element["turn"] == row["turn"]
        assert element["sample_index"] == row["sample_index"]
    for element in available:
        assert element["record_turn"] is not None
        assert len(element["factual"]) == len(element["chosen"])
        for factual, chosen in zip(element["factual"], element["chosen"]):
            assert factual["coordinates"].shape == chosen["coordinates"].shape
            assert factual["coordinates"].shape[0] == 1
            assert factual["coordinates"].shape[-2] == element["k"]
            assert factual["attention"].shape[0] == 1
            assert factual["state_identity"] == chosen["state_identity"]
            assert factual["depth_index"] == chosen["depth_index"]
    unavailable = [item for item in geometry if "unavailable_reason" in item]
    for element in unavailable:
        assert isinstance(element["unavailable_reason"], str)


def test_replay_keeps_the_factual_batch_composition_when_rows_halt() -> None:
    """B>1 with a mixed accept/halt batch must not change the replay shape.

    Replaying only the surviving rows would change kernel tiling and reduction
    order, which can break the strict C1 tolerance under AMP.  The record keeps
    every row of the factual forward, so the replay reproduces it exactly.
    """

    net = _net()
    images = _images(4)
    with torch.no_grad():
        probe = net.infer(images, mode="self_audit", tau_accept=-1e9, t_max=1)
        threshold = float(probe["audits"][0]["delta_q"].reshape(-1).median())
        output = net.infer(images, mode="self_audit", tau_accept=threshold, t_max=3)

    accepted_at_zero = output["audits"][0]["accepted"]
    assert bool(accepted_at_zero.any()) and not bool(accepted_at_zero.all()), "need a mixed batch"

    replayed = [
        row for row in output["candidate_c_diagnostics"] if row["c1_passed"] is not None
    ]
    assert replayed, "at least one surviving row must replay its accepted record"
    for row in replayed:
        assert row["c1_passed"] is True
        assert row["c1_max_abs_err"] == 0.0
        assert row["record_kind"] == "ordinary"
        assert row["record_turn"] == row["turn"] - 1
        # The whole factual forward is replayed, not just the survivors.
        assert row["solver_group_size"] == 4
        assert row["sample_index"] in set(range(4))


def test_record_turn_is_present_on_every_diagnostic_row() -> None:
    net = _net()
    with torch.no_grad():
        rows = net.infer(_images(1), mode="always_accept_refinement", t_max=3)[
            "candidate_c_diagnostics"
        ]
    assert all("record_turn" in row for row in rows)
    assert rows[0]["record_turn"] is None
    replayed = [row for row in rows if row["record_kind"] == "ordinary"]
    assert replayed
    for row in replayed:
        assert isinstance(row["record_turn"], int)
        assert 0 <= row["record_turn"] < row["turn"]


def test_candidate_c_adds_no_parameters_or_buffers() -> None:
    baseline = _net("current")
    for mode in ("feature_only", "free_offsets", "candidate_c", "candidate_c_no_fix", "direct_rollback"):
        net = _net(mode)
        assert set(net.state_dict()) == set(baseline.state_dict())
        assert sum(p.numel() for p in net.parameters()) == sum(
            p.numel() for p in baseline.parameters()
        )
    # The replay generation is an ordinary attribute, never persisted state.
    assert not any("replay" in key or "candidate_c" in key for key in baseline.state_dict())


# ---------------------------------------------------------------------------
# C1 failure must disable Candidate C and fall back to NORMAL annotation
# ---------------------------------------------------------------------------


def _corrupt_record_at_turn(net: SelfAuditNet, turn: int):
    """Fault injection: make the record captured at ``turn`` fail C1 later."""

    from dataclasses import replace

    original = net.annotation_expert.build_replay_record

    def patched(**kwargs):
        record = original(**kwargs)
        if int(kwargs["turn_index"]) == turn:
            return replace(
                record,
                factual_candidate_logits=record.factual_candidate_logits + 5.0,
            )
        return record

    net.annotation_expert.build_replay_record = patched
    return original


def test_c1_failure_routes_the_row_to_normal_annotation_in_real_infer() -> None:
    """Stored-factual-stale fixture: the RECORD itself was corrupted.

    This perturbs the stored factual logits, so it trips the retained-state
    staleness check as well as the numeric comparison.  It is deliberately kept
    separate from the numeric-only fixture below, which corrupts nothing but
    the replay output and is the one that actually exercises the tri-state
    ``c1_passed`` branch.
    """

    net = _net(fresh=True)
    images = _images(1)
    _corrupt_record_at_turn(net, 0)
    with torch.no_grad():
        corrupted = net.infer(images, mode="always_accept_refinement", t_max=3)

    # Reference: the same weights running the ORDINARY annotator at every turn.
    # Turn 0 is ordinary in both runs and is accepted in both, so the state and
    # the recurrent audit evidence entering turn 1 are identical.
    ordinary = _net("current", fresh=True)
    with torch.no_grad():
        reference = ordinary.infer(images, mode="always_accept_refinement", t_max=3)

    rows = corrupted["candidate_c_diagnostics"]
    failed = [row for row in rows if row["fallback_reason"] == "replay_failure"]
    assert len(failed) == 1, [row["fallback_reason"] for row in rows]
    row = failed[0]
    assert row["turn"] == 1
    assert row["accepted_path"] == "ordinary", "C1 failure must use NORMAL annotation"
    assert row["eligible"] is False
    assert row["c1_passed"] is False
    assert row["c1_max_abs_err"] is not None and row["c1_max_abs_err"] > 0.0
    assert row["record_kind"] == "ordinary"
    assert row["c1_failure_kind"] == "numeric"
    # No solver work was spent on a row that cannot be replayed.
    assert row["evals"]["coordinate_backward"] == 0
    assert row["evals"]["candidate_checks"] == 0
    assert row["coordinate_displacement"] is None
    assert row["innovation_magnitude"] is None
    # The official Auditor still judged the fallback candidate.
    assert row["delta_q"] is not None and row["accepted"] is not None

    # The fallback really is the ordinary annotator: turn 1's candidate matches
    # the ordinary run, and it is NOT the frozen identity the old code produced.
    assert torch.equal(
        corrupted["transition_candidates"][1], reference["transition_candidates"][1]
    )
    assert not torch.equal(
        corrupted["transition_candidates"][1], corrupted["transition_previous"][1]
    )


def test_fallback_output_becomes_a_new_ordinary_record_not_a_candidate_c_one() -> None:
    net = _net(fresh=True)
    _corrupt_record_at_turn(net, 0)
    with torch.no_grad():
        rows = net.infer(_images(1), mode="always_accept_refinement", t_max=3)[
            "candidate_c_diagnostics"
        ]
    assert rows[1]["fallback_reason"] == "replay_failure"
    # Turn 1 took the ordinary path and its accepted output was recorded as an
    # ORDINARY record for the current turn, so turn 2 can replay it exactly.
    assert rows[2]["record_kind"] == "ordinary"
    assert rows[2]["record_turn"] == 1
    assert rows[2]["c1_passed"] is True
    assert rows[2]["c1_max_abs_err"] == 0.0
    assert rows[2]["accepted_path"] != "ordinary" or rows[2]["eligible"] is True


def test_stale_replay_record_state_does_not_crash_infer() -> None:
    """A record that no longer matches the live model state falls back, narrowly."""

    net = _net(fresh=True)
    calls: list[int] = []

    def raising(*args, **kwargs):
        calls.append(1)
        raise StaleReplayRecordError("injected state-generation mismatch")

    net.annotation_expert.replay = raising
    with torch.no_grad():
        output = net.infer(_images(1), mode="always_accept_refinement", t_max=3)
    assert calls, "the solver must actually have attempted a replay"
    rows = output["candidate_c_diagnostics"]
    fallen = [row for row in rows if str(row["fallback_reason"]).startswith("stale_record_state")]
    assert fallen
    for row in fallen:
        assert row["accepted_path"] == "ordinary"
        assert row["eligible"] is False
        assert row["evals"]["candidate_checks"] == 0
        assert row["delta_q"] is not None
    assert torch.isfinite(output["logits"]).all()


def test_unrelated_replay_errors_are_not_swallowed() -> None:
    net = _net(fresh=True)

    def broken(*args, **kwargs):
        raise RuntimeError("programmer error inside replay")

    net.annotation_expert.replay = broken
    with pytest.raises(RuntimeError, match="programmer error"):
        with torch.no_grad():
            net.infer(_images(1), mode="always_accept_refinement", t_max=3)


def test_stale_retained_state_also_routes_to_normal_annotation() -> None:
    net = _net(fresh=True)
    images = _images()
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.95)
    result = _run_restitution_solver(
        net.annotation_expert,
        record,
        state + 1.0,
        config=CandidateCConfig(rho_feature_pixels=0.05),
        enforce_fix=True,
    )
    # ``stale`` is exposed so the caller can route these rows to the ordinary
    # annotator rather than transferring a zero innovation.
    assert bool(result["stale"].all())
    assert all(reason == "stale_record" for reason in result["reason"])
    assert result["evals"]["coordinate_backward"] == 0
    assert result["evals"]["candidate_checks"] == 0


def test_rejected_record_rows_are_never_eligible() -> None:
    net = _net(fresh=True)
    images = _images(2)
    record, state, _ = _record(net, images, turn=1, regress=0.95, fix=0.95)
    record.accepted[1] = False
    result = _run_restitution_solver(
        net.annotation_expert,
        record,
        state,
        config=CandidateCConfig(rho_feature_pixels=0.05),
        enforce_fix=True,
    )
    # Row 1 stays in the replay purely to preserve the factual batch
    # composition; eligibility is an explicit intersection with ``accepted``.
    assert bool(result["eligible"][0])
    assert not bool(result["eligible"][1])
    assert result["reason"][1] == "record_row_not_accepted"
    assert not bool(result["settled"][1])
    assert int(torch.count_nonzero(result["innovation"][1])) == 0
    assert int(result["c1_passed"].numel()) == 2, "the whole group is still replayed"


def test_config_bounds_are_enforced_at_the_runtime_constructor() -> None:
    """Not only the YAML schema: direct construction must reject these too."""

    with pytest.raises(ValueError, match="0..2"):
        CandidateCConfig(max_backtracks=3)
    with pytest.raises(ValueError, match="0..2"):
        CandidateCConfig(max_backtracks=-1)
    with pytest.raises(ValueError, match="strictly positive"):
        CandidateCConfig(rho_feature_pixels=0.0)
    for field in ("fix_threshold", "regress_threshold", "margin_fraction", "min_regress_mass"):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            CandidateCConfig(**{field: 1.5})
        with pytest.raises(ValueError, match="non-negative"):
            CandidateCConfig(**{field: -0.5})
    with pytest.raises(ValueError, match="0..2"):
        SelfAuditNet(window_mode="candidate_c", candidate_c={"max_backtracks": 5})
    with pytest.raises(ValueError, match="non-negative"):
        SelfAuditNet(window_mode="candidate_c", candidate_c={"rho_feature_pixels": -1.0})
    with pytest.raises(ValueError, match="strictly positive"):
        SelfAuditNet(window_mode="candidate_c", candidate_c={"rho_feature_pixels": 0.0})
    with pytest.raises(ValueError, match="0..2"):
        resolve_candidate_c_config({"max_backtracks": 99})
    # The solver can never run more checks than the hard ceiling.
    assert CandidateCConfig().max_backtracks <= MAX_CANDIDATE_CHECKS


def test_coordinate_override_rejects_a_different_device_index() -> None:
    block = DynamicWindowAttention(8, heads=2, k=2)
    features = torch.randn(1, 8, 3, 3)
    override = torch.zeros(1, 3, 3, 2, 2)
    block(features, coordinate_override=override)
    # Full device equality, index included.
    assert override.device == features.device
    with pytest.raises(ValueError, match="but features are on"):
        validate_coordinate_override(
            override,
            batch=1,
            height=3,
            width=3,
            k=2,
            device=torch.device("cuda", 1),
        )


def test_mixed_group_c1_failure_still_yields_a_replayable_ordinary_record() -> None:
    """One row fails C1 and one does not, in the same turn.

    The failing row joins the ordinary partition before its single forward
    runs, so the replay record built from that forward covers exactly the rows
    of that forward and can be replayed bit-exactly on the next turn.
    """

    from dataclasses import replace

    net = _net(fresh=True)
    expert = net.annotation_expert
    original = expert.build_replay_record

    def patched(**kwargs):
        record = original(**kwargs)
        if int(kwargs["turn_index"]) != 0:
            return record
        corrupted = record.factual_candidate_logits.clone()
        corrupted[0] = corrupted[0] + 5.0  # only sample 0 will fail C1
        return replace(record, factual_candidate_logits=corrupted)

    expert.build_replay_record = patched
    with torch.no_grad():
        output = net.infer(_images(2), mode="always_accept_refinement", t_max=3)

    rows = output["candidate_c_diagnostics"]
    turn_one = {row["sample_index"]: row for row in rows if row["turn"] == 1}
    assert turn_one[0]["fallback_reason"] == "replay_failure"
    assert turn_one[0]["accepted_path"] == "ordinary"
    assert turn_one[0]["c1_passed"] is False
    assert turn_one[1]["c1_passed"] is True
    assert turn_one[1]["accepted_path"] != "ordinary"

    # The mixed turn produced a sound ordinary record for the fallback row.
    turn_two = {row["sample_index"]: row for row in rows if row["turn"] == 2}
    assert turn_two[0]["record_kind"] == "ordinary"
    assert turn_two[0]["record_turn"] == 1
    assert turn_two[0]["c1_passed"] is True
    assert turn_two[0]["c1_max_abs_err"] == 0.0
    assert torch.isfinite(output["logits"]).all()


def test_free_offset_control_has_the_same_per_axis_reach_as_structured() -> None:
    """The free control must not be handicapped by a tighter bound.

    ``structured`` reaches centre + radius + residual along one axis; giving
    ``free`` only the residual bound would measure the bound rather than the
    parameterization.
    """

    torch.manual_seed(5)
    structured = DynamicWindowAttention(16, heads=4, k=4, offset_mode="structured")
    free = DynamicWindowAttention(16, heads=4, k=4, offset_mode="free")
    free.load_state_dict(structured.state_dict())
    generator = structured.generator
    reach = generator.free_support_reach
    assert reach == pytest.approx(
        generator.max_center_displacement + generator.max_radius + generator.max_residual_offset
    )
    assert reach > generator.max_residual_offset

    features = torch.randn(2, 16, 5, 7)
    free_parameters = free.generate_coordinates(features)
    base = free._base_grid(5, 7, features.device, features.dtype)
    offsets = free_parameters.coordinates_preclamp - base
    # The free support is exactly the unit tanh scaled to the structured reach.
    assert torch.allclose(offsets, free_parameters.residual_unit * reach, atol=1e-6)
    bounded = free_parameters.residual_unit * generator.max_residual_offset
    assert float(offsets.abs().max()) > float(bounded.abs().max())
    assert torch.isfinite(free_parameters.coordinates).all()
    # Still no ring contribution and still no extra parameters.
    structured_parameters = structured.generate_coordinates(features)
    assert not torch.allclose(
        structured_parameters.coordinates, free_parameters.coordinates
    )
    assert sum(p.numel() for p in free.parameters()) == sum(
        p.numel() for p in structured.parameters()
    )


def test_free_offset_reach_is_defined_when_the_residual_bound_is_zero() -> None:
    block = DynamicWindowAttention(16, heads=4, k=4, offset_mode="free", max_residual_offset=0.0)
    features = torch.randn(1, 16, 4, 4)
    parameters = block.generate_coordinates(features)
    assert torch.isfinite(parameters.coordinates).all()
    assert block.generator.free_support_reach > 0.0
    # The bounded residual collapses to zero but the control keeps its reach:
    # the unit tanh is rescaled, never divided by the zero bound.
    assert float(parameters.residual_offsets.abs().max()) == 0.0
    base = block._base_grid(4, 4, features.device, features.dtype)
    assert float((parameters.coordinates_preclamp - base).abs().max()) > 0.0


def test_rows_sharing_one_solver_invocation_are_identifiable() -> None:
    net = _net(fresh=True)
    with torch.no_grad():
        rows = net.infer(_images(3), mode="always_accept_refinement", t_max=3)[
            "candidate_c_diagnostics"
        ]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["solver_invocation_id"] is not None:
            grouped.setdefault(row["solver_invocation_id"], []).append(row)
    assert grouped, "at least one turn must run the solver"
    for identifier, members in grouped.items():
        counts = {tuple(sorted(member["evals"].items())) for member in members}
        # Every row of one invocation reports that invocation's counts, so a
        # consumer must deduplicate on the id before summing.
        assert len(counts) == 1
        assert len({member["turn"] for member in members}) == 1
        assert all(member["solver_group_size"] == members[0]["solver_group_size"] for member in members)
    ordinary = [row for row in rows if row["solver_invocation_id"] is None]
    assert all(row["evals"]["total_forward"] == 0 for row in ordinary)


def test_numeric_only_replay_failure_takes_the_normal_annotation_path() -> None:
    """The decisive fixture: ONLY the replay output is wrong.

    The record, the retained state and the model state are all untouched, so
    the staleness checks stay clean and the routing decision rests entirely on
    the numeric C1 comparison.  ``c1_passed`` is a 0-dim ``torch.bool`` tensor
    here, and ``tensor(False) is False`` is always false -- that identity check
    silently left every numeric replay failure on the identity path, and the
    stored-factual fixture above could not catch it because corrupting the
    record also set the stale flag.
    """

    from dataclasses import replace

    net = _net(fresh=True)
    images = _images(1)
    expert = net.annotation_expert
    original = expert.replay

    def perturbed(record, coordinates=None, **kwargs):
        output = original(record, coordinates, **kwargs)
        return replace(output, candidate_logits=output.candidate_logits + 1.0)

    expert.replay = perturbed
    with torch.no_grad():
        corrupted = net.infer(images, mode="always_accept_refinement", t_max=3)

    ordinary = _net("current", fresh=True)
    with torch.no_grad():
        reference = ordinary.infer(images, mode="always_accept_refinement", t_max=3)

    rows = corrupted["candidate_c_diagnostics"]
    failed = [row for row in rows if row["fallback_reason"] == "replay_failure"]
    assert failed, [row["fallback_reason"] for row in rows]
    row = failed[0]
    assert row["turn"] == 1
    assert row["c1_passed"] is False
    assert row["c1_failure_kind"] == "numeric"
    assert row["c1_max_abs_err"] == pytest.approx(1.0)
    # The regression this fixture exists for: NOT the identity fallback.
    assert row["accepted_path"] == "ordinary"
    assert row["accepted_path"] != "factual_support"
    assert row["eligible"] is False
    # Nothing was stale: the record and the retained state are untouched.
    assert row["fallback_reason"] != "stale_record"
    assert row["coordinate_displacement"] is None
    assert row["innovation_magnitude"] is None
    # The replay ran once for the C1 check and nothing else was spent.
    assert row["evals"]["factual_replay"] == 1
    assert row["evals"]["coordinate_backward"] == 0
    assert row["evals"]["candidate_checks"] == 0
    # The official Auditor still judged the fallback candidate.
    assert row["delta_q"] is not None and row["accepted"] is not None

    # The fallback candidate IS the ordinary annotator's, and is not the frozen
    # identity the identity-comparison bug produced.
    assert torch.equal(
        corrupted["transition_candidates"][1], reference["transition_candidates"][1]
    )
    assert not torch.equal(
        corrupted["transition_candidates"][1], corrupted["transition_previous"][1]
    )


def test_numeric_only_replay_failure_is_not_reported_as_stale() -> None:
    """The solver's own staleness flags stay clean under a numeric-only fault."""

    from dataclasses import replace

    net = _net(fresh=True)
    record, state, _ = _record(net, _images(2), turn=1, regress=0.95, fix=0.95)
    expert = net.annotation_expert
    original = expert.replay
    expert.replay = lambda rec, coords=None, **kw: replace(
        original(rec, coords, **kw),
        candidate_logits=original(rec, coords, **kw).candidate_logits + 1.0,
    )
    result = _run_restitution_solver(
        expert,
        record,
        state,
        config=CandidateCConfig(rho_feature_pixels=0.05),
        enforce_fix=True,
    )
    assert not bool(result["c1_passed"].any())
    assert not bool(result["stale"].any()), "a numeric fault must not look like staleness"
    assert all(reason == "replay_failure" for reason in result["reason"])
    assert not bool(result["eligible"].any())
    assert result["evals"]["coordinate_backward"] == 0
    assert result["evals"]["candidate_checks"] == 0


def test_preflight_state_failure_is_counted_as_a_c1_failure() -> None:
    """C1 covers a valid recorded state, not only the numeric comparison.

    A ``StaleReplayRecordError`` happens before any comparison, so the numeric
    error is genuinely unmeasured and stays null -- but the row must still
    report ``c1_passed: False`` so a trainer counting replay failures observes
    invalid-state attempts instead of skipping them.
    """

    net = _net(fresh=True)

    def raising(*args, **kwargs):
        raise StaleReplayRecordError("injected state-generation mismatch")

    net.annotation_expert.replay = raising
    with torch.no_grad():
        rows = net.infer(_images(1), mode="always_accept_refinement", t_max=3)[
            "candidate_c_diagnostics"
        ]
    fallen = [row for row in rows if str(row["fallback_reason"]).startswith("stale_record_state")]
    assert fallen
    for row in fallen:
        assert row["c1_passed"] is False, "an invalid-state replay attempt is a C1 failure"
        assert row["c1_failure_kind"] == "record_state"
        # Not measured, so not invented.
        assert row["c1_max_abs_err"] is None
        assert row["c1_max_rel_err"] is None
        assert row["accepted_path"] == "ordinary"
        assert row["record_turn"] is not None
    # A consumer counting replay failures sees both kinds and no false zero.
    failures = [row for row in rows if row["c1_passed"] is False]
    assert len(failures) == len(fallen)
    assert {row["c1_failure_kind"] for row in failures} == {"record_state"}


def test_geometry_reports_the_effective_conditioning_identity() -> None:
    """Depth 3 at turn 2 requests iteration 4 but the embedding table stops at 3."""

    net = _net(fresh=True)
    record, state, output = _record(net, _images(1), turn=2, regress=0.95, fix=0.95)
    expert_record = record.expert_record
    assert expert_record.depth == 3
    table = net.annotation_expert.refinement_block.generator.iteration_embedding
    assert table.num_embeddings == 4

    captured = [int(item["iteration_index"].reshape(-1)[0]) for item in output.geometry]
    assert captured == [2, 3, 3], "the clamp is what the generator actually applied"
    assert expert_record.effective_iteration_indices == (2, 3, 3)
    assert expert_record.effective_turn_indices == (2, 2, 2)

    result = _run_restitution_solver(
        net.annotation_expert,
        record,
        state,
        config=CandidateCConfig(rho_feature_pixels=0.05),
        enforce_fix=True,
        capture_geometry=True,
    )
    geometry = result["geometry"]
    for side in ("factual", "chosen"):
        assert [entry["iteration_index"] for entry in geometry[side]] == [2, 3, 3]
        assert [entry["turn_index"] for entry in geometry[side]] == [2, 2, 2]
        # The raw request is kept separately rather than discarded.
        assert [entry["requested_iteration_index"] for entry in geometry[side]] == [2, 3, 4]
    # Runtime replay identities are unchanged: the record still replays exactly.
    replayed = net.annotation_expert.replay(expert_record)
    assert torch.equal(replayed.candidate_logits, expert_record.factual_candidate_logits)


# ---------------------------------------------------------------------------
# Tied and near-tied protected FIX pixels
#
# A real forward pass cannot be asked to place a protected pixel at an EXACT
# logit tie, so these tests script the replay instead.  The scripted replay is
# exact at the factual supports (the displacement it responds to is identically
# zero there), so C1 holds by construction and the solver runs its ordinary
# path; only the logits it observes are controlled.
# ---------------------------------------------------------------------------


class _ScriptedReplayExpert:
    """Stands in for ``AnnotationExpert`` for the one call the solver makes.

    ``candidate_logits = base + shift * sensitivity`` where ``shift`` is the
    signed total coordinate displacement of the proposal from the factual
    supports.  At the factual supports ``shift`` is exactly zero, so the replay
    reproduces the stored factual output bit-for-bit.
    """

    def __init__(
        self,
        base_logits: torch.Tensor,
        supports: tuple[torch.Tensor, ...],
        sensitivity: torch.Tensor,
        *,
        moved_logits: torch.Tensor | None = None,
        moved_mask: torch.Tensor | None = None,
    ) -> None:
        self._base = base_logits
        self._supports = supports
        self._sensitivity = sensitivity
        # Optional pixels that jump to an EXACT prescribed value as soon as the
        # proposal moves at all.  Exact logit equality cannot be reached by a
        # smooth response to a floating displacement, and one constraint below
        # is only ever decided at exact equality.
        self._moved_logits = moved_logits
        self._moved_mask = moved_mask

    def replay(self, record, coordinates=None, *, capture_geometry: bool = False):
        supports = self._supports if coordinates is None else tuple(coordinates)
        shift = torch.zeros(self._base.shape[0], dtype=self._base.dtype)
        for new, base in zip(supports, self._supports):
            shift = shift + (new - base).flatten(1).sum(1)
        logits = self._base + shift.view(-1, 1, 1, 1) * self._sensitivity
        if self._moved_logits is not None and self._moved_mask is not None:
            moved = (shift != 0).view(-1, 1, 1, 1) & self._moved_mask
            logits = torch.where(moved, self._moved_logits, logits)
        return AnnotationExpertOutput(
            delta_logits=torch.zeros_like(logits),
            update_gate=torch.zeros_like(logits),
            candidate_logits=logits,
            depth=len(supports),
            geometry=(),
        )


#: Four pixels on a 2x2 grid, one per case the FIX contract has to cover.
_REGRESS_PIXEL = (0, 0)
_TIE_PIXEL = (0, 1)
_NEAR_TIE_PIXEL = (1, 0)
_HEALTHY_PIXEL = (1, 1)
#: 1.0 - 5e-7 rounds to a float32 winning margin of ~4.77e-7: strictly
#: positive, and at or below ``MARGIN_TIE_EPSILON``.
_NEAR_TIE_GAP = 5e-7


def _tie_case(*, tie_sensitivity: float, retained_nudge: float = 0.0):
    """A scripted one-row case with tied, near-tied and healthy FIX pixels."""

    base = torch.zeros(1, 3, 2, 2)
    base[0, :, _TIE_PIXEL[0], _TIE_PIXEL[1]] = torch.tensor([1.0, 1.0, 0.0])
    base[0, :, _NEAR_TIE_PIXEL[0], _NEAR_TIE_PIXEL[1]] = torch.tensor(
        [1.0, 1.0 - _NEAR_TIE_GAP, 0.0]
    )
    base[0, :, _HEALTHY_PIXEL[0], _HEALTHY_PIXEL[1]] = torch.tensor([2.0, 0.0, 0.0])

    # Only the REGRESS pixel responds toward the restoration target; the tied
    # and near-tied FIX pixels respond by handing their win to class 1.
    sensitivity = torch.zeros(1, 3, 2, 2)
    sensitivity[0, 2, _REGRESS_PIXEL[0], _REGRESS_PIXEL[1]] = 4.0
    for row, column in (_TIE_PIXEL, _NEAR_TIE_PIXEL):
        sensitivity[0, 1, row, column] = tie_sensitivity

    evidence = torch.zeros(1, 3, 2, 2)
    evidence[0, LOCAL_FIX] = 0.95
    evidence[0, LOCAL_FIX, _REGRESS_PIXEL[0], _REGRESS_PIXEL[1]] = 0.0
    evidence[0, LOCAL_REGRESS, _REGRESS_PIXEL[0], _REGRESS_PIXEL[1]] = 0.95
    evidence[0, LOCAL_UNCHANGED] = 1.0 - evidence[0, LOCAL_FIX] - evidence[0, LOCAL_REGRESS]

    supports = (torch.zeros(1, 2, 2, 2, 2),)
    record = _scripted_record(base, evidence, supports)
    retained = base.clone()
    if retained_nudge:
        retained[0, 1, _TIE_PIXEL[0], _TIE_PIXEL[1]] += retained_nudge
    expert = _ScriptedReplayExpert(base.clone(), supports, sensitivity)
    return expert, record, retained


def _scripted_record(
    base: torch.Tensor, evidence: torch.Tensor, supports: tuple[torch.Tensor, ...]
):
    """One accepted-ordinary record around scripted factual logits."""

    # The restoration target is class 2 everywhere; only the REGRESS pixel
    # carries weight, so that is the only pixel it acts on.
    annotation = torch.zeros_like(base)
    annotation[:, 2] = 1.0
    expert_record = ExpertReplayRecord(
        shared_features=torch.zeros(1, 1, 2, 2),
        annotation_logits=annotation,
        previous_audit_evidence=None,
        turn_index=1,
        iteration_index=1,
        depth=1,
        coordinates=supports,
        coordinates_preclamp=tuple(item.clone() for item in supports),
        factual_candidate_logits=base.clone(),
        factual_update_gate=torch.zeros_like(base),
        state_identity="scripted",
        record_kind=RECORD_KIND_ORDINARY,
        audit_conditioning="full",
        offset_mode="structured",
    )
    return _AcceptedOrdinaryRecord(
        indices=torch.arange(base.shape[0]),
        expert_record=expert_record,
        audit_evidence=evidence,
        turn=1,
        accepted=torch.ones(base.shape[0], dtype=torch.bool),
    )


def _fix_pixels(record, config: CandidateCConfig) -> torch.Tensor:
    return record.audit_evidence[:, LOCAL_FIX] >= config.fix_threshold


def test_tied_and_near_tied_fix_pixels_are_protected_not_excluded() -> None:
    """Every thresholded predicted FIX pixel is protected, ties included."""

    config = CandidateCConfig(rho_feature_pixels=0.1, lam=1.0, margin_fraction=0.5)
    expert, record, retained = _tie_case(tie_sensitivity=0.5)
    factual = record.expert_record.factual_candidate_logits
    winner = factual.argmax(dim=1)
    margin = _winning_margin(factual, winner)
    # The fixture really does contain an exact tie and a strictly positive
    # margin at or below the epsilon: neither case is hypothetical.
    assert float(margin[0, _TIE_PIXEL[0], _TIE_PIXEL[1]]) == 0.0
    near_tie = float(margin[0, _NEAR_TIE_PIXEL[0], _NEAR_TIE_PIXEL[1]])
    assert 0.0 < near_tie <= 1e-6
    assert float(margin[0, _HEALTHY_PIXEL[0], _HEALTHY_PIXEL[1]]) == 2.0
    # argmax is deterministic at the tie: the predicted class is class 0.
    assert int(winner[0, _TIE_PIXEL[0], _TIE_PIXEL[1]]) == 0

    result = _run_restitution_solver(
        expert, record, retained, config=config, enforce_fix=True
    )
    # C1 is exact, the row is eligible, and a real proposal was evaluated.
    assert bool(result["c1_passed"].all())
    assert float(result["c1_abs_error"].max()) == 0.0
    assert bool(result["eligible"].all())
    assert result["evals"]["candidate_checks"] == 2

    # Diagnostics count the FULL protected set, and nothing is excluded.
    assert int(result["num_protected"][0]) == int(_fix_pixels(record, config).sum()) == 3
    assert int(result["num_ties_excluded"][0]) == 0
    assert int(result["num_fix_ties"][0]) == 2

    # The proposal genuinely improves the objective and would destroy the
    # tied and near-tied FIX classes, so it is refused for infeasibility --
    # not for non-improvement -- and exactly nothing is transferred.
    assert result["checked_improved"][0] is True
    assert result["checked_feasible"][0] is False
    assert result["violation_after"][0] > 0.0
    assert not bool(result["settled"].any())
    assert result["reason"][0] == "infeasible"
    assert int(torch.count_nonzero(result["innovation"])) == 0
    assert result["selected_displacement"][0] == 0.0


def test_the_refused_step_is_feasible_for_strict_margin_pixels_alone() -> None:
    """Only the tie/near-tie protection refuses it: healthy pixels are fine.

    This pins the change down to the protected SET.  Had protection stayed
    restricted to strictly-positive-margin pixels, the very same proposal would
    have been accepted and the tied FIX classes would have been destroyed.
    """

    config = CandidateCConfig(rho_feature_pixels=0.1, lam=1.0, margin_fraction=0.5)
    expert, record, retained = _tie_case(tie_sensitivity=0.5)
    unconstrained = _run_restitution_solver(
        expert, record, retained, config=config, enforce_fix=False
    )
    # The no_fix control removes ONLY the constraint: the same first proposal
    # is checked, improves, and is therefore accepted.
    assert bool(unconstrained["settled"].all())
    assert unconstrained["reason"][0] is None
    assert int(unconstrained["num_protected"][0]) == 3

    factual = record.expert_record.factual_candidate_logits
    winner = factual.argmax(dim=1)
    margin = _winning_margin(factual, winner)
    protected = _fix_pixels(record, config)
    strict_only = protected & (margin > 1e-6)
    assert int(strict_only.sum()) == 1

    accepted_candidate = factual + unconstrained["innovation"]
    candidate_margin = _winning_margin(accepted_candidate, winner)
    required = float(config.margin_fraction) * margin
    # Under the old, strict-margin-only protected set the accepted proposal is
    # perfectly feasible ...
    assert float((required - candidate_margin)[strict_only].max()) <= 0.0
    assert bool((accepted_candidate.argmax(dim=1) == winner)[strict_only].all())
    # ... while the tied and near-tied predicted FIX classes are destroyed,
    # both in the counterfactual and after the actual gated transfer.
    flipped = protected & ~strict_only
    assert bool((accepted_candidate.argmax(dim=1) != winner)[flipped].all())
    transferred = retained + unconstrained["gate"] * unconstrained["innovation"]
    assert bool((transferred.argmax(dim=1) != winner)[flipped].all())


def test_final_c3_check_catches_a_tie_that_survived_the_counterfactual() -> None:
    """The final check covers the tied pixels too, and it has to.

    The C3 mixture is convex between the FACTUAL output and the candidate, but
    the retained state is only required to match the factual output to within
    the C1 tolerance.  A tied protected pixel can therefore still flip after a
    feasible counterfactual, and the final check is what catches it.
    """

    config = CandidateCConfig(rho_feature_pixels=0.1, lam=1.0, margin_fraction=0.5)
    # The tied pixels do not move under the proposal, so the counterfactual
    # constraint is satisfied; the retained state carries the tie the other way
    # by 2e-6, well inside the 1e-5 replay tolerance.
    expert, record, retained = _tie_case(tie_sensitivity=0.0, retained_nudge=2e-6)
    factual = record.expert_record.factual_candidate_logits
    winner = factual.argmax(dim=1)

    result = _run_restitution_solver(
        expert, record, retained, config=config, enforce_fix=True
    )
    # Not stale, eligible, and the counterfactual really did pass both halves
    # of the FIX constraint.
    assert not bool(result["stale"].any())
    assert bool(result["eligible"].all())
    assert result["checked_improved"][0] is True
    assert result["checked_feasible"][0] is True
    assert result["violation_after"][0] == 0.0
    # The transfer is nevertheless refused, and refused for C3.
    assert not bool(result["settled"].any())
    assert result["reason"][0] == "c3_preservation_failed"
    assert int(torch.count_nonzero(result["innovation"])) == 0
    assert result["selected_displacement"][0] == 0.0

    # The no_fix control accepts the same step and exposes the flip the
    # constraint prevents: the tied predicted FIX class changes.
    unconstrained = _run_restitution_solver(
        expert, record, retained, config=config, enforce_fix=False
    )
    assert bool(unconstrained["settled"].all())
    transferred = retained + unconstrained["gate"] * unconstrained["innovation"]
    assert int(transferred.argmax(dim=1)[0, _TIE_PIXEL[0], _TIE_PIXEL[1]]) == 1
    assert int(winner[0, _TIE_PIXEL[0], _TIE_PIXEL[1]]) == 0


def _exact_flip_case():
    """A protected tie whose candidate flips class at an EXACT logit tie.

    The fractional-margin test cannot see this case: the factual margin is
    zero, so the required margin is zero, and the candidate's winning margin is
    also exactly zero.  Only the explicit class-identity test refuses it.
    """

    base = torch.zeros(1, 3, 2, 2)
    # winner = class 1 (first maximum), runner-up = class 2, margin = 0.
    base[0, :, _TIE_PIXEL[0], _TIE_PIXEL[1]] = torch.tensor([0.0, 1.0, 1.0])
    base[0, :, _NEAR_TIE_PIXEL[0], _NEAR_TIE_PIXEL[1]] = torch.tensor([2.0, 0.0, 0.0])
    base[0, :, _HEALTHY_PIXEL[0], _HEALTHY_PIXEL[1]] = torch.tensor([2.0, 0.0, 0.0])

    sensitivity = torch.zeros(1, 3, 2, 2)
    sensitivity[0, 2, _REGRESS_PIXEL[0], _REGRESS_PIXEL[1]] = 4.0

    # As soon as the proposal moves, class 0 rises to exactly the tied value:
    # the winning margin against class 1 stays exactly 0.0, but argmax now
    # returns class 0 instead of class 1.
    moved_logits = base.clone()
    moved_logits[0, 0, _TIE_PIXEL[0], _TIE_PIXEL[1]] = 1.0
    moved_mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    moved_mask[0, 0, _TIE_PIXEL[0], _TIE_PIXEL[1]] = True

    evidence = torch.zeros(1, 3, 2, 2)
    evidence[0, LOCAL_FIX] = 0.95
    evidence[0, LOCAL_FIX, _REGRESS_PIXEL[0], _REGRESS_PIXEL[1]] = 0.0
    evidence[0, LOCAL_REGRESS, _REGRESS_PIXEL[0], _REGRESS_PIXEL[1]] = 0.95
    evidence[0, LOCAL_UNCHANGED] = 1.0 - evidence[0, LOCAL_FIX] - evidence[0, LOCAL_REGRESS]

    supports = (torch.zeros(1, 2, 2, 2, 2),)
    record = _scripted_record(base, evidence, supports)
    expert = _ScriptedReplayExpert(
        base.clone(),
        supports,
        sensitivity,
        moved_logits=moved_logits,
        moved_mask=moved_mask,
    )
    return expert, record, base.clone()


def test_counterfactual_class_check_refuses_an_exactly_tied_flip() -> None:
    """The class-identity half of the counterfactual constraint, on its own."""

    config = CandidateCConfig(
        rho_feature_pixels=0.1, lam=1.0, margin_fraction=0.5, max_backtracks=1
    )
    expert, record, retained = _exact_flip_case()
    factual = record.expert_record.factual_candidate_logits
    winner = factual.argmax(dim=1)
    margin = _winning_margin(factual, winner)
    assert int(winner[0, _TIE_PIXEL[0], _TIE_PIXEL[1]]) == 1
    assert float(margin[0, _TIE_PIXEL[0], _TIE_PIXEL[1]]) == 0.0

    result = _run_restitution_solver(
        expert, record, retained, config=config, enforce_fix=True
    )
    assert bool(result["c1_passed"].all())
    assert result["evals"]["candidate_checks"] == 1
    assert result["checked_improved"][0] is True
    # The margin test alone measures NO violation -- and the step is still
    # refused, because the predicted class changed.
    assert result["violation_after"][0] == 0.0
    assert result["checked_feasible"][0] is False
    assert result["reason"][0] == "infeasible"
    assert not bool(result["settled"].any())
    assert int(torch.count_nonzero(result["innovation"])) == 0

    # The no_fix control accepts the same step and exposes the class change.
    unconstrained = _run_restitution_solver(
        expert, record, retained, config=config, enforce_fix=False
    )
    assert bool(unconstrained["settled"].all())
    flipped = factual + unconstrained["innovation"]
    assert int(flipped.argmax(dim=1)[0, _TIE_PIXEL[0], _TIE_PIXEL[1]]) == 0
    assert float(_winning_margin(flipped, winner)[0, _TIE_PIXEL[0], _TIE_PIXEL[1]]) == 0.0
