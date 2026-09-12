"""Tests for running the existing gated joint objective from the first epoch.

The user chose to train the Annotator and the Auditor jointly, with the
Auditor's accept/reject feedback live from the first step, accepting that the
Auditor starts randomly initialised.  No new objective, loss, or acceptance
policy is introduced by that choice: the schedule places the already-approved
``retained_final_annotation`` / ``threshold_gate`` interval at epoch 0.

**How the accept and reject cases are reached.**  Not by a seed, and not by
overriding any decision.  Each test pins the Auditor's *final scoring layer* to
a constant (`_pin_gate`): the last `Linear` of `global_head` gets a zero weight
matrix and a bias of exactly `+margin` or `-margin`, so `delta_q` is that
constant for every row on every backend.  The real
`SelfAuditNet.infer` gate -- `delta_q > tau_accept`, strict -- then does the
deciding on its own.  Nothing in the decision code is patched, no boolean is
forced, and the result does not depend on random sign, PyTorch version, timm
version, or backend.

What is pinned and what is not:

* pinned: two Auditor scoring **parameters** (a test-only initialisation).
* untouched: the gate expression, `tau`, the halt rule, the annotation path,
  the audit loss, and every other Auditor parameter -- the local head still
  produces real logits and still receives real gradient.

All tests run on a tiny CPU network.  No GPU, no dataset download, no real
training run.  Observed environment: torch 2.14.0, numpy 2.4.6.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from self_audit.models.self_audit_net import SelfAuditNet
from self_audit.training.finetune_joint import compute_joint_losses
from self_audit.training.schedule import Schedule, ScheduleInterval
from self_audit.training.unified_config import load_unified_config
from self_audit.training.unified_trainer import SELECTION_METRIC_KEY, UnifiedTrainer


CONFIG_PATH = "configs/self_audit_joint_from_start.yaml"

TURNS = 3

#: Distance from ``tau`` that the pinned scoring bias sits at.  Any positive
#: value works; the gate compares strictly against ``tau``.
GATE_MARGIN = 0.05


def _pin_gate(model: SelfAuditNet, *, accept: bool, margin: float = GATE_MARGIN) -> float:
    """Pin the Auditor's scalar score to a constant, deterministically.

    Test-only *initialisation*, not a decision override: the final `Linear` of
    `auditor.global_head` is set to a zero weight matrix and a constant bias, so
    its output is that bias for every row and every batch.  The real gate in
    `SelfAuditNet.infer` (`delta_q > tau_accept`) is what then accepts or
    rejects; this function never touches it.

    Returns the pinned score so a test can assert the gate saw what was pinned.
    """

    head = model.auditor.global_head[-1]
    assert isinstance(head, torch.nn.Linear) and head.out_features == 1
    score = float(margin if accept else -margin)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(score)
    return score


def _tiny_net(seed: int = 0, *, window_mode: str = "current", max_turns: int = TURNS) -> SelfAuditNet:
    torch.manual_seed(seed)
    return SelfAuditNet(
        pretrained_encoder=False,
        encoder_allow_fallback=True,
        shared_channels=16,
        window_k=4,
        max_turns=max_turns,
        window_mode=window_mode,
    )


def _batch(batch_size: int = 2, size: int = 32, seed: int = 11) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "image": torch.randn(batch_size, 3, size, size),
        "mask": torch.randint(0, 4, (batch_size, size, size)),
    }


def _grad_sum(model: torch.nn.Module, name_match) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if not name_match(name):
            continue
        if param.grad is not None:
            total += float(param.grad.detach().abs().sum())
    return total


def _is_auditor(name: str) -> bool:
    return name.startswith("auditor")


class _TinyDataset(Dataset):
    def __init__(self, count: int = 4, size: int = 32) -> None:
        torch.manual_seed(5)
        self.images = torch.randn(count, 3, size, size)
        self.masks = torch.randint(0, 4, (count, size, size))

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"image": self.images[index], "mask": self.masks[index]}


# ============================================================================
# 1. The zero-based single-interval schedule actually works
# ============================================================================


def test_joint_from_start_schedule_is_valid_and_gated_at_epoch_zero() -> None:
    config = load_unified_config(CONFIG_PATH)
    schedule = config.training.schedule

    assert schedule.total_epochs == 130
    assert len(schedule.intervals) == 1
    interval = schedule.get_interval(0)
    assert interval.start_epoch == 0 and interval.end_epoch == 130
    assert interval.objective == "retained_final_annotation"
    assert interval.rollout == "threshold_gate"
    assert interval.transition_population == "active_attempted"
    assert interval.trainable == "all"
    # The approved contract for this objective requires a reset at its own
    # boundary; at epoch 0 that boundary is the start of training.
    assert interval.reset_optimizer is True
    assert schedule.is_interval_start(0) is True
    assert schedule.is_reset_boundary(0) is True
    assert schedule.get_interval_index(129) == 0

    # The same interval, rebuilt from its dict, still satisfies the unchanged
    # cross-field contract: nothing was relaxed to let it start at epoch 0.
    assert ScheduleInterval.from_dict(interval.to_dict()) == interval
    assert Schedule.from_dict(schedule.to_dict()) == schedule
    assert config.training.rollout.tau == 0.0


def test_trainer_builds_all_three_parameter_groups_at_epoch_zero() -> None:
    config = load_unified_config(CONFIG_PATH)
    model = _tiny_net()
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)
    trainer.amp_enabled = False
    interval = config.training.schedule.get_interval(0)

    trainer.setup_interval_optimizer_and_scheduler(interval, num_batches=2)
    names = {group.get("name") for group in trainer.optimizer.param_groups}
    assert names == {"encoder", "annotation_heads", "auditor"}
    assert all(param.requires_grad for _, param in model.named_parameters())


# ============================================================================
# 2. The pinned score drives the real, unmodified gate
# ============================================================================


@pytest.mark.parametrize("accept", [True, False])
def test_pinned_score_drives_the_real_unmodified_gate(accept: bool) -> None:
    model = _tiny_net()
    score = _pin_gate(model, accept=accept)
    batch = _batch()

    output = model.infer(batch["image"], mode="self_audit", tau_accept=0.0, t_max=TURNS)

    # The gate really did see the pinned score, on every scored transition.
    for audit in output["audits"]:
        observed = audit["delta_q"].detach().reshape(-1)
        assert torch.allclose(observed, torch.full_like(observed, score))

    batch_size = int(batch["image"].shape[0])
    if accept:
        # Strictly above tau on every turn, so every row runs to the cap.
        assert int(output["accepted_count"].sum()) == batch_size * TURNS
        assert int(output["num_attempted_turns"].min()) == TURNS
    else:
        # Strictly at-or-below tau, so nothing is accepted and every row halts
        # after its first attempted turn.  The halt rule is unmodified.
        assert int(output["accepted_count"].sum()) == 0
        assert int(output["num_attempted_turns"].max()) == 1
        assert int(output["halt_turn"].max()) == 0


# ============================================================================
# 3. Accepted case: a unit gradient step moves both families
# ============================================================================


def test_accepted_unit_gradient_step_trains_both_families() -> None:
    """One **unit** gradient step, not the profile's first scheduled step.

    The optimizer here uses a plain unit learning rate and ``weight_decay=0``,
    so a parameter that moves moved because of its gradient and not because of
    decay.  The real run's first step is far smaller: the warmup-cosine
    schedule returns a factor of ``max(step / warmup_steps, 1e-8)``, so the
    profile's very first optimizer step applies ``base_lr * 1e-8``.  This test
    therefore says *that a gradient exists and is applied*, never that the
    first real step moves weights by this much.
    """

    model = _tiny_net()
    _pin_gate(model, accept=True)
    model.train()
    batch = _batch()

    total, details = compute_joint_losses(model, batch, tau_accept=0.0, t_max=TURNS)
    assert int(details["output"]["accepted_count"].sum()) > 0
    assert details["transition_count"] == TURNS

    # Gradients are asserted BEFORE any optimizer step, and per term, so the
    # claim is about which loss produces which gradient.
    model.zero_grad(set_to_none=True)
    details["annotation_loss_tensor"].backward(retain_graph=True)
    annotation_expert_from_annotation = _grad_sum(model, lambda n: n.startswith("annotation_expert"))
    auditor_from_annotation = _grad_sum(model, _is_auditor)
    assert annotation_expert_from_annotation > 0.0
    assert auditor_from_annotation == 0.0

    model.zero_grad(set_to_none=True)
    details["audit_loss_tensor"].backward(retain_graph=True)
    auditor_from_audit = _grad_sum(model, _is_auditor)
    annotator_from_audit = _grad_sum(model, lambda n: not _is_auditor(n))
    assert auditor_from_audit > 0.0
    assert annotator_from_audit == 0.0

    # Now the combined step.  weight_decay=0 means movement cannot come from
    # decay on an otherwise-gradientless parameter.
    watched = {
        "encoder": next(p for n, p in model.named_parameters() if "encoder" in n and p.requires_grad),
        "expert": next(p for n, p in model.named_parameters() if n.startswith("annotation_expert")),
        "auditor_local": model.auditor.local_head.weight,
    }
    before = {key: param.detach().clone() for key, param in watched.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for group in optimizer.param_groups:
        assert group["weight_decay"] == 0.0
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    for key, param in watched.items():
        assert param.grad is not None and float(param.grad.abs().sum()) > 0.0, f"{key} has no gradient"
    optimizer.step()

    for key, param in watched.items():
        assert not torch.equal(before[key], param.detach()), f"{key} did not move"


# ============================================================================
# 4. All-reject case: the explicit limitation
# ============================================================================


def test_all_reject_trains_the_auditor_but_only_a0_and_the_encoder() -> None:
    model = _tiny_net()
    _pin_gate(model, accept=False)
    model.train()
    batch = _batch()

    _, details = compute_joint_losses(model, batch, tau_accept=0.0, t_max=TURNS)
    output = details["output"]
    assert int(output["accepted_count"].sum()) == 0
    # A rejected row halts: it attempts exactly one turn and no more.
    assert int(output["num_attempted_turns"].max()) == 1
    assert details["transition_count"] == 1

    # The Auditor still learns from the rejected transition.
    model.zero_grad(set_to_none=True)
    details["audit_loss_tensor"].backward(retain_graph=True)
    assert _grad_sum(model, _is_auditor) > 0.0
    assert _grad_sum(model, lambda n: not _is_auditor(n)) == 0.0

    # The annotation term, however, reaches only the retained A0 state: the
    # rejected candidate is discarded from the graph, so the refinement expert
    # receives nothing.  This is the accepted limitation of gated joint
    # training from a random Auditor, stated here rather than hidden.
    model.zero_grad(set_to_none=True)
    details["annotation_loss_tensor"].backward()
    assert _grad_sum(model, lambda n: n.startswith("annotation_expert")) == 0.0
    assert _grad_sum(model, lambda n: n.startswith("initial_head")) > 0.0
    assert _grad_sum(model, lambda n: "encoder" in n and not _is_auditor(n)) > 0.0
    assert _grad_sum(model, _is_auditor) == 0.0


# ============================================================================
# 5. Runtime invariants that must survive starting joint at epoch 0
# ============================================================================


def test_auditor_inputs_are_detached_and_ground_truth_never_enters_the_model() -> None:
    model = _tiny_net()
    _pin_gate(model, accept=True)
    model.train()
    batch = _batch()

    captured: list[tuple[Any, ...]] = []
    real_auditor_forward = model.auditor.forward

    def capturing(*args: Any, **kwargs: Any) -> Any:
        captured.append((args, kwargs))
        return real_auditor_forward(*args, **kwargs)

    model.auditor.forward = capturing  # type: ignore[method-assign]

    infer_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real_infer = model.infer

    def recording_infer(*args: Any, **kwargs: Any) -> Any:
        infer_calls.append((args, kwargs))
        return real_infer(*args, **kwargs)

    model.infer = recording_infer  # type: ignore[method-assign]

    compute_joint_losses(model, batch, tau_accept=0.0, t_max=TURNS)

    assert captured, "the gated rollout must consult the Auditor"
    for args, kwargs in captured:
        for value in list(args) + list(kwargs.values()):
            if torch.is_tensor(value):
                assert not value.requires_grad, "every Auditor input must be a detached leaf"

    # Ground truth is never handed to the model: the rollout is called with the
    # image only, and no argument is the mask.
    assert len(infer_calls) == 1
    args, kwargs = infer_calls[0]
    passed = [value for value in list(args) + list(kwargs.values()) if torch.is_tensor(value)]
    assert len(passed) == 1
    assert passed[0].shape == batch["image"].shape
    for value in passed:
        assert value.shape != batch["mask"].shape


@pytest.mark.parametrize("exposure", [False, True])
def test_joint_runs_one_rollout_whatever_the_auxiliary_exposure_flag_says(exposure: bool) -> None:
    """The joint objective ignores ``predicted_history_exposure`` entirely.

    That flag arms an *optional second* rollout of the same evidence for the
    staged objectives.  The gated joint objective already consumes accepted
    predicted Auditor feedback inside its own single rollout, so it must run
    exactly one ``infer`` and one ``encode`` per batch under either setting, and
    must never add an auxiliary pass.
    """

    config = load_unified_config(CONFIG_PATH)
    model = _tiny_net()
    _pin_gate(model, accept=True)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)
    trainer.amp_enabled = False
    trainer.predicted_history_settings = lambda: (exposure, 0.1)  # type: ignore[method-assign]
    interval = config.training.schedule.get_interval(0)

    calls = {"infer": 0, "encode": 0}
    real_infer = model.infer
    real_encode = model.encode

    def counting_infer(*args: Any, **kwargs: Any) -> Any:
        calls["infer"] += 1
        return real_infer(*args, **kwargs)

    def counting_encode(*args: Any, **kwargs: Any) -> Any:
        calls["encode"] += 1
        return real_encode(*args, **kwargs)

    model.infer = counting_infer  # type: ignore[method-assign]
    model.encode = counting_encode  # type: ignore[method-assign]

    _, details = trainer.compute_batch_loss(_batch(), interval)

    assert calls["infer"] == 1, "the joint objective must roll out exactly once per batch"
    assert calls["encode"] == 1
    assert details["predicted_history_exposure"] is exposure
    assert details["auxiliary_batches"] == 0
    assert details["auxiliary_annotation_loss"] == 0.0
    assert details["auxiliary_audit_loss"] == 0.0

    # Positive evidence that Auditor feedback was live in that single rollout:
    # the counters come from the joint rollout itself, and it consumed accepted
    # predicted history.
    counters = details["rollout_counters"]
    assert counters["rollout_batches"] == 1
    assert counters["accepted_history_count"] > 0
    assert counters["real_evidence_attempts"] > 0


# ============================================================================
# 6. Candidate-C record precondition, on a real candidate_c model
# ============================================================================


def test_candidate_c_record_is_unavailable_at_turn_zero_and_consumed_only_next_turn() -> None:
    """The record precondition, checked on a real ``window_mode="candidate_c"`` model.

    The assertions are about *availability and timing*, which the diagnostics
    report unconditionally.  Whether the solve turns out feasible or improving
    on this tiny fixture is deliberately NOT asserted -- that would make the
    test depend on the numerical outcome of the solve rather than on the
    precondition under test.
    """

    model = _tiny_net(window_mode="candidate_c", max_turns=2)
    assert model.window_mode == "candidate_c"
    assert model.uses_transition_records is True
    _pin_gate(model, accept=True)

    batch = _batch(batch_size=1)
    output = model.infer(batch["image"], mode="self_audit", tau_accept=0.0, t_max=2)
    rows = output["candidate_c_diagnostics"]

    turn0 = [row for row in rows if row["turn"] == 0]
    turn1 = [row for row in rows if row["turn"] == 1]
    assert turn0 and turn1, "both turns must publish a diagnostics row"

    # Turn 0: no accepted ordinary transition exists yet, so the solver is not
    # attempted at all -- no invocation id, no record turn, and the row names
    # the reason.
    for row in turn0:
        assert row["solver_invocation_id"] is None
        assert row["record_turn"] is None
        assert row["eligible"] is False
        assert row["fallback_reason"] == "no_accepted_history"
        assert row["accepted_path"] == "ordinary"
        assert row["c1_passed"] is None

    # Turn 1: the record built by the accepted ordinary transition at turn 0 is
    # consumed here, and only here.
    for row in turn1:
        assert row["solver_invocation_id"] == "turn1"
        assert row["record_turn"] == 0

    # And no record survives the ``infer`` call that produced it.
    generation_before = model.annotation_expert._replay_generation
    assert model.invalidate_candidate_c_records() == generation_before + 1


def test_candidate_c_records_are_invalidated_around_every_optimizer_step() -> None:
    """The invalidation really bumps the replay generation, on a candidate_c model.

    A record is only valid against the exact weights that produced it, so the
    generation counter must advance at the epoch start, before each batch, and
    after each optimizer step.  This asserts the counter actually moves, not
    merely that a hook was called.
    """

    config = load_unified_config(CONFIG_PATH)
    model = _tiny_net(window_mode="candidate_c", max_turns=2)
    _pin_gate(model, accept=True)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)
    trainer.amp_enabled = False
    interval = config.training.schedule.get_interval(0)

    loader = DataLoader(_TinyDataset(count=4), batch_size=2)
    trainer.setup_interval_optimizer_and_scheduler(interval, num_batches=len(loader))

    generation_before = model.annotation_expert._replay_generation
    train_stats = trainer.train_epoch(interval, loader)
    generation_after = model.annotation_expert._replay_generation

    reported = int(train_stats["rollout/candidate_c_invalidations"])
    # Once at epoch start, once before each of the two batches, and once after
    # each of the two optimizer steps.
    assert reported >= 5
    assert train_stats["epoch_optimizer_steps"] == 2
    # Every reported invalidation is a real generation bump.
    assert generation_after - generation_before == reported


def test_train_epoch_logs_feedback_as_active_and_joint_from_the_first_epoch() -> None:
    config = load_unified_config(CONFIG_PATH)
    model = _tiny_net()
    _pin_gate(model, accept=True)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)
    trainer.amp_enabled = False
    interval = config.training.schedule.get_interval(0)

    loader = DataLoader(_TinyDataset(count=4), batch_size=2)
    trainer.setup_interval_optimizer_and_scheduler(interval, num_batches=len(loader))
    train_stats = trainer.train_epoch(interval, loader)

    assert train_stats["annotation_loss"] > 0.0
    assert train_stats["audit_loss"] > 0.0
    # The gate is live in this stage's own rollout.  The auxiliary exposure
    # flag is a different quantity and its zero must not be read as "no
    # Auditor feedback".
    assert train_stats["rollout/feedback_gates"] == "active"
    assert train_stats["rollout/joint_trainable"] == 1.0
    assert train_stats["rollout/interval_rollout"] == "threshold_gate"
    assert train_stats["rollout/predicted_history_exposure"] == 0.0
    assert train_stats["rollout/accepted_history_count"] > 0
    assert train_stats["rollout/evidence_calibration"] == "nonstationary_training_predicted"

    payload = trainer._build_wandb_payload(0, interval, 0, train_stats, {})
    assert payload["schedule/feedback_gates"] == "active"
    assert payload["schedule/joint_trainable"] == 1.0
    assert payload["schedule/rollout_mode"] == "threshold_gate"


# ============================================================================
# 7. Best selection from epoch 0, but only through gated validation
# ============================================================================


def test_best_selection_opens_at_epoch_zero_only_through_the_gated_metric() -> None:
    config = load_unified_config(CONFIG_PATH)
    interval = config.training.schedule.get_interval(0)

    assert config.checkpoint.best_selection_min_epoch == 0
    assert config.checkpoint.best_metric == SELECTION_METRIC_KEY
    # Selection is gated in the trainer on the interval's rollout, so it is open
    # from epoch 0 here precisely because this interval is the gated one.
    assert interval.rollout == "threshold_gate"

    model = _tiny_net()
    _pin_gate(model, accept=True)
    trainer = UnifiedTrainer(config, model=model, device=torch.device("cpu"), disable_tqdm=True)
    trainer.amp_enabled = False
    trainer.global_epoch = 0
    loader = DataLoader(_TinyDataset(count=4), batch_size=2)

    # A full validation really produces the selection metric at epoch 0, so
    # selection has a measured quantity to act on rather than a fabricated one.
    val_stats = trainer.validate_epoch(interval, loader)
    assert SELECTION_METRIC_KEY in val_stats
    assert val_stats["primary_metric"] == val_stats[SELECTION_METRIC_KEY]


@pytest.mark.parametrize("epoch", [0, 129])
def test_every_epoch_resolves_to_the_single_gated_interval(epoch: int) -> None:
    schedule = load_unified_config(CONFIG_PATH).training.schedule
    interval = schedule.get_interval(epoch)
    assert interval.name == "joint_self_audit"
    assert interval.rollout == "threshold_gate"
    assert schedule.get_interval_index(epoch) == 0
