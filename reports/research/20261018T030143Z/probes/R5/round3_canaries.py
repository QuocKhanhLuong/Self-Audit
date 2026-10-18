"""Manufactured CPU checks for R3's review; no production/patient imports."""
import copy
import hashlib
import json
import platform
from pathlib import Path

import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def kl(q, p):
    return (q * (q.log() - p.log())).sum().item()


def protection_scope():
    # Feature-only update changes a shared encoder, while the semantic head is fixed.
    encoder = torch.tensor(1.0, requires_grad=True)
    head = torch.tensor(1.0)
    q = torch.stack([torch.sigmoid(encoder.detach()), 1 - torch.sigmoid(encoder.detach())])
    before = kl(q, q)
    feature_loss = 0.5 * (encoder - 2.0).square()
    feature_loss.backward()
    after_encoder = encoder.detach() - 0.1 * encoder.grad
    p = torch.stack([torch.sigmoid(head * after_encoder), 1 - torch.sigmoid(head * after_encoder)])
    after = kl(q, p)
    assert before == 0 and after > before
    # Generic full-Q support does not fabricate a named RV lineage.
    diffuse_q = [0.7, 0.1, 0.1, 0.1]
    named_rv_lineages = []
    rv_group_created = bool(named_rv_lineages)
    assert diffuse_q[1] > 0 and not rv_group_created
    return {
        "batch_semantics": "INACTIVE", "fixed_positive_support_proxy_panel": "CHECKED",
        "named_RV_without_lineage": "UNPROTECTED", "zero_support_panel_group": "INACTIVE",
        "panel_KL_before": before, "panel_KL_after_feature_only_step": after,
        "batch_active_only_acceptance": "vacuous", "fixed_panel_hard_guard_accepts": after <= before,
        "diffuse_Q": diffuse_q, "named_RV_group_created": rv_group_created,
    }


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.4))
        self.register_buffer("running", torch.tensor(0.0))

    def forward(self, x, rng):
        noise = torch.rand((), generator=rng) * 0.1
        if self.training:
            self.running.add_(x.detach())
        return self.theta * (x + noise)


class Runtime:
    def __init__(self):
        self.model = Toy()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=0.04, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=0.9)
        self.aug_rng = torch.Generator().manual_seed(101)
        self.model_rng = torch.Generator().manual_seed(202)
        self.cursor = 0
        self.attempts = 0
        self.accepted = 0

    def consume_exposure(self):
        x = torch.tensor(1.0 + 0.2 * self.cursor) + torch.rand((), generator=self.aug_rng) * 0.1
        self.cursor += 1
        self.attempts += 1
        return x

    def snapshot_update(self):
        return copy.deepcopy({
            "model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(), "model_rng": self.model_rng.get_state(),
            "training": self.model.training, "accepted": self.accepted,
        })

    def restore_update(self, state):
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.model_rng.set_state(state["model_rng"])
        self.model.train(state["training"])
        self.accepted = state["accepted"]
        self.opt.zero_grad(set_to_none=True)

    def update(self, x):
        self.opt.zero_grad(set_to_none=True)
        loss = (self.model(x, self.model_rng) - 0.8).square()
        loss.backward()
        old = self.model.theta.detach().clone()
        self.opt.step()
        assert self.model.theta.detach() != old
        self.scheduler.step()
        self.accepted += 1
        self.opt.zero_grad(set_to_none=True)


def max_difference(a, b):
    if torch.is_tensor(a):
        return (a.to(torch.float64) - b.to(torch.float64)).abs().max().item()
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max([max_difference(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max([max_difference(x, y) for x, y in zip(a, b)] or [0.0])
    assert a == b, (a, b)
    return 0.0


def transaction_canary():
    trial, skip = Runtime(), Runtime()
    for rt in [trial, skip]:
        rt.update(rt.consume_exposure())
    # Exposure and augmentation are consumed in both arms before the transaction.
    x_trial, x_skip = trial.consume_exposure(), skip.consume_exposure()
    assert torch.equal(x_trial, x_skip)
    saved = trial.snapshot_update()
    trial.update(x_trial)
    mutated_buffer = trial.model.running.item() != saved["model"]["running"].item()
    mutated_rng = not torch.equal(trial.model_rng.get_state(), saved["model_rng"])
    trial.restore_update(saved)
    after_reject_gap = max_difference(trial.snapshot_update(), skip.snapshot_update())
    assert after_reject_gap == 0
    assert torch.equal(trial.aug_rng.get_state(), skip.aug_rng.get_state())
    assert (trial.cursor, trial.attempts, trial.accepted) == (2, 2, 1)
    for rt in [trial, skip]:
        rt.update(rt.consume_exposure())
    subsequent_gap = max_difference(trial.snapshot_update(), skip.snapshot_update())
    assert subsequent_gap == 0
    assert torch.equal(trial.aug_rng.get_state(), skip.aug_rng.get_state())

    # A supposedly deterministic acceptance closure must not consume model RNG
    # or mutate buffers. Intentionally bad closures expose both violations.
    saved = trial.snapshot_update()
    with torch.no_grad():
        trial.model(torch.tensor(2.0), trial.model_rng)
    buffer_canary_detected = not torch.equal(trial.model.running, saved["model"]["running"])
    stochastic_canary_detected = not torch.equal(trial.model_rng.get_state(), saved["model_rng"])
    trial.restore_update(saved)
    with torch.no_grad():
        old_mode = trial.model.training
        trial.model.eval()
        deterministic_panel_value = torch.sigmoid(trial.model.theta * torch.tensor(2.0)).item()
        trial.model.train(old_mode)
    deterministic_gap = max_difference(trial.snapshot_update(), saved)
    assert mutated_buffer and mutated_rng and buffer_canary_detected and stochastic_canary_detected
    assert deterministic_gap == 0
    return {
        "rejected_stage_mutated_buffer": mutated_buffer,
        "rejected_stage_consumed_model_rng": mutated_rng,
        "after_rejection_vs_same_exposure_skip_control_max_gap": after_reject_gap,
        "subsequent_accepted_update_max_gap": subsequent_gap,
        "cursor": trial.cursor, "attempts": trial.attempts, "accepted_positive_updates": trial.accepted,
        "scheduler_clock": "accepted_positive_updates",
        "bad_closure_buffer_detected": buffer_canary_detected,
        "bad_closure_rng_detected": stochastic_canary_detected,
        "deterministic_closure_update_state_max_gap": deterministic_gap,
        "deterministic_panel_value": deterministic_panel_value,
        "verified_state": ["parameters", "AdamW moments/step/groups", "registered buffer", "explicit CPU model RNG", "StepLR state", "mode", "grad=None policy", "retained exposure/augmentation RNG"],
        "not_verified": ["production implementation", "GradScaler/AMP", "CUDA RNG", "worker prefetch/dataloader queues", "distributed state", "arbitrary external mutable state"],
    }


def outside_panel_and_stall():
    theta0 = torch.tensor([0.0, 0.0])
    theta1 = torch.tensor([0.0, 3.0])
    p0, p1 = torch.sigmoid(theta0), torch.sigmoid(theta1)
    panel_kl0 = kl(torch.tensor([0.5, 0.5]), torch.stack([p0[0], 1 - p0[0]]))
    panel_kl1 = kl(torch.tensor([0.5, 0.5]), torch.stack([p1[0], 1 - p1[0]]))
    assert panel_kl0 == panel_kl1 == 0 and p1[1] != p0[1]
    attempts, displacements = 16, [0.0] * 16
    stall = attempts == 16 and all(d == 0 for d in displacements)
    assert stall
    return {
        "panel_KL_before": panel_kl0, "panel_KL_after": panel_kl1,
        "guard_accepts_outside_panel_change": True,
        "outside_probability_before": p0[1].item(), "outside_probability_after": p1[1].item(),
        "stall_fixture_attempt_budget": attempts, "accepted_nonzero_displacements": 0,
        "stop": stall, "automatic_relaxations": 0,
        "scope": "finite-panel retention only; outside anatomical validity unobserved",
    }


if __name__ == "__main__":
    report = {
        "status": "MANUFACTURED_CPU_EXECUTED", "python": platform.python_version(),
        "torch": torch.__version__, "dtype": "float64", "production_imports": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protection_scope": protection_scope(), "transaction": transaction_canary(),
        "outside_panel_and_stall": outside_panel_and_stall(),
    }
    output = Path(__file__).with_name("round3_canaries.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
