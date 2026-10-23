"""Bounded manufactured CPU probes; no patient/pretrained/GPU evidence."""
from pathlib import Path
import copy
import hashlib
import json
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from scipy.optimize import minimize
import torch
from torch import nn
from torch.nn import functional as F
from self_audit.control import (Allocation, audit_weights, controlled_head_step, make_candidates,
                                response_moments, scalar_allocation, solve_allocation)
import self_audit.decoder_control as decoder_control
from self_audit.decoder_control import decoder_transaction
from self_audit.model import NativeStudent, TinyEncoder


def normalized_scalar_solve(a, b, ridge):
    """Probe-only positive rescaling; same ridge and same zero-budget feasible set."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), ridge)
    result = minimize(lambda x: float((-.1e1*b*x[0]+.5*a*x[0]**2).mean()
                                      +.5*ridge*x[0]**2)/scale,
                      [0.], jac=lambda x: np.array([(a.mean()*x[0]-b.mean()+ridge*x[0])/scale]),
                      method="SLSQP", bounds=[(0., 1.)],
                      constraints=[dict(type="ineq", fun=lambda x: (b*x[0]-.5*a*x[0]**2)/scale,
                                        jac=lambda x: ((b-a*x[0])/scale)[:, None])],
                      options=dict(ftol=1e-12, maxiter=300))
    return dict(alpha=float(result.x[0]), success=bool(result.success), scale=scale)


def scalar_scale_sweep():
    rows = []
    for response in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        # Consistent least-squares moments from residual r=1, response v=response.
        a, b, e, ridge = response**2, response, .5, 1e-6
        numerical = solve_allocation([[[a]]], [[b]], [e], ridge=ridge)
        exact = scalar_allocation([a], [b], ridge=ridge)
        value = lambda x: e-b*x+.5*(a+ridge)*x*x
        rows.append(dict(response=response, numerical=float(numerical.alpha[0]), exact=exact,
                         status=numerical.status, objective_gap=value(numerical.alpha[0])-value(exact),
                         exact_improvement=e-value(exact),
                         equivalent_normalized_probe=normalized_scalar_solve([a], [b], ridge)))
    return rows


def head_scale_sweep():
    rows = []
    for eta in [1e-3, 1e-4, 1e-5, 1e-6]:
        head = nn.Conv2d(1, 4, 1).double()
        nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        features = torch.ones(1, 1, 2, 2, dtype=torch.float64)
        labels = torch.ones(1, 2, 2, dtype=torch.long)
        q = F.one_hot(labels, 4).permute(0, 3, 1, 2).double()
        directions, ids, _ = make_candidates(head, features, q, .7, eta)
        p = head(features).softmax(1)
        dz = torch.stack([F.conv2d(features, *d) for d in directions])
        dp = p[None] * (dz-(p[None]*dz).sum(2, keepdim=True))
        w, t, _ = audit_weights(labels)
        a, b, e = response_moments(p, dp, w, t)
        exact = scalar_allocation(a[:, 0, 0], b[:, 0], ridge=1e-6)
        report = controlled_head_step(head, features, labels, directions, ids, ridge=1e-6)
        rows.append(dict(pseudo_lr=eta, exact_scalar_alpha=exact,
                         maximum_alignment=float(np.max(b)), report=report,
                         equivalent_normalized_probe=normalized_scalar_solve(a[:, 0, 0], b[:, 0], 1e-6)))
    return rows


def decoder_plain_equivalence_and_cost():
    torch.manual_seed(2301)
    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32)).eval()
    simple = copy.deepcopy(model)
    strong, audit = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    labels = torch.arange(32*32).reshape(1, 32, 32) % 4
    q = F.one_hot(labels, 4).permute(0, 3, 1, 2).float()
    selected = lambda m: [p for n, p in m.named_parameters()
                          if not n.startswith(("encoder.", "input_adapter."))]
    calls = [0]
    hook = model.encoder.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0]+1))
    start = time.perf_counter()
    _, report = decoder_transaction(model, strong, q, audit, labels, method="ema_block",
                                    threshold=.7, step_size=.001)
    transaction_seconds = time.perf_counter()-start
    hook.remove()
    start = time.perf_counter()
    loss = F.cross_entropy(simple(strong), labels)
    grads = torch.autograd.grad(loss, selected(simple))
    with torch.no_grad():
        for p, g in zip(selected(simple), grads):
            p.add_(g, alpha=-.001)
    simple_seconds = time.perf_counter()-start
    max_error = max(float((a-b).detach().abs().max()) for a, b in zip(model.parameters(), simple.parameters()))
    assert max_error < 1e-6
    return dict(production_encoder_forward_calls=calls[0], production_jvp_count=report["jvp_count"],
                production_trials=len(report["trials"]), production_seconds=transaction_seconds,
                direct_sgd_seconds=simple_seconds, maximum_parameter_error=max_error,
                caveat="One tiny CPU fixture; direct SGD omits audit diagnostics, no GPU speedup claim.")


def head_rejected_trial_atomicity():
    head = nn.Conv2d(4, 4, 1)
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    features = torch.eye(4).reshape(1, 4, 2, 2)
    labels = torch.arange(4).reshape(1, 2, 2)
    optimizer = torch.optim.AdamW(head.parameters())
    F.cross_entropy(head(features), labels).backward(); optimizer.step()
    state = copy.deepcopy(head.state_dict()); optstate = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    direction = (torch.zeros_like(head.weight), torch.tensor([1., 0., 0., 0.]))
    report = controlled_head_step(head, features, labels, [direction], ["harmful_bg"], method="joint_guard")
    model_same = all(torch.equal(v, state[k]) for k, v in head.state_dict().items())
    opt_same = all(torch.equal(v, optstate["state"][k][n]) if isinstance(v, torch.Tensor)
                   else v == optstate["state"][k][n]
                   for k, s in optimizer.state_dict()["state"].items() for n, v in s.items())
    rng_same = torch.equal(rng, torch.get_rng_state())
    assert report["scale"] == 0 and model_same and opt_same and rng_same
    return dict(report=report, parameters_unchanged=model_same, optimizer_unchanged=opt_same,
                torch_rng_unchanged=rng_same,
                caveat="Zero allocation path; this probe does not force failed nonzero trial evaluations.")


def decoder_forced_trial_rejection():
    torch.manual_seed(2302)
    model = NativeStudent(TinyEncoder(), (8, 16, 24, 32)).eval()
    nn.init.zeros_(model.head.weight); nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.AdamW(model.parameters())
    strong, audit = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    labels = torch.ones(1, 32, 32, dtype=torch.long)
    q = torch.zeros(1, 4, 32, 32); q[:, 0] = 1.
    state = copy.deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
    original = decoder_control.solve_allocation
    def forced(a, b, e, **kwargs):
        return Allocation(np.ones(b.shape[1]), "FAULT_INJECTION_ALPHA_ONE", float(np.mean(e)), e)
    decoder_control.solve_allocation = forced
    try:
        _, report = decoder_transaction(model, strong, q, audit, labels, method="joint_guard",
                                        threshold=.7, step_size=1., backtracks=3)
    finally:
        decoder_control.solve_allocation = original
    state_same = all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    rng_same = torch.equal(rng, torch.get_rng_state())
    assert report["scale"] == 0 and len(report["trials"]) == 3 and state_same and rng_same
    assert not optimizer.state
    return dict(report=report, parameters_and_buffers_unchanged=state_same,
                optimizer_state_untouched=not bool(optimizer.state), torch_rng_unchanged=rng_same,
                caveat="Fault injection forces a harmful nonzero local allocation; verifies actual trial rejection only.")


if __name__ == "__main__":
    torch.set_num_threads(2)
    result = dict(scope="EXECUTED_MANUFACTURED_CPU_ONLY", python=platform.python_version(),
                  torch=torch.__version__, scipy=scipy.__version__, numpy=np.__version__,
                  source_commit="637f38ced217fe1db2347a87924695533056d8f0",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scalar_scale_sweep=scalar_scale_sweep(), head_scale_sweep=head_scale_sweep(),
                  decoder_plain=decoder_plain_equivalence_and_cost(),
                  rejection_atomicity=head_rejected_trial_atomicity(),
                  decoder_forced_rejection=decoder_forced_trial_rejection())
    output = ROOT / "reports/research/20261023_astra_ultra/evidence/A_source_probes.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False))
