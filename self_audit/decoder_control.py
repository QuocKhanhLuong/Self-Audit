"""Audited decoder displacements using real parameter-to-output JVPs.

Unlike the affine-head fast path, decoder logits are nonlinear in parameters.
Only actual functional trials receive the finite-step audit guarantee. No trial
writes to model parameters, optimizer, EMA or running-statistic buffers.
"""
import numpy as np
import torch
from torch.func import functional_call, jvp
from torch.nn import functional as F
from .control import audit_weights, risks, response_moments, solve_allocation, scalar_allocation


def decoder_transaction(model, strong, teacher_probs, audit_x, labels, *, method,
                        threshold, step_size, ridge=1e-6, budget=0., backtracks=8, tolerance=1e-8, valid_mask=None):
    if method not in {"ema_block", "sign", "independent_guard", "joint_no_class", "joint_no_guard", "joint_guard"}:
        raise ValueError("unknown allocation method")
    if not 0 <= threshold <= 1 or step_size < 0 or budget < 0 or tolerance < 0 or backtracks < 1:
        raise ValueError("invalid transaction policy")
    if model.training:
        raise ValueError("transaction must use deterministic eval mode")
    selected = {n: p for n, p in model.named_parameters() if not n.startswith(("encoder.", "input_adapter."))}
    params = tuple(selected.values()); names = tuple(selected)
    conf, py = teacher_probs.detach().max(1)
    accepted = conf >= threshold
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or valid_mask.shape != accepted.shape:
            raise ValueError("image geometry validity must be boolean BHW")
        accepted = accepted & valid_mask
    denominator = accepted.sum()
    weights, targets, positives = audit_weights(labels)
    with torch.no_grad():
        p = model(audit_x).softmax(1)
        e = risks(p, weights, targets).cpu().numpy()
    info = {"accepted_pixels": int(denominator), "class_pixels": [], "scope": "decoder"}
    report = {"method": method, "before": e.tolist(), "positive_pixels": positives,
              "scope": "observed_TRAIN_one_vs_rest_Brier", "optimizer_steps_in_transaction": 0,
              "ema_updates_in_transaction": 0, "update_scope": "decoder"}
    if not denominator or step_size == 0:
        return info, {**report, "alpha": [], "scale": 0., "after": e.tolist(), "status": "NO_CANDIDATES"}
    logits = model(strong)
    lm = F.cross_entropy(logits, py, reduction="none")
    directions, ids = [], []
    for c in range(4):
        mask = accepted & (py == c); info["class_pixels"].append(int(mask.sum()))
        if mask.any():
            grad = torch.autograd.grad((lm * mask).sum() / denominator, params, retain_graph=True)
            d = tuple(-step_size * g.detach() for g in grad)
            if not all(torch.isfinite(x).all() for x in d):
                raise FloatingPointError("nonfinite decoder direction")
            directions.append(d); ids.append(f"pseudo_class_{c}")
    del logits, lm
    base = tuple(x.detach().clone() for x in params)
    # Buffer copies prevent an incompatible future module from writing live state.
    def predict(values):
        buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        return functional_call(model, (dict(zip(names, values)), buffers), (audit_x,), strict=False).softmax(1)
    with torch.no_grad():
        effects = [jvp(predict, (base,), (direction,))[1].detach() for direction in directions]
        A, b, _ = response_moments(p, torch.stack(effects), weights, targets)
        if method == "ema_block":
            alpha, status = np.ones(len(ids)), "PLAIN_DECODER_SGD"
        elif method == "sign":
            alpha, status = (b.mean(0) > 0).astype(float), "AGGREGATE_SIGN"
        elif method == "independent_guard":
            alpha = np.array([scalar_allocation(A[:, j, j], b[:, j], ridge, budget) for j in range(len(ids))])
            status = "INDEPENDENT_SCALARS"
        else:
            solved = solve_allocation(A, b, e, ridge=ridge, budget=budget,
                                      constrained=method != "joint_no_class", source_ids=ids)
            alpha, status = solved.alpha, solved.status
        guarded = method in {"joint_guard", "joint_no_class", "independent_guard"}
        delta = tuple(sum(float(a) * d[k] for a, d in zip(alpha, directions)) for k in range(len(base)))
        trials, scale, after, chosen = [], 0., e, base
        if np.any(alpha > 0):
            for k in range(backtracks if guarded else 1):
                s = .5**k; proposal = tuple(x + s*d for x, d in zip(base, delta))
                ep = risks(predict(proposal), weights, targets).cpu().numpy()
                ok = np.isfinite(ep).all() and (not guarded or (np.all(ep <= e + budget + tolerance) and ep.mean() <= e.mean()+tolerance))
                trials.append({"scale": s, "risks": ep.tolist(), "accepted": bool(ok)})
                if ok:
                    scale, after, chosen = s, ep, proposal; break
        if scale:
            for parameter, value in zip(params, chosen): parameter.copy_(value)
    return info, {**report, "source_ids": ids, "alpha": alpha.tolist(), "scale": scale,
                 "after": after.tolist(), "status": status, "guarded": guarded, "trials": trials,
                 "jvp_count": len(directions), "realized_max_regression": float(np.max(after-e)),
                 "linearized_after": (e-b@(alpha*scale)+.5*np.einsum("i,gij,j->g",alpha*scale,A,alpha*scale)).tolist()}
