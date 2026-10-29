"""Joint local allocation and an actual-output guard, not a Dice guarantee.

One canonical proposal per predicted class. Candidate SGD displacements act on
an affine native head; body features are frozen during this transaction. The
supervised body/head optimizers have already stepped and never step in trials.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
import torch
from torch.nn import functional as F


@dataclass
class Allocation:
    alpha: np.ndarray
    status: str
    objective: float
    local_risks: np.ndarray


def solve_allocation(grams, alignment, baseline, *, ridge=1e-6, budget=0.0,
                     constrained=True, source_ids=None):
    """Solve a convex box-QCQP; invalid/failed solves fail closed to zero.

    SLSQP returns a numerical candidate, NOT an exact optimum certificate.
    Feasibility and improvement relative to zero are verified independently.
    """
    A, b, e = (np.asarray(x, dtype=np.float64) for x in (grams, alignment, baseline))
    if A.ndim != 3 or A.shape[1] != A.shape[2] or b.shape != A.shape[:2] or e.shape != A.shape[:1]:
        raise ValueError("expected GxMxM, GxM, G")
    if not all(np.isfinite(x).all() for x in (A, b, e)) or ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("finite moments and positive ridge required")
    eps = np.broadcast_to(np.asarray(budget, dtype=float), e.shape)
    if (eps < 0).any() or not np.isfinite(eps).all() or (e < 0).any():
        raise ValueError("invalid baseline or budgets")
    m = b.shape[1]
    if source_ids is not None and (len(source_ids) != m or len(set(source_ids)) != m):
        raise ValueError("duplicate/missing source IDs: aggregate by class before allocation")
    if m == 0:
        return Allocation(np.zeros(0), "NO_CANDIDATES", float(e.mean()), e.copy())
    if not np.allclose(A, A.swapaxes(1, 2), atol=1e-10) or np.linalg.eigvalsh(A).min() < -1e-9:
        raise ValueError("Gram matrices must be symmetric positive semidefinite")
    def risks(x):
        return e - b @ x + .5 * np.einsum("i,gij,j->g", x, A, x)
    def objective(x):
        return float(risks(x).mean() + .5 * ridge * (x @ x))
    def jac(x):
        return A.mean(0) @ x - b.mean(0) + ridge * x
    constraints = []
    if constrained:
        constraints = [{"type": "ineq", "fun": lambda x: e + eps - risks(x),
                        "jac": lambda x: b - np.einsum("gij,j->gi", A, x)}]
    zero = np.zeros(m)
    result = minimize(objective, zero, jac=jac, method="SLSQP", bounds=[(0., 1.)] * m,
                      constraints=constraints, options={"ftol": 1e-12, "maxiter": 300})
    x = np.clip(result.x, 0., 1.)
    feasible = np.isfinite(x).all() and (not constrained or np.all(risks(x) <= e + eps + 1e-10))
    if not result.success or not feasible or objective(x) > objective(zero) + 1e-12:
        return Allocation(zero, "SOLVER_FALLBACK_ZERO", objective(zero), e.copy())
    return Allocation(x, "NUMERICAL_FEASIBLE", objective(x), risks(x))


def scalar_allocation(a, b, ridge=1e-6, budget=0.0):
    """Exact scalar specialization for consistent squared-response moments."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    eps = np.broadcast_to(np.asarray(budget, float), a.shape)
    if a.shape != b.shape or (a < 0).any() or (eps < 0).any() or ridge <= 0:
        raise ValueError("invalid scalar problem")
    if not all(np.isfinite(x).all() for x in (a, b, eps)):
        raise ValueError("nonfinite scalar problem")
    if np.any((a == 0) & (b != 0)):
        raise ValueError("zero response must have zero alignment")
    upper = 1.0
    for ai, bi, ei in zip(a.flat, b.flat, eps.flat):
        if ai > 0:
            root = np.sqrt(bi * bi + 2 * ai * ei)
            bound = 2 * ei / (root - bi) if bi < 0 else (bi + root) / ai
            upper = min(upper, bound)
    return float(np.clip(b.mean() / (a.mean() + ridge), 0, max(0, upper)))


def audit_weights(labels, classes=4):
    """Foreground one-v-rest Brier groups, including negatives/absent classes.

    Each patient receives equal mass. A present class balances positives and
    negatives 50/50; absent classes test false positives only and are reported.
    IGNORE=255 has exactly zero mass; it is not background.
    """
    if labels.ndim != 3 or not torch.isin(labels, labels.new_tensor([0, 1, 2, 3, 255])).all():
        raise ValueError("expected BHW class IDs or 255")
    valid = labels != 255
    if not valid.flatten(1).any(1).all():
        raise ValueError("an audit patient has no observed pixels")
    weights, targets, positives = [], [], []
    for c in range(1, classes):
        target = labels == c
        w = torch.zeros_like(labels, dtype=torch.float64)
        pos_counts = []
        for n in range(len(labels)):
            pos, neg = target[n] & valid[n], ~target[n] & valid[n]
            np_, nn_ = int(pos.sum()), int(neg.sum())
            pos_counts.append(np_)
            if np_ and nn_:
                w[n, pos] = .5 / np_
                w[n, neg] = .5 / nn_
            else:
                w[n, valid[n]] = 1.0 / int(valid[n].sum())
        weights.append(w / len(labels))
        targets.append(target.to(torch.float64))
        positives.append(pos_counts)
    return torch.stack(weights), torch.stack(targets), positives


def risks(probabilities, weights, targets):
    p = probabilities[:, 1:].permute(1, 0, 2, 3).double()
    return (.5 * weights * (p - targets).square()).flatten(1).sum(1)


def response_moments(p, dp, weights, targets):
    """dp is MxBxCxHxW; reduce on device, transfer only small moments to CPU."""
    m = dp.shape[0]
    r = (targets - p[:, 1:].permute(1, 0, 2, 3).double()).flatten(1)
    w = weights.flatten(1)
    v = dp[:, :, 1:].permute(2, 1, 3, 4, 0).double().reshape(3, -1, m)
    A = torch.einsum("gnm,gn,gnk->gmk", v, w, v)
    b = torch.einsum("gn,gn,gnm->gm", r, w, v)
    e = .5 * (w * r.square()).sum(1)
    return tuple(x.detach().cpu().numpy() for x in (A, b, e))


def make_candidates(head, features, teacher_probs, threshold, step_size, valid_mask=None):
    """Per-class hard pseudo-CE with ONE shared accepted-pixel denominator.

    Summing the returned directions equals one plain head-only SGD pseudo-step.
    There is no per-class renormalization or access to unlabeled reference masks.
    """
    if not 0 <= threshold <= 1 or step_size < 0:
        raise ValueError("invalid proposal policy")
    teacher_probs = teacher_probs.detach()
    conf, y = teacher_probs.max(1)
    valid = conf >= threshold
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or valid_mask.shape != valid.shape:
            raise ValueError("image geometry validity must be boolean BHW")
        valid = valid & valid_mask
    denom = valid.sum()
    if not denom or step_size == 0:
        return [], [], {"accepted_pixels": int(denom), "class_pixels": [0] * 4}
    logits = head(features.detach())
    loss_map = F.cross_entropy(logits, y, reduction="none")
    params = tuple(head.parameters())
    if len(params) != 2 or head.kernel_size != (1, 1):
        raise ValueError("v2 requires affine 1x1 head with weight and bias")
    ds, ids, counts = [], [], []
    for c in range(4):
        mask = valid & (y == c)
        counts.append(int(mask.sum()))
        if mask.any():
            g = torch.autograd.grad((loss_map * mask).sum() / denom, params, retain_graph=True)
            ds.append(tuple(-step_size * x.detach() for x in g))
            ids.append(f"pseudo_class_{c}")
    return ds, ids, {"accepted_pixels": int(denom), "class_pixels": counts}


def controlled_head_step(head, features, labels, directions, source_ids, *, method="joint_guard",
                         ridge=1e-6, budget=0., backtracks=8, tolerance=1e-8):
    """Atomic extra unlabeled update, measured relative to supervised-only head.

    No optimizer or EMA step occurs here. All trial evaluations are functional:
    rejected trials cannot contaminate parameters or optimizer/buffer/RNG state.
    Accepted guard bounds apply ONLY to this observed TRAIN audit measure.
    """
    methods = {"ema_head", "sign", "independent_guard", "joint_no_class",
               "joint_no_guard", "joint_guard"}
    if method not in methods or backtracks < 1 or budget < 0 or tolerance < 0:
        raise ValueError("invalid control configuration")
    if len(directions) != len(source_ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate/missing source IDs")
    weights, targets, positives = audit_weights(labels)
    with torch.no_grad():
        f = features.detach()
        base_w, base_b = head.weight.detach().clone(), head.bias.detach().clone()
        base_logits = F.conv2d(f, base_w, base_b)
        p = base_logits.softmax(1)
        e = risks(p, weights, targets).cpu().numpy()
        report = {"method": method, "source_ids": source_ids, "before": e.tolist(),
                  "positive_pixels": positives, "scope": "observed_TRAIN_one_vs_rest_Brier",
                  "optimizer_steps_in_transaction": 0, "ema_updates_in_transaction": 0}
        if not directions:
            return {**report, "alpha": [], "scale": 0., "after": e.tolist(), "status": "NO_CANDIDATES"}
        for dw, db in directions:
            if dw.shape != base_w.shape or db.shape != base_b.shape or not torch.isfinite(dw).all() or not torch.isfinite(db).all():
                raise ValueError("invalid head displacement")
        dz = torch.stack([F.conv2d(f, dw, db) for dw, db in directions])
        dp = p[None] * (dz - (p[None] * dz).sum(2, keepdim=True))
        A, b, _ = response_moments(p, dp, weights, targets)
        if method == "ema_head":
            alpha, status = np.ones(len(directions)), "PLAIN"
        elif method == "sign":
            alpha, status = (b.mean(0) > 0).astype(float), "AGGREGATE_SIGN"
        elif method == "independent_guard":
            alpha = np.array([scalar_allocation(A[:, j, j], b[:, j], ridge, budget) for j in range(len(directions))])
            status = "INDEPENDENT_SCALARS"
        else:
            solved = solve_allocation(A, b, e, ridge=ridge, budget=budget,
                                      constrained=method != "joint_no_class", source_ids=source_ids)
            alpha, status = solved.alpha, solved.status
        guarded = method in {"joint_guard", "joint_no_class", "independent_guard"}
        trials, scale, chosen_w, chosen_b, after = [], 0., base_w, base_b, e
        delta_w = sum(float(a) * d[0] for a, d in zip(alpha, directions))
        delta_b = sum(float(a) * d[1] for a, d in zip(alpha, directions))
        if np.any(alpha > 0):
            for k in range(backtracks if guarded else 1):
                s = 0.5 ** k
                tw, tb = base_w + s * delta_w, base_b + s * delta_b
                ep = risks(F.conv2d(f, tw, tb).softmax(1), weights, targets).cpu().numpy()
                finite = np.isfinite(ep).all()
                within = np.all(ep <= e + budget + tolerance)
                improves = ep.mean() <= e.mean() + tolerance
                ok = finite and (not guarded or (within and improves))
                trials.append({"scale": s, "risks": ep.tolist(), "accepted": bool(ok)})
                if ok:
                    scale, chosen_w, chosen_b, after = s, tw, tb, ep
                    break
        if scale:
            head.weight.copy_(chosen_w)
            head.bias.copy_(chosen_b)
        return {**report, "alpha": alpha.tolist(), "scale": scale, "after": after.tolist(),
                "status": status, "guarded": guarded, "trials": trials,
                "realized_max_regression": float(np.max(after - e)),
                "linearized_after": (e - b @ (alpha * scale) + .5 * np.einsum("i,gij,j->g", alpha * scale, A, alpha * scale)).tolist()}
