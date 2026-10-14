"""Read-only, label-free diagnostics; these measure fidelity, NOT anatomy.

No training, evaluator or reference discovery is called. Diagnostics preserve
the original soft-weight permission and batch-activation conventions exactly.
"""
import numpy as np
import torch
from torch.nn import functional as F


def semantic_terms(logits, q, weights, semantic_enabled=True):
    """Same logits interpolation/float32/detach/denominator as guidance_loss.

    All three terms are zero on inactive batches, matching the historical
    epoch average. Target entropy on nonzero weights is also reported before
    activation, so inactivity cannot be mistaken for zero target uncertainty.
    """
    q, w = q.detach().float(), weights.detach().float()
    if (q.ndim != 4 or q.shape[1] != 4 or w.shape != (q.shape[0], *q.shape[-2:])
            or not torch.isfinite(q).all() or not torch.isfinite(w).all()
            or (q < 0).any() or (w < 0).any() or (w > 1).any()
            or not torch.allclose(q.sum(1), torch.ones_like(w), atol=2e-4, rtol=0)):
        raise ValueError('invalid soft targets or weights')
    lp = F.interpolate(logits.float(), q.shape[-2:], mode='bilinear', align_corners=False).log_softmax(1)
    active = bool(semantic_enabled and ((q.argmax(1) != 0) & (w > 0)).any())
    denom = w.sum().clamp_min(1e-8)
    # x log x = 0 at x=0; no artificial floor for positive target probabilities.
    qlogq = torch.special.xlogy(q, q)
    entropy = -(qlogq.sum(1)*w).sum()/denom
    ce_class = -(q*lp*w[:, None]).sum((0, 2, 3))/denom
    kl = ((qlogq-q*lp).sum(1)*w).sum()/denom
    zero = lp.sum()*0
    return dict(active=active, weight_sum=w.sum(), denominator=denom,
                ce=ce_class.sum() if active else zero,
                entropy=entropy if active else zero, kl=kl if active else zero,
                target_entropy_before_activation=entropy,
                per_class_ce=ce_class if active else ce_class*0,
                per_class_target_mass=(q*w[:, None]).sum((0, 2, 3)),
                positive_weight_pixels=int((w > 0).sum()))


def scalar_terms(terms):
    return {k: v.detach().cpu().tolist() if torch.is_tensor(v) else v for k, v in terms.items()}


def epoch_terms(rows):
    """Equal batch means including inactive zeros, NOT pooled-pixel CE."""
    n = len(rows)
    active = [r for r in rows if r['active']]
    return dict(batches=n, active_batches=len(active),
                active_fraction=len(active)/n if n else None,
                aggregation='equal_batch_mean_including_inactive_zero',
                **{k: sum(r[k] for r in rows)/n if n else None for k in ('ce', 'entropy', 'kl')},
                active_batch_ce=sum(r['ce'] for r in active)/len(active) if active else None,
                per_class_ce=np.mean([r['per_class_ce'] for r in rows], axis=0).tolist() if n else None,
                per_class_target_mass=np.sum([r['per_class_target_mass'] for r in rows], axis=0).tolist() if n else None)


def replay_active_schedule(frame_activity, patients, seed, epochs, batch_size):
    """Reproduce ONLY the fixed-bank NumPy batch/permission schedule.

    This yields eligible semantic-update counts, not measured finite gradients.
    Exact comparison with the recorded checkpoint update counter is mandatory.
    No model, optimizer or reference is used.
    """
    if set(frame_activity) != set(patients):
        raise ValueError('all TRAIN patients required for schedule replay')
    rng = np.random.default_rng(seed)
    rows = []
    for epoch in range(epochs):
        total, active = 0, 0
        for pid in rng.permutation(patients):
            flags = np.asarray(frame_activity[str(pid)], dtype=bool)
            ids = np.array(list(np.ndindex(flags.shape)))
            rng.shuffle(ids)
            for start in range(0, len(ids), batch_size):
                t, z = ids[start:start+batch_size].T
                active += int(flags[t, z].any()); total += 1
        rows.append(dict(epoch=epoch+1, batches=total, active_batches=active,
                         active_fraction=active/total if total else None))
    return rows


def shared_gradient_norms(model, feature_loss, semantic_loss):
    """No optimizer, no .grad accumulation and no weight mutation."""
    params = [p for block in (model.enc0, model.enc1, model.enc2)
              for p in block.parameters() if p.requires_grad]
    vectors = []
    for loss in (feature_loss, semantic_loss):
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                  for p, g in zip(params, grads)]))
    a, b = vectors
    an, bn = float(a.detach().norm()), float(b.detach().norm())
    return dict(feature_norm=an, semantic_norm=bn,
                cosine=float(a.dot(b)/(a.norm()*b.norm())) if an > 0 and bn > 0 else None,
                weights_updated=False, gradient_scope='enc0_enc1_enc2_before_clipping')


def gate_probabilities(p, config, allowed, representation_only=False):
    top = p.topk(2, 1).values
    probability = top[:, 0] >= config['min_probability']
    margin = top[:, 0]-top[:, 1] >= config['min_margin']
    permission = torch.as_tensor(allowed, device=p.device)[p.argmax(1)]
    valid = probability & margin & permission & (not representation_only)
    labels = torch.where(valid, p.argmax(1), 255)
    return labels, dict(pixels=valid.numel(), accepted=int(valid.sum()),
                        # Overlapping vetoes: do not sum them as a partition.
                        probability_rejected=int((~probability).sum()),
                        margin_rejected=int((~margin).sum()),
                        permission_rejected=int((~permission).sum()),
                        representation_only=representation_only)


def resolution_control(logits, bank_hw):
    """Predetermined attribution control, NOT an alternative export policy.

    Downsample LOGITS exactly as training, softmax on bank grid, then bilinear
    transport probabilities to native. Interpolation is not boundary evidence.
    """
    coarse = F.interpolate(logits.float(), bank_hw, mode='bilinear', align_corners=False).softmax(1)
    projected = F.interpolate(coarse, logits.shape[-2:], mode='bilinear', align_corners=False)
    return coarse, projected


def bank_statistics(data, allowed, config):
    q, r0, rel, hard = (data[k] for k in ('q', 'r0', 'reliability', 'labels'))
    w = rel*np.asarray(allowed)[q.argmax(-1)]
    top = np.sort(q, axis=-1)
    entropy = -(np.where(q > 0, q*np.log(np.maximum(q, 1e-38)), 0)).sum(-1)
    informative = np.max(np.abs(r0-.25), axis=-1) > 1e-7
    result = dict(pixels=int(hard.size), initial_informative_pixels=int(informative.sum()),
                  unknown_pixels=int((hard == 255).sum()),
                  accepted_initially_neutral=int(((hard != 255) & ~informative).sum()),
                  hard_rejection_counts_overlapping={
                      'probability': int((top[..., -1] < config['min_probability']).sum()),
                      'margin': int((top[..., -1]-top[..., -2] < config['min_margin']).sum()),
                      'reliability': int((rel < config['min_reliability']).sum())},
                  weight_sum=float(w.sum()),
                  zero_reliability_pixels=int((rel == 0).sum()),
                  permission_veto_pixels=int((~np.asarray(allowed, dtype=bool)[q.argmax(-1)]).sum()),
                  weighted_target_entropy=float((entropy*w).sum()/w.sum()) if w.sum()>0 else None,
                  classes=[])
    for c in range(4):
        mask = q.argmax(-1) == c
        result['classes'].append(dict(class_id=c, permitted=bool(allowed[c]),
            r0_argmax_informative=int(((r0.argmax(-1) == c) & informative).sum()),
            q_argmax=int(mask.sum()), hard_accepted=int((hard == c).sum()),
            soft_weight_pixels=int((mask & (w > 0)).sum()),
            soft_weight_on_unknown=int((mask & (w > 0) & (hard == 255)).sum()),
            target_mass=float((w*q[..., c]).sum()),
            mean_target_entropy=float(entropy[mask].mean()) if mask.any() else None,
            mean_target_confidence=float(q[..., c][mask].mean()) if mask.any() else None))
    return result
