"""Evaluation-only metrics. No tensor returned here can enter an optimizer graph."""
from __future__ import annotations

import math
import numpy as np
import torch


@torch.no_grad()
def class_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per example foreground Dice; both-empty classes are NaN, not perfect."""
    pred = logits.detach().argmax(1)
    values = []
    for cls in range(1, logits.shape[1]):
        p, y = pred == cls, target.detach() == cls
        dims = tuple(range(1, p.ndim))
        den = p.sum(dims) + y.sum(dims)
        values.append(torch.where(den > 0, 2 * (p & y).sum(dims) / den.clamp_min(1), float('nan')))
    return torch.stack(values, dim=1)


@torch.no_grad()
def oracle_delta(skip_logits, audit_logits, target):
    """Offline oracle contrast ONLY. Never used by training or action selection."""
    return torch.nanmean(class_dice(audit_logits, target), 1) - torch.nanmean(class_dice(skip_logits, target), 1)


def _ranks(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def _corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def binary_ranking(labels, scores):
    """Tie-aware ROC AUC and grouped average precision, without sklearn."""
    y, s = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return {'auroc': None, 'auprc': None, 'reason': 'both outcome classes required'}
    auc = (_ranks(s)[y].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(-s, kind='stable')
    tp, seen, ap, previous_tp = 0, 0, 0., 0
    while seen < len(y):
        end = seen + 1
        while end < len(y) and s[order[end]] == s[order[seen]]:
            end += 1
        tp += int(y[order[seen:end]].sum())
        ap += (tp - previous_tp) / positives * tp / end
        previous_tp, seen = tp, end
    return {'auroc': float(auc), 'auprc': float(ap), 'reason': None}


def transition_metrics(rf_delta, true_delta, predicted_value=None, decisions=None):
    q, d = np.asarray(rf_delta, float), np.asarray(true_delta, float)
    valid = np.isfinite(q) & np.isfinite(d)
    q, d = q[valid], d[valid]
    out = {'n': len(d), 'pearson': _corr(q, d), 'spearman': _corr(_ranks(q), _ranks(d)),
           'rf_sign_agreement': float(np.mean((q > 0) == (d > 0))) if len(d) else None,
           'true_beneficial_fraction': float(np.mean(d > 0)) if len(d) else None,
           'rf_beneficial_fraction': float(np.mean(q > 0)) if len(d) else None,
           'mean_true_delta': float(d.mean()) if len(d) else None,
           'oracle_mean_retained_delta': float(np.maximum(d, 0).mean()) if len(d) else None,
           'rf_auroc_auprc': binary_ranking(d > 0, q) if len(d) else None}
    if predicted_value is not None:
        v = np.asarray(predicted_value, float)[valid]
        out['trigger_auroc_auprc'] = binary_ranking(d > 0, v)
        out['value_bins'] = [dict(n=int(mask.sum()), predicted_value=float(v[mask].mean()),
                                 observed_true_delta=float(d[mask].mean()),
                                 beneficial_frequency=float((d[mask] > 0).mean()))
                             for ids in np.array_split(np.argsort(v), max(1, min(5, len(v))))
                             if len(ids) and (mask := np.isin(np.arange(len(v)), ids)).any()]
        out['value_bin_note'] = 'Value is in intrinsic proxy units; bins are diagnostic, not calibrated Dice/probability.'
    if decisions is not None:
        z = np.asarray(decisions, bool)[valid]
        kept = d * z
        out.update(audit_fraction=float(z.mean()) if len(z) else None,
                   harm_rate=float((kept < 0).mean()) if len(z) else None,
                   true_delta_per_audit=float(d[z].mean()) if z.any() else None,
                   false_intervention_rate=float((d[z] <= 0).mean()) if z.any() else None,
                   beneficial_recall=float(z[d > 0].mean()) if (d > 0).any() else None,
                   missed_beneficial_rate=float((~z[d > 0]).mean()) if (d > 0).any() else None,
                   utility_regret=float((np.maximum(d, 0) - kept).mean()) if len(d) else None)
    return out


def convergence_metrics(curve, thresholds=(.5, .7, .8)):
    """Scheduled validation only; unreached thresholds are right-censored."""
    result = {'thresholds': {}, 'area_under_curve': {}, 'best_validation_dice': None}
    valid = [r for r in curve if r.get('final_dice') is not None]
    if not valid:
        return result
    result['best_validation_dice'] = max(r['final_dice'] for r in valid)
    for threshold in thresholds:
        hit = next((r for r in valid if r['final_dice'] >= threshold), None)
        result['thresholds'][str(threshold)] = {'updates': hit['updates'] if hit else None,
              'examples_seen': hit['examples_seen'] if hit else None,
              'wall_seconds': hit['wall_seconds'] if hit else None, 'right_censored': hit is None}
    for axis in ('updates', 'examples_seen', 'wall_seconds'):
        area = sum((b[axis]-a[axis])*(b['final_dice']+a['final_dice'])/2 for a,b in zip(valid, valid[1:]))
        width = valid[-1][axis]-valid[0][axis]
        result['area_under_curve'][axis] = {'integral': area, 'normalized': area/width if width > 0 else None}
    return result


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    return value
