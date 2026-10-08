"""Experimental Counterfactual Quotient Auditing (CQA), not anatomy certification.

A quotient groups existing, confident, connected atoms without adding pixels.
We compare keeping vs deleting an observed interface using two-fold image
prediction risk. Real adjacent frames are additional *fixed-grid* witnesses;
they are correlated, not independent validation data. Neither semantic logits,
a prototype bank, topology scores nor reference masks select merges.

The audit provides detached same/different-region relations for the anonymous
head. Classical region merging, cross-fitting and affinity losses are prior
art; novelty of this combined research hypothesis is NOT established here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CQAConfig:
    enabled: bool = True
    distill_weight: float = 0.1
    max_edges: int = 512
    max_merges: int = 32
    max_atoms_per_group: int = 8
    min_fold_pixels: int = 2
    min_interface_pixels: int = 2
    min_real_neighbors: int = 1
    merge_risk_max: float = 0.01
    keep_risk_min: float = 0.04
    merge_boundary_max: float = 0.10
    keep_boundary_min: float = 0.35
    require_both_targets: bool = True

    def __post_init__(self):
        for name in ('enabled', 'require_both_targets'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'CQA {name} must be boolean')
        bounds = {'max_edges': (1, 4096), 'max_merges': (1, 256),
                  'max_atoms_per_group': (2, 32), 'min_fold_pixels': (1, 128),
                  'min_interface_pixels': (1, 1024), 'min_real_neighbors': (1, 2)}
        for name, (lo, hi) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'CQA {name} must be an integer in [{lo},{hi}]')
        for name in ('distill_weight', 'merge_risk_max', 'keep_risk_min',
                     'merge_boundary_max', 'keep_boundary_min'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'CQA {name} must be a finite nonnegative number')
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'CQA {name} must be a finite nonnegative number')
        if self.keep_risk_min <= self.merge_risk_max:
            raise ValueError('CQA requires an abstention gap between risk decisions')
        if self.keep_boundary_min <= self.merge_boundary_max:
            raise ValueError('CQA requires an abstention gap between edge decisions')


def _integer_at(batch: dict, name: str, index: int) -> int | None:
    value = batch.get(name)
    if value is None:
        return None
    try:
        value = value[index]
    except (IndexError, TypeError):
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    if isinstance(value, np.integer):
        value = int(value)
    return value if type(value) is int else None


def _views(batch: dict, index: int, hw: tuple[int, int], cfg: CQAConfig):
    t, total = _integer_at(batch, 't', index), _integer_at(batch, 'num_frames', index)
    names = ['cur']
    if t is not None and total is not None and total >= 2 and 0 <= t < total:
        if t > 0:
            names.append('prev')
        if t + 1 < total:
            names.append('nxt')
    if len(names) - 1 < cfg.min_real_neighbors:
        return None, 'no_genuine_neighbor'
    images = []
    for name in names:
        tensor = batch.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError('CQA requires finite [B,3,H,W] temporal image context')
        if index >= len(tensor) or not bool(torch.isfinite(tensor).all()):
            raise ValueError('invalid CQA temporal image')
        image = F.interpolate(tensor[index:index+1, 1:2].detach().float(), hw,
                              mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        scale = float(image.std())
        # Flat witnesses may not license a merge. Never count replicated endpoints.
        if scale < 1e-6:
            return None, 'flat_witness'
        images.append((image - float(image.mean())) / scale)
    return np.stack(images).astype(np.float64), None


def _interfaces(ids: np.ndarray):
    """4-neighbour interfaces with physical endpoint indices; never wrap borders."""
    grid = np.arange(ids.size).reshape(ids.shape)
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b, ia, ib in ((ids[:-1], ids[1:], grid[:-1], grid[1:]),
                          (ids[:, :-1], ids[:, 1:], grid[:, :-1], grid[:, 1:])):
        mask = (a != b) & (a > 0) & (b > 0)
        for ra, rb, pa, pb in zip(a[mask], b[mask], ia[mask], ib[mask]):
            key = tuple(sorted((int(ra), int(rb))))
            edges.setdefault(key, []).append((int(pa), int(pb)))
    return edges


def _risk_delta(images: np.ndarray, atoms: dict[int, np.ndarray],
                members: set[int], parity: np.ndarray, minimum: int):
    """R(merged)-R(finest constituent partition), both fit/test fold directions.

    Comparison to the original atoms, not just the last merge, prevents chaining
    individually tolerable intensity jumps into an unrestricted merged region.
    This is a deterministic surrogate, NOT a calibrated significance test.
    """
    values = images.reshape(len(images), -1)
    deltas = []
    for fit_fold in (0, 1):
        fit, test = [], []
        split_risk = np.zeros(len(images), dtype=np.float64)
        for atom in sorted(members):
            pixels = atoms[atom]
            train = pixels[parity[pixels] == fit_fold]
            check = pixels[parity[pixels] != fit_fold]
            if min(len(train), len(check)) < minimum:
                return None
            mean = values[:, train].mean(1, keepdims=True)
            split_risk += ((values[:, check] - mean) ** 2).sum(1)
            fit.append(train)
            test.append(check)
        fit_all, test_all = np.concatenate(fit), np.concatenate(test)
        merged_mean = values[:, fit_all].mean(1, keepdims=True)
        merged_risk = ((values[:, test_all] - merged_mean) ** 2).sum(1)
        deltas.extend(((merged_risk - split_risk) / len(test_all)).tolist())
    return np.asarray(deltas)


def audit_partition(ids: np.ndarray, images: np.ndarray, cfg: CQAConfig | None = None):
    """Pure, bounded, anonymous split-vs-merge audit on a single image grid.

    ids=0 is immutable void. Returned relations reference original atom IDs.
    Inputs contain no semantic classes. Missing temporal witness handling belongs
    to audit_components; images must include cur plus real neighbour witnesses.
    """
    cfg = cfg or CQAConfig()
    ids, images = np.asarray(ids), np.asarray(images)
    if ids.ndim != 2 or min(ids.shape) < 2 or ids.size > 262144:
        raise ValueError('CQA IDs require a bounded two-dimensional grid')
    if not np.issubdtype(ids.dtype, np.integer) or ids.min() < 0:
        raise ValueError('CQA IDs must be nonnegative integers')
    if (images.ndim != 3 or images.shape[1:] != ids.shape
            or len(images) < 1 + cfg.min_real_neighbors or len(images) > 3
            or not np.isfinite(images).all()):
        raise ValueError('CQA requires current and genuine temporal image witnesses')
    keys = [int(k) for k in np.unique(ids) if k]
    if len(keys) > 256:
        raise ValueError('CQA supports at most 256 nonvoid atoms')
    from scipy.ndimage import label
    if any(label(ids == k)[1] != 1 for k in keys):
        raise ValueError('CQA atoms must each be four-connected')
    atoms = {k: np.flatnonzero(ids.ravel() == k) for k in keys}
    metrics = {'cqa_atoms': len(keys), 'cqa_edges': 0, 'cqa_edges_truncated': 0,
               'cqa_merge_relations': 0, 'cqa_keep_relations': 0,
               'cqa_abstained_edges': 0, 'cqa_chain_vetoes': 0,
               'cqa_budget_vetoes': 0, 'cqa_groups': len(keys)}
    if not cfg.enabled or not keys or np.any(images.std(axis=(1, 2)) < 1e-6):
        return ids.copy(), [], metrics
    parity = (np.indices(ids.shape).sum(0) % 2).ravel()
    edges = _interfaces(ids)
    # Canonical ordering depends on physical support, never anonymous channel ID.
    ordered = sorted(edges, key=lambda ab: tuple(sorted((atoms[ab[0]][0], atoms[ab[1]][0]))))
    metrics['cqa_edges_truncated'] = max(0, len(ordered) - cfg.max_edges)
    ordered = ordered[:cfg.max_edges]
    metrics['cqa_edges'] = len(ordered)
    positives, negatives = [], []
    values = images.reshape(len(images), -1)
    for a, b in ordered:
        interface = np.asarray(edges[(a, b)])
        if len(interface) < cfg.min_interface_pixels:
            metrics['cqa_abstained_edges'] += 1
            continue
        delta = _risk_delta(images, atoms, {a, b}, parity, cfg.min_fold_pixels)
        boundary = np.quantile(np.abs(values[:, interface[:, 0]]
                                      - values[:, interface[:, 1]]), .9, axis=1)
        if delta is None:
            metrics['cqa_abstained_edges'] += 1
        elif float(delta.max()) <= cfg.merge_risk_max and float(boundary.max()) <= cfg.merge_boundary_max:
            positives.append((float(delta.max()), a, b))
        elif float(delta.min()) >= cfg.keep_risk_min and float(boundary.min()) >= cfg.keep_boundary_min:
            negatives.append((a, b, 0))
        else:
            metrics['cqa_abstained_edges'] += 1
    parent = {k: k for k in keys}
    groups = {k: {k} for k in keys}

    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k

    relations = list(negatives)
    certified = {tuple(sorted((x, y))) for _, x, y in positives}
    # Audit weakest-risk interfaces first; physical support breaks numerical ties.
    positives.sort(key=lambda row: (round(row[0], 12),
                                    *sorted((atoms[row[1]][0], atoms[row[2]][0]))))
    for _, a, b in positives:
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        union = groups[ra] | groups[rb]
        if (metrics['cqa_merge_relations'] >= cfg.max_merges
                or len(union) > cfg.max_atoms_per_group):
            metrics['cqa_budget_vetoes'] += 1
            continue
        # Every interface that would disappear must itself have positive image
        # evidence. A path of positive edges cannot erase a negative/unknown edge.
        cross = [(x, y) for x in groups[ra] for y in groups[rb]
                 if tuple(sorted((x, y))) in edges]
        if any(tuple(sorted(pair)) not in certified for pair in cross):
            metrics['cqa_chain_vetoes'] += 1
            continue
        delta = _risk_delta(images, atoms, union, parity, cfg.min_fold_pixels)
        if delta is None or float(delta.max()) > cfg.merge_risk_max:
            metrics['cqa_chain_vetoes'] += 1
            continue
        # Physical order makes the same quotient independent of input numbering.
        if min(atoms[k][0] for k in groups[ra]) > min(atoms[k][0] for k in groups[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        groups[ra] = union
        del groups[rb]
        relations.append((a, b, 1))
        metrics['cqa_merge_relations'] += 1
    result = np.zeros_like(ids)
    ordered_groups = sorted(groups.values(), key=lambda s: min(atoms[k][0] for k in s))
    for output_id, members in enumerate(ordered_groups, 1):
        for atom in members:
            result.ravel()[atoms[atom]] = output_id
    metrics['cqa_groups'] = len(ordered_groups)
    metrics['cqa_keep_relations'] = len(negatives)
    return result, relations, metrics


def relation_distillation_loss(q: torch.Tensor, atomic_ids: torch.Tensor,
                               relations: torch.Tensor, *, require_both: bool = True):
    """Permutation-invariant co-assignment BCE; no names or selected cluster IDs.

    Relations are [N,4]: batch, original_atom_a, original_atom_b, same(0/1).
    Each sample gives equal weight to positive/negative strata. By default, a
    sample with no credible boundary negatives cannot train an all-merge target.
    Float32 accumulation avoids half-precision area overflow. Targets and masks
    are detached, while q retains its autograd graph.
    """
    if (q.ndim != 4 or not q.is_floating_point() or not bool(torch.isfinite(q).all())
            or atomic_ids.shape != (q.shape[0], *q.shape[-2:])
            or atomic_ids.dtype != torch.long or atomic_ids.device != q.device):
        raise ValueError('CQA q/atomic_ids grid, dtype or device mismatch')
    if relations.ndim != 2 or relations.shape[1] != 4 or relations.dtype != torch.long:
        raise ValueError('CQA relations must be long [N,4]')
    if len(relations) > q.shape[0] * 4096:
        raise ValueError('CQA relation budget exceeded')
    if type(require_both) is not bool:
        raise ValueError('require_both must be boolean')
    if bool((q < 0).any()) or not torch.allclose(q.detach().float().sum(1),
            torch.ones_like(q[:, 0], dtype=torch.float32), atol=2e-3, rtol=0):
        raise ValueError('CQA q must be normalized probabilities')
    rows = relations.detach().cpu().tolist()
    grouped: dict[int, list] = {}
    for bi, a, b, same in rows:
        if not (0 <= bi < q.shape[0]) or min(a, b) < 1 or a == b or same not in (0, 1):
            raise ValueError('invalid CQA relation')
        grouped.setdefault(bi, []).append((a, b, same))
    losses = []
    for bi, pairs in grouped.items():
        kinds = {row[2] for row in pairs}
        if require_both and kinds != {0, 1}:
            continue
        ids = atomic_ids[bi].detach().flatten()
        flat = q[bi].float().flatten(1)
        flat = flat / flat.sum(0, keepdim=True).clamp_min(1e-8)
        pooled = {}
        for atom in {a for a, _, _ in pairs} | {b for _, b, _ in pairs}:
            mask = ids == atom
            if not bool(mask.any()):
                raise ValueError('CQA relation references an absent atom')
            pooled[atom] = flat[:, mask].mean(1)
        terms = {0: [], 1: []}
        for a, b, same in pairs:
            coassign = (pooled[a] * pooled[b]).sum()
            # Affine smoothing, not hard clipping: gradients survive at 0/1.
            probability = 1e-6 + (1 - 2e-6) * coassign
            terms[same].append(-torch.log(probability if same else 1 - probability))
        strata = [torch.stack(terms[k]).mean() for k in (0, 1) if terms[k]]
        losses.append(torch.stack(strata).mean())
    return torch.stack(losses).mean() if losses else q.sum() * 0.0


def audit_components(base: dict[str, Any], batch: dict, teacher,
                     config: CQAConfig | None = None) -> dict[str, Any]:
    """Adapt a components_v3 output, preserving confidence and pixel ownership.

    No state is kept across calls. Inference cannot advance readiness or write a
    bank. Semantic probabilities never vote on grouping. Confidence is inherited
    from original assignments, never from a newly one-hot quotient.
    """
    cfg = config or CQAConfig()
    if not cfg.enabled:
        return base
    required = ('component_pixel_region', 'component_pixel_eligible',
                'component_confidence', 'anonymous_region_prob', 'fused_features',
                'motion_features', 'component_eligible')
    if any(name not in base for name in required):
        raise ValueError('CQA requires the confidence-gated component adapter')
    ids = base['component_pixel_region'].detach()
    q = base['anonymous_region_prob']
    if (ids.dtype != torch.long or ids.ndim != 3
            or ids.shape != (q.shape[0], *q.shape[-2:])
            or ids.numel() > 16777216):
        raise ValueError('invalid CQA component grid')
    if bool((ids < 0).any()) or int(ids.max()) > 256:
        raise ValueError('CQA component IDs outside the adapter bound')
    inherited = base['component_eligible'].detach()
    if (inherited.ndim != 2 or inherited.dtype != torch.bool or len(inherited) != len(ids)
            or int(ids.max()) >= inherited.shape[1]):
        raise ValueError('invalid inherited CQA region eligibility')
    allowed = inherited.gather(1, ids.flatten(1)).reshape_as(ids)
    if bool(((ids > 0) & ~allowed).any()):
        raise ValueError('CQA may not promote an ineligible component')
    confidence = base['component_confidence'].detach()
    if (confidence.shape != ids.shape or not bool(torch.isfinite(confidence).all())
            or bool(((confidence < 0) | (confidence > 1)).any())):
        raise ValueError('invalid inherited CQA confidence')
    support = base['component_pixel_eligible'].detach()
    if support.shape != ids.shape or support.dtype != torch.bool:
        raise ValueError('invalid CQA pixel support')
    if not torch.equal(ids > 0, support):
        raise ValueError('unsupported pixels may not enter CQA atoms')
    partitions, all_relations, metrics = [], [], {}
    for bi in range(len(ids)):
        views, reason = _views(batch, bi, tuple(ids.shape[-2:]), cfg)
        original = ids[bi].cpu().numpy()
        if reason:
            partition, relations = original.copy(), []
            observed = {'cqa_' + reason + '_samples': 1,
                        'cqa_atoms': int((np.unique(original) > 0).sum()),
                        'cqa_groups': int((np.unique(original) > 0).sum())}
        else:
            partition, relations, observed = audit_partition(original, views, cfg)
        partitions.append(partition)
        all_relations.extend((bi, a, b, target) for a, b, target in relations)
        for key, value in observed.items():
            metrics[key] = metrics.get(key, 0) + value
    for key in ('cqa_merge_relations', 'cqa_keep_relations', 'cqa_abstained_edges',
                'cqa_edges', 'cqa_edges_truncated', 'cqa_chain_vetoes',
                'cqa_budget_vetoes', 'cqa_no_genuine_neighbor_samples', 'cqa_flat_witness_samples'):
        metrics.setdefault(key, 0)
    owner = torch.from_numpy(np.stack(partitions)).to(ids.device)
    if not torch.equal(owner == 0, ids == 0):
        raise RuntimeError('CQA must preserve void exactly')
    channels = int(owner.max()) + 1
    if channels * owner.numel() > 16777216:
        raise ValueError('CQA quotient allocation exceeds component adapter bound')
    assign = q.new_zeros((len(ids), channels, *ids.shape[-2:]))
    assign.scatter_(1, owner[:, None], 1)
    sizes = assign.float().sum((-2, -1))
    eligible = sizes > 0
    eligible[:, 0] = False
    confidence = base['component_confidence'].detach()
    mean_conf = torch.einsum('bkhw,bhw->bk', assign.float(), confidence.float()) / sizes.clamp_min(1)
    mean_conf = mean_conf.masked_fill(~eligible, 0).to(q.dtype)
    from .system_v3 import pool_regions
    features = pool_regions(assign.float(), base['fused_features'].float(),
                            batch['cur'][:, 1:2].float(), base['motion_features'].float())
    features = features.to(base['fused_features'].dtype) * eligible[:, :, None]
    logits = teacher.semantic(features)
    if logits.shape != (len(ids), channels, 4) or not bool(torch.isfinite(logits).all()):
        raise ValueError('invalid CQA semantic repooling')
    out = dict(base)
    # An assembled group has no single source prototype; -2 is descriptive only.
    source = torch.full_like(sizes, -2, dtype=torch.long).masked_fill(~eligible, -1)
    out.update(region_prob=assign, region_features=features, semantic_logits=logits,
               semantic_prob=logits.softmax(-1), component_eligible=eligible,
               component_candidate=eligible,
               component_pixel_region=owner, component_pixel_eligible=support,
               component_region_confidence=mean_conf, component_source_id=source,
               component_sizes=sizes.long(), cqa_atomic_ids=ids,
               cqa_relations=torch.tensor(all_relations, dtype=torch.long,
                                          device=q.device).reshape(-1, 4),
               cqa_metrics=metrics)
    return out
