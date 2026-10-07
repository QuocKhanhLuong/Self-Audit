"""Opt-in, image-local regions for the experimental teacher bootstrap.

Anonymous prototypes are useful for representation learning, but a prototype can
occur in several unrelated places. This adapter splits its hard assignments into
four-connected components of confident assignments before semantic evidence is
considered. It neither names anatomy nor modifies a mask with morphology, and it
never consults GT.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
import torch

from .system_v3 import pool_regions

# Explicit implementation bounds include the void channel. They prevent a very
# fragmented/high-resolution assignment from creating an unbounded dense tensor.
MAX_GRID_PIXELS = 262144
MAX_ASSIGNMENT_ELEMENTS = 16777216
MAX_COMPONENTS = 256


def component_regions(base, cur, teacher, *, min_region_pixels=4,
                      max_components=256, min_probability=.70):
    """Return a new base dictionary with bounded component-local region tensors.

    Inputs require ``region_prob`` [B,K,H,W], ``fused_features`` [B,D,H,W],
    ``motion_features`` [B,M,H,W], and a finite image ``cur`` [B,3,H',W'].
    ``teacher.semantic`` consumes D+3 pooled features, just like the original
    prototype head. The original base dictionary and its tensors are untouched.

    Output ``region_prob`` is a detached one-hot physical partition [B,C,H,W].
    Channel zero is always void; C <= max_components+1. Components smaller than
    min_region_pixels and components beyond the cap go to void. Exact argmax
    ties also abstain, because breaking them by anonymous ID is not invariant.
    Pixels whose original winning probability is below min_probability go to
    void BEFORE component labeling. An uncertain bridge or seal therefore cannot
    supply topology, pooled features or readiness-history support.
    Retained components are ordered by descending size, then first raster pixel;
    changing anonymous prototype IDs does not change the physical partition.

    ``component_eligible`` [B,C] rejects void, padding, and regions whose mean
    original winning probability is below min_probability.
    ``component_region_confidence`` [B,C] stores that mean (zero on void/padding).
    ``component_confidence`` [B,H,W] is the original winning probability on a
    retained component, zero on void. ``component_pixel_eligible`` [B,H,W]
    additionally enforces the original *pixel* probability and region eligibility.
    Callers must gate semantic evidence with component_eligible and dense labels
    with component_pixel_eligible (nearest-neighbour upsampling). Hard one-hot
    membership alone must never be mistaken for confidence or certification.

    Original region tensors are available under ``anonymous_region_prob``,
    ``anonymous_region_features``, ``anonymous_semantic_logits`` and
    ``anonymous_semantic_prob``. In particular, unsupervised prototype losses must
    use anonymous_region_prob, which retains its original autograd graph. Local
    pooled features and teacher.semantic retain gradients; the partition and all
    eligibility/confidence metadata are intentionally detached.
    """
    from scipy.ndimage import label

    if type(min_region_pixels) is not int or min_region_pixels < 1:
        raise ValueError('min_region_pixels must be a positive integer')
    if type(max_components) is not int or not 1 <= max_components <= MAX_COMPONENTS:
        raise ValueError(f'max_components must be an integer in [1,{MAX_COMPONENTS}]')
    if (isinstance(min_probability, bool) or not isinstance(min_probability, Real)
            or not math.isfinite(min_probability) or not 0 <= min_probability <= 1):
        raise ValueError('min_probability must be finite and in [0,1]')
    if 'anonymous_region_prob' in base:
        raise ValueError('component_regions cannot be applied twice to the same base')
    q = base['region_prob']
    fused = base['fused_features']
    motion = base['motion_features']
    if q.ndim != 4 or not q.is_floating_point() or any(d < 1 for d in q.shape):
        raise ValueError('region_prob must be floating [B,K,H,W] with nonempty axes')
    b, k, h, w = q.shape
    if h*w > MAX_GRID_PIXELS or b*h*w > MAX_ASSIGNMENT_ELEMENTS:
        raise ValueError('component assignment grid exceeds the declared memory bound')
    for name, value in (('fused_features', fused), ('motion_features', motion)):
        if (value.ndim != 4 or value.shape[0] != b or value.shape[1] < 1
                or value.shape[-2:] != (h,w) or value.device != q.device
                or value.dtype != q.dtype):
            raise ValueError(f'{name} must match the assignment batch, grid, device and dtype')
    if (cur.ndim != 4 or cur.shape[:2] != (b,3) or min(cur.shape[-2:]) < 1
            or cur.device != q.device or cur.dtype != q.dtype):
        raise ValueError('cur must be [B,3,H,W] on the assignment device and dtype')
    if not all(bool(torch.isfinite(v).all()) for v in (q, fused, motion, cur)):
        raise ValueError('component inputs must be finite')
    tol = max(1e-4, 4*torch.finfo(q.dtype).eps)
    if (bool((q < 0).any()) or bool((q > 1).any())
            or not bool(torch.allclose(q.detach().float().sum(1),
                                      torch.ones_like(q[:,0], dtype=torch.float32),
                                      atol=tol, rtol=0))):
        raise ValueError('region_prob must contain normalized probabilities')
    if getattr(teacher, 'region_dim', fused.shape[1]+3) != fused.shape[1]+3:
        raise ValueError('teacher region_dim must equal fused feature width + 3')

    # Partition construction is discrete and intentionally outside autograd.
    with torch.no_grad():
        winning, hard = q.detach().max(1)
        unique = (q.detach() == winning[:,None]).sum(1) == 1
        hard_cpu = hard.cpu().numpy()
        # Confidence applies before connectivity/topology, not merely after
        # decoding. An uncertain radial seam must not certify a closed wall.
        supported_cpu = (unique & (winning >= min_probability)).cpu().numpy()
        partitions = np.zeros((b,h,w), dtype=np.int64)
        source_rows, size_rows, counts, dropped = [], [], [], []
        # The effective cap also accounts for the batch/grid product. A reduced
        # cap only withholds unsupported outputs; no omitted pixels are relabelled.
        capacity = min(max_components, MAX_ASSIGNMENT_ELEMENTS//(b*h*w)-1)
        if capacity < 1:
            raise ValueError('component assignment grid leaves no bounded region capacity')
        for bi in range(b):
            local = np.zeros((h,w), dtype=np.int64)
            candidates = []
            offset = 0
            component_count = 0
            # scipy's default connectivity is four-neighbour in two dimensions.
            for source_id in range(k):
                cc, n = label((hard_cpu[bi] == source_id) & supported_cpu[bi])
                component_count += n
                if n == 0:
                    continue
                sizes = np.bincount(cc.ravel(), minlength=n+1)
                first = np.full(n+1, h*w, dtype=np.int64)
                np.minimum.at(first, cc.ravel(), np.arange(h*w, dtype=np.int64))
                mask = cc != 0
                local[mask] = cc[mask] + offset
                for cid in np.flatnonzero(sizes[1:] >= min_region_pixels)+1:
                    candidates.append((int(sizes[cid]), int(first[cid]),
                                       source_id, int(cid)+offset))
                offset += n
            candidates.sort(key=lambda row: (-row[0], row[1]))
            kept = candidates[:capacity]
            # A lookup performs one grid pass regardless of component count.
            mapping = np.zeros(offset+1, dtype=np.int64)
            for rid, row in enumerate(kept, 1):
                mapping[row[3]] = rid
            partitions[bi] = mapping[local]
            source_rows.append([row[2] for row in kept])
            size_rows.append([row[0] for row in kept])
            counts.append(component_count)
            dropped.append(component_count-len(kept))
        channels = max(len(row) for row in source_rows)+1
        ids = torch.from_numpy(partitions).to(q.device)
        # Avoid F.one_hot's extra int64 B*C*H*W allocation.
        assign = q.new_zeros((b,channels,h,w))
        assign.scatter_(1, ids[:,None], 1)
        source = torch.full((b,channels), -1, device=q.device, dtype=torch.long)
        sizes = torch.zeros((b,channels), device=q.device, dtype=torch.long)
        sizes[:,0] = (ids == 0).sum((-2,-1))
        for bi, row in enumerate(source_rows):
            if row:
                source[bi,1:len(row)+1] = torch.tensor(row, device=q.device)
                sizes[bi,1:len(row)+1] = torch.tensor(size_rows[bi], device=q.device)
        represented = source >= 0
        # Accumulate low-precision probabilities in float32: a large valid
        # component must not overflow a float16 sum before taking its mean.
        confidence_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        region_confidence = torch.zeros((b,channels), device=q.device, dtype=confidence_dtype)
        region_confidence.scatter_add_(1, ids.flatten(1), winning.flatten(1).to(confidence_dtype))
        region_confidence = (region_confidence/sizes.clamp_min(1).to(confidence_dtype)).to(q.dtype)
        region_confidence = region_confidence.masked_fill(~represented, 0)
        eligible = represented & (region_confidence >= min_probability)
        pixel_confidence = winning.masked_fill(ids == 0, 0)
        pixel_eligible = eligible.gather(1, ids.flatten(1)).reshape(b,h,w)
        pixel_eligible &= pixel_confidence >= min_probability

    # Each feature mean uses only that physical component. Void and padded slots
    # cannot collect a semantic training gradient or borrow any local evidence.
    pool_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    features = pool_regions(assign.to(pool_dtype), fused.to(pool_dtype),
                            cur[:,1:2].to(pool_dtype), motion.to(pool_dtype)).to(fused.dtype)
    features = features*represented[:,:,None].to(features.dtype)
    logits = teacher.semantic(features)
    if logits.shape != (b,channels,4) or not bool(torch.isfinite(logits).all()):
        raise ValueError('teacher.semantic must return finite [B,C,4] logits')
    logits = logits*represented[:,:,None].to(logits.dtype)
    out = dict(base)
    for key in ('region_prob', 'region_features', 'semantic_logits', 'semantic_prob'):
        if key in base:
            out['anonymous_'+key] = base[key]
    out.update(region_prob=assign, region_features=features,
               semantic_logits=logits, semantic_prob=logits.softmax(-1),
               component_eligible=eligible,
               component_region_confidence=region_confidence,
               component_confidence=pixel_confidence,
               component_pixel_confidence=winning[:,None],
               component_pixel_eligible=pixel_eligible,
               component_pixel_region=ids, component_source_id=source,
               component_sizes=sizes,
               component_count=torch.tensor(counts, device=q.device),
               component_dropped_count=torch.tensor(dropped, device=q.device))
    return out
