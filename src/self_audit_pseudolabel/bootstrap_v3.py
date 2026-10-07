"""Conservative image-only permission to begin semantic training, not anatomy certification.

This optional research recipe keeps the existing topology resolver authoritative.
Neither a neural semantic prediction, prototype-bank score nor elapsed epoch can
satisfy the gate. The declared numerical defaults are engineering safeguards,
not thresholds calibrated for ACDC or M&Ms accuracy. Fixed-grid corroboration of
true neighbouring frames is deliberately conservative under large motion.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

BG, RV, MYO, LV = 0, 1, 2, 3


@dataclass(frozen=True)
class BootstrapConfig:
    min_successful_updates: int = 5
    min_train_samples: int = 4
    min_train_patients: int = 2
    min_observations_per_sample: int = 2
    min_mask_iou: float = .80
    min_region_pixels: int = 8
    min_assignment_probability: float = .70
    min_contrast: float = 1.0
    min_edge_enrichment: float = 1.25
    min_temporal_neighbors: int = 1
    max_history_samples: int = 128
    max_mask_pixels: int = 262144

    def __post_init__(self):
        lower = {'min_successful_updates': 0, 'min_train_samples': 2,
                 'min_train_patients': 2, 'min_observations_per_sample': 2,
                 'min_region_pixels': 4, 'min_temporal_neighbors': 1,
                 'max_history_samples': 2, 'max_mask_pixels': 16}
        for name, minimum in lower.items():
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.min_temporal_neighbors > 2:
            raise ValueError('at most two adjacent frames are available')
        if self.min_train_samples < self.min_train_patients:
            raise ValueError('sample requirement must cover the patient requirement')
        if self.max_history_samples < self.min_train_samples:
            raise ValueError('history cannot hold the required distinct samples')
        for name in ('min_mask_iou', 'min_contrast', 'min_edge_enrichment', 'min_assignment_probability'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if not .5 < self.min_assignment_probability <= 1:
            raise ValueError('min_assignment_probability must be in (.5,1]')
        if not 0 < self.min_mask_iou <= 1 or self.min_contrast <= 0 or self.min_edge_enrichment <= 1:
            raise ValueError('invalid image-evidence thresholds')


@dataclass
class CandidateSupport:
    region_eligible: torch.Tensor
    sample_eligible: torch.Tensor
    masks: list[torch.Tensor | None]
    diagnostics: list[dict[str, Any]]


def _at(batch, name, index):
    if name not in batch:
        return None
    values = batch[name]
    value = values[index] if not isinstance(values, str) else values
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    return value


def _identity(batch, index):
    pid, t, z = (_at(batch, name, index) for name in ('patient_id', 't', 'z'))
    if not isinstance(pid, str) or not pid or type(t) is not int or type(z) is not int or min(t, z) < 0:
        raise ValueError('bootstrap requires patient_id and nonnegative integer t,z identities')
    return pid, t, z


def _dilate(mask):
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask)


def _image_support(image, wall, cavity, cfg):
    """Sign-free robust contrast plus local boundary enrichment; no shape template."""
    if min(int(wall.sum()), int(cavity.sum())) < cfg.min_region_pixels:
        return False, {'contrast': 0., 'edge_enrichment': 0.}
    # Boundary pixels are excluded from the comparison region, not used to set
    # their own noise floor. A local neighbourhood avoids image-border dominance.
    interface = (_dilate(cavity) & wall) | (_dilate(wall) & cavity)
    vicinity = _dilate(_dilate(wall | cavity))
    control = vicinity & ~_dilate(interface)
    if not interface.any() or int(control.sum()) < cfg.min_region_pixels:
        return False, {'contrast': 0., 'edge_enrichment': 0.}
    w, c = image[wall], image[cavity]
    wm, cm = float(np.median(w)), float(np.median(c))
    span = float(np.percentile(image[vicinity], 95) - np.percentile(image[vicinity], 5))
    if not np.isfinite(span) or span <= 1e-6:
        return False, {'contrast': 0., 'edge_enrichment': 0.}
    noise = 1.4826 * .5 * (float(np.median(np.abs(w-wm))) + float(np.median(np.abs(c-cm))))
    contrast = abs(cm-wm) / max(noise, .05*span, 1e-6)
    gy, gx = np.gradient(image)
    grad = np.hypot(gx, gy)
    boundary = float(np.mean(grad[interface]))
    # Nonzero absolute scale prevents a numerically flat image from passing a
    # ratio of almost-zero gradients. Control mean rejects diffuse noise edges.
    enrichment = boundary / max(float(np.mean(grad[control])), .02*span, 1e-6)
    good = contrast >= cfg.min_contrast and enrichment >= cfg.min_edge_enrichment
    return bool(good), {'contrast': contrast, 'edge_enrichment': enrichment}


def assess_bootstrap_candidates(batch, region_prob, raw_logits, raw_valid, diagnostics, config=None,
                                *, assignment_confidence=None):
    """Pure local eligibility; no readiness or prototype state is mutated.

    raw_valid must be the pre-bank raw-evidence confidence/size gate. The
    diagnostics must come from the existing resolver. We recheck its complete
    connected-region topology, so a disconnected cluster cannot borrow one
    component's evidence. assignment_confidence is the original winning-pixel
    confidence [B,H,W] when regions have been adapted to one-hot components; by
    default it is derived from region_prob. Every wall/cavity pixel must pass the
    declared confidence floor: no uncertain seam can certify closure, and a
    trusted subset never licenses naming a larger region. BG is allowed only on
    currently supported images.
    """
    from scipy.ndimage import binary_fill_holes, label
    cfg = config or BootstrapConfig()
    if region_prob.ndim != 4 or min(region_prob.shape[-2:]) < 2:
        raise ValueError('bootstrap region_prob must be [B,K,H,W] with H,W >= 2')
    b, k, h, w = region_prob.shape
    if h*w > cfg.max_mask_pixels:
        raise ValueError('bootstrap assignment grid exceeds the declared mask bound')
    if raw_logits.shape != (b, k, 4) or raw_valid.shape != (b, k) or raw_valid.dtype != torch.bool:
        raise ValueError('bootstrap requires raw logits [B,K,4] and bool raw_valid [B,K]')
    if len(diagnostics) != b or not bool(torch.isfinite(region_prob).all()) or not bool(torch.isfinite(raw_logits).all()):
        raise ValueError('invalid bootstrap evidence')
    if assignment_confidence is None:
        assignment_confidence = region_prob.detach().max(1).values
    if (not isinstance(assignment_confidence,torch.Tensor)
            or assignment_confidence.shape != (b,h,w)
            or not assignment_confidence.is_floating_point()
            or not bool(torch.isfinite(assignment_confidence).all())
            or bool(((assignment_confidence<0)|(assignment_confidence>1)).any())):
        raise ValueError('assignment_confidence must be finite floating [B,H,W] in [0,1]')
    supported_pixels = (assignment_confidence.detach() >= cfg.min_assignment_probability).cpu().numpy()
    labels = region_prob.detach().argmax(1).cpu().numpy()
    classes = raw_logits.detach().argmax(-1).cpu().numpy()
    valid = raw_valid.detach().cpu().numpy()
    images = {}
    for name in ('cur', 'prev', 'nxt'):
        value = batch.get(name)
        if value is None or value.ndim != 4 or value.shape[:2] != (b, 3) or not bool(torch.isfinite(value).all()):
            raise ValueError('bootstrap requires finite [B,3,H,W] temporal image context')
        images[name] = F.interpolate(value.detach()[:,1:2], (h,w), mode='bilinear', align_corners=False)[:,0].cpu().numpy()
    eligible = np.zeros((b,k), dtype=bool)
    sample_ok = np.zeros(b, dtype=bool)
    masks, output_diag = [], []
    for bi in range(b):
        _identity(batch, bi)
        t, frames = _at(batch, 't', bi), _at(batch, 'num_frames', bi)
        neighbors = []
        if type(frames) is int and frames >= 2 and t < frames:
            if t > 0: neighbors.append('prev')
            if t+1 < frames: neighbors.append('nxt')
        trusted = []
        details = []
        for pair in diagnostics[bi].get('enclosures', []):
            outer, inner = int(pair[0]), int(pair[1])
            if not (0 <= outer < k and 0 <= inner < k) or outer == inner:
                raise ValueError('invalid enclosure region identity')
            if not (valid[bi,outer] and valid[bi,inner] and classes[bi,outer] == MYO and classes[bi,inner] == LV):
                continue
            wall, cavity = labels[bi] == outer, labels[bi] == inner
            if ((wall|cavity)&~supported_pixels[bi]).any():
                details.append({'outer':outer,'inner':inner,'eligible':False,
                                'reason':'unsupported_wall_or_cavity_pixels'})
                continue
            border = lambda m: bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any())
            holes = binary_fill_holes(wall) & ~wall
            if border(wall) or border(cavity) or label(wall)[1] != 1 or label(cavity)[1] != 1:
                continue
            if label(holes)[1] != 1 or (binary_fill_holes(cavity) & ~cavity).any() or not (_dilate(wall)&cavity).any():
                continue
            overlap = int((holes&cavity).sum())
            if overlap/max(int(cavity.sum()),1) < .8 or overlap/max(int(holes.sum()),1) < .8:
                continue
            good, detail = _image_support(images['cur'][bi],wall,cavity,cfg)
            temporal = sum(_image_support(images[name][bi],wall,cavity,cfg)[0] for name in neighbors)
            good = good and temporal >= cfg.min_temporal_neighbors
            details.append({'outer':outer,'inner':inner,**detail,'temporal_neighbors':temporal,'eligible':bool(good)})
            if good: trusted.append((outer,inner))
        if trusted:
            sample_ok[bi] = True
            eligible[bi] = valid[bi] & (classes[bi] == BG)
            wall_union = np.zeros((h,w),dtype=bool)
            cavity_union = np.zeros((h,w),dtype=bool)
            for outer,inner in trusted:
                eligible[bi,outer] = eligible[bi,inner] = True
                wall = labels[bi] == outer
                wall_union |= wall
                cavity_union |= labels[bi] == inner
                for rid in range(k):
                    if not valid[bi,rid] or classes[bi,rid] != RV: continue
                    rv = labels[bi] == rid
                    if (rv&~supported_pixels[bi]).any(): continue
                    # Keep RV tied to this trusted wall and the original raw
                    # orientation gate; no new semantic voting is introduced.
                    if border(rv) or label(rv)[1] != 1 or (binary_fill_holes(rv)&~rv).any(): continue
                    if (_dilate(wall)&rv).any(): eligible[bi,rid] = True
            masks.append(torch.from_numpy(np.stack([wall_union,cavity_union])).clone())
        else:
            masks.append(None)
        output_diag.append({'trusted_pairs':len(trusted),'pairs':details,'genuine_neighbors':len(neighbors)})
    return CandidateSupport(torch.from_numpy(eligible).to(raw_valid.device),
                            torch.from_numpy(sample_ok).to(raw_valid.device),masks,output_diag)


class BootstrapReadiness:
    """Training-only evidence latch with bounded, consecutive per-sample support.

    Call observe exactly once AFTER each successful optimizer update. Repeated
    updates on one image never manufacture distinct subjects. Mask IoU compares
    physical wall/cavity roles, so anonymous-channel permutations have no effect.
    Support anchors are selected by stable identity hashes, with one per patient
    retained where the budget allows. Unlike LRU, unrelated epoch samples cannot
    erase all repeat-observation opportunities. Selection uses no image score,
    semantic class, process-randomized hash, or RNG. A failed local check removes
    that anchor; geometric instability restarts its consecutive support count.
    An activated latch only grants permission: local eligibility remains required
    in every training and inference batch. No evaluation sample may update it.
    """
    STATE_VERSION = 2

    def __init__(self, config=None, train_patient_ids=()):
        self.config = config or BootstrapConfig()
        ids = tuple(sorted(train_patient_ids))
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(p,str) or not p for p in ids):
            raise ValueError('readiness needs distinct locked training patient IDs')
        self.train_patient_ids = ids
        self.successful_updates = 0
        self.ready = False
        self.activation = None
        self._history = OrderedDict()

    def _qualified(self):
        return [(key,row) for key,row in self._history.items()
                if row['observations'] >= self.config.min_observations_per_sample]

    @property
    def qualified_samples(self):
        return len(self._qualified())

    @property
    def qualified_patients(self):
        return len({key[0] for key,_ in self._qualified()})

    def validate_training_batch(self,batch):
        """Validate identities/cohort BEFORE the caller performs an optimizer step.

        This method is pure and is also repeated by observe as defense in depth.
        Metadata must identify every sample; no implicit patient or sample count
        may turn evaluation images into training observations.
        """
        image = batch.get('cur')
        if not isinstance(image,torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
            raise ValueError('training identity validation requires nonempty cur [B,C,H,W]')
        n = image.shape[0]
        for name in ('patient_id','t','z'):
            values = batch.get(name)
            if isinstance(values,str):
                if name != 'patient_id' or n != 1:
                    raise ValueError('training metadata must identify every sample')
            else:
                try:
                    if len(values) != n: raise ValueError('training metadata length mismatch')
                except TypeError as exc:
                    raise ValueError('training metadata must identify every sample') from exc
        try:
            keys = [_identity(batch,i) for i in range(n)]
        except (IndexError,TypeError) as exc:
            raise ValueError('invalid training sample identity metadata') from exc
        if any(key[0] not in self.train_patient_ids for key in keys):
            raise ValueError('only locked TRAIN samples can advance bootstrap readiness')
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate sample identities within one update')
        return tuple(keys)

    @staticmethod
    def _identity_rank(key):
        encoded = json.dumps(list(key),ensure_ascii=True,separators=(',',':')).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest(),key

    def _retain_anchors(self,history):
        # Preserve each patient's best identity anchor when there is capacity;
        # fill remaining slots by the same global identity rank. For >budget
        # patients, patient-identity hashes choose the representative subjects.
        # This mergeable rule is independent of iteration/arrival order for a
        # stream of eligible identities, and does not privilege image quality.
        ordered = sorted(history,key=self._identity_rank)
        representatives = {}
        for key in ordered: representatives.setdefault(key[0],key)
        patient_order = sorted(representatives,key=lambda pid:self._identity_rank((pid,)))
        budget = self.config.max_history_samples
        chosen = [representatives[pid] for pid in patient_order[:budget]]
        selected = set(chosen)
        chosen.extend(key for key in ordered if key not in selected)
        chosen = chosen[:budget]
        return OrderedDict((key,history[key]) for key in sorted(chosen,key=self._identity_rank))

    def observe(self,batch,support):
        keys = self.validate_training_batch(batch)
        n = len(keys)
        if len(support.masks) != n or support.sample_eligible.shape != (n,) or support.sample_eligible.dtype != torch.bool:
            raise ValueError('invalid candidate support batch')
        # Validate first so a malformed observation cannot partially advance state.
        for i,mask in enumerate(support.masks):
            if bool(support.sample_eligible[i]): self._validate_mask(mask)
            elif mask is not None: raise ValueError('ineligible samples cannot carry support masks')
        self.successful_updates += 1
        for i,key in enumerate(keys):
            previous = self._history.pop(key,None)
            mask = support.masks[i]
            if not bool(support.sample_eligible[i]): continue
            mask = mask.detach().cpu()
            count = 1
            if previous is not None and previous['mask'].shape == mask.shape:
                inter = (previous['mask'] & mask).sum((-2,-1)).double()
                union = (previous['mask'] | mask).sum((-2,-1)).double()
                if bool((inter/union.clamp_min(1) >= self.config.min_mask_iou).all()):
                    count = previous['observations']+1
            self._history[key] = {'mask':mask.detach().cpu().clone(),
                                  'observations':count,'last_update':self.successful_updates}
        self._history = self._retain_anchors(self._history)
        if (self.successful_updates >= self.config.min_successful_updates
                and self.qualified_samples >= self.config.min_train_samples
                and self.qualified_patients >= self.config.min_train_patients):
            if not self.ready:
                self.activation = {'successful_update':self.successful_updates,
                    'samples':[{'key':list(key),'observations':row['observations']}
                               for key,row in self._qualified()]}
            self.ready = True
        return self.ready

    def _validate_mask(self,mask):
        if not isinstance(mask,torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 3 or mask.shape[0] != 2:
            raise ValueError('history masks must be bool [2,H,W] tensors')
        if min(mask.shape[-2:]) < 2 or mask.shape[-2]*mask.shape[-1] > self.config.max_mask_pixels:
            raise ValueError('invalid bounded history mask shape')
        if not bool(mask.any(-1).any(-1).all()) or bool((mask[0]&mask[1]).any()):
            raise ValueError('wall/cavity history masks must be nonempty and disjoint')

    def state_dict(self):
        return {'version':self.STATE_VERSION,'config':asdict(self.config),
                'train_patient_ids':list(self.train_patient_ids),
                'selection_policy':'sha256_patient_representative_then_sample_v1',
                'successful_updates':self.successful_updates,'ready':self.ready,
                'activation':None if self.activation is None else {'successful_update':self.activation['successful_update'],
                    'samples':[{'key':list(row['key']),'observations':row['observations']} for row in self.activation['samples']]},
                'history':[{'key':list(key),'observations':row['observations'],
                            'last_update':row['last_update'],'mask':row['mask'].clone()}
                           for key,row in self._history.items()]}

    def load_state_dict(self,state):
        required={'version','config','train_patient_ids','successful_updates','ready','history','activation','selection_policy'}
        if not isinstance(state,dict) or set(state)!=required or type(state['version']) is not int or state['version']!=self.STATE_VERSION:
            raise ValueError('invalid bootstrap state schema')
        if state['selection_policy']!='sha256_patient_representative_then_sample_v1':
            raise ValueError('bootstrap anchor selection policy mismatch')
        if state['config']!=asdict(self.config) or state['train_patient_ids']!=list(self.train_patient_ids):
            raise ValueError('bootstrap state/config or training-cohort mismatch')
        updates = state['successful_updates']
        if type(updates) is not int or updates < 0 or type(state['ready']) is not bool:
            raise ValueError('invalid bootstrap update count/readiness')
        rows = state['history']
        if not isinstance(rows,list) or len(rows)>self.config.max_history_samples:
            raise ValueError('invalid bounded bootstrap history')
        history = OrderedDict()
        for row in rows:
            if not isinstance(row,dict) or set(row)!={'key','observations','last_update','mask'}:
                raise ValueError('invalid bootstrap history record')
            key = row['key']
            if not isinstance(key,list) or len(key)!=3 or key[0] not in self.train_patient_ids:
                raise ValueError('invalid bootstrap history identity')
            if any(type(v) is not int or v<0 for v in key[1:]): raise ValueError('invalid bootstrap history indices')
            key = tuple(key)
            if key in history: raise ValueError('duplicate bootstrap history identity')
            obs,last = row['observations'],row['last_update']
            if type(obs) is not int or type(last) is not int or not 1<=obs<=last<=updates:
                raise ValueError('invalid bootstrap history counters')
            self._validate_mask(row['mask'])
            history[key]={'mask':row['mask'].detach().cpu().clone(),'observations':obs,'last_update':last}
        if list(history) != list(self._retain_anchors(history)):
            raise ValueError('bootstrap anchor history is not in canonical identity order')
        # The activation receipt preserves independent support even if those
        # samples later leave the bounded live history. It is artifact metadata,
        # not an anatomical correctness certificate.
        activation = state['activation']
        if state['ready']:
            if not isinstance(activation,dict) or set(activation)!={'successful_update','samples'}:
                raise ValueError('ready state requires an activation evidence receipt')
            step = activation['successful_update']
            if type(step) is not int or not self.config.min_successful_updates<=step<=updates:
                raise ValueError('ready state violates successful-update floor')
            proof = activation['samples']
            if not isinstance(proof,list) or not self.config.min_train_samples<=len(proof)<=self.config.max_history_samples:
                raise ValueError('invalid activation sample support')
            proof_keys = []
            for row in proof:
                if not isinstance(row,dict) or set(row)!={'key','observations'}:
                    raise ValueError('invalid activation sample record')
                key,observations = row['key'],row['observations']
                if not isinstance(key,list) or len(key)!=3 or key[0] not in self.train_patient_ids:
                    raise ValueError('invalid activation sample identity')
                if any(type(v) is not int or v<0 for v in key[1:]):
                    raise ValueError('invalid activation sample indices')
                if type(observations) is not int or not self.config.min_observations_per_sample<=observations<=step:
                    raise ValueError('invalid activation sample observations')
                proof_keys.append(tuple(key))
            if len(set(proof_keys))!=len(proof_keys) or len({key[0] for key in proof_keys})<self.config.min_train_patients:
                raise ValueError('activation lacks distinct training sample/patient support')
            activation={'successful_update':step,'samples':[{'key':list(row['key']),'observations':row['observations']} for row in proof]}
        elif activation is not None:
            raise ValueError('inactive state cannot have activation evidence')
        self.successful_updates,self.ready,self._history = updates,state['ready'],history
        self.activation = activation
