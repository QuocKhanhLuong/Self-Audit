"""Opt-in supervised Read--Evaluate--Write (REW), not the mask-free pipeline.

One shared writer, three reads (local / local+near / local+wide), exact KEEP,
tile selection and an independent audit of the actual assembled output.
GT is not accepted by any inference API. Candidate C and legacy models are
unchanged; their replay/calibration/checkpoints do not certify this model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .encoder import ConvNeXtTinyEncoder
from .fpn import LightweightFPN
from .annotation_head import InitialAnnotationHead

CC, FIX, REGRESS, WW = range(4)
READ_NAMES = ("local", "near", "wide")


def _block(inputs: int, channels: int) -> nn.Sequential:
    groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
    return nn.Sequential(nn.Conv2d(inputs, channels, 1), nn.GroupNorm(groups, channels), nn.GELU())


def _resize(x: Tensor, size: tuple[int, int]) -> Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _entropy(p: Tensor) -> Tensor:
    return -(p * p.clamp_min(1e-8).log()).sum(1, keepdim=True) / math.log(p.shape[1])


@dataclass
class Outcome:
    local_logits: Tensor
    probabilities: Tensor
    delta_q: Tensor

    @property
    def candidate_correct(self) -> Tensor:
        return self.probabilities[:, CC:CC+1] + self.probabilities[:, FIX:FIX+1]


class OutcomeAuditor(nn.Module):
    """Four correctness transitions plus signed slice-Dice change.

    Inputs are explicitly LOGITS, never guessed from their numeric range.
    Actor features and all predictions are detached here. Identity state audits
    distinguish CC from WW; identical hard predictions have exact delta_q=0.
    """
    def __init__(self, channels: int, classes: int = 4) -> None:
        super().__init__()
        self.classes = classes
        self.trunk = nn.Sequential(
            _block(channels + 3 * classes + 2, channels),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
        )
        self.local_head = nn.Conv2d(channels, 4, 1)
        self.global_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 1))

    def forward(self, features: Tensor, previous: Tensor, candidate: Tensor) -> Outcome:
        if previous.shape != candidate.shape or previous.ndim != 4 or previous.shape[1] != self.classes:
            raise ValueError("auditor expects equal [B,4,H,W] logits")
        size = features.shape[-2:]
        p = previous.detach().float().softmax(1)
        c = candidate.detach().float().softmax(1)
        pl, cl = _resize(p, size), _resize(c, size)
        x = self.trunk(torch.cat([features.detach(), pl, cl, cl-pl, _entropy(pl), _entropy(cl)], 1))
        logits = _resize(self.local_head(x).float(), previous.shape[-2:])
        same = previous.detach().argmax(1) == candidate.detach().argmax(1)
        # Both-correct is impossible for unequal labels; FIX/REGRESS impossible
        # for equal labels. These masks use predictions only, not GT.
        allowed = torch.stack([same, ~same, ~same, torch.ones_like(same)], 1)
        logits = logits.masked_fill(~allowed, -1e4)
        change = self.global_head(x).float().squeeze(1)
        change = torch.where(same.flatten(1).all(1), torch.zeros_like(change), change)
        return Outcome(logits, logits.softmax(1), change)

    def state(self, features: Tensor, logits: Tensor) -> Outcome:
        return self(features, logits, logits)


class SharedReadWriter(nn.Module):
    """Pair-message reader without QK dot products; shared parameters for all reads.

    Eight samples per query. Local uses a fixed compass; near/wide use four
    fixed anchors and four ellipse points. Audit enters the coordinate generator
    only. Pixel-coordinate bounds keep ellipses inside the image without
    independently clipping their points. Fixed anchors use border padding.
    """
    def __init__(self, channels: int = 96, classes: int = 4, max_radius: float = 8.0,
                 checkpoint_reads: bool = True) -> None:
        super().__init__()
        if max_radius < 1 or not math.isfinite(max_radius):
            raise ValueError("max_radius must be finite and >=1 feature pixel")
        self.channels, self.classes = channels, classes
        self.max_radius, self.checkpoint_reads = float(max_radius), bool(checkpoint_reads)
        self.content = _block(channels + classes + 1, channels)
        self.geometry = nn.Sequential(_block(channels + 4, max(channels//2, 16)),
                                      nn.Conv2d(max(channels//2, 16), 10, 1))
        # Nonzero last weights allow geometry conditioning to learn immediately.
        nn.init.normal_(self.geometry[-1].weight, std=0.01)
        nn.init.zeros_(self.geometry[-1].bias)
        with torch.no_grad():
            for branch, radius in enumerate((1.5, 3.0)):
                proportion = min(0.95, max(0.05, (radius-0.5)/(max_radius-0.5)))
                self.geometry[-1].bias[branch*5+2:branch*5+4] = math.log(proportion/(1-proportion))
        self.message = nn.Sequential(_block(2*channels + 2*classes + 2, channels),
                                     nn.Conv2d(channels, channels, 1), nn.GELU())
        self.refine = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(channels, channels, 1))
        self.delta = nn.Conv2d(channels, classes, 1)
        self.gate = nn.Conv2d(channels, 1, 1)
        self.turn_embedding = nn.Embedding(3, channels)
        self.register_buffer("compass", torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],
                                                      [1.,1.],[-1.,1.],[-1.,-1.],[1.,-1.]]), persistent=True)

    @staticmethod
    def _base(x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=torch.float32),
                                torch.arange(w, device=x.device, dtype=torch.float32), indexing="ij")
        return torch.stack([xx, yy], -1)[None]

    def coordinates(self, z: Tensor, feedback: Tensor) -> tuple[Tensor, ...]:
        b, _, h, w = z.shape
        base = self._base(z)
        local = (base[..., None, :] + self.compass).expand(b, -1, -1, -1, -1)
        raw = self.geometry(torch.cat([z, feedback.detach().to(z.dtype)], 1)).float()
        reads = [local]
        for branch in range(2):
            r = raw[:, branch*5:(branch+1)*5].permute(0,2,3,1)
            radius = 0.5 + (self.max_radius-0.5) * r[..., 2:4].sigmoid()
            theta = math.pi * r[..., 4].tanh()
            cos, sin = theta.cos(), theta.sin()
            # Fit the entire rotated ellipse in the image. It can degenerate
            # on size-one axes, which is reported rather than faking diversity.
            extx = ((radius[...,0]*cos)**2 + (radius[...,1]*sin)**2 + 1e-12).sqrt()
            exty = ((radius[...,0]*sin)**2 + (radius[...,1]*cos)**2 + 1e-12).sqrt()
            scale = torch.minimum(torch.ones_like(extx), torch.minimum((w-1)/2/extx, (h-1)/2/exty))
            radius = radius * scale[..., None]
            extx, exty = extx*scale, exty*scale
            ext = torch.stack([extx, exty], -1)
            upper = z.new_tensor([w-1., h-1.], dtype=torch.float32) - ext
            center = base + 2.0*r[..., :2].tanh()
            center = torch.maximum(ext, torch.minimum(center, upper))
            unit = self.compass[:4].to(radius)
            xy = unit * radius[..., None, :]
            rotated = torch.stack([xy[...,0]*cos[...,None]-xy[...,1]*sin[...,None],
                                   xy[...,0]*sin[...,None]+xy[...,1]*cos[...,None]], -1)
            probes = center[..., None, :] + rotated
            reads.append(torch.cat([local[..., :4, :], probes], -2))
        return tuple(reads)

    def _write(self, z: Tensor, probabilities: Tensor, coordinates: Tensor) -> Tensor:
        h, w = z.shape[-2:]
        # Float32 sampling avoids half-precision coordinate rounding; convolution
        # layers still follow the caller's explicit autocast policy.
        xy = coordinates.float()
        gx = 2*xy[...,0]/max(w-1,1)-1 if w > 1 else xy[...,0]*0
        gy = 2*xy[...,1]/max(h-1,1)-1 if h > 1 else xy[...,1]*0
        grid = torch.stack([gx,gy], -1)
        source = torch.cat([z.float(), probabilities.float()], 1)
        base = self._base(z)
        aggregate = None
        for k in range(8):
            sampled = F.grid_sample(source, grid[...,k,:], mode="bilinear", padding_mode="border", align_corners=True)
            relative = (xy[...,k,:]-base).permute(0,3,1,2) / self.max_radius
            pair = torch.cat([z, sampled[:,:self.channels]-z,
                              probabilities, sampled[:,self.channels:], relative], 1)
            message = self.message(pair)
            aggregate = message if aggregate is None else aggregate+message
        state = z + self.refine(aggregate / 8.0)
        return torch.sigmoid(self.gate(state)) * self.delta(state)

    def forward(self, features: Tensor, annotation: Tensor, feedback: Tensor, turn: int) -> tuple[list[Tensor], tuple[Tensor,...]]:
        p = _resize(annotation.float().softmax(1), features.shape[-2:])
        z = self.content(torch.cat([features, p, _entropy(p)], 1))
        z = z + self.turn_embedding.weight[turn].to(z.dtype)[None,:,None,None]
        reads = self.coordinates(z, _resize(feedback.detach().float(), z.shape[-2:]))
        candidates = []
        for coordinates in reads:
            if self.checkpoint_reads and self.training and torch.is_grad_enabled():
                delta = checkpoint(self._write, z, p, coordinates, use_reentrant=False)
            else:
                delta = self._write(z, p, coordinates)
            candidates.append(annotation + _resize(delta.float(), annotation.shape[-2:]))
        return candidates, reads


def select_tiles(previous: Tensor, candidates: list[Tensor], outcomes: list[Outcome],
                 tile: int, harm_weight: float, min_gain: float) -> tuple[Tensor, Tensor]:
    """Select one candidate per nonoverlapping tile, or exact KEEP; no averaging."""
    if not candidates or len(candidates) != len(outcomes) or tile < 1:
        raise ValueError("invalid candidate bank or tile size")
    with torch.no_grad():
        h,w = previous.shape[-2:]
        pad = (0,(-w)%tile,0,(-h)%tile)
        support = F.pad(torch.ones_like(previous[:,:1]),pad)
        count = F.avg_pool2d(support,tile,tile)
        values = []
        for o in outcomes:
            utility = o.probabilities[:,FIX:FIX+1] - harm_weight*o.probabilities[:,REGRESS:REGRESS+1]
            values.append(F.avg_pool2d(F.pad(utility,pad),tile,tile)/count)
        keep = torch.full_like(values[0], min_gain)
        choice = torch.cat([keep,*values],1).argmax(1,keepdim=True)
        full = choice.repeat_interleave(tile,2).repeat_interleave(tile,3)[...,:h,:w]
    bank = torch.stack([previous,*candidates],1)
    selected = bank.gather(1, full.unsqueeze(2).expand(-1,1,previous.shape[1],-1,-1)).squeeze(1)
    return selected, full.squeeze(1)


class ReadEvaluateWriteNet(nn.Module):
    """Experimental supervised model. A forward never accepts ground truth."""
    def __init__(self, channels: int = 96, max_turns: int = 3, tile_size: int = 32,
                 harm_weight: float = 2., min_gain: float = 0.01, tau_accept: float = 0.,
                 pretrained: bool = True, checkpoint_reads: bool = True,
                 encoder: nn.Module | None = None, fpn: nn.Module | None = None) -> None:
        super().__init__()
        if max_turns not in (1,2,3) or tile_size < 1 or harm_weight < 0 or min_gain < 0:
            raise ValueError("invalid REW controls")
        self.encoder = encoder if encoder is not None else ConvNeXtTinyEncoder(pretrained=pretrained, allow_fallback=False)
        self.fpn = fpn if fpn is not None else LightweightFPN(self.encoder.out_channels, channels)
        self.initial_head = InitialAnnotationHead(channels,4)
        self.annotation_expert = SharedReadWriter(channels, checkpoint_reads=checkpoint_reads)
        self.auditor = OutcomeAuditor(channels)
        self.max_turns, self.tile_size = max_turns, tile_size
        self.harm_weight, self.min_gain, self.tau_accept = harm_weight, min_gain, tau_accept

    def forward(self, image: Tensor) -> dict:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("REW requires [B,3,H,W] input")
        features = self.fpn(self.encoder(image))
        initial = self.initial_head(features, image.shape[-2:]).float()
        state = initial
        state_outcome = self.auditor.state(features, initial)
        feedback = torch.cat([1-state_outcome.candidate_correct.detach(), torch.zeros_like(initial[:,:3])],1)
        active = torch.ones(image.shape[0],dtype=torch.bool,device=image.device)
        turns = []
        for turn in range(self.max_turns):
            ids = active.nonzero().flatten()
            if not ids.numel():
                break
            prev = state.index_select(0,ids)
            feat = features.index_select(0,ids)
            proposals, coordinates = self.annotation_expert(feat,prev,feedback.index_select(0,ids),turn)
            outcomes = [self.auditor(feat,prev,c) for c in proposals]
            assembled, choice = select_tiles(prev,proposals,outcomes,self.tile_size,self.harm_weight,self.min_gain)
            final_audit = self.auditor(feat,prev,assembled)
            accepted = final_audit.delta_q.detach() > self.tau_accept
            changed = (assembled.detach().argmax(1) != prev.detach().argmax(1)).flatten(1).any(1)
            accepted = accepted & changed
            retained = torch.where(accepted[:,None,None,None],assembled,prev)
            state = state.index_copy(0,ids,retained)
            next_feedback = torch.cat([1-final_audit.candidate_correct.detach(),
                final_audit.probabilities[:,FIX:FIX+1].detach(),
                final_audit.probabilities[:,REGRESS:REGRESS+1].detach(),
                torch.ones_like(prev[:,:1])],1)
            feedback = feedback.index_copy(0,ids,torch.where(accepted[:,None,None,None],next_feedback,feedback.index_select(0,ids)))
            active = active.clone();active[ids] = accepted
            turns.append(dict(ids=ids,previous=prev,candidates=proposals,outcomes=outcomes,
                              assembled=assembled,assembled_outcome=final_audit,accepted=accepted,
                              choice=choice,features=feat,turn=turn))
        return dict(logits=state,initial=initial,features=features,state_outcome=state_outcome,turns=turns)
