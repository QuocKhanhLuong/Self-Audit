"""Bounded progressive training utilities for the v3 pseudo-label teacher."""
from __future__ import annotations
from dataclasses import dataclass
from .evidence import build_region_evidence, EvidenceConfig
from .evolution import PrototypeBank, accepted_region_mask
from .losses_v3 import teacher_loss

@dataclass(frozen=True)
class ProgressiveConfig:
    min_prob: float=.70
    min_margin: float=.20
    prototype_scale: float=1.5
    prototype_min_confidence: float=.80
    w_recon: float=1.0
    w_proto: float=.1
    w_seed: float=1.0

class ProgressiveTeacherTrainer:
    def __init__(self,teacher,optimizer,config=None,evidence_config=None):
        self.teacher=teacher; self.optimizer=optimizer
        self.cfg=config or ProgressiveConfig(); self.evidence_config=evidence_config or EvidenceConfig()
        self.bank=PrototypeBank(4,teacher.region_dim).to(next(teacher.parameters()).device)

    def _axes(self,batch):
        axes=batch.get("patient_left_axis")
        if axes is None: return [None]*batch["cur"].shape[0]
        if isinstance(axes,str): return [axes]*batch["cur"].shape[0]
        return list(axes)

    def build_evidence(self,batch,base):
        raw,diag=build_region_evidence(base["region_prob"],batch["cur"][:,1:2],base["motion_features"],
            patient_left_axis=self._axes(batch),config=self.evidence_config)
        proto=self.bank.logits(base["region_features"].detach(),scale=self.cfg.prototype_scale)
        evidence=(raw+proto).detach()
        valid=accepted_region_mask(evidence.softmax(-1),min_prob=self.cfg.min_prob,min_margin=self.cfg.min_margin)
        return evidence,valid,diag

    def train_batch(self,batch):
        prev,cur,nxt=batch["prev"],batch["cur"],batch["nxt"]
        base=self.teacher(prev,cur,nxt,min_prob=self.cfg.min_prob,min_margin=self.cfg.min_margin)
        evidence,valid,diag=self.build_evidence(batch,base)
        out=self.teacher(prev,cur,nxt,evidence_logits=evidence,min_prob=self.cfg.min_prob,min_margin=self.cfg.min_margin)
        losses=teacher_loss(out,cur,evidence,valid,w_recon=self.cfg.w_recon,w_proto=self.cfg.w_proto,w_seed=self.cfg.w_seed)
        self.optimizer.zero_grad(set_to_none=True); losses["total"].backward(); self.optimizer.step()
        with __import__("torch").no_grad():
            self.bank.update(out["region_features"].detach(),out["semantic_prob"].detach(),valid,
                min_confidence=self.cfg.prototype_min_confidence)
        return {k:float(v.detach().cpu()) for k,v in losses.items()},int(valid.sum().item()),diag

    def infer_batch(self,batch):
        import torch
        with torch.no_grad():
            prev,cur,nxt=batch["prev"],batch["cur"],batch["nxt"]
            base=self.teacher(prev,cur,nxt,min_prob=self.cfg.min_prob,min_margin=self.cfg.min_margin)
            evidence,_,diag=self.build_evidence(batch,base)
            out=self.teacher(prev,cur,nxt,evidence_logits=evidence,min_prob=self.cfg.min_prob,min_margin=self.cfg.min_margin)
            return out,evidence,diag
