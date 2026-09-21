"""One teacher encoding per batch; bootstrap seeds are not certified by their student."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .evidence import build_region_evidence,EvidenceConfig
from .evolution import PrototypeBank,accepted_region_mask
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
    w_motion: float=1.0
    w_motion_smooth: float=.05

class ProgressiveTeacherTrainer:
    def __init__(self,teacher,optimizer,config=None,evidence_config=None):
        self.teacher=teacher; self.optimizer=optimizer
        self.cfg=config or ProgressiveConfig(); self.evidence_config=evidence_config or EvidenceConfig()
        self.bank=PrototypeBank(4,teacher.region_dim).to(next(teacher.parameters()).device)

    def build_evidence(self,batch,base):
        axes=batch.get("patient_left_axis")
        if isinstance(axes,str): axes=[axes]*batch["cur"].shape[0]
        # Use displacement fields, not arbitrary learned-feature magnitude, as motion evidence.
        motion=torch.cat([base["flow_prev"],base["flow_next"]],dim=1)
        raw,diag=build_region_evidence(base["region_prob"],batch["cur"][:,1:2],motion,
                                      patient_left_axis=axes,config=self.evidence_config)
        valid=accepted_region_mask(raw.softmax(-1),min_prob=self.cfg.min_prob,min_margin=self.cfg.min_margin)
        hard=base["region_prob"].detach().argmax(1)
        counts=torch.stack([(hard==k).sum((-2,-1)) for k in range(raw.shape[1])],1)
        valid &= counts>=self.evidence_config.min_region_pixels
        proto=self.bank.logits(base["region_features"].detach(),scale=self.cfg.prototype_scale).detach()
        guided=(raw+proto).detach()
        # Prototype history may corroborate a seed, but cannot invent/rename one.
        valid &= guided.argmax(-1)==raw.argmax(-1)
        return raw.detach(),guided,valid,diag

    def train_batch(self,batch):
        self.teacher.train()
        base=self.teacher(batch["prev"],batch["cur"],batch["nxt"])
        raw,_,valid,diag=self.build_evidence(batch,base)
        losses=teacher_loss(base,batch["cur"],raw,valid,w_recon=self.cfg.w_recon,
                            w_proto=self.cfg.w_proto,w_seed=self.cfg.w_seed,w_motion=self.cfg.w_motion,
                            w_motion_smooth=self.cfg.w_motion_smooth)
        if not bool(torch.isfinite(losses["total"])): raise FloatingPointError("non-finite teacher loss")
        self.optimizer.zero_grad(set_to_none=True); losses["total"].backward(); self.optimizer.step()
        # Raw evidence supplies class identity, NOT the network prediction being trained.
        self.bank.update(base["region_features"].detach(),raw.softmax(-1),valid,
                         min_confidence=self.cfg.prototype_min_confidence)
        return {k:float(v.detach().cpu()) for k,v in losses.items()},int(valid.sum()),diag

    @torch.no_grad()
    def infer_batch(self,batch):
        self.teacher.eval()
        base=self.teacher(batch["prev"],batch["cur"],batch["nxt"])
        _,guided,valid,diag=self.build_evidence(batch,base)
        out=self.teacher.decode_evidence(base,batch["cur"].shape[-2:],evidence_logits=guided,
                evidence_valid=valid,min_prob=self.cfg.min_prob,min_margin=self.cfg.min_margin)
        return out,guided,diag
