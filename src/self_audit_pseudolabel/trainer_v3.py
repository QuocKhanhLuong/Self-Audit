"""One teacher encoding per batch; bootstrap seeds are not certified by their student."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .evidence import (build_region_evidence,EvidenceConfig,
                       split_connected_region_components)
from .evolution import PrototypeBank,accepted_region_mask
from .losses_v3 import teacher_loss
from .system_v3 import pool_regions

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
    w_region_spatial: float=.05

class ProgressiveTeacherTrainer:
    def __init__(self,teacher,optimizer,config=None,evidence_config=None):
        self.teacher=teacher; self.optimizer=optimizer
        self.cfg=config or ProgressiveConfig(); self.evidence_config=evidence_config or EvidenceConfig()
        self.bank=PrototypeBank(4,teacher.region_dim).to(next(teacher.parameters()).device)
        self.raw_evidence_counts=[0]*4; self.accepted_evidence_counts=[0]*4
        self.decodable_evidence_counts=[0]*4; self.reset_epoch_support()

    def reset_epoch_support(self):
        self.epoch_raw_evidence_counts=[0]*4
        self.epoch_accepted_evidence_counts=[0]*4
        self.epoch_decodable_evidence_counts=[0]*4

    @staticmethod
    def _counts(classes,mask):
        if not bool(mask.any()): return [0]*4
        return [int(x) for x in torch.bincount(classes[mask],minlength=4).detach().cpu().tolist()]

    @staticmethod
    def _accumulate(target,values):
        for i,value in enumerate(values): target[i]+=int(value)

    def _component_view(self,batch,base):
        regions,candidates=split_connected_region_components(
            base["region_prob"],min_region_pixels=self.evidence_config.min_region_pixels,
            max_components_per_prototype=self.evidence_config.max_components_per_prototype)
        features=pool_regions(regions,base["fused_features"],batch["cur"][:,1:2],base["motion_features"])
        logits=self.teacher.semantic(features)
        return {**base,"region_prob":regions,"region_features":features,
                "semantic_logits":logits,"semantic_prob":logits.softmax(-1),
                "component_candidate":candidates}

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
        if "component_candidate" in base: valid &= base["component_candidate"]
        proto=self.bank.logits(base["region_features"].detach(),scale=self.cfg.prototype_scale).detach()
        guided=(raw+proto).detach()
        # Prototype history may corroborate a seed, but cannot invent/rename one.
        valid &= guided.argmax(-1)==raw.argmax(-1)
        return raw.detach(),guided,valid,diag

    def _bootstrap_mask(self,raw,valid):
        classes=raw.argmax(-1); foreground=valid&(classes!=0)
        ready=bool((self.bank.counts[1:]>0).any())
        return valid if ready or bool(foreground.any()) else foreground

    def _decodable_mask(self,base,raw,guided,valid):
        probability=(base["semantic_logits"].detach()+guided).softmax(-1)
        top=probability.topk(2,-1).values
        reliable=(top[...,0]>=self.cfg.min_prob)&((top[...,0]-top[...,1])>=self.cfg.min_margin)
        return valid&reliable&(probability.argmax(-1)==raw.argmax(-1))

    def _record_support(self,raw,raw_valid,accepted,decodable):
        classes=raw.argmax(-1)
        for total,epoch,mask in (
            (self.raw_evidence_counts,self.epoch_raw_evidence_counts,raw_valid),
            (self.accepted_evidence_counts,self.epoch_accepted_evidence_counts,accepted),
            (self.decodable_evidence_counts,self.epoch_decodable_evidence_counts,decodable)):
            counts=self._counts(classes,mask); self._accumulate(total,counts); self._accumulate(epoch,counts)

    def train_batch(self,batch,*,collect_metrics=False):
        self.teacher.train()
        base=self.teacher(batch["prev"],batch["cur"],batch["nxt"])
        semantic_base=self._component_view(batch,base)
        raw,guided,raw_valid,diag=self.build_evidence(batch,semantic_base)
        valid=self._bootstrap_mask(raw,raw_valid)
        loss_base={**base,"semantic_logits":semantic_base["semantic_logits"]}
        losses=teacher_loss(loss_base,batch["cur"],raw,valid,w_recon=self.cfg.w_recon,
                            w_proto=self.cfg.w_proto,w_seed=self.cfg.w_seed,w_motion=self.cfg.w_motion,
                            w_motion_smooth=self.cfg.w_motion_smooth,w_region_spatial=self.cfg.w_region_spatial)
        if not bool(torch.isfinite(losses["total"])): raise FloatingPointError("non-finite teacher loss")
        self.optimizer.zero_grad(set_to_none=True); losses["total"].backward()
        # A connected zero loss would still let AdamW decay an unbootstrapped
        # semantic head.  Keep those gradients absent until foreground exists.
        if not bool(valid.any()):
            for parameter in self.teacher.semantic.parameters(): parameter.grad=None
        self.optimizer.step()
        # Raw evidence supplies class identity, NOT the network prediction being trained.
        self.bank.update(semantic_base["region_features"].detach(),raw.softmax(-1),valid,
                         min_confidence=self.cfg.prototype_min_confidence)
        decodable=self._decodable_mask(semantic_base,raw,guided,valid)
        self._record_support(raw,raw_valid,valid,decodable)
        self.last_metrics = {}
        if collect_metrics:
            # Reuse the pre-update encoding and evidence; no extra forward or GT.
            # Diagnostic decode only: these pixels are NOT the region-loss targets.
            from .progress_v3 import coverage_metrics
            with torch.no_grad():
                decoded = self.teacher.decode_evidence(semantic_base,batch["cur"].shape[-2:],
                    evidence_logits=guided,evidence_valid=valid,
                    min_prob=self.cfg.min_prob,min_margin=self.cfg.min_margin)
                self.last_metrics = coverage_metrics(decoded["pseudo_label"],decoded["valid"],prefix="decoded")
                self.last_metrics["candidate_regions"] = int(semantic_base["component_candidate"].sum())
                self.last_metrics["foreground_bootstrap_ready"] = bool((self.bank.counts[1:]>0).any())
                for c,name in enumerate(("bg","rv","myo","lv")):
                    self.last_metrics[f"accepted_{name}_regions"] = int((valid & (raw.argmax(-1)==c)).sum())
                    self.last_metrics[f"raw_{name}_regions"] = int((raw_valid & (raw.argmax(-1)==c)).sum())
        return {k:float(v.detach().cpu()) for k,v in losses.items()},int(valid.sum()),diag

    @torch.no_grad()
    def infer_batch(self,batch):
        self.teacher.eval()
        base=self.teacher(batch["prev"],batch["cur"],batch["nxt"])
        semantic_base=self._component_view(batch,base)
        raw,guided,raw_valid,diag=self.build_evidence(batch,semantic_base)
        valid=self._bootstrap_mask(raw,raw_valid)
        out=self.teacher.decode_evidence(semantic_base,batch["cur"].shape[-2:],evidence_logits=guided,
                evidence_valid=valid,min_prob=self.cfg.min_prob,min_margin=self.cfg.min_margin)
        return out,guided,diag
