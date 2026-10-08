"""One teacher encoding per batch; bootstrap seeds are not certified by their student."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .evidence import build_region_evidence,EvidenceConfig
from .evolution import PrototypeBank,accepted_region_mask
from .losses_v3 import (teacher_loss,BootstrapLossConfig,spatial_continuity_loss,
                        region_reconstruction_loss)

def region_class_counts(valid, classes, prefix):
    """Support counts, not accuracy; detached diagnostics never select targets."""
    counts=torch.bincount(classes[valid].detach(),minlength=4).cpu().tolist()
    return {f'{prefix}_{name}_regions':int(counts[c]) for c,name in enumerate(('bg','rv','myo','lv'))}

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
    def __init__(self,teacher,optimizer,config=None,evidence_config=None,*,
                 bootstrap_config=None,bootstrap_losses=None,component_config=None,train_patient_ids=()):
        self.teacher=teacher; self.optimizer=optimizer
        self.cfg=config or ProgressiveConfig(); self.evidence_config=evidence_config or EvidenceConfig()
        self.bank=PrototypeBank(4,teacher.region_dim).to(next(teacher.parameters()).device)
        self.raw_evidence_counts=[0]*4; self.accepted_evidence_counts=[0]*4
        self.decodable_evidence_counts=[0]*4; self.reset_epoch_support()
        self.bootstrap=None
        if bootstrap_config is not None:
            from .bootstrap_v3 import BootstrapReadiness
            self.bootstrap=BootstrapReadiness(bootstrap_config,train_patient_ids)
        self.bootstrap_losses=bootstrap_losses or BootstrapLossConfig()
        self.component_config=component_config or {'mode':'components','min_region_pixels':4,
                                                   'max_components':256,'min_probability':.70}
        self.cqa_config=None
        if self.bootstrap_losses.counterfactual_audit is not None:
            from .cqa_v3 import CQAConfig
            self.cqa_config=CQAConfig(**self.bootstrap_losses.counterfactual_audit)
            if self.cqa_config.enabled and (self.bootstrap is None or self.component_config['mode']!='components'):
                raise ValueError('CQA requires explicit bootstrap with confidence-gated components')

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

    def _encode(self,batch):
        base=self.teacher(batch['prev'],batch['cur'],batch['nxt'])
        if self.bootstrap is not None and self.component_config['mode']=='components':
            from .components_v3 import component_regions
            base=component_regions(base,batch['cur'],self.teacher,
                                   **{k:v for k,v in self.component_config.items() if k!='mode'})
            if self.cqa_config is not None and self.cqa_config.enabled:
                from .cqa_v3 import audit_components
                base=audit_components(base,batch,self.teacher,self.cqa_config)
        return base

    def _decode(self,base,output_hw,guided,valid):
        out=self.teacher.decode_evidence(base,output_hw,evidence_logits=guided,evidence_valid=valid,
                                        min_prob=self.cfg.min_prob,min_margin=self.cfg.min_margin)
        if self.bootstrap is not None or 'component_pixel_eligible' in base:
            if 'component_pixel_eligible' in base:
                ids=base['component_pixel_region'];support=base['component_pixel_eligible']
            else:
                confidence,ids=base['region_prob'].detach().max(1)
                support=confidence>=self.bootstrap.config.min_assignment_probability
            owner_valid=out['region_valid'].gather(1,ids.flatten(1)).reshape_as(ids)
            owner_class=guided.argmax(-1).gather(1,ids.flatten(1)).reshape_as(ids)
            support=support&owner_valid
            support=F.interpolate(support[:,None].float(),output_hw,mode='nearest')[:,0].bool()
            owner_class=F.interpolate(owner_class[:,None].float(),output_hw,mode='nearest')[:,0].long()
            # Bilinear probability mixing may otherwise lend accepted anatomy
            # to a rejected neighbouring component, especially at small corners.
            out['valid']=out['valid']&support&(out['soft_label'].argmax(1)==owner_class)
            out['pseudo_label']=torch.where(out['valid'],out['soft_label'].argmax(1),
                                           torch.full_like(out['pseudo_label'],255))
        return out

    def bootstrap_summary(self):
        if self.bootstrap is None: return {'enabled':False,'stage':'legacy'}
        return {'enabled':True,'stage':'semantic' if self.bootstrap.ready else 'anonymous_warmup',
                'ready':self.bootstrap.ready,'successful_updates':self.bootstrap.successful_updates,
                'qualified_samples':self.bootstrap.qualified_samples,'qualified_patients':self.bootstrap.qualified_patients,
                'quality_status':'NOT_CERTIFIED'}

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
        if 'component_eligible' in base: valid &= base['component_eligible']
        if 'component_candidate' in base: valid &= base['component_candidate']
        classes=raw.argmax(-1)
        self.seed_metrics=region_class_counts(valid,classes,'raw_seed')
        self.seed_metrics.update(base.get('cqa_metrics',{}))
        self.seed_metrics['enclosure_pairs']=sum(len(d['enclosures']) for d in diag)
        if self.bootstrap is not None:
            from .bootstrap_v3 import assess_bootstrap_candidates
            self.bootstrap_support=assess_bootstrap_candidates(batch,base['region_prob'],raw,valid,diag,
                self.bootstrap.config,assignment_confidence=base.get('component_confidence'))
            valid &= self.bootstrap_support.region_eligible
            self.seed_metrics.update(region_class_counts(valid,classes,'bootstrap_candidate'))
            self.seed_metrics['bootstrap_candidate_samples']=int(self.bootstrap_support.sample_eligible.sum())
            self.seed_metrics['bootstrap_ready_samples']=batch['cur'].shape[0] if self.bootstrap.ready else 0
            if not self.bootstrap.ready: valid=torch.zeros_like(valid)
        proto=self.bank.logits(base["region_features"].detach(),scale=self.cfg.prototype_scale).detach()
        guided=(raw+proto).detach()
        # Prototype history may corroborate a seed, but cannot invent/rename one.
        valid &= guided.argmax(-1)==raw.argmax(-1)
        if self.bootstrap is not None:
            # A later gate must not turn a credible-pair batch back into a
            # background-only semantic update.
            valid &= (valid&(classes!=0)).any(1,keepdim=True)
        self.seed_metrics.update(region_class_counts(valid,classes,'accepted'))
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
        if self.bootstrap is not None: self.bootstrap.validate_training_batch(batch)
        self.teacher.train()
        base=self._encode(batch)
        raw,guided,raw_valid,diag=self.build_evidence(batch,base)
        valid=self._bootstrap_mask(raw,raw_valid) if self.bootstrap is None else raw_valid
        if not hasattr(self,'seed_metrics'): self.seed_metrics={}
        self.seed_metrics.update(region_class_counts(valid,raw.argmax(-1),'accepted'))
        losses=teacher_loss(base,batch["cur"],raw,valid,w_recon=self.cfg.w_recon,
                            w_proto=self.cfg.w_proto,w_seed=self.cfg.w_seed,w_motion=self.cfg.w_motion,
                            w_motion_smooth=self.cfg.w_motion_smooth,w_region_spatial=self.cfg.w_region_spatial)
        if self.bootstrap is not None:
            q=base.get('anonymous_region_prob',base['region_prob']); image=batch['cur'][:,1:2]
            losses['spatial_continuity']=spatial_continuity_loss(q,image,edge_scale=self.bootstrap_losses.edge_scale)
            losses['region_reconstruction']=region_reconstruction_loss(q,image)
            losses['total']=(losses['total']+self.bootstrap_losses.spatial_weight*losses['spatial_continuity']
                             +self.bootstrap_losses.region_reconstruction_weight*losses['region_reconstruction'])
            if self.cqa_config is not None and self.cqa_config.enabled:
                from .cqa_v3 import relation_distillation_loss
                losses['cqa_relation']=relation_distillation_loss(q,base['cqa_atomic_ids'],base['cqa_relations'],
                    require_both=self.cqa_config.require_both_targets)
                losses['total']=losses['total']+self.cqa_config.distill_weight*losses['cqa_relation']
        if not bool(torch.isfinite(losses["total"])): raise FloatingPointError("non-finite teacher loss")
        self.optimizer.zero_grad(set_to_none=True); losses["total"].backward()
        if not bool(valid.any()):
            # Zero CE gradients still permit AdamW momentum/decay. No named
            # semantic parameter may move on a warmup or unsupported update.
            for p in self.teacher.semantic.parameters(): p.grad=None
        self.optimizer.step()
        # Raw evidence supplies class identity, NOT the network prediction being trained.
        self.bank.update(base["region_features"].detach(),raw.softmax(-1),valid,
                         min_confidence=self.cfg.prototype_min_confidence)
        decodable=self._decodable_mask(base,raw,guided,valid)
        self._record_support(raw,raw_valid,valid,decodable)
        self.last_seed_metrics=dict(self.seed_metrics)
        if self.bootstrap is not None:
            self.bootstrap.observe(batch,self.bootstrap_support)
            self.last_seed_metrics.update(bootstrap_ready_after_update=int(self.bootstrap.ready),
                                          bootstrap_successful_updates=self.bootstrap.successful_updates)
        self.last_metrics = {}
        if collect_metrics:
            # Reuse the pre-update encoding and evidence; no extra forward or GT.
            # Diagnostic decode only: these pixels are NOT the region-loss targets.
            from .progress_v3 import coverage_metrics
            with torch.no_grad():
                decoded = self._decode(base,batch["cur"].shape[-2:],guided,valid)
                self.last_metrics = coverage_metrics(decoded["pseudo_label"],decoded["valid"],prefix="decoded")
                self.last_metrics["candidate_regions"] = int(base["component_candidate"].sum()) if "component_candidate" in base else valid.numel()
                self.last_metrics["foreground_bootstrap_ready"] = bool((self.bank.counts[1:]>0).any())
                for c,name in enumerate(("bg","rv","myo","lv")):
                    self.last_metrics[f"accepted_{name}_regions"] = int((valid & (raw.argmax(-1)==c)).sum())
                    self.last_metrics[f"raw_{name}_regions"] = int((raw_valid & (raw.argmax(-1)==c)).sum())
                self.last_metrics.update(self.last_seed_metrics)
                self.last_metrics.update(region_class_counts(decoded['region_valid'],guided.argmax(-1),'decoded'))
        return {k:float(v.detach().cpu()) for k,v in losses.items()},int(valid.sum()),diag

    @torch.no_grad()
    def infer_batch(self,batch):
        self.teacher.eval()
        base=self._encode(batch)
        raw,guided,raw_valid,diag=self.build_evidence(batch,base)
        valid=self._bootstrap_mask(raw,raw_valid) if self.bootstrap is None else raw_valid
        out=self._decode(base,batch["cur"].shape[-2:],guided,valid)
        out['seed_diagnostics']={**getattr(self,'seed_metrics',{}),
            **region_class_counts(out['region_valid'],guided.argmax(-1),'decoded')}
        return out,guided,diag
