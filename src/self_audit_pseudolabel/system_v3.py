"""Integrated research scaffold for Self-Audit v3.

Offline teacher and deployment annotator are intentionally separated.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

NUM_CLASSES=4
UNKNOWN=255

def _gn(c):
    for g in (8,4,2):
        if c%g==0 and c//g>=2: return nn.GroupNorm(g,c)
    return nn.GroupNorm(1,c)

class Block(nn.Module):
    def __init__(self,ci,co,stride=1):
        super().__init__(); self.net=nn.Sequential(
            nn.Conv2d(ci,co,3,stride=stride,padding=1,bias=False),_gn(co),nn.GELU(),
            nn.Conv2d(co,co,3,padding=1,bias=False),_gn(co),nn.GELU())
    def forward(self,x): return self.net(x)

class AppearanceEncoder(nn.Module):
    def __init__(self,width=24,dim=48):
        super().__init__(); self.e0=Block(3,width); self.e1=Block(width,width*2,2); self.e2=Block(width*2,width*4,2)
        self.out=nn.Conv2d(width*4,dim,1); self.recon=nn.Conv2d(dim,1,1)
    def forward(self,x):
        h=self.e2(self.e1(self.e0(x))); f=self.out(h)
        return f,self.recon(F.interpolate(f,size=x.shape[-2:],mode="bilinear",align_corners=False))

class MotionBranch(nn.Module):
    """Lightweight unsupervised pairwise registration + motion descriptor."""
    def __init__(self,dim=16,max_disp=.15):
        super().__init__(); self.max_disp=float(max_disp)
        hidden=max(dim,16)
        self.flow=nn.Sequential(
            nn.Conv2d(2,hidden,3,padding=1,bias=False),_gn(hidden),nn.GELU(),
            nn.Conv2d(hidden,hidden,3,stride=2,padding=1,bias=False),_gn(hidden),nn.GELU(),
            nn.Conv2d(hidden,hidden,3,stride=2,padding=1,bias=False),_gn(hidden),nn.GELU(),
            nn.Conv2d(hidden,2,3,padding=1))
        self.feat=nn.Sequential(nn.Conv2d(6,dim,3,padding=1,bias=False),_gn(dim),nn.GELU())
    @staticmethod
    def _grid(h,w,device,dtype):
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=device,dtype=dtype),
                             torch.linspace(-1,1,w,device=device,dtype=dtype),indexing="ij")
        return torch.stack([xx,yy],-1)[None]
    def _register(self,src,cur):
        low=torch.tanh(self.flow(torch.cat([src,cur],1)))*self.max_disp
        flow=F.interpolate(low,size=cur.shape[-2:],mode="bilinear",align_corners=False)
        grid=self._grid(cur.shape[-2],cur.shape[-1],cur.device,cur.dtype)+flow.permute(0,2,3,1)
        warped=F.grid_sample(src,grid,mode="bilinear",padding_mode="border",align_corners=True)
        return low,warped
    def forward(self,prev,cur,nxt):
        p,c,n=prev[:,1:2],cur[:,1:2],nxt[:,1:2]
        fp,wp=self._register(p,c); fn,wn=self._register(n,c)
        c_low=F.interpolate(c,size=fp.shape[-2:],mode="bilinear",align_corners=False)
        wp_low=F.interpolate(wp,size=fp.shape[-2:],mode="bilinear",align_corners=False)
        wn_low=F.interpolate(wn,size=fp.shape[-2:],mode="bilinear",align_corners=False)
        descriptor=torch.cat([fp,fn,(c_low-wp_low).abs(),(c_low-wn_low).abs()],1)
        return self.feat(descriptor),{"flow_prev":fp,"flow_next":fn,"warped_prev":wp,"warped_next":wn}

class RegionPrototypeHead(nn.Module):
    def __init__(self,dim,k=12,temperature=.1):
        super().__init__(); self.p=nn.Parameter(torch.randn(k,dim)*.02); self.temperature=float(temperature)
    def forward(self,f):
        f=F.normalize(f,dim=1); p=F.normalize(self.p,dim=1)
        return (torch.einsum("bchw,kc->bkhw",f,p)/self.temperature).softmax(1)

def pool_regions(assign,feat,image,motion):
    _,_,h,w=assign.shape; mass=assign.sum((-2,-1)).clamp_min(1e-6)
    fv=torch.einsum("bkhw,bchw->bkc",assign,feat)/mass[...,None]
    img=F.interpolate(image,(h,w),mode="bilinear",align_corners=False)
    mot=F.interpolate((motion.pow(2).mean(1,keepdim=True)+1e-12).sqrt(),(h,w),mode="bilinear",align_corners=False)
    inten=torch.einsum("bkhw,bchw->bkc",assign,img)/mass[...,None]
    mov=torch.einsum("bkhw,bchw->bkc",assign,mot)/mass[...,None]
    area=(mass/float(h*w))[...,None]
    return torch.cat([fv,inten,mov,area],-1)

class CinePseudoTeacher(nn.Module):
    """Offline teacher. Named output is accepted only through reliability gates."""
    def __init__(self,width=24,appearance_dim=48,motion_dim=16,fused_dim=64,k=12):
        super().__init__(); self.appearance=AppearanceEncoder(width,appearance_dim); self.motion=MotionBranch(motion_dim)
        self.fuse=nn.Sequential(nn.Conv2d(appearance_dim+motion_dim,fused_dim,1,bias=False),_gn(fused_dim),nn.GELU())
        self.regions=RegionPrototypeHead(fused_dim,k); self.region_dim=fused_dim+3
        self.semantic=nn.Sequential(nn.Linear(self.region_dim,96),nn.GELU(),nn.Linear(96,NUM_CLASSES))
    def forward(self,prev,cur,nxt,*,evidence_logits=None,min_prob=.70,min_margin=.20):
        if cur.ndim != 4 or cur.shape[1] != 3 or min(cur.shape[-2:]) < 4:
            raise ValueError("cine context must be [B,3,H,W], H/W >= 4")
        if prev.shape != cur.shape or nxt.shape != cur.shape:
            raise ValueError("temporal context shapes must match")
        if not all(bool(torch.isfinite(x).all()) for x in (prev,cur,nxt)):
            raise ValueError("non-finite cine input")
        app,recon=self.appearance(cur); motion,motion_aux=self.motion(prev,cur,nxt)
        fused=self.fuse(torch.cat([app,motion],1)); q=self.regions(fused)
        r=pool_regions(q,fused,cur[:,1:2],motion); logits=self.semantic(r)
        base={"region_prob":q,"region_features":r,"semantic_logits":logits,
              "semantic_prob":logits.softmax(-1),"reconstruction":recon,
              "appearance_features":app,"motion_features":motion,**motion_aux}
        return self.decode_evidence(base,cur.shape[-2:],evidence_logits=evidence_logits,
                                    min_prob=min_prob,min_margin=min_margin)

    def decode_evidence(self,base,output_hw,*,evidence_logits=None,evidence_valid=None,
                        min_prob=.70,min_margin=.20):
        """Reuse encoded tensors. Scores are not calibrated correctness probabilities.

        The neural semantic logits never include their own supervision target.
        No external image-only evidence means exact abstention, even for a
        spuriously confident neural head. UNKNOWN is not a trainable class.
        """
        if not 0 <= min_prob <= 1 or not 0 <= min_margin <= 1:
            raise ValueError("probability/margin thresholds must be in [0,1]")
        logits=base["semantic_logits"]; q=base["region_prob"]
        if evidence_logits is None:
            prob=logits.softmax(-1)
            valid_region=torch.zeros(logits.shape[:2],device=logits.device,dtype=torch.bool)
        else:
            ev=evidence_logits.detach()
            if ev.shape != logits.shape or not bool(torch.isfinite(ev).all()):
                raise ValueError("finite evidence_logits must have shape [B,K,4]")
            evidence_prob=ev.softmax(-1); top=evidence_prob.topk(2,-1).values
            valid_region=(top[...,0]>=min_prob)&((top[...,0]-top[...,1])>=min_margin)
            if evidence_valid is not None:
                if evidence_valid.shape != valid_region.shape or evidence_valid.dtype != torch.bool:
                    raise ValueError("evidence_valid must be bool [B,K]")
                valid_region &= evidence_valid
            prob=(logits+ev).softmax(-1)
            # A neural guess may not override a contradictory accepted seed.
            valid_region &= prob.argmax(-1)==evidence_prob.argmax(-1)
        weights=q*valid_region[:,:,None,None].to(q.dtype)
        mass=weights.sum(1,keepdim=True)
        accepted=torch.einsum("bkhw,bkc->bchw",weights,prob)/mass.clamp_min(1e-8)
        fallback=torch.einsum("bkhw,bkc->bchw",q,prob)
        dense_low=torch.where(mass>1e-8,accepted,fallback)
        dense=F.interpolate(dense_low,output_hw,mode="bilinear",align_corners=False)
        dense_mass=F.interpolate(mass,output_hw,mode="bilinear",align_corners=False)[:,0]
        top=dense.topk(2,dim=1).values
        valid=(dense_mass>=.5)&(top[:,0]>=min_prob)&((top[:,0]-top[:,1])>=min_margin)
        labels=dense.argmax(1)
        return {**base,"raw_semantic_prob":logits.softmax(-1),
                "semantic_prob":prob,"guided_semantic_prob":prob,"region_valid":valid_region,
                "soft_label":dense,"valid":valid,
                "pseudo_label":torch.where(valid,labels,torch.full_like(labels,UNKNOWN))}

@dataclass(frozen=True)
class ResourceProfile:
    name:str
    turns:int

PROFILES={"compact":ResourceProfile("compact",0),"balanced":ResourceProfile("balanced",1),"accurate":ResourceProfile("accurate",2)}

class DeploymentEncoder(nn.Module):
    def __init__(self,width=32):
        super().__init__(); self.net=nn.Sequential(Block(3,width),Block(width,width,2),Block(width,width,2))
    def forward(self,x): return self.net(x)

class AdaptiveAnnotationStudent(nn.Module):
    """Final model: shared encoder -> A0 -> optional Dynamic Window turns."""
    def __init__(self,width=32,window_k=8):
        from self_audit.models.annotation_expert import AnnotationExpert
        super().__init__(); self.encoder=DeploymentEncoder(width); self.a0_head=nn.Conv2d(width,NUM_CLASSES,1)
        self.refiner=AnnotationExpert(feature_channels=width,num_classes=NUM_CLASSES,audit_channels=3,
            window_k=window_k,max_turns=2,audit_conditioning="feature_only",offset_mode="structured")
    def encode(self,x): return self.encoder(x)
    def initial_logits(self,feat,output_hw):
        return F.interpolate(self.a0_head(feat),output_hw,mode="bilinear",align_corners=False)
    def refine_from_features(self,feat,a0,*,profile="balanced",return_metadata=False):
        if profile not in PROFILES: raise ValueError(f"profile must be one of {sorted(PROFILES)}")
        logits=a0; stages=[a0]; metadata=[]
        for turn in range(PROFILES[profile].turns):
            out=self.refiner(feat,logits,previous_audit_evidence=None,turn_index=turn,return_metadata=return_metadata)
            logits=out.candidate_logits; stages.append(logits)
            if return_metadata: metadata.append(out.window_metadata)
        return {"a0_logits":a0,"final_logits":logits,"stages":tuple(stages),"profile":profile,"window_metadata":tuple(metadata)}
    def forward(self,x,*,profile="balanced",return_metadata=False):
        feat=self.encode(x); a0=self.initial_logits(feat,x.shape[-2:])
        return self.refine_from_features(feat,a0,profile=profile,return_metadata=return_metadata)

def pseudo_supervision_loss(outputs,target,valid,a0_weight=.25):
    """Ignore UNKNOWN BEFORE cross entropy, including mixed-validity batches."""
    if target.dtype != torch.long or valid.dtype != torch.bool or valid.shape != target.shape:
        raise ValueError("target long and valid bool must both be [B,H,W]")
    final=outputs["final_logits"]; a0=outputs["a0_logits"]
    if target.shape != (final.shape[0],*final.shape[2:]) or a0.shape != final.shape:
        raise ValueError("logit/target shape mismatch")
    if not bool(valid.any()):
        return final.sum()*0.0+a0.sum()*0.0
    if bool(((target[valid]<0)|(target[valid]>=NUM_CLASSES)).any()):
        raise ValueError("accepted target must be a semantic class 0..3")
    safe=target.detach().masked_fill(~valid,UNKNOWN)
    def ce(x):
        return F.cross_entropy(x,safe,ignore_index=UNKNOWN,reduction="sum")/valid.sum()
    return ce(final)+float(a0_weight)*ce(a0)
