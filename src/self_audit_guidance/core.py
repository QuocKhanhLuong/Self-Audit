"""Fixed image evidence, explicit anatomy priors and a certified sparse bank solve.

This is an experimental readout, NOT a reproduced pretrained segmentation head.
No target masks, class matching against references or student predictions enter.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy import ndimage as ndi, sparse


@dataclass(frozen=True)
class GuidanceConfig:
    grid: int = 48
    feature_dim: int = 64
    clusters: tuple = (4, 8, 12)
    kmeans_steps: int = 15
    max_fit_points: int = 4096
    max_nodes: int = 1500000
    max_region_candidates: int = 96
    min_region_pixels: int = 6
    min_enclosure: float = .65
    min_contrast: float = .25
    spatial_scale: float = .35
    temporal: bool = True
    match_radius: int = 2
    match_cosine: float = .85
    match_margin: float = .02
    match_photo: float = .5
    alpha: float = .35
    tolerance: float = 1e-6
    max_iterations: int = 500
    min_probability: float = .70
    min_margin: float = .20
    min_reliability: float = .05

    def __post_init__(self):
        for k, lo, hi in [('grid', 8, 128), ('feature_dim', 4, 256),
                         ('kmeans_steps', 1, 100), ('max_fit_points', 32, 65536),
                         ('max_nodes', 16, 4000000), ('max_region_candidates', 4, 512),
                         ('min_region_pixels', 2, 512), ('match_radius', 1, 4),
                         ('max_iterations', 1, 10000)]:
            if type(getattr(self,k)) is not int or not lo <= getattr(self,k) <= hi:
                raise ValueError(f'{k} must be an integer in [{lo}, {hi}]')
        if type(self.temporal) is not bool:
            raise ValueError('temporal must be bool')
        if not isinstance(self.clusters, (list, tuple)) or not self.clusters:
            raise ValueError('clusters must be a nonempty list')
        if any(type(k) is not int or not 2 <= k <= 32 for k in self.clusters):
            raise ValueError('cluster counts must be in [2,32]')
        object.__setattr__(self, 'clusters', tuple(self.clusters))
        for k in ('min_enclosure', 'min_contrast', 'spatial_scale', 'match_cosine',
                  'match_margin', 'match_photo', 'alpha', 'tolerance',
                  'min_probability', 'min_margin', 'min_reliability'):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'{k} must be finite and positive')
        for k in ('min_enclosure','match_cosine','match_margin','alpha',
                  'min_probability','min_margin','min_reliability'):
            if getattr(self,k) > 1: raise ValueError(f'{k} must not exceed 1')

    def dictionary(self):
        return asdict(self)


def unit_features(f):
    f = np.asarray(f, dtype=np.float32)
    if not np.isfinite(f).all(): raise ValueError('nonfinite features')
    return f / np.maximum(np.linalg.norm(f,axis=-1,keepdims=True), 1e-8)


def validate_cine(image, features):
    if image.ndim != 4 or min(image.shape) < 1 or min(image.shape[-2:]) < 2:
        raise ValueError('image must be [T,Z,H,W]')
    if features.ndim != 5 or features.shape[:-1] != image.shape or features.shape[-1] < 1:
        raise ValueError('features must be [T,Z,H,W,D]')
    if not np.isfinite(image).all() or not np.isfinite(features).all():
        raise ValueError('nonfinite cine evidence')


def _nearest(a, b, radius):
    """Local cosine nearest and runner-up, ties rejected by the caller."""
    h,w,d = a.shape
    yy,xx = np.mgrid[:h,:w]
    best = np.full((h,w), -np.inf, np.float32)
    second = best.copy(); owner = np.full((h,w), -1, np.int64)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            y,x = yy+dy,xx+dx
            inside=(y>=0)&(y<h)&(x>=0)&(x<w)
            yc,xc=np.clip(y,0,h-1),np.clip(x,0,w-1)
            score=np.sum(a*b[yc,xc],axis=-1)
            score=np.where(inside,score,-np.inf)
            take=score>best
            second=np.where(take,best,np.maximum(second,score))
            best=np.maximum(best,score)
            owner=np.where(take,yc*w+xc,owner)
    return owner,best,best-second


def temporal_matches(a,b,ia,ib,cfg):
    """Mutual local feature correspondence, with margin/photo veto; not registration GT."""
    f,score,margin=_nearest(a,b,cfg.match_radius)
    r,rs,rm=_nearest(b,a,cfg.match_radius)
    ff=f.ravel(); valid=ff>=0; target=np.maximum(ff,0)
    valid &= r.ravel()[target] == np.arange(ff.size)
    valid &= (score.ravel()>=cfg.match_cosine)&(margin.ravel()>=cfg.match_margin)
    valid &= (rs.ravel()[target]>=cfg.match_cosine)&(rm.ravel()[target]>=cfg.match_margin)
    valid &= np.abs(ia.ravel()-ib.ravel()[target])<=cfg.match_photo
    src=np.flatnonzero(valid)
    return src,ff[src],np.maximum(score.ravel()[src],0)


def build_graph(image,features,cfg):
    """One patient, no cross-patient or unregistered cross-slice edges."""
    validate_cine(image,features)
    n=image.size
    if n>cfg.max_nodes: raise ValueError(f'{n} graph nodes exceed explicit max_nodes; change grid explicitly')
    f=unit_features(features); ids=np.arange(n).reshape(image.shape)
    rows=[np.arange(n)]; cols=[np.arange(n)]; weights=[np.ones(n,np.float32)]
    metrics={'spatial_edges':0,'temporal_edges':0,'nodes':int(n)}
    def add(a,b,w,name):
        a,b,w=a.ravel(),b.ravel(),w.ravel()
        ok=np.isfinite(w)&(w>1e-8)
        a,b,w=a[ok],b[ok],w[ok]
        rows.extend([a,b]);cols.extend([b,a]);weights.extend([w,w])
        metrics[name]+=len(a)
    for axis in (-2,-1):
        s1=[slice(None)]*4;s2=s1.copy();s1[axis]=slice(None,-1);s2[axis]=slice(1,None)
        s1,s2=tuple(s1),tuple(s2)
        cosine=np.sum(f[s1]*f[s2],axis=-1)
        photo=(image[s1]-image[s2])/cfg.spatial_scale
        weight=np.exp(-(1-np.clip(cosine,-1,1))/.15-.5*photo**2)
        add(ids[s1],ids[s2],weight,'spatial_edges')
    if cfg.temporal:
        t,z,h,w=image.shape
        for ti in range(t-1):
            for zi in range(z):
                a,b,weight=temporal_matches(f[ti,zi],f[ti+1,zi],image[ti,zi],image[ti+1,zi],cfg)
                add(ids[ti,zi].ravel()[a],ids[ti+1,zi].ravel()[b],weight,'temporal_edges')
    p=sparse.coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(n,n)).tocsr()
    p=sparse.diags(1/np.asarray(p.sum(1)).ravel())@p
    return p.tocsr(),metrics


def anchored_solve(p,r0,alpha=.35,tolerance=1e-6,max_iterations=500,initial=None):
    """Fixed-operator contraction, certified ||Q-Q*||max <= residual/alpha."""
    if not 0<alpha<=1 or not 0<tolerance<1 or type(max_iterations) is not int or max_iterations<1:
        raise ValueError('invalid solver settings')
    p=sparse.csr_matrix(p,dtype=np.float64)
    r0=np.asarray(r0,dtype=np.float64)
    if (r0.ndim!=2 or r0.shape[1]<1 or p.shape!=(len(r0),len(r0))
        or not np.isfinite(r0).all() or np.any(r0<0) or not np.isfinite(p.data).all()
        or np.any(p.data<0) or not np.allclose(p.sum(1).A.ravel(),1,atol=2e-6,rtol=0)
        or not np.allclose(r0.sum(1),1,atol=2e-6,rtol=0)):
        raise ValueError('stochastic P and simplex R0 required')
    # Normalize rounding error once, then leave the operator fixed.
    p=sparse.diags(1/p.sum(1).A.ravel())@p
    r0=r0/r0.sum(1,keepdims=True)
    q=r0.copy() if initial is None else np.asarray(initial,dtype=np.float64).copy()
    if q.shape!=r0.shape or not np.isfinite(q).all(): raise ValueError('invalid initial iterate')
    for step in range(1,max_iterations+1):
        q=alpha*r0+(1-alpha)*(p@q)
        residual=float(np.max(np.abs(alpha*r0+(1-alpha)*(p@q)-q)))
        if residual<=alpha*tolerance:
            return q,{'converged':True,'iterations':step,'residual':residual,'error_bound':residual/alpha}
    raise RuntimeError(f'BANK_NOT_CONVERGED: residual={residual:.6g}, bound={residual/alpha:.6g}')


def _cluster(f,k,steps,max_points):
    """Deterministic farthest-point initialization; no label-conditioned restart."""
    flat=f.reshape(-1,f.shape[-1]).astype(np.float32)
    fit=flat[np.linspace(0,len(flat)-1,min(max_points,len(flat)),dtype=int)]
    centers=[fit[0]]; dist=np.sum((fit-centers[0])**2,axis=1)
    for _ in range(1,min(k,len(fit))):
        j=int(np.argmax(dist))
        if dist[j]<1e-8: break
        centers.append(fit[j]);dist=np.minimum(dist,np.sum((fit-fit[j])**2,axis=1))
    centers=np.asarray(centers)
    def assign(x):
        return np.maximum((x*x).sum(1)[:,None]+(centers*centers).sum(1)[None]-2*x@centers.T,0).argmin(1)
    for _ in range(steps):
        lab=assign(fit)
        updated=np.stack([fit[lab==j].mean(0) if (lab==j).any() else centers[j] for j in range(len(centers))])
        if np.allclose(updated,centers,atol=1e-5):break
        centers=updated
    # Avoid NxKxD allocation for full cine.
    labels=np.concatenate([assign(flat[i:i+8192]) for i in range(0,len(flat),8192)])
    return labels.reshape(f.shape[:-1])


def anatomical_anchors(image,features,left_axis,cfg):
    """Experimental multiscale component hypotheses plus soft conflicting votes.

    Priors: an internally supported wall surrounds a cavity; an adjacent cavity
    on the patient's right may be RV. No image-border component can be a wall.
    Atoms have no confidence gate. Partial enclosure is scored, not repaired.
    No missing pixel is filled. These priors do not identify the heart uniquely.
    """
    validate_cine(image,features)
    votes=np.zeros((*image.shape,4),np.float32); masses=np.zeros(image.shape,np.float32)
    diag={'wall_cavity_hypotheses':0,'rv_hypotheses':0,'slices_with_hypotheses':0,
          'no_orientation_slices':0,'candidate_cap_slices':0,'hypotheses_before_anatomy':0}
    # Frozen features plus an explicit normalized-intensity term, not semantic logits.
    descriptor=np.concatenate([unit_features(features),image[...,None]*.5],-1)
    for k in cfg.clusters:
        labels=_cluster(descriptor,k,cfg.kmeans_steps,cfg.max_fit_points)
        for t in range(image.shape[0]):
            for z in range(image.shape[1]):
                lab,im=labels[t,z],image[t,z]
                parts=[]
                for c in np.unique(lab):
                    cc,n=ndi.label(lab==c)
                    sizes=np.bincount(cc.ravel())
                    for j in np.flatnonzero(sizes[1:]>=cfg.min_region_pixels)+1:
                        mask=cc==j
                        parts.append((int(sizes[j]),int(np.flatnonzero(mask)[0]),mask))
                parts.sort(key=lambda x:(-x[0],x[1]))
                if len(parts)>cfg.max_region_candidates:diag['candidate_cap_slices']+=1
                parts=parts[:cfg.max_region_candidates]
                stats=[]
                for n,_,m in parts:
                    border=bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any())
                    holes=ndi.binary_fill_holes(m)&~m
                    stats.append((m,holes,border,n))
                found=False
                for wi,(wall,holes,border,wn) in enumerate(stats):
                    if border or not holes.any():continue
                    for ci,(cavity,ch,cb,cn) in enumerate(stats):
                        if ci==wi or cb or ch.any():continue
                        overlap=int((holes&cavity).sum())
                        enclosure=min(overlap/cn,overlap/max(int(holes.sum()),1))
                        if enclosure<cfg.min_enclosure:continue
                        diag['hypotheses_before_anatomy']+=1
                        interface=ndi.binary_dilation(cavity)&wall
                        if not interface.any():continue
                        contrast=abs(float(np.median(im[cavity])-np.median(im[wall])))
                        if contrast<cfg.min_contrast:continue
                        score=float(enclosure*(1-np.exp(-contrast)))
                        # The geometry already observed is evidence; do NOT fill holes to make output masks.
                        votes[t,z,wall,2]+=score;votes[t,z,cavity,3]+=score
                        masses[t,z,wall]+=score;masses[t,z,cavity]+=score
                        diag['wall_cavity_hypotheses']+=1;found=True
                        if left_axis not in ('+x','-x','+y','-y'):
                            diag['no_orientation_slices']+=1;continue
                        dim=1 if left_axis.endswith('x') else 0
                        sign=1 if left_axis[0]=='+' else -1
                        lvpos=sign*np.nonzero(cavity)[dim].mean()
                        for ri,(rv,rh,rb,rn) in enumerate(stats):
                            if ri in (wi,ci) or rb or rh.any():continue
                            if not (ndi.binary_dilation(wall)&rv).any():continue
                            if sign*np.nonzero(rv)[dim].mean()>=lvpos:continue
                            # Require input support for blood-pool appearance; orientation alone is not a label.
                            gap=abs(float(np.median(im[rv])-np.median(im[cavity])))
                            if gap>max(contrast*.5,.1):continue
                            rs=score*np.exp(-gap/max(contrast,1e-6))
                            votes[t,z,rv,1]+=rs;masses[t,z,rv]+=rs
                            diag['rv_hypotheses']+=1
                if found:
                    diag['slices_with_hypotheses']+=1
                    # Border evidence is permitted only in slices with foreground hypotheses.
                    for m,_,border,_ in stats:
                        if border: votes[t,z,m,0]+=.5;masses[t,z,m]+=.5
    r0=np.full(votes.shape,.25,np.float32)
    supported=masses>0
    r0[supported]=votes[supported]/masses[supported,None]
    strength=1-np.exp(-masses)
    r0=strength[...,None]*r0+(1-strength[...,None])*.25
    diag['anchor_supported_pixels']=int(supported.sum())
    return r0,strength,diag


def make_bank(image,features,left_axis,cfg):
    r0,strength,anatomy=anatomical_anchors(image,features,left_axis,cfg)
    p,graph=build_graph(image,features,cfg)
    q,receipt=anchored_solve(p,r0.reshape(-1,4),cfg.alpha,cfg.tolerance,cfg.max_iterations)
    # Reliability cannot appear just because a random semantic head is confident.
    support=np.stack([strength.ravel(),1-strength.ravel()],-1)
    propagated,_=anchored_solve(p,support,cfg.alpha,cfg.tolerance,cfg.max_iterations)
    q=q.reshape(*image.shape,4).astype(np.float32)
    entropy=-(q*np.log(np.maximum(q,1e-8))).sum(-1)/np.log(4)
    reliability=(propagated[:,0].reshape(image.shape)*(1-entropy)).clip(0,1).astype(np.float32)
    top=np.sort(q,axis=-1)
    valid=(top[...,-1]>=cfg.min_probability)&(top[...,-1]-top[...,-2]>=cfg.min_margin)&(reliability>=cfg.min_reliability)
    labels=np.where(valid,q.argmax(-1),255).astype(np.uint8)
    counts=[int((labels==c).sum()) for c in range(4)]
    return dict(q=q,r0=r0,reliability=reliability,labels=labels),dict(
        anatomy=anatomy,graph=graph,solver=receipt,class_pixels=counts,
        status='SUPPORT_NOT_ACCURACY' if sum(counts[1:]) else 'NO_FOREGROUND_SUPPORT')
