"""Frozen CineMA MAE adapter. The official weight hash is pinned, not a file-name claim.

CineMA package/weights are optional external dependencies. Mock features in tests
are never a production fallback. Upstream pretraining lineage is not certified
merely by recognizing an MAE checkpoint.
"""
from __future__ import annotations
import itertools, math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .io import digest

WEIGHT_SHA256='a841b7e9300d60ea8feb73e3a7d8c60ddedc1d1d49ec1049ebfcfa337c5d5590'
WEIGHT_FILE='pretrained/cinema.safetensors'
SOURCE_COMMIT='c10daa1d93f0ea28d8b9ad9206b0f673d25805c1'
HF_REVISION='e0466ed5bd7687a32121434875bf2168aed41c03'


def validate_device(device):
    d=torch.device(device)
    if d.type not in ('cpu','cuda'):raise ValueError('device must be cpu or cuda')
    if d.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent fallback')
    return d


def starts(n,size):
    if n<=size:return [0]
    # Half-overlap tiles; final tile always covers the far border, never a selected ROI.
    return list(range(0,n-size+1,max(size//2,1))) + ([] if (n-size)%max(size//2,1)==0 else [n-size])


def validate_teacher_config(c):
    expected={'patch_size':[4,4,1],'scale_factor':[2,2,1],'enc_conv_chans':[64,128],'enc_conv_n_blocks':2,'size':'base'}
    if any(c['model'].get(k)!=v for k,v in expected.items()):raise ValueError('unexpected CineMA architecture')
    if c['data']['sax']['patch_size']!=[192,192,16] or c['data']['sax']['in_chans']!=1:
        raise ValueError('unexpected CineMA input grid')


class FrozenCineMA:
    def __init__(self,device='cuda',checkpoint=None,config=None,cache_dir=None,offline=False):
        self.device=validate_device(device)
        try:
            from cinema.mae.mae import get_model
            from omegaconf import OmegaConf
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError('Install the documented isolated CineMA environment; do not substitute a segmentation model') from e
        if (checkpoint is None)!=(config is None):raise ValueError('supply both MAE checkpoint and config, or neither')
        if checkpoint is None:
            checkpoint=hf_hub_download('mathpluscode/CineMA',WEIGHT_FILE,cache_dir=cache_dir,local_files_only=offline,revision=HF_REVISION)
            config=hf_hub_download('mathpluscode/CineMA','pretrained/config.yaml',cache_dir=cache_dir,local_files_only=offline,revision=HF_REVISION)
        checkpoint,config=Path(checkpoint),Path(config)
        if digest(checkpoint)!=WEIGHT_SHA256:raise ValueError('checkpoint is not the pinned official MAE; finetuned/unknown weights refused')
        cfg=OmegaConf.load(config)
        validate_teacher_config(OmegaConf.to_container(cfg,resolve=True))
        self.model=get_model(cfg)
        self.model.load_state_dict(load_file(str(checkpoint),device='cpu'),strict=True)
        self.model.requires_grad_(False).eval().to(self.device)
        self.model.set_grad_ckpt(False)
        self.tile=tuple(cfg.data.sax.patch_size)
        self.token_grid=tuple(self.model.enc_down_dict['sax'].patch_embed.grid_size)
        if self.token_grid!=(12,12,16):raise ValueError(f'unexpected token layout: {self.token_grid}')
        import cinema.mae.mae as source
        self.provenance=dict(provider='CineMA-MAE',checkpoint_sha256=WEIGHT_SHA256,
            config_sha256=digest(config),adapter_source_sha256=digest(__file__),
            cinema_mae_source_sha256=digest(source.__file__),
            checkpoint_filename=WEIGHT_FILE,model_revision=HF_REVISION,source_commit_expected=SOURCE_COMMIT,upstream_lineage='SSL_DECLARED_PREPROCESSING_NOT_FULLY_AUDITED',
            frozen=True,segmentation_finetune=False)

    @torch.inference_mode()
    def extract(self,cine,grid=48,feature_dim=64,tile_batch=1):
        return extract_tiled(self.model,cine,self.tile,self.token_grid,self.device,grid,feature_dim,tile_batch)


@torch.inference_mode()
def extract_tiled(model,cine,tile,token_grid,device,grid,feature_dim,tile_batch=1):
    """Native TZYX -> full-coverage fixed XYZ tiles -> registered TZYXD features.

    Model tokens flatten XYZ in that order. D is reduced by a fixed Gaussian
    projection to bound the cache, not by a learned or label-selected projector.
    In-plane features are interpolated; they do not gain true fine detail.
    """
    if (cine.ndim!=4 or min(cine.shape)<1 or not np.isfinite(cine).all()
        or type(tile_batch) is not int or tile_batch<1 or grid<2 or feature_dim<1):
        raise ValueError('invalid teacher input/budget')
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('teacher must be frozen and eval')
    t,z,h,w=cine.shape;tx,ty,tz=tile
    # Compress only feature output. Every native pixel enters at least one teacher window.
    projected_frames=[];projection=None
    for ti in range(t):
        native=torch.from_numpy(cine[ti].copy()).float().permute(2,1,0)  # XYZ
        lo,hi=native.min(),native.max();native=(native-lo)/(hi-lo).clamp_min(1e-6)
        xyz=torch.zeros(max(w,tx),max(h,ty),max(z,tz))
        xyz[:w,:h,:z]=native
        coords=list(itertools.product(starts(xyz.shape[0],tx),starts(xyz.shape[1],ty),starts(xyz.shape[2],tz)))
        if len(coords)>4096:raise ValueError('teacher tile budget exceeded')
        acc=torch.zeros(feature_dim,xyz.shape[2],grid,grid)
        count=torch.zeros(1,xyz.shape[2],grid,grid)
        # Accumulate at a global feature lattice, with coordinates derived from native pixels.
        for first in range(0,len(coords),tile_batch):
            chunk=coords[first:first+tile_batch]
            batch=torch.stack([xyz[x:x+tx,y:y+ty,zz:zz+tz] for x,y,zz in chunk])[:,None].to(device)
            tokens=model.feature_forward({'sax':batch})['sax']
            if tokens.ndim!=3 or tokens.shape[1]!=math.prod(token_grid) or not torch.isfinite(tokens).all():
                raise ValueError('unexpected/nonfinite CineMA token output')
            if projection is None:
                g=torch.Generator(device='cpu').manual_seed(20261011)
                projection=torch.randn(tokens.shape[-1],feature_dim,generator=g).to(device)/math.sqrt(feature_dim)
            f=(tokens.float()@projection).reshape(len(chunk),*token_grid,feature_dim).permute(0,4,3,2,1)
            # Sampling every global output center inside this tile avoids seams from rounded resize boundaries.
            gy=(torch.arange(grid)+.5)*h/grid-.5
            gx=(torch.arange(grid)+.5)*w/grid-.5
            for j,(x,y,zz) in enumerate(chunk):
                iy=torch.where((gy>=y-.5)&(gy<min(y+ty,h)-.5))[0]
                ix=torch.where((gx>=x-.5)&(gx<min(x+tx,w)-.5))[0]
                depth=min(tz,z-zz)
                if not len(iy) or not len(ix) or depth<=0:continue
                sz=2*(torch.arange(depth)+.5)/tz-1
                sy=2*(gy[iy]-y+.5)/ty-1;sx=2*(gx[ix]-x+.5)/tx-1
                gz,gyy,gxx=torch.meshgrid(sz,sy,sx,indexing='ij')
                sampling=torch.stack([gxx,gyy,gz],-1)[None].to(device)
                sampled=F.grid_sample(f[j:j+1],sampling,align_corners=False,padding_mode='border')[0].cpu()
                for yi,out_y in enumerate(iy.tolist()):
                    acc[:,zz:zz+depth,out_y,ix]+=sampled[:,:,yi,:]
                    count[:,zz:zz+depth,out_y,ix]+=1
        if not bool((count[:,:z]>0).all()):raise RuntimeError('teacher tiling left uncovered native support')
        dense=acc[:,:z]/count[:,:z]
        dense=F.normalize(dense,dim=0).permute(1,2,3,0).numpy().astype(np.float16)
        projected_frames.append(dense)
        print(f'[SSL] frame={ti+1}/{t} tiles={len(coords)} frozen_features={list(dense.shape)}',flush=True)
    return np.stack(projected_frames)
