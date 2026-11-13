"""Code-free native UNETR++-lite student with optional FUSE feature gates."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..candidate_q3d.alignment import GridPhase, grid_phase
from ..candidate_q3d.student import _ConvNormAct, _EPA3DBlock, _ResidualConv3D, _valid_grid_mask
from .fusion import GatedFusion, align_feature


def _shape(geometry: Any, image: Tensor) -> tuple[int, int, int]:
    if geometry is None:
        return tuple(int(v) for v in image.shape[-3:])
    if isinstance(geometry, Mapping):
        s = geometry.get("shape")
        if s is not None and len(s) == 3:
            result = tuple(int(v) for v in s)
            if result[0] != image.shape[-3] or result[1] > image.shape[-2] or result[2] > image.shape[-1]:
                raise ValueError("geometry shape is incompatible with native image")
            return result
    raise ValueError("geometry must provide native shape D,H,W")


def teacher_dropout_probability(update: int, total_updates: int) -> float:
    if total_updates <= 0 or update < 0:
        raise ValueError("update and total_updates must be valid")
    return 0.5 * update / max(total_updates - 1, 1)


def sample_keep_mask(batch_size: int, update: int, total_updates: int, *, device: torch.device | str,
                     generator: torch.Generator | None = None) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    p = teacher_dropout_probability(update, total_updates)
    keep = torch.rand((batch_size, 1, 1, 1, 1), device=device, generator=generator) >= p
    return keep.to(torch.float32)


class _EPAStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, count: int) -> None:
        super().__init__()
        self.down = _ConvNormAct(in_ch, out_ch, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.blocks = nn.ModuleList([_EPA3DBlock(out_ch, "uniform", count, 0.5) for _ in range(2)])

class FuseStudent(nn.Module):
    def __init__(self, config: Mapping[str, Any], *, teacher_channels: tuple[int, ...] | list[int] | None = None,
                 teacher_phases: tuple[GridPhase, ...] | list[GridPhase] | None = None,
                 include_fusion: bool = True, include_mimic: bool = True) -> None:
        super().__init__()
        model = config.get("model", config)
        student = model.get("student", {})
        channels = tuple(int(v) for v in student.get("channels", (32,64,128,256,512)))
        if len(channels) != 5 or any(c < 4 or c % 4 for c in channels):
            raise ValueError("student channels must be five positive EPA-compatible widths")
        if student.get("num_classes", 4) != 4:
            raise ValueError("FUSE requires four classes")
        self.channels = channels
        self.include_fusion = bool(include_fusion)
        self.include_mimic = bool(include_mimic)
        self.stem = nn.Sequential(_ConvNormAct(1, channels[0], kernel_size=(1,3,3)), _ResidualConv3D(channels[0],channels[0]))
        self.stage1 = nn.Sequential(_ConvNormAct(channels[0],channels[1],kernel_size=(1,3,3),stride=(1,2,2)),_ResidualConv3D(channels[1],channels[1]))
        self.stage2 = nn.Sequential(_ConvNormAct(channels[1],channels[2],kernel_size=(1,3,3),stride=(1,2,2)),_ResidualConv3D(channels[2],channels[2]))
        self.stage3 = _EPAStage(channels[2],channels[3],min(4*4*4,64))
        self.stage4 = _EPAStage(channels[3],channels[4],min(4*4*4,64))
        self.decoder_reduce = nn.ModuleList(nn.Conv3d(a,b,1) for a,b in zip((channels[4],channels[3],channels[2],channels[1]),(channels[3],channels[2],channels[1],channels[0])))
        self.decoder_blocks = nn.ModuleList(_ResidualConv3D(c,c) for c in (channels[3],channels[2],channels[1],channels[0]))
        self.logits_head = nn.Conv3d(channels[0],4,1)
        self.teacher_channels = tuple(int(v) for v in (teacher_channels or (32,64,128,256,320)))
        self.teacher_phases = tuple(teacher_phases or (((1,1,1),(0.,0.,0.)),((1,2,2),(0.,0.,0.)),((1,4,4),(0.,0.,0.)),((2,8,8),(0.,0.,0.)),((2,16,16),(0.,0.,0.))))
        if len(self.teacher_channels) != 5 or len(self.teacher_phases) != 5:
            raise ValueError("teacher channels and phases must describe five levels")
        self.fusions = nn.ModuleList(GatedFusion(s,t) for s,t in zip(channels,self.teacher_channels)) if include_fusion else nn.ModuleList()
        self.mimic_heads = nn.ModuleList(nn.Conv3d(channels[i],self.teacher_channels[i],1) for i in (2,3,4)) if include_mimic else nn.ModuleList()
        # Derive each phase from actual main-path sampling modules.
        self.phases = tuple(grid_phase(chain) for chain in (
            (self.stem[0].conv,),
            (self.stem[0].conv,self.stage1[0].conv),
            (self.stem[0].conv,self.stage1[0].conv,self.stage2[0].conv),
            (self.stem[0].conv,self.stage1[0].conv,self.stage2[0].conv,self.stage3.down.conv),
            (self.stem[0].conv,self.stage1[0].conv,self.stage2[0].conv,self.stage3.down.conv,self.stage4.down.conv),))

    def _decode(self, f4: Tensor, skips: tuple[Tensor,...], native_hw: tuple[int,int]) -> Tensor:
        x=f4
        for reduce,block,skip in zip(self.decoder_reduce,self.decoder_blocks,(skips[0],skips[1],skips[2],skips[3])):
            x=F.interpolate(x,size=skip.shape[-3:],mode="trilinear",align_corners=False)
            x=block(reduce(x)+skip)
        return x[...,:native_hw[0],:native_hw[1]]

    def forward(self, image: Tensor, geometry: Any, *, teacher_features: tuple[Tensor,...] | list[Tensor] | None = None,
                keep_mask: Tensor | None = None) -> dict[str, Any]:
        if image.ndim != 5 or image.shape[1] != 1 or not image.is_floating_point() or not torch.isfinite(image).all():
            raise ValueError("FUSE input must be finite floating [B,1,D,H,W]")
        native = _shape(geometry,image)
        x=image[...,:native[1],:native[2]]
        if torch.any(x<0) or torch.any(x>1): raise ValueError("FUSE image must be normalized to [0,1]")
        x,native_hw=self._pad(x)
        any_kept = teacher_features is not None and bool((keep_mask != 0).any()) if keep_mask is not None else False
        if teacher_features is not None:
            if not self.include_fusion: raise ValueError("fusion modules are absent from this student")
            if len(teacher_features)!=5 or keep_mask is None: raise ValueError("teacher features require five levels and keep_mask")
            if tuple(keep_mask.shape)!=(image.shape[0],1,1,1,1): raise ValueError("keep_mask must have shape [B,1,1,1,1]")
            if not torch.isfinite(keep_mask).all() or not torch.all((keep_mask == 0) | (keep_mask == 1)):
                raise ValueError("keep_mask must contain only finite binary 0/1 values")
        elif keep_mask is not None:
            raise ValueError("keep_mask cannot be supplied without teacher features")
        stages=[]; gates=[]; mimic=[]
        def fuse(index: int, feature: Tensor) -> Tensor:
            if teacher_features is None or not self.include_fusion or not any_kept:
                gates.append(0.0 if teacher_features is None else None); return feature
            aligned=align_feature(teacher_features[index],native,self.teacher_phases[index],self.phases[index],feature.shape[-3:])
            output,gate=self.fusions[index](feature,aligned,keep_mask)
            gates.append(float(gate.detach().mean().item()))
            return output
        f0=fuse(0,self.stem[0](x)); f0=self.stem[1](f0); stages.append(f0)
        s1=fuse(1,self.stage1[0](f0)); s1=self.stage1[1](s1); stages.append(s1)
        s2_pre=self.stage2[0](s1); s2=fuse(2,s2_pre); s2=self.stage2[1](s2); stages.append(s2)
        s3_pre=self.stage3.down(s2); s3=fuse(3,s3_pre); s3=self._epa_blocks(s3,s2.shape[-3:],self.stage3,8,native); stages.append(s3)
        s4_pre=self.stage4.down(s3); s4=fuse(4,s4_pre); s4=self._epa_blocks(s4,s3.shape[-3:],self.stage4,16,native); stages.append(s4)
        mimic_loss=stages[-1].sum()*0.0
        if teacher_features is not None and self.training and self.include_mimic and bool((keep_mask==0).any()):
            terms=[]
            for head,idx in zip(self.mimic_heads,(2,3,4)):
                aligned=align_feature(teacher_features[idx],native,self.teacher_phases[idx],self.phases[idx],stages[idx].shape[-3:])
                pred=F.normalize(head(stages[idx]),dim=1,eps=1e-6); target=F.normalize(aligned,dim=1,eps=1e-6)
                cosine=1-(pred*target).sum(1)
                valid=self._valid_centers(native,self.phases[idx],cosine.shape[-3:],cosine.device)
                per=(cosine*valid).sum((1,2,3))/valid.sum().clamp_min(1)
                terms.append(per)
            per_sample=torch.stack(terms).mean(0)
            mimic_loss=(per_sample*(keep_mask[:,0,0,0,0]==0).to(per_sample.dtype)).mean()
        decoded=self._decode(s4,(s3,s2,s1,f0),native_hw)
        logits=self.logits_head(decoded)[...,:native[1],:native[2]]
        return {"logits":logits,"features":tuple(stages),"gate_means":gates if teacher_features is not None else [0.0]*5,"mimic_loss":mimic_loss}

    @staticmethod
    def _valid_centers(native: tuple[int,int,int], phase: GridPhase, size: tuple[int,int,int], device: torch.device)->Tensor:
        step,offset=phase; axes=[offset[i]+torch.arange(size[i],device=device)*step[i]<native[i] for i in range(3)]
        return (axes[0][:,None,None]&axes[1][None,:,None]&axes[2][None,None,:]).to(torch.float32)

    @staticmethod
    def _epa_blocks(x: Tensor, _previous: tuple[int,int,int], stage: _EPAStage, stride: int, native: tuple[int,int,int])->Tensor:
        d,h,w=x.shape[-3:]
        mask=_valid_grid_mask(x.shape[0],d,h,w,native_shape=native,stride_yx=stride,device=x.device).to(x.dtype)
        for block in stage.blocks: x=block(x,mask,native,stride)
        return x

    @staticmethod
    def _pad(x: Tensor)->tuple[Tensor,tuple[int,int]]:
        h,w=x.shape[-2:]; ph=(-h)%16; pw=(-w)%16
        if ph or pw: x=F.pad(x,(0,pw,0,ph,0,0),mode="replicate")
        return x,(h,w)

    def infer_logits(self,image:Tensor,geometry:Any)->Tensor:
        return self.forward(image,geometry)["logits"]


__all__=["FuseStudent","teacher_dropout_probability","sample_keep_mask"]
