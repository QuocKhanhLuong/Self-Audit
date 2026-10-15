"""Dense student wrapper. Official pretrained weights must be supplied explicitly.

This does not import a segmentation head or download anything automatically.
The random torchvision factory is for architecture/runtime tests ONLY.
"""
from __future__ import annotations
from collections.abc import Sequence
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .layers import CenterDifferenceInput, compose_logits


class OfficialConvNeXtStages(nn.Module):
    """Use the raw native stages of a supplied official DINOv3 ConvNeXt object.

    Bypasses get_intermediate_layers' optional common-grid resize. Does not
    assert checkpoint ancestry or perform weight loading. The source API was
    inspected, but this package's runtime tests use a random ConvNeXt fixture.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        if not all(hasattr(backbone, k) for k in ('downsample_layers','stages','embed_dims')):
            raise TypeError('expected an official DINOv3 ConvNeXt backbone object')
        if len(backbone.stages) != 4 or len(backbone.downsample_layers) != 4:
            raise ValueError('expected four native stages')
        self.downsample_layers = backbone.downsample_layers
        self.stages = backbone.stages
        self.channels = tuple(backbone.embed_dims)

    def forward(self, x: Tensor) -> list[Tensor]:
        out = []
        for down, stage in zip(self.downsample_layers, self.stages):
            x = stage(down(x)); out.append(x)
        return out


class ResidualUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels))
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class LayerGuidedStudent(nn.Module):
    """Multiscale encoder-decoder with exact coarse/detail logit factorization.

    Output native logits are cropped; coarse supervision uses the UNcropped
    padded coarse tensor. A validity mask prevents padded image evidence.
    Input alignment to cached teacher features remains the trainer's duty.
    """
    def __init__(self, encoder: nn.Module, *, classes: int = 4,
                 decoder_channels: Sequence[int] = (96,128,192,256),
                 teacher_dim: int = 64, detail_bound: float = 1.0):
        super().__init__()
        chans = tuple(getattr(encoder, 'channels', ()))
        dc = tuple(decoder_channels)
        if len(chans) != 4 or len(dc) != 4 or any(c % 8 for c in dc):
            raise ValueError('four encoder channels and decoder widths divisible by 8 required')
        if not math.isfinite(detail_bound) or detail_bound <= 0:
            raise ValueError("detail_bound must be finite and positive")
        self.register_buffer("detail_bound", torch.tensor(float(detail_bound)))
        self.encoder = encoder
        self.input_adapter = CenterDifferenceInput()
        self.lateral = nn.ModuleList([nn.Conv2d(c,d,1) for c,d in zip(chans,dc)])
        self.up_channels = nn.ModuleList([nn.Conv2d(dc[i+1],dc[i],1) for i in range(3)])
        self.refine = nn.ModuleList([nn.Sequential(ResidualUnit(d),ResidualUnit(d)) for d in dc])
        self.coarse_head = nn.Conv2d(dc[0], classes, 1)
        self.feature_project = nn.Conv2d(chans[2], teacher_dim, 1)
        self.detail_stem = nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.GELU(),ResidualUnit(32))
        self.detail_fuse = nn.Sequential(nn.Conv2d(dc[0]+32,32,1),nn.GELU(),ResidualUnit(32))
        self.detail_head = nn.Conv2d(32,classes,1)
        # Start without high-frequency random class assignments; this branch
        # receives native image/supervision gradients after initialization.
        nn.init.zeros_(self.detail_head.weight); nn.init.zeros_(self.detail_head.bias)
        self.classes = classes

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        h,w = x.shape[-2:]
        norm = self.input_adapter(x)
        ph,pw = (-h)%32,(-w)%32
        norm = F.pad(norm, (0,pw,0,ph), mode='replicate')
        raw = F.pad(x, (0,pw,0,ph), mode='replicate')
        feats = self.encoder(norm)
        for i, (f,c) in enumerate(zip(feats,self.encoder.channels)):
            expected=(norm.shape[-2]//(4*2**i),norm.shape[-1]//(4*2**i))
            if f.ndim!=4 or f.shape[1]!=c or f.shape[-2:]!=expected:
                raise ValueError(f'encoder stage {i} has incorrect native shape')
        d = self.refine[3](self.lateral[3](feats[3]))
        for i in (2,1,0):
            skip = self.lateral[i](feats[i])
            d = self.refine[i](skip + self.up_channels[i](F.interpolate(
                d, skip.shape[-2:], mode='bilinear',align_corners=False)))
        coarse = self.coarse_head(d)
        high = self.detail_stem(raw)
        high = self.detail_fuse(torch.cat((high,F.interpolate(
            d,raw.shape[-2:],mode='bilinear',align_corners=False)),1))
        detail = self.detail_head(high)
        bounded_detail = self.detail_bound * detail.tanh()
        logits = compose_logits(coarse,bounded_detail,4)
        valid = torch.zeros_like(logits[:,:1],dtype=torch.bool)
        valid[...,:h,:w] = True
        return dict(logits=logits[...,:h,:w],logits_padded=logits,
                    coarse_logits=coarse,raw_detail=detail,bounded_detail=bounded_detail,
                    features=self.feature_project(feats[2]),native_valid=valid,
                    native_hw=torch.tensor([h,w],device=x.device))


def build_random_convnext_demo() -> LayerGuidedStudent:
    """Full-size random ConvNeXt-Tiny smoke factory; NOT pretrained DINOv3."""
    from torchvision.models import convnext_tiny
    original = convnext_tiny(weights=None, stochastic_depth_prob=0.0)

    class Stages(nn.Module):
        channels = (96,192,384,768)
        def __init__(self, features: nn.Module):
            super().__init__(); self.features = features
        def forward(self, x: Tensor) -> list[Tensor]:
            out=[]
            for i,layer in enumerate(self.features):
                x=layer(x)
                if i % 2: out.append(x)
            return out
    return LayerGuidedStudent(Stages(original.features))
