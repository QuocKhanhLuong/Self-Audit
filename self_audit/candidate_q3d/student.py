"""Student-only UNETR++-lite network for the Q-3D candidate.

This module deliberately has no teacher dependency.  It accepts one native
``[B, 1, D, H, W]`` frame, keeps the z axis at stride one, and crops all dense
outputs back to the unpadded in-plane shape.  In VQ arms the teacher-anchored
codebook is a non-trainable buffer; gradients use a straight-through estimate
through the selected code vectors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_DEFAULT_CHANNELS = (16, 32, 64, 128, 256)
_VQ_ARMS = {"V1", "V1-U", "V0", "B1", "E1", "O1", "S1"}
_CONTINUOUS_ARMS = {"C0", "C1"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read the public nested config while allowing a direct student mapping."""
    outer: Mapping[str, Any] = _mapping(config)
    root: Mapping[str, Any] = outer
    for key in ("candidate_q3d", "model"):
        nested = _mapping(root.get(key))
        if nested:
            root = nested
            break
    student = _mapping(root.get("student")) or root
    tokenizer = _mapping(outer.get("tokenizer")) or _mapping(root.get("tokenizer"))
    vq = (
        _mapping(student.get("vq"))
        or _mapping(root.get("vq"))
        or _mapping(tokenizer.get("vq"))
        or tokenizer
    )
    attention = _mapping(student.get("attention")) or _mapping(root.get("attention"))
    residual = (
        _mapping(student.get("residual_vq"))
        or _mapping(student.get("residual_level"))
        or _mapping(student.get("residual2"))
        or _mapping(root.get("residual_vq"))
        or _mapping(root.get("residual_level"))
        or _mapping(root.get("residual2"))
    )
    teacher = _mapping(root.get("teacher")) or _mapping(outer.get("teacher"))
    teacher_channels = teacher.get(
        "feature_channels", teacher.get("channels", outer.get("teacher_channels"))
    )

    channels = student.get(
        "channels", student.get("stage_channels", student.get("encoder_channels", _DEFAULT_CHANNELS))
    )
    if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes)) or len(channels) != 5:
        raise ValueError("student.channels must contain five positive stage widths")
    widths = tuple(int(value) for value in channels)
    if any(value <= 0 for value in widths):
        raise ValueError("student.channels must contain five positive stage widths")
    if widths[0] % 4 or any(value % 4 for value in widths[1:]):
        raise ValueError("student channel widths must be divisible by four EPA heads")

    arm = str(student.get("arm", root.get("arm", outer.get("arm", "V1")))).upper()
    arm = "V1-U" if arm.replace("-", "") == "V1U" else arm
    if arm not in _VQ_ARMS | _CONTINUOUS_ARMS:
        raise ValueError(f"unsupported Q-3D student arm: {arm}")
    codebook_size = int(vq.get("size", vq.get("codebook_size", 256)))
    codebook_dim = int(
        vq.get(
            "dim",
            vq.get("codebook_dim", vq.get("embedding_dim", vq.get("dimension", 64))),
        )
    )
    tau = float(vq.get("tau", vq.get("temperature", 0.1)))
    if codebook_size < 2 or codebook_dim < 1 or tau <= 0:
        raise ValueError("student VQ size/dim/tau must be positive (size at least two)")
    support_mode = str(
        attention.get(
            "support", attention.get(
                "support_mode", student.get("epa_support", "uniform_fixed")
            )
        )
    ).lower()
    if support_mode in {"uniform_fixed", "fixed_uniform"}:
        support_mode = "uniform"
    if attention.get("adaptive_support", False) or support_mode == "adaptive":
        support_mode = "learned_content"
    if support_mode not in {"uniform", "learned_content"}:
        raise ValueError("student.attention.support must be 'uniform' or 'learned_content'")
    support_pool = student.get("epa_pool_zyx", (4, 4, 4))
    if not isinstance(support_pool, Sequence) or isinstance(support_pool, (str, bytes)) \
            or len(support_pool) != 3:
        raise ValueError("student.epa_pool_zyx must contain z,y,x support dimensions")
    support_pool = tuple(int(value) for value in support_pool)
    if support_pool[0] < 1 or support_pool[0] > 4 or support_pool[1:] != (4, 4):
        raise ValueError("EPA support is fixed to min(D,4) x 4 x 4")
    support_count = int(attention.get("support_count", student.get("support_count", 64)))
    if support_count != support_pool[0] * support_pool[1] * support_pool[2]:
        raise ValueError("EPA support_count must match student.epa_pool_zyx")
    geometric_fraction = float(
        attention.get("geometric_fraction", student.get("epa_geometric_fraction", 0.5))
    )
    if not 0.0 < geometric_fraction < 1.0:
        raise ValueError("EPA geometric_fraction must be strictly between zero and one")
    residual_enabled = bool(residual.get("enabled", False))
    gate_init = float(student.get("code_gate_init", 0.0))
    if gate_init != 0.0:
        raise ValueError("Q-3D code gate must initialize to zero")
    residual_zero_init = bool(residual.get("zero_init", True))
    if residual_enabled and not residual_zero_init:
        raise ValueError("Q-3D residual level must initialize its gate to zero")
    sdf_option = student.get("sdf_head", {})
    if isinstance(sdf_option, Mapping):
        sdf_enabled = bool(sdf_option.get("enabled", True))
    elif isinstance(sdf_option, bool):
        sdf_enabled = sdf_option
    else:
        raise ValueError("model.student.sdf_head must be a boolean or mapping")
    sdf_enabled = bool(student.get("sdf_head_enabled", sdf_enabled))
    return {
        "channels": widths,
        "num_classes": int(student.get("num_classes", len(student.get("classes", ("BG", "RV", "MYO", "LV"))))),
        "in_channels": int(student.get("in_channels", student.get("input_channels", 1))),
        "arm": arm,
        "codebook_size": codebook_size,
        "codebook_dim": codebook_dim,
        "tau": tau,
        "support_mode": support_mode,
        "support_count": support_count,
        "geometric_fraction": geometric_fraction,
        "residual_enabled": residual_enabled,
        "residual_codebook_size": int(residual.get("size", codebook_size)),
        "teacher_channels": None if teacher_channels is None else int(teacher_channels),
        "sdf_enabled": sdf_enabled,
    }


class _ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = tuple(size // 2 for size in kernel_size)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True, eps=1e-5)
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class _ResidualConv3D(nn.Module):
    """Two k3 convolutions with an identity or projected residual."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = _ConvNormAct(
            in_channels, out_channels, kernel_size=(3, 3, 3), activation=True
        )
        self.conv2 = _ConvNormAct(
            out_channels, out_channels, kernel_size=(3, 3, 3), activation=False
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.out_act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.out_act(self.conv2(self.conv1(x)) + self.skip(x))


def _valid_grid_mask(
    batch: int,
    depth: int,
    height: int,
    width: int,
    *,
    native_shape: tuple[int, int, int],
    stride_yx: int,
    device: torch.device,
) -> Tensor:
    """Return native-pixel area fractions for each in-plane feature cell."""
    native_d, native_h, native_w = native_shape
    z = (torch.arange(depth, device=device) < native_d).to(torch.float32)
    y_start = torch.arange(height, device=device, dtype=torch.float32) * stride_yx
    x_start = torch.arange(width, device=device, dtype=torch.float32) * stride_yx
    y_fraction = ((native_h - y_start).clamp(min=0, max=stride_yx) / stride_yx)
    x_fraction = ((native_w - x_start).clamp(min=0, max=stride_yx) / stride_yx)
    fractions = (
        z.view(1, 1, depth, 1, 1)
        * y_fraction.view(1, 1, 1, height, 1)
        * x_fraction.view(1, 1, 1, 1, width)
    )
    return fractions.expand(batch, 1, depth, height, width)


def _masked_support_pool(
    features: Tensor,
    valid_mask: Tensor,
    *,
    native_shape: tuple[int, int, int],
    stride_yx: int,
    support_mode: str,
) -> tuple[Tensor, Tensor]:
    """Pool to the mask-aware uniform support grid used by the main arm."""
    if support_mode != "uniform":
        raise ValueError("masked average pooling is only used by uniform EPA support")
    native_d, native_h, native_w = native_shape
    del native_h, native_w, stride_yx
    output_size = (min(native_d, 4), 4, 4)
    mask = valid_mask.to(dtype=features.dtype)
    numerator = F.adaptive_avg_pool3d(features * mask, output_size)
    denominator = F.adaptive_avg_pool3d(mask, output_size)
    support = numerator / denominator.clamp_min(torch.finfo(features.dtype).eps)
    support_valid = denominator.flatten(2).squeeze(1) > 0
    return support, support_valid


class _EPA3DBlock(nn.Module):
    """Shape-variable EPA with masked spatial support and channel attention."""

    def __init__(
        self,
        channels: int,
        support_mode: str,
        support_count: int,
        geometric_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.heads = 4
        self.head_dim = channels // self.heads
        self.support_mode = support_mode
        if support_count < 1 or support_count > 64:
            raise ValueError("EPA support_count must be between one and 64")
        if not 0.0 < geometric_fraction < 1.0:
            raise ValueError("EPA geometric_fraction must be strictly between zero and one")
        self.support_count = support_count
        self.geometric_fraction = geometric_fraction
        self.norm = nn.LayerNorm(channels)
        # EPA shares Q/K between spatial and channel branches, with a separate
        # value projection for each branch.
        self.qk = nn.Linear(channels, 2 * channels, bias=False)
        self.spatial_value = nn.Linear(channels, channels, bias=False)
        self.channel_value = nn.Linear(channels, channels, bias=False)
        if support_mode == "learned_content":
            # A_d scores only student features within geometrically partitioned
            # windows; neither teacher outputs nor GT enter this projection.
            self.detail_score = nn.Linear(channels, 1, bias=False)
        else:
            self.detail_score = None
        self.branch_projection = nn.Linear(2 * channels, channels, bias=False)
        self.ff_depthwise = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.ff_norm = nn.InstanceNorm3d(channels, affine=True, eps=1e-5)
        self.ff_pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.ff_activation = nn.GELU()

    @staticmethod
    def _coordinates(
        shape: tuple[int, int, int], *, device: torch.device
    ) -> Tensor:
        axes = []
        for size in shape:
            axis = torch.arange(size, device=device, dtype=torch.float32)
            if size > 1:
                axis = axis / float(size - 1)
            axes.append(axis)
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return grid.reshape(-1, 3)

    @staticmethod
    def _coverage_anchors(valid_count: int, count: int, *, device: torch.device) -> Tensor:
        """Choose deterministic, evenly spaced positions over valid coordinates."""
        if count <= 0 or count >= valid_count:
            raise ValueError("EPA geometric supports require fewer rows than valid tokens")
        return torch.linspace(
            0, valid_count - 1, steps=count, device=device, dtype=torch.float32
        ).round().long()

    def _build_adaptive_support(
        self,
        normalized_tokens: Tensor,
        token_weight: Tensor,
        native_shape: tuple[int, int, int],
        lattice_shape: tuple[int, int, int],
    ) -> Tensor:
        """Build A=[A_c;A_d], one normalized support row per pooled token.

        A_c uses valid native-grid coordinates and area fractions to partition
        tokens into geometrically covering Voronoi windows. A_d uses a learned
        scalar score from student features within separate geometric windows.
        Both matrices weight the same projected K and spatial V. If support is
        denser than the valid grid, the fallback is the identity over valid
        tokens, with no empty-set softmax.
        """
        if normalized_tokens.ndim != 3:
            raise ValueError("EPA support tokens must have shape [B,N,C]")
        batch, token_count, channels = normalized_tokens.shape
        if tuple(token_weight.shape) != (batch, token_count):
            raise ValueError("EPA token mask/weights must match the [B,N] token lattice")
        if len(lattice_shape) != 3 or any(int(size) <= 0 for size in lattice_shape):
            raise ValueError("EPA lattice_shape must contain positive D,H,W")
        if int(lattice_shape[0]) * int(lattice_shape[1]) * int(lattice_shape[2]) != token_count:
            raise ValueError("EPA lattice_shape product must equal the token count")
        if len(native_shape) != 3 or any(int(size) <= 0 for size in native_shape):
            raise ValueError("EPA native_shape must contain positive D,H,W")
        if not torch.isfinite(token_weight).all() or bool((token_weight < 0).any()):
            raise ValueError("EPA token weights must be finite and nonnegative")
        if bool((token_weight > 1.0 + 1e-6).any()):
            raise ValueError("EPA token weights cannot exceed one")
        if any(not torch.equal(token_weight[0] > 0, token_weight[index] > 0)
               for index in range(1, batch)):
            raise ValueError("EPA adaptive support requires one shared native mask per batch")
        native_depth = native_shape[0]
        support_capacity = min(native_depth, 4) * 4 * 4
        if support_capacity > self.support_count:
            raise ValueError("configured EPA support_count is below the Q-3D support grid")
        valid = token_weight[0] > 0
        valid_indices = valid.nonzero(as_tuple=False).flatten()
        valid_count = int(valid_indices.numel())
        if valid_count == 0:
            raise ValueError("EPA adaptive support has no valid tokens")
        support_count = min(support_capacity, valid_count)
        if valid_count <= support_capacity:
            assignment = torch.arange(valid_count, device=normalized_tokens.device)
            matrix = torch.zeros(
                batch, valid_count, token_count,
                device=normalized_tokens.device, dtype=torch.float32,
            )
            matrix[:, assignment, valid_indices] = 1.0
            return matrix

        geometric_count = int(round(support_count * self.geometric_fraction))
        geometric_count = min(max(geometric_count, 1), support_count - 1)
        detail_count = support_count - geometric_count
        coordinates = self._coordinates(lattice_shape, device=normalized_tokens.device)
        valid_coordinates = coordinates.index_select(0, valid_indices)
        geom_indices = self._coverage_anchors(
            valid_count, geometric_count, device=normalized_tokens.device
        )
        # Detail windows sample the center of evenly spaced valid-token bins;
        # their score weights remain fully content-dependent.
        detail_indices = (
            (torch.arange(detail_count, device=normalized_tokens.device, dtype=torch.float32) + 0.5)
            * (valid_count / detail_count)
        ).round().long().clamp_max(valid_count - 1)
        geom_anchors = valid_coordinates.index_select(0, geom_indices)
        detail_anchors = valid_coordinates.index_select(0, detail_indices)
        geom_windows = torch.cdist(valid_coordinates, geom_anchors).argmin(dim=-1)
        detail_windows = torch.cdist(valid_coordinates, detail_anchors).argmin(dim=-1)
        if self.detail_score is None:
            raise RuntimeError("adaptive EPA support is missing its student detail scorer")
        detail_logits = self.detail_score(normalized_tokens).squeeze(-1)

        matrix = torch.zeros(
            batch, support_count, token_count,
            device=normalized_tokens.device, dtype=torch.float32,
        )
        for row in range(geometric_count):
            local = geom_windows.eq(row).nonzero(as_tuple=False).flatten()
            indices = valid_indices.index_select(0, local)
            weight = token_weight.index_select(1, indices)
            matrix[:, row, indices] = weight / weight.sum(dim=1, keepdim=True)
        for row in range(detail_count):
            local = detail_windows.eq(row).nonzero(as_tuple=False).flatten()
            indices = valid_indices.index_select(0, local)
            logits = detail_logits.index_select(1, indices)
            weights = token_weight.index_select(1, indices)
            logits = logits + weights.clamp_min(1e-12).log()
            matrix[:, geometric_count + row, indices] = torch.softmax(logits, dim=-1)
        row_sums = matrix.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=0.0, atol=1e-6):
            raise RuntimeError("EPA support rows must sum to one")
        return matrix

    def forward(
        self,
        x: Tensor,
        valid_mask: Tensor,
        native_shape: tuple[int, int, int],
        stride_yx: int,
    ) -> Tensor:
        batch, channels, depth, height, width = x.shape
        token_weight = valid_mask.flatten(2).squeeze(1).float()
        token_mask = token_weight > 0
        mask_5d = token_mask.reshape(batch, 1, depth, height, width).to(dtype=x.dtype)
        # Invalid cells can contain arbitrary convolutional padding values;
        # remove them before every attention/FFN path and the residual.
        valid_x = x * mask_5d
        tokens = valid_x.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        qk = self.qk(normalized).reshape(batch, -1, 2, channels)
        q, k = qk[:, :, 0], qk[:, :, 1]
        spatial_v_full = self.spatial_value(normalized)
        channel_v = self.channel_value(normalized)

        q_spatial = q.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q_spatial = q_spatial.contiguous()
        if self.support_mode == "uniform":
            k_map = k.transpose(1, 2).reshape(batch, channels, depth, height, width)
            v_map = spatial_v_full.transpose(1, 2).reshape(
                batch, channels, depth, height, width
            )
            pooled_k, support_valid = _masked_support_pool(
                k_map,
                valid_mask,
                native_shape=native_shape,
                stride_yx=stride_yx,
                support_mode=self.support_mode,
            )
            pooled_v, v_support_valid = _masked_support_pool(
                v_map,
                valid_mask,
                native_shape=native_shape,
                stride_yx=stride_yx,
                support_mode=self.support_mode,
            )
            if not torch.equal(support_valid, v_support_valid):
                raise RuntimeError("uniform EPA K/V pooling produced mismatched support masks")
            k_support = pooled_k.flatten(2).transpose(1, 2)
            v_support = pooled_v.flatten(2).transpose(1, 2)
        else:
            support_matrix = self._build_adaptive_support(
                normalized,
                token_weight,
                native_shape,
                (depth, height, width),
            )
            k_support = torch.bmm(support_matrix.to(dtype=k.dtype), k)
            v_support = torch.bmm(support_matrix.to(dtype=spatial_v_full.dtype), spatial_v_full)
            support_valid = torch.ones(
                batch, support_matrix.shape[1], dtype=torch.bool, device=x.device
            )
        k_spatial = k_support.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        v_spatial = v_support.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        spatial_mask = support_valid[:, None, None, :]
        spatial = F.scaled_dot_product_attention(
            q_spatial,
            k_spatial,
            v_spatial,
            attn_mask=spatial_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        spatial = spatial.transpose(1, 2).reshape(batch, -1, channels)

        # Channel attention is C x C.  Its cosine scores only include valid
        # tokens, and padded values cannot contribute to the value aggregate.
        q_channel = q.reshape(batch, -1, self.heads, self.head_dim).permute(0, 2, 3, 1)
        k_channel = k.reshape(batch, -1, self.heads, self.head_dim).permute(0, 2, 3, 1)
        v_channel = channel_v.reshape(batch, -1, self.heads, self.head_dim).permute(0, 2, 3, 1)
        channel_weight = token_weight[:, None, None, :].to(dtype=q_channel.dtype)
        channel_sqrt_weight = channel_weight.clamp_min(0).sqrt()
        q_channel = F.normalize(q_channel * channel_sqrt_weight, p=2.0, dim=-1, eps=1e-6)
        k_channel = F.normalize(k_channel * channel_sqrt_weight, p=2.0, dim=-1, eps=1e-6)
        v_channel = v_channel * channel_weight
        channel_scores = torch.matmul(q_channel, k_channel.transpose(-2, -1))
        channel_weights = torch.softmax(channel_scores, dim=-1)
        channel = torch.matmul(channel_weights, v_channel)
        channel = channel.permute(0, 3, 1, 2).reshape(batch, -1, channels)

        mixed = self.branch_projection(torch.cat((spatial, channel), dim=-1))
        mixed = mixed * token_mask.unsqueeze(-1).to(dtype=mixed.dtype)
        output = valid_x + mixed.transpose(1, 2).reshape(
            batch, channels, depth, height, width
        )
        ff_input = output * mask_5d
        feed_forward = self.ff_pointwise(
            self.ff_activation(self.ff_norm(self.ff_depthwise(ff_input)))
        )
        return (output + feed_forward) * mask_5d


class _EPAStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        support_mode: str,
        support_count: int,
        geometric_fraction: float,
    ) -> None:
        super().__init__()
        self.down = _ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            activation=True,
        )
        self.blocks = nn.ModuleList(
            [
                _EPA3DBlock(
                    out_channels, support_mode, support_count, geometric_fraction
                )
                for _ in range(2)
            ]
        )

    def forward(
        self,
        x: Tensor,
        *,
        native_shape: tuple[int, int, int],
        stride_yx: int,
    ) -> Tensor:
        x = self.down(x)
        mask = _valid_grid_mask(
            x.shape[0],
            x.shape[2],
            x.shape[3],
            x.shape[4],
            native_shape=native_shape,
            stride_yx=stride_yx,
            device=x.device,
        )
        for block in self.blocks:
            x = block(x, mask, native_shape, stride_yx)
        return x


class CandidateQ3DStudent(nn.Module):
    """Native-volume UNETR++-lite student.

    Args:
        config: Optional nested Q-3D config.  The relevant keys live under
            ``student`` (channels, arm, attention, vq, residual_vq); a direct
            student mapping is also accepted.
        codebook: Optional ``[K, d]`` tensor, or mapping with ``codes`` or
            ``codebook`` and optional ``version``.  It is normalized and kept
            as a buffer, never as a trainable parameter.

    ``forward`` returns native-grid logits and the stage-4 representation.
    In training mode it also returns the auxiliary SDF head.  VQ arms add
    assignment probabilities, hard indices and straight-through vectors.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        codebook: Tensor | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = _read_config(config)
        channels = cfg["channels"]
        self.arm = cfg["arm"]
        self.in_channels = cfg["in_channels"]
        self.num_classes = cfg["num_classes"]
        self.codebook_dim = cfg["codebook_dim"]
        self.codebook_size = cfg["codebook_size"]
        self.tau = cfg["tau"]
        self.support_mode = cfg["support_mode"]
        self.teacher_channels = cfg["teacher_channels"]
        self.use_vq = self.arm in _VQ_ARMS
        self.use_feature_adapter = self.arm in {"C1", *_VQ_ARMS}
        self.code_only = self.arm == "B1"
        self.uses_dense_skips = not self.code_only

        self.stem = nn.Sequential(
            _ConvNormAct(
                self.in_channels,
                channels[0],
                kernel_size=(1, 3, 3),
                activation=True,
            ),
            _ResidualConv3D(channels[0], channels[0]),
        )
        self.stage1 = nn.Sequential(
            _ConvNormAct(
                channels[0], channels[1], kernel_size=(1, 3, 3),
                stride=(1, 2, 2), activation=True,
            ),
            _ResidualConv3D(channels[1], channels[1]),
        )
        self.stage2 = nn.Sequential(
            _ConvNormAct(
                channels[1], channels[2], kernel_size=(1, 3, 3),
                stride=(1, 2, 2), activation=True,
            ),
            _ResidualConv3D(channels[2], channels[2]),
        )
        self.stage3 = _EPAStage(
            channels[2], channels[3], self.support_mode, cfg["support_count"],
            cfg["geometric_fraction"],
        )
        self.stage4 = _EPAStage(
            channels[3], channels[4], self.support_mode, cfg["support_count"],
            cfg["geometric_fraction"],
        )

        self.adapter = (
            nn.Sequential(
                nn.Conv3d(channels[4], 128, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv3d(128, self.codebook_dim, kernel_size=1, bias=True),
            )
            if self.use_feature_adapter
            else None
        )
        if self.arm == "C1":
            if self.teacher_channels is None or self.teacher_channels <= 0:
                raise ValueError(
                    "C1 requires model.teacher.feature_channels for exact F_T_aln matching"
                )
            self.continuous_feature_head = nn.Conv3d(
                self.codebook_dim, self.teacher_channels, kernel_size=1, bias=True
            )
        else:
            self.continuous_feature_head = None
        if self.use_vq:
            self.quantized_projector = nn.Conv3d(
                self.codebook_dim, channels[4], kernel_size=1, bias=True
            )
            self.vq_gate = nn.Parameter(torch.zeros(()))
            base_codes, version_id, version_label, initialized = self._prepare_codebook(codebook, cfg)
        else:
            self.quantized_projector = None
            self.register_parameter("vq_gate", None)
            base_codes = torch.empty((0, 0), dtype=torch.float32)
            version_id, version_label, initialized = 0, "", False
        self.register_buffer("C0", base_codes, persistent=True)
        # The engine installs the frozen tokenizer C0 after stage Q and updates
        # this scalar in-place.  A zero ID means the placeholder buffer is not
        # yet safe to use for a VQ forward.
        self.register_buffer("codebook_version", torch.tensor(version_id, dtype=torch.int64))
        self.register_buffer("codebook_is_frozen", torch.tensor(initialized, dtype=torch.bool))
        self.codebook_version_label = version_label

        self.residual_level_enabled = self.use_vq and cfg["residual_enabled"]
        if self.num_classes != 4 or self.in_channels != 1:
            raise ValueError("Q-3D student contract requires one input channel and four classes")
        if cfg["residual_enabled"] and not self.use_vq:
            raise ValueError("the optional residual second level is available only to VQ arms")
        if self.residual_level_enabled:
            self.residual_projector = nn.Conv3d(
                self.codebook_dim, channels[4], kernel_size=1, bias=True
            )
            self.residual_gate = nn.Parameter(torch.zeros(()))
            second_size = cfg["residual_codebook_size"]
            if second_size != self.codebook_size:
                raise ValueError("residual VQ level must reuse the frozen C0 codebook size")
            self.register_buffer("_empty_residual_codebook", torch.empty((0, 0)), persistent=False)
        else:
            self.residual_projector = None
            self.register_parameter("residual_gate", None)
            self.register_buffer("_empty_residual_codebook", torch.empty((0, 0)), persistent=False)

        # Four reductions, each matching a same-stage encoder skip.
        decoder_channels = (channels[3], channels[2], channels[1], channels[0])
        decoder_inputs = (channels[4], *decoder_channels[:-1])
        self.decoder_reduce = nn.ModuleList(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=True)
            for in_ch, out_ch in zip(decoder_inputs, decoder_channels)
        )
        self.decoder_blocks = nn.ModuleList(
            _ResidualConv3D(out_ch, out_ch) for out_ch in decoder_channels
        )
        self.logits_head = nn.Conv3d(channels[0], self.num_classes, kernel_size=1)
        self.sdf_head = (
            nn.Sequential(
                nn.Conv3d(channels[0], self.num_classes, kernel_size=1), nn.Tanh()
            )
            if cfg["sdf_enabled"]
            else None
        )

    @staticmethod
    def _prepare_codebook(
        codebook: Tensor | Mapping[str, Any] | None, cfg: Mapping[str, Any]
    ) -> tuple[Tensor, int, str, bool]:
        expected = (int(cfg["codebook_size"]), int(cfg["codebook_dim"]))
        if codebook is None:
            # The engine creates the student before Q has frozen C0.  Keep a
            # correctly shaped placeholder for checkpoint compatibility, but
            # do not invent codes that could accidentally enter training.
            return torch.zeros(expected, dtype=torch.float32), 0, "", False
        if not isinstance(codebook, Mapping):
            raise ValueError("VQ codebook requires frozen codes, version, sha256, and frozen=true")
        if codebook.get("schema", "self_audit.q3d.codebook.v1") != "self_audit.q3d.codebook.v1":
            raise ValueError("VQ codebook mapping schema is unsupported")
        version = codebook.get("version")
        if not isinstance(version, (str, int)) or version == "" or version == 0:
            raise ValueError("frozen VQ codebook mapping must include a nonzero version")
        if codebook.get("frozen") is not True:
            raise ValueError("VQ codebook mapping must declare frozen=true")
        if codebook.get("initialized") is not True:
            raise ValueError("VQ codebook mapping must declare initialized=true")
        recorded_sha = codebook.get("sha256")
        if not isinstance(recorded_sha, str) or len(recorded_sha) != 64 or any(
            char not in "0123456789abcdef" for char in recorded_sha.lower()
        ):
            raise ValueError("frozen VQ codebook mapping must include a SHA-256")
        value = codebook.get("codes", codebook.get("codebook", codebook.get("C0")))
        if value is None:
            raise ValueError("codebook mapping must provide codes or codebook")
        value = torch.as_tensor(value, dtype=torch.float32).detach().clone()
        expected = (int(cfg["codebook_size"]), int(cfg["codebook_dim"]))
        if tuple(value.shape) != expected:
            raise ValueError(f"codebook must have shape {expected}, got {tuple(value.shape)}")
        if int(codebook.get("size", -1)) != expected[0] or int(
            codebook.get("dimension", -1)
        ) != expected[1]:
            raise ValueError("VQ codebook size/dimension metadata does not match its codes")
        if not torch.isfinite(value).all():
            raise ValueError("codebook values must be finite")
        if torch.any(torch.linalg.vector_norm(value, dim=-1) <= 1e-12):
            raise ValueError("codebook vectors must have nonzero norm")
        digest = CandidateQ3DStudent._tensor_sha256(value)
        if digest != recorded_sha.lower():
            raise ValueError("frozen VQ codebook SHA-256 does not match its codes")
        version_id = CandidateQ3DStudent._version_id(version)
        if version_id <= 0:
            raise ValueError("frozen VQ codebook version must include a positive vN suffix or integer")
        return value, version_id, version, True

    @staticmethod
    def _version_id(version: str | int) -> int:
        if isinstance(version, int):
            return version
        suffix = version.rsplit("v", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    @staticmethod
    def _tensor_sha256(value: Tensor) -> str:
        payload = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    @property
    def codebook(self) -> Tensor:
        """Compatibility view of the frozen teacher-anchored C0 buffer."""
        return self.C0

    @property
    def codebook_version_id(self) -> Tensor:
        """Compatibility alias for the engine-mutated persistent version ID."""
        return self.codebook_version

    @property
    def codebook_version_name(self) -> str | int | None:
        if not self.codebook_frozen:
            return None
        version_id = int(self.codebook_version.item())
        return self.codebook_version_label or f"q3d_teacher_codebook.v{version_id}"

    @property
    def codebook_frozen(self) -> bool:
        return bool(
            self.use_vq and self.codebook_is_frozen.item()
            and int(self.codebook_version.item()) > 0
        )

    @property
    def codebook_sha256(self) -> str | None:
        if not self.codebook_frozen:
            return None
        return self._tensor_sha256(self.C0)

    def codebook_metadata(self) -> dict[str, Any]:
        frozen = self.codebook_frozen
        return {
            "version": self.codebook_version_name,
            "sha256": self.codebook_sha256,
            "size": self.codebook_size,
            "dimension": self.codebook_dim,
            "initialized": frozen,
            "frozen": frozen,
        }

    @torch.no_grad()
    def install_codebook(self, mapping: Mapping[str, Any]) -> dict[str, Any]:
        """Install tokenizer C0 only after validating its frozen artifact metadata."""
        if not self.use_vq:
            raise ValueError("continuous Q-3D student cannot install a VQ codebook")
        codes, version_id, version_label, initialized = self._prepare_codebook(
            mapping, {
                "codebook_size": self.codebook_size,
                "codebook_dim": self.codebook_dim,
            }
        )
        if not initialized:
            raise ValueError("Q-3D codebook installation requires initialized frozen C0")
        if self.codebook_frozen:
            if (
                self.codebook_version_name != version_label
                or self.codebook_sha256 != self._tensor_sha256(codes)
            ):
                raise ValueError("cannot replace an already frozen Q-3D student codebook")
            return self.codebook_metadata()
        self.C0.copy_(codes.to(device=self.C0.device, dtype=self.C0.dtype))
        self.codebook_version.fill_(version_id)
        self.codebook_is_frozen.fill_(True)
        self.codebook_version_label = version_label
        return self.codebook_metadata()

    @property
    def residual_codebook(self) -> Tensor:
        return self.C0 if self.residual_level_enabled else self._empty_residual_codebook

    def get_extra_state(self) -> dict[str, Any]:
        """Persist the non-tensor identity of the frozen tokenizer codebook."""
        return {"schema": "self_audit.q3d.codebook.v1", **self.codebook_metadata()}

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != "self_audit.q3d.codebook.v1":
            raise ValueError("Q-3D student codebook metadata is missing or unsupported")
        version = state.get("version")
        recorded_sha = state.get("sha256")
        frozen = state.get("frozen")
        initialized = state.get("initialized")
        if int(state.get("size", -1)) != self.codebook_size or int(
            state.get("dimension", -1)
        ) != self.codebook_dim:
            raise ValueError("Q-3D student codebook size/dimension metadata mismatch")
        version_id = int(self.codebook_version.item())
        if self.use_vq and version_id > 0:
            if (
                frozen is not True
                or initialized is not True
                or not bool(self.codebook_is_frozen.item())
                or not isinstance(version, (str, int))
            ):
                raise ValueError("Q-3D frozen C0 checkpoint is missing version provenance")
            if self._version_id(version) != version_id:
                raise ValueError("Q-3D frozen C0 checkpoint version ID mismatch")
            if not isinstance(recorded_sha, str) or self._tensor_sha256(self.C0) != recorded_sha:
                raise ValueError("Q-3D frozen C0 checkpoint SHA-256 mismatch")
            self.codebook_version_label = version
        elif version_id == 0 and not bool(self.codebook_is_frozen.item()):
            if (
                frozen is not False or initialized is not False
                or version is not None or recorded_sha is not None
            ):
                raise ValueError("uninitialized Q-3D C0 checkpoint carries frozen provenance")
            self.codebook_version_label = ""
        else:
            raise ValueError("Q-3D C0 version/frozen metadata is inconsistent")

    @staticmethod
    def _geometry_shape(geometry: Any) -> tuple[int, int, int] | None:
        if geometry is None:
            return None
        try:
            from ..candidate_a3d.geometry import canonical_geometry

            canonical = canonical_geometry(geometry)
            shape = tuple(int(value) for value in canonical["shape"])
            if len(shape) != 3:
                raise ValueError("geometry.shape must be D,H,W")
            return shape
        except ImportError as exc:  # pragma: no cover - package import failure
            raise ValueError("native geometry validation is unavailable") from exc

    @staticmethod
    def _pad_inplane(x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        height, width = x.shape[-2:]
        pad_h = (-height) % 16
        pad_w = (-width) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
        return x, (height, width)

    def _quantize(
        self, u: Tensor, codebook: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized = F.normalize(u, p=2.0, dim=1, eps=1e-6)
        codes = F.normalize(codebook, p=2.0, dim=-1, eps=1e-6)
        # [B,K,D,H,W], then keep the hard indices on the stage-4 lattice.
        cosine = torch.einsum("bcdhw,kc->bkdhw", normalized, codes)
        pi = torch.softmax(cosine / self.tau, dim=1)
        indices = cosine.argmax(dim=1)
        hard = F.embedding(indices, codes).permute(0, 4, 1, 2, 3).contiguous()
        straight_through = normalized + (hard - normalized).detach()
        return pi, indices, straight_through

    def _decode(self, f4: Tensor, skips: tuple[Tensor, ...], native_hw: tuple[int, int]) -> Tensor:
        x = f4
        # Encoder features in decode order: F3, S2, S1, F0.
        for index, (reduce, block) in enumerate(zip(self.decoder_reduce, self.decoder_blocks)):
            skip = None if self.code_only else skips[index]
            target_size = skip.shape[-3:] if skip is not None else (
                x.shape[-3], x.shape[-2] * 2, x.shape[-1] * 2
            )
            x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
            x = reduce(x)
            if skip is not None:
                x = x + skip
            x = block(x)
        return x[..., : native_hw[0], : native_hw[1]]

    def forward(self, x: Tensor, geometry: Any = None) -> dict[str, Any]:
        if x.ndim != 5:
            raise ValueError("Q-3D student input must have shape [B,1,D,H,W]")
        if x.shape[0] <= 0:
            raise ValueError("Q-3D student batch size must be positive")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Q-3D student expects {self.in_channels} input channel")
        if any(int(size) <= 0 for size in x.shape[2:]):
            raise ValueError("Q-3D student requires positive native D,H,W")
        if not x.is_floating_point():
            raise TypeError("Q-3D student input must be floating point")
        stored_shape = tuple(int(value) for value in x.shape[-3:])
        geometry_shape = self._geometry_shape(geometry)
        if geometry_shape is None:
            native_shape = stored_shape
        else:
            native_shape = geometry_shape
            if native_shape[0] != stored_shape[0]:
                raise ValueError("Q-3D geometry cannot pad or crop the native z axis")
            if native_shape[1] > stored_shape[1] or native_shape[2] > stored_shape[2]:
                raise ValueError("input is smaller than the full native geometry")
            # Remove any caller-provided in-plane padding before normalization,
            # InstanceNorm, attention, or convolution can observe its values.
            x = x[..., : native_shape[1], : native_shape[2]]
        if not torch.isfinite(x).all():
            raise ValueError("Q-3D student input must be finite")
        if torch.any(x < 0) or torch.any(x > 1):
            raise ValueError("Q-3D student input must be normalized to [0,1]")

        padded, native_hw = self._pad_inplane(x)
        f0 = self.stem(padded)
        s1 = self.stage1(f0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2, native_shape=native_shape, stride_yx=8)
        f4 = self.stage4(s3, native_shape=native_shape, stride_yx=16)

        u: Tensor | None = None
        pi: Tensor | None = None
        codes: Tensor | None = None
        quantized: Tensor | None = None
        residual_result: dict[str, Tensor] = {}
        decoder_f4 = f4
        if self.adapter is not None:
            u = F.normalize(self.adapter(f4), p=2.0, dim=1, eps=1e-6)
        feature_kd: Tensor | None = None
        if self.training and self.continuous_feature_head is not None:
            assert u is not None
            feature_kd = F.normalize(
                self.continuous_feature_head(u), p=2.0, dim=1, eps=1e-6
            )
        if self.use_vq:
            assert u is not None and self.quantized_projector is not None
            if not self.codebook_frozen:
                raise RuntimeError(
                    "Q-3D VQ forward requires frozen tokenizer C0 after the Q gate"
                )
            pi, codes, quantized = self._quantize(u, self.codebook)
            quantized_features = self.quantized_projector(quantized)
            decoder_f4 = (
                quantized_features
                if self.code_only
                else f4 + self.vq_gate * quantized_features
            )
            if self.residual_level_enabled:
                assert self.residual_projector is not None
                residual_u = F.normalize(
                    u - quantized.detach(), p=2.0, dim=1, eps=1e-6
                )
                residual_pi, residual_codes, residual_quantized = self._quantize(
                    residual_u, self.C0
                )
                decoder_f4 = decoder_f4 + self.residual_gate * self.residual_projector(
                    residual_quantized
                )
                residual_result = {
                    "residual_u": residual_u,
                    "residual_pi": residual_pi,
                    "residual_codes": residual_codes,
                    "residual_quantized": residual_quantized,
                }

        decoded = self._decode(decoder_f4, (s3, s2, s1, f0), native_hw)
        # Decoder output includes padded boundary context internally; the final
        # logits and train-only SDF are strictly native-grid tensors.
        logits = self.logits_head(decoded)[..., : native_shape[1], : native_shape[2]]
        result: dict[str, Any] = {
            "logits": logits,
            "f4": f4,
            "u": u,
            "pi": pi,
            "codes": codes,
            "quantized": quantized,
            "feature_kd": feature_kd,
            "codebook_version": self.codebook_version_name,
            "codebook_version_id": self.codebook_version,
        }
        if self.training and self.sdf_head is not None:
            assert self.sdf_head is not None
            result["sdf"] = self.sdf_head(decoded)[..., : native_shape[1], : native_shape[2]]
        result.update(residual_result)
        return result

    def infer_logits(self, x: Tensor, geometry: Any = None) -> Tensor:
        """Return the deploy output while preserving the caller's train/eval mode."""
        return self.forward(x, geometry=geometry)["logits"]


__all__ = ["CandidateQ3DStudent"]
