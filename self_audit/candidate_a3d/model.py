"""Code-only 3-D student for Candidate A.

The encoder and typed value heads create the finite measurement values.  The
decoder is deliberately independent: it receives only IDs, quantized values,
and canonical public volume geometry.  No raw image feature, teacher feature,
or skip connection can reach the decoder.  Boundary value heads use the bank's
declared nearest-native-voxel point; region heads pool the declared native
ROI.  The encoder is stride one so those footprints align exactly with its
input-resolution feature map.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .geometry import canonical_float32_geometry


def _embedding_with_stable_rng(num_embeddings: int, embedding_dim: int) -> nn.Embedding:
    """Allocate a bank-sized table without shifting shared RNG initialization."""
    state = torch.random.get_rng_state()
    try:
        return nn.Embedding(num_embeddings, embedding_dim)
    finally:
        torch.random.set_rng_state(state)


def _semantic_embedding(query_ids: Sequence[str], embedding_dim: int) -> nn.Embedding:
    table = _embedding_with_stable_rng(len(query_ids), embedding_dim)
    with torch.no_grad():
        for row, query_id in enumerate(query_ids):
            seed = int.from_bytes(hashlib.sha256(query_id.encode("utf-8")).digest()[:8], "big")
            generator = torch.Generator(device="cpu").manual_seed(seed)
            table.weight[row].copy_(torch.randn(embedding_dim, generator=generator))
    return table


@dataclass(frozen=True)
class StudentConfig3D:
    in_channels: int = 1
    num_classes: int = 4
    width: int = 16
    decoder_dim: int = 32
    num_heads: int = 4
    chunk_size: int = 4096
    local_queries: bool = True
    depth: int = 2

    def __post_init__(self) -> None:
        integer_fields = {
            "in_channels": self.in_channels, "num_classes": self.num_classes,
            "width": self.width, "decoder_dim": self.decoder_dim,
            "num_heads": self.num_heads, "chunk_size": self.chunk_size,
            "depth": self.depth,
        }
        if any(isinstance(value, bool) or int(value) != value for value in integer_fields.values()):
            raise ValueError("A-3D student architecture fields must be integers")
        if self.in_channels != 1 or self.num_classes != 4:
            raise ValueError("A-3D student expects one input and four output classes")
        if self.width < 4 or self.decoder_dim < 4 or self.num_heads < 1:
            raise ValueError("width, decoder_dim and num_heads must be positive")
        if self.decoder_dim % self.num_heads:
            raise ValueError("decoder_dim must be divisible by num_heads")
        if self.chunk_size < 1 or self.depth < 1:
            raise ValueError("chunk_size and depth must be positive")
        if not isinstance(self.local_queries, bool):
            raise ValueError("local_queries must be boolean")


def _kind_code(kind: str) -> int:
    if kind == "REGION":
        return 0
    if kind == "BOUNDARY":
        return 1
    raise ValueError(f"unknown query kind: {kind!r}")


def _geometry(value: Mapping[str, Any]) -> tuple[tuple[int, int, int], tuple[float, float, float], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("A-3D geometry must be a mapping")
    required = {"shape", "spacing_zyx", "affine_zyx", "context"}
    if set(value) != required:
        raise ValueError("A-3D geometry must contain exactly shape, spacing_zyx, affine_zyx and context")
    canonical = canonical_float32_geometry(value)
    shape = tuple(int(item) for item in canonical["shape"])
    spacing = tuple(float(item) for item in canonical["spacing_zyx"])
    return shape, spacing, canonical


class CandidateA3DStudent(nn.Module):
    """Stride-one local/global encoder with a finite code-only decoder."""

    BOUNDARY_SAMPLE_MODE = "nearest_voxel"

    def __init__(self, bank: Any, config: StudentConfig3D) -> None:
        super().__init__()
        if not isinstance(config, StudentConfig3D):
            raise TypeError("config must be StudentConfig3D")
        required = ("query", "descriptors", "region_slices", "point_index")
        if any(not callable(getattr(bank, name, None)) for name in required) or not hasattr(bank, "__len__"):
            raise TypeError("bank must expose query, descriptors, region_slices and point_index")
        self.bank = bank
        self.config = config
        queries = tuple(bank)
        if not queries:
            raise ValueError("query bank must not be empty")
        self.query_ids = tuple(str(q.query_id) for q in queries)
        self._id_to_index = {q.query_id: i for i, q in enumerate(queries)}
        self.num_queries = len(queries)
        self.query_kinds = torch.tensor([_kind_code(q.kind) for q in queries], dtype=torch.long)
        descriptors = bank.descriptors(device="cpu", dtype=torch.float32)
        expected = (self.num_queries, 2 + len(getattr(bank.config, "classes", (1, 2, 3))) + 4)
        if descriptors.shape != expected:
            raise ValueError(f"3-D bank descriptors must have shape {expected}")
        if not torch.isfinite(descriptors).all():
            raise ValueError("bank descriptors must be finite")
        self.register_buffer("query_descriptors", descriptors, persistent=True)

        w, d = config.width, config.decoder_dim
        layers: list[nn.Module] = [nn.Conv3d(config.in_channels, w, 3, padding=1), nn.GELU()]
        for _ in range(config.depth - 1):
            layers.extend((nn.Conv3d(w, w, 3, padding=1), nn.GELU()))
        self.encoder = nn.Sequential(*layers)
        self.query_value = nn.Sequential(nn.Linear(descriptors.shape[1], w), nn.GELU(), nn.Linear(w, w))
        # Physical spacing and native extent are part of the value-head input:
        # equal voxel tensors at different dz/dy/dx must not be forced to
        # predict identical signed-distance values.  This is public geometry,
        # not a hidden feature path, and is shared by local/global ablations.
        self.geometry_context = nn.Sequential(nn.Linear(6, w), nn.GELU(), nn.Linear(w, w))
        value_layers = lambda: nn.Sequential(nn.Linear(4 * w, w), nn.GELU(), nn.Linear(w, 1))
        self.value_heads = nn.ModuleDict({"REGION": value_layers(), "BOUNDARY": value_layers()})

        # Query ID semantic seeds are stable when REGION-only/BOTH banks have
        # different row counts.  This prevents variable table allocation from
        # changing common shared decoder/policy initialization.
        self.id_embedding = _semantic_embedding(self.query_ids, d)
        self.policy_id_embedding = _semantic_embedding(self.query_ids, w)
        self.type_embedding = nn.Embedding(2, d)
        self.policy_type_embedding = nn.Embedding(2, w)
        self.value_embedding = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.policy_value_embedding = nn.Sequential(nn.Linear(1, w), nn.GELU(), nn.Linear(w, w))
        self.decoder_descriptor = nn.Sequential(nn.Linear(descriptors.shape[1], d), nn.GELU(), nn.Linear(d, d))
        self.policy_descriptor = nn.Sequential(nn.Linear(descriptors.shape[1], w), nn.GELU(), nn.Linear(w, w))
        self.coordinate_embedding = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.cross_attention = nn.MultiheadAttention(d, config.num_heads, dropout=0.0, batch_first=True)
        self.decoder_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, config.num_classes))
        self.null_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.null_token, std=0.02)
        self.policy_head = nn.Sequential(nn.Linear(4 * w, w), nn.GELU(), nn.Linear(w, 1))
        self.policy_stop = nn.Sequential(nn.Linear(3 * w, w), nn.GELU(), nn.Linear(w, 1))

    # ---------------------------------------------------------------- inputs
    def _indices(self, ids: Any, *, device: torch.device) -> Tensor:
        if isinstance(ids, Tensor):
            if ids.ndim != 1:
                raise ValueError("query IDs must be one-dimensional")
            if ids.dtype.is_floating_point or ids.dtype.is_complex:
                if not torch.isfinite(ids).all() or not torch.equal(ids, ids.round()):
                    raise ValueError("numeric query IDs must be finite integers")
            raw = ids.detach().cpu().tolist()
        else:
            raw = list(ids) if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)) else [ids]
        result: list[int] = []
        for value in raw:
            if isinstance(value, bool):
                raise ValueError("query IDs must not be bool")
            if value in self._id_to_index:
                result.append(self._id_to_index[value])
            elif isinstance(value, int) and 0 <= value < self.num_queries:
                result.append(int(value))
            else:
                raise ValueError(f"unknown query ID: {value!r}")
        output = torch.tensor(result, dtype=torch.long, device=device)
        if output.numel() != output.unique().numel():
            raise ValueError("code query IDs must not repeat")
        return output

    def _batch_indices(self, code_ids: Any, batch: int, *, device: torch.device) -> list[Tensor]:
        if code_ids is None:
            return [torch.empty(0, dtype=torch.long, device=device) for _ in range(batch)]
        if isinstance(code_ids, Tensor) and code_ids.ndim == 2:
            if code_ids.shape[0] != batch:
                raise ValueError("code ID batch does not match input batch")
            rows = [code_ids[b] for b in range(batch)]
        elif isinstance(code_ids, Sequence) and not isinstance(code_ids, (str, bytes)) and len(code_ids) == batch and any(isinstance(row, (Tensor, list, tuple)) for row in code_ids):
            rows = list(code_ids)
        else:
            rows = [code_ids] * batch
        return [self._indices(row, device=device) for row in rows]

    @staticmethod
    def _batch_values(code_values: Any, rows: list[Tensor], *, device, dtype) -> list[Tensor]:
        if code_values is None:
            return [torch.empty(row.numel(), device=device, dtype=dtype) for row in rows]
        values = torch.as_tensor(code_values, device=device, dtype=dtype)
        if values.ndim == 1:
            values = values.unsqueeze(0).expand(len(rows), -1)
        if values.ndim != 2 or values.shape[0] != len(rows):
            raise ValueError("code_values must have shape [K] or [B,K]")
        result = []
        for row_index, row in enumerate(rows):
            if values.shape[1] != row.numel():
                raise ValueError("code_values length must match code_ids")
            result.append(values[row_index])
        return result

    def _bounds(self, indices: Tensor, *, dtype) -> tuple[Tensor, Tensor]:
        kinds = self.query_kinds.to(indices.device)[indices]
        lower = torch.where(kinds == 1, torch.full_like(kinds, -1, dtype=dtype), torch.zeros_like(kinds, dtype=dtype))
        return lower, torch.ones_like(lower)

    # ---------------------------------------------------------------- encoder
    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[1] != self.config.in_channels:
            raise ValueError("expected [B,1,D,H,W] input")
        if not torch.isfinite(x).all():
            raise ValueError("input contains NaN or Inf")
        return self.encoder(x)

    def _local_features(self, features: Tensor, geometry: Mapping[str, Any]) -> Tensor:
        shape, _, _ = _geometry(geometry)
        if tuple(features.shape[-3:]) != shape:
            raise ValueError("geometry shape does not match encoder feature shape")
        rows: list[Tensor] = []
        for q in tuple(self.bank):
            if q.kind == "REGION":
                z_slice, y_slice, x_slice = self.bank.region_slices(q, shape)
                patch = features[..., z_slice, y_slice, x_slice]
                if patch.shape[-3] == 0 or patch.shape[-2] == 0 or patch.shape[-1] == 0:
                    raise ValueError("region query resolved to an empty native footprint")
                rows.append(patch.mean(dim=(-3, -2, -1)))
            else:
                z, y, x = self.bank.point_index(q, shape)
                rows.append(features[..., z, y, x])
        local = torch.stack(rows, dim=1)
        return local if self.config.local_queries else torch.zeros_like(local)

    def predict_values(self, features: Tensor, geometry: Mapping[str, Any]) -> Tensor:
        if features.ndim != 5 or features.shape[1] != self.config.width:
            raise ValueError("expected encoder features [B,width,D,H,W]")
        if not torch.isfinite(features).all():
            raise ValueError("encoder features contain NaN or Inf")
        shape, spacing, _ = _geometry(geometry)
        local = self._local_features(features, geometry)
        global_context = features.mean(dim=(-3, -2, -1))[:, None, :].expand(-1, self.num_queries, -1)
        query = self.query_value(self.query_descriptors.to(features.device, features.dtype))[None].expand(features.shape[0], -1, -1)
        spacing_tensor = torch.tensor(spacing, device=features.device, dtype=features.dtype)
        extent_tensor = spacing_tensor * torch.tensor(shape, device=features.device, dtype=features.dtype)
        physical = torch.log(torch.cat((spacing_tensor, extent_tensor), dim=0).clamp_min(torch.finfo(features.dtype).tiny))
        geometry_context = self.geometry_context(physical)[None, None, :].expand(features.shape[0], self.num_queries, -1)
        joint = torch.cat((local, global_context, query, geometry_context), dim=-1)
        raw = torch.empty((features.shape[0], self.num_queries), device=features.device, dtype=features.dtype)
        for kind, code in (("REGION", 0), ("BOUNDARY", 1)):
            mask = self.query_kinds.to(features.device) == code
            if bool(mask.any()):
                raw[:, mask] = self.value_heads[kind](joint[:, mask]).squeeze(-1)
        kinds = self.query_kinds.to(features.device)
        return torch.where(kinds[None] == 1, raw.tanh(), raw.sigmoid())

    # ---------------------------------------------------------------- policy
    def policy_logits(self, features: Tensor, code_ids: Any, code_values: Any,
                      remaining_bits: int | Tensor | Sequence[float] | None = None) -> Tensor:
        if features.ndim != 5 or features.shape[1] != self.config.width:
            raise ValueError("expected encoder features [B,width,D,H,W]")
        batch = features.shape[0]
        rows = self._batch_indices(code_ids, batch, device=features.device)
        values = self._batch_values(code_values, rows, device=features.device, dtype=features.dtype)
        pooled = features.mean(dim=(-3, -2, -1))
        kinds = self.query_kinds.to(features.device)
        descriptors = self.policy_descriptor(self.query_descriptors.to(features.device, features.dtype))
        qbase = self.policy_id_embedding.weight + self.policy_type_embedding(kinds) + descriptors
        summary = torch.zeros(batch, self.config.width, device=features.device, dtype=features.dtype)
        used = torch.zeros(batch, self.num_queries, device=features.device, dtype=torch.bool)
        for b, row in enumerate(rows):
            if row.numel():
                used[b, row] = True
                token = self.policy_id_embedding(row) + self.policy_type_embedding(kinds[row])
                token = token + self.policy_value_embedding(values[b][:, None])
                summary[b] = token.mean(0)
        if remaining_bits is None:
            remaining = torch.ones(batch, device=features.device, dtype=features.dtype)
        else:
            remaining = torch.as_tensor(remaining_bits, device=features.device, dtype=features.dtype).flatten()
            if remaining.numel() == 1:
                remaining = remaining.expand(batch)
            if remaining.numel() != batch:
                raise ValueError("remaining_bits must be scalar or [B]")
            remaining = remaining.clamp(-1e6, 1e6) / 4096.0
        common = torch.cat((pooled, summary, remaining[:, None].expand(-1, self.config.width)), dim=-1)
        query_input = torch.cat((common[:, None].expand(-1, self.num_queries, -1), qbase[None].expand(batch, -1, -1)), dim=-1)
        logits = self.policy_head(query_input).squeeze(-1)
        if remaining_bits is not None:
            logits = logits.masked_fill(remaining[:, None] <= 0, float("-inf"))
        logits = logits.masked_fill(used, float("-inf"))
        stop = self.policy_stop(common).squeeze(-1)[:, None]
        return torch.cat((logits, stop), dim=-1)

    @property
    def stop_index(self) -> int:
        return self.num_queries

    # ---------------------------------------------------------------- decoder
    def _tokens(self, indices: Tensor, values: Tensor, *, dtype: torch.dtype) -> Tensor:
        if indices.numel() == 0:
            return self.null_token.to(device=values.device, dtype=dtype)
        if values.ndim != 1 or values.numel() != indices.numel() or not torch.isfinite(values).all():
            raise ValueError("decoder code values must be finite and one per selected ID")
        lower, upper = self._bounds(indices, dtype=dtype)
        if bool((values < lower).any()) or bool((values > upper).any()):
            raise ValueError("decoder requires bounded codec-dequantized values")
        order = torch.argsort(indices)
        indices, values = indices[order], values[order]
        kinds = self.query_kinds.to(indices.device)[indices]
        descriptors = self.query_descriptors.to(values.device, dtype=dtype)[indices]
        return (self.id_embedding(indices) + self.type_embedding(kinds) + self.decoder_descriptor(descriptors) + self.value_embedding(values[:, None])).unsqueeze(0)

    def decode(self, code_ids: Any, code_values: Any, geometry: Mapping[str, Any], chunk_size: int | None = None) -> Tensor:
        """Decode only code and public canonical geometry, in voxel chunks."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        if isinstance(code_ids, Tensor) and code_ids.ndim == 2:
            batch = code_ids.shape[0]
        elif isinstance(code_ids, Sequence) and not isinstance(code_ids, (str, bytes)) and code_ids and isinstance(code_ids[0], (list, tuple, Tensor)):
            batch = len(code_ids)
        else:
            batch = 1
        rows = self._batch_indices(code_ids, batch, device=device)
        values = self._batch_values(code_values, rows, device=device, dtype=dtype)
        (depth, height, width), spacing, canonical = _geometry(geometry)
        del canonical  # validation above is intentional; decoder uses native shape/spacing below
        chunk = int(chunk_size or self.config.chunk_size)
        if chunk <= 0:
            raise ValueError("chunk_size must be positive")
        z = torch.linspace(-1, 1, depth, device=device, dtype=dtype)
        y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        n = depth * height * width
        log_spacing = torch.log1p(torch.tensor(spacing, device=device, dtype=dtype))
        coords = torch.cat((torch.stack((zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)), dim=-1),
                            log_spacing.expand(n, -1)), dim=-1)
        outputs = []
        for row, value in zip(rows, values):
            tokens = self._tokens(row, value, dtype=dtype)
            pieces = []
            for start in range(0, n, chunk):
                query = self.coordinate_embedding(coords[start:start + chunk]).unsqueeze(0)
                attended, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
                pieces.append(self.decoder_head(query + attended).squeeze(0))
            flat = torch.cat(pieces, dim=0)
            outputs.append(flat.transpose(0, 1).reshape(self.config.num_classes, depth, height, width))
        return torch.stack(outputs, dim=0)


__all__ = ["StudentConfig3D", "CandidateA3DStudent"]
