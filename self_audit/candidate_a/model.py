"""Student architecture for candidate A.

This is the small reference 2-D/2.5-D implementation.  The student encoder
predicts one bounded value for every fixed :class:`QueryBank` entry.  K1 and
``MeasurementCodec`` own rollout, budget and quantization; this module never
constructs a teacher target or a budget from a feature tensor.

The decoder consumes only ``(query_ids, quantized_student_values, geometry)``.
It has no skip connection or hidden feature argument.  The coordinate grid is
decoded in chunks to keep the ``H*W`` cross-attention matrix bounded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StudentConfig:
    """Only architecture fields belong to the student checkpoint."""

    in_channels: int = 3
    num_classes: int = 4
    width: int = 32
    decoder_dim: int = 48
    num_heads: int = 4
    chunk_size: int = 4096

    def __post_init__(self) -> None:
        if self.in_channels != 3 or self.num_classes != 4:
            raise ValueError("candidate A reference student expects 3 input and 4 output channels")
        if self.width < 4 or self.decoder_dim < 4:
            raise ValueError("width and decoder_dim must be at least four")
        if self.num_heads < 1 or self.decoder_dim % self.num_heads:
            raise ValueError("decoder_dim must be divisible by num_heads")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")


def _kind_code(kind: Any) -> int:
    text = str(kind).upper()
    return int(text in {"BOUNDARY", "B", "1"} or "BOUND" in text or "DIST" in text)


def _query_id(query: Any) -> Any:
    if isinstance(query, Mapping):
        return query.get("id", query.get("query_id"))
    return getattr(query, "id", getattr(query, "query_id", query))


class CandidateAStudent(nn.Module):
    """A deterministic image-to-code student and code-to-mask decoder.

    Parameters
    ----------
    bank:
        A concrete finite ``QueryBank``.  It must be iterable and expose
        ``descriptors(device=..., dtype=...)`` returning exactly ``[Q, 8]``.
    config:
        Architecture-only :class:`StudentConfig`.
    """

    def __init__(self, bank: Any, config: StudentConfig) -> None:
        super().__init__()
        if not isinstance(config, StudentConfig):
            raise TypeError("config must be StudentConfig; bank/codec/rollout fields do not belong here")
        self.bank = bank
        self.config = config
        try:
            queries = tuple(bank)
        except TypeError as exc:
            raise TypeError("bank must be a finite iterable of query records") from exc
        if not queries:
            raise ValueError("query bank must not be empty")
        self.query_ids = tuple(_query_id(q) for q in queries)
        if any(q is None for q in self.query_ids) or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("bank query IDs must be unique and non-null")
        self._id_to_index = {query_id: index for index, query_id in enumerate(self.query_ids)}
        self.num_queries = len(self.query_ids)
        self.query_kinds = torch.tensor([_kind_code(getattr(q, "kind", q.get("kind", q.get("type")))
                                                  if isinstance(q, Mapping)
                                                  else getattr(q, "kind", getattr(q, "type", "REGION")))
                                        for q in queries], dtype=torch.long)
        try:
            descriptors = bank.descriptors(device="cpu", dtype=torch.float32)
        except TypeError:
            descriptors = bank.descriptors()
        descriptors = torch.as_tensor(descriptors, dtype=torch.float32)
        if descriptors.ndim != 2 or descriptors.shape[0] != self.num_queries:
            raise ValueError(f"bank descriptors must have shape [{self.num_queries}, D]")
        # QueryBank controls may intentionally restrict the foreground class
        # set.  Its canonical layout is ``type(2), class(C), (u,v,scale)``;
        # embed C<3 controls into the fixed three-class student descriptor
        # without inventing a location/type value.  Arbitrary dimensions are
        # rejected rather than silently truncated.
        if descriptors.shape[1] != 8:
            class_count = descriptors.shape[1] - 5
            if class_count not in (1, 2) or descriptors.shape[1] < 6:
                raise ValueError(f"bank descriptors must have fixed canonical dimension 8 (received {tuple(descriptors.shape)})")
            canonical = torch.zeros((self.num_queries, 8), dtype=descriptors.dtype)
            canonical[:, : 2 + class_count] = descriptors[:, : 2 + class_count]
            canonical[:, 5:] = descriptors[:, 2 + class_count :]
            descriptors = canonical
        if not torch.isfinite(descriptors).all():
            raise ValueError("bank descriptors must be finite")
        self.register_buffer("query_descriptors", descriptors, persistent=True)

        w, d = config.width, config.decoder_dim
        # No BatchNorm or Dropout: K1's before/after trials see the same
        # decoder state in train and eval mode.
        self.encoder = nn.Sequential(
            nn.Conv2d(config.in_channels, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, w, 1), nn.GELU(),
        )
        self.query_value = nn.Sequential(nn.Linear(8, w), nn.GELU(), nn.Linear(w, w))
        self.value_head = nn.Sequential(nn.Linear(2 * w, w), nn.GELU(), nn.Linear(w, 1))

        # The policy and decoder use learned ID/type descriptors in addition
        # to the fixed bank metadata.  IDs are index-backed internally while
        # public K1 code IDs remain the bank's stable strings.
        self.id_embedding = nn.Embedding(self.num_queries, d)
        self.type_embedding = nn.Embedding(2, d)
        self.policy_id_embedding = nn.Embedding(self.num_queries, w)
        self.policy_type_embedding = nn.Embedding(2, w)
        self.value_embedding = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.policy_value_embedding = nn.Sequential(nn.Linear(1, w), nn.GELU(), nn.Linear(w, w))
        self.decoder_descriptor = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.coordinate_embedding = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.cross_attention = nn.MultiheadAttention(d, config.num_heads, dropout=0.0, batch_first=True)
        self.decoder_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, config.num_classes))
        self.null_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.null_token, std=0.02)
        self.policy_head = nn.Sequential(nn.Linear(4 * w, w), nn.GELU(), nn.Linear(w, 1))
        self.policy_stop = nn.Sequential(nn.Linear(3 * w, w), nn.GELU(), nn.Linear(w, 1))

    # -------------------------------------------------------------- public IDs
    def _indices(self, ids: Sequence[Any] | Tensor, *, device: torch.device) -> Tensor:
        if isinstance(ids, Tensor):
            if ids.ndim != 1:
                raise ValueError("query IDs must be one-dimensional per sample")
            if ids.dtype.is_floating_point or ids.dtype.is_complex:
                if not torch.isfinite(ids).all() or not torch.equal(ids, ids.round()):
                    raise ValueError("numeric query IDs must be finite integers")
            raw = ids.detach().cpu().tolist()
        else:
            raw = list(ids)
        result = []
        for query_id in raw:
            if query_id in self._id_to_index:
                result.append(self._id_to_index[query_id])
            elif isinstance(query_id, int) and 0 <= query_id < self.num_queries:
                result.append(int(query_id))
            else:
                raise ValueError(f"unknown query ID: {query_id!r}")
        result = torch.tensor(result, dtype=torch.long, device=device)
        if result.numel() != result.unique().numel():
            raise ValueError("code query IDs must not repeat")
        return result

    def _batch_indices(self, code_ids: Any, batch: int, *, device: torch.device) -> list[Tensor]:
        if code_ids is None:
            return [torch.empty(0, dtype=torch.long, device=device) for _ in range(batch)]
        if isinstance(code_ids, Tensor) and code_ids.ndim == 2:
            if code_ids.shape[0] != batch:
                raise ValueError("code ID batch does not match image batch")
            rows = [code_ids[b] for b in range(batch)]
        elif isinstance(code_ids, Sequence) and not isinstance(code_ids, (str, bytes)) and len(code_ids) == batch and any(isinstance(row, (Tensor, list, tuple)) for row in code_ids):
            rows = list(code_ids)
        else:
            rows = [code_ids] * batch
        result = []
        for row in rows:
            if isinstance(row, Tensor):
                row = row[row >= 0] if row.ndim == 1 and row.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) else row
            result.append(self._indices(row, device=device))
        return result

    @staticmethod
    def _batch_values(code_values: Any, rows: list[Tensor], *, device: torch.device, dtype: torch.dtype) -> list[Tensor]:
        if code_values is None:
            return [torch.empty(row.numel(), device=device, dtype=dtype) for row in rows]
        values = torch.as_tensor(code_values, device=device, dtype=dtype)
        if values.ndim == 1:
            values = values.unsqueeze(0).expand(len(rows), -1)
        if values.ndim != 2 or values.shape[0] != len(rows):
            raise ValueError("code_values must have shape [K] or [B,K]")
        out = []
        for b, row in enumerate(rows):
            if values.shape[1] != row.numel():
                raise ValueError("code_values length must match code_ids per sample")
            out.append(values[b])
        return out

    def _bounds(self, indices: Tensor, *, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        kinds = self.query_kinds.to(device=indices.device)[indices]
        lower = torch.where(kinds == 1, torch.full_like(kinds, -1, dtype=dtype), torch.zeros_like(kinds, dtype=dtype))
        return lower, torch.ones_like(lower)

    # --------------------------------------------------------------- encoder
    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.config.in_channels:
            raise ValueError("expected image tensor with shape [B,3,H,W]")
        if not torch.isfinite(x).all():
            raise ValueError("image contains NaN or Inf")
        return self.encoder(x)

    features = encode

    def predict_values(self, features: Tensor, *, quantize: bool | None = None) -> Tensor:
        """Return raw bounded student values, before codec quantization.

        ``quantize`` is accepted only as a source-compatibility shim for an
        older K1 caller that passed ``quantize=False``.  Quantization is always
        performed by ``MeasurementCodec``; ``quantize=True`` is rejected so a
        second model-side codec cannot silently enter the training path.
        """
        if quantize is True:
            raise ValueError("student values are raw; use MeasurementCodec.quantize_values")
        if features.ndim != 4 or features.shape[1] != self.config.width:
            raise ValueError("expected encoder features [B,width,H,W]")
        pooled = features.mean(dim=(-2, -1))
        query = self.query_value(self.query_descriptors.to(device=features.device, dtype=features.dtype))
        joint = torch.cat((pooled[:, None, :].expand(-1, self.num_queries, -1),
                           query[None].expand(features.shape[0], -1, -1)), dim=-1)
        raw = self.value_head(joint).squeeze(-1)
        kinds = self.query_kinds.to(device=features.device)
        return torch.where(kinds[None] == 1, raw.tanh(), raw.sigmoid())

    # --------------------------------------------------------------- policy
    def policy_logits(self, features: Tensor, code_ids: Any, code_values: Any,
                      remaining_bits: int | Tensor | Sequence[float] | None = None) -> Tensor:
        """Return bank-order query logits followed by an always-valid STOP.

        K1 masks candidates that exceed the codec budget before its restricted
        policy loss.  This method masks used IDs itself and accepts the
        remaining budget as a normalized state feature; if it is non-positive,
        all additions are masked while STOP remains valid.
        """
        if features.ndim != 4 or features.shape[1] != self.config.width:
            raise ValueError("expected encoder features [B,width,H,W]")
        batch = features.shape[0]
        rows = self._batch_indices(code_ids, batch, device=features.device)
        vals = self._batch_values(code_values, rows, device=features.device, dtype=features.dtype)
        pooled = features.mean(dim=(-2, -1))
        descriptors = self.query_value(self.query_descriptors.to(device=features.device, dtype=features.dtype))
        kinds = self.query_kinds.to(device=features.device)
        qbase = self.policy_id_embedding.weight + self.policy_type_embedding(kinds) + descriptors
        summary = torch.zeros(batch, self.config.width, device=features.device, dtype=features.dtype)
        used = torch.zeros(batch, self.num_queries, device=features.device, dtype=torch.bool)
        for b, row in enumerate(rows):
            if row.numel():
                used[b, row] = True
                token = self.policy_id_embedding(row) + self.policy_type_embedding(kinds[row])
                token = token + self.policy_value_embedding(vals[b][:, None])
                summary[b] = token.mean(0)
        if remaining_bits is None:
            remaining = torch.ones(batch, device=features.device, dtype=features.dtype)
        else:
            remaining = torch.as_tensor(remaining_bits, device=features.device, dtype=features.dtype).flatten()
            if remaining.numel() == 1:
                remaining = remaining.expand(batch)
            if remaining.numel() != batch:
                raise ValueError("remaining_bits must be scalar or [B]")
            # The absolute bit scale is deliberately not invented here; K1's
            # codec feasibility check is authoritative.  The bounded state
            # feature avoids unbounded logits for large packet budgets.
            remaining = remaining.clamp(-1e6, 1e6) / 1024.0
        common = torch.cat((pooled, summary, remaining[:, None].expand(-1, self.config.width)), dim=-1)
        query_input = torch.cat((common[:, None].expand(-1, self.num_queries, -1), qbase[None].expand(batch, -1, -1)), dim=-1)
        logits = self.policy_head(query_input).squeeze(-1)
        stop = self.policy_stop(common).squeeze(-1,).unsqueeze(-1)
        logits = logits.masked_fill(used, float("-inf"))
        if remaining_bits is not None:
            logits = logits.masked_fill(remaining <= 0, float("-inf"))
        return torch.cat((logits, stop), dim=-1)

    @property
    def stop_index(self) -> int:
        return self.num_queries

    # --------------------------------------------------------------- decoder
    def _tokens(self, indices: Tensor, values: Tensor, *, dtype: torch.dtype) -> Tensor:
        if indices.numel() == 0:
            return self.null_token.to(device=values.device, dtype=dtype)
        if values.ndim != 1 or values.numel() != indices.numel():
            raise ValueError("decoder code values must be one value per selected ID")
        if not torch.isfinite(values).all():
            raise ValueError("decoder code values must be finite")
        lower, upper = self._bounds(indices, dtype=dtype)
        if bool((values < lower).any()) or bool((values > upper).any()):
            raise ValueError("decoder requires bounded, codec-dequantized values")
        # Codes are sets in the protocol and MeasurementCodec serializes IDs in
        # canonical ascending order.  Canonicalize here too so an unsorted K1
        # rollout and its packet decode have identical attention token order.
        order = torch.argsort(indices)
        indices, values = indices[order], values[order]
        kinds = self.query_kinds.to(device=indices.device)[indices]
        desc = self.query_descriptors.to(device=values.device, dtype=dtype)[indices]
        return (self.id_embedding(indices) + self.type_embedding(kinds) +
                self.decoder_descriptor(desc) + self.value_embedding(values[:, None])).unsqueeze(0)

    @staticmethod
    def _geometry(geometry: Mapping[str, Any]) -> tuple[int, int, float, float]:
        if not isinstance(geometry, Mapping) or set(("shape", "spacing_yx")) - set(geometry):
            raise ValueError("geometry must provide shape=(H,W) and spacing_yx=(dy,dx)")
        shape, spacing = geometry["shape"], geometry["spacing_yx"]
        if isinstance(shape, Tensor):
            shape = shape.detach().cpu().tolist()
        if isinstance(spacing, Tensor):
            spacing = spacing.detach().cpu().tolist()
        if len(shape) != 2 or any(int(v) != v or int(v) <= 0 for v in shape):
            raise ValueError("geometry shape must be positive [H,W]")
        if len(spacing) != 2:
            raise ValueError("geometry spacing_yx must contain (dy,dx)")
        h, w = int(shape[0]), int(shape[1])
        sy, sx = float(spacing[0]), float(spacing[1])
        if not torch.isfinite(torch.tensor([sy, sx])).all() or sy <= 0 or sx <= 0:
            raise ValueError("geometry spacing_yx must be finite and positive")
        # MeasurementCodec stores spacing in the packet header as IEEE float32
        # (``ff``).  Canonicalising before coordinate embedding makes a rollout
        # decoded from its serialized packet bitwise-stable with the in-memory
        # rollout, instead of leaving a 1e-5--1e-4 spacing round-trip drift.
        packed_spacing = torch.tensor((sy, sx), dtype=torch.float32)
        sy, sx = float(packed_spacing[0]), float(packed_spacing[1])
        return h, w, sy, sx

    def decode(self, code_ids: Any, code_values: Any, geometry: Mapping[str, Any],
               chunk_size: int | None = None) -> Tensor:
        """Decode only IDs, quantized values and public geometry to logits."""
        device = next(self.parameters()).device
        if isinstance(code_values, Tensor):
            device = code_values.device
        if isinstance(code_ids, Tensor) and code_ids.ndim == 2:
            batch = code_ids.shape[0]
        elif isinstance(code_ids, Sequence) and not isinstance(code_ids, (str, bytes)) and code_ids and isinstance(code_ids[0], (list, tuple, Tensor)):
            batch = len(code_ids)
        else:
            batch = 1
        rows = self._batch_indices(code_ids, batch, device=device)
        vals = self._batch_values(code_values, rows, device=device, dtype=next(self.parameters()).dtype)
        h, w, sy, sx = self._geometry(geometry)
        chunk = int(chunk_size or self.config.chunk_size)
        if chunk <= 0:
            raise ValueError("chunk_size must be positive")
        ys = torch.linspace(-1, 1, h, device=device, dtype=vals[0].dtype if vals else torch.float32)
        xs = torch.linspace(-1, 1, w, device=device, dtype=ys.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((xx.reshape(-1), yy.reshape(-1),
                              torch.full((h * w,), sy, device=device, dtype=ys.dtype).log1p(),
                              torch.full((h * w,), sx, device=device, dtype=ys.dtype).log1p()), dim=-1)
        outputs = []
        for row, value in zip(rows, vals):
            token = self._tokens(row, value, dtype=ys.dtype)
            pieces = []
            for start in range(0, h * w, chunk):
                q = self.coordinate_embedding(coords[start:start + chunk]).unsqueeze(0)
                attended, _ = self.cross_attention(q, token, token, need_weights=False)
                pieces.append(self.decoder_head(q + attended).squeeze(0))
            flat = torch.cat(pieces, dim=0)
            outputs.append(flat.transpose(0, 1).reshape(self.config.num_classes, h, w))
        return torch.stack(outputs, dim=0)

    # -------------------------------------------------------------- export aid
    def export_student(self) -> dict[str, Any]:
        """Teacher-free in-memory export; filesystem/checkpoint policy is in io.py."""
        return {"format": "candidate_a_student_v1", "config": asdict(self.config),
                "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}}

    @classmethod
    def from_export(cls, payload: Mapping[str, Any], bank: Any, *, map_location: str | torch.device = "cpu") -> "CandidateAStudent":
        if payload.get("format") not in (None, "candidate_a_student_v1"):
            raise ValueError("unsupported student export format")
        model = cls(bank, StudentConfig(**payload["config"]))
        state = {k: v.to(map_location) if isinstance(v, Tensor) else v for k, v in payload["state_dict"].items()}
        model.load_state_dict(state, strict=True)
        return model


# Existing candidate-A integration names this class CandidateStudent.
CandidateStudent = CandidateAStudent
StudentModel = CandidateAStudent
