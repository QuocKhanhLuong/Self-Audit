"""Finite-bit serialization for candidate A measurement codes.

The model-side representation is a list of unique bank IDs and *quantized*
student values.  This module keeps the differentiable straight-through value
separate from deployment serialization: training may use
``x + (round(x) - x).detach()``; a serialized packet contains only integer
levels.  The packet contains a bank fingerprint and public per-image geometry,
so an ID from a different protocol or an ambiguous spacing cannot be decoded.

Bit accounting is based on the actual padded packet bytes (header, fingerprint,
shape, spacing, count, value bit width, IDs, and value levels).  Thus an
incremental cost includes the fixed code/R0 overhead rather than treating a
floating-point vector as a code with an unbounded or imaginary budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Sequence

import torch

from .bank import QueryBank


_MAGIC = b"CA01"
_VERSION = 1
_HEADER = struct.Struct(">4sB32sIIffHBB")
_RECORD_ID = struct.Struct(">I")
_CHECKSUM_SIZE = 8


def _finite_shape(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) != 2 or any(isinstance(v, bool) or int(v) != v or int(v) <= 0 for v in shape):
        raise ValueError("shape must be two positive integer in-plane dimensions (H, W)")
    return int(shape[0]), int(shape[1])


def _finite_spacing(spacing: Sequence[float]) -> tuple[float, float]:
    if len(spacing) != 2:
        raise ValueError("spacing must contain exactly (dy, dx)")
    out = (float(spacing[0]), float(spacing[1]))
    if any(not math.isfinite(v) or v <= 0.0 for v in out):
        raise ValueError("spacing must be finite and positive (dy, dx)")
    return out


@dataclass(frozen=True)
class QuantizedValues:
    """Quantized student values and corresponding bank IDs.

    ``values`` is the dequantized value tensor used by the decoder.  With
    ``ste=True`` it retains a straight-through gradient to the input prediction;
    ``levels`` is always an integer tensor suitable for packet serialization.
    """

    ids: tuple[int, ...]
    levels: torch.Tensor
    values: torch.Tensor

@dataclass(frozen=True)
class CodePacket:
    """Decoded, validated integer packet metadata."""

    bank_fingerprint: str
    shape: tuple[int, int]
    spacing: tuple[float, float]
    ids: tuple[int, ...]
    levels: tuple[int, ...]
    value_bits: int

    @property
    def count(self) -> int:
        return len(self.ids)


class MeasurementCodec:
    """Uniform finite-bit codec tied to exactly one :class:`QueryBank`.

    Parameters
    ----------
    bank:
        Immutable finite query protocol.  Its SHA-256 fingerprint is embedded
        in every packet.
    value_bits:
        Number of bits per value level, used for both REGION and BOUNDARY.
    base_ids:
        Fixed R0 IDs.  They are included in every serialized packet and in
        ``cost(0)``.  The caller must provide their *student-predicted*
        quantized levels to ``serialize``; the codec never invents an R0 value.
    """

    def __init__(self, bank: QueryBank, value_bits: int = 8, *, base_ids: Sequence[int] = ()) -> None:
        if not isinstance(bank, QueryBank):
            raise TypeError("bank must be a QueryBank")
        if isinstance(value_bits, bool) or int(value_bits) != value_bits or not 1 <= int(value_bits) <= 24:
            raise ValueError("value_bits must be an integer in [1, 24]")
        self.bank = bank
        self.value_bits = int(value_bits)
        self.level_count = 1 << self.value_bits
        self.max_level = self.level_count - 1
        self.base_ids = self._validate_ids(base_ids, unique=True)

    @property
    def bank_fingerprint(self) -> str:
        return self.bank.fingerprint

    def _validate_ids(self, ids: Sequence[int] | torch.Tensor, *, unique: bool = True) -> tuple[int, ...]:
        if isinstance(ids, torch.Tensor):
            if ids.ndim != 1:
                raise ValueError("query IDs must be a one-dimensional sequence")
            if ids.dtype.is_floating_point or ids.dtype.is_complex:
                if not torch.isfinite(ids).all() or not torch.equal(ids, ids.round()):
                    raise ValueError("query IDs must be finite integers")
            ids = ids.detach().cpu().tolist()
        try:
            raw_values = tuple(ids)
            if any(isinstance(v, bool) for v in raw_values):
                raise ValueError("query IDs must be integers")
            for value in raw_values:
                if not math.isfinite(float(value)) or int(value) != value:
                    raise ValueError("query IDs must be finite integers")
            values = tuple(int(v) for v in raw_values)
        except (TypeError, ValueError) as exc:
            raise ValueError("query IDs must be integers") from exc
        if any(v < 0 or v >= len(self.bank) for v in values):
            raise ValueError("query ID is outside the finite bank")
        if unique and len(set(values)) != len(values):
            raise ValueError("duplicate query IDs are not permitted in a code")
        return values

    def _bounds(self, ids: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        lower, upper = [], []
        for idx in ids:
            q = self.bank.query(idx)
            if q.kind == "REGION":
                lower.append(0.0)
                upper.append(1.0)
            else:
                lower.append(-1.0)
                upper.append(1.0)
        return torch.tensor(lower, device=device, dtype=dtype), torch.tensor(upper, device=device, dtype=dtype)

    def quantize_values(
        self,
        values: torch.Tensor | Sequence[float],
        ids: Sequence[int] | torch.Tensor | None = None,
        *,
        ste: bool = False,
    ) -> QuantizedValues:
        """Quantize values for a common ID list, retaining optional STE gradients.

        ``values`` may be ``[N]`` or ``[B, N]``.  If ``ids`` is omitted, N must
        equal the whole bank size and IDs are ``range(len(bank))``.  For a
        rollout with different IDs per sample call this method once per sample;
        silently broadcasting one sample's query IDs to another would make the
        code ambiguous.
        """

        raw = torch.as_tensor(values) if not isinstance(values, torch.Tensor) else values
        if raw.ndim not in (1, 2):
            raise ValueError("values must have shape [N] or [B, N]")
        if not raw.dtype.is_floating_point:
            raw = raw.float()
        if not torch.isfinite(raw).all():
            raise ValueError("cannot quantize non-finite values")
        n = raw.shape[-1]
        query_ids = tuple(range(len(self.bank))) if ids is None else self._validate_ids(ids)
        if len(query_ids) != n:
            raise ValueError("number of values must equal number of query IDs")
        lower, upper = self._bounds(query_ids, device=raw.device, dtype=raw.dtype)
        view = (1, -1) if raw.ndim == 2 else (-1,)
        lo, hi = lower.view(*view), upper.view(*view)
        normalized = ((raw - lo) / (hi - lo)).clamp(0.0, 1.0)
        levels = torch.round(normalized * self.max_level).to(torch.int64)
        dequantized = lo + levels.to(raw.dtype) / self.max_level * (hi - lo)
        if ste:
            # Forward value is exactly the finite quantized level while the
            # backward path is the student prediction.  No teacher/GT value is
            # introduced here.
            dequantized = raw + (dequantized - raw).detach()
        return QuantizedValues(query_ids, levels, dequantized)

    def dequantize_levels(
        self,
        ids: Sequence[int] | torch.Tensor,
        levels: Sequence[int] | torch.Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        query_ids = self._validate_ids(ids)
        lev = torch.as_tensor(levels, device=device, dtype=torch.int64)
        if lev.ndim != 1 or lev.numel() != len(query_ids):
            raise ValueError("levels must be one-dimensional and match IDs")
        if (lev < 0).any() or (lev > self.max_level).any():
            raise ValueError("quantization level outside configured value_bits")
        lower, upper = self._bounds(query_ids, device=lev.device, dtype=dtype)
        return lower + lev.to(dtype) / self.max_level * (upper - lower)

    def _with_base(self, ids: Sequence[int], levels: Sequence[int] | None, *, require_base_values: bool = True) -> tuple[tuple[int, ...], tuple[int, ...]]:
        query_ids = self._validate_ids(ids)
        if levels is None:
            if query_ids:
                # ``None`` is used only by cost(), where value contents do not
                # affect packet size.  It must never silently become a model
                # code: serialize() requires explicit student levels.
                lev = [0] * len(query_ids)
            else:
                lev = []
        else:
            lev = []
            for value in levels:
                if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != value:
                    raise ValueError("serialized levels must be finite integers")
                lev.append(int(value))
            if len(lev) != len(query_ids):
                raise ValueError("levels must match IDs")
            if any(v < 0 or v > self.max_level for v in lev):
                raise ValueError("quantization level outside configured value_bits")
        pairs = dict(zip(query_ids, lev))
        for idx in self.base_ids:
            if idx not in pairs:
                if require_base_values:
                    raise ValueError("serialized code must contain student-quantized values for every R0 query")
                pairs[idx] = 0
        ordered = tuple(sorted(pairs))
        return ordered, tuple(pairs[idx] for idx in ordered)

    def _packet_bytes(self, ids: tuple[int, ...], levels: tuple[int, ...], shape, spacing) -> bytes:
        shape = _finite_shape(shape)
        spacing = _finite_spacing(spacing)
        if len(ids) != len(levels):
            raise ValueError("IDs and levels must have the same count")
        if len(ids) > 65535:
            raise ValueError("packet count exceeds uint16 protocol field")
        fp = bytes.fromhex(self.bank_fingerprint)
        header = _HEADER.pack(_MAGIC, _VERSION, fp, shape[0], shape[1], spacing[0], spacing[1], len(ids), self.value_bits, 0)
        level_bytes = (self.value_bits + 7) // 8
        body = bytearray()
        for idx, level in zip(ids, levels):
            body.extend(_RECORD_ID.pack(idx))
            body.extend(int(level).to_bytes(level_bytes, "big"))
        unsigned = header + bytes(body)
        # The bank fingerprint prevents cross-protocol decoding; this short
        # integrity trailer additionally makes accidental/tampered payload
        # changes fail closed rather than silently changing a valid level.
        return unsigned + hashlib.sha256(unsigned).digest()[:_CHECKSUM_SIZE]

    def serialize(
        self,
        ids: Sequence[int] | torch.Tensor,
        levels: Sequence[int] | torch.Tensor | None,
        shape: Sequence[int],
        spacing: Sequence[float],
        *,
        budget_bits: int | None = None,
    ) -> bytes:
        """Serialize an integer code and reject packets exceeding budget."""

        if levels is None:
            raise ValueError("serialize requires integer student levels, including fixed R0 queries")
        level_values = torch.as_tensor(levels)
        if level_values.ndim != 1:
            raise ValueError("levels must be a one-dimensional sequence")
        query_ids, lev = self._with_base(ids, level_values.detach().cpu().tolist())
        packet = self._packet_bytes(query_ids, lev, shape, spacing)
        if budget_bits is not None:
            if isinstance(budget_bits, bool) or int(budget_bits) != budget_bits or int(budget_bits) < 0:
                raise ValueError("budget_bits must be a non-negative integer")
            if len(packet) * 8 > int(budget_bits):
                raise ValueError(f"code packet needs {len(packet) * 8} bits, budget is {int(budget_bits)}")
        return packet

    def deserialize(self, payload: bytes | bytearray | memoryview, *, budget_bits: int | None = None) -> CodePacket:
        """Validate and parse a packet, including bank fingerprint and geometry."""

        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        payload = bytes(payload)
        if len(payload) < _HEADER.size:
            raise ValueError("truncated candidate-A code header")
        magic, version, fingerprint, h, w, dy, dx, count, bits, reserved = _HEADER.unpack_from(payload)
        if magic != _MAGIC or version != _VERSION or reserved != 0:
            raise ValueError("unsupported or corrupted candidate-A code header")
        if fingerprint.hex() != self.bank_fingerprint:
            raise ValueError("bank fingerprint mismatch")
        if bits != self.value_bits:
            raise ValueError("value_bits mismatch")
        shape = _finite_shape((h, w))
        spacing = _finite_spacing((dy, dx))
        level_bytes = (bits + 7) // 8
        record_size = _RECORD_ID.size + level_bytes
        expected = _HEADER.size + int(count) * record_size + _CHECKSUM_SIZE
        if len(payload) != expected:
            raise ValueError("truncated or trailing candidate-A code bytes")
        unsigned, checksum = payload[:-_CHECKSUM_SIZE], payload[-_CHECKSUM_SIZE:]
        if hashlib.sha256(unsigned).digest()[:_CHECKSUM_SIZE] != checksum:
            raise ValueError("candidate-A code checksum mismatch")
        ids, levels = [], []
        offset = _HEADER.size
        for _ in range(count):
            idx = _RECORD_ID.unpack_from(payload, offset)[0]
            offset += _RECORD_ID.size
            level = int.from_bytes(payload[offset : offset + level_bytes], "big")
            offset += level_bytes
            ids.append(idx)
            levels.append(level)
        query_ids = self._validate_ids(ids)
        if any(v > self.max_level for v in levels):
            raise ValueError("quantization level outside configured value_bits")
        if not set(self.base_ids).issubset(query_ids):
            raise ValueError("packet is missing fixed R0 query IDs")
        if budget_bits is not None and len(payload) * 8 > int(budget_bits):
            raise ValueError("serialized packet exceeds supplied bit budget")
        return CodePacket(self.bank_fingerprint, shape, spacing, query_ids, tuple(levels), bits)

    def dequantize(self, packet: CodePacket | bytes, *, device=None, dtype=torch.float32) -> torch.Tensor:
        if isinstance(packet, (bytes, bytearray, memoryview)):
            packet = self.deserialize(packet)
        if not isinstance(packet, CodePacket) or packet.bank_fingerprint != self.bank_fingerprint:
            raise ValueError("packet does not belong to this codec")
        return self.dequantize_levels(packet.ids, packet.levels, device=device, dtype=dtype)

    def packet_bits(self, payload: bytes | bytearray | memoryview) -> int:
        return len(payload) * 8

    def _ids_for_count(self, count: int) -> tuple[int, ...]:
        if isinstance(count, bool) or int(count) != count or int(count) < 0:
            raise ValueError("query count must be a non-negative integer")
        selected = list(self.base_ids)
        for idx in range(len(self.bank)):
            if idx not in selected and len(selected) - len(self.base_ids) < int(count):
                selected.append(idx)
        if len(selected) - len(self.base_ids) != int(count):
            raise ValueError("query count exceeds available non-R0 bank queries")
        return tuple(selected)

    def cost(
        self,
        ids_or_count: int | Sequence[int] | torch.Tensor,
        *,
        shape: Sequence[int] = (1, 1),
        spacing: Sequence[float] = (1.0, 1.0),
    ) -> int:
        """Return actual padded packet size in bits, including fixed R0 cost."""

        if isinstance(ids_or_count, (int, float)) and not isinstance(ids_or_count, bool):
            ids = self._ids_for_count(int(ids_or_count))
        else:
            ids = self._validate_ids(ids_or_count)
        # Packet size is value-independent.  For accounting only, materialize
        # zero levels for every actual query (including R0); this is not exposed
        # as a deployment packet and therefore cannot replace predicted R0
        # values in serialize().
        packet_ids, _ = self._with_base(ids, None, require_base_values=False)
        payload = self._packet_bytes(packet_ids, tuple(0 for _ in packet_ids), shape, spacing)
        return len(payload) * 8

    def incremental_cost(
        self,
        existing_ids: Sequence[int] | torch.Tensor,
        new_id: int,
        *,
        shape: Sequence[int] = (1, 1),
        spacing: Sequence[float] = (1.0, 1.0),
    ) -> int:
        """Actual byte-padded cost increase.

        The explicit ``incremental_cost(existing_ids, new_id)`` form is used
        for K1 trials so the exact selected IDs and fixed R0 packet are
        accounted for.
        """
        existing = self._validate_ids(existing_ids)
        if isinstance(new_id, bool) or int(new_id) != new_id:
            raise ValueError("new_id must be an integer")
        new_id = int(new_id)
        if new_id in existing or new_id in self.base_ids:
            return 0
        return self.cost(existing + (new_id,), shape=shape, spacing=spacing) - self.cost(
            existing, shape=shape, spacing=spacing
        )

    def fits(
        self,
        ids: Sequence[int] | torch.Tensor,
        budget_bits: int,
        *,
        shape: Sequence[int] = (1, 1),
        spacing: Sequence[float] = (1.0, 1.0),
    ) -> bool:
        if isinstance(budget_bits, bool) or int(budget_bits) != budget_bits or int(budget_bits) < 0:
            raise ValueError("budget_bits must be a non-negative integer")
        return self.cost(ids, shape=shape, spacing=spacing) <= int(budget_bits)
