"""Finite-bit native-volume packet codec for Candidate-A 3-D.

The packet stores the bank fingerprint, native ``D,H,W`` shape, float32
spacing, float32 affine, fixed context marker, query IDs and integer value
levels.  Cost is computed from the exact serialized bytes, including the
fixed R0/header and checksum; K1 therefore never treats an unconstrained
floating-point vector as a finite code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Sequence

import torch

from .bank import QueryBank3D
from .geometry import canonical_float32_geometry


_MAGIC = b"CA3D"
_VERSION = 1
_CONTEXT = b"full_volume"
# magic/version/fingerprint/D/H/W/spacing(3)/affine(16)/count/bits/reserved/context
_HEADER = struct.Struct(">4sB32sIIIfff16fHBB11s")
_RECORD_ID = struct.Struct(">I")
_CHECKSUM_SIZE = 8


def _shape(value: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("shape must contain positive integer D,H,W")
    result = tuple(value)
    if any(isinstance(v, bool) or int(v) != v or int(v) <= 0 for v in result):
        raise ValueError("shape must contain positive integer D,H,W")
    return tuple(int(v) for v in result)


def _spacing(value: Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError("spacing must contain positive (dz,dy,dx)")
    result = tuple(float(v) for v in value)
    if any(not math.isfinite(v) or v <= 0 for v in result):
        raise ValueError("spacing must be finite and positive")
    return result


def _identity(spacing: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    dz, dy, dx = _spacing(spacing)
    return ((dz, 0.0, 0.0, 0.0), (0.0, dy, 0.0, 0.0), (0.0, 0.0, dx, 0.0),
            (0.0, 0.0, 0.0, 1.0))


def _geometry(shape, spacing, affine_zyx=None) -> tuple[tuple[int, int, int], tuple[float, float, float], tuple[tuple[float, ...], ...]]:
    shape = _shape(shape)
    spacing = _spacing(spacing)
    if affine_zyx is None:
        affine_zyx = _identity(spacing)
    validated = canonical_float32_geometry({
        "shape": shape, "spacing_zyx": spacing, "affine_zyx": affine_zyx, "context": "full_volume",
    })
    affine = tuple(tuple(float(v) for v in row) for row in validated["affine_zyx"])
    # ``canonical_float32_geometry`` is the packet's authority.  Re-read the
    # packed values so cost/serialize/decode all see identical bytes.
    packed_spacing = tuple(float(v) for v in validated["spacing_zyx"])
    return shape, packed_spacing, affine


@dataclass(frozen=True)
class QuantizedValues3D:
    ids: tuple[int, ...]
    levels: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class CodePacket3D:
    bank_fingerprint: str
    shape: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]
    affine_zyx: tuple[tuple[float, ...], ...]
    ids: tuple[int, ...]
    levels: tuple[int, ...]
    value_bits: int

    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Canonical short alias used by the A-3D runtime contract."""
        return self.spacing_zyx


class MeasurementCodec3D:
    """Strict finite-bit codec bound to one :class:`QueryBank3D`."""

    def __init__(self, bank: QueryBank3D, value_bits: int = 8, *, base_ids: Sequence[int] = ()):
        if not isinstance(bank, QueryBank3D):
            raise TypeError("bank must be QueryBank3D")
        if isinstance(value_bits, bool) or int(value_bits) != value_bits or not 1 <= int(value_bits) <= 24:
            raise ValueError("value_bits must be an integer in [1,24]")
        self.bank = bank
        self.value_bits = int(value_bits)
        self.level_count = 1 << self.value_bits
        self.max_level = self.level_count - 1
        self.base_ids = self._validate_ids(base_ids)

    @property
    def bank_fingerprint(self) -> str:
        return self.bank.fingerprint

    def _validate_ids(self, ids: Sequence[int] | torch.Tensor, *, unique: bool = True) -> tuple[int, ...]:
        if isinstance(ids, torch.Tensor):
            if ids.ndim != 1:
                raise ValueError("query IDs must be one-dimensional")
            ids = ids.detach().cpu().tolist()
        try:
            raw_values = tuple(ids)
            if any(isinstance(v, bool) or not math.isfinite(float(v)) or int(v) != v for v in raw_values):
                raise ValueError("query IDs must be finite integers")
            values = tuple(int(v) for v in raw_values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("query IDs must be integers") from exc
        if any(not 0 <= value < len(self.bank) for value in values):
            raise ValueError("query ID is outside the finite bank")
        if unique and len(set(values)) != len(values):
            raise ValueError("duplicate query IDs are not permitted")
        return values

    def _bounds(self, ids: tuple[int, ...], *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        lower, upper = [], []
        for idx in ids:
            if self.bank.query(idx).kind == "REGION":
                lower.append(0.0); upper.append(1.0)
            else:
                lower.append(-1.0); upper.append(1.0)
        return torch.tensor(lower, device=device, dtype=dtype), torch.tensor(upper, device=device, dtype=dtype)

    def quantize_values(self, values: torch.Tensor | Sequence[float], ids=None, *, ste: bool = False) -> QuantizedValues3D:
        raw = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
        if raw.ndim not in (1, 2):
            raise ValueError("values must have shape [N] or [B,N]")
        if not raw.dtype.is_floating_point:
            raw = raw.float()
        if not torch.isfinite(raw).all():
            raise ValueError("cannot quantize non-finite values")
        query_ids = tuple(range(len(self.bank))) if ids is None else self._validate_ids(ids)
        if len(query_ids) != raw.shape[-1]:
            raise ValueError("number of values must equal number of query IDs")
        lower, upper = self._bounds(query_ids, device=raw.device, dtype=raw.dtype)
        view = (1, -1) if raw.ndim == 2 else (-1,)
        normalized = ((raw - lower.view(*view)) / (upper - lower).view(*view)).clamp(0, 1)
        levels = torch.round(normalized * self.max_level).to(torch.int64)
        dequantized = lower.view(*view) + levels.to(raw.dtype) / self.max_level * (upper - lower).view(*view)
        if ste:
            dequantized = raw + (dequantized - raw).detach()
        return QuantizedValues3D(query_ids, levels, dequantized)

    def dequantize_levels(self, ids, levels, *, device=None, dtype=torch.float32) -> torch.Tensor:
        query_ids = self._validate_ids(ids)
        lev = torch.as_tensor(levels, device=device, dtype=torch.int64)
        if lev.ndim != 1 or lev.numel() != len(query_ids):
            raise ValueError("levels must be one-dimensional and match IDs")
        if bool((lev < 0).any()) or bool((lev > self.max_level).any()):
            raise ValueError("quantization level outside configured value_bits")
        lower, upper = self._bounds(query_ids, device=lev.device, dtype=dtype)
        return lower + lev.to(dtype) / self.max_level * (upper - lower)

    def _with_base(self, ids, levels=None, *, require_base_values=True):
        query_ids = self._validate_ids(ids)
        if levels is None:
            values = [0] * len(query_ids)
        else:
            values = []
            for value in levels:
                if isinstance(value, bool) or int(value) != value or not math.isfinite(float(value)):
                    raise ValueError("serialized levels must be finite integers")
                values.append(int(value))
            if len(values) != len(query_ids) or any(v < 0 or v > self.max_level for v in values):
                raise ValueError("serialized levels must match IDs and value_bits")
        pairs = dict(zip(query_ids, values))
        for idx in self.base_ids:
            if idx not in pairs:
                if require_base_values:
                    raise ValueError("serialized code is missing an R0 query")
                pairs[idx] = 0
        ordered = tuple(sorted(pairs))
        return ordered, tuple(pairs[idx] for idx in ordered)

    def _packet_bytes(self, ids, levels, shape, spacing, affine_zyx):
        shape, spacing, affine = _geometry(shape, spacing, affine_zyx)
        if len(ids) != len(levels) or len(ids) > 65535:
            raise ValueError("invalid packet IDs/levels")
        flat_affine = tuple(float(v) for row in affine for v in row)
        header = _HEADER.pack(_MAGIC, _VERSION, bytes.fromhex(self.bank_fingerprint), *shape, *spacing,
                              *flat_affine, len(ids), self.value_bits, 0, _CONTEXT)
        level_bytes = (self.value_bits + 7) // 8
        body = bytearray()
        for idx, level in zip(ids, levels):
            body.extend(_RECORD_ID.pack(int(idx)))
            body.extend(int(level).to_bytes(level_bytes, "big"))
        unsigned = header + bytes(body)
        return unsigned + hashlib.sha256(unsigned).digest()[:_CHECKSUM_SIZE]

    def serialize(self, ids, levels, shape, spacing, *, affine_zyx=None, budget_bits=None) -> bytes:
        if levels is None:
            raise ValueError("serialize requires quantized student levels")
        level_values = torch.as_tensor(levels)
        if level_values.ndim != 1:
            raise ValueError("levels must be one-dimensional")
        query_ids, packed_levels = self._with_base(ids, level_values.detach().cpu().tolist())
        packet = self._packet_bytes(query_ids, packed_levels, shape, spacing, affine_zyx)
        if budget_bits is not None:
            if isinstance(budget_bits, bool) or int(budget_bits) != budget_bits or int(budget_bits) < 0:
                raise ValueError("budget_bits must be a non-negative integer")
            if len(packet) * 8 > int(budget_bits):
                raise ValueError("serialized packet exceeds bit budget")
        return packet

    def deserialize(self, payload, *, budget_bits=None) -> CodePacket3D:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        payload = bytes(payload)
        if len(payload) < _HEADER.size + _CHECKSUM_SIZE:
            raise ValueError("truncated A-3D packet")
        unpacked = _HEADER.unpack_from(payload)
        magic, version, fingerprint = unpacked[:3]
        if magic != _MAGIC or version != _VERSION or fingerprint.hex() != self.bank_fingerprint:
            raise ValueError("unsupported/cross-bank A-3D packet")
        d, h, w = unpacked[3:6]
        spacing = unpacked[6:9]
        affine_flat = unpacked[9:25]
        count, bits, reserved, context = unpacked[25:]
        if reserved != 0 or context != _CONTEXT or bits != self.value_bits:
            raise ValueError("A-3D packet metadata mismatch")
        shape, spacing, affine = _geometry((d, h, w), spacing,
                                            tuple(tuple(affine_flat[r * 4:(r + 1) * 4]) for r in range(4)))
        level_bytes = (bits + 7) // 8
        expected = _HEADER.size + count * (_RECORD_ID.size + level_bytes) + _CHECKSUM_SIZE
        if len(payload) != expected:
            raise ValueError("truncated or trailing A-3D packet bytes")
        unsigned, checksum = payload[:-_CHECKSUM_SIZE], payload[-_CHECKSUM_SIZE:]
        if hashlib.sha256(unsigned).digest()[:_CHECKSUM_SIZE] != checksum:
            raise ValueError("A-3D packet checksum mismatch")
        ids, levels, offset = [], [], _HEADER.size
        for _ in range(count):
            idx = _RECORD_ID.unpack_from(payload, offset)[0]; offset += _RECORD_ID.size
            level = int.from_bytes(payload[offset:offset + level_bytes], "big"); offset += level_bytes
            ids.append(idx); levels.append(level)
        query_ids = self._validate_ids(ids)
        if any(level > self.max_level for level in levels) or not set(self.base_ids).issubset(query_ids):
            raise ValueError("invalid A-3D packet values or missing R0")
        if budget_bits is not None:
            if isinstance(budget_bits, bool) or int(budget_bits) != budget_bits or int(budget_bits) < 0:
                raise ValueError("budget_bits must be a non-negative integer")
            if len(payload) * 8 > int(budget_bits):
                raise ValueError("A-3D packet exceeds bit budget")
        return CodePacket3D(self.bank_fingerprint, shape, spacing, affine, query_ids, tuple(levels), bits)

    def dequantize(self, packet, *, device=None, dtype=torch.float32) -> torch.Tensor:
        if isinstance(packet, (bytes, bytearray, memoryview)):
            packet = self.deserialize(packet)
        if not isinstance(packet, CodePacket3D) or packet.bank_fingerprint != self.bank_fingerprint:
            raise ValueError("packet does not belong to this codec")
        return self.dequantize_levels(packet.ids, packet.levels, device=device, dtype=dtype)

    @staticmethod
    def packet_bits(payload: bytes | bytearray | memoryview) -> int:
        """Return the exact serialized size used by the bit budget."""
        return len(bytes(payload)) * 8

    def _ids_for_count(self, count: int) -> tuple[int, ...]:
        if isinstance(count, bool) or int(count) != count or int(count) < 0:
            raise ValueError("count must be a non-negative integer")
        ids = list(self.base_ids)
        for idx in range(len(self.bank)):
            if idx not in ids and len(ids) - len(self.base_ids) < int(count):
                ids.append(idx)
        if len(ids) - len(self.base_ids) != int(count):
            raise ValueError("count exceeds non-R0 bank queries")
        return tuple(ids)

    def cost(self, ids_or_count, shape=(1, 1, 1), spacing=(1.0, 1.0, 1.0), *, affine_zyx=None) -> int:
        if isinstance(ids_or_count, (int, float)) and not isinstance(ids_or_count, bool):
            if not math.isfinite(float(ids_or_count)) or int(ids_or_count) != ids_or_count or int(ids_or_count) < 0:
                raise ValueError("query count must be a non-negative integer")
            ids = self._ids_for_count(int(ids_or_count))
        else:
            ids = self._validate_ids(ids_or_count)
        packet_ids, _ = self._with_base(ids, None, require_base_values=False)
        return len(self._packet_bytes(packet_ids, [0] * len(packet_ids), shape, spacing, affine_zyx)) * 8

    def incremental_cost(self, existing_ids, new_id, shape=(1, 1, 1), spacing=(1.0, 1.0, 1.0), *, affine_zyx=None) -> int:
        existing = self._validate_ids(existing_ids)
        if isinstance(new_id, bool) or int(new_id) != new_id:
            raise ValueError("new_id must be an integer")
        new_id = int(new_id)
        if new_id in existing or new_id in self.base_ids:
            return 0
        return self.cost(existing + (new_id,), shape=shape, spacing=spacing, affine_zyx=affine_zyx) - self.cost(existing, shape=shape, spacing=spacing, affine_zyx=affine_zyx)

    def fits(self, ids, budget_bits: int, shape=(1, 1, 1), spacing=(1.0, 1.0, 1.0), *, affine_zyx=None) -> bool:
        if isinstance(budget_bits, bool) or int(budget_bits) != budget_bits or int(budget_bits) < 0:
            raise ValueError("budget_bits must be a non-negative integer")
        return self.cost(ids, shape=shape, spacing=spacing, affine_zyx=affine_zyx) <= int(budget_bits)


__all__ = ["QuantizedValues3D", "CodePacket3D", "MeasurementCodec3D"]
