"""Fixed, hash-pinned boundary feature memory and its paired retrieval module.

The serialized bank is an NPZ of numeric arrays plus UTF-8 JSON metadata. It
never relies on Python pickle. The tensors are registered as persistent buffers
so a student checkpoint contains everything needed for teacher-free inference.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


CLASS_MAP = ("BG", "RV", "MYO", "LV")
CLASS_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
MAX_ENTRIES_PER_GROUP = 8
_SCHEMA_VERSION = 1


def _canonical_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("bank metadata must be a mapping")
    result = dict(metadata)
    result.setdefault("schema_version", _SCHEMA_VERSION)
    result.setdefault("class_map", list(CLASS_MAP))
    result.setdefault("pairs", [list(pair) for pair in CLASS_PAIRS])
    if result["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("unsupported memory bank schema version")
    if tuple(result["class_map"]) != CLASS_MAP:
        raise ValueError("memory bank class map must be BG,RV,MYO,LV")
    if tuple(tuple(pair) for pair in result["pairs"]) != CLASS_PAIRS:
        raise ValueError("memory bank class-pair order is not canonical")
    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("bank provenance must be a nonempty mapping")
    # JSON encoding is also a validation step: reject tensors, NaNs and custom
    # objects instead of silently persisting non-portable representations.
    try:
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("bank metadata must be finite JSON data") from exc
    return result


def _validate_arrays(
    endpoints: Any, confidences: Any, valid: Any, metadata: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoints = np.asarray(endpoints, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32)
    valid = np.asarray(valid)
    if valid.dtype != np.bool_:
        raise ValueError("valid must be a boolean [6,N] array")
    if endpoints.ndim != 4 or endpoints.shape[0] != 6 or endpoints.shape[2] != 2:
        raise ValueError("endpoints must have shape [6,N,2,D]")
    groups, count, _, dim = endpoints.shape
    if not 1 <= count <= MAX_ENTRIES_PER_GROUP or dim < 1:
        raise ValueError("bank must have 1..8 entries/group and positive feature dimension")
    if confidences.shape != (groups, count, 2) or valid.shape != (groups, count):
        raise ValueError("confidence/valid arrays do not match endpoint shape")
    if not np.isfinite(endpoints).all() or not np.isfinite(confidences).all():
        raise ValueError("bank vectors and confidences must be finite")
    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise ValueError("endpoint confidence must lie in [0,1]")
    if np.any(valid[..., None] & (confidences < 0.8)):
        raise ValueError("valid endpoints require confidence >= 0.8")
    if np.any((~valid)[..., None] & (confidences != 0.0)):
        raise ValueError("invalid entries must have zero confidence")
    if np.any((~valid)[..., None, None] & (endpoints != 0.0)):
        raise ValueError("invalid entries must use zero vectors")
    norms = np.linalg.norm(endpoints, axis=-1)
    if np.any(valid[..., None] & (np.abs(norms - 1.0) > 1e-4)):
        raise ValueError("valid endpoint vectors must be unit normalized")
    entry_metadata = metadata.get("entry_metadata")
    if entry_metadata is not None:
        if not isinstance(entry_metadata, list) or len(entry_metadata) != groups:
            raise ValueError("entry_metadata must be indexed [6][N][2]")
        for group in entry_metadata:
            if not isinstance(group, list) or len(group) != count:
                raise ValueError("entry_metadata must be indexed [6][N][2]")
            if any(not isinstance(entry, list) or len(entry) != 2 for entry in group):
                raise ValueError("entry_metadata must identify both endpoints")
    # Empty groups are represented only by invalid zero slots. They do not
    # acquire a fabricated endpoint or class label.
    return np.ascontiguousarray(endpoints), np.ascontiguousarray(confidences), np.ascontiguousarray(valid)


class MemoryBank(nn.Module):
    """Immutable bank tensors stored as persistent, non-trainable buffers.

    Args:
        endpoints: normalized endpoint vectors ``[6,N,2,D]``.
        confidences: positive endpoint reliabilities ``[6,N,2]`` for valid
            entries, zero for padded entries.
        valid: boolean occupancy ``[6,N]``; an all-false group means empty.
        metadata: JSON-compatible provenance and canonical class metadata.
    """

    def __init__(
        self,
        endpoints: Any,
        confidences: Any,
        valid: Any,
        metadata: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.metadata = _canonical_metadata(metadata)
        ep, conf, mask = _validate_arrays(endpoints, confidences, valid, self.metadata)
        self.register_buffer("endpoints", torch.from_numpy(ep.copy()), persistent=True)
        self.register_buffer("confidences", torch.from_numpy(conf.copy()), persistent=True)
        self.register_buffer("valid", torch.from_numpy(mask.copy()), persistent=True)

    @property
    def feature_dim(self) -> int:
        return int(self.endpoints.shape[-1])

    @property
    def entries_per_group(self) -> int:
        return int(self.endpoints.shape[1])

    def extra_repr(self) -> str:
        return f"entries/group={self.entries_per_group}, feature_dim={self.feature_dim}, immutable=True"


def _bank_bytes(bank: MemoryBank) -> bytes:
    # Serialize canonical arrays on CPU and metadata as bytes, never as an
    # object-dtype array. allow_pickle=False is required by the loader.
    arrays = {
        "endpoints": bank.endpoints.detach().cpu().numpy().astype(np.float32, copy=False),
        "confidences": bank.confidences.detach().cpu().numpy().astype(np.float32, copy=False),
        "valid": bank.valid.detach().cpu().numpy().astype(np.bool_, copy=False),
        "metadata_json": np.frombuffer(
            json.dumps(bank.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"),
            dtype=np.uint8,
        ),
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def save_bank(bank: MemoryBank, path: str | os.PathLike[str]) -> str:
    """Atomically save a fixed bank and return SHA-256 of the exact file bytes."""
    if not isinstance(bank, MemoryBank):
        raise TypeError("save_bank expects a MemoryBank")
    payload = _bank_bytes(bank)
    digest = hashlib.sha256(payload).hexdigest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, destination)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return digest


def load_bank(
    path: str | os.PathLike[str],
    expected_sha256: str,
    map_location: str | torch.device = "cpu",
) -> MemoryBank:
    """Load and validate an NPZ bank; optional hash is checked before parsing."""
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("a 64-character expected_sha256 is required")
    if digest.lower() != expected_sha256.lower():
        raise ValueError("memory bank SHA-256 mismatch")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            required = {"endpoints", "confidences", "valid", "metadata_json"}
            if set(archive.files) != required:
                raise ValueError("memory bank archive has unexpected/missing arrays")
            metadata_bytes = np.asarray(archive["metadata_json"])
            if metadata_bytes.dtype != np.uint8 or metadata_bytes.ndim != 1:
                raise ValueError("metadata_json must be a one-dimensional uint8 array")
            metadata = json.loads(metadata_bytes.tobytes().decode("utf-8"))
            ep = np.asarray(archive["endpoints"])
            conf = np.asarray(archive["confidences"])
            valid = np.asarray(archive["valid"])
    except (OSError, ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid safe memory bank archive") from exc
    bank = MemoryBank(ep, conf, valid, metadata)
    bank.to(map_location)
    # Keep the verified file identity available without placing it in a tensor
    # state dict. Training code may store this digest in checkpoint metadata.
    bank.bank_sha256 = digest
    return bank


def _normalized_log_conf(confidence: Tensor) -> Tensor:
    # Invalid slots have zero confidence and are separately masked to -inf.
    return confidence.clamp_min(torch.finfo(confidence.dtype).tiny).log()


class TeacherMemory(nn.Module):
    """A2/A3 memory residual using one shared projection and six pair scales.

    ``h`` is the stride-4 feature after ``NativeStudent.refine[0]`` with shape
    ``[B,C,H4,W4]``. ``base_logits`` is the unmodified native-resolution head
    output ``[B,4,H,W]``. The return is a native-size class-logit displacement.

    In A3, each pair's softmax energy is ``(s_c+s_d)/(2*tau) +
    log(conf_c*conf_d)``. A2 independently marginalizes class endpoint
    evidence and then averages the nonlinear pair scores over the outer
    product; this uses the exact same memory vectors, projection and scales.
    Pair updates are antisymmetric, so their four-class sum is zero.
    """

    def __init__(
        self,
        channels: int,
        bank: MemoryBank,
        mode: str = "paired",
        tau: float = 0.2,
        gamma: float = 0.5,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not isinstance(bank, MemoryBank):
            raise TypeError("bank must be a MemoryBank")
        if mode not in {"paired", "unpaired"}:
            raise ValueError("mode must be 'paired' (A3) or 'unpaired' (A2)")
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        if not np.isfinite(gamma) or not 0.0 < gamma < 1.0:
            raise ValueError("gamma initialization must lie strictly between zero and one")
        self.channels = int(channels)
        self.mode = mode
        self.tau = float(tau)
        self.gamma_max = 0.5
        self.projection = nn.Parameter(torch.empty(bank.feature_dim, channels))
        nn.init.orthogonal_(self.projection)
        # gamma_k = 0.5 * sigmoid(alpha_k); alpha starts at -4 per contract.
        self.alpha = nn.Parameter(torch.full((len(CLASS_PAIRS),), -4.0))
        if gamma != 0.5:
            # Preserve the requested initialization relation for custom caps.
            self.gamma_max = float(gamma)
        self.bank = bank

    def _pool_paired(self, query: Tensor, group: int) -> Tensor:
        # query [B,D,H4,W4], bank [N,2,D], output [B,H4,W4]
        endpoints = self.bank.endpoints[group].to(dtype=query.dtype)
        conf = self.bank.confidences[group].to(dtype=query.dtype)
        valid = self.bank.valid[group]
        if not bool(valid.any()):
            return query.new_zeros(query.shape[0], *query.shape[-2:])
        scores = torch.einsum("bdhw,nkd->bnhwk", query, endpoints)
        sc, sd = scores[..., 0], scores[..., 1]  # [B,N,H4,W4]
        energy = (sc + sd) / (2.0 * self.tau)
        energy = energy + (conf[:, 0] * conf[:, 1]).clamp_min(1e-12).log()[None, :, None, None]
        energy = energy.masked_fill(~valid[None, :, None, None], -torch.inf)
        attention = energy.softmax(dim=1)
        return (attention * torch.tanh((sc - sd) / self.tau)).sum(dim=1)

    def _pool_unpaired(self, query: Tensor, group: int) -> Tensor:
        # Each endpoint class gets its own marginal over N entries, followed by
        # an N x N nonlinear comparison for every query location.
        endpoints = self.bank.endpoints[group].to(dtype=query.dtype)
        conf = self.bank.confidences[group].to(dtype=query.dtype)
        valid = self.bank.valid[group]
        if not bool(valid.any()):
            return query.new_zeros(query.shape[0], *query.shape[-2:])
        score = torch.einsum("bdhw,nkd->bnhwk", query, endpoints)
        sc, sd = score[..., 0], score[..., 1]
        mask = valid[None, :, None, None]
        ac_logits = (sc / (2.0 * self.tau) + _normalized_log_conf(conf[:, 0])[None, :, None, None])
        ad_logits = (sd / (2.0 * self.tau) + _normalized_log_conf(conf[:, 1])[None, :, None, None])
        ac = ac_logits.masked_fill(~mask, -torch.inf).softmax(dim=1)
        ad = ad_logits.masked_fill(~mask, -torch.inf).softmax(dim=1)
        pair_score = torch.tanh((sc[:, :, None] - sd[:, None, :]) / self.tau)  # [B,N,N,H4,W4]
        weights = ac[:, :, None] * ad[:, None, :]
        return (weights * pair_score).sum(dim=(1, 2))

    def forward(self, h: Tensor, base_logits: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if h.ndim != 4 or h.shape[1] != self.channels:
            raise ValueError(f"h must have shape [B,{self.channels},H4,W4]")
        if base_logits.ndim != 4 or base_logits.shape[:2] != (h.shape[0], 4):
            raise ValueError("base_logits must have shape [B,4,H,W]")
        if not torch.is_floating_point(h) or not torch.is_floating_point(base_logits):
            raise TypeError("h and base_logits must be floating point tensors")
        if h.device != base_logits.device:
            raise ValueError("h and base_logits must be on the same device")
        native_h, native_w = base_logits.shape[-2:]
        padded_h, padded_w = 4 * h.shape[-2], 4 * h.shape[-1]
        if native_h > padded_h or native_w > padded_w:
            raise ValueError("stride-4 feature grid is too small for native logits")
        if valid_mask is not None:
            if valid_mask.shape != (h.shape[0], *base_logits.shape[-2:]) or valid_mask.dtype is not torch.bool:
                raise ValueError("valid_mask must be bool [B,H,W] at native logit resolution")
            valid_mask = valid_mask.to(device=h.device)
        if self.bank.endpoints.device != h.device:
            raise ValueError("TeacherMemory and its bank must be moved to h.device together")

        # W maps the post-refine[0] feature to the bank's 32D feature space.
        q = F.conv2d(h, self.projection[:, :, None, None])
        q = F.normalize(q, p=2, dim=1, eps=1e-8)
        # Keep the original pixel grid aligned: native probabilities and the
        # geometric mask are padded right/bottom to the full decoder footprint,
        # then pooled over exact non-overlapping 4x4 stride-4 cells. Resizing the
        # native map directly to h would compress non-divisible dimensions.
        pad_h, pad_w = padded_h - native_h, padded_w - native_w
        p0_native = base_logits.softmax(dim=1).detach()
        p0_padded = F.pad(p0_native, (0, pad_w, 0, pad_h), value=0.0)
        p0 = F.avg_pool2d(p0_padded, kernel_size=4, stride=4)
        if valid_mask is None:
            native_valid = torch.ones(
                (h.shape[0], 1, native_h, native_w), dtype=torch.bool, device=h.device
            )
        else:
            native_valid = valid_mask[:, None]
        padded_valid = F.pad(native_valid, (0, pad_w, 0, pad_h), value=False)
        mask4 = F.avg_pool2d(padded_valid.float(), kernel_size=4, stride=4).ge(1.0 - 1e-6)
        delta4 = h.new_zeros((h.shape[0], 4, *h.shape[-2:]))
        scales = self.gamma_max * self.alpha.sigmoid()
        for idx, (c, d) in enumerate(CLASS_PAIRS):
            residual = self._pool_paired(q, idx) if self.mode == "paired" else self._pool_unpaired(q, idx)
            gate = (4.0 * p0[:, c] * p0[:, d]).clamp_(0.0, 1.0)
            gate = gate * mask4[:, 0].to(dtype=gate.dtype)
            pair_delta = (scales[idx] * gate * residual).to(dtype=delta4.dtype)
            delta4[:, c] = delta4[:, c] + pair_delta
            delta4[:, d] = delta4[:, d] - pair_delta
        # For each class c, sum_d 4*p_c*p_d = 4*p_c*(1-p_c) <= 1. Since
        # |r_cd|<=1 and every gamma_k<=gamma_max, the class residual obeys
        # |delta_c|<=gamma_max, with the same bound preserved by interpolation.
        delta_padded = F.interpolate(
            delta4, size=(padded_h, padded_w), mode="bilinear", align_corners=False
        )
        delta = delta_padded[..., :native_h, :native_w]
        delta = delta * native_valid.to(dtype=delta.dtype)
        return delta


@torch.no_grad()
def permute_endpoints(bank: MemoryBank, seed: int = 0) -> MemoryBank:
    """A4: shuffle d endpoints and their confidence metadata within each group.

    The c endpoints stay fixed, preserving class marginals. Groups with fewer
    than two valid entries remain unchanged and are recorded as identities.
    """
    ep = bank.endpoints.detach().cpu().numpy().copy()
    conf = bank.confidences.detach().cpu().numpy().copy()
    valid = bank.valid.detach().cpu().numpy().copy()
    metadata = json.loads(json.dumps(bank.metadata))
    rng = np.random.default_rng(seed)
    identity_groups = []
    for group in range(len(CLASS_PAIRS)):
        indices = np.flatnonzero(valid[group])
        if len(indices) < 2:
            identity_groups.append(group)
            continue
        order = rng.permutation(indices)
        if np.array_equal(order, indices):
            # A4 is an association-breaking control, so a chance identity
            # permutation for a populated group would silently erase the factor.
            order = np.roll(indices, 1)
        ep[group, indices, 1] = ep[group, order, 1]
        conf[group, indices, 1] = conf[group, order, 1]
        if "entry_metadata" in bank.metadata:
            # Keep source/coordinate/timepoint lineage attached to the endpoint
            # whose feature and confidence are being permuted.
            entries = metadata["entry_metadata"][group]
            for target, source in zip(indices.tolist(), order.tolist()):
                entries[target][1] = bank.metadata["entry_metadata"][group][source][1]
    metadata["control"] = {"name": "A4_endpoint_permutation", "seed": int(seed),
                           "identity_groups": identity_groups}
    return MemoryBank(ep, conf, valid, metadata)


@torch.no_grad()
def random_feature_control(bank: MemoryBank, seed: int = 0) -> MemoryBank:
    """A6: fixed-seed random normalized vectors with original occupancy/weights."""
    rng = np.random.default_rng(seed)
    ep = bank.endpoints.detach().cpu().numpy().copy()
    valid = bank.valid.detach().cpu().numpy().copy()
    for group, row in zip(*np.where(valid)):
        vector = rng.normal(size=ep[group, row].shape).astype(np.float32)
        norms = np.linalg.norm(vector, axis=-1, keepdims=True)
        ep[group, row] = vector / np.maximum(norms, 1e-12)
    metadata = json.loads(json.dumps(bank.metadata))
    original_provenance = metadata["provenance"]
    metadata["provenance"] = {
        "kind": "random_control",
        "seed": int(seed),
        "derived_from": original_provenance,
        "feature_provenance": "random_normalized;not_teacher_features",
    }
    metadata["control"] = {"name": "A6_random_normalized_features", "seed": int(seed),
                           "source_provenance": "random_control;not_teacher_features"}
    return MemoryBank(ep, bank.confidences.detach().cpu().numpy().copy(), valid, metadata)

