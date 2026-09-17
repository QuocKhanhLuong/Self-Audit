"""One shared recurrent Annotation Expert for soft annotation refinement."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call
import torch.nn.functional as F

from .dynamic_window import DynamicWindowAttention


MODEL_ENTROPY_VERSION: str = "2.0.0"


def entropy_from_logits(logits: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute normalized Shannon entropy from unnormalized logits.

    The channel dimension (dim=1) represents classes.
    Computes Shannon entropy normalized by ln(num_classes) so that:
      - Uniform distribution produces entropy ≈ 1.0
      - One-hot (pure certainty) distribution produces entropy = 0.0

    This implementation is strictly invariant to constant shifts along the
    channel dimension (shift invariance) and invariant to other batch items
    (batch-composition invariance).

    Computations are performed in float32 for float16 and bfloat16 inputs
    to ensure numerical stability against overflow/underflow, while preserving
    the caller's original output dtype, float64 precision, and backpropagation gradients.

    Parameters
    ----------
    logits : Tensor
        Unnormalized logit tensor of shape [B, C, H, W].
    eps : float, default 1e-8
        Numerical stability epsilon (reserved for API parity).

    Returns
    -------
    Tensor
        Normalized entropy tensor of shape [B, 1, H, W] in range [0, 1],
        matching the device and dtype of the input logits.
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected 4D logits tensor [B, C, H, W], got shape {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("Logits must contain only finite values (found NaN or Inf)")
    num_classes = logits.shape[1]
    if num_classes <= 1:
        return logits.new_zeros((logits.shape[0], 1, logits.shape[2], logits.shape[3]))

    orig_dtype = logits.dtype
    compute_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    logits_comp = logits.to(compute_dtype)

    # Invariant to constant shifts along channel dimension via log_softmax
    probs = F.softmax(logits_comp, dim=1)
    log_probs = F.log_softmax(logits_comp, dim=1)
    # Clamp extreme negative infinities to avoid 0 * -inf = NaN in pathological logit differences
    log_probs_safe = torch.nan_to_num(log_probs, neginf=-1e30)
    p_log_p = torch.where(probs > 0, probs * log_probs_safe, torch.zeros_like(probs))
    entropy = -p_log_p.sum(dim=1, keepdim=True)
    normalizer = float(math.log(num_classes))
    normalized_entropy = entropy / normalizer
    return normalized_entropy.to(orig_dtype)


def entropy_from_probabilities(
    probs: Tensor,
    eps: float = 1e-8,
    *,
    atol: float = 1e-3,
) -> Tensor:
    """Compute normalized Shannon entropy from probability simplex tensors.

    Validates that the input represents a valid probability distribution on
    the simplex:
      1. 4D shape [B, C, H, W]
      2. Strictly finite values (no NaNs or Infs)
      3. Valid probability range [0, 1] within numerical tolerance `atol`
      4. Valid per-pixel probability simplex (sum over classes equals 1.0 within `atol`)

    Does NOT silently renormalize invalid inputs.

    Computations are performed in float32 for float16 and bfloat16 inputs
    to prevent underflow of eps or overflow during exponentiation/logarithms,
    while preserving the caller's original output dtype, float64 precision,
    and backpropagation gradients.

    Parameters
    ----------
    probs : Tensor
        Probability tensor of shape [B, C, H, W].
    eps : float, default 1e-8
        Numerical clamping epsilon for logarithm in compute precision.
    atol : float, default 1e-3
        Absolute numerical tolerance for range [0, 1] and simplex sum == 1.0.

    Returns
    -------
    Tensor
        Normalized entropy tensor of shape [B, 1, H, W] in range [0, 1],
        matching the device and dtype of the input probabilities.
    """
    if probs.ndim != 4:
        raise ValueError(f"Expected 4D probabilities tensor [B, C, H, W], got shape {tuple(probs.shape)}")
    if not torch.isfinite(probs).all():
        raise ValueError("Probabilities must contain only finite values (found NaN or Inf)")
    num_classes = probs.shape[1]
    if num_classes <= 1:
        return probs.new_zeros((probs.shape[0], 1, probs.shape[2], probs.shape[3]))

    orig_dtype = probs.dtype
    compute_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    probs_comp = probs.to(compute_dtype)

    if (probs_comp < -atol).any() or (probs_comp > 1.0 + atol).any():
        min_val = float(probs_comp.min().detach().item())
        max_val = float(probs_comp.max().detach().item())
        raise ValueError(
            f"Probabilities out of valid range [0, 1] (min={min_val:.6f}, max={max_val:.6f})"
        )

    pixel_sums = probs_comp.sum(dim=1, keepdim=True)
    discrepancy = torch.abs(pixel_sums - 1.0)
    if (discrepancy > atol).any():
        max_disc = float(discrepancy.max().detach().item())
        raise ValueError(
            f"Probabilities do not sum to 1.0 along channel dimension (max deviation={max_disc:.6f} > atol={atol}). "
            "Silent renormalization of invalid inputs is prohibited."
        )

    # Standard convention 0 * log(0) -> 0 limit for Shannon entropy
    safe_probs = torch.clamp(probs_comp, min=eps, max=1.0)
    log_probs = torch.log(safe_probs)
    p_log_p = torch.where(probs_comp > 0, probs_comp * log_probs, torch.zeros_like(probs_comp))
    entropy = -p_log_p.sum(dim=1, keepdim=True)
    normalizer = float(math.log(num_classes))
    normalized_entropy = entropy / normalizer
    return normalized_entropy.to(orig_dtype)


def annotation_entropy(
    x: Tensor,
    eps: float = 1e-8,
    *,
    input_type: str = "logits",
    atol: float = 1e-3,
) -> Tensor:
    """Compute normalized Shannon entropy from logits or probabilities.

    .. note::
        Range-based heuristic dispatch has been removed. Callers must use
        explicit input typing: 'logits' (default) or 'probabilities'.
        Direct callers should prefer :func:`entropy_from_logits` or
        :func:`entropy_from_probabilities`.

    Parameters
    ----------
    x : Tensor
        Input 4D tensor [B, C, H, W].
    eps : float, default 1e-8
        Numerical stability epsilon.
    input_type : {"logits", "probabilities", "probs"}, default "logits"
        Explicit input representation type. Range-based heuristic dispatch is not performed.
    atol : float, default 1e-3
        Simplex tolerance when input_type is 'probabilities'.

    Returns
    -------
    Tensor
        Normalized entropy tensor [B, 1, H, W].
    """
    clean_type = str(input_type).lower().strip()
    if clean_type in ("logits", "logit"):
        return entropy_from_logits(x, eps=eps)
    elif clean_type in ("probabilities", "probs", "probability"):
        return entropy_from_probabilities(x, eps=eps, atol=atol)
    else:
        raise ValueError(
            f"Invalid input_type: {input_type!r}. Must be 'logits' or 'probabilities'. "
            "Heuristic range-based dispatch has been removed."
        )


@dataclass
class AnnotationExpertOutput:
    delta_logits: Tensor
    update_gate: Tensor
    candidate_logits: Tensor
    depth: int
    window_metadata: dict[str, Tensor] | None = None
    # Appended with defaults so existing positional construction keeps working.
    geometry: tuple[dict[str, Any], ...] = ()
    state_identity: str | None = None

    @property
    def A_candidate(self) -> Tensor:
        return self.candidate_logits

    def __iter__(self):
        yield self.delta_logits
        yield self.update_gate


class StaleReplayRecordError(RuntimeError):
    """Raised when a replay record no longer matches the live model state."""


@dataclass(frozen=True)
class ExpertReplayRecord:
    """Bounded, runtime-only replay record for ONE accepted transition.

    Every tensor is a detached ordinary clone (never an inference tensor and
    never a view of a live autograd graph), so the record can be replayed
    inside a gradient-enabled region even when it was captured under
    ``torch.no_grad()`` or ``torch.inference_mode()``.

    The record is deliberately *not* a buffer and is never serialized into a
    ``state_dict`` or a checkpoint: it describes one in-flight inference, not
    model state.
    """

    shared_features: Tensor
    annotation_logits: Tensor
    previous_audit_evidence: Tensor | None
    turn_index: int
    iteration_index: int
    depth: int
    coordinates: tuple[Tensor, ...]
    coordinates_preclamp: tuple[Tensor, ...]
    factual_candidate_logits: Tensor
    factual_update_gate: Tensor
    state_identity: str
    record_kind: str
    audit_conditioning: str
    offset_mode: str
    #: The conditioning identities the window generator ACTUALLY used, one per
    #: internal depth, after the embedding-table clamp.  The raw request
    #: ``iteration_index + depth`` can exceed the table and be clamped, so
    #: reporting the raw value would misdescribe the computation.  These are
    #: read back from the captured geometry, never recomputed.
    effective_turn_indices: tuple[int, ...] = ()
    effective_iteration_indices: tuple[int, ...] = ()

    @property
    def feature_hw(self) -> tuple[int, int]:
        return int(self.shared_features.shape[-2]), int(self.shared_features.shape[-1])

    @property
    def batch_size(self) -> int:
        return int(self.shared_features.shape[0])


#: Record kinds.  Only ``ordinary`` transitions are replay-eligible: their
#: factual output *is* an ordinary AnnotationExpert update of their immediate
#: pre-state, which is exactly what C1 re-executes.  A transition produced by
#: the Candidate C solver or by direct rollback is a different function of the
#: pre-state and is therefore recorded as ineligible, never relabelled.
RECORD_KIND_ORDINARY: str = "ordinary"
RECORD_KIND_CANDIDATE_C: str = "candidate_c"
RECORD_KIND_ROLLBACK: str = "rollback"
REPLAY_ELIGIBLE_RECORD_KINDS: frozenset[str] = frozenset({RECORD_KIND_ORDINARY})

AUDIT_CONDITIONING_MODES: tuple[str, ...] = ("full", "feature_only")


def _freeze_tensor(value: Tensor | None) -> Tensor | None:
    """Materialise an ordinary detached clone usable by autograd later."""

    if value is None:
        return None
    if not torch.is_tensor(value):
        raise TypeError(f"expected tensor, got {type(value)!r}")
    with torch.inference_mode(False), torch.no_grad():
        return value.detach().clone()


class AnnotationExpert(nn.Module):
    """Shared-weight recurrent residual updater.

    The dynamic window is conditioned by the projected previous local audit
    evidence.  A high audit value is not assigned a hand-written window size;
    the generator learns all support parameters from the joint state.
    """

    entropy_version: str = MODEL_ENTROPY_VERSION

    def __init__(
        self,
        feature_channels: int = 96,
        num_classes: int = 4,
        *,
        audit_channels: int = 3,
        window_k: int = 8,
        max_turns: int = 3,
        audit_conditioning: str = "full",
        offset_mode: str = "structured",
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.num_classes = int(num_classes)
        self.audit_channels = int(audit_channels)
        self.max_turns = int(max_turns)
        conditioning = str(audit_conditioning).lower().strip()
        if conditioning not in AUDIT_CONDITIONING_MODES:
            raise ValueError(
                f"audit_conditioning must be one of {list(AUDIT_CONDITIONING_MODES)}, got {audit_conditioning!r}"
            )
        #: ``feature_only`` removes BOTH explicit audit-evidence entry paths:
        #: the audit channels of the input projection AND the window
        #: generator's conditioning map.  The input channel count is kept so
        #: the ablation shares the baseline parameter shapes exactly.
        self.audit_conditioning = conditioning
        self._replay_generation = 0
        input_channels = self.feature_channels + self.num_classes + 1 + self.audit_channels
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, self.feature_channels, 1),
            nn.GroupNorm(8 if self.feature_channels % 8 == 0 else 1, self.feature_channels),
            nn.GELU(),
        )
        # Exactly one shared dynamic-window refinement block is reused for all
        # recurrent turns and all depth iterations.
        self.refinement_block = DynamicWindowAttention(
            self.feature_channels,
            k=int(window_k),
            condition_channels=self.audit_channels,
            max_turns=max(self.max_turns, 3),
            offset_mode=offset_mode,
        )
        self.refinement_residual = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.feature_channels, 3, padding=1),
            nn.GroupNorm(8 if self.feature_channels % 8 == 0 else 1, self.feature_channels),
            nn.GELU(),
            nn.Conv2d(self.feature_channels, self.feature_channels, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.feature_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feature_channels // 2, self.num_classes, 1),
        )
        self.gate_head = nn.Conv2d(self.feature_channels, 1, 1)
        self.last_window_metadata: dict[str, Tensor] | None = None

    def forward(
        self,
        shared_features: Tensor,
        annotation_logits: Tensor,
        entropy: Tensor | None = None,
        previous_audit_evidence: Tensor | None = None,
        *,
        turn_index: int | Tensor = 0,
        iteration_index: int | Tensor = 0,
        return_metadata: bool = False,
        coordinate_overrides: Any = None,
        capture_geometry: bool = False,
    ) -> AnnotationExpertOutput:
        if shared_features.ndim != 4 or annotation_logits.ndim != 4:
            raise ValueError("shared_features and annotation_logits must be four-dimensional")
        if annotation_logits.shape[1] != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} annotation classes, got {annotation_logits.shape[1]}")
        spatial = shared_features.shape[-2:]
        annotation_low = F.interpolate(annotation_logits, size=spatial, mode="bilinear", align_corners=False)
        entropy = entropy_from_logits(annotation_logits) if entropy is None else entropy
        entropy_low = F.interpolate(entropy, size=spatial, mode="bilinear", align_corners=False)
        if previous_audit_evidence is None or self.audit_conditioning == "feature_only":
            audit_low = shared_features.new_zeros((shared_features.shape[0], self.audit_channels, *spatial))
        else:
            if previous_audit_evidence.ndim != 4:
                raise ValueError("previous_audit_evidence must be [B,C,H,W]")
            audit_low = F.interpolate(previous_audit_evidence, size=spatial, mode="bilinear", align_corners=False)
            if audit_low.shape[1] == 1:
                audit_low = audit_low.expand(-1, self.audit_channels, -1, -1)
            elif audit_low.shape[1] != self.audit_channels:
                audit_low = audit_low[:, : self.audit_channels]
                if audit_low.shape[1] < self.audit_channels:
                    padding = audit_low.new_zeros(
                        audit_low.shape[0],
                        self.audit_channels - audit_low.shape[1],
                        audit_low.shape[2],
                        audit_low.shape[3],
                    )
                    audit_low = torch.cat([audit_low, padding], dim=1)
        state = self.input_projection(torch.cat([shared_features, annotation_low, entropy_low, audit_low], dim=1))
        if torch.is_tensor(turn_index):
            turn_value = int(turn_index.reshape(-1)[0].item())
        else:
            turn_value = int(turn_index)
        depth = min(turn_value + 1, 3)
        depth = max(depth, 1)
        overrides: tuple[Tensor, ...] | None = None
        if coordinate_overrides is not None:
            overrides = tuple(coordinate_overrides)
            if len(overrides) != depth:
                raise ValueError(
                    "coordinate_overrides must supply exactly one realized support per internal "
                    f"iteration: expected {depth} for turn_index={turn_value}, got {len(overrides)}"
                )
        # ``feature_only`` also removes the window generator's conditioning map,
        # so neither audit entry path survives the ablation.
        window_condition = None if self.audit_conditioning == "feature_only" else audit_low
        metadata: dict[str, Tensor] | None = None
        geometry: list[dict[str, Any]] = []
        identity = self.state_identity() if capture_geometry else None
        want_metadata = bool(return_metadata or capture_geometry)
        for iteration in range(depth):
            iteration_value = iteration
            if isinstance(iteration_index, int):
                iteration_value += int(iteration_index)
            block_result = self.refinement_block(
                state,
                condition=window_condition,
                turn_index=turn_index,
                iteration_index=iteration_value,
                return_metadata=want_metadata,
                coordinate_override=None if overrides is None else overrides[iteration],
            )
            if want_metadata:
                window_output, window_metadata = block_result
            else:
                window_output, window_metadata = block_result, None
            state = state + self.refinement_residual(window_output)
            if return_metadata:
                metadata = window_metadata
            if capture_geometry and window_metadata is not None:
                # Reported geometry is detached so a training graph is never
                # retained by diagnostics; the returned candidate logits still
                # carry the intervention's gradient.
                geometry.append(
                    {
                        "turn_index": window_metadata["turn_index"].detach().clone(),
                        "iteration_index": window_metadata["iteration_index"].detach().clone(),
                        "internal_iteration": int(iteration),
                        "coordinates": window_metadata["coordinates"].detach(),
                        "coordinates_preclamp": window_metadata["coordinates_preclamp"].detach(),
                        "attention": window_metadata["attention"].detach(),
                        "override_used": bool(window_metadata["override_used"].item()),
                        "feature_hw": (int(spatial[0]), int(spatial[1])),
                        "state_identity": identity,
                    }
                )
        delta_low = self.delta_head(state)
        gate_low = self.gate_head(state)
        delta_logits = F.interpolate(delta_low, size=annotation_logits.shape[-2:], mode="bilinear", align_corners=False)
        update_gate = F.interpolate(gate_low, size=annotation_logits.shape[-2:], mode="bilinear", align_corners=False)
        candidate_logits = annotation_logits + torch.sigmoid(update_gate) * delta_logits
        self.last_window_metadata = metadata
        return AnnotationExpertOutput(
            delta_logits,
            update_gate,
            candidate_logits,
            depth,
            metadata,
            tuple(geometry),
            identity,
        )

    # ------------------------------------------------------------------
    # Candidate C: exact frozen replay of one accepted ordinary transition
    # ------------------------------------------------------------------

    def state_identity(self) -> str:
        """A cheap fail-closed signature of the exact replay state.

        Includes per-parameter and per-buffer object identity *and* version
        counters (an optimizer step bumps the version in place), dtype, device,
        the module train/eval mode and an explicit invalidation generation.
        A sum of versions is deliberately not used: it collides trivially.
        """

        parts: list[str] = [
            f"gen={int(self._replay_generation)}",
            f"mode={'train' if self.training else 'eval'}",
            f"audit={self.audit_conditioning}",
            f"offsets={self.refinement_block.offset_mode}",
        ]
        for name, parameter in self.named_parameters(recurse=True):
            parts.append(f"p|{name}|{id(parameter)}|{parameter._version}|{parameter.dtype}|{parameter.device}")
        for name, buffer in self.named_buffers(recurse=True):
            parts.append(f"b|{name}|{id(buffer)}|{buffer._version}|{buffer.dtype}|{buffer.device}")
        digest = hashlib.blake2s("\n".join(parts).encode("utf-8"), digest_size=16).hexdigest()
        return f"annotation_expert_v1:{digest}"

    def invalidate_replay_records(self) -> int:
        """Bump the generation so every outstanding record fails closed."""

        self._replay_generation = int(self._replay_generation) + 1
        return self._replay_generation

    def build_replay_record(
        self,
        *,
        shared_features: Tensor,
        annotation_logits: Tensor,
        previous_audit_evidence: Tensor | None,
        turn_index: int,
        iteration_index: int,
        output: AnnotationExpertOutput,
        record_kind: str = RECORD_KIND_ORDINARY,
    ) -> ExpertReplayRecord:
        """Freeze one transition into a bounded runtime replay record.

        ``output`` must come from a ``capture_geometry=True`` forward so the
        *realized* (post-clamp) supports of every internal iteration are known.
        """

        if not output.geometry:
            raise ValueError("build_replay_record requires a capture_geometry=True forward")
        if len(output.geometry) != int(output.depth):
            raise ValueError(
                f"captured geometry has {len(output.geometry)} entries for depth {output.depth}"
            )
        coordinates = tuple(_freeze_tensor(item["coordinates"]) for item in output.geometry)
        preclamp = tuple(_freeze_tensor(item["coordinates_preclamp"]) for item in output.geometry)
        # This architecture conditions every row of a call on the same scalar
        # turn/iteration identity, so one resolved scalar per depth describes
        # the call exactly; the values come from the capture, not from a
        # re-derivation of the clamp.
        effective_turns = tuple(int(item["turn_index"].reshape(-1)[0]) for item in output.geometry)
        effective_iterations = tuple(
            int(item["iteration_index"].reshape(-1)[0]) for item in output.geometry
        )
        return ExpertReplayRecord(
            shared_features=_freeze_tensor(shared_features),
            annotation_logits=_freeze_tensor(annotation_logits),
            previous_audit_evidence=_freeze_tensor(previous_audit_evidence),
            turn_index=int(turn_index),
            iteration_index=int(iteration_index),
            depth=int(output.depth),
            coordinates=coordinates,
            coordinates_preclamp=preclamp,
            factual_candidate_logits=_freeze_tensor(output.candidate_logits),
            factual_update_gate=_freeze_tensor(output.update_gate),
            state_identity=self.state_identity(),
            record_kind=str(record_kind),
            audit_conditioning=self.audit_conditioning,
            offset_mode=self.refinement_block.offset_mode,
            effective_turn_indices=effective_turns,
            effective_iteration_indices=effective_iterations,
        )

    def _frozen_state(self) -> dict[str, Tensor]:
        frozen: dict[str, Tensor] = {}
        for name, parameter in self.named_parameters(recurse=True):
            frozen[name] = parameter.detach()
        for name, buffer in self.named_buffers(recurse=True):
            frozen[name] = buffer.detach()
        return frozen

    def replay(
        self,
        record: ExpertReplayRecord,
        coordinates: Any = None,
        *,
        capture_geometry: bool = False,
    ) -> AnnotationExpertOutput:
        """Re-execute a recorded transition, optionally at alternative supports.

        Only the realized sampling coordinates may differ from the record; every
        other input (features, pre-transition logits, prior audit evidence, turn
        and internal-iteration identities, depth) is the recorded one.  The
        replay is functional: parameters and buffers are passed in detached, so
        no solver gradient can reach the annotator, the encoder, the Auditor or
        the patient image, and no ``requires_grad`` flag is mutated.
        """

        if record.record_kind not in REPLAY_ELIGIBLE_RECORD_KINDS:
            raise ValueError(
                f"record_kind {record.record_kind!r} is not replay eligible; "
                f"eligible kinds are {sorted(REPLAY_ELIGIBLE_RECORD_KINDS)}"
            )
        if record.audit_conditioning != self.audit_conditioning:
            raise ValueError(
                f"record captured with audit_conditioning={record.audit_conditioning!r} "
                f"but module is {self.audit_conditioning!r}"
            )
        if record.offset_mode != self.refinement_block.offset_mode:
            raise ValueError(
                f"record captured with offset_mode={record.offset_mode!r} "
                f"but module is {self.refinement_block.offset_mode!r}"
            )
        identity = self.state_identity()
        if identity != record.state_identity:
            raise StaleReplayRecordError(
                "annotation expert state changed since the record was captured "
                f"({record.state_identity} -> {identity})"
            )
        supports = record.coordinates if coordinates is None else tuple(coordinates)
        if len(supports) != record.depth:
            raise ValueError(
                f"replay needs exactly {record.depth} realized supports, got {len(supports)}"
            )
        with torch.inference_mode(False):
            return functional_call(
                self,
                self._frozen_state(),
                args=(),
                kwargs={
                    "shared_features": record.shared_features,
                    "annotation_logits": record.annotation_logits,
                    "entropy": None,
                    "previous_audit_evidence": record.previous_audit_evidence,
                    "turn_index": record.turn_index,
                    "iteration_index": record.iteration_index,
                    "return_metadata": False,
                    "coordinate_overrides": supports,
                    "capture_geometry": capture_geometry,
                },
            )


SharedAnnotationExpert = AnnotationExpert
