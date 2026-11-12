"""Q-3D ResEnc-L teacher backed by the nnFoundationCNN encoder.

The encoder construction mirrors ``nnssl`` at pinned commit
``3d774e942256bd70e5e264c6b94fa38e25f7ed74`` (``get_res_enc_l``):
``dynamic_network_architectures==0.4.4`` ``ResidualEncoder`` with feature
widths ``[32, 64, 128, 256, 320, 320]`` and residual block counts
``[1, 3, 4, 6, 6, 6]``.  The checkpoint's decoder and segmentation heads are
deliberately ignored.  Q-3D creates a fresh decoder and boundary branch so
the only imported weights are the strict ``encoder.stem`` and
``encoder.stages`` tensors.

The teacher normalises its own input (volume-wise z-score), pads only on the
high side to the declared cumulative encoder stride, and crops all outputs
back to native geometry.  Student inputs are never normalised by this class.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:  # dynamic_network_architectures is a pinned runtime dependency.
    from dynamic_network_architectures.architectures.unet import (
        ResidualEncoderUNet,
        UNetDecoder,
    )
    from dynamic_network_architectures.building_blocks.residual_encoders import (
        ResidualEncoder,
    )
except ImportError as exc:  # pragma: no cover - exercised by environment setup
    raise ImportError(
        "Q-3D ResEnc-L requires dynamic_network_architectures==0.4.4"
    ) from exc

from ..candidate_a3d.teacher import CLASS_NAMES, _BoundaryPath3D

RESENC_L_FEATURES = (32, 64, 128, 256, 320, 320)
RESENC_L_BLOCKS = (1, 3, 4, 6, 6, 6)
RESENC_L_KERNELS = ((3, 3, 3),) * 6
RESENC_L_DEFAULT_STRIDES = ((1, 1, 1),) + ((2, 2, 2),) * 5
NNSSL_COMMIT = "3d774e942256bd70e5e264c6b94fa38e25f7ed74"
NNSSL_SOURCE_URL = "https://github.com/MIC-DKFZ/nnssl"
NNFOUNDATION_WEIGHTS_SHA256 = (
    "ac262d3e8c226c79f9567fddc34d38e011284730ddf2294b5c6c319ce2f22bbd"
)
NNFOUNDATION_HF_REVISION = "f443c09319988e833b3d9589bf95909e4443d7b2"
_HF_ARCHITECTURE = "ResEncL"


def _triples(value: Sequence[int] | int, *, count: int, name: str) -> tuple[tuple[int, int, int], ...]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        values: Sequence[Any] = (value,) * count
    else:
        values = value
    if not isinstance(values, Sequence) or len(values) != count:
        raise ValueError(f"{name} must contain {count} three-axis strides")
    parsed: list[tuple[int, int, int]] = []
    for index, item in enumerate(values):
        if isinstance(item, Integral) and not isinstance(item, bool):
            item = (item,) * 3
        if not isinstance(item, Sequence) or len(item) != 3:
            raise ValueError(f"{name}[{index}] must contain z,y,x")
        if any(not isinstance(axis, Integral) or isinstance(axis, bool) for axis in item):
            raise ValueError(f"{name}[{index}] must contain positive integers")
        axes = tuple(int(axis) for axis in item)
        if any(axis <= 0 for axis in axes):
            raise ValueError(f"{name}[{index}] must contain positive integers")
        parsed.append(axes)
    return tuple(parsed)


def cumulative_strides(strides: Sequence[Sequence[int]]) -> tuple[int, int, int]:
    parsed = _triples(strides, count=len(strides), name="strides")
    product = [1, 1, 1]
    for stride in parsed:
        for axis in range(3):
            product[axis] *= stride[axis]
    return tuple(product)


def _phase_offset_map(
    values: Mapping[str, Sequence[float]] | None,
) -> dict[str, tuple[float, float, float]]:
    """Validate the measured Phase-0 offsets consumed by the teacher.

    A zero offset is the analytical default for the pinned same-padded
    encoder.  When Phase 0 measures a lattice displacement greater than the
    tolerance, callers pass the locked value here; it is then used by both
    feature metadata and alignment rather than being silently discarded.
    """

    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError("phase_offsets must be a mapping from feature source to z,y,x")
    aliases = {
        "stride8": "encoder_stride8",
        "encoder_stride8": "encoder_stride8",
        "stride16": "encoder_stride16",
        "encoder_stride16": "encoder_stride16",
    }
    result: dict[str, tuple[float, float, float]] = {}
    for raw_name, raw_offset in values.items():
        canonical = aliases.get(raw_name)
        if canonical is None:
            raise ValueError(f"unsupported ResEnc phase offset source: {raw_name!r}")
        if isinstance(raw_offset, (str, bytes)) or not isinstance(raw_offset, Sequence) \
                or len(raw_offset) != 3:
            raise ValueError(f"phase_offsets[{raw_name!r}] must contain z,y,x")
        offset = tuple(float(value) for value in raw_offset)
        if not all(isfinite(value) for value in offset):
            raise ValueError(f"phase_offsets[{raw_name!r}] must be finite")
        if canonical in result and result[canonical] != offset:
            raise ValueError(f"conflicting aliases for phase_offsets[{canonical!r}]")
        result[canonical] = offset
    return result


def build_res_enc_l(
    strides: Sequence[Sequence[int]] | None = None,
    *,
    input_channels: int = 1,
    features_per_stage: Sequence[int] = RESENC_L_FEATURES,
    n_blocks_per_stage: Sequence[int] = RESENC_L_BLOCKS,
) -> ResidualEncoderUNet:
    """Build the pinned nnssl ResEnc-L encoder with ``return_skips=True``.

    The default arguments are copied from ``nnssl``'s ``get_res_enc_l``.  A
    caller may provide anisotropic strides from the locked Phase-0 geometry
    audit.  Smaller widths/block counts are allowed only for explicit
    synthetic smoke construction; they do not alter the real default.
    """

    stage_count = len(tuple(features_per_stage))
    if stage_count != 6:
        raise ValueError("ResEnc-L requires six encoder stages")
    features = tuple(int(value) for value in features_per_stage)
    blocks = tuple(int(value) for value in n_blocks_per_stage)
    if len(blocks) != 6 or any(value <= 0 for value in (*features, *blocks)):
        raise ValueError("ResEnc-L feature widths and block counts must be positive")
    parsed_strides = _triples(
        RESENC_L_DEFAULT_STRIDES if strides is None else strides,
        count=6,
        name="strides",
    )
    # ``get_res_enc_l`` returns a ResidualEncoderUNet.  We construct that
    # exact wrapper here (including its fresh decoder/head) and Q3D later
    # consumes only ``model.encoder``.  Keeping the wrapper makes parameter
    # and state-dict comparison with the pinned nnssl factory unambiguous.
    return ResidualEncoderUNet(
        input_channels=int(input_channels),
        n_stages=6,
        features_per_stage=features,
        conv_op=nn.Conv3d,
        kernel_sizes=RESENC_L_KERNELS,
        strides=parsed_strides,
        n_blocks_per_stage=blocks,
        num_classes=4,
        n_conv_per_stage_decoder=[1] * 5,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )


def build_resenc_l(*args: Any, **kwargs: Any) -> ResidualEncoderUNet:
    """Canonical spelling for :func:`build_res_enc_l` used by Q3D callers.

    ``build_res_enc_l`` mirrors the upstream nnssl name; this compact alias is
    kept as the public Q3D factory spelling requested by the active spec.
    """

    return build_res_enc_l(*args, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_encoder_state(payload: Mapping[str, Any]) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise ValueError("nnFoundation checkpoint must be a mapping")
    plan = payload.get("nnssl_adaptation_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("nnFoundation checkpoint adaptation plan is missing")
    architecture_plans = plan.get("architecture_plans")
    if not isinstance(architecture_plans, Mapping):
        raise ValueError("nnFoundation adaptation plan architecture_plans is missing")
    if architecture_plans.get("arch_class_name") != _HF_ARCHITECTURE:
        raise ValueError("nnFoundation checkpoint architecture_plans.arch_class_name must be ResEncL")
    if plan.get("key_to_encoder") != "encoder.stages" \
            or plan.get("key_to_stem") != "encoder.stem":
        raise ValueError("nnFoundation adaptation plan does not declare encoder.stages/encoder.stem")
    network = payload.get("network_weights")
    if not isinstance(network, Mapping) or not network:
        raise ValueError("nnFoundation checkpoint network_weights is missing")
    selected = {
        key[len("encoder."):]: value
        for key, value in network.items()
        if isinstance(key, str) and key.startswith(("encoder.stem.", "encoder.stages."))
    }
    if not selected or any(not isinstance(value, Tensor) for value in selected.values()):
        raise ValueError("nnFoundation encoder weights are missing or contain non-tensor values")
    return selected


def load_nnfoundation_encoder(
    path: str | Path,
    sha256: str,
    *,
    strides: Sequence[Sequence[int]] | None = None,
    input_channels: int = 1,
) -> tuple[ResidualEncoder, dict[str, Any]]:
    """Load only the strict ResEnc-L encoder from a safe HF checkpoint.

    ``decoder.*`` and segmentation-head keys are intentionally never passed
    to ``load_state_dict``.  A hash or architecture mismatch fails before any
    module is returned.
    """

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"nnFoundation checkpoint not found: {checkpoint_path}")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("nnFoundation weights_sha256 must be a 64-character hash")
    actual_sha = _sha256(checkpoint_path)
    if actual_sha.lower() != sha256.lower():
        raise ValueError("nnFoundation checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("nnFoundation checkpoint must be a mapping")
    state = _strict_encoder_state(payload)
    encoder_model = build_res_enc_l(strides, input_channels=input_channels)
    encoder = encoder_model.encoder
    expected = encoder.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(f"strict ResEnc encoder key mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}")
    for key, expected_tensor in expected.items():
        if tuple(state[key].shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"strict ResEnc encoder shape mismatch at {key}: "
                f"checkpoint={tuple(state[key].shape)} model={tuple(expected_tensor.shape)}"
            )
    load_result = encoder.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:  # pragma: no cover - defensive
        raise RuntimeError(f"strict ResEnc load reported {load_result}")
    return encoder, {
        "architecture": _HF_ARCHITECTURE,
        "weights_sha256": actual_sha,
        "checkpoint_path": str(checkpoint_path),
        # A local file can be structurally compatible without being the
        # pinned public artifact.  Never attach the HF revision to an
        # arbitrary checkpoint merely because its keys happen to match.
        "hf_revision": (
            NNFOUNDATION_HF_REVISION
            if actual_sha.lower() == NNFOUNDATION_WEIGHTS_SHA256
            else None
        ),
        "nnssl_commit": NNSSL_COMMIT,
        "loaded_prefixes": ["encoder.stem", "encoder.stages"],
        "encoder_key_count": len(state),
        "decoder_loaded": False,
        "segmentation_head_loaded": False,
        "missing_keys": [],
        "unexpected_keys": [],
    }


def _normalise_geometry(
    geometry: Mapping[str, Any],
    batch: int,
    native_shape: Sequence[int],
    device: torch.device,
) -> Tensor:
    if not isinstance(geometry, Mapping):
        raise ValueError("Q3D ResEnc teacher requires explicit geometry")
    declared_shape = geometry.get("shape")
    if (not isinstance(declared_shape, Sequence)
            or tuple(int(value) for value in declared_shape) != tuple(int(value) for value in native_shape)):
        raise ValueError("geometry.shape must equal the native input shape")
    if geometry.get("context") != "full_volume":
        raise ValueError("Q3D ResEnc teacher requires geometry.context=full_volume")
    affine = torch.as_tensor(geometry.get("affine_zyx"), dtype=torch.float64)
    if affine.shape != (4, 4) or not torch.isfinite(affine).all():
        raise ValueError("geometry.affine_zyx must be a finite 4x4 matrix")
    spacing = torch.as_tensor(geometry.get("spacing_zyx"), dtype=torch.float32, device=device)
    if spacing.ndim == 1:
        if spacing.numel() != 3:
            raise ValueError("geometry.spacing_zyx must contain z,y,x")
        spacing = spacing[None].expand(batch, -1)
    if spacing.shape != (batch, 3) or not torch.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("geometry.spacing_zyx must be finite positive [B,3]")
    return spacing


class Q3DResEncTeacher(nn.Module):
    """Dual-head Q-3D teacher using a pretrained ResEnc-L encoder."""

    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        strides: Sequence[Sequence[int]] | None = None,
        phase_offsets: Mapping[str, Sequence[float]] | None = None,
        input_channels: int = 1,
        num_classes: int = 4,
        boundary_width: int | None = None,
        synthetic: bool = False,
    ) -> None:
        super().__init__()
        if num_classes != 4:
            raise ValueError("Q3DResEncTeacher requires BG/RV/MYO/LV")
        if encoder is None:
            if not synthetic:
                raise ValueError(
                    "real Q3DResEncTeacher requires an explicitly loaded encoder; "
                    "use load_nnfoundation_encoder first"
                )
            encoder = build_res_enc_l(
                strides,
                input_channels=input_channels,
                features_per_stage=(4, 8, 12, 16, 20, 20),
                n_blocks_per_stage=(1, 1, 1, 1, 1, 1),
            )
        if isinstance(encoder, ResidualEncoderUNet):
            encoder = encoder.encoder
        if not isinstance(encoder, nn.Module) or not hasattr(encoder, "output_channels"):
            raise TypeError("Q3DResEncTeacher encoder must be a ResidualEncoder-like module")
        self.encoder = encoder
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.encoder_channels = tuple(int(value) for value in encoder.output_channels)
        if len(self.encoder_channels) != 6:
            raise ValueError("Q3D ResEnc teacher requires six feature stages")
        self.encoder_strides = tuple(tuple(int(axis) for axis in stride) for stride in encoder.strides)
        self.cumulative_stride = cumulative_strides(self.encoder_strides)
        self.padding_multiple = self.cumulative_stride
        self.synthetic = bool(synthetic)
        self.class_mapping = CLASS_NAMES
        self.phase_offsets = _phase_offset_map(phase_offsets)
        self.region_decoder = UNetDecoder(
            encoder,
            num_classes=num_classes,
            n_conv_per_stage=[1] * 5,
            deep_supervision=False,
            conv_bias=True,
        )
        width = int(boundary_width or self.encoder_channels[0])
        self.boundary_path = _BoundaryPath3D(
            self.encoder_channels[0], self.encoder_channels[-1], width, num_classes,
        )
        self.feature_phases = self._derive_feature_phases()
        self.backbone_meta: dict[str, Any] = {
            "architecture": "nnfoundation_resenc_l",
            "nnssl_commit": NNSSL_COMMIT,
            "strides": [list(item) for item in self.encoder_strides],
            "cumulative_stride": list(self.cumulative_stride),
            "phase_offsets": {
                name: list(offset) for name, offset in self.phase_offsets.items()
            },
        }

    def _derive_feature_phases(self) -> dict[str, tuple[tuple[int, int, int], tuple[float, float, float]]]:
        product = [1, 1, 1]
        phases: dict[str, tuple[tuple[int, int, int], tuple[float, float, float]]] = {}
        for index, stride in enumerate(self.encoder_strides):
            for axis in range(3):
                product[axis] *= stride[axis]
            if index == 3:
                phases["encoder_stride8"] = (
                    tuple(product), self.phase_offsets.get(
                        "encoder_stride8", (0.0, 0.0, 0.0)
                    ),
                )
            if index == 4:
                phases["encoder_stride16"] = (
                    tuple(product), self.phase_offsets.get(
                        "encoder_stride16", (0.0, 0.0, 0.0)
                    ),
                )
        return phases

    def phase_for_source(self, source: str) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
        canonical = {"stride8": "encoder_stride8", "stride16": "encoder_stride16"}.get(source, source)
        if canonical not in self.feature_phases:
            raise ValueError("ResEnc teacher source must be encoder_stride8 or encoder_stride16")
        return self.feature_phases[canonical]

    def _pad_high(self, x: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        shape = tuple(int(value) for value in x.shape[-3:])
        pads: list[int] = []
        for length, multiple in zip(shape, self.padding_multiple):
            # High-side padding is only the smallest multiple of the declared
            # cumulative stride.  There is deliberately no hidden minimum
            # (such as two deepest cells); thin synthetic fixtures must choose
            # an explicit anisotropic stride configuration that is valid for
            # their geometry instead of changing this native-volume rule.
            remainder = length % multiple
            pads.append(0 if remainder == 0 else multiple - remainder)
        # F.pad takes W,H,D low/high pairs.  All low-side values are zero.
        padded = F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0]), mode="replicate")
        return padded, tuple(pads)

    @staticmethod
    def _crop(value: Tensor, native_shape: Sequence[int]) -> Tensor:
        return value[..., : int(native_shape[0]), : int(native_shape[1]), : int(native_shape[2])]

    def forward(
        self,
        x: Tensor,
        *,
        geometry: Mapping[str, Any],
        return_features: bool = False,
    ) -> dict[str, Tensor]:
        if x.ndim != 5 or x.shape[1] != self.input_channels:
            raise ValueError("Q3D ResEnc teacher input must be [B,1,D,H,W]")
        native_shape = tuple(int(value) for value in x.shape[-3:])
        if any(int(value) <= 0 for value in native_shape):
            raise ValueError("Q3D ResEnc teacher input dimensions must be positive")
        spacing = _normalise_geometry(geometry, x.shape[0], native_shape, x.device)
        mean = x.float().mean(dim=(1, 2, 3, 4), keepdim=True)
        std = x.float().std(dim=(1, 2, 3, 4), unbiased=False, keepdim=True).clamp_min(1e-6)
        normalised = (x.float() - mean) / std
        padded, _pads = self._pad_high(normalised)
        skips = self.encoder(padded)
        if not isinstance(skips, (tuple, list)) or len(skips) != 6:
            raise RuntimeError("ResEnc encoder must return six skip tensors")
        region = self.region_decoder(skips)
        if isinstance(region, (tuple, list)):
            region = region[0]
        boundary, sdf = self.boundary_path(skips[0], skips[-1], spacing)
        outputs: dict[str, Tensor] = {
            "region_logits": self._crop(region, native_shape),
            "boundary_logits": self._crop(boundary, native_shape),
            "sdf": self._crop(sdf, native_shape),
        }
        if return_features:
            # These are encoder-grid tensors, not native-volume tensors.  The
            # high-side pad is represented by their declared phase/shape and
            # is handled by alignment; cropping them with native D/H/W would
            # silently change the feature lattice for anisotropic volumes.
            outputs["feature_encoder_stride8"] = skips[3]
            outputs["feature_encoder_stride16"] = skips[4]
        return outputs

    def forward_with_features(self, x: Tensor, *, geometry: Mapping[str, Any]) -> dict[str, Tensor]:
        return self.forward(x, geometry=geometry, return_features=True)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Reject divergent values for the decoder's shared encoder alias.

        ``UNetDecoder`` keeps a registered reference to ``self.encoder`` so
        PyTorch emits both ``encoder.*`` and ``region_decoder.encoder.*`` in a
        state dict.  They are the same module, not two independently trainable
        trunks.  Checking both copies before loading prevents traversal order
        from silently choosing one if a hand-edited checkpoint diverges.
        """

        if isinstance(state_dict, Mapping):
            prefix = "encoder."
            for key, value in state_dict.items():
                if not isinstance(key, str) or not key.startswith(prefix):
                    continue
                alias = f"region_decoder.{key}"
                if alias not in state_dict:
                    continue
                alias_value = state_dict[alias]
                if isinstance(value, Tensor) and isinstance(alias_value, Tensor) \
                        and not torch.equal(value, alias_value):
                    raise ValueError(
                        f"Q3D ResEnc checkpoint has divergent shared encoder aliases: {key} / {alias}"
                    )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


__all__ = [
    "NNFOUNDATION_HF_REVISION",
    "NNFOUNDATION_WEIGHTS_SHA256",
    "NNSSL_COMMIT",
    "NNSSL_SOURCE_URL",
    "RESENC_L_BLOCKS",
    "RESENC_L_DEFAULT_STRIDES",
    "RESENC_L_FEATURES",
    "Q3DResEncTeacher",
    "build_res_enc_l",
    "build_resenc_l",
    "cumulative_strides",
    "load_nnfoundation_encoder",
]
