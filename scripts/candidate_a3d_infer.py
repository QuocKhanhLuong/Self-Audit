#!/usr/bin/env python3
"""Run a serialized Candidate-A 3-D student on one native full volume.

The input is a normalized ``[1,D,H,W]`` cache frame.  This command never
opens a label, teacher checkpoint, or training manifest.  Geometry is required
explicitly: use a complete runtime geometry JSON object, or provide both
``--spacing-zyx`` and ``--affine``.  No identity affine or unit-spacing
fallback is permitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from self_audit.candidate_a3d.evaluation import _serialize_student, _student_rollout
from self_audit.candidate_a3d.geometry import (
    VolumeGeometry3D,
    require_full_volume_shape,
)
from self_audit.candidate_a3d.io import load_student
from self_audit.candidate_a3d.k1 import K1Config
from self_audit.candidate_a3d.profile import VolumeProfile, validate_device
from self_audit.model import digest


def _read_affine(path: str | Path) -> list[list[float]]:
    """Read a declared 4x4 affine from ``.npy`` or JSON without inference."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"affine file does not exist: {source}")
    if source.suffix.lower() == ".npy":
        value = np.load(source, allow_pickle=False)
    else:
        value = json.loads(source.read_text())
        if isinstance(value, dict):
            value = value.get("affine_zyx")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError("--affine must contain a finite 4x4 matrix")
    return array.tolist()


def _geometry(args: argparse.Namespace, shape: tuple[int, int, int]) -> dict:
    if args.geometry_json and (args.spacing_zyx is not None or args.affine is not None):
        raise ValueError("use either --geometry-json or --spacing-zyx/--affine, not both")
    if args.geometry_json:
        geometry_path = Path(args.geometry_json).resolve()
        if not geometry_path.is_file():
            raise ValueError(f"geometry JSON does not exist: {geometry_path}")
        value = json.loads(geometry_path.read_text())
        if not isinstance(value, dict):
            raise ValueError("--geometry-json must contain one runtime geometry object")
    else:
        if args.spacing_zyx is None or args.affine is None:
            raise ValueError("geometry is required: provide --geometry-json or both --spacing-zyx and --affine")
        value = {
            "shape": shape,
            "spacing_zyx": tuple(float(item) for item in args.spacing_zyx),
            "affine_zyx": _read_affine(args.affine),
            "context": "full_volume",
        }
    if tuple(value.get("shape", ())) != shape:
        raise ValueError("declared geometry shape does not match input [D,H,W]")
    geometry = VolumeGeometry3D.from_mapping(value)
    require_full_volume_shape(geometry, shape)
    return geometry.as_dict()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--image", required=True, help="normalized .npy array with shape [1,D,H,W]")
    parser.add_argument("--geometry-json", help="runtime geometry object with shape/spacing/affine/context")
    parser.add_argument("--spacing-zyx", nargs=3, type=float, metavar=("DZ", "DY", "DX"))
    parser.add_argument("--affine", "--affine-zyx", dest="affine",
                        help=".npy or JSON file containing the explicit 4x4 affine")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    image_path = Path(args.image).resolve()
    image = np.load(image_path, allow_pickle=False)
    if (image.ndim != 4 or image.shape[0] != 1 or not np.isfinite(image).all()
            or image.min() < 0 or image.max() > 1):
        raise ValueError("--image must be finite normalized [1,D,H,W]")
    shape = tuple(int(value) for value in image.shape[1:])
    geometry = _geometry(args, shape)
    device = validate_device(args.device)
    model, bank, codec, metadata = load_student(args.checkpoint, args.checkpoint_sha256, args.device)
    try:
        rollout_config = K1Config(**metadata["config"]["rollout"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("student checkpoint is missing the exact A-3D K1 rollout config") from exc

    x = torch.from_numpy(np.asarray(image[0], dtype=np.float32))[None, None].to(device)
    profile = VolumeProfile()
    with torch.inference_mode():
        with profile.measure("rollout_ms", device):
            rollout = _student_rollout(model, x, bank, codec, geometry, rollout_config)
        output = rollout.final_logits
        expected = (1, 4, *shape)
        if output.shape != expected or not torch.isfinite(output).all():
            raise ValueError("A-3D student decoder output shape/value mismatch")
        code = {"ids": tuple(rollout.code_ids), "values": rollout.code_values}
        with profile.measure("packet_serialize_ms", device):
            blob = _serialize_student(codec, code, geometry, rollout_config.bit_budget)
        with profile.measure("packet_roundtrip_ms", device):
            packet = codec.deserialize(blob, budget_bits=rollout_config.bit_budget)
            values = codec.dequantize(packet, device=output.device, dtype=output.dtype)
        with profile.measure("decoder_roundtrip_ms", device):
            decoded = model.decode(
                packet.ids, values.unsqueeze(0),
                {"shape": packet.shape, "spacing_zyx": packet.spacing_zyx,
                 "affine_zyx": packet.affine_zyx, "context": "full_volume"},
            )
        if not torch.allclose(decoded, output, atol=1e-5, rtol=1e-5):
            raise RuntimeError("serialized/dequantized code does not reproduce decoder output")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    probability_path = out / "probabilities.npy"
    np.save(probability_path, output[0].softmax(0).cpu().numpy().astype(np.float32))
    packet_path = out / "code.bin"
    packet_path.write_bytes(blob)
    profile.fields["total_ms"] = float(sum(profile.fields.values()))
    metadata_out = {
        "checkpoint_sha256": args.checkpoint_sha256,
        "image_sha256": digest(image_path),
        "geometry": {
            "shape": list(geometry["shape"]),
            "spacing_zyx": list(geometry["spacing_zyx"]),
            "affine_zyx": np.asarray(geometry["affine_zyx"], dtype=float).tolist(),
            "context": geometry["context"],
        },
        "bytes": int(packet_path.stat().st_size),
        "packet_sha256": digest(packet_path),
        "query_count": int(rollout.query_count),
        "profile_ms": profile.fields,
        "timing_scope": "rollout_ms=encoder+query_value+policy+full_volume_decoder; packet roundtrip and decoder roundtrip are separate",
        "prediction_model": "student_code_only",
        "labels_used": False,
    }
    (out / "METADATA.json").write_text(json.dumps(metadata_out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metadata_out, indent=2))


if __name__ == "__main__":
    main()
