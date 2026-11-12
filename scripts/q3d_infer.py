"""Run one fail-closed Q-3D student artifact on a native-volume NPZ input."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.candidate_q3d.evaluation import validate_native_image  # noqa: E402
from self_audit.candidate_q3d.export import load_export  # noqa: E402


def _load_input(path: str | Path) -> tuple[torch.Tensor, dict]:
    source = Path(path).resolve()
    with np.load(source, allow_pickle=False) as data:
        required = {"image", "shape_zyx", "spacing_zyx", "affine_zyx", "context"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"external Q-3D input is missing fields: {sorted(missing)}")
        context = str(np.asarray(data["context"]).item())
        if context != "full_volume":
            raise ValueError("external Q-3D inference accepts only context=full_volume")
        shape = tuple(int(value) for value in np.asarray(data["shape_zyx"]).tolist())
        geometry = {
            "shape": shape,
            "spacing_zyx": np.asarray(data["spacing_zyx"], dtype=np.float64).tolist(),
            "affine_zyx": np.asarray(data["affine_zyx"], dtype=np.float64).tolist(),
            "context": context,
        }
        image = np.array(data["image"], copy=True)
    tensor, geometry = validate_native_image(image, geometry)
    if tuple(tensor.shape[-3:]) != shape:
        raise ValueError("external Q-3D image shape does not match shape_zyx")
    return tensor, geometry


def infer(artifact_path: str | Path, input_path: str | Path, output_path: str | Path, *,
          device: str = "cpu", expected_artifact_sha256: str | None = None,
          exploratory_quality_waiver: str | None = None) -> dict:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no device fallback is allowed")
    model, metadata = load_export(
        artifact_path, expected_sha256=expected_artifact_sha256, device=device,
        **({"exploratory_quality_waiver": exploratory_quality_waiver}
           if exploratory_quality_waiver is not None else {}),
    )
    image, geometry = _load_input(input_path)
    with torch.inference_mode():
        logits = model.infer_logits(image.to(device), geometry=geometry)
    expected = (1, 4, *geometry["shape"])
    if tuple(logits.shape) != expected or not torch.isfinite(logits).all():
        raise ValueError(f"Q-3D deploy logits must be finite with shape {expected}")
    logits_np = logits[0].detach().float().cpu().numpy()
    mask = logits_np.argmax(axis=0).astype(np.uint8)
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    exploratory = {key: metadata[key] for key in (
        "exploratory", "exploratory_quality_waiver", "measured_quality_gates",
    ) if key in metadata}
    with target.open("wb") as stream:
        np.savez_compressed(
            stream,
            logits=logits_np,
            mask=mask,
            shape_zyx=np.asarray(geometry["shape"], dtype=np.int32),
            spacing_zyx=np.asarray(geometry["spacing_zyx"], dtype=np.float64),
            affine_zyx=np.asarray(geometry["affine_zyx"], dtype=np.float64),
            context=np.asarray(geometry["context"]),
            class_mapping=np.asarray(json.dumps({"0": "BG", "1": "RV", "2": "MYO", "3": "LV"})),
            **({"exploratory_metadata": np.asarray(json.dumps(exploratory, allow_nan=False))}
               if exploratory else {}),
        )
    output_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"output": str(target), "sha256": output_sha, "shape": list(mask.shape),
            "artifact_sha256": metadata["artifact_sha256"], "device": device,
            **exploratory,
            "evidence_status": "INFERENCE_ONLY_NO_LABELS_READ"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--input", required=True, help="NPZ with image, shape_zyx, spacing_zyx, affine_zyx, context")
    parser.add_argument("--output", required=True, help="output NPZ path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-sha256")
    parser.add_argument("--q3d-exploratory-quality-waiver", metavar="REASON",
                        help="exact explicit reason recorded in an exploratory artifact")
    args = parser.parse_args(argv)
    result = infer(args.artifact, args.input, args.output, device=args.device,
                   expected_artifact_sha256=args.artifact_sha256,
                   **({"exploratory_quality_waiver": args.q3d_exploratory_quality_waiver}
                      if args.q3d_exploratory_quality_waiver is not None else {}))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
