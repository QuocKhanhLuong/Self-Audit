#!/usr/bin/env python3
"""Run a serialized candidate-A student on one 2.5-D image context.

The input is a single ``[3,H,W]`` normalized context.  This command never reads a
label, teacher checkpoint, or training manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from self_audit.candidate_a.evaluation import _load_student, _serialize, _student_infer
from self_audit.candidate_a.k1 import K1Config
from self_audit.model import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Candidate-A student-only inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--image", required=True, help="normalized .npy array with shape [3,H,W]")
    parser.add_argument("--spacing-yx", required=True, nargs=2, type=float, metavar=("DY", "DX"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    image_path = Path(args.image).resolve()
    image = np.load(image_path, allow_pickle=False)
    if image.ndim != 3 or image.shape[0] != 3 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("--image must be finite normalized [3,H,W]")
    device = torch.device(args.device)
    model, bank, codec, metadata = _load_student(args.checkpoint, args.checkpoint_sha256, args.device)
    rollout_config = K1Config(**metadata["config"]["rollout"])
    x = torch.from_numpy(np.asarray(image, dtype=np.float32))[None].to(device)
    with torch.inference_mode():
        output, code, trace = _student_infer(model, bank, codec, x, tuple(args.spacing_yx), rollout_config)
    blob = _serialize(codec, code, tuple(args.spacing_yx), tuple(image.shape[-2:]),
                      budget_bits=rollout_config.bit_budget)
    packet = codec.deserialize(blob, budget_bits=rollout_config.bit_budget)
    dequantized = codec.dequantize(packet, device=output.device, dtype=output.dtype)
    with torch.inference_mode():
        decoded = model.decode(packet.ids, dequantized.unsqueeze(0),
                               {"shape": packet.shape, "spacing_yx": packet.spacing},
                               chunk_size=model.config.chunk_size)
    if not torch.allclose(decoded, output, atol=1e-5, rtol=1e-5):
        raise RuntimeError("serialized/dequantized code does not reproduce decoder output")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "probabilities.npy", output[0].softmax(0).cpu().numpy().astype(np.float32))
    (out / "code.bin").write_bytes(blob)
    record = {"checkpoint_sha256": args.checkpoint_sha256, "image_sha256": digest(image_path),
              "spacing_yx": list(map(float, args.spacing_yx)), "shape": list(image.shape),
              "bytes": len(blob), "query_count": trace.get("query_count", trace.get("num_queries"))
              if isinstance(trace, dict) else None, "prediction_model": "student_code_only"}
    (out / "METADATA.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
