#!/usr/bin/env python3
"""Optional GPU memory smoke for the locked 2.5-D Self-Audit model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.models.self_audit_net import SelfAuditNet
from self_audit.losses.annotation import annotation_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        print("CUDA unavailable; memory smoke skipped")
        return
    device = torch.device(args.device)
    model = SelfAuditNet(pretrained_encoder=False, shared_channels=96, window_k=8, max_turns=3).to(device)
    images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    target = torch.randint(0, 4, (args.batch_size, args.image_size, args.image_size), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    output = model.forward_annotation(images)
    loss = annotation_loss(output["logits"], target)[0]
    loss.backward()
    print(f"allocated_mb={torch.cuda.memory_allocated(device) / 2**20:.1f}")
    print(f"peak_allocated_mb={torch.cuda.max_memory_allocated(device) / 2**20:.1f}")
    print(f"reserved_mb={torch.cuda.memory_reserved(device) / 2**20:.1f}")


if __name__ == "__main__":  # pragma: no cover
    main()
