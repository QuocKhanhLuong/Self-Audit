"""DINO is a licensed backbone dependency, never the unlicensed DSS-US source."""
import subprocess
from pathlib import Path

import numpy as np
import torch

from shared_benchmark.native_protocol import file_hash

DINO_COMMIT = "7c446df5b9f45747937fb0d72314eb9f7b66930a"


class DinoKeys:
    def __init__(self, repository, checkpoint, *, checkpoint_sha256, device):
        repository = Path(repository).resolve()
        actual = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
        if actual != DINO_COMMIT:
            raise ValueError("DINO checkout is not pinned")
        dirty = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True)
        if dirty:
            raise ValueError("DINO source is dirty")
        if file_hash(Path(checkpoint)) != checkpoint_sha256:
            raise ValueError("backbone checkpoint hash mismatch")
        self.model = torch.hub.load(str(repository), "dino_vits8", source="local", pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(device)
        self.device = device
        self.receipt = {"repository_commit": actual, "checkpoint_sha256": checkpoint_sha256}

    @torch.no_grad()
    def keys(self, tensor, *, crop_to_patch_multiple):
        """Capture last-attention qkv, return keys excluding CLS."""
        height, width = tensor.shape[-2:]
        if height % 8 or width % 8:
            if not crop_to_patch_multiple:
                raise ValueError("native patch-edge policy must be explicit")
            tensor = tensor[..., :height // 8 * 8, :width // 8 * 8]
        saved = []
        hook = self.model.blocks[-1].attn.qkv.register_forward_hook(lambda module, inputs, value: saved.append(value))
        try:
            self.model.get_last_selfattention(tensor.to(self.device))
        finally:
            hook.remove()
        batch, tokens, triple_width = saved[0].shape
        heads = self.model.blocks[-1].attn.num_heads
        qkv = saved[0].reshape(batch, tokens, 3, heads, triple_width // (3 * heads))
        keys = qkv[:, 1:, 1].reshape(batch, tokens - 1, triple_width // 3)
        return keys.cpu().numpy(), (tensor.shape[-2] // 8, tensor.shape[-1] // 8)
