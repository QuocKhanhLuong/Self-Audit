"""S1 SSL out-of-domain cardiac benchmark shell for STEGO.

STEGO uses a DINO ViT-Base backbone pretrained on ImageNet (self-supervised)
with STEGO contrastive heads pretrained on Cityscapes (unsupervised).
The anonymous partition is produced without cardiac semantic labels.
"""

from .manifest import MANIFEST_SCHEMA_VERSION, load_manifest, require_scientific_manifest

__all__ = ["MANIFEST_SCHEMA_VERSION", "load_manifest", "require_scientific_manifest"]
