"""Optional local licensed-architecture check, never a CAMUS/pretrained result."""
from pathlib import Path
import sys

import pytest
import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src")]
from dss_us.features import DinoKeys
from shared_benchmark.native_protocol import file_hash


def test_pinned_licensed_architecture_checkpoint_and_key_hook(tmp_path):
    repository = ROOT / ".scratch/dino"
    if not (repository / "hubconf.py").exists():
        pytest.skip("run scripts/pin_native_references.py for licensed backbone check")
    pytest.importorskip("torchvision")
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model = torch.hub.load(str(repository), "dino_vits8", source="local", pretrained=False)
    checkpoint = tmp_path / "synthetic_checkpoint.pth"
    torch.save(model.state_dict(), checkpoint)
    provider = DinoKeys(repository, checkpoint, checkpoint_sha256=file_hash(checkpoint), device="cpu")
    keys, grid = provider.keys(torch.zeros(1, 3, 32, 40), crop_to_patch_multiple=False)
    assert grid == (4, 5) and keys.shape == (1, 20, 384)
    with pytest.raises(ValueError, match="checkpoint hash"):
        DinoKeys(repository, checkpoint, checkpoint_sha256="0" * 64, device="cpu")
