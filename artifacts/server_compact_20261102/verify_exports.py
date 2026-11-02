"""Verify published bytes and reload real weights on synthetic CPU inputs only."""
from pathlib import Path
import gzip
import hashlib
import json
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from self_audit.compact_artifacts import digest, load_compact_export
from self_audit.compact_model import build_compact_model


def main():
    torch.set_num_threads(2)
    publication = json.loads((ROOT / "PUBLICATION.json").read_text())
    verified_files = 0
    for line in (ROOT / "SERVER_SHA256SUMS.txt").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT)
        assert digest(path) == expected, relative
        verified_files += 1
    result = {
        "status": "PASS",
        "scope": "Physical weights and CPU fixture reload parity; no new patient evaluation",
        "torch_version": torch.__version__,
        "server_snapshot_files_verified": verified_files,
        "arms": {},
    }
    generator = torch.Generator().manual_seed(42)
    inputs = torch.rand((2, 3, 37, 45), generator=generator)
    for arm, expected in publication["arms"].items():
        folder = ROOT / arm
        best = folder / "best.pt"
        assert best.stat().st_size == expected["best_pt_bytes"]
        assert digest(best) == expected["best_pt_sha256"]
        with best.open("rb") as stream:
            assert not stream.read(128).startswith(b"version https://git-lfs.github.com/spec")
        model, metadata = load_compact_export(
            folder / "export", expected["export_manifest_sha256"], device="cpu"
        )
        assert digest(best) == metadata["payload_sha256"]
        direct = build_compact_model(metadata["architecture"]).eval()
        direct.load_state_dict(torch.load(best, map_location="cpu", weights_only=True)["student"], strict=True)
        with torch.inference_mode():
            output = model(inputs)
            direct_output = direct(inputs)
        assert tuple(output.shape) == (2, 4, 37, 45)
        assert torch.isfinite(output).all()
        assert torch.equal(output, direct_output)
        assert torch.equal(output.argmax(1), direct_output.argmax(1))
        for log in expected["training_logs"]:
            h = hashlib.sha256()
            with gzip.open(ROOT / log["archive"], "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            assert h.hexdigest() == log["sha256"]
        result["arms"][arm] = {
            "best_pt_sha256": digest(best),
            "export_manifest_sha256": metadata["manifest_sha256"],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "prediction_parity": "EXACT_CPU_FIXTURE",
            "external_training_assets_required": False,
            "training_logs_round_trip": "PASS",
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
