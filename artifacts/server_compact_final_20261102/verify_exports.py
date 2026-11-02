"""Verify exact transferred bytes and strict real-weight reload on CPU fixtures."""
from pathlib import Path
import hashlib
import json
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from self_audit.compact_artifacts import load_compact_export
from self_audit.compact_model import build_compact_model


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inventory = json.loads((ROOT / 'TRANSFER_MANIFEST.json').read_text())
    for relative, entry in inventory.items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT)
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], relative
    torch.set_num_threads(2)
    inputs = torch.rand((2, 3, 37, 45), generator=torch.Generator().manual_seed(42))
    receipt = {'status': 'PASS', 'scope': 'Strict real student export reload and synthetic CPU parity; no new patient/GPU evaluation', 'torch_version': torch.__version__, 'transferred_files_verified': len(inventory), 'arms': {}}
    for arm in ('S0', 'S1', 'S2'):
        folder = ROOT / arm
        best = json.loads((folder / 'BEST.json').read_text())
        model, meta = load_compact_export(folder / 'best_export', best['export_manifest_sha256'], device='cpu')
        payload = folder / 'best.pt'
        assert sha(payload) == best['best_pt_sha256'] == meta['payload_sha256']
        assert payload.stat().st_size == meta['serialization_bytes']
        assert not payload.read_bytes()[:128].startswith(b'version https://git-lfs.github.com/spec')
        state = torch.load(payload, map_location='cpu', weights_only=True)
        assert set(state) == {'student'}
        direct = build_compact_model(meta['architecture']).eval()
        direct.load_state_dict(state['student'], strict=True)
        with torch.inference_mode():
            output, direct_output = model(inputs), direct(inputs)
        assert tuple(output.shape) == (2, 4, 37, 45)
        assert torch.isfinite(output).all() and torch.equal(output, direct_output)
        receipt['arms'][arm] = {'best_pt_sha256': sha(payload), 'best_pt_bytes': payload.stat().st_size, 'export_manifest_sha256': meta['manifest_sha256'], 'parameter_count': sum(p.numel() for p in model.parameters()), 'prediction_parity': 'EXACT_CPU_FIXTURE', 'external_teacher_required': meta['external_teacher_required']}
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
