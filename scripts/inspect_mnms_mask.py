"""Print the raw-to-ACDC label mapping used for one M&Ms mask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from self_audit.data.common import load_array
from self_audit.data.mnms import DEFAULT_MNMS_TO_ACDC, MNMSClassMapping


def _mapping(value: str | None) -> MNMSClassMapping:
    if value is None:
        return MNMSClassMapping(DEFAULT_MNMS_TO_ACDC)
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8") if path.exists() else value)
    if not isinstance(payload, dict):
        raise ValueError("mapping must be a JSON object or a path to one")
    return MNMSClassMapping({int(key): int(target) for key, target in payload.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", required=True, help="Path to one .npy/.npz/.nii/.nii.gz mask")
    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional JSON object or JSON file overriding the default raw-to-ACDC mapping",
    )
    args = parser.parse_args()
    raw_mask, _ = load_array(args.mask)
    mapping = _mapping(args.mapping)
    mapped = mapping.apply(np.asarray(raw_mask))
    print(f"unique raw labels: {sorted(int(value) for value in np.unique(raw_mask))}")
    print(f"configured mapping: {dict(mapping.raw_to_acdc)}")
    print(f"mapped labels: {sorted(int(value) for value in np.unique(mapped))}")


if __name__ == "__main__":  # pragma: no cover
    main()
