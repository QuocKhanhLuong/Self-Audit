#!/usr/bin/env python
"""Create a new, separately hashed native profile by filling declared required fields.

Only fields declared in ``paper_unspecified``, ``implementation_conventions``,
``conditional_values`` or ``required_data`` of the base profile may be set, only while
they are still null, and every value must carry a ``source``. Values are recorded as
USER_SUPPLIED with the base profile hash; the base profile and its index entry are never
modified. Scientific settings cannot be changed through this tool.

    python scripts/instantiate_native_profile.py --base baseline/SGSCN/config/native/ph2_paper_faithful.yaml \
        --values my_values.json --profile ph2_paper_faithful_userA
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_benchmark.native_protocol import value_hash  # noqa: E402

GROUPS = ("paper_unspecified", "implementation_conventions", "conditional_values", "required_data")


def instantiate(base_path: Path, values: dict, profile: str) -> Path:
    base_path = Path(base_path).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    index_path = base_path.parent / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get(base_path.name) != value_hash(base):
        raise ValueError("base profile differs from its frozen index")
    target = base_path.parent / f"{profile}.yaml"
    if target.exists() or target.name in index:
        raise FileExistsError(f"profile already exists: {target.name}")
    if not values:
        raise ValueError("no values supplied")
    derived = copy.deepcopy(base)
    for name, supplied in values.items():
        groups = [group for group in GROUPS if name in derived.get(group, {})]
        if len(groups) != 1:
            raise ValueError(f"not a declared required field: {name}")
        entry = derived[groups[0]][name]
        if entry.get("value") is not None:
            raise ValueError(f"field already has a value and cannot be overridden: {name}")
        if not isinstance(supplied, dict) or "value" not in supplied or not supplied.get("source"):
            raise ValueError(f"{name}: supply {{'value': ..., 'source': '...'}}")
        if supplied["value"] is None:
            raise ValueError(f"{name}: value must not be null")
        entry.update({"value": supplied["value"], "status": f"USER_SUPPLIED ({entry.get('status')})",
                      "user_source": supplied["source"]})
    derived["profile"] = profile
    derived["derived_from"] = {"profile": base["profile"], "config_sha256": value_hash(base)}
    target.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    index[target.name] = value_hash(derived)
    index_path.write_text(json.dumps(dict(sorted(index.items())), indent=2) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    print(instantiate(args.base, json.loads(args.values.read_text(encoding="utf-8")), args.profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
