"""Development-only protocol index/receipt writer; execution never auto-refreezes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


if __name__ == "__main__":
    for method in ("DSS_US", "SGSCN"):
        base = ROOT / "baseline" / method / "config/native"
        index = {}
        for path in sorted(base.glob("*.yaml")):
            index[path.name] = hashlib.sha256(canonical(json.loads(path.read_text()))).hexdigest()
        (base / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    base = ROOT / "baseline/SGSCN"
    names = ("upstream/demo_final.py", "upstream/src/center.py", "upstream/README.md", "LICENSE")
    receipt = {"upstream_commit": "592efb6e72ceeef15c8be0630a4673eda5dce6f5",
               "files": {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in names}}
    (base / "upstream/receipt.json").write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
