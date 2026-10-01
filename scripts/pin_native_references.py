"""Fetch exact local references; never execute unlicensed DSS-US source."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def pin_references(destination: Path) -> dict:
    pins = json.loads((ROOT / "configs/native_baselines/references.json").read_text())
    for name, pin in pins.items():
        target = destination / {"dss_us": "dss-us", "sgscn": "sgscn", "dino": "dino"}[name]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", pin["url"], str(target)], check=True)
        remote = subprocess.check_output(["git", "-C", str(target), "remote", "get-url", "origin"], text=True).strip()
        if remote.rstrip("/").removesuffix(".git") != pin["url"].removesuffix(".git"):
            raise ValueError(f"unexpected origin for {name}")
        # Runtime Python bytecode is not a scientific source edit. Do not modify
        # tracked upstream ignore files or exclude arbitrary untracked source.
        exclude = target / ".git/info/exclude"
        text = exclude.read_text() if exclude.exists() else ""
        if "__pycache__/" not in text:
            exclude.write_text(text + "\n__pycache__/\n")
        dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
        if dirty:
            raise ValueError(f"refuse to change dirty reference: {target}")
        exists = subprocess.run(["git", "-C", str(target), "cat-file", "-e", pin["commit"] + "^{commit}"], capture_output=True)
        if exists.returncode:
            subprocess.run(["git", "-C", str(target), "fetch", "origin", pin["commit"]], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", pin["commit"]], check=True)
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != pin["commit"]:
            raise ValueError(f"pin mismatch: {name}")
    return pins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / ".scratch")
    args = parser.parse_args()
    print(json.dumps(pin_references(args.destination), indent=2))
