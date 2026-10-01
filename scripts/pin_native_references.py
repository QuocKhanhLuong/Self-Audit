"""Fetch exact local references; never execute unlicensed DSS-US source."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGET_NAMES = {"dss_us": "dss-us", "sgscn": "sgscn", "dino": "dino"}


def _git(target, *args):
    subprocess.run(["git", "-C", str(target), *args], check=True)


def _git_output(target, *args):
    return subprocess.check_output(["git", "-C", str(target), *args], text=True).strip()


def _never_checked_out(target: Path) -> bool:
    """A --no-checkout clone has an empty index and no work-tree files besides .git.

    Its `git status` lists every tracked file as deleted, which is not a user edit.
    Anything else in the work tree is real content that must not be overwritten.
    """
    return not _git_output(target, "ls-files") and {p.name for p in target.iterdir()} == {".git"}


def pin_reference(target: Path, url: str, commit: str) -> str:
    target = Path(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", url, str(target)], check=True)
    remote = _git_output(target, "remote", "get-url", "origin")
    if remote.rstrip("/").removesuffix(".git") != url.rstrip("/").removesuffix(".git"):
        raise ValueError(f"unexpected origin for {target}")
    # Runtime Python bytecode is not a scientific source edit. Do not modify
    # tracked upstream ignore files or exclude arbitrary untracked source.
    exclude = target / ".git/info/exclude"
    text = exclude.read_text() if exclude.exists() else ""
    if "__pycache__/" not in text:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text(text + "\n__pycache__/\n")
    # The empty index of a never-populated --no-checkout clone is not dirty; edits are.
    if not _never_checked_out(target) and _git_output(target, "status", "--porcelain"):
        raise ValueError(f"refuse to change dirty reference: {target}")
    exists = subprocess.run(["git", "-C", str(target), "cat-file", "-e", commit + "^{commit}"], capture_output=True)
    if exists.returncode:
        _git(target, "fetch", "origin", commit)
    _git(target, "checkout", "--detach", commit)
    actual = _git_output(target, "rev-parse", "HEAD")
    if actual != commit:
        raise ValueError(f"pin mismatch: {target}")
    if _git_output(target, "status", "--porcelain"):
        raise ValueError(f"pinned reference is not clean after checkout: {target}")
    return actual


def pin_references(destination: Path) -> dict:
    pins = json.loads((ROOT / "configs/native_baselines/references.json").read_text())
    for name, pin in pins.items():
        pin_reference(destination / TARGET_NAMES[name], pin["url"], pin["commit"])
    return pins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / ".scratch")
    args = parser.parse_args()
    print(json.dumps(pin_references(args.destination), indent=2))
