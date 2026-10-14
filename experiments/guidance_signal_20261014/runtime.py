"""Hash-pinned imports from the executed snapshot, never production ``src``."""
from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

SNAPSHOT_SHA = "b937e3b746608725e43d3bc3cef8861666d152dc"
REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = REPO_ROOT / "reports/stable_guidance_world_v2_20261014"
SOURCE_ROOT = SNAPSHOT_ROOT / "executed_source/src/self_audit_guidance"
MANIFEST_SHA256 = "d7a891fd887a44aebd26e65288b13c0c313b896a69ca244b42a22f13b1e3a41e"
EXPECTED_SOURCE_SHA256 = {
    "__init__.py": "1c840d42867f0e07dfc13c656c82ad85b0b9abe36fb0870fc87b338df5609e38",
    "core.py": "08a0e3ed1987e446d0367c6b15b102c95ffbc083cf6077c7f1985e6adb346c63",
    "evaluate.py": "44cfff418bc1510d61c4f3b9b6296d1ece9421b14739398d3900d2d7028315d6",
    "experimental_diagnostics.py": "2aaa412959199105c1ce572e10df5b58d70f247838d113bd28a8ae9146977bc4",
    "experimental_geometry.py": "93355bcffabd5471950bbef62dbd94aedf13c2d3989aecd594bdbce373df056b",
    "full.py": "8f9036fb717c4edab731649a4ccb861ef40784b2d0f5d3c78c7651af3cccdf91",
    "io.py": "42c73e38a59463ec409fa1915d425f1e549f510be68d5c2c6fbd135eb5204b5d",
    "pipeline.py": "0f23f07bb3f77fd6c7724b7170a75c543b211fa36ec1c704fd84f5bcf23e2029",
    "preflight.py": "bf81ecacbb194e5ae6f3bbc6a5386e4703f9e95fc3618cf2329518234430584b",
    "student.py": "d216bdc7ca8e9b01b368457aed2b91065e8124a2d3be2a9875d1f6284b3387fa",
    "teacher.py": "8a7b88548223c30ef570f90d977c71a14f2ca6ec219bf216e0f16efafdada273",
}
PRIVATE_PACKAGE = "_guidance_executed_b937e3b746608725"
TEACHER_SHA256 = "a841b7e9300d60ea8feb73e3a7d8c60ddedc1d1d49ec1049ebfcfa337c5d5590"
TEACHER_CONFIG_SHA256 = "3147bbd6d0350f9fe30d14989db0f6d53ecef2ed32900d843a3fc3155a3e0a32"
TEACHER_REVISION = "e0466ed5bd7687a32121434875bf2168aed41c03"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source():
    """Check every module and its published receipt BEFORE any snapshot import."""
    manifest = SNAPSHOT_ROOT / "MANIFEST.json"
    if sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("executed-source publication manifest hash mismatch")
    entries = {e["published_path"]: e for e in json.loads(manifest.read_text())["files"]}
    if {p.name for p in SOURCE_ROOT.glob("*.py")} != set(EXPECTED_SOURCE_SHA256):
        raise ValueError("executed-source module inventory mismatch")
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        path = SOURCE_ROOT / name
        receipt = entries.get("executed_source/src/self_audit_guidance/" + name, {})
        if (path.is_symlink() or sha256(path) != expected
                or receipt.get("original_sha256") != expected
                or receipt.get("published_sha256") != expected
                or receipt.get("sanitized") is not False):
            raise ValueError("executed-source hash mismatch: " + name)
    return dict(EXPECTED_SOURCE_SHA256)


class _VerifiedLoader(importlib.abc.Loader):
    def __init__(self, path, package=False):
        self.path, self.package = path, package

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Compile verified bytes directly; an unverified pre-existing pyc must
        # not bypass the source-byte pins through Python's timestamp cache.
        data = self.path.read_bytes()
        if hashlib.sha256(data).hexdigest() != EXPECTED_SOURCE_SHA256[self.path.name]:
            raise ImportError("executed source changed during import")
        module.__file__ = str(self.path)
        if self.package:
            module.__path__ = [str(SOURCE_ROOT)]
        exec(compile(data, str(self.path), "exec"), module.__dict__)


class _VerifiedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == PRIVATE_PACKAGE:
            name, package = "__init__.py", True
        elif fullname.startswith(PRIVATE_PACKAGE + "."):
            name, package = fullname[len(PRIVATE_PACKAGE)+1:] + ".py", False
            if name not in EXPECTED_SOURCE_SHA256:
                raise ImportError("unlisted snapshot module refused: " + name)
        else:
            return None
        return importlib.util.spec_from_loader(fullname, _VerifiedLoader(SOURCE_ROOT/name, package),
                                               is_package=package)


def module(name):
    if name + ".py" not in EXPECTED_SOURCE_SHA256 or name == "__init__":
        raise ValueError("unknown snapshot module")
    verify_source()
    if not any(isinstance(f, _VerifiedFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _VerifiedFinder())
    result = importlib.import_module("." + name, PRIVATE_PACKAGE)
    if Path(result.__file__).resolve() != (SOURCE_ROOT / (name + ".py")).resolve():
        raise ImportError("private snapshot module identity mismatch: " + name)
    return result


class _Runtime(SimpleNamespace):
    def __getattr__(self, name):
        if name == "evaluate":
            value = module(name)
            setattr(self, name, value)
            return value
        raise AttributeError(name)


def load_runtime():
    """Return private snapshot modules; importing does not instantiate a teacher."""
    hashes = verify_source()
    modules = {}
    for name in ("core", "io", "student", "teacher", "pipeline", "experimental_geometry"):
        modules[name] = module(name)
    return _Runtime(**modules, geometry=modules["experimental_geometry"],
                    diagnostics=module("experimental_diagnostics"),
                    source_sha256=hashes, source_hashes=hashes, snapshot_sha=SNAPSHOT_SHA,
                    SNAPSHOT_SHA=SNAPSHOT_SHA, module=module)
