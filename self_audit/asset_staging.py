"""CPU-only, metadata-first asset staging. Never imports a model or trainer."""
from __future__ import annotations

import contextlib
import configparser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone

SOURCE_COMMIT = "8dc9e7446be23e2b8533871c2a1e418b4a115da7"
PINS = {
    "teacher_manifest": "ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa",
    "teacher_weights": "0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6",
    "teacher_runtime": "3c7fa85f00d54a6e087af3cd6c7045351fb72d532f1069bf09d95f9cd72359b7",
    "student_manifest": "197dd4367877c51bc21b91d172f25442549ef9a3d0e422d28118bb07308067de",
    "student_weights": "32a35c1fc3b83d57d083de4ed17ab80bda6ba38a8ebc50133e5aede17ab00380",
    "student_runtime": "bbfbdab02cf7703904121f39720adcb1dc358b6d76b60fbbcb4366fa30244b3c",
    "train_manifest": "e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4",
    "val_manifest": "cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)


def relative_path(value):
    p = PurePosixPath(value)
    if not value or not p.parts or p.is_absolute() or ".." in p.parts or ":" in value or "\\" in value or "\n" in value:
        raise ValueError("unsafe relative asset path")
    return p.as_posix()


@contextlib.contextmanager
def rclone_config(environ=None):
    """Credentials stay in a mode-0600 /tmp file, never on the Volume."""
    env = os.environ if environ is None else environ
    text = env.get("RCLONE_CONFIG_TEXT")
    remote = env.get("RCLONE_REMOTE_NAME", "").rstrip(":")
    if not text or not re.fullmatch(r"[\w .-]+", remote):
        raise ValueError("selfaudit-rclone needs RCLONE_CONFIG_TEXT and RCLONE_REMOTE_NAME")
    parsed = configparser.ConfigParser(interpolation=None)
    try:
        parsed.read_string(text)
    except configparser.Error:
        raise ValueError("RCLONE_CONFIG_TEXT is not valid INI; config contents withheld") from None
    if remote not in parsed:
        raise ValueError("RCLONE_REMOTE_NAME has no matching section in RCLONE_CONFIG_TEXT")
    fd, name = tempfile.mkstemp(prefix="selfaudit-rclone-", suffix=".conf", dir="/tmp")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        yield name, remote
    finally:
        Path(name).unlink(missing_ok=True)


class Rclone:
    def __init__(self, config, remote):
        self.config, self.remote = config, remote

    def run(self, *args):
        # Do not propagate secret text or print stderr: OAuth errors can contain credentials.
        env = {k: v for k, v in os.environ.items() if k not in ("RCLONE_CONFIG_TEXT", "RCLONE_REMOTE_NAME")}
        p = subprocess.run(["rclone", "--config", self.config, "--retries", "2",
                            "--low-level-retries", "2", "--contimeout", "20s",
                            "--timeout", "120s", *args], capture_output=True, env=env, timeout=1800)
        if p.returncode:
            err = p.stderr.decode(errors="replace").lower()
            reason = "unclassified"
            for needle, category in (("invalid_grant", "OAUTH_INVALID_GRANT"),
                                     ("invalid_client", "OAUTH_INVALID_CLIENT"),
                                     ("unauthorized_client", "OAUTH_UNAUTHORIZED_CLIENT"),
                                     ("couldn't find root directory", "ROOT_DIRECTORY_UNAVAILABLE"),
                                     ("directory not found", "DIRECTORY_NOT_FOUND"),
                                     ("didn't find section", "CONFIG_SECTION_NOT_FOUND"),
                                     ("empty token", "OAUTH_EMPTY_TOKEN"),
                                     ("token expired", "OAUTH_TOKEN_EXPIRED"),
                                     ("failed to get token", "OAUTH_TOKEN_FAILURE"),
                                     ("unknown flag", "UNSUPPORTED_RCLONE_FLAG"),
                                     ("ratelimitexceeded", "DRIVE_RATE_LIMIT"),
                                     ("permission denied", "PERMISSION_DENIED")):
                if needle in err:
                    reason = category
                    break
            raise RuntimeError(f"RCLONE_FAILED operation={args[0]} exit={p.returncode} category={reason}; raw stderr withheld")
        return p.stdout

    def source(self, path):
        return self.remote + ":" + (relative_path(path) if path else "")

    def listing(self, path="", *, recursive=False, depth=1):
        if recursive and not path:
            raise ValueError("recursive Drive-root traversal is forbidden")
        args = ["lsjson", self.source(path), "--max-depth", str(depth)]
        if recursive:
            args.append("--recursive")
        rows = json.loads(self.run(*args))
        if len(rows) > 20000:
            raise ValueError("scoped inventory exceeds 20000 entries")
        return rows

    def copy_file(self, source, destination):
        self.run("copyto", self.source(source), str(destination))


def source_digest(root):
    root = Path(root)
    files = sorted((root / "self_audit").glob("*.py")) + [root / "train.py"]
    if len(files) < 2 or not all(p.is_file() for p in files):
        raise ValueError("incomplete pinned runtime")
    return hashlib.sha256("".join(f"{p.name}:{sha256(p)}\n" for p in files).encode()).hexdigest()


def initial_receipt():
    return {"schema": "self_audit.modal_staging.v1", "source_commit": SOURCE_COMMIT,
            "status": "BLOCKED", "gpu_requested": False, "dataset_through_mac": False,
            "files": [], "assets": {}, "discoveries": [], "missing": list(PINS),
            "prepared_data_verified": False, "persistence_verified": False}


def refresh_status(receipt):
    missing = [key for key, expected in PINS.items()
               if receipt["assets"].get(key, {}).get("sha256") != expected]
    if not receipt.get("prepared_data_verified"):
        missing.append("manifest_referenced_ACDC_TRAIN_VAL18_prepared_files")
    receipt["missing"] = missing
    receipt["status"] = "BLOCKED" if missing else "READY"
    return receipt


def discover(root, client, paths, *, recursive=False, depth=1):
    """List only observed/receipt-backed paths; never download directory trees."""
    root = Path(root)
    results = []
    for path in paths:
        rows = client.listing(path, recursive=recursive, depth=depth)
        if not path:
            rows = [r for r in rows if re.search(r"self.?audit|acdc|data|cardiac|medical|training", r["Name"], re.I)]
        result = {"source": client.source(path), "path": path, "recursive": recursive,
                  "max_depth": depth, "entries": rows}
        ident = hashlib.sha256(json.dumps([path, recursive, depth]).encode()).hexdigest()[:16]
        atomic_json(root / "receipts" / "discovery" / (ident + ".json"), result)
        results.append(result)
    return results


def copy_payload(root, client, item, receipt):
    asset = item["asset"]
    if asset not in ("teacher_weights", "student_weights") or item["sha256"] != PINS[asset]:
        raise ValueError("only the two pinned payloads may be staged")
    role = asset.split("_")[0]
    meta_record = receipt["assets"].get(role + "_manifest")
    if not meta_record or sha256(Path(root) / meta_record["path"]) != PINS[role + "_manifest"]:
        raise ValueError("copy and verify exact checkpoint manifest before payload")
    meta = json.loads((Path(root) / meta_record["path"]).read_text())
    if meta.get("payload_sha256") != PINS[asset]:
        raise ValueError("manifest/payload identity mismatch")
    source, target = relative_path(item["source"]), relative_path(item["target"])
    rows = client.listing(source)
    if len(rows) != 1 or rows[0].get("IsDir") or not 0 < rows[0]["Size"] < 2 << 30:
        raise ValueError("payload source unavailable or outside bounded size")
    dst = Path(root) / target
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    try:
        if dst.exists():
            if sha256(dst) != PINS[asset]:
                raise ValueError("existing payload differs; refusing overwrite")
        else:
            client.copy_file(source, tmp)
            if tmp.stat().st_size != rows[0]["Size"] or sha256(tmp) != PINS[asset]:
                raise ValueError("payload full SHA256/size mismatch")
            os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    record = {"path": target, "source": client.source(source), "sha256": PINS[asset],
              "bytes": dst.stat().st_size, "kind": "weights"}
    receipt["files"] = [f for f in receipt["files"] if f["path"] != target] + [record]
    receipt["assets"][asset] = record
    return record


def stage_git_runtime(root, item, receipt):
    """Fetch exact known source bytes directly to Modal, not datasets via client."""
    asset, commit = item["asset"], item["commit"]
    if asset not in ("teacher_runtime", "student_runtime") or not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("invalid runtime source identity")
    target = relative_path(item["target"])
    folder = Path(root) / target
    for name, expected in item["files"].items():
        name = relative_path(name)
        if not (name == "train.py" or name.startswith("self_audit/")) or not name.endswith(".py"):
            raise ValueError("runtime download only permits Python source in declared package")
        url = f"https://raw.githubusercontent.com/QuocKhanhLuong/Self-Audit/{commit}/{name}"
        dst = folder / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if sha256(dst) != expected:
                raise ValueError("existing source differs; refusing overwrite")
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read((2 << 20) + 1)
            if len(data) > 2 << 20 or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("pinned Git source byte hash mismatch")
            dst.write_bytes(data)
        record = {"path": f"{target}/{name}", "source": url, "sha256": expected,
                  "bytes": dst.stat().st_size, "kind": "runtime_source"}
        receipt["files"] = [f for f in receipt["files"] if f["path"] != record["path"]] + [record]
    actual = source_digest(folder)
    if actual != PINS[asset]:
        raise ValueError("runtime full source closure differs from diagnostic pin")
    receipt["assets"][asset] = {"path": target, "sha256": actual, "git_commit": commit}
    return receipt["assets"][asset]


def stage_git_metadata(root, entries, receipt):
    results = []
    for item in entries:
        asset = item["asset"]
        if asset not in ("teacher_manifest", "student_manifest", "train_manifest", "val_manifest"):
            raise ValueError("Git metadata only permits pinned manifests")
        source, target = relative_path(item["source"]), relative_path(item["target"])
        url = f"https://raw.githubusercontent.com/QuocKhanhLuong/Self-Audit/{SOURCE_COMMIT}/{source}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read((2 << 20) + 1)
        if len(data) > 2 << 20 or hashlib.sha256(data).hexdigest() != PINS[asset]:
            raise ValueError("Git manifest byte hash mismatch")
        dst = Path(root) / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha256(dst) != PINS[asset]:
            raise ValueError("existing manifest differs; refusing overwrite")
        dst.write_bytes(data)
        record = {"path": target, "source": url, "sha256": PINS[asset],
                  "bytes": len(data), "kind": "metadata"}
        receipt["files"] = [f for f in receipt["files"] if f["path"] != target] + [record]
        receipt["assets"][asset] = record
        results.append(record)
    return results


def copy_metadata(root, client, entries, receipt):
    """Only exact small metadata files before both payload/runtime sources exist."""
    results = []
    for item in entries:
        source = relative_path(item["source"])
        target = relative_path(item["target"])
        if Path(source).suffix.lower() not in (".json", ".txt", ".md", ".py", ".patch"):
            raise ValueError("metadata copy excludes data/weights/archives")
        rows = client.listing(source)
        if len(rows) != 1 or rows[0].get("IsDir") or not 0 <= rows[0]["Size"] <= 2 << 20:
            raise ValueError("metadata must be one existing file at most 2 MiB")
        expected = item.get("sha256")
        asset = item.get("asset")
        if asset and (asset not in PINS or PINS[asset] != expected):
            raise ValueError("metadata identity differs from diagnostic pins")
        dst = Path(root) / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".partial")
        try:
            client.copy_file(source, tmp)
            actual = sha256(tmp)
            if expected and actual != expected:
                raise ValueError("copied metadata SHA256 mismatch")
            if dst.exists() and sha256(dst) != actual:
                raise ValueError("existing asset differs; refusing overwrite")
            os.replace(tmp, dst)
        finally:
            tmp.unlink(missing_ok=True)
        record = {"path": target, "source": client.source(source), "sha256": actual,
                  "bytes": dst.stat().st_size, "kind": "metadata"}
        receipt["files"] = [f for f in receipt["files"] if f["path"] != target] + [record]
        if asset:
            receipt["assets"][asset] = record
        results.append(record)
    return results


def verify_files(root, receipt):
    failures = []
    for entry in receipt["files"]:
        path = Path(root) / relative_path(entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            failures.append(entry["path"])
    for name in ("teacher_runtime", "student_runtime"):
        item = receipt.get("assets", {}).get(name)
        if item:
            try:
                valid = source_digest(Path(root) / relative_path(item["path"])) == PINS[name]
            except (ValueError, OSError):
                valid = False
            if not valid:
                failures.append(name + "_closure")
    return failures


def stage(root, action, request):
    root = Path(root)
    p = root / "receipts/STAGING.json"
    receipt = json.loads(p.read_text()) if p.exists() else initial_receipt()
    receipt["stage_container"] = os.environ.get("MODAL_TASK_ID", "local-test")
    receipt["stage_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["stager_source_sha256"] = sha256(__file__)
    receipt["persistence_verified"] = False
    result = None
    config = None
    try:
        with rclone_config() as (config, remote):
            receipt["credential_file_mode"] = oct(os.stat(config).st_mode & 0o777)
            receipt["credential_file_outside_volume"] = not Path(config).is_relative_to(root)
            client = Rclone(config, remote)
            receipt["remote_name"] = remote
            if action == "discover":
                result = discover(root, client, request.get("paths", [""]),
                                  recursive=request.get("recursive", False), depth=request.get("depth", 1))
                receipt["discoveries"] += [{k: r[k] for k in ("source", "recursive", "max_depth")} for r in result]
            elif action == "metadata":
                result = copy_metadata(root, client, request["entries"], receipt)
            elif action == "payload":
                result = copy_payload(root, client, request, receipt)
            elif action == "runtime":
                result = stage_git_runtime(root, request, receipt)
            elif action == "git_metadata":
                result = stage_git_metadata(root, request["entries"], receipt)
            elif action == "record_blockers":
                receipt["source_findings"] = request["findings"]
                result = {"findings_recorded": True}
            else:
                raise ValueError("unsupported staging action")
        receipt.pop("error", None)
    except Exception as exc:
        # No exception text from external libraries/config is allowed into receipts.
        receipt["error"] = {"type": type(exc).__name__, "operation": action,
                            "reason": str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "operation failed; raw diagnostic withheld"}
    refresh_status(receipt)
    receipt["credential_file_removed"] = config is not None and not Path(config).exists()
    if receipt.get("error"):
        receipt["status"] = "BLOCKED"
    atomic_json(p, receipt)
    return {"status": receipt["status"], "missing": receipt["missing"],
            "error": receipt.get("error"), "result": result, "receipt": "/assets/receipts/STAGING.json"}
