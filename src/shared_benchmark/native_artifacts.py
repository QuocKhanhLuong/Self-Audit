"""Method-agnostic native image allowlists, raw seals and verification.

No algorithm, semantic adapter, GT loader or GPL package is imported here.
The Python access guard supplements image-only mounts; it is not an OS sandbox.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys

import numpy as np

from .native_protocol import canonical_json, file_hash, value_hash


class NativeContractError(ValueError):
    pass


def _annotation_path(path) -> bool:
    return bool(re.search(r"(^|[/\\_.-])(gt|labels?|masks?|annotations?|groundtruth|ground_truth)([/\\_.-]|$)",
                          str(path).lower()))


def _within_path(path, root) -> bool:
    path, root = Path(path), Path(root)
    return path == root or root in path.parents


def _safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise NativeContractError("unsafe sample ID")
    return value


def array_hash(array):
    value = np.ascontiguousarray(array)
    header = canonical_json({"shape": list(value.shape), "dtype": value.dtype.str})
    import hashlib
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def _image_only_metadata(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"gt", "ground_truth", "groundtruth", "mask", "masks",
                                    "label", "labels", "oracle", "dice", "hungarian", "annotation"}:
                raise NativeContractError("GT/oracle fields cannot enter producer metadata")
            _image_only_metadata(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _image_only_metadata(child)


def environment_receipt():
    packages = {}
    for name in ("numpy", "torch", "torchvision", "scipy", "scikit-learn", "opencv-python", "pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    installed = sorted(
        [{"name": dist.metadata["Name"], "version": dist.version}
         for dist in importlib.metadata.distributions()],
        key=lambda item: (item["name"].lower(), item["version"]),
    )
    framework = sys.modules.get("torch")
    accelerator = None if framework is None else {
        "torch_cuda_build": framework.version.cuda,
        "cudnn_version": framework.backends.cudnn.version(),
    }
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "installed_packages": installed, "installed_packages_sha256": value_hash(installed),
            "accelerator": accelerator}


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    temporary.replace(path)


class ImageInventory:
    """Explicit inventory, no globbing or GT existence/header checks.

    Every image must live under `image_root`, an image-only staging directory that
    contains nothing besides the listed images (and optionally this manifest).
    """
    def __init__(self, manifest_path, *, image_root):
        root = Path(image_root).absolute()
        for component in (root, *root.parents):
            if component.is_symlink():
                raise NativeContractError("image-only staging root cannot contain symbolic links")
        if not root.is_dir():
            raise NativeContractError("image-only staging root must be an existing directory")
        self.image_root = root.resolve()
        if _annotation_path(self.image_root):
            raise NativeContractError("annotation-shaped staging root")
        self.path = Path(manifest_path).absolute()
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if set(document) != {"schema", "dataset", "records"}:
            raise NativeContractError("image manifest has unexpected fields")
        if document["schema"] != "medical-native.image-only.v1":
            raise NativeContractError("unknown image-only manifest schema")
        if not isinstance(document["records"], list) or not document["records"]:
            raise NativeContractError("empty image inventory")
        self.document = document
        self.records = []
        seen = set()
        for record in document["records"]:
            if set(record) != {"sample_id", "role", "image_path", "image_sha256"} or record["role"] != "image":
                raise NativeContractError("record must contain image-only fields")
            sample = _safe_id(record["sample_id"])
            if sample in seen:
                raise NativeContractError("duplicate sample ID")
            seen.add(sample)
            path = Path(record["image_path"])
            if _annotation_path(path):
                raise NativeContractError("annotation-shaped image path rejected before probing")
            if not path.is_absolute():
                path = self.path.parent / path
            # Reject links before resolving/reading, including linked parents.
            for component in (path, *path.parents):
                if component.is_symlink():
                    raise NativeContractError("image locator cannot contain symbolic links")
            path = path.resolve()
            if _annotation_path(path):
                raise NativeContractError("annotation locator")
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                raise NativeContractError("native inventory requires explicitly prepared image files")
            if not re.fullmatch("[0-9a-f]{64}", record["image_sha256"]):
                raise NativeContractError("image must be hash-bound")
            if not _within_path(path, self.image_root):
                raise NativeContractError("image outside the image-only staging root")
            self.records.append({**record, "resolved_image_path": path})
        self.sha256 = value_hash(document)
        self._verify_image_only_root()

    def _verify_image_only_root(self):
        """Fail closed unless the staging root holds exactly the inventory images.

        Runs before the producer starts. Unlisted entry names are never reported,
        so a misplaced annotation cannot leak through the error message either.
        """
        allowed_files = {r["resolved_image_path"] for r in self.records}
        if _within_path(self.path, self.image_root):
            allowed_files.add(self.path.resolve())
        allowed_directories = {self.image_root}
        for path in allowed_files:
            allowed_directories.update(p for p in path.parents if _within_path(p, self.image_root))
        unlisted = 0
        for directory, subdirectories, files in os.walk(self.image_root, followlinks=False):
            for name in subdirectories:
                entry = Path(directory) / name
                if entry.is_symlink() or entry not in allowed_directories:
                    unlisted += 1
            for name in files:
                entry = Path(directory) / name
                if entry.is_symlink() or entry not in allowed_files:
                    unlisted += 1
        if unlisted:
            raise NativeContractError(
                f"image-only staging root contains {unlisted} unlisted entries; stage images separately from GT")

    def read_bgr(self, record):
        # imdecode avoids an untraced OpenCV filesystem read.
        import cv2
        path = record["resolved_image_path"]
        encoded = path.read_bytes()
        import hashlib
        if hashlib.sha256(encoded).hexdigest() != record["image_sha256"]:
            raise NativeContractError("input image hash changed")
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
            raise NativeContractError("invalid native image")
        return image


class ProducerAccessGuard:
    """Deny dataset reads, stat/existence probes and listing outside an explicit allowlist.

    Fail closed: producers may open/stat only the exact listed images and readable
    files. Directory listing/globbing is permitted only inside the output root and
    trusted code roots, never in or above the image-only staging root, so sibling
    GT directories cannot be enumerated. Directory metadata (stat) is allowed only
    for the staging root's internal image ancestors and for ancestors of allowed
    non-image paths. Python-level guard; it supplements OS mount isolation.
    """
    _CODE_SUFFIXES = {".py", ".pyc", ".pyd", ".dll", ".so", ".json", ".txt", ".h", ".cpp"}

    def __init__(self, *, images, output_root, image_root, trusted_roots=(), readable_files=()):
        norm = lambda p: os.path.normcase(os.path.abspath(os.fsdecode(p)))
        self.image_root = norm(image_root)
        self.images = {norm(p) for p in images}
        self.readable = {norm(p) for p in readable_files}
        self.files = self.images | self.readable
        self.output = norm(output_root)
        self.trusted = [norm(p) for p in trusted_roots]
        for image in self.images:
            if not self._within(image, self.image_root):
                raise NativeContractError("guarded image outside the image-only staging root")
        for root in (self.output, *self.trusted):
            if self._within(root, self.image_root) or self._within(self.image_root, root):
                raise NativeContractError("output/trusted roots must not overlap the image-only staging root")
        for path in self.readable:
            if self._within(path, self.image_root) and path not in self.images and Path(path).suffix != ".json":
                raise NativeContractError("only images and the inventory may be read from the staging root")
        # Directory metadata: never strict ancestors of the staging root unless they
        # are also needed as ancestors of non-image allowed paths.
        self.stat_directories = {self.image_root}
        for image in self.images:
            self.stat_directories.update(norm(p) for p in Path(image).parents if self._within(norm(p), self.image_root))
        for path in (*self.readable, self.output, *self.trusted):
            self.stat_directories.update(norm(p) for p in Path(path).parents)
        self.log = []
        self.active = False

    @staticmethod
    def _within(path, root):
        return path == root or path.startswith(root.rstrip(os.sep) + os.sep)

    def _check(self, path, *, event, writing=False):
        if not self.active or isinstance(path, int):
            return
        location = os.path.normcase(os.path.abspath(os.fsdecode(path)))
        output = self._within(location, self.output)
        trusted = any(self._within(location, root) for root in self.trusted)
        staged = self._within(location, self.image_root)
        above_staging = self._within(self.image_root, location) and not staged
        suffix = Path(location).suffix.lower()
        if output:
            allowed = True
        elif writing:
            allowed = False
        elif event in {"listdir", "scandir", "glob"}:
            # Listing is how siblings are discovered: only code and output trees.
            allowed = trusted and not staged and not above_staging
        elif event == "open":
            allowed = location in self.files or (trusted and suffix in self._CODE_SUFFIXES)
        else:  # stat / lstat / access / existence probes
            allowed = location in self.files or location in self.stat_directories or trusted
        if _annotation_path(location) and not (trusted and suffix in {".py", ".pyc"}) and not output:
            allowed = False
        self.log.append({"event": event, "path": location, "writing": writing, "allowed": allowed})
        if not allowed:
            raise NativeContractError("GT firewall: unapproved filesystem access")

    def _audit(self, event, args):
        if not self.active:
            return
        if event == "open":
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            self._check(args[0], event="open", writing=writing)
        elif event in {"os.listdir", "os.scandir"}:
            self._check(args[0] or os.getcwd(), event=event.split(".")[1])
        elif event in {"glob.glob", "glob.glob/2"}:
            pattern = os.fsdecode(args[0])
            root_dir = args[2] if event == "glob.glob/2" and len(args) > 2 else None
            if root_dir is not None and not os.path.isabs(pattern):
                pattern = os.path.join(os.fsdecode(root_dir), pattern)
            # Check the deepest non-wildcard directory of the pattern.
            fixed = []
            for part in Path(pattern).parts:
                if any(c in part for c in "*?["):
                    break
                fixed.append(part)
            self._check(os.path.join(*fixed) if fixed else os.getcwd(), event="glob")

    def __enter__(self):
        sys.addaudithook(self._audit)
        self.original_stat, self.original_lstat, self.original_access = os.stat, os.lstat, os.access
        def guarded_stat(path, *args, **kwargs):
            self._check(path, event="stat")
            return self.original_stat(path, *args, **kwargs)
        def guarded_lstat(path, *args, **kwargs):
            self._check(path, event="lstat")
            return self.original_lstat(path, *args, **kwargs)
        def guarded_access(path, *args, **kwargs):
            self._check(path, event="access")
            return self.original_access(path, *args, **kwargs)
        os.stat, os.lstat, os.access = guarded_stat, guarded_lstat, guarded_access
        self.active = True
        return self

    def __exit__(self, exc_type, *args):
        self.active = False
        os.stat, os.lstat, os.access = self.original_stat, self.original_lstat, self.original_access
        # Existence helpers (os.path.exists, pathlib glob) swallow the denial; fail closed anyway.
        if exc_type is None and any(not row["allowed"] for row in self.log):
            raise NativeContractError("GT firewall: denied filesystem access was suppressed by caller")


class RawRunWriter:
    def __init__(self, output_root, *, inventory, config, provenance):
        self.root = Path(output_root).resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise NativeContractError("refuse nonempty output root")
        self.root.mkdir(parents=True, exist_ok=True)
        self.samples = []
        self.inventory = inventory
        self.run = {"schema": "medical-native.raw-run.v1", "method": config["method"],
                    "dataset": config["dataset"], "profile": config["profile"],
                    "stage": config.get("stage"), "config": config,
                    "config_sha256": value_hash(config), "image_manifest_sha256": inventory.sha256,
                    "expected_samples": [r["sample_id"] for r in inventory.records],
                    "provenance": provenance}
        _write_json(self.root / "run.json", self.run)

    def _refuse_if_sealed(self):
        if (self.root / "raw_seal.json").exists():
            raise NativeContractError("raw run is already sealed; resealing/rewriting is not permitted")

    def write(self, record, partition, metadata):
        self._refuse_if_sealed()
        _image_only_metadata(metadata)
        sample = _safe_id(record["sample_id"])
        if sample not in self.run["expected_samples"] or any(r["sample_id"] == sample for r in self.samples):
            raise NativeContractError("unknown or duplicate output sample")
        partition = np.asarray(partition)
        if partition.ndim != 2 or not np.issubdtype(partition.dtype, np.integer) or np.any(partition < 0):
            raise NativeContractError("raw partition must be a nonnegative 2-D integer map")
        if np.max(partition) > np.iinfo(np.int32).max:
            raise NativeContractError("raw ID exceeds int32")
        partition = np.ascontiguousarray(partition, dtype="<i4")
        if metadata.get("input_native_hw") != list(partition.shape):
            raise NativeContractError("native prediction/input geometry mismatch")
        if not re.fullmatch("[0-9a-f]{64}", metadata.get("decoded_image_sha256", "")):
            raise NativeContractError("decoded image must be hash-bound")
        directory = self.root / sample
        directory.mkdir()
        np.savez_compressed(directory / "raw.npz", partition=partition)
        receipt = {"schema": "medical-native.raw-sample.v1", "sample_id": sample,
                   "image_sha256": record["image_sha256"], "partition_sha256": array_hash(partition),
                   "native_hw": list(partition.shape), "metadata": metadata}
        _write_json(directory / "metadata.json", receipt)
        self.samples.append({"sample_id": sample,
                             "raw_file_sha256": file_hash(directory / "raw.npz"),
                             "metadata_sha256": file_hash(directory / "metadata.json"),
                             "partition_sha256": receipt["partition_sha256"]})

    def seal(self, access_log):
        self._refuse_if_sealed()
        actual = {r["sample_id"] for r in self.samples}
        if actual != set(self.run["expected_samples"]):
            raise NativeContractError("cannot seal incomplete inventory")
        _write_json(self.root / "access_log.json", access_log)
        if any(not entry["allowed"] for entry in access_log):
            raise NativeContractError("cannot seal a GT-firewall violation")
        seal = {"schema": "medical-native.raw-seal.v1", "status": "RAW_COMPLETE",
                "run_sha256": file_hash(self.root / "run.json"),
                "access_log_sha256": file_hash(self.root / "access_log.json"),
                "samples": sorted(self.samples, key=lambda r: r["sample_id"])}
        seal["seal_sha256"] = value_hash(seal)
        with (self.root / "raw_seal.json").open("xb") as stream:  # exclusive: never replace a seal
            stream.write(canonical_json(seal) + b"\n")
        return verify_raw_run(self.root)


def verify_raw_run(root):
    root = Path(root).resolve()
    seal = json.loads((root / "raw_seal.json").read_text())
    unsigned = {k: v for k, v in seal.items() if k != "seal_sha256"}
    if seal.get("schema") != "medical-native.raw-seal.v1" or seal.get("status") != "RAW_COMPLETE":
        raise NativeContractError("invalid raw seal")
    if value_hash(unsigned) != seal.get("seal_sha256"):
        raise NativeContractError("seal hash mismatch")
    for name, key in (("run.json", "run_sha256"), ("access_log.json", "access_log_sha256")):
        if file_hash(root / name) != seal[key]:
            raise NativeContractError(f"sealed file changed: {name}")
    run = json.loads((root / "run.json").read_text())
    if value_hash(run["config"]) != run["config_sha256"]:
        raise NativeContractError("config receipt changed")
    if any(not r["allowed"] for r in json.loads((root / "access_log.json").read_text())):
        raise NativeContractError("sealed access violation")
    expected = run["expected_samples"]
    actual = [r["sample_id"] for r in seal["samples"]]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise NativeContractError("sealed inventory mismatch")
    allowed_root = {"run.json", "access_log.json", "raw_seal.json", *actual}
    if {p.name for p in root.iterdir()} != allowed_root:
        raise NativeContractError("unsealed entries in raw root")
    for record in seal["samples"]:
        directory = root / _safe_id(record["sample_id"])
        if {p.name for p in directory.iterdir()} != {"raw.npz", "metadata.json"}:
            raise NativeContractError("unsealed sample entries")
        if file_hash(directory / "raw.npz") != record["raw_file_sha256"]:
            raise NativeContractError("raw file changed")
        if file_hash(directory / "metadata.json") != record["metadata_sha256"]:
            raise NativeContractError("metadata changed")
        metadata = json.loads((directory / "metadata.json").read_text())
        _image_only_metadata(metadata["metadata"])
        with np.load(directory / "raw.npz", allow_pickle=False) as values:
            if set(values.files) != {"partition"}:
                raise NativeContractError("raw contains non-partition fields")
            array = values["partition"]
        if array.ndim != 2 or array.dtype != np.dtype("<i4") or np.any(array < 0):
            raise NativeContractError("raw shape/dtype invalid")
        if metadata["native_hw"] != list(array.shape) or metadata["sample_id"] != record["sample_id"]:
            raise NativeContractError("raw geometry/sample mismatch")
        if array_hash(array) != record["partition_sha256"] or record["partition_sha256"] != metadata["partition_sha256"]:
            raise NativeContractError("partition seal mismatch")
    return {"run": run, "seal": seal}


SEAL_RECEIPT_SCHEMA = "medical-native.raw-seal-receipt.v1"


def _seal_receipt_body(verified, root):
    run, seal = verified["run"], verified["seal"]
    return {"schema": SEAL_RECEIPT_SCHEMA, "status": "RAW_SEAL_FINALIZED",
            "method": run["method"], "dataset": run["dataset"], "profile": run["profile"],
            "stage": run.get("stage"), "config_sha256": run["config_sha256"],
            "image_manifest_sha256": run["image_manifest_sha256"],
            "seal_sha256": seal["seal_sha256"], "raw_seal_file_sha256": file_hash(root / "raw_seal.json"),
            "run_sha256": seal["run_sha256"], "access_log_sha256": seal["access_log_sha256"],
            "samples": [{"sample_id": r["sample_id"], "partition_sha256": r["partition_sha256"],
                         "raw_file_sha256": r["raw_file_sha256"]} for r in seal["samples"]]}


def write_seal_receipt(raw_root, receipt_path):
    """Finalize a verified raw seal in a separate, write-once receipt outside the raw root.

    Commit or archive the receipt before any evaluation; evaluators verify against it,
    so a modified and resealed raw run no longer matches.
    """
    root = Path(raw_root).resolve()
    verified = verify_raw_run(root)
    receipt_path = Path(receipt_path).absolute()
    if _within_path(receipt_path.parent.resolve() / receipt_path.name, root):
        raise NativeContractError("seal receipt must be stored outside the raw root")
    receipt = _seal_receipt_body(verified, root)
    receipt["receipt_sha256"] = value_hash(receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("xb") as stream:  # FileExistsError: never overwrite a receipt
        stream.write(canonical_json(receipt) + b"\n")
    os.chmod(receipt_path, 0o444)
    return receipt


def verify_seal_receipt(raw_root, receipt_path):
    """Evaluator preflight: external receipt AND raw seal must verify before GT access."""
    root = Path(raw_root).resolve()
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt.get("schema") != SEAL_RECEIPT_SCHEMA or receipt.get("status") != "RAW_SEAL_FINALIZED":
        raise NativeContractError("invalid seal receipt")
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if value_hash(unsigned) != receipt.get("receipt_sha256"):
        raise NativeContractError("seal receipt hash mismatch")
    verified = verify_raw_run(root)
    if _seal_receipt_body(verified, root) != unsigned:
        raise NativeContractError("raw run does not match its finalized seal receipt")
    return {**verified, "receipt": receipt}


def compare_repeat_runs(first, second):
    a, b = verify_raw_run(first), verify_raw_run(second)
    if a["run"]["config_sha256"] != b["run"]["config_sha256"] or a["run"]["image_manifest_sha256"] != b["run"]["image_manifest_sha256"]:
        raise NativeContractError("repeat identity mismatch")
    if a["run"]["provenance"] != b["run"]["provenance"]:
        raise NativeContractError("repeat seed/code/environment mismatch")
    rows = []
    for record in a["seal"]["samples"]:
        sample = record["sample_id"]
        with np.load(Path(first) / sample / "raw.npz", allow_pickle=False) as data:
            x = data["partition"]
        with np.load(Path(second) / sample / "raw.npz", allow_pickle=False) as data:
            y = data["partition"]
        if x.shape != y.shape:
            raise NativeContractError("repeat grid mismatch")
        pairs = np.unique(np.stack([x.ravel(), y.ravel()], axis=1), axis=0)
        equivalent = len(pairs) == len(np.unique(x)) == len(np.unique(y))
        rows.append({"sample_id": sample, "first_array_sha256": array_hash(x),
                     "second_array_sha256": array_hash(y), "equal": bool(np.array_equal(x, y)),
                     "raw_id_mismatch_fraction": float(np.mean(x != y)),
                     "partition_equivalent_up_to_id_permutation": equivalent})
    return {"first_seal_sha256": a["seal"]["seal_sha256"], "second_seal_sha256": b["seal"]["seal_sha256"],
            "all_equal": all(row["equal"] for row in rows), "samples": rows}
