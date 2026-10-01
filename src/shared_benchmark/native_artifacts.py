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
    """Explicit inventory, no globbing or GT existence/header checks."""
    def __init__(self, manifest_path):
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
            self.records.append({**record, "resolved_image_path": path})
        self.sha256 = value_hash(document)

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
    """Deny arbitrary dataset reads/stat/listing; log allowed and denied accesses."""
    def __init__(self, *, images, output_root, trusted_roots=(), readable_files=()):
        self.images = {os.path.normcase(os.path.abspath(p)) for p in images}
        self.files = self.images | {os.path.normcase(os.path.abspath(p)) for p in readable_files}
        self.output = os.path.normcase(os.path.abspath(output_root))
        self.trusted = [os.path.normcase(os.path.abspath(p)) for p in trusted_roots]
        self.directories = set()
        for path in self.files | {self.output}:
            self.directories.update(os.path.normcase(str(p)) for p in Path(path).parents)
        self.log = []
        self.active = False

    @staticmethod
    def _within(path, root):
        return path == root or path.startswith(root + os.sep)

    def _check(self, path, *, event, writing=False):
        if not self.active or isinstance(path, int):
            return
        location = os.path.normcase(os.path.abspath(os.fsdecode(path)))
        output = self._within(location, self.output)
        trusted = any(self._within(location, root) for root in self.trusted)
        suffix = Path(location).suffix.lower()
        trusted_code = trusted and suffix in {".py", ".pyc", ".pyd", ".dll", ".so", ".json", ".txt", ".h", ".cpp"}
        allowed = output or (not writing and (location in self.files or
                  (event != "open" and location in self.directories) or
                  (trusted and event != "open") or trusted_code))
        if _annotation_path(location) and not (trusted and suffix in {".py", ".pyc"}) and not output:
            allowed = False
        self.log.append({"event": event, "path": location, "writing": writing, "allowed": allowed})
        if not allowed:
            raise NativeContractError("GT firewall: unapproved filesystem access")

    def _audit(self, event, args):
        if event == "open":
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            self._check(args[0], event="open", writing=writing)
        elif event in {"os.listdir", "os.scandir"}:
            self._check(args[0] or os.getcwd(), event=event)

    def __enter__(self):
        sys.addaudithook(self._audit)
        self.original_stat, self.original_lstat = os.stat, os.lstat
        def guarded_stat(path, *args, **kwargs):
            self._check(path, event="stat")
            return self.original_stat(path, *args, **kwargs)
        def guarded_lstat(path, *args, **kwargs):
            self._check(path, event="lstat")
            return self.original_lstat(path, *args, **kwargs)
        os.stat, os.lstat = guarded_stat, guarded_lstat
        self.active = True
        return self

    def __exit__(self, *args):
        self.active = False
        os.stat, os.lstat = self.original_stat, self.original_lstat


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

    def write(self, record, partition, metadata):
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
        _write_json(self.root / "raw_seal.json", seal)
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
