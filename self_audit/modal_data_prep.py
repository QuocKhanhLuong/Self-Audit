"""Independent fresh ACDC preparation; never changes the historical staging receipt."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time

from .asset_staging import Rclone, atomic_json, rclone_config, sha256

SPLIT_SHA = "b826eaf494760079aa559b0c0c12f7444951d779ebbb5ed291c0398f20bf835f"
HISTORICAL_TRAIN_SHA = "e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4"
HISTORICAL_VAL_SHA = "cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50"
L8 = ["patient007", "patient010", "patient011", "patient024", "patient026", "patient041", "patient077", "patient099"]
OLD_STAGING_SHA = "c4ffe94c921820f0bc3c67586f3ad64f5c6ec02fe2359146c1f2fa8c7b22029d"


class BoundedRclone(Rclone):
    """Slow scoped requests and retry only transient rate limits, never indefinitely."""
    def run(self, *args):
        for attempt in range(3):
            try:
                return super().run(*args, "--tpslimit", "2", "--tpslimit-burst", "2",
                                   "--drive-pacer-min-sleep", "500ms", "--max-duration", "600s",
                                   "--cutoff-mode", "HARD")
            except RuntimeError as exc:
                if "category=DRIVE_RATE_LIMIT" not in str(exc) or attempt == 2:
                    raise
                print(json.dumps({"stage": "DRIVE_RATE_LIMIT", "retry": attempt + 1,
                                  "max_attempts": 3, "delay_seconds": 30}), flush=True)
                time.sleep(30)


def locked_split(path):
    from .data import label_subset
    if sha256(path) != SPLIT_SHA:
        raise ValueError("locked split SHA256 mismatch")
    obj = json.loads(Path(path).read_text())
    train, val = obj["train"], obj["val"]
    assert len(train) == len(set(train)) == 80
    assert len(val) == len(set(val)) == 18
    assert not set(train) & set(val)
    assert not {"patient055", "patient076"} & set(train + val)
    assert sorted(label_subset(train, 8, 42)) == L8
    return obj


def download_plan(rows, split):
    ids = set(split["train"] + split["val"])
    mask_ids = set(L8 + split["val"])
    result = []
    for row in rows:
        path = row["Path"]
        if row.get("IsDir"):
            continue
        parts = path.split("/")
        if len(parts) != 2 or parts[0] not in ids:
            continue
        pid, name = parts
        cine = re.fullmatch(re.escape(pid) + r"_4d\.nii(?:\.gz)?", name)
        reference = re.fullmatch(re.escape(pid) + r"_frame\d+(?:_gt)?\.nii(?:\.gz)?", name)
        if cine or (reference and pid in mask_ids):
            result.append({"path": path, "size": row["Size"], "drive_id": row.get("ID"),
                           "kind": "cine" if cine else "reference"})
    for pid in ids:
        assert sum(x["path"].startswith(pid + "/") and x["kind"] == "cine" for x in result) == 1
        refs = [x["path"] for x in result if x["path"].startswith(pid + "/") and x["kind"] == "reference"]
        if pid in mask_ids:
            masks = [x for x in refs if "_gt.nii" in x]
            assert len(masks) == 2, (pid, "expected two references")
            assert all(x.replace("_gt.nii", ".nii") in refs for x in masks)
        else:
            assert not refs
    assert len(result) == 98 + 26 * 4
    return sorted(result, key=lambda x: x["path"])


def manifest_comparison(path, historical_path=None):
    """Compare to recovered exact historical bytes, allowing only path/serialization changes."""
    import copy
    raw = Path(path).read_bytes()
    obj = json.loads(raw)
    result = {"actual_sha256": sha256(path), "expected_sha256": HISTORICAL_TRAIN_SHA,
              "exact_bytes": sha256(path) == HISTORICAL_TRAIN_SHA, "path_only_witness": None}
    if result["exact_bytes"]:
        return result
    if historical_path is None:
        historical_path = Path(__file__).resolve().parents[1] / "configs/fresh_s0/acdc_l8_historical_train.json"
    if Path(historical_path).is_file():
        assert sha256(historical_path) == HISTORICAL_TRAIN_SHA
        historical = json.loads(Path(historical_path).read_text())
        def without_paths(value):
            value = copy.deepcopy(value)
            for group in ("labeled", "unlabeled"):
                for rec in value[group]:
                    rec["image"] = Path(rec["image"]).name
                    for m in rec.get("masks", {}).values():
                        m["path"] = Path(m["path"]).name
            return value
        if without_paths(obj) == without_paths(historical):
            result["path_only_witness"] = {"historical_manifest_sha256": HISTORICAL_TRAIN_SHA,
                "comparison": "complete parsed manifest equality after stripping image/mask parent paths",
                "payload_hashes_unchanged": True}
            return result
        result["historical_semantic_differences"] = [k for k in sorted(set(obj) | set(historical))
            if without_paths(obj).get(k) != without_paths(historical).get(k)]
        return result
    raise FileNotFoundError("exact historical manifest is required to investigate changed bytes")


def validate_prepared(raw, prepared, split, historical_val):
    import nibabel as nib
    import numpy as np
    raw, prepared = Path(raw), Path(prepared)
    train = json.loads((prepared / "train.json").read_text())
    dev = json.loads((prepared / "development.json").read_text())
    assert sorted(x["id"] for x in train["labeled"]) == L8
    assert len(train["unlabeled"]) == 72 and all("masks" not in x for x in train["unlabeled"])
    assert set(x["id"] for x in train["labeled"] + train["unlabeled"]) == set(split["train"])
    assert set(x["id"] for x in dev["patients"]) == set(split["val"])
    hist = {x["id"]: x for x in json.loads(Path(historical_val).read_text())["patients"]}
    assert sha256(historical_val) == HISTORICAL_VAL_SHA
    geometry, files, val_checks = [], [], []
    native = dict(dev, geometry_provenance="raw_nifti_affine_and_exact_frame_cine_bridge")
    native["patients"] = []
    for rec in train["labeled"] + train["unlabeled"] + dev["patients"]:
        pid = rec["id"]
        cine = next((raw / pid).glob(pid + "_4d.nii*"))
        nii = nib.load(cine)
        image = prepared / rec["image"]
        arr = np.load(image, mmap_mode="r", allow_pickle=False)
        assert arr.dtype == np.float32 and arr.shape == tuple(reversed(nii.shape))
        assert np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1
        assert sha256(image) == rec["image_sha256"]
        files.append({"path": str(image), "sha256": rec["image_sha256"], "bytes": image.stat().st_size})
        frame_checks = []
        for frame, m in rec.get("masks", {}).items():
            mask_path = prepared / m["path"]
            y = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            assert y.shape == arr.shape[1:] and y.dtype == np.uint8
            assert np.isin(y, [0, 1, 2, 3, 255]).all() and sha256(mask_path) == m["sha256"]
            original = next((raw / pid).glob(f"{pid}_frame{int(frame)+1:02d}.nii*"))
            fr = nib.load(original)
            assert np.allclose(fr.affine, nii.affine, atol=1e-5, rtol=1e-5)
            assert np.array_equal(np.asarray(fr.dataobj), np.asarray(nii.dataobj)[..., int(frame)])
            files.append({"path": str(mask_path), "sha256": m["sha256"], "bytes": mask_path.stat().st_size})
            frame_checks.append({"frame": int(frame), "labels": np.unique(y).tolist(), "bridge": "EXACT"})
        geometry.append({"id": pid, "shape_tzyx": list(arr.shape), "affine": nii.affine.tolist(),
                         "axes": ["T", "Z", "Y", "X"], "frames": frame_checks, "raw_sha256": sha256(cine)})
        if pid in hist:
            h = hist[pid]
            val_checks.append({"id": pid, "image_matches": rec["image_sha256"] == h["image_sha256"],
                 "mask_hashes_match": {f: m["sha256"] for f, m in rec["masks"].items()} == {f: m["sha256"] for f, m in h["masks"].items()},
                 "raw_matches": sha256(cine) == h["geometry_source_image"]["sha256"],
                 "affine_matches": bool(np.allclose(nii.affine, h["affine"], atol=1e-5, rtol=1e-5))})
            native["patients"].append(dict(rec, axes=["T", "Z", "Y", "X"], affine=nii.affine.tolist(),
                spatial_units="mm", geometry_source_image={"path": str(cine), "sha256": sha256(cine)}))
    assert len(files) == 98 + 52
    assert all(all(x[k] for k in ("image_matches", "mask_hashes_match", "raw_matches", "affine_matches")) for x in val_checks)
    atomic_json(prepared / "val18_fresh_native.json", native)
    return {"files": files, "geometry": geometry, "historical_val_payload_comparison": val_checks,
            "train_manifest_comparison": manifest_comparison(prepared / "train.json"),
            "val_manifest_sha256": sha256(prepared / "val18_fresh_native.json"),
            "development_manifest_sha256": sha256(prepared / "development.json")}


def prepare_on_modal(namespace, repo="/repo", use_observed_inventory=False):
    import sys
    sys.path.insert(0, repo)
    from prepare_acdc import prepare
    base = Path(namespace)
    historical_receipt = Path("/assets/receipts/STAGING.json")
    assert sha256(historical_receipt) == OLD_STAGING_SHA
    split_path = Path(repo) / "splits/acdc_reviewed_locked_train80_val18.json"
    split = locked_split(split_path)
    receipt_path = base / "DATA_PREP.json"
    if receipt_path.exists():
        raise ValueError("fresh namespace already has DATA_PREP.json; explicit resume/review required")
    base.mkdir(parents=True, exist_ok=False)
    raw, prepared = base / "raw", base / "prepared"
    receipt = {"schema": "self_audit.fresh_s0.data.v1", "status": "PREPARING", "seed": 42,
        "label_budget": 8, "labeled_ids": L8, "split_sha256": SPLIT_SHA,
        "stage_container": os.environ.get("MODAL_TASK_ID"), "namespace": str(base),
        "historical_staging_sha256": OLD_STAGING_SHA, "teacher_required": False,
        "source_files": {p: sha256(Path(repo)/p) for p in ("prepare_acdc.py", "self_audit/data.py", "self_audit/modal_data_prep.py")}}
    atomic_json(receipt_path, receipt)
    started = time.monotonic()
    try:
        with rclone_config() as (config, remote):
            client = BoundedRclone(config, remote)
            root_flags = []
            source = client.source("Dataset/ACDC/training")
            if use_observed_inventory:
                inventory_path = Path("/assets/receipts/discovery/e7f5be8897f16d63.json")
                inventory = json.loads(inventory_path.read_text())
                assert inventory["path"] == "Dataset/ACDC" and inventory["max_depth"] == 3
                roots = [r for r in inventory["entries"] if r["Path"] == "training" and r["IsDir"]]
                assert len(roots) == 1 and roots[0]["ID"]
                rows = [dict(r, Path=r["Path"].removeprefix("training/")) for r in inventory["entries"]
                        if r["Path"].startswith("training/")]
                root_flags = ["--drive-root-folder-id", roots[0]["ID"]]
                source = client.source("")
                receipt["observed_inventory"] = {"path": str(inventory_path), "sha256": sha256(inventory_path),
                    "root_id": roots[0]["ID"], "fresh_listing": False, "payloads_verified_after_copy": True}
            else:
                rows = client.listing("Dataset/ACDC/training", recursive=True, depth=2)
            plan = download_plan(rows, split)
            atomic_json(base / "DOWNLOAD_PLAN.json", {"remote": remote, "files": plan})
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as listing:
                listing.write("\n".join(x["path"] for x in plan) + "\n"); listing.flush()
                print(json.dumps({"stage": "DOWNLOAD", "files": len(plan), "bytes": sum(x["size"] for x in plan)}), flush=True)
                raw.mkdir()
                client.run("copy", source, str(raw), "--files-from-raw", listing.name,
                           "--transfers", "2", "--checkers", "2", "--no-traverse", *root_flags)
        receipt["credential_file_removed"] = not Path(config).exists()
        assert {str(p.relative_to(raw)) for p in raw.rglob("*") if p.is_file()} == {x["path"] for x in plan}
        for x in plan:
            p = raw / x["path"]
            assert p.stat().st_size == x["size"]
            x["sha256"] = sha256(p)
        atomic_json(base / "RAW_FILES.json", plan)
        print(json.dumps({"stage": "PREPARE", "entrypoint": "prepare_acdc.py", "labels": 8, "seed": 42, "evaluation": True}), flush=True)
        train_sha = prepare(raw, split_path, prepared, 8, 42, evaluation=True)
        receipt.update(validate_prepared(raw, prepared, split,
            Path(repo)/"historical_val18_native.json"))
        receipt.update(train_manifest_sha256=train_sha, train_manifest=str(prepared / "train.json"),
                       val_manifest=str(prepared / "val18_fresh_native.json"), raw_file_count=len(plan),
                       prepared_image_count=98, prepared_mask_count=52, unlabeled_mask_downloads=0,
                       unlabeled_mask_reads=0, train_count=80, val_count=18)
        comp = receipt["train_manifest_comparison"]
        receipt["status"] = "PREPARED_PENDING_FRESH_CONTAINER" if comp["exact_bytes"] or comp["path_only_witness"] else "BLOCKED_HISTORICAL_TRAIN_COMPARISON"
    except Exception as exc:
        receipt.update(status="BLOCKED", error={"type": type(exc).__name__, "message": str(exc)[:1000]})
    receipt["elapsed_seconds"] = time.monotonic() - started
    assert sha256(historical_receipt) == OLD_STAGING_SHA
    receipt["historical_receipt_unchanged"] = True
    atomic_json(receipt_path, receipt)
    return {k: receipt.get(k) for k in ("status", "namespace", "train_manifest_sha256", "train_manifest_comparison", "error", "elapsed_seconds")}


def verify_fresh_container(namespace):
    import numpy as np
    from .compact_data import CompactTrainingData
    base = Path(namespace)
    receipt = json.loads((base / "DATA_PREP.json").read_text())
    assert receipt["stage_container"] != os.environ["MODAL_TASK_ID"]
    assert "RCLONE_CONFIG_TEXT" not in os.environ
    assert "SELFAUDIT_DRIVE_AUTHORIZED_USER_JSON" not in os.environ
    assert sha256("/assets/receipts/STAGING.json") == OLD_STAGING_SHA
    if receipt["status"] not in ("PREPARED_PENDING_FRESH_CONTAINER", "PASS"):
        return {"status": receipt["status"], "error": receipt.get("error"), "comparison": receipt.get("train_manifest_comparison")}
    for f in receipt["files"]:
        assert sha256(f["path"]) == f["sha256"]
    data = CompactTrainingData(receipt["train_manifest"], receipt["train_manifest_sha256"], synthetic=False)
    rng = np.random.default_rng(42)
    li, _, _ = data.sample_indices(0, rng)
    x, y, meta = data.sample_batch(li, rng, 8, labeled=True, crop_size=160)
    assert x.shape == (8,3,160,160) and y.shape == (8,160,160)
    assert set(data.label_reads).issubset(L8)
    val = json.loads(Path(receipt["val_manifest"]).read_text())
    for r in val["patients"]:
        assert r["axes"] == ["T", "Z", "Y", "X"] and np.asarray(r["affine"]).shape == (4,4)
    receipt.update(status="PASS", persistence_verified=True, verification_container=os.environ["MODAL_TASK_ID"],
        cpu_batch={"x_shape": list(x.shape), "y_shape": list(y.shape), "label_reads": sorted(data.label_reads)},
        s0_namespace="fresh_s0", historical_reproduction=receipt["train_manifest_comparison"]["exact_bytes"])
    data.close()
    atomic_json(base / "DATA_PREP.json", receipt)
    return {k: receipt[k] for k in ("status", "namespace", "train_manifest_sha256", "val_manifest_sha256", "train_manifest_comparison", "persistence_verified", "cpu_batch")}
