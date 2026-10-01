"""DSS-US native producer.

Original paper-reproduction profiles stay BLOCKED_PROTOCOL. Paper-faithful profiles
with declared conventions run the independent paper-faithful pipeline end to end:
image-only staging inventory -> DINO ViT-S/8 (pinned source, hashed checkpoint) ->
Step I eigensegments -> Step II (DSS step2) -> sealed anonymous raw partitions.
Exit codes: 0 RAW_COMPLETE / ready, 2 BLOCKED_PROTOCOL, 3 BLOCKED_DATA.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src")]
from shared_benchmark.native_protocol import ProtocolBlocked, file_hash, load_lock  # noqa: E402

DINO_COMMIT = "7c446df5b9f45747937fb0d72314eb9f7b66930a"


class BlockedData(Exception):
    pass


def _emit(status, reasons, code):
    print(json.dumps({"status": status, "reasons": reasons}))
    return code


def check_data(args):
    """Precise BLOCKED_DATA reasons; nothing is downloaded and no output is created."""
    reasons = []
    if args.images_manifest is None or not args.images_manifest.is_file():
        reasons.append(f"CAMUS image-only inventory not found: {args.images_manifest}")
    if args.image_root is None or not args.image_root.is_dir():
        reasons.append(f"CAMUS image-only staging root not found: {args.image_root}")
    repository = args.dino_repo
    if not (repository / "hubconf.py").is_file():
        reasons.append(f"pinned DINO source not found at {repository} (run scripts/pin_native_references.py)")
    else:
        head = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != DINO_COMMIT:
            reasons.append(f"DINO source is not pinned to {DINO_COMMIT}")
    if args.dino_checkpoint is None or not args.dino_checkpoint.is_file():
        reasons.append("DINO ViT-S/8 checkpoint dino_deitsmall8_pretrain.pth not supplied (--dino-checkpoint)")
    elif not args.dino_checkpoint_sha256:
        reasons.append("--dino-checkpoint-sha256 is required to bind the checkpoint")
    elif file_hash(args.dino_checkpoint) != args.dino_checkpoint_sha256:
        reasons.append("DINO checkpoint SHA-256 does not match --dino-checkpoint-sha256")
    if reasons:
        raise BlockedData(reasons)


def produce(inventory, settings, *, keys_provider, image_records=None):
    """Run the paper-faithful Step II (DSS step2) pipeline; returns {sample_id: (partition, receipt)}.

    keys_provider exposes ``keys(tensor, crop_to_patch_multiple=True)`` and ``embedding(tensor)``.
    """
    import cv2
    import numpy as np
    from dss_us import paper_faithful as pf
    from dss_us.preprocessing import imagenet_tensor, ultrasound_preprocess
    from shared_benchmark.native_artifacts import array_hash
    if settings.get("dino_input_policy") != "rgb_totensor_imagenet_crop_to_patch_multiple":
        raise pf.ProtocolSettingError("unsupported dino_input_policy")
    items, receipts = [], {}
    for record in image_records if image_records is not None else inventory.records:
        bgr = inventory.read_bgr(record)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        if settings.get("preprocessing"):
            method = settings["preprocessing_method"]
            parameters = settings["preprocessing_parameters"]
            rgb = ultrasound_preprocess(rgb, histogram_equalization="histogram_equalization" in method,
                                        gaussian_kernel=parameters.get("gaussian_kernel") if "gaussian_blur" in method else None,
                                        gaussian_sigma=parameters.get("gaussian_sigma") if "gaussian_blur" in method else None)
        tensor = imagenet_tensor(rgb)[None]
        keys, grid = keys_provider.keys(tensor, crop_to_patch_multiple=True)
        height, width = grid[0] * pf.PATCH_SIZE, grid[1] * pf.PATCH_SIZE
        gray = cv2.cvtColor(rgb[:height, :width], cv2.COLOR_RGB2GRAY)
        segmap, fit = pf.presegment(keys[0], gray, grid, settings)
        items.append((record["sample_id"], tensor, segmap))
        receipts[record["sample_id"]] = {"input_native_hw": list(bgr.shape[:2]), "decoded_image_sha256": array_hash(bgr),
                                         "patch_grid": list(grid), "step1_eigenvalues": [float(v) for v in fit["eigenvalues"][:16]],
                                         "step1_segments": int(len(np.unique(segmap)))}
    semantic, step2 = pf.step2_dss_official(items, settings, embed=keys_provider.embedding)
    outputs = {}
    for sample_id, _tensor, _segmap in items:
        receipt = receipts[sample_id]
        partition = pf.official_resize_to_image(semantic[sample_id], receipt["input_native_hw"])
        outputs[sample_id] = (partition, {**receipt, "step2": step2})
    return outputs


def warm_up_frameworks(dino):
    """Initialise framework/library state before the GT firewall is active (no dataset access).

    scikit-learn's thread-pool controller and lazy torch/skimage imports inspect system
    libraries on first use; doing that here keeps the guarded block limited to the
    approved images and outputs.
    """
    import numpy as np
    import torch
    from skimage.morphology import binary_dilation, binary_erosion  # noqa: F401
    from sklearn.cluster import KMeans, MiniBatchKMeans
    points = np.arange(8, dtype=np.float64).reshape(4, 2)
    KMeans(n_clusters=2, n_init=1, random_state=0).fit(points)
    MiniBatchKMeans(n_clusters=2, n_init=1, random_state=0, batch_size=4).fit(points)
    zeros = torch.zeros(1, 3, 16, 16)
    dino.keys(zeros, crop_to_patch_multiple=True)
    dino.embedding(zeros)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images-manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dino-repo", type=Path, default=ROOT / ".scratch/dino")
    parser.add_argument("--dino-checkpoint", type=Path)
    parser.add_argument("--dino-checkpoint-sha256")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--check-protocol", action="store_true")
    args = parser.parse_args()
    try:
        config = load_lock(args.config)
    except ProtocolBlocked as error:
        return _emit(error.status, error.reasons, 2)
    if config.get("implementation") != "paper_faithful_reimplementation":
        # Only a fully unblocked original profile reaches this point.
        from environment_contract import require_official_environment
        require_official_environment()
        return _emit("BLOCKED_PROTOCOL", ["Full native orchestration awaits evidenced row recipes and CRF correspondence"], 2)
    from dss_us import paper_faithful as pf
    settings = pf.dss_settings(config)
    if config["stage"] != "II":
        return _emit("BLOCKED_PROTOCOL", ["Step I requires CRF; no dense-CRF backend in the canonical environment"], 2)
    try:
        pf.validate_dss_step2_settings(settings)
    except pf.ProtocolSettingError as error:
        return _emit("BLOCKED_PROTOCOL", [str(error)], 2)
    if args.check_protocol:
        print(json.dumps({"status": "PAPER_FAITHFUL_READY", "profile_class": config["profile_class"],
                          "paper_equivalence": config["paper_equivalence"]}))
        return 0
    from environment_contract import require_official_environment
    environment = require_official_environment()
    try:
        check_data(args)
    except BlockedData as error:
        return _emit("BLOCKED_DATA", error.args[0], 3)
    if args.output is None or args.threads < 1:
        parser.error("--output and positive --threads required")
    import torch
    import cv2
    import sklearn
    import skimage
    from dss_us.features import DinoKeys
    from shared_benchmark.native_artifacts import (ImageInventory, ProducerAccessGuard, RawRunWriter,
                                                   environment_receipt)
    torch.set_num_threads(args.threads)
    inventory = ImageInventory(args.images_manifest, image_root=args.image_root)
    if inventory.document["dataset"] != config["dataset"]:
        raise ValueError("native dataset/profile mismatch")
    dino = DinoKeys(args.dino_repo, args.dino_checkpoint, checkpoint_sha256=args.dino_checkpoint_sha256, device="cpu")
    warm_up_frameworks(dino)
    sources = [*sorted((BASE / "src").rglob("*.py")), Path(__file__), ROOT / "src/shared_benchmark/native_artifacts.py",
               ROOT / "src/shared_benchmark/native_protocol.py", ROOT / "src/environment_contract.py"]
    provenance = {"code_files": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                  "environment": environment_receipt(), "environment_contract": environment, "threads": args.threads,
                  "implementation_kind": config["implementation"], "profile_class": config["profile_class"],
                  "dino": dino.receipt, "cohort_inventory_sha256": inventory.sha256,
                  "upstream_evidence_commit": "d4ac44c60df18b921c590796f6994a4c8ac0726c"}
    output = args.output.resolve()
    if output == inventory.image_root or inventory.image_root in output.parents or output in inventory.image_root.parents:
        raise ValueError("output root must not overlap the image-only staging root")
    writer = RawRunWriter(args.output, inventory=inventory, config=config, provenance=provenance)
    guard = ProducerAccessGuard(images=[r["resolved_image_path"] for r in inventory.records], output_root=writer.root,
                                image_root=inventory.image_root, readable_files=[inventory.path, args.config, *sources],
                                trusted_roots=[sys.prefix, sys.base_prefix, BASE / "src", ROOT / "src", args.dino_repo,
                                               Path(torch.__file__).parent, Path(cv2.__file__).parent,
                                               Path(sklearn.__file__).parent, Path(skimage.__file__).parent])
    try:
        with guard:
            outputs = produce(inventory, settings, keys_provider=dino)
            for record in inventory.records:
                partition, receipt = outputs[record["sample_id"]]
                writer.write(record, partition, receipt)
        verified = writer.seal(guard.log)
    except Exception:
        (writer.root / "FAILED.json").write_text(json.dumps({"status": "RAW_INCOMPLETE", "access_log": guard.log}),
                                                 encoding="utf-8")
        raise
    print(json.dumps({"status": "RAW_COMPLETE", "seal_sha256": verified["seal"]["seal_sha256"],
                      "profile_class": config["profile_class"], "paper_equivalence": config["paper_equivalence"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
