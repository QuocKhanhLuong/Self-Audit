#!/usr/bin/env python3
"""Bounded training-transaction profiles for the native teacher-memory pilot.

Manifest profiles use ``self_audit.engine.train``. The separate
``--manufactured-native-cost`` path uses the same supervised/pseudo/EMA helpers
with the real pinned DINO backbone and manufactured inputs. Both are execution
cost probes, not accuracy evaluations. Synthetic mode uses ``tiny_test`` and
has no patient-quality interpretation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ARMS = tuple(f"A{i}" for i in range(7))
MAX_MEASURED_UPDATES = 3
DEFAULT_BATCHES = (4, 8, 16, 32)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_bank_spec(spec: str) -> tuple[str, Path, str]:
    """Parse ``ARM=PATH=SHA256`` without accepting unpinned bank assets."""
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("bank must use ARM=PATH=SHA256")
    arm, path_text, expected = parts
    if arm not in ARMS[2:]:
        raise argparse.ArgumentTypeError("bank arm must be A2, A3, A4, A5, or A6")
    path = Path(path_text).expanduser().resolve()
    expected = expected.lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise argparse.ArgumentTypeError("bank SHA-256 must be 64 hexadecimal characters")
    return arm, path, expected


def parse_batches(value: str) -> tuple[int, ...]:
    try:
        batches = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batches must be comma-separated positive integers") from exc
    if not batches or any(batch < 1 for batch in batches) or len(set(batches)) != len(batches):
        raise argparse.ArgumentTypeError("batches must be a nonempty list of unique positive integers")
    return batches


def parse_arms(value: str) -> tuple[str, ...]:
    arms = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not arms or any(arm not in ARMS for arm in arms) or len(set(arms)) != len(arms):
        raise argparse.ArgumentTypeError("arms must be unique values from A0,A1,A2,A3,A4,A5,A6")
    return arms


def summarize_run(rows: list[dict[str, Any]], report: dict[str, Any], *,
                  batch_size: int, warmup_updates: int) -> dict[str, Any]:
    """Summarize measured engine rows; count L and U separately and explicitly."""
    if len(rows) <= warmup_updates:
        raise ValueError("run contains no measured updates after warmup")
    measured = rows[warmup_updates:]
    intervals = []
    previous = 0.0
    for row in rows:
        current = float(row["elapsed_seconds"])
        intervals.append(current - previous)
        previous = current
    seconds = sum(intervals[warmup_updates:])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("measured elapsed time must be finite and positive")
    def role_count(row: dict[str, Any], role: str) -> int:
        metadata = row[role]
        return len(metadata.get("samples", [metadata]))

    l_examples = sum(role_count(row, "supervised") for row in measured)
    u_examples = sum(role_count(row, "unlabeled") for row in measured)
    audit_examples = sum(role_count(row, "audit") for row in measured)
    return {
        "warmup_updates_excluded": warmup_updates,
        "measured_updates": len(measured),
        "measured_seconds_from_engine_rows": seconds,
        "configured_microbatch_size": batch_size,
        "supervised_labeled_examples": l_examples,
        "audit_labeled_examples_not_used_for_quality": audit_examples,
        "unlabeled_examples": u_examples,
        "labeled_plus_unlabeled_examples": l_examples + u_examples,
        "labeled_plus_unlabeled_examples_per_second": (l_examples + u_examples) / seconds,
        "all_role_examples_including_audit": l_examples + u_examples + audit_examples,
        "optimizer_step_counts": {
            "supervised_transactions": int(report["supervised_updates"]),
            "pseudo_transactions": int(report["pseudo_accepted"]),
            "ema_updates": int(report["ema_updates"]),
            # engine.train uses one body AdamW and one head SGD optimizer per
            # committed supervised or pseudo transaction.
            "body_optimizer_steps": int(report["supervised_updates"] + report["pseudo_accepted"]),
            "head_optimizer_steps": int(report["supervised_updates"] + report["pseudo_accepted"]),
        },
    }


def operator_cost_counts(batch_size: int, measured_updates: int, total_updates: int,
                         measured_seconds: float) -> dict[str, Any]:
    """Counts for the manifest-free real-model operator-cost probe."""
    if min(batch_size, measured_updates, total_updates) < 1 or total_updates < measured_updates:
        raise ValueError("operator-cost counts require positive, consistent update counts")
    if not math.isfinite(measured_seconds) or measured_seconds <= 0:
        raise ValueError("measured_seconds must be finite and positive")
    labeled = batch_size * measured_updates
    unlabeled = batch_size * measured_updates
    optimizer_steps = 2 * total_updates
    return {
        "supervised_labeled_examples": labeled,
        "unlabeled_examples": unlabeled,
        "labeled_plus_unlabeled_examples": labeled + unlabeled,
        "labeled_plus_unlabeled_examples_per_second": (labeled + unlabeled) / measured_seconds,
        "optimizer_step_counts": {
            "supervised_transactions": total_updates,
            "pseudo_transactions": total_updates,
            "body_optimizer_steps": optimizer_steps,
            "head_optimizer_steps": optimizer_steps,
            "ema_updates": total_updates,
        },
    }


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _make_manufactured_bank(path: Path, seed: int) -> str:
    """Create an explicitly synthetic bank for tiny-engine smoke only."""
    import numpy as np
    from self_audit.teacher_memory import MemoryBank, save_bank

    rng = np.random.default_rng(seed)
    endpoints = rng.normal(size=(6, 8, 2, 32)).astype(np.float32)
    endpoints /= np.maximum(np.linalg.norm(endpoints, axis=-1, keepdims=True), 1e-8)
    confidences = np.full((6, 8, 2), 0.9, dtype=np.float32)
    valid = np.ones((6, 8), dtype=np.bool_)
    bank = MemoryBank(endpoints, confidences, valid, {
        "provenance": {"source": "manufactured_profile_control", "seed": int(seed),
                       "scientific_claim": "NONE"},
    })
    return save_bank(bank, path)


def _make_manufactured_cache(manifest: Path, out: Path) -> tuple[Path, str]:
    """Write a normalized uniform-probability cache for synthetic U examples."""
    import numpy as np

    data = json.loads(manifest.read_text())
    records = []
    native_geometry_by_patient = {}
    for sample in data["unlabeled"]:
        image_path = Path(sample["image"])
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        shape = (int(image.shape[0]), int(image.shape[1]), 4,
                 int(image.shape[2]), int(image.shape[3]))
        target = np.full(shape, 0.25, dtype=np.float32)
        target_path = out / f"{sample['id']}_uniform_probabilities.npy"
        np.save(target_path, target, allow_pickle=False)
        support = np.ones(image.shape, dtype=np.bool_)
        support_path = out / f"{sample['id']}_manufactured_valid_support.npy"
        np.save(support_path, support, allow_pickle=False)
        records.append({
            "id": sample["id"], "image_sha256": sample["image_sha256"],
            "probabilities": target_path.name,
            "probabilities_sha256": _sha256(target_path), "shape": list(shape),
        })
        native_geometry_by_patient[sample["id"]] = {
            "shape_tzyx": list(image.shape),
            # Explicitly synthetic index coordinates, not patient geometry.
            "affine_xyz_to_world_mm": [[1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0]],
            "probability_shape_tzcyx": list(shape),
            "axis_mapping": "native_NIfTI_XYZT_to_prepared_TZYX=transpose(3,2,1,0); probability_TZCYX_plane=class,Y,X from the same native affine XYZ grid",
            "valid_support": {"path": support_path.name, "sha256": _sha256(support_path),
                              "shape_tzyx": list(image.shape)},
        }
    cache_path = out / "manufactured_external_pseudo.json"
    _write_json(cache_path, {
        "schema": "self_audit.external_pseudo.v1", "role": "training", "synthetic": True,
        "class_map": ["BG", "RV", "MYO", "LV"],
        "probability_layout": "TZCYX", "records": records, "evaluation_ids": [],
        "provenance": {"writer": "profile_teacher_memory", "source": "manufactured_uniform_control",
                       "scientific_claim": "NONE",
                       "native_geometry_by_patient": native_geometry_by_patient},
    })
    return cache_path, _sha256(cache_path)


def _load_arm_config(config_dir: Path, arm: str) -> dict[str, Any]:
    config_path = config_dir / f"{arm}.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing arm config: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("teacher_memory", {}).get("arm") != arm:
        raise ValueError(f"arm config {config_path} does not declare {arm}")
    return config


def _model_asset_identity(model_config_path: Path | None, model_config: dict[str, Any], synthetic: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "backbone": model_config["backbone"],
        "model_config_sha256": _sha256(model_config_path) if model_config_path else None,
        "source_repo": model_config.get("source_repo"),
        "source_commit": model_config.get("source_commit"),
        "weights_declared_sha256": model_config.get("weights_sha256"),
        "weights_actual_sha256": None,
    }
    weight_path = Path(model_config.get("weights", "")).expanduser() if model_config.get("weights") else None
    if weight_path is not None and weight_path.is_file():
        actual = _sha256(weight_path.resolve())
        result["weights_actual_sha256"] = actual
        declared = model_config.get("weights_sha256")
        if declared and actual != declared.lower():
            raise ValueError("model weight file does not match the declared SHA-256")
    elif not synthetic:
        raise ValueError("real-data profiles require an existing pinned model weight file")
    return result


def _manufactured_native_cost(args: argparse.Namespace) -> dict[str, Any]:
    """Profile real pinned DINO operators on manufactured tensors, with no data claims."""
    import copy
    import gc
    import numpy as np
    import torch
    from self_audit import engine
    from self_audit.engine import ema_update, full_pseudo_step, source_digest, supervised_step
    from self_audit.model import build_model
    from self_audit.teacher_memory import MemoryBank, permute_endpoints, random_feature_control

    if args.synthetic or args.manufactured_bank_control or args.manifest or args.manifest_sha256:
        raise ValueError("manufactured-native-cost is a standalone no-manifest operator-cost mode")
    if not args.model_config:
        raise ValueError("manufactured-native-cost requires a pinned --model-config")
    if not (1 <= args.native_batch_size <= 32) or not (64 <= args.native_crop_size <= 512):
        raise ValueError("native batch/crop size is outside bounded profile limits")
    if not (0 <= args.warmup_updates <= 1) or not (1 <= args.measured_updates <= MAX_MEASURED_UPDATES):
        raise ValueError("native cost mode supports at most one warmup and three measured transactions")
    if args.bank or args.cache or args.cache_sha256:
        raise ValueError("native cost mode manufactures in-memory operator controls; no data cache/bank input")

    model_config_path = args.model_config.expanduser().resolve()
    model_config = json.loads(model_config_path.read_text())
    if model_config.get("backbone") != "dinov3_convnext_tiny":
        raise ValueError("manufactured-native-cost requires the real pinned DINO backbone")
    model_identity = _model_asset_identity(model_config_path, model_config, synthetic=False)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent device fallback")
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"output must be a new directory: {out}")
    out.mkdir(parents=True, exist_ok=False)

    # Same dense bank capacity and common manufactured vectors for all memory
    # arms. A5 is operator cost only: this bank is not student-derived.
    bank_rng = np.random.default_rng(41)
    endpoints = bank_rng.normal(size=(6, 8, 2, 32)).astype(np.float32)
    endpoints /= np.maximum(np.linalg.norm(endpoints, axis=-1, keepdims=True), 1e-8)
    base_bank = MemoryBank(
        endpoints,
        np.full((6, 8, 2), 0.9, dtype=np.float32),
        np.ones((6, 8), dtype=np.bool_),
        {"provenance": {"source": "manufactured_native_operator_cost_control",
                        "seed": 41, "student_checkpoint_payload_sha256": None,
                        "teacher_weights_sha256": None, "scientific_claim": "NONE"}},
    )

    # One common pair of tensors and targets is reused by every arm. These
    # are manufactured cost inputs, not image-shaped patient samples.
    device = torch.device(args.device)
    b, size = args.native_batch_size, args.native_crop_size
    data_rng = np.random.default_rng(4242)
    x_l = torch.from_numpy(data_rng.random((b, 3, size, size), dtype=np.float32)).to(device)
    x_u = torch.from_numpy(data_rng.random((b, 3, size, size), dtype=np.float32)).to(device)
    y_l = torch.from_numpy(data_rng.integers(0, 4, size=(b, size, size), dtype=np.int64)).to(device)
    strong = x_u.clone()
    external_probabilities = torch.full((b, 4, size, size), 0.25, dtype=x_u.dtype, device=device)
    external_accepted = torch.ones((b, size, size), dtype=torch.bool, device=device)
    total_steps = args.warmup_updates + args.measured_updates
    config_dir = args.config_dir.expanduser().resolve()
    base_config = _load_arm_config(config_dir, "A0")
    torch.set_num_threads(base_config["threads"])
    matched_keys = ("seed", "body_lr", "head_lr", "weight_decay", "ema_decay", "window")
    matched_values = {key: base_config[key] for key in matched_keys}

    report: dict[str, Any] = {
        "schema": "self_audit.teacher_memory_profile.v1", "status": "RUNNING",
        "data_mode": "MODEL_REAL_DINO_INPUTS_AND_BANKS_MANUFACTURED_COST_ONLY",
        "label_mode": "MODEL_REAL_DINO_INPUTS_AND_BANKS_MANUFACTURED_COST_ONLY",
        "device": args.device, "quality": "NOT_EVALUATED", "patient_dice": "NOT_MEASURED",
        "evaluation": "NOT_RUN", "source_sha256": source_digest(), "model_assets": model_identity,
        "profile_script_sha256": _sha256(Path(__file__).resolve()),
        "class_map": ["BG", "RV", "MYO", "LV"],
        "manufactured_bank_content_sha256": None,
        "protocol": {
            "arms": list(args.arms), "common_batch_size": b, "crop_size": size,
            "dense_entries_per_class_pair": 8, "warmup_full_transactions": args.warmup_updates,
            "measured_full_transactions": args.measured_updates,
            "same_inputs_initialization_and_update_count_across_arms": True,
            "full_engine_train": "NOT_CALLED; existing supervised_step/full_pseudo_step/ema_update helpers used",
            "A1": "external soft-target optimizer branch cost only; manufactured uniform targets, no cache evidence",
            "A5": "memory operator branch cost only; random bank is not student-derived and is not A5 bank evidence",
            "offline_teacher_runtime": "EXCLUDED; no external teacher import or inference",
            "checkpoint_persistence": "NOT_SAVED_PROFILE_ONLY",
        },
        "runs": [],
    }
    bank_hash = hashlib.sha256()
    bank_hash.update(base_bank.endpoints.detach().cpu().numpy().tobytes())
    bank_hash.update(base_bank.confidences.detach().cpu().numpy().tobytes())
    bank_hash.update(base_bank.valid.detach().cpu().numpy().tobytes())
    bank_hash.update(json.dumps(base_bank.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    report["manufactured_bank_content_sha256"] = bank_hash.hexdigest()
    _write_json(out / "PROFILE.json", report)

    for arm in args.arms:
        try:
            arm_config_path = config_dir / f"{arm}.json"
            config = _load_arm_config(config_dir, arm)
            if any(config[key] != matched_values[key] for key in matched_keys):
                raise ValueError("native cost arms must share seed, optimizers, EMA decay, and fixed-window config")
            if args.device.startswith("cuda"):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                torch.cuda.synchronize(args.device)
            torch.manual_seed(config["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config["seed"])
            bank = None
            mode = "paired"
            if arm in ARMS[2:]:
                bank = copy.deepcopy(base_bank)
                if arm == "A4":
                    bank = permute_endpoints(bank, seed=config["seed"])
                elif arm == "A6":
                    bank = random_feature_control(bank, seed=config["seed"])
                if arm == "A2":
                    mode = "unpaired"
            model, provenance = build_model(
                model_config, synthetic=False, window=config["window"], memory_bank=bank,
                memory_mode=mode, memory_tau=config["teacher_memory"]["tau"],
            )
            model = model.to(device)
            teacher = copy.deepcopy(model).eval().requires_grad_(False)
            body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
            optimizers = [
                torch.optim.AdamW(body, lr=config["body_lr"], weight_decay=config["weight_decay"]),
                torch.optim.SGD(model.head.parameters(), lr=config["head_lr"], momentum=0., weight_decay=0.),
            ]
            bank_identity = hashlib.sha256()
            if bank is not None:
                for tensor in (bank.endpoints, bank.confidences, bank.valid):
                    bank_identity.update(tensor.detach().cpu().numpy().tobytes())
                bank_identity.update(json.dumps(bank.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            arm_bank_sha = bank_identity.hexdigest() if bank is not None else None

            rows = []
            memory_gradients = []
            memory = None
            for step in range(total_steps):
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                tick = time.perf_counter()
                supervised_loss_value, supervised_grad_norm = supervised_step(model, optimizers, x_l, y_l)
                model.eval()
                with torch.no_grad():
                    q = teacher(x_u).softmax(1).detach()
                    confidence, pseudo = q.max(1)
                    accepted = confidence >= 0.0
                if arm == "A1":
                    pseudo_loss_value, pseudo_grad_norm = full_pseudo_step(
                        model, optimizers, strong, pseudo, accepted,
                        external_probabilities=external_probabilities,
                        external_accepted=external_accepted,
                        external_weight=config["teacher_memory"]["external_weight"],
                    )
                else:
                    pseudo_loss_value, pseudo_grad_norm = full_pseudo_step(
                        model, optimizers, strong, pseudo, accepted,
                    )
                if bank is not None:
                    memory = model.teacher_memory
                    memory_gradients.append({
                        "projection_grad_present": memory.projection.grad is not None,
                        "projection_grad_norm": float(memory.projection.grad.detach().norm()) if memory.projection.grad is not None else None,
                        "alpha_gamma_grad_present": memory.alpha.grad is not None,
                        "alpha_gamma_grad_norm": float(memory.alpha.grad.detach().norm()) if memory.alpha.grad is not None else None,
                    })
                ema_update(teacher, model, config["ema_decay"])
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                elapsed = time.perf_counter() - tick
                rows.append({"step": step + 1, "seconds": elapsed,
                             "supervised_loss": supervised_loss_value,
                             "supervised_gradient_norm": supervised_grad_norm,
                             "pseudo_loss": pseudo_loss_value,
                             "pseudo_gradient_norm": pseudo_grad_norm,
                             "accepted_pseudo_pixels": int(accepted.sum())})

            measured = rows[args.warmup_updates:]
            measured_seconds = sum(row["seconds"] for row in measured)
            if not math.isfinite(measured_seconds) or measured_seconds <= 0:
                raise RuntimeError("measured native cost interval is invalid")
            if bank is not None and (len(memory_gradients) != total_steps or any(
                    not (item["projection_grad_present"] and item["alpha_gamma_grad_present"])
                    for item in memory_gradients)):
                raise RuntimeError("memory projection/W or alpha/gamma backward was absent")
            allocated = int(torch.cuda.max_memory_allocated(args.device)) if args.device.startswith("cuda") else None
            reserved = int(torch.cuda.max_memory_reserved(args.device)) if args.device.startswith("cuda") else None
            cost_counts = operator_cost_counts(b, len(measured), total_steps, measured_seconds)
            record = {
                "arm": arm, "status": "PROFILED_OPERATOR_COST_ONLY", "data_mode": report["data_mode"],
                "arm_config_sha256": _sha256(arm_config_path),
                "batch_size": b, "crop_size": size, "warmup_updates_excluded": args.warmup_updates,
                "measured_updates": len(measured), "measured_seconds": measured_seconds,
                **cost_counts,
                "cuda_peak_allocated_bytes_including_model_and_teacher": allocated,
                "cuda_peak_reserved_bytes_including_model_and_teacher": reserved,
                "backbone_provenance": provenance, "bank_content_sha256": arm_bank_sha,
                "memory_gradients_after_pseudo_backward": memory_gradients,
                "rows": rows, "quality": "NOT_EVALUATED", "patient_dice": "NOT_MEASURED",
            }
            report["runs"].append(record)
            _write_json(out / "PROFILE.json", report)
            # Drop every per-arm CUDA reference, including optimizer parameter
            # lists and the bank alias, before the next arm resets peak stats.
            del teacher, model, optimizers, body, bank, memory
            del q, confidence, pseudo, accepted
            gc.collect()
        except BaseException as exc:
            oom = _is_cuda_oom(exc)
            report["status"] = "FAILED_OOM" if oom else "FAILED"
            report["failure"] = {"arm": arm, "error_type": type(exc).__name__, "error": str(exc),
                                 "cuda_oom": oom, "successful_profile_claim": False}
            _write_json(out / "PROFILE.json", report)
            raise

    report["status"] = "COMPLETED_OPERATOR_COST_ONLY"
    _write_json(out / "PROFILE.json", report)
    return report


def _prepare_config(args: argparse.Namespace, arm: str, batch: int, *,
                    config_dir: Path, model_config: dict[str, Any],
                    bank_assets: dict[str, tuple[Path, str]],
                    cache_asset: tuple[Path, str] | None) -> dict[str, Any]:
    config = _load_arm_config(config_dir, arm)
    config["model"] = copy.deepcopy(model_config)
    config["steps"] = args.warmup_updates + args.measured_updates
    # Engine-level adaptive warmup remains disabled; the profiler's first
    # complete EMA pseudo transaction is excluded only from cost summaries.
    config["warmup_steps"] = 0
    config["batch_size"] = batch
    if args.crop_size is not None:
        config["crop_size"] = args.crop_size
    elif args.synthetic:
        config["crop_size"] = 32
    # A zero threshold guarantees support in manufactured smoke. Real profiles
    # inherit the declared experiment threshold unless explicitly overridden.
    if args.synthetic:
        config["pseudo_threshold"] = 0.0
    if args.pseudo_threshold is not None:
        config["pseudo_threshold"] = args.pseudo_threshold

    tm = config["teacher_memory"]
    if arm in bank_assets:
        path, sha = bank_assets[arm]
        tm["bank_path"], tm["bank_sha256"] = str(path), sha
    if arm == "A1" and cache_asset is not None:
        path, sha = cache_asset
        tm["cache_path"], tm["cache_sha256"] = str(path), sha
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs/teacher_memory_pilot",
                        help="directory containing A0.json through A6.json")
    parser.add_argument("--model-config", type=Path,
                        help="pinned backbone model JSON; required for real-data profiles")
    parser.add_argument("--manifest", type=Path, help="hash-pinned TRAIN-only data manifest")
    parser.add_argument("--manifest-sha256", help="SHA-256 of --manifest")
    parser.add_argument("--synthetic", action="store_true",
                        help="manufacture toy data, bank/cache, and use tiny_test; no patient claims")
    parser.add_argument("--manufactured-bank-control", action="store_true",
                        help="real TRAIN/model cost probe with an explicitly random bank; never an accuracy arm")
    parser.add_argument("--manufactured-native-cost", action="store_true",
                        help="real pinned DINO operator-cost probe on manufactured tensors; no manifest/quality evidence")
    parser.add_argument("--native-batch-size", type=int, default=4,
                        help="shared batch size for --manufactured-native-cost (default: 4)")
    parser.add_argument("--native-crop-size", type=int, default=256,
                        help="shared crop size for --manufactured-native-cost (default: 256)")
    parser.add_argument("--device", default="cuda", help="explicit device, e.g. cuda or cpu; no fallback")
    parser.add_argument("--arms", type=parse_arms, default=ARMS,
                        help="comma-separated arms (default: A0,A1,A2,A3,A4,A5,A6)")
    parser.add_argument("--batches", "--batch-sizes", dest="batches", type=parse_batches,
                        default=DEFAULT_BATCHES,
                        help="candidate microbatch sizes (default: 4,8,16,32)")
    parser.add_argument("--warmup-updates", type=int, default=1,
                        help="complete supervised+pseudo+EMA transactions excluded from throughput (0 or 1)")
    parser.add_argument("--measured-updates", type=int, default=2,
                        help="measured full transactions per arm/batch (1..3)")
    parser.add_argument("--crop-size", type=int,
                        help="crop override; synthetic defaults to 32, real defaults to arm config")
    parser.add_argument("--pseudo-threshold", type=float,
                        help="optional shared override; synthetic defaults to zero to exercise pseudo backward")
    parser.add_argument("--bank", action="append", type=parse_bank_spec, default=[], metavar="ARM=PATH=SHA256",
                        help="hash-pinned source bank for A2-A6; repeat once per arm")
    parser.add_argument("--cache", type=Path, help="hash-pinned A1 external pseudo cache JSON")
    parser.add_argument("--cache-sha256", help="SHA-256 of --cache")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    import gc
    import torch
    from self_audit import engine
    from self_audit.data import synthetic_fixture
    from self_audit.engine import source_digest, train
    from self_audit.model import digest
    from self_audit.teacher_memory import load_bank, permute_endpoints, random_feature_control, save_bank

    if args.manufactured_native_cost:
        return _manufactured_native_cost(args)

    if not (0 <= args.warmup_updates <= 1):
        raise ValueError("warmup-updates must be 0 or 1")
    if not (1 <= args.measured_updates <= MAX_MEASURED_UPDATES):
        raise ValueError("measured-updates must be between 1 and 3")
    if args.crop_size is not None and args.crop_size < 0:
        raise ValueError("crop-size must be nonnegative")
    if args.pseudo_threshold is not None and not (0 <= args.pseudo_threshold <= 1):
        raise ValueError("pseudo-threshold must lie in [0,1]")
    if args.synthetic:
        if args.manifest or args.manifest_sha256:
            raise ValueError("synthetic mode creates its own manifest; do not pass patient data")
        if args.model_config:
            raise ValueError("synthetic mode uses the declared tiny_test model, not a pinned patient model")
        if args.bank or args.cache or args.cache_sha256:
            raise ValueError("synthetic mode manufactures its explicitly declared control bank/cache")
    else:
        if not args.manifest or not args.manifest_sha256 or not args.model_config:
            raise ValueError("real-data mode requires --manifest, --manifest-sha256, and --model-config")
        if len(args.manifest_sha256) != 64 or digest(args.manifest) != args.manifest_sha256.lower():
            raise ValueError("TRAIN manifest SHA-256 mismatch")
        if args.model_config.resolve() == args.out.resolve():
            raise ValueError("model config cannot be the output directory")
        if args.manufactured_bank_control and args.bank:
            raise ValueError("manufactured-bank-control creates its own bank; do not also pass --bank")
    if args.synthetic and args.manufactured_bank_control:
        raise ValueError("manufactured-bank-control is for real TRAIN/model cost probes; synthetic already creates controls")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent device fallback")

    bank_assets: dict[str, tuple[Path, str]] = {}
    for arm, path, expected_sha in args.bank:
        if arm in bank_assets:
            raise ValueError(f"duplicate bank assignment for {arm}")
        if not path.is_file() or digest(path) != expected_sha:
            raise ValueError(f"bank file/hash mismatch for {arm}: {path}")
        bank_assets[arm] = (path, expected_sha)
    cache_asset: tuple[Path, str] | None = None
    if bool(args.cache) != bool(args.cache_sha256):
        raise ValueError("--cache and --cache-sha256 must be supplied together")
    if args.cache:
        if not args.cache.is_file() or digest(args.cache) != args.cache_sha256.lower():
            raise ValueError("external pseudo cache SHA-256 mismatch")
        cache_asset = (args.cache.resolve(), args.cache_sha256.lower())

    missing_banks = [arm for arm in args.arms if arm in ARMS[2:] and arm not in bank_assets]
    if missing_banks and not args.synthetic and not args.manufactured_bank_control:
        raise ValueError(f"missing hash-pinned banks for arms: {', '.join(missing_banks)}")
    if "A1" in args.arms and cache_asset is None and not args.synthetic and not args.manufactured_bank_control:
        raise ValueError("real-data A1 profile requires --cache and --cache-sha256")

    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"output must be a new directory: {out}")
    out.mkdir(parents=True, exist_ok=False)
    config_dir = args.config_dir.expanduser().resolve()
    assets_dir = out / "manufactured_assets" if (args.synthetic or args.manufactured_bank_control) else out / "profile_assets"
    assets_dir.mkdir()

    if args.synthetic:
        manifest, manifest_sha = synthetic_fixture(assets_dir / "synthetic_data", seed=17,
                                                   n_labeled=3, n_unlabeled=3, size=32)
        model_config = {"backbone": "tiny_test", "source_repo": "", "source_commit": "",
                        "weights": "", "weights_sha256": ""}
        bank_assets = {}
        if any(arm in args.arms for arm in ARMS[2:]):
            bank_path = assets_dir / "manufactured_memory_bank.npz"
            bank_sha = _make_manufactured_bank(bank_path, seed=41)
            bank_assets.update({arm: (bank_path, bank_sha) for arm in ARMS[2:]
                                if arm in args.arms and arm != "A5"})
        cache_asset = None
        if "A1" in args.arms:
            cache_asset = _make_manufactured_cache(manifest, assets_dir)
        data_mode = "SYNTHETIC_TINY_TEST"
        label_mode = "MIXED_MANUFACTURED_CONTROL" if any(arm != "A0" for arm in args.arms) else "SYNTHETIC_NO_MEMORY"
    else:
        manifest, manifest_sha = args.manifest.resolve(), args.manifest_sha256.lower()
        model_config = json.loads(args.model_config.read_text())
        data_mode = "TRAIN_MANIFEST_ONLY"
        label_mode = "MIXED_MANUFACTURED_CONTROL" if args.manufactured_bank_control else "REAL_TRAIN_ASSETS"
        if args.manufactured_bank_control and any(arm in args.arms for arm in ARMS[2:]):
            bank_path = assets_dir / "manufactured_memory_bank.npz"
            bank_sha = _make_manufactured_bank(bank_path, seed=41)
            bank_assets = {arm: (bank_path, bank_sha) for arm in ARMS[2:]
                           if arm in args.arms and arm != "A5"}

    skipped_arms = []
    if "A5" in args.arms and (args.synthetic or args.manufactured_bank_control):
        skipped_arms.append({"arm": "A5", "reason": "no frozen-student-derived bank was supplied or fabricated"})
    if ("A1" in args.arms and cache_asset is None and args.manufactured_bank_control):
        skipped_arms.append({"arm": "A1", "reason": "no hash-pinned A1 cache; affine geometry will not be fabricated"})

    model_identity = _model_asset_identity(args.model_config.resolve() if args.model_config else None,
                                           model_config, args.synthetic)
    effective_bank_hashes: dict[str, str] = {}
    for arm in args.arms:
        if arm not in ARMS[2:]:
            continue
        if any(item["arm"] == arm for item in skipped_arms):
            continue
        path, expected_sha = bank_assets[arm]
        bank = load_bank(path, expected_sha)
        if arm == "A4":
            bank = permute_endpoints(bank, seed=42)
        elif arm == "A6":
            bank = random_feature_control(bank, seed=42)
        if arm in {"A4", "A6"}:
            derived = assets_dir / f"effective_{arm}_bank.npz"
            effective_bank_hashes[arm] = save_bank(bank, derived)
        else:
            effective_bank_hashes[arm] = expected_sha

    root_report: dict[str, Any] = {
        "schema": "self_audit.teacher_memory_profile.v1", "status": "RUNNING",
        "data_mode": data_mode, "label_mode": label_mode,
        "quality": "NOT_EVALUATED", "patient_dice": "NOT_MEASURED",
        "evaluation": "NOT_RUN", "device": args.device,
        "source_sha256": source_digest(), "manifest_sha256": manifest_sha,
        "profile_script_sha256": _sha256(Path(__file__).resolve()),
        "model_assets": model_identity,
        "bank_sha256_by_arm": {arm: sha for arm, (_, sha) in bank_assets.items() if arm in args.arms},
        "effective_bank_sha256_by_arm": effective_bank_hashes,
        "cache_sha256": cache_asset[1] if cache_asset else None,
        "protocol": {
            "arms": list(args.arms), "candidate_batch_sizes": list(args.batches),
            "warmup_full_transactions_per_run": args.warmup_updates,
            "measured_full_transactions_per_run": args.measured_updates,
            "same_batch_and_update_count_within_each_arm_comparison": True,
            "different_batch_sizes_are_exploratory_profiles_not_matched_scientific_arms": len(args.batches) > 1,
            "gradient_accumulation": "NOT_USED; engine has no supported accumulation protocol",
            "pseudo_threshold_override": 0.0 if args.synthetic and args.pseudo_threshold is None else args.pseudo_threshold,
            "offline_teacher_runtime": "EXCLUDED; no external teacher import or inference",
        },
        "runs": [],
        "skipped_arms": skipped_arms,
    }
    _write_json(out / "PROFILE.json", root_report)

    original_full_pseudo_step = engine.full_pseudo_step
    original_save_checkpoint = engine.save_checkpoint
    original_monotonic = engine.time.monotonic

    def no_profile_checkpoint(*_args, **_kwargs):
        # A bounded probe must not emit a student/teacher/optimizer checkpoint.
        return None

    def synchronized_monotonic():
        # engine rows use monotonic() as their transaction boundary; synchronize
        # immediately before every boundary so CUDA work is included.
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        return original_monotonic()

    for arm in args.arms:
        if any(item["arm"] == arm for item in skipped_arms):
            continue
        for batch in args.batches:
            config = _prepare_config(args, arm, batch, config_dir=config_dir,
                                     model_config=model_config, bank_assets=bank_assets,
                                     cache_asset=cache_asset)
            run_dir = out / "runs" / f"{arm}_batch{batch}"
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            observed_memory_grads: list[dict[str, float | bool | None]] = []

            def monitored_pseudo_step(*call_args, **call_kwargs):
                result = original_full_pseudo_step(*call_args, **call_kwargs)
                model = call_args[0]
                memory = getattr(model, "teacher_memory", None)
                if memory is not None:
                    details: dict[str, float | bool | None] = {}
                    for name, parameter in (("projection", memory.projection), ("alpha_gamma", memory.alpha)):
                        grad = parameter.grad
                        details[f"{name}_grad_present"] = grad is not None
                        details[f"{name}_grad_norm"] = float(grad.detach().norm()) if grad is not None else None
                        details[f"{name}_grad_finite"] = bool(torch.isfinite(grad).all()) if grad is not None else False
                    observed_memory_grads.append(details)
                return result

            engine.full_pseudo_step = monitored_pseudo_step
            engine.save_checkpoint = no_profile_checkpoint
            engine.time.monotonic = synchronized_monotonic
            try:
                if args.device.startswith("cuda"):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(args.device)
                    torch.cuda.reset_peak_memory_stats(args.device)
                    torch.cuda.synchronize(args.device)
                wall_start = time.perf_counter()
                result = train(config, manifest, manifest_sha, run_dir,
                               device=args.device, synthetic=args.synthetic)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                result["final_manifest_sha256"] = None
                result["checkpoint_persistence"] = "NOT_SAVED_PROFILE_ONLY"
                _write_json(run_dir / "REPORT.json", result)
                wall_seconds = time.perf_counter() - wall_start
                rows = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
                expected_steps = args.warmup_updates + args.measured_updates
                if len(rows) != expected_steps:
                    raise RuntimeError(f"engine emitted {len(rows)} rows for {expected_steps} bounded transactions")
                if int(result["supervised_updates"]) != expected_steps or int(result["ema_updates"]) != expected_steps:
                    raise RuntimeError("engine transaction counters do not match bounded step count")
                if int(result["pseudo_accepted"]) != expected_steps:
                    raise RuntimeError("not every bounded step executed a full EMA pseudo optimizer transaction")
                if arm in ARMS[2:]:
                    if len(observed_memory_grads) != expected_steps:
                        raise RuntimeError("memory gradient probe did not observe each full pseudo backward")
                    for grad in observed_memory_grads:
                        if not all(grad[name] for name in ("projection_grad_present", "alpha_gamma_grad_present",
                                                           "projection_grad_finite", "alpha_gamma_grad_finite")):
                            raise RuntimeError("memory projection/W or alpha/gamma backward was absent/nonfinite")

                counts = summarize_run(rows, result, batch_size=batch,
                                       warmup_updates=args.warmup_updates)
                allocated_peak = reserved_peak = None
                if args.device.startswith("cuda"):
                    allocated_peak = int(torch.cuda.max_memory_allocated(args.device))
                    reserved_peak = int(torch.cuda.max_memory_reserved(args.device))
                run_record = {
                    "arm": arm, "batch_size": batch, "status": "PROFILED",
                    "arm_config_sha256": _sha256(config_dir / f"{arm}.json"),
                    "synthetic": bool(args.synthetic), "label_mode": label_mode,
                    "configured_steps_including_warmup": expected_steps,
                    "pseudo_threshold": config["pseudo_threshold"],
                    "wall_seconds_including_startup_and_warmup": wall_seconds,
                    "cuda_peak_allocated_bytes_including_warmup": allocated_peak,
                    "cuda_peak_reserved_bytes_including_warmup": reserved_peak,
                    "source_bank_sha256": bank_assets[arm][1] if arm in bank_assets else None,
                    "effective_bank_sha256": effective_bank_hashes.get(arm),
                    "external_cache_sha256": cache_asset[1] if arm == "A1" and cache_asset else None,
                    "checkpoint_persistence": "NOT_SAVED_PROFILE_ONLY",
                    "counts_and_throughput": counts,
                    "memory_gradients_after_pseudo_backward": observed_memory_grads,
                    "engine_report": result,
                    "run_directory": str(run_dir),
                }
                root_report["runs"].append(run_record)
                _write_json(out / "PROFILE.json", root_report)
                _write_json(out / "profiles" / f"{arm}_batch{batch}.json", run_record)
            except BaseException as exc:
                oom = _is_cuda_oom(exc)
                root_report["status"] = "FAILED_OOM" if oom else "FAILED"
                root_report["failure"] = {
                    "arm": arm, "batch_size": batch,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "cuda_oom": oom,
                    "successful_profile_claim": False,
                    "partial_run_directory": str(run_dir),
                }
                _write_json(out / "PROFILE.json", root_report)
                if oom:
                    print(json.dumps({"status": "FAILED_OOM", "arm": arm, "batch_size": batch,
                                      "successful_profile": False}), file=sys.stderr)
                raise
            finally:
                engine.full_pseudo_step = original_full_pseudo_step
                engine.save_checkpoint = original_save_checkpoint
                engine.time.monotonic = original_monotonic
            # The fixed update cap is the primary bound; retain actual wall time
            # to expose startup/I/O overhead instead of hiding it.

    root_report["status"] = "COMPLETED_WITH_SKIPS" if root_report["skipped_arms"] else "COMPLETED"
    _write_json(out / "PROFILE.json", root_report)
    return root_report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except BaseException as exc:
        oom = _is_cuda_oom(exc)
        print(json.dumps({"status": "FAILED_OOM" if oom else "FAILED",
                          "error_type": type(exc).__name__, "error": str(exc),
                          "successful_profile": False}), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]),
                      "receipt": str(args.out.resolve() / "PROFILE.json"),
                      "quality": "NOT_EVALUATED", "patient_dice": "NOT_MEASURED"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
