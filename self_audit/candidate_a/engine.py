"""Candidate A stages behind the canonical train.py entry point.

Only dense budgeted TRAIN labels supervise this measurement-code pipeline.
"""
from pathlib import Path
import json
import time
import numpy as np
import torch
from self_audit.data import write_json
from self_audit.losses import aligned_flip
from self_audit.metrics import (
    ConfusionAccumulator,
    append_csv,
    append_jsonl,
    epoch_rows,
    validation_rows as _validation_rows,
)
from . import CLASS_MAPPING as CANDIDATE_CLASS_MAPPING, SCHEMA
from .config import validate_config
from .io import build_student, read_checkpoint, save_checkpoint, source_inventory


def _teacher(config, *, synthetic):
    from .teacher import build_teacher, set_teacher_stage
    c = config
    init = c["teacher_init"]
    if not synthetic and c["stage"] != "teacher_pretrain" and init["kind"] == "none":
        raise ValueError("real warm-up/joint requires a verified teacher/segmentation checkpoint")
    if not synthetic and c["stage"] == "joint" and init["kind"] != "candidate_a":
        raise ValueError("joint teacher proposals require a candidate-A teacher; warm up new branches first")
    payload = meta = None
    if init["kind"] != "none":
        payload, meta = read_checkpoint(init["path"], init["sha256"])
        if bool(meta["synthetic"]) != synthetic:
            raise ValueError("teacher checkpoint synthetic/patient mismatch")
    native_state = payload["student"] if init["kind"] == "native_v2" else None
    native_meta = meta if init["kind"] == "native_v2" else None
    teacher_config = {**c["model"], "class_mapping": c["class_mapping"]}
    teacher, provenance = build_teacher(teacher_config, synthetic=synthetic,
                                        native_state=native_state, native_metadata=native_meta)
    if init["kind"] == "candidate_a":
        if meta.get("class_mapping") != CANDIDATE_CLASS_MAPPING or payload.get("teacher") is None:
            raise ValueError("candidate-A teacher checkpoint/class mapping missing")
        if meta["config"]["bank"]["radius_mm"] != c["bank"]["radius_mm"]:
            raise ValueError("teacher SDF normalization radius differs; retrain explicitly")
        teacher.load_state_dict(payload["teacher"], strict=True)
        teacher.segmentation_ready = True
        provenance.update(segmentation_pretrained=True,
            pretrained_claim="loaded_supervised_candidate_a_teacher;accuracy_NOT_CERTIFIED",
            initialization_stage=meta["config"]["stage"])
    if meta is not None:
        provenance["initialization_manifest_sha256"] = init["sha256"]
        provenance["initialization_payload_sha256"] = meta["payload_sha256"]
    set_teacher_stage(teacher, c["stage"], freeze=c["freeze_teacher"])
    return teacher, provenance, meta


def _labeled_slice_keys(data) -> list[tuple[int, int, int]]:
    """Index all labeled TRAIN frame/slices without retaining tensor copies.

    A full ACDC/M&Ms cohort can contain many thousands of slices.  The plan
    therefore stores only ``(record_index, frame, slice)`` references; image
    and mask tensors are materialized one update at a time below.
    """

    return data.base.labeled_slice_keys()


def _epoch_plan(data, rng: np.random.Generator, *, keys=None) -> list[tuple[int, int, int]]:
    """Return every annotated TRAIN frame/slice once in deterministic order."""

    keys = _labeled_slice_keys(data) if keys is None else keys
    order = rng.permutation(len(keys))
    return [keys[int(index)] for index in order]


def _fixed_labeled_sample(data, planned):
    """Return a canonical planned sample and its in-plane spacing."""

    x, y, metadata = data.base.load_labeled_slice(planned)
    patient = metadata["patient"]
    spacing = data.spacing_yx_for(patient)
    sample = {**metadata, "spacing_yx": list(spacing)}
    return x, y, sample, spacing


def _metric_row_set(epoch: int, accumulators: dict[str, ConfusionAccumulator]) -> list[dict]:
    rows: list[dict] = []
    for branch, accumulator in accumulators.items():
        rows.extend(epoch_rows(accumulator.matrix, epoch=epoch, split="train_online", branch=branch))
    return rows


def _validate_epoch_manifest(path, expected_sha256, *, train_ids, synthetic, expected_dataset):
    """Validate development provenance before any optimizer update occurs."""

    # Keep one manifest/geometry/hash validator for standalone evaluation and
    # epoch validation.  Training is allowed to consume only development data;
    # ``external_frozen`` is reserved for the final, standalone evaluator.
    from .evaluation import _validated_eval_manifest
    manifest_path, manifest, *_ = _validated_eval_manifest(
        path, expected_sha256, train_ids=train_ids, synthetic=synthetic)
    if manifest.get("role") != "development":
        raise ValueError("epoch validation requires a development manifest")
    if not manifest.get("patients"):
        raise ValueError("epoch validation manifest must contain at least one patient")
    manifest_dataset = manifest.get("dataset")
    if synthetic:
        if manifest_dataset != "synthetic":
            raise ValueError("synthetic validation manifest must declare dataset=synthetic")
    elif manifest_dataset != expected_dataset:
        raise ValueError("validation dataset does not match the declared TRAIN benchmark")
    return manifest_path


def train(config, manifest, manifest_sha256, out, *, device="cuda", synthetic=False,
          validation_manifest=None, validation_sha256=None):
    from .data import CandidateAData
    from .k1 import K1Config, student_rollout
    from .teacher import teacher_loss
    c = validate_config(config)
    if c["validation_every"] and not (validation_manifest and validation_sha256):
        raise ValueError("validation_every > 0 requires --validation-data and --validation-data-sha256")
    if bool(validation_manifest) != bool(validation_sha256):
        raise ValueError("validation requires both manifest path and SHA256")
    if validation_manifest and not c["validation_every"]:
        raise ValueError("validation data was supplied while validation_every is disabled")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent device fallback")
    if synthetic and c["model"]["backbone"] != "tiny_test":
        raise ValueError("synthetic smoke must declare the unpretrained tiny_test teacher")
    data = CandidateAData(manifest, manifest_sha256, synthetic=synthetic)
    train_dataset = data.manifest.get("dataset")
    if synthetic:
        if train_dataset != "synthetic":
            raise ValueError("synthetic TRAIN manifest must declare dataset=synthetic")
    elif train_dataset != c["dataset"]["name"]:
        raise ValueError("TRAIN manifest dataset does not match the declared benchmark")
    if validation_manifest:
        _validate_epoch_manifest(validation_manifest, validation_sha256,
                                 train_ids=data.lids + data.uids, synthetic=synthetic,
                                 expected_dataset=c["dataset"]["name"])
    out = Path(out).resolve()
    if out.exists() or out.is_relative_to(data.path.parent) or data.path.parent.is_relative_to(out):
        raise ValueError("new output directory outside source data required")
    torch.set_num_threads(c["threads"])
    torch.manual_seed(c["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c["seed"])
    # The student initialization is identical with and without teacher. Teacher
    # initialization cannot consume the student or data/action random streams.
    student, bank, codec = build_student(c)
    student = student.to(device)
    teacher = None
    provenance = {"used": False}
    if c["use_teacher"]:
        with torch.random.fork_rng():
            torch.manual_seed(c["seed"] + 1000)
            teacher, provenance, init_meta = _teacher(c, synthetic=synthetic)
        if init_meta is not None:
            init_ids = set(init_meta.get("labeled_ids", [])) | set(init_meta.get("unlabeled_ids", []))
            if not init_ids.issubset(set(data.lids + data.uids)):
                raise ValueError("teacher initialization contains patients outside the declared TRAIN cohort")
            if not set(init_meta.get("labeled_ids", [])).issubset(set(data.lids)):
                raise ValueError("all teacher pretraining labels must count in this dense TRAIN budget")
            if (init_meta.get("data_sha256") != manifest_sha256
                    or init_meta.get("label_budget") != len(data.lids)
                    or set(init_meta.get("labeled_ids", [])) != set(data.lids)
                    or set(init_meta.get("unlabeled_ids", [])) != set(data.uids)):
                raise ValueError("teacher stages require the exact hashed TRAIN manifest and labeled/U budget")
        teacher = teacher.to(device)
    student_optimizer = torch.optim.AdamW(student.parameters(), lr=c["student_lr"], weight_decay=c["weight_decay"])
    teacher_parameters = [p for p in teacher.parameters() if p.requires_grad] if teacher is not None else []
    teacher_optimizer = torch.optim.AdamW(teacher_parameters, lr=c["teacher_lr"], weight_decay=c["weight_decay"]) if teacher_parameters else None
    rng = np.random.default_rng(c["seed"])
    generator = torch.Generator(device="cpu").manual_seed(c["rollout"]["seed"])
    rollout_config = K1Config(**c["rollout"])
    slice_keys = _labeled_slice_keys(data)
    out.mkdir(parents=True)
    write_json(out / "CONFIG.json", c)
    write_json(out / "STARTUP.json", {"schema": SCHEMA, "source_files": source_inventory(),
        "synthetic": synthetic, "device": device, "teacher": provenance,
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "teacher_parameters": sum(p.numel() for p in teacher.parameters()) if teacher is not None else 0,
        "label_budget": len(data.lids), "supervision": "budgeted dense TRAIN only; U unused",
        "distance_protocol": "2D in-plane physical distance; 3-slice input context",
        "quality": "NOT_EVALUATED"})
    start = time.perf_counter()
    counts = {"teacher_updates": 0, "student_updates": 0, "teacher_snapshots": 0}
    metric_jsonl = out / "metrics.jsonl"
    metric_csv = out / "metrics.csv"
    epoch_log = out / "epochs.jsonl"
    update_log = out / "updates.jsonl"
    global_update = 0
    try:
        with epoch_log.open("x") as epoch_stream, update_log.open("x") as update_stream:
            for epoch in range(1, c["epochs"] + 1):
                plan = _epoch_plan(data, rng, keys=slice_keys)
                accumulators = {}
                epoch_losses = []
                for update_index, plan_item in enumerate(plan):
                    x, y, sample, spacing = _fixed_labeled_sample(data, plan_item)
                    if c["augmentation"] == "aligned_flip":
                        x, y = aligned_flip(x, y, rng)
                    x, y = x.to(device), y.to(device)
                    geometry = {"shape": tuple(y.shape[-2:]), "spacing_yx": spacing}
                    targets = bank.targets(y, spacing) if c["stage"] == "joint" else None
                    row = {"epoch": epoch, "update": update_index + 1, "sample": sample}
                    teacher_metrics_recorded = False
                    if teacher_optimizer is not None and (c["stage"] != "joint" or global_update % c["teacher_update_every"] == 0):
                        teacher.train()
                        if c["stage"] == "teacher_warmup":
                            teacher.encoder.eval()
                        teacher_optimizer.zero_grad(set_to_none=True)
                        outputs = teacher(x)
                        lt = teacher_loss(outputs, y, bank.sdf_targets(y, spacing), c["lambda_distance"])
                        if not torch.isfinite(lt):
                            raise FloatingPointError("nonfinite teacher loss")
                        for branch, logits in (("teacher_region", outputs["region_logits"]),
                                               ("teacher_boundary", outputs["boundary_logits"])):
                            accumulators.setdefault(branch, ConfusionAccumulator.create())
                            accumulators[branch].update(y.detach().cpu().numpy(), logits.detach().argmax(1).cpu().numpy())
                        teacher_metrics_recorded = True
                        lt.backward()
                        torch.nn.utils.clip_grad_norm_(teacher_parameters, 5., error_if_nonfinite=True)
                        teacher_optimizer.step()
                        teacher_optimizer.zero_grad(set_to_none=True)
                        row["teacher_loss"] = float(lt.detach())
                        counts["teacher_updates"] += 1
                    if c["stage"] == "joint":
                        teacher_values = None
                        if teacher is not None:
                            teacher.eval()
                            # One fixed snapshot for the entire actual student rollout.
                            with torch.no_grad():
                                output = teacher(x)
                                if not teacher_metrics_recorded:
                                    accumulators.setdefault("teacher_region", ConfusionAccumulator.create())
                                    accumulators.setdefault("teacher_boundary", ConfusionAccumulator.create())
                                    accumulators["teacher_region"].update(y.detach().cpu().numpy(), output["region_logits"].argmax(1).cpu().numpy())
                                    accumulators["teacher_boundary"].update(y.detach().cpu().numpy(), output["boundary_logits"].argmax(1).cpu().numpy())
                                teacher_values = bank.teacher_values(output["region_logits"].softmax(1), output["sdf"]).detach()
                            counts["teacher_snapshots"] += 1
                        student.train()
                        student_optimizer.zero_grad(set_to_none=True)
                        result = student_rollout(student, x, bank, labels=y, codec=codec,
                            geometry=geometry, teacher_values=teacher_values, query_targets=targets,
                            config=rollout_config, generator=generator)
                        loss = (result.segmentation_loss + c["loss"]["region"] * result.region_query_loss
                                + c["loss"]["boundary"] * result.boundary_query_loss
                                + c["loss"]["policy"] * result.policy_loss)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite student loss")
                        loss.backward()
                        if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
                            raise RuntimeError("student loss reached teacher parameters")
                        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 5., error_if_nonfinite=True)
                        student_optimizer.step()
                        counts["student_updates"] += 1
                        row.update(student_loss=float(loss.detach()), gradient_norm=float(norm),
                            segmentation_loss=float(result.segmentation_loss.detach()),
                            region_query_loss=float(result.region_query_loss.detach()),
                            boundary_query_loss=float(result.boundary_query_loss.detach()),
                            policy_loss=float(result.policy_loss.detach()), code_bits=result.code_bits,
                            query_count=result.query_count, code_ids=list(result.code_ids))
                        accumulators.setdefault("student", ConfusionAccumulator.create())
                        accumulators["student"].update(y.detach().cpu().numpy(), result.final_logits.detach().argmax(1).cpu().numpy())
                        epoch_losses.append(float(loss.detach()))
                    row["elapsed_seconds"] = time.perf_counter() - start
                    # Per-update data are useful for debugging; per-epoch metrics
                    # below are the comparison artifact.
                    update_stream.write(json.dumps(row, allow_nan=False) + "\n")
                    update_stream.flush()
                    global_update += 1
                rows = _metric_row_set(epoch, accumulators)
                append_jsonl(metric_jsonl, rows)
                append_csv(metric_csv, rows)
                epoch_record = {"epoch": epoch, "updates": len(plan),
                    "mean_student_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                    "metric_rows": len(rows), "metrics_file": metric_csv.name,
                    "elapsed_seconds": time.perf_counter() - start}
                if validation_manifest and epoch % c["validation_every"] == 0:
                    validation_out = out / "validation" / f"epoch_{epoch:04d}"
                    if c["stage"] == "joint":
                        from .evaluation import evaluate_epoch
                        validation = evaluate_epoch(
                            student, bank, codec, rollout_config, validation_manifest,
                            validation_sha256, validation_out, epoch=epoch, device=device,
                            train_ids=data.lids + data.uids, synthetic=synthetic)
                        validation_rows = _validation_rows(validation, epoch=epoch, branch="student")
                        validation_summary = {"per_class": validation.get("per_class", {})}
                    else:
                        # A student-only validation score would be misleading here:
                        # teacher_pretrain and teacher_warmup have not optimized the
                        # student yet.  Score both actual teacher branches instead.
                        from .evaluation import evaluate_teacher_epoch
                        validation = evaluate_teacher_epoch(
                            teacher, validation_manifest, validation_sha256, validation_out,
                            epoch=epoch, device=device, train_ids=data.lids + data.uids,
                            synthetic=synthetic)
                        validation_rows = []
                        for branch, branch_result in validation.get("branches", {}).items():
                            validation_rows.extend(_validation_rows(
                                branch_result, epoch=epoch, branch=f"teacher_{branch}"))
                        validation_summary = {"branches": validation.get("branches", {})}
                    append_jsonl(metric_jsonl, validation_rows)
                    append_csv(metric_csv, validation_rows)
                    epoch_record["validation"] = {"directory": str(validation_out),
                        **validation_summary}
                epoch_stream.write(json.dumps(epoch_record, allow_nan=False) + "\n")
                epoch_stream.flush()
                if c.get("checkpoint_every_epochs", 1) and epoch % c["checkpoint_every_epochs"] == 0:
                    optimizers = [student_optimizer] + ([teacher_optimizer] if teacher_optimizer is not None else [])
                    save_checkpoint(out / f"epoch_{epoch:04d}", student=student, teacher=teacher,
                                    optimizers=optimizers, config=c, data=data, epoch=epoch,
                                    updates=global_update,
                                    provenance=provenance)
        optimizers = [student_optimizer] + ([teacher_optimizer] if teacher_optimizer is not None else [])
        sha = save_checkpoint(out / "final", student=student, teacher=teacher, optimizers=optimizers,
                              config=c, data=data, epoch=c["epochs"], updates=global_update, provenance=provenance)
        report = {"schema": SCHEMA, "status": "COMPLETED", "synthetic": synthetic,
            "quality": "NOT_EVALUATED", "stage": c["stage"], "epochs": c["epochs"], **counts,
            "label_budget": len(data.lids), "label_reads": sorted(data.label_reads),
            "unlabeled_usage": "UNUSED; no U reference paths opened",
            "metric_files": {"epoch": str(epoch_log.name), "per_class_jsonl": str(metric_jsonl.name),
                             "per_class_csv": str(metric_csv.name), "updates": str(update_log.name)},
            "validation_manifest_sha256": validation_sha256,
            "final_manifest_sha256": sha, "elapsed_seconds": time.perf_counter() - start}
        write_json(out / "REPORT.json", report)
        return report
    except Exception as exc:
        write_json(out / "FAILURE.json", {"status": "FAILED", "error": repr(exc), **counts})
        raise
