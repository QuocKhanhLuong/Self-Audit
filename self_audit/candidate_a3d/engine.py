"""Epoch-based native full-volume training for the Candidate-A 3-D track.

The 3-D runner intentionally mirrors the tested Candidate-A lifecycle while
keeping its protocol separate: one annotated TRAIN frame is one update, the
input is the complete native ``[1,1,D,H,W]`` volume, and geometry is carried
explicitly to every measurement/decoder operation.  No validation mask is
opened before the epoch validation preflight and no teacher tensor is passed
to the student decoder.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from self_audit.data import write_json
from self_audit.metrics import (
    ConfusionAccumulator,
    append_csv,
    append_jsonl,
    epoch_rows,
    validation_rows as _validation_rows,
)

from . import SCHEMA
from .config import validate_config
from .geometry import canonical_geometry
from .io import build_student, read_checkpoint, save_checkpoint, source_inventory


def _sample_parts(data: Any, key: Any) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """Load one native full-volume sample through the data-agent API."""

    image, labels, geometry, metadata = data.load_labeled_frame(key)
    image = torch.as_tensor(image, dtype=torch.float32)
    labels = torch.as_tensor(labels)
    if image.ndim != 5 or tuple(image.shape[:2]) != (1, 1):
        raise ValueError("A-3D input must have shape [1,1,D,H,W]")
    if labels.ndim != 4 or labels.shape[0] != 1:
        raise ValueError("A-3D labels must have shape [1,D,H,W]")
    if not torch.isfinite(image).all():
        raise ValueError("A-3D input contains NaN or Inf")
    geometry = canonical_geometry(geometry)
    if tuple(image.shape[-3:]) != tuple(geometry["shape"]):
        raise ValueError("A-3D sample shape does not match native full-volume geometry")
    if tuple(labels.shape[-3:]) != tuple(geometry["shape"]):
        raise ValueError("A-3D label shape does not match native full-volume geometry")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    return image, labels.long(), geometry, metadata


def _validate_teacher_output(output: Any, shape: tuple[int, int, int]) -> dict[str, torch.Tensor]:
    required = {"region_logits", "boundary_logits", "sdf"}
    if not isinstance(output, Mapping) or set(output) != required:
        raise ValueError("A-3D teacher must return exactly region_logits, boundary_logits and sdf")
    expected = (1, 4, *shape)
    result = {}
    for name in ("region_logits", "boundary_logits", "sdf"):
        value = output[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"A-3D teacher {name} must have shape {expected}")
        if not torch.isfinite(value).all():
            raise ValueError(f"A-3D teacher {name} contains NaN or Inf")
        result[name] = value
    return result


def _teacher(config: dict, *, synthetic: bool):
    from .teacher import build_teacher3d, set_teacher_stage

    init = config["teacher_init"]
    if not synthetic and config["stage"] != "teacher_pretrain" and init["kind"] == "none":
        raise ValueError("real A-3D warm-up/joint requires a verified teacher checkpoint")
    payload = metadata = None
    if init["kind"] != "none":
        payload, metadata = read_checkpoint(init["path"], init["sha256"])
        if bool(metadata.get("synthetic")) != bool(synthetic):
            raise ValueError("teacher checkpoint synthetic/patient identity mismatch")
    state = None
    if init["kind"] == "candidate_a3d":
        state = payload.get("teacher")
        if state is None:
            raise ValueError("candidate-A3D teacher checkpoint has no teacher state")
    elif init["kind"] != "none":
        raise ValueError(f"unsupported A-3D teacher initialization kind: {init['kind']}")
    teacher_config = {**config["model"], "class_mapping": config["class_mapping"]}
    teacher, provenance = build_teacher3d(
        teacher_config, synthetic=synthetic,
    )
    if state is not None:
        # The checkpoint is loaded only after the adapter has been built from
        # the declared model config.  Do not rewrite metadata to satisfy the
        # adapter: shared checkpoint metadata is validated below as-is.
        if metadata.get("model_class") not in {"CandidateA3DStudent", "DualBranchTeacher3D"}:
            raise ValueError("A-3D teacher checkpoint model class is not recognized")
        mapping = metadata.get("class_mapping")
        if mapping not in ({"0": "background", "1": "RV", "2": "MYO", "3": "LV", "255": "ignore"},
                          ["BG", "RV", "MYO", "LV"]):
            raise ValueError("A-3D teacher checkpoint class mapping mismatch")
        teacher.load_state_dict(state, strict=True)
        provenance.update(quality="TRAINED_CHECKPOINT", accuracy_claim="NOT_MEASURED")
    set_teacher_stage(teacher, config["stage"], freeze=config["freeze_teacher"])
    if metadata is not None:
        provenance["initialization_manifest_sha256"] = init["sha256"]
        provenance["initialization_payload_sha256"] = metadata["payload_sha256"]
    return teacher, provenance, metadata


def _validate_epoch_manifest(path, expected_sha256, *, train_ids, synthetic, expected_dataset):
    """Validate frozen development provenance before any optimizer update."""

    from .evaluation import _validated_manifest

    manifest_path, manifest, *_ = _validated_manifest(
        path, expected_sha256, train_ids=train_ids, synthetic=synthetic)
    if manifest.get("role") != "development":
        raise ValueError("A-3D epoch validation requires a development manifest")
    if not manifest.get("patients"):
        raise ValueError("A-3D epoch validation manifest must contain at least one patient")
    dataset = manifest.get("dataset")
    if synthetic:
        if dataset != "synthetic":
            raise ValueError("synthetic A-3D validation must declare dataset=synthetic")
    elif dataset != expected_dataset:
        raise ValueError("A-3D validation dataset does not match the declared benchmark")
    return manifest_path


def _metric_rows(epoch: int, accumulators: dict[str, ConfusionAccumulator]) -> list[dict]:
    rows: list[dict] = []
    for branch, accumulator in accumulators.items():
        rows.extend(epoch_rows(accumulator.matrix, epoch=epoch,
                               split="train_online", branch=branch))
    return rows


def _teacher_values(bank, output):
    return bank.teacher_values(
        output["region_logits"].softmax(1), output["sdf"]
    ).detach()


def _validate_pretraining_provenance(provenance: Mapping[str, Any], data: Any,
                                     manifest_sha256: str) -> None:
    """Bind external segmentation weights to this exact dense TRAIN cohort."""

    backbone = provenance.get("backbone", {})
    metadata = backbone.get("pretraining_data") if isinstance(backbone, Mapping) else None
    if metadata is None:
        return
    if not isinstance(metadata, Mapping):
        raise TypeError("teacher pretraining provenance must be a mapping")
    required = {"labeled_ids", "unlabeled_ids", "label_budget", "data_sha256", "class_mapping"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"teacher pretraining provenance missing {missing}")
    labeled = {str(value) for value in metadata["labeled_ids"]}
    unlabeled = {str(value) for value in metadata["unlabeled_ids"]}
    if labeled & unlabeled or labeled != set(data.lids) or unlabeled != set(data.uids):
        raise ValueError("teacher pretraining IDs do not match the declared TRAIN cohort")
    if metadata["label_budget"] != len(data.lids) or metadata["data_sha256"] != manifest_sha256:
        raise ValueError("teacher pretraining weights are not bound to this TRAIN manifest")
    mapping = metadata["class_mapping"]
    if isinstance(mapping, Mapping):
        mapping = [mapping.get(str(index), mapping.get(index)) for index in range(4)]
    normalized = tuple(str(value).strip().upper() for value in mapping)
    if normalized != ("BG", "RV", "MYO", "LV"):
        raise ValueError("teacher pretraining class mapping does not match the native ontology")


def _repeat_shuffled(items, count, rng):
    """Balanced cycles; every item is visited before a new cycle starts."""
    result = []
    while len(result) < count:
        result.extend(items[int(i)] for i in rng.permutation(len(items)))
    return result[:count]


def _semi_epoch_plan(frame_keys, unlabeled_keys, steps, rng, visits):
    """Use TRAIN images only; U frame selection never depends on annotations."""
    by_patient = {}
    for record, frame in unlabeled_keys:
        by_patient.setdefault(record, []).append((record, frame))
    if not by_patient or steps < max(len(frame_keys), len(by_patient)):
        raise ValueError("semi epoch must cover all labeled frames and every U patient")
    labeled = _repeat_shuffled(frame_keys, steps, rng)
    patients = _repeat_shuffled(sorted(by_patient), steps, rng)
    unlabeled = []
    for patient in patients:
        frames = by_patient[patient]
        unlabeled.append(frames[visits.get(patient, 0) % len(frames)])
        visits[patient] = visits.get(patient, 0) + 1
    return labeled, unlabeled


def _unlabeled_update(*, data, key, student, student_ema, student_optimizer,
                      teacher, teacher_ema, teacher_optimizer, teacher_parameters,
                      update_teacher, bank, codec, rollout_config, config, epoch,
                      device, augmentation_generator, rollout_generator, counts):
    """Image-only consistency; no label loader, GT target or K1 utility path."""
    from .k1 import student_rollout
    from .semisupervised import (
        ramp_weight, strong_intensity, student_unlabeled_loss,
        teacher_consistency_loss, update_ema,
    )

    semi = config["semi_supervised"]
    image, geometry, sample = data.load_unlabeled_frame(key)
    strong = strong_intensity(image, semi, generator=augmentation_generator).to(device)
    weak = image.to(device)
    weight = ramp_weight(epoch, semi)
    row = {"sample": sample, "ramp_weight": weight}
    counts["unlabeled_samples"] += 1
    if update_teacher and teacher_optimizer is not None:
        teacher_ema.eval()
        with torch.no_grad():
            target = _validate_teacher_output(teacher_ema(weak, geometry=geometry), tuple(weak.shape[-3:]))
        teacher.train()
        if config["stage"] == "teacher_warmup":
            teacher.encoder.eval()
        teacher_optimizer.zero_grad(set_to_none=True)
        online = _validate_teacher_output(teacher(strong, geometry=geometry), tuple(strong.shape[-3:]))
        raw_loss, stats = teacher_consistency_loss(online, target, semi)
        loss = weight * raw_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite teacher U consistency loss")
        applied = float(loss.detach()) > 0.0
        if applied:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher_parameters, 5., error_if_nonfinite=True)
            teacher_optimizer.step()
            update_ema(teacher_ema, teacher, semi["ema_decay"])
            counts["teacher_unlabeled_updates"] += 1
        teacher_optimizer.zero_grad(set_to_none=True)
        row.update(teacher_unlabeled_loss=float(loss.detach()),
                   teacher_unlabeled_update=applied, teacher_consistency=stats)
        del loss, raw_loss, online, target
    if config["stage"] == "joint":
        # Both passes select from the feasible bank without GT or mentor values.
        # Random-query controls use the same random stream for weak/strong views.
        random_state = rollout_generator.get_state()
        student_ema.eval()
        with torch.no_grad():
            target = student_rollout(
                student_ema, weak, bank, codec=codec, geometry=geometry,
                config=rollout_config, generator=rollout_generator,
            )
        rollout_generator.set_state(random_state)
        student.train()
        student_optimizer.zero_grad(set_to_none=True)
        online = student_rollout(
            student, strong, bank, codec=codec, geometry=geometry,
            config=rollout_config, generator=rollout_generator, quantize_ste=True,
        )
        raw_loss, stats = student_unlabeled_loss(online.final_logits, target.final_logits, semi)
        loss = weight * raw_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite student U consistency loss")
        applied = float(loss.detach()) > 0.0
        if applied:
            loss.backward()
            if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
                raise RuntimeError("student U loss reached main teacher parameters")
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5., error_if_nonfinite=True)
            student_optimizer.step()
            update_ema(student_ema, student, semi["ema_decay"])
            counts["student_unlabeled_updates"] += 1
        student_optimizer.zero_grad(set_to_none=True)
        if any(trace.target_probs is not None for trace in online.traces):
            raise RuntimeError("U rollout unexpectedly created a policy utility target")
        row.update(student_unlabeled_loss=float(loss.detach()),
                   student_unlabeled_update=applied, student_consistency=stats,
                   student_code_bits=int(online.code_bits), student_query_count=int(online.query_count),
                   policy_targets=0, ground_truth_reads=0)
    return row


def train(config, manifest, manifest_sha256, out, *, device="cuda", synthetic=False,
          validation_manifest=None, validation_sha256=None):
    from .data import CandidateA3DData
    from .k1 import K1Config, student_rollout
    from .teacher import teacher_loss
    from .semisupervised import make_ema, strong_intensity, update_ema

    config = validate_config(config)
    if config["validation_every"] and not (validation_manifest and validation_sha256):
        raise ValueError("validation_every > 0 requires validation manifest and SHA256")
    if bool(validation_manifest) != bool(validation_sha256):
        raise ValueError("validation requires both manifest path and SHA256")
    if validation_manifest and not config["validation_every"]:
        raise ValueError("validation data was supplied while validation_every is disabled")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent device fallback")
    if synthetic and config["model"]["backbone"] != "tiny_test3d":
        raise ValueError("synthetic A-3D smoke must use tiny_test3d")

    data = CandidateA3DData(manifest, manifest_sha256, synthetic=synthetic)
    semi = config.get("semi_supervised", {})
    semi_enabled = semi.get("enabled", False)
    if semi_enabled and not data.uids:
        raise ValueError("semi-supervised training requires image-only U patients")
    dataset = data.manifest.get("dataset")
    if synthetic:
        if dataset != "synthetic":
            raise ValueError("synthetic A-3D TRAIN manifest must declare dataset=synthetic")
    elif dataset != config["dataset"]["name"]:
        raise ValueError("A-3D TRAIN dataset does not match the declared benchmark")
    if validation_manifest:
        _validate_epoch_manifest(
            validation_manifest, validation_sha256,
            train_ids=data.lids + data.uids, synthetic=synthetic,
            expected_dataset=config["dataset"]["name"],
        )

    out = Path(out).resolve()
    if out.exists() or out.is_relative_to(data.path.parent) or data.path.parent.is_relative_to(out):
        raise ValueError("new A-3D output directory must be outside source data")
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    student, bank, codec = build_student(config)
    student = student.to(device)
    teacher = None
    provenance = {"used": False}
    init_metadata = None
    if config["use_teacher"]:
        with torch.random.fork_rng():
            torch.manual_seed(config["seed"] + 1000)
            teacher, provenance, init_metadata = _teacher(config, synthetic=synthetic)
        if init_metadata is not None:
            init_ids = set(init_metadata.get("labeled_ids", [])) | set(init_metadata.get("unlabeled_ids", []))
            if not init_ids.issubset(set(data.lids + data.uids)):
                raise ValueError("teacher checkpoint contains patients outside TRAIN")
            if (init_metadata.get("data_sha256") != manifest_sha256
                    or init_metadata.get("label_budget") != len(data.lids)
                    or set(init_metadata.get("labeled_ids", [])) != set(data.lids)
                    or set(init_metadata.get("unlabeled_ids", [])) != set(data.uids)):
                raise ValueError("teacher stages require the exact hashed TRAIN manifest and label budget")
        _validate_pretraining_provenance(provenance, data, manifest_sha256)
        teacher = teacher.to(device)

    student_optimizer = torch.optim.AdamW(
        student.parameters(), lr=config["student_lr"], weight_decay=config["weight_decay"])
    teacher_parameters = [p for p in teacher.parameters() if p.requires_grad] if teacher is not None else []
    teacher_optimizer = (torch.optim.AdamW(
        teacher_parameters, lr=config["teacher_lr"], weight_decay=config["weight_decay"])
        if teacher_parameters else None)
    rng = np.random.default_rng(config["seed"])
    generator = torch.Generator(device="cpu").manual_seed(config["rollout"]["seed"])
    rollout_config = K1Config(**config["rollout"])
    frame_keys = list(data.labeled_frame_keys())
    if not frame_keys:
        raise ValueError("A-3D TRAIN manifest has no annotated frames")

    unlabeled_keys = list(data.unlabeled_frame_keys()) if semi_enabled else []
    semi_steps = ((semi["steps_per_epoch"] or max(len(frame_keys), len(data.lids) + len(data.uids)))
                  if semi_enabled else len(frame_keys))
    if semi_enabled and semi_steps < max(len(frame_keys), len(data.uids)):
        raise ValueError("semi steps_per_epoch must cover all labeled frames and U patients")
    teacher_ema = make_ema(teacher, semi["ema_decay"]) if semi_enabled and teacher_optimizer is not None else None
    student_ema = make_ema(student, semi["ema_decay"]) if semi_enabled and config["stage"] == "joint" else None
    augmentation_generator = torch.Generator(device="cpu").manual_seed(config["seed"] + 2000)
    unlabeled_generator = torch.Generator(device="cpu").manual_seed(config["seed"] + 3000)
    u_visits = {}

    def training_state():
        if not semi_enabled:
            return None
        return {"student_ema": student_ema.state_dict() if student_ema is not None else None,
                "teacher_ema": teacher_ema.state_dict() if teacher_ema is not None else None,
                "ema_initialization": "online weights at start of each stage; no resume",
                "semi_updates": dict(counts), "unlabeled_frame_visits": dict(u_visits),
                "augmentation_rng": augmentation_generator.get_state(),
                "unlabeled_rollout_rng": unlabeled_generator.get_state()}

    out.mkdir(parents=True)
    write_json(out / "CONFIG.json", config)
    write_json(out / "STARTUP.json", {
        "schema": SCHEMA, "source_files": source_inventory(),
        "synthetic": synthetic, "device": device, "teacher": provenance,
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "teacher_parameters": sum(p.numel() for p in teacher.parameters()) if teacher is not None else 0,
        "label_budget": len(data.lids),
        "supervision": ("semi-supervised: dense L labels plus image-only U EMA consistency"
                        if semi_enabled else "dense labeled TRAIN frames only; U unused"),
        "unlabeled_patient_count": len(data.uids), "updates_per_epoch": semi_steps,
        "semi_supervised": semi if semi_enabled else {"enabled": False},
        "unlabeled_policy_targets": "NONE; K1 utility and policy targets use L only",
        "ema_stage_initialization": "reset from online stage initialization",
        "geometry_protocol": "native full_volume [D,H,W], spacing_zyx and affine_zyx",
        "quality": "NOT_EVALUATED",
    })

    start = time.perf_counter()
    counts = {"teacher_updates": 0, "student_updates": 0, "teacher_snapshots": 0,
              "unlabeled_samples": 0, "teacher_unlabeled_updates": 0, "student_unlabeled_updates": 0}
    metric_jsonl, metric_csv = out / "metrics.jsonl", out / "metrics.csv"
    epoch_log, update_log = out / "epochs.jsonl", out / "updates.jsonl"
    global_update = 0
    try:
        with epoch_log.open("x") as epoch_stream, update_log.open("x") as update_stream:
            for epoch in range(1, config["epochs"] + 1):
                if semi_enabled:
                    plan, u_plan = _semi_epoch_plan(frame_keys, unlabeled_keys, semi_steps, rng, u_visits)
                else:
                    order = rng.permutation(len(frame_keys))
                    plan = [frame_keys[int(index)] for index in order]
                    u_plan = []
                accumulators: dict[str, ConfusionAccumulator] = {}
                epoch_losses: list[float] = []
                rollout_stats: list[dict[str, float]] = []
                u_rows = []
                epoch_teacher_losses = []
                for update_index, key in enumerate(plan):
                    x, y, geometry, sample = _sample_parts(data, key)
                    if semi_enabled:
                        x = strong_intensity(x, semi, generator=augmentation_generator)
                    x, y = x.to(device), y.to(device)
                    shape = tuple(int(v) for v in x.shape[-3:])
                    row = {"epoch": epoch, "update": update_index + 1,
                           "sample": sample, "geometry": geometry}
                    teacher_metrics_recorded = False
                    update_teacher = (teacher_optimizer is not None and (
                        config["stage"] != "joint" or global_update % config["teacher_update_every"] == 0))
                    if update_teacher:
                        teacher.train()
                        if config["stage"] == "teacher_warmup":
                            teacher.encoder.eval()
                        teacher_optimizer.zero_grad(set_to_none=True)
                        outputs = _validate_teacher_output(teacher(x, geometry=geometry), shape)
                        sdf_target = bank.sdf_targets(y, geometry["spacing_zyx"])
                        teacher_total = teacher_loss(outputs, y, sdf_target, config["lambda_distance"])
                        if not torch.isfinite(teacher_total):
                            raise FloatingPointError("nonfinite A-3D teacher loss")
                        for branch, logits in (("teacher_region", outputs["region_logits"]),
                                               ("teacher_boundary", outputs["boundary_logits"])):
                            accumulators.setdefault(branch, ConfusionAccumulator.create())
                            accumulators[branch].update(
                                y.detach().cpu().numpy(), logits.detach().argmax(1).cpu().numpy())
                        teacher_metrics_recorded = True
                        teacher_total.backward()
                        torch.nn.utils.clip_grad_norm_(teacher_parameters, 5., error_if_nonfinite=True)
                        teacher_optimizer.step()
                        if teacher_ema is not None:
                            update_ema(teacher_ema, teacher, semi["ema_decay"])
                        teacher_optimizer.zero_grad(set_to_none=True)
                        row["teacher_loss"] = float(teacher_total.detach())
                        counts["teacher_updates"] += 1
                        epoch_teacher_losses.append(row["teacher_loss"])
                        del teacher_total, sdf_target

                    if config["stage"] == "joint":
                        teacher_values = None
                        if teacher is not None:
                            teacher.eval()
                            with torch.no_grad():
                                outputs = _validate_teacher_output(teacher(x, geometry=geometry), shape)
                                if not teacher_metrics_recorded:
                                    for branch, logits in (("teacher_region", outputs["region_logits"]),
                                                           ("teacher_boundary", outputs["boundary_logits"])):
                                        accumulators.setdefault(branch, ConfusionAccumulator.create())
                                        accumulators[branch].update(
                                            y.detach().cpu().numpy(), logits.detach().argmax(1).cpu().numpy())
                                teacher_values = _teacher_values(bank, outputs)
                            counts["teacher_snapshots"] += 1
                        targets = bank.targets(y, geometry["spacing_zyx"])
                        student.train()
                        student_optimizer.zero_grad(set_to_none=True)
                        result = student_rollout(
                            student, x, bank, labels=y, codec=codec,
                            geometry=geometry, teacher_values=teacher_values,
                            query_targets=targets, config=rollout_config,
                            generator=generator,
                        )
                        loss = (result.segmentation_loss
                                + config["loss"]["region"] * result.region_query_loss
                                + config["loss"]["boundary"] * result.boundary_query_loss
                                + config["loss"]["policy"] * result.policy_loss)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite A-3D student loss")
                        loss.backward()
                        if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
                            raise RuntimeError("A-3D student loss reached teacher parameters")
                        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 5., error_if_nonfinite=True)
                        student_optimizer.step()
                        if student_ema is not None:
                            update_ema(student_ema, student, semi["ema_decay"])
                        student_optimizer.zero_grad(set_to_none=True)
                        counts["student_updates"] += 1
                        traces = tuple(result.traces)
                        region_count = sum(
                            bank.query(query_id).kind == "REGION" for query_id in result.code_ids)
                        boundary_count = sum(
                            bank.query(query_id).kind == "BOUNDARY" for query_id in result.code_ids)
                        stop_selected = any(int(trace.selected_id) == -1 for trace in traces)
                        candidate_trials = sum(len(trace.candidate_ids) for trace in traces)
                        row.update(
                            student_loss=float(loss.detach()), gradient_norm=float(norm),
                            segmentation_loss=float(result.segmentation_loss.detach()),
                            region_query_loss=float(result.region_query_loss.detach()),
                            boundary_query_loss=float(result.boundary_query_loss.detach()),
                            policy_loss=float(result.policy_loss.detach()),
                            code_bits=int(result.code_bits), query_count=int(result.query_count),
                            code_ids=list(result.code_ids), region_query_count=region_count,
                            boundary_query_count=boundary_count, stop_selected=stop_selected,
                            rollout_steps=len(traces), candidate_trials=candidate_trials,
                        )
                        rollout_stats.append({
                            "code_bits": float(result.code_bits),
                            "query_count": float(result.query_count),
                            "stop_selected": float(stop_selected),
                            "rollout_steps": float(len(traces)),
                            "candidate_trials": float(candidate_trials),
                        })
                        accumulators.setdefault("student", ConfusionAccumulator.create())
                        accumulators["student"].update(
                            y.detach().cpu().numpy(), result.final_logits.detach().argmax(1).cpu().numpy())
                        epoch_losses.append(float(loss.detach()))
                        del result, loss, targets, traces
                    if teacher is not None:
                        del outputs
                    del x, y
                    if semi_enabled:
                        u_row = _unlabeled_update(
                            data=data, key=u_plan[update_index], student=student, student_ema=student_ema,
                            student_optimizer=student_optimizer, teacher=teacher, teacher_ema=teacher_ema,
                            teacher_optimizer=teacher_optimizer, teacher_parameters=teacher_parameters,
                            update_teacher=update_teacher, bank=bank, codec=codec, rollout_config=rollout_config,
                            config=config, epoch=epoch, device=device, augmentation_generator=augmentation_generator,
                            rollout_generator=unlabeled_generator, counts=counts)
                        row["unlabeled"] = u_row
                        u_rows.append(u_row)
                    row["elapsed_seconds"] = time.perf_counter() - start
                    update_stream.write(json.dumps(row, allow_nan=False, default=str) + "\n")
                    update_stream.flush()
                    global_update += 1

                rows = _metric_rows(epoch, accumulators)
                append_jsonl(metric_jsonl, rows)
                append_csv(metric_csv, rows)
                epoch_record = {
                    "epoch": epoch, "updates": len(plan),
                    "mean_student_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                    "mean_teacher_loss": float(np.mean(epoch_teacher_losses)) if epoch_teacher_losses else None,
                    "mean_code_bits": float(np.mean([item["code_bits"] for item in rollout_stats])) if rollout_stats else None,
                    "mean_query_count": float(np.mean([item["query_count"] for item in rollout_stats])) if rollout_stats else None,
                    "stop_fraction": float(np.mean([item["stop_selected"] for item in rollout_stats])) if rollout_stats else None,
                    "mean_rollout_steps": float(np.mean([item["rollout_steps"] for item in rollout_stats])) if rollout_stats else None,
                    "mean_candidate_trials": float(np.mean([item["candidate_trials"] for item in rollout_stats])) if rollout_stats else None,
                    "metric_rows": len(rows), "metrics_file": metric_csv.name,
                    "elapsed_seconds": time.perf_counter() - start,
                }
                if semi_enabled:
                    epoch_record["semi_supervised"] = {
                        "unlabeled_samples": len(u_rows),
                        "unlabeled_patients": sorted({item["sample"]["patient"] for item in u_rows}),
                        "labeled_frame_visits": len(plan), "unique_labeled_frames": len(set(plan)),
                        "teacher_unlabeled_updates": sum(item.get("teacher_unlabeled_update", False) for item in u_rows),
                        "student_unlabeled_updates": sum(item.get("student_unlabeled_update", False) for item in u_rows),
                    }
                    for name in ("teacher_unlabeled_loss", "student_unlabeled_loss"):
                        values = [item[name] for item in u_rows if name in item]
                        epoch_record["semi_supervised"]["mean_" + name] = float(np.mean(values)) if values else None
                    if set(epoch_record["semi_supervised"]["unlabeled_patients"]) != set(data.uids):
                        raise RuntimeError("semi epoch failed to visit every U patient")
                if validation_manifest and epoch % config["validation_every"] == 0:
                    validation_out = out / "validation" / f"epoch_{epoch:04d}"
                    if config["stage"] == "joint":
                        from .evaluation import evaluate_epoch
                        validation = evaluate_epoch(
                            student, bank, codec, rollout_config,
                            validation_manifest, validation_sha256, validation_out,
                            epoch=epoch, device=device,
                            train_ids=data.lids + data.uids, synthetic=synthetic,
                        )
                        validation_rows = _validation_rows(validation, epoch=epoch, branch="student")
                        summary = {"per_class": validation.get("per_class", {})}
                    else:
                        from .evaluation import evaluate_teacher_epoch
                        validation = evaluate_teacher_epoch(
                            teacher, validation_manifest, validation_sha256, validation_out,
                            epoch=epoch, device=device,
                            train_ids=data.lids + data.uids, synthetic=synthetic,
                        )
                        validation_rows = []
                        for branch, branch_result in validation.get("branches", {}).items():
                            validation_rows.extend(_validation_rows(
                                branch_result, epoch=epoch, branch=f"teacher_{branch}"))
                        summary = {"branches": validation.get("branches", {})}
                    append_jsonl(metric_jsonl, validation_rows)
                    append_csv(metric_csv, validation_rows)
                    epoch_record["validation"] = {"directory": str(validation_out), **summary}
                epoch_stream.write(json.dumps(epoch_record, allow_nan=False, default=str) + "\n")
                epoch_stream.flush()
                if config.get("checkpoint_every_epochs", 1) and epoch % config["checkpoint_every_epochs"] == 0:
                    save_checkpoint(
                        out / f"epoch_{epoch:04d}", student=student, teacher=teacher,
                        optimizers=[student_optimizer] + ([teacher_optimizer] if teacher_optimizer else []),
                        config=config, data=data, epoch=epoch, updates=global_update,
                        provenance=provenance, training_state=training_state(),
                    )
        sha = save_checkpoint(
            out / "final", student=student, teacher=teacher,
            optimizers=[student_optimizer] + ([teacher_optimizer] if teacher_optimizer else []),
            config=config, data=data, epoch=config["epochs"], updates=global_update,
            provenance=provenance, training_state=training_state(),
        )
        report = {
            "schema": SCHEMA, "status": "COMPLETED", "synthetic": synthetic,
            "quality": "NOT_EVALUATED", "stage": config["stage"],
            "epochs": config["epochs"], **counts,
            "label_budget": len(data.lids), "label_reads": sorted(data.label_reads),
            "unlabeled_usage": ("TRAIN images only; self-EMA consistency; no U masks or K1 utility"
                                if semi_enabled else "UNUSED; no U reference paths opened"),
            "unlabeled_image_reads": sorted(data.unlabeled_image_reads),
            "semi_supervised": semi if semi_enabled else {"enabled": False},
            "updates_per_epoch": semi_steps,
            "metric_files": {"epoch": epoch_log.name, "per_class_jsonl": metric_jsonl.name,
                             "per_class_csv": metric_csv.name, "updates": update_log.name},
            "validation_manifest_sha256": validation_sha256,
            "final_manifest_sha256": sha, "elapsed_seconds": time.perf_counter() - start,
        }
        write_json(out / "REPORT.json", report)
        return report
    except Exception as exc:
        write_json(out / "FAILURE.json", {"status": "FAILED", "error": repr(exc), **counts})
        raise


__all__ = ["train"]
