"""Candidate A stages behind the canonical train.py entry point.

This is a separately declared experimental factor, not a replacement for the
native v2 audit transaction. Only dense budgeted TRAIN labels supervise A.
"""
from pathlib import Path
import json
import time
import numpy as np
import torch
from self_audit.data import write_json
from self_audit.losses import aligned_flip
from . import CLASS_MAPPING, SCHEMA
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
        if meta.get("class_mapping") != CLASS_MAPPING or payload.get("teacher") is None:
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


def train(config, manifest, manifest_sha256, out, *, device="cuda", synthetic=False):
    from .data import CandidateAData
    from .k1 import K1Config, student_rollout
    from .teacher import teacher_loss
    c = validate_config(config)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent device fallback")
    if synthetic and c["model"]["backbone"] != "tiny_test":
        raise ValueError("synthetic smoke must declare the unpretrained tiny_test teacher")
    data = CandidateAData(manifest, manifest_sha256, synthetic=synthetic)
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
    try:
        with (out / "steps.jsonl").open("x") as log:
            for step in range(c["steps"]):
                x, y, sample = data.base.sample_labeled(step % len(data.lids), rng)
                spacing = data.spacing_yx_for(sample["patient"])
                sample["spacing_yx"] = list(spacing)
                if c["augmentation"] == "aligned_flip":
                    x, y = aligned_flip(x, y, rng)
                x, y = x.to(device), y.to(device)
                geometry = {"shape": tuple(y.shape[-2:]), "spacing_yx": spacing}
                targets = bank.targets(y, spacing)
                row = {"step": step + 1, "sample": sample}
                if teacher_optimizer is not None and (c["stage"] != "joint" or step % c["teacher_update_every"] == 0):
                    teacher.train()
                    if c["stage"] == "teacher_warmup":
                        teacher.encoder.eval()
                    teacher_optimizer.zero_grad(set_to_none=True)
                    outputs = teacher(x)
                    lt = teacher_loss(outputs, y, bank.sdf_targets(y, spacing), c["lambda_distance"])
                    if not torch.isfinite(lt):
                        raise FloatingPointError("nonfinite teacher loss")
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
                        # No teacher optimizer runs during candidate utility trials.
                        with torch.no_grad():
                            output = teacher(x)
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
                row["elapsed_seconds"] = time.perf_counter() - start
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
        optimizers = [student_optimizer] + ([teacher_optimizer] if teacher_optimizer is not None else [])
        sha = save_checkpoint(out / "final", student=student, teacher=teacher, optimizers=optimizers,
                              config=c, data=data, step=c["steps"], provenance=provenance)
        report = {"schema": SCHEMA, "status": "COMPLETED", "synthetic": synthetic,
            "quality": "NOT_EVALUATED", "stage": c["stage"], "steps": c["steps"], **counts,
            "label_budget": len(data.lids), "label_reads": sorted(data.label_reads),
            "unlabeled_usage": "UNUSED; no U reference paths opened",
            "final_manifest_sha256": sha, "elapsed_seconds": time.perf_counter() - start}
        write_json(out / "REPORT.json", report)
        return report
    except Exception as exc:
        write_json(out / "FAILURE.json", {"status": "FAILED", "error": repr(exc), **counts})
        raise
