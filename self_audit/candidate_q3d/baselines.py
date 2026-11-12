"""Runnable Q-3D baselines with isolated artifacts and evidence labels.

The A-3D baseline delegates to the frozen public Candidate A-3D trainer. The
teacher-supervised ResEnc baseline delegates its T-only loop to the Q-3D engine so model,
loss, checkpoint, and resume behavior stay in one implementation.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

from . import CHECKPOINT_SCHEMA


_ROOT = Path(__file__).resolve().parents[2]
_A3D_REPORT_SCHEMA = "self_audit.candidate_q3d.baseline.a3d_teacher_free.v1"
_TEACHER_REPORT_SCHEMA = "self_audit.candidate_q3d.baseline.teacher_supervised.v1"
_A3D_CHECKPOINT_SCHEMA = "self_audit.candidate_a3d.checkpoint.v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest(config: Mapping[str, Any], name: str) -> tuple[Path, str]:
    manifests = config["dataset"]["manifests"]
    record = manifests[name]
    if not isinstance(record, Mapping) or record.get("status") != "READY":
        raise ValueError(f"Q-3D baseline requires a READY {name} manifest")
    raw_path, expected = record.get("path"), record.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Q-3D baseline {name} manifest path is missing")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Q-3D baseline {name} manifest does not exist: {path}")
    actual = _sha256_file(path)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"Q-3D baseline {name} manifest SHA-256 mismatch")
    return path, actual


def _validated_config(
    config: Mapping[str, Any], *, arm: str, device: str, smoke: bool,
) -> tuple[dict[str, Any], bool]:
    if not isinstance(config, Mapping):
        raise TypeError("Q-3D baseline config must be a mapping")
    from .config import validate_config

    synthetic = bool(config.get("runtime", {}).get("synthetic", False))
    if config.get("arm") != arm:
        raise ValueError(f"baseline runner for {arm!r} received arm {config.get('arm')!r}")
    if smoke != synthetic:
        raise ValueError("SOFTWARE_ONLY smoke requires runtime.synthetic=true and real runs require false")
    if smoke and device != "cpu":
        raise ValueError("SOFTWARE_ONLY baseline smoke is CPU-only")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Q-3D baseline device must be cpu or cuda")
    # Real baseline paths are validated below without invoking the generic
    # Protocol-B all-TRAIN teacher-data requirement. Both baselines are K-only.
    normalized = validate_config(config, real_run=False)
    return normalized, synthetic


def _stage_budget(config: Mapping[str, Any], stage: str, *, smoke: bool) -> int:
    if smoke:
        overrides = config["runtime"].get("smoke_update_overrides", {})
        budget = overrides.get(stage)
        if type(budget) is not int or budget <= 0:
            raise ValueError(f"SOFTWARE_ONLY smoke requires a positive {stage} update override")
        return budget
    budget = config["training"]["stage_updates"].get(stage)
    if type(budget) is not int or budget <= 0:
        raise ValueError(f"real baseline config requires a positive {stage} update budget")
    return budget


def _a3d_template(config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    baseline = config.get("baselines", {}).get("a3d_teacher_free", {})
    template_value = baseline.get("config_template") if isinstance(baseline, Mapping) else None
    template = Path(template_value) if isinstance(template_value, str) else Path(
        "configs/candidate_a3d/teacher_free_acdc_k8.json"
    )
    if not template.is_absolute():
        template = _ROOT / template
    template = template.resolve()
    if not template.is_file():
        raise FileNotFoundError(f"A-3D teacher_free config template is missing: {template}")
    raw = json.loads(template.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("method") != "candidate_a3d":
        raise ValueError("A-3D teacher_free template must declare method=candidate_a3d")
    return raw, template


def _prepare_a3d_config(
    config: Mapping[str, Any], *, budget: int,
    frame_updates_per_epoch: int, smoke: bool,
) -> tuple[dict[str, Any], Path]:
    raw, template_path = _a3d_template(config)
    if budget % frame_updates_per_epoch:
        raise ValueError(
            "A-3D epoch schedule cannot exactly match the declared J optimizer budget: "
            f"budget={budget}, labeled_frames_per_epoch={frame_updates_per_epoch}"
        )
    raw["epochs"] = budget // frame_updates_per_epoch
    raw["seed"] = int(config["seed"])
    raw["rollout"]["seed"] = int(config["seed"])
    raw["dataset"]["name"] = config["dataset"]["name"]
    raw["dataset"]["protocol"] = "native_3d"
    # Preserve the frozen Candidate A-3D template's real backbone declaration;
    # this baseline runner must not hard-code a retired teacher architecture.
    raw["model"]["backbone"] = "tiny_test3d" if smoke else raw["model"].get("backbone")
    raw["use_teacher"] = False
    raw["freeze_teacher"] = True
    raw["teacher_init"] = {"kind": "none", "path": "", "sha256": ""}
    raw["semi_supervised"]["enabled"] = False
    raw["semi_supervised"]["steps_per_epoch"] = 0
    return raw, template_path


def run_a3d_teacher_free(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    device: str,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the unchanged A-3D teacher-free public trainer on K labels only."""

    normalized, synthetic = _validated_config(
        config, arm="a3d_teacher_free", device=device, smoke=smoke,
    )
    train_path, train_sha256 = _manifest(normalized, "train")
    development_path, development_sha256 = _manifest(normalized, "development")
    budget = _stage_budget(normalized, "J", smoke=smoke)

    from ..candidate_a3d.data import CandidateA3DData

    data = CandidateA3DData(train_path, train_sha256, synthetic=synthetic)
    k = int(normalized["protocol"]["k"])
    if len(data.lids) != k:
        raise ValueError(f"A-3D teacher_free requires exactly K={k} labeled TRAIN patients")
    if not data.labeled_frame_keys():
        raise ValueError("A-3D teacher_free TRAIN manifest has no labeled frames")
    frame_updates = len(data.labeled_frame_keys())
    a3d_config, template_path = _prepare_a3d_config(
        normalized, budget=budget, frame_updates_per_epoch=frame_updates, smoke=smoke,
    )

    from ..candidate_a3d.config import validate_config as validate_a3d_config
    from ..candidate_a3d.engine import train as train_a3d

    validated_a3d = validate_a3d_config(a3d_config)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"A-3D baseline output directory is not empty: {output}")
    result = train_a3d(
        validated_a3d,
        str(train_path),
        train_sha256,
        str(output),
        device=device,
        synthetic=synthetic,
        validation_manifest=str(development_path),
        validation_sha256=development_sha256,
    )
    logical_updates = int(result["epochs"]) * int(result["updates_per_epoch"])
    student_updates = int(result["student_updates"])
    u_updates = int(result.get("student_unlabeled_updates", 0))
    if logical_updates != budget or student_updates != budget or u_updates != 0:
        raise RuntimeError(
            "A-3D teacher_free update accounting differs from the matched J budget: "
            f"logical={logical_updates}, student_optimizer={student_updates}, "
            f"unlabeled_optimizer={u_updates}, expected={budget}"
        )
    checkpoint_path = output / "final"
    manifest_path = checkpoint_path / "MANIFEST.json"
    checkpoint_sha256 = _sha256_file(manifest_path) if manifest_path.is_file() else result.get("final_manifest_sha256")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise RuntimeError("A-3D baseline trainer did not return a valid final checkpoint manifest SHA-256")
    if manifest_path.is_file() and checkpoint_sha256 != result.get("final_manifest_sha256"):
        raise RuntimeError("A-3D checkpoint manifest hash differs from the public trainer report")
    report = {
        "schema": _A3D_REPORT_SCHEMA,
        "status": "COMPLETED_SOFTWARE_ONLY" if synthetic else "COMPLETED",
        "evidence": "SOFTWARE_ONLY" if synthetic else "REAL_DATA_TRAINING",
        "method": "candidate_a3d",
        "arm": "teacher_free",
        "trainer_schema": result.get("schema"),
        "checkpoint_schema": _A3D_CHECKPOINT_SCHEMA,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "checkpoint_manifest_path": str(manifest_path),
        "config_template": str(template_path),
        "config_template_sha256": _sha256_file(template_path),
        "source_hashes": {
            "baseline_runner": _sha256_file(Path(__file__)),
            "legacy_engine": _sha256_file(_ROOT / "self_audit/candidate_a3d/engine.py"),
            "legacy_config": _sha256_file(_ROOT / "self_audit/candidate_a3d/config.py"),
            "legacy_data": _sha256_file(_ROOT / "self_audit/candidate_a3d/data.py"),
        },
        "train_manifest_sha256": train_sha256,
        "development_manifest_sha256": development_sha256,
        "protocol": normalized["protocol"]["variant"],
        "label_budget_k": k,
        "labeled_patient_ids": list(data.lids),
        "seed": int(normalized["seed"]),
        "logical_update_unit": "L_only",
        "logical_updates": logical_updates,
        "optimizer_updates": student_updates,
        "optimizer_update_counts": {
            "student_labeled": student_updates,
            "student_unlabeled": u_updates,
            "teacher": int(result.get("teacher_updates", 0)),
        },
        "training_forward_counts": {"labeled": logical_updates, "unlabeled": 0, "teacher": 0},
        "image_only_u_policy": "not_forwarded; no U loss or optimizer update",
        "supervision_source": "K labeled TRAIN patients only",
        "development_selection": "legacy A-3D validation at each configured epoch; final checkpoint retained",
        "gpu_evidence": "NOT_RUN" if device == "cpu" else "RUNTIME_REQUESTED",
        "real_data_evidence": "NOT_RUN" if synthetic else "TRAINED",
        "trainer_report": result,
    }
    _json_write(output / "BASELINE_REPORT.json", report)
    return report


def _run_teacher_t_only(
    config: Mapping[str, Any],
    data: Any,
    output_dir: Path,
    *,
    device: str,
    smoke: bool,
    resume_from: Path | None,
    update_budget: int,
) -> dict[str, Any]:
    """Compose the existing Q-3D T stage without student/Q/P/J execution."""

    import torch
    from torch import nn

    from . import engine

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("teacher-supervised baseline requested CUDA but CUDA is unavailable")
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("teacher-supervised baseline supports CPU or CUDA only")
    if bool(config["runtime"]["synthetic"]) != smoke or (smoke and target_device.type != "cpu"):
        raise ValueError("synthetic ResEnc proxy requires explicit SOFTWARE_ONLY CPU mode")
    if config["tokenizer"]["enabled"]:
        raise ValueError("teacher-supervised baseline must not construct or run a tokenizer")
    if data.protocol != "A" or data.label_budget != int(config["protocol"]["k"]):
        raise ValueError("teacher-supervised baseline data must be an explicit K-only logical Protocol-A view")
    if set(data.teacher_patients) != set(data.labeled_patients):
        raise ValueError("teacher-supervised baseline teacher pool must contain exactly the student K labeled patients")

    seed = int(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    spatial_rng, intensity_generator = engine._seed_all(seed)
    stage_rng = random.Random(seed + 47)
    data_identity = engine._data_identity_without_labels(data)
    config_sha256 = engine._object_sha256(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    root_checkpoint = output_dir / "latest.pt"
    if resume_from is None and root_checkpoint.is_file():
        resume_from = root_checkpoint
    if resume_from is None and any(output_dir.iterdir()):
        raise FileExistsError("teacher-supervised baseline output contains no resumable latest.pt")

    teacher, student, tokenizer, provenance = engine._build_models(
        config,
        device=target_device,
        teacher=None,
        student=nn.Identity(),
        tokenizer=None,
    )
    if teacher is None or tokenizer is not None or not isinstance(student, nn.Identity):
        raise RuntimeError("teacher-supervised T-only construction unexpectedly created a student or tokenizer")
    if provenance.get("teacher") is None:
        raise RuntimeError("teacher-supervised T-only construction did not retain teacher provenance")

    if resume_from is None:
        state = engine._initial_run_state(config_sha256, data_identity)
    else:
        state = engine._load_checkpoint(
            resume_from, config_sha256=config_sha256, data_identity=data_identity,
        )
        completed = list(state.get("completed_stages", []))
        if any(stage != "T" for stage in completed) or state.get("stage") not in {None, "T"}:
            raise ValueError("teacher-supervised baseline resume checkpoint contains stages outside T")
        engine._restore_run_state(
            state,
            teacher=teacher,
            student=student,
            tokenizer=None,
            ema_model=None,
            spatial_rng=spatial_rng,
            intensity_generator=intensity_generator,
            stage_rng=stage_rng,
        )

    baseline_metadata = {
        "schema": _TEACHER_REPORT_SCHEMA,
        # This public runner is a standalone compatibility path.  Campaign
        # plans do not launch it: they bind the teacher-supervised reference
        # to a shared T producer and set independent_baseline=false only in
        # that verified reuse record.
        "independent_baseline": "NOT_RUN",
        "shared_teacher_stage": "NOT_REUSED",
        "stage_contract": "T_ONLY_STANDALONE",
        "parent_config_sha256": config_sha256,
        "parent_protocol": config["protocol"]["variant"],
        "baseline_supervision_protocol": "K_only",
        "train_manifest_sha256": config["dataset"]["manifests"]["train"]["sha256"],
        "development_manifest_sha256": config["dataset"]["manifests"]["development"]["sha256"],
        "label_budget": int(data.label_budget),
        "labeled_patient_ids": sorted(data.labeled_patients),
        "teacher_train_manifest_used": False,
        "seed": seed,
        "stage": "T",
        "source_pretraining_hashes": dict(provenance["teacher"]),
    }
    saved_metadata = state.get("baseline_metadata")
    if saved_metadata is not None:
        if any(saved_metadata.get(key) != value for key, value in baseline_metadata.items()):
            raise ValueError("teacher-supervised resume baseline metadata differs from the parent config or K pool")
        baseline_metadata.update(saved_metadata)
    state["baseline_metadata"] = baseline_metadata

    if state.get("stage") not in {None, "T"}:
        raise ValueError("teacher-supervised baseline checkpoint is not at a resumable T-stage boundary")
    completed_stages = list(state.get("completed_stages", []))
    if "T" not in completed_stages:
        completed = engine._run_teacher_stage(
            config,
            data=data,
            teacher=teacher,
            student=student,
            tokenizer=None,
            ema_model=None,
            state=state,
            output_dir=output_dir,
            device=target_device,
            spatial_rng=spatial_rng,
            intensity_generator=intensity_generator,
            stage_rng=stage_rng,
            sampler=None,
            smoke=smoke,
            update_limit_active=False,
            dev_evaluator=None,
            update_budget=update_budget,
        )
    else:
        completed = int(state.get("stage_cursor", update_budget) or update_budget)
        best = state.get("best_teacher_state")
        if not isinstance(best, Mapping):
            raise ValueError("completed teacher-supervised resume checkpoint has no DEV-selected teacher weights")
        teacher.load_state_dict(best, strict=True)

    if completed != update_budget or state.get("best_teacher_state") is None:
        raise RuntimeError("teacher-supervised T-only run did not complete its declared DEV-selected budget")
    if "T" not in state.get("completed_stages", []):
        raise RuntimeError("teacher-supervised T-only helper returned before the T stage was marked complete")

    selections = [
        row for row in state.get("metrics", [])
        if row.get("stage") == "T" and row.get("event") == "development_selection"
    ]
    if not selections:
        raise RuntimeError("teacher-supervised T-only run has no DEV checkpoint-selection record")
    best_selection = max(selections, key=lambda row: float(row["score"]))
    if float(best_selection["score"]) != float(state["best_teacher_score"]):
        raise RuntimeError("teacher-supervised selected score differs from its DEV selection history")

    teacher_state_sha256 = engine._state_sha256(teacher)
    state["baseline_metadata"].update({
        "logical_updates": int(completed),
        "optimizer_updates": int(completed),
        "best_development_score": float(state["best_teacher_score"]),
        "best_development_update": int(best_selection["update"]),
        "teacher_dev_fg_dice": state.get("teacher_dev_fg_dice"),
        "teacher_dev_per_class": state.get("teacher_dev_per_class"),
        "teacher_selected_dev_update": state.get("teacher_selected_dev_update"),
        "teacher_terminal_dev_fg_dice": state.get("teacher_terminal_dev_fg_dice"),
        "teacher_terminal_dev_per_class": state.get("teacher_terminal_dev_per_class"),
        "dev_curve_still_improving": state.get("dev_curve_still_improving"),
        "development_selection": "development_patient_macro_fg_dice",
        "best_teacher_state_sha256": teacher_state_sha256,
    })
    stage_checkpoint = engine._stage_directory(output_dir, config, "T") / "latest.pt"
    engine._atomic_torch_save(state, stage_checkpoint)
    root_checkpoint = output_dir / "latest.pt"
    checkpoint_sha256 = engine._atomic_torch_save(state, root_checkpoint)
    teacher_source = dict(provenance["teacher"])
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_path": str(root_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "logical_updates": int(completed),
        "optimizer_updates": int(completed),
        "training_forward_counts": {
            "labeled": int(completed), "teacher": int(completed), "unlabeled": 0,
        },
        "best_development_score": float(state["best_teacher_score"]),
        "best_development_update": int(best_selection["update"]),
        "teacher_dev_fg_dice": state.get("teacher_dev_fg_dice"),
        "teacher_dev_per_class": state.get("teacher_dev_per_class"),
        "teacher_selected_dev_update": state.get("teacher_selected_dev_update"),
        "teacher_terminal_dev_fg_dice": state.get("teacher_terminal_dev_fg_dice"),
        "teacher_terminal_dev_per_class": state.get("teacher_terminal_dev_per_class"),
        "dev_curve_still_improving": state.get("dev_curve_still_improving"),
        "development_selection": "development_patient_macro_fg_dice",
        "source_pretraining_hashes": teacher_source,
        "resume_from": str(resume_from) if resume_from else None,
        "resume_checkpoint_path": str(root_checkpoint),
        "resume_checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "teacher_state_sha256": teacher_state_sha256,
    }


def _validate_teacher_pretraining(config: Mapping[str, Any]) -> dict[str, Any]:
    from .config import validate_real_teacher_provenance

    validate_real_teacher_provenance(config)
    teacher = config["model"]["teacher"]
    pretrained = teacher["pretrained"]
    source_repo, source_commit = teacher.get("source_repo"), teacher.get("source_commit")
    if teacher.get("architecture") != "nnfoundation_resenc_l":
        raise ValueError("teacher-supervised baseline requires compatible nnfoundation ResEnc-L weights")
    if not isinstance(source_repo, str) or not source_repo.strip() or not isinstance(source_commit, str) or not source_commit.strip():
        raise ValueError("real teacher-supervised baseline requires validated source_repo and source_commit")
    pretraining_dataset = pretrained.get("pretraining_dataset")
    overlap_acdc, overlap_mnms = pretrained.get("overlap_acdc"), pretrained.get("overlap_mnms")
    if not isinstance(pretraining_dataset, str) or not pretraining_dataset.strip():
        raise ValueError("real teacher-supervised baseline requires declared pretraining_dataset")
    if type(overlap_acdc) is not bool or type(overlap_mnms) is not bool:
        raise ValueError("real teacher-supervised baseline requires explicit ACDC and M&Ms overlap provenance")
    return {
        "source_repo": source_repo,
        "source_commit": source_commit,
        "weights_path": str(Path(pretrained["weights_path"]).expanduser().resolve()),
        "weights_sha256": pretrained["weights_sha256"],
        "pretraining_dataset": pretraining_dataset,
        "overlap_acdc": overlap_acdc,
        "overlap_mnms": overlap_mnms,
    }


def run_teacher_supervised(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    device: str,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the shared ResEnc teacher as a K-only Q-3D T-stage baseline."""

    normalized, synthetic = _validated_config(
        config, arm="teacher_supervised", device=device, smoke=smoke,
    )
    train_path, train_sha256 = _manifest(normalized, "train")
    development_path, development_sha256 = _manifest(normalized, "development")
    budget = _stage_budget(normalized, "T", smoke=smoke)
    k = int(normalized["protocol"]["k"])

    if not synthetic:
        raise ValueError(
            "teacher-supervised real evaluation must reuse the campaign's shared T artifact; "
            "no standalone teacher training job is permitted"
        )

    from ..candidate_a3d.data import CandidateA3DData

    data = CandidateA3DData(train_path, train_sha256, synthetic=synthetic)
    if len(data.lids) != k:
        raise ValueError(f"teacher-supervised baseline requires exactly K={k} labeled TRAIN patients")
    provenance = None if synthetic else _validate_teacher_pretraining(normalized)
    output = Path(output_dir).expanduser().resolve()
    resume_from: Path | None = None
    baseline_settings = normalized.get("baselines", {}).get("teacher_supervised", {})
    configured_resume = baseline_settings.get("resume_from") if isinstance(baseline_settings, Mapping) else None
    if configured_resume:
        resume_from = Path(configured_resume).expanduser().resolve()
    elif (output / "latest.pt").is_file():
        resume_from = output / "latest.pt"
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("teacher-supervised baseline output has files but no resumable latest.pt")

    from .data import Q3DData

    k_only_data = Q3DData(data, protocol="A")
    helper_result = _run_teacher_t_only(
        normalized,
        k_only_data,
        output,
        device=device,
        smoke=smoke,
        resume_from=resume_from,
        update_budget=budget,
    )
    logical_updates = int(helper_result.get("logical_updates", helper_result.get("updates", -1)))
    optimizer_updates = int(helper_result.get("optimizer_updates", -1))
    if logical_updates != budget or optimizer_updates != budget:
        raise RuntimeError(
            "teacher-supervised T-only update accounting differs from its declared budget: "
            f"logical={logical_updates}, optimizer={optimizer_updates}, expected={budget}"
        )
    checkpoint_path = helper_result.get("checkpoint_path", helper_result.get("checkpoint"))
    checkpoint_sha256 = helper_result.get("checkpoint_sha256")
    if not isinstance(checkpoint_path, str) or not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise RuntimeError("teacher-supervised T-only helper did not return checkpoint path and SHA-256")
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise RuntimeError("teacher-supervised T-only helper checkpoint path is not a file")
    if _sha256_file(checkpoint_file) != checkpoint_sha256:
        raise RuntimeError("teacher-supervised T-only checkpoint SHA-256 differs from its bytes")
    forward_counts = helper_result.get("training_forward_counts") or {
        "labeled": logical_updates, "teacher": logical_updates, "unlabeled": 0,
    }
    if int(forward_counts.get("unlabeled", 0)) != 0:
        raise RuntimeError("teacher-supervised T stage must not forward image-only U")
    report = {
        "schema": _TEACHER_REPORT_SCHEMA,
        "status": "COMPLETED_SOFTWARE_ONLY" if synthetic else "COMPLETED",
        "evidence": "SOFTWARE_ONLY" if synthetic else "REAL_DATA_TRAINING",
        "method": "candidate_q3d",
        "arm": "teacher_supervised",
        "independent_baseline": "NOT_RUN",
        "shared_teacher_stage": "NOT_REUSED",
        "stage_contract": "T_ONLY_STANDALONE",
        "checkpoint_schema": helper_result.get("checkpoint_schema"),
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_sha256": checkpoint_sha256,
        "train_manifest_sha256": train_sha256,
        "development_manifest_sha256": development_sha256,
        "protocol": normalized["protocol"]["variant"],
        "baseline_supervision_protocol": "K_only",
        "teacher_train_manifest_used": False,
        "label_budget_k": k,
        "labeled_patient_ids": list(data.lids),
        "seed": int(normalized["seed"]),
        "stages": ["T"],
        "tokenizer_enabled": False,
        "student_artifact": False,
        "logical_update_unit": "L_only",
        "logical_updates": logical_updates,
        "optimizer_updates": optimizer_updates,
        "training_forward_counts": forward_counts,
        "source_pretraining_hashes": provenance or helper_result.get("source_pretraining_hashes"),
        "source_hashes": {
            "baseline_runner": _sha256_file(Path(__file__)),
            "q3d_engine": _sha256_file(_ROOT / "self_audit/candidate_q3d/engine.py"),
            "q3d_teacher": _sha256_file(_ROOT / "self_audit/candidate_q3d/teacher_resenc.py"),
            "q3d_config": _sha256_file(_ROOT / "self_audit/candidate_q3d/config.py"),
        },
        "development_selection": helper_result.get("development_selection", "patient_macro_fg_dice"),
        "teacher_dev_fg_dice": helper_result.get("best_development_score"),
        "teacher_dev_per_class": helper_result.get("teacher_dev_per_class"),
        "teacher_selected_dev_update": helper_result.get("best_development_update"),
        "teacher_terminal_dev_fg_dice": helper_result.get("teacher_terminal_dev_fg_dice"),
        "teacher_terminal_dev_per_class": helper_result.get("teacher_terminal_dev_per_class"),
        "dev_curve_still_improving": helper_result.get("dev_curve_still_improving"),
        "evaluation": {
            "kind": "teacher_supervised_region_baseline",
            "manifest_sha256": development_sha256,
            "artifact_path": str(Path(checkpoint_path).resolve()),
            "separate_from_student_export": True,
            "selected_score": helper_result.get("best_development_score"),
        },
        "resume_from": str(resume_from) if resume_from else None,
        "gpu_evidence": "NOT_RUN" if device == "cpu" else "RUNTIME_REQUESTED",
        "real_data_evidence": "NOT_RUN" if synthetic else "TRAINED",
        "engine_result": helper_result,
    }
    _json_write(output / "BASELINE_REPORT.json", report)
    return report


def evaluate_teacher_supervised_baseline(
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    config: Mapping[str, Any],
    data_manifest: str | Path,
    data_sha256: str,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    split: str = "dev",
) -> dict[str, Any]:
    """Evaluate the DEV-selected shared ResEnc region checkpoint separately."""

    import torch
    from torch import nn

    from ..candidate_a3d.data import CandidateA3DData
    from . import engine
    from .config import validate_config
    from .data import Q3DData
    from .evaluation import evaluate_manifest

    normalized = validate_config(config, real_run=False)
    if normalized.get("arm") != "teacher_supervised":
        raise ValueError("teacher-supervised evaluation requires arm=teacher_supervised")
    synthetic = bool(normalized["runtime"]["synthetic"])
    if synthetic and device != "cpu":
        raise ValueError("SOFTWARE_ONLY heavy baseline evaluation is CPU-only")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("heavy baseline evaluation requested unavailable CUDA")
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("heavy baseline evaluation supports CPU or CUDA only")

    train_path, train_sha = _manifest(normalized, "train")
    split_key = {"dev": "development", "development": "development", "test": "test"}.get(split)
    if split_key is None:
        raise ValueError("teacher-supervised evaluation split must be dev, development, or test")
    declared_path, declared_sha = _manifest(normalized, split_key)
    requested_path = Path(data_manifest).expanduser().resolve()
    if requested_path != declared_path or data_sha256 != declared_sha:
        raise ValueError("heavy baseline evaluation manifest differs from the nested hash-pinned config")

    artifact = Path(checkpoint_path).expanduser().resolve()
    if not artifact.is_file() or _sha256_file(artifact) != checkpoint_sha256:
        raise ValueError("teacher-supervised checkpoint bytes or SHA-256 do not match")
    try:
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("teacher-supervised checkpoint is not a safe Q-3D resume.v3 payload") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("teacher-supervised evaluation requires Q-3D resume.v3 checkpoint schema")

    config_sha256 = engine._object_sha256(normalized)
    shared_t = payload.get("stage_artifact_stage") == "T"
    sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
    if sidecar.is_file():
        try:
            sidecar_sha256 = sidecar.read_text(encoding="ascii").split()[0]
        except (OSError, IndexError) as exc:
            raise ValueError("teacher-supervised checkpoint SHA-256 sidecar is malformed") from exc
        if sidecar_sha256 != checkpoint_sha256:
            raise ValueError("teacher-supervised checkpoint sidecar SHA-256 mismatch")
    elif not shared_t:
        # Standalone baseline checkpoints are passed around with their sidecar;
        # shared immutable T artifacts already receive the SHA from their stage
        # manifest and need not mutate that artifact directory with a second
        # mutable receipt.
        raise ValueError("teacher-supervised checkpoint requires its SHA-256 sidecar")
    if shared_t:
        # A shared T artifact is sealed against the arm-independent T source
        # contract, not the consumer's full arm config.
        from .stages import stage_source_config

        source_sha256 = engine._object_sha256(stage_source_config(normalized, "T"))
        if payload.get("config_sha256") != source_sha256:
            raise ValueError("shared T checkpoint source-config SHA-256 mismatch")
    elif payload.get("config_sha256") != config_sha256:
        raise ValueError("teacher-supervised checkpoint parent config SHA-256 mismatch")
    train_data = CandidateA3DData(train_path, train_sha, synthetic=synthetic)
    k_only_data = Q3DData(train_data, protocol="A")
    if len(k_only_data.labeled_patients) != int(normalized["protocol"]["k"]):
        raise ValueError("teacher-supervised evaluation TRAIN data does not match parent K")
    data_identity = engine._data_identity_without_labels(k_only_data)
    if payload.get("data_identity") != data_identity:
        raise ValueError("teacher-supervised checkpoint K-only data identity mismatch")

    expected_patient_ids = sorted(k_only_data.labeled_patients)
    metadata = payload.get("baseline_metadata")
    if shared_t:
        if metadata is not None:
            raise ValueError("shared T checkpoint must not carry standalone baseline metadata")
        data_identity_sha = payload.get("data_identity_sha256")
        if data_identity_sha != engine._object_sha256(data_identity):
            raise ValueError("shared T checkpoint data-identity hash differs from the requested evaluation")
    else:
        if not isinstance(metadata, Mapping):
            raise ValueError("resume.v3 checkpoint lacks teacher-supervised baseline metadata")
        if (metadata.get("schema") != _TEACHER_REPORT_SCHEMA
                or metadata.get("parent_config_sha256") != config_sha256
                or metadata.get("parent_protocol") != normalized["protocol"]["variant"]
                or metadata.get("baseline_supervision_protocol") != "K_only"
                or metadata.get("label_budget") != int(normalized["protocol"]["k"])
                or metadata.get("labeled_patient_ids") != expected_patient_ids
                or metadata.get("teacher_train_manifest_used") is not False
                or metadata.get("stage") != "T"
                or metadata.get("train_manifest_sha256") != train_sha
                or metadata.get("development_manifest_sha256") != normalized["dataset"]["manifests"]["development"]["sha256"]):
            raise ValueError("teacher-supervised checkpoint baseline identity differs from the requested evaluation")
    if payload.get("completed_stages") != ["T"] or payload.get("stage") is not None:
        raise ValueError("teacher-supervised checkpoint must contain completed T only, with no Q/P/J stages")
    if payload.get("tokenizer_state") is not None or payload.get("ema_state") is not None:
        raise ValueError("teacher-supervised checkpoint must not carry a tokenizer or EMA student")
    best_teacher_state = payload.get("best_teacher_state")
    if not isinstance(best_teacher_state, Mapping):
        raise ValueError("teacher-supervised checkpoint lacks its DEV-selected best_teacher_state")
    selections = [
        row for row in payload.get("metrics", [])
        if isinstance(row, Mapping) and row.get("stage") == "T" and row.get("event") == "development_selection"
    ]
    if not selections:
        raise ValueError("teacher-supervised checkpoint has no DEV selection history")
    best_selection = max(selections, key=lambda row: float(row["score"]))
    best_score = float(payload.get("best_teacher_score"))
    metadata_score = best_score if shared_t else metadata.get("best_development_score")
    metadata_update = int(best_selection["update"]) if shared_t else metadata.get("best_development_update")
    if (float(best_selection["score"]) != best_score
            or metadata_score != best_score
            or metadata_update != int(best_selection["update"])):
        raise ValueError("teacher-supervised best_teacher_state is not the recorded DEV-selected checkpoint")

    source_hashes = (
        {"shared_t": True, "source_config_sha256": payload.get("config_sha256")}
        if shared_t else metadata.get("source_pretraining_hashes")
    )
    if not isinstance(source_hashes, Mapping):
        raise ValueError("teacher-supervised checkpoint lacks source/pretraining hashes")
    if not synthetic and not shared_t:
        expected_source = _validate_teacher_pretraining(normalized)
        if source_hashes.get("weights_sha256") != expected_source["weights_sha256"]:
            raise ValueError("teacher-supervised checkpoint pretrained-weight hash differs from config")
        if source_hashes.get("source_commit") not in {None, expected_source["source_commit"]}:
            raise ValueError("teacher-supervised checkpoint source commit differs from config")

    teacher, student, tokenizer, provenance = engine._build_models(
        normalized,
        device=target_device,
        teacher=None,
        student=nn.Identity(),
        tokenizer=None,
    )
    if teacher is None or tokenizer is not None or not isinstance(student, nn.Identity):
        raise RuntimeError("teacher-supervised evaluation unexpectedly constructed a student or tokenizer")
    teacher.load_state_dict(best_teacher_state, strict=True)
    teacher_state_sha256 = engine._state_sha256(teacher)
    expected_teacher_state_sha = payload.get("teacher_best_sha256") if shared_t else metadata.get("best_teacher_state_sha256")
    if expected_teacher_state_sha != teacher_state_sha256:
        raise ValueError("teacher-supervised DEV-selected teacher state SHA-256 mismatch")

    class RegionLogitsAdapter(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, image: torch.Tensor, *, geometry: Mapping[str, Any]):
            output = self.model(image, geometry=geometry)
            logits = output.get("region_logits") if isinstance(output, Mapping) else None
            if not torch.is_tensor(logits):
                raise ValueError("teacher-supervised region baseline did not return region_logits")
            return {"logits": logits}

    result = evaluate_manifest(
        RegionLogitsAdapter(teacher),
        requested_path,
        declared_sha,
        output_dir,
        device=device,
        split=split,
        train_patient_ids=k_only_data.train_patients,
        expected_synthetic=synthetic,
    )
    output = Path(output_dir).expanduser().resolve()
    report = {
        "schema": "self_audit.candidate_q3d.teacher_supervised.evaluation.v1",
        "artifact_kind": "teacher_supervised_region_baseline",
        "student_export": False,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_path": str(artifact),
        "checkpoint_sha256": checkpoint_sha256,
        "parent_config_sha256": config_sha256,
        "parent_protocol": normalized["protocol"]["variant"],
        "independent_baseline": False if shared_t else "NOT_RUN",
        "shared_teacher_stage": "T" if shared_t else "NOT_REUSED",
        "stage_contract": "T0J" if shared_t else "T_ONLY_STANDALONE",
        "shared_t_source_config_sha256": payload.get("config_sha256") if shared_t else None,
        "baseline_supervision_protocol": "K_only",
        "teacher_train_manifest_used": False,
        "label_budget_k": int(normalized["protocol"]["k"]),
        "labeled_patient_ids": expected_patient_ids,
        "split": split_key,
        "evaluation_manifest_sha256": declared_sha,
        "training_patient_ids_for_overlap_check": list(k_only_data.train_patients),
        "dev_selected_update": int(best_selection["update"]),
        "dev_selected_fg_dice": best_score,
        "teacher_dev_fg_dice": payload.get("teacher_dev_fg_dice", best_score),
        "teacher_dev_per_class": payload.get("teacher_dev_per_class"),
        "teacher_selected_dev_update": payload.get("teacher_selected_dev_update", best_selection["update"]),
        "teacher_terminal_dev_fg_dice": payload.get("teacher_terminal_dev_fg_dice"),
        "teacher_terminal_dev_per_class": payload.get("teacher_terminal_dev_per_class"),
        "dev_curve_still_improving": payload.get("dev_curve_still_improving"),
        "teacher_state_sha256": teacher_state_sha256,
        "source_pretraining_hashes": dict(source_hashes),
        "evaluation": result,
    }
    _json_write(output / "BASELINE_EVALUATION.json", report)
    return report


def _baseline_train_patient_ids(path: Path, expected_sha256: str) -> tuple[str, ...]:
    if _sha256_file(path) != expected_sha256:
        raise ValueError("A-3D Q-3D baseline TRAIN manifest SHA-256 mismatch")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("A-3D Q-3D baseline TRAIN manifest is invalid JSON") from exc
    if manifest.get("schema") != "self_audit.data.v2" or manifest.get("role") != "training":
        raise ValueError("A-3D Q-3D evaluation requires a hash-pinned training manifest")
    ids: list[str] = []
    for field in ("labeled", "unlabeled"):
        records = manifest.get(field)
        if not isinstance(records, list):
            raise ValueError(f"A-3D Q-3D TRAIN {field} records are invalid")
        for record in records:
            patient_id = record.get("id") if isinstance(record, Mapping) else None
            if not isinstance(patient_id, str) or not patient_id:
                raise ValueError("A-3D Q-3D TRAIN patient IDs must be nonempty strings")
            ids.append(patient_id)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("A-3D Q-3D TRAIN patient IDs must be nonempty and unique")
    return tuple(ids)


def evaluate_a3d_teacher_free_baseline(
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    config: Mapping[str, Any],
    data_manifest: str | Path,
    data_sha256: str,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate the frozen A-3D student with Q-3D TEST metrics and checks."""

    import torch
    from torch import nn

    from ..candidate_a3d.io import load_student
    from ..candidate_a3d.k1 import K1Config, infer_student
    from .config import validate_config
    from .evaluation import evaluate_manifest

    normalized = validate_config(config, real_run=False)
    if normalized.get("method") != "candidate_q3d" or normalized.get("arm") != "a3d_teacher_free":
        raise ValueError("A-3D Q-3D baseline evaluation requires method=candidate_q3d and arm=a3d_teacher_free")
    synthetic = bool(normalized["runtime"]["synthetic"])
    if synthetic and device != "cpu":
        raise ValueError("SOFTWARE_ONLY A-3D baseline evaluation is CPU-only")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A-3D baseline evaluation requested unavailable CUDA")
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("A-3D baseline evaluation supports CPU or CUDA only")

    train_path, train_sha = _manifest(normalized, "train")
    split_key = {"dev": "development", "development": "development", "test": "test"}.get(split)
    if split_key is None:
        raise ValueError("A-3D Q-3D baseline split must be dev, development, or test")
    declared_path, declared_sha = _manifest(normalized, split_key)
    requested_path = Path(data_manifest).expanduser().resolve()
    if requested_path != declared_path or data_sha256 != declared_sha:
        raise ValueError("A-3D baseline evaluation manifest differs from the nested hash-pinned config")

    train_ids = _baseline_train_patient_ids(train_path, train_sha)
    # The shared loader checks hashes, schema, safe weights-only loading, and
    # exact source_inventory equality before constructing the student.
    model, bank, codec, metadata = load_student(
        checkpoint_path, checkpoint_sha256, device=str(target_device),
    )
    if metadata.get("data_sha256") != train_sha:
        raise ValueError("A-3D checkpoint TRAIN manifest SHA-256 differs from Q-3D config")
    checkpoint_train_ids = set(metadata.get("labeled_ids", [])) | set(metadata.get("unlabeled_ids", []))
    if checkpoint_train_ids != set(train_ids):
        raise ValueError("A-3D checkpoint TRAIN patient inventory differs from Q-3D config")
    if bool(metadata.get("synthetic", False)) != synthetic:
        raise ValueError("A-3D checkpoint/config synthetic identity mismatch")
    if metadata.get("prediction_model") != "student":
        raise ValueError("A-3D Q-3D evaluation requires the student-only prediction path")
    try:
        rollout_config = K1Config(**metadata["config"]["rollout"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("A-3D checkpoint is missing its exact native query-budget config") from exc

    class A3DStudentPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.student = model

        def forward(self, image: torch.Tensor, *, geometry: Mapping[str, Any]) -> dict[str, torch.Tensor]:
            rollout = infer_student(
                self.student, image, bank, codec=codec, geometry=geometry,
                config=rollout_config,
            )
            return {"logits": rollout.final_logits}

    result = evaluate_manifest(
        A3DStudentPredictor(), requested_path, data_sha256, output_dir,
        device=target_device, split=split, train_patient_ids=train_ids,
        expected_synthetic=synthetic,
    )
    result.update({
        "artifact_kind": "a3d_teacher_free_student",
        "checkpoint_schema": metadata["schema"],
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "source_inventory_sha256": metadata.get("source_sha256"),
        "query_budget_bits": rollout_config.bit_budget,
        "query_budget_max_steps": rollout_config.max_steps,
        "student_only_inference": True,
    })
    _json_write(Path(output_dir).resolve() / "BASELINE_EVALUATION.json", result)
    return result


__all__ = ["evaluate_a3d_teacher_free_baseline", "evaluate_teacher_supervised_baseline",
           "run_a3d_teacher_free", "run_teacher_supervised"]
