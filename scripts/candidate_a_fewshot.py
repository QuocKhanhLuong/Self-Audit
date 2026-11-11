#!/usr/bin/env python3
"""Plan and run the full Candidate-A 3-D few-shot campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.candidate_a_ablations import _load_semi_config, create_plan, run_plan
from self_audit.data import write_json
from self_audit.fewshot import build_subset_manifest
from self_audit.model import digest

DEFAULT_CONFIG = ROOT / "configs/fewshot/candidate_a3d.json"
CATALOGUE = ROOT / "configs/ablations/candidate_a3d.json"
PLAN_SCHEMA = "self_audit.candidate_a3d.fewshot_plan.v1"
RECEIPT_SCHEMA = "self_audit.candidate_a3d.fewshot_receipt.v1"
EXPECTED_SHOTS = (2, 5, 8, 10)
EXPECTED_SEED = 42
_UNSET = object()


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON file must contain one object: {path}")
    return value


def _validate_config(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    if config.get("schema") != "self_audit.candidate_a3d.fewshot.v1":
        raise ValueError("few-shot config schema mismatch")
    if config.get("track") != "candidate_a3d" or config.get("seed") != EXPECTED_SEED:
        raise ValueError("few-shot config must target candidate_a3d with seed 42")
    if tuple(config.get("shots", ())) != EXPECTED_SHOTS:
        raise ValueError("few-shot shots must be exactly [2,5,8,10]")
    for field in ("epochs", "pretrain_epochs", "warmup_epochs"):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    checkpoint_every = config.get("checkpoint_every_epochs", 0)
    if (isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int)
            or checkpoint_every < 0):
        raise ValueError("checkpoint_every_epochs must be a non-negative integer")
    chunk_size = config.get("decoder_chunk_size")
    if (chunk_size is not None and
            (isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1)):
        raise ValueError("decoder_chunk_size must be a positive integer")
    catalogue = _load_json(CATALOGUE)
    known = [arm["name"] for arm in catalogue.get("arms", [])]
    arms = config.get("arms")
    if (not isinstance(arms, list) or len(arms) != len(known)
            or len(set(arms)) != len(arms) or set(arms) != set(known)):
        raise ValueError("few-shot config must enumerate every canonical A-3D arm exactly once")
    return config


def _validate_model_config(path: Path | None, *, synthetic: bool) -> dict[str, Any] | None:
    if synthetic and path is None:
        return None
    if path is None:
        raise ValueError("real few-shot campaign requires --model-config")
    config = _load_json(path)
    if config.get("backbone") != "mednext_l_k3":
        raise ValueError("few-shot campaign requires the verified mednext_l_k3 backend")
    if not config.get("source_repo") or not config.get("source_commit"):
        raise ValueError("model config must pin source_repo and source_commit")
    if config.get("weights") not in (None, "") or config.get("weights_sha256") not in (None, ""):
        raise ValueError("one external weight checkpoint cannot be reused across few-shot budgets; use weights:null and pretrain per k")
    return config


def _validate_development(path: Path, source: dict[str, Any], *, synthetic: bool, dataset: str) -> None:
    development = _load_json(path)
    if development.get("schema") != "self_audit.data.v2" or development.get("role") != "development":
        raise ValueError("development manifest must be a self_audit.data.v2 development manifest")
    if bool(development.get("synthetic", False)) != synthetic:
        raise ValueError("development synthetic/patient identity differs from TRAIN")
    if not synthetic and development.get("dataset") != dataset:
        raise ValueError("development dataset differs from requested dataset")
    eval_ids = source.get("evaluation_ids", [])
    development_ids = [record.get("id") for record in development.get("patients", [])]
    if (not development_ids or len(set(development_ids)) != len(development_ids)
            or not set(development_ids).issubset(set(eval_ids))):
        raise ValueError("development IDs must be a unique subset of source evaluation_ids")


def _validate_source(train_path: Path, development_path: Path, *, dataset: str, synthetic: bool) -> tuple[str, dict[str, Any]]:
    train_sha = digest(train_path)
    source = _load_json(train_path)
    if source.get("schema") != "self_audit.data.v2" or source.get("role") != "training":
        raise ValueError("few-shot source must be a self_audit.data.v2 training manifest")
    if bool(source.get("synthetic", False)) != synthetic:
        raise ValueError("source synthetic/patient identity differs from request")
    if not synthetic and source.get("dataset") != dataset:
        raise ValueError("source dataset differs from requested dataset")
    if source.get("label_budget") != len(source.get("labeled", [])):
        raise ValueError("source label_budget must equal source labeled pool")
    if len(source.get("labeled", [])) < max(EXPECTED_SHOTS):
        raise ValueError("source TRAIN manifest must contain at least 10 labeled patients")
    _validate_development(development_path, source, synthetic=synthetic, dataset=dataset)
    return train_sha, source


def _path_hash(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": digest(path)}


def _validate_child_plan(child_path: Path, *, expected_train: Path, expected_dev: Path,
                         k: int, arms: list[str], seed: int,
                         checkpoint_every_epochs: int | None | object = _UNSET,
                         decoder_chunk_size: int | None | object = _UNSET,
                         semi_supervised: dict[str, Any] | None | object = _UNSET) -> dict[str, Any]:
    expected_train, expected_dev = _resolve(expected_train), _resolve(expected_dev)
    child = _load_json(child_path)
    if child.get("schema") != "self_audit.candidate_a3d.plan.v1" or child.get("track") != "candidate_a3d":
        raise ValueError(f"invalid child plan: {child_path}")
    if child.get("training_seeds") != [seed] or child.get("ablation_arms") != arms:
        raise ValueError(f"child plan seed/arms mismatch: {child_path}")
    if _resolve(child.get("train", "")) != expected_train or _resolve(child.get("development", "")) != expected_dev:
        raise ValueError(f"child plan manifest mismatch: {child_path}")
    if child.get("train_sha256") != digest(expected_train):
        raise ValueError(f"child plan TRAIN manifest changed: {child_path}")
    if child.get("development_sha256") != digest(expected_dev):
        raise ValueError(f"child plan development manifest changed: {child_path}")
    runtime = child.get("runtime_overrides")
    if runtime is None and checkpoint_every_epochs is _UNSET and decoder_chunk_size is _UNSET:
        runtime = {}
    if not isinstance(runtime, dict):
        raise TypeError(f"child plan runtime overrides are missing: {child_path}")
    if (checkpoint_every_epochs is not _UNSET
            and runtime.get("checkpoint_every_epochs") != checkpoint_every_epochs):
        raise ValueError(f"child plan checkpoint override mismatch: {child_path}")
    if (decoder_chunk_size is not _UNSET
            and runtime.get("decoder_chunk_size") != decoder_chunk_size):
        raise ValueError(f"child plan decoder override mismatch: {child_path}")
    train = _load_json(expected_train)
    if train.get("label_budget") != k or train.get("fewshot", {}).get("k") != k:
        raise ValueError(f"child plan is not the expected k={k} manifest")
    if len(child.get("jobs", [])) != len(arms) + 2:
        raise ValueError(f"child plan job count mismatch: {child_path}")
    for job in child["jobs"]:
        config_path = _resolve(job.get("config", ""))
        if digest(config_path) != job.get("config_sha256"):
            raise ValueError(f"child recipe changed: {config_path}")
        config = _load_json(config_path)
        if (checkpoint_every_epochs is not _UNSET
                and config.get("checkpoint_every_epochs") != checkpoint_every_epochs):
            raise ValueError(f"child recipe checkpoint override mismatch: {config_path}")
        if decoder_chunk_size is not _UNSET and decoder_chunk_size is not None:
            student = config.get("student")
            if not isinstance(student, dict) or student.get("chunk_size") != decoder_chunk_size:
                raise ValueError(f"child recipe decoder override mismatch: {config_path}")
        if semi_supervised is not _UNSET and config.get("semi_supervised") != semi_supervised:
            raise ValueError(f"child recipe semi-supervised block mismatch: {config_path}")
    if semi_supervised is not _UNSET:
        if child.get("semi_supervised") != semi_supervised:
            raise ValueError(f"child plan semi-supervised block mismatch: {child_path}")
    return child


def create_fewshot_plan(*, dataset: str, train_manifest: str | Path,
                        development_manifest: str | Path, model_config: str | Path | None,
                        out: str | Path, config: str | Path = DEFAULT_CONFIG,
                        synthetic: bool = False, checkpoint_every_epochs: int | None = None,
                        decoder_chunk_size: int | None = None,
                        semi_config: str | Path | None = None) -> Path:
    if dataset not in {"acdc", "mnms"}:
        raise ValueError("dataset must be acdc or mnms")
    campaign = _resolve(out)
    if campaign.exists():
        raise ValueError("few-shot campaign output must be a new directory")
    config_path = _resolve(config)
    settings = _validate_config(config_path)
    train_path, development_path = _resolve(train_manifest), _resolve(development_manifest)
    if not train_path.is_file() or not development_path.is_file():
        raise ValueError("TRAIN and development manifests must exist")
    train_sha, _source = _validate_source(train_path, development_path, dataset=dataset, synthetic=synthetic)
    if campaign.is_relative_to(train_path.parent) or train_path.parent.is_relative_to(campaign):
        raise ValueError("campaign output must be separate from the source cache")
    if campaign.is_relative_to(development_path.parent) or development_path.parent.is_relative_to(campaign):
        raise ValueError("campaign output must be separate from the development cache")
    model_path = _resolve(model_config) if model_config is not None else None
    _validate_model_config(model_path, synthetic=synthetic)
    semi_path, semi_sha256, semi_block = _load_semi_config(semi_config, track="candidate_a3d")
    arms = list(settings["arms"])
    checkpoint_every = (settings["checkpoint_every_epochs"]
                        if checkpoint_every_epochs is None else checkpoint_every_epochs)
    decoder_chunk = (settings["decoder_chunk_size"]
                     if decoder_chunk_size is None else decoder_chunk_size)
    if (isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int)
            or checkpoint_every < 0):
        raise ValueError("checkpoint_every_epochs must be a non-negative integer")
    if (decoder_chunk is not None and
            (isinstance(decoder_chunk, bool) or not isinstance(decoder_chunk, int) or decoder_chunk < 1)):
        raise ValueError("decoder_chunk_size must be a positive integer")
    prepared: list[tuple[int, dict[str, Any], list[str]]] = []
    for k in settings["shots"]:
        manifest = build_subset_manifest(train_path, train_sha, k=k, seed=settings["seed"],
                                          dataset=dataset, synthetic=synthetic)
        prepared.append((k, manifest, manifest["fewshot"]["selected_ids"]))
    campaign.mkdir(parents=True)
    manifests, children = [], []
    for k, manifest, selected_ids in prepared:
        manifest_path = campaign / "manifests" / f"k{k}" / "train.json"
        manifest_path.parent.mkdir(parents=True)
        write_json(manifest_path, manifest)
        child_path = campaign / f"k{k}" / "PLAN.json"
        child_plan_path = create_plan(track="candidate_a3d", dataset=dataset,
            train_manifest=manifest_path, development_manifest=development_path,
            model_config=model_path, out=child_path.parent, seeds=[settings["seed"]],
            arms=arms, epochs=settings["epochs"], pretrain_epochs=settings["pretrain_epochs"],
            warmup_epochs=settings["warmup_epochs"], synthetic=synthetic,
            checkpoint_every_epochs=checkpoint_every, decoder_chunk_size=decoder_chunk,
            semi_config=semi_path)
        child = _validate_child_plan(child_plan_path, expected_train=manifest_path,
            expected_dev=development_path, k=k, arms=arms, seed=settings["seed"],
            checkpoint_every_epochs=checkpoint_every, decoder_chunk_size=decoder_chunk,
            semi_supervised=semi_block if semi_path is not None else _UNSET)
        manifests.append({"k": k, "selected_ids": selected_ids, **_path_hash(manifest_path)})
        children.append({"k": k, "plan": str(child_plan_path), "plan_sha256": digest(child_plan_path),
                         "jobs": len(child["jobs"]), "manifest_sha256": digest(manifest_path)})
    plan = {"schema": PLAN_SCHEMA, "track": "candidate_a3d", "dataset": dataset,
        "synthetic": synthetic, "seed": settings["seed"], "shots": list(settings["shots"]),
        "arms": arms, "epochs": settings["epochs"], "pretrain_epochs": settings["pretrain_epochs"],
        "warmup_epochs": settings["warmup_epochs"], "source_train": _path_hash(train_path),
        "config": _path_hash(config_path),
        "checkpoint_every_epochs": checkpoint_every, "decoder_chunk_size": decoder_chunk,
        "runtime_override_source": {
            "checkpoint_every_epochs": "cli" if checkpoint_every_epochs is not None else "config",
            "decoder_chunk_size": "cli" if decoder_chunk_size is not None else "config",
        },
        "source_development": _path_hash(development_path),
        "model_config": _path_hash(model_path) if model_path is not None else None,
        "semi_config": ({"path": str(semi_path), "sha256": semi_sha256}
                        if semi_path is not None else None),
        "semi_supervised": semi_block,
        "weights_policy": "weights:null; each k has independent supervised teacher pretrain",
        "selection": "nested_sha256_ranked_source_labeled_patients", "unit": "dense_labeled_train_patients",
        "arms_per_k": len(arms), "jobs_per_k": len(arms) + 2,
        "total_jobs": len(children) * (len(arms) + 2), "manifests": manifests, "child_plans": children}
    plan_path = campaign / "FEWSHOT_PLAN.json"
    write_json(plan_path, plan)
    return plan_path


def _verify_root_plan(path: Path) -> dict[str, Any]:
    plan = _load_json(path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("track") != "candidate_a3d":
        raise ValueError("few-shot plan schema/track mismatch")
    if plan.get("seed") != EXPECTED_SEED or plan.get("shots") != list(EXPECTED_SHOTS):
        raise ValueError("few-shot plan must use seed 42 and k=[2,5,8,10]")
    if len(plan.get("arms", [])) != 21 or len(set(plan["arms"])) != 21:
        raise ValueError("few-shot plan must contain all 21 arms")
    if (isinstance(plan.get("checkpoint_every_epochs"), bool)
            or not isinstance(plan.get("checkpoint_every_epochs"), int)
            or plan["checkpoint_every_epochs"] < 0):
        raise ValueError("few-shot plan checkpoint cadence is invalid")
    if (plan.get("decoder_chunk_size") is not None and
            (isinstance(plan["decoder_chunk_size"], bool)
             or not isinstance(plan["decoder_chunk_size"], int)
             or plan["decoder_chunk_size"] < 1)):
        raise ValueError("few-shot plan decoder chunk size is invalid")
    for field in ("epochs", "pretrain_epochs", "warmup_epochs"):
        value = plan.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"few-shot plan {field} is invalid")
    for key in ("source_train", "source_development"):
        item = plan.get(key)
        if not isinstance(item, dict) or digest(_resolve(item["path"])) != item["sha256"]:
            raise ValueError(f"{key} changed after planning")
    config_item = plan.get("config")
    config_raw_path = config_item.get("path") if isinstance(config_item, dict) else None
    if (not isinstance(config_raw_path, str) or not config_raw_path
            or not _resolve(config_raw_path).is_file()
            or digest(_resolve(config_raw_path)) != config_item.get("sha256")):
        raise ValueError("few-shot config changed after planning")
    settings = _validate_config(_resolve(config_raw_path))
    for field in ("arms", "epochs", "pretrain_epochs", "warmup_epochs"):
        if plan.get(field) != settings[field]:
            raise ValueError(f"few-shot plan {field} differs from pinned config")
    override_source = plan.get("runtime_override_source")
    if not isinstance(override_source, dict):
        raise TypeError("few-shot runtime override provenance is missing")
    for field in ("checkpoint_every_epochs", "decoder_chunk_size"):
        source = override_source.get(field)
        if source not in {"config", "cli"}:
            raise ValueError(f"few-shot runtime override provenance is invalid: {field}")
        config_value = settings.get(field, 0) if field == "checkpoint_every_epochs" else settings.get(field)
        if source == "config" and plan.get(field) != config_value:
            raise ValueError(f"few-shot plan {field} differs from pinned config")
    if plan.get("model_config") is not None:
        item = plan["model_config"]
        if digest(_resolve(item["path"])) != item["sha256"]:
            raise ValueError("model config changed after planning")
        _validate_model_config(_resolve(item["path"]), synthetic=bool(plan.get("synthetic", False)))
    semi_descriptor = plan.get("semi_config")
    if semi_descriptor is not None:
        if not isinstance(semi_descriptor, dict):
            raise ValueError("few-shot semi_config provenance is invalid")
        semi_path, semi_sha256, semi_block = _load_semi_config(semi_descriptor.get("path"), track="candidate_a3d")
        if semi_sha256 != semi_descriptor.get("sha256") or semi_block != plan.get("semi_supervised"):
            raise ValueError("few-shot semi-supervised config changed after planning")
    elif plan.get("semi_supervised") is not None:
        raise ValueError("few-shot semi-supervised block has no pinned source config")
    if len(plan.get("child_plans", [])) != 4:
        raise ValueError("few-shot plan must contain four child plans")
    seen = set()
    for record in plan["child_plans"]:
        k = record.get("k")
        if k in seen or k not in EXPECTED_SHOTS:
            raise ValueError("child plan k set is invalid")
        seen.add(k)
        child_path = _resolve(record["plan"])
        if digest(child_path) != record["plan_sha256"]:
            raise ValueError(f"child plan changed after planning: {child_path}")
        manifest_record = next((m for m in plan["manifests"] if m["k"] == k), None)
        if manifest_record is None:
            raise ValueError(f"missing few-shot manifest for k={k}")
        manifest_path = _resolve(manifest_record["path"])
        if digest(manifest_path) != record["manifest_sha256"]:
            raise ValueError(f"few-shot manifest changed after planning: k={k}")
        _validate_child_plan(child_path, expected_train=manifest_path,
            expected_dev=_resolve(plan["source_development"]["path"]),
            k=k, arms=plan["arms"], seed=plan["seed"],
            checkpoint_every_epochs=plan["checkpoint_every_epochs"],
            decoder_chunk_size=plan["decoder_chunk_size"],
            semi_supervised=plan.get("semi_supervised", _UNSET))
    return plan


def run_fewshot_plan(path: str | Path, *, execute: bool = False, device: str = "cuda") -> dict[str, Any]:
    plan_path = _resolve(path)
    plan = _verify_root_plan(plan_path)
    if not execute:
        return {"status": "PLAN_VALIDATED_NOT_RUN", "dataset": plan["dataset"],
                "shots": plan["shots"], "total_jobs": plan["total_jobs"]}
    if any(_resolve(record["plan"]).parent.joinpath("RUN_RECEIPT.json").exists()
           for record in plan["child_plans"]):
        raise ValueError("one or more few-shot child plans already ran; no implicit resume")
    results = []
    for record in plan["child_plans"]:
        child_path = _resolve(record["plan"])
        result = run_plan(child_path, execute=True, device=device)
        if result.get("status") != "COMPLETED":
            raise RuntimeError(f"few-shot child plan failed for k={record['k']}")
        results.append({"k": record["k"], "plan": str(child_path), "result": result})
    receipt = {"schema": RECEIPT_SCHEMA, "status": "COMPLETED", "plan": str(plan_path),
        "dataset": plan["dataset"], "shots": plan["shots"], "total_jobs": plan["total_jobs"],
        "results": results}
    write_json(plan_path.parent / "FEWSHOT_RUN_RECEIPT.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--dataset", choices=("acdc", "mnms"), required=True)
    plan_parser.add_argument("--train", required=True)
    plan_parser.add_argument("--development", required=True)
    plan_parser.add_argument("--model-config")
    plan_parser.add_argument("--out", required=True)
    plan_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    plan_parser.add_argument("--checkpoint-every-epochs", type=int,
                             help="override periodic checkpoint cadence; 0 keeps final only")
    plan_parser.add_argument("--decoder-chunk-size", type=int,
                             help="override student decoder chunk size")
    plan_parser.add_argument("--semi-config", help="pinned JSON file containing the A-3D semi_supervised block")
    plan_parser.add_argument("--synthetic", action="store_true")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--plan", required=True)
    run_parser.add_argument("--execute", action="store_true")
    run_parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = create_fewshot_plan(dataset=args.dataset, train_manifest=args.train,
            development_manifest=args.development, model_config=args.model_config,
            out=args.out, config=args.config, synthetic=args.synthetic,
            checkpoint_every_epochs=args.checkpoint_every_epochs,
            decoder_chunk_size=args.decoder_chunk_size,
            semi_config=args.semi_config)
        print(json.dumps({"status": "NOT_RUN", "plan": str(plan), "sha256": digest(plan)}))
    else:
        print(json.dumps(run_fewshot_plan(args.plan, execute=args.execute, device=args.device), indent=2))


if __name__ == "__main__":
    main()
