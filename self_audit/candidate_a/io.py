"""Hash-pinned checkpoints and a self-contained, teacher-free student export."""
from pathlib import Path
import hashlib
import json
import torch
from self_audit.model import digest
from self_audit.data import write_json
from . import CLASS_MAPPING

CHECKPOINT_SCHEMA = "self_audit.candidate_a.checkpoint.v2"
EXPORT_SCHEMA = "self_audit.candidate_a.student.v2"


def source_inventory():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "self_audit").rglob("*.py")) + [root / "train.py", root / "evaluate.py"]
    return {str(p.relative_to(root)): digest(p) for p in paths}


def read_checkpoint(folder, expected_sha256):
    folder = Path(folder).resolve()
    manifest = folder / "MANIFEST.json"
    if digest(manifest) != expected_sha256:
        raise ValueError("checkpoint manifest hash mismatch")
    meta = json.loads(manifest.read_text())
    if meta.get("schema") not in {CHECKPOINT_SCHEMA, EXPORT_SCHEMA, "self_audit.checkpoint.v2"}:
        raise ValueError("unsupported checkpoint schema")
    if digest(folder / "weights.pt") != meta["payload_sha256"]:
        raise ValueError("checkpoint payload hash mismatch")
    if meta["schema"] in {CHECKPOINT_SCHEMA, EXPORT_SCHEMA}:
        current = source_inventory()
        current_sha = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
        if meta.get("source_files") != current or meta.get("source_sha256") != current_sha:
            raise ValueError("executed source differs from candidate-A checkpoint; use its pinned worktree")
    payload = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    return payload, meta


def build_student(config):
    from .bank import BankConfig, QueryBank
    from .codec import MeasurementCodec
    from .model import CandidateAStudent, StudentConfig
    bank = QueryBank(BankConfig(**config["bank"]))
    codec = MeasurementCodec(bank, **config["codec"])
    model = CandidateAStudent(bank, StudentConfig(**config["student"]))
    return model, bank, codec


def save_checkpoint(folder, *, student, teacher, optimizers, config, data, epoch, updates, provenance):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    payload = {"student": student.state_dict(), "teacher": teacher.state_dict() if teacher is not None else None,
               "optimizers": [o.state_dict() for o in optimizers], "torch_rng": torch.get_rng_state()}
    torch.save(payload, folder / "weights.pt")
    sources = source_inventory()
    meta = {"schema": CHECKPOINT_SCHEMA, "config": config, "epoch": epoch,
            "completed_sample_updates": updates,
            "class_mapping": CLASS_MAPPING, "model_class": "CandidateAStudent",
            "labeled_ids": data.lids, "unlabeled_ids": data.uids,
            "label_budget": len(data.lids), "label_reads": sorted(data.label_reads),
            "data_sha256": digest(data.path), "synthetic": bool(data.manifest["synthetic"]),
            "teacher_provenance": provenance, "source_files": sources,
            "source_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            "payload_sha256": digest(folder / "weights.pt"), "quality": "NOT_EVALUATED",
            "supervision": "dense labeled TRAIN subset; U images unused by this candidate",
            "resume": "NOT_SUPPORTED; teacher stage transfer supported", "prediction_model": "student"}
    write_json(folder / "MANIFEST.json", meta)
    return digest(folder / "MANIFEST.json")


def load_student(folder, expected_sha256, device="cpu"):
    payload, meta = read_checkpoint(folder, expected_sha256)
    if meta["schema"] not in {CHECKPOINT_SCHEMA, EXPORT_SCHEMA} or meta.get("class_mapping") != CLASS_MAPPING:
        raise ValueError("not a candidate-A student with the declared four-class ontology")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent device fallback")
    model, bank, codec = build_student(meta["config"])
    model.load_state_dict(payload["student"], strict=True)
    return model.to(device).eval(), bank, codec, meta


def export_student(checkpoint, checkpoint_sha256, out):
    # Load and validate exact architecture/state before emitting a new artifact.
    model, _, _, meta = load_student(checkpoint, checkpoint_sha256)
    if meta["schema"] == CHECKPOINT_SCHEMA and meta["config"]["stage"] != "joint":
        raise ValueError("student export requires joint/student training; teacher-only stages leave student untrained")
    out = Path(out).resolve()
    source = Path(checkpoint).resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("export must be separate from checkpoint")
    out.mkdir(parents=True, exist_ok=False)
    torch.save({"student": model.state_dict()}, out / "weights.pt")
    # Only inference recipes are retained. No teacher paths, weights or training
    # labels are needed to instantiate or run the exported model.
    config = {k: meta["config"][k] for k in ("bank", "codec", "student", "rollout", "class_mapping")}
    export = {k: meta[k] for k in ("class_mapping", "model_class", "labeled_ids", "unlabeled_ids",
                                 "label_budget", "synthetic", "quality", "source_files", "source_sha256")}
    export.update(schema=EXPORT_SCHEMA, config=config, payload_sha256=digest(out / "weights.pt"),
                  parent_manifest_sha256=checkpoint_sha256, prediction_model="student",
                  teacher_dependency=False, external_backbone_dependency=False)
    write_json(out / "MANIFEST.json", export)
    return {"path": str(out), "manifest_sha256": digest(out / "MANIFEST.json"),
            "payload_sha256": export["payload_sha256"], "synthetic": export["synthetic"]}
