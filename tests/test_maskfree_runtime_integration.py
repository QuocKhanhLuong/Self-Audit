"""Runtime integration checks; synthetic CPU evidence, never rental throughput."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from self_audit_maskfree import auditor
from self_audit_maskfree.export import export_prediction
from self_audit_maskfree.trainer import MaskfreeTrainer, TrainerContractError, ResumeIdentityError, _scalar, _scalar_metrics


def test_real_preflight_receipt_satisfies_full_run_validator(tmp_path, monkeypatch):
    """Exercise the writer and the exact validator used by run(), without CUDA."""
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import test_maskfree_trainer as fixture

    components = replace(fixture.build_components(), audit_banks=auditor.audit_banks,
                         export_prediction=export_prediction)
    config = fixture.build_config(tmp_path, total_epochs=150)
    probe = MaskfreeTrainer(config, components=components)
    receipt = probe.preflight()
    assert receipt["status"] == "pass", receipt
    assert receipt["audit_device_identity"] == receipt["audit_device"]
    runner = MaskfreeTrainer(config, components=components)
    runner.setup()
    original = runner.paths.gate_receipt.read_bytes()
    runner._validate_preflight_gate()
    assert runner.paths.gate_receipt.read_bytes() == original

    # Old/mismatched receipts remain rejected; the validator never repairs them.
    gate = json.loads(original)
    gate.pop("audit_device_identity")
    runner.paths.gate_receipt.write_text(json.dumps(gate))
    with pytest.raises(TrainerContractError, match="audit_device_identity"):
        runner._validate_preflight_gate()
    gate["audit_device_identity"] = {"type": "incorrect"}
    runner.paths.gate_receipt.write_text(json.dumps(gate))
    with pytest.raises(TrainerContractError, match="audit_device_identity"):
        runner._validate_preflight_gate()


def test_bulk_metric_scalars_preserve_values_and_filtering():
    metrics = {"fp32": torch.tensor(0.123456789), "fp64": torch.tensor(0.123456789, dtype=torch.float64),
               "flag": torch.tensor(True), "int": 5, "none": None, "vector": torch.ones(2),
               "nan": torch.tensor(float("nan"))}
    expected = {key: _scalar(value) for key, value in metrics.items()}
    assert _scalar_metrics(metrics) == expected


def test_checkpoint_offsets_are_durable_and_truncated_on_resume(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import test_maskfree_trainer as fixture
    config = fixture.build_config(tmp_path, total_epochs=3, max_steps=1)
    components = fixture.build_components()
    trainer = MaskfreeTrainer(config, components=components)
    trainer.run()
    checkpoint = trainer.paths.last_checkpoint
    payload = torch.load(checkpoint, weights_only=False)
    relative = "reports/label_lineage.jsonl"
    path = trainer.paths.root / relative
    durable = path.read_bytes()
    assert payload["log_offsets"][relative] == len(durable) > 0
    with path.open("ab") as handle:
        handle.write(b'{"uncheckpointed":true}\n')
    resumed = MaskfreeTrainer(config.replace(resume=str(checkpoint), max_steps=2), components=components)
    resumed.setup()
    resumed._load_checkpoint(checkpoint)
    assert path.read_bytes() == durable
    path.write_bytes(durable[:-1])
    with pytest.raises(ResumeIdentityError, match="shorter than durable offset"):
        resumed._load_checkpoint(checkpoint)


def test_changed_runtime_options_require_fresh_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import test_maskfree_trainer as fixture
    config = fixture.build_config(tmp_path, total_epochs=3, max_steps=1)
    trainer = MaskfreeTrainer(config, components=fixture.build_components())
    trainer.run()
    changed = config.replace(resume=str(trainer.paths.last_checkpoint), logging_mode="sync")
    with pytest.raises(ResumeIdentityError, match="runtime_options"):
        MaskfreeTrainer(changed, components=fixture.build_components()).setup()


@pytest.mark.parametrize("workers,prefetch", [(0, 1), (2, 1)])
def test_runtime_paths_preserve_scientific_batch_and_rng(tmp_path, monkeypatch, workers, prefetch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import test_maskfree_trainer as fixture
    import test_maskfree_trainer_performance as checks
    from self_audit_maskfree.data.prefetch import iter_batches

    class OrderedFixture(fixture.FakeImageOnlyDataset):
        def iter_batches(self, batches, **kwargs):
            return iter_batches(self, batches, **kwargs)

    def run_variant(root, *, worker_count, ahead, logging, steps=2, resume=None):
        captures = {key: [] for key in ("producer_loss", "producer_metrics", "student_targets",
                    "student_loss", "student_metrics", "gradients", "audits", "model_outputs")}
        base = replace(fixture.build_components(), dataset_factory=OrderedFixture)
        components = checks._instrumented_components(base, captures, bulk=True)
        trainer = MaskfreeTrainer(fixture.build_config(root, total_epochs=3, max_steps=steps, resume=resume,
                                  candidate_workers=worker_count, prefetch_batches=ahead,
                                  logging_mode=logging), components=components)
        checks._instrument_trainer(trainer, captures)
        trainer.run()
        checkpoint = torch.load(trainer.paths.last_checkpoint, weights_only=False)
        return captures, checkpoint, trainer.paths.last_checkpoint

    ref, ref_state, _ = run_variant(tmp_path / "ref", worker_count=0, ahead=0, logging="sync")
    fast, fast_state, _ = run_variant(tmp_path / "fast", worker_count=workers, ahead=prefetch, logging="buffered")
    checks._assert_nested_exact(ref, fast)
    for field in ("models", "optimizers", "rng", "sampling_generator", "component_steps"):
        checks._assert_nested_exact(ref_state[field], fast_state[field], path=field)
    _, _, partial_path = run_variant(tmp_path / "resume", worker_count=workers, ahead=prefetch, logging="buffered", steps=1)
    _, resumed_state, _ = run_variant(tmp_path / "resume", worker_count=workers, ahead=prefetch, logging="buffered",
                                      resume=str(partial_path))
    for field in ("models", "optimizers", "rng", "sampling_generator", "component_steps"):
        checks._assert_nested_exact(fast_state[field], resumed_state[field], path="resume." + field)
