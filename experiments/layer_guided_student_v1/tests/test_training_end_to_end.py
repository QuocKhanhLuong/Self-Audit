"""Exercise the real bank, loop, serialization and inference on manufactured data.

Only the external pretrained constructor is replaced. This is software evidence
with a small encoder, not official DINO weights or patient/GPU training.
"""

import copy
import json

import numpy as np
import torch
from integration import artifacts
from test_training_checkpoint import make_model
from test_training_data import _make_bank, _sha
from training_v1 import checkpoint, runner
from training_v1.config import DEFAULT_RECIPE


def test_bank_to_updates_to_real_checkpoint_to_native_prediction(tmp_path, monkeypatch):
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    _make_bank(bank_dir)
    source = tmp_path / "official_source"
    source.mkdir()
    (source / "hubconf.py").write_text("# TEST ONLY; never executed\n")
    weights = tmp_path / "fixture.pth"
    weights.write_bytes(b"manufactured constructor fixture")
    approval = {
        "schema": "layer_guided.backbone_approval.v1",
        "approved": True,
        "model": "dinov3_convnext_tiny",
        "lineage": "official_self_supervised_dinov3",
        "repo": str(source),
        "weights": str(weights),
        "weights_sha256": artifacts.sha256(weights),
        "source_commit": "a" * 40,
        "review_record": "TEST_ONLY_NOT_PRETRAINED",
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval))
    provenance = {**approval, "verified_weights_sha256": approval["weights_sha256"]}
    constructed = []

    def construct(recipe, approval, device="cpu"):
        model = make_model(recipe).to(device)
        model.integration_identity["backbone"] = copy.deepcopy(provenance)
        constructed.append((model, model.coarse_head.weight.detach().clone()))
        return model, copy.deepcopy(provenance)

    monkeypatch.setattr(runner, "construct_student", construct)
    monkeypatch.setattr(artifacts, "construct_student", construct)
    config = {**DEFAULT_RECIPE, "epochs": 2, "max_steps": 5, "threads": 1}
    output = tmp_path / "run"
    report = runner.run_training(
        bank=bank_dir,
        expected_bank_sha256=_sha(bank_dir),
        approval=approval_path,
        expected_approval_sha256=artifacts.sha256(approval_path),
        config=config,
        out=output,
        device="cpu",
    )
    assert report["optimizer_steps"] == 5
    assert report["semantic_updates"] > 0
    assert report["epochs_completed"] == 0
    assert report["epochs_started"] == 1
    trained, initial_head = constructed[0]
    assert not torch.equal(trained.coarse_head.weight, initial_head)
    assert torch.count_nonzero(trained.detail_head.weight) == 0
    assert trained.detail_head.weight.grad is None
    restored, metadata = checkpoint.load_checkpoint(
        output / "final",
        report["final_manifest_sha256"],
        approval,
    )
    assert metadata["status"] == checkpoint.STATUS_PSEUDO
    assert len(metadata["lineage"]["train_patients"]) == 2
    assert len(metadata["lineage"]["planned_train_patients"]) == 3
    images = torch.rand(1, 3, 17, 19)
    with torch.no_grad():
        torch.testing.assert_close(
            trained.eval()(images)["logits"], restored(images)["logits"], rtol=0, atol=0
        )
    cine = np.arange(2 * 2 * 17 * 19, dtype=np.float32).reshape(2, 2, 17, 19)
    dense, gated, confidence, entropy, receipt = checkpoint.predict_cine(
        restored,
        metadata,
        cine,
        image_sha256="e" * 64,
    )
    assert dense.shape == gated.shape == confidence.shape == entropy.shape == cine.shape
    assert set(np.unique(dense)) <= {0, 1, 2, 3}
    assert receipt["bank_or_training_gate_mutated"] is False
