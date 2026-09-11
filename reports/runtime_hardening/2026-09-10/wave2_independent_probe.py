"""Astra independent CPU artifact-boundary probes; synthetic evidence only."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
import test_transition_bank as bank_fixture
from scripts.evaluate_external_mnms import run_external_evaluation
from self_audit.training._utils import save_checkpoint
from self_audit.provenance import file_sha256, state_digest
from self_audit.evaluation.threshold import save_calibration, load_calibration
from self_audit.evaluation.transition_bank import TransitionEvaluator, dump_bank, load_bank

def strict_read(path):
    def reject(value):
        raise AssertionError(value)
    return json.loads(path.read_text(), parse_constant=reject)

with tempfile.TemporaryDirectory(prefix="astra-wave2-") as directory:
    root = Path(directory)
    model = bank_fixture.build_model()
    cfg = {"pretrained_encoder": False, "shared_channels": 16, "window_k": 4, "max_turns": 3}
    checkpoint = save_checkpoint(root / "best.pt", model, config={"dataset": "acdc", "model": cfg})
    for name, array in (("volumes", np.zeros((32,32,2), dtype=np.float32)), ("masks", np.zeros((32,32,2), dtype=np.uint8))):
        folder = root / "testing" / name
        folder.mkdir(parents=True)
        np.save(folder / "A100_t00.npy", array)
    output = root / "external.json"
    external_config = {"external_test": {"dataset": "mnms", "split": "testing", "raw_to_acdc": {0:0,1:3,2:2,3:1}}, "model": cfg, "num_classes": 4, "image_size": 32, "depth_axis": 2, "audit": {"tau_accept": 0.25, "t_max": 1}}
    # Only the metric payload is injected; real dataset dispatch and checkpoint binding run.
    with patch("scripts.evaluate_external_mnms.evaluate_volume_cohort", return_value={"nested": {"undefined": np.array([np.nan, np.inf]), "zero": 0.0}}):
        result = run_external_evaluation(config=external_config, checkpoint=checkpoint, data_root=root, device="cpu", output=output)
    loaded = strict_read(output)
    assert loaded["checkpoint_binding"]["checkpoint_sha256"] == file_sha256(checkpoint)
    assert loaded["checkpoint_binding"]["state_digest"] == state_digest(model)
    assert loaded["metrics"]["nested"] == {"undefined": [None, None], "zero": 0.0}
    print("PASS external: real factory, real checkpoint binding, nested undefined JSON")

    calibration = root / "calibration.json"
    kwargs = dict(tau_accept=0.1, neutral_margin=0.005, source_split="val", checkpoint_path=None, t_max=3, threshold_grid=[0.0,0.1,0.2], selected_row={"tau_accept":0.1,"final_macro_dice":0.8}, metric_space="slice_proxy")
    save_calibration(calibration, **kwargs)
    load_calibration(calibration)
    old = calibration.read_bytes()
    with patch("os.fsync", side_effect=OSError(28,"injected fsync failure")):
        try: save_calibration(calibration, **kwargs)
        except OSError: pass
        else: raise AssertionError("fsync failure swallowed")
    assert calibration.read_bytes() == old
    load_calibration(calibration)
    print("PASS calibration: valid artifact remains loadable after fsync failure")

    proposals = bank_fixture.make_proposals(model, bank_fixture.fixed_images(), tau_accept=-1.0)
    rows = TransitionEvaluator().evaluate(proposals, targets=bank_fixture.targets_from(bank_fixture.structured_reference()))
    bank = bank_fixture.assemble(rows, model=model, tau_accept=-1.0)
    destination = dump_bank(bank, root / "bank.json")
    original = destination.read_bytes()
    load_bank(destination)
    with patch("os.replace", side_effect=OSError(28,"injected replace failure")):
        try: dump_bank(bank, destination)
        except OSError: pass
        else: raise AssertionError("replace failure swallowed")
    assert destination.read_bytes() == original
    assert load_bank(destination)["integrity"]["content_signature"] == bank["integrity"]["content_signature"]
    assert not list(root.glob(".*.tmp"))
    print("PASS bank: real generated sealed bank survives failed replacement with signature intact")
print("ASTRA_WAVE2_REAL_ENTRYPOINT_PROBES_PASS")
