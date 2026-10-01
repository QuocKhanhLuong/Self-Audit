import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from shared_benchmark.native_protocol import ProtocolBlocked, load_lock, native_status


class ProtocolTests(unittest.TestCase):
    def test_all_paper_profiles_block_before_execution(self):
        dss = ROOT / "baseline/DSS_US/config/native"
        self.assertEqual(len(list(dss.glob("*.yaml"))), 11)
        for path in dss.glob("*.yaml"):
            with self.subTest(path=path), self.assertRaises(ProtocolBlocked):
                load_lock(path)
            with self.assertRaises(ProtocolBlocked):
                load_lock(path, purpose="native_track_b")
        for name in ("ph2", "sysu_us"):
            with self.assertRaises(ProtocolBlocked):
                load_lock(ROOT / f"baseline/SGSCN/config/native/{name}_paper.yaml")

    def test_reference_profiles_preserve_settings(self):
        for name, lr in (("ph2", .1), ("sysu_us", .05)):
            cfg = load_lock(ROOT / f"baseline/SGSCN/config/native/{name}_official_reference.yaml")
            self.assertEqual(cfg["scientific"]["learning_rate"], lr)
            self.assertTrue(cfg["scientific"]["context_loss"])
            self.assertEqual(cfg["scientific"]["max_iterations"], 50)
            self.assertEqual(cfg["paper_equivalence"], "UNRESOLVED")

    def test_edit_cannot_remove_gate_without_invalidating_index(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            source = ROOT / "baseline/DSS_US/config/native"
            config = json.loads((source / "step1_ours_comb.yaml").read_text())
            config["gates"]["producer"] = []
            (target / "step1_ours_comb.yaml").write_text(json.dumps(config))
            (target / "index.json").write_bytes((source / "index.json").read_bytes())
            with self.assertRaisesRegex(ProtocolBlocked, "frozen index"):
                load_lock(target / "step1_ours_comb.yaml")

    def test_gpl_source_fingerprints(self):
        base = ROOT / "baseline/SGSCN"
        receipt = json.loads((base / "upstream/receipt.json").read_text())
        self.assertEqual(receipt["upstream_commit"], "592efb6e72ceeef15c8be0630a4673eda5dce6f5")
        for name, expected in receipt["files"].items():
            self.assertEqual(hashlib.sha256((base / name).read_bytes()).hexdigest(), expected)

    def test_evaluator_edit_cannot_reuse_a_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            config_dir = target / "config/native"
            config_dir.mkdir(parents=True)
            evaluation_dir = target / "evaluation/track_b"
            evaluation_dir.mkdir(parents=True)
            source = ROOT / "baseline/SGSCN"
            for name in ("index.json", "ph2_official_reference.yaml"):
                shutil.copyfile(source / "config/native" / name, config_dir / name)
            spec = json.loads((source / "evaluation/track_b/spec.json").read_text())
            spec["tie_policy"] = "invented"
            (evaluation_dir / "spec.json").write_text(json.dumps(spec))
            with self.assertRaisesRegex(ProtocolBlocked, "evaluator specification differs"):
                load_lock(config_dir / "ph2_official_reference.yaml", purpose="native_track_b")

    def test_sgscn_context_loss_evidence_and_paper_gates(self):
        for path in sorted((ROOT / "baseline/SGSCN/config/native").glob("*.yaml")):
            config = json.loads(path.read_text())
            evidence = config["evidence"]
            self.assertEqual(evidence["context_loss"]["status"], "VERIFIED_PAPER")
            self.assertIn("--center", evidence["context_loss_code_invocation"]["value"])
            self.assertIn("DISCREPANCY", evidence["spatial_weight"]["source"])
            self.assertEqual(config["scientific"]["spatial_weight"], 5)
            if path.stem.endswith("_official_reference"):
                self.assertIn("not exact paper reproduction", config["reference_invocation"])
                self.assertEqual(config["paper_equivalence"], "UNRESOLVED")
            else:
                self.assertTrue(any("spatial weight" in gate for gate in config["gates"]["producer"]))

    def test_native_statuses_independent_and_stochastic(self):
        status = native_status(producer_complete=True, track_b_status="COMPLETE",
                               track_a_status="BLOCKED_ADAPTER", same_seed_equal=False,
                               stochastic_behavior_documented=True)
        self.assertEqual(status["NATIVE_REPRODUCTION_STATUS"], "COMPLETE")
        self.assertEqual(status["NATIVE_TRACK_A_STATUS"], "BLOCKED_ADAPTER")
        blocked = native_status(producer_complete=True, track_b_status="COMPLETE",
                                track_a_status="COMPLETE", verification_reliable=False)
        self.assertEqual(blocked["NATIVE_REPRODUCTION_STATUS"], "BLOCKED_REPRODUCIBILITY")

    def test_missing_native_dataset_remains_partial(self):
        status = native_status(producer_complete=True, track_b_status="PARTIAL",
                               track_a_status="BLOCKED_ADAPTER", all_required_data=False)
        self.assertEqual(status["NATIVE_REPRODUCTION_STATUS"], "PARTIAL")


if __name__ == "__main__":
    unittest.main()
