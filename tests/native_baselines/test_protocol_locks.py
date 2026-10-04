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
        dss = [path for path in (ROOT / "baseline/DSS_US/config/native").glob("*.yaml")
               if not "_paper_faithful" in path.name]  # paper-faithful profiles: own tests
        self.assertEqual(len(dss), 11)
        for path in dss:
            with self.subTest(path=path), self.assertRaises(ProtocolBlocked):
                load_lock(path)
            if path.name.startswith("step2_"):
                with self.assertRaises(ProtocolBlocked):
                    load_lock(path, purpose="native_track_b")
            else:  # Step I evaluator resolved from paper + official evaluation path
                self.assertEqual(load_lock(path, purpose="native_track_b")["native_track_b"]["n_classes"], 4)
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
            if "_paper_faithful" in path.name:
                continue  # covered by baseline/SGSCN/tests/test_sgscn_paper_faithful.py
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

    def test_dss_us_evidence_keeps_producers_blocked_and_resolves_only_step1_evaluator(self):
        spec = json.loads((ROOT / "baseline/DSS_US/evaluation/track_b/spec.json").read_text())
        self.assertEqual(spec["step_I"]["status"], "RESOLVED")
        self.assertEqual(spec["step_I"]["iou_thresh"]["value"], 0.0)
        self.assertTrue(spec["step_II"]["status"].endswith("BLOCKED_PROTOCOL"))
        for path in sorted((ROOT / "baseline/DSS_US/config/native").glob("*.yaml")):
            if "_paper_faithful" in path.name:
                continue  # covered by baseline/DSS_US/tests/test_dss_us_paper_faithful.py
            config = json.loads(path.read_text())
            producer = " ".join(config["gates"]["producer"])
            for reason in ("row preprocessing/affinity/spectral/CRF recipe", "CAMUS cohort", "CRF parameter set"):
                self.assertIn(reason, producer)
            self.assertEqual(config["evidence"]["camus_cohort"]["status"], "UNRESOLVED")
            self.assertIn("dino_deitsmall8_pretrain.pth", config["evidence"]["backbone_checkpoint"]["value"])
            if config["stage"] == "I":
                self.assertEqual(config["gates"]["native_track_b"], [])
                self.assertEqual(config["native_track_b"]["evaluated_stage"], "crf_multi_region")
            else:
                self.assertEqual(len(config["gates"]["native_track_b"]), 2)
                self.assertIn("semantic clusters", producer)

    def test_sgscn_paper_gates_record_every_unresolved_discrepancy(self):
        spec = json.loads((ROOT / "baseline/SGSCN/evaluation/track_b/spec.json").read_text())
        self.assertEqual(spec["evidence"]["HM"]["type"], "UNRESOLVED")
        self.assertIn("130.8", spec["evidence"]["HM"]["source"])
        for name in ("ph2", "sysu_us"):
            paper = json.loads((ROOT / f"baseline/SGSCN/config/native/{name}_paper.yaml").read_text())
            gates = " ".join(paper["gates"]["producer"])
            for reason in ("Architecture", "Loss", "Stopping"):
                self.assertIn(reason, gates)
            self.assertIn("SYSU-US cohort" if name == "sysu_us" else "PH2 input format", gates)
            reference = json.loads((ROOT / f"baseline/SGSCN/config/native/{name}_official_reference.yaml").read_text())
            self.assertEqual(reference["gates"]["producer"], [])
            self.assertEqual(reference["scientific"], paper["scientific"])
            self.assertEqual(reference["scientific"]["spatial_weight"], 5)
            self.assertEqual(len(reference["gates"]["native_track_b"]), 2)

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
