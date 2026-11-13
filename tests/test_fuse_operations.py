from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location("test_" + name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FuseOperationTests(unittest.TestCase):
    def test_campaign_state_order_and_test_once_ledger(self):
        campaign = load_script("fuse_campaign.py")
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "RUN_STATE.json"
            state = campaign.load_state(state_path)
            self.assertEqual(state["test_ledger"]["status"], "NOT_RUN")
            state.update(completed=["T", "J", "VAL3", "EXPORT", "TEST_ONCE"],
                         test_ledger={"status": "COMMITTED"})
            state_path.write_text(json.dumps(state))
            self.assertEqual(campaign.load_state(state_path)["completed"][-1], "TEST_ONCE")
            state["completed"] = ["T", "J", "VAL3", "EXPORT", "EXPORT"]
            state_path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                campaign.load_state(state_path)

    def test_campaign_builds_three_mode_bundle_and_calls_single_transaction(self):
        campaign = load_script("fuse_campaign.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exports = root / "exports"
            exports.mkdir()
            for mode in ("teacher", "student", "fusion"):
                artifact = exports / f"{mode}.pt"
                artifact.write_bytes(mode.encode())
                import hashlib
                sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
                artifact.with_suffix(".pt.manifest.json").write_text(json.dumps({
                    "schema": "self_audit.candidate_fuse.export.v1",
                    "artifact": artifact.name,
                    "artifact_sha256": sha,
                    "artifact_bytes": artifact.stat().st_size,
                    "mode": mode,
                }))
            bundle = campaign.export_bundle(root)
            self.assertEqual(set(bundle), {"teacher", "student", "fusion"})
            plan = campaign.commands(root / "config.json", root / "run", device="cpu")
            self.assertIn("cpu", plan["T"])
            self.assertIn("cpu", plan["J"])
            config = root / "config.json"
            config.write_text(json.dumps({"data": {"test_manifest": "test.json",
                                                       "test_manifest_sha256": "a" * 64}}))
            transaction = {"status": "COMMITTED", "role": "test"}
            with mock.patch("self_audit.candidate_fuse.evaluation.evaluate_exports",
                            return_value=transaction) as evaluate_exports:
                result = campaign.evaluate_test_once(config, root, device="cpu")
            self.assertEqual(result, transaction)
            evaluate_exports.assert_called_once_with(bundle, "test.json", root / "TEST_ONCE",
                                                      "cpu", "test", "a" * 64)

    def test_uncommitted_t_state_is_archived_before_fresh_attempt(self):
        campaign = load_script("fuse_campaign.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "metrics.jsonl").write_text("old metrics\n")
            (root / "STATE.json").write_text(json.dumps({"status": "FAILED", "stage": "T"}))
            logs = root / "campaign_logs"
            logs.mkdir()
            (logs / "T.log").write_text("failed launch\n")
            state = {"schema": campaign.STATE_SCHEMA, "status": "FAILED", "completed": [],
                     "active": "T", "test_ledger": {"status": "NOT_RUN"}}
            archive = campaign.archive_uncommitted_t_attempt(root, state)
            self.assertIsNotNone(archive)
            self.assertFalse((root / "metrics.jsonl").exists())
            self.assertTrue((archive / "metrics.jsonl").exists())
            self.assertTrue((archive / "STATE.json").exists())
            self.assertTrue((archive / "campaign_logs" / "T.log").exists())
            self.assertTrue((archive / "ARCHIVE.json").is_file())

    def test_t_resume_requires_matching_checkpoint_state_sha_and_receipt(self):
        campaign = load_script("fuse_campaign.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "resume.pt"
            checkpoint.write_bytes(b"committed")
            import hashlib
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (root / "resume.sha256.json").write_text(json.dumps({"path": "resume.pt", "sha256": digest}))
            (root / "STATE.json").write_text(json.dumps({"stage": "T", "checkpoint_path": str(checkpoint),
                                                           "checkpoint_sha256": digest}))
            self.assertTrue(campaign.has_committed_t_checkpoint(root))
            checkpoint.write_bytes(b"corrupted")
            self.assertFalse(campaign.has_committed_t_checkpoint(root))

    def test_campaign_refuses_config_drift_on_resume(self):
        campaign = load_script("fuse_campaign.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "config.json"
            config.write_text(json.dumps({"runtime": {"synthetic": True}}))
            output = root / "run"
            output.mkdir()
            (output / "RUN_STATE.json").write_text(json.dumps({
                "schema": campaign.STATE_SCHEMA, "status": "FAILED", "completed": [],
                "active": "T", "attempts": {}, "test_ledger": {"status": "NOT_RUN"},
                "config_sha256": "0" * 64,
            }))
            with self.assertRaisesRegex(ValueError, "config SHA"):
                campaign.run_campaign(config, output, execute=True, resume=True, device="cpu")

    def test_watchdog_healthy_run_is_observational(self):
        watchdog = load_script("fuse_watchdog.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "STATE.json").write_text(json.dumps({"status": "RUNNING", "stage": "J",
                "epoch": 1, "update": 2, "updated_utc": "2026-11-13T00:00:00+00:00"}))
            (root / "RUN_STATE.json").write_text(json.dumps({"status": "RUNNING", "active": "J"}))
            (root / "metrics.jsonl").write_text(json.dumps({"type": "validation", "stage": "J",
                "epoch": 1, "update": 2, "patient_id": "must-not-leak",
                "modes": {"fusion": {"fg_dice": .6, "per_class": {
                    "RV": {"dice": .5}, "MYO": {"dice": .6}, "LV": {"dice": .7}}},
                    "per_patient": {"patient-secret": {"fg_dice": .1}}},
                "gate_means": [.1]*5}) + "\n")
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            result = watchdog.inspect_run(root, now=1794528000, stale_seconds=1800)
            self.assertTrue(result["healthy"])
            self.assertFalse(any("stale" in x for x in result["alerts"]))
            self.assertNotIn("patient_id", result["latest_metric"])
            self.assertNotIn("per_patient", result["latest_metric"]["modes"]["fusion"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_watchdog_follows_campaign_when_engine_stage_is_completed(self):
        watchdog = load_script("fuse_watchdog.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "STATE.json").write_text(json.dumps({"status": "COMPLETED", "stage": "T",
                "epoch": 4, "update": 8, "updated_utc": "2026-11-13T00:00:00+00:00"}))
            (root / "RUN_STATE.json").write_text(json.dumps({"status": "RUNNING", "active": "J"}))
            result = watchdog.inspect_run(root, now=1794528000, stale_seconds=1800)
            self.assertEqual(result["status"], "RUNNING")
            self.assertEqual(result["stage"], "J")
            self.assertTrue(result["healthy"])

    def test_switch_snapshot_hashes_without_deleting_history(self):
        switch = load_script("fuse_switch.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "old"
            root.mkdir()
            checkpoint = root / "resume.pt"
            checkpoint.write_bytes(b"checkpoint boundary")
            log = root / "train.log"
            log.write_text("historical log")
            receipt = Path(td) / "SWITCH_RECEIPT.json"
            result = switch.snapshot_boundary(root, receipt, checkpoint=checkpoint,
                                              stage="J", update=200,
                                              logs=[log], metadata=[], writers_stopped=True)
            self.assertTrue(any(x["path"] == "train.log" for x in result["artifacts"]))
            self.assertEqual(result["checkpoint_boundary"]["update"], 200)
            self.assertTrue((root / "train.log").exists())
            switch.mark_superseded(receipt, ["old-job"])
            with self.assertRaises(ValueError):
                switch.mark_superseded(receipt, ["old-job"])

    def test_metric_parser_shows_modes_and_five_gates(self):
        metrics = load_script("fuse_watch_metrics.py")
        row = {"type": "validation", "stage": "J", "epoch": 1, "update": 80,
               "modes": {m: {"fg_dice": .7, "per_class": {
                   "RV": {"dice": .6}, "MYO": {"dice": .7}, "LV": {"dice": .8}}}
                         for m in ("teacher", "student", "fusion")},
               "gate_means": [.1, .2, .3, .4, .5]}
        shown = metrics.format_metric(row)
        self.assertIn("teacher FG/RV/MYO/LV", shown)
        self.assertIn("60.00", shown)
        self.assertIn("G=", shown)
        self.assertIsNone(metrics.format_metric({"event": "other"}))
        train = metrics.format_metric({"type": "train_update", "stage": "J", "epoch": 1,
            "update": 1, "losses": {"loss": 2.0, "gate_means_l": [.1]*5,
                                      "gate_means_u": [.3]*5}})
        self.assertIn("loss=2", train)
        self.assertIn("G=", train)

    def test_live_metric_handles_dropped_pairs_and_nested_fusion_gates(self):
        metrics = load_script("fuse_watch_metrics.py")
        row = {"type": "train_update", "stage": "T", "epoch": 0, "update": 1,
               "loss": 1.25,
               "losses": {"gate_means_l": [0.2, None, None, 0.4, 0.5],
                          "gate_means_u": [0.4, 0.6, None, None, 0.7]}}
        shown = metrics.format_metric(row)
        self.assertIn("epoch=1", shown)
        self.assertIn("loss=1.25", shown)
        self.assertIn("G= 30.00/ 60.00/    NA/ 40.00/ 60.00", shown)
        self.assertNotIn("teacher FG/RV/MYO/LV", shown)
        validation = {"type": "validation", "stage": "J", "epoch": 1, "update": 8,
                      "modes": {"fusion": {"gate_means": [0.1, 0.2, 0.3, 0.4, 0.5]}}}
        self.assertIn("G= 10.00/ 20.00/ 30.00/ 40.00/ 50.00",
                      metrics.format_metric(validation))

    def test_profile_frame_selection_contains_largest_l_and_u(self):
        profile = load_script("fuse_profile.py")

        class FakeData:
            role = "train"
            patients = tuple(f"p{i:02d}" for i in range(20))

            def labeled_keys(self):
                return [(patient, frame) for patient in self.patients for frame in (0, 1)]

            def unlabeled_keys(self):
                return [(patient, 2) for patient in self.patients]

            def load(self, key, *, labeled):
                patient, frame_index = key
                depth = int(patient[1:]) + 2
                geometry = {"shape": (depth, 8, 8)}
                target = object() if labeled else None
                return SimpleNamespace(geometry=geometry, target=target,
                                       patient_id=patient, frame_index=frame_index)

        labeled, pairs, largest_u = profile._select_profile_frames(FakeData(), count=20)
        self.assertEqual(len(labeled), 20)
        self.assertEqual(len(pairs), 20)
        self.assertEqual(max(labeled, key=lambda x: x[1][0])[0][0], "p19")
        self.assertIn(("p19", 2), [pair[1] for pair in pairs])
        self.assertEqual(largest_u, ("p19", 2))
        self.assertEqual(len({pair[1][0] for pair in pairs}), 20)


if __name__ == "__main__":
    unittest.main()
