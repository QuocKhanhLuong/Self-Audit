from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_profile():
    path = ROOT / "scripts" / "fuse_profile.py"
    spec = importlib.util.spec_from_file_location("test_fuse_profile_semi", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


import unittest


class K8ProfileTests(unittest.TestCase):
    def test_k8_repeats_only_authorized_l_keys_and_includes_largest_native_u(self):
        profile = load_profile()

        class FakeData:
            role = "train"
            patients = tuple(f"p{i:02d}" for i in range(80))

            def labeled_keys(self):
                # K8: two permitted ED/ES frames per L patient (16 total).
                return [(patient, frame) for patient in self.patients[:8] for frame in (0, 1)]

            def unlabeled_keys(self):
                # Image-only U: non-L patients and non-ED/ES L frames.
                return [(patient, 2) for patient in self.patients]

            def load(self, key, *, labeled):
                patient, frame_index = key
                if labeled and key not in set(self.labeled_keys()):
                    raise AssertionError(f"masked non-L read attempted: {key}")
                if not labeled and frame_index in {0, 1} and patient in self.patients[:8]:
                    raise AssertionError(f"U included L ED/ES frame: {key}")
                depth = 100 + int(patient[1:]) if frame_index == 2 else 10 + int(patient[1:])
                geometry = {"shape": (depth, 8, 8)}
                return SimpleNamespace(geometry=geometry, target=object() if labeled else None)

        data = FakeData()
        labeled, pairs, largest_u = profile._select_profile_frames(data, count=20)
        authorized = set(data.labeled_keys())
        self.assertEqual(len(labeled), 20)
        self.assertEqual(len(set(key for key, _ in labeled)), 16)
        self.assertTrue(all(key in authorized for key, _ in labeled))
        self.assertEqual(len(pairs), 20)
        self.assertTrue(all(lkey in authorized for lkey, _ in pairs))
        self.assertEqual(largest_u, ("p79", 2))
        self.assertIn(largest_u, [ukey for _lkey, ukey in pairs])

    def test_profile_rejects_empty_l_inventory(self):
        profile = load_profile()

        class EmptyData:
            role = "train"
            patients = tuple(f"p{i:02d}" for i in range(20))
            def labeled_keys(self): return []
            def unlabeled_keys(self): return [(patient, 2) for patient in self.patients]
            def load(self, key, *, labeled): raise AssertionError("no data should be loaded")

        import pytest
        with pytest.raises(ValueError, match="authorized TRAIN L frame"):
            profile._select_profile_frames(EmptyData(), count=20)
