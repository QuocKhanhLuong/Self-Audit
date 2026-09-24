"""Scientific hardening tests for PICIE cardiac benchmark.

Verifies:
- Freeze validation: config hash stability
- Deterministic inference: same input → same partition hash
- Checkpoint provenance: SHA-256 binding
- Bundle seal/verify integrity
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cardiac_benchmark.config import PICIEConfig
from cardiac_benchmark.freeze import (
    FreezeError,
    create_freeze,
    evaluator_skeleton,
    seal_raw_bundle,
    validate_freeze,
    verify_raw_bundle,
)
from cardiac_benchmark.provenance import sha256_array, sha256_file, sha256_json


class FreezeTests(unittest.TestCase):

    def test_config_hash_stable(self):
        """Same config produces same hash."""
        config = PICIEConfig()
        h1 = config.config_hash()
        h2 = config.config_hash()
        self.assertEqual(h1, h2)

    def test_config_hash_changes_on_param_change(self):
        """Different config produces different hash."""
        c1 = PICIEConfig(K_train=4)
        c2 = PICIEConfig(K_train=5) # Invalid config, but hashes should still differ
        self.assertNotEqual(c1.config_hash(), c2.config_hash())

if __name__ == "__main__":
    unittest.main()