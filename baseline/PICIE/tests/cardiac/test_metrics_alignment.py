"""Metrics alignment tests for PICIE cardiac benchmark.

Verifies that PICIE's HD95 and ASSD implementations exactly match the Self-Audit spec:
- Synthetic 3D volume verification
- Edge cases handling: empty slices
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from medical_metrics import compute_medical_metrics


class MetricsAlignmentTests(unittest.TestCase):
    def test_synthetic_3d_metrics(self):
        """Verify exact match for 3D synthetic volumes."""
        pred = np.zeros((10, 10, 3), dtype=np.int32)
        gt = np.zeros((10, 10, 3), dtype=np.int32)

        # Add foreground object in slice z=1
        pred[4:6, 4:6, 1] = 1
        gt[4:6, 5:7, 1] = 1 # Shifted by 1 pixel in X

        spacing = (1.5, 1.0, 2.0) # y, x, z
        metrics = compute_medical_metrics(pred, gt, spacing, num_classes=2)

        # Dice: intersection=2, union=4+4=8 => 4/8 = 0.5
        self.assertAlmostEqual(metrics['dice'][1], 0.5)

        # Both objects have 4 pixels. Border of pred: (4,4), (4,5), (5,4), (5,5)
        # Border of gt: (4,5), (4,6), (5,5), (5,6)
        # Distances from pred border to gt:
        # (4,4)->(4,5) = 1.0 (X diff)
        # (4,5)->(4,5) = 0.0
        # (5,4)->(5,5) = 1.0
        # (5,5)->(5,5) = 0.0
        # Distances from gt border to pred:
        # (4,5)->(4,5) = 0.0
        # (4,6)->(4,5) = 1.0
        # (5,5)->(5,5) = 0.0
        # (5,6)->(5,5) = 1.0
        # All distances: [1, 0, 1, 0, 0, 1, 0, 1] => mean is 4/8 = 0.5
        self.assertAlmostEqual(metrics['assd'][1], 0.5)
        self.assertAlmostEqual(metrics['hd95'][1], 1.0)

    def test_empty_slices_edge_cases(self):
        """Verify edge cases with empty volumes."""
        # 1 slice empty in GT but not pred -> distance is inf
        pred = np.zeros((10, 10, 3), dtype=np.int32)
        gt = np.zeros((10, 10, 3), dtype=np.int32)

        pred[4:6, 4:6, 1] = 1
        # GT is entirely empty

        spacing = (1.0, 1.0, 1.0)
        metrics = compute_medical_metrics(pred, gt, spacing, num_classes=2)
        self.assertEqual(metrics['dice'][1], 0.0)
        self.assertTrue(np.isnan(metrics['hd95'][1]))
        self.assertTrue(np.isnan(metrics['assd'][1]))

if __name__ == "__main__":
    unittest.main()
