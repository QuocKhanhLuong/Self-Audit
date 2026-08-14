from __future__ import annotations

import numpy as np
import pytest
import torch

from src.self_audit.evaluation.metrics import annotation_metrics
from src.self_audit.evaluation.volume_inference import build_25d_batch, reconstruct_volume


def test_volume_reconstruction_from_2d_predictions() -> None:
    predictions = [torch.zeros(4, 5, dtype=torch.long), torch.ones(4, 5, dtype=torch.long)]
    volume = reconstruct_volume(predictions, num_slices=2)
    assert volume.shape == (2, 4, 5)
    assert volume[1].unique().tolist() == [1]


def test_volume_reconstruction_rejects_duplicate_or_missing_slice_indices() -> None:
    predictions = [torch.zeros(2, 2), torch.ones(2, 2)]
    with pytest.raises(ValueError, match="unique complete range"):
        reconstruct_volume(predictions, num_slices=2, slice_indices=[0, 0])


def test_volume_25d_batch_keeps_depth_and_boundary_contract() -> None:
    volume = np.arange(4 * 6 * 5, dtype=np.float32).reshape(4, 6, 5)
    batch = build_25d_batch(volume, image_size=8)
    assert batch.shape == (4, 3, 8, 8)
    assert torch.allclose(batch[0, 0], batch[0, 1])
    assert torch.allclose(batch[-1, -1], batch[-1, 1])


def test_annotation_metrics_report_required_fields() -> None:
    target = np.zeros((3, 8, 8), dtype=np.int64)
    target[:, 2:6, 2:6] = 1
    result = annotation_metrics(target, target, num_classes=4)
    assert {"dice", "hd95", "assd", "precision", "recall"}.issubset(result)
    assert result["dice"] == 1.0
