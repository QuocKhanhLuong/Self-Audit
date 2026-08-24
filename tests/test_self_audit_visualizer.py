"""Unit tests for Self-Audit visualizer module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.self_audit.evaluation.visualizer import (
    log_figures_to_wandb,
    plot_patient_volume_qa,
    plot_phase_a_samples,
    plot_phase_b_transitions,
    plot_phase_c_audit_trace,
    render_difference_map,
    render_overlay,
    render_transition_overlay,
    save_figure,
)


def test_render_overlay_shapes_and_values() -> None:
    img = np.random.randn(32, 32).astype(np.float32)
    mask = np.random.randint(0, 4, size=(32, 32), dtype=np.int64)
    overlay = render_overlay(img, mask, num_classes=4)
    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.float32
    assert 0.0 <= overlay.min() and overlay.max() <= 1.0


def test_render_transition_overlay_shapes() -> None:
    img = np.random.randn(32, 32).astype(np.float32)
    trans = np.random.randint(0, 3, size=(32, 32), dtype=np.int64)
    overlay = render_transition_overlay(img, trans)
    assert overlay.shape == (32, 32, 3)
    assert 0.0 <= overlay.min() and overlay.max() <= 1.0


def test_render_difference_map_shapes() -> None:
    img = np.random.randn(32, 32).astype(np.float32)
    gt = np.random.randint(0, 2, size=(32, 32), dtype=np.int64)
    pred = np.random.randint(0, 2, size=(32, 32), dtype=np.int64)
    diff = render_difference_map(img, gt, pred)
    assert diff.shape == (32, 32, 3)
    assert 0.0 <= diff.min() and diff.max() <= 1.0


def test_plot_phase_a_samples_and_save(tmp_path: Path) -> None:
    images = torch.randn(3, 3, 32, 32)
    masks = torch.randint(0, 4, size=(3, 32, 32))
    initial_preds = torch.randint(0, 4, size=(3, 32, 32))
    final_preds = torch.randint(0, 4, size=(3, 32, 32))

    fig = plot_phase_a_samples(images, masks, initial_preds, final_preds, max_samples=2)
    out_file = tmp_path / "vis" / "phase_a.png"
    saved = save_figure(fig, out_file)
    assert Path(saved).is_file()
    assert Path(saved).stat().st_size > 0


def test_plot_phase_b_transitions_and_save(tmp_path: Path) -> None:
    images = torch.randn(2, 3, 32, 32)
    masks = torch.randint(0, 4, size=(2, 32, 32))
    prev_states = torch.randint(0, 4, size=(2, 32, 32))
    cand_states = torch.randint(0, 4, size=(2, 32, 32))
    local_preds = torch.randint(0, 3, size=(2, 32, 32))
    local_tgts = torch.randint(0, 3, size=(2, 32, 32))
    delta_qs = [0.05, -0.02]
    delta_dices = [0.04, -0.03]

    fig = plot_phase_b_transitions(
        images, masks, prev_states, cand_states, local_preds, local_tgts,
        delta_qs=delta_qs, delta_dices=delta_dices, max_samples=2,
    )
    out_file = tmp_path / "phase_b.png"
    saved = save_figure(fig, out_file)
    assert Path(saved).is_file()
    assert Path(saved).stat().st_size > 0


def test_plot_phase_c_audit_trace_and_save(tmp_path: Path) -> None:
    images = torch.randn(2, 3, 32, 32)
    masks = torch.randint(0, 4, size=(2, 32, 32))
    initial_logits = torch.randn(2, 4, 32, 32)
    final_logits = torch.randn(2, 4, 32, 32)
    candidates = [torch.randn(2, 4, 32, 32)]
    halted = [1, 2]

    fig = plot_phase_c_audit_trace(
        images, masks, initial_logits, final_logits,
        transition_candidates=candidates,
        halted_turns=halted,
        max_samples=2,
    )
    out_file = tmp_path / "phase_c.png"
    saved = save_figure(fig, out_file)
    assert Path(saved).is_file()
    assert Path(saved).stat().st_size > 0


def test_plot_patient_volume_qa_and_save(tmp_path: Path) -> None:
    vol_image = np.random.randn(8, 32, 32).astype(np.float32)
    gt_vol = np.random.randint(0, 4, size=(8, 32, 32), dtype=np.int64)
    pred_vol = np.random.randint(0, 4, size=(8, 32, 32), dtype=np.int64)
    init_vol = np.random.randint(0, 4, size=(8, 32, 32), dtype=np.int64)

    fig = plot_patient_volume_qa(
        vol_image, gt_vol, pred_vol,
        initial_pred_volume=init_vol,
        slice_indices=[1, 3, 5],
        patient_id="patient001_test",
    )
    out_file = tmp_path / "patient_qa.png"
    saved = save_figure(fig, out_file)
    assert Path(saved).is_file()
    assert Path(saved).stat().st_size > 0


def test_log_figures_to_wandb_safe_when_no_wandb(tmp_path: Path) -> None:
    # Should execute without throwing any error even when wandb is disabled
    dummy_img = tmp_path / "dummy.png"
    dummy_img.write_text("fake")
    log_figures_to_wandb({"test_img": dummy_img}, step=1)
