from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.self_audit.training._utils import (
    WandbLogger,
    add_wandb_and_tqdm_args,
    setup_wandb_logger,
)
from src.self_audit.training.train_annotation import (
    parse_args as parse_args_a,
    train_annotation_epoch,
    validate_annotation_epoch,
)
from src.self_audit.training.train_auditor import (
    parse_args as parse_args_b,
)
from src.self_audit.training.finetune_joint import (
    parse_args as parse_args_c,
)


def test_wandb_logger_disabled_is_noop() -> None:
    logger = WandbLogger(enabled=False)
    assert not logger.enabled
    logger.log({"loss": 0.5}, step=1)
    logger.finish()


def test_wandb_logger_handles_logging_with_mock() -> None:
    with patch.dict("sys.modules", {"wandb": MagicMock()}):
        import wandb
        mock_run = MagicMock()
        wandb.init.return_value = mock_run

        logger = WandbLogger(enabled=True, project="test-proj", mode="offline")
        assert logger.enabled
        logger.log({"train/loss": torch.tensor(0.25), "val/dice": 0.85}, step=2)
        wandb.log.assert_called_once()
        logger.finish()
        wandb.finish.assert_called_once()


def test_add_wandb_and_tqdm_args_integration() -> None:
    parser = argparse.ArgumentParser()
    add_wandb_and_tqdm_args(parser)
    args = parser.parse_args(["--wandb", "--wandb_project", "custom-proj", "--no_tqdm"])
    assert args.wandb is True
    assert args.wandb_project == "custom-proj"
    assert args.no_tqdm is True


def test_setup_wandb_logger_cli_overrides_config() -> None:
    parser = argparse.ArgumentParser()
    add_wandb_and_tqdm_args(parser)
    args = parser.parse_args(["--no_wandb", "--wandb_project", "cli-proj"])
    config = {"wandb": {"enabled": True, "project": "config-proj"}}
    logger = setup_wandb_logger(args, config, phase="annotation")
    assert not logger.enabled
    assert logger.project == "cli-proj"


def test_train_annotation_epoch_runs_with_tqdm_disabled_and_enabled() -> None:
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=1)

        def forward_annotation(self, x):
            out = self.conv(x)
            return {"initial_logits": out, "logits": out}

    dataset = TensorDataset(torch.randn(2, 3, 16, 16), torch.zeros(2, 16, 16, dtype=torch.long))

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        masks = torch.stack([item[1] for item in batch])
        return {"image": images, "mask": masks}

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    # Disabled TQDM
    stats = train_annotation_epoch(
        model, loader, optimizer, device, epoch=0, total_epochs=1, disable_tqdm=True
    )
    assert "loss" in stats
    assert "lr" in stats

    # Enabled TQDM
    val_stats = validate_annotation_epoch(
        model, loader, device, epoch=0, total_epochs=1, disable_tqdm=False
    )
    assert "val_loss" in val_stats
    assert "val_macro_foreground_dice" in val_stats


def test_parsers_contain_wandb_and_tqdm_flags() -> None:
    for parse_fn in (parse_args_a, parse_args_b, parse_args_c):
        with patch("sys.argv", ["script_name", "--help"]):
            with pytest.raises(SystemExit):
                parse_fn()
