import csv
import json

import pytest

from self_audit.candidate_pg.logging import EPOCH_COLUMNS, EpochLogger, select_best_epoch


def test_epoch_console_line_has_exact_requested_format():
    from self_audit.candidate_pg.logging import epoch_console_line
    assert epoch_console_line("A4", 37, 100, .412, .951, .836, .841, 31, .792, 41.2) == (
        "[A4] ep 37/100 | loss 0.412 | train FG 0.951 | val FG 0.836 (best 0.841@31) | test FG 0.792 | 41s"
    )


def _row(epoch, split, path, *, score=None, loss=None):
    return {
        "stage": "J", "arm": "A4", "epoch": epoch, "update": epoch * 80,
        "split": split, "path": path, "loss_total": loss,
        "fg_dice": score,
    }


def test_epoch_logger_fixed_rows_and_val_only_selection(tmp_path):
    logger = EpochLogger(tmp_path)
    assert logger.csv_path.name == "epochs.csv"
    assert logger.jsonl_path.name == "epochs.jsonl"
    logger.log_epoch(_row(1, "train", "student_only", loss=1.0))
    logger.log_epoch(_row(1, "VAL18", "student_only", score=0.4))
    logger.log_epoch(_row(1, "TEST50", "student_only", score=0.99))
    logger.log_epoch(_row(2, "VAL18", "student_only", score=0.6))
    selected = logger.select_best()
    assert selected["epoch"] == 2
    assert logger.best_val["epoch"] == 2
    with logger.csv_path.open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    json_rows = [json.loads(line) for line in logger.jsonl_path.read_text().splitlines()]
    assert tuple(csv_rows[0]) == EPOCH_COLUMNS
    assert len(csv_rows) == len(json_rows) == 4
    assert sum(row["best_val"] == "True" for row in csv_rows) == 1
    with pytest.raises(ValueError, match="training loss"):
        logger.log_epoch(_row(3, "VAL18", "student_only", loss=2.0, score=0.2))


def test_epoch_logger_rejects_duplicate_and_non_val_mark(tmp_path):
    logger = EpochLogger(tmp_path)
    logger.log_epoch(_row(1, "VAL18", "student_only", score=0.2))
    with pytest.raises(ValueError, match="duplicate"):
        logger.log_epoch(_row(1, "VAL18", "student_only", score=0.3))
    with pytest.raises(ValueError, match="VAL"):
        logger.mark_best(epoch=1, split="TEST50", path="student_only")


def test_logger_crash_after_log_before_checkpoint_archives_suffix(tmp_path):
    logger = EpochLogger(tmp_path)
    logger.log_epoch(_row(1, "VAL18", "student_only", score=0.4))
    logger.commit_epoch(epoch=1, update=80)
    logger.log_epoch(_row(2, "VAL18", "student_only", score=0.9))

    recovered = EpochLogger(tmp_path)
    assert recovered.committed_epoch == 1
    assert [row["epoch"] for row in recovered.rows] == [1]
    archives = list(tmp_path.glob("epochs.uncommitted.*.jsonl"))
    assert len(archives) == 1
    assert '"epoch":2' in archives[0].read_text()


def test_select_best_epoch_is_central_val_only_api():
    rows = [_row(1, "VAL18", "student_only", score=0.6), _row(2, "VAL18", "student_only", score=0.6), _row(3, "TEST50", "student_only", score=1.0)]
    assert select_best_epoch(rows)["epoch"] == 1
