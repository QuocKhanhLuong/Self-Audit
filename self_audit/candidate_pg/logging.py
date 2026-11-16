"""Fixed-schema epoch and patient logging for Candidate PG."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping


EPOCH_COLUMNS = (
    "stage", "arm", "epoch", "update", "split", "path",
    "loss_total", "loss_sup", "loss_pg", "loss_cbkd",
    "fg_dice", "rv_dice", "myo_dice", "lv_dice",
    "fg_hd95", "rv_hd95", "myo_hd95", "lv_hd95",
    "fg_assd", "rv_assd", "myo_assd", "lv_assd",
    "lr", "seconds", "best_val",
)
_PREFIX = EPOCH_COLUMNS[:-1]
_TRAIN_SPLITS = {"train", "TRAIN", "train_labeled", "TRAIN16"}
_VAL_SPLITS = {"val", "VAL", "VAL18", "development", "dev"}
_TEST_SPLITS = {"test", "TEST", "TEST50"}


def _empty_row() -> dict[str, Any]:
    return {column: None for column in EPOCH_COLUMNS}


def _as_row(value: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if value is not None and not isinstance(value, Mapping):
        raise TypeError("epoch log row must be a mapping")
    merged = dict(value or {})
    merged.update(kwargs)
    unknown = set(merged) - set(EPOCH_COLUMNS)
    if unknown:
        raise ValueError(f"unknown epoch log columns: {sorted(unknown)}")
    row = _empty_row()
    row.update(merged)
    if row["stage"] is None or row["arm"] is None or row["epoch"] is None or row["update"] is None or row["split"] is None or row["path"] is None:
        raise ValueError("epoch rows require stage, arm, epoch, update, split and path")
    if type(row["epoch"]) is not int or row["epoch"] < 0 or type(row["update"]) is not int or row["update"] < 0:
        raise ValueError("epoch and update must be non-negative integers")
    if row["best_val"] is None:
        row["best_val"] = False
    if not isinstance(row["best_val"], bool):
        raise ValueError("best_val must be a boolean marker")
    split = str(row["split"])
    loss_columns = ("loss_total", "loss_sup", "loss_pg", "loss_cbkd")
    if split not in _TRAIN_SPLITS and any(row[name] not in (None, "") for name in loss_columns):
        raise ValueError("training loss columns are permitted only on train rows")
    return row


class EpochLogger:
    """Append-only fixed epoch schema with atomic rewrites for best markers.

    The logger keeps a small row index in memory and rewrites the two receipt
    files after each append/selection.  This makes CSV and JSONL byte contents
    derive from the same ordered rows and prevents duplicate
    ``(epoch, split, path)`` entries after a resume.
    """

    columns = EPOCH_COLUMNS

    def __init__(self, output_dir: str | Path, *, csv_name: str = "epochs.csv", jsonl_name: str = "epochs.jsonl", stage: str | None = None, arm: str | None = None):
        requested = Path(output_dir)
        if requested.suffix.lower() in {".csv", ".jsonl"}:
            self.output_dir = requested.parent
            if requested.suffix.lower() == ".csv":
                csv_name = requested.name
            else:
                jsonl_name = requested.name
        else:
            self.output_dir = requested
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / csv_name
        self.jsonl_path = self.output_dir / jsonl_name
        self.default_stage = stage
        self.default_arm = arm
        self.rows: list[dict[str, Any]] = []
        self._keys: set[tuple[int, str, str]] = set()
        self.commit_path = self.output_dir / "epochs.commit.json"
        self._committed_epoch: int | None = None
        self._committed_update: int | None = None
        self._recover_committed_suffix()
        self._load_existing()

    def _recover_committed_suffix(self) -> None:
        """Recover a crash after logging but before checkpoint commit.

        A checkpoint records the committed row count and byte offsets.  On a
        fresh logger construction, rows after that boundary are archived and
        the two receipts are rewritten from the committed prefix.  This is
        intentionally conservative: if a receipt is malformed, recovery
        fails closed rather than guessing a prefix.
        """

        if not self.commit_path.exists() or not (self.csv_path.exists() and self.jsonl_path.exists()):
            return
        try:
            state = json.loads(self.commit_path.read_text(encoding="utf-8"))
            committed_count = int(state["row_count"])
            committed_epoch = int(state["epoch"])
            committed_update = int(state["update"])
            csv_lines = self.csv_path.read_text(encoding="utf-8").splitlines(keepends=True)
            jsonl_lines = self.jsonl_path.read_text(encoding="utf-8").splitlines(keepends=True)
            if not csv_lines or not jsonl_lines or committed_count < 0:
                raise ValueError("invalid committed epoch receipt")
            # CSV has one header; JSONL has one line per row.
            csv_row_count = max(0, len(csv_lines) - 1)
            jsonl_row_count = len(jsonl_lines)
            if csv_row_count != jsonl_row_count or committed_count > jsonl_row_count:
                raise ValueError("committed epoch receipt exceeds logged rows")
            if jsonl_row_count == committed_count:
                self._committed_epoch, self._committed_update = committed_epoch, committed_update
                return
            stamp = f"{time.time_ns()}"
            csv_archive = self.output_dir / f"epochs.uncommitted.{stamp}.csv"
            jsonl_archive = self.output_dir / f"epochs.uncommitted.{stamp}.jsonl"
            csv_archive.write_text("".join(csv_lines[1 + committed_count:]), encoding="utf-8")
            jsonl_archive.write_text("".join(jsonl_lines[committed_count:]), encoding="utf-8")
            self.csv_path.write_text("".join(csv_lines[: 1 + committed_count]), encoding="utf-8")
            self.jsonl_path.write_text("".join(jsonl_lines[:committed_count]), encoding="utf-8")
            self._committed_epoch, self._committed_update = committed_epoch, committed_update
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid PG committed epoch recovery receipt") from exc

    def _load_existing(self) -> None:
        if not self.jsonl_path.exists():
            return
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = _as_row(json.loads(line), {})
            key = (row["epoch"], str(row["split"]), str(row["path"]))
            if key in self._keys:
                raise ValueError("duplicate epoch/split/path in existing PG JSONL")
            self._keys.add(key); self.rows.append(row)
        if self.csv_path.exists():
            with self.csv_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != EPOCH_COLUMNS:
                    raise ValueError("existing PG CSV schema does not match fixed epoch columns")
                csv_keys = {(int(item["epoch"]), item["split"], item["path"]) for item in reader}
            if csv_keys != self._keys:
                raise ValueError("PG CSV/JSONL epoch rows differ")

    @property
    def path(self) -> Path:
        return self.csv_path

    def _write(self) -> None:
        temp_csv = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        temp_jsonl = self.jsonl_path.with_suffix(self.jsonl_path.suffix + ".tmp")
        with temp_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=EPOCH_COLUMNS, lineterminator="\n")
            writer.writeheader(); writer.writerows(self.rows)
        with temp_jsonl.open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, sort_keys=False, allow_nan=False, separators=(",", ":")) + "\n")
        temp_csv.replace(self.csv_path); temp_jsonl.replace(self.jsonl_path)

    @property
    def committed_epoch(self) -> int | None:
        return self._committed_epoch

    @property
    def committed_update(self) -> int | None:
        return self._committed_update

    def commit_epoch(self, *, epoch: int, update: int) -> dict[str, Any]:
        """Persist the committed epoch boundary used by checkpoint resume."""

        if type(epoch) is not int or type(update) is not int or epoch < 0 or update < 0:
            raise ValueError("committed epoch/update must be non-negative integers")
        if not self.rows or update < max(int(row["update"]) for row in self.rows):
            raise ValueError("cannot commit before all logged rows at this update")
        state = {
            "schema": "self_audit.candidate_pg.epoch_commit.v1",
            "epoch": epoch,
            "update": update,
            "row_count": len(self.rows),
            "csv_offset": self.csv_path.stat().st_size if self.csv_path.exists() else 0,
            "jsonl_offset": self.jsonl_path.stat().st_size if self.jsonl_path.exists() else 0,
        }
        state["metrics_offset"] = state["jsonl_offset"]
        temp = self.commit_path.with_suffix(self.commit_path.suffix + ".tmp")
        temp.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        temp.replace(self.commit_path)
        self._committed_epoch, self._committed_update = epoch, update
        return dict(state)

    checkpoint_committed = commit_epoch

    def recover_from_checkpoint(self, checkpoint: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Apply an external checkpoint's committed epoch/offset boundary.

        Checkpoints from the engine may carry ``row_count`` directly or only a
        JSONL ``metrics_offset``/``offset``.  The latter is converted to a
        newline count without guessing incomplete rows.
        """

        if checkpoint is not None:
            source = checkpoint.get("metrics_receipt", checkpoint) if isinstance(checkpoint, Mapping) else {}
            offset = source.get("metrics_offset", source.get("jsonl_offset", source.get("offset")))
            row_count = source.get("row_count")
            if row_count is None and offset is not None and self.jsonl_path.exists():
                raw = self.jsonl_path.read_bytes()[: int(offset)]
                row_count = len(raw.splitlines())
            if row_count is None:
                raise ValueError("checkpoint lacks committed epoch row_count/offset")
            state = {
                "schema": "self_audit.candidate_pg.epoch_commit.v1",
                "epoch": int(checkpoint.get("epoch", source.get("epoch", 0))),
                "update": int(checkpoint.get("update", source.get("update", 0))),
                "row_count": int(row_count),
                "csv_offset": int(source.get("csv_offset", self.csv_path.stat().st_size if self.csv_path.exists() else 0)),
                "jsonl_offset": int(offset if offset is not None else self.jsonl_path.stat().st_size if self.jsonl_path.exists() else 0),
            }
            self.commit_path.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            self.rows.clear(); self._keys.clear()
            self._recover_committed_suffix(); self._load_existing()
            return state
        self.rows.clear(); self._keys.clear()
        self._recover_committed_suffix(); self._load_existing()
        return {"epoch": self._committed_epoch, "update": self._committed_update, "row_count": len(self.rows)}

    recover = recover_from_checkpoint
    restore_committed = recover_from_checkpoint

    def log_epoch(self, row: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        if self.default_stage is not None and (row is None or "stage" not in row) and "stage" not in kwargs:
            kwargs["stage"] = self.default_stage
        if self.default_arm is not None and (row is None or "arm" not in row) and "arm" not in kwargs:
            kwargs["arm"] = self.default_arm
        normalized = _as_row(row, kwargs)
        key = (normalized["epoch"], str(normalized["split"]), str(normalized["path"]))
        if key in self._keys:
            raise ValueError(f"duplicate PG epoch row: {key}")
        self._keys.add(key); self.rows.append(normalized); self._write()
        return dict(normalized)

    append = log_epoch
    log = log_epoch
    write_epoch = log_epoch
    record_epoch = log_epoch

    def log_patient_rows(self, epoch: int, rows: Iterable[Mapping[str, Any]]) -> Path:
        """Write patient-macro rows to ``patients_epochXXX.csv``.

        Each row must identify split and path; duplicate patient records for the
        same split/path/epoch are rejected.  The metrics are intentionally
        caller-defined so existing per-class absent-class conventions survive.
        """

        values = [dict(item) for item in rows]
        if type(epoch) is not int or epoch < 0:
            raise ValueError("patient epoch must be a non-negative integer")
        if not values:
            raise ValueError("patient rows cannot be empty")
        seen: set[tuple[str, str, str]] = set()
        for row in values:
            for key in ("patient_id", "split", "path"):
                if key not in row:
                    raise ValueError(f"patient row requires {key}")
            identity = (str(row["patient_id"]), str(row["split"]), str(row["path"]))
            if identity in seen:
                raise ValueError(f"duplicate patient row: {identity}")
            seen.add(identity)
            row.setdefault("epoch", epoch)
            if row["epoch"] != epoch:
                raise ValueError("patient row epoch does not match filename epoch")
        fields = ["epoch", "split", "path", "patient_id"]
        extras = sorted({key for row in values for key in row if key not in fields})
        fields.extend(extras)
        path = self.output_dir / f"patients_epoch{epoch:03d}.csv"
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader(); writer.writerows(values)
        temp.replace(path)
        return path

    log_patients = log_patient_rows

    def mark_best(self, *, epoch: int, split: str = "VAL18", path: str = "student_only") -> dict[str, Any]:
        """Mark one VAL row and clear all other markers.

        Selection is deliberately VAL-only; asking to mark TEST or TRAIN is a
        contract error even if those rows happen to contain a high metric.
        """

        if split not in _VAL_SPLITS:
            raise ValueError("PG best_val can only mark a VAL row")
        matches = [row for row in self.rows if row["epoch"] == epoch and str(row["split"]) == split and str(row["path"]) == path]
        if len(matches) != 1:
            raise ValueError("best_val target must identify exactly one VAL epoch row")
        for row in self.rows:
            row["best_val"] = row is matches[0]
        self._write()
        return dict(matches[0])

    def select_best(self, *, split: str = "VAL18", path: str = "student_only", metric: str = "fg_dice") -> dict[str, Any]:
        """Select highest VAL metric, earliest epoch on ties.

        TEST rows are never considered, so changing a TEST metric cannot change
        the selected epoch.
        """

        if split not in _VAL_SPLITS:
            raise ValueError("PG selection split must be VAL18")
        best = select_best_epoch(self.rows, split=split, path=path, metric=metric)
        return self.mark_best(epoch=int(best["epoch"]), split=split, path=path)

    @property
    def best_val(self) -> dict[str, Any] | None:
        selected = [row for row in self.rows if row.get("best_val") is True]
        return dict(selected[0]) if selected else None


def select_best_epoch(rows: Iterable[Mapping[str, Any]], *, split: str = "VAL18", path: str = "student_only", metric: str = "fg_dice") -> dict[str, Any]:
    """Central VAL-only earliest-tie selector shared by engine/evaluation."""

    if split not in _VAL_SPLITS:
        raise ValueError("PG selection split must be VAL18")
    candidates = [dict(row) for row in rows if str(row.get("split")) == split and str(row.get("path")) == path and row.get(metric) not in (None, "")]
    if not candidates:
        raise ValueError("no VAL rows available for PG selection")
    try:
        best = min(candidates, key=lambda row: (-float(row[metric]), int(row["epoch"]), int(row["update"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"selection metric {metric} must be numeric") from exc
    return best


__all__ = ["EPOCH_COLUMNS", "EpochLogger", "select_best_epoch"]
