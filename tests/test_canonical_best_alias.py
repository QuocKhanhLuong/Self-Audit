"""Tests for the committed-best-alias guard at the evaluation bind entrypoints.

The Wave 4 commit protocol writes the selected snapshot under ``selected_best/``,
records a hash-verified reference to it in ``last.pt``, and only then
republishes the public ``best.pt`` alias. A crash between the last two steps
leaves ``last.pt`` correct and ``best.pt`` stale.

Evaluation is frozen, so the guard refuses a stale alias and names the recovery
(resume from ``last.pt``, which resolves the committed reference and
republishes) rather than repairing anything itself.

Reference environment: Python 3.10.21 / torch 2.4.1 (CPU), single-threaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.self_audit.provenance import file_sha256
from src.self_audit.training._utils import (
    COMMITTED_BEST_REFERENCE_KEY,
    bind_evaluation_checkpoint,
    bind_existing_evaluation_state,
    save_checkpoint,
)
from src.self_audit.training.checkpoint_commit import (
    SELECTED_BEST_DIRNAME,
    make_best_reference,
    publish_best_alias,
)


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Linear(4, 2)


def _snapshot(run: Path, model: nn.Module, *, epoch: int, token: str) -> Path:
    store = run / SELECTED_BEST_DIRNAME
    store.mkdir(parents=True, exist_ok=True)
    return save_checkpoint(
        store / f"epoch_{epoch}_{token}.pt", model, epoch=epoch, global_step=epoch
    )


def _write_last(run: Path, model: nn.Module, reference: Any, *, include_key: bool = True) -> Path:
    extra = {COMMITTED_BEST_REFERENCE_KEY: reference} if include_key else {}
    return save_checkpoint(run / "last.pt", model, epoch=1, global_step=1, extra=extra)


def _committed_run(tmp_path: Path) -> tuple[Path, nn.Module, dict[str, Any]]:
    """A run whose public alias matches the reference committed in ``last.pt``."""

    run = tmp_path / "run"
    run.mkdir()
    model = _model(1)
    snapshot = _snapshot(run, model, epoch=1, token="c1")
    reference = dict(make_best_reference(snapshot, run, epoch=1, metric=0.72))
    publish_best_alias(reference, run)
    _write_last(run, model, reference)
    return run, model, reference


def _tree_digests(run: Path) -> dict[str, str]:
    return {
        str(path.relative_to(run)): file_sha256(path)
        for path in sorted(run.rglob("*"))
        if path.is_file()
    }


def _candidates(run: Path) -> list[tuple[str, Path]]:
    return [("best", run / "best.pt")]


# ---------------------------------------------------------------------------
# 1. A committed alias binds normally through both entrypoints
# ---------------------------------------------------------------------------

def test_committed_alias_binds_and_loads_the_actual_model(tmp_path: Path) -> None:
    run, trained, reference = _committed_run(tmp_path)

    target = _model(9)
    assert not torch.equal(target.weight, trained.weight), "fixture must start from other weights"

    binding = bind_evaluation_checkpoint(target, _candidates(run))

    assert binding.role == "best"
    assert binding.path == run / "best.pt"
    assert binding.restored == ("model",)
    assert binding.epoch == 1
    assert torch.equal(target.weight, trained.weight)
    assert torch.equal(target.bias, trained.bias)
    assert reference["sha256"] == file_sha256(run / "best.pt")


def test_committed_alias_binds_existing_live_state(tmp_path: Path) -> None:
    run, trained, _reference = _committed_run(tmp_path)

    binding = bind_existing_evaluation_state(trained, _candidates(run))

    assert binding.role == "best"
    assert binding.restored == ()
    assert binding.state_digest == binding.file_state_digest


# ---------------------------------------------------------------------------
# 2. A stale alias is refused at both entrypoints
# ---------------------------------------------------------------------------

def _stale_alias_run(tmp_path: Path) -> tuple[Path, nn.Module, dict[str, Any]]:
    """Interrupted alias publication: last.pt commits epoch 2, best.pt is epoch 1."""

    run = tmp_path / "run"
    run.mkdir()
    old_model = _model(1)
    old_snapshot = _snapshot(run, old_model, epoch=1, token="s1")
    old_reference = dict(make_best_reference(old_snapshot, run, epoch=1, metric=0.40))
    publish_best_alias(old_reference, run)

    new_model = _model(2)
    new_snapshot = _snapshot(run, new_model, epoch=2, token="s2")
    new_reference = dict(make_best_reference(new_snapshot, run, epoch=2, metric=0.61))
    # last.pt commits the new selection; the alias republish never landed.
    _write_last(run, new_model, new_reference)
    assert file_sha256(run / "best.pt") == old_reference["sha256"]
    return run, old_model, new_reference


@pytest.mark.parametrize("entrypoint", ["bind_evaluation_checkpoint", "bind_existing_evaluation_state"])
def test_stale_alias_is_refused_and_names_the_recovery(tmp_path: Path, entrypoint: str) -> None:
    run, alias_model, new_reference = _stale_alias_run(tmp_path)
    before = _tree_digests(run)

    if entrypoint == "bind_evaluation_checkpoint":
        call = lambda: bind_evaluation_checkpoint(_model(9), _candidates(run))
    else:
        call = lambda: bind_existing_evaluation_state(alias_model, _candidates(run))

    with pytest.raises(ValueError) as excinfo:
        call()

    message = str(excinfo.value)
    assert "alias is stale" in message
    assert new_reference["sha256"] in message
    assert "resume from last.pt" in message
    assert "will not repair" in message
    # Evaluation stays frozen: nothing on disk moved, least of all the alias.
    assert _tree_digests(run) == before


def test_refusal_reads_the_sibling_on_cpu_with_weights_only(tmp_path: Path) -> None:
    """The guard's own read is a safe CPU load with no pickle fallback."""

    run, _alias_model, _reference = _stale_alias_run(tmp_path)
    observed: list[dict[str, Any]] = []
    real_load = torch.load

    def _spy(path: Any, **kwargs: Any) -> Any:
        observed.append({"path": str(path), **kwargs})
        return real_load(path, **kwargs)

    with patch.object(torch, "load", side_effect=_spy):
        with pytest.raises(ValueError, match="alias is stale"):
            bind_evaluation_checkpoint(_model(9), _candidates(run))

    last_reads = [call for call in observed if call["path"].endswith("last.pt")]
    assert last_reads, "the guard must consult the sibling last.pt"
    for call in last_reads:
        assert call["map_location"] == "cpu"
        assert call["weights_only"] is True


# ---------------------------------------------------------------------------
# 3. Unverifiable committed references are refused
# ---------------------------------------------------------------------------

def test_missing_snapshot_is_refused(tmp_path: Path) -> None:
    run, _model_, reference = _committed_run(tmp_path)
    (run / reference["path"]).unlink()
    before = _tree_digests(run)

    with pytest.raises(ValueError, match="does not verify"):
        bind_evaluation_checkpoint(_model(9), _candidates(run))
    assert _tree_digests(run) == before


def test_corrupt_snapshot_is_refused(tmp_path: Path) -> None:
    run, _model_, reference = _committed_run(tmp_path)
    (run / reference["path"]).write_bytes(b"not-the-committed-bytes")
    before = _tree_digests(run)

    with pytest.raises(ValueError, match="does not verify"):
        bind_evaluation_checkpoint(_model(9), _candidates(run))
    assert _tree_digests(run) == before


def test_malformed_reference_is_refused(tmp_path: Path) -> None:
    run, model, reference = _committed_run(tmp_path)
    _write_last(run, model, dict(reference, path=f"{SELECTED_BEST_DIRNAME}/../best.pt"))

    with pytest.raises(ValueError, match="does not verify"):
        bind_evaluation_checkpoint(_model(9), _candidates(run))


def test_explicit_absent_selection_refuses_an_unexplained_best(tmp_path: Path) -> None:
    """``best_reference: None`` records that no selection was committed."""

    run, model, _reference = _committed_run(tmp_path)
    _write_last(run, model, None)

    with pytest.raises(ValueError, match="explicitly records no committed best selection"):
        bind_evaluation_checkpoint(_model(9), _candidates(run))


# ---------------------------------------------------------------------------
# 4. Backward compatibility
# ---------------------------------------------------------------------------

def test_standalone_best_without_a_sibling_last_still_binds(tmp_path: Path) -> None:
    run = tmp_path / "copied"
    run.mkdir()
    trained = _model(3)
    save_checkpoint(run / "best.pt", trained, epoch=4, global_step=4)
    assert not (run / "last.pt").exists()

    target = _model(9)
    binding = bind_evaluation_checkpoint(target, _candidates(run))

    assert binding.epoch == 4
    assert torch.equal(target.weight, trained.weight)


def test_historical_last_without_the_reference_key_still_binds(tmp_path: Path) -> None:
    run = tmp_path / "legacy"
    run.mkdir()
    trained = _model(4)
    save_checkpoint(run / "best.pt", trained, epoch=5, global_step=5)
    last = _write_last(run, trained, None, include_key=False)
    payload = torch.load(last, map_location="cpu", weights_only=True)
    assert COMMITTED_BEST_REFERENCE_KEY not in payload

    target = _model(9)
    binding = bind_evaluation_checkpoint(target, _candidates(run))

    assert binding.epoch == 5
    assert torch.equal(target.weight, trained.weight)


def test_unreadable_sibling_last_fails_closed(tmp_path: Path) -> None:
    """An unsafe-to-load sibling must raise, never be read as "no reference".

    Unreadable bytes are not evidence that ``best_reference`` is absent: a
    corrupted ``last.pt`` could equally commit a selection the alias
    contradicts. Treating it as historical would let the stale alias through.
    """

    run, _model_, _reference = _committed_run(tmp_path)
    (run / "last.pt").write_bytes(b"not a torch archive at all")
    before = _tree_digests(run)

    with pytest.raises(ValueError) as excinfo:
        bind_evaluation_checkpoint(_model(9), _candidates(run))
    message = str(excinfo.value)
    assert "could not be safely read" in message
    assert "not treated as one without a committed selection" in message

    with pytest.raises(ValueError, match="could not be safely read"):
        bind_existing_evaluation_state(_model(9), _candidates(run))

    assert _tree_digests(run) == before


def test_non_canonical_basename_is_outside_the_protocol(tmp_path: Path) -> None:
    """The historical ``phase_c_best.pt`` naming is untouched by the guard."""

    run = tmp_path / "phases"
    run.mkdir()
    trained = _model(5)
    save_checkpoint(run / "phase_c_best.pt", trained, epoch=6, global_step=6)
    # A last.pt that would refuse a canonical best.pt is irrelevant here.
    _write_last(run, trained, None)

    target = _model(9)
    binding = bind_evaluation_checkpoint(target, [("best", run / "phase_c_best.pt")])

    assert binding.epoch == 6
    assert torch.equal(target.weight, trained.weight)
