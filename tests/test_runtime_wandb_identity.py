"""Tests for the WandbLogger run-identity wrapper API.

Everything is mocked: no network call and no real W&B run is created anywhere
in this module. The wrapper's job is to keep *requested* identity separate from
what actually took effect, so these tests mostly assert that distinction --
especially that an offline attempt never claims a backend resume it cannot
perform.

Delaying logger construction until a validated resume identity is known, and
preserving RNG around telemetry, are the trainer's responsibilities; this file
only covers the wrapper it calls.

Reference environment: Python 3.10.21 / torch 2.4.1 (CPU), single-threaded.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.self_audit.training._utils import WandbLogger

_UNSET = object()


def _mock_sdk(
    run_id: Any = "server-run-1", resumed: Any = _UNSET
) -> tuple[MagicMock, MagicMock]:
    """A stand-in wandb module whose init returns a run with a chosen identity.

    ``resumed`` left unset means the run object reports no usable ``resumed``
    flag, which is how a partial or older SDK behaves: unknown, never a resume.
    """

    module = MagicMock()
    run = MagicMock()
    run.id = run_id
    run.name = "mock-run"
    run.summary = {}
    if resumed is not _UNSET:
        run.resumed = resumed
    module.init.return_value = run
    return module, run


def _init_kwargs(module: MagicMock) -> dict[str, Any]:
    assert module.init.call_count == 1
    return module.init.call_args.kwargs


SUMMARY_KEYS = {
    "schema_version",
    "enabled_requested",
    "enabled_effective",
    "mode",
    "requested_run_id",
    "actual_run_id",
    "requested_resume",
    "effective_resume",
    "backend_resume_performed",
    "resume_limitation",
    "sdk_available",
    "init_status",
    "init_error",
    "finish_status",
    "finish_count",
    "failed_log_count",
    "telemetry_error_count",
}


def _assert_summary_is_checkpoint_safe(summary: dict[str, Any]) -> None:
    """Only plain scalars: no logger object, no exception, no torch anything."""

    assert set(summary) == SUMMARY_KEYS
    for key, value in summary.items():
        assert isinstance(value, (str, bool, int, float, type(None))), f"{key}={value!r}"
        assert not isinstance(value, BaseException)
    json.dumps(summary)  # must round-trip as JSON metadata


# ---------------------------------------------------------------------------
# 1. Online: forwarding, and resume proven only by run.resumed
# ---------------------------------------------------------------------------

def test_online_resume_is_confirmed_only_by_the_sdk_resumed_flag() -> None:
    module, _run = _mock_sdk(run_id="server-run-1", resumed=True)
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(
            enabled=True,
            project="p",
            mode="online",
            run_id="server-run-1",
            resume="must",
        )

    kwargs = _init_kwargs(module)
    assert kwargs["id"] == "server-run-1"
    assert kwargs["resume"] == "must"
    assert kwargs["mode"] == "online"

    assert logger.enabled is True
    assert logger.init_status == "ok"
    assert logger.sdk_available is True
    assert logger.requested_run_id == "server-run-1"
    assert logger.actual_run_id == "server-run-1"
    assert logger.requested_resume == "must"
    assert logger.effective_resume == "must"
    assert logger.sdk_reported_resumed is True
    assert logger.backend_resume_performed is True
    assert logger.resume_limitation is None

    summary = logger.identity_summary
    _assert_summary_is_checkpoint_safe(summary)
    assert summary["backend_resume_performed"] is True
    assert summary["actual_run_id"] == "server-run-1"
    assert summary["schema_version"] == WandbLogger.IDENTITY_SCHEMA_VERSION


def test_actual_run_id_is_read_from_the_sdk_and_never_guessed() -> None:
    """A run object without a real string id yields None, not the requested id."""

    module, _run = _mock_sdk(run_id=MagicMock())  # not a str
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=True, mode="online", run_id="asked-for", resume="allow")

    assert logger.requested_run_id == "asked-for"
    assert logger.actual_run_id is None
    assert logger.sdk_reported_resumed is None
    assert logger.backend_resume_performed is False
    assert "did not report run.resumed" in (logger.resume_limitation or "")


def test_online_resume_with_a_different_returned_id_is_not_claimed_as_resumed() -> None:
    module, _run = _mock_sdk(run_id="a-fresh-run", resumed=False)
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=True, mode="online", run_id="wanted-run", resume="allow")

    assert logger.effective_resume == "allow"
    assert logger.actual_run_id == "a-fresh-run"
    assert logger.backend_resume_performed is False
    assert "run.resumed=False" in (logger.resume_limitation or "")


def test_online_resume_without_an_id_cannot_confirm_continuation() -> None:
    module, _run = _mock_sdk(run_id="auto-picked")
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=True, mode="online", resume="auto")

    assert _init_kwargs(module)["resume"] == "auto"
    assert "id" not in _init_kwargs(module)
    assert logger.actual_run_id == "auto-picked"
    assert logger.backend_resume_performed is False
    assert logger.resume_limitation is not None


def test_matching_run_id_alone_is_never_treated_as_a_resume() -> None:
    """``resume="allow"`` creates a NEW run under the requested id when none exists.

    So a returned id equal to the requested one is not evidence of anything.
    Only ``run.resumed`` is, and here the SDK explicitly says it did not resume.
    """

    module, _run = _mock_sdk(run_id="same-id", resumed=False)
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=True, mode="online", run_id="same-id", resume="allow")

    assert _init_kwargs(module)["resume"] == "allow"
    assert logger.requested_run_id == logger.actual_run_id == "same-id"
    assert logger.sdk_reported_resumed is False
    assert logger.backend_resume_performed is False
    limitation = logger.resume_limitation or ""
    assert "run.resumed=False" in limitation and "new backend run was created" in limitation
    assert logger.identity_summary["backend_resume_performed"] is False


def test_matching_run_id_with_no_resumed_flag_stays_unconfirmed() -> None:
    """An SDK that reports no usable ``resumed`` leaves the resume unconfirmed."""

    module, _run = _mock_sdk(run_id="same-id")  # no resumed attribute set
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=True, mode="online", run_id="same-id", resume="must")

    assert logger.requested_run_id == logger.actual_run_id == "same-id"
    assert logger.sdk_reported_resumed is None
    assert logger.backend_resume_performed is False
    limitation = logger.resume_limitation or ""
    assert "did not report run.resumed" in limitation
    assert "not evidence of a resume" in limitation


# ---------------------------------------------------------------------------
# 2. Offline: the resume limitation must be stated, never papered over
# ---------------------------------------------------------------------------

def test_offline_requested_resume_is_recorded_but_never_forwarded() -> None:
    module, _run = _mock_sdk(run_id="offline-run")
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(
            enabled=True, mode="offline", run_id="offline-run", resume="must"
        )

    kwargs = _init_kwargs(module)
    # The id is still useful offline (it names the local run directory); resume is not.
    assert kwargs["id"] == "offline-run"
    assert "resume" not in kwargs

    assert logger.requested_resume == "must"
    assert logger.effective_resume is None
    assert logger.backend_resume_performed is False
    limitation = logger.resume_limitation or ""
    assert "not forwarded" in limitation and "offline" in limitation


def test_offline_reusing_the_same_id_does_not_claim_a_continuing_backend_run() -> None:
    """The exact failure mode the wrapper exists to prevent."""

    module, _run = _mock_sdk(run_id="reused-id", resumed=True)
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=True, mode="offline", run_id="reused-id", resume="allow")

    # Requested and actual ids agree, and it still must not be called a resume.
    assert logger.requested_run_id == logger.actual_run_id == "reused-id"
    assert logger.backend_resume_performed is False
    assert logger.effective_resume is None
    assert logger.identity_summary["backend_resume_performed"] is False
    assert logger.identity_summary["resume_limitation"]
    # resume was never forwarded, so no returned flag can make this a resume.
    assert "not forwarded" in (logger.resume_limitation or "")


@pytest.mark.parametrize("mode", ["offline", "disabled", "OFFLINE"])
def test_non_online_modes_never_forward_resume(mode: str) -> None:
    module, _run = _mock_sdk()
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=True, mode=mode, resume="never")

    assert "resume" not in _init_kwargs(module)
    assert logger.effective_resume is None
    assert logger.resume_limitation is not None


def test_online_mode_detection_is_case_insensitive() -> None:
    module, _run = _mock_sdk(run_id="r1", resumed=True)
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=True, mode="Online", run_id="r1", resume="Must")

    assert _init_kwargs(module)["resume"] == "must"
    assert logger.requested_resume == "must"
    assert logger.backend_resume_performed is True


# ---------------------------------------------------------------------------
# 3. Argument validation against the documented wandb.init contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("resume", ["allow", "never", "must", "auto", "ALLOW", " must "])
def test_documented_resume_values_are_accepted(resume: str) -> None:
    logger = WandbLogger(enabled=False, resume=resume)
    assert logger.requested_resume == resume.strip().lower()
    assert logger.requested_resume in WandbLogger.ALLOWED_RESUME_VALUES


@pytest.mark.parametrize("resume", ["yes", "", "continue", True, False, 1, 0, object()])
def test_unknown_or_deprecated_resume_values_are_refused(resume: Any) -> None:
    """wandb documents exactly allow/never/must/auto; True/False are deprecated."""

    with pytest.raises(ValueError, match="resume must be one of"):
        WandbLogger(enabled=False, resume=resume)


@pytest.mark.parametrize("run_id", ["", "   ", True, 7, object()])
def test_malformed_run_id_is_refused(run_id: Any) -> None:
    with pytest.raises(ValueError, match="run_id must"):
        WandbLogger(enabled=False, run_id=run_id)


# ---------------------------------------------------------------------------
# 4. Disabled, missing SDK, init failure: truthful requested-vs-effective
# ---------------------------------------------------------------------------

def test_disabled_logger_never_starts_a_run() -> None:
    module, _run = _mock_sdk()
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=False, mode="online", run_id="x", resume="must")

    module.init.assert_not_called()
    summary = logger.identity_summary
    _assert_summary_is_checkpoint_safe(summary)
    assert summary["enabled_requested"] is False
    assert summary["enabled_effective"] is False
    assert summary["init_status"] == "not_attempted"
    assert summary["sdk_available"] is None
    assert summary["requested_run_id"] == "x"
    assert summary["actual_run_id"] is None
    assert summary["requested_resume"] == "must"
    assert summary["effective_resume"] is None
    assert summary["backend_resume_performed"] is False


def test_uninstalled_sdk_is_reported_as_missing_not_as_a_failure() -> None:
    # A ``None`` entry in sys.modules makes ``import wandb`` raise ImportError.
    with patch.dict("sys.modules", {"wandb": None}):
        logger = WandbLogger(enabled=True, mode="online", run_id="x", resume="must")

    assert logger.enabled is False
    assert logger.enabled_requested is True
    assert logger.sdk_available is False
    assert logger.init_status == "sdk_missing"
    assert logger.init_error == "wandb is not installed"
    assert logger.failed_log_count == 0
    assert logger.backend_resume_performed is False
    assert "no backend resume was performed" in (logger.resume_limitation or "")
    _assert_summary_is_checkpoint_safe(logger.identity_summary)


def test_shadowed_wandb_module_without_init_is_reported_as_missing_sdk() -> None:
    """A local ./wandb output directory imports cleanly but has no ``init``."""

    shadow = MagicMock()
    del shadow.init  # attribute lookup now raises AttributeError -> not callable
    with patch.dict("sys.modules", {"wandb": shadow}):
        logger = WandbLogger(enabled=True, mode="online", run_id="x")

    assert logger.enabled is False
    assert logger.sdk_available is False
    assert logger.init_status == "sdk_missing"
    assert "no callable init" in (logger.init_error or "")


def test_init_failure_is_recorded_truthfully_and_disables_logging() -> None:
    module, _run = _mock_sdk()
    module.init.side_effect = RuntimeError("backend refused the run id")
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=True, mode="online", run_id="taken", resume="must")

    assert logger.enabled is False
    assert logger.enabled_requested is True
    assert logger.sdk_available is True
    assert logger.init_status == "failed"
    assert logger.init_error == "backend refused the run id"
    assert logger.failed_log_count == 1
    assert logger.telemetry_errors == ["init: backend refused the run id"]
    assert isinstance(logger.last_error, RuntimeError)
    assert logger.actual_run_id is None
    assert logger.effective_resume is None
    assert logger.backend_resume_performed is False
    assert "no backend resume was performed" in (logger.resume_limitation or "")
    # Telemetry after a failed init stays a graceful no-op.
    logger.log({"loss": 0.1}, step=1)
    logger.set_summary({"final": 1.0})
    assert logger.failed_log_count == 1
    _assert_summary_is_checkpoint_safe(logger.identity_summary)


# ---------------------------------------------------------------------------
# 5. finish: idempotent, with a consistent status and failure count
# ---------------------------------------------------------------------------

def test_finish_is_idempotent_and_records_success_once() -> None:
    module, _run = _mock_sdk(run_id="r")
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=True, mode="online", run_id="r", resume="must")
        logger.finish()
        logger.finish(exit_code=1)
        logger.finish()

    module.finish.assert_called_once_with()
    assert logger.finish_count == 1
    assert logger.finish_status == "ok"
    assert logger.telemetry_errors == []
    assert logger.identity_summary["finish_status"] == "ok"
    assert logger.identity_summary["finish_count"] == 1


def test_finish_failure_is_recorded_once_and_not_retried() -> None:
    module, _run = _mock_sdk(run_id="r")
    module.finish.side_effect = RuntimeError("finish exploded")
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=True, mode="offline", run_id="r")
        logger.finish(exit_code=1)
        logger.finish(exit_code=1)

    module.finish.assert_called_once_with(exit_code=1)
    assert logger.finish_count == 1
    assert logger.finish_status == "failed"
    assert logger.telemetry_errors == ["finish: finish exploded"]
    assert isinstance(logger.last_error, RuntimeError)
    _assert_summary_is_checkpoint_safe(logger.identity_summary)


def test_finish_on_a_logger_that_never_started_is_a_noop() -> None:
    logger = WandbLogger(enabled=False)
    logger.finish()
    logger.finish()
    assert logger.finish_count == 0
    assert logger.finish_status == "not_finished"


# ---------------------------------------------------------------------------
# 6. Backward compatibility of the existing no-id / no-resume behaviour
# ---------------------------------------------------------------------------

def test_existing_construction_without_id_or_resume_is_unchanged() -> None:
    module, _run = _mock_sdk(run_id="legacy")
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(enabled=True, project="test-proj", mode="offline")
        logger.log({"train/loss": 0.25, "val/dice": float("nan")}, step=2)

    kwargs = _init_kwargs(module)
    assert "id" not in kwargs and "resume" not in kwargs
    assert kwargs["project"] == "test-proj" and kwargs["mode"] == "offline"
    assert logger.enabled is True
    assert logger.requested_run_id is None
    assert logger.requested_resume is None
    assert logger.effective_resume is None
    assert logger.resume_limitation is None
    assert logger.backend_resume_performed is False
    # The strict payload conversion is untouched: NaN becomes an explicit null.
    logged = module.log.call_args[0][0]
    assert logged["train/loss"] == 0.25
    assert logged["val/dice"] is None


def test_positional_construction_still_matches_the_previous_signature() -> None:
    """The two new parameters are appended, so positional callers are unaffected."""

    module, _run = _mock_sdk()
    with patch.dict("sys.modules", {"wandb": module}):
        logger = WandbLogger(True, "proj", "ent", "name", "grp", ["t"], {"lr": 0.1}, "offline")

    kwargs = _init_kwargs(module)
    assert kwargs["project"] == "proj"
    assert kwargs["entity"] == "ent"
    assert kwargs["name"] == "name"
    assert kwargs["group"] == "grp"
    assert kwargs["tags"] == ["t"]
    assert kwargs["config"] == {"lr": 0.1}
    assert kwargs["mode"] == "offline"
    assert logger.requested_run_id is None and logger.requested_resume is None
