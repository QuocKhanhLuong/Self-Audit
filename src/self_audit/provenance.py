"""Narrow producer-side provenance helpers for checkpoint binding (W3.1).

The unified runner used to collect its validation transition cache from the
live last-epoch model and only afterwards hash ``phase_c_best.pt`` for the
artifact identity.  Hashing one file while measuring a different set of
weights is not identity: this module supplies the primitives that let the
producer bind the *actual* state it evaluates.

Three separate things are recorded and never conflated:

``checkpoint_sha256``
    The bytes of the file that was selected.
``state_digest``
    A deterministic digest over the parameter/buffer tensors that are live in
    the model *after* loading, so a later mutation of the live module is
    detectable at every consumer boundary.
``producer_git_sha``
    The source revision that *wrote* the checkpoint, read out of the
    checkpoint payload.  A legacy checkpoint without that record stays
    ``"unknown"`` forever; the current evaluation revision is recorded
    separately and is never back-filled into a historical slot.

Nothing here reads environment variables, credentials, or provider/account
state.  The only external process consulted is ``git`` in the repository
working tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch
from torch import nn

PROVENANCE_SCHEMA_VERSION = 1
LINEAGE_SCHEMA_VERSION = 1
UNKNOWN = "unknown"

#: Membership signatures cover *which* records a split contained, never the
#: pixel content of those records.  Consumers must not read a matching
#: membership signature as proof that the underlying files are unchanged.
MEMBERSHIP_COVERAGE = "membership_only_not_content_hash"

#: Paths whose modification means the *model/evaluation code* changed.  A
#: dirty ``reports/`` tree is a report-only commit and must not be reported as
#: a source change.
SOURCE_PATHS = ("src", "scripts", "configs")

#: Version of the source-content signature algorithm.  Bump this whenever the
#: scope or the framing below changes, so an old signature can never be
#: compared against a new one as though they meant the same thing.
SOURCE_SIGNATURE_VERSION = 2

#: The *active source* whose content decides what a measurement means.  These
#: are enumerated explicitly rather than discovered: no whole-workspace walk,
#: no data, no checkpoints, and nothing outside the repository.
SOURCE_SIGNATURE_ROOTS = ("src/self_audit", "scripts", "configs")

#: Only code and configuration participate.  Notebooks, docs and binaries do
#: not change what the pipeline computes. Active shell and PowerShell script
#: wrappers in scripts/ are included.
SOURCE_SIGNATURE_SUFFIXES = (".py", ".yaml", ".yml", ".sh", ".ps1")

#: Directory names pruned anywhere under a root.  ``reports`` and ``tests``
#: are excluded on purpose: a report-only commit and a test-only edit must
#: both leave the signature untouched.  ``data`` is not excluded here because
#: ``src/self_audit/data`` is an active code package; raw data at repository
#: root is kept out by explicit scoping to SOURCE_SIGNATURE_ROOTS.
SOURCE_SIGNATURE_EXCLUDED_DIRS = (
    "__pycache__",
    ".git",
    ".pytest_cache",
    "checkpoints",
    "external",
    "preprocessed_data",
    "reports",
    "scratch",
    "splits",
    "tests",
    "wandb",
    "weights",
)

SOURCE_SIGNATURE_EXCLUDED_SUFFIXES = (".pyc", ".pyo")

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return the raw little-endian bytes backing ``tensor``.

    The uint8 view keeps ``bfloat16`` (and any other dtype NumPy cannot
    represent) hashable: the conversion never goes through a NumPy dtype of
    the tensor's own type.
    """

    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    if flat.numel() == 0:
        return b""
    return flat.view(torch.uint8).numpy().tobytes()


def _iter_state(state: Any) -> Mapping[str, Any]:
    if isinstance(state, nn.Module):
        return state.state_dict()
    if isinstance(state, Mapping):
        return state
    raise TypeError(f"state_digest expects nn.Module or mapping, got {type(state).__name__}")


def state_digest(state: nn.Module | Mapping[str, Any]) -> str:
    """Deterministic SHA-256 over every parameter/buffer entry of ``state``.

    Keys are visited in sorted order and each entry contributes its key, its
    shape, its dtype and its raw bytes with explicit length framing, so two
    different states cannot collide by concatenation.  Scalar (0-dim) integer
    buffers and ``bfloat16`` tensors are included like any other entry.
    """

    mapping = _iter_state(state)
    digest = hashlib.sha256()
    digest.update(b"self_audit.state_digest.v1\x00")
    for key in sorted(mapping):
        value = mapping[key]
        digest.update(b"key\x00")
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\x00")
        if torch.is_tensor(value):
            payload = _tensor_bytes(value)
            digest.update(b"tensor\x00")
            digest.update(str(tuple(int(dim) for dim in value.shape)).encode("utf-8"))
            digest.update(b"\x00")
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(b"\x00")
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
        else:
            encoded = repr(value).encode("utf-8")
            digest.update(b"nontensor\x00")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        digest.update(b"\x00end\x00")
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of ``path``; a missing file is an error here."""

    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_signature(payload: Mapping[str, Any], *, domain: str) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{domain}\x00{encoded}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Source revision
# ---------------------------------------------------------------------------


def _git(*args: str, root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _scoped_source_files(base: Path) -> list[Path]:
    """List the scoped source files, deterministically and without wandering.

    Walks only :data:`SOURCE_SIGNATURE_ROOTS`, prunes
    :data:`SOURCE_SIGNATURE_EXCLUDED_DIRS` at every level, follows no symlinks
    out of the tree, and returns paths sorted by their repository-relative
    POSIX form so the order never depends on the filesystem.
    Missing required scope roots raise FileNotFoundError; directory read
    errors raise OSError so callers can fail closed rather than claim a
    partial signature.
    """

    found: list[Path] = []
    for relative_root in SOURCE_SIGNATURE_ROOTS:
        start = base / relative_root
        if not start.is_dir():
            raise FileNotFoundError(f"Missing required scope root: {relative_root}")
        stack = [start]
        while stack:
            directory = stack.pop()
            entries = sorted(directory.iterdir())
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if entry.name in SOURCE_SIGNATURE_EXCLUDED_DIRS:
                        continue
                    if entry.name == "data" and entry.relative_to(base).as_posix() != "src/self_audit/data":
                        continue
                    stack.append(entry)
                    continue
                if not entry.is_file():
                    continue
                if entry.suffix in SOURCE_SIGNATURE_EXCLUDED_SUFFIXES:
                    continue
                if entry.suffix not in SOURCE_SIGNATURE_SUFFIXES:
                    continue
                found.append(entry)
    return sorted(found, key=lambda path: path.relative_to(base).as_posix())


def source_content_signature(root: str | Path | None = None) -> dict[str, Any]:
    """Digest the content of the active source and configuration.

    What this is *for*: a calibration produced under one version of the
    evaluation code must not be silently reused under another.  A git SHA
    cannot answer that -- it changes on every report-only commit and misses
    uncommitted edits entirely -- so the content of the scoped files is
    hashed directly.  Each file contributes its repository-relative path, its
    byte length and the digest of its bytes, all length-framed, so neither a
    rename nor a content change can be hidden by another.

    Nothing outside :data:`SOURCE_SIGNATURE_ROOTS` is read: no data, no
    checkpoints, no reports, no tests, and nothing belonging to a provider,
    account or environment.  When the scope cannot be read the signature is
    :data:`UNKNOWN` with ``source_content_signature_known`` ``False`` -- an
    unknown signature is never a verified one.
    """

    base = Path(root) if root is not None else _REPO_ROOT
    scope: dict[str, Any] = {
        "roots": list(SOURCE_SIGNATURE_ROOTS),
        "suffixes": list(SOURCE_SIGNATURE_SUFFIXES),
        "excluded_dirs": list(SOURCE_SIGNATURE_EXCLUDED_DIRS),
        "excluded_suffixes": list(SOURCE_SIGNATURE_EXCLUDED_SUFFIXES),
        "file_count": 0,
    }
    try:
        files = _scoped_source_files(base)
    except OSError:
        return {
            "source_content_signature": UNKNOWN,
            "source_content_signature_known": False,
            "source_signature_version": int(SOURCE_SIGNATURE_VERSION),
            "source_signature_scope": scope,
        }
    if not files:
        return {
            "source_content_signature": UNKNOWN,
            "source_content_signature_known": False,
            "source_signature_version": int(SOURCE_SIGNATURE_VERSION),
            "source_signature_scope": scope,
        }
    digest = hashlib.sha256()
    digest.update(f"self_audit.source_content_signature.v{SOURCE_SIGNATURE_VERSION}\x00".encode("utf-8"))
    counted = 0
    for path in files:
        try:
            payload = path.read_bytes()
        except OSError:
            # One unreadable scoped file makes the whole signature a guess.
            return {
                "source_content_signature": UNKNOWN,
                "source_content_signature_known": False,
                "source_signature_version": int(SOURCE_SIGNATURE_VERSION),
                "source_signature_scope": scope,
            }
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(hashlib.sha256(payload).digest())
        counted += 1
    scope["file_count"] = counted
    return {
        "source_content_signature": digest.hexdigest(),
        "source_content_signature_known": True,
        "source_signature_version": int(SOURCE_SIGNATURE_VERSION),
        "source_signature_scope": scope,
    }


def git_source_provenance(root: str | Path | None = None) -> dict[str, Any]:
    """Describe the revision of the working tree that is running right now.

    ``source_dirty`` is restricted to :data:`SOURCE_PATHS` so that an
    uncommitted report or scratch file is not reported as a model-code change.
    ``dirty`` covers the whole tree and is kept separately.  Both are ``None``
    when git is unavailable -- never silently ``False``.

    ``source_content_signature`` answers the question the git SHA cannot: did
    the code that decides what a measurement *means* change?  A commit SHA
    moves on every report-only commit and says nothing about uncommitted
    edits; the content signature is stable across both and changes only when
    scoped source or configuration content changes.
    """

    base = Path(root) if root is not None else _REPO_ROOT
    signature = source_content_signature(base)
    commit = _git("rev-parse", "HEAD", root=base)
    if commit is None:
        return {
            "git_sha": UNKNOWN,
            "dirty": None,
            "source_dirty": None,
            "source_paths": list(SOURCE_PATHS),
            **signature,
        }
    whole = _git("status", "--porcelain", root=base)
    source = _git("status", "--porcelain", "--", *SOURCE_PATHS, root=base)
    return {
        "git_sha": commit.strip(),
        "dirty": None if whole is None else bool(whole.strip()),
        "source_dirty": None if source is None else bool(source.strip()),
        "source_paths": list(SOURCE_PATHS),
        **signature,
    }


def producer_provenance_record(root: str | Path | None = None) -> dict[str, Any]:
    """Build the block ``save_checkpoint`` stamps into a new checkpoint."""

    source = git_source_provenance(root)
    return {
        "provenance_schema_version": int(PROVENANCE_SCHEMA_VERSION),
        "producer_git_sha": str(source["git_sha"]),
        "producer_dirty": source["dirty"],
        "producer_source_dirty": source["source_dirty"],
        # The source that *produced* this checkpoint, recorded separately from
        # whatever source later reads it back.
        "producer_source_content_signature": str(source["source_content_signature"]),
        "producer_source_content_signature_known": bool(source["source_content_signature_known"]),
        "producer_source_signature_version": int(source["source_signature_version"]),
        "producer_source_signature_scope": dict(source["source_signature_scope"]),
        "producer_torch_version": str(torch.__version__),
    }


def checkpoint_producer(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read the producing revision out of a checkpoint payload.

    A checkpoint written before this record existed reports
    ``producer_git_sha == "unknown"`` with ``producer_recorded == False``.  The
    current HEAD is *never* substituted: an unknown producer must stay
    visibly unknown rather than be dressed up as history.
    """

    record = None
    if isinstance(payload, Mapping):
        candidate = payload.get("provenance")
        if isinstance(candidate, Mapping):
            record = candidate
    if record is None:
        return {
            "producer_recorded": False,
            "provenance_schema_version": None,
            "producer_git_sha": UNKNOWN,
            "producer_dirty": None,
            "producer_source_dirty": None,
            "producer_torch_version": None,
            "producer_state_digest": None,
            "producer_source_content_signature": None,
            "producer_source_content_signature_known": False,
            "producer_source_signature_version": None,
            "producer_source_signature_scope": None,
        }
    return {
        "producer_recorded": True,
        "provenance_schema_version": record.get("provenance_schema_version"),
        "producer_git_sha": str(record.get("producer_git_sha", UNKNOWN)),
        "producer_dirty": record.get("producer_dirty"),
        "producer_source_dirty": record.get("producer_source_dirty"),
        "producer_torch_version": record.get("producer_torch_version"),
        "producer_state_digest": record.get("state_digest"),
        # A checkpoint written before source signatures existed reports an
        # unknown producing source; the reader's own signature is never
        # substituted for it.
        "producer_source_content_signature": record.get("producer_source_content_signature"),
        "producer_source_content_signature_known": bool(
            record.get("producer_source_content_signature_known", False)
        ),
        "producer_source_signature_version": record.get("producer_source_signature_version"),
        "producer_source_signature_scope": record.get("producer_source_signature_scope"),
    }


# ---------------------------------------------------------------------------
# Resolved model identity
# ---------------------------------------------------------------------------


#: The historical execution mechanism.  A model running this mode computes
#: exactly what every pre-Candidate-C revision computed, so it contributes
#: nothing to the signed identity and historical signatures are preserved
#: byte for byte.  The authoritative list of valid modes lives with the model
#: and the configuration layer; this module deliberately knows only which
#: single value is the no-op, so it never has to be kept in sync with the rest
#: and never imports from either layer.
DEFAULT_WINDOW_MODE = "current"

#: Identity fields that are recorded but not signed.  ``parameter_count`` is
#: excluded because it is derivable; the mechanism fields are excluded from
#: the *unconditional* payload because they are folded in only when the model
#: actually runs a non-default mechanism (see :func:`resolve_model_identity`).
_UNSIGNED_IDENTITY_FIELDS = frozenset(
    {"parameter_count", "window_mode", "candidate_c_settings", "candidate_c_signature"}
)


def _resolve_window_mode(model: nn.Module) -> str:
    """Read the live execution mode, or report it unknown.

    A module that predates the execution-mode switches has no attribute and is
    reported as :data:`UNKNOWN` -- never silently as ``"current"``, because
    "this model cannot tell me" and "this model told me it is the baseline"
    are different facts and the consumer decides what to do about each.
    """

    value = getattr(model, "window_mode", None)
    if value is None:
        return UNKNOWN
    if not isinstance(value, str) or not value.strip():
        return UNKNOWN
    return str(value)


def _resolve_candidate_c_settings(model: nn.Module) -> dict[str, Any] | None:
    """Read the live Candidate C solver settings as plain primitives.

    Accepts either an object exposing ``as_dict()`` or a plain mapping, so no
    import of the model package is needed here and no import cycle is created.
    ``None`` means the live model could not report settings at all.
    """

    value = getattr(model, "candidate_c", None)
    if value is None:
        return None
    raw: Any = value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        try:
            raw = as_dict()
        except Exception:  # pragma: no cover - a broken accessor is "unreadable"
            return None
    if not isinstance(raw, Mapping):
        return None
    resolved: dict[str, Any] = {}
    for key in sorted(raw):
        entry = raw[key]
        if entry is None or isinstance(entry, str):
            resolved[str(key)] = entry
        elif isinstance(entry, bool):
            resolved[str(key)] = bool(entry)
        elif isinstance(entry, int):
            resolved[str(key)] = int(entry)
        elif isinstance(entry, float):
            resolved[str(key)] = float(entry)
        else:
            # An unreportable value makes the whole settings block a guess.
            return None
    return resolved


def _candidate_c_signature(settings: Mapping[str, Any] | None) -> str:
    """Sign the solver settings, or report them unknown."""

    if settings is None:
        return UNKNOWN
    return _stable_signature(dict(settings), domain="candidate_c_settings.v1")


def resolve_model_identity(model: nn.Module) -> dict[str, Any]:
    """Describe the model that was actually constructed and loaded.

    Every field is read off live modules.  Nothing is inferred from a
    checkpoint filename, a directory name, or the YAML that was *requested* --
    a fallback encoder that silently replaced the ImageNet backbone shows up
    here as ``encoder_backend == "fallback_synthetic"``.

    The *mechanism* the model runs is recorded alongside its architecture.
    Equal weights are not equal models: Candidate C adds no parameter, so a
    ``current`` network and a ``candidate_c`` network share every tensor, every
    state digest and every file hash while computing different transitions.
    ``window_mode`` and ``candidate_c_signature`` are therefore always recorded,
    and they are folded into ``signature`` exactly when the live model runs a
    non-default mechanism.  A model running :data:`DEFAULT_WINDOW_MODE`, and a
    legacy module that cannot report a mode at all, keep the historical
    signature unchanged, so existing artifacts stay comparable; anything else
    signs differently and cannot be mistaken for the baseline.

    Solver settings are signed only under a non-default mode because they are
    inert otherwise: two ``current`` runs whose configuration happens to carry
    different Candidate C values compute the same thing.
    """

    encoder = getattr(model, "encoder", None)
    expert = getattr(model, "annotation_expert", None)
    window = getattr(expert, "refinement_block", None)
    fpn = getattr(model, "fpn", None)
    using_timm = getattr(encoder, "using_timm", None)
    identity: dict[str, Any] = {
        "class_name": type(model).__name__,
        "num_classes": int(getattr(model, "num_classes", getattr(expert, "num_classes", -1))),
        "shared_channels": int(getattr(fpn, "out_channels", -1)),
        "window_k": int(getattr(window, "k", -1)),
        "max_turns": int(getattr(model, "max_turns", -1)),
        "encoder_name": str(getattr(encoder, "name", UNKNOWN)),
        "encoder_pretrained_requested": getattr(encoder, "pretrained", None),
        "encoder_backend": (
            UNKNOWN if using_timm is None else ("timm" if bool(using_timm) else "fallback_synthetic")
        ),
        "entropy_version": str(getattr(expert, "entropy_version", UNKNOWN)),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "buffer_count": int(sum(1 for _ in model.buffers())),
        "torch_version": str(torch.__version__),
    }
    window_mode = _resolve_window_mode(model)
    candidate_c_settings = _resolve_candidate_c_settings(model)
    identity["window_mode"] = window_mode
    identity["candidate_c_settings"] = candidate_c_settings
    identity["candidate_c_signature"] = _candidate_c_signature(candidate_c_settings)
    signed = {
        key: identity[key] for key in sorted(identity) if key not in _UNSIGNED_IDENTITY_FIELDS
    }
    if window_mode not in (DEFAULT_WINDOW_MODE, UNKNOWN):
        # A non-default mechanism is part of what the weights mean, so it joins
        # the signature together with the settings that parameterise it.  An
        # unreadable settings block signs as "unknown" rather than as absent:
        # it must not collide with a model whose settings were readable.
        signed["window_mode"] = window_mode
        signed["candidate_c_signature"] = identity["candidate_c_signature"]
    identity["signature"] = _stable_signature(signed, domain="model_identity.v1")
    return identity


def _verify_execution_mechanism(
    configured: Mapping[str, Any],
    identity: Mapping[str, Any],
    mismatches: list[str],
    unresolved: list[str],
) -> None:
    """Compare the configured mechanism against the live one, failing closed.

    Appends to ``mismatches``/``unresolved`` rather than raising, so a
    mechanism disagreement is reported in the same refusal as an architecture
    disagreement instead of masking it.
    """

    live_mode = str(identity.get("window_mode", UNKNOWN))
    if "window_mode" in configured:
        declared = configured["window_mode"]
        if not isinstance(declared, str) or not declared.strip():
            unresolved.append(f"window_mode (config={declared!r} is not a mode name)")
        elif live_mode == UNKNOWN:
            if declared != DEFAULT_WINDOW_MODE:
                unresolved.append(
                    f"window_mode (config={declared!r}; the live model reports no execution "
                    f"mode, and only {DEFAULT_WINDOW_MODE!r} can be satisfied without one)"
                )
        elif declared != live_mode:
            mismatches.append(f"window_mode: config={declared!r}, model={live_mode!r}")

    if "candidate_c" not in configured:
        return
    if live_mode in (DEFAULT_WINDOW_MODE, UNKNOWN):
        # The solver never runs, so its settings do not describe this model and
        # comparing them would refuse runs that compute identical results.
        return
    declared_settings = configured["candidate_c"]
    if declared_settings is None:
        # An explicit null means "the documented defaults", which the live model
        # has already resolved; there is nothing to contradict.
        return
    if not isinstance(declared_settings, Mapping):
        unresolved.append(
            f"candidate_c (config={type(declared_settings).__name__} is not a settings mapping)"
        )
        return
    live_settings = identity.get("candidate_c_settings")
    if not isinstance(live_settings, Mapping):
        unresolved.append(
            f"candidate_c (config declares solver settings under window_mode={live_mode!r} "
            "but the live model cannot report any)"
        )
        return
    for key in sorted(declared_settings):
        name = str(key)
        if name not in live_settings:
            unresolved.append(
                f"candidate_c.{name} (config={declared_settings[key]!r}; the live model "
                "reports no such setting)"
            )
            continue
        if not _mechanism_values_equal(declared_settings[key], live_settings[name]):
            mismatches.append(
                f"candidate_c.{name}: config={declared_settings[key]!r}, "
                f"model={live_settings[name]!r}"
            )


def _mechanism_values_equal(declared: Any, live: Any) -> bool:
    """Compare one solver setting without coercing across kinds.

    ``None`` (the documented "use the normalized proposal rule" sentinel) is
    only equal to ``None``; booleans never compare equal to numbers; numbers
    compare by exact float value so ``1`` and ``1.0`` agree.
    """

    if declared is None or live is None:
        return declared is None and live is None
    if isinstance(declared, bool) or isinstance(live, bool):
        return isinstance(declared, bool) and isinstance(live, bool) and declared is live
    if isinstance(declared, (int, float)) and isinstance(live, (int, float)):
        return float(declared) == float(live)
    return str(declared) == str(live)


def verify_model_config(model: nn.Module, config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fail when the live model disagrees with the configuration it claims.

    Returns the resolved identity when compatible.  Unset configuration keys
    are not invented, and a value the live model could not report (``-1`` /
    ``"unknown"``) is skipped rather than treated as a mismatch.

    The execution mechanism is verified here too.  A configured
    ``model.window_mode`` must equal the mode the live model reports, and under
    a non-default mode the configured ``model.candidate_c`` settings must equal
    the live solver settings key for key.  A module that cannot report a mode
    binds only against the configured default, which is the one claim a
    pre-mechanism module can honestly satisfy; a non-default claim against such
    a module fails closed rather than running the baseline under another
    mode's name.
    """

    identity = resolve_model_identity(model)
    if config is None:
        return identity
    if not isinstance(config, Mapping):
        raise ValueError("model config must be a mapping")
    configured = config.get("model", config)
    if not isinstance(configured, Mapping):
        return identity
    mismatches: list[str] = []
    unresolved: list[str] = []
    for key in ("num_classes", "shared_channels", "window_k", "max_turns", "encoder_name"):
        if key not in configured:
            continue
        actual = identity.get(key)
        if actual in {-1, UNKNOWN, ""}:
            # A configured key the live model cannot report is a failure, not a
            # free pass: an unreadable attribute must never bypass the check.
            unresolved.append(f"{key} (config={configured[key]!r})")
            continue
        if str(configured[key]) != str(actual):
            mismatches.append(f"{key}: config={configured[key]!r}, model={actual!r}")
    _verify_execution_mechanism(configured, identity, mismatches, unresolved)
    if unresolved:
        raise ValueError(
            "Cannot resolve live model attribute(s) required by the configuration; "
            "refusing to bind without verification: " + "; ".join(unresolved)
        )
    if mismatches:
        raise ValueError(
            "Resolved model does not match its configuration; refusing to bind: "
            + "; ".join(mismatches)
        )
    return identity


# ---------------------------------------------------------------------------
# Data-side descriptors
# ---------------------------------------------------------------------------


#: Samplers whose coverage of a dataset can be characterised exactly.  Anything
#: else is refused rather than described with a guess.
_FULL_COVERAGE_SAMPLERS = ("SequentialSampler", "RandomSampler")
_SUBSET_SAMPLERS = ("SubsetRandomSampler",)


def _unwrap_dataset(source: Any) -> tuple[Any, list[int] | None]:
    """Return the base dataset and, for a ``Subset``, its exact indices.

    A ``Subset`` is not transparent: dropping it would let a slice of a split
    be described with the whole split's identity.  The indices are carried out
    so the caller can record what was really iterated.
    """

    current = source
    indices: list[int] | None = None
    seen = 0
    while hasattr(current, "dataset") and seen < 8:
        level = getattr(current, "indices", None)
        if level is not None:
            # ``level`` maps this level's positions onto its parent's.  Indices
            # accumulated so far are expressed in this level's space, so they
            # are re-expressed in the parent's space; outer-to-inner order.
            resolved = [int(value) for value in level]
            indices = resolved if indices is None else [resolved[position] for position in indices]
        current = current.dataset
        seen += 1
    return current, indices


def _known_preprocessing_pipeline(dataset: Any) -> bool:
    """True only for the project's own volume-slice pipeline.

    A generic dataset must not be described as percentile-clipped and
    z-scored just because it was passed to a loader.
    """

    return all(
        hasattr(dataset, attribute)
        for attribute in ("records", "index_map", "lower_percentile", "upper_percentile", "image_size")
    )


#: Geometry metadata contract of the project's volume-slice pipeline.  These
#: are declared recipe properties, identical for every sample the pipeline
#: emits; they are recorded without touching the data.
_PIPELINE_GEOMETRY_KEYS = (
    "source_shape",
    "network_shape",
    "source_spacing",
    "effective_spacing",
    "spacing_known",
    "spacing_units",
    "source_axis_order",
    "network_axis_order",
)


def _geometry_semantics(dataset: Any) -> dict[str, Any]:
    """Record the geometry metadata contract this dataset's recipe declares.

    Deliberately does **not** index the dataset: fetching a sample would load
    ground truth, run the transform and consume RNG, and one sample's spacing
    could never speak for a whole cohort anyway.  Per-record spacing
    availability is cohort content and is reported by :func:`cohort_identity`.
    """

    if not _known_preprocessing_pipeline(dataset):
        return {
            "status": "unknown_dataset_no_geometry_contract",
            "declared_keys": [],
            "resize_semantics": None,
        }
    return {
        "status": "declared_by_pipeline_recipe",
        "declared_keys": list(_PIPELINE_GEOMETRY_KEYS),
        "source_axis_order": "ZHW",
        "network_axis_order": "ZHW",
        # In-plane resize only: Z is preserved exactly, images are bilinear and
        # masks nearest, and in-plane spacing is rescaled by the H/W ratio.
        "resize_semantics": "in_plane_bilinear_image_nearest_mask_z_preserved",
        "spacing_semantics": "effective_spacing_rescaled_by_inplane_resize_ratio",
        "spacing_units_policy": "mm_when_source_spacing_known_else_pixel",
    }


def preprocessing_descriptor(source: Any) -> dict[str, Any]:
    """Describe the preprocessing recipe the *actual* dataset performs.

    Read off the constructed dataset, not off the config, so an override that
    never reached the dataset cannot masquerade as the effective setting.

    This is a *recipe* identity: it deliberately excludes how many records or
    slices were present, so two disjoint cohorts processed the same way share
    one preprocessing signature.  Cohort size lives in
    :func:`cohort_identity`.
    """

    dataset, _ = _unwrap_dataset(source)
    image_size = getattr(dataset, "image_size", None)
    if isinstance(image_size, (list, tuple)):
        image_size = [int(value) for value in image_size]
    elif image_size is not None:
        image_size = int(image_size)
    transform = getattr(dataset, "transform", None)
    known = _known_preprocessing_pipeline(dataset)
    descriptor: dict[str, Any] = {
        "dataset_class": type(dataset).__name__,
        "image_size": image_size,
        "depth_axis": getattr(dataset, "depth_axis", None),
        "expected_slices": getattr(dataset, "expected_slices", None),
        "foreground_only": bool(getattr(dataset, "foreground_only", False)),
        "augment": bool(getattr(dataset, "augment", False)),
        "lower_percentile": getattr(dataset, "lower_percentile", None),
        "upper_percentile": getattr(dataset, "upper_percentile", None),
        # An arbitrary dataset gets no claim to this pipeline's normalization.
        "normalization": "percentile_clip_and_zscore" if known else UNKNOWN,
        "input_construction": "25d_triplet_replicated_boundary" if known else UNKNOWN,
        "transform": None if transform is None else type(transform).__name__,
        "geometry": _geometry_semantics(dataset),
    }
    descriptor["signature"] = _stable_signature(descriptor, domain="preprocessing.v1")
    return descriptor


def split_membership_descriptor(source: Any, *, split_name: str) -> dict[str, Any]:
    """Describe which records the evaluated split actually holds.

    The signature is a *membership* signature: it proves the same case ids
    were present, never that their pixel content is unchanged.  That limit is
    recorded verbatim in ``coverage``.
    """

    from .data.common import compute_split_signature

    dataset, _ = _unwrap_dataset(source)
    records = list(getattr(dataset, "records", []) or [])
    case_ids = sorted(str(getattr(record, "case_id", record)) for record in records)
    patient_ids = sorted({str(getattr(record, "patient_id", "")) for record in records} - {""})
    descriptor: dict[str, Any] = {
        "split_name": str(split_name),
        "num_records": len(case_ids),
        "num_patients": len(patient_ids),
        "case_ids": case_ids,
        "patient_ids": patient_ids,
        "coverage": MEMBERSHIP_COVERAGE,
    }
    try:
        descriptor["membership_signature"] = compute_split_signature({str(split_name): records})
    except Exception:  # pragma: no cover - defensive; falls back to local hash
        descriptor["membership_signature"] = _stable_signature(
            {"split": str(split_name), "case_ids": case_ids}, domain="split_membership.v1"
        )
    return descriptor


def _sampling_identity(loader: Any, dataset: Any, subset_indices: list[int] | None) -> dict[str, Any]:
    """Characterise exactly what the loader iterates, or refuse to guess.

    A sampler this function cannot characterise is an error: describing an
    unknown iteration order as full coverage is precisely the dishonesty this
    record exists to prevent.
    """

    identity: dict[str, Any] = {
        "sampler": None,
        "batch_sampler": None,
        "drop_last": None,
        "shuffled": None,
        "subset_size": None,
        "subset_indices_signature": None,
        "iterates_every_record": None,
    }
    if subset_indices is not None:
        identity["subset_size"] = len(subset_indices)
        identity["subset_indices_signature"] = _stable_signature(
            {"indices": list(subset_indices)}, domain="subset_indices.v1"
        )
    if loader is dataset or not hasattr(loader, "sampler"):
        identity["iterates_every_record"] = subset_indices is None
        return identity

    sampler = getattr(loader, "sampler", None)
    batch_sampler = getattr(loader, "batch_sampler", None)
    sampler_name = type(sampler).__name__ if sampler is not None else None
    batch_sampler_name = type(batch_sampler).__name__ if batch_sampler is not None else None
    identity["sampler"] = sampler_name
    identity["batch_sampler"] = batch_sampler_name
    identity["drop_last"] = bool(getattr(loader, "drop_last", False))

    if batch_sampler_name not in (None, "BatchSampler"):
        raise ValueError(
            f"Unsupported batch_sampler {batch_sampler_name!r}: cohort identity cannot be "
            "certified for a custom batching scheme"
        )
    if sampler_name is None:
        identity["shuffled"] = False
        covered = subset_indices is None
    elif sampler_name in _FULL_COVERAGE_SAMPLERS:
        identity["shuffled"] = sampler_name == "RandomSampler"
        replacement = bool(getattr(sampler, "replacement", False))
        num_samples = getattr(sampler, "_num_samples", None)
        if replacement or num_samples is not None:
            raise ValueError(
                f"Unsupported {sampler_name} configuration (replacement={replacement}, "
                f"num_samples={num_samples}): cohort identity cannot be certified"
            )
        covered = subset_indices is None
    elif sampler_name in _SUBSET_SAMPLERS:
        indices = [int(value) for value in getattr(sampler, "indices", [])]
        identity["shuffled"] = True
        identity["subset_size"] = len(indices)
        identity["subset_indices_signature"] = _stable_signature(
            {"indices": indices}, domain="subset_indices.v1"
        )
        covered = False
    else:
        raise ValueError(
            f"Unsupported sampler {sampler_name!r}: cohort identity cannot be certified. "
            "Record the exact observed cohort or use a supported sampler."
        )

    identity["iterates_every_record"] = bool(covered)
    return identity


def cohort_identity(
    source: Any,
    *,
    split_name: str,
    max_batches: int | None = None,
    batch_size: int | None = None,
    observed_samples: int | None = None,
) -> dict[str, Any]:
    """Identify the exact cohort a measurement ran over.

    ``covers_full_split`` is ``True`` only when every record of the split was
    actually iterated: no ``max_batches`` truncation, no ``Subset``, no
    subset sampler, and no ``drop_last`` remainder.  ``max_batches`` is
    protocol metadata, never an equivalent full-cohort measurement.
    """

    dataset, subset_indices = _unwrap_dataset(source)
    available = int(len(dataset)) if hasattr(dataset, "__len__") else None
    iterated = len(subset_indices) if subset_indices is not None else available
    loader_batches = None
    if hasattr(source, "__len__") and source is not dataset:
        try:
            loader_batches = int(len(source))
        except TypeError:  # pragma: no cover - iterable-style loaders
            loader_batches = None
    sampling = _sampling_identity(source, dataset, subset_indices)

    truncated = max_batches is not None and loader_batches is not None and int(max_batches) < loader_batches
    dropped = bool(sampling["drop_last"]) and bool(batch_size) and bool(iterated) and iterated % int(batch_size) != 0
    short = (
        observed_samples is not None
        and iterated is not None
        and int(observed_samples) < int(iterated)
    )
    if truncated and bool(sampling["shuffled"]):
        # A shuffled loader truncated to a prefix produces a cohort whose
        # membership is decided by the epoch's random permutation and is not
        # recorded anywhere.  There is no honest identity to write down, so
        # this combination is refused rather than described.
        raise ValueError(
            "Cannot certify cohort identity for a truncated prefix of a shuffled loader: "
            f"sampler={sampling['sampler']!r}, max_batches={max_batches}, "
            f"loader_batches={loader_batches}. Iterate the full split, disable shuffling, "
            "or record the observed indices explicitly."
        )
    reasons: list[str] = []
    if truncated:
        reasons.append("max_batches truncated the measurement")
    if dropped:
        reasons.append("drop_last discarded a trailing partial batch")
    if sampling["iterates_every_record"] is False:
        reasons.append("a subset of the split was iterated")
    if short:
        reasons.append(
            f"only {int(observed_samples)} of {int(iterated)} available samples were observed"
        )

    if reasons:
        # An observed shortfall is decisive on its own: it is direct evidence
        # that the whole split was not measured.
        covers_full_split: bool | None = False
    elif sampling["iterates_every_record"] is None or loader_batches is None:
        covers_full_split: bool | None = None
    else:
        covers_full_split = True

    membership = split_membership_descriptor(source, split_name=split_name)
    records = list(getattr(dataset, "records", []) or [])
    # Declared record metadata only -- no volume is loaded to answer this.
    spacing_known_records = sum(1 for record in records if getattr(record, "spacing", None) is not None)
    return {
        "split_name": str(split_name),
        "membership_signature": membership["membership_signature"],
        "coverage": MEMBERSHIP_COVERAGE,
        "num_records": membership["num_records"],
        "spacing_known_records": spacing_known_records,
        "spacing_known_all_records": bool(records) and spacing_known_records == len(records),
        "available_slices": available,
        "iterated_slices": iterated,
        "loader_batches": loader_batches,
        "batch_size": None if batch_size is None else int(batch_size),
        "max_batches": None if max_batches is None else int(max_batches),
        "observed_samples": None if observed_samples is None else int(observed_samples),
        "sampling": sampling,
        "covers_full_split": covers_full_split,
        "subset_note": (
            "; ".join(reasons) + " -- protocol metadata, not a full-cohort measurement"
            if reasons
            else "every record of the split was iterated for this measurement"
        ),
    }


# ---------------------------------------------------------------------------
# Lineage assembly
# ---------------------------------------------------------------------------


def _agree_or_fail(label: str, provided: Any, derived: Any, *, tolerance: float | None = None) -> None:
    """Refuse a caller argument that contradicts the authoritative value.

    Silently preferring one of two disagreeing values is how a lineage record
    ends up describing semantics the measurement was not taken under.
    """

    if provided is None:
        return
    from .evaluation.contracts import ContractMismatchError

    if tolerance is not None:
        if abs(float(provided) - float(derived)) <= tolerance:
            return
    elif str(provided) == str(derived):
        return
    raise ContractMismatchError(
        f"lineage {label} argument {provided!r} contradicts the measurement's own "
        f"{derived!r}; refusing to record semantics the cache was not collected under"
    )


def resolve_lineage_semantics(
    *,
    cache: Mapping[str, Any] | None = None,
    metric_contract: Any = None,
    metric_space: str | None = None,
    neutral_margin: float | None = None,
) -> dict[str, Any]:
    """Derive the metric semantics of a measurement from the measurement.

    When a transition ``cache`` is supplied it is authoritative: the contract
    name it recorded is resolved to its full versioned definition, the cache's
    own embedded ``metric_space``/``neutral_margin``/version fields are checked
    against that definition, and any caller argument that disagrees is
    rejected rather than silently overridden.  ``metric_space`` and
    ``neutral_margin`` are therefore never left ``None`` once a contract is
    known.
    """

    from .evaluation.contracts import resolve_metric_contract, validate_contract

    cache_map: Mapping[str, Any] | None = cache if isinstance(cache, Mapping) else None
    contract_source: Any = metric_contract
    if cache_map is not None and cache_map.get("metric_contract") is not None:
        cached_contract = cache_map["metric_contract"]
        _agree_or_fail("metric_contract", metric_contract, cached_contract)
        contract_source = cached_contract

    if contract_source is None:
        # Nothing to derive from: the caller's own values stand alone and are
        # recorded as given, including their absence.
        return {
            "metric_contract": None,
            "metric_contract_version": None,
            "metric_contract_definition": None,
            "metric_space": None if metric_space is None else str(metric_space),
            "neutral_margin": None if neutral_margin is None else float(neutral_margin),
            "empty_class_policy": None,
            "cache_schema_version": None if cache_map is None else cache_map.get("cache_schema_version"),
            "semantics_source": "caller_arguments" if cache_map is None else "cache_without_contract",
        }

    # A cache carries the contract *name*; the full versioned definition is
    # resolved from the registry so the lineage records classes, empty policy,
    # aggregation and margin, not just a string.
    if isinstance(contract_source, Mapping):
        contract = validate_contract(dict(contract_source))
    else:
        contract = resolve_metric_contract(contract_source)
    record = contract.to_dict()

    if cache_map is not None:
        # The cache must not disagree with the contract it names.
        _agree_or_fail("cache metric_space", cache_map.get("metric_space"), record["metric_space"])
        _agree_or_fail(
            "cache neutral_margin", cache_map.get("neutral_margin"), record["neutral_margin"], tolerance=1e-7
        )
        _agree_or_fail(
            "cache metric_contract_version", cache_map.get("metric_contract_version"), record["version"]
        )
        _agree_or_fail("cache empty_class_policy", cache_map.get("empty_class_policy"), record["empty_policy"])

    _agree_or_fail("metric_space", metric_space, record["metric_space"])
    _agree_or_fail("neutral_margin", neutral_margin, record["neutral_margin"], tolerance=1e-7)

    return {
        "metric_contract": record["name"],
        "metric_contract_version": record["version"],
        "metric_contract_definition": record,
        "metric_space": record["metric_space"],
        "neutral_margin": float(record["neutral_margin"]),
        "empty_class_policy": record["empty_policy"],
        "cache_schema_version": None if cache_map is None else cache_map.get("cache_schema_version"),
        "semantics_source": "cache_metadata" if cache_map is not None else "resolved_contract",
    }


def build_lineage(
    *,
    binding: Mapping[str, Any],
    loader: Any = None,
    split_name: str = "val",
    cache: Mapping[str, Any] | None = None,
    metric_contract: Any = None,
    metric_space: str | None = None,
    neutral_margin: float | None = None,
    t_max: int | None = None,
    max_batches: int | None = None,
    batch_size: int | None = None,
    observed_samples: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the producer-side lineage record for a cached measurement.

    The evaluation revision (``evaluation_code``) is recorded separately from
    the checkpoint's producing revision (inside ``binding``) so a report-only
    commit can never read as "the model code changed".

    Pass the transition ``cache`` whenever one exists: its recorded metric
    semantics are authoritative, so ``metric_space`` and ``neutral_margin``
    come out of the measurement instead of being left ``None`` or supplied by
    a caller who could disagree with it.
    """

    from .data.common import CLASS_NAMES, NUM_CLASSES

    semantics = resolve_lineage_semantics(
        cache=cache,
        metric_contract=metric_contract,
        metric_space=metric_space,
        neutral_margin=neutral_margin,
    )
    lineage: dict[str, Any] = {
        "lineage_schema_version": int(LINEAGE_SCHEMA_VERSION),
        "checkpoint": dict(binding),
        "evaluation_code": git_source_provenance(),
        "semantics": {
            "entropy_version": dict(binding).get("model_identity", {}).get("entropy_version", UNKNOWN),
            **semantics,
            "t_max": None if t_max is None else int(t_max),
            "num_classes": int(NUM_CLASSES),
            "class_names": list(CLASS_NAMES),
        },
    }
    if loader is not None:
        lineage["preprocessing"] = preprocessing_descriptor(loader)
        lineage["split"] = split_membership_descriptor(loader, split_name=split_name)
        lineage["cohort"] = cohort_identity(
            loader,
            split_name=split_name,
            max_batches=max_batches,
            batch_size=batch_size,
            observed_samples=observed_samples,
        )
    if extra:
        lineage["extra"] = dict(extra)
    return lineage


@dataclass(frozen=True)
class CheckpointBinding:
    """The exact state a measurement is bound to.

    ``state_digest`` is the digest of the live model *after* the load, so
    :func:`verify_bound_state` at a consumer boundary detects any mutation of
    the module between binding and use.
    """

    path: Path
    role: str
    checkpoint_sha256: str
    state_digest: str
    file_state_digest: str
    producer: dict[str, Any] = field(default_factory=dict)
    model_identity: dict[str, Any] = field(default_factory=dict)
    epoch: int = 0
    global_step: int = 0
    fallback_used: bool = False
    considered: tuple[str, ...] = ()
    restored: tuple[str, ...] = ("model",)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.path),
            "checkpoint_role": str(self.role),
            "checkpoint_sha256": str(self.checkpoint_sha256),
            "state_digest": str(self.state_digest),
            "file_state_digest": str(self.file_state_digest),
            "producer": dict(self.producer),
            "model_identity": dict(self.model_identity),
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
            "fallback_used": bool(self.fallback_used),
            "considered": list(self.considered),
            "restored": list(self.restored),
        }


__all__ = [
    "CheckpointBinding",
    "LINEAGE_SCHEMA_VERSION",
    "MEMBERSHIP_COVERAGE",
    "PROVENANCE_SCHEMA_VERSION",
    "SOURCE_PATHS",
    "SOURCE_SIGNATURE_EXCLUDED_DIRS",
    "SOURCE_SIGNATURE_EXCLUDED_SUFFIXES",
    "SOURCE_SIGNATURE_ROOTS",
    "SOURCE_SIGNATURE_SUFFIXES",
    "SOURCE_SIGNATURE_VERSION",
    "UNKNOWN",
    "build_lineage",
    "checkpoint_producer",
    "cohort_identity",
    "file_sha256",
    "git_source_provenance",
    "preprocessing_descriptor",
    "producer_provenance_record",
    "resolve_model_identity",
    "source_content_signature",
    "split_membership_descriptor",
    "state_digest",
    "verify_model_config",
]
